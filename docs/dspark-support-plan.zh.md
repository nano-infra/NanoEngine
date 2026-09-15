# DSpark 支持计划：Kimi-K3 首个接入目标

状态：支持计划，待实施；本文中的新增配置、模块和验收标准为后续实施目标，尚未落地。

跟踪：[DSpark 实现 Workstream #371](https://github.com/JimyMa/NanoDeploy/issues/371)；本文档交付由 [Task #372](https://github.com/JimyMa/NanoDeploy/issues/372) 跟踪，文档合并不代表运行时功能已经完成。

调研日期：2026-09-07。以用户提供的 K3 target 和 DSpark 本地权重为首个验收组合；部署目标沿用 `attention_dp=2 / attention_tp=8 / ffn_ep=16`。设计依据是当前 NanoDeploy 工作区、SGLang `a58fa0388e30` 和 vLLM `51da0ca66c80` 的实际代码，固定版本链接见文末。

## 1. 目标与交付边界

为 DLEngine 增加 DSpark 推测解码：draft 并行生成一个候选块，轻量 Markov head 补充块内依赖，target 一次验证多个候选，最终只提交通过验证的前缀和一个 target token。confidence head 用于后续按请求分配验证长度。SGLang 将这些职责拆为 proposer、verify executor、KV injector 和 planner；vLLM 在 DFlash 的块式 draft 基础上增加 DSpark speculator。[SGLang worker][sg-worker]、[vLLM speculator][vl-speculator]

分两级交付，避免把“能够生成”与“具备完整调度收益”混在一起：

- **基础支持**：当前 K3 + 已提供的 GQA DSpark draft，固定 7 个候选、8 个 target 验证输入；支持正确的 greedy 和随机采样、KDA 状态恢复、现有并行配置及长 prompt 分块 prefill。
- **完整支持**：在基础正确性上增加低显存状态恢复、CUDA Graph 和 confidence 驱动的变长验证，并提供相同资源条件下的性能报告。

首轮覆盖文本、OpenAI/Anthropic 普通与流式接口、reasoning 和工具调用。PP、P/D 分离、多模态、其他 draft 架构及 draft 量化作为后续任务；这些组合不能因底层已有部分能力而直接宣称支持。现有 MTP 和非 speculative 路径必须保持兼容。

本文只提出工程接入和验证方案，不训练或转换 draft 权重，也不预设性能提升倍数。

## 2. 已确认的本地权重契约

已只读检查本地 `config.json` 和 `model.safetensors` 头部，共 62 个张量；没有加载 GPU 或执行权重目录中的 Python 文件。本地配置与公开 [RadixArk/Kimi-K3-DSpark][draft-card] 的固定 revision `3c5bac301d9cf392706189d82ed947feca6c2f0f` 一致；这不等于已经核验完整权重内容相同。[公开配置][draft-config]

| 项目                | 已确认值                                      | 对接要求                                                            |
| ------------------- | --------------------------------------------- | ------------------------------------------------------------------- |
| draft 架构          | `DSparkDraftModel`，Qwen3 风格 GQA，5 层      | 独立 draft loader，不能当作 target 的 MTP 层加载                    |
| hidden / FFN        | `7168 / 14336`                                | target 特征投影和共享 embedding/head 的维度必须匹配                 |
| attention           | 64 个 Q heads、16 个 KV heads、`head_dim=64`  | 支持块内非因果 GQA；TP8 时每 rank 8 Q / 2 KV heads                  |
| 候选块              | `block_size=7`                                | 7 个 draft 候选，target 验证宽度为 8                                |
| target 特征层       | `[7, 23, 51, 67, 83]`                         | 按 SGLang 的零基 layer 后特征约定捕获                               |
| 特征投影            | `fc.weight=[7168,35840]`                      | 按指定顺序拼接 5 个 7168 维特征；不是取平均或只取最后一层           |
| Markov              | `vanilla`、rank 256，两个 `[163840,256]` 权重 | 后一位置使用实际选出的前一 token；保留修正后的 draft 分布           |
| confidence          | 权重 `[1,7424]`、bias `[1]`                   | 输入包含 7168 维 hidden 和 256 维 Markov embedding                  |
| vocabulary / mask   | `163840 / 163824`                             | mask 是输入占位；使用 target tokenizer 和显式 token 映射校验        |
| embedding / LM head | 本地权重中未包含                              | 按 checkpoint 契约共享 target 模块，验证 dtype、词表及 TP 分片      |
| precision           | 所有已列权重 BF16                             | 首版 draft 权重和 draft KV 均 BF16；target MLA KV 可单独为 FP8      |
| RoPE                | YaRN，factor 16，原始长度 65536，上限 1048576 | 使用 draft 自己的 RoPE 配置；上限不代表 NanoDeploy 已验证 1M 上下文 |

`num_target_layers=93` 表示 target 总层数，不能误作需要捕获 93 个特征。当前 target 配置 `num_nextn_predict_layers=0`，也没有发现声明 DSpark 内嵌权重的配置字段；接入应显式使用这份独立 draft。

**首个 loader 只承诺支持这份 GQA checkpoint。** vLLM 另有 `K3DSparkForCausalLM` / `k3_dspark` 的 dense MLA draft：投影、KV 格式及配置均不同，其固定版本 loader 还将 confidence 权重标为不加载。不能把它与本地 GQA draft 混用，也不能由它推断本地 confidence head 不需要加载。[vLLM K3 MLA draft][vl-mla-draft]

## 3. 从上游借鉴哪些设计

| 主题          | SGLang 的参考点                                                          | vLLM 的参考点                                                    | NanoDeploy 的选择                                    |
| ------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------- | ---------------------------------------------------- |
| 编排          | `DSparkWorkerV2` 组合 draft、verify、injector、planner                   | GPU worker 下的 `DSparkSpeculator` 复用 DFlash 基础设施          | 增加独立 DSpark runner；保留 MTP runner 及兼容入口   |
| 块生成        | 一次 backbone forward，再串行执行轻量 Markov head                        | 区分 anchor 预测布局和 fill-in 布局                              | 按已选 checkpoint + SGLang worker 固定布局，禁止猜测 |
| target 特征   | K3 `_dspark_capture_stream` 导出下一消费者所需的 pre-norm AttnRes stream | K3 有独立 aux stream 捕获及层编号转换                            | 首先对齐 SGLang 训练/推理语义，显式记录层编号约定    |
| draft context | `TargetHiddenKvInjector` 把已提交 target 特征投影到 draft KV             | context KV precompute 与 query block 分离                        | 独立 draft KV 生命周期；拒绝分支不进入持久 context   |
| 采样          | draft 保留 Markov-corrected logits，verify 处理接受与恢复                | speculator 保留实际 draft logits，处理词表映射                   | 先 greedy，再实现一般 `p/q` 拒绝采样                 |
| KDA 恢复      | ReplaySSM 保存短输入窗口，在接受后重放状态递推                           | RecoverSSM 有独立 verify/recovery 缓冲与提交上下文               | 先快照参考实现，再对齐本地精度的低显存恢复           |
| confidence    | planner、SPS 成本表、STS 校准和 ragged verify 分离                       | adaptive verification 检查 checkpoint 是否有可用 confidence head | 固定块是可比较的基线；变长调度单独验收               |

参考入口：[worker][sg-worker]、[draft][sg-draft]、[K3 特征][sg-k3]、[KV injector][sg-injector]、[planner][sg-planner]、[ReplaySSM][sg-replay]、[vLLM speculator][vl-speculator]、[vLLM K3 特征][vl-k3]、[RecoverSSM][vl-recover]。

上游“支持 DSPARK”不等于某个 backend/并行组合可直接复用。例如，本次固定版本的 SGLang 参数入口仍限制 PP，并对 DP attention 的 MoE backend 有检查；vLLM 新 GPU worker 的 DSpark 入口也不能与旧 `DFlashProposer` 的入口混为一谈。对照实验必须记录实际 worker、backend、配置和提交版本。[SGLang 参数校验][sg-hook]

## 4. NanoDeploy 当前差距与代码落点

当前工作区包含前序 K3 cache/capacity 集成预览。实施时先对齐 `Pure_dp`，确认 #363、#368、#369 的实际合入状态；已合并的 #370 为协议解析回归提供基础。不能把工作区中的预览功能自动视为目标分支已有功能。

| 当前代码                                                                                                                   | 已有能力或限制                                                            | 计划改动                                                                  |
| -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| [config.py](../dlengine/config.py)                                                                                         | `num_speculative_tokens>0` 绑定原生 MTP；多步验证限制特定模型             | 按 algorithm 校验；DSpark 由独立 draft 配置确定能力                       |
| [模型注册](../dlengine/runtime/models/registry.py)、[loader](../dlengine/runtime/runner/loader.py)                         | target 与 MTP 加载入口                                                    | 新增 draft registry/loader，共享模块在加载后显式绑定                      |
| [K3 模型](../dlengine/runtime/models/kimi_k3/kimi_k3.py)                                                                   | 只返回主干结果；AttnRes 带可变 bank                                       | 增加只读的多层特征导出；关闭 DSpark 时无额外捕获                          |
| [model_runner.py](../dlengine/runtime/runner/model_runner.py)                                                              | 推测执行与 `mtp_runner` 紧耦合                                            | 增加最小 runner 分发契约；不要为 DSpark 重写整个执行器                    |
| [mtp_runner.py](../dlengine/runtime/runner/mtp_runner.py)                                                                  | 已有线性 greedy 验证、one-hot draft 的拒绝采样；GDN 回滚仅覆盖旧 N=1 路径 | 提取可复用的提交结果结构；DSpark 单独实现一般 draft 分布和任意前缀恢复    |
| [KDA backend](../dlengine/runtime/layers/backends/delta_net/kda.py)、[GDN cache](../dlengine/runtime/context/cache/gdn.py) | 单 token recurrent decode；现有 active/backup 状态池                      | 多 token causal verify，conv + recurrent 状态按接受长度提交               |
| [MLA backend](../dlengine/runtime/layers/backends/mla/trtllm.py)                                                           | 已有 `num_tokens_per_seq` reshape 和 BF16/FP8 cache 路径                  | 验证 N+1 的因果 mask、positions、seq_lens 和 FP8 shape 契约               |
| [attention backends](../dlengine/runtime/layers/backends/attention/)                                                       | GQA/MLA 按目标模型选择                                                    | draft 单独选择支持 context + 非因果 query block 的 GQA backend            |
| [BatchContext](../dlengine/runtime/context/batch.py)、[input_preparer](../dlengine/runtime/runner/input_preparer.py)       | decode/prefill 与 MTP 扩展 metadata                                       | 显式区分 target verify 和 draft propose 的 mask、长度、slot、缓存写入目的 |
| [graph_runner.py](../dlengine/runtime/runner/graph_runner.py)                                                              | 普通 decode、MTP 和线性 verify graph                                      | 独立 capture draft、target verify 和恢复；固定地址及 bucket 容量          |
| [Rust cache_flow](../rust/src/scheduler/cache_flow.rs)、[types](../rust/src/scheduler/types.rs)                            | 已预留 MTP lookahead，但公式服务于其双窗口生命周期                        | 审计 target/draft reservation、释放、slot 复用和实际提交 token 数         |

建议新增模块边界（文件名可在实现时调整）：

```text
runtime/models/dspark/                 # 本地 GQA draft 结构与严格权重加载
runtime/runner/speculative/base.py     # proposal / verify / commit 数据契约
runtime/runner/speculative/dspark.py   # DSpark 执行编排
runtime/runner/speculative/verify.py   # 一般拒绝采样与接受前缀
runtime/runner/speculative/planner.py  # static / confidence 验证计划
runtime/context/cache/draft.py        # draft KV 及请求所有权
```

## 5. 一轮执行必须遵守的语义

### 5.1 anchor、候选、验证与提交

统一术语：`γ=7` 是候选数；每次最大 target query 数 `W=γ+1=8`。轮开始时 anchor `a` 已被 target 选出并发送给客户端，但还未写入本轮 target KV；持久缓存覆盖 `a` 之前的前缀。

```text
已提交前缀的 target aux features ──投影──> draft context KV
                                      + anchor / mask query block
                                      ↓
                            并行 backbone + 串行 Markov head
                                      ↓
                                d1, ..., dγ
                                      ↓
target causal verify:          [a, d1, ..., dγ]
                                      ↓
接受 r 个候选：               [d1, ..., dr] + 新 target token b
                                      ↓
提交输入状态：               [a, d1, ..., dr]；b 成为下一轮 anchor
```

- 本地选定的 SGLang 路径使用 **7 个 draft query：anchor + 6 个 mask**，每个 query 都产生下一位置的预测；不要误写成产生 7 个候选就一定需要 8 个 draft query。
- target 的第 `j-1` 行 logits 验证 `dj`；全部接受后，第 `γ` 行产生 bonus token。
- `accepted_drafts=r`，`committed_input_len=r+1`。`r=0` 时仍提交 anchor 的状态；不能恢复到 anchor 之前后直接跳到下一轮。
- 输出为 `r` 个候选加 1 个恢复/bonus token；已发送的 anchor 不得重复发送。新 token `b` 尚未经过 target forward，不能提前声称其 KV/aux features 已就绪。
- EOS、stop、`max_tokens` 和 context 上限可截断输出；token 数、状态所有权和释放必须按实际终止位置处理。

这些长度和状态语义应成为独立测试的契约，不能靠 `num_speculative_tokens` 一个整数在各模块中隐式推断。[SGLang draft 输入][sg-draft]

### 5.2 特征与 draft attention

捕获的是指定 layer 后、下一层 attention 消费前的 AttnRes 聚合流。不能直接保存 `hidden`、`prefix` 或最终 norm 输出作为替代。导出操作不得改变残差 bank、写入计数或后续 target 输出；prefill 和 verify 必须使用相同定义。[SGLang K3][sg-k3]

实现时按 checkpoint 顺序拼接 5 个特征，经 `fc` 和 `hidden_norm` 生成 context，再由每个 draft layer 的 K/V 投影写入自己的池。核对 q/k norm、RoPE、position 和 TP 行顺序；模型和算子均不允许继承 target 的量化选项后误解释 BF16 draft 权重。

draft attention 能访问历史 context 和当前 query block；当前块内部是非因果注意力。target verify 仍为因果注意力。显式提供各自的 backend capability 和 mask，不能共用一个 `is_prefill=False` 分支就认为语义一致。

分块 prefill 每完成一个 chunk 就捕获并投影该 chunk 的特征，及时释放临时特征；新 draft query 写入的临时 KV 不得当作持久 target-derived context。接受后用对应的 target 特征覆盖/提交正确前缀，废弃后缀从逻辑长度中移除。

### 5.3 正确的随机采样

首个可验证版本使用 temperature 0。之后实现一般 draft 分布下的拒绝采样：候选 `y` 的接受概率为 `min(1, p(y)/q(y))`；首次拒绝后，从归一化的 `max(p-q,0)` 采样恢复 token；全部接受则从最后一行 target 分布采样 bonus。

这里 `q` 必须是 **加入 Markov bias 并应用实际 draft 采样变换后的分布**，不是 backbone 原始 logits，也不是 confidence。仅保存 `q(y)` 不足以构造恢复分布；需要保留或准确重建该行完整分布。当前 `linear_rejection_sample` 假设 greedy draft 的 q 为 one-hot，不能直接用于 probabilistic DSpark。[vLLM draft 采样][vl-speculator]

target 的温度、已支持的 logits 处理及请求约束必须在每个候选前缀上生效。超出已验证采样能力的请求应有明确处理策略，不能静默忽略参数。随机采样比较分布和质量，不要求同 seed 与普通解码逐 token 相同；greedy 则以相同精度 target 的输出为基准。

## 6. KDA 状态与显存设计

### 6.1 正确性基线和低显存实现

K3 同时有 MLA KV、KDA recurrent state 和 causal-conv state。只缩短 MLA `context_lens` 不会撤销 KDA 的状态更新。

第一步建立小 batch 的正确性参考：从持久状态开始，在 verify 中保留每个输入位置之后的 recurrent/conv 状态，按 `committed_input_len` 选择状态。逐一验证 `r=0..7`，特别是首次拒绝、全部接受和拒绝后连续下一轮。

随后实现 ReplaySSM/RecoverSSM 风格的恢复：保留轮前状态和短输入/修正窗口，接受结果确定后恢复正确前缀；conv 历史一起提交。具体采用重放还是修正重建，由与本地 kernel 的精度对齐及实际成本决定。[SGLang ReplaySSM][sg-replay]、[vLLM RecoverSSM][vl-recover]

不能直接移植上游“bitwise exact”的结论：本地 Blackwell KDA 状态池默认 BF16，而本次 SGLang 专用 CuTe verify 分支要求 FP32 state。必须固定本地 state dtype、每步舍入、gate lower_bound、L2 norm 和运算次序后建立参考；若需要改为 FP32，单独评估质量、显存和非 speculative 基线。[SGLang KDA backend][sg-kda]

### 6.2 按当前形状算出的预算

以下为配置和张量形状推算，尚非运行峰值测量；不含 allocator 对齐、workspace、graph pool 和通信 buffer。

| 项目                    | 当前形状下的估算                                                | 实施要求                                                                   |
| ----------------------- | --------------------------------------------------------------- | -------------------------------------------------------------------------- |
| draft 权重原始存储      | 2,249,289,601 个 BF16 参数，约 4.19 GiB                         | 这不是 TP8 的单 rank 实际占用；Markov/投影可能复制，需逐模块核算           |
| 5 层 aux 特征           | chunk 8192 约 560 MiB；16384 约 1120 MiB                        | 是完整 feature rows 的大小；若保留分层输出又 concat，临时峰值还会增加      |
| draft GQA KV            | TP8 且 16 KV heads 按 8 rank 分片时，每 rank 2560 bytes/token   | 公式 `5 × 2(K,V) × (16/8) × 64 × 2(BF16)`；32K context 约 80 MiB/请求/rank |
| KDA 单份状态            | 69 层、每 rank 12 heads、128×128、BF16，约 25.875 MiB/请求/rank | 仅 recurrent state，不含 conv                                              |
| 8 个位置的完整 KDA 快照 | 约 207 MiB/请求/rank                                            | 16 请求约 3.23 GiB/rank；参考实现不能不设限地变成生产默认                  |

缓存预算必须联合考虑 target MLA KV、draft GQA KV、KDA 状态、恢复窗口、draft/target logits、aux buffers 和 graph pools。`--kv_cache_dtype fp8_e4m3` 首版只控制已支持的 target MLA cache，draft KV 保持 BF16；不能据此宣称所有 cache 已减半。

draft KV 的 sharding 按其实际 attention TP 计算，不能除以全局 world size 或 EP。每个 attention-DP 组持有自己的 draft 上下文；共享逻辑 slot 表时，也要独立建模存储、dtype、容量及回收责任。

已有 K3 MegaMoE 容量计算需纳入 `max_num_seqs × W` 和 graph padding，不能因 confidence 平均验证长度较短而缩小最大容量。chunk 8192、16384 只改变单次 prefill 工作量，完整 context 长度决定持久 KV 占用，二者应分别报告。

## 7. 并行与生命周期

首版 draft 使用 target 的 attention TP group，每个 DP 组一份逻辑 draft，暂不提供独立 draft TP 大小。draft 为 dense GQA，不进入 target MegaMoE EP group；采样 token、接受长度、RNG ownership 和下一轮 anchor 在同一 attention TP 组内一致。

真实 K3 从现有 DP2/TP8/EP16 拓扑验收，先 batch 1，再增加并发。小模型/单层 fixture 可以单卡运行，不能把单卡 fixture 通过写成整模型 K3 单卡支持。

必须覆盖以下情况：

- 只有一个 DP 组有请求，另一组为空；target EP collectives 的调用顺序仍一致，空组执行所需 dummy 工作。
- target/draft context 切换不污染 TP group、量化策略、cache pointers 或 metadata；CUDA Graph padding 不写真实请求的 state slot。
- 请求重排、插入、结束、abort、slot 复用时，proposal 与状态绑定 `seq_id + generation/epoch`，避免复用过期候选。
- KV 页边界、cache pressure、抢占重算：target 与 draft 同时回收/重建，不能只有 target prefix 命中而假定 draft KV/aux 已存在。
- server 只接收已验证输出 token bundle；#370 的结构化 parser 不接触未接受 draft，usage 也不计入被拒绝候选。

PP 和 P/D 需要额外传递 aux features、draft KV 与 KDA checkpoint，留到核心版本完成后再设计，不在本次首轮验收中隐式开启。

## 8. Confidence 调度与 CUDA Graph

固定验证先跑通，再引入 `verify_lens[B]`。每条请求只能选择连续候选前缀，额外保留 anchor 行；验证预算按 `sum(1 + verify_lens[i])` 计算。用 confidence 估计前缀存活率，用实测 target verify 成本选择预算，不能把 confidence 当接受判定。[SGLang planner][sg-planner]

阶段顺序：

1. **Static eager**：所有请求固定 7 个候选，建立可比较的正确性和性能基线。
2. **Static graph**：分别捕获 draft backbone + Markov、target verify、状态恢复；把 scratch 和 cache metadata 预分配为固定地址。
3. **Confidence eager / bucket**：验证 ragged offset、状态提交和请求公平性；先证明采样一致性，再开放随机采样下的 adaptive 模式。
4. **Confidence graph + cost model**：引入实际收益明确的 graph tiers、SPS 成本表及必要的 STS 校准；表必须绑定硬件、TP/DP/EP、cache dtype、block size 和 kernel 版本。

当前 SGLang K3 CuTe verify 快路径对验证宽度、ragged layout 和 dtype 有严格要求，变长模式可能切换 kernel。性能比较需要注明实际 backend；不能只观察“验证 token 变少”就认定变快。[SGLang KDA 条件][sg-kda]

## 9. 配置提案

以下参数**尚未实现**。建议沿用 NanoDeploy 的下划线命名，并让一个字段统一表达最大候选数。

| 参数                                            | 提案语义                                                                            |
| ----------------------------------------------- | ----------------------------------------------------------------------------------- |
| `speculative_algorithm=auto\|none\|mtp\|dspark` | 默认 auto 保留旧行为：旧配置 N>0 走 MTP，否则普通解码；显式 dspark 走独立 loader    |
| `speculative_draft_model`                       | DSpark 必填，指向已提供的本地 draft；不因 target 存在同名目录就自动推断             |
| `num_speculative_tokens`                        | 对 DSpark 为 γ；未显式指定时从 draft 读取 7，显式与当前 checkpoint 不兼容时启动报错 |
| `speculative_verify_mode=static\|confidence`    | 默认 static；confidence 要求完整权重及对应 backend/graph 能力                       |
| `speculative_draft_kv_cache_dtype`              | 首版仅 BF16；与 target 的 `kv_cache_dtype` 分开                                     |

兼容实现需区分用户未提供 N 与显式 N=0；现有 N=0 的普通解码语义不能因增加 auto 模式被意外改变。当前 checkpoint 的 γ 固定为 7；γ=1/2/4 用于支持相应布局的测试 fixture，不能把 checkpoint 配置直接改小就宣称兼容。

拟支持的启动形式：

```bash
# 设计示例：新增参数尚不可用。将两个占位路径替换为本地 target / draft。
dlengine serve /path/to/Kimi-K3 \
  --attention_dp 2 --attention_tp 8 --ffn_ep 16 \
  --max_num_batched_tokens 8192 \
  --kv_cache_dtype fp8_e4m3 \
  --speculative_algorithm dspark \
  --speculative_draft_model /path/to/Kimi-K3-DSpark \
  --num_speculative_tokens 7 \
  --speculative_verify_mode static
```

每次启动输出解析后的 target/draft 架构、特征层、γ/W、state/cache dtype、draft TP group、attention backend 和各类 buffer 预算。运行指标至少包括 proposed/verified/accepted draft tokens、实际输出 tokens、接受长度分布、draft/verify/commit 耗时、显存及请求等待时间。

## 10. 实施拆分与验收顺序

由 [Workstream #371](https://github.com/JimyMa/NanoDeploy/issues/371) 跟踪 DSpark 实施。下面各阶段开始前建立独立叶子 Task，并设置正式 Sub-issue、负责人和依赖；阶段名称不是已创建的 Issue。代码 PR 以 `Pure_dp` 为基线，文档发布任务 #372 仅完成本计划的归档。

| 阶段                 | 交付物                                                                                          | 通过标准                                                                                       | 依赖          |
| -------------------- | ----------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ------------- |
| P0：冻结参考         | 本地权重 manifest、配置验证、SGLang target-only/DSpark 参考 trace；记录 backend 与精度          | 明确 aux 特征语义、anchor 对齐、实际模型版本和参考采样设置；本地头部核对已完成                 | 无            |
| P1：draft 与特征     | algorithm 配置/注册、GQA loader、只读 K3 aux capture、共享模块、独立 draft KV、非因果块 forward | 选定 checkpoint 所有必需权重加载；分层 features、投影、logits、Markov 和 confidence 与参考对齐 | P0            |
| P2：固定块 greedy    | DSpark runner、target MLA/KDA 多 token verify、快照参考恢复、调度预留和 bundle 输出             | 小 batch 的全部接受长度通过；真实 K3 batch 1，DP 空组可运行；输出与同精度 target-only 对齐     | P1            |
| P3：采样与服务完整性 | 一般 p/q 采样、EOS/stop/预算/abort、重排和 slot 复用                                            | 小词表分布测试、真实 K3 随机采样、普通/流式工具调用和 usage 全部通过                           | P2            |
| P4：生产容量与图     | 低显存 KDA 恢复、static CUDA Graph、各池预算和 DP/EP 压力测试                                   | 8192/16384 chunk、长 prompt、混合并发和 cache pressure 无状态泄漏；提交显存与延迟报告          | P3            |
| P5：confidence 调度  | verify_lens、ragged/bucket、成本表和指标                                                        | 变长提交与采样正确，含低 confidence/公平性测试；相对 static 在目标负载上有实测收益             | P4            |
| P6：扩展组合         | PP/P/D、多模态、其他 draft/量化                                                                 | 分组合建立新的协议、测试和性能报告；不作为 P0–P5 的隐式承诺                                    | P5 后另立任务 |

若某阶段只覆盖 greedy、eager 或固定 block，应在功能文档中保留该限制，不能提前写成完整 DSpark 支持。后续必要工作必须关联 Issue，不留口头 follow-up。

## 11. 测试与性能验收矩阵

| 层级       | 必测内容                                                                                             | 判定方式                                                                                |
| ---------- | ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| 配置与权重 | 62 个张量映射、缺失/额外参数、mask 越界、特征顺序、γ 不匹配、GQA head_dim、TP divisibility、共享权重 | 明确失败或完整加载；不静默随机初始化缺失 Markov/confidence 参数                         |
| 数值       | target aux、`fc+norm`、draft K/V、块 mask、base/Markov logits、confidence                            | 保存分阶段 reference；与相同 dtype 参考比较，近似容差写入测试                           |
| 接受与恢复 | r=0..7、连续多轮拒绝/全收、conv 窗口、KDA state、MLA KV、bonus 下标                                  | 每轮与逐 token target 重放比较；greedy 输出严格一致，数值差异需定位                     |
| 随机采样   | 小词表可枚举 p/q、q=0 边界、p=q、混合温度、Markov 条件分布                                           | 精确小例子及固定统计置信区间；不靠一段自然语言输出证明无偏                              |
| 缓存与调度 | 页边界、batch 重排、DP 单侧空闲、active/dummy slot、抢占、结束和复用                                 | 没有越界、重复提交、跨请求污染或 collective hang                                        |
| 长上下文   | chunk 8192/16384；prompt 小于、等于、大于 chunk；至少 33K prompt 跨多个 chunk，再测 64K/128K         | target/draft context 连续；输出可精确核验的长文检索与标识符复制；更长长度按显存逐级扩展 |
| 服务接口   | OpenAI/Anthropic 普通与 SSE，reasoning、多个工具调用、EOS/stop/max_tokens、usage                     | 与非 speculative 接口契约一致；未接受 token 不进入 parser，工具参数不重复               |
| 回归       | DSpark 关闭、现有 GLM/Qwen MTP、K3 BF16/FP8 KV                                                       | 原有相关测试继续通过；开关关闭后不增加 draft 分配或 aux capture                         |

性能对照固定 target/draft 权重版本、GPU 数、DP2/TP8/EP16、请求集合、输入/输出长度、采样参数、target KV dtype 和 chunk。比较普通解码、static DSpark、confidence DSpark；SGLang/vLLM 仅在实际支持相同 checkpoint/拓扑时作为外部对照。

并发依次为 1/2/4/8/16/32，在共同可运行的 batch 上比较延迟，同时单独报告各模式最大稳定并发，防止把减少 KV 容量换来的吞吐损失隐藏掉。记录 TTFT、TPOT/ITL P50/P95/P99、请求吞吐、实际输出 tok/s、GPU 峰值显存，以及 aux/draft/verify/commit 分项时间。

候选数或接受率不能替代端到端收益。warmup 后至少重复三轮，报告离散程度；低并发与代表性服务负载需要有超出测量波动的收益，TTFT/尾延迟变化一并呈现。若没有稳定收益，保留实验入口并定位成本，不默认启用。公开 checkpoint 的 1M 配置和上游成绩只作为后续测试依据，不作为本引擎已经通过的验收结果。

## 12. 实施基线

1. 首个正式支持组合固定为当前 K3 + 已提供的 RadixArk 风格 GQA draft，γ=7，先 static、后 confidence。
2. 保留当前 DP2/TP8/EP16 部署作为整模型验收目标；draft 初期不单独配置 TP，PP/P/D 后续处理。
3. target MLA FP8 与 draft BF16 KV 分开配置；低显存 KDA 恢复作为生产容量阶段的必要工作。
4. 在 #371 下按 P0–P5 逐阶段登记 Task/PR；本次交付为计划文档，实现、模型验证和性能测试由后续任务完成。

## 参考源码与权重

以下均为原始项目来源。链接固定到调研版本；实施前再次检查差异，避免把上游后续改动当作本文已覆盖的能力。

- [SGLang DSpark worker][sg-worker]、[draft proposal][sg-draft]、[KV injector][sg-injector]、[confidence planner][sg-planner]、[参数校验][sg-hook]。
- [SGLang K3 特征捕获][sg-k3]、[KDA verify 条件][sg-kda]、[KDA ReplaySSM][sg-replay]。
- [vLLM DSpark speculator][vl-speculator]、[K3 aux stream][vl-k3]、[K3 RecoverSSM][vl-recover]、[另一种 K3 MLA draft][vl-mla-draft]。
- [RadixArk K3 DSpark 模型说明][draft-card]、[固定版本配置][draft-config]。正文中的具体张量形状也已用本地 safetensors 头部复核。

[draft-card]: https://huggingface.co/RadixArk/Kimi-K3-DSpark
[draft-config]: https://huggingface.co/RadixArk/Kimi-K3-DSpark/blob/3c5bac301d9cf392706189d82ed947feca6c2f0f/config.json
[sg-draft]: https://github.com/sgl-project/sglang/blob/a58fa0388e30315e641b9666fc5c096033ace5d9/python/sglang/srt/speculative/dspark_components/dspark_draft.py
[sg-hook]: https://github.com/sgl-project/sglang/blob/a58fa0388e30315e641b9666fc5c096033ace5d9/python/sglang/srt/arg_groups/speculative_hook.py
[sg-injector]: https://github.com/sgl-project/sglang/blob/a58fa0388e30315e641b9666fc5c096033ace5d9/python/sglang/srt/speculative/dspark_components/dspark_kv_inject.py
[sg-k3]: https://github.com/sgl-project/sglang/blob/a58fa0388e30315e641b9666fc5c096033ace5d9/python/sglang/srt/models/kimi_k3.py
[sg-kda]: https://github.com/sgl-project/sglang/blob/a58fa0388e30315e641b9666fc5c096033ace5d9/python/sglang/srt/layers/attention/linear/kda_backend.py
[sg-planner]: https://github.com/sgl-project/sglang/blob/a58fa0388e30315e641b9666fc5c096033ace5d9/python/sglang/srt/speculative/dspark_components/dspark_planner.py
[sg-replay]: https://github.com/sgl-project/sglang/blob/a58fa0388e30315e641b9666fc5c096033ace5d9/python/sglang/kernels/ops/attention/fla/kda_replayssm_spec_decode.py
[sg-worker]: https://github.com/sgl-project/sglang/blob/a58fa0388e30315e641b9666fc5c096033ace5d9/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py
[vl-k3]: https://github.com/vllm-project/vllm/blob/51da0ca66c8065619c79e35dff97aa99aeaf5644/vllm/models/kimi_k3/nvidia/model.py
[vl-mla-draft]: https://github.com/vllm-project/vllm/blob/51da0ca66c8065619c79e35dff97aa99aeaf5644/vllm/models/kimi_k3/nvidia/dspark_mla.py
[vl-recover]: https://github.com/vllm-project/vllm/blob/51da0ca66c8065619c79e35dff97aa99aeaf5644/vllm/models/kimi_k3/nvidia/ops/recoverssm.py
[vl-speculator]: https://github.com/vllm-project/vllm/blob/51da0ca66c8065619c79e35dff97aa99aeaf5644/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py
