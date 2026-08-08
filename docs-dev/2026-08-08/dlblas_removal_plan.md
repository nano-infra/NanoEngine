# gemm-update 去除 dlBLAS 迁移计划

日期：2026-08-08

状态：native backend 代码迁移和 CPU/mock 验证已完成；GPU 验证按用户要求留待换机后执行

主路径：DeepSeek-V3 / Kimi-K2 decode（两者共用 `DeepseekV2ForCausalLM` 和
`DeepseekV2MoE`）

## 1. 已完成的依赖源码更新

当前 NanoDeploy 分支实际名称是 `gemm-update`，HEAD 为 `55c3233`，工作区在本次
计划文档落盘前是干净的。

两个目标依赖仓库已 fetch 并切换为 detached HEAD：

| 组件 | 路径 | 目标提交 | 源码构建版本 | 当前状态 |
| --- | --- | --- | --- | --- |
| DeepEP | `/mnt/nvme1n1/ml_research/linbinbin1/DeepEP-aug` | `73b6ea4a439ba03a695563f9fd242c8e4b02b37c` | `1.2.1+73b6ea4` | 已切换、tracked tree 干净 |
| DeepGEMM | `/mnt/nvme1n1/ml_research/linbinbin1/DeepGEMM-aug` | `477618cd51baffca09c4b0b87e97c03fe827ef03` | `2.3.0+477618c` | 已切换、submodule 对齐且干净 |

DeepEP 原目录中的未跟踪旧扩展
`deep_ep_cpp.cpython-312-x86_64-linux-gnu.so` 已可恢复地存入该仓库的
`stash@{0}`，说明为 `pre-73b6ea4-stale-extension`。这样不会把旧 native 扩展误当成
新源码的构建结果。

目标源码已构建为本地 wheel 并安装。当前 Python 环境的安装元数据是：

```text
deep_ep   1.2.1+73b6ea4
deep_gemm 2.3.0+477618c
```

两者均由 wheel 非 editable 安装；DeepEP 构建时使用
`NVSHMEM_DIR=/sgl-workspace/nvshmem/install`。当前构建环境满足 vLLM 参考脚本给出的 DeepGEMM CUDA 12.8+ 前提：
PyTorch 为 `2.8.0+cu128`，`nvcc` 为 12.9；现有 NVSHMEM 安装位于
`/sgl-workspace/nvshmem/install`，host library 为 3.4.5。

## 2. 目标与边界

本轮完成后：

1. NanoDeploy runtime、测试和项目依赖不再 import 或安装 `dlblas`。
2. DeepSeek-V3 和 Kimi-K2 decode 直接执行：

   ```text
   native DeepEP low_latency_dispatch
     -> DeepGEMM masked grouped FP8 gate/up GEMM
     -> NanoDeploy Triton SiLU * up + block-128 FP8 quant
     -> DeepGEMM masked grouped FP8 down GEMM
     -> native DeepEP low_latency_combine
   ```

3. Dense/shared-expert FP8 linear 继续直接调用 DeepGEMM。
4. 一个 worker/EP group 只拥有一个 DeepEP `Buffer`；模型层不能各自创建通信
   runtime。
5. DeepEP buffer 必须在 CUDA Graph capture 前初始化，并在 process group 销毁前
   显式 `destroy()`。
6. 固定且启动期严格检查 `deep_ep==1.2.1+73b6ea4`、
   `deep_gemm==2.3.0+477618c`，不保留旧版本 fallback。
7. DeepSeek/Kimi 现有实验语义保持不变：`distribution="uniform"`、随机专家 ID、
   top-k 8 和 FP8 block 128x128 均不在本轮修改。

主验收范围仍是当前分支已经验证过的 decode-only 路径。为了真正删除项目依赖，
Qwen3-MoE 的顶层 dlBLAS import 也必须迁移到本地实现；normal/prefill dispatcher
至少保持可用或在未支持配置上明确失败，不能留下因缺少 dlBLAS 导致的 import-time
崩溃。

## 3. 当前 dlBLAS 依赖面

直接触点只有以下几类，但 `model_runner.py` 顶层同时 import DeepSeek 和 Qwen3-MoE，
所以不能只改 DeepSeek 文件：

| 文件 | 当前职责 | 目标 |
| --- | --- | --- |
| `pyproject.toml` | 声明 `dlblas==0.0.7` | 删除依赖 |
| `nanodeploy/models/deepseek_v2.py` | import/call `build_deepep_moe` | 改用 NanoDeploy 本地 MoE backend |
| `nanodeploy/models/qwen3_moe.py` | 同上 | 同步迁移，避免 import-time 失败 |
| `nanodeploy/worker/model_runner.py` | 通过 dlBLAS `DeepEPBuffer` 设置显式销毁和回收 | 直接持有/销毁 `deep_ep.Buffer` |
| `nanodeploy/worker/decode_backend_compat.py` | 校验 dlBLAS 版本和 wrapper 符号 | 只校验 DeepEP/DeepGEMM native contract |
| `tests/test_decode_backend_compat.py` | mock dlBLAS wrapper | 改为 mock 本地 owner/dispatcher |
| debug 运行脚本和透传环境变量 | 使用 `DLBLAS_MOE_GEMM_DEBUG*` | 改名为 `NANODEPLOY_MOE_GEMM_DEBUG*` |

dlBLAS 的 `build_deepep_moe` 当前还隐含提供四类代码，删除 dependency 不等于只替换
一个 import：

- DeepEP buffer singleton 与 normal/low-latency dispatcher；
- DeepEP dispatch/combine 参数、event/hook 和 handle 管理；
- 两次 masked/contiguous DeepGEMM 及 tensor layout；
- per-token FP8 quant、masked SiLU-and-mul post-quant、prefill scatter/gather Triton
  kernels。

## 4. 两个参考实现给出的结论

### 4.1 NanoDeploy-Pure_dp

`NanoDeploy-Pure_dp` 已经把上述能力收回项目内部，最值得移植的是：

- `dlengine/context_v2/expert.py`：进程级 DeepEP buffer owner；
- `dlengine/layers/token_dispatcher.py`：直接调用 DeepEP normal/LL API；
- `dlengine/kernel/triton/hopper/fp8.py`：per-token FP8 与 masked activation
  post-quant；
- `dlengine/kernel/triton/hopper/fused_moe_v3.py`：prefill contiguous grouped GEMM
  所需的 scatter/gather 和 scale layout；
- `dlengine/layers/hopper/experts.py::_compute_decode_ep`：native LL dispatch ->
  masked DeepGEMM -> native combine 的完整顺序。

不直接整文件复制其新 HAL/`DistributedRoutedExperts` 架构；`gemm-update` 仍沿用旧模型
组织方式。本轮只抽取 native backend 和 kernels，并保留现有模型权重参数名、loader
和 forward 接口，以降低回归范围。

### 4.2 vllm-v0180

本地 vLLM checkout 正好 pin 相同目标提交。需要跟随的新版 contract 是：

- DeepGEMM 顶层扩展内部从 `deep_gemm_cpp` 改成 `deep_gemm._C`，业务代码只 import
  顶层 `deep_gemm` 即不受影响。
- Dense API 仍是 `fp8_gemm_nt((A, A_scale), (B, B_scale), out, ...)`。
- Masked API 的 canonical 名为
  `m_grouped_fp8_gemm_nt_masked((A, As), (B, Bs), out, masked_m,
  expected_m, ...)`；目标版本也保留 `fp8_m_grouped_gemm_nt_masked` alias。
- 新 GEMM API 增加 `recipe_a`/`recipe_b`，旧的前五个位置参数仍兼容。
- Hopper 的 float32 scale 路径应显式传 `disable_ue8m0_cast=True`；不要让 scale
  策略依赖默认值。Blackwell UE8M0 是另一条策略，不在本轮启用。
- DeepEP 新增 `deep_ep.topk_idx_t`；默认构建为 int64，但调用方应按导出的 dtype
  转换或在启动期检查，不能静默假设构建参数。
- Native LL buffer 使用 `explicitly_destroy=True`，由 owner 逐实例销毁，不再需要
  dlBLAS 的 `set_explicitly_destroy()` 全局开关。
- `low_latency_dispatch` 返回 `(expert_x, expert_count, handle, event, hook)`；若请求
  recv hook，消费 `expert_x` 前必须调用 hook。
- `low_latency_combine` 内部完成 top-k 权重与 reduction；不能在 NanoDeploy 再加一次
  权重。
- 目标 DeepEP 增加硬约束：

  ```text
  NVSHMEM_QP_DEPTH >= 2 * (DEEPEP_MAX_TOKENS_PER_RANK + 1)
  ```

  默认 1024 只覆盖最大 511 tokens；当前 NanoDeploy 配置允许 512，因此必须显式
  配置/校验，不能等第一次 dispatch 才失败。
- DeepEP LL 内部只有双 buffer。NanoDeploy 当前单请求顺序执行可维持单个 in-flight
  handle；如果以后引入 microbatch overlap，handle 必须按 slot 保存且最多两组。

## 5. 目标代码结构

### 5.1 DeepGEMM 单一适配层

新增 `nanodeploy/kernels/deep_gemm_backend.py`，所有业务代码通过该模块调用
DeepGEMM。职责：

1. lazy import `deep_gemm`，统一设置/记录 `DG_JIT_CACHE_DIR`；
2. 暴露 dense `fp8_gemm_nt`、masked `m_grouped_fp8_gemm_nt_masked` 和 normal
   contiguous `m_grouped_fp8_gemm_nt_contiguous`；
3. Hopper float32 scale 一律显式传 `disable_ue8m0_cast=True`；
4. 统一 alignment helper；内部使用新 `get_mk_alignment_for_contiguous_layout()`，
   对现有调用方按需返回单值或 `[align, align]`；
5. 不支持旧 DeepGEMM 名称 fallback。缺符号时启动失败，而不是首次模型执行才失败。

`nanodeploy/kernels/block_gemm_fp8.py` 改用该适配层。这样 dense、MoE 和未来 API
变动只有一个审计点。

### 5.2 进程级 DeepEP owner

把现有未实现的 `nanodeploy/worker/ep_context.py` 改成真正的 owner，或者新增等价的
`deep_ep_context.py` 后删除旧 skeleton。建议接口：

```python
context.initialize(
    ep_group=...,
    ep_size=...,
    num_experts=...,
    hidden_size=...,
    max_tokens_per_rank=...,
    num_sms=...,
    allow_mnnvl=...,
)
buffer = context.get_buffer()
context.destroy()
```

初始化发生在 `ModelRunner` 设置 CUDA device、创建 distributed groups 之后且模型构造
之前。owner 负责：

- 用 `Buffer.get_low_latency_rdma_size_hint()` 计算 LL RDMA 大小；如保留 normal
  dispatcher，则再计算 dispatch/combine 的 NVL/RDMA hints 并取最大值；
- 构造 `deep_ep.Buffer(..., low_latency_mode=True, explicitly_destroy=True)`；
- 初期保留已验证的 QP 公式
  `max(DEEPEP_SMS, num_local_experts)`，避免在移除依赖时同时改变通信调参；vLLM 的
  `num_local_experts` 公式后续单独 A/B；
- 在 Buffer 构造前解析并校验 `NVSHMEM_QP_DEPTH`，不足时明确报错或提升到不小于
  `2 * (max_tokens + 1)` 的值；最终值加入 worker env、collective fingerprint 和日志；
- 记录 `deep_ep.topk_idx_t`，dispatcher 统一转换 top-k IDs；默认目标构建应为
  `torch.int64`；
- `destroy()` 幂等，先销毁 Buffer，再由 `ModelRunner.exit()` barrier，最后销毁
  process group。

`DEEPEP_SMS` 对 low-latency communication 本身不是 SM 占用开关，但为了 normal
dispatcher 和当前 QP 兼容计算先保留；日志要将 `num_sms` 与 `num_qps_per_rank`
分开。

### 5.3 本地 dispatcher

新增 `nanodeploy/layers/token_dispatcher.py`，以 Pure_dp 为骨架，但只依赖本地
EP context 和原生 `deep_ep`：

- `DeepEPTokenDispatcherLowLatency.dispatch()`：
  top-k dtype 转换、调用 native dispatch、等待 event/hook、保存本次 handle、返回
  packed FP8 activation/scales、`masked_m` 和 `expected_m`；
- `combine()`：使用相同 handle、原始 top-k IDs 和 FP32 weights 调用 native
  combine；完成后清空 handle；
- normal dispatcher：移植 native layout/dispatch/combine 三阶段，供 Qwen3-MoE
  和非 decode 调用保持 import/runtime 完整；
- 明确断言每个 dispatcher 当前最多一个未 combine 的 handle。若后续需要 overlap，
  再引入双 slot，而不是静默覆盖。

### 5.4 本地 FP8 MoE compute

新增 `nanodeploy/kernels/moe_fp8.py`（具体文件可按现有 kernels 目录拆分），移植并
精简以下能力：

- `silu_and_mul_masked_post_quant_fwd`；
- `per_token_group_quant_fp8(..., column_major_scales=True)`；
- normal/prefill 所需 `tma_align_input_scale`、scatter/gather 和 contiguous grouped
  GEMM glue。

decode 的 scale layout 必须保持 DeepEP/DeepGEMM contract：

```text
activation: [local_experts, max_tokens * ep_size, hidden] FP8
scale:      [local_experts, max_tokens * ep_size, hidden / 128] FP32
masked_m:   [local_experts] int32
```

scale 的后两维必须保持 column-major stride；不能为了“看起来连续”调用
`.contiguous()`。`expected_m` 必须裁剪到实际 `m` 且大于 0。

新增本地 `FusedMoELowLatency`/`FusedMoENormal` facade，保持现有
`build_deepep_moe(...)` 参数和 `forward(...)` 形状不变。模型文件只改 import，
先不重构权重组织和 loader。

### 5.5 生命周期、兼容检查与配置

修改 `decode_backend_compat.py`：

- `EXPECTED_BACKEND_VERSIONS` 只保留目标 DeepGEMM/DeepEP；
- 必需 DeepGEMM 符号：dense、masked、contiguous 和 alignment helper；
- 必需 DeepEP 符号：`Buffer`、size hints、native dispatch/combine、destroy、
  `topk_idx_t`；
- 检查 `NVSHMEM_QP_DEPTH` 与 max tokens 关系；
- 保留现有跨 rank 版本/配置 all-gather，错误仍带 rank、期望和实际值；
- 删除 dlBLAS module import 和 wrapper symbol 检查。

现有 Ray/deployment env 传播补充 `NVSHMEM_QP_DEPTH`。原
`DLBLAS_MOE_GEMM_DEBUG*` 改为 `NANODEPLOY_MOE_GEMM_DEBUG*`，相应脚本、透传列表、
日志前缀和测试一起改，避免删除 dependency 后仍留下误导性控制面。

最后从 `pyproject.toml` 删除 `dlblas==0.0.7`，并执行：

```bash
rg -n --hidden -S '(^|[^A-Za-z])dlblas([^A-Za-z]|$)' \
  nanodeploy tests scripts pyproject.toml
```

允许历史 `docs-dev` 记录继续描述旧部署，但运行时代码、测试和新脚本必须为零。

## 6. 实施顺序与提交拆分

每一步先完成 CPU/static 测试并单独提交；GPU 操作另行申请 elevated permission。

### 提交 1：`build: pin native DeepEP and DeepGEMM contracts`

- 更新版本检查到目标版本；
- 增加 target symbols、top-k dtype、QP depth 配置测试；
- 更新安装手册，固定两个完整 SHA；
- 不切换模型路径。

### 提交 2：`feat: add native DeepEP runtime owner`

- 实现 EP context、Buffer sizing/creation/destroy；
- `ModelRunner` 从 dlBLAS lifecycle 改成 direct Buffer lifecycle；
- env 传播、fingerprint、跨 rank 报告加入 QP depth 和 direct backend 信息；
- owner/退出路径使用 mock 做 CPU 回归。

### 提交 3：`feat: add native DeepEP token dispatchers`

- 移植 LL dispatcher；
- 移植 normal dispatcher，保证 Qwen3-MoE 不因移除依赖而退化为 import failure；
- 覆盖 handle、event/hook、dtype、shape 和 combine 权重 contract。

### 提交 4：`feat: run MoE through native DeepGEMM`

- 新增 DeepGEMM adapter 和本地 FP8/Triton kernels；
- 接通两次 masked GEMM 和 normal contiguous GEMM；
- DeepSeek/Kimi/Qwen 模型 import 改到本地 `build_deepep_moe`；
- 保留 DeepSeek/Kimi uniform routing 和现有 GEMM debug 内容。

### 提交 5：`build: remove dlblas dependency`

- 删除 `pyproject.toml` dependency；
- 删除 compat/test 中所有 dlBLAS mock；
- debug env 和脚本改名；
- 运行静态零引用检查与 CPU suite；
- 更新迁移/回滚文档。

### 提交 6：`test: validate native DeepEP decode backends`

- 重建并安装目标 DeepGEMM/DeepEP；
- 执行 GPU smoke、CUDA Graph 和生产拓扑验证；
- 仅提交测试代码与精简结果，不提交大体积日志。

## 7. 依赖重建计划

代码接通后再重建。DeepGEMM 继续用本地 wheel，避免 editable 安装缺失 JIT
headers；DeepEP 使用带 NVSHMEM 的本地构建。安装前必须确认两个 worktree HEAD 和
cleanliness。

关键约束：

1. DeepGEMM：`DG_FORCE_BUILD=1`、`DG_USE_LOCAL_VERSION=1`，wheel 名必须包含
   `2.3.0+477618c`；目标版本内部 extension 是 `deep_gemm._C`。
2. DeepEP：`NVSHMEM_DIR=/sgl-workspace/nvshmem/install`、
   `TORCH_CUDA_ARCH_LIST=9.0`、不设置 `TOPK_IDX_BITS`（保持默认 64）；版本必须是
   `1.2.1+73b6ea4`。
3. 不修改 DeepEP、DeepGEMM 或其他第三方源码。
4. 安装后先做不启动 kernel 的 metadata/symbol/header 验收，再进行 GPU smoke。
5. vLLM 参考环境使用 NVSHMEM 3.3.24；本机现有 3.4.5 不在本轮静默替换。先验证
   target DeepEP 能否在现有 NVSHMEM 上构建和运行，如不兼容再单独提出版本变更。

## 8. 验证矩阵

### 8.1 CPU/static（无需 GPU）

- exact version mismatch、missing symbol、top-k dtype mismatch 清晰失败；
- QP depth：256 通过，511 通过，512 在默认 1024 下失败或被提升；
- one Buffer per worker，重复 initialize/destroy 幂等；
- LL dispatch/combine mock 检查参数、hook 顺序、handle 清空和 FP32 weights；
- scale shape/stride contract；
- DeepSeek 256 experts / EP32、Kimi 384 experts / EP16 的 local expert 数；
- `distribution="uniform"` 和随机 expert 分配保持；
- model_runner 导入不再要求 dlBLAS；
- 现有 hierarchical control-plane tests 不回归。

建议命令：

```bash
python -m pytest tests/test_decode_backend_compat.py
python -m pytest tests/test_hierarchical_control_plane.py
python -m pytest tests/test_hierarchical_contract.py
python -m pytest tests/test_native_moe_backend.py
python tests/test_sequence_proxy.py
```

### 8.2 GPU（每次操作前申请 elevated permission）

启动任何实验前设置：

```bash
export SLIME_QP_NUM=4
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
```

验证顺序：

1. 单卡/单 rank dense FP8：覆盖 `fp8_gemm_nt` 首次 JIT 与 cache hit。
2. 单节点 2 rank native LL：dispatch -> 两次 GEMM -> combine -> explicit destroy。
3. 已知可工作的 DP8/EP8 DeepSeek eager smoke，作为旧 dlBLAS 结果的功能基线。
4. Kimi DP16/EP16 eager：60 个 MoE 层均命中 native path，保留现有 gate-up/down
   input trace 能力。
5. full CUDA Graph 和 piecewise CUDA Graph，各覆盖至少两个 batch buckets，连续
   replay 无 handle 覆盖、buffer 污染或 JIT 插入 capture。
6. Kimi 2 nodes/16 GPUs/EP16：24 local experts；DeepSeek 4 nodes/32 GPUs/EP32：
   8 local experts。分别记录 QP、RDMA bytes、QP depth 和 teardown。
7. max tokens 256 与 512 边界；512 必须使用满足新 DeepEP 公式的 QP depth。

性能验收在功能/数值通过后进行，至少对比旧 dlBLAS 基线的 dispatch、gate-up GEMM、
activation quant、down GEMM、combine 分段时间和整体 decode tok/s。

截至 2026-08-08，本机只执行了版本/符号验收、Python 编译检查、shell 语法检查和
CPU/mock pytest；未启动 CUDA kernel、Ray 实验或 benchmark。GPU 矩阵将在换机后按
用户指示继续。

## 9. 主要风险与控制

1. **旧 native extension 混入新源码**：DeepEP 旧 `.so` 已 stash；构建后检查
   `metadata.version()` 和 module 路径。
2. **DeepGEMM 2.3 scale 策略变化**：Hopper 路径显式
   `disable_ue8m0_cast=True`，并断言 scale dtype/stride。
3. **QP depth 的 512 边界**：初始化前检查并纳入 collective fingerprint。
4. **Kimi 与 DeepSeek QP/local expert 差异**：先保持当前
   `max(num_sms, local_experts)`，不在迁移提交中顺便调优。
5. **event/hook 次序错误**：把 recv hook 作为 dispatcher contract 测试；未完成接收
   不进入 DeepGEMM。
6. **handle 被覆盖**：当前先断言一个 in-flight；CUDA Graph 和未来 overlap 需要明确
   双 slot 设计。
7. **normal -> LL buffer 污染**：如果同一进程真正执行 normal 后再 decode，进入 LL
   前调用一次 `clean_low_latency_buffer()`；纯 decode 进程不增加每步 clean 开销。
8. **Qwen3-MoE import 回归**：在删除 dependency 前完成本地 normal/LL facade，至少
   加一个 model module import contract test。
9. **首次 JIT 卡住 collective**：安装验收后、服务前按目标 shape 预热 dense 和两类
   grouped GEMM；JIT 不应发生在 CUDA Graph capture 或一部分 rank 已进入 combine 时。

## 10. 验收标准

- `nanodeploy/`、`tests/`、新运行脚本和 `pyproject.toml` 无 dlBLAS 引用；
- `pip show nanodeploy` 不再把 dlBLAS 列为 required dependency；
- 所有 rank 在 Buffer 创建前确认完全相同的 DeepEP/DeepGEMM 版本和通信配置；
- DeepSeek-V3/Kimi-K2 decode 只经过 native DeepEP + native DeepGEMM；
- Kimi EP16、DeepSeek EP32 eager/full graph/piecewise graph smoke 通过；
- max tokens 512 不触发 DeepEP QP depth assertion；
- actor 正常退出显式 destroy Buffer，随后 barrier 和 destroy process group，无 hang；
- uniform routing、输出 shape、现有控制面和 GEMM debug 能力不回归；
- 性能无无法解释的显著下降；如有下降，分段 trace 能定位到 dispatch、GEMM、quant
  或 combine，而不是回退到旧 dlBLAS。
