# NanoDeploy decode-only 后端升级方案

日期：2026-08-06
状态：已实施（CPU 回归通过，GPU 验证待执行）
目标版本：dlBLAS v0.0.7、DeepGEMM v2.1.1.post3、DeepEP v1.2.1

## 1. 目标与固定前提

本方案用于把 NanoDeploy 的 DeepSeek-V3 与 Kimi-K2 decode 推理路径升级到以下固定组合：

- dlBLAS：`v0.0.7`，commit `6bc37b092d96531b5cdcb2bf5d9f9b84df3d960c`
- DeepGEMM：`v2.1.1.post3`，commit `c9f8b34dcdacc20aa746b786f983492c51072870`
- DeepEP：`v1.2.1`，commit `9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee`

方案基于以下固定决策，不在本轮讨论或修改：

1. 只支持 decode，不考虑 prefill。
2. DeepEP 只运行 low-latency dispatch/combine，不运行 normal dispatch/combine。
3. 保留 `DeepseekV2MoE.distribution = "uniform"`。
4. 保留随机专家分配逻辑，不改成模型原生 sigmoid/group-limited routing。
5. DeepSeek-V3 与 Kimi-K2 继续共用 `DeepseekV2ForCausalLM` 和 `DeepseekV2MoE`。
6. 目标硬件仍以当前 H200/SM90 集群为主；本轮不为 SM100/B200 单独设计 scale 预打包路径。

本方案只描述 NanoDeploy 需要修改的内容，不修改 dlBLAS、DeepGEMM 或 DeepEP 源码。

## 2. 当前路径与升级影响面

两种模型的共同调用链为：

```text
ModelRunner
  -> DeepseekV2ForCausalLM
    -> DeepseekV2Model
      -> DeepseekV2MoE
        -> dlblas.layers.moe.ep_moe.build_deepep_moe(low_latency_mode=True)
          -> DeepEP low_latency_dispatch
          -> DeepGEMM masked grouped FP8 GEMM
          -> DeepEP low_latency_combine

Dense/attention/shared-expert FP8 Linear
  -> nanodeploy.kernels.block_gemm_fp8.deep_gemm_fp8
    -> DeepGEMM dense FP8 GEMM
```

静态 API 审计结果：

- `build_deepep_moe(...)` 在 dlBLAS v0.0.7 中保持参数兼容；NanoDeploy 现有调用不需要改签名。
- dlBLAS v0.0.7 已同时适配 DeepGEMM 新旧 masked-grouped GEMM 名称；MoE 内部不需要 NanoDeploy 再做一层 DeepGEMM 名称适配。
- NanoDeploy 自己的 dense FP8 wrapper 仍调用已被删除的旧 DeepGEMM API，必须修改。
- DeepEP v1.2.1 新增显式销毁能力，应该接入 Ray actor 生命周期。
- decode-only 不会发生 normal -> low-latency 切换，因此不需要 `clean_low_latency_buffer()`。

## 3. 必须修改的代码

### 3.1 使用 DeepGEMM v2 dense API

文件：`nanodeploy/kernels/block_gemm_fp8.py`

当前代码：

```python
from deep_gemm import gemm_fp8_fp8_bf16_nt

gemm_fp8_fp8_bf16_nt((A, A_scale), (B, B_scale), C)
```

目标代码：

```python
from deep_gemm import fp8_gemm_nt

fp8_gemm_nt((A, A_scale), (B, B_scale), C)
```

本轮采用目标版本单栈，不保留旧 `gemm_fp8_fp8_bf16_nt` fallback。理由：

- dlBLAS、DeepGEMM、DeepEP 必须作为一个经过验证的固定组合部署。
- 保留旧 fallback 会掩盖节点安装版本不一致的问题。
- 启动期版本检查会在模型运行前报告错误，而不是让不同 rank 进入不同 API 分支。

以下内容不需要修改：

- `ceil_div`
- `get_m_alignment_for_contiguous_layout`
- `quant_fp8_tma()` 当前生成的 A scale 布局
- 模型权重的 `[N / 128, K / 128]` FP32 block scale

DeepGEMM v2.1.1.post3 仍导出前两个工具函数的兼容别名。对 H200/SM90，`fp8_gemm_nt` 会使用 `(1, 128, 128)` recipe；当前 A scale 已是 TMA-aligned MN-major 布局，DeepGEMM 会命中已有布局的 fast path。

### 3.2 重构 DeepEP actor 初始化

文件：`nanodeploy/worker/model_runner.py`

删除当前直接修改第三方类变量的代码：

```python
import deep_ep
deep_ep.Buffer.num_sms = 16
```

在构造 `self.model` 之前增加一个 decode-only DeepEP 初始化 helper。建议接口：

```python
def _configure_decode_deepep(self, ep_size: int) -> None:
    ...
```

该 helper 只在 `ep_size > 1` 时执行，并完成以下工作。

#### 3.2.1 固化有效参数

有效值优先级：driver 传入的环境变量优先；缺省时使用 NanoDeploy 默认值。

```text
DEEPEP_SMS=16
DEEPEP_MAX_TOKENS_PER_RANK=<config.max_num_seqs>
DEEPEP_ENABLE_MNNVL=0
DEEPEP_MODE=auto
```

约束：

- `DEEPEP_SMS` 必须为正偶数。
- `DEEPEP_MAX_TOKENS_PER_RANK` 必须为正数，且不得小于该 worker 可能进入 MoE 的最大 decode 行数。
- 当前普通 IBGDA 集群使用 `DEEPEP_ENABLE_MNNVL=0`；只有确认存在 MNNVL 拓扑时才允许设为 `1`。
- `DEEPEP_MODE` 必须保持 `auto`。

虽然系统只跑 low-latency decode，也不能设置 `DEEPEP_MODE=low_latency`。dlBLAS v0.0.7 的 `DeepEPBuffer.get_buffer_common()` 在初始化 shared buffer 时仍要求内部模式为 `AUTO`；强制 `low_latency` 会在 buffer 创建阶段触发断言。

`DEEPEP_MAX_TOKENS_PER_RANK` 默认从 `config.max_num_seqs` 推导，不继续依赖 dlBLAS 的隐式默认值 `128`。Kimi 的现有运行脚本使用过 batch 256，DeepSeek 也存在大于 128 的 batch 配置，因此必须把容量作为显式运行参数记录。

#### 3.2.2 开启显式销毁

在任何 `build_deepep_moe()` 调用发生前执行：

```python
from dlblas.layers.moe.token_dispatcher import DeepEPBuffer

enabled = DeepEPBuffer.set_explicitly_destroy()
```

首次初始化时 `enabled` 应为 `True`。若返回 `False` 且尚未开始模型构造，应直接报错，避免在 buffer 已创建后才尝试改变生命周期模式。

记录 actor 状态：

```python
self._deepep_enabled = ep_size > 1
self._deepep_destroyed = False
```

### 3.3 不增加 DeepEP 模式切换与清理

以下代码不应加入本轮实现：

```python
DeepEPBuffer.set_deepep_mode(DeepEPMode.NORMAL)
DeepEPBuffer.set_deepep_mode(DeepEPMode.LOW_LATENCY)
DeepEPBuffer.clean_low_latency_buffer(...)
```

原因：

- 进程从初始化到退出只运行 low-latency kernel。
- `clean_low_latency_buffer()` 的必要条件是 shared buffer 曾被 normal dispatch/combine 污染。
- 无条件清理会增加 decode host/GPU 开销，还可能被错误放入 CUDA Graph capture/replay 边界。

CUDA Graph 捕获前也不额外清理。初始 buffer 是干净的，连续 low-latency 调用属于 DeepEP 支持的正常路径。

### 3.4 在 actor 退出前显式销毁 DeepEP

文件：`nanodeploy/worker/model_runner.py`

在 `ModelRunner.exit()` 中：

1. 删除并释放 CUDA Graph 引用。
2. `torch.cuda.synchronize()`。
3. 所有 EP rank 调用 `DeepEPBuffer.destroy()`。
4. 执行一次 EP/CUDA world barrier。
5. 最后调用 `dist.destroy_process_group()`。

建议骨架：

```python
if self._deepep_enabled and not self._deepep_destroyed:
    destroyed = DeepEPBuffer.destroy()
    self._deepep_destroyed = True
    dist.barrier(group=get_dist_context().cuda_world_group)

dist.destroy_process_group()
```

要求销毁逻辑幂等，避免正常退出路径被重复调用时二次销毁 runtime。若 actor
尚未执行首个 MoE forward，dlBLAS 不会创建 shared buffer，此时 `destroy()` 返回
`False` 是正常的“无 runtime 可销毁”，不应把无请求的正常退出误报为失败。

硬 `ray.kill()` 或进程崩溃无法保证执行 Python 清理；显式销毁主要覆盖正常 shutdown 和可控异常退出路径。

### 3.5 传播并校验 DeepEP 通信环境

需要修改：

- `nanodeploy/engine/ray_executor.py`
- `nanodeploy/engine/deployment_manager.py`
- `nanodeploy/config.py`

把以下变量加入 worker runtime environment 传播：

```text
SLIME_QP_NUM
DEEPEP_SMS
DEEPEP_MAX_TOKENS_PER_RANK
DEEPEP_ENABLE_MNNVL
```

`DEEPEP_MODE` 不作为可调参数传播；actor 内部只接受或写入 `auto`。

把同一组 DeepEP 有效值加入 `Config.collective_fingerprint()`，确保同一个 collective group 的所有 rank 使用完全相同的配置。任何不一致都必须在 actor READY 前失败，不能等到 NVSHMEM/DeepEP 初始化后才暴露。

注意：

- `SLIME_QP_NUM=4` 仍按仓库要求设置并传播。
- `SLIME_QP_NUM` 不是 DeepEP 的 `num_qps_per_rank`。
- dlBLAS v0.0.7 实际使用 `max(DEEPEP_SMS, num_local_experts)` 计算 DeepEP QP 数。

### 3.6 固定和检查版本

文件：`pyproject.toml`

将：

```toml
"dlblas",
```

修改为：

```toml
"dlblas==0.0.7",
```

DeepGEMM 和 DeepEP 仍通过本地 CUDA 源码安装，不加入普通 PyPI dependency。新增一个轻量启动检查模块，例如：

```text
nanodeploy/worker/decode_backend_compat.py
```

检查内容：

1. 使用 `importlib.metadata.version()` 检查安装元数据：
   - `dlblas == 0.0.7`
   - 本地 DeepGEMM 构建应为 `2.1.1+c9f8b34`
   - 本地 DeepEP 构建应为 `1.2.1+9af0e0d`
2. 在 actor 已设置 CUDA device 后检查 DeepGEMM 符号：
   - 必须有 `fp8_gemm_nt`
   - 必须有 `m_grouped_fp8_gemm_nt_masked`
3. 检查 DeepEP 符号：
   - `Buffer.set_num_sms`
   - `Buffer.destroy`
   - `Buffer.low_latency_dispatch`
   - `Buffer.low_latency_combine`
4. 检查 dlBLAS 符号：
   - `DeepEPBuffer.set_explicitly_destroy`
   - `DeepEPBuffer.destroy`

版本或符号不匹配时，错误信息必须同时打印期望值、实际值和当前 rank。

## 4. 模型路径保持不变的部分

文件：`nanodeploy/models/deepseek_v2.py`

以下代码明确保持不变：

```python
self.distribution = "uniform"
```

以及：

```python
if self.distribution == "uniform":
    selected_experts = torch.randint(...)
```

本轮不实施以下内容：

- sigmoid gate
- `e_score_correction_bias`
- group-limited top-k
- `routed_scaling_factor`
- 模型原生专家路由数值对齐

`routing_weights` 仍沿用当前计算方式，`selected_experts` 仍被 uniform random IDs 覆盖。这是预期行为，不作为 bug 修复。

现有 `build_deepep_moe(...)` 参数保持不变：

```python
build_deepep_moe(
    low_latency_mode=True,
    ep_size=self.ep_size,
    ep_group=self.ep_group,
    num_experts=self.num_experts,
    hidden_dim=self.hidden_size,
    block_size=128,
    top_k=8,
    out_dtype=torch.bfloat16,
    ...,
)
```

dlBLAS v0.0.7 新增的 `expert_alignment` 默认值为 128，正好符合当前 FP8 grouped GEMM 路径，不需要显式传参。

## 5. DeepSeek-V3 与 Kimi-K2 的运行约束

共同配置：

- hidden size：7168
- MoE intermediate size：2048
- top-k：8
- FP8 block：128 x 128
- output dtype：BF16

模型差异：

| 项目 | DeepSeek-V3 | Kimi-K2 |
|---|---:|---:|
| routed experts | 256 | 384 |
| 常用生产 EP | 32 | 16 |
| local experts | 8 | 24 |
| `DEEPEP_SMS=16` 时的 dlBLAS QP 数 | 16 | 24 |
| first MoE layer | 3 | 1 |

启动前增加以下断言：

```python
assert hf_config.n_routed_experts % ep_size == 0
assert hf_config.num_experts_per_tok == 8
assert hf_config.quantization_config["weight_block_size"] == [128, 128]
```

DeepEP v1.2.1 已支持 top-k 10，因此两种模型的 top-k 8 均在支持范围内。

不要把 DeepSeek EP32 的 QP、buffer 占用和性能结果直接套用到 Kimi EP16。Kimi 每 rank 有 24 个本地专家，DeepEP 初始化资源和 masked grouped GEMM 的 group 数均不同。

## 6. 文件级修改清单

必须修改：

1. `nanodeploy/kernels/block_gemm_fp8.py`
   - dense FP8 API 改为 `fp8_gemm_nt`。
2. `nanodeploy/worker/model_runner.py`
   - decode-only DeepEP 环境初始化。
   - 启动版本/API 检查。
   - `DeepEPBuffer.set_explicitly_destroy()`。
   - 正常退出时 `DeepEPBuffer.destroy()`。
3. `nanodeploy/engine/ray_executor.py`
   - 传播 DeepEP 环境变量。
4. `nanodeploy/engine/deployment_manager.py`
   - hierarchical actor 同步传播 DeepEP 环境变量。
5. `nanodeploy/config.py`
   - collective fingerprint 纳入 DeepEP 有效配置。
6. `pyproject.toml`
   - 固定 `dlblas==0.0.7`。
7. `nanodeploy/worker/decode_backend_compat.py`
   - 新增版本、符号和配置检查。
8. `tests/test_decode_backend_compat.py`
   - 新增 CPU-friendly compatibility tests。

按运行入口更新并记录有效配置：

9. `scripts/run_kimi_conversation_16gpu.sh`
10. `scripts/decent-e2e/run_decent_e2e_longshort_dp4sp8_ep32.sh`

不修改：

- `nanodeploy/models/deepseek_v2.py` 中的 uniform distribution 逻辑。
- 模型目录 `/mnt/nvme1n1/ml_research/linbinbin1/DeepSeek-V3`。
- 模型目录 `/mnt/nvme1n1/ml_research/linbinbin1/Kimi-K2-Instruct-0905`。
- dlBLAS、DeepGEMM、DeepEP worktree 源码。
- prefill、normal DeepEP 或 normal/low-latency 模式切换代码。

## 7. 测试方案

### 7.1 CPU-friendly 测试

新增 `tests/test_decode_backend_compat.py`，覆盖：

1. 正确版本组合通过。
2. dlBLAS 版本不匹配时失败。
3. DeepGEMM 缺少 `fp8_gemm_nt` 时失败。
4. DeepEP 缺少显式 destroy API 时失败。
5. `DEEPEP_SMS` 非偶数时失败。
6. `DEEPEP_MAX_TOKENS_PER_RANK <= 0` 时失败。
7. `DEEPEP_MODE != auto` 时失败。
8. collective fingerprint 会随任一 DeepEP 参数改变。
9. DeepSeek 256 experts 与 Kimi 384 experts 对目标 EP 均可整除。
10. 静态或轻量 contract test 确认 `DeepseekV2MoE.distribution` 仍为 `uniform`。

建议命令：

```bash
python -m pytest tests/test_decode_backend_compat.py
python -m pytest tests/test_hierarchical_control_plane.py
python -m pytest tests/test_hierarchical_contract.py
```

### 7.2 GPU decode 验证

所有 GPU 操作前单独申请 elevated permission，并设置：

```bash
export SLIME_QP_NUM=4
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
```

验证阶段全程不调用 prefill，不运行 normal DeepEP kernel。KV/cache 与 decode 输入使用现有 decode-only 测试准备方式。

阶段 A：单节点 smoke

- dummy weight 或最小可运行配置。
- eager decode。
- 验证 dense `fp8_gemm_nt` 首次 JIT 和缓存命中。
- 验证 low-latency dispatch/combine 能重复执行。
- 验证 actor 正常退出时 DeepEP destroy 成功。

阶段 B：CUDA Graph

- full graph decode capture + replay。
- piecewise graph decode capture + replay。
- 至少覆盖两个 graph batch bucket。
- 确认没有加入 `clean_low_latency_buffer()`，且连续 replay 输出 shape、事件和通信状态稳定。

阶段 C：Kimi 生产拓扑

- 模型：`/mnt/nvme1n1/ml_research/linbinbin1/Kimi-K2-Instruct-0905`
- 2 nodes / 16 GPUs / EP16。
- local experts = 24。
- 检查实际 `NVSHMEM_IBGDA_NUM_RC_PER_PE=24`。
- 覆盖实际最大 decode batch，验证 `DEEPEP_MAX_TOKENS_PER_RANK` 容量。

阶段 D：DeepSeek 生产拓扑

- 模型：`/mnt/nvme1n1/ml_research/linbinbin1/DeepSeek-V3`
- 4 nodes / 32 GPUs / EP32。
- local experts = 8。
- `DEEPEP_SMS=16` 时检查实际 `NVSHMEM_IBGDA_NUM_RC_PER_PE=16`。
- 覆盖 full graph 和当前主力 scheduler/backend 组合。

### 7.3 必须记录的运行信息

每次运行日志至少记录：

```text
dlblas version
deep_gemm version and API family
deep_ep version
model path
EP size
local expert count
DEEPEP_SMS
DEEPEP_MAX_TOKENS_PER_RANK
DEEPEP_ENABLE_MNNVL
effective num_qps_per_rank
CUDA graph mode
SLIME_QP_NUM
```

## 8. 实施顺序与提交拆分

建议拆成三个窄提交：

1. `fix: adapt dense fp8 gemm to DeepGEMM v2`
   - `block_gemm_fp8.py`
   - DeepGEMM API contract test
2. `fix: configure decode-only DeepEP lifecycle`
   - actor 初始化、环境传播、fingerprint、destroy
   - compatibility tests
3. `build: pin dlblas decode backend versions`
   - `pyproject.toml`
   - 启动版本检查
   - 运行脚本与日志字段

每个提交先跑对应 CPU 测试。三个提交完成后再开始 GPU smoke 和多节点验证。

## 9. 验收标准

代码层面：

- NanoDeploy 不再引用 `gemm_fp8_fp8_bf16_nt`。
- NanoDeploy dense FP8 路径只调用 `fp8_gemm_nt`。
- `build_deepep_moe(...)` 仍只以 `low_latency_mode=True` 用于实际推理。
- 没有引入 normal DeepEP、模式切换或 buffer clean 调用。
- `self.distribution = "uniform"` 及随机专家选择保持不变。
- 所有 rank 在 DeepEP 初始化前完成版本和 collective 配置一致性检查。
- 正常 actor shutdown 在 process group 销毁前显式销毁 DeepEP runtime。

运行层面：

- Kimi EP16 decode eager/full graph/piecewise graph 至少各通过一次目标配置 smoke。
- DeepSeek EP32 decode eager/full graph/piecewise graph 至少各通过一次目标配置 smoke。
- 连续 decode replay 无 hang、无 buffer size assertion、无 DeepGEMM symbol error。
- actor 正常退出无 NVSHMEM/DeepEP 析构 hang。
- 日志可还原每个 rank 的完整后端版本和 DeepEP 有效配置。

## 10. 风险与回滚

主要风险：

1. 节点仍加载旧 `deep_gemm`，在首个 dense FP8 Linear 处报缺少 `fp8_gemm_nt`。
2. `DEEPEP_MAX_TOKENS_PER_RANK` 小于实际 decode 行数，导致 low-latency buffer 容量不足。
3. 部分 rank 的 DeepEP 环境不同，导致 NVSHMEM 初始化 hang，而不是干净失败。
4. 当前集群错误开启 MNNVL，改变 NVSHMEM 初始化路径。
5. Kimi EP16 的 24 QP 资源占用和 DeepSeek EP32 不同，需要独立确认。

回滚单位必须是整个后端组合，不能只回滚其中一个库：

```text
dlBLAS 0.0.5-era checkout
DeepGEMM 03d0be3
DeepEP 9fe9021
NanoDeploy 对应旧 API commit
```

禁止把 NanoDeploy dense wrapper 回退到旧 API，同时继续保留 dlBLAS v0.0.7/DeepGEMM v2 的混合部署。
