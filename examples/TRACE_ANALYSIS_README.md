# Trace-Driven Hyperparameter Tuning Guide

本指南说明如何使用 Trace 数据来离线推导最优的 SP 调度超参数。

## 工作流程

### 阶段 1: 收集 Trace 数据（32DP, SP=1）

在纯 DP 模式下运行，收集详细的系统状态快照：

```bash
python bench_serving.py \
    --dp 32 --sp 1 \
    --sp-size-mode load_aware \
    --export-hyperparams trace_data.jsonl \  # 注意：必须是 .jsonl 结尾
    --num-requests 10000 \
    --request-rate 10
```

这会生成 `trace_data.jsonl`，每行一个 JSON 对象，包含：
- 请求特征：`prompt_length`, `timestamp_ms`
- 系统状态：`free_blocks_per_rank`, `batch_size_per_rank`, `long_used_per_rank`
- 负载指标：`kvcache_imbalance_ratio`, `batch_size_cv`, `memory_pressure`
- 决策结果：`decision.sp_size`, `decision.dispatch_tokens`

### 阶段 2: 离线分析 Trace 数据

使用分析脚本从 Trace 数据推导最优超参数：

```bash
python analyze_trace.py trace_data.jsonl --output optimal_hyperparams.json
```

**分析策略：**

1. **`learned_long_req_threshold`** (长请求阈值)
   - 识别导致 `batch_size_cv > 0.3` 的"痛点时刻"
   - 分析这些时刻的请求特征
   - 推导阈值乘数，用于区分"问题长请求"和正常请求
   - 公式：`is_long_request = prompt_length > learned_long_req_threshold * avg_prompt_length`

2. **`learned_imbalance_threshold`** (失衡阈值)
   - 识别"持续性失衡"：高 `kvcache_imbalance_ratio` 且高 `batch_size_cv`
   - 这些时刻表明 DP 路由无法解决的持续问题
   - 推导阈值，用于触发 Path B (Attention Balance)
   - 公式：`should_trigger_sp = (current_imbalance > threshold * 1.5) && (avg_imbalance > threshold)`

3. **统计信息**
   - 从 Trace 中提取 `avg_short_prompt_length`, `avg_short_output_length`, `avg_short_batch_size`
   - 这些用于 Path A (Workload-Aware Reservation) 的内存预留计算

### 阶段 3: 使用优化后的超参数（4DP8SP）

在生产环境中加载优化后的超参数：

```bash
python bench_serving.py \
    --dp 4 --sp 8 \
    --sp-size-mode load_aware \
    --load-hyperparams optimal_hyperparams.json \  # 加载离线学习的超参数
    --num-requests 100000 \
    --request-rate 20
```

## 分析脚本输出示例

```
============================================================
Analyzing Trace Data for Hyperparameter Tuning
============================================================
Loaded 10000 trace samples from trace_data.jsonl

[1] Analyzing Long Request Threshold...
  Average prompt length: 936.72
  Pain points (batch_size_cv > 0.3): 1247/10000
  Average prompt length at pain points: 2847.35
  Derived threshold multiplier: 3.037

[2] Analyzing Imbalance Threshold...
  Persistent imbalance points: 892/10000
  Derived imbalance threshold: 1.623

[3] Extracting Statistics...
  Average prompt length: 936.72
  Average short prompt length: 517.18
  Average short batch size: 121.84

============================================================
Derived Hyperparameters:
============================================================
  learned_long_req_threshold: 3.037
  learned_imbalance_threshold: 1.623
  avg_short_prompt_length: 517.18
  avg_short_output_length: 257.97
  avg_short_batch_size: 121.84
  avg_prompt_length: 936.72
  avg_output_length: 262.06
  sp_decision_count: 0
  running_benefit_ratio: 0.5
============================================================

✓ Hyperparameters saved to: optimal_hyperparams.json
```

## 超参数说明

### `learned_long_req_threshold` (默认: 3.0)
- **含义**: 用于判断请求是否为"长请求"的乘数
- **公式**: `is_long_request = prompt_length > learned_long_req_threshold * avg_prompt_length`
- **影响**: 
  - 值越小 → 更多请求被识别为"长请求" → 更激进的内存预留
  - 值越大 → 更少请求被识别为"长请求" → 更宽松的内存预留
- **范围**: [1.5, 5.0]

### `learned_imbalance_threshold` (默认: 1.5)
- **含义**: KVCache 失衡比率阈值，用于触发 Path B (Attention Balance)
- **公式**: `should_trigger_sp = (current_imbalance > threshold * 1.5) && (avg_imbalance > threshold)`
- **影响**:
  - 值越小 → 更容易触发 SP 来平衡 Attention 计算
  - 值越大 → 更保守，只在严重失衡时触发 SP
- **范围**: [1.1, 2.5]

## 验证和调优

1. **检查 Trace 数据质量**
   ```bash
   # 查看前几条记录
   head -n 3 trace_data.jsonl | python -m json.tool
   
   # 统计行数（应该等于请求数）
   wc -l trace_data.jsonl
   ```

2. **分析结果合理性**
   - `learned_long_req_threshold` 应该在 [2.0, 4.0] 范围内
   - `learned_imbalance_threshold` 应该在 [1.2, 2.0] 范围内
   - 如果值异常，可能需要：
     - 收集更多 Trace 数据
     - 调整分析脚本中的阈值参数

3. **生产环境验证**
   - 使用优化后的超参数运行一段时间
   - 监控 `batch_size_cv` 和 `kvcache_imbalance_ratio`
   - 如果性能不理想，可以手动微调超参数

## 高级用法

### 自定义分析参数

可以修改 `analyze_trace.py` 中的参数：

```python
# 调整"痛点"识别阈值
pain_threshold_cv = 0.3  # batch_size_cv 阈值

# 调整"持续性失衡"识别阈值
persistent_imbalance_threshold_cv = 0.25  # batch_size_cv 阈值
```

### 批量分析多个 Trace 文件

```bash
for trace_file in trace_*.jsonl; do
    python analyze_trace.py "$trace_file" --output "hyperparams_${trace_file%.jsonl}.json"
done
```

## 故障排查

1. **Trace 文件为空或格式错误**
   - 检查 `--export-hyperparams` 参数是否以 `.jsonl` 结尾
   - 确认程序正常运行并记录了 Trace

2. **分析结果异常**
   - 检查 Trace 数据是否包含足够的样本（建议 > 1000）
   - 确认 Trace 数据覆盖了不同的负载场景

3. **超参数加载失败**
   - 检查 JSON 文件格式是否正确
   - 确认所有必需字段都存在

## 参考

- SP 调度策略详细说明：见代码注释
- Path A (Workload-Aware Reservation): `sp_size_policy.cpp`
- Path B (Attention Balance): `sp_size_policy.cpp`
