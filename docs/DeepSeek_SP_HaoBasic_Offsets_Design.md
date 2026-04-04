# DeepSeek SP 下 all2all `offsets` 设计与 DLSlime `hao_basic` 对齐方案

## 1. 目标与边界

1. 先确认当前 DeepSeek-SP 路径的真实调用图，明确 `offsets` 在 MLA 与 GQA 里的生效范围。
2. 给 `hao_basic` 后端给出一套可以落地的 `offsets` 接口设计，不改动业务逻辑，仅调整底层通信语义与适配层。
3. 设计中以现有 Nano + DLSlime 代码位置为依据，尽量保持 `attention.py` 上层调用接口不变。

## 2. DeepSeek SP 推理路径里 `offsets` 的真实状态

1. DeepSeek-V2 当前模型在 SP decode 下使用 MLA 分支。
2. 这一点可在模型构造时看到 `attention_type="MLA"`，在注意力层里实际走 `FlashMLAImpl`。
3. `FlashMLAImpl` 的 SP decode 分支在 `q/res/lse` all2all 时只传 `mask` 与 `is_transpose`，没有传 `offsets`，例如 `q_buffer.all_to_all_ll(..., mask=..., is_transpose=False)` 和 `res/lse` 的 `is_transpose=True`。
4. `q_offsets` 在 MLA 下虽然在上下文里存在，但只作为字段透传，不参与 MLA 注意力计算；当前代码在 MLA 里没有读取它。

参考代码位置：
1. [`nanodeploy/models/deepseek_v2.py#L577`](./../../NanoDeploy-April/nanodeploy/models/deepseek_v2.py#L577)
2. [`nanodeploy/layers/attention.py#L201`](./../../NanoDeploy-April/nanodeploy/layers/attention.py#L201)
3. [`nanodeploy/layers/attention.py#L261`](./../../NanoDeploy-April/nanodeploy/layers/attention.py#L261)

结论：**如果目标是 DeepSeek 现网 MLA 推理，`offsets` 本身不在主干推理路径生效**；但必须在实现层保持接口可兼容，以支持未来 GQA 或其他 `attention_type=GQA` 场景。

## 3. GQA 场景里应复用的 offsets 语义（参考现有实现）

1. Nano 已有 GQA 版本 `q_offsets` 的构造逻辑。
2. 在 C++ 侧 `prepare_decode_cpp` 中，先统计每个 SP source 的有效请求数 `sp_valid_request_counts`，再做前缀和得到 `q_offsets`，长度 `sp_size + 1`。
3. `q_offsets` 的语义是：`q_offsets[i]` 到 `q_offsets[i+1]` 是 source rank `i` 的打包片段。
4. GQA 调用处在 `FlashAttentionImpl` 的 `q = q_buffer.all_to_all_ll(..., offsets=context.q_offsets)`，`offsets` 直接用于把不同 source 的有效段写到统一输出视图中，保持目标 rank 侧可聚合。
5. 这个语义与 DLSlime 的旧 intra-ll `all_to_all_intra_ll` 测试保持一致（固定目标是 `is_transpose` / `mask` 组合下按 source 段打包）。

参考代码位置：
1. [`csrc/nanodeploy/worker/model_runner_utils.cpp#L217`](./../../NanoDeploy-April/csrc/nanodeploy/worker/model_runner_utils.cpp#L217)
2. [`csrc/nanodeploy/worker/model_runner_utils.h#L55`](./../../NanoDeploy-April/csrc/nanodeploy/worker/model_runner_utils.h#L55)
3. [`nanodeploy/worker/model_runner.py#L428`](./../../NanoDeploy-April/nanodeploy/worker/model_runner.py#L428)
4. [`nanodeploy/layers/attention.py#L85`](./../../NanoDeploy-April/nanodeploy/layers/attention.py#L85)
5. [`tests/test_sp_attention_cudagraph.py#L249`](./../../NanoDeploy-April/tests/test_sp_attention_cudagraph.py#L249)

## 4. 当前 `hao_basic` 的限制与不一致

1. Nano 侧 `HaoAllToAllBufferAdapter` 在 `offsets` 非空时直接 `NotImplementedError`。
2. `Hao` 后端目前调用的是 `AllToAllBuffer::all_to_all(...)`，但该接口签名不支持 `offsets`。
3. DLSlime 的 `alltoall_buffer.cpp` 当前实现走 `intranode_alltoall` 内核；该内核签名里没有 `offsets`。
4. 同一个仓库里另一路旧实现 `AllToAllIntraLLBuffer` 已经支持 `offsets`，并且有完整的 `test_intra_all_to_all_offsets.py` 覆盖。

参考代码位置：
1. [`nanodeploy/worker/sp_backend.py#L173`](./../../NanoDeploy-April/nanodeploy/worker/sp_backend.py#L173)
2. [`csrc/dlslime/ops/intra_ll/all_to_all/alltoall_buffer.h#L33`](./../../DLSlime/csrc/dlslime/ops/intra_ll/all_to_all/alltoall_buffer.h#L33)
3. [`csrc/dlslime/ops/intra_ll/all_to_all/alltoall_buffer.cpp#L169`](./../../DLSlime/csrc/dlslime/ops/intra_ll/all_to_all/alltoall_buffer.cpp#L169)
4. [`tests/python/test_intra_all_to_all_offsets.py#L75`](./../../DLSlime/tests/python/test_intra_all_to_all_offsets.py#L75)

## 5. `hao_basic` 增加 `offsets` 的推荐落地设计

### 5.1 总体原则

1. 保持 Nano 上层接口不变：仍然调用 `buffer.all_to_all_ll(..., is_transpose=?, mask=?, offsets=?)`。
2. 尽量在 DLSlime 层把 `offsets` 语义补齐，这样 `HaoAllToAllBufferAdapter` 不再长期做 mask/布局重排。
3. 优先复用现有 `all_to_all_intra_ll` 的 offsets 分支，避免再引入新 kernel 行为。

### 5.2 DLSlime 侧实现步骤

1. 在 `csrc/dlslime/ops/intra_ll/all_to_all/alltoall_buffer.h` / `.cpp` 扩展 `all_to_all` 与 `dispatch_basic` 的签名，新增 `offsets: c10::optional<torch::Tensor>`，并传入 `dispatch_basic`。
2. 修改 `AllToAllBuffer::dispatch_basic`：
   1. 当 `offsets` 为 `nullopt` 时保持当前 `intranode_alltoall` 路径。
   2. 当 `offsets` 有值时改走 `all_to_all_intra_ll(...)`，沿用同一 `is_transpose`、`mask`、`offsets` 语义。
3. 引入/使用 `all_to_all_intra_ll.h` 中声明，复用已经验证过的 `all_to_all_intra_ll` 调用约定。
4. 加强参数校验：
   1. `offsets` dtype 必须 `int32`。
   2. `offsets` 形状必须是 `world_size + 1`。
   3. `offsets` 单调不减，且 `offsets[0]==0`。
   4. `total_messages = offsets[world_size]` 必须 `<= world_size * max_batch_size_`。
   5. 优先要求 `offsets` 在 CUDA 上；若是 CPU，内部拷贝到 CUDA。
5. 在 Python binding 中同步新增参数，改为 `all_to_all(..., offsets=none())`，并更新文档字符串。
6. 仅在 offsets 分支中保留返回形状 `[world_size, max_batch_size, msg]` 的既有约定。

### 5.3 Nano 侧适配步骤

1. 修改 `nanodeploy/worker/sp_backend.py`：
   1. `HaoAllToAllBufferAdapter.all_to_all_ll(...)` 不再在 `offsets` 非空时直接报错。
   2. 当 `_native_local_buffer` 存在（真实 `hao_basic` 路径）时将 `offsets` 原样透传给 `_buffer.all_to_all(...)`。
   3. `_compat_mode` 分支可以保留用于历史兼容，但明确标注：不保证 offsets 性能优于原生路径，建议逐步移除。
2. 保留 `_patch_self_slice`、`_pad_masked_non_transpose_input` 等兼容逻辑作为保底，不在 offsets 热路径中引入额外张量重排。

参考代码位置：
1. [`csrc/dlslime/ops/intra_ll/all_to_all/all_to_all_intra_ll.h#L10`](./../../DLSlime/csrc/dlslime/ops/intra_ll/all_to_all/all_to_all_intra_ll.h#L10)
2. [`csrc/dlslime/ops/intra_ll/all_to_all/alltoall_buffer.cpp#L155`](./../../DLSlime/csrc/dlslime/ops/intra_ll/all_to_all/alltoall_buffer.cpp#L155)
3. [`csrc/dlslime/ops/intra_ll/all_to_all/all_to_all_intra_ll.cu`](./../../DLSlime/csrc/dlslime/ops/intra_ll/all_to_all/all_to_all_intra_ll.cu)
4. [`csrc/python/bind.cpp#L306`](./../../DLSlime/csrc/python/bind.cpp#L306)
5. [`nanodeploy/worker/sp_backend.py#L173`](./../../NanoDeploy-April/nanodeploy/worker/sp_backend.py#L173)

## 6. DeepSeek-SP 的 `offsets` 设计建议（给开发直接用）

1. 当运行 DeepSeek MLA 推理时，`offsets` 直接保持可选参数，不参与 MLA 通信路径，`q_offsets` 可以保持空值或默认值由上层框架传入。
2. 当切到 GQA 流水线时，`offsets` 只给 Q 路径（非 transpose）使用，仍保持长度 `sp_size + 1` 的前缀和。
3. `offsets` 值必须可验证地对应 `sp_size` 个 source 的待发送条数，否则 `all_to_all` 的 packed 区间会重叠导致覆盖。
4. `mask` 仍保持 `shape = [sp_size, max_bs]` 的 target-major（`mask[target_rank, slot]`）定义，不要在 `HaoAllToAllBufferAdapter` 热路径再做转置 copy。

## 7. 验收要点

1. 在 GQA 单测中验证 `offsets` 下的 non-transpose/masked 与 transpose/masked 都通过，优先复用现有 `tests/python/test_intra_all_to_all_offsets.py`。
2. 在 Nano `sp_backend` 单测中新增用例：`HaoAllToAllBufferAdapter` 传入 `offsets` 时应进入 _buffer 的 offsets 透传路径，不再抛异常。
3. DeepSeek MLA 正常推理必须保持 bitwise 一致性；`offsets` 改造不得改变 MLA 现有结果和 latency 主路径行为。
