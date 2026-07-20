# 任务进度罗盘

更新时间：2026-07-20 03:06 UTC

任务目标：确认当前 LoongServe-style 调度版本能否使用 Ray 地址
`10.102.252.174:6380` 跑通单机测试。

当前状态：DP1×SP8、8×H200 的真实 Issue-1% CSV eager serving smoke 已 exit 0，
`8 / 8` 请求全部完成，manifest 为 `success`。功能路径可以运行；性能不达 100 ms
TPOT SLO。完整命令、指标、产物和退出后 GPU0 残留 context 风险见
`loongserve_single_node_ray6380_smoke_20260720.md`。

代码状态：本次没有修改 NanoDeploy 运行时代码，也没有修改用户既有的
`nanodeploy/engine/ray_executor.py` buffer override。待办仅为提交本次报告/manifest；
GPU0 的驱动/container context 不应在没有节点所有者授权时 reset。

---

更新时间：2026-07-20 03:44 UTC

任务目标：在当前 LoongServe-style 调度中安全支持 `loop_count=16`，使用
Ray `10.102.252.174:6380` 做单机真实 GPU 验证，并确认实际 DoP 与扩缩容行为。

当前实现：调度器会按本轮所有 RUNNING 请求的剩余 token 数计算实际执行块
`K=min(16, min_remaining_tokens)`，把 K 贯穿 KV 容量估计、iteration-master
规划/校验/事务提交、Ray executor、worker model runner 和 engine 后处理；因此尾轮
可自动从 16 降为 1，避免超生成。LoongServe-style 配置目前允许 `loop_count=1..16`，
且要求小于 KV block size。

验证状态：独立 C++ harness 7 项通过；两组相关 Python/pybind CPU 测试分别
99 与 101 项通过。正在执行 8×H200、8 请求、prompt=4096、max_tokens=34、
configured loop_count=16 的单节点 smoke，预期实际 decode chunk 为 `16,16,1`。

后续：读取 GPU smoke JSON/退出状态；补跑能触发 DoP 扩容的场景并核对缩容；
整理报告，只暂存本任务修改（保留用户原有 ray executor buffer override），提交代码。

---

更新时间：2026-07-20 03:53 UTC

完成状态：loop_count=16 实现与验证完成。Ray `10.102.252.174:6380`
单机 8×H200 的 8/64/128 请求测试全部 exit 0，三组实际 Decode chunk 均为
`16,16,1` 并精确完成。8/64 请求保持 master/KV DoP=1；128 请求发生
compute scale-up，首轮 master DoP=2、KV DoP=1，后续 KV DoP=2。由于请求
同长度同时结束，本 workload 没有 scale-down 窗口。

最终回归：两组测试分别 99、101 项通过，共 200 项；C++ harness 7/7 包含
K=16 跨 KV block 边界预留。详细报告见
`loongserve_loop_count16_single_node_validation_20260720.md`。提交范围已核对：
只包含本任务修改，用户原有 ray executor buffer override 与其他工作树改动未纳入。

---

更新时间：2026-07-20 07:50 UTC

任务目标：修复 LoongServe-style `loop_count=16` 在双节点 DP2×SP8
稀疏启动时首轮 Decode 报 `IndexError: Block index out of range` 的问题。

根因：首个请求只被分配到 DP0。组合调度器仅在某个 DP 自己存在真实 Decode
group 时才为该 DP 的空闲 SP rank 添加持久化 dummy，导致 DP1 的 8 个 rank
收到空 `dp_seqs`。worker 随后临时构造了没有 KV block table 的 dummy，读取
`last_block_page_id` 时越界。

修复：只要本轮任意 DP 存在真实 Decode，就为所有没有真实负载的 DP/SP rank
补充调度器拥有、已分配 KV block 的持久化 dummy，以保持全局同步的 FFN cadence；
纯 admission 步骤不额外创建 dummy。新增 DP2×SP8、K=16 的空闲 DP 回归测试，
验证 DP1 获得 8 个 KV-backed dummy。

验证：修改 C++ 后已执行 `pip install -v -e .`；相关 CPU 回归两组分别
100、101 项通过，共 201 项。Ray `10.102.243.60:8776` 上完成 DP2×SP8、
loop_count=16 的双机真实 GPU 稀疏启动 smoke：1/1 请求完成、进程 exit 0、
manifest 为 `success`，原来触发异常的全局 rank 8 路径已通过。

资源与文档状态：smoke 退出后 Ray 为 `0.0/16.0 GPU`、无 pending demand，
本机没有残留 CUDA 计算进程；详细记录见
`loongserve_loop16_idle_dp_fix_20260720.md`。本轮修复将随对应代码提交落盘。
