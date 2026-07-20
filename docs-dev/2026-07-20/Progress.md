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

---

更新时间：2026-07-20 11:54 UTC

任务目标：修复双节点 DP2×SP8、`.85`、K=16 正式 workload 在约 21 秒后报
`request cannot fit the full empty SP pool exactly` 的新问题。

根因：原始 811259-token 长请求必须跨 rank。入口预检和正式 placement 都先按
原始 free-token 容量填满 master rank，之后才计算 bootstrap token 和
`reserved_blocks_per_req=1`，导致 master 精确 block 数超限；每个候选 DoP 都
重复填满第一个 rank，因此把总 pool 容量足够的请求错误判为永久不可调度。

修复：每个候选 DoP 在 prompt 打包前先扣除所有 master 的 reserved blocks 和
新增 master 的 bootstrap tokens，之后继续执行原有逐请求 block rounding 与
精确容量校验。入口 `_ls_batch_fits_empty_system` 和实际
`_plan_ls_initial_placement` 同步修改。

验证：新增 DP2×SP2、K=16 的纯 CPU 跨-rank回归，旧实现稳定失败、新实现通过；
失败 workload 的精确 811259/754 样本和首 7200 条最大 prompt 971548 均在 CPU
上成功选择 KV DoP=2。相关回归共 `201 passed`。双节点 DP2×SP8、`.85`、K=16
真实 GPU smoke 使用 811259-token prompt，`1/1` 请求完成、17 tokens 输出、
manifest `success`、exit 0。退出后 Ray 为 `0.0/16.0 GPU` 且本机无残留进程。
详细记录见 `loongserve_cross_rank_admission_fix_20260720.md`。

---

更新时间：2026-07-20 12:49 UTC

任务目标：持续监控修复后的 DP2×SP8、`.85`、`loop_count=16`、20 req/s
正式 workload，解释运行速度异常缓慢的原因；本阶段只做只读诊断，未停止任务或
修改代码。

最新阶段性状态：正式日志为
`ls_style_loop16_2node_dp2sp8_.85_r20_6min_20260720.log`。截至 12:47:57 UTC，
进度停在 `676/7200`、平均延迟约 57.45 秒，日志 mtime 为 12:44:43.748，已超过
3 分钟没有新增内容，且没有 traceback。Ray 仍完整占用 16/16 GPU、无 pending
demand；当前节点 8 张 GPU 连续采样均为 0% utilization、约 133–135 GiB 显存占用，
说明 worker/模型仍驻留，但长时间没有收到执行任务。

负载证据：按 seed=0 的开放环到达序列，125 秒约到达 2427 个请求、220 秒约到达
4364 个、308 秒约到达 6224 个，而 308 秒时只完成 676 个，积压已约 5548 个；
360 秒后 7200 个请求都会到达，脚本还需继续排空队列，所以配置中的“6 分钟”只是
注入持续时间，不是整体运行完成时间。

当前根因判断：主要瓶颈在 host 端 LoongServe admission/scheduling，而非 GPU kernel。
组合调度路径会在每一步遍历不断增长的 waiting 队列；每个候选又调用 future-KV
容量检查和完整 admission placement 规划，其中包含 running envelope 重建、排序、
donor/placement 扫描。`ls_max_num_ooe=8` 当前只决定是否允许 OOE，却没有把一次扫描
限制为跳过 8 个候选，因此在数千请求积压后出现近似 waiting×planning 的超线性开销。
日志已出现约 72 秒、约 90 秒的无 worker metric 间隔，随后演变为超过 3 分钟的
调度空窗，与 GPU 0% 利用率吻合。下一步继续采样确认是否恢复或已实质卡在单轮调度。

---

更新时间：2026-07-20 13:02 UTC

最终日志分析：用户于 12:53 UTC 手动关闭任务；manifest 随后记录
`status=failed`、`BrokenPipeError`。这是关闭 stdout/任务后的结果，不是原始性能故障。
正式运行没有出现新的 `IndexError`、`UnschedulableRequestError` 或 scheduler traceback，
最后完成 `680/7200`。

时间轴：full CUDA graph 初始化从 12:35:51 到 12:39:14，engine 到 12:39:28
才开始正式 workload，这部分约 3 分 49 秒是一次性启动成本。正式处理最初完成吞吐
约 4.6--4.9 req/s：100/200/300/400/500/600 个请求分别在 37/51/67/85/102/124
秒完成。随后进度从 664 开始塌缩，ModelRunner metric 的相邻间隔从稳定的 5--6 秒
变成 72、91、249 秒；249 秒空窗后仅从 676 推进到 680，下一轮又持续无下发直至
手动关闭。空窗期间 8 张本机 GPU 实测均为 0% utilization、模型显存仍驻留。

开放环积压：复现 seed=0 的 arrival sequence 后，145/220/308/356 秒累计到达
2812/4364/6224/7200 个请求，对应只完成 664/674/676/约 676；最后 559 秒时完成
680，有 6520 个 outstanding。DP2 且每池 running cap=1000，因此此时 waiting 至少
4520。所谓 6 分钟只是请求注入窗口，脚本会在注入结束后继续排空全部 7200 个请求。

代码根因：组合 admission 路径逐项遍历整个 `step->waiting[dp_idx]`；每个仍有机会的
候选会重复执行 future-KV 和 exact placement，其中 `_ls_pool_future_kv_fits` 每次又
重新收集 running、遍历完整 canonical waiting 统计 paused、分配哈希集合并排序 future
envelope。future 检查在外层、`make_admission_plan`、`_plan_ls_initial_placement` 内最多
重复三次。候选还会逐次复制 `selected_fifo`，每轮 admission 又复制完整 scheduler
shadow。`max_num_ooe=8` 的布尔语义与 LoongServe 上游一致，表示允许连续 8 轮 OOE，
不是“最多检查 8 个候选”，所以允许 OOE 时仍可全队列扫描；但 NanoDeploy 缺少上游在
running 已满时的立即返回，并且把上游单次 future policy 检查扩展成多次 exact planner，
使几千 waiting 时形成严重的超线性 host 开销。driver 在一次 `engine.step()` 中同步阻塞，
返回后 benchmark 又突发补交阻塞期间累计的 arrivals，形成正反馈。

归因结论：`loop_count=16` 和本日两项正确性修复不是此次分钟级停顿的直接原因；两项
修复没有改 admission queue loop，后者由 2026-07-19 的 `b8e0b7d3` 引入。K=16 反而
减少 scheduler 调用次数。下一步应先优化/缓存 admission fast policy、补 running-full
fast path、避免 per-candidate 完整 planner 和队列扫描，再按同一 workload 复跑；仅降
request rate 可绕开积压但不能验证正式 r20 场景。
