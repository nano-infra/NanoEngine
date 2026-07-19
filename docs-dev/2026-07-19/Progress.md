# 任务进度罗盘

更新时间：2026-07-19 13:32 UTC

目标：实现
`docs-dev/2026-07-18/loongserve_source_aligned_decode_baseline_plan_20260718.md`
定义的 LoongServe source-aligned Decode-only baseline；先完成单机 CPU 验证，GPU
操作必须先提权。

完整的实施状态、工作树避让项和恢复顺序见同目录
`loongserve_source_aligned_decode_baseline_progress_20260719.md`。压缩恢复后必须先读
这两个文件，再查看 `git status`/`git diff`，不得重做已完成工作。

最新检查点：13:32 UTC 学术 demo 范围的实现与单机 CPU 验收已完成，正在做显式分组暂存/提交；
未运行 GPU。两次按规则提权的非 GPU `pip install -v -e .` 均完成 C++ 全量重编译。最终安装后
联合套件在 `CUDA_VISIBLE_DEVICES='' PYTHONMALLOC=debug` 下为 `221 passed in 34.49s`，独立
Sequence proxy 通过，新增 KV plan 跨 Scheduler 生命周期回归普通模式 `4 passed`、ASan/UBSan
`4 passed`。三条终审（combined transaction、Sequence ABI、low-KV）均确认无剩余 correctness blocker。
联合回归期间发现并修复：OFFLOAD 空 context 遗留 `master_sp_idx_=0`、RESERVED guard 继承上一 step
publication flag、typed fatal 未继承 Python `RuntimeError`、DISPATCHED KV plan 晚于 Scheduler 析构
造成 prepared mutation UAF，以及一个漏算 cadence dummy block 的测试容量。此前 stable native heap
corruption 现已消失，full low-KV/P2P 为 `43 passed` 且进程正常退出。

13:15 UTC 自动压缩恢复后已按协议完整重读两份罗盘。目标仍为完成学术 demo 范围的
LoongServe source-aligned Decode baseline，并先完成单机 CPU 验收；未运行 GPU。生产级 exact
telemetry 已完整裁剪：`ExactPlanKey`、reservation fingerprint、attempt/duplicate 字段及 Python/binding/
fake/test 路径均已删除，跨 admission-overlay 的 prepared cache 也已彻底删除。C++ schedule 热路径又
删除了入口 stable canonical-owner 全量 census 和 publication 后的重复全量复核；最终每-DP composition
validate、global publication barrier、pool-local rollback、RAII commit/abort 与 typed fatal 仍保留。
该裁剪由子任务完成后通过 scheduler C++20 syntax、`git diff --check` 和 source-built CPU transaction
harness `17 passed`。主线程刚完成 Python action/KV postcommit 热路径的同类学术范围精简，正在重新
取得 fake-engine subset 结果并做统一源代码验证；installed extension 仍过期，随后需提权执行非 GPU
editable reinstall，再跑列明的单机 CPU suite。

12:51 UTC 用户将目标进一步明确为“学术 demo”，不需要生产级审计/telemetry。主线程已停止新增
combined fingerprint、initial adapter count/pre-solve dedupe、waiting age、统一事件、fatal state hash和
两套stable Decode fault hook；只保留跨admission-overlay prepared cache删除、rollback前RAII owner
reset/abort、pool-local optional admission rollback与global required-Decode publication barrier等
correctness核心。
12:44 UTC 已向用户报告剩余任务与完成距离，并写回压缩恢复点；单机功能实现约
80%，但因最新 C++ 尚未统一重装，验收完成度约 65%。12:31 UTC 自动压缩恢复后已按协议完整
重读两份罗盘。
语义边界进一步校正为 LoongServe-style 的“post-mandatory base + per-DP admission overlay”：
某个 DP 的 admission overlay 或其后 required-Decode component 构建失败时，只撤销该 DP 的
admission，并从相同 post-mandatory base 为该 DP 重建 decode-only；其他 DP 已有效 admission
继续保留。所有稳定 required Decode component 仍须全局 validate 后统一 publication；真正的
required Decode 内部失败保持 typed fatal/global-zero-publication。这既保持 LoongServe 独立 pool
决策，又满足 NanoDeploy 共享 FFN collective 的原子执行边界。transaction safety 子任务正在
落地该关键修复及 2-DP post-admission failure 回归。

Sequence raw/pickle BlockContext 共用校验、token tail、高水位恢复已补强；public legacy
`ScheduleResult.ls_initial_*` ABI 已删除；low-KV frozen-plan/worker census/epoch getter 已合入；
resource epoch 只在显式 preempt/manual low-KV 新边界强制递增。用户已纠正 Python runtime 检查
边界：固定 pybind ABI 不在每步热路径用 `hasattr/getattr/type/shape` 反射式扫描；这些 schema、
DP/SP shape、master Counter与exact telemetry结构改由 C++/binding/unit tests验证。Python只保留
action互斥、admission owner/running等真正跨 publication 的动态关系，以及统一 post-publication
fail-close。对应 fake negative tests已删除，纯 CPU engine fake path为 `22 passed, 2 deselected`。
最近一次统一安装后的 targeted suite 仍为 `126 passed in 15.50s`；其后 C++ 改动尚未重新安装，
installed extension 已过期。未运行 GPU。transaction 的 post-admission per-DP rollback/retry 已终审
收紧：只有显式 pool-local TENTATIVE 可局部退回 decode-only，其余 unexpected failure 均为
INTERNAL/global-zero-publication。exact agent 正在实现 combined admission+iteration fingerprint、
overlay-safe key/cache 与真实 adapter attempt count；Sequence agent 已完成 raw/pickle ownership、
optimized raw、高水位及 legacy/version gate，待主线程终审。Issue-1% 六个 fixed-length fixtures 经
只读审计确认当前为零覆盖，required Decode stable-component fault hook/global-zero typed-fatal test
同样缺失。下一步先合并终审这些 C++/测试改动，再做 syntax/harness、提权执行非 GPU
`pip install -v -e .`，随后只跑单机 CPU tests。固定ABI反射清理已扩展到LS ingress/fatal/plan/
epoch路径，fake engine仍为`22 passed, 2 deselected`；KV consolidation同类清理正在进行。六个
Issue-1% authoritative window及DP4 RR/FIFO/no-reroute测试已新增，仍缺真实SP8 long-blocker/short-
OOE frontier运行时场景。required Decode stable prepare/global-validate fault hook仍待实现。
