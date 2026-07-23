# LoongServe-style iteration 118 精准重放

时间：2026-07-23 09:49–09:58 UTC

## 目标

精准重放
`ls_style_loop16_dp2sp8_r20_diag01_3.log` 中 zero-based iteration 118
（2026-07-23 06:26:05）的 Decode 数据面：

- batch 1040，DP2 × SP8；
- loop count 16；
- master max batch 138；
- sequence DoP histogram：D1/D2/D3 = 574/347/119；
- max KV utilization 99.44%；
- 历史 model runner 1450.6053 ms，即 90.6628 ms/loop；
- 历史 step ITL 94.0128 ms/loop。

## 重建方法和精确性

新增 `scripts/bench_ls_decode_snapshot_replay.py`。它从 admission、
此前每轮的 sequence/master assignment 和目标轮的 group telemetry
重建 1040 条 active sequence 的 `committed_tokens_by_sp`。

目标轮只有两条请求初始就是 D2；它们的 prompt split 可由所属 group 的
per-rank token residual 唯一求出：

- seq 1500，prompt 923230：
  rank 1 / rank 6 = 174174 / 749056；
- seq 2359，prompt 747799：
  rank 1 / rank 7 = 463895 / 283904。

CPU 校验逐 group、逐 rank 对齐以下全部数据：

- committed KV tokens；
- committed KV blocks（逐 sequence 向上取整后求和）；
- master batch；
- KV participant DoP；
- loop16 的 pending input + output reservation blocks；
- 全局 batch 和 D1/D2/D3 histogram。

12/12 groups 全部 exact。随后按真实 12372 blocks/rank 做 C++
`SPStateManager` allocation，520+520 条请求成功分配并完整清理。

这里的“精准”是 scheduler-visible metadata exact。历史日志没有记录 sampled
token value 和物理 KV block ID，因此 replay 使用 token 0，并由新
`SPStateManager` 重新分配等量 block。这两个未记录维度不声称 bitwise exact。

## GPU 配置

- Ray：`10.102.206.14:7789`，两节点 16 × H200；
- DP2 × SP8，FFN EP16；
- GPU memory limit 141 GB，utilization 0.85；
- full CUDA graph；
- loop count 16；
- 3 rounds；每轮 1 warmup + 5 measured repeats；
- 正式 DLSlime metadata transport 和 `ModelRunner.run`；
- scheduler 不在 replay timed path；
- 同时记录 wall time 和 16-rank CUDA-event distributed critical path。

运行前后 Ray 均为 `0/16 GPU`。

## 结果

15 次正式测量：

| metric | median | mean | p95 | min–max |
|---|---:|---:|---:|---:|
| replay wall / loop | 90.1087 ms | 90.0786 ms | 91.0050 ms | 88.7316–91.4384 ms |
| GPU critical path / loop | 89.3284 ms | 89.3024 ms | 90.2158 ms | 88.0240–90.7432 ms |

逐轮 wall median：

- round 0：88.8074 ms；
- round 1：90.7425 ms；
- round 2：90.0971 ms。

与历史目标轮对比：

- 历史 model runner：90.6628 ms/loop；
- replay wall median：90.1087 ms/loop；
- 差值：-0.5541 ms/loop，-0.61%；
- 比值：0.9939×。

历史 step ITL 94.0128 ms/loop 与 model runner 的差为 3.34995 ms/loop；
乘以 loop16 恰好是 53.5992 ms，与该轮 `sch_ovhd=53.60ms` 对齐。也就是说，
该轮 ITL 中约 96.44% 是 model-runner 数据面，约 3.56% 是被 loop16 摊薄后的
scheduler；`post_sch_ovhd=1.90ms` 是另列的 postprocess，不包含在这条
`step_itl_ms` 中。

## 结论

这次 replay 在 0.61% 内复现了历史 model-runner 耗时。因而可以排除
“此前看到的约 90 ms 数据面只是日志计时误差或 scheduler 混入”的解释：
真实的 sequence KV placement、master assignment、receiver shape 和长
context 分布本身，足以重现这轮 LoongServe-style 数据面耗时。

它也解释了为什么固定 800-token 的静态 T11 只有 67.0009 ms：精准快照有
5,968,143 committed tokens，平均 5738.6、median 483.5、p99 72667、max
923790；synthetic T11 只有 832,000 tokens。精准快照 GPU critical path
比 synthetic T11 高 22.3275 ms（+33.32%）。这个差来自真实 context/KV
shape，而不是 scheduler，因为两个 harness 都绕过 scheduler。

但本实验只校准了 observed E11，不单独给出“CP 扩散”和“master skew”在真实
快照上的反事实边际。此前 800-token 2×2 已证明两者都有代价；若要把真实
90.1 ms 再严格拆分，下一步应在同一 snapshot 上构造 capacity-feasible 的：

1. owner-local master rebalance（保持每条 sequence 的 KV placement）；
2. 只折叠非必要的短请求 CP（保留两条容量上必须 D2 的超长请求）；
3. 二者同时。

不能简单把所有请求强制 D1：同一快照已有单请求 923790-token context，超过
单 rank KV token capacity，且当前 master skew 下部分 rank 也会超容量。
反事实必须显式保持 capacity feasibility，否则测到的是不可部署布局。

## 产物

- snapshot：
  `docs-dev/2026-07-23/ls_decode_iteration118_snapshot.json`
- replay result：
  `docs-dev/2026-07-23/ls_decode_iteration118_exact_replay.json`
- replay log：
  `docs-dev/2026-07-23/ls_decode_iteration118_exact_replay.log`
- harness：
  `scripts/bench_ls_decode_snapshot_replay.py`
- tests：
  `tests/test_ls_decode_snapshot_replay.py`
