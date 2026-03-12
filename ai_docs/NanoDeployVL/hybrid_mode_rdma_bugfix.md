# Hybrid 模式 PeerAgent 未启动导致 RDMA Vision Embed 传输失败

> 日期: 2026-03-11 | 关联设计: [ep_separation_design.md](ep_separation_design.md)

## 问题现象

在 EP 分离 VL 集成测试中，Encoder 端编码正常，但 LLM 端 prefill 输出结果完全错误（模型看不到图像内容）。

**错误日志关键行**：

```
# NanoCtrl 返回 422
(LLMComponent) HTTP error registering engine with NanoCtrl: Client error '422 Unprocessable Entity'
(LLMComponent) Response: Failed to deserialize the JSON body into the target type: peer_addrs[0]: invalid type: null, expected a string

# LLM worker 无法 RDMA 拉取 vision embeds
(ModelRunner) PeerAgent not available, cannot RDMA-fetch vision embeds

# Prefill 走了无 vision embed 分支
(ModelRunner) [RUN_MODEL] Prefill WITHOUT vision embeds, input_ids.shape=torch.Size([1218])
```

## 根因分析

```
cache.py: start_peer_agent()
│
├── if mode == "hybrid":   ◄── 直接 return，跳过 PeerAgent 启动
│       return
│
└── 后续: 创建 PeerAgent、注册 KV cache MR、注册 GDN MR
```

调用链：

```
ModelRunner.setup()
  └── cache_context.start_peer_agent(mode="hybrid")
        └── 直接 return（PeerAgent 未启动）

LLMComponent.__init__()
  └── _register_with_nanoctrl()
        └── get_peer_agent_addrs()  →  [None]   ◄── worker 返回 None
              └── POST /register_engine  peer_addrs=[null]
                    └── NanoCtrl: 422 (null 不是 string)

ModelRunner.run_from_bytes()
  └── _fetch_vision_embeds_rdma(vision_slots)
        └── peer_agent is None  →  WARNING + return
              └── Prefill WITHOUT vision embeds  →  错误输出
```

**核心矛盾**：原设计假定 hybrid 模式不需要 P2P（因为 KV cache 不做跨节点传输），但 EP 分离引入 RDMA 后，hybrid 模式的 LLM worker 也需要 PeerAgent 来从 Encoder 拉取 vision embeddings。

## 修复方案

**文件**: `NanoDeploy/nanodeploy/context/cache.py` — `start_peer_agent()`

将 `if mode == "hybrid": return` 从方法入口移到 KV cache MR 注册之前：

```python
def start_peer_agent(self, mode: str = "hybrid"):
    # 不再在入口处 return
    if self.nanoctrl_address is None or self.engine_id is None:
        return

    # ... 创建 PeerAgent（所有模式都需要） ...
    self._peer_agent = start_peer_agent_fn(...)
    self._peer_agent_addr = agent_alias

    # hybrid 模式只需 PeerAgent（用于 RDMA 拉取 vision embeds），
    # 不注册 KV cache / GDN MR
    if mode == "hybrid":
        logger.info(f"PeerAgent started (hybrid, no KV MR): ...")
        return

    # prefill / decode 模式继续注册 KV cache MR、GDN MR ...
```

**影响**：

| 行为              | 修复前   | 修复后                              |
| ----------------- | -------- | ----------------------------------- |
| PeerAgent 启动    | 跳过     | 正常启动                            |
| `peer_addrs`      | `[None]` | `["engine_id:0"]`                   |
| NanoCtrl 注册     | 422      | 200 OK                              |
| RDMA 连接建立     | 无       | `Link Established: LLM <-> Encoder` |
| Vision embed 注入 | 跳过     | 成功注入 1200 tokens                |
| 推理结果          | 错误     | 正确                                |

## 修复后验证日志

```
# PeerAgent 正常启动（hybrid, 不注册 KV MR）
(ModelRunner) PeerAgent started (hybrid, no KV MR): alias=4f156869-...:0

# NanoCtrl 注册成功
(LLMComponent) peer_addrs: ['4f156869-c639-492f-a83a-577fd9063ee9:0']
(LLMComponent) Successfully registered engine 4f156869-... with NanoCtrl

# RDMA 链路建立
Link Established: 4f156869-...:0 <-> e64c3a5e-...:0

# Vision embeds 成功 RDMA 拉取并注入
(ModelRunner) [VISION_RDMA] Stored vision embeds: shape=torch.Size([1200, 2048]),
              dtype=torch.bfloat16, norm=153.0000, nonzero=2457600/2457600
(ModelRunner) [VISION_INJECT] Successfully injected 1200 image tokens

# Encoder slot 通过 P2P 正常释放
Received FreeVisionSlots from 4f156869-...: slots=[0]
Freed slots [0], pool free=8/8
```

**性能数据**（修复后，Qwen3.5-35B-A3B，单卡 H200）：

| 指标                   | 值         |
| ---------------------- | ---------- |
| Prefill 吞吐           | 1181 tok/s |
| Decode 吞吐            | 44 tok/s   |
| Decode ITL             | ~22.5 ms   |
| Scheduler 开销         | ~2.1 ms    |
| 生成 128 tokens 总耗时 | ~3 s       |

## 涉及文件

| 文件                                           | 改动                                                |
| ---------------------------------------------- | --------------------------------------------------- |
| `NanoDeploy/nanodeploy/context/cache.py`       | `start_peer_agent()` — 移动 hybrid 检查到 MR 注册前 |
| `NanoDeploy/nanodeploy/worker/model_runner.py` | 更新注释（hybrid 模式不再跳过 PeerAgent）           |
