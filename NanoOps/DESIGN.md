# NanoOps 设计文档

NanoOps 是 NanoInfra 分布式 LLM 推理的操作层 CLI 与编排逻辑，负责会话生命周期、组件部署、状态管理以及与 Redis / Ray / NanoCtrl 的集成。

______________________________________________________________________

## 1. 概述

### 1.1 目标

- **会话隔离**：以 session 为单位管理一组 route / prefill / decode 组件，支持多租户或多环境并存。
- **云原生配置**：连接信息（Redis、Ray、NanoCtrl）优先从环境变量读取，便于 K8s/容器部署。
- **一键编排**：通过 `nanoctrl` CLI 完成 create → set model → deploy 组件 → status 的完整流程。
- **可观测**：提供 status、job logs、job status 等运维能力。

### 1.2 与 NanoInfra 其他组件的关系

| 组件                    | 职责                                                    | NanoOps 的交互                                                                 |
| ----------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------ |
| **Redis**               | 会话元数据、组件注册、NanoCtrl 的 engine/agent 发现     | 读写 session 配置、component 注册、placement group 记录                        |
| **NanoCtrl** (Rust)     | 控制面：engine/agent 注册、Redis 键与 Pub/Sub、HTTP API | 可选由 NanoOps 拉起；NanoOps 通过 HTTP 查询 list_engines 等                    |
| **Ray**                 | 作业调度与资源管理                                      | 提交 route/prefill/decode 作业、创建 STRICT_PACK placement group、查状态与日志 |
| **NanoRoute** (Rust)    | 推理入口：HTTP API、调度、与 engine 通信                | 由 NanoOps 以 Ray Job 形式部署，配置由 NanoOps 生成                            |
| **NanoDeploy** (Python) | Prefill/Decode 引擎                                     | 由 NanoOps 以 Ray Job + placement group 形式部署，配置由 NanoOps 生成          |

NanoOps 不实现推理逻辑，只做「会话 + 部署 + 状态」的编排与 CLI。

______________________________________________________________________

## 2. 架构

### 2.1 整体数据流

```
用户 / 脚本
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  nanoctrl CLI (Typer + Rich)                                     │
│  create | attach | set | deploy | status | stop | list | job    │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  SessionOrchestrator                                             │
│  start_session / set_model_config / spawn_component /           │
│  wait_for_ready / stop_session                                  │
└─────────────────────────────────────────────────────────────────┘
    │
    ├──► RedisClient          (会话与组件元数据)
    ├──► RayJobManager        (作业与 placement group)
    ├──► NanoCtrlClient       (控制面 HTTP、可选拉起 NanoCtrl)
    └──► HealthChecker        (Ray 状态 + NanoCtrl 注册 + 端口)
```

### 2.2 模块划分

| 模块         | 文件                 | 职责                                                                                  |
| ------------ | -------------------- | ------------------------------------------------------------------------------------- |
| **CLI**      | `cli.py`             | 命令解析、会话解析（`--session-id` / `NANOCTRL_SESSION`）、Rich 输出、attach 子 shell |
| **编排**     | `orchestrator.py`    | 会话生命周期、模型配置、组件生成配置与 Ray 提交、placement group 创建                 |
| **Redis**    | `redis_client.py`    | Session CRUD、组件注册、placement group 记录、key 命名规范                            |
| **Ray**      | `ray_manager.py`     | Placement group（STRICT_PACK）、Job 提交/状态/日志/停止                               |
| **NanoCtrl** | `nanoctrl_client.py` | 健康检查、ensure_running（可选启动）、list_engines、生成 config                       |
| **健康**     | `health_checker.py`  | 等待 session 就绪：Ray RUNNING + NanoCtrl 注册 + route 端口监听                       |
| **配置**     | `config.py`          | NanoOpsConfig、from_file / from_env、load_config 优先级                               |
| **工具**     | `utils.py`           | session_id 校验、端口分配、端口检测、二进制查找                                       |
| **异常**     | `exceptions.py`      | NanoOpsError 及子类（Session/Config/Ray/NanoCtrl/Health 等）                          |
| **清理**     | `cleanup.py`         | 清理 Redis key、杀进程（engine_server、nanoroute 等）                                 |

______________________________________________________________________

## 3. 会话模型 (Session)

### 3.1 会话生命周期

1. **create**：校验 session_id、检查不存在、可选启动 NanoCtrl、在 Redis 创建 session 元数据。
2. **set**：写入 model_path（及可选 session 级参数），不包含并行度（并行度在 deploy 时按组件指定）。
3. **deploy**：按组件类型（route / prefill / decode）生成配置、创建 PG（仅 engine）、提交 Ray Job、在 Redis 注册组件。
4. **status**：展示 session 与各组件状态、NanoCtrl 注册数、route 端点。
5. **stop**：停止所有相关 Ray Job、删除 placement group、可选清理 Redis 中该 session 所有 key。

同一 session_id 下可多次 deploy 同类型组件（例如多个 decode），每个对应一个 Ray Job，用 `ray_job_id` 区分。

### 3.2 Redis Key 设计

- **Session 配置**

  - Key: `{session_id}:session:config`
  - 类型: Hash
  - 内容: session_id, status, redis_url, ray_address, nanoctrl_address, model_path, created_at, 等

- **组件注册**

  - Key: `{session_id}:components:{component_type}:{component_id}`
  - `component_id` 一般为 Ray Job ID
  - 类型: Hash
  - 内容: ray_job_id, placement_group_id, config_path, host, port, status, spawned_at

- **Placement Group**

  - Key: `{session_id}:placement_groups:{pg_id}`
  - 类型: Hash
  - 内容: component_type, num_gpus, strategy, created_at

NanoCtrl 使用的 key（如 `{scope}:engine:*`）由 NanoCtrl 与引擎侧维护；NanoOps 只通过 NanoCtrl HTTP API 查询引擎列表，不直接写这些 key。

### 3.3 会话与连接配置的解析顺序

- **session_id**：`--session-id` > 环境变量 `NANOCTRL_SESSION`，缺一不可则报错。
- **连接配置**（redis_url, ray_address, nanoctrl_address）：
  **CLI 显式参数 > 环境变量 > 已存在 session 中保存的值 > 配置文件 > 默认值**
  若提供 `session_id` 且未显式传连接参数，会尝试从该 session 的 Redis 配置中补全未设置的项（环境变量仍优先）。

______________________________________________________________________

## 4. 配置 (Configuration)

### 4.1 NanoOpsConfig 字段

- **连接**：redis_url, ray_address, nanoctrl_address
- **NanoCtrl**：nanoctrl_auto_start, nanoctrl_binary_path, nanoctrl_config_path
- **Ray**：code_mode, working_dir
- **引擎默认值**：default_kvcache_blocks, default_kvcache_block_size, default_max_model_len, default_max_num_batched_tokens

### 4.2 加载优先级 (load_config)

1. 默认值
2. 配置文件：`--config` 指定文件或当前目录 `nanoops.toml`
3. 环境变量：仅对「已设置」的变量写进 config，便于 exclude_unset 与云原生覆盖

环境变量名：`NANOCTRL_REDIS_URL`, `RAY_ADDRESS`, `NANOCTRL_ADDRESS`。

______________________________________________________________________

## 5. CLI 设计

### 5.1 命令一览

| 命令                                                                | 说明                                                       |
| ------------------------------------------------------------------- | ---------------------------------------------------------- |
| `nanoctrl create --session-id <id>`                                 | 创建会话并可选启动 NanoCtrl；输出提示 attach 与 set/deploy |
| `nanoctrl attach <session-id>`                                      | 进入带 session 环境的交互 shell，后续命令自动带 session    |
| `nanoctrl detach`                                                   | 说明如何退出 session shell（exit / Ctrl-D）                |
| `nanoctrl current`                                                  | 显示当前 attach 的 session 与相关环境变量                  |
| `nanoctrl set --model <path>`                                       | 设置 session 的模型路径（并行度在 deploy 时指定）          |
| `nanoctrl deploy route \| prefill \| decode [--attention-tp ...]`   | 部署单个组件，engine 支持并行度参数                        |
| `nanoctrl status`                                                   | 会话与组件状态表、NanoCtrl 注册数、route 端点              |
| `nanoctrl stop [--no-cleanup]`                                      | 停止 session 所有 Ray 作业与 PG，可选清理 Redis            |
| `nanoctrl list [--all]`                                             | 列出会话（默认排除已停止）                                 |
| `nanoctrl job stop <job_id>`                                        | 停止 Ray 作业并从 session 的 Redis 组件表移除              |
| `nanoctrl job rm <job_id>`                                          | 仅从 Redis 移除组件记录，不停止作业                        |
| `nanoctrl job logs <job_id> [--tail N]`                             | 查看 Ray 作业日志                                          |
| `nanoctrl job status <job_id>`                                      | 查看 Ray 作业状态                                          |
| `nanoctrl cleanup [--redis-url] [--sessions] [--no-kill-processes]` | 清理陈旧进程与 Redis key                                   |

### 5.2 会话 Shell (attach)

- **目的**：一次 attach 后，后续 `set` / `deploy` / `status` 等不再需要写 `--session-id`。
- **实现**：子进程继承环境，并注入 `NANOCTRL_SESSION`、`NANOCTRL_SCOPE`、`NANOCTRL_REDIS_URL`、`NANOCTRL_ADDRESS`、`RAY_ADDRESS`。
- **Shell 检测**：优先从 `/proc/<ppid>/exe` 识别 zsh/bash/fish，否则用 `$SHELL`。
- **Zsh**：通过临时 `ZDOTDIR` 与 `.zshenv`/`.zshrc` 保留原有主题与插件，并用 `add-zsh-hook precmd` 在 PS1 前加 `(nanoctrl/<session_id>)`。
- **Bash**：通过 `--rcfile` 注入带 session 前缀的 PS1。
- **detach**：在 shell 中执行 `exit` 或 Ctrl-D，不提供单独 detach 子命令。

### 5.3 全局选项

- `--config <file>`：指定 NanoOps 配置文件。
- `--session-id / -s`：指定 session（与 attach 后的 `NANOCTRL_SESSION` 二选一生效）。

______________________________________________________________________

## 6. 编排器 (SessionOrchestrator)

### 6.1 初始化

- 入参：redis_url, ray_address, nanoctrl_address。
- 内部构造：RedisClient、RayJobManager、NanoCtrlClient、HealthChecker。
- 将 ray_address（Dashboard HTTP）转换为 GCS 地址（如 `ray://host:7078`），供引擎或配置使用。

### 6.2 start_session(session_id, ensure_nanoctrl=True)

- 校验 session_id 格式（字母数字、横线、下划线）。
- 若 session 已存在则抛 SessionExistsError。
- 若 ensure_nanoctrl 为 True，调用 NanoCtrlClient.ensure_running（不健康则拉起，并传入 NANOCTRL_REDIS_URL）。
- 在 Redis 创建 session config hash，写入 status=initializing、连接信息、可选 nanoctrl_pid。

### 6.3 set_model_config(session_id, model_path, \*\*kwargs)

- 要求 session 存在。
- 更新 Redis 中 session config 的 model_path 及 kwargs（如 max_model_len）。
- 不写入并行度；并行度在 spawn_component 时通过 overrides 传入。

### 6.4 spawn_component(session_id, component_type, \*\*overrides)

- **前置**：session 存在；对 prefill/decode 要求已 set model_path。
- **GPU 数**：仅对 prefill/decode，由 `_calculate_num_gpus(overrides)` 得到：
  `attention_tp * attention_sp * attention_dp`（与 NanoDeploy 的 attention 并行度一致）。
- **Placement Group**：仅对 prefill/decode 创建 STRICT_PACK PG，bundle 为单 GPU，并登记到 Redis。
- **配置生成**：
  - **route**：`_generate_route_config` → TOML，含 server/tokenizer/engine（mode=Disaggregated, scope=session_id）/scheduler。
  - **prefill/decode**：`_generate_engine_config` → YAML，含 model、mode、attention_tp/sp/dp、ffn_tp/dp/ep、host、port、nanoctrl_address、ray_address、master_address、loop_count、max_num_batched_tokens、max_model_len、kvcache 等。
- **环境变量**：NANOCTRL_SCOPE=session_id、NANOCTRL_REDIS_URL、SESSION_ID、NANOCTRL_ADDRESS；若有 PG 则 NANOOPS_PLACEMENT_GROUP_ID。
- **Ray 提交**：
  - route：entrypoint 为 NanoRoute 二进制 + `--config <path>`。
  - prefill/decode：entrypoint 为 `python -m nanodeploy.server.engine_server --config <path> --log_level INFO`，runtime_env 带 placement_group_id 与 env_vars。
- **Redis**：注册组件，component_id = ray_job_id，记录 ray_job_id、placement_group_id、config_path、host、port、status=spawning。

### 6.5 wait_for_ready(session_id, timeout=300)

- 委托 HealthChecker.wait_for_session_ready：轮询所有已注册组件，直到 Ray 状态为 RUNNING、prefill/decode 在 NanoCtrl 可见、route 端口监听，或超时。

### 6.6 stop_session(session_id, cleanup=True)

- 停止该 session 下所有已登记 Ray Job。
- 删除该 session 下所有 placement group。
- 若 cleanup 为 True，删除该 session 在 Redis 上的所有 key（`{session_id}:*`）。

______________________________________________________________________

## 7. 后端客户端

### 7.1 RedisClient

- **Session**：session_exists, create_session, get_session_config, update_session_config。
- **Component**：register_component, get_component_info, get_session_components, remove_component（按 component_id 删除，用于 job stop 后从 session 摘除）。
- **Placement Group**：register_placement_group, get_placement_groups。
- **批量**：cleanup_session（按前缀删 key）、list_sessions（支持排除 stopped）。
- Hash 值序列化：dict/list → JSON 字符串，bool → "0"/"1"，反序列化时按类型解析。

### 7.2 RayJobManager

- **地址**：支持 `http://...`、`ray://...` 或 host:port，统一为 Dashboard HTTP 地址用于 JobSubmissionClient。
- **Placement Group**：placement_group(..., strategy="STRICT_PACK")，bundle 为 `{"CPU": 0.1, "GPU": 1.0}` × num_gpus，ray.get(pg.ready())，返回 pg.id.hex()。
- **作业**：submit_job(entrypoint, runtime_env, job_id=...)；get_job_status；get_job_logs；stop_job。
- **PG 清理**：PlacementGroup.from_hex + remove_placement_group。
- 创建 PG 前若未初始化 Ray，会临时 unset RAY_ADDRESS 再 ray.init(address="auto")，避免 Dashboard URL 被误用于 init。

### 7.3 NanoCtrlClient

- **健康**：POST /get_redis_address，2xx 视为健康。
- **ensure_running**：不健康则 \_start_nanoctrl（查找二进制、生成临时 config、传入 NANOCTRL_REDIS_URL、Popen 后轮询健康至多 30s）。
- **list_engines**：POST /list_engines，body 含 `scope: session_id`，用于 status 与健康检查。
- **配置生成**：临时 TOML，server host/port + redis url。
- 不根据 NanoCtrl 返回的 redis_url 做强制 kill/restart；连接信息以 NANOCTRL_ADDRESS 为准，由调用方保证一致。

______________________________________________________________________

## 8. 健康检查 (HealthChecker)

- **输入**：session_id、timeout；从 Redis 取该 session 所有已注册组件。
- **单组件**：
  - Ray job 非 RUNNING 则未就绪；FAILED/STOPPED 视为永久失败不再重试。
  - prefill/decode：NanoCtrl list_engines(session_id, role=...) 数量 > 0。
  - route：从 Redis 取 host/port，检测 TCP 监听（0.0.0.0 视为 localhost）。
- **聚合**：轮询直到全部就绪或超时，带简单进度；就绪后返回 endpoints（如 route URL）、components、summary（engine 数量等）。

______________________________________________________________________

## 9. 部署流程小结

1. **create**：创建 session、可选启动 NanoCtrl、写 Redis session config。
2. **set**：写 model_path 到 session config。
3. **deploy route**：生成 route TOML（含 scope=session_id）、提交 Ray Job（无 PG）、注册组件。
4. **deploy prefill**：计算 GPU 数、创建 PG、生成 engine YAML、提交 Ray Job（带 PG 与 NANOCTRL_SCOPE 等）、注册组件。
5. **deploy decode**：同 prefill，仅 mode 与端口等不同。
6. **status**：读 Redis 组件列表、拉 Ray 状态、拉 NanoCtrl 引擎数、展示表与 route 端点。
7. **job stop**：停 Ray Job + 从 Redis 移除该 component 记录；**job rm** 仅移除记录。

______________________________________________________________________

## 10. 异常与错误处理

- **SessionExistsError**：create 时 session 已存在。
- **SessionNotFoundError**：对不存在的 session 做 set/deploy/status/stop 等。
- **ConfigError**：例如 deploy engine 前未 set model。
- **NanoCtrlError**：NanoCtrl 不可用或启动失败。
- **RayJobError**：PG 创建或 Job 提交失败。
- **HealthCheckTimeout**：wait_for_ready 超时。

CLI 层捕获后以 Rich 输出错误与提示并 typer.Exit(1)。

______________________________________________________________________

## 11. 端口与二进制

- **端口分配**（utils.allocate_port）：基于 session_id + component_type 的哈希，在 base（route 3001、prefill 5000、decode 6000）上加 0–999 的偏移，保证同 session 同类型端口稳定。
- **NanoRoute 二进制**：ray_manager 中写死路径（如 `NanoRoute/target/release/nanoroute`），不存在时提交会报错并提示 build。
- **NanoCtrl 二进制**：nanoctrl_client 在若干相对路径与 NANOINFRA_ROOT 下查找，找不到则报错并提示 build。

______________________________________________________________________

## 12. 文件与依赖

- **包**：typer、rich、pydantic、redis、httpx、ray、tqdm、pyyaml 等。
- **入口**：通过 setuptools 或类似将 `nanoctrl` 指向 `nanoops.cli:app()`。
- **设计文档**：本文档（DESIGN.md）；README.md 侧重安装与快速开始。

以上为 NanoOps 的详细设计说明，便于维护、扩展与对接 NanoInfra 其他组件。
