# NanoOps 设计文档

NanoOps 是 NanoInfra 的运维编排层，以 **session** 为单位管理 route / prefill / decode 组件的部署与生命周期。

## 架构

```
用户 / 脚本
    │
    ▼
┌────────────────────────────────────────────────────┐
│  nanoctrl CLI  (Typer + Rich)                      │
│  create | attach | set | deploy | status | stop    │
└────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────┐
│  SessionOrchestrator                               │
└────────────────────────────────────────────────────┘
    │
    ├──► RedisClient          会话与组件元数据
    ├──► RayJobManager        作业与 placement group
    ├──► NanoCtrlClient       控制面 HTTP / 可选拉起
    └──► HealthChecker        Ray + NanoCtrl + 端口
```

## 模块

| 文件                 | 职责                                             |
| -------------------- | ------------------------------------------------ |
| `cli.py`             | 命令解析、session shell（attach）、Rich 输出     |
| `orchestrator.py`    | 会话生命周期、配置生成、Ray 提交                 |
| `redis_client.py`    | Session / Component / PG 的 CRUD                 |
| `ray_manager.py`     | Placement group、Job 提交 / 状态 / 日志 / 停止   |
| `nanoctrl_client.py` | 健康检查、ensure_running、list_engines           |
| `health_checker.py`  | 等待就绪：Ray RUNNING + NanoCtrl 注册 + 端口     |
| `config.py`          | NanoOpsConfig、from_file / from_env、load_config |
| `utils.py`           | session_id 校验、端口分配、二进制查找            |
| `exceptions.py`      | NanoOpsError 及子类                              |
| `cleanup.py`         | 清理 Redis key、杀进程                           |

## 会话生命周期

1. **create** — 校验 session_id → 可选启动 NanoCtrl → Redis 创建 session config
2. **set** — 写入 model_path（并行度在 deploy 时指定）
3. **deploy** — 生成组件配置 → 创建 PG（仅 engine）→ 提交 Ray Job → Redis 注册组件
4. **status** — 读 Redis + Ray 状态 + NanoCtrl 引擎数 → 展示表格
5. **stop** — 停 Ray Job → 删 PG → 可选清理 Redis key

同一 session 可多次 deploy 同类型组件，每个对应独立 Ray Job。

## Redis Key 设计

| Key 模式                                  | 类型 | 用途                                             |
| ----------------------------------------- | ---- | ------------------------------------------------ |
| `{session_id}:session:config`             | Hash | session 元数据（status, model_path, 连接信息等） |
| `{session_id}:components:{type}:{job_id}` | Hash | 组件注册（ray_job_id, port, status 等）          |
| `{session_id}:placement_groups:{pg_id}`   | Hash | PG 信息（component_type, num_gpus, strategy）    |

> NanoCtrl 的 `{scope}:engine:*` 键由 NanoCtrl / 引擎维护，NanoOps 仅通过 HTTP API 查询。

## 配置优先级

```
CLI 参数  >  环境变量  >  session 已保存值  >  配置文件  >  默认值
```

| 环境变量             | 默认值                   | 说明          |
| -------------------- | ------------------------ | ------------- |
| `NANOCTRL_REDIS_URL` | `redis://localhost:6379` | Redis         |
| `RAY_ADDRESS`        | `http://localhost:8265`  | Ray Dashboard |
| `NANOCTRL_ADDRESS`   | `http://localhost:3000`  | NanoCtrl HTTP |

配置文件为 `nanoops.toml`（或 `--config` 指定），格式与 `NanoOpsConfig` 字段一致。

## 组件部署细节

### Route

- 生成 TOML 配置（server / tokenizer / engine scope=session_id / scheduler）
- 提交 Ray Job，entrypoint 为 NanoRoute 二进制

### Prefill / Decode

- GPU 数 = `attention_tp × attention_sp × attention_dp`
- 创建 STRICT_PACK placement group（每 bundle 1 GPU）
- 生成 YAML 配置（model、并行度、nanoctrl_address、master_address 等）
- 提交 Ray Job，entrypoint 为 `python -m nanodeploy.server.engine_server`
- 环境变量注入 `NANOCTRL_SCOPE=session_id`

## Session Shell (attach)

`nanoctrl attach <id>` 启动子 shell，注入环境变量（`NANOCTRL_SESSION`, `NANOCTRL_SCOPE`, 连接信息），后续命令自动使用该 session。

- **Zsh**：通过临时 ZDOTDIR 保留原有主题，precmd hook 添加 `(nanoctrl/<id>)` 前缀
- **Bash**：通过 `--rcfile` 注入 PS1 前缀
- `exit` / Ctrl-D 退出

## 健康检查

对每个已注册组件轮询（默认 timeout 300s）：

1. Ray Job 状态 = RUNNING（FAILED/STOPPED 视为永久失败）
2. Engine：NanoCtrl `list_engines(scope, role)` 数量 > 0
3. Route：TCP 端口可连接

## 端口分配

基于 `hash(session_id + component_type) % 1000` 的确定性偏移：

| 组件    | 基础端口 | 范围      |
| ------- | -------- | --------- |
| route   | 3001     | 3001–4000 |
| prefill | 5000     | 5000–5999 |
| decode  | 6000     | 6000–6999 |

## 异常体系

| 异常                   | 场景                         |
| ---------------------- | ---------------------------- |
| `SessionExistsError`   | create 时 session 已存在     |
| `SessionNotFoundError` | 操作不存在的 session         |
| `ConfigError`          | deploy engine 前未 set model |
| `NanoCtrlError`        | NanoCtrl 不可用或启动失败    |
| `RayJobError`          | PG 创建或 Job 提交失败       |
| `HealthCheckTimeout`   | wait_for_ready 超时          |

## NanoRoute 二进制查找

`ray_manager.py` 按以下顺序定位 `nanoroute`：

1. 系统 PATH（`which nanoroute`）
2. 环境变量 `NANOCTRL_NANOROUTE_PATH`

均未找到则报错并提示构建。
