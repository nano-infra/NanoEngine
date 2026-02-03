# Engine 动态发现功能使用指南

## 问题诊断

如果遇到 "Found 0 engines from NanoCtrl" 或 503 Service Unavailable 错误，请按以下步骤检查：

### 1. 检查配置

确保 `NanoRoute/config.toml` 中配置了 `redis_url` 以启用动态发现：

```toml
[engine]
mode = "Unified"
host = "127.0.0.1"
port = 5555
nanoctrl_address = "http://127.0.0.1:3000"
redis_url = "redis://127.0.0.1:6379"  # 必须配置此项
```

### 2. 检查服务状态

运行诊断脚本：

```bash
./test_engine_discovery.sh
```

### 3. 注册 Engine

如果 Redis 中没有 Engine，需要先注册：

```bash
curl -X POST http://localhost:3000/register_engine \
  -H "Content-Type: application/json" \
  -d '{
    "engine_id": "engine-1",
    "role": "prefill",
    "host": "127.0.0.1",
    "port": 5555,
    "world_size": 1,
    "num_blocks": 100,
    "peer_addrs": []
  }'
```

### 4. 验证注册

检查 Redis 中的数据：

```bash
# 查看所有 Engine
redis-cli KEYS "engine:*"

# 查看特定 Engine 信息
redis-cli HGETALL "engine:engine-1"

# 查看当前 revision
redis-cli GET "nano_meta:engine_revision"
```

### 5. 检查 NanoRouter 日志

启动 NanoRouter 时应该看到：

- 如果配置了 `redis_url`：`Starting dynamic service discovery with Redis: ...`
- 如果未配置：`Connecting to Engines (Static Config)...`

## 工作流程

### 动态发现模式（推荐）

1. **启动 Redis**

   ```bash
   redis-server
   ```

2. **启动 NanoCtrl**

   ```bash
   cd NanoCtrl
   cargo run
   ```

3. **注册 Engine**

   ```bash
   curl -X POST http://localhost:3000/register_engine ...
   ```

4. **启动 NanoRouter**（配置了 `redis_url`）

   ```bash
   cd NanoRoute
   cargo run -- --config config.toml
   ```

5. **NanoRouter 会自动**：

   - 从 NanoCtrl API 或 Redis 加载初始 Engine 列表（Snapshot）
   - 订阅 Redis Pub/Sub 监听 Engine 变化
   - 自动连接新注册的 Engine
   - 自动断开已注销的 Engine

### 静态配置模式

如果未配置 `redis_url`，NanoRouter 会：

- 从 `config.toml` 读取 Engine 列表
- 或从 NanoCtrl API 查询（如果配置了 `nanoctrl_address`）
- 不会监听动态变化

## 常见问题

### Q: 为什么显示 "Found 0 engines"？

A: 可能的原因：

1. NanoCtrl 中没有注册任何 Engine
2. NanoCtrl API 无法访问
3. Redis 中没有 Engine 数据

**解决方案**：

- 检查 NanoCtrl 是否运行：`curl http://localhost:3000/health`
- 检查 Redis 中的 Engine：`redis-cli KEYS "engine:*"`
- 注册一个 Engine（见上面的 curl 命令）

### Q: 为什么返回 503 Service Unavailable？

A: 因为 NanoRouter 没有可用的 Engine。

**解决方案**：

1. 确保至少有一个 Engine 已注册
2. 确保 Engine 进程正在运行
3. 检查 Engine 的连接信息（host, port）是否正确

### Q: 如何查看 Engine 是否已连接？

A: 查看 NanoRouter 启动日志：

```
Connected engines: X prefill, Y decode
```

如果都是 0，说明没有 Engine 可用。

### Q: 动态发现不工作怎么办？

A: 检查：

1. `config.toml` 中是否配置了 `redis_url`
2. Redis 是否运行
3. NanoRouter 日志中是否有错误信息
4. 如果动态发现失败，会自动回退到静态配置模式

## 测试命令

### 注册 Engine

```bash
curl -X POST http://localhost:3000/register_engine \
  -H "Content-Type: application/json" \
  -d '{
    "engine_id": "engine-1",
    "role": "prefill",
    "host": "127.0.0.1",
    "port": 5555,
    "world_size": 1,
    "num_blocks": 100,
    "peer_addrs": []
  }'
```

### 注销 Engine

```bash
curl -X POST http://localhost:3000/unregister_engine \
  -H "Content-Type: application/json" \
  -d '{
    "engine_id": "engine-1"
  }'
```

### 列出所有 Engine

```bash
curl -X POST http://localhost:3000/list_engines \
  -H "Content-Type: application/json" \
  -d '{}'
```

### 查看 Engine 信息

```bash
curl -X POST http://localhost:3000/get_engine_info \
  -H "Content-Type: application/json" \
  -d '{
    "engine_id": "engine-1"
  }'
```
