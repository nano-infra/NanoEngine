# 动态服务发现功能更新日志

## 最新更新：简化配置，Redis URL 从 NanoCtrl 获取

### 变更内容

1. **NanoCtrl 端修改**

   - `get_redis_address` API 现在返回完整的 Redis URL（例如：`redis://127.0.0.1:6379`）
   - 移除了 host:port 格式的转换逻辑，直接返回配置的 Redis URL

2. **NanoRouter 端修改**

   - **只支持动态发现模式**，移除了所有静态配置的 fallback
   - **必须配置 `nanoctrl_address`**，不再需要配置 `redis_url`
   - 启动时自动从 NanoCtrl 获取 Redis URL
   - 如果无法从 NanoCtrl 获取 Redis URL，启动会失败（不再 fallback）

### 配置要求

**NanoRouter 的 `config.toml` 只需要配置：**

```toml
[engine]
mode = "Unified"
host = "127.0.0.1"  # 可选，仅用于兼容性
port = 5555          # 可选，仅用于兼容性
nanoctrl_address = "http://127.0.0.1:3000"  # 必须配置
# redis_url 不再需要配置，会自动从 NanoCtrl 获取
```

### 工作流程

1. **NanoRouter 启动时**：

   - 从配置读取 `nanoctrl_address`
   - 调用 `POST /get_redis_address` 从 NanoCtrl 获取 Redis URL
   - 使用获取的 Redis URL 启动动态服务发现
   - 如果任何步骤失败，启动会失败（不再 fallback）

2. **动态发现过程**：

   - 从 NanoCtrl API 或 Redis 加载初始 Engine 列表（Snapshot）
   - 订阅 Redis Pub/Sub 监听 Engine 变化
   - 自动连接新注册的 Engine
   - 自动断开已注销的 Engine

### 优势

1. **配置简化**：只需要配置 NanoCtrl 地址，Redis URL 自动获取
2. **统一管理**：Redis URL 在 NanoCtrl 中统一配置，避免多处配置不一致
3. **更可靠**：如果无法获取 Redis URL，启动失败，避免静默使用错误配置

### 迁移指南

**旧配置（需要移除 `redis_url`）：**

```toml
[engine]
mode = "Unified"
nanoctrl_address = "http://127.0.0.1:3000"
redis_url = "redis://127.0.0.1:6379"  # 移除此行
```

**新配置（只需要 `nanoctrl_address`）：**

```toml
[engine]
mode = "Unified"
nanoctrl_address = "http://127.0.0.1:3000"  # 仅此一项必需
```

### 错误处理

如果 NanoRouter 启动时无法从 NanoCtrl 获取 Redis URL，会输出错误并退出：

```
ERROR: Failed to get Redis URL from NanoCtrl: <error>
ERROR: nanoctrl_address must be configured in config.toml
```

**解决方案**：

1. 确保 NanoCtrl 正在运行
2. 确保 `nanoctrl_address` 配置正确
3. 检查网络连接
