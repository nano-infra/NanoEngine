# Docker 安装与配置指南

## 1. Docker 安装

### Ubuntu/Debian 系统

```bash
# 更新软件包索引
sudo apt-get update

# 安装必要的依赖
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# 添加 Docker 官方 GPG 密钥
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# 设置 Docker 仓库
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 安装 Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 启动 Docker 服务
sudo systemctl start docker
sudo systemctl enable docker

# 将当前用户添加到 docker 组（避免每次使用 sudo）
sudo usermod -aG docker $USER
# 注意：需要重新登录或执行 newgrp docker 使组权限生效
```

### 验证安装

```bash
docker --version
docker run hello-world
```

## 2. Docker Compose 安装

### 方法一：使用 Docker 官方插件（推荐）

Docker Compose 已经作为插件包含在 Docker Engine 中，安装 Docker 时会自动安装。

```bash
# 验证安装
docker compose version
```

### 方法二：独立安装 Docker Compose

如果使用旧版本的 Docker，可以单独安装：

```bash
# 下载最新版本的 Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

# 添加执行权限
sudo chmod +x /usr/local/bin/docker-compose

# 验证安装
docker-compose --version
```

## 3. Docker 代理配置

### 3.1 为 Docker 守护进程配置代理（拉取镜像时使用）

创建或编辑 `/etc/docker/daemon.json`：

```bash
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "proxies": {
    "http-proxy": "http://proxy.example.com:8080",
    "https-proxy": "http://proxy.example.com:8080",
    "no-proxy": "localhost,127.0.0.1,docker-registry.example.com,.corp"
  }
}
EOF
```

**注意**：将 `proxy.example.com:8080` 替换为你的实际代理地址。

### 3.2 为 Docker 服务配置系统代理

创建 systemd 服务覆盖目录：

```bash
sudo mkdir -p /etc/systemd/system/docker.service.d
```

创建代理配置文件 `/etc/systemd/system/docker.service.d/http-proxy.conf`：

```bash
sudo tee /etc/systemd/system/docker.service.d/http-proxy.conf > /dev/null <<EOF
[Service]
Environment="HTTP_PROXY=http://127.0.0.1:17897"
Environment="HTTPS_PROXY=http://127.0.0.1:17897"
Environment="NO_PROXY=localhost,127.0.0.1,docker-registry.example.com,.corp"
EOF
```

### 3.3 重启 Docker 服务使配置生效

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

### 3.4 验证代理配置

```bash
# 检查 Docker 服务环境变量
sudo systemctl show --property=Environment docker

# 测试拉取镜像
docker pull hello-world
```

## 4. Docker Compose 代理配置

### 4.1 在 docker-compose.yml 中配置代理

在 `docker-compose.yml` 文件中为服务添加环境变量：

```yaml
services:
  redis:
    image: redis:7-alpine
    environment:
      - HTTP_PROXY=http://proxy.example.com:8080
      - HTTPS_PROXY=http://proxy.example.com:8080
      - NO_PROXY=localhost,127.0.0.1
    # ... 其他配置
```

### 4.2 使用 .env 文件配置代理

创建 `.env` 文件（与 docker-compose.yml 同目录）：

```bash
HTTP_PROXY=http://proxy.example.com:8080
HTTPS_PROXY=http://proxy.example.com:8080
NO_PROXY=localhost,127.0.0.1
```

在 `docker-compose.yml` 中引用：

```yaml
services:
  redis:
    image: redis:7-alpine
    environment:
      - HTTP_PROXY=${HTTP_PROXY}
      - HTTPS_PROXY=${HTTPS_PROXY}
      - NO_PROXY=${NO_PROXY}
```

### 4.3 在构建时使用代理

如果需要构建镜像时使用代理，在 `docker-compose.yml` 中配置：

```yaml
services:
  app:
    build:
      context: .
      args:
        - HTTP_PROXY=http://proxy.example.com:8080
        - HTTPS_PROXY=http://proxy.example.com:8080
```

## 5. 常用代理配置示例

### 5.1 国内镜像加速（推荐）

如果在中国大陆，可以使用镜像加速器，编辑 `/etc/docker/daemon.json`：

```json
{
  "registry-mirrors": [
    "https://docker.mirrors.ustc.edu.cn",
    "https://hub-mirror.c.163.com",
    "https://mirror.baidubce.com"
  ]
}
```

然后重启 Docker：

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

### 5.2 企业内网代理示例

```json
{
  "proxies": {
    "http-proxy": "http://proxy.company.com:3128",
    "https-proxy": "http://proxy.company.com:3128",
    "no-proxy": "localhost,127.0.0.1,*.company.local,10.0.0.0/8"
  }
}
```

## 6. 故障排查

### 检查 Docker 日志

```bash
sudo journalctl -u docker.service
```

### 测试代理连接

```bash
# 测试 HTTP 代理
curl -x http://proxy.example.com:8080 http://www.google.com

# 测试 HTTPS 代理
curl -x http://proxy.example.com:8080 https://www.google.com
```

### 清除代理配置

如果需要清除代理配置：

```bash
# 删除代理配置文件
sudo rm /etc/systemd/system/docker.service.d/http-proxy.conf
sudo systemctl daemon-reload
sudo systemctl restart docker
```

## 7. 使用说明

启动 Redis 和 RedisInsight：

```bash
# 进入 docker 目录
cd docker

# 启动服务
docker compose up -d

# 查看服务状态
docker compose ps

# 查看日志
docker compose logs -f

# 停止服务
docker compose down

# 停止并删除数据卷
docker compose down -v
```

访问：

- Redis: `localhost:6379`
- RedisInsight: `http://localhost:8001`
