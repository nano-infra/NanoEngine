# DLEngine Compose Monitoring

`docker-compose.monitor.yml` starts DLEngine, Prometheus, and Grafana on the
same compose network.

- DLEngine listens on `8104`.
- Prometheus listens on `9090`.
- Grafana listens on `3000`.
- Prometheus scrapes DLEngine at `dlengine:8104/metrics`.
- Grafana provisions the DLEngine dashboard automatically.

Start the full stack:

```bash
cd DLEngine/docker
DLENGINE_MODEL=/path/to/model docker compose -f docker-compose.monitor.yml up -d
```

Restart the full stack after config changes:

```bash
cd DLEngine/docker
docker compose -f docker-compose.monitor.yml down
DLENGINE_MODEL=/path/to/model docker compose -f docker-compose.monitor.yml up -d
```

Open:

- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (`admin/admin`)

Extra serve flags can be passed with `DLENGINE_SERVE_ARGS`:

```bash
DLENGINE_MODEL=/path/to/model \
DLENGINE_SERVE_ARGS="--ray_address 127.0.0.1:6379" \
docker compose -f docker-compose.monitor.yml up -d
```
