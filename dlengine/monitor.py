"""Grafana/Prometheus monitoring stack generation for DLEngine."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Sequence

DEFAULT_OUTPUT_DIR = "monitoring"
DEFAULT_DLENGINE_TARGET = "host.docker.internal:5000"
PROMETHEUS_PORT = 9090
GRAFANA_PORT = 3000


def default_dlengine_target(port: int) -> str:
    return f"host.docker.internal:{port}"


def _prometheus_config(dlengine_target: str, scrape_interval: str) -> str:
    return f"""global:
  scrape_interval: {scrape_interval}

scrape_configs:
  - job_name: dlengine
    metrics_path: /metrics
    static_configs:
      - targets:
          - {dlengine_target}
"""


def _datasource_config() -> str:
    return """apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
"""


def _dashboard_provider_config() -> str:
    return """apiVersion: 1

providers:
  - name: DLEngine
    orgId: 1
    folder: DLEngine
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
"""


def _panel(
    panel_id: int,
    title: str,
    expr: str,
    x: int,
    y: int,
    w: int,
    h: int,
    unit: str = "short",
    panel_type: str = "timeseries",
) -> dict:
    return {
        "id": panel_id,
        "type": panel_type,
        "title": title,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "fieldConfig": {
            "defaults": {"unit": unit},
            "overrides": [],
        },
        "targets": [
            {
                "datasource": {"type": "prometheus", "uid": "Prometheus"},
                "expr": expr,
                "legendFormat": "{{phase}}{{state}}{{type}}{{direction}}{{dp}}",
                "refId": "A",
            }
        ],
        "options": {
            "legend": {"displayMode": "list", "placement": "bottom"},
            "tooltip": {"mode": "multi", "sort": "none"},
        },
    }


def _panel_multi(
    panel_id: int,
    title: str,
    targets: list[tuple[str, str]],
    x: int,
    y: int,
    w: int,
    h: int,
    unit: str = "short",
    defaults_extra: dict | None = None,
) -> dict:
    """A timeseries panel with explicit (expr, legend) targets."""
    defaults = {"unit": unit}
    if defaults_extra:
        defaults.update(defaults_extra)
    return {
        "id": panel_id,
        "type": "timeseries",
        "title": title,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "targets": [
            {
                "datasource": {"type": "prometheus", "uid": "Prometheus"},
                "expr": expr,
                "legendFormat": legend,
                "refId": chr(ord("A") + i),
            }
            for i, (expr, legend) in enumerate(targets)
        ],
        "options": {
            "legend": {"displayMode": "list", "placement": "bottom"},
            "tooltip": {"mode": "multi", "sort": "none"},
        },
    }


def _dashboard_config() -> dict:
    return {
        "uid": "dlengine-overview",
        "title": "DLEngine Overview",
        "tags": ["dlengine"],
        "timezone": "browser",
        "schemaVersion": 39,
        "version": 1,
        "refresh": "5s",
        "time": {"from": "now-15m", "to": "now"},
        "panels": [
            _panel(
                1,
                "Requests",
                "dlengine_requests",
                0,
                0,
                12,
                8,
            ),
            _panel(
                2,
                "Throughput",
                "dlengine_throughput_tokens_per_second",
                12,
                0,
                12,
                8,
                "tps",
            ),
            _panel(
                3,
                "Tokens",
                "dlengine_tokens",
                0,
                8,
                8,
                8,
            ),
            _panel(
                4,
                "KV Blocks",
                "dlengine_kv_blocks",
                8,
                8,
                8,
                8,
            ),
            _panel(
                5,
                "Step Latency",
                "dlengine_step_latency_ms",
                16,
                8,
                8,
                8,
                "ms",
            ),
            _panel(
                6,
                "Forward Transfer",
                "dlengine_forward_bytes",
                0,
                16,
                12,
                8,
                "bytes",
            ),
            _panel(
                7,
                "Transfer Latency",
                "dlengine_transfer_latency_ms",
                12,
                16,
                12,
                8,
                "ms",
            ),
            _panel_multi(
                8,
                "Prefix Cache Hit Rate",
                [
                    ("dlengine_prefix_cache_hit_rate_per_dp", "dp={{dp}}"),
                    ("dlengine_prefix_cache_hit_rate", "overall"),
                ],
                0,
                24,
                8,
                8,
                "percentunit",
                {"min": 0, "max": 1},
            ),
            _panel_multi(
                9,
                "TTFT",
                [
                    (
                        "histogram_quantile(0.5, rate(dlengine_ttft_seconds_bucket[5m]))",
                        "p50",
                    ),
                    (
                        "histogram_quantile(0.9, rate(dlengine_ttft_seconds_bucket[5m]))",
                        "p90",
                    ),
                    (
                        "histogram_quantile(0.99, rate(dlengine_ttft_seconds_bucket[5m]))",
                        "p99",
                    ),
                    (
                        "rate(dlengine_ttft_seconds_sum[5m]) / "
                        "clamp_min(rate(dlengine_ttft_seconds_count[5m]), 1e-9)",
                        "avg",
                    ),
                ],
                8,
                24,
                8,
                8,
                "s",
            ),
            _panel_multi(
                10,
                "TPOT",
                [
                    (
                        "histogram_quantile(0.5, rate(dlengine_tpot_seconds_bucket[5m]))",
                        "p50",
                    ),
                    (
                        "histogram_quantile(0.9, rate(dlengine_tpot_seconds_bucket[5m]))",
                        "p90",
                    ),
                    (
                        "histogram_quantile(0.99, rate(dlengine_tpot_seconds_bucket[5m]))",
                        "p99",
                    ),
                    (
                        "rate(dlengine_tpot_seconds_sum[5m]) / "
                        "clamp_min(rate(dlengine_tpot_seconds_count[5m]), 1e-9)",
                        "avg",
                    ),
                ],
                16,
                24,
                8,
                8,
                "s",
            ),
        ],
    }


def _compose_config() -> str:
    return f"""services:
  prometheus:
    image: prom/prometheus:latest
    command:
      - --config.file=/etc/prometheus/prometheus.yml
      - --storage.tsdb.path=/prometheus
    ports:
      - "{PROMETHEUS_PORT}:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    extra_hosts:
      - "host.docker.internal:host-gateway"

  grafana:
    image: grafana/grafana-oss:latest
    ports:
      - "{GRAFANA_PORT}:3000"
    environment:
      GF_SECURITY_ADMIN_USER: admin
      GF_SECURITY_ADMIN_PASSWORD: admin
      GF_USERS_ALLOW_SIGN_UP: "false"
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
      - ./grafana/dashboards:/var/lib/grafana/dashboards:ro
    depends_on:
      - prometheus
"""


def create_monitor_stack(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    dlengine_target: str = DEFAULT_DLENGINE_TARGET,
    scrape_interval: str = "5s",
) -> Path:
    """Create Grafana and Prometheus config files for a DLEngine dashboard."""
    root = Path(output_dir)
    prometheus_dir = root / "prometheus"
    grafana_provisioning = root / "grafana" / "provisioning"
    grafana_dashboards = root / "grafana" / "dashboards"

    prometheus_dir.mkdir(parents=True, exist_ok=True)
    (grafana_provisioning / "datasources").mkdir(parents=True, exist_ok=True)
    (grafana_provisioning / "dashboards").mkdir(parents=True, exist_ok=True)
    grafana_dashboards.mkdir(parents=True, exist_ok=True)

    (root / "docker-compose.yml").write_text(_compose_config(), encoding="utf-8")
    (prometheus_dir / "prometheus.yml").write_text(
        _prometheus_config(dlengine_target, scrape_interval),
        encoding="utf-8",
    )
    (grafana_provisioning / "datasources" / "prometheus.yml").write_text(
        _datasource_config(),
        encoding="utf-8",
    )
    (grafana_provisioning / "dashboards" / "dlengine.yml").write_text(
        _dashboard_provider_config(),
        encoding="utf-8",
    )
    (grafana_dashboards / "dlengine-overview.json").write_text(
        json.dumps(_dashboard_config(), indent=2),
        encoding="utf-8",
    )
    return root


def start_monitor_stack(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    dlengine_target: str = DEFAULT_DLENGINE_TARGET,
    scrape_interval: str = "5s",
) -> Path:
    """Create the stack files and start Prometheus/Grafana with docker compose."""
    root = create_monitor_stack(
        output_dir,
        dlengine_target=dlengine_target,
        scrape_interval=scrape_interval,
    )
    subprocess.run(["docker", "compose", "up", "-d"], cwd=root, check=True)
    return root


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Grafana/Prometheus monitoring stack for DLEngine."
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated config (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--dlengine-target",
        default=DEFAULT_DLENGINE_TARGET,
        help=(
            "Prometheus scrape target for dlengine serve "
            f"(default: {DEFAULT_DLENGINE_TARGET})"
        ),
    )
    parser.add_argument(
        "--scrape-interval",
        default="5s",
        help="Prometheus scrape interval (default: 5s)",
    )
    parser.add_argument(
        "--up",
        action="store_true",
        help="Run docker compose up -d after generating config.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.up:
        root = start_monitor_stack(
            args.output_dir,
            dlengine_target=args.dlengine_target,
            scrape_interval=args.scrape_interval,
        )
    else:
        root = create_monitor_stack(
            args.output_dir,
            dlengine_target=args.dlengine_target,
            scrape_interval=args.scrape_interval,
        )
    print(f"monitoring config written to {root}")
    print(f"prometheus: http://localhost:{PROMETHEUS_PORT}")
    print(f"grafana:    http://localhost:{GRAFANA_PORT} (admin/admin)")


if __name__ == "__main__":
    main()
