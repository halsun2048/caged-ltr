#!/usr/bin/env bash
set -Eeuo pipefail

API_URL="${API_URL:-http://127.0.0.1:8001}"
UI_URL="${UI_URL:-http://127.0.0.1:8502}"
PROM_URL="${PROM_URL:-http://127.0.0.1:9090}"
GRAFANA_URL="${GRAFANA_URL:-http://127.0.0.1:3000}"

check() {
  local name="$1" url="$2"
  if curl -fsS --max-time 5 "$url" >/dev/null; then
    echo "[ok]   $name $url"
  else
    echo "[fail] $name $url"
    return 1
  fi
}

echo "[1/5] API health"
check api "$API_URL/health"
echo "[2/5] API functional smoke"
PYTHONPATH=src uv run --frozen python scripts/smoke_demo_http.py --base-url "$API_URL"
echo "[3/5] Streamlit"
check streamlit "$UI_URL/_stcore/health"
echo "[4/5] Prometheus"
check prometheus "$PROM_URL/-/ready"
echo "[5/5] Grafana"
check grafana "$GRAFANA_URL/api/health"
echo "[########################] full stack verification complete"
