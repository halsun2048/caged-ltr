#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PROJECT="${COMPOSE_PROJECT_NAME:-caged-ltr}"
DOCKER=(docker)

if ! docker info >/dev/null 2>&1; then
  if sudo -n docker info >/dev/null 2>&1; then
    DOCKER=(sudo -n docker)
    echo "[info] 当前 shell 未刷新 docker 组，使用 sudo docker 继续"
  else
    echo "[error] 当前终端无法访问 Docker daemon。请执行："
    echo "        sg docker -c './scripts/run_full_stack.sh'"
    echo "或重新登录终端后再运行脚本。"
    exit 2
  fi
fi

COMPOSE=("${DOCKER[@]}" compose -p "$PROJECT" -f docker-compose.full.yml)

free_port() {
  local candidate="$1"
  while ss -ltn "sport = :$candidate" 2>/dev/null | grep -q LISTEN; do
    candidate=$((candidate + 1))
  done
  echo "$candidate"
}

export CAGED_API_PORT="${CAGED_API_PORT:-$(free_port 8001)}"
export CAGED_MCP_PORT="${CAGED_MCP_PORT:-$(free_port 8766)}"
export CAGED_UI_PORT="${CAGED_UI_PORT:-$(free_port 8502)}"
export PROMETHEUS_PORT="${PROMETHEUS_PORT:-$(free_port 9090)}"
export GRAFANA_PORT="${GRAFANA_PORT:-$(free_port 3000)}"
echo "端口: API=$CAGED_API_PORT MCP=$CAGED_MCP_PORT UI=$CAGED_UI_PORT Prometheus=$PROMETHEUS_PORT Grafana=$GRAFANA_PORT"

bar() {
  local step="$1" total="$2" label="$3"
  local width=24 filled=$(( step * 24 / total ))
  local hashes dashes
  hashes="$(printf '%*s' "$filled" '' | tr ' ' '#')"
  dashes="$(printf '%*s' "$((width-filled))" '' | tr ' ' '-')"
  printf '[%s%s] %s (%d/%d)\n' \
    "$hashes" "$dashes" \
    "$label" "$step" "$total"
}

wait_http() {
  local name="$1" url="$2" max="${3:-60}"
  for ((i=1; i<=max; i++)); do
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
      echo "[ready] $name $url"
      return 0
    fi
    printf '\r[wait]  %s %s (%d/%d)' "$name" "$url" "$i" "$max"
    sleep 2
  done
  echo
  echo "[warn] $name 未在限定时间内就绪；继续输出容器日志"
  return 0
}

echo "项目目录: $ROOT_DIR"
echo "Compose 项目: $PROJECT"
bar 1 7 "启动 PostgreSQL、Redis、Qdrant"
"${COMPOSE[@]}" up -d postgres redis qdrant

bar 2 7 "等待核心依赖健康"
for ((i=1; i<=45; i++)); do
  status="$("${COMPOSE[@]}" ps --format json 2>/dev/null || true)"
  if grep -q '"Health":"healthy"' <<<"$status" 2>/dev/null && \
     [[ "$("${COMPOSE[@]}" ps -q postgres redis qdrant | wc -l)" -ge 3 ]]; then
    echo "[ready] postgres/redis/qdrant"
    break
  fi
  printf '\r[wait]  core health (%d/45)' "$i"
  sleep 2
done
echo

bar 3 7 "构建并启动 FastAPI"
"${COMPOSE[@]}" up -d --build caged-api
wait_http "FastAPI" "http://127.0.0.1:${CAGED_API_PORT}/health" 90

bar 4 7 "构建并启动 MCP 服务"
"${COMPOSE[@]}" up -d --build caged-mcp
if ! curl -fsS --max-time 3 http://127.0.0.1:8765/ >/dev/null 2>&1; then
  echo "[info] MCP 根路径未提供 HTTP 页面；检查容器端口与日志"
fi

bar 5 7 "构建并启动 Streamlit"
"${COMPOSE[@]}" up -d --build caged-ui
wait_http "Streamlit" "http://127.0.0.1:${CAGED_UI_PORT}/_stcore/health" 60

bar 6 7 "启动 Prometheus"
"${COMPOSE[@]}" up -d prometheus
wait_http "Prometheus" "http://127.0.0.1:${PROMETHEUS_PORT}/-/ready" 60

bar 7 7 "下载并启动 Grafana"
"${COMPOSE[@]}" up -d grafana
wait_http "Grafana" "http://127.0.0.1:${GRAFANA_PORT}/api/health" 90

echo
echo "===== 完整服务状态 ====="
"${COMPOSE[@]}" ps
echo
echo "访问地址：API http://127.0.0.1:${CAGED_API_PORT}/docs | MCP http://127.0.0.1:${CAGED_MCP_PORT} | Streamlit http://127.0.0.1:${CAGED_UI_PORT} | Prometheus http://127.0.0.1:${PROMETHEUS_PORT} | Grafana http://127.0.0.1:${GRAFANA_PORT}"
echo "实时日志：${COMPOSE[*]} logs -f --tail=100 caged-api caged-mcp caged-ui prometheus grafana"
