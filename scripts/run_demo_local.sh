#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${PROJECT_ROOT}/runs/demo_local"
mkdir -p "${RUN_DIR}"

if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
  echo "FastAPI already running at http://127.0.0.1:8000"
else
  cd "${PROJECT_ROOT}"
  UV_CACHE_DIR=.uv-cache PYTHONPATH=src R16_BACKEND=cached \
    nohup uv run --frozen --with-requirements requirements-app.txt \
    uvicorn scripts.run_r16_api:app --host 127.0.0.1 --port 8000 \
    > "${RUN_DIR}/api.log" 2>&1 < /dev/null &
  echo $! > "${RUN_DIR}/api.pid"
fi

for _ in $(seq 1 30); do
  curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1 && break
  sleep 1
done
curl -fsS http://127.0.0.1:8000/health >/dev/null

if curl -fsS http://127.0.0.1:8501/ >/dev/null 2>&1; then
  echo "Streamlit already running at http://127.0.0.1:8501"
else
  cd "${PROJECT_ROOT}"
  UV_CACHE_DIR=.uv-cache PYTHONPATH=src R18_API_URL=http://127.0.0.1:8000 \
    nohup uv run --frozen --with-requirements requirements-app.txt \
    streamlit run app/streamlit_app.py --server.address=127.0.0.1 \
    --server.port=8501 --server.headless=true \
    > "${RUN_DIR}/streamlit.log" 2>&1 < /dev/null &
  echo $! > "${RUN_DIR}/streamlit.pid"
fi

echo "API: http://127.0.0.1:8000"
echo "UI:  http://127.0.0.1:8501"
echo "Logs: ${RUN_DIR}"
