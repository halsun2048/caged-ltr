#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-cached}"
if [[ "$MODE" != "cached" && "$MODE" != "replay" && "$MODE" != "cpu" ]]; then
  echo "usage: $0 [cached|replay|cpu]" >&2
  exit 2
fi

export R16_BACKEND="$MODE"
export R16_MODEL_PATH="${R16_MODEL_PATH:-artifacts/models/all-MiniLM-L6-v2}"
export R16_STUDENT_CHECKPOINT="${R16_STUDENT_CHECKPOINT:-artifacts/r16_runtime/mind_r13_reweight_mild.pt}"
export R16_FIRST_RESULTS="${R16_FIRST_RESULTS:-runs/mind_r10_0/dev_first/results.jsonl}"
export R16_DEVICE=cpu
export CAGED_DENSE_PROVIDER="${CAGED_DENSE_PROVIDER:-minilm_cpu}"
export CAGED_EVENT_DB="${CAGED_EVENT_DB:-runs/cpu-demo/events.sqlite3}"
export PYTHONPATH=src:scripts
export UV_CACHE_DIR=.uv-cache

exec uv run --frozen --with-requirements requirements-app.txt \
  uvicorn scripts.run_r16_api:app --host 127.0.0.1 --port "${CAGED_API_PORT:-8000}"
