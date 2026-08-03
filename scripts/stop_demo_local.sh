#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${PROJECT_ROOT}/runs/demo_local"

for service in api streamlit; do
  pid_file="${RUN_DIR}/${service}.pid"
  if [[ -f "${pid_file}" ]]; then
    pid="$(<"${pid_file}")"
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}"
      echo "Stopped ${service} (PID ${pid})"
    fi
    rm -f "${pid_file}"
  fi
done
