#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python3 scripts/build_local_asset_manifest.py
git add Dockerfile.cpu Dockerfile.demo docker-compose.full.yml \
  src/caged_ltr/mcp_server.py \
  scripts/verify_infra_observability.sh \
  docs/r48_r53_closeout.md monitoring/grafana monitoring/prometheus.demo.yml \
  reports/experiments/r48_r53_acceptance.json reports/experiments/r48_r53_runtime_smoke.json \
  reports/data/local_asset_manifest.json scripts/run_full_stack.sh \
  scripts/verify_full_stack.sh scripts/verify_infra_observability.sh \
  scripts/build_local_asset_manifest.py scripts/commit_closeout.sh scripts/finalize_release.sh

if git diff --cached --quiet; then
  echo "没有新的 Git 变更"
  exit 0
fi

git commit -m "Complete CPU deployment and observability closeout"
git push origin main
echo "Git 收尾提交并推送完成"
