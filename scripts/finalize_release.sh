#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TAG="${1:-v1.0.0-cpu-closeout}"
./scripts/commit_closeout.sh
if git rev-parse "$TAG" >/dev/null 2>&1; then
  echo "tag 已存在：$TAG"
else
  git tag -a "$TAG" -m "CAGED-LTR CPU deployment and thesis closeout"
  git push origin "$TAG"
  echo "release tag 已创建并推送：$TAG"
fi
