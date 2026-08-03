# GPU server backup — 2026-08-03

Local-only backup copied from `/root/caged-ltr` before releasing the GPU server.

- Archive: `caged-ltr-gpu-essential-20260803.tgz` (about 1.1 GB)
- Archive SHA-256: `42ff3b740bf0ec28584db42727e9fc96f1ab426acfbf3604d610547d000d9c92`
- File manifest: `SHA256SUMS.txt`
- Manifest SHA-256: `d101eef40daba8196d685c087225e06da2e0dfa17669237879b4dbc0b11ec6e2`

The archive contains final checkpoints, Gate artifacts, reports, code, configs,
FIRST outputs/logs and the MiniLM base model. Regenerable prompt JSONL files,
duplicate `*_latest.pt` resume checkpoints, Parquet caches and SQLite runtime
databases were intentionally excluded.

Verify with:

```bash
sha256sum caged-ltr-gpu-essential-20260803.tgz
tar -tzf caged-ltr-gpu-essential-20260803.tgz >/dev/null
```
