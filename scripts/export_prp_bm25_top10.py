"""Export the PRP BM25 Top-K text snapshot from a Pyserini prebuilt index."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_top_k(path: Path, top_k: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    with path.open("rt", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            fields = line.split()
            if len(fields) != 6:
                raise ValueError(f"invalid TREC run row at {path}:{line_number}")
            query_id, _, passage_id, raw_rank, raw_score, run_name = fields
            rank = int(raw_rank)
            if rank > top_k:
                continue
            identity = (query_id, passage_id)
            if identity in seen:
                raise ValueError(f"duplicate run pair: {query_id}/{passage_id}")
            seen.add(identity)
            rows.append(
                {
                    "query_id": query_id,
                    "passage_id": passage_id,
                    "bm25_rank": rank,
                    "bm25_score": float(raw_score),
                    "run_name": run_name,
                }
            )
    return rows


def _progress(done: int, total: int, started: float) -> None:
    width = 24
    filled = round(width * done / total)
    elapsed = max(0, round(time.monotonic() - started))
    sys.stderr.write(
        f"\r\033[2K[export] [{'#' * filled}{'-' * (width - filled)}] "
        f"passages={done}/{total} elapsed={elapsed // 60:02d}m{elapsed % 60:02d}s"
    )
    if done == total:
        sys.stderr.write("\n")
    sys.stderr.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        action="append",
        required=True,
        help="TREC run path; repeat once per evaluation year",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/prp_trec_dl/pyserini_bm25_top10.jsonl.gz"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/raw/prp_trec_dl/pyserini_bm25_top10_manifest.json"),
    )
    parser.add_argument("--index", default="msmarco-v1-passage")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if args.top_k <= 1:
        raise ValueError("top_k must be greater than one")

    from pyserini.search.lucene import LuceneSearcher

    rows = [
        row
        for run_path in args.run
        for row in _read_top_k(run_path, args.top_k)
    ]
    identities = {
        (str(row["query_id"]), str(row["passage_id"])) for row in rows
    }
    if len(identities) != len(rows):
        raise ValueError("duplicate query/passage pair across run files")
    rows.sort(key=lambda row: (int(str(row["query_id"])), int(row["bm25_rank"])))

    searcher = LuceneSearcher.from_prebuilt_index(args.index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    started = time.monotonic()
    with temporary.open("wb") as raw_output:
        with gzip.GzipFile(fileobj=raw_output, mode="wb", mtime=0) as compressed:
            for index, row in enumerate(rows, start=1):
                document = searcher.doc(str(row["passage_id"]))
                if document is None:
                    raise ValueError(f"passage missing from index: {row['passage_id']}")
                raw = json.loads(document.raw())
                text = raw.get("contents")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"passage text missing: {row['passage_id']}")
                payload = {**row, "passage": text}
                compressed.write(
                    (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
                )
                if args.progress and (index % 25 == 0 or index == len(rows)):
                    _progress(index, len(rows), started)
        raw_output.flush()
        os.fsync(raw_output.fileno())
    temporary.replace(args.output)

    manifest = {
        "stage": "complete",
        "result_type": "deterministic BM25 retrieval text snapshot; no model inference",
        "pyserini_version": importlib.metadata.version("pyserini"),
        "index": args.index,
        "bm25": {"k1": 0.9, "b": 0.4},
        "top_k": args.top_k,
        "queries": len({str(row["query_id"]) for row in rows}),
        "candidates": len(rows),
        "runs": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in args.run
        },
        "snapshot": {
            "path": str(args.output),
            "bytes": args.output.stat().st_size,
            "sha256": _sha256(args.output),
        },
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
