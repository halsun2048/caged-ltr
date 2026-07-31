"""Freeze the leakage-safe 1k-query MS MARCO R4 distillation dataset."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path
from typing import Any

from caged_ltr.data.instruction_distillation import (
    MSMARCO_QUERY_ARCHIVE_SHA256,
    RetrievedPassage,
    evaluation_identities,
    read_msmarco_train_queries,
    retrieve_distillation_candidates,
    select_distillation_queries,
    write_distillation_dataset,
)

DEFAULT_INDEX_PATH = Path(
    "data/raw/prp_trec_dl/pyserini_cache/indexes/"
    "lucene-inverted.msmarco-v1-passage.20221004."
    "252b5e.678876e8c99a89933d553609a0fd8793"
)


def _duration(seconds: float) -> str:
    total = max(0, round(seconds))
    minutes, seconds = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}h{minutes:02d}m" if hours else f"{minutes:02d}m{seconds:02d}s"


class _Progress:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self.started = time.monotonic()
        self.line_open = False

    def __call__(self, event: dict[str, object]) -> None:
        if not self.enabled:
            return
        done = int(event["done"])
        total = int(event["total"])
        width = 24
        filled = round(width * done / max(total, 1))
        elapsed = time.monotonic() - self.started
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else 0.0
        sys.stderr.write("\r\033[2K")
        sys.stderr.write(
            f"[retrieve] [{'#' * filled}{'-' * (width - filled)}] "
            f"queries={done:>4}/{total:<4} rate={rate:>5.2f}/s "
            f"elapsed={_duration(elapsed)} ETA={_duration(eta) if rate else '--'}"
        )
        sys.stderr.flush()
        self.line_open = True
        if done == total:
            sys.stderr.write("\n")
            sys.stderr.flush()
            self.line_open = False

    def finish(self) -> None:
        if self.enabled and self.line_open:
            sys.stderr.write("\n")
            sys.stderr.flush()
            self.line_open = False


class _PyseriniRetriever:
    def __init__(self, index_path: Path) -> None:
        try:
            from pyserini.search.lucene import LuceneSearcher
        except ImportError as error:
            raise RuntimeError(
                "Pyserini is required; run with `uv run --with pyserini==2.3.0 ...`"
            ) from error
        if importlib.metadata.version("pyserini") != "2.3.0":
            raise RuntimeError("R4 requires the pinned pyserini==2.3.0")
        if not index_path.is_dir():
            raise FileNotFoundError(f"MS MARCO Lucene index not found: {index_path}")
        self.searcher: Any = LuceneSearcher(str(index_path))
        self.searcher.set_bm25(0.9, 0.4)

    def __call__(self, query: str, top_k: int) -> list[RetrievedPassage]:
        passages = []
        for hit in self.searcher.search(query, k=top_k):
            document = self.searcher.doc(hit.docid)
            if document is None:
                raise ValueError(f"passage missing from index: {hit.docid}")
            raw = json.loads(document.raw())
            text = raw.get("contents")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"passage text missing from index: {hit.docid}")
            passages.append(
                RetrievedPassage(
                    passage_id=str(hit.docid),
                    score=float(hit.score),
                    passage=text,
                )
            )
        return passages


def _write_report(path: Path, manifest: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-archive",
        type=Path,
        default=Path("data/raw/prp_trec_dl/queries.tar.gz"),
    )
    parser.add_argument(
        "--query-archive-sha256",
        default=MSMARCO_QUERY_ARCHIVE_SHA256,
    )
    parser.add_argument(
        "--evaluation-queries",
        type=Path,
        action="append",
        default=None,
        help="Query-only parquet used for identity exclusion; repeat as needed.",
    )
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/r4_msmarco_1k"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/data/r4_msmarco_1k_summary.json"),
    )
    parser.add_argument("--query-count", type=int, default=1_000)
    parser.add_argument("--validation-count", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    evaluation_paths = args.evaluation_queries or [
        Path("data/processed/prp_trec_dl_top100/queries.parquet")
    ]
    test_query_ids, test_query_texts = evaluation_identities(evaluation_paths)
    queries, selection_audit = select_distillation_queries(
        read_msmarco_train_queries(
            args.query_archive,
            expected_sha256=args.query_archive_sha256,
        ),
        evaluation_query_ids=test_query_ids,
        evaluation_query_texts=test_query_texts,
        query_count=args.query_count,
        validation_count=args.validation_count,
        seed=args.seed,
    )
    selected_ids = {query.query_id for query in queries}
    selected_texts = {query.query.casefold() for query in queries}
    if selected_ids & test_query_ids or selected_texts & test_query_texts:
        raise AssertionError("test query leakage detected after selection")

    progress = _Progress(enabled=args.progress)
    candidates, controls = retrieve_distillation_candidates(
        queries,
        _PyseriniRetriever(args.index_path),
        top_k=args.top_k,
        random_seed=args.seed,
        progress_callback=progress,
    )
    progress.finish()
    manifest = write_distillation_dataset(
        args.output_dir,
        queries,
        candidates,
        controls,
        source_archive=args.query_archive,
        selection_audit=selection_audit,
        seed=args.seed,
        top_k=args.top_k,
    )
    manifest["retrieval"] = {
        "pyserini_version": "2.3.0",
        "index_path": str(args.index_path),
        "index": "msmarco-v1-passage",
        "bm25": {"k1": 0.9, "b": 0.4},
    }
    manifest["test_isolation"]["query_only_sources"] = [
        str(path) for path in evaluation_paths
    ]
    _write_report(args.output_dir / "manifest.json", manifest)
    _write_report(args.report, manifest)
    print(
        json.dumps(
            {
                "stage": manifest["stage"],
                "queries": manifest["selection"]["selected_queries"],
                "train": manifest["selection"]["train_queries"],
                "validation": manifest["selection"]["validation_queries"],
                "candidates": len(candidates),
                "qrels_accessed": manifest["test_isolation"]["qrels_accessed"],
                "report": str(args.report),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
