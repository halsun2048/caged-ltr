"""Prepare the official TREC-DL 2019/2020 passage reranking subset for PRP."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import urllib.request
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from caged_ltr.evaluation.metrics import ranking_metrics
from caged_ltr.reproducibility import sha256_file

ProgressCallback = Callable[[dict[str, object]], None]
MSMARCO_BASE = "https://msmarco.z22.web.core.windows.net/msmarcoranking"
NIST_BASE = "https://trec.nist.gov/data/deep"


@dataclass(frozen=True, slots=True)
class DownloadSpec:
    """One immutable official download identity."""

    filename: str
    url: str
    expected_md5: str

    def __post_init__(self) -> None:
        if not self.filename or not self.url:
            raise ValueError("download filename and URL must not be empty")
        if len(self.expected_md5) != 32:
            raise ValueError("expected_md5 must be a hexadecimal MD5 digest")
        try:
            int(self.expected_md5, 16)
        except ValueError as error:
            raise ValueError("expected_md5 must be a hexadecimal MD5 digest") from error


@dataclass(frozen=True, slots=True)
class TRECYearSources:
    """Official files and expected record counts for one evaluation year."""

    year: int
    queries: DownloadSpec
    top1000: DownloadSpec
    qrels: DownloadSpec
    expected_query_records: int
    expected_top1000_records: int
    expected_qrels_records: int
    expected_judged_queries: int


DEFAULT_SOURCES = (
    TRECYearSources(
        year=2019,
        queries=DownloadSpec(
            filename="msmarco-test2019-queries.tsv.gz",
            url=f"{MSMARCO_BASE}/msmarco-test2019-queries.tsv.gz",
            expected_md5="eda71eccbe4d251af83150abe065368c",
        ),
        top1000=DownloadSpec(
            filename="msmarco-passagetest2019-top1000.tsv.gz",
            url=f"{MSMARCO_BASE}/msmarco-passagetest2019-top1000.tsv.gz",
            expected_md5="ec9e012746aa9763c7ff10b3336a3ce1",
        ),
        qrels=DownloadSpec(
            filename="2019qrels-pass.txt",
            url=f"{NIST_BASE}/2019qrels-pass.txt",
            expected_md5="2f4be390198da108f6845c822e5ada14",
        ),
        expected_query_records=200,
        expected_top1000_records=189_877,
        expected_qrels_records=9_260,
        expected_judged_queries=43,
    ),
    TRECYearSources(
        year=2020,
        queries=DownloadSpec(
            filename="msmarco-test2020-queries.tsv.gz",
            url=f"{MSMARCO_BASE}/msmarco-test2020-queries.tsv.gz",
            expected_md5="00a406fb0d14ed3752d70d1e4eb98600",
        ),
        top1000=DownloadSpec(
            filename="msmarco-passagetest2020-top1000.tsv.gz",
            url=f"{MSMARCO_BASE}/msmarco-passagetest2020-top1000.tsv.gz",
            expected_md5="aa6fbc51d66bd1dc745964c0e140a727",
        ),
        qrels=DownloadSpec(
            filename="2020qrels-pass.txt",
            url=f"{NIST_BASE}/2020qrels-pass.txt",
            expected_md5="0355ccee7509ac0463e8278186cdd8d1",
        ),
        expected_query_records=200,
        expected_top1000_records=190_699,
        expected_qrels_records=11_386,
        expected_judged_queries=54,
    ),
)


def _md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _download_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _response_total(response: BinaryIO, existing_size: int) -> int | None:
    content_range = response.headers.get("Content-Range")
    if content_range and "/" in content_range:
        total = content_range.rsplit("/", maxsplit=1)[1]
        return int(total) if total.isdigit() else None
    content_length = response.headers.get("Content-Length")
    return existing_size + int(content_length) if content_length else None


def download_file(
    spec: DownloadSpec,
    raw_dir: Path,
    *,
    progress_callback: ProgressCallback | None = None,
    chunk_size: int = 1024 * 1024,
) -> dict[str, object]:
    """Download one file with Range resume and reject any MD5 mismatch."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / spec.filename
    if destination.is_file() and _md5_file(destination) == spec.expected_md5:
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "download",
                    "filename": spec.filename,
                    "done": destination.stat().st_size,
                    "total": destination.stat().st_size,
                    "cached": True,
                }
            )
        return {
            "filename": spec.filename,
            "url": spec.url,
            "bytes": destination.stat().st_size,
            "md5": spec.expected_md5,
            "sha256": sha256_file(destination),
            "cached": True,
        }
    if destination.is_file():
        raise ValueError(f"downloaded file MD5 mismatch: {destination}")

    partial = destination.with_suffix(destination.suffix + ".part")
    existing_size = partial.stat().st_size if partial.is_file() else 0
    request = urllib.request.Request(
        spec.url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; caged-ltr research data preparation)"
            )
        },
    )
    if existing_size:
        request.add_header("Range", f"bytes={existing_size}-")
    with _download_opener().open(request, timeout=60) as response:
        status = getattr(response, "status", None)
        append = existing_size > 0 and status == 206
        if existing_size and not append:
            existing_size = 0
        total = _response_total(response, existing_size)
        mode = "ab" if append else "wb"
        done = existing_size
        with partial.open(mode) as output:
            while chunk := response.read(chunk_size):
                output.write(chunk)
                done += len(chunk)
                if progress_callback is not None:
                    progress_callback(
                        {
                            "stage": "download",
                            "filename": spec.filename,
                            "done": done,
                            "total": total,
                            "cached": False,
                        }
                    )
            output.flush()
            os.fsync(output.fileno())
    if _md5_file(partial) != spec.expected_md5:
        raise ValueError(f"downloaded file MD5 mismatch: {partial}")
    partial.replace(destination)
    return {
        "filename": spec.filename,
        "url": spec.url,
        "bytes": destination.stat().st_size,
        "md5": spec.expected_md5,
        "sha256": sha256_file(destination),
        "cached": False,
    }


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _read_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            fields = line.rstrip("\n").split("\t", maxsplit=1)
            if len(fields) != 2 or not all(fields):
                raise ValueError(f"invalid query TSV at {path}:{line_number}")
            query_id, text = fields
            if query_id in queries:
                raise ValueError(f"duplicate query ID in {path}: {query_id}")
            queries[query_id] = _normalize_text(text)
    return queries


def _read_qrels(path: Path) -> tuple[dict[str, dict[str, int]], int]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    record_count = 0
    with path.open("rt", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            fields = line.split()
            if len(fields) != 4:
                raise ValueError(f"invalid qrels row at {path}:{line_number}")
            query_id, _, passage_id, raw_relevance = fields
            relevance = int(raw_relevance)
            if relevance not in {0, 1, 2, 3}:
                raise ValueError(f"invalid qrels relevance at {path}:{line_number}")
            if passage_id in qrels[query_id]:
                raise ValueError(
                    f"duplicate qrels pair at {path}:{line_number}: "
                    f"{query_id}/{passage_id}"
                )
            qrels[query_id][passage_id] = relevance
            record_count += 1
    return dict(qrels), record_count


def _read_top_candidates(
    path: Path,
    *,
    target_query_ids: set[str],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    year: int,
    top_k: int,
    expected_records: int,
    progress_callback: ProgressCallback | None,
) -> tuple[list[dict[str, object]], int]:
    candidates: list[dict[str, object]] = []
    seen_per_query: dict[str, int] = defaultdict(int)
    passage_ids_per_query: dict[str, set[str]] = defaultdict(set)
    record_count = 0
    with gzip.open(path, "rt", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            fields = line.rstrip("\n").split("\t", maxsplit=3)
            if len(fields) != 4:
                raise ValueError(f"invalid top1000 TSV at {path}:{line_number}")
            query_id, passage_id, embedded_query, passage = fields
            record_count += 1
            if progress_callback is not None and record_count % 2000 == 0:
                progress_callback(
                    {
                        "stage": "parse",
                        "filename": path.name,
                        "done": record_count,
                        "total": expected_records,
                        "cached": False,
                    }
                )
            if query_id not in target_query_ids:
                continue
            seen_per_query[query_id] += 1
            if passage_id in passage_ids_per_query[query_id]:
                raise ValueError(f"duplicate passage in candidate list: {query_id}/{passage_id}")
            passage_ids_per_query[query_id].add(passage_id)
            if _normalize_text(embedded_query) != queries[query_id]:
                raise ValueError(f"query text mismatch in {path}: {query_id}")
            rank = seen_per_query[query_id]
            if rank <= top_k:
                relevance = qrels[query_id].get(passage_id, 0)
                candidates.append(
                    {
                        "request_id": f"dl{year}-{query_id}",
                        "year": year,
                        "query_id": query_id,
                        "passage_id": passage_id,
                        "bm25_rank": rank,
                        "query": queries[query_id],
                        "passage": passage,
                        "judged": passage_id in qrels[query_id],
                        "graded_relevance": relevance,
                        "trec_relevance": relevance if relevance >= 2 else 0,
                        "binary_relevance": int(relevance >= 2),
                    }
                )
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "parse",
                "filename": path.name,
                "done": record_count,
                "total": expected_records,
                "cached": False,
            }
        )
    if set(seen_per_query) != target_query_ids:
        missing = sorted(target_query_ids - set(seen_per_query))
        raise ValueError(f"top1000 file is missing judged query IDs: {missing[:5]}")
    return candidates, record_count


def _read_bm25_snapshot(
    path: Path,
    *,
    top_k: int,
) -> dict[str, list[dict[str, object]]]:
    candidates: dict[str, list[dict[str, object]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    with gzip.open(path, "rt", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            try:
                row = json.loads(line)
                query_id = str(row["query_id"])
                passage_id = str(row["passage_id"])
                rank = int(row["bm25_rank"])
                score = float(row["bm25_score"])
                passage = str(row["passage"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid BM25 snapshot row at {path}:{line_number}"
                ) from error
            if not query_id or not passage_id or not passage.strip():
                raise ValueError(f"empty BM25 snapshot field at {path}:{line_number}")
            if rank < 1 or rank > top_k:
                raise ValueError(f"invalid BM25 rank at {path}:{line_number}")
            identity = (query_id, passage_id)
            if identity in seen:
                raise ValueError(f"duplicate BM25 snapshot pair: {query_id}/{passage_id}")
            seen.add(identity)
            candidates[query_id].append(
                {
                    "passage_id": passage_id,
                    "bm25_rank": rank,
                    "bm25_score": score,
                    "passage": passage,
                }
            )
    for query_id, rows in candidates.items():
        rows.sort(key=lambda row: int(row["bm25_rank"]))
        ranks = [int(row["bm25_rank"]) for row in rows]
        if ranks != list(range(1, len(rows) + 1)):
            raise ValueError(f"non-contiguous BM25 ranks for query: {query_id}")
    return dict(candidates)


def _candidates_from_bm25_snapshot(
    snapshot: dict[str, list[dict[str, object]]],
    *,
    target_query_ids: set[str],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    year: int,
    top_k: int,
) -> list[dict[str, object]]:
    missing = sorted(target_query_ids - set(snapshot))
    if missing:
        raise ValueError(f"BM25 snapshot is missing judged query IDs: {missing[:5]}")
    candidates: list[dict[str, object]] = []
    for query_id in sorted(target_query_ids, key=int):
        rows = snapshot[query_id]
        if len(rows) != top_k:
            raise ValueError(
                f"BM25 snapshot candidate count mismatch: "
                f"{query_id} has {len(rows)}, expected {top_k}"
            )
        for row in rows:
            passage_id = str(row["passage_id"])
            relevance = qrels[query_id].get(passage_id, 0)
            candidates.append(
                {
                    "request_id": f"dl{year}-{query_id}",
                    "year": year,
                    "query_id": query_id,
                    "passage_id": passage_id,
                    "bm25_rank": int(row["bm25_rank"]),
                    "bm25_score": float(row["bm25_score"]),
                    "query": queries[query_id],
                    "passage": str(row["passage"]),
                    "judged": passage_id in qrels[query_id],
                    "graded_relevance": relevance,
                    "trec_relevance": relevance if relevance >= 2 else 0,
                    "binary_relevance": int(relevance >= 2),
                }
            )
    return candidates


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_teacher_inputs(
    candidates: pd.DataFrame,
    path: Path,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for request_id, group in candidates.groupby("request_id", sort=False):
            first = group.iloc[0]
            payload = {
                "request_id": request_id,
                "year": int(first["year"]),
                "query_id": str(first["query_id"]),
                "query": str(first["query"]),
                "candidates": [
                    {
                        "passage_id": str(row["passage_id"]),
                        "bm25_rank": int(row["bm25_rank"]),
                        "passage": str(row["passage"]),
                    }
                    for _, row in group.sort_values("bm25_rank").iterrows()
                ],
            }
            output.write(json.dumps(payload, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def _qrels_aware_ranking_metrics(
    candidates: pd.DataFrame,
    qrels: pd.DataFrame,
    *,
    label_column: str,
    cutoffs: tuple[int, ...] = (5, 10),
    exponential_gain: bool = True,
) -> dict[str, float]:
    """Score candidate rankings against the complete qrels ideal and recall set."""
    qrels_by_request = {
        str(request_id): group[label_column].astype(int).tolist()
        for request_id, group in qrels.groupby("request_id", sort=False)
    }
    reciprocal_ranks: list[float] = []
    recall_values = {cutoff: [] for cutoff in cutoffs}
    ndcg_values = {cutoff: [] for cutoff in cutoffs}
    for request_id, group in candidates.groupby("request_id", sort=False):
        ranked = (
            group.sort_values("bm25_rank", kind="stable")[label_column]
            .astype(int)
            .tolist()
        )
        full_labels = qrels_by_request[str(request_id)]
        first_relevant = next(
            (rank for rank, relevance in enumerate(ranked, start=1) if relevance > 0),
            None,
        )
        reciprocal_ranks.append(1.0 / first_relevant if first_relevant else 0.0)
        total_relevant = sum(relevance > 0 for relevance in full_labels)
        ideal = sorted(full_labels, reverse=True)
        for cutoff in cutoffs:
            observed = ranked[:cutoff]
            recall_values[cutoff].append(
                sum(relevance > 0 for relevance in observed) / total_relevant
                if total_relevant
                else 0.0
            )
            dcg = sum(
                ((2**relevance - 1) if exponential_gain else relevance)
                / math.log2(rank + 1)
                for rank, relevance in enumerate(observed, start=1)
            )
            idcg = sum(
                ((2**relevance - 1) if exponential_gain else relevance)
                / math.log2(rank + 1)
                for rank, relevance in enumerate(ideal[:cutoff], start=1)
            )
            ndcg_values[cutoff].append(dcg / idcg if idcg else 0.0)
    metrics = {"MRR": sum(reciprocal_ranks) / len(reciprocal_ranks)}
    for cutoff in cutoffs:
        metrics[f"Recall@{cutoff}"] = (
            sum(recall_values[cutoff]) / len(recall_values[cutoff])
        )
        metrics[f"NDCG@{cutoff}"] = (
            sum(ndcg_values[cutoff]) / len(ndcg_values[cutoff])
        )
    return metrics


def _audit(
    queries: pd.DataFrame,
    candidates: pd.DataFrame,
    qrels: pd.DataFrame,
    *,
    top_k: int,
) -> dict[str, object]:
    list_sizes = candidates.groupby("request_id").size()
    candidate_conditional_ranking = ranking_metrics(
        candidates["graded_relevance"].to_numpy(),
        -candidates["bm25_rank"].to_numpy(),
        candidates["request_id"].to_numpy(),
        cutoffs=(5, 10),
    )
    candidate_conditional_binary_ranking = ranking_metrics(
        candidates["binary_relevance"].to_numpy(),
        -candidates["bm25_rank"].to_numpy(),
        candidates["request_id"].to_numpy(),
        cutoffs=(5, 10),
    )
    candidate_conditional_trec_ranking = _qrels_aware_ranking_metrics(
        candidates,
        candidates,
        label_column="trec_relevance",
        exponential_gain=False,
    )
    ranking = _qrels_aware_ranking_metrics(
        candidates,
        qrels,
        label_column="graded_relevance",
    )
    trec_eval_ranking = _qrels_aware_ranking_metrics(
        candidates,
        qrels,
        label_column="graded_relevance",
        exponential_gain=False,
    )
    binary_ranking = _qrels_aware_ranking_metrics(
        candidates,
        qrels,
        label_column="binary_relevance",
    )
    trec_ranking = _qrels_aware_ranking_metrics(
        candidates,
        qrels,
        label_column="trec_relevance",
        exponential_gain=False,
    )
    by_year: dict[str, object] = {}
    for year, year_queries in queries.groupby("year", sort=True):
        year_candidates = candidates[candidates["year"] == year]
        year_qrels = qrels[qrels["year"] == year]
        by_year[str(year)] = {
            "queries": len(year_queries),
            "candidates": len(year_candidates),
            "top10_candidates": len(year_candidates),
            "qrels": len(year_qrels),
            "qrels_relevant_at_least_2": int(
                (year_qrels["graded_relevance"] >= 2).sum()
            ),
            "judged": int(year_candidates["judged"].sum()),
            "top10_judged": int(year_candidates["judged"].sum()),
            "relevant_at_least_2": int(
                (year_candidates["graded_relevance"] >= 2).sum()
            ),
            "top10_relevant_at_least_2": int(
                (year_candidates["graded_relevance"] >= 2).sum()
            ),
            "queries_with_relevant_at_least_2": int(
                year_candidates.groupby("request_id")["binary_relevance"]
                .max()
                .sum()
            ),
            "queries_with_top10_relevant_at_least_2": int(
                year_candidates.groupby("request_id")["binary_relevance"]
                .max()
                .sum()
            ),
            "bm25_graded": _qrels_aware_ranking_metrics(
                year_candidates,
                year_qrels,
                label_column="graded_relevance",
            ),
            "bm25_trec_eval": _qrels_aware_ranking_metrics(
                year_candidates,
                year_qrels,
                label_column="graded_relevance",
                exponential_gain=False,
            ),
            "bm25_binary_relevance_at_least_2": _qrels_aware_ranking_metrics(
                year_candidates,
                year_qrels,
                label_column="binary_relevance",
            ),
            "bm25_trec_relevance_at_least_2": _qrels_aware_ranking_metrics(
                year_candidates,
                year_qrels,
                label_column="trec_relevance",
                exponential_gain=False,
            ),
        }
    return {
        "queries": len(queries),
        "candidates": len(candidates),
        "top10_candidates": len(candidates),
        "candidate_limit": top_k,
        "candidate_list_size_min": int(list_sizes.min()),
        "candidate_list_size_max": int(list_sizes.max()),
        "short_candidate_lists": {
            str(request_id): int(size)
            for request_id, size in list_sizes.items()
            if size < top_k
        },
        "unique_passages": int(candidates["passage_id"].nunique()),
        "judged_rate": float(candidates["judged"].mean()),
        "top10_judged_rate": float(candidates["judged"].mean()),
        "relevant_at_least_2": int(
            (candidates["graded_relevance"] >= 2).sum()
        ),
        "top10_relevant_at_least_2": int(
            (candidates["graded_relevance"] >= 2).sum()
        ),
        "queries_with_relevant_at_least_2": int(
            candidates.groupby("request_id")["binary_relevance"].max().sum()
        ),
        "queries_with_top10_relevant_at_least_2": int(
            candidates.groupby("request_id")["binary_relevance"].max().sum()
        ),
        "bm25_graded": ranking,
        "bm25_trec_eval": trec_eval_ranking,
        "bm25_binary_relevance_at_least_2": binary_ranking,
        "bm25_trec_relevance_at_least_2": trec_ranking,
        "bm25_candidate_conditional_graded": candidate_conditional_ranking,
        "bm25_candidate_conditional_binary_relevance_at_least_2": (
            candidate_conditional_binary_ranking
        ),
        "bm25_candidate_conditional_trec_relevance_at_least_2": (
            candidate_conditional_trec_ranking
        ),
        "primary_metric": "bm25_trec_eval",
        "primary_metric_denominator": "complete official qrels",
        "trec_eval_definition": (
            "bm25_trec_eval uses linear graded gain over complete qrels; "
            "binary relevance separately treats only grades 2 and 3 as relevant"
        ),
        "by_year": by_year,
        "binary_relevance_rule": (
            "NIST passage relevance 1 is Related but non-relevant; "
            "only grades 2 and 3 are binary relevant"
        ),
    }


def prepare_prp_trec_dl(
    raw_dir: Path,
    processed_dir: Path,
    *,
    top_k: int = 10,
    sources: Sequence[TRECYearSources] = DEFAULT_SOURCES,
    candidate_snapshot: Path | None = None,
    candidate_snapshot_sha256: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, object]:
    """Download, validate, and prepare the judged DL19/DL20 Top-K passage lists."""
    if top_k <= 1:
        raise ValueError("top_k must be greater than one")
    if not sources:
        raise ValueError("at least one source year is required")
    if candidate_snapshot is not None and not candidate_snapshot.is_file():
        raise FileNotFoundError(
            f"BM25 candidate snapshot not found: {candidate_snapshot}; "
            "generate the Pyserini runs and run scripts/export_prp_bm25_top10.py"
        )
    snapshot_sha256 = (
        sha256_file(candidate_snapshot) if candidate_snapshot is not None else None
    )
    if (
        candidate_snapshot_sha256 is not None
        and snapshot_sha256 != candidate_snapshot_sha256
    ):
        raise ValueError(
            f"BM25 candidate snapshot SHA-256 mismatch: "
            f"{snapshot_sha256} != {candidate_snapshot_sha256}"
        )
    processed_dir.mkdir(parents=True, exist_ok=True)
    downloads: dict[str, dict[str, object]] = {}
    for source in sources:
        specs = [source.queries, source.qrels]
        if candidate_snapshot is None:
            specs.insert(1, source.top1000)
        for spec in specs:
            if spec.filename in downloads:
                continue
            downloads[spec.filename] = download_file(
                spec,
                raw_dir,
                progress_callback=progress_callback,
            )

    snapshot = (
        _read_bm25_snapshot(candidate_snapshot, top_k=top_k)
        if candidate_snapshot is not None
        else None
    )
    query_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    qrel_rows: list[dict[str, object]] = []
    source_audits: dict[str, object] = {}
    request_ids: set[str] = set()
    for source in sources:
        queries = _read_queries(raw_dir / source.queries.filename)
        qrels, qrels_count = _read_qrels(raw_dir / source.qrels.filename)
        if len(queries) != source.expected_query_records:
            raise ValueError(
                f"{source.year} query record mismatch: "
                f"{len(queries)} != {source.expected_query_records}"
            )
        if qrels_count != source.expected_qrels_records:
            raise ValueError(
                f"{source.year} qrels record mismatch: "
                f"{qrels_count} != {source.expected_qrels_records}"
            )
        target_query_ids = set(qrels)
        if len(target_query_ids) != source.expected_judged_queries:
            raise ValueError(
                f"{source.year} judged query mismatch: "
                f"{len(target_query_ids)} != {source.expected_judged_queries}"
            )
        if not target_query_ids.issubset(queries):
            raise ValueError(f"{source.year} qrels contain unknown query IDs")

        if snapshot is not None:
            year_candidates = _candidates_from_bm25_snapshot(
                snapshot,
                target_query_ids=target_query_ids,
                queries=queries,
                qrels=qrels,
                year=source.year,
                top_k=top_k,
            )
            top_records: int | None = None
        else:
            year_candidates, top_records = _read_top_candidates(
                raw_dir / source.top1000.filename,
                target_query_ids=target_query_ids,
                queries=queries,
                qrels=qrels,
                year=source.year,
                top_k=top_k,
                expected_records=source.expected_top1000_records,
                progress_callback=progress_callback,
            )
            if top_records != source.expected_top1000_records:
                raise ValueError(
                    f"{source.year} top1000 record mismatch: "
                    f"{top_records} != {source.expected_top1000_records}"
                )
        for query_id in sorted(target_query_ids, key=int):
            request_id = f"dl{source.year}-{query_id}"
            if request_id in request_ids:
                raise ValueError(f"duplicate request ID: {request_id}")
            request_ids.add(request_id)
            query_rows.append(
                {
                    "request_id": request_id,
                    "year": source.year,
                    "query_id": query_id,
                    "query": queries[query_id],
                }
            )
            for passage_id, relevance in qrels[query_id].items():
                qrel_rows.append(
                    {
                        "request_id": request_id,
                        "year": source.year,
                        "query_id": query_id,
                        "passage_id": passage_id,
                        "graded_relevance": relevance,
                        "trec_relevance": relevance if relevance >= 2 else 0,
                        "binary_relevance": int(relevance >= 2),
                    }
                )
        candidate_rows.extend(year_candidates)
        source_audit = {
            "query_records": len(queries),
            "judged_queries": len(target_query_ids),
            "qrels_records": qrels_count,
        }
        if top_records is not None:
            source_audit["top1000_records"] = top_records
        source_audits[str(source.year)] = source_audit

    queries_frame = pd.DataFrame(query_rows).sort_values(
        ["year", "query_id"],
        kind="stable",
    )
    candidates_frame = pd.DataFrame(candidate_rows).sort_values(
        ["year", "query_id", "bm25_rank"],
        kind="stable",
    )
    qrels_frame = pd.DataFrame(qrel_rows).sort_values(
        ["year", "query_id", "passage_id"],
        kind="stable",
    )
    list_sizes = candidates_frame.groupby("request_id").size()
    if set(list_sizes.index) != request_ids:
        raise ValueError("one or more judged queries have no candidates")
    if (list_sizes > top_k).any():
        raise ValueError("one or more prepared requests exceed top_k")

    queries_path = processed_dir / "queries.parquet"
    candidates_path = processed_dir / "candidates.parquet"
    qrels_path = processed_dir / "qrels.parquet"
    teacher_inputs_path = processed_dir / "teacher_inputs.jsonl"
    _atomic_parquet(queries_frame, queries_path)
    _atomic_parquet(candidates_frame, candidates_path)
    _atomic_parquet(qrels_frame, qrels_path)
    _write_teacher_inputs(candidates_frame, teacher_inputs_path)
    audit = _audit(
        queries_frame,
        candidates_frame,
        qrels_frame,
        top_k=top_k,
    )
    artifact_paths = (
        queries_path,
        candidates_path,
        qrels_path,
        teacher_inputs_path,
    )
    manifest: dict[str, object] = {
        "stage": "complete",
        "dataset": "TREC-DL 2019/2020 passage reranking",
        "result_type": "official evaluation data preparation; no model inference",
        "top_k": top_k,
        "license_boundary": "MS MARCO non-commercial research use",
        "sources": (
            [
                {
                    "year": source.year,
                    "queries": asdict(source.queries),
                    "qrels": asdict(source.qrels),
                    "expected_query_records": source.expected_query_records,
                    "expected_qrels_records": source.expected_qrels_records,
                    "expected_judged_queries": source.expected_judged_queries,
                }
                for source in sources
            ]
            if candidate_snapshot is not None
            else [asdict(source) for source in sources]
        ),
        "downloads": downloads,
        "candidate_source": (
            {
                "type": "pyserini_bm25_snapshot",
                "path": str(candidate_snapshot),
                "bytes": candidate_snapshot.stat().st_size,
                "sha256": snapshot_sha256,
            }
            if candidate_snapshot is not None
            else {"type": "legacy_top1000_file_order_fixture"}
        ),
        "source_audit": source_audits,
        "audit": audit,
        "artifacts": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in artifact_paths
        },
        "teacher_input_excludes_qrels": True,
    }
    _atomic_json(processed_dir / "manifest.json", manifest)
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "complete",
                "filename": "manifest.json",
                "done": len(query_rows),
                "total": len(query_rows),
                "cached": False,
            }
        )
    return manifest
