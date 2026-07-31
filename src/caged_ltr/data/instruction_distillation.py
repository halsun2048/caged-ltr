"""Leakage-safe MS MARCO data preparation for R4 instruction distillation."""

from __future__ import annotations

import hashlib
import heapq
import json
import tarfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from caged_ltr.reproducibility import sha256_file

MSMARCO_QUERY_ARCHIVE_SHA256 = (
    "05e4c62c9c8520cd695725340d4c990627fd1a92eb8a71b88e3c746031ba6de8"
)
MSMARCO_TRAIN_MEMBER = "queries.train.tsv"


@dataclass(frozen=True, slots=True)
class DistillationQuery:
    """One immutable train or validation query."""

    query_id: str
    query: str
    split: str
    selection_sha256: str

    def __post_init__(self) -> None:
        if not self.query_id or not self.query:
            raise ValueError("query_id and query must not be empty")
        if self.split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        if len(self.selection_sha256) != 64:
            raise ValueError("selection_sha256 must be a full SHA-256 digest")


@dataclass(frozen=True, slots=True)
class RetrievedPassage:
    """One BM25 candidate returned by a text retriever."""

    passage_id: str
    score: float
    passage: str

    def __post_init__(self) -> None:
        if not self.passage_id or not self.passage.strip():
            raise ValueError("passage_id and passage must not be empty")


class PassageRetriever(Protocol):
    """Minimal retrieval interface used by the deterministic builder."""

    def __call__(self, query: str, top_k: int) -> Sequence[RetrievedPassage]: ...


ProgressCallback = Callable[[dict[str, object]], None]


def normalize_query(text: str) -> str:
    """Normalize whitespace and case for exact-text leakage checks."""
    return " ".join(text.split()).casefold()


def _selection_digest(namespace: str, seed: int, query_id: str, query: str) -> str:
    payload = f"{namespace}\0{seed}\0{query_id}\0{query}".encode()
    return hashlib.sha256(payload).hexdigest()


def read_msmarco_train_queries(
    archive_path: Path,
    *,
    expected_sha256: str = MSMARCO_QUERY_ARCHIVE_SHA256,
    member_name: str = MSMARCO_TRAIN_MEMBER,
) -> Iterable[tuple[str, str]]:
    """Yield the official MS MARCO train TSV directly from its pinned archive."""
    if sha256_file(archive_path) != expected_sha256:
        raise ValueError(f"MS MARCO query archive SHA-256 mismatch: {archive_path}")
    with tarfile.open(archive_path, mode="r:gz") as archive:
        member = archive.getmember(member_name)
        extracted = archive.extractfile(member)
        if extracted is None:
            raise ValueError(f"query member is not a regular file: {member_name}")
        seen_ids: set[str] = set()
        for line_number, raw_line in enumerate(extracted, start=1):
            fields = raw_line.decode("utf-8").rstrip("\n").split("\t", maxsplit=1)
            if len(fields) != 2 or not all(fields):
                raise ValueError(
                    f"invalid query row at {archive_path}:{member_name}:{line_number}"
                )
            query_id, query = fields
            if query_id in seen_ids:
                raise ValueError(f"duplicate train query ID: {query_id}")
            seen_ids.add(query_id)
            yield query_id, " ".join(query.split())


def evaluation_identities(
    query_paths: Sequence[Path],
) -> tuple[set[str], set[str]]:
    """Read test query IDs and normalized texts without touching qrels."""
    if not query_paths:
        raise ValueError("at least one evaluation query file is required")
    query_ids: set[str] = set()
    query_texts: set[str] = set()
    for path in query_paths:
        frame = pd.read_parquet(path, columns=["query_id", "query"])
        for row in frame.itertuples(index=False):
            query_ids.add(str(row.query_id))
            query_texts.add(normalize_query(str(row.query)))
    if not query_ids or not query_texts:
        raise ValueError("evaluation query identities must not be empty")
    return query_ids, query_texts


def select_distillation_queries(
    queries: Iterable[tuple[str, str]],
    *,
    evaluation_query_ids: set[str],
    evaluation_query_texts: set[str],
    query_count: int = 1_000,
    validation_count: int = 100,
    seed: int = 42,
) -> tuple[list[DistillationQuery], dict[str, int]]:
    """Select a stable query sample and split it without test-set overlap."""
    if query_count <= 1:
        raise ValueError("query_count must be greater than one")
    if not 0 < validation_count < query_count:
        raise ValueError("validation_count must lie strictly within query_count")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    eligible_by_text: dict[str, tuple[str, str, str]] = {}
    total = 0
    excluded_id = 0
    excluded_text = 0
    duplicate_text = 0
    for query_id, query in queries:
        total += 1
        normalized = normalize_query(query)
        if query_id in evaluation_query_ids:
            excluded_id += 1
            continue
        if normalized in evaluation_query_texts:
            excluded_text += 1
            continue
        digest = _selection_digest("r4-query-sample-v1", seed, query_id, query)
        candidate = (digest, query_id, query)
        previous = eligible_by_text.get(normalized)
        if previous is not None:
            duplicate_text += 1
        if previous is None or candidate < previous:
            eligible_by_text[normalized] = candidate

    selected = heapq.nsmallest(query_count, eligible_by_text.values())
    if len(selected) != query_count:
        raise ValueError(
            f"only {len(selected)} leakage-safe unique queries for requested {query_count}"
        )
    validation_ids = {
        query_id
        for _, query_id, query in heapq.nsmallest(
            validation_count,
            selected,
            key=lambda row: _selection_digest(
                "r4-validation-split-v1", seed, row[1], row[2]
            ),
        )
    }
    records = [
        DistillationQuery(
            query_id=query_id,
            query=query,
            split="validation" if query_id in validation_ids else "train",
            selection_sha256=digest,
        )
        for digest, query_id, query in selected
    ]
    audit = {
        "source_queries": total,
        "excluded_test_query_id": excluded_id,
        "excluded_test_query_text": excluded_text,
        "deduplicated_train_query_text": duplicate_text,
        "eligible_unique_queries": len(eligible_by_text),
        "selected_queries": len(records),
        "train_queries": query_count - validation_count,
        "validation_queries": validation_count,
    }
    return records, audit


def _control_rank(
    namespace: str,
    seed: int,
    query_id: str,
    passage_id: str,
) -> str:
    return hashlib.sha256(
        f"{namespace}\0{seed}\0{query_id}\0{passage_id}".encode()
    ).hexdigest()


def retrieve_distillation_candidates(
    queries: Sequence[DistillationQuery],
    retriever: PassageRetriever,
    *,
    top_k: int = 10,
    random_seed: int = 42,
    progress_callback: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrieve candidates and materialize BM25/random RankNet controls."""
    if not queries:
        raise ValueError("queries must not be empty")
    if top_k <= 1:
        raise ValueError("top_k must be greater than one")
    candidate_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    for query_index, query in enumerate(queries, start=1):
        passages = list(retriever(query.query, top_k))
        if len(passages) != top_k:
            raise ValueError(
                f"retriever returned {len(passages)} candidates for {query.query_id}"
            )
        passage_ids = [passage.passage_id for passage in passages]
        if len(set(passage_ids)) != top_k:
            raise ValueError(f"duplicate passage IDs for query {query.query_id}")
        random_order = sorted(
            passage_ids,
            key=lambda passage_id: _control_rank(
                "r4-random-permutation-v1",
                random_seed,
                query.query_id,
                passage_id,
            ),
        )
        random_ranks = {
            passage_id: rank for rank, passage_id in enumerate(random_order, start=1)
        }
        for bm25_rank, passage in enumerate(passages, start=1):
            candidate_rows.append(
                {
                    "query_id": query.query_id,
                    "split": query.split,
                    "query": query.query,
                    "passage_id": passage.passage_id,
                    "bm25_rank": bm25_rank,
                    "bm25_score": float(passage.score),
                    "passage": " ".join(passage.passage.split()),
                }
            )
            control_rows.append(
                {
                    "query_id": query.query_id,
                    "split": query.split,
                    "passage_id": passage.passage_id,
                    "bm25_teacher_rank": bm25_rank,
                    "bm25_teacher_label": float(top_k - bm25_rank + 1),
                    "random_teacher_rank": random_ranks[passage.passage_id],
                    "random_teacher_label": float(
                        top_k - random_ranks[passage.passage_id] + 1
                    ),
                }
            )
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "retrieve",
                    "done": query_index,
                    "total": len(queries),
                    "query_id": query.query_id,
                }
            )
    return pd.DataFrame(candidate_rows), pd.DataFrame(control_rows)


def _canonical_sha256(records: object) -> str:
    content = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(content).hexdigest()


def write_distillation_dataset(
    output_dir: Path,
    queries: Sequence[DistillationQuery],
    candidates: pd.DataFrame,
    controls: pd.DataFrame,
    *,
    source_archive: Path,
    selection_audit: dict[str, int],
    seed: int,
    top_k: int,
) -> dict[str, object]:
    """Atomically write the frozen R4.0 artifacts and their identities."""
    expected_candidates = len(queries) * top_k
    if len(candidates) != expected_candidates or len(controls) != expected_candidates:
        raise ValueError("candidate/control counts do not match queries times top_k")
    if set(candidates["query_id"]) != {query.query_id for query in queries}:
        raise ValueError("candidate query identities do not match selected queries")
    if candidates.duplicated(["query_id", "passage_id"]).any():
        raise ValueError("duplicate query/passage candidate identity")
    if {"graded_relevance", "binary_relevance", "trec_relevance"} & set(candidates):
        raise ValueError("teacher inputs must not contain evaluation labels")

    output_dir.mkdir(parents=True, exist_ok=True)
    query_frame = pd.DataFrame(asdict(query) for query in queries)
    paths = {
        "queries": output_dir / "queries.parquet",
        "candidates": output_dir / "candidates.parquet",
        "controls": output_dir / "control_labels.parquet",
        "teacher_inputs": output_dir / "teacher_inputs.jsonl",
    }
    query_frame.to_parquet(paths["queries"], index=False)
    candidates.to_parquet(paths["candidates"], index=False)
    controls.to_parquet(paths["controls"], index=False)
    temporary = paths["teacher_inputs"].with_suffix(".jsonl.tmp")
    grouped = {str(query_id): frame for query_id, frame in candidates.groupby("query_id")}
    with temporary.open("w", encoding="utf-8") as output:
        for query in queries:
            frame = grouped[query.query_id].sort_values("bm25_rank")
            payload = {
                "request_id": f"r4-{query.query_id}",
                "query_id": query.query_id,
                "split": query.split,
                "query": query.query,
                "candidates": [
                    {
                        "passage_id": str(row.passage_id),
                        "bm25_rank": int(row.bm25_rank),
                        "bm25_score": float(row.bm25_score),
                        "passage": str(row.passage),
                    }
                    for row in frame.itertuples(index=False)
                ],
            }
            output.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(paths["teacher_inputs"])

    split_identities = {
        split: sorted(
            query.query_id for query in queries if query.split == split
        )
        for split in ("train", "validation")
    }
    manifest: dict[str, object] = {
        "stage": "complete",
        "schema": "r4_instruction_distillation_v1",
        "source": {
            "archive": str(source_archive),
            "archive_sha256": sha256_file(source_archive),
            "member": MSMARCO_TRAIN_MEMBER,
            "license_boundary": "MS MARCO non-commercial research use",
        },
        "selection": {
            "seed": seed,
            "top_k": top_k,
            **selection_audit,
            "split_query_ids_sha256": {
                split: _canonical_sha256(query_ids)
                for split, query_ids in split_identities.items()
            },
        },
        "controls": {
            "vanilla_pointwise": "untrained pretrained checkpoint; no pseudo-label file",
            "bm25": "descending BM25 rank",
            "random": "deterministic per-query random permutation",
            "prp_allpair": "deferred until frozen teacher inference",
        },
        "test_isolation": {
            "qrels_accessed": False,
            "evaluation_query_id_overlap": 0,
            "evaluation_normalized_text_overlap": 0,
        },
        "artifacts": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
    }
    manifest_path = output_dir / "manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    return manifest


def export_prp_teacher_labels(
    summary_path: Path,
    candidates_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """Convert a complete qrels-free Allpair summary into RankNet labels."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("stage") != "inference_complete":
        raise ValueError("PRP teacher summary must be inference_complete")
    if summary.get("qrels_accessed") is not False:
        raise ValueError("PRP teacher summary must prove qrels were not accessed")
    rankings = summary.get("rankings")
    if not isinstance(rankings, list) or not rankings:
        raise ValueError("PRP teacher summary contains no rankings")

    candidates = pd.read_parquet(candidates_path)
    grouped = {
        str(query_id): frame
        for query_id, frame in candidates.groupby("query_id", sort=False)
    }
    rows: list[dict[str, object]] = []
    seen_queries: set[str] = set()
    for ranking in rankings:
        if not isinstance(ranking, dict):
            raise ValueError("invalid PRP ranking payload")
        query_id = str(ranking["query_id"])
        if query_id in seen_queries or query_id not in grouped:
            raise ValueError(f"duplicate or unexpected PRP query: {query_id}")
        seen_queries.add(query_id)
        order = [str(passage_id) for passage_id in ranking["ranking"]]
        expected = set(grouped[query_id]["passage_id"].astype(str))
        if len(order) != len(expected) or set(order) != expected:
            raise ValueError(f"PRP ranking is not an exact candidate permutation: {query_id}")
        raw_scores = ranking.get("scores")
        if not isinstance(raw_scores, dict):
            raise ValueError(f"PRP ranking has no Borda scores: {query_id}")
        split = str(grouped[query_id]["split"].iloc[0])
        size = len(order)
        for rank, passage_id in enumerate(order, start=1):
            rows.append(
                {
                    "query_id": query_id,
                    "split": split,
                    "passage_id": passage_id,
                    "prp_teacher_rank": rank,
                    "prp_teacher_label": float(size - rank + 1),
                    "prp_borda_score": float(raw_scores[passage_id]),
                    "swap_agreement": float(ranking["swap_agreement"]),
                    "tie_ratio": float(ranking["tie_ratio"]),
                }
            )
    if seen_queries != set(grouped):
        missing = sorted(set(grouped) - seen_queries)
        raise ValueError(f"PRP summary is missing {len(missing)} candidate queries")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    temporary.replace(output_path)
    return {
        "stage": "complete",
        "result_type": "qrels-free PRP Allpair RankNet labels",
        "queries": len(seen_queries),
        "candidates": len(rows),
        "qrels_accessed": False,
        "source_summary_sha256": sha256_file(summary_path),
        "source_candidates_sha256": sha256_file(candidates_path),
        "output_sha256": sha256_file(output_path),
        "mean_swap_agreement": sum(
            float(ranking["swap_agreement"]) for ranking in rankings
        )
        / len(rankings),
        "mean_tie_ratio": sum(float(ranking["tie_ratio"]) for ranking in rankings)
        / len(rankings),
    }
