from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pandas as pd

from caged_ltr.data.instruction_distillation import (
    DistillationQuery,
    RetrievedPassage,
    evaluation_identities,
    export_prp_teacher_labels,
    read_msmarco_train_queries,
    retrieve_distillation_candidates,
    select_distillation_queries,
    write_distillation_dataset,
)


def _query_archive(path: Path, rows: list[tuple[str, str]]) -> str:
    content = "".join(f"{query_id}\t{query}\n" for query_id, query in rows).encode()
    info = tarfile.TarInfo("queries.train.tsv")
    info.size = len(content)
    with tarfile.open(path, mode="w:gz") as archive:
        archive.addfile(info, io.BytesIO(content))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_query_selection_excludes_test_id_and_normalized_text(tmp_path: Path) -> None:
    archive = tmp_path / "queries.tar.gz"
    digest = _query_archive(
        archive,
        [
            ("test-id", "different text"),
            ("2", "TEST   query"),
            ("3", "same train text"),
            ("4", "Same Train Text"),
            ("5", "eligible five"),
            ("6", "eligible six"),
            ("7", "eligible seven"),
        ],
    )

    rows = list(read_msmarco_train_queries(archive, expected_sha256=digest))
    selected, audit = select_distillation_queries(
        rows,
        evaluation_query_ids={"test-id"},
        evaluation_query_texts={"test query"},
        query_count=3,
        validation_count=1,
        seed=42,
    )

    assert len(selected) == 3
    assert {query.split for query in selected} == {"train", "validation"}
    assert "test-id" not in {query.query_id for query in selected}
    assert "2" not in {query.query_id for query in selected}
    assert audit["excluded_test_query_id"] == 1
    assert audit["excluded_test_query_text"] == 1
    assert audit["deduplicated_train_query_text"] == 1


def test_evaluation_identities_need_only_query_columns(tmp_path: Path) -> None:
    path = tmp_path / "queries.parquet"
    pd.DataFrame(
        [{"query_id": "7", "query": "  Mixed CASE query  ", "year": 2020}]
    ).to_parquet(path, index=False)

    query_ids, query_texts = evaluation_identities([path])

    assert query_ids == {"7"}
    assert query_texts == {"mixed case query"}


def test_retrieval_controls_and_manifest_are_label_free(tmp_path: Path) -> None:
    queries = [
        DistillationQuery("1", "query one", "train", "a" * 64),
        DistillationQuery("2", "query two", "validation", "b" * 64),
    ]

    def retrieve(query: str, top_k: int) -> list[RetrievedPassage]:
        assert top_k == 3
        prefix = query.rsplit(maxsplit=1)[-1]
        return [
            RetrievedPassage(f"{prefix}-{index}", 10.0 - index, f"passage {index}")
            for index in range(1, 4)
        ]

    candidates, controls = retrieve_distillation_candidates(
        queries,
        retrieve,
        top_k=3,
        random_seed=42,
    )
    source = tmp_path / "source.tar.gz"
    source.write_bytes(b"source")
    manifest = write_distillation_dataset(
        tmp_path / "processed",
        queries,
        candidates,
        controls,
        source_archive=source,
        selection_audit={
            "source_queries": 2,
            "excluded_test_query_id": 0,
            "excluded_test_query_text": 0,
            "deduplicated_train_query_text": 0,
            "eligible_unique_queries": 2,
            "selected_queries": 2,
            "train_queries": 1,
            "validation_queries": 1,
        },
        seed=42,
        top_k=3,
    )

    assert len(candidates) == 6
    assert controls.groupby("query_id")["random_teacher_rank"].nunique().tolist() == [3, 3]
    assert manifest["test_isolation"]["qrels_accessed"] is False
    teacher_rows = [
        json.loads(line)
        for line in (tmp_path / "processed" / "teacher_inputs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert set(teacher_rows[0]["candidates"][0]) == {
        "passage_id",
        "bm25_rank",
        "bm25_score",
        "passage",
    }


def test_prp_summary_exports_exact_qrels_free_ranknet_labels(tmp_path: Path) -> None:
    candidates_path = tmp_path / "candidates.parquet"
    pd.DataFrame(
        [
            {
                "query_id": "1",
                "split": "train",
                "passage_id": passage_id,
            }
            for passage_id in ("a", "b", "c")
        ]
    ).to_parquet(candidates_path, index=False)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "stage": "inference_complete",
                "qrels_accessed": False,
                "rankings": [
                    {
                        "query_id": "1",
                        "ranking": ["b", "a", "c"],
                        "scores": {"a": 1.0, "b": 2.0, "c": 0.0},
                        "swap_agreement": 1.0,
                        "tie_ratio": 0.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "labels.parquet"

    manifest = export_prp_teacher_labels(
        summary_path,
        candidates_path,
        output_path,
    )

    labels = pd.read_parquet(output_path).sort_values("prp_teacher_rank")
    assert labels["passage_id"].tolist() == ["b", "a", "c"]
    assert labels["prp_teacher_label"].tolist() == [3.0, 2.0, 1.0]
    assert manifest["qrels_accessed"] is False
