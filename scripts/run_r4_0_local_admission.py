"""Run the no-GPU R4.0 data, leakage, control, and loss admission checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from caged_ltr.data.instruction_distillation import normalize_query
from caged_ltr.distillation import load_text_ranking_groups
from caged_ltr.losses import ranknet_loss
from caged_ltr.models import DEFAULT_DEBERTA_V3_BASE
from caged_ltr.reproducibility import sha256_file
from caged_ltr.teachers.prp_real import load_teacher_inputs


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/r4_msmarco_1k"),
    )
    parser.add_argument(
        "--evaluation-queries",
        type=Path,
        default=Path("data/processed/prp_trec_dl_top100/queries.parquet"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/experiments/r4_0_local_admission.json"),
    )
    args = parser.parse_args()

    manifest = json.loads(
        (args.data_dir / "manifest.json").read_text(encoding="utf-8")
    )
    queries = pd.read_parquet(args.data_dir / "queries.parquet")
    candidates = pd.read_parquet(args.data_dir / "candidates.parquet")
    controls = pd.read_parquet(args.data_dir / "control_labels.parquet")
    evaluation = pd.read_parquet(
        args.evaluation_queries,
        columns=["query_id", "query"],
    )
    teacher_inputs = load_teacher_inputs(args.data_dir / "teacher_inputs.jsonl")
    bm25_groups = load_text_ranking_groups(
        args.data_dir / "candidates.parquet",
        args.data_dir / "control_labels.parquet",
        control="bm25",
    )
    random_groups = load_text_ranking_groups(
        args.data_dir / "candidates.parquet",
        args.data_dir / "control_labels.parquet",
        control="random",
    )

    scores = torch.zeros(2, requires_grad=True)
    loss = ranknet_loss(scores, torch.tensor([2.0, 1.0]), torch.tensor([2]))
    loss.backward()
    gradient_direction_correct = bool(
        scores.grad is not None and scores.grad[0] < 0 and scores.grad[1] > 0
    )
    query_ids = set(queries["query_id"].astype(str))
    evaluation_ids = set(evaluation["query_id"].astype(str))
    query_texts = {normalize_query(query) for query in queries["query"]}
    evaluation_texts = {normalize_query(query) for query in evaluation["query"]}
    group_sizes = candidates.groupby("query_id").size()
    artifact_hashes_match = all(
        sha256_file(Path(artifact["path"])) == artifact["sha256"]
        for artifact in manifest["artifacts"].values()
    )
    bm25_label_direction = bool(
        (
            controls.sort_values(["query_id", "bm25_teacher_rank"])
            .groupby("query_id")["bm25_teacher_label"]
            .apply(lambda values: values.is_monotonic_decreasing)
        ).all()
    )
    random_differs = sum(
        bm25.labels != random.labels
        for bm25, random in zip(bm25_groups, random_groups, strict=True)
    )
    acceptance = {
        "artifact_hashes_match": artifact_hashes_match,
        "query_count_is_1000": len(queries) == 1_000,
        "split_is_900_train_100_validation": queries["split"].value_counts().to_dict()
        == {"train": 900, "validation": 100},
        "train_validation_ids_disjoint": not (
            set(queries.loc[queries["split"] == "train", "query_id"])
            & set(queries.loc[queries["split"] == "validation", "query_id"])
        ),
        "trec_query_id_overlap_zero": not (query_ids & evaluation_ids),
        "trec_query_text_overlap_zero": not (query_texts & evaluation_texts),
        "qrels_not_accessed": manifest["test_isolation"]["qrels_accessed"] is False,
        "all_candidate_groups_are_top10": len(candidates) == 10_000
        and group_sizes.eq(10).all(),
        "control_labels_cover_every_candidate": len(controls) == len(candidates),
        "bm25_label_direction_correct": bm25_label_direction,
        "random_control_is_nonidentity": random_differs > 0,
        "teacher_inputs_are_qrels_free_and_loadable": len(teacher_inputs) == 1_000
        and all(query.year is None and len(query.candidates) == 10 for query in teacher_inputs),
        "ranknet_gradient_direction_correct": gradient_direction_correct,
        "author_pointwise_model_is_deberta": DEFAULT_DEBERTA_V3_BASE
        == "microsoft/deberta-v3-base",
    }
    acceptance = {name: bool(passed) for name, passed in acceptance.items()}
    report = {
        "stage": "complete",
        "result_type": "R4.0 local admission; no teacher or student model inference",
        "data_manifest": str(args.data_dir / "manifest.json"),
        "summary": {
            "queries": len(queries),
            "train_queries": int((queries["split"] == "train").sum()),
            "validation_queries": int((queries["split"] == "validation").sum()),
            "candidates": len(candidates),
            "teacher_ordered_prompts_expected": len(queries) * 10 * 9,
            "random_groups_different_from_bm25": random_differs,
        },
        "acceptance": acceptance,
        "all_acceptance_pass": all(acceptance.values()),
        "next": (
            "Generate the frozen 90,000 ordered FLAN-T5-XL Allpair prompts, "
            "export PRP ranks, then train the three RankNet controls."
        ),
    }
    _write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    if not report["all_acceptance_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
