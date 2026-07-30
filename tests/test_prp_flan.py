from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from caged_ltr.teachers import (
    AUTHOR_PRP_PROMPT,
    FlanT5PairwiseTeacher,
    OrderedPairRequest,
    TeacherMetadata,
    TeacherResponse,
    parse_prp_generation,
)
from caged_ltr.teachers.prp import prompt_sha256
from caged_ltr.teachers.prp_real import load_teacher_inputs, run_prp_r3_1b


class _FakeTokenizer:
    def __call__(
        self,
        texts: str | list[str],
        *,
        add_special_tokens: bool,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
    ) -> dict[str, object]:
        del add_special_tokens, truncation
        values = [texts] if isinstance(texts, str) else texts
        if values == ["Passage A", "Passage B"]:
            ids = [[3, 1], [4, 1]]
        elif values == ["Passage"]:
            ids = [[2]]
        else:
            ids = [
                [5, 6, 7, 8, 1][:max_length]
                if max_length is not None
                else [5, 6, 7, 8, 1]
                for _ in values
            ]
        masks = [[1] * len(row) for row in ids]
        if padding:
            width = max(len(row) for row in ids)
            ids = [row + [0] * (width - len(row)) for row in ids]
            masks = [row + [0] * (width - len(row)) for row in masks]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(ids),
                "attention_mask": torch.tensor(masks),
            }
        return {"input_ids": ids, "attention_mask": masks}

    def batch_decode(
        self,
        output_ids: torch.Tensor,
        *,
        skip_special_tokens: bool,
    ) -> list[str]:
        del skip_special_tokens
        return ["Passage A" for _ in output_ids]


class _FakeLikelihoodModel:
    config = SimpleNamespace(decoder_start_token_id=0)

    def get_encoder(self) -> _FakeLikelihoodModel:
        return self

    def __call__(self, **kwargs: torch.Tensor) -> SimpleNamespace:
        if "labels" not in kwargs:
            input_ids = kwargs["input_ids"]
            return SimpleNamespace(
                last_hidden_state=torch.zeros(
                    (*input_ids.shape, 4),
                    dtype=torch.float32,
                )
            )
        labels = kwargs["labels"]
        logits = torch.zeros((*labels.shape, 8), dtype=torch.float32)
        for row in range(labels.shape[0]):
            preferred = 3 if row % 2 == 0 else 4
            logits[row, 0, preferred] = 4.0 if row % 2 == 0 else 1.0
            logits[row, 1, 1] = 2.0
        return SimpleNamespace(logits=logits)


class _DeterministicBatchTeacher:
    def __init__(self, revision: str = "fixture-r1") -> None:
        self.calls = 0
        self._metadata = TeacherMetadata(
            backend="fixture",
            model_name="fixture",
            model_revision=revision,
            tokenizer_name="fixture",
            tokenizer_revision=revision,
            quantization="not_applicable",
            prompt_name="fixture",
            prompt_version="1",
            prompt_sha256=prompt_sha256(AUTHOR_PRP_PROMPT),
            generation_parameters={"mode": "fixture"},
        )

    @property
    def metadata(self) -> TeacherMetadata:
        return self._metadata

    def compare_many(
        self,
        requests: list[OrderedPairRequest],
    ) -> list[TeacherResponse]:
        self.calls += 1
        responses = []
        for request in requests:
            first_wins = request.first_id < request.second_id
            responses.append(
                TeacherResponse(
                    choice="first" if first_wins else "second",
                    input_tokens=10,
                    output_tokens=2,
                    latency_ms=1.0,
                    raw_output="Passage A" if first_wins else "Passage B",
                    score_first=0.0 if first_wins else -1.0,
                    score_second=-1.0 if first_wins else 0.0,
                )
            )
        return responses


def _write_fixture(root: Path) -> tuple[Path, Path]:
    teacher_input = root / "teacher_inputs.jsonl"
    rows = []
    qrels = []
    for query_index in range(2):
        request_id = f"dl2019-q{query_index}"
        candidates = [
            {
                "passage_id": f"d{candidate_index}",
                "bm25_rank": candidate_index + 1,
                "passage": (
                    f"passage {candidate_index}"
                    if candidate_index
                    else "â\u0098\u0085 durable equipment"
                ),
            }
            for candidate_index in range(3)
        ]
        rows.append(
            {
                "request_id": request_id,
                "year": 2019,
                "query_id": f"q{query_index}",
                "query": "fixture query",
                "candidates": candidates,
            }
        )
        qrels.extend(
            {
                "request_id": request_id,
                "year": 2019,
                "query_id": f"q{query_index}",
                "passage_id": f"d{candidate_index}",
                "graded_relevance": 3 - candidate_index,
                "trec_relevance": 3 - candidate_index,
                "binary_relevance": int(candidate_index < 2),
            }
            for candidate_index in range(3)
        )
    teacher_input.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    qrels_path = root / "qrels.parquet"
    pd.DataFrame(qrels).to_parquet(qrels_path, index=False)
    return teacher_input, qrels_path


def test_author_prompt_and_strict_generation_parser() -> None:
    request = OrderedPairRequest(
        request_id="r1",
        query_id="q1",
        query="query",
        first_id="a",
        first="first",
        second_id="b",
        second="second",
    )
    assert request.prompt().startswith('Given a query "query"')
    assert parse_prp_generation(" Passage   A\n") == "first"
    assert parse_prp_generation("Passage B") == "second"
    assert parse_prp_generation("A") == "tie"
    assert parse_prp_generation("Passage A because it is relevant") == "tie"
    with pytest.raises(ValueError, match="must differ"):
        OrderedPairRequest("r1", "q1", "q", "a", "x", "a", "y")


def test_likelihood_backend_batches_two_targets_without_model_download() -> None:
    teacher = FlanT5PairwiseTeacher(
        model=_FakeLikelihoodModel(),
        tokenizer=_FakeTokenizer(),
        model_name="fixture",
        model_revision="revision",
        tokenizer_name="fixture",
        tokenizer_revision="revision",
        device="cpu",
        dtype="float32",
        scoring_mode="likelihood",
        max_input_tokens=4,
    )
    response = teacher.compare_many(
        (
            OrderedPairRequest("r1", "q1", "q", "a", "first", "b", "second"),
        )
    )[0]

    assert response.choice == "first"
    assert response.score_first > response.score_second
    assert response.input_truncated


def test_real_runner_admission_resume_and_delayed_qrels_access(tmp_path) -> None:
    teacher_input, qrels_path = _write_fixture(tmp_path)
    output_dir = tmp_path / "run"
    progress: list[dict[str, object]] = []
    teacher = _DeterministicBatchTeacher()
    admission = run_prp_r3_1b(
        teacher,
        teacher_input_path=teacher_input,
        output_dir=output_dir,
        qrels_path=tmp_path / "not-yet-accessed.parquet",
        query_limit=2,
        batch_size=3,
        max_ordered_prompts=4,
        progress_callback=lambda event: progress.append(dict(event)),
    )

    assert admission["stage"] == "admission_complete"
    assert admission["cached_ordered_prompts"] == 4
    assert admission["qrels_accessed"] is False
    assert progress[-1]["prompt_done"] == 4

    complete = run_prp_r3_1b(
        teacher,
        teacher_input_path=teacher_input,
        output_dir=output_dir,
        qrels_path=qrels_path,
        query_limit=2,
        batch_size=3,
    )
    calls_after_completion = teacher.calls
    cached = run_prp_r3_1b(
        teacher,
        teacher_input_path=teacher_input,
        output_dir=output_dir,
        qrels_path=qrels_path,
        query_limit=2,
        batch_size=3,
    )

    assert complete["stage"] == "complete"
    assert complete["cached_ordered_prompts"] == 12
    assert complete["diagnostics"]["mean_swap_agreement"] == 1.0
    assert complete["evaluation"]["overall"]["absolute_gain"] == pytest.approx(0.0)
    assert cached["newly_scored_ordered_prompts"] == 0
    assert teacher.calls == calls_after_completion
    response_lines = []
    for path in sorted(
        (output_dir / "ordered_pair_responses").glob("*.jsonl")
    ):
        with path.open(encoding="utf-8") as response_file:
            response_lines.extend(response_file)
    assert len(response_lines) == 12
    assert "graded_relevance" not in response_lines[0]
    assert "prompt" not in json.loads(response_lines[0])
    assert complete["cache_layout"] == "per_query_compact_jsonl_v2"
    assert complete["cache_shards"] == 2
    assert complete["batch_order"] == "input"
    assert complete["exact_likelihood_ties"] == 0
    assert complete["invalid_outputs"] == 0

    with pytest.raises(ValueError, match="identity mismatch"):
        run_prp_r3_1b(
            _DeterministicBatchTeacher(revision="different"),
            teacher_input_path=teacher_input,
            output_dir=output_dir,
            qrels_path=qrels_path,
            query_limit=2,
            batch_size=3,
        )


def test_real_runner_can_batch_pending_requests_by_length(tmp_path) -> None:
    teacher_input, qrels_path = _write_fixture(tmp_path)
    output_dir = tmp_path / "length-ordered"
    teacher = _DeterministicBatchTeacher()

    summary = run_prp_r3_1b(
        teacher,
        teacher_input_path=teacher_input,
        output_dir=output_dir,
        qrels_path=qrels_path,
        query_limit=2,
        batch_size=3,
        batch_order="length",
    )

    assert summary["stage"] == "complete"
    assert summary["batch_order"] == "length"
    assert summary["cached_ordered_prompts"] == 12


def test_real_runner_can_defer_qrels_until_shards_are_merged(tmp_path) -> None:
    teacher_input, _ = _write_fixture(tmp_path)
    summary = run_prp_r3_1b(
        _DeterministicBatchTeacher(),
        teacher_input_path=teacher_input,
        output_dir=tmp_path / "deferred",
        qrels_path=tmp_path / "must-not-be-read.parquet",
        query_limit=2,
        batch_size=3,
        evaluate_when_complete=False,
    )

    assert summary["stage"] == "inference_complete"
    assert summary["cached_ordered_prompts"] == 12
    assert summary["qrels_accessed"] is False
    assert "evaluation" not in summary
    assert summary["diagnostics"]["mean_swap_agreement"] == 1.0


def test_teacher_input_rejects_evaluation_fields(tmp_path) -> None:
    path = tmp_path / "leaky.jsonl"
    path.write_text(
        json.dumps(
            {
                "request_id": "r1",
                "year": 2019,
                "query_id": "q1",
                "query": "q",
                "candidates": [
                    {
                        "passage_id": "a",
                        "bm25_rank": 1,
                        "passage": "a",
                        "graded_relevance": 3,
                    },
                    {"passage_id": "b", "bm25_rank": 2, "passage": "b"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="evaluation-only"):
        load_teacher_inputs(path)
