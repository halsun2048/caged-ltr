from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from caged_ltr.data.prp_trec_dl import (
    DownloadSpec,
    TRECYearSources,
    download_file,
    prepare_prp_trec_dl,
)


def _write_gzip(path: Path, lines: list[str]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as output:
        output.writelines(lines)


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def _write_snapshot(path: Path, rows: list[dict[str, object]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row) + "\n")


def _source(
    raw_dir: Path,
    *,
    year: int,
    query_lines: list[str],
    top_lines: list[str],
    qrel_lines: list[str],
    judged_queries: int,
) -> TRECYearSources:
    query_name = f"{year}-queries.tsv.gz"
    top_name = f"{year}-top1000.tsv.gz"
    qrel_name = f"{year}-qrels.txt"
    _write_gzip(raw_dir / query_name, query_lines)
    _write_gzip(raw_dir / top_name, top_lines)
    (raw_dir / qrel_name).write_text("".join(qrel_lines), encoding="utf-8")

    def spec(filename: str) -> DownloadSpec:
        path = raw_dir / filename
        return DownloadSpec(
            filename=filename,
            url=f"https://invalid.example/{filename}",
            expected_md5=_md5(path),
        )

    return TRECYearSources(
        year=year,
        queries=spec(query_name),
        top1000=spec(top_name),
        qrels=spec(qrel_name),
        expected_query_records=len(query_lines),
        expected_top1000_records=len(top_lines),
        expected_qrels_records=len(qrel_lines),
        expected_judged_queries=judged_queries,
    )


def test_prepare_prp_trec_dl_from_cached_official_shape_files(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    source_2019 = _source(
        raw_dir,
        year=2019,
        query_lines=["1\tquery one\n", "2\tquery two\n", "3\tunused query\n"],
        top_lines=[
            "1\tp1\tquery one\tpassage one\n",
            "1\tp2\tquery one\tpassage two\n",
            "1\tp3\tquery one\tpassage three\n",
            "2\tp4\tquery two\tpassage four\n",
            "2\tp5\tquery two\tpassage five\n",
            "2\tp6\tquery two\tpassage six\n",
            "3\tp7\tunused query\tpassage seven\n",
        ],
        qrel_lines=[
            "1 Q0 p1 3\n",
            "1 Q0 p2 1\n",
            "2 Q0 p4 2\n",
        ],
        judged_queries=2,
    )
    source_2020 = _source(
        raw_dir,
        year=2020,
        query_lines=["4\tquery four\n", "5\tunused query\n"],
        top_lines=[
            "4\tp8\tquery four\tpassage eight\n",
            "4\tp9\tquery four\tpassage nine\n",
            "4\tp10\tquery four\tpassage ten\n",
            "5\tp11\tunused query\tpassage eleven\n",
        ],
        qrel_lines=["4 Q0 p8 2\n", "4 Q0 missing 0\n"],
        judged_queries=1,
    )
    events: list[dict[str, object]] = []
    processed_dir = tmp_path / "processed"

    manifest = prepare_prp_trec_dl(
        raw_dir,
        processed_dir,
        top_k=2,
        sources=(source_2019, source_2020),
        progress_callback=events.append,
    )

    assert manifest["audit"]["queries"] == 3
    assert manifest["audit"]["top10_candidates"] == 6
    assert manifest["teacher_input_excludes_qrels"] is True
    assert manifest["downloads"]["2019-queries.tsv.gz"]["cached"] is True
    candidates = pd.read_parquet(processed_dir / "candidates.parquet")
    assert candidates["bm25_rank"].tolist() == [1, 2, 1, 2, 1, 2]
    assert candidates["binary_relevance"].tolist() == [1, 0, 1, 0, 1, 0]
    teacher_inputs = [
        json.loads(line)
        for line in (processed_dir / "teacher_inputs.jsonl").read_text().splitlines()
    ]
    assert len(teacher_inputs) == 3
    assert set(teacher_inputs[0]["candidates"][0]) == {
        "passage_id",
        "bm25_rank",
        "passage",
    }
    assert events[-1]["stage"] == "complete"


def test_download_rejects_existing_hash_mismatch(tmp_path) -> None:
    path = tmp_path / "artifact.txt"
    path.write_text("corrupt", encoding="utf-8")
    spec = DownloadSpec(
        filename=path.name,
        url="https://invalid.example/artifact.txt",
        expected_md5="0" * 32,
    )

    with pytest.raises(ValueError, match="MD5 mismatch"):
        download_file(spec, tmp_path)


def test_source_preserves_official_short_candidate_list(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    source = _source(
        raw_dir,
        year=2019,
        query_lines=["1\tquery one\n"],
        top_lines=["1\tp1\tquery one\tpassage one\n"],
        qrel_lines=["1 Q0 p1 3\n"],
        judged_queries=1,
    )

    manifest = prepare_prp_trec_dl(
        raw_dir,
        tmp_path / "processed",
        top_k=2,
        sources=(source,),
    )

    assert manifest["audit"]["top10_candidates"] == 1
    assert manifest["audit"]["candidate_list_size_min"] == 1
    assert manifest["audit"]["short_candidate_lists"] == {"dl2019-1": 1}


def test_snapshot_rank_overrides_unranked_top1000_file_order(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    source = _source(
        raw_dir,
        year=2019,
        query_lines=["1\tquery one\n"],
        top_lines=[
            "1\tp1\tquery one\tpassage one\n",
            "1\tp2\tquery one\tpassage two\n",
        ],
        qrel_lines=["1 Q0 p2 3\n"],
        judged_queries=1,
    )
    snapshot = raw_dir / "bm25.jsonl.gz"
    _write_snapshot(
        snapshot,
        [
            {
                "query_id": "1",
                "passage_id": "p2",
                "bm25_rank": 1,
                "bm25_score": 2.0,
                "passage": "passage two",
            },
            {
                "query_id": "1",
                "passage_id": "p1",
                "bm25_rank": 2,
                "bm25_score": 1.0,
                "passage": "passage one",
            },
        ],
    )

    manifest = prepare_prp_trec_dl(
        raw_dir,
        tmp_path / "processed",
        top_k=2,
        sources=(source,),
        candidate_snapshot=snapshot,
    )

    candidates = pd.read_parquet(tmp_path / "processed" / "candidates.parquet")
    assert candidates["passage_id"].tolist() == ["p2", "p1"]
    assert candidates["bm25_score"].tolist() == [2.0, 1.0]
    assert manifest["candidate_source"]["type"] == "pyserini_bm25_snapshot"
    assert source.top1000.filename not in manifest["downloads"]
