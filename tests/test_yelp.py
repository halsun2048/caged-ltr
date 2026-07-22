from __future__ import annotations

import io
import json
import tarfile
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import duckdb

from caged_ltr.data.yelp import YelpPreparationConfig, extract_yelp_sources, prepare_yelp


def _write_fixture_archive(path: Path) -> None:
    start = datetime(2020, 1, 1)
    user_items = {
        "u1": ["a", "b", "c", "d"],
        "u2": ["e", "a", "f"],
        "u3-filtered": ["a", "b"],
        "u4": ["g", "h", "i"],
    }
    reviews: list[dict[str, object]] = []
    for user_offset, (user_id, items) in enumerate(user_items.items()):
        for position, item_id in enumerate(items):
            review_id = f"{user_id}-r{position + 1}"
            reviews.append(
                {
                    "review_id": review_id,
                    "user_id": user_id,
                    "business_id": item_id,
                    "stars": 5.0,
                    "useful": position,
                    "funny": 0,
                    "cool": 0,
                    "text": f"training-safe fixture text {review_id}",
                    "date": str(start + timedelta(days=user_offset * 10 + position)),
                }
            )
    businesses = [
        {
            "business_id": item_id,
            "name": f"Business {item_id}",
            "city": "Test City",
            "state": "TS",
            "postal_code": "00000",
            "latitude": 0.0,
            "longitude": 0.0,
            "stars": 4.0,
            "review_count": 10,
            "is_open": 1,
            "categories": "Test",
            "attributes": {"WiFi": "free"},
            "hours": {"Monday": "09:00-17:00"},
        }
        for item_id in "abcdefghi"
    ]
    review_lines = "\n".join(json.dumps(review) for review in reviews) + "\n"
    business_lines = "\n".join(json.dumps(business) for business in businesses) + "\n"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("Yelp JSON/yelp_academic_dataset_review.json", review_lines)
        archive.writestr("Yelp JSON/yelp_academic_dataset_business.json", business_lines)


def test_extract_yelp_sources_supports_official_nested_archive(tmp_path: Path) -> None:
    nested_buffer = io.BytesIO()
    contents = {
        "yelp_academic_dataset_review.json": b'{"review_id":"r1"}\n',
        "yelp_academic_dataset_business.json": b'{"business_id":"b1"}\n',
    }
    with tarfile.open(fileobj=nested_buffer, mode="w:gz") as nested_archive:
        for name, content in contents.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            nested_archive.addfile(member, io.BytesIO(content))
    archive_path = tmp_path / "nested.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("Yelp JSON/yelp_dataset.tar", nested_buffer.getvalue())

    extracted = extract_yelp_sources(archive_path, tmp_path / "source")

    assert extracted["review_json"].read_bytes() == contents["yelp_academic_dataset_review.json"]
    assert extracted["business_json"].read_bytes() == contents[
        "yelp_academic_dataset_business.json"
    ]


def test_prepare_yelp_enforces_leave_two_out_and_train_only_text(tmp_path: Path) -> None:
    archive = tmp_path / "Yelp-JSON.zip"
    _write_fixture_archive(archive)
    processed = tmp_path / "processed"
    report = tmp_path / "report.json"

    manifest = prepare_yelp(
        YelpPreparationConfig(
            archive=archive,
            interim_dir=tmp_path / "interim",
            processed_dir=processed,
            report_path=report,
            min_item_interactions=1,
            event_time_max="2025-01-01 00:00:00",
            memory_limit="1GB",
            threads=1,
        )
    )

    statistics = manifest["statistics"]
    assert statistics["raw_reviews"] == 12
    assert statistics["eligible_interactions"] == 10
    assert statistics["users"] == 3
    assert statistics["train_interactions"] == 4
    assert statistics["valid_interactions"] == 3
    assert statistics["test_interactions"] == 3
    assert statistics["sequence_rows"] == 3
    assert statistics["profile_train_rows"] == 4
    assert statistics["split_invariant_violations"] == 0
    assert statistics["cold_start_items"] == 5
    assert statistics["missing_business_snapshots"] == 0
    assert statistics["cold_start_valid_interactions"] == 2
    assert statistics["cold_start_test_interactions"] == 3

    connection = duckdb.connect()
    interactions = connection.execute(
        "SELECT review_id, user_id, split, history_position, history_length "
        "FROM read_parquet(?) ORDER BY user_id, history_position",
        [str(processed / "interactions.parquet")],
    ).fetchall()
    profile_review_ids = {
        row[0]
        for row in connection.execute(
            "SELECT review_id FROM read_parquet(?)",
            [str(processed / "profile_reviews_train.parquet")],
        ).fetchall()
    }
    sequence_rows = connection.execute(
        "SELECT train_length, valid_event_timestamp, test_event_timestamp "
        "FROM read_parquet(?)",
        [str(processed / "sequences.parquet")],
    ).fetchall()
    connection.close()

    assert all(
        split == ("test" if position == length else "valid" if position == length - 1 else "train")
        for _, _, split, position, length in interactions
    )
    train_review_ids = {review_id for review_id, _, split, _, _ in interactions if split == "train"}
    held_out_review_ids = {
        review_id for review_id, _, split, _, _ in interactions if split != "train"
    }
    assert profile_review_ids == train_review_ids
    assert profile_review_ids.isdisjoint(held_out_review_ids)
    assert sorted(row[0] for row in sequence_rows) == [1, 1, 2]
    assert all(row[1] < row[2] for row in sequence_rows)
    assert report.exists()
    assert manifest["processed_fingerprint"]


def test_prepare_yelp_is_deterministic(tmp_path: Path) -> None:
    archive = tmp_path / "Yelp-JSON.zip"
    _write_fixture_archive(archive)
    fingerprints = []
    for run in ("first", "second"):
        manifest = prepare_yelp(
            YelpPreparationConfig(
                archive=archive,
                interim_dir=tmp_path / run / "interim",
                processed_dir=tmp_path / run / "processed",
                report_path=tmp_path / run / "report.json",
                min_item_interactions=1,
                event_time_max="2025-01-01 00:00:00",
                memory_limit="1GB",
                threads=1,
            )
        )
        fingerprints.append(manifest["processed_fingerprint"])

    assert fingerprints[0] == fingerprints[1]
