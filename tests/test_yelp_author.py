from __future__ import annotations

import zipfile
from pathlib import Path

import duckdb

from caged_ltr.data.yelp_author import YelpAuthorPreparationConfig, prepare_yelp_author


def test_prepare_yelp_author_preserves_sequence_and_does_not_unpickle(tmp_path: Path) -> None:
    archive_path = tmp_path / "author.zip"
    interactions = "1 1\n1 2\n1 3\n1 4\n2 2\n2 3\n2 5\n"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("LLMESR/beauty/inter.txt", "99 99\n")
        archive.writestr("LLMESR/yelp/inter.txt", interactions)
        for name in (
            "itm_emb_np.pkl",
            "pca64_itm_emb_np.pkl",
            "sim_user_100.pkl",
            "usr_emb_np.pkl",
        ):
            archive.writestr(f"LLMESR/beauty/{name}", b"decoy")
            archive.writestr(f"LLMESR/yelp/{name}", b"not-a-trusted-pickle")

    processed = tmp_path / "processed"
    manifest = prepare_yelp_author(
        YelpAuthorPreparationConfig(
            archive=archive_path,
            processed_dir=processed,
            report_path=tmp_path / "report.json",
            memory_limit="1GB",
            threads=1,
        )
    )

    assert manifest["statistics"]["users"] == 2
    assert manifest["statistics"]["items"] == 5
    assert manifest["statistics"]["interactions"] == 7
    assert manifest["statistics"]["train_interactions"] == 3
    assert manifest["statistics"]["valid_interactions"] == 2
    assert manifest["statistics"]["test_interactions"] == 2
    assert manifest["statistics"]["split_invariant_violations"] == 0
    assert all(
        not metadata["unpickled_during_preparation"]
        for metadata in manifest["author_assets"].values()
    )

    connection = duckdb.connect()
    rows = connection.execute(
        "SELECT author_user_id, author_item_id, history_position, split "
        "FROM read_parquet(?) ORDER BY interaction_order",
        [str(processed / "interactions.parquet")],
    ).fetchall()
    connection.close()
    assert rows[:4] == [
        (1, 1, 1, "train"),
        (1, 2, 2, "train"),
        (1, 3, 3, "valid"),
        (1, 4, 4, "test"),
    ]
