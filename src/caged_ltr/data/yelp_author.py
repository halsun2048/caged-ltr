"""Convert an LLM-ESR author bundle dataset into the common project schema."""

from __future__ import annotations

import json
import math
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from caged_ltr.data.yelp import (
    _combined_fingerprint,
    _copy_parquet,
    _entity_query,
    _extract_member,
    _scalar,
    _sql_path,
)
from caged_ltr.reproducibility import sha256_file

AUTHOR_ASSET_NAMES = (
    "inter.txt",
    "itm_emb_np.pkl",
    "pca64_itm_emb_np.pkl",
    "sim_user_100.pkl",
    "usr_emb_np.pkl",
)
PAPER_REFERENCES: dict[str, dict[str, float | int]] = {
    "yelp": {"users": 15720, "items": 11383, "average_sequence_length": 12.23},
    "fashion": {"users": 9049, "items": 4722},
    "beauty": {"users": 52204, "items": 57289},
}


@dataclass(frozen=True, slots=True)
class YelpAuthorPreparationConfig:
    archive: Path
    processed_dir: Path
    report_path: Path
    dataset_name: str = "yelp"
    tail_quantile: float = 0.2
    head_quantile: float = 0.8
    paper_head_fraction: float = 0.2
    memory_limit: str = "8GB"
    threads: int = 1

    def __post_init__(self) -> None:
        if self.dataset_name not in PAPER_REFERENCES:
            raise ValueError(
                f"dataset_name must be one of {sorted(PAPER_REFERENCES)}"
            )
        if not 0.0 < self.tail_quantile < self.head_quantile < 1.0:
            raise ValueError("bucket quantiles must satisfy 0 < tail < head < 1")
        if not 0.0 < self.paper_head_fraction < 1.0:
            raise ValueError("paper_head_fraction must lie in (0, 1)")
        if self.threads <= 0:
            raise ValueError("threads must be positive")


def _extract_author_assets(
    archive_path: Path,
    destination: Path,
    *,
    dataset_name: str,
) -> dict[str, Path]:
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        return {
            name: _extract_member(archive, f"{dataset_name}/{name}", destination)
            for name in AUTHOR_ASSET_NAMES
        }


def _build_author_tables(
    connection: duckdb.DuckDBPyConnection,
    inter_path: Path,
    processed_dir: Path,
    config: YelpAuthorPreparationConfig,
) -> dict[str, Path]:
    base = processed_dir / "interactions_base.parquet"
    users = processed_dir / "users.parquet"
    items = processed_dir / "items.parquet"
    interactions = processed_dir / "interactions.parquet"
    sequences = processed_dir / "sequences.parquet"

    _copy_parquet(
        connection,
        f"""
            WITH source AS (
                SELECT
                    ROW_NUMBER() OVER () AS interaction_order,
                    user_id::BIGINT AS user_id,
                    item_id::BIGINT AS item_id
                FROM read_csv(
                    '{_sql_path(inter_path)}',
                    delim = ' ',
                    header = false,
                    columns = {{user_id: 'BIGINT', item_id: 'BIGINT'}}
                )
            ),
            positioned AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY user_id ORDER BY interaction_order
                    ) AS history_position,
                    COUNT(*) OVER (PARTITION BY user_id) AS history_length
                FROM source
            )
            SELECT
                interaction_order,
                interaction_order AS interaction_id,
                user_id,
                item_id,
                NULL::TIMESTAMP AS event_time,
                history_position,
                history_length,
                CASE
                    WHEN history_length < 3 THEN 'train'
                    WHEN history_position = history_length THEN 'test'
                    WHEN history_position = history_length - 1 THEN 'valid'
                    ELSE 'train'
                END AS split
            FROM positioned
            ORDER BY interaction_order
        """,
        base,
    )
    _copy_parquet(
        connection,
        _entity_query(
            base,
            "user_id",
            "user_idx",
            config.tail_quantile,
            config.head_quantile,
            config.paper_head_fraction,
        ),
        users,
    )
    _copy_parquet(
        connection,
        _entity_query(
            base,
            "item_id",
            "item_idx",
            config.tail_quantile,
            config.head_quantile,
            config.paper_head_fraction,
        ),
        items,
    )
    _copy_parquet(
        connection,
        f"""
            SELECT
                base.interaction_id,
                base.interaction_order,
                base.user_id AS author_user_id,
                users.user_idx,
                base.item_id AS author_item_id,
                items.item_idx,
                base.history_position,
                base.history_length,
                base.split,
                users.frequency_bucket AS user_frequency_bucket,
                users.paper_bucket AS user_paper_bucket,
                items.frequency_bucket AS item_frequency_bucket,
                items.paper_bucket AS item_paper_bucket,
                items.train_interactions = 0 AS item_is_cold_start
            FROM read_parquet('{_sql_path(base)}') AS base
            JOIN read_parquet('{_sql_path(users)}') AS users USING (user_id)
            JOIN read_parquet('{_sql_path(items)}') AS items USING (item_id)
            ORDER BY base.interaction_order
        """,
        interactions,
    )
    _copy_parquet(
        connection,
        f"""
            SELECT
                author_user_id,
                user_idx,
                ANY_VALUE(user_frequency_bucket) AS user_frequency_bucket,
                ANY_VALUE(user_paper_bucket) AS user_paper_bucket,
                list(item_idx ORDER BY history_position)
                    FILTER (WHERE split = 'train') AS train_item_ids,
                MAX(item_idx) FILTER (WHERE split = 'valid') AS valid_item_id,
                MAX(item_idx) FILTER (WHERE split = 'test') AS test_item_id,
                COUNT(*) FILTER (WHERE split = 'train') AS train_length
            FROM read_parquet('{_sql_path(interactions)}')
            GROUP BY author_user_id, user_idx
            ORDER BY user_idx
        """,
        sequences,
    )
    return {
        "interactions": interactions,
        "users": users,
        "items": items,
        "sequences": sequences,
    }


def _author_statistics(
    connection: duckdb.DuckDBPyConnection,
    outputs: dict[str, Path],
) -> dict[str, Any]:
    interactions = f"read_parquet('{_sql_path(outputs['interactions'])}')"
    items = f"read_parquet('{_sql_path(outputs['items'])}')"
    users = f"read_parquet('{_sql_path(outputs['users'])}')"
    return {
        "interactions": int(_scalar(connection, f"SELECT COUNT(*) FROM {interactions}")),
        "users": int(_scalar(connection, f"SELECT COUNT(*) FROM {users}")),
        "items": int(_scalar(connection, f"SELECT COUNT(*) FROM {items}")),
        "train_interactions": int(
            _scalar(connection, f"SELECT COUNT(*) FROM {interactions} WHERE split='train'")
        ),
        "valid_interactions": int(
            _scalar(connection, f"SELECT COUNT(*) FROM {interactions} WHERE split='valid'")
        ),
        "test_interactions": int(
            _scalar(connection, f"SELECT COUNT(*) FROM {interactions} WHERE split='test'")
        ),
        "evaluable_users": int(
            _scalar(
                connection,
                f"""SELECT COUNT(*) FROM (
                    SELECT author_user_id
                    FROM {interactions}
                    GROUP BY author_user_id
                    HAVING COUNT(*) FILTER (WHERE split='valid') = 1
                        AND COUNT(*) FILTER (WHERE split='test') = 1
                )""",
            )
        ),
        "short_sequence_users": int(
            _scalar(
                connection,
                f"""SELECT COUNT(*) FROM (
                    SELECT author_user_id
                    FROM {interactions}
                    GROUP BY author_user_id
                    HAVING COUNT(*) < 3
                )""",
            )
        ),
        "average_sequence_length": float(
            _scalar(connection, f"SELECT AVG(total_interactions) FROM {users}")
        ),
        "cold_start_items": int(
            _scalar(connection, f"SELECT COUNT(*) FROM {items} WHERE train_interactions=0")
        ),
        "max_author_user_id": int(
            _scalar(connection, f"SELECT MAX(author_user_id) FROM {interactions}")
        ),
        "max_author_item_id": int(
            _scalar(connection, f"SELECT MAX(author_item_id) FROM {interactions}")
        ),
        "split_invariant_violations": int(
            _scalar(
                connection,
                f"""SELECT COUNT(*) FROM (
                    SELECT author_user_id
                    FROM {interactions}
                    GROUP BY author_user_id
                    HAVING (
                        COUNT(*) < 3
                        AND (
                            COUNT(*) FILTER (WHERE split='train') <> COUNT(*)
                            OR COUNT(*) FILTER (WHERE split='valid') <> 0
                            OR COUNT(*) FILTER (WHERE split='test') <> 0
                        )
                    )
                    OR (
                        COUNT(*) >= 3
                        AND (
                            COUNT(*) FILTER (WHERE split='valid') <> 1
                            OR COUNT(*) FILTER (WHERE split='test') <> 1
                            OR COUNT(*) FILTER (WHERE split='train') < 1
                        )
                    )
                )""",
            )
        ),
    }


def prepare_yelp_author(config: YelpAuthorPreparationConfig) -> dict[str, Any]:
    """Extract and convert ordered author interactions without unpickling assets."""
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    assets = _extract_author_assets(
        config.archive,
        config.processed_dir / "author_assets",
        dataset_name=config.dataset_name,
    )
    connection = duckdb.connect()
    connection.execute("SET preserve_insertion_order = true")
    connection.execute(f"SET memory_limit = '{config.memory_limit}'")
    connection.execute(f"SET threads = {config.threads}")
    try:
        outputs = _build_author_tables(
            connection,
            assets["inter.txt"],
            config.processed_dir,
            config,
        )
        statistics = _author_statistics(connection, outputs)
    finally:
        connection.close()

    output_hashes = {name: sha256_file(path) for name, path in outputs.items()}
    asset_hashes = {name: sha256_file(path) for name, path in assets.items()}
    paper_reference = PAPER_REFERENCES[config.dataset_name]
    paper_reference_match = {
        key: (
            statistics[key] == value
            if isinstance(value, int)
            else math.isclose(statistics[key], value, abs_tol=0.005)
        )
        for key, value in paper_reference.items()
    }
    manifest: dict[str, Any] = {
        "pipeline_version": (
            "yelp-llmesr-author-v1"
            if config.dataset_name == "yelp"
            else "llmesr-author-v1"
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_name": config.dataset_name,
        "dataset_variant": (
            "faithful_author_processed"
            if config.dataset_name == "yelp"
            else f"{config.dataset_name}_faithful_author_processed"
        ),
        "source": {
            "repository": "https://github.com/liuqidong07/LLM-ESR",
            "archive_url": (
                "https://drive.google.com/file/d/1MpBUjCDLiFIEODTnopSCzDAnS8RzO9aV/view"
            ),
            "archive_bytes": config.archive.stat().st_size,
            "archive_sha256": sha256_file(config.archive),
        },
        "config": {
            **asdict(config),
            "archive": str(config.archive),
            "processed_dir": str(config.processed_dir),
            "report_path": str(config.report_path),
        },
        "statistics": statistics,
        "paper_reference": paper_reference,
        "paper_reference_match": paper_reference_match,
        "outputs": {
            name: {"path": str(path), "bytes": path.stat().st_size, "sha256": output_hashes[name]}
            for name, path in outputs.items()
        },
        "author_assets": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": asset_hashes[name],
                "unpickled_during_preparation": False,
            }
            for name, path in assets.items()
        },
        "processed_fingerprint": _combined_fingerprint({**output_hashes, **asset_hashes}),
        "limitations": {
            "raw_id_mapping": "not included in the author bundle",
            "event_timestamps": "not included; author sequence order is preserved",
            "short_sequence_split": (
                "users with fewer than three interactions remain train-only, "
                "matching the author loader"
            ),
            "pickle_security": "external pickle assets were hashed but not deserialized",
            "paper_statistic_mismatch": (
                "none"
                if all(paper_reference_match.values())
                else "author-bundle statistics differ from the published dataset table"
            ),
            "user_embedding_leakage": (
                "provenance cutoff is not established; do not use until separately audited"
            ),
        },
    }
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    config.report_path.write_text(serialized, encoding="utf-8")
    (config.processed_dir / "manifest.json").write_text(serialized, encoding="utf-8")
    return manifest


LLMESRAuthorPreparationConfig = YelpAuthorPreparationConfig
prepare_llmesr_author = prepare_yelp_author
