"""Leakage-aware preparation of the current official Yelp Open Dataset."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from caged_ltr.reproducibility import sha256_file

PIPELINE_VERSION = "yelp-current-v1"


@dataclass(frozen=True, slots=True)
class YelpPreparationConfig:
    archive: Path
    interim_dir: Path
    processed_dir: Path
    report_path: Path
    min_user_interactions: int = 3
    min_item_interactions: int = 3
    event_time_min: str = "2000-01-01 00:00:00"
    event_time_max: str = "2019-12-31 00:00:00"
    rating_min_exclusive: float = 0.0
    tail_quantile: float = 0.2
    head_quantile: float = 0.8
    paper_head_fraction: float = 0.2
    memory_limit: str = "24GB"
    threads: int = 8

    def __post_init__(self) -> None:
        if self.min_user_interactions < 3:
            raise ValueError("leave-two-out requires min_user_interactions >= 3")
        if self.min_item_interactions < 1:
            raise ValueError("min_item_interactions must be positive")
        if self.event_time_min > self.event_time_max:
            raise ValueError("event_time_min must not exceed event_time_max")
        if not 0.0 < self.tail_quantile < self.head_quantile < 1.0:
            raise ValueError("bucket quantiles must satisfy 0 < tail < head < 1")
        if not 0.0 < self.paper_head_fraction < 1.0:
            raise ValueError("paper_head_fraction must lie in (0, 1)")
        if self.threads <= 0:
            raise ValueError("threads must be positive")


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _extract_member(archive: zipfile.ZipFile, suffix: str, destination: Path) -> Path:
    matches = [
        member
        for member in archive.infolist()
        if (
            member.filename.rstrip("/").endswith(suffix)
            if "/" in suffix
            else Path(member.filename).name == suffix
        )
    ]
    if len(matches) != 1:
        message = f"expected one archive member ending in {suffix!r}, found {len(matches)}"
        raise ValueError(message)
    member = matches[0]
    output = destination / Path(member.filename).name
    if output.exists() and output.stat().st_size == member.file_size:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    with archive.open(member) as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    temporary.replace(output)
    return output


def _extract_nested_tar_sources(
    archive: zipfile.ZipFile,
    source_dir: Path,
) -> dict[str, Path]:
    tar_members = [
        member
        for member in archive.infolist()
        if Path(member.filename).name == "yelp_dataset.tar"
    ]
    if len(tar_members) != 1:
        raise ValueError("archive contains neither direct Yelp JSON nor one yelp_dataset.tar")
    expected = {
        "yelp_academic_dataset_review.json": source_dir / "yelp_academic_dataset_review.json",
        "yelp_academic_dataset_business.json": source_dir / "yelp_academic_dataset_business.json",
    }
    source_dir.mkdir(parents=True, exist_ok=True)
    remaining = {name for name, output in expected.items() if not output.exists()}
    if remaining:
        with (
            archive.open(tar_members[0]) as nested_stream,
            tarfile.open(fileobj=nested_stream, mode="r|*") as nested_archive,
        ):
            for member in nested_archive:
                basename = Path(member.name).name
                if basename not in remaining or not member.isfile():
                    continue
                extracted = nested_archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"could not read nested TAR member {member.name}")
                output = expected[basename]
                temporary = output.with_suffix(output.suffix + ".partial")
                with extracted, temporary.open("wb") as target:
                    shutil.copyfileobj(extracted, target, length=8 * 1024 * 1024)
                if temporary.stat().st_size != member.size:
                    raise ValueError(f"size mismatch while extracting {member.name}")
                temporary.replace(output)
                remaining.remove(basename)
                if not remaining:
                    break
    if remaining:
        raise ValueError(f"nested TAR is missing required files: {sorted(remaining)}")
    return {
        "review_json": expected["yelp_academic_dataset_review.json"],
        "business_json": expected["yelp_academic_dataset_business.json"],
    }


def extract_yelp_sources(archive_path: Path, source_dir: Path) -> dict[str, Path]:
    """Extract only review/business JSON and the agreement, validating ZIP CRC while reading."""
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        direct_review = [
            member
            for member in archive.infolist()
            if member.filename.endswith("yelp_academic_dataset_review.json")
        ]
        direct_business = [
            member
            for member in archive.infolist()
            if member.filename.endswith("yelp_academic_dataset_business.json")
        ]
        if direct_review and direct_business:
            paths = {
                "review_json": _extract_member(
                    archive, "yelp_academic_dataset_review.json", source_dir
                ),
                "business_json": _extract_member(
                    archive, "yelp_academic_dataset_business.json", source_dir
                ),
            }
        else:
            paths = _extract_nested_tar_sources(archive, source_dir)
        pdf_members = [
            member for member in archive.infolist() if member.filename.lower().endswith(".pdf")
        ]
        agreement = None
        if pdf_members:
            agreement = _extract_member(archive, Path(pdf_members[0].filename).name, source_dir)
    if agreement is not None:
        paths["agreement_pdf"] = agreement
    return paths


def _copy_parquet(connection: duckdb.DuckDBPyConnection, query: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    connection.execute(
        f"COPY ({query}) TO '{_sql_path(output)}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
    )


def _stage_reviews(
    connection: duckdb.DuckDBPyConnection,
    review_json: Path,
    output: Path,
) -> None:
    query = f"""
        SELECT
            review_id::VARCHAR AS review_id,
            user_id::VARCHAR AS user_id,
            business_id::VARCHAR AS item_id,
            stars::DOUBLE AS stars,
            useful::BIGINT AS useful,
            funny::BIGINT AS funny,
            cool::BIGINT AS cool,
            text::VARCHAR AS text,
            date::TIMESTAMP AS event_time
        FROM read_json(
            '{_sql_path(review_json)}',
            format = 'newline_delimited',
            columns = {{
                review_id: 'VARCHAR', user_id: 'VARCHAR', business_id: 'VARCHAR',
                stars: 'DOUBLE', useful: 'BIGINT', funny: 'BIGINT', cool: 'BIGINT',
                text: 'VARCHAR', date: 'TIMESTAMP'
            }}
        )
        WHERE review_id IS NOT NULL
          AND user_id IS NOT NULL
          AND business_id IS NOT NULL
          AND date IS NOT NULL
    """
    _copy_parquet(connection, query, output)


def _stage_businesses(
    connection: duckdb.DuckDBPyConnection,
    business_json: Path,
    output: Path,
) -> None:
    query = f"""
        SELECT
            business_id::VARCHAR AS item_id,
            name::VARCHAR AS name,
            city::VARCHAR AS city,
            state::VARCHAR AS state,
            postal_code::VARCHAR AS postal_code,
            latitude::DOUBLE AS latitude,
            longitude::DOUBLE AS longitude,
            stars::DOUBLE AS stars_snapshot,
            review_count::BIGINT AS review_count_snapshot,
            is_open::INTEGER AS is_open_snapshot,
            categories::VARCHAR AS categories,
            attributes::JSON AS attributes_json,
            hours::JSON AS hours_json
        FROM read_json(
            '{_sql_path(business_json)}',
            format = 'newline_delimited',
            columns = {{
                business_id: 'VARCHAR', name: 'VARCHAR', city: 'VARCHAR', state: 'VARCHAR',
                postal_code: 'VARCHAR', latitude: 'DOUBLE', longitude: 'DOUBLE', stars: 'DOUBLE',
                review_count: 'BIGINT', is_open: 'INTEGER', categories: 'VARCHAR',
                attributes: 'JSON', hours: 'JSON'
            }}
        )
        WHERE business_id IS NOT NULL
    """
    _copy_parquet(connection, query, output)


def _build_base_interactions(
    connection: duckdb.DuckDBPyConnection,
    staged_reviews: Path,
    output: Path,
    min_user_interactions: int,
    min_item_interactions: int,
    event_time_min: str,
    event_time_max: str,
    rating_min_exclusive: float,
) -> None:
    query = f"""
        WITH source_window AS (
            SELECT *
            FROM read_parquet('{_sql_path(staged_reviews)}')
            WHERE event_time >= TIMESTAMP '{event_time_min}'
              AND event_time <= TIMESTAMP '{event_time_max}'
              AND stars > {float(rating_min_exclusive)}
        ),
        raw_counts AS (
            SELECT
                review_id, user_id, item_id, event_time, stars, useful, funny, cool,
                COUNT(*) OVER (PARTITION BY user_id) AS raw_user_interactions,
                COUNT(*) OVER (PARTITION BY item_id) AS raw_item_interactions
            FROM source_window
        ),
        common_filtered AS (
            SELECT *
            FROM raw_counts
            WHERE raw_user_interactions >= {int(min_user_interactions)}
              AND raw_item_interactions >= {int(min_item_interactions)}
        ),
        recounted AS (
            SELECT
                *,
                COUNT(*) OVER (PARTITION BY user_id) AS history_length
            FROM common_filtered
        ),
        ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id ORDER BY event_time ASC, review_id ASC
                ) AS history_position
            FROM recounted
            WHERE history_length >= {int(min_user_interactions)}
        )
        SELECT
            review_id, user_id, item_id, event_time, stars, useful, funny, cool,
            history_position, history_length,
            CASE
                WHEN history_position = history_length THEN 'test'
                WHEN history_position = history_length - 1 THEN 'valid'
                ELSE 'train'
            END AS split
        FROM ranked
        ORDER BY user_id ASC, history_position ASC
    """
    _copy_parquet(connection, query, output)


def _entity_query(
    interactions: Path,
    entity_column: str,
    index_column: str,
    tail_quantile: float,
    head_quantile: float,
    paper_head_fraction: float,
) -> str:
    return f"""
        WITH counts AS (
            SELECT
                {entity_column},
                COUNT(*) AS total_interactions,
                COUNT(*) FILTER (WHERE split = 'train') AS train_interactions,
                MIN(event_time) AS first_event_time,
                MAX(event_time) AS last_event_time
            FROM read_parquet('{_sql_path(interactions)}')
            GROUP BY {entity_column}
        ),
        positive_thresholds AS (
            SELECT
                quantile_disc(train_interactions, {float(tail_quantile)})
                    FILTER (WHERE train_interactions > 0) AS tail_max,
                quantile_disc(train_interactions, {float(head_quantile)})
                    FILTER (WHERE train_interactions > 0) AS head_min
            FROM counts
        ),
        frequency_ranked AS (
            SELECT
                counts.*,
                ROW_NUMBER() OVER (
                    ORDER BY train_interactions DESC, {entity_column} ASC
                ) AS frequency_rank,
                COUNT(*) OVER () AS entity_count
            FROM counts
        )
        SELECT
            {entity_column},
            ROW_NUMBER() OVER (ORDER BY {entity_column} ASC) - 1 AS {index_column},
            total_interactions,
            train_interactions,
            first_event_time,
            last_event_time,
            CASE
                WHEN train_interactions = 0 THEN 'cold_start'
                WHEN tail_max >= head_min THEN 'torso'
                WHEN train_interactions <= tail_max THEN 'tail'
                WHEN train_interactions >= head_min THEN 'head'
                ELSE 'torso'
            END AS frequency_bucket,
            CASE
                WHEN train_interactions = 0 THEN 'cold_start'
                WHEN frequency_rank <= CEIL(entity_count * {float(paper_head_fraction)}) THEN 'head'
                ELSE 'tail'
            END AS paper_bucket,
            tail_max AS frequency_tail_max,
            head_min AS frequency_head_min
        FROM frequency_ranked
        CROSS JOIN positive_thresholds
        ORDER BY {entity_column} ASC
    """


def _build_outputs(
    connection: duckdb.DuckDBPyConnection,
    staged_reviews: Path,
    staged_businesses: Path,
    base_interactions: Path,
    processed_dir: Path,
    config: YelpPreparationConfig,
) -> dict[str, Path]:
    users = processed_dir / "users.parquet"
    item_counts = config.interim_dir / "item_counts.parquet"
    items = processed_dir / "items.parquet"
    interactions = processed_dir / "interactions.parquet"
    sequences = processed_dir / "sequences.parquet"
    profile_reviews = processed_dir / "profile_reviews_train.parquet"

    _copy_parquet(
        connection,
        _entity_query(
            base_interactions,
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
            base_interactions,
            "item_id",
            "item_idx",
            config.tail_quantile,
            config.head_quantile,
            config.paper_head_fraction,
        ),
        item_counts,
    )
    _copy_parquet(
        connection,
        f"""
            SELECT
                counts.*,
                business.name,
                business.city,
                business.state,
                business.postal_code,
                business.latitude,
                business.longitude,
                business.stars_snapshot,
                business.review_count_snapshot,
                business.is_open_snapshot,
                business.categories,
                business.attributes_json,
                business.hours_json,
                CASE WHEN business.item_id IS NULL THEN FALSE ELSE TRUE END AS has_business_snapshot
            FROM read_parquet('{_sql_path(item_counts)}') AS counts
            LEFT JOIN read_parquet('{_sql_path(staged_businesses)}') AS business USING (item_id)
            ORDER BY counts.item_idx ASC
        """,
        items,
    )
    _copy_parquet(
        connection,
        f"""
            SELECT
                base.review_id,
                base.user_id,
                users.user_idx,
                base.item_id,
                items.item_idx,
                base.event_time,
                base.stars,
                base.useful,
                base.funny,
                base.cool,
                base.history_position,
                base.history_length,
                base.split,
                users.frequency_bucket AS user_frequency_bucket,
                users.paper_bucket AS user_paper_bucket,
                items.frequency_bucket AS item_frequency_bucket,
                items.paper_bucket AS item_paper_bucket,
                items.train_interactions = 0 AS item_is_cold_start
            FROM read_parquet('{_sql_path(base_interactions)}') AS base
            JOIN read_parquet('{_sql_path(users)}') AS users USING (user_id)
            JOIN read_parquet('{_sql_path(items)}') AS items USING (item_id)
            ORDER BY users.user_idx ASC, base.history_position ASC
        """,
        interactions,
    )
    _copy_parquet(
        connection,
        f"""
            SELECT
                user_id,
                user_idx,
                ANY_VALUE(user_frequency_bucket) AS user_frequency_bucket,
                ANY_VALUE(user_paper_bucket) AS user_paper_bucket,
                list(item_idx ORDER BY history_position)
                    FILTER (WHERE split = 'train') AS train_item_ids,
                list(epoch(event_time)::BIGINT ORDER BY history_position)
                    FILTER (WHERE split = 'train') AS train_event_timestamps,
                MAX(item_idx) FILTER (WHERE split = 'valid') AS valid_item_id,
                MAX(epoch(event_time)::BIGINT)
                    FILTER (WHERE split = 'valid') AS valid_event_timestamp,
                MAX(item_idx) FILTER (WHERE split = 'test') AS test_item_id,
                MAX(epoch(event_time)::BIGINT)
                    FILTER (WHERE split = 'test') AS test_event_timestamp,
                COUNT(*) FILTER (WHERE split = 'train') AS train_length
            FROM read_parquet('{_sql_path(interactions)}')
            GROUP BY user_id, user_idx
            ORDER BY user_idx ASC
        """,
        sequences,
    )
    _copy_parquet(
        connection,
        f"""
            SELECT
                interactions.review_id,
                interactions.user_id,
                interactions.user_idx,
                interactions.item_id,
                interactions.item_idx,
                interactions.event_time,
                interactions.history_position,
                reviews.stars,
                reviews.text
            FROM read_parquet('{_sql_path(interactions)}') AS interactions
            JOIN read_parquet('{_sql_path(staged_reviews)}') AS reviews USING (review_id)
            WHERE interactions.split = 'train'
            ORDER BY interactions.user_idx ASC, interactions.history_position ASC
        """,
        profile_reviews,
    )
    return {
        "interactions": interactions,
        "users": users,
        "items": items,
        "sequences": sequences,
        "profile_reviews_train": profile_reviews,
    }


def _scalar(connection: duckdb.DuckDBPyConnection, query: str) -> Any:
    return connection.execute(query).fetchone()[0]


def _statistics(
    connection: duckdb.DuckDBPyConnection,
    staged_reviews: Path,
    outputs: dict[str, Path],
    config: YelpPreparationConfig,
) -> dict[str, Any]:
    interactions = outputs["interactions"]
    items = outputs["items"]
    sequences = outputs["sequences"]
    profile_reviews = outputs["profile_reviews_train"]
    interactions_table = f"read_parquet('{_sql_path(interactions)}')"
    items_table = f"read_parquet('{_sql_path(items)}')"
    sequences_table = f"read_parquet('{_sql_path(sequences)}')"
    profile_reviews_table = f"read_parquet('{_sql_path(profile_reviews)}')"
    staged_reviews_table = f"read_parquet('{_sql_path(staged_reviews)}')"
    return {
        "raw_reviews": int(
            _scalar(connection, f"SELECT COUNT(*) FROM {staged_reviews_table}")
        ),
        "source_window_interactions": int(
            _scalar(
                connection,
                f"""SELECT COUNT(*) FROM {staged_reviews_table}
                    WHERE event_time >= TIMESTAMP '{config.event_time_min}'
                      AND event_time <= TIMESTAMP '{config.event_time_max}'
                      AND stars > {float(config.rating_min_exclusive)}""",
            )
        ),
        "eligible_interactions": int(
            _scalar(connection, f"SELECT COUNT(*) FROM {interactions_table}")
        ),
        "users": int(
            _scalar(connection, f"SELECT COUNT(DISTINCT user_id) FROM {interactions_table}")
        ),
        "items": int(
            _scalar(connection, f"SELECT COUNT(*) FROM {items_table}")
        ),
        "train_interactions": int(
            _scalar(connection, f"SELECT COUNT(*) FROM {interactions_table} WHERE split='train'")
        ),
        "valid_interactions": int(
            _scalar(connection, f"SELECT COUNT(*) FROM {interactions_table} WHERE split='valid'")
        ),
        "test_interactions": int(
            _scalar(connection, f"SELECT COUNT(*) FROM {interactions_table} WHERE split='test'")
        ),
        "sequence_rows": int(
            _scalar(connection, f"SELECT COUNT(*) FROM {sequences_table}")
        ),
        "profile_train_rows": int(
            _scalar(connection, f"SELECT COUNT(*) FROM {profile_reviews_table}")
        ),
        "split_invariant_violations": int(
            _scalar(
                connection,
                f"""SELECT COUNT(*) FROM (
                    SELECT user_id
                    FROM {interactions_table}
                    GROUP BY user_id
                    HAVING COUNT(*) FILTER (WHERE split='valid') <> 1
                        OR COUNT(*) FILTER (WHERE split='test') <> 1
                        OR COUNT(*) FILTER (WHERE split='train') < 1
                )""",
            )
        ),
        "cold_start_items": int(
            _scalar(connection, f"SELECT COUNT(*) FROM {items_table} WHERE train_interactions=0")
        ),
        "missing_business_snapshots": int(
            _scalar(
                connection,
                f"SELECT COUNT(*) FROM {items_table} WHERE NOT has_business_snapshot",
            )
        ),
        "cold_start_valid_interactions": int(
            _scalar(
                connection,
                f"""SELECT COUNT(*) FROM {interactions_table}
                    WHERE split='valid' AND item_is_cold_start""",
            )
        ),
        "cold_start_test_interactions": int(
            _scalar(
                connection,
                f"""SELECT COUNT(*) FROM {interactions_table}
                    WHERE split='test' AND item_is_cold_start""",
            )
        ),
        "duplicate_user_item_pairs": int(
            _scalar(
                connection,
                f"""SELECT COUNT(*) FROM (
                    SELECT user_id, item_id FROM {interactions_table}
                    GROUP BY user_id, item_id HAVING COUNT(*) > 1
                )""",
            )
        ),
        "event_time_min": str(
            _scalar(connection, f"SELECT MIN(event_time) FROM {interactions_table}")
        ),
        "event_time_max": str(
            _scalar(connection, f"SELECT MAX(event_time) FROM {interactions_table}")
        ),
    }


def _combined_fingerprint(file_hashes: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name, file_hash in sorted(file_hashes.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def prepare_yelp(config: YelpPreparationConfig) -> dict[str, Any]:
    """Prepare Yelp into split-safe Parquet tables and return the reproducibility manifest."""
    config.interim_dir.mkdir(parents=True, exist_ok=True)
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    source_paths = extract_yelp_sources(config.archive, config.interim_dir / "source")
    staged_reviews = config.interim_dir / "reviews_full.parquet"
    staged_businesses = config.interim_dir / "businesses.parquet"
    base_interactions = config.interim_dir / "interactions_base.parquet"
    temporary_dir = config.interim_dir / "duckdb_tmp"
    temporary_dir.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect()
    connection.execute(f"SET memory_limit = '{config.memory_limit}'")
    connection.execute(f"SET threads = {config.threads}")
    connection.execute(f"SET temp_directory = '{_sql_path(temporary_dir)}'")
    try:
        print("[1/6] Staging review JSON as compressed Parquet", flush=True)
        _stage_reviews(connection, source_paths["review_json"], staged_reviews)
        print("[2/6] Staging current business metadata snapshot", flush=True)
        _stage_businesses(connection, source_paths["business_json"], staged_businesses)
        print("[3/6] Applying user filter and chronological leave-two-out split", flush=True)
        _build_base_interactions(
            connection,
            staged_reviews,
            base_interactions,
            config.min_user_interactions,
            config.min_item_interactions,
            config.event_time_min,
            config.event_time_max,
            config.rating_min_exclusive,
        )
        print(
            "[4/6] Building ID maps, buckets, sequences, and train-only profile sources",
            flush=True,
        )
        connection.execute("SET threads = 1")
        outputs = _build_outputs(
            connection,
            staged_reviews,
            staged_businesses,
            base_interactions,
            config.processed_dir,
            config,
        )
        print("[5/6] Auditing counts and cold-start separation", flush=True)
        statistics = _statistics(connection, staged_reviews, outputs, config)
    finally:
        connection.close()

    print("[6/6] Computing SHA-256 fingerprints", flush=True)
    output_hashes = {name: sha256_file(path) for name, path in outputs.items()}
    manifest: dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_variant": "current_official_extension_not_faithful_paper_snapshot",
        "source": {
            "official_page": "https://business.yelp.com/data/resources/open-dataset/",
            "archive_url": "https://business.yelp.com/external-assets/files/Yelp-JSON.zip",
            "archive_bytes": config.archive.stat().st_size,
            "archive_sha256": sha256_file(config.archive),
            "license": "Educational use; see the Yelp agreement included in the archive",
            "author_rule_source": (
                "https://github.com/liuqidong07/LLM-ESR/blob/master/data/data_process.py"
            ),
            "author_rule_blob_sha": "eb1f1df65274b29be958c9739c6c19d5773093c4",
        },
        "config": {
            **asdict(config),
            "archive": str(config.archive),
            "interim_dir": str(config.interim_dir),
            "processed_dir": str(config.processed_dir),
            "report_path": str(config.report_path),
        },
        "statistics": statistics,
        "outputs": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": output_hashes[name],
            }
            for name, path in outputs.items()
        },
        "processed_fingerprint": _combined_fingerprint(output_hashes),
        "reproducibility": {
            "staging_threads": config.threads,
            "output_write_threads": 1,
            "reason": "single-threaded final Parquet writes stabilize physical artifact hashes",
        },
        "leakage_controls": {
            "split_order": ["event_time", "review_id"],
            "valid": "per-user second-to-last interaction",
            "test": "per-user last interaction",
            "profiles": "profile_reviews_train.parquet contains train rows only",
            "frequency_buckets": "fit from train interaction counts only",
            "business_metadata_limitation": (
                "current archive snapshot has no historical versions and must not be claimed "
                "as event-time-frozen text"
            ),
            "cross_user_time_limitation": (
                "per-user leave-two-out is not a single global wall-clock cutoff"
            ),
        },
    }
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (config.processed_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
