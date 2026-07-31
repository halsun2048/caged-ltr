"""Train one author-style DeBERTa pointwise student with grouped RankNet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from transformers import AutoTokenizer

from caged_ltr.distillation import (
    DistillationControl,
    TextRankingGroup,
    collate_text_ranking_groups,
    load_text_ranking_groups,
    teacher_ndcg_at_k,
)
from caged_ltr.losses import ranknet_loss
from caged_ltr.models import (
    DEFAULT_DEBERTA_V3_BASE,
    DEFAULT_DEBERTA_V3_BASE_REVISION,
    PointwiseCrossEncoder,
)
from caged_ltr.reproducibility import seed_everything, sha256_file


class _GroupDataset(Dataset[TextRankingGroup]):
    def __init__(self, groups: list[TextRankingGroup]) -> None:
        if not groups:
            raise ValueError("ranking-group dataset must not be empty")
        self.groups = tuple(groups)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> TextRankingGroup:
        return self.groups[index]


class _StridedDistributedSampler(Sampler[int]):
    """Partition validation data without padding or duplicate queries."""

    def __init__(self, size: int, *, rank: int, world_size: int) -> None:
        self.indices = tuple(range(rank, size, world_size))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def _base_model(model: nn.Module) -> PointwiseCrossEncoder:
    if isinstance(model, DistributedDataParallel):
        return model.module
    if not isinstance(model, PointwiseCrossEncoder):
        raise TypeError(f"unexpected pointwise model type: {type(model)!r}")
    return model


def _duration(seconds: float) -> str:
    total = max(0, round(seconds))
    minutes, seconds = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}h{minutes:02d}m" if hours else f"{minutes:02d}m{seconds:02d}s"


def _identity(args: argparse.Namespace, *, world_size: int) -> dict[str, object]:
    payload = {
        "schema": "r4_pointwise_student_v1",
        "control": args.control,
        "candidates_sha256": sha256_file(args.candidates),
        "labels_sha256": sha256_file(args.labels),
        "model": args.model,
        "revision": args.revision,
        "seed": args.seed,
        "query_batch_size": args.query_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "world_size": world_size,
        "effective_global_query_batch": (
            args.query_batch_size
            * args.gradient_accumulation_steps
            * world_size
        ),
        "learning_rate": args.learning_rate,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "max_length": args.max_length,
        "precision": args.precision,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "identity_sha256": hashlib.sha256(encoded).hexdigest()}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _save_checkpoint(
    path: Path,
    *,
    identity_sha256: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    completed_epoch: int,
    best_epoch: int,
    best_ndcg: float,
    stale_epochs: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "identity_sha256": identity_sha256,
            "model": _base_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "completed_epoch": completed_epoch,
            "best_epoch": best_epoch,
            "best_validation_teacher_ndcg_at_10": best_ndcg,
            "stale_epochs": stale_epochs,
        },
        temporary,
    )
    os.replace(temporary, path)


def _move_batch(
    batch: dict[str, object],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    encoded = {
        name: tensor.to(device, non_blocking=True)
        for name, tensor in dict(batch["encoded"]).items()
    }
    labels = batch["labels"].to(device, non_blocking=True)
    group_sizes = batch["group_sizes"].to(device, non_blocking=True)
    return encoded, labels, group_sizes


def _loader(
    groups: list[TextRankingGroup],
    *,
    tokenizer: Any,
    batch_size: int,
    max_length: int,
    shuffle: bool,
    seed: int,
    rank: int,
    world_size: int,
) -> DataLoader[TextRankingGroup]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = _GroupDataset(groups)
    sampler: Sampler[int] | None = None
    if world_size > 1:
        if shuffle:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=seed,
                drop_last=False,
            )
        else:
            sampler = _StridedDistributedSampler(
                len(dataset),
                rank=rank,
                world_size=world_size,
            )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=partial(
            collate_text_ranking_groups,
            tokenizer,
            max_length=max_length,
        ),
    )


def _autocast(
    device: torch.device,
    precision: str,
) -> torch.autocast:
    dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    enabled = device.type == "cuda" and precision in dtypes
    return torch.autocast(
        device_type=device.type,
        dtype=dtypes.get(precision, torch.float32),
        enabled=enabled,
    )


@torch.no_grad()
def _validate(
    model: nn.Module,
    loader: DataLoader[TextRankingGroup],
    *,
    device: torch.device,
    precision: str,
    distributed: bool,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    ndcg_sum = 0.0
    group_count = 0
    for batch in loader:
        encoded, batch_labels, group_sizes = _move_batch(batch, device)
        with _autocast(device, precision):
            batch_scores = model(**encoded)
            loss = ranknet_loss(batch_scores, batch_labels, group_sizes)
        batch_group_count = int(group_sizes.numel())
        loss_sum += float(loss) * batch_group_count
        ndcg_sum += (
            teacher_ndcg_at_k(
                batch_scores.float().cpu(),
                batch_labels.cpu(),
                group_sizes.cpu(),
                cutoff=10,
            )
            * batch_group_count
        )
        group_count += batch_group_count
    totals = torch.tensor(
        [loss_sum, ndcg_sum, float(group_count)],
        dtype=torch.float64,
        device=device,
    )
    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    if totals[2].item() <= 0:
        raise ValueError("validation loader must contain at least one query group")
    return (
        float((totals[0] / totals[2]).item()),
        float((totals[1] / totals[2]).item()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", choices=("bm25", "random", "prp"), required=True)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("data/processed/r4_msmarco_1k/candidates.parquet"),
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_DEBERTA_V3_BASE)
    parser.add_argument("--revision", default=DEFAULT_DEBERTA_V3_BASE_REVISION)
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf-cache"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=500)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    if min(
        args.query_batch_size,
        args.gradient_accumulation_steps,
        args.max_epochs,
        args.patience,
        args.max_length,
        args.progress_every,
    ) <= 0:
        parser.error("batch, epoch, patience, length, and progress values must be positive")
    if args.learning_rate <= 0 or args.seed < 0:
        parser.error("learning rate must be positive and seed must be non-negative")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if args.device != "cuda":
            raise ValueError("distributed R4 training currently requires --device cuda")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl", device_id=device)
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed_everything(args.seed + rank, deterministic_algorithms=False)
    identity = _identity(args, world_size=world_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("identity_sha256") != identity["identity_sha256"]:
            raise ValueError("existing R4 student run has a different identity")
    elif rank == 0:
        _write_json(
            manifest_path,
            {
                **identity,
                "author_reference": (
                    "sunnweiwei/RankGPT specialization.py: "
                    "DeBERTa-v3-base, scalar cross-encoder, RankNet, 3 epochs"
                ),
                "qrels_accessed": False,
            },
        )
    if distributed:
        dist.barrier()

    control: DistillationControl = args.control
    groups = load_text_ranking_groups(
        args.candidates,
        args.labels,
        control=control,
    )
    train_groups = [group for group in groups if group.split == "train"]
    validation_groups = [group for group in groups if group.split == "validation"]
    if not train_groups or not validation_groups:
        raise ValueError("both train and validation groups are required")

    if rank == 0:
        print(
            f"[model] loading {args.model}@{args.revision} on {world_size} device(s); "
            f"control={args.control} train={len(train_groups)} "
            f"validation={len(validation_groups)} "
            f"effective_global_query_batch="
            f"{identity['effective_global_query_batch']}",
            file=sys.stderr,
            flush=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        use_fast=True,
    )
    base_model = PointwiseCrossEncoder.from_pretrained(
        model_name=args.model,
        revision=args.revision,
        cache_dir=str(args.cache_dir),
    ).to(device)
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and args.precision == "float16",
    )

    latest_path = args.output_dir / "latest.pt"
    best_path = args.output_dir / "best.pt"
    start_epoch = 0
    best_epoch = -1
    best_ndcg = float("-inf")
    stale_epochs = 0
    history = []
    if latest_path.is_file():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        if checkpoint.get("identity_sha256") != identity["identity_sha256"]:
            raise ValueError("latest checkpoint identity mismatch")
        base_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["completed_epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_ndcg = float(checkpoint["best_validation_teacher_ndcg_at_10"])
        stale_epochs = int(checkpoint["stale_epochs"])
        summary_path = args.output_dir / "summary.json"
        if summary_path.is_file():
            history = json.loads(summary_path.read_text(encoding="utf-8")).get(
                "history", []
            )
        if rank == 0:
            print(
                f"[resume] completed_epoch={start_epoch - 1} "
                f"best_epoch={best_epoch} best={best_ndcg:.6f}",
                file=sys.stderr,
                flush=True,
            )
    model: nn.Module = base_model
    if distributed:
        model = DistributedDataParallel(
            base_model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

    validation_loader = _loader(
        validation_groups,
        tokenizer=tokenizer,
        batch_size=args.query_batch_size,
        max_length=args.max_length,
        shuffle=False,
        seed=args.seed,
        rank=rank,
        world_size=world_size,
    )
    started = time.monotonic()
    for epoch in range(start_epoch, args.max_epochs):
        train_loader = _loader(
            train_groups,
            tokenizer=tokenizer,
            batch_size=args.query_batch_size,
            max_length=args.max_length,
            shuffle=True,
            seed=args.seed,
            rank=rank,
            world_size=world_size,
        )
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, batch in enumerate(train_loader, start=1):
            encoded, labels, group_sizes = _move_batch(batch, device)
            with _autocast(device, args.precision):
                scores = model(**encoded)
                loss = ranknet_loss(scores, labels, group_sizes)
                scaled_loss = loss / args.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            update = (
                step % args.gradient_accumulation_steps == 0
                or step == len(train_loader)
            )
            if update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            if rank == 0 and (
                step % args.progress_every == 0 or step == len(train_loader)
            ):
                width = 24
                filled = round(width * step / len(train_loader))
                sys.stderr.write("\r\033[2K")
                completed_queries = min(
                    step * args.query_batch_size * world_size,
                    len(train_groups),
                )
                sys.stderr.write(
                    f"[epoch {epoch + 1}/{args.max_epochs}] "
                    f"[{'#' * filled}{'-' * (width - filled)}] "
                    f"queries={completed_queries:>4}/"
                    f"{len(train_groups):<4} loss={sum(losses) / len(losses):.5f} "
                    f"elapsed={_duration(time.monotonic() - started)}"
                )
                sys.stderr.flush()
        if rank == 0:
            sys.stderr.write("\n")
        train_totals = torch.tensor(
            [sum(losses), float(len(losses))],
            dtype=torch.float64,
            device=device,
        )
        if distributed:
            dist.all_reduce(train_totals, op=dist.ReduceOp.SUM)
        train_loss = float((train_totals[0] / train_totals[1]).item())
        validation_loss, validation_ndcg = _validate(
            model,
            validation_loader,
            device=device,
            precision=args.precision,
            distributed=distributed,
        )
        improved = validation_ndcg > best_ndcg + 1e-12
        if improved:
            best_ndcg = validation_ndcg
            best_epoch = epoch
            stale_epochs = 0
            if rank == 0:
                _save_checkpoint(
                    best_path,
                    identity_sha256=str(identity["identity_sha256"]),
                    model=model,
                    optimizer=optimizer,
                    completed_epoch=epoch,
                    best_epoch=best_epoch,
                    best_ndcg=best_ndcg,
                    stale_epochs=stale_epochs,
                )
        else:
            stale_epochs += 1
        if rank == 0:
            _save_checkpoint(
                latest_path,
                identity_sha256=str(identity["identity_sha256"]),
                model=model,
                optimizer=optimizer,
                completed_epoch=epoch,
                best_epoch=best_epoch,
                best_ndcg=best_ndcg,
                stale_epochs=stale_epochs,
            )
        if distributed:
            dist.barrier()
        history.append(
            {
                "epoch": epoch,
                "train_ranknet_loss": train_loss,
                "validation_ranknet_loss": validation_loss,
                "validation_teacher_ndcg_at_10": validation_ndcg,
                "best": improved,
            }
        )
        summary = {
            "stage": "training",
            **identity,
            "train_queries": len(train_groups),
            "validation_queries": len(validation_groups),
            "distributed_sampler_padding_queries_per_epoch": (
                len(train_loader) * args.query_batch_size * world_size
                - len(train_groups)
            ),
            "best_epoch": best_epoch,
            "best_validation_teacher_ndcg_at_10": best_ndcg,
            "history": history,
            "qrels_accessed": False,
            "test_accessed": False,
            "wall_seconds": time.monotonic() - started,
        }
        if rank == 0:
            _write_json(args.output_dir / "summary.json", summary)
            print(
                f"[valid] epoch={epoch + 1} loss={validation_loss:.6f} "
                f"teacher-NDCG@10={validation_ndcg:.6f} "
                f"best_epoch={best_epoch + 1} stale={stale_epochs}/{args.patience}",
                file=sys.stderr,
                flush=True,
            )
        if stale_epochs >= args.patience:
            break

    if rank == 0:
        summary_path = args.output_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["stage"] = "complete"
        summary["early_stopped"] = stale_epochs >= args.patience
        summary["best_checkpoint"] = str(best_path)
        _write_json(summary_path, summary)
        print(
            json.dumps(
                {
                    "stage": summary["stage"],
                    "control": args.control,
                    "best_epoch": best_epoch,
                    "best_validation_teacher_ndcg_at_10": best_ndcg,
                    "qrels_accessed": False,
                    "test_accessed": False,
                    "report": str(summary_path),
                },
                ensure_ascii=False,
            )
        )
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
