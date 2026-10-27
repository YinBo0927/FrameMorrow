"""Train the selector on frozen features; the generator and encoders are absent."""
import argparse
from contextlib import nullcontext
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader, RandomSampler

from framemorrow import FrameMorrow, SelectorConfig, ranking_loss, teacher_scores
from framemorrow.data import (FeatureDataset, atomic_save, check_video_split,
                             collate_features, read_manifest, student_inputs)


def run_batch(model, batch, device, bf16):
    batch = {k: v.to(device) for k, v in batch.items()}
    # Realized continuation is consumed only by this detached teacher.
    target = teacher_scores(batch["history"], batch["future"],
                            batch["history_mask"], batch["future_mask"])
    with torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else nullcontext():
        scores = model(**student_inputs(batch))
    return ranking_loss(scores, target, batch["history_mask"]), scores, target, batch["history_mask"]


@torch.inference_mode()
def validate(model, loader, device, bf16):
    model.eval()
    totals = {"loss": 0., "listwise": 0., "pairwise": 0., "teacher_top4_recall": 0.}
    count = 0
    for batch in loader:
        losses, scores, target, mask = run_batch(model, batch, device, bf16)
        n = len(scores)
        count += n
        for key, loss in losses.items():
            totals[key] += loss.item() * n
        for s, t, m in zip(scores, target, mask):
            k = min(4, int(m.sum()))
            si = set(s.argsort(descending=True, stable=True)[:k].tolist())
            ti = set(t.argsort(descending=True, stable=True)[:k].tolist())
            totals["teacher_top4_recall"] += len(si & ti) / k
    return {k: v / count for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--settings", default=str(Path(__file__).resolve().parent / "configs/default.json"))
    parser.add_argument("--config", default="configs/text.json")
    parser.add_argument("--output", default="runs/text")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--require-videos", type=int)
    parser.add_argument("--resume", help="last.pt from the same run and data manifest")
    parser.add_argument("--workers", type=int, default=4)
    initial, _ = parser.parse_known_args()
    settings = json.loads(Path(initial.settings).read_text())["training"]
    allowed = {"epochs", "batch_size", "accumulate", "lr", "seed", "device", "precision", "workers"}
    if not isinstance(settings, dict) or set(settings) - allowed:
        parser.error("unknown training settings; see configs/default.json")
    parser.set_defaults(**settings)
    args = parser.parse_args()
    if (min(args.epochs, args.batch_size, args.accumulate) < 1 or not math.isfinite(args.lr)
            or args.lr <= 0 or args.workers < 0 or args.precision not in ("bf16", "fp32")):
        raise ValueError("invalid training settings: positive epochs/batch/accumulate/lr, nonnegative workers, bf16/fp32")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("choose an empty run directory to preserve previous results")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    check_video_split(args.manifest)
    config = SelectorConfig(**json.loads(Path(args.config).read_text()))
    manifest_rows = read_manifest(args.manifest)
    video_ids = {split: {row["video_id"] for row in manifest_rows if row["split"] == split} for split in ("train", "val")}
    if args.require_videos and sum(map(len, video_ids.values())) != args.require_videos:
        raise ValueError("actual encoded video count differs from the required corpus size")
    metadata = manifest_rows[0].get("encoder_metadata")
    if any(r.get("encoder_metadata") != metadata for r in manifest_rows):
        raise ValueError("mixed encoder provenance in feature manifest")
    if metadata and metadata["modality"] != config.modality:
        raise ValueError("feature modality differs from selector config")
    train_set, val_set = (FeatureDataset(args.manifest, split) for split in ("train", "val"))
    sampler_rng = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train_set, batch_size=args.batch_size,
                        sampler=RandomSampler(train_set, generator=sampler_rng), collate_fn=collate_features,
                        num_workers=args.workers, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, collate_fn=collate_features,
                            num_workers=args.workers, persistent_workers=args.workers > 0)
    device = torch.device(args.device)
    bf16 = args.precision == "bf16" and device.type == "cuda"
    if args.precision == "bf16" and device.type != "cuda":
        raise ValueError("use --precision fp32 for CPU training")
    if bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU does not support BF16; use --precision fp32")
    model = FrameMorrow(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01, betas=(.9, .999))
    updates = math.ceil(len(loader) / args.accumulate) * args.epochs
    warmup = max(1, math.ceil(updates * .05))

    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        return .5 * (1 + math.cos(math.pi * min(1, (step - warmup) / max(1, updates - warmup))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    provenance = {**vars(args), "paper": "https://arxiv.org/abs/2609.38839v1",
                  "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                  "train_examples": len(train_set), "val_examples": len(val_set),
                  "train_videos": len(video_ids["train"]), "val_videos": len(video_ids["val"]),
                  "trainable_parameters": sum(p.numel() for p in model.parameters()),
                  "torch": str(torch.__version__), "hip": torch.version.hip,
                  "encoder_metadata": metadata, "global_batch_size": args.batch_size * args.accumulate}
    (output / "settings.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance), flush=True)
    best = math.inf
    start_epoch = 0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        previous = saved["provenance"]
        for key in ("manifest_sha256", "seed", "global_batch_size", "epochs", "lr", "precision"):
            if previous[key] != provenance[key]:
                raise ValueError(f"resume setting differs: {key}")
        if SelectorConfig(**saved["config"]) != model.config:
            raise ValueError("resume selector architecture differs")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        torch.set_rng_state(saved["torch_rng"])
        sampler_rng.set_state(saved["sampler_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        start_epoch, best = saved["epoch"], saved["best_listwise"]
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total, count = 0., 0
        iterator = iter(loader)
        epoch_started = time.time()
        for update in range(math.ceil(len(loader) / args.accumulate)):
            group = list(itertools.islice(iterator, args.accumulate))
            optimizer.zero_grad(set_to_none=True)
            group_size = sum(len(b["history"]) for b in group)
            for batch in group:
                losses, *_ = run_batch(model, batch, device, bf16)
                n = len(batch["history"])
                (losses["loss"] * n / group_size).backward()
                total += losses["loss"].item() * n
                count += n
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            if update % 25 == 0:
                print(json.dumps({"stage": "training", "epoch": epoch + 1, "update": update + 1,
                                  "updates_per_epoch": math.ceil(len(loader) / args.accumulate),
                                  "train_loss_so_far": total / count, "elapsed_seconds": time.time() - epoch_started}), flush=True)
        metrics = {"epoch": epoch + 1, "train_loss": total / count,
                   **{"val_" + k: v for k, v in validate(model, val_loader, device, bf16).items()}}
        if not all(math.isfinite(v) for v in metrics.values()):
            raise RuntimeError("nonfinite training or validation metrics")
        checkpoint = {"config": model.checkpoint_config(), "model": model.state_dict(),
                      "epoch": epoch + 1, "metrics": metrics, "provenance": provenance}
        if metrics["val_listwise"] < best:
            best = metrics["val_listwise"]
            atomic_save(checkpoint, output / "best.pt")
        atomic_save({**checkpoint, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                     "sampler_rng": sampler_rng.get_state(),
                     "best_listwise": best}, output / "last.pt")
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
