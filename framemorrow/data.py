"""Portable JSONL manifests and cached frozen feature bundles."""
import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def read_manifest(path):
    path = Path(path).resolve()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        row["_base"] = str(path.parent)
    return rows


def resolve_path(row, value):
    return Path(row["_base"]) / value


def validate_times(history, recent, future=None, condition_end=None):
    for name, values in (("history", history), ("recent", recent), ("future", future)):
        if values is None:
            continue
        values = torch.as_tensor(values, dtype=torch.float64)
        if values.ndim != 1 or not torch.isfinite(values).all():
            raise ValueError(f"{name} timestamps must be a finite vector")
        if len(values) > 1 and not (values[1:] > values[:-1]).all():
            raise ValueError(f"{name} timestamps must be strictly increasing")
    if len(history) and len(recent) and history[-1] >= recent[0]:
        raise ValueError("eligible history overlaps recent context")
    if future is not None:
        if not len(history) or not len(recent) or not len(future):
            raise ValueError("training requires history, recent context, and future")
        if future[0] <= recent[-1]:
            raise ValueError("teacher continuation must be later than observed context")
        if condition_end is None or future[-1] > condition_end:
            raise ValueError("teacher frames extend outside the known condition scope")


def validate_bundle(row, training=False):
    keys = ["history", "recent", "condition"] + (["future"] if training else [])
    for key in keys:
        x = row[key]
        if x.ndim != 2 or not torch.isfinite(x).all():
            raise ValueError(f"{key} must contain finite [sequence,feature] values")
    if len(row["condition"]) < 1:
        raise ValueError("known condition cannot be empty")
    for key in ("history", "recent") + (("future",) if training else ()):
        if len(row[key]) != len(row[key + "_times"]):
            raise ValueError(f"{key} feature/timestamp count mismatch")
    validate_times(row["history_times"], row["recent_times"],
                   row["future_times"] if training else None, row.get("condition_end"))
    if training and len(row["history"]) > 128:
        raise ValueError("paper training limit is 128 historical candidates")
    return row


class FeatureDataset(Dataset):
    def __init__(self, manifest, split):
        self.rows = [r for r in read_manifest(manifest) if r["split"] == split]
        if not self.rows:
            raise ValueError(f"no {split} examples in manifest")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return validate_bundle(torch.load(resolve_path(row, row["features"]),
                                          map_location="cpu", weights_only=True), training=True)


def collate_features(rows):
    result = {}
    for key in ("history", "recent", "condition", "future"):
        result[key] = pad_sequence([r[key].float() for r in rows], batch_first=True)
        lengths = torch.tensor([len(r[key]) for r in rows])
        result[key + "_mask"] = torch.arange(result[key].shape[1])[None] < lengths[:, None]
    return result


def student_inputs(batch):
    return {key: batch[key] for key in ("history", "recent", "condition",
            "history_mask", "recent_mask", "condition_mask")}


def check_video_split(manifest):
    seen = {}
    sources = {}
    for row in read_manifest(manifest):
        if row["split"] not in ("train", "val"):
            raise ValueError("split must be train or val")
        previous = seen.setdefault(row["video_id"], row["split"])
        if previous != row["split"]:
            raise ValueError(f"video leaks across train/val: {row['video_id']}")
        source = row.get('source_id') or row['video_id']
        if sources.setdefault(source, row['split']) != row['split']:
            raise ValueError(f"source leaks across train/val: {source}")


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)
