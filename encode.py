"""Precompute frozen features from a timestamped RGB tuple manifest."""
import argparse
import json
from pathlib import Path

from framemorrow.data import atomic_save, read_manifest, resolve_path, validate_times
from framemorrow.encoders import FrozenEncoders


def encode_row(row, encoders, teacher=True):
    keys = ("history", "recent", "future") if teacher else ("history", "recent")
    times = {key + "_times": [f["time"] for f in row[key]] for key in keys}
    validate_times(times["history_times"], times["recent_times"],
                   times.get("future_times"), row.get("condition_end"))
    result = {**times, "condition_end": row.get("condition_end"), "prompt": row["prompt"],
              "video_id": row["video_id"], "encoder_metadata": encoders.metadata()}
    if row.get('source_id'):
        result['source_id'] = row['source_id']
    # Cache each observed image only within this tuple; teacher features use the same frozen encoder.
    paths = list(dict.fromkeys(str(resolve_path(row, f["path"])) for key in keys for f in row[key]))
    features = dict(zip(paths, encoders.images(paths)))
    import torch
    for key in keys:
        result[key] = (torch.stack([features[str(resolve_path(row, f["path"]))] for f in row[key]])
                       if row[key] else torch.empty(0, 768))
    result["condition"] = encoders.text(row["prompt"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("--modality", choices=["text", "action"], default="text")
    parser.add_argument("--visual-model", default="facebook/dinov2-base")
    parser.add_argument("--condition-model")
    parser.add_argument("--tokenizer")
    parser.add_argument("--max-condition-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "features.jsonl"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite {manifest_path}; choose a new output directory")
    rows = read_manifest(args.manifest)
    encoders = FrozenEncoders(args.modality, args.visual_model, args.condition_model, args.tokenizer,
                              args.device, args.max_condition_tokens)
    with manifest_path.with_suffix(".jsonl.tmp").open("w") as manifest:
        for i, row in enumerate(rows):
            bundle = encode_row(row, encoders)
            filename = f"tuple_{i:07d}.pt"
            atomic_save(bundle, output / filename)
            record = {"features": filename, "video_id": row["video_id"],
                      "split": row["split"], "encoder_metadata": encoders.metadata()}
            if row.get('source_id'):
                record['source_id'] = row['source_id']
            manifest.write(json.dumps(record) + "\n")
            manifest.flush()
            print(f"encoded {i + 1}/{len(rows)}", flush=True)
    manifest_path.with_suffix(".jsonl.tmp").replace(manifest_path)


if __name__ == "__main__":
    main()
