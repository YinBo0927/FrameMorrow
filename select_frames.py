"""Select explicit observed historical frames; accepts no realized continuation."""
import argparse
import json
from pathlib import Path

import torch

from framemorrow import FrameMorrow, SelectorConfig
from framemorrow.data import validate_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--features", help="observed-only .pt bundle")
    group.add_argument("--input", help="observed-only RGB JSON with history/recent/prompt")
    settings = json.loads((Path(__file__).resolve().parent / "configs/default.json").read_text())
    parser.add_argument("--k", type=int, default=settings["inference"]["k"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--visual-model", default="facebook/dinov2-base")
    parser.add_argument("--condition-model")
    parser.add_argument("--tokenizer")
    parser.add_argument("--output")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = SelectorConfig(**checkpoint["config"])
    model = FrameMorrow(config).to(args.device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    frames = None
    if args.features:
        bundle = torch.load(args.features, map_location="cpu", weights_only=True)
        if any(key.startswith("future") for key in bundle):
            raise ValueError("inference bundle must contain observed inputs only")
    else:
        from encode import encode_row
        from framemorrow.encoders import FrozenEncoders
        row = json.loads(Path(args.input).read_text())
        if any(key.startswith("future") for key in row):
            raise ValueError("inference JSON must contain observed inputs only")
        row["_base"] = str(Path(args.input).resolve().parent)
        frames = [{**frame, 'path': str((Path(row['_base']) / frame['path']).resolve())}
                  for frame in row['history']]
        limit = (checkpoint.get("provenance", {}).get("encoder_metadata") or {}).get("max_condition_tokens", 128)
        encoders = FrozenEncoders(config.modality, args.visual_model, args.condition_model,
                                  args.tokenizer, args.device, limit)
        bundle = encode_row(row, encoders, teacher=False)
    validate_bundle(bundle)
    if bundle.get("encoder_metadata", {}).get("modality", config.modality) != config.modality:
        raise ValueError("input modality differs from checkpoint")
    features = {key: bundle[key].float().unsqueeze(0).to(args.device)
                for key in ("history", "recent", "condition")}
    times = torch.tensor(bundle["history_times"], dtype=torch.float64, device=args.device)[None]
    selected, scores = model.select(**features, timestamps=times, k=args.k)
    indices = selected[0].tolist()
    result = {"indices": indices, "timestamps": times[0, selected[0]].tolist(),
              "scores": scores[0, selected[0]].tolist(), "candidate_count": len(bundle["history"])}
    if frames is not None:
        result["frames"] = [frames[i] for i in indices]
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
