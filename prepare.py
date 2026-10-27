"""Extract two-fps RGB frames and paper training tuples from condition intervals.

Input JSONL: video_id, video, segments [{start,end,prompt}]. Conditions must be
known at their start; source annotations are separate from selector inputs.
"""
import argparse
import bisect
import json
import math
from pathlib import Path
import random

from framemorrow.data import read_manifest, resolve_path, validate_times
from framemorrow.video import sample_video


def uniform(items, maximum):
    if maximum < 1:
        raise ValueError("sample limit must be positive")
    if len(items) <= maximum:
        return items
    if maximum == 1:
        return [items[len(items) // 2]]
    return [items[round(i * (len(items) - 1) / (maximum - 1))] for i in range(maximum)]


def make_tuple(frames, start, end, prompt, recent_seconds=4., horizon=4., duration=None):
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("condition intervals require finite start < end")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("condition prompt is empty")
    times = [f["time"] for f in frames]
    # t=condition start; history <= t-4; recent (t-4,t]; teacher (t,min(t+4,end)].
    history_end = bisect.bisect_right(times, start - recent_seconds)
    observed_end = bisect.bisect_right(times, start)
    future_end = bisect.bisect_right(times, min(start + horizon, end))
    history = uniform(frames[:history_end], 128)
    recent = uniform(frames[history_end:observed_end], 8)
    future = uniform(frames[observed_end:future_end], 8)
    if not history or len(recent) != 8 or not future:
        return None
    # Skip incomplete four-second futures unless the condition itself is shorter.
    available_until = times[-1] if duration is None and times else duration
    if not times or available_until < min(start + horizon, end):
        return None
    validate_times([f["time"] for f in history], [f["time"] for f in recent],
                   [f["time"] for f in future], end)
    return {"history": history, "recent": recent, "future": future,
            "prompt": prompt.strip(), "condition_start": start, "condition_end": end}


def split_videos(rows, seed=0):
    ids = [row['video_id'] for row in rows]
    groups = {row['video_id']: row.get('source_id') or row['video_id'] for row in rows}
    if len(ids) != len(set(ids)) or len(set(groups.values())) < 2:
        raise ValueError('need unique video IDs and at least two independent sources')
    supplied = ['split' in row for row in rows]
    if any(supplied):
        if not all(supplied) or any(row['split'] not in ('train', 'val') for row in rows):
            raise ValueError('supply a valid split for every video, or omit all splits')
        seen = {}
        for row in rows:
            group = groups[row['video_id']]
            if seen.setdefault(group, row['split']) != row['split']:
                raise ValueError('original source crosses the train/validation boundary')
        assignments = {row['video_id']: row['split'] for row in rows}
    else:
        sources = sorted(set(groups.values()))
        random.Random(seed).shuffle(sources)
        validation = set(sources[:max(1, round(len(sources) * .1))])
        assignments = {video: 'val' if group in validation else 'train' for video, group in groups.items()}
    if set(assignments.values()) != {'train', 'val'}:
        raise ValueError('both training and validation sources are required')
    return assignments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos")
    parser.add_argument("output")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rows = read_manifest(args.videos)
    assignments = split_videos(rows, args.seed)
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("choose an empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    manifests, counts = [], {}
    for i, row in enumerate(rows):
        counts[row['video_id']] = 0
        if not row['segments']:
            continue
        directory = output / "frames" / f"video_{i:06d}"
        sampled = sample_video(resolve_path(row, row["video"]), directory)
        frames = [{**frame, "path": str(Path(frame["path"]).relative_to(output))}
                  for frame in sampled["frames"]]
        for segment in row["segments"]:
            sample = make_tuple(frames, float(segment["start"]), float(segment["end"]), segment["prompt"], duration=sampled["duration"])
            if sample:
                manifests.append({"video_id": row["video_id"],
                                  "source_id": row.get('source_id') or row['video_id'],
                                  "split": assignments[row['video_id']], **sample})
                counts[row['video_id']] += 1
    (output / 'preparation_report.json').write_text(json.dumps({'video_tuple_counts': counts,
        'usable_videos': sum(count > 0 for count in counts.values()),
        'skipped_video_ids': [video for video, count in counts.items() if not count],
        'tuples': len(manifests)}, indent=2) + '\n')
    if {r["split"] for r in manifests} != {"train", "val"}:
        raise ValueError("no eligible tuples in one split; supply longer videos")
    (output / "tuples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in manifests))
    example = next(row for row in manifests if row['split'] == 'val')
    observed = {key: example[key] for key in ('video_id', 'history', 'recent', 'prompt')}
    (output / 'observed.json').write_text(json.dumps(observed, indent=2) + '\n')
    (output / "split.json").write_text(json.dumps({"seed": args.seed,
        "val_video_ids": sorted(video for video, split in assignments.items() if split == 'val'),
        "all_video_ids": sorted(assignments)}, indent=2) + "\n")
    print(json.dumps({"videos": len(rows), "tuples": len(manifests), "manifest": str(output / "tuples.jsonl")}))


if __name__ == "__main__":
    main()
