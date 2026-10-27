"""Fresh Qwen3-VL segment annotations; no old selector annotations are needed."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import torch

from framemorrow.data import read_manifest, resolve_path

FIELDS = ("identities", "scene", "actions", "state_changes")
INSTRUCTION = (
    "Describe these four ordered frames of one two-second video unit. Return only a JSON object "
    "with exactly these string fields: identities, scene, actions, state_changes. Preserve visible "
    "subject appearance, objects, scene attributes, actions and state changes. Do not invent "
    "unsupported objects or events. Do not include timestamps, predictions or later events. "
    "Use concise descriptions, at most 35 words total; use an empty string when a field is unsupported."
)


def parse_annotation(text):
    text = text.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    annotation = json.loads(text)
    if set(annotation) != set(FIELDS) or not all(isinstance(annotation[k], str) for k in FIELDS):
        raise ValueError("annotation schema mismatch")
    if not any(v.strip() for v in annotation.values()):
        raise ValueError("empty annotation")
    return annotation


def read_annotation_cache(path):
    cache = {}
    if not path.exists():
        return cache
    with path.open('r+b') as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if not line.endswith(b'\n') and not stream.read(1):
                    stream.truncate(offset)
                    break
                raise
            cache[record['video_id'], record['start']] = record
            if not line.endswith(b'\n'):
                stream.write(b'\n')
                break
    return cache


@torch.inference_mode()
def describe(paths, model, processor, attempts=2, strict=False):
    for attempt in range(attempts):
        prompt = INSTRUCTION + (" Strictly output valid JSON without Markdown or commentary." if strict or attempt else "")
        content = [{"type": "image", "url": str(p)} for p in paths]
        content.append({"type": "text", "text": prompt})
        inputs = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                    add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
        inputs.pop("token_type_ids", None)
        ids = model.generate(**inputs, do_sample=False, max_new_tokens=256)
        text = processor.decode(ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        try:
            return parse_annotation(text)
        except (ValueError, TypeError):
            pass
    return None


@torch.inference_mode()
def describe_many(path_groups, model, processor):
    """Batch first attempts; each malformed result gets exactly one strict retry."""
    processor.tokenizer.padding_side = "left"
    conversations = []
    for paths in path_groups:
        content = [{"type": "image", "url": str(path)} for path in paths]
        content.append({"type": "text", "text": INSTRUCTION})
        conversations.append([{"role": "user", "content": content}])
    inputs = processor.apply_chat_template(conversations, tokenize=True, padding=True,
                add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
    inputs.pop("token_type_ids", None)
    ids = model.generate(**inputs, do_sample=False, max_new_tokens=256)
    texts = processor.batch_decode(ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    results = []
    for paths, text in zip(path_groups, texts):
        try:
            results.append(parse_annotation(text))
        except (ValueError, TypeError):
            results.append(describe(paths, model, processor, attempts=1, strict=True))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", help="JSONL rows with video_id, video path, optional source_id/split")
    parser.add_argument("output")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--revision", help="optional pinned Qwen model commit")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--condition-seconds", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="reuse successful units under identical settings")
    args = parser.parse_args()
    if args.condition_seconds < 2 or args.condition_seconds % 2:
        raise ValueError("condition duration must be a positive multiple of the two-second annotation unit")
    rows = read_manifest(args.videos)
    ids = [row['video_id'] for row in rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('annotation requires nonempty, unique video IDs')
    output = Path(args.output).resolve()
    settings_path = output / 'annotation_settings.json'
    log_path = output / 'annotations.jsonl'
    expected = {"model": args.model, "revision": args.revision, "instruction": INSTRUCTION,
                "sampling": "greedy (temperature zero)", "max_new_tokens": 256,
                "unit_seconds": 2, "frames_per_unit": 4, "condition_seconds": args.condition_seconds,
                "input_manifest_sha256": hashlib.sha256(Path(args.videos).read_bytes()).hexdigest()}
    cache = {}
    if args.resume:
        previous = json.loads(settings_path.read_text())
        if any(previous.get(key) != value for key, value in expected.items()):
            raise ValueError('resume settings or input manifest changed; choose a new annotation directory')
        cache = read_annotation_cache(log_path)
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError("choose an empty annotation output directory, or use --resume")
    output.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(expected, indent=2) + '\n')
    model, processor = None, None
    videos, failures = [], []
    with log_path.open('a' if args.resume else 'w') as annotations:
        for i, row in enumerate(rows):
            video = resolve_path(row, row['video']).resolve()
            signature = [video.stat().st_size, video.stat().st_mtime_ns]
            probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries",
                    "format=duration", "-of", "json", str(video)], text=True))
            duration = float(probe['format']['duration'])
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError('annotation video must have a finite positive duration')
            item = {'video_id': row['video_id'], 'video': str(video), 'segments': []}
            item.update({key: row[key] for key in ('source_id', 'split') if key in row})
            if duration < 2:
                failures.append({'video_id': row['video_id'], 'reason': 'shorter than two-second annotation unit',
                                 'duration': duration})
                videos.append(item)
                print(f"skipped video {i + 1}: duration {duration:.3f}s < 2s", flush=True)
                continue
            units = []
            for unit in range(math.floor(duration / 2)):
                cached = cache.get((row['video_id'], 2 * unit))
                if cached is not None and cached.get('video_signature') != signature:
                    raise ValueError('source video changed since annotation; choose a new output directory')
                annotation = cached.get('annotation') if cached else None
                if annotation is not None:
                    annotation = parse_annotation(json.dumps(annotation))
                else:
                    if model is None:
                        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
                        model = Qwen3VLForConditionalGeneration.from_pretrained(args.model,
                            revision=args.revision, torch_dtype=torch.bfloat16,
                            attn_implementation='eager').to(args.device).eval()
                        model.requires_grad_(False)
                        processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
                    directory = output / 'frames' / f'{i:06d}' / f'{unit:05d}'
                    directory.mkdir(parents=True, exist_ok=True)
                    paths = []
                    for j in range(4):
                        path = directory / f'{j}.jpg'
                        subprocess.run(['ffmpeg', '-nostdin', '-y', '-hide_banner', '-loglevel', 'error',
                            '-ss', str(2 * unit + j * .5), '-i', str(video), '-frames:v', '1',
                            '-vf', 'scale=448:-2', str(path)], check=True)
                        paths.append(path)
                    annotation = describe(paths, model, processor)
                    annotations.write(json.dumps({'video_id': row['video_id'], 'start': 2 * unit,
                        'end': 2 * unit + 2, 'annotation': annotation, 'video_signature': signature}) + '\n')
                    annotations.flush()
                units.append(annotation)
                if annotation is None:
                    failures.append({'video_id': row['video_id'], 'unit': unit})
                print(f'annotated or restored video {i + 1}, unit {unit + 1}/{math.floor(duration / 2)}', flush=True)
            group_size = args.condition_seconds // 2
            segments = []
            for begin in range(0, len(units) - group_size + 1, group_size):
                group = units[begin:begin + group_size]
                if any(a is None for a in group):
                    continue
                descriptions = ['; '.join(a[field].strip() for field in FIELDS if a[field].strip()) for a in group]
                segments.append({'start': begin * 2, 'end': (begin + group_size) * 2,
                                 'prompt': '. Then '.join(descriptions)})
            item['segments'] = segments
            videos.append(item)
    destination = output / 'videos.jsonl'
    temporary = destination.with_suffix('.jsonl.tmp')
    temporary.write_text(''.join(json.dumps(row) + '\n' for row in videos))
    temporary.replace(destination)
    settings_path.write_text(json.dumps({**expected, 'failures': failures,
        'videos_with_segments': sum(bool(row['segments']) for row in videos)}, indent=2) + '\n')


if __name__ == "__main__":
    main()
