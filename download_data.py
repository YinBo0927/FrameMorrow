"""Plan and fetch OpenVidHD with source-disjoint training/validation splits.

plan downloads only metadata. fetch reads only the selected ZIP members via
verified HTTP ranges, including split ZIPs; it never falls back to full archives.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re

from framemorrow.downloads import REPO, fetch_video, save_json, source_id


def video_count(value):
    return None if str(value).lower() == 'all' else int(str(value))


def part_files(files):
    parts = {}
    for path in files:
        match = re.fullmatch(r'OpenVidHD/OpenVidHD_part_(\d+)(\.zip|_part_[a-z]{2})', path)
        if match:
            parts.setdefault('part' + match[1], []).append(path)
    result = {}
    for part, names in parts.items():
        whole = [name for name in names if name.endswith('.zip')]
        result[part] = whole or sorted(names)
        if not whole:
            suffixes = [name.rsplit('_', 1)[1] for name in result[part]]
            expected = [chr(97 + i // 26) + chr(97 + i % 26) for i in range(len(suffixes))]
            if suffixes != expected:
                raise ValueError(f'incomplete split archive: {part}')
    return result


def select_videos(csv_path, index_path, archives, count, seed, min_seconds, min_aesthetic=None,
                  parts=None, val_fraction=.1):
    if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 2):
        raise ValueError('count must be at least 2, or all/null')
    if not math.isfinite(min_seconds) or min_seconds < 0:
        raise ValueError('min-seconds must be finite and nonnegative')
    if min_aesthetic is not None and not math.isfinite(min_aesthetic):
        raise ValueError('min-aesthetic must be finite')
    if not math.isfinite(val_fraction) or not 0 < val_fraction < 1:
        raise ValueError('val-fraction must be between 0 and 1')
    mapping = {}
    for item in json.loads(Path(index_path).read_text()):
        for part, names in item.items():
            if parts and part not in parts:
                continue
            for name in names:
                if name in mapping:
                    raise ValueError('duplicate filename in OpenVidHD archive index')
                mapping[name] = part
    choices, unknown_sources = {}, 0
    with Path(csv_path).open(newline='') as stream:
        for row in csv.DictReader(stream):
            name = row['video']
            if name not in mapping or mapping[name] not in archives:
                continue
            if Path(name).name != name or '\\' in name or not name.endswith('.mp4'):
                raise ValueError('unsafe video filename in metadata')
            duration, aesthetic = float(row['seconds']), float(row['aesthetic score'])
            if (not math.isfinite(duration) or not math.isfinite(aesthetic) or duration < min_seconds
                    or (min_aesthetic is not None and aesthetic < min_aesthetic)):
                continue
            try:
                source = source_id(name)
            except ValueError:
                # Keep unrecognized names together rather than assuming independent sources.
                source = 'unknown-source'
                unknown_sources += 1
            candidate = {'video_id': Path(name).stem, 'source_id': source, 'member': name,
                         'part': mapping[name], 'seconds': duration, 'aesthetic_score': aesthetic}
            if name in choices:
                raise ValueError('duplicate video filename in metadata')
            choices[name] = candidate
    if count is not None and len(choices) < count:
        raise ValueError(f'only {len(choices)} eligible videos; requested {count}')
    records = [choices[key] for key in sorted(choices)]
    random.Random(seed).shuffle(records)
    if count is not None:
        selected = records[:count]
        if selected and len({row['source_id'] for row in selected}) == 1:
            other = next((row for row in records[count:]
                          if row['source_id'] != selected[0]['source_id']), None)
            if other is not None:
                selected[-1] = other
        records = selected
    sources = sorted({row['source_id'] for row in records})
    if len(sources) < 2:
        raise ValueError('need at least two source groups for training and validation')
    random.Random(seed).shuffle(sources)
    val_count = min(len(sources) - 1, max(1, round(len(sources) * val_fraction)))
    validation = set(sources[:val_count])
    for row in records:
        row['split'] = 'val' if row['source_id'] in validation else 'train'
        row['remote_files'] = archives[row['part']]
    return records, unknown_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    plan = sub.add_parser('plan', help='freeze a download plan; defaults to all OpenVidHD videos')
    plan.add_argument('output')
    plan.add_argument('--settings', default=str(Path(__file__).resolve().parent / 'configs/default.json'))
    plan.add_argument('--count', type=video_count, default=None, help='video limit, or all (default)')
    plan.add_argument('--seed', type=int, default=0)
    plan.add_argument('--min-seconds', type=float, default=0.)
    plan.add_argument('--min-aesthetic', type=float)
    plan.add_argument('--val-fraction', type=float, default=.1, help='fraction of source groups held out')
    plan.add_argument('--parts', type=int, nargs='+', help='optional archive-pool restriction, e.g. 1 2')
    plan.add_argument('--revision', default='main')
    plan.add_argument('--endpoint', default=os.environ.get('HF_ENDPOINT', 'https://huggingface.co'))
    plan.add_argument('--metadata-dir', help='reuse local CSV/index; their hashes and origin are recorded')
    fetch = sub.add_parser('fetch', help='fetch all selected videos; safe to resume the same plan')
    fetch.add_argument('plan')
    initial, _ = parser.parse_known_args()
    if initial.command == 'plan':
        settings = json.loads(Path(initial.settings).read_text())['data']
        allowed = {'count', 'seed', 'min_seconds', 'min_aesthetic', 'val_fraction', 'parts', 'revision'}
        if not isinstance(settings, dict) or set(settings) - allowed:
            parser.error('unknown data settings; see configs/default.json')
        if 'count' in settings and settings['count'] is not None:
            settings['count'] = video_count(settings['count'])
        plan.set_defaults(**settings)
    args = parser.parse_args()
    if args.command == 'plan':
        output = Path(args.output).resolve()
        target = output / 'download_plan.json'
        if target.exists():
            raise FileExistsError('preserve the frozen plan; fetch it or choose a new output directory')
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi(endpoint=args.endpoint)
        info = api.repo_info(REPO, repo_type='dataset',
                             revision=None if args.revision == 'main' else args.revision, timeout=30)
        revision = info.sha
        files = [item.rfilename for item in info.siblings or []]
        metadata = Path(args.metadata_dir).resolve() if args.metadata_dir else output / 'metadata'
        if not args.metadata_dir:
            for name in ('data/train/OpenVidHD.csv', 'OpenVidHD/OpenVidHD.json'):
                hf_hub_download(REPO, name, repo_type='dataset', revision=revision,
                                endpoint=args.endpoint, local_dir=metadata)
        csv_path, index_path = metadata / 'data/train/OpenVidHD.csv', metadata / 'OpenVidHD/OpenVidHD.json'
        records, unknown = select_videos(csv_path, index_path, part_files(files), args.count,
                                         args.seed, args.min_seconds, args.min_aesthetic,
                                         {'part' + str(part) for part in args.parts} if args.parts else None,
                                         args.val_fraction)
        for row in records:
            row.update(video=str(output / 'videos' / row['member']), revision=revision,
                       endpoint=args.endpoint.rstrip('/'))
        save_json({'repo': REPO, 'revision': revision, 'seed': args.seed,
                   'count': args.count, 'val_fraction': args.val_fraction,
                   'min_seconds': args.min_seconds, 'min_aesthetic': args.min_aesthetic,
                   'archive_pool': args.parts, 'unknown_source_names': unknown,
                   'metadata_origin': 'local override' if args.metadata_dir else 'pinned repository files',
                   'metadata_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (csv_path, index_path)},
                   'records': records}, target)
        print(json.dumps({'plan': str(target), 'videos': len(records), 'revision': revision,
                          'video_bytes_transferred': 0}, indent=2))
    else:
        path = Path(args.plan).resolve()
        data = json.loads(path.read_text())
        records = data['records']
        if not records or data['repo'] != REPO:
            raise ValueError('invalid OpenVidHD download plan')
        for index, row in enumerate(records):
            fetch_video(row)
            print(f'downloaded or verified {index + 1}/{len(records)}', flush=True)
        manifest = path.parent / 'videos.jsonl'
        temporary = manifest.with_suffix('.jsonl.tmp')
        temporary.write_text(''.join(json.dumps({key: row[key] for key in
                              ('video_id', 'source_id', 'video', 'split')}) + '\n' for row in records))
        temporary.replace(manifest)
        print(json.dumps({'manifest': str(manifest), 'videos': len(records)}))


if __name__ == '__main__':
    main()
