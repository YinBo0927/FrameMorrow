"""Source grouping and verified selective downloads from OpenVidHD ZIP archives.

Only selected members are transferred. ZIP byte ranges, member CRC and saved
SHA256 are checked. Downloads use advisory file locks on Linux/macOS.
"""
import fcntl
import hashlib
import io
import json
from pathlib import Path
import re
import threading
import time
import zipfile

REPO = "nkp37/OpenVid-1M"


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def source_id(name):
    stem = Path(name).stem
    match = re.fullmatch(r"([A-Za-z0-9_-]{11})_\d+_\d+to\d+", stem)
    if match:
        return "youtube:" + match[1]
    match = re.fullmatch(r"magictime_(\d+)(?:_\d+)*", stem)
    if match:
        return "magictime:" + match[1]
    raise ValueError(f"cannot safely group unknown source filename: {name}")



class RangeArchive(io.RawIOBase):
    """Seekable concatenation of public HTTP files; every byte range is verified."""
    def __init__(self, urls):
        import requests
        super().__init__()
        self.session = requests.Session()
        self.urls = urls
        self.sizes = []
        self.position = 0
        self.cache = b""
        self.cache_start = 0
        try:
            for url in urls:
                _, total = self.request(url, 0, 1)
                self.sizes.append(total)
        except BaseException:
            self.close()
            raise
        self.size = sum(self.sizes)

    def request(self, url, start, length):
        error = None
        for attempt in range(5):
            try:
                # Different resolve queries also avoid intermediary caches that omit
                # Range from the cache key. Never log signed redirect URLs.
                query = f"?download=true&range={start}-{start + length - 1}"
                with self.session.get(url + query, headers={"Range": f"bytes={start}-{start + length - 1}"},
                                      stream=True, timeout=(20, 120)) as response:
                    if response.status_code != 206:
                        raise IOError(f"range request returned HTTP {response.status_code}")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not match or tuple(map(int, match.groups()[:2])) != (start, start + length - 1):
                        raise IOError("server returned a different byte range")
                    content = response.content
                    if len(content) != length:
                        raise IOError("truncated byte range")
                    return content, int(match[3])
            except Exception as failure:
                # requests exceptions may contain signed URLs; retain a safe message.
                error = type(failure).__name__
                if attempt < 4:
                    time.sleep(min(2 ** attempt, 8))
        raise IOError(f"verified range download failed after five attempts ({error})")

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if position < 0:
            raise ValueError("negative archive seek")
        self.position = position
        return position

    def read(self, length=-1):
        if length < 0:
            length = self.size - self.position
        length = min(length, self.size - self.position)
        if length <= 0:
            return b""
        start = self.position
        end = start + length
        if not (self.cache_start <= start and end <= self.cache_start + len(self.cache)):
            amount = min(max(length, 256 * 1024), self.size - start)
            chunks, offset, remaining, absolute = [], start, amount, 0
            for url, size in zip(self.urls, self.sizes):
                if offset < absolute + size and remaining:
                    local = max(0, offset - absolute)
                    count = min(remaining, size - local)
                    content, total = self.request(url, local, count)
                    if total != size:
                        raise IOError("archive resource size changed")
                    chunks.append(content)
                    remaining -= count
                    offset += count
                absolute += size
            self.cache_start, self.cache = start, b"".join(chunks)
        self.position = end
        return self.cache[start - self.cache_start:end - self.cache_start]

    def close(self):
        if hasattr(self, "session"):
            self.session.close()
        super().close()


_archive_threads = threading.local()


def fetch_video(row):
    destination = Path(row["video"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Prefetch and GPU preparation can request the same clip concurrently.
    with destination.with_suffix('.download.lock').open('w') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _fetch_video_locked(row)


def _fetch_video_locked(row):
    destination = Path(row["video"])
    if Path(row['member']).name != row['member'] or '\\' in row['member']:
        raise ValueError('unsafe selected member name')
    provenance = destination.with_suffix(".provenance.json")
    if destination.exists():
        if not provenance.is_file():
            raise ValueError("existing video lacks download provenance; choose a new destination")
        saved = json.loads(provenance.read_text())
        if (saved.get('repo') != REPO or saved.get('video_id') != row['video_id']
                or Path(saved.get('zip_member', '')).name != row['member']
                or saved.get("revision") != row["revision"] or saved.get("remote_files") != row["remote_files"]
                or saved.get("sha256") != hashlib.sha256(destination.read_bytes()).hexdigest()):
            raise ValueError("existing video or provenance changed; preserve it and use a new destination")
        return destination
    key = (row["endpoint"], row["revision"], tuple(row["remote_files"]))
    if not hasattr(_archive_threads, 'archives'):
        _archive_threads.archives = {}
    archives = _archive_threads.archives
    started = time.time()
    if key not in archives:
        # Each archive owns HTTP connections; bound the cache in long prefetch runs.
        if len(archives) >= 4:
            previous = archives.pop(next(iter(archives)))
            reader = previous.fp
            previous.close()
            reader.close()
        urls = [f"{row['endpoint']}/datasets/{REPO}/resolve/{row['revision']}/{name}"
                for name in row["remote_files"]]
        reader = RangeArchive(urls)
        try:
            archives[key] = zipfile.ZipFile(reader)
        except BaseException:
            reader.close()
            raise
    archive = archives.pop(key)
    archives[key] = archive
    member = row['member']
    if member not in archive.namelist():
        matches = [name for name in archive.namelist() if Path(name).name == member]
        if len(matches) != 1:
            raise ValueError('selected filename does not identify exactly one archive member')
        member = matches[0]
    # ZipExtFile verifies the member CRC when read completely. No arbitrary paths
    # are extracted and no full 50GB archive is downloaded for one video.
    payload = archive.read(member)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".mp4.tmp")
    temporary.write_bytes(payload)
    save_json({"video_id": row["video_id"], "repo": REPO, "revision": row["revision"],
               "remote_files": row["remote_files"], "zip_member": member,
               "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
               "download_seconds_including_archive_lookup": time.time() - started},
              destination.with_suffix(".provenance.json"))
    temporary.replace(destination)
    return destination
