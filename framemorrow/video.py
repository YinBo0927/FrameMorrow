"""Original RGB frames near the centers of a two-fps sampling grid."""
import json
from pathlib import Path
import re
import subprocess


def sample_video(video, directory):
    directory = Path(directory)
    metadata = directory / "frames.json"
    if metadata.exists():
        result = json.loads(metadata.read_text())
        if all(Path(frame["path"]).is_file() for frame in result["frames"]):
            return result
        raise ValueError("saved sampling metadata refers to missing RGB frames")
    directory.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(["ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "info",
                    "-threads", "2", "-i", str(video), "-vf",
                    "select='gte(t,0.25+selected_n*0.5)',showinfo,scale=448:-2", "-vsync", "0",
                    "-q:v", "2", str(directory / "%06d.jpg")], check=True, stderr=subprocess.PIPE, text=True)
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration,start_time", "-of", "json", str(video)], text=True))
    duration = float(probe["format"]["duration"])
    origin = float(probe["format"].get("start_time", 0))
    # showinfo reads the actual decoder timestamps; a second full video decode is unnecessary.
    times = [float(time) for time in re.findall(r"\bn:\s*\d+.*?\bpts_time:([-+\d.eE]+)", result.stderr)]
    files = sorted(directory.glob("*.jpg"))
    if len(files) != len(times) or not files or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("FFmpeg frame selection differs from source presentation timestamps")
    result = {"duration": duration, "source_start_time": origin,
              "frames": [{"path": str(path.resolve()), "time": time} for path, time in zip(files, times)]}
    temporary = metadata.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result) + "\n")
    temporary.replace(metadata)
    return result
