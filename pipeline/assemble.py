"""Functional FFmpeg timeline assembly and audio mixing module.

Concatenates sequential shot clips, mixes speech and ambient music,
and exports the final master video.
"""

import subprocess
from pathlib import Path
from typing import Optional, Sequence


def build_concat_list_file(video_paths: Sequence[str], list_file_path: str) -> str:
    """Writes an FFmpeg concat demuxer text file."""
    path = Path(list_file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for vp in video_paths:
            f.write(f"file '{Path(vp).resolve()}'\n")
    return str(path.resolve())


def concatenate_clips(concat_list_path: str, output_path: str) -> str:
    """Concatenates video clips into a single continuous video stream using FFmpeg."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list_path,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "fast",
        str(out.resolve()),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return str(out.resolve())


def mix_scene_audio(
    video_path: str,
    output_path: str,
    speech_audio_path: Optional[str] = None,
    ambient_audio_path: Optional[str] = None,
    ambient_volume: float = 0.15,
) -> str:
    """Layers speech dialogue and ducked ambient audio under the video track."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-y", "-i", video_path]
    inputs = 1

    if speech_audio_path and Path(speech_audio_path).exists():
        cmd.extend(["-i", speech_audio_path])
        inputs += 1

    if ambient_audio_path and Path(ambient_audio_path).exists():
        cmd.extend(["-i", ambient_audio_path])
        inputs += 1

    if inputs == 1:
        # No extra audio, copy or silent audio
        cmd.extend(["-c:v", "copy", str(out.resolve())])
    elif inputs == 2:
        # 1 audio stream (speech or ambient)
        cmd.extend([
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(out.resolve()),
        ])
    else:
        # Both speech and ambient: mix and duck ambient
        filter_complex = (
            f"[1:a]volume=1.0[speech];"
            f"[2:a]volume={ambient_volume}[bg];"
            f"[speech][bg]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        )
        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            str(out.resolve()),
        ])

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return str(out.resolve())
