#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: str) -> Path:
    target = Path(path)
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    return target.resolve()


def parse_args():
    parser = argparse.ArgumentParser(description="Assemble avatar6 idle frame images into an mp4 preview.")
    parser.add_argument(
        "--frames_dir",
        default="data/avatars/avatar6/echomimicv3/idle_frames",
        help="directory containing ordered idle frames",
    )
    parser.add_argument(
        "--output",
        default="outputs/avatar6_idle_preview.mp4",
        help="output mp4 path",
    )
    parser.add_argument("--fps", type=int, default=25, help="output fps")
    parser.add_argument(
        "--mirror_loop",
        type=int,
        default=0,
        help="append reversed middle frames for a smoother loop preview",
    )
    parser.add_argument(
        "--codec",
        choices=["h264", "mp4v"],
        default="h264",
        help="output codec; h264 is best for browser/IDE preview",
    )
    return parser.parse_args()


def list_frame_paths(frames_dir: Path):
    paths = []
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        paths.extend(frames_dir.glob(pattern))
    return sorted(paths)


def write_mp4v(frame_paths, output: Path, fps: int):
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise RuntimeError(f"cannot read first frame: {frame_paths[0]}")

    height, width = first.shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {output}")

    written = 0
    try:
        for path in frame_paths:
            frame = cv2.imread(str(path))
            if frame is None:
                print(f"[assemble-idle] skip unreadable frame: {path}", flush=True)
                continue
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
            written += 1
    finally:
        writer.release()

    if written <= 0:
        raise RuntimeError("no readable frames")
    return written


def write_h264(frame_paths, output: Path, fps: int):
    if shutil.which("ffmpeg") is None:
        print("[assemble-idle] ffmpeg not found, fallback to mp4v", flush=True)
        return write_mp4v(frame_paths, output, fps)

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise RuntimeError(f"cannot read first frame: {frame_paths[0]}")

    height, width = first.shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="avatar6_idle_frames_") as temp_dir:
        temp_dir = Path(temp_dir)
        written = 0
        for index, path in enumerate(frame_paths):
            frame = cv2.imread(str(path))
            if frame is None:
                print(f"[assemble-idle] skip unreadable frame: {path}", flush=True)
                continue
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            temp_path = temp_dir / f"{written:06d}.png"
            if not cv2.imwrite(str(temp_path), frame):
                raise RuntimeError(f"failed to write temp frame: {temp_path}")
            written += 1

        if written <= 0:
            raise RuntimeError("no readable frames")

        command = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(temp_dir / "%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
        subprocess.run(command, check=True)
        return written


def main():
    args = parse_args()
    frames_dir = resolve_project_path(args.frames_dir)
    output = resolve_project_path(args.output)
    frame_paths = list_frame_paths(frames_dir)

    if not frame_paths:
        raise FileNotFoundError(f"no frame images found in: {frames_dir}")

    if args.mirror_loop and len(frame_paths) > 2:
        frame_paths = frame_paths + list(reversed(frame_paths[1:-1]))

    if args.codec == "h264":
        written = write_h264(frame_paths, output, args.fps)
    else:
        written = write_mp4v(frame_paths, output, args.fps)

    print(f"[assemble-idle] frames_dir={frames_dir}", flush=True)
    print(f"[assemble-idle] frames={written} fps={args.fps}", flush=True)
    print(f"[assemble-idle] output={output}", flush=True)


if __name__ == "__main__":
    main()
