#!/usr/bin/env python3
import argparse
import importlib
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def status(message: str):
    print(f"[avatar6-idle] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a silent EchoMimicV3 idle video for avatar6 from a single reference image.",
        add_help=True,
    )
    parser.add_argument("--duration", type=float, default=4.0, help="idle video duration in seconds")
    parser.add_argument("--fps", type=int, default=25, help="output fps")
    parser.add_argument("--output", default="outputs/avatar6_idle_echomimicv3.mp4", help="output mp4 path")
    parser.add_argument(
        "--frames_dir",
        default="data/avatars/avatar6/echomimicv3/idle_frames",
        help="directory to save extracted idle frames",
    )
    parser.add_argument(
        "--ref_image",
        default="assets/avatars/avatar6.png",
        help="avatar6 reference image path",
    )
    parser.add_argument("--sample_size", type=int, nargs=2, default=[768, 768], help="EchoMimicV3 sample size H W")
    parser.add_argument("--num_steps", type=int, default=30, help="EchoMimicV3 diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=5.5)
    parser.add_argument("--audio_guidance_scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument(
        "--prompt",
        default=(
            "A realistic front-facing male digital human presenter is idle and silent, "
            "breathing subtly, blinking naturally, showing tiny micro-expressions, "
            "with very slight head movement, calm broadcast-style expression, stable body, "
            "sharp eyes, realistic skin texture, studio-quality portrait lighting."
        ),
        help="EchoMimicV3 positive prompt for idle motion",
    )
    parser.add_argument(
        "--negative_prompt",
        default=(
            "speaking, talking, open mouth, exaggerated motion, large head movement, "
            "blur, low quality, distorted face, deformed mouth, bad teeth, unnatural lips, "
            "jitter, flicker, warped face, cross-eyed, text, watermark, logo, cartoon, anime, painting"
        ),
        help="EchoMimicV3 negative prompt",
    )

    script_args, server_args = parser.parse_known_args()
    return script_args, server_args


def parse_server_opt(server_args: list[str]):
    from config import parse_args as parse_server_args

    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0]] + server_args
        return parse_server_args()
    finally:
        sys.argv = old_argv


def resolve_project_path(path: str) -> Path:
    target = Path(path)
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    return target.resolve()


def apply_defaults(opt, args):
    opt.model = "echomimicv3"
    opt.avatar_id = "avatar6"
    opt.transport = "webrtc"
    opt.tts = "edgetts"
    opt.sessionid = "generate-avatar6-idle"
    opt.fps = args.fps
    opt.echomimicv3_sample_size = args.sample_size
    opt.echomimicv3_video_length = max(1, int(round(args.duration * args.fps)))
    opt.echomimicv3_num_steps = args.num_steps
    opt.echomimicv3_guidance_scale = args.guidance_scale
    opt.echomimicv3_audio_guidance_scale = args.audio_guidance_scale
    opt.echomimicv3_seed = args.seed
    opt.echomimicv3_prompt = args.prompt

    defaults = {
        "echomimicv3_repo": "third_party/echomimic_v3",
        "echomimicv3_model_dir": "EchoMimicV3",
        "echomimicv3_base_model_dir": "Wan2.1-Fun-1.3B-InP",
        "echomimicv3_wav2vec_dir": "chinese-wav2vec2-base",
    }
    for attr, default in defaults.items():
        value = getattr(opt, attr, "") or default
        setattr(opt, attr, str(resolve_project_path(value)))

    if getattr(opt, "echomimicv3_transformer_path", ""):
        opt.echomimicv3_transformer_path = str(resolve_project_path(opt.echomimicv3_transformer_path))
    if getattr(opt, "echomimicv3_config_path", ""):
        opt.echomimicv3_config_path = str(resolve_project_path(opt.echomimicv3_config_path))
    return opt


def ensure_avatar6_ref_image(ref_image: Path):
    if not ref_image.exists():
        raise FileNotFoundError(f"avatar6 reference image not found: {ref_image}")

    avatar_dir = PROJECT_ROOT / "data" / "avatars" / "avatar6" / "echomimicv3"
    avatar_dir.mkdir(parents=True, exist_ok=True)


def save_video(frames: list[np.ndarray], output_path: Path, fps: int):
    if not frames:
        raise RuntimeError("EchoMimicV3 returned empty frames")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {output_path}")
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()


def save_frames(frames: list[np.ndarray], frames_dir: Path):
    frames_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        path = frames_dir / f"{index:06d}.png"
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"failed to write frame: {path}")


def main():
    args, server_args = parse_args()
    opt = apply_defaults(parse_server_opt(server_args), args)
    os.chdir(PROJECT_ROOT)

    ref_image = resolve_project_path(args.ref_image)
    output_path = resolve_project_path(args.output)
    frames_dir = resolve_project_path(args.frames_dir)
    ensure_avatar6_ref_image(ref_image)

    target_frames = max(1, int(round(args.duration * args.fps)))
    audio_samples = target_frames * 2 * (16000 // (args.fps * 2))
    silent_audio = np.zeros(audio_samples, dtype=np.float32)

    status(f"project_root={PROJECT_ROOT}")
    status(f"ref_image={ref_image}")
    status(f"duration={args.duration:.2f}s fps={args.fps} target_frames={target_frames}")
    status(f"output={output_path}")
    status(f"frames_dir={frames_dir}")
    status(f"wav2vec_dir={opt.echomimicv3_wav2vec_dir}")
    status(f"base_model_dir={opt.echomimicv3_base_model_dir}")
    status(f"num_steps={opt.echomimicv3_num_steps}")

    avatar_mod = importlib.import_module("avatars.echomimicv3_avatar")
    status("loading EchoMimicV3 model; this can take several minutes")
    model = avatar_mod.load_model(opt)

    started_at = time.perf_counter()
    status("generating silent idle motion")
    frames = model.generate_frames(
        str(ref_image),
        silent_audio,
        args.prompt,
        args.negative_prompt,
        max_video_length=target_frames,
    )
    frames = frames[:target_frames]
    if not frames:
        raise RuntimeError("EchoMimicV3 returned empty frames")

    status(f"saving video frames={len(frames)}")
    save_video(frames, output_path, args.fps)
    save_frames(frames, frames_dir)

    elapsed = time.perf_counter() - started_at
    status(f"done elapsed={elapsed:.2f}s video={output_path}")
    status(f"idle frames saved={frames_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
