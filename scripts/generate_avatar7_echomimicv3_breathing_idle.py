#!/usr/bin/env python3
import argparse
import importlib
import json
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
    print(f"[avatar7-idle] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate silent, loopable EchoMimicV3 breathing idle material for avatar7.",
    )
    parser.add_argument("--duration", type=float, default=6.0, help="final loop duration in seconds")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--ref_image", default="assets/avatars/avatar7.png")
    parser.add_argument("--frames_dir", default="data/avatars/avatar7/echomimicv3/idle_frames")
    parser.add_argument("--output", default="outputs/avatar7_breathing_idle_loop.mp4")
    parser.add_argument("--sample_size", type=int, nargs=2, default=[768, 768], help="H W")
    parser.add_argument("--num_steps", type=int, default=80)
    parser.add_argument("--guidance_scale", type=float, default=5.5)
    parser.add_argument("--audio_guidance_scale", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument(
        "--prompt",
        default=(
            "A realistic front-facing digital human is silent and idle, breathing very subtly, "
            "with tiny natural chest movement, occasional soft blinking, calm neutral expression, "
            "mouth closed, stable head pose, stable camera, realistic portrait lighting."
        ),
    )
    parser.add_argument(
        "--negative_prompt",
        default=(
            "speaking, talking, open mouth, lip movement, large gesture, large head movement, "
            "body sway, jitter, flicker, blur, low quality, distorted face, deformed mouth, "
            "bad teeth, text, watermark, logo, cartoon, anime, painting"
        ),
    )
    script_args, server_args = parser.parse_known_args()
    return script_args, server_args


def parse_server_opt(server_args):
    from config import parse_args as parse_server_args

    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0]] + server_args
        return parse_server_args()
    finally:
        sys.argv = old_argv


def resolve(path: str) -> Path:
    target = Path(path)
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    return target.resolve()


def apply_defaults(opt, args):
    opt.model = "echomimicv3"
    opt.avatar_id = "avatar7"
    opt.transport = "webrtc"
    opt.tts = "edgetts"
    opt.sessionid = "generate-avatar7-breathing-idle"
    opt.fps = args.fps
    opt.echomimicv3_sample_size = args.sample_size
    opt.echomimicv3_video_length = max(1, int(round(args.duration * args.fps / 2)))
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
        setattr(opt, attr, str(resolve(value)))

    if getattr(opt, "echomimicv3_transformer_path", ""):
        opt.echomimicv3_transformer_path = str(resolve(opt.echomimicv3_transformer_path))
    if getattr(opt, "echomimicv3_config_path", ""):
        opt.echomimicv3_config_path = str(resolve(opt.echomimicv3_config_path))
    return opt


def clear_old_frames(frames_dir: Path):
    frames_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        for path in frames_dir.glob(pattern):
            path.unlink()


def save_frames(frames, frames_dir: Path):
    clear_old_frames(frames_dir)
    for index, frame in enumerate(frames):
        path = frames_dir / f"{index:06d}.png"
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"failed to write frame: {path}")


def save_avatar_config(args):
    config_dir = PROJECT_ROOT / "data" / "avatars" / "avatar7" / "echomimicv3"
    config_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "avatar7 silent breathing idle loop",
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
    }
    with (config_dir / "avatar_config.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def save_video(frames, output: Path, fps: int):
    output.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {output}")
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()


def main():
    args, server_args = parse_args()
    opt = apply_defaults(parse_server_opt(server_args), args)
    os.chdir(PROJECT_ROOT)

    ref_image = resolve(args.ref_image)
    frames_dir = resolve(args.frames_dir)
    output = resolve(args.output)
    if not ref_image.exists():
        raise FileNotFoundError(f"avatar7 reference image not found: {ref_image}")

    final_frames = max(2, int(round(args.duration * args.fps)))
    seed_frames = max(2, final_frames // 2 + 1)
    chunk = 16000 // (args.fps * 2)
    silent_audio = np.zeros(seed_frames * 2 * chunk, dtype=np.float32)

    status(f"project_root={PROJECT_ROOT}")
    status(f"ref_image={ref_image}")
    status(f"frames_dir={frames_dir}")
    status(f"output={output}")
    status(f"final_duration={args.duration:.2f}s fps={args.fps} final_frames={final_frames}")
    status(f"seed_frames={seed_frames} num_steps={args.num_steps}")

    avatar_mod = importlib.import_module("avatars.echomimicv3_avatar")
    status("loading EchoMimicV3 model; this can take several minutes")
    model = avatar_mod.load_model(opt)

    started_at = time.perf_counter()
    status("generating silent subtle breathing seed motion")
    seed = model.generate_frames(
        str(ref_image),
        silent_audio,
        args.prompt,
        args.negative_prompt,
        max_video_length=seed_frames,
    )
    if not seed:
        raise RuntimeError("EchoMimicV3 returned empty frames")

    seed = seed[:seed_frames]
    loop_frames = seed + list(reversed(seed[:-1]))
    loop_frames = loop_frames[:final_frames]
    if len(loop_frames) < final_frames:
        loop_frames.extend(loop_frames[: final_frames - len(loop_frames)])

    save_frames(loop_frames, frames_dir)
    save_video(loop_frames, output, args.fps)
    save_avatar_config(args)

    elapsed = time.perf_counter() - started_at
    status(f"done elapsed={elapsed:.2f}s")
    status(f"idle frames saved={frames_dir}")
    status(f"silent preview video={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
