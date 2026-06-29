#!/usr/bin/env python3
import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml


TWO_STAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TWO_STAGE_ROOT.parent
DEFAULT_CONFIG = TWO_STAGE_ROOT / "configs" / "two_stage_avatar7_dongqing.yaml"
DEFAULT_STAGE1_SCRIPT = PROJECT_ROOT / "Echo_mimicV3_test" / "run_echomimicv3_dongqing_batch.py"
DEFAULT_STAGE2_SCRIPT = PROJECT_ROOT / "LatentSync_test" / "run_latentsync_dongqing_batch.py"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def require_key(data: dict, key: str) -> object:
    if key not in data or data[key] in (None, ""):
        raise ValueError(f"Missing required config key: {key}")
    return data[key]


def resolve_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def ffprobe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-hide_banner",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def run_command(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def add_flag(cmd: list[str], flag: str, value: object | None = None) -> None:
    if value is None:
        cmd.append(flag)
    else:
        cmd.extend([flag, str(value)])


def bool_value(config: dict, key: str, default: bool = False) -> bool:
    return bool(config.get(key, default))


def calculate_video_length(audio_seconds: float, fps: int, length_safety_frames: int) -> int:
    return int(math.ceil(audio_seconds * fps)) + length_safety_frames


def stage1_output_path(stage1_dir: Path, audio_path: Path) -> Path:
    return stage1_dir / f"{audio_path.stem}.mp4"


def load_stage1_item(manifest_path: Path, expected_audio: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest.get("items", [])
    for item in items:
        item_audio = Path(item.get("audio", "")).resolve()
        if item_audio == expected_audio.resolve() and item.get("output"):
            return item
    usable = [item for item in items if item.get("output")]
    if len(usable) == 1:
        return usable[0]
    raise RuntimeError(f"Could not identify Stage 1 item for {expected_audio} in {manifest_path}")


def copy_final_video(source: Path, target: Path, dry_run: bool) -> None:
    print(f"\nFinal video: {source}")
    print(f"Requested video_path: {target}")
    if dry_run:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)


def build_stage1_command(config: dict, image_path: Path, audio_path: Path, stage1_dir: Path, video_length: int) -> list[str]:
    params = config.get("echomimicv3", {}) or {}
    cmd = [
        sys.executable,
        str(DEFAULT_STAGE1_SCRIPT),
        "--workflow",
        str(params.get("workflow", "jimeng")),
        "--image",
        str(image_path),
        "--drive-audio",
        str(audio_path),
        "--output-dir",
        str(stage1_dir),
        "--video-length",
        str(video_length),
    ]

    list_options = {
        "sample_size": "--sample-size",
    }
    scalar_options = {
        "quality_preset": "--quality-preset",
        "num_inference_steps": "--num-inference-steps",
        "guidance_scale": "--guidance-scale",
        "audio_guidance_scale": "--audio-guidance-scale",
        "audio_scale": "--audio-scale",
        "neg_scale": "--neg-scale",
        "neg_steps": "--neg-steps",
        "seed": "--seed",
        "teacache_threshold": "--teacache-threshold",
        "num_skip_start_steps": "--num-skip-start-steps",
        "gpu_memory_mode": "--gpu-memory-mode",
        "weight_dtype": "--weight-dtype",
        "fps": "--fps",
        "tail_padding_seconds": "--tail-padding-seconds",
        "length_safety_frames": "--length-safety-frames",
        "max_segment_seconds": "--max-segment-seconds",
        "cfg_skip_ratio": "--cfg-skip-ratio",
        "shift": "--shift",
        "prompt": "--prompt",
        "negative_prompt": "--negative-prompt",
        "mouth_prompts": "--mouth-prompts",
    }
    flag_options = {
        "overwrite": "--overwrite",
        "use_dynamic_acfg": "--use-dynamic-acfg",
        "use_dynamic_cfg": "--use-dynamic-cfg",
    }

    for key, flag in scalar_options.items():
        if key in params and params[key] is not None:
            add_flag(cmd, flag, params[key])
    for key, flag in list_options.items():
        values = params.get(key)
        if values:
            cmd.append(flag)
            cmd.extend(str(value) for value in values)
    for key, flag in flag_options.items():
        if bool_value(params, key):
            add_flag(cmd, flag)

    auto_trim = params.get("auto_trim")
    if auto_trim is False:
        add_flag(cmd, "--no-auto-trim")
    elif auto_trim is True:
        add_flag(cmd, "--auto-trim")
    return cmd


def build_stage2_command(
    config: dict,
    stage1_video: Path,
    stage1_audio: Path,
    stage2_dir: Path,
) -> list[str]:
    params = config.get("latentsync", {}) or {}
    output_dir = stage2_dir / stage1_video.stem
    cmd = [
        sys.executable,
        str(DEFAULT_STAGE2_SCRIPT),
        "--video",
        str(stage1_video),
        "--audio-dir",
        str(stage1_audio.parent),
        "--only",
        stage1_audio.name,
        "--output-dir",
        str(output_dir),
    ]
    scalar_options = {
        "inference_steps": "--inference-steps",
        "guidance_scale": "--guidance-scale",
        "seed": "--seed",
    }
    for key, flag in scalar_options.items():
        if key in params and params[key] is not None:
            add_flag(cmd, flag, params[key])
    if bool_value(params, "skip_dep_check", False):
        add_flag(cmd, "--skip-dep-check")
    if bool_value(params, "enable_deepcache", True):
        add_flag(cmd, "--enable-deepcache")
    else:
        add_flag(cmd, "--disable-deepcache")
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a YAML-driven EchoMimicV3 -> LatentSync pipeline.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--skip-stage2", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    require_path(config_path, "Pipeline config")
    config = load_config(config_path)

    image_path = resolve_path(require_key(config, "image_path"))
    audio_path = resolve_path(require_key(config, "audio_path"))
    final_video_path = resolve_path(require_key(config, "video_path"))
    output_root = resolve_path(config.get("output_root", "LatentSync_test/generated_videos_yaml_two_stage"))
    stage1_dir = resolve_path(config.get("stage1_output_dir", output_root / "stage1_echomimicv3"))
    stage2_dir = resolve_path(config.get("stage2_output_dir", output_root / "stage2_latentsync"))

    require_path(image_path, "Input image")
    require_path(audio_path, "Input audio")

    echo_params = config.get("echomimicv3", {}) or {}
    fps = int(echo_params.get("fps", 25))
    length_safety_frames = int(echo_params.get("length_safety_frames", 4))
    tail_padding_seconds = float(echo_params.get("tail_padding_seconds", 0.0))
    audio_duration = ffprobe_duration(audio_path)
    target_duration = audio_duration + tail_padding_seconds + (length_safety_frames / fps)
    video_length = calculate_video_length(target_duration, fps, length_safety_frames)

    if "video_length" in echo_params:
        requested = int(echo_params["video_length"])
        if requested < video_length:
            raise ValueError(
                f"echomimicv3.video_length={requested} is shorter than the audio-based "
                f"minimum {video_length}. Remove it or raise it."
            )
        video_length = requested

    print(f"Config: {config_path}")
    print(f"Image: {image_path}")
    print(f"Audio: {audio_path} ({audio_duration:.3f}s)")
    print(f"Stage 1 video_length: {video_length} frames at {fps} FPS")

    if not args.skip_stage1:
        stage1_cmd = build_stage1_command(config, image_path, audio_path, stage1_dir, video_length)
        run_command(stage1_cmd, PROJECT_ROOT, args.dry_run)

    manifest_path = stage1_dir / "manifest.json"
    if args.dry_run:
        print(f"\nDry run: Stage 1 manifest would be read from {manifest_path}")
        return 0
    require_path(manifest_path, "Stage 1 manifest")
    stage1_item = load_stage1_item(manifest_path, audio_path)
    stage1_video = Path(stage1_item["output"]).resolve()
    stage1_audio = Path(stage1_item.get("padded_audio") or stage1_item["audio"]).resolve()
    require_path(stage1_video, "Stage 1 video")
    require_path(stage1_audio, "Stage 1 aligned audio")

    if not args.skip_stage2:
        stage2_cmd = build_stage2_command(config, stage1_video, stage1_audio, stage2_dir)
        run_command(stage2_cmd, PROJECT_ROOT, args.dry_run)

    stage2_manifest = stage2_dir / stage1_video.stem / "manifest.json"
    require_path(stage2_manifest, "Stage 2 manifest")
    stage2_data = json.loads(stage2_manifest.read_text(encoding="utf-8"))
    items = stage2_data.get("items", [])
    if not items or not items[0].get("output"):
        raise RuntimeError(f"No Stage 2 output found in {stage2_manifest}")
    stage2_video = Path(items[0]["output"]).resolve()
    require_path(stage2_video, "Stage 2 video")
    copy_final_video(stage2_video, final_video_path, args.dry_run)

    output_manifest = output_root / "manifest.json"
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": str(config_path),
        "image_path": str(image_path),
        "audio_path": str(audio_path),
        "video_path": str(final_video_path),
        "audio_duration_seconds": audio_duration,
        "fps": fps,
        "calculated_stage1_video_length": video_length,
        "stage1_manifest": str(manifest_path),
        "stage1_video": str(stage1_video),
        "stage1_audio": str(stage1_audio),
        "stage2_manifest": str(stage2_manifest),
        "stage2_video": str(stage2_video),
    }
    output_manifest.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nPipeline manifest: {output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
