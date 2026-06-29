#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE1_DIR = (
    PROJECT_ROOT
    / "Echo_mimicV3_test"
    / "generated_videos_woman04_dongqing_morning_jimeng_sync"
)
DEFAULT_STAGE2_DIR = PROJECT_ROOT / "LatentSync_test" / "generated_videos_dongqing_two_stage"
DEFAULT_IMAGE = PROJECT_ROOT / "Echo_mimicV3_test" / "official_materials" / "imgs" / "demo_ch_woman_04.png"
DEFAULT_DRIVE_AUDIO = (
    PROJECT_ROOT
    / "gpt_sovits_official_materials"
    / "generated_bilibili_refs_tts"
    / "DongQing_6s"
    / "01_morning_breakfast.wav"
)


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def run_command(cmd: list[str], cwd: Path) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a 24GB-friendly EchoMimicV3 -> LatentSync two-stage DongQing pipeline."
    )
    parser.add_argument("--stage1-dir", type=Path, default=DEFAULT_STAGE1_DIR)
    parser.add_argument("--stage2-dir", type=Path, default=DEFAULT_STAGE2_DIR)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--drive-audio", type=Path, default=DEFAULT_DRIVE_AUDIO)
    parser.add_argument("--only", default="01_morning_breakfast")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--skip-dep-check", action="store_true")
    parser.add_argument("--no-auto-trim", action="store_true")
    parser.add_argument("--stage1-sample-size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--stage1-steps", type=int, default=15)
    parser.add_argument("--latentsync-steps", type=int, default=30)
    parser.add_argument("--latentsync-guidance-scale", type=float, default=2.2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stage1_dir = args.stage1_dir.resolve()
    stage2_dir = args.stage2_dir.resolve()
    image_path = args.image.resolve()
    drive_audio = args.drive_audio.resolve()

    require_path(image_path, "Stage 1 image")
    require_path(drive_audio, "Stage 1 drive audio")

    if not args.skip_stage1:
        stage1_cmd = [
            "Echo_mimicV3_test/run_echomimicv3_dongqing_batch.sh",
            "--workflow",
            "jimeng",
            "--image",
            str(image_path),
            "--drive-audio",
            str(drive_audio),
            "--output-dir",
            str(stage1_dir),
            "--only",
            args.only,
            "--sample-size",
            str(args.stage1_sample_size[0]),
            str(args.stage1_sample_size[1]),
            "--num-inference-steps",
            str(args.stage1_steps),
        ]
        if args.no_auto_trim:
            stage1_cmd.append("--no-auto-trim")
        if args.overwrite:
            stage1_cmd.append("--overwrite")
        run_command(stage1_cmd, PROJECT_ROOT)

    manifest_path = stage1_dir / "manifest.json"
    require_path(manifest_path, "Stage 1 manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = [
        item
        for item in manifest.get("items", [])
        if item.get("status") in {"generated", "skipped"} and item.get("output")
    ]
    if not items:
        raise RuntimeError(f"No usable Stage 1 items found in {manifest_path}")

    stage2_items = []
    for item in items:
        video_path = Path(item["output"]).resolve()
        audio_path = Path(item.get("padded_audio") or item["audio"]).resolve()
        require_path(video_path, "Stage 1 output video")
        require_path(audio_path, "Stage 1 aligned audio")

        audio_dir = audio_path.parent
        output_dir = stage2_dir / video_path.stem
        stage2_cmd = [
            "LatentSync_test/run_latentsync_dongqing_batch.sh",
            "--video",
            str(video_path),
            "--audio-dir",
            str(audio_dir),
            "--only",
            audio_path.name,
            "--output-dir",
            str(output_dir),
            "--inference-steps",
            str(args.latentsync_steps),
            "--guidance-scale",
            str(args.latentsync_guidance_scale),
        ]
        if args.skip_dep_check:
            stage2_cmd.append("--skip-dep-check")
        run_command(stage2_cmd, PROJECT_ROOT)
        stage2_items.append(
            {
                "stage1_video": str(video_path),
                "stage1_audio": str(audio_path),
                "stage2_output_dir": str(output_dir),
                "stage2_output": str(output_dir / f"{audio_path.stem}.mp4"),
            }
        )

    two_stage_manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "stage1_manifest": str(manifest_path),
        "stage2_dir": str(stage2_dir),
        "latentsync_steps": args.latentsync_steps,
        "latentsync_guidance_scale": args.latentsync_guidance_scale,
        "items": stage2_items,
    }
    stage2_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = stage2_dir / "manifest.json"
    output_manifest.write_text(
        json.dumps(two_stage_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nTwo-stage manifest: {output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
