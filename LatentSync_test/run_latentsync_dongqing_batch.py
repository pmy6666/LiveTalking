#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = Path(__file__).resolve().parent
LATENTSYNC_ROOT = PROJECT_ROOT / "third_party" / "LatentSync"

DEFAULT_VIDEO = LATENTSYNC_ROOT / "assets" / "demo1_video.mp4"
DEFAULT_AUDIO_DIR = PROJECT_ROOT / "gpt_sovits_official_materials" / "generated_bilibili_refs_tts" / "DongQing_6s"
DEFAULT_OUTPUT_DIR = TEST_ROOT / "generated_videos_dongqing_sync"
DEFAULT_INFERENCE_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 2.2


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def check_required_files() -> None:
    required = [
        LATENTSYNC_ROOT / "scripts" / "inference.py",
        LATENTSYNC_ROOT / "configs" / "unet" / "stage2_512.yaml",
        LATENTSYNC_ROOT / "checkpoints" / "latentsync_unet.pt",
        LATENTSYNC_ROOT / "checkpoints" / "whisper" / "tiny.pt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("LatentSync files are incomplete:\n" + "\n".join(f"  - {item}" for item in missing))


def check_python_deps() -> None:
    modules = [
        "diffusers",
        "transformers",
        "decord",
        "accelerate",
        "einops",
        "omegaconf",
        "cv2",
        "mediapipe",
        "python_speech_features",
        "scenedetect",
        "ffmpeg",
        "imageio",
        "lpips",
        "face_alignment",
        "kornia",
        "insightface",
        "onnxruntime",
        "DeepCache",
    ]
    missing = []
    for name in modules:
        try:
            __import__(name)
        except Exception as exc:
            missing.append(f"{name}: {exc}")
    if missing:
        raise RuntimeError(
            "LatentSync Python dependencies are missing or broken:\n"
            + "\n".join(f"  - {item}" for item in missing)
            + "\n\nInstall them with:\n"
            + "  ../envs/livetalking/bin/python -m pip install -r "
            + str(LATENTSYNC_ROOT / "requirements.txt")
        )


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


def ensure_local_vae_cache() -> None:
    local_vae = PROJECT_ROOT / "models" / "sd-vae"
    if not local_vae.exists():
        return

    hf_model_dir = Path.home() / ".cache" / "huggingface" / "hub" / "models--stabilityai--sd-vae-ft-mse"
    snapshot_dir = hf_model_dir / "snapshots" / "local"
    refs_dir = hf_model_dir / "refs"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)
    (refs_dir / "main").write_text("local\n", encoding="utf-8")

    for filename in ["config.json", "diffusion_pytorch_model.bin"]:
        source = local_vae / filename
        target = snapshot_dir / filename
        if source.exists() and not target.exists():
            try:
                target.symlink_to(source)
            except FileExistsError:
                pass
            except OSError:
                shutil.copy2(source, target)


def list_audio_files(audio_dir: Path, only: str, all_files: bool) -> list[Path]:
    files = sorted(path for path in audio_dir.glob("*.wav") if path.is_file())
    if not all_files:
        wanted = {item.strip() for item in only.split(",") if item.strip()}
        files = [path for path in files if path.stem in wanted or path.name in wanted]
    if not files:
        raise FileNotFoundError(f"No matching wav files found in {audio_dir}")
    return files


def run_one(args, audio_path: Path, output_path: Path, temp_dir: Path) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "scripts.inference",
        "--unet_config_path",
        "configs/unet/stage2_512.yaml",
        "--inference_ckpt_path",
        "checkpoints/latentsync_unet.pt",
        "--inference_steps",
        str(args.inference_steps),
        "--guidance_scale",
        str(args.guidance_scale),
        "--video_path",
        str(args.video.resolve()),
        "--audio_path",
        str(audio_path.resolve()),
        "--video_out_path",
        str(output_path.resolve()),
        "--temp_dir",
        str(temp_dir.resolve()),
        "--seed",
        str(args.seed),
    ]
    if args.enable_deepcache:
        cmd.append("--enable_deepcache")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{LATENTSYNC_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    subprocess.run(cmd, cwd=LATENTSYNC_ROOT, env=env, check=True)
    elapsed = time.time() - start

    return {
        "video": str(args.video.resolve()),
        "audio": str(audio_path.resolve()),
        "output": str(output_path.resolve()),
        "audio_duration_seconds": ffprobe_duration(audio_path),
        "output_duration_seconds": ffprobe_duration(output_path) if output_path.exists() else None,
        "elapsed_seconds": round(elapsed, 3),
        "inference_steps": args.inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "enable_deepcache": args.enable_deepcache,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch LatentSync 1.6 test with DongQing GPT-SoVITS audios.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--only", default="01_morning_breakfast", help="Comma-separated stems or filenames.")
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--inference-steps",
        type=int,
        default=DEFAULT_INFERENCE_STEPS,
        help="Higher values improve visual quality but slow inference. Sync-focused default: 30.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=DEFAULT_GUIDANCE_SCALE,
        help="Higher values strengthen audio-lip alignment but may add jitter. Sync-focused default: 2.2.",
    )
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--enable-deepcache", action="store_true", default=True)
    parser.add_argument("--disable-deepcache", action="store_false", dest="enable_deepcache")
    parser.add_argument("--skip-dep-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_path(LATENTSYNC_ROOT, "LatentSync repository")
    require_path(args.video, "Input video")
    require_path(args.audio_dir, "Audio directory")
    check_required_files()
    if not args.skip_dep_check:
        check_python_deps()
    ensure_local_vae_cache()

    audio_files = list_audio_files(args.audio_dir, args.only, args.all)
    output_dir = args.output_dir.resolve()
    temp_root = TEST_ROOT / "_work"
    temp_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "latentsync_root": str(LATENTSYNC_ROOT),
        "video": str(args.video.resolve()),
        "audio_dir": str(args.audio_dir.resolve()),
        "output_dir": str(output_dir),
        "inference_steps": args.inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "enable_deepcache": args.enable_deepcache,
        "items": [],
    }

    for audio_path in audio_files:
        output_path = output_dir / f"{audio_path.stem}.mp4"
        temp_dir = temp_root / audio_path.stem
        item = run_one(args, audio_path, output_path, temp_dir)
        manifest["items"].append(item)

    manifest_path = output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
