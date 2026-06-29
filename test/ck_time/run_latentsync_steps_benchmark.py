#!/usr/bin/env python3
import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from run_llm_tts_latentsync import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    build_runtime_args,
    generate_one_sentence,
    get_nested,
    load_json,
    load_prompt_text,
    load_yaml,
    project_path,
    request_tts,
    require_path,
    set_nested,
)


SCRIPT_DIR = Path(__file__).resolve().parent


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


def ensure_local_vae_cache(config: dict) -> None:
    local_vae = project_path(get_nested(config, "latentsync.vae_path", "models/sd-vae"))
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


def check_latentsync_files(config: dict) -> None:
    root = project_path(get_nested(config, "latentsync.root", "third_party/LatentSync"))
    required = [
        root / "scripts" / "inference.py",
        root / get_nested(config, "latentsync.unet_config", "configs/unet/stage2_512.yaml"),
        root / get_nested(config, "latentsync.checkpoint", "checkpoints/latentsync_unet.pt"),
        root / get_nested(config, "latentsync.whisper_tiny", "checkpoints/whisper/tiny.pt"),
        project_path(get_nested(config, "latentsync.stage1_video")),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("LatentSync benchmark files are incomplete:\n" + "\n".join(f"  - {item}" for item in missing))


def run_worker(config_path: Path, audio_path: Path, run_dir: Path) -> dict:
    config = load_yaml(config_path)
    check_latentsync_files(config)
    ensure_local_vae_cache(config)

    latentsync_root = project_path(get_nested(config, "latentsync.root", "third_party/LatentSync"))
    worker_manifest = run_dir / "worker_manifest.json"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker",
        "--config",
        str(config_path.resolve()),
        "--audio-path",
        str(audio_path.resolve()),
        "--run-dir",
        str(run_dir.resolve()),
        "--worker-manifest",
        str(worker_manifest.resolve()),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{latentsync_root}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["LIVETALKING_ROOT"] = str(PROJECT_ROOT)
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    start = time.perf_counter()
    subprocess.run(cmd, cwd=latentsync_root, env=env, check=True)
    total_worker_seconds = round(time.perf_counter() - start, 3)

    result = load_json(worker_manifest)
    result["worker_process_seconds"] = total_worker_seconds
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark LatentSync 20-step vs 30-step without counting first model load.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--deepseek-api-key", default="")
    parser.add_argument("--fixed-sentence", default="", help="Use this sentence instead of calling DeepSeek.")
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    if args.out_dir is not None:
        set_nested(config, "output.benchmark_out_dir", str(args.out_dir))

    runtime_args = build_runtime_args(
        argparse.Namespace(
            config=args.config,
            prompt=None,
            deepseek_api_key=args.deepseek_api_key,
            tts_speed_factor=None,
            inference_steps=None,
            guidance_scale=None,
            out_dir=None,
        )
    )

    require_path(runtime_args.tts_params, "GPT-SoVITs params file")
    require_path(runtime_args.content, "reference transcript file")
    require_path(runtime_args.ref_audio, "TTS reference audio")
    require_path(project_path(get_nested(config, "latentsync.stage1_video")), "LatentSync stage1 video")

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_root = project_path(get_nested(config, "output.benchmark_out_dir", "test/ck_time/outputs_steps_benchmark"))
    run_dir = out_root / run_id
    audio_dir = run_dir / "audio"
    audio_path = audio_dir / "fixed_tts.wav"
    run_dir.mkdir(parents=True, exist_ok=True)

    timing = {}
    total_start = time.perf_counter()
    if args.fixed_sentence:
        sentence = args.fixed_sentence.strip()
        timing["llm_seconds"] = 0.0
    else:
        llm_start = time.perf_counter()
        sentence = generate_one_sentence(runtime_args)
        timing["llm_seconds"] = round(time.perf_counter() - llm_start, 3)
    (run_dir / "fixed_sentence.txt").write_text(sentence + "\n", encoding="utf-8")
    print(f"fixed_sentence: {sentence}", flush=True)

    tts_start = time.perf_counter()
    prompt_text = load_prompt_text(runtime_args.content, runtime_args.ref_id)
    tts_result = request_tts(runtime_args, sentence, runtime_args.ref_audio, prompt_text, audio_path)
    timing["tts_seconds"] = round(time.perf_counter() - tts_start, 3)
    print(f"audio: {audio_path}", flush=True)

    worker_start = time.perf_counter()
    worker_result = run_worker(args.config, audio_path, run_dir)
    timing["latentsync_worker_seconds"] = round(time.perf_counter() - worker_start, 3)
    timing["total_seconds"] = round(time.perf_counter() - total_start, 3)

    measured = [item for item in worker_result["runs"] if not item.get("warmup")]
    summary = {
        "measure_20_seconds": next((item["elapsed_seconds"] for item in measured if item["inference_steps"] == 20), None),
        "measure_30_seconds": next((item["elapsed_seconds"] for item in measured if item["inference_steps"] == 30), None),
    }
    if summary["measure_20_seconds"] and summary["measure_30_seconds"]:
        summary["step_30_minus_20_seconds"] = round(summary["measure_30_seconds"] - summary["measure_20_seconds"], 3)
        summary["step_30_over_20_ratio"] = round(summary["measure_30_seconds"] / summary["measure_20_seconds"], 4)

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": str(args.config.resolve()),
        "run_dir": str(run_dir.resolve()),
        "fixed_sentence": sentence,
        "timing": timing,
        "summary": summary,
        "tts": tts_result,
        "latentsync": worker_result,
        "effective_config": config,
    }
    manifest_path = run_dir / "steps_benchmark_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("done")
    print(f"measure_20_seconds: {summary['measure_20_seconds']}")
    print(f"measure_30_seconds: {summary['measure_30_seconds']}")
    print(f"manifest: {manifest_path}")
    return 0


def worker_main() -> int:
    parser = argparse.ArgumentParser(description="Internal LatentSync steps benchmark worker.")
    parser.add_argument("--_worker", action="store_true")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--worker-manifest", required=True)
    args = parser.parse_args()

    import torch
    from accelerate.utils import set_seed
    from DeepCache import DeepCacheSDHelper
    from diffusers import AutoencoderKL, DDIMScheduler
    from diffusers.utils import logging as diffusers_logging
    from latentsync.models.unet import UNet3DConditionModel
    from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
    from latentsync.whisper.audio2feature import Audio2Feature
    from omegaconf import OmegaConf

    config = load_yaml(args.config)
    latentsync_root = project_path(get_nested(config, "latentsync.root", "third_party/LatentSync"))
    unet_config_path = latentsync_root / get_nested(config, "latentsync.unet_config", "configs/unet/stage2_512.yaml")
    ckpt_path = latentsync_root / get_nested(config, "latentsync.checkpoint", "checkpoints/latentsync_unet.pt")
    whisper_path = latentsync_root / get_nested(config, "latentsync.whisper_tiny", "checkpoints/whisper/tiny.pt")
    stage1_video = project_path(get_nested(config, "latentsync.stage1_video"))

    diffusers_logging.set_verbosity_error()
    model_config = OmegaConf.load(unet_config_path)
    is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16_supported else torch.float32

    load_start = time.perf_counter()
    scheduler = DDIMScheduler.from_pretrained(str(latentsync_root / "configs"))
    audio_encoder = Audio2Feature(
        model_path=str(whisper_path),
        device="cuda",
        num_frames=model_config.data.num_frames,
        audio_feat_length=model_config.data.audio_feat_length,
    )

    local_vae_path = project_path(get_nested(config, "latentsync.vae_path", "models/sd-vae"))
    vae_source = str(local_vae_path) if local_vae_path.exists() else "stabilityai/sd-vae-ft-mse"
    vae = AutoencoderKL.from_pretrained(vae_source, torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(model_config.model),
        str(ckpt_path),
        device="cpu",
    )
    unet = unet.to(dtype=dtype)

    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
    ).to("cuda")

    benchmark_deepcache = bool(
        get_nested(
            config,
            "latentsync.benchmark_enable_deepcache",
            get_nested(config, "latentsync.enable_deepcache", True),
        )
    )
    if benchmark_deepcache:
        helper = DeepCacheSDHelper(pipe=pipeline)
        helper.set_params(cache_interval=3, cache_branch_id=0)
        helper.enable()

    seed = int(get_nested(config, "latentsync.seed", 1247))
    if seed != -1:
        set_seed(seed)
    else:
        torch.seed()
    load_seconds = round(time.perf_counter() - load_start, 3)

    benchmark_steps = get_nested(config, "latentsync.benchmark_steps", [])
    if not benchmark_steps:
        benchmark_steps = [
            {"name": "warmup_20", "inference_steps": 20, "warmup": True},
            {"name": "measure_20", "inference_steps": 20, "warmup": False},
            {"name": "measure_30", "inference_steps": 30, "warmup": False},
        ]

    runs = []
    for index, item in enumerate(benchmark_steps, start=1):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        name = str(item.get("name", f"run_{index:02d}"))
        steps = int(item.get("inference_steps", 20))
        is_warmup = bool(item.get("warmup", False))
        out_dir = Path(args.run_dir) / "videos" / f"{index:02d}_{name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_video = out_dir / f"{Path(args.audio_path).stem}_{steps}step.mp4"
        temp_dir = project_path(get_nested(config, "output.benchmark_work_dir", "test/ck_time/_work_steps_benchmark")) / Path(args.run_dir).name / name

        print(f"latentsync_benchmark_run {index}/{len(benchmark_steps)}: {name}, steps={steps}, warmup={is_warmup}", flush=True)
        start = time.perf_counter()
        pipeline(
            video_path=str(stage1_video),
            audio_path=args.audio_path,
            video_out_path=str(out_video),
            num_frames=model_config.data.num_frames,
            num_inference_steps=steps,
            guidance_scale=float(get_nested(config, "latentsync.guidance_scale", 2.2)),
            weight_dtype=dtype,
            width=model_config.data.resolution,
            height=model_config.data.resolution,
            mask_image_path=model_config.data.mask_image_path,
            temp_dir=str(temp_dir),
        )
        elapsed = round(time.perf_counter() - start, 3)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        runs.append(
            {
                "index": index,
                "name": name,
                "warmup": is_warmup,
                "inference_steps": steps,
                "elapsed_seconds": elapsed,
                "output_video": str(out_video.resolve()),
                "output_duration_seconds": ffprobe_duration(out_video),
            }
        )
        print(f"{name}_seconds: {elapsed}", flush=True)

    worker_manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "load_model_seconds": load_seconds,
        "video": str(stage1_video.resolve()),
        "audio": str(Path(args.audio_path).resolve()),
        "audio_duration_seconds": ffprobe_duration(Path(args.audio_path)),
        "benchmark_enable_deepcache": benchmark_deepcache,
        "runs": runs,
    }
    Path(args.worker_manifest).write_text(json.dumps(worker_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    if "--_worker" in sys.argv:
        raise SystemExit(worker_main())
    raise SystemExit(main())
