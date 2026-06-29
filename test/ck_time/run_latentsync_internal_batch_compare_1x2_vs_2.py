#!/usr/bin/env python3
import argparse
import gc
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LATENTSYNC_ROOT = PROJECT_ROOT / "third_party" / "LatentSync"
DEFAULT_AUDIO_DIR = PROJECT_ROOT / "test" / "ck_time" / "tts_internal_batch_compare_dongqing_20char" / "batch2"
DEFAULT_VIDEO = PROJECT_ROOT / "LatentSync_test" / "api_stage1.mp4"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "test" / "ck_time" / "latentsync_internal_batch_compare"


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


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
    return round(float(result.stdout.strip()), 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare LatentSync internal batch=2 vs batch=1 twice.")
    parser.add_argument("--video", default=str(DEFAULT_VIDEO))
    parser.add_argument("--audio-dir", default=str(DEFAULT_AUDIO_DIR))
    parser.add_argument("--only", default="dongqing_01,dongqing_02")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--mode", choices=["compare", "batch1_once", "batch1_twice", "batch2"], default="compare")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=2.2)
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--enable-deepcache", action="store_true", default=True)
    parser.add_argument("--disable-deepcache", action="store_false", dest="enable_deepcache")
    parser.add_argument("--skip-video-write", action="store_true")
    return parser.parse_args()


def pad_first_dim(tensor, target_len: int):
    import torch

    if tensor.shape[0] == target_len:
        return tensor
    pad_shape = (target_len - tensor.shape[0],) + tuple(tensor.shape[1:])
    pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad], dim=0)


def slice_or_pad_chunks(chunks: list, start: int, end: int):
    import torch

    selected = chunks[start : min(end, len(chunks))]
    if not selected:
        selected = [torch.zeros_like(chunks[-1])]
    if len(selected) < end - start:
        selected.extend(torch.zeros_like(selected[-1]) for _ in range(end - start - len(selected)))
    return torch.stack(selected)


def sync_if_cuda(torch_module) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def clear_cuda(torch_module) -> None:
    gc.collect()
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    if not LATENTSYNC_ROOT.exists():
        raise FileNotFoundError(f"LatentSync repo not found: {LATENTSYNC_ROOT}")

    video_path = project_path(args.video)
    audio_dir = project_path(args.audio_dir)
    stems = [item.strip() for item in args.only.split(",") if item.strip()]
    expected_stems = 1 if args.mode == "batch1_once" else 2
    if len(stems) != expected_stems:
        raise ValueError(f"--only must name exactly {expected_stems} wav stem(s) for --mode {args.mode}")
    audio_paths = [audio_dir / f"{stem}.wav" for stem in stems]
    missing = [str(path) for path in [video_path, *audio_paths] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else DEFAULT_OUT_ROOT / run_id
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(LATENTSYNC_ROOT)
    sys.path.insert(0, str(LATENTSYNC_ROOT))
    os.environ["PYTHONPATH"] = f"{LATENTSYNC_ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
    os.environ["LIVETALKING_ROOT"] = str(PROJECT_ROOT)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    import cv2
    import soundfile as sf
    import torch
    import torchvision
    from accelerate.utils import set_seed
    from DeepCache import DeepCacheSDHelper
    from diffusers import AutoencoderKL, DDIMScheduler
    from diffusers.utils import logging as diffusers_logging
    from einops import rearrange
    from latentsync.models.unet import UNet3DConditionModel
    from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
    from latentsync.utils.image_processor import ImageProcessor, load_fixed_mask
    from latentsync.utils.util import check_ffmpeg_installed, read_audio, read_video, write_video
    from latentsync.whisper.audio2feature import Audio2Feature
    from omegaconf import OmegaConf
    from torchvision import transforms

    _ = (cv2, torchvision, transforms)
    check_ffmpeg_installed()
    diffusers_logging.set_verbosity_error()
    config = OmegaConf.load(LATENTSYNC_ROOT / "configs" / "unet" / "stage2_512.yaml")
    if not torch.cuda.is_available():
        raise RuntimeError("LatentSync benchmark requires CUDA.")
    is_fp16_supported = torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16_supported else torch.float32
    device = torch.device("cuda")

    load_start = time.perf_counter()
    scheduler = DDIMScheduler.from_pretrained(str(LATENTSYNC_ROOT / "configs"))
    audio_encoder = Audio2Feature(
        model_path=str(LATENTSYNC_ROOT / "checkpoints" / "whisper" / "tiny.pt"),
        device="cuda",
        num_frames=config.data.num_frames,
        audio_feat_length=config.data.audio_feat_length,
    )
    local_vae = PROJECT_ROOT / "models" / "sd-vae"
    vae_source = str(local_vae) if local_vae.exists() else "stabilityai/sd-vae-ft-mse"
    vae = AutoencoderKL.from_pretrained(vae_source, torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0
    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        str(LATENTSYNC_ROOT / "checkpoints" / "latentsync_unet.pt"),
        device="cpu",
    )
    unet = unet.to(dtype=dtype)
    pipeline = LipsyncPipeline(vae=vae, audio_encoder=audio_encoder, unet=unet, scheduler=scheduler).to("cuda")
    if args.enable_deepcache:
        helper = DeepCacheSDHelper(pipe=pipeline)
        helper.set_params(cache_interval=3, cache_branch_id=0)
        helper.enable()
    if args.seed != -1:
        set_seed(args.seed)
    else:
        torch.seed()
    sync_if_cuda(torch)
    load_seconds = round(time.perf_counter() - load_start, 3)

    height = width = int(config.data.resolution)
    num_frames = int(config.data.num_frames)
    video_fps = 25
    audio_sample_rate = 16000
    do_cfg = args.guidance_scale > 1.0

    current_official_evidence = None
    original_unet_forward = pipeline.unet.forward

    def traced_unet_forward(sample, timestep, encoder_hidden_states=None, *forward_args, **forward_kwargs):
        if current_official_evidence is not None and not current_official_evidence["unet_forward_calls"]:
            current_official_evidence["unet_forward_calls"].append(
                {
                    "unet_input_shape": list(sample.shape),
                    "encoder_hidden_states_shape": list(encoder_hidden_states.shape)
                    if hasattr(encoder_hidden_states, "shape")
                    else None,
                    "observed_unet_batch": int(sample.shape[0]),
                    "cfg_enabled": do_cfg,
                }
            )
        return original_unet_forward(sample, timestep, encoder_hidden_states=encoder_hidden_states, *forward_args, **forward_kwargs)

    pipeline.unet.forward = traced_unet_forward

    def run_official_pipeline_mode(mode_name: str, selected_audio_paths: list[Path], record_evidence: bool):
        nonlocal current_official_evidence
        mode_dir = out_dir / mode_name
        mode_dir.mkdir(parents=True, exist_ok=True)
        runs = []
        started_at = time.perf_counter()
        for run_index, audio_path in enumerate(selected_audio_paths, start=1):
            evidence = {
                "uses_official_lipsync_pipeline_call": True,
                "unet_forward_calls": [],
                "expected_sample_batch": 1,
                "cfg_enabled": do_cfg,
            }
            current_official_evidence = evidence if record_evidence else None
            out_video = mode_dir / f"{audio_path.stem}.mp4"
            temp_dir = out_dir / "_work_official" / mode_name / audio_path.stem
            run_start = time.perf_counter()
            pipeline(
                video_path=str(video_path),
                audio_path=str(audio_path),
                video_out_path=str(out_video),
                num_frames=config.data.num_frames,
                num_inference_steps=args.inference_steps,
                guidance_scale=args.guidance_scale,
                weight_dtype=dtype,
                width=config.data.resolution,
                height=config.data.resolution,
                mask_image_path=config.data.mask_image_path,
                temp_dir=str(temp_dir),
            )
            sync_if_cuda(torch)
            current_official_evidence = None
            elapsed = round(time.perf_counter() - run_start, 3)
            item = {
                "run_index": run_index,
                "audio": str(audio_path),
                "output": str(out_video),
                "elapsed_seconds": elapsed,
                "audio_duration_seconds": ffprobe_duration(audio_path),
                "video_duration_seconds": ffprobe_duration(out_video),
            }
            if record_evidence:
                item["evidence"] = evidence
                observed = [call["observed_unet_batch"] for call in evidence["unet_forward_calls"]]
                item["is_true_batch1_pipeline_call"] = observed == ([2] if do_cfg else [1])
            runs.append(item)
            clear_cuda(torch)
        return {
            "mode": mode_name,
            "batch_size": 1,
            "num_runs": len(selected_audio_paths),
            "total_seconds": round(time.perf_counter() - started_at, 3),
            "run_seconds": [item["elapsed_seconds"] for item in runs],
            "runs": runs,
        }

    if args.mode in ("batch1_once", "batch1_twice"):
        selected_audio_paths = audio_paths[:1] if args.mode == "batch1_once" else audio_paths
        warmup = []
        for _ in range(max(0, args.warmup_runs)):
            warmup.append(run_official_pipeline_mode(f"warmup_{args.mode}", selected_audio_paths, record_evidence=False))

        official_result = run_official_pipeline_mode(args.mode, selected_audio_paths, record_evidence=True)
        manifest = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "stage": "talking_head",
            "engine": "LatentSync",
            "mode": args.mode,
            "algorithm": "official_LipsyncPipeline.__call__",
            "model_load_seconds": load_seconds,
            "warmup_runs": max(0, args.warmup_runs),
            "warmup": warmup,
            "video": str(video_path),
            "audio_paths": [str(path) for path in selected_audio_paths],
            "inference_steps": args.inference_steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "enable_deepcache": args.enable_deepcache,
            args.mode: official_result,
        }
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.mode == "batch1_once":
            print(f"talking_head_official_batch1_once_seconds: {official_result['total_seconds']}")
        else:
            print(f"talking_head_official_batch1_twice_seconds: {official_result['total_seconds']}")
        print(f"talking_head_official_run_seconds: {official_result['run_seconds']}")
        print(f"manifest: {manifest_path}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return 0

    mask_image = load_fixed_mask(height, config.data.mask_image_path)
    pipeline.image_processor = ImageProcessor(height, device="cuda", mask_image=mask_image)
    pipeline.scheduler.set_timesteps(args.inference_steps, device=device)
    timesteps = pipeline.scheduler.timesteps
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(None, 0.0)
    num_channels_latents = pipeline.vae.config.latent_channels
    latent_h = height // pipeline.vae_scale_factor
    latent_w = width // pipeline.vae_scale_factor

    preprocess_start = time.perf_counter()
    samples = []
    for index, audio_path in enumerate(audio_paths):
        whisper_feature = pipeline.audio_encoder.audio2feat(str(audio_path))
        whisper_chunks = pipeline.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)
        audio_samples = read_audio(str(audio_path))
        video_frames = read_video(str(video_path), use_decord=False)
        video_frames, faces, boxes, affine_matrices = pipeline.loop_video(whisper_chunks, video_frames)
        samples.append(
            {
                "id": stems[index],
                "audio_path": audio_path,
                "whisper_chunks": whisper_chunks,
                "audio_samples": audio_samples,
                "video_frames": video_frames,
                "faces": faces,
                "boxes": boxes,
                "affine_matrices": affine_matrices,
            }
        )
    sync_if_cuda(torch)
    preprocess_seconds = round(time.perf_counter() - preprocess_start, 3)

    def run_group(group_indices: list[int], record_evidence: bool, evidence: dict | None):
        group_samples = [samples[index] for index in group_indices]
        sample_batch = len(group_samples)
        max_chunks = max(len(sample["whisper_chunks"]) for sample in group_samples)
        num_inferences = math.ceil(max_chunks / num_frames)
        all_latents = torch.randn(
            (sample_batch, num_channels_latents, max_chunks, latent_h, latent_w),
            device=device,
            dtype=dtype,
        ) * pipeline.scheduler.init_noise_sigma
        synced_chunks = [[] for _ in group_samples]

        for chunk_index in range(num_inferences):
            start = chunk_index * num_frames
            end = min((chunk_index + 1) * num_frames, max_chunks)
            chunk_len = end - start
            target_len = num_frames

            batch_audio_embeds = []
            batch_ref_pixels = []
            batch_masked_pixels = []
            batch_masks = []
            for sample in group_samples:
                audio_embeds = slice_or_pad_chunks(sample["whisper_chunks"], start, start + target_len).to(device, dtype=dtype)
                inference_faces = pad_first_dim(sample["faces"][start : min(start + target_len, len(sample["faces"]))], target_len)
                ref_pixel_values, masked_pixel_values, masks = pipeline.image_processor.prepare_masks_and_masked_images(
                    inference_faces, affine_transform=False
                )
                batch_audio_embeds.append(audio_embeds)
                batch_ref_pixels.append(ref_pixel_values)
                batch_masked_pixels.append(masked_pixel_values)
                batch_masks.append(masks)

            audio_embeds = torch.stack(batch_audio_embeds, dim=0)
            if do_cfg:
                audio_embeds = torch.cat([torch.zeros_like(audio_embeds), audio_embeds], dim=0)

            ref_pixel_values = torch.stack(batch_ref_pixels, dim=0)
            masked_pixel_values = torch.stack(batch_masked_pixels, dim=0)
            masks = torch.stack(batch_masks, dim=0)

            ref_flat = rearrange(ref_pixel_values, "b f c h w -> (b f) c h w").to(device=device, dtype=dtype)
            masked_flat = rearrange(masked_pixel_values, "b f c h w -> (b f) c h w").to(device=device, dtype=dtype)
            masks_small = rearrange(masks, "b f c h w -> (b f) c h w")
            masks_small = torch.nn.functional.interpolate(masks_small, size=(latent_h, latent_w))
            masks_latents = rearrange(masks_small.to(device=device, dtype=dtype), "(b f) c h w -> b c f h w", b=sample_batch)

            masked_image_latents = pipeline.vae.encode(masked_flat).latent_dist.sample()
            masked_image_latents = (masked_image_latents - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor
            masked_image_latents = rearrange(masked_image_latents.to(device=device, dtype=dtype), "(b f) c h w -> b c f h w", b=sample_batch)

            ref_latents = pipeline.vae.encode(ref_flat).latent_dist.sample()
            ref_latents = (ref_latents - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor
            ref_latents = rearrange(ref_latents.to(device=device, dtype=dtype), "(b f) c h w -> b c f h w", b=sample_batch)

            if do_cfg:
                masks_latents = torch.cat([masks_latents, masks_latents], dim=0)
                masked_image_latents = torch.cat([masked_image_latents, masked_image_latents], dim=0)
                ref_latents = torch.cat([ref_latents, ref_latents], dim=0)

            latents = all_latents[:, :, start : start + target_len]
            if latents.shape[2] < target_len:
                pad = torch.zeros(
                    (sample_batch, num_channels_latents, target_len - latents.shape[2], latent_h, latent_w),
                    device=device,
                    dtype=dtype,
                )
                latents = torch.cat([latents, pad], dim=2)

            for step_index, t in enumerate(timesteps):
                unet_latents = torch.cat([latents, latents], dim=0) if do_cfg else latents
                unet_input = pipeline.scheduler.scale_model_input(unet_latents, t)
                unet_input = torch.cat([unet_input, masks_latents, masked_image_latents, ref_latents], dim=1)
                if record_evidence and chunk_index == 0 and step_index == 0:
                    evidence["instrumented_tensor_shapes"].append(
                        {
                            "group_indices": group_indices,
                            "sample_batch": sample_batch,
                            "effective_unet_batch": int(unet_input.shape[0]),
                            "latents": list(latents.shape),
                            "audio_embeds": list(audio_embeds.shape),
                            "masks_latents": list(masks_latents.shape),
                            "masked_image_latents": list(masked_image_latents.shape),
                            "ref_latents": list(ref_latents.shape),
                            "unet_input": list(unet_input.shape),
                        }
                    )
                noise_pred = pipeline.unet(unet_input, t, encoder_hidden_states=audio_embeds).sample
                if do_cfg:
                    noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_audio - noise_pred_uncond)
                latents = pipeline.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            all_latents[:, :, start : start + chunk_len] = latents[:, :, :chunk_len]
            decoded_latents = pipeline.decode_latents(latents[:, :, :chunk_len])
            decoded_latents = rearrange(decoded_latents, "(b f) c h w -> b f c h w", b=sample_batch)
            ref_original = ref_pixel_values[:, :chunk_len].to(device=device, dtype=dtype)
            masks_original = masks[:, :chunk_len].to(device=device, dtype=dtype)
            for local_index, sample in enumerate(group_samples):
                valid_len = max(0, min(chunk_len, len(sample["whisper_chunks"]) - start))
                if valid_len <= 0:
                    continue
                pasted = (
                    decoded_latents[local_index, :valid_len] * (1 - masks_original[local_index, :valid_len])
                    + ref_original[local_index, :valid_len] * masks_original[local_index, :valid_len]
                )
                synced_chunks[local_index].append(pasted.detach().cpu())

        return {sample_index: torch.cat(synced_chunks[local], dim=0) for local, sample_index in enumerate(group_indices)}

    def run_mode(mode_name: str, batch_size: int, record_evidence: bool):
        evidence = {
            "input_sample_count": len(samples),
            "requested_batch_size": batch_size,
            "cfg_enabled": do_cfg,
            "instrumented_tensor_shapes": [],
        }
        groups = [list(range(start, min(start + batch_size, len(samples)))) for start in range(0, len(samples), batch_size)]
        denoise_start = time.perf_counter()
        is_train = pipeline.unet.training
        pipeline.unet.eval()
        synced_by_index = {}
        group_seconds = []
        for group_indices in groups:
            group_start = time.perf_counter()
            synced_by_index.update(run_group(group_indices, record_evidence, evidence if record_evidence else None))
            sync_if_cuda(torch)
            group_seconds.append(round(time.perf_counter() - group_start, 3))
        if is_train:
            pipeline.unet.train()
        sync_if_cuda(torch)
        denoise_seconds = round(time.perf_counter() - denoise_start, 3)
        if record_evidence:
            observed_sample_batches = [item["sample_batch"] for item in evidence["instrumented_tensor_shapes"]]
            observed_unet_batches = [item["effective_unet_batch"] for item in evidence["instrumented_tensor_shapes"]]
            evidence["observed_sample_batches"] = observed_sample_batches
            evidence["observed_unet_batches"] = observed_unet_batches

        postprocess_seconds = 0.0
        items = []
        if not args.skip_video_write:
            post_start = time.perf_counter()
            mode_dir = out_dir / mode_name
            mode_dir.mkdir(parents=True, exist_ok=True)
            for sample_index, sample in enumerate(samples):
                synced_faces = synced_by_index[sample_index]
                synced_video_frames = pipeline.restore_video(
                    synced_faces,
                    sample["video_frames"],
                    sample["boxes"],
                    sample["affine_matrices"],
                )
                audio_samples_remain_length = int(synced_video_frames.shape[0] / video_fps * audio_sample_rate)
                audio_samples = sample["audio_samples"][:audio_samples_remain_length].cpu().numpy()
                temp_dir = mode_dir / "_work" / sample["id"]
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                temp_dir.mkdir(parents=True, exist_ok=True)
                raw_video = temp_dir / "video.mp4"
                raw_audio = temp_dir / "audio.wav"
                output = mode_dir / f"{sample['id']}.mp4"
                write_video(str(raw_video), synced_video_frames, fps=video_fps)
                sf.write(str(raw_audio), audio_samples, audio_sample_rate)
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-loglevel",
                        "error",
                        "-nostdin",
                        "-i",
                        str(raw_video),
                        "-i",
                        str(raw_audio),
                        "-c:v",
                        "libx264",
                        "-crf",
                        "18",
                        "-c:a",
                        "aac",
                        "-q:v",
                        "0",
                        "-q:a",
                        "0",
                        str(output),
                    ],
                    check=True,
                )
                items.append(
                    {
                        "id": sample["id"],
                        "audio": str(sample["audio_path"]),
                        "output": str(output),
                        "audio_duration_seconds": ffprobe_duration(sample["audio_path"]),
                        "video_duration_seconds": ffprobe_duration(output),
                    }
                )
            postprocess_seconds = round(time.perf_counter() - post_start, 3)

        total_seconds = round(denoise_seconds + postprocess_seconds, 3)
        result = {
            "batch_size": batch_size,
            "groups": groups,
            "group_seconds": group_seconds,
            "denoise_seconds": denoise_seconds,
            "postprocess_seconds": postprocess_seconds,
            "total_seconds": total_seconds,
            "evidence": evidence,
            "items": items,
        }
        del synced_by_index
        clear_cuda(torch)
        return result

    warmup = {"batch2_seconds": [], "batch1_twice_seconds": [], "batch1_once_seconds": []}
    for _ in range(max(0, args.warmup_runs)):
        if args.mode in ("compare", "batch2"):
            warmup["batch2_seconds"].append(run_mode("warmup_batch2", 2, record_evidence=False)["denoise_seconds"])
        if args.mode in ("compare", "batch1_twice"):
            warmup["batch1_twice_seconds"].append(run_mode("warmup_batch1_twice", 1, record_evidence=False)["denoise_seconds"])
        if args.mode == "batch1_once":
            warmup["batch1_once_seconds"].append(run_mode("warmup_batch1_once", 1, record_evidence=False)["denoise_seconds"])

    batch2 = None
    batch1_once = None
    batch1_twice = None
    if args.mode in ("compare", "batch2"):
        batch2 = run_mode("batch2", 2, record_evidence=True)
        batch2_sample_ok = batch2["evidence"].get("observed_sample_batches") == [2]
        batch2_unet_ok = all(value in (2, 4) for value in batch2["evidence"].get("observed_unet_batches", []))
        batch2["is_true_model_batch"] = batch2_sample_ok and batch2_unet_ok

    if args.mode == "batch1_once":
        batch1_once = run_mode("batch1_once", 1, record_evidence=True)
        batch1_once_sample_ok = batch1_once["evidence"].get("observed_sample_batches") == [1]
        batch1_once_unet_ok = all(value in (1, 2) for value in batch1_once["evidence"].get("observed_unet_batches", []))
        batch1_once["is_true_model_batch"] = batch1_once_sample_ok and batch1_once_unet_ok

    if args.mode in ("compare", "batch1_twice"):
        batch1_twice = run_mode("batch1_twice", 1, record_evidence=True)
        batch1_sample_ok = batch1_twice["evidence"].get("observed_sample_batches") == [1, 1]
        batch1_unet_ok = all(value in (1, 2) for value in batch1_twice["evidence"].get("observed_unet_batches", []))
        batch1_twice["is_true_model_batch"] = batch1_sample_ok and batch1_unet_ok

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "stage": "talking_head",
        "engine": "LatentSync",
        "comparison": "internal_batch2_vs_internal_batch1_twice",
        "mode": args.mode,
        "timing_mode": "warm",
        "model_load_seconds": load_seconds,
        "preprocess_seconds": preprocess_seconds,
        "warmup_runs": max(0, args.warmup_runs),
        "warmup": warmup,
        "video": str(video_path),
        "audio_paths": [str(path) for path in audio_paths],
        "inference_steps": args.inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "enable_deepcache": args.enable_deepcache,
        "skip_video_write": args.skip_video_write,
    }
    if batch2 is not None:
        manifest["batch2"] = batch2
    if batch1_once is not None:
        manifest["batch1_once"] = batch1_once
    if batch1_twice is not None:
        manifest["batch1_twice"] = batch1_twice
    if batch2 is not None and batch1_twice is not None:
        manifest["speedup"] = {
            "batch1_twice_over_batch2": round(batch1_twice["denoise_seconds"] / batch2["denoise_seconds"], 4)
            if batch2["denoise_seconds"] > 0
            else None,
            "denoise_seconds_saved_by_batch2": round(batch1_twice["denoise_seconds"] - batch2["denoise_seconds"], 3),
        }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if batch2 is not None:
        print(f"talking_head_internal_batch2_denoise_seconds: {batch2['denoise_seconds']}")
    if batch1_once is not None:
        print(f"talking_head_internal_batch1_once_denoise_seconds: {batch1_once['denoise_seconds']}")
    if batch1_twice is not None:
        print(f"talking_head_internal_batch1_twice_denoise_seconds: {batch1_twice['denoise_seconds']}")
    if "speedup" in manifest:
        print(f"talking_head_internal_batch1_over_batch2_speedup: {manifest['speedup']['batch1_twice_over_batch2']}")
    print(f"manifest: {manifest_path}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
