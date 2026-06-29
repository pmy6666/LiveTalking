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
SCRIPT_DIR = Path(__file__).resolve().parent
LATENTSYNC_ROOT = PROJECT_ROOT / "third_party" / "LatentSync"
DEFAULT_AUDIO_DIR = PROJECT_ROOT / "test" / "ck_time" / "tts_internal_batch_compare_dongqing_20char" / "batch2"
DEFAULT_VIDEO = PROJECT_ROOT / "LatentSync_test" / "api_stage1.mp4"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "test" / "ck_time" / "outputs_internal_batch_size_2"


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
    parser = argparse.ArgumentParser(description="Experimental true internal batch_size=2 LatentSync benchmark.")
    parser.add_argument("--video", default=str(DEFAULT_VIDEO))
    parser.add_argument("--audio-dir", default=str(DEFAULT_AUDIO_DIR))
    parser.add_argument("--only", default="dongqing_batch2_01,dongqing_batch2_02")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=2.2)
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--vae-batch-size", type=int, default=8)
    parser.add_argument("--enable-deepcache", action="store_true", default=False)
    parser.add_argument("--disable-deepcache", action="store_false", dest="enable_deepcache")
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

    selected = chunks[start:min(end, len(chunks))]
    if not selected:
        selected = [torch.zeros_like(chunks[-1])]
    if len(selected) < end - start:
        selected.extend(torch.zeros_like(selected[-1]) for _ in range(end - start - len(selected)))
    return torch.stack(selected)


def encode_vae_in_chunks(vae, pixels, chunk_size: int):
    import torch

    if chunk_size <= 0:
        raise ValueError("--vae-batch-size must be positive")
    latents = []
    for start in range(0, pixels.shape[0], chunk_size):
        encoded = vae.encode(pixels[start:start + chunk_size]).latent_dist.sample()
        latents.append(encoded)
    return torch.cat(latents, dim=0)


def decode_vae_in_chunks(pipeline, latents, chunk_size: int):
    import torch
    from einops import rearrange

    if chunk_size <= 0:
        raise ValueError("--vae-batch-size must be positive")
    latents = latents / pipeline.vae.config.scaling_factor + pipeline.vae.config.shift_factor
    latents = rearrange(latents, "b c f h w -> (b f) c h w")
    decoded = []
    for start in range(0, latents.shape[0], chunk_size):
        decoded.append(pipeline.vae.decode(latents[start:start + chunk_size]).sample)
    return torch.cat(decoded, dim=0)


def main() -> int:
    args = parse_args()
    if not LATENTSYNC_ROOT.exists():
        raise FileNotFoundError(f"LatentSync repo not found: {LATENTSYNC_ROOT}")

    video_path = project_path(args.video)
    audio_dir = project_path(args.audio_dir)
    stems = [item.strip() for item in args.only.split(",") if item.strip()]
    if len(stems) != 2:
        raise ValueError("--only must name exactly two wav stems for internal batch_size=2")
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

    torch.set_grad_enabled(False)
    check_ffmpeg_installed()
    diffusers_logging.set_verbosity_error()
    config = OmegaConf.load(LATENTSYNC_ROOT / "configs" / "unet" / "stage2_512.yaml")
    is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
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
    load_seconds = round(time.perf_counter() - load_start, 3)

    height = width = int(config.data.resolution)
    num_frames = int(config.data.num_frames)
    video_fps = 25
    audio_sample_rate = 16000
    do_cfg = args.guidance_scale > 1.0
    mask_image = load_fixed_mask(height, config.data.mask_image_path)
    pipeline.image_processor = ImageProcessor(height, device="cuda", mask_image=mask_image)
    pipeline.scheduler.set_timesteps(args.inference_steps, device=device)
    timesteps = pipeline.scheduler.timesteps
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(None, 0.0)

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
                "synced_chunks": [],
            }
        )
    preprocess_seconds = round(time.perf_counter() - preprocess_start, 3)

    max_chunks = max(len(sample["whisper_chunks"]) for sample in samples)
    num_inferences = math.ceil(max_chunks / num_frames)
    num_channels_latents = pipeline.vae.config.latent_channels
    latent_h = height // pipeline.vae_scale_factor
    latent_w = width // pipeline.vae_scale_factor
    all_latents = torch.randn(
        (2, num_channels_latents, max_chunks, latent_h, latent_w),
        device=device,
        dtype=dtype,
    ) * pipeline.scheduler.init_noise_sigma

    evidence = {
        "input_sample_count": 2,
        "observed_sample_batch": 2,
        "cfg_enabled": do_cfg,
        "instrumented_tensor_shapes": [],
    }

    denoise_start = time.perf_counter()
    is_train = pipeline.unet.training
    pipeline.unet.eval()
    for i in range(num_inferences):
        start = i * num_frames
        end = min((i + 1) * num_frames, max_chunks)
        chunk_len = end - start
        target_len = num_frames

        batch_audio_embeds = []
        batch_ref_pixels = []
        batch_masked_pixels = []
        batch_masks = []
        for sample in samples:
            audio_embeds = slice_or_pad_chunks(sample["whisper_chunks"], start, start + target_len).to(device, dtype=dtype)
            inference_faces = pad_first_dim(sample["faces"][start:min(start + target_len, len(sample["faces"]))], target_len)
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
        masks_latents = rearrange(masks_small.to(device=device, dtype=dtype), "(b f) c h w -> b c f h w", b=2)

        masked_image_latents = encode_vae_in_chunks(pipeline.vae, masked_flat, args.vae_batch_size)
        masked_image_latents = (masked_image_latents - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor
        masked_image_latents = rearrange(masked_image_latents.to(device=device, dtype=dtype), "(b f) c h w -> b c f h w", b=2)

        ref_latents = encode_vae_in_chunks(pipeline.vae, ref_flat, args.vae_batch_size)
        ref_latents = (ref_latents - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor
        ref_latents = rearrange(ref_latents.to(device=device, dtype=dtype), "(b f) c h w -> b c f h w", b=2)

        if do_cfg:
            masks_latents = torch.cat([masks_latents, masks_latents], dim=0)
            masked_image_latents = torch.cat([masked_image_latents, masked_image_latents], dim=0)
            ref_latents = torch.cat([ref_latents, ref_latents], dim=0)

        latents = all_latents[:, :, start:start + target_len]
        if latents.shape[2] < target_len:
            latents = torch.cat(
                [
                    latents,
                    torch.zeros((2, num_channels_latents, target_len - latents.shape[2], latent_h, latent_w), device=device, dtype=dtype),
                ],
                dim=2,
            )

        for step_index, t in enumerate(timesteps):
            unet_latents = torch.cat([latents, latents], dim=0) if do_cfg else latents
            unet_input = pipeline.scheduler.scale_model_input(unet_latents, t)
            unet_input = torch.cat([unet_input, masks_latents, masked_image_latents, ref_latents], dim=1)
            if i == 0 and step_index == 0:
                evidence["instrumented_tensor_shapes"].append(
                    {
                        "chunk_index": i,
                        "latents": list(latents.shape),
                        "audio_embeds": list(audio_embeds.shape),
                        "masks_latents": list(masks_latents.shape),
                        "masked_image_latents": list(masked_image_latents.shape),
                        "ref_latents": list(ref_latents.shape),
                        "unet_input": list(unet_input.shape),
                    }
                )
                evidence["observed_unet_batch"] = int(unet_input.shape[0])
            noise_pred = pipeline.unet(unet_input, t, encoder_hidden_states=audio_embeds).sample
            if do_cfg:
                noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_audio - noise_pred_uncond)
            latents = pipeline.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        all_latents[:, :, start:start + chunk_len] = latents[:, :, :chunk_len]
        del masked_image_latents, ref_latents, masks_latents
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        decoded_latents = decode_vae_in_chunks(pipeline, latents[:, :, :chunk_len], args.vae_batch_size)
        decoded_latents = rearrange(decoded_latents, "(b f) c h w -> b f c h w", b=2)
        ref_original = ref_pixel_values[:, :chunk_len].to(device=device, dtype=dtype)
        masks_original = masks[:, :chunk_len].to(device=device, dtype=dtype)
        for sample_index, sample in enumerate(samples):
            valid_len = max(0, min(chunk_len, len(sample["whisper_chunks"]) - start))
            if valid_len <= 0:
                continue
            generation_mask = 1 - masks_original[sample_index, :valid_len]
            pasted = decoded_latents[sample_index, :valid_len] * generation_mask + ref_original[sample_index, :valid_len] * (1 - generation_mask)
            sample["synced_chunks"].append(pasted.detach().cpu())

    denoise_seconds = round(time.perf_counter() - denoise_start, 3)
    if is_train:
        pipeline.unet.train()

    post_start = time.perf_counter()
    items = []
    for sample in samples:
        synced_faces = torch.cat(sample["synced_chunks"], dim=0).to(device=device, dtype=dtype)
        synced_video_frames = pipeline.restore_video(
            synced_faces,
            sample["video_frames"],
            sample["boxes"],
            sample["affine_matrices"],
        )
        audio_samples_remain_length = int(synced_video_frames.shape[0] / video_fps * audio_sample_rate)
        audio_samples = sample["audio_samples"][:audio_samples_remain_length].cpu().numpy()
        temp_dir = out_dir / "_work" / sample["id"]
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        raw_video = temp_dir / "video.mp4"
        raw_audio = temp_dir / "audio.wav"
        output = out_dir / f"{sample['id']}.mp4"
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
    total_seconds = round(preprocess_seconds + denoise_seconds + postprocess_seconds, 3)

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "stage": "talking_head",
        "engine": "LatentSync",
        "batch_mode": "model_internal_experimental",
        "batch_size": 2,
        "is_true_model_batch": evidence.get("observed_sample_batch") == 2 and evidence.get("observed_unet_batch") in (2, 4),
        "model_load_seconds": load_seconds,
        "preprocess_seconds": preprocess_seconds,
        "denoise_seconds": denoise_seconds,
        "postprocess_seconds": postprocess_seconds,
        "total_seconds": total_seconds,
        "video": str(video_path),
        "inference_steps": args.inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "vae_batch_size": args.vae_batch_size,
        "enable_deepcache": args.enable_deepcache,
        "evidence": evidence,
        "items": items,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"talking_head_internal_batch2_seconds: {total_seconds}")
    print(f"manifest: {manifest_path}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
