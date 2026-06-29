# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import time
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from accelerate.utils import set_seed
from latentsync.whisper.audio2feature import Audio2Feature
from DeepCache import DeepCacheSDHelper


def split_csv(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def print_profile_summary(profile, summary_path):
    phases = profile.get("phases", {})
    total = profile.get("total", 0)
    success = profile.get("success", 0)
    failure = profile.get("failure", max(0, total - success))
    print("=" * 60)
    print(f"Batch inference completed in {profile.get('total_seconds', 0.0):.1f}s")
    print(f"  Mode: {profile.get('mode', 'unknown')}")
    print(f"  Batch size: {profile.get('batch_size', 1)}")
    if "chunk_batch_size" in profile:
        print(f"  Chunk batch size: {profile.get('chunk_batch_size', 1)}")
    print(f"  Success: {success}/{total}")
    print(f"  Failure: {failure}/{total}")
    print("  Pipeline profiling:")
    print(f"    Phase 1 (preprocess): {phases.get('preprocess', 0.0):.3f}s")
    print(f"    Phase 2 (denoise):    {phases.get('denoise', 0.0):.3f}s")
    print(f"    Phase 3 (postprocess): {phases.get('postprocess', 0.0):.3f}s")
    print(f"    GPU peak memory: {profile.get('gpu_peak_memory_mb', 0.0):.1f} MB")
    print(f"    UNet forward calls: {profile.get('num_unet_forward_calls', 0)}")
    print(f"    Avg UNet forward: {profile.get('avg_unet_forward_ms', 0.0):.3f} ms")
    print(f"  Summary saved to: {summary_path}")
    print("=" * 60)


def main(config, args):
    video_paths = split_csv(args.video_paths) if args.video_paths else []
    audio_paths = split_csv(args.audio_paths) if args.audio_paths else []
    video_out_paths = split_csv(args.video_out_paths) if args.video_out_paths else []

    if video_paths or audio_paths or video_out_paths:
        if not (video_paths and audio_paths and video_out_paths):
            raise RuntimeError("--video_paths, --audio_paths, and --video_out_paths must be provided together")
        if len(video_paths) == 1 and len(audio_paths) > 1:
            video_paths = video_paths * len(audio_paths)
        if len(video_out_paths) != len(audio_paths):
            raise RuntimeError("--video_out_paths must have the same item count as --audio_paths")
        if len(video_paths) != len(audio_paths):
            raise RuntimeError("--video_paths must have one item or the same item count as --audio_paths")
    else:
        if not args.video_path or not args.audio_path or not args.video_out_path:
            raise RuntimeError("--video_path, --audio_path, and --video_out_path are required for single inference")
        video_paths = [args.video_path]
        audio_paths = [args.audio_path]
        video_out_paths = [args.video_out_path]

    for video_path in video_paths:
        if not os.path.exists(video_path):
            raise RuntimeError(f"Video path '{video_path}' not found")
    for audio_path in audio_paths:
        if not os.path.exists(audio_path):
            raise RuntimeError(f"Audio path '{audio_path}' not found")

    # Check if the GPU supports float16
    is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16_supported else torch.float32

    print(f"Input video paths: {video_paths}")
    print(f"Input audio paths: {audio_paths}")
    print(f"Loaded checkpoint path: {args.inference_ckpt_path}")

    scheduler = DDIMScheduler.from_pretrained("configs")

    if config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")

    audio_encoder = Audio2Feature(
        model_path=whisper_model_path,
        device="cuda",
        num_frames=config.data.num_frames,
        audio_feat_length=config.data.audio_feat_length,
    )

    project_root = os.environ.get(
        "LIVETALKING_ROOT",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")),
    )
    local_vae_path = os.environ.get(
        "LATENTSYNC_VAE_PATH",
        os.path.join(project_root, "models", "sd-vae"),
    )
    vae_source = local_vae_path if os.path.exists(local_vae_path) else "stabilityai/sd-vae-ft-mse"
    vae = AutoencoderKL.from_pretrained(vae_source, torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        args.inference_ckpt_path,
        device="cpu",
    )

    unet = unet.to(dtype=dtype)

    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
    ).to("cuda")

    # use DeepCache
    if args.enable_deepcache:
        helper = DeepCacheSDHelper(pipe=pipeline)
        helper.set_params(cache_interval=3, cache_branch_id=0)
        helper.enable()

    if args.seed != -1:
        set_seed(args.seed)
    else:
        torch.seed()

    print(f"Initial seed: {torch.initial_seed()}")
    trace_unet_shapes_enabled = args.trace_unet_shapes or bool(args.trace_unet_shapes_path)

    if len(audio_paths) == 1 and args.batch_size == 1:
        profile = pipeline(
            video_path=video_paths[0],
            audio_path=audio_paths[0],
            video_out_path=video_out_paths[0],
            num_frames=config.data.num_frames,
            num_inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            weight_dtype=dtype,
            width=config.data.resolution,
            height=config.data.resolution,
            mask_image_path=config.data.mask_image_path,
            temp_dir=args.temp_dir,
            chunk_batch_size=args.chunk_batch_size,
            trace_unet_shapes=trace_unet_shapes_enabled,
            trace_unet_shapes_path=args.trace_unet_shapes_path,
        )
    else:
        profile = pipeline.batch_inference(
            video_paths=video_paths,
            audio_paths=audio_paths,
            video_out_paths=video_out_paths,
            batch_size=args.batch_size,
            num_frames=config.data.num_frames,
            num_inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            weight_dtype=dtype,
            width=config.data.resolution,
            height=config.data.resolution,
            mask_image_path=config.data.mask_image_path,
            temp_dir=args.temp_dir,
            trace_unet_shapes=trace_unet_shapes_enabled,
            trace_unet_shapes_path=args.trace_unet_shapes_path,
        )

    profile.update(
        {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "video_paths": video_paths,
            "audio_paths": audio_paths,
            "video_out_paths": video_out_paths,
            "unet_config_path": args.unet_config_path,
            "inference_ckpt_path": args.inference_ckpt_path,
            "inference_steps": args.inference_steps,
            "guidance_scale": args.guidance_scale,
            "chunk_batch_size": args.chunk_batch_size,
            "seed": args.seed,
            "enable_deepcache": args.enable_deepcache,
            "trace_unet_shapes": trace_unet_shapes_enabled,
            "trace_unet_shapes_path": args.trace_unet_shapes_path,
        }
    )
    summary_path = args.profile_summary_path or os.path.join(args.temp_dir, "inference_profile_summary.json")
    summary_dir = os.path.dirname(summary_path)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    print_profile_summary(profile, summary_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_config_path", type=str, default="configs/unet.yaml")
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--audio_path", type=str)
    parser.add_argument("--video_out_path", type=str)
    parser.add_argument("--video_paths", type=str, help="Comma-separated video paths for internal batch inference")
    parser.add_argument("--audio_paths", type=str, help="Comma-separated audio paths for internal batch inference")
    parser.add_argument("--video_out_paths", type=str, help="Comma-separated output paths for internal batch inference")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--chunk_batch_size", type=int, default=1, help="Micro-batch chunks within a single video")
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--temp_dir", type=str, default="temp")
    parser.add_argument("--profile_summary_path", type=str)
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--enable_deepcache", action="store_true")
    parser.add_argument("--trace_unet_shapes", action="store_true", help="Print denoise and UNet stage tensor shapes")
    parser.add_argument(
        "--trace_unet_shapes_path",
        type=str,
        help="Write denoise and UNet stage tensor shapes to this file; also enables shape tracing",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.unet_config_path)

    main(config, args)
