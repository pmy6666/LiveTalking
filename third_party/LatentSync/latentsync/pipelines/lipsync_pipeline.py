# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py

import inspect
import math
import os
import shutil
import time
from typing import Callable, List, Optional, Union
import subprocess

import numpy as np
import torch
import torchvision
from torchvision import transforms

from packaging import version

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipelines import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging

from einops import rearrange
import cv2

from ..models.unet import UNet3DConditionModel
from ..utils.util import read_video, read_audio, write_video, check_ffmpeg_installed
from ..utils.image_processor import ImageProcessor, load_fixed_mask
from ..whisper.audio2feature import Audio2Feature
import tqdm
import soundfile as sf

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class LipsyncPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        audio_encoder: Audio2Feature,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.set_progress_bar_config(desc="Steps")

    @staticmethod
    def _sync_device(device):
        if torch.cuda.is_available() and torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)

    @classmethod
    def _profile_start(cls, device):
        cls._sync_device(device)
        return time.perf_counter()

    @classmethod
    def _profile_elapsed(cls, started_at, device):
        cls._sync_device(device)
        return time.perf_counter() - started_at

    @staticmethod
    def _new_profile(mode: str, batch_size: int, total: int, device):
        if torch.cuda.is_available() and torch.device(device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        return {
            "mode": mode,
            "batch_size": batch_size,
            "total": total,
            "success": 0,
            "failure": 0,
            "phases": {
                "preprocess": 0.0,
                "denoise": 0.0,
                "postprocess": 0.0,
            },
            "cache": {
                "video_hits": 0,
                "video_misses": 0,
                "vae_hits": 0,
                "vae_misses": 0,
            },
            "gpu_peak_memory_mb": 0.0,
            "num_unet_forward_calls": 0,
            "avg_unet_forward_ms": 0.0,
            "outputs": [],
        }

    @staticmethod
    def _finish_profile(profile, device, total_started_at):
        if torch.cuda.is_available() and torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
            profile["gpu_peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        profile["total_seconds"] = time.perf_counter() - total_started_at
        profile["phases"]["postprocess"] = max(
            0.0,
            profile["total_seconds"] - profile["phases"]["preprocess"] - profile["phases"]["denoise"],
        )
        if profile.get("num_unet_forward_calls", 0) > 0:
            profile["avg_unet_forward_ms"] = (
                profile["phases"]["denoise"] * 1000.0 / profile["num_unet_forward_calls"]
            )
        profile["failure"] = profile["total"] - profile["success"]
        return profile

    @staticmethod
    def _trace_pipeline_shape(enabled, name, tensor, note=None, chunk=None, step=None, timestep=None, writer=None):
        if not enabled:
            return
        lines = [
            "",
            "[DenoiseShape]",
            f"context: chunk={chunk} step={step} timestep={timestep}",
            f"event: {name}",
        ]
        if torch.is_tensor(tensor):
            if tensor.dim() == 5:
                meaning = "5D latent/video tensor: (B, C, F, H, W) = batch/CFG batch, channels, frames, latent height, latent width"
            elif tensor.dim() == 4:
                meaning = "4D tensor; audio uses (B, F, S, D), images use (B, C, H, W)"
            elif tensor.dim() == 3:
                meaning = "3D token tensor: (B, S, D) = batch, sequence length, feature dim"
            elif tensor.dim() == 2:
                meaning = "2D tensor: (B, D) = batch, feature dim"
            else:
                meaning = f"{tensor.dim()}D tensor"
            lines.extend(
                [
                    f"shape: {tuple(tensor.shape)}",
                    f"dtype/device: {tensor.dtype} / {tensor.device}",
                    "meaning:",
                    f"  {meaning}",
                ]
            )
        else:
            lines.append(f"value: {tensor}")
        if note:
            lines.extend(["note:", f"  {note}"])
        message = "\n".join(lines)
        if writer is None:
            print(message)
        else:
            writer.write(message + "\n")
            writer.flush()

    @staticmethod
    def _should_trace_unet_step(enabled, chunk, step):
        return enabled and chunk == 0 and step == 0

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        decoded_latents = self.vae.decode(latents).sample
        return decoded_latents

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, height, width, callback_steps):
        assert height == width, "Height and width must be equal"

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, num_frames, num_channels_latents, height, width, dtype, device, generator):
        shape = (
            1,
            num_channels_latents,
            1,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )  # (b, c, f, h, w)
        rand_device = "cpu" if device.type == "mps" else device
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        latents = latents.repeat(1, 1, num_frames, 1, 1)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_batch_latents(self, batch_size, num_frames, num_channels_latents, height, width, dtype, device, generator):
        shape = (
            batch_size,
            num_channels_latents,
            1,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        ) # (b, c, f, h, w)
        rand_device = "cpu" if device.type == "mps" else device
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        latents = latents.repeat(1, 1, num_frames, 1, 1)
        return latents * self.scheduler.init_noise_sigma

    def prepare_chunk_latents(self, num_channels_latents, num_frames, height, width, dtype, device, generator):
        shape = (
            1,
            num_channels_latents,
            num_frames,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        rand_device = torch.device("cpu") if torch.device(device).type != "mps" else device
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype)
        latents = latents * self.scheduler.init_noise_sigma
        return latents.to(device)

    @staticmethod
    def pad_first_dim(tensor, target_len: int):
        if tensor.shape[0] == target_len:
            return tensor
        pad_shape = (target_len - tensor.shape[0],) + tuple(tensor.shape[1:])
        pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, pad], dim=0)

    @staticmethod
    def slice_or_pad_chunks(chunks: list, start: int, target_len: int):
        selected = chunks[start : min(start + target_len, len(chunks))]
        if not selected:
            selected = [torch.zeros_like(chunks[-1])]
        if len(selected) < target_len:
            selected.extend(torch.zeros_like(selected[-1]) for _ in range(target_len - len(selected)))
        return torch.stack(selected)

    def prepare_mask_latents(
        self, mask, masked_image, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        masked_image = masked_image.to(device=device, dtype=dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
        masked_image_latents = (masked_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        mask = mask.to(device=device, dtype=dtype)

        # assume batch size = 1
        mask = rearrange(mask, "f c h w -> 1 c f h w")
        masked_image_latents = rearrange(masked_image_latents, "f c h w -> 1 c f h w")

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )
        return mask, masked_image_latents

    def prepare_image_latents(self, images, device, dtype, generator, do_classifier_free_guidance):
        images = images.to(device=device, dtype=dtype)
        image_latents = self.vae.encode(images).latent_dist.sample(generator=generator)
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        image_latents = rearrange(image_latents, "f c h w -> 1 c f h w")
        image_latents = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents

        return image_latents

    def set_progress_bar_config(self, **kwargs):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(kwargs)

    @staticmethod
    def paste_surrounding_pixels_back(decoded_latents, pixel_values, masks, device, weight_dtype):
        # Paste the surrounding pixels back, because we only want to change the mouth region
        pixel_values = pixel_values.to(device=device, dtype=weight_dtype)
        masks = masks.to(device=device, dtype=weight_dtype)
        combined_pixel_values = decoded_latents * masks + pixel_values * (1 - masks)
        return combined_pixel_values

    @staticmethod
    def pixel_values_to_images(pixel_values: torch.Tensor):
        pixel_values = rearrange(pixel_values, "f c h w -> f h w c")
        pixel_values = (pixel_values / 2 + 0.5).clamp(0, 1)
        images = (pixel_values * 255).to(torch.uint8)
        images = images.cpu().numpy()
        return images

    def affine_transform_video(self, video_frames: np.ndarray):
        faces = []
        boxes = []
        affine_matrices = []
        print(f"Affine transforming {len(video_frames)} faces...")
        for frame in tqdm.tqdm(video_frames):
            face, box, affine_matrix = self.image_processor.affine_transform(frame)
            faces.append(face)
            boxes.append(box)
            affine_matrices.append(affine_matrix)

        faces = torch.stack(faces)
        return faces, boxes, affine_matrices

    def restore_video(self, faces: torch.Tensor, video_frames: np.ndarray, boxes: list, affine_matrices: list):
        video_frames = video_frames[: len(faces)]
        out_frames = []
        print(f"Restoring {len(faces)} faces...")
        for index, face in enumerate(tqdm.tqdm(faces)):
            x1, y1, x2, y2 = boxes[index]
            height = int(y2 - y1)
            width = int(x2 - x1)
            face = torchvision.transforms.functional.resize(
                face, size=(height, width), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
            )
            out_frame = self.image_processor.restorer.restore_img(video_frames[index], face, affine_matrices[index])
            out_frames.append(out_frame)
        return np.stack(out_frames, axis=0)

    def loop_video(self, whisper_chunks: list, video_frames: np.ndarray):
        # If the audio is longer than the video, we need to loop the video
        if len(whisper_chunks) > len(video_frames):
            faces, boxes, affine_matrices = self.affine_transform_video(video_frames)
            num_loops = math.ceil(len(whisper_chunks) / len(video_frames))
            loop_video_frames = []
            loop_faces = []
            loop_boxes = []
            loop_affine_matrices = []
            for i in range(num_loops):
                if i % 2 == 0:
                    loop_video_frames.append(video_frames)
                    loop_faces.append(faces)
                    loop_boxes += boxes
                    loop_affine_matrices += affine_matrices
                else:
                    loop_video_frames.append(video_frames[::-1])
                    loop_faces.append(faces.flip(0))
                    loop_boxes += boxes[::-1]
                    loop_affine_matrices += affine_matrices[::-1]

            video_frames = np.concatenate(loop_video_frames, axis=0)[: len(whisper_chunks)]
            faces = torch.cat(loop_faces, dim=0)[: len(whisper_chunks)]
            boxes = loop_boxes[: len(whisper_chunks)]
            affine_matrices = loop_affine_matrices[: len(whisper_chunks)]
        else:
            video_frames = video_frames[: len(whisper_chunks)]
            faces, boxes, affine_matrices = self.affine_transform_video(video_frames)

        return video_frames, faces, boxes, affine_matrices

    def get_prepared_video_cache(self, video_path: str, video_cache: dict, profile: dict = None):
        video_key = os.path.abspath(video_path)
        if video_key not in video_cache:
            if profile is not None:
                profile["cache"]["video_misses"] += 1
            print(f"Preparing cached video tensors: {video_path}")
            video_frames = read_video(video_path, use_decord=False)
            faces, boxes, affine_matrices = self.affine_transform_video(video_frames)
            video_cache[video_key] = {
                "video_key": video_key,
                "video_frames": video_frames,
                "faces": faces,
                "boxes": boxes,
                "affine_matrices": affine_matrices,
            }
        else:
            if profile is not None:
                profile["cache"]["video_hits"] += 1
            print(f"Using cached video tensors: {video_path}")
        return video_cache[video_key]

    def loop_prepared_video(self, target_len: int, cached_video: dict):
        video_frames = cached_video["video_frames"]
        faces = cached_video["faces"]
        boxes = cached_video["boxes"]
        affine_matrices = cached_video["affine_matrices"]

        if target_len > len(video_frames):
            num_loops = math.ceil(target_len / len(video_frames))
            loop_video_frames = []
            loop_faces = []
            loop_boxes = []
            loop_affine_matrices = []
            for i in range(num_loops):
                if i % 2 == 0:
                    loop_video_frames.append(video_frames)
                    loop_faces.append(faces)
                    loop_boxes += boxes
                    loop_affine_matrices += affine_matrices
                else:
                    loop_video_frames.append(video_frames[::-1])
                    loop_faces.append(faces.flip(0))
                    loop_boxes += boxes[::-1]
                    loop_affine_matrices += affine_matrices[::-1]

            video_frames = np.concatenate(loop_video_frames, axis=0)[:target_len]
            faces = torch.cat(loop_faces, dim=0)[:target_len]
            boxes = loop_boxes[:target_len]
            affine_matrices = loop_affine_matrices[:target_len]
        else:
            video_frames = video_frames[:target_len]
            faces = faces[:target_len]
            boxes = boxes[:target_len]
            affine_matrices = affine_matrices[:target_len]

        return video_frames, faces, boxes, affine_matrices

    def get_chunk_vae_cache(
        self,
        sample: dict,
        start: int,
        num_frames: int,
        latent_h: int,
        latent_w: int,
        weight_dtype: torch.dtype,
        device,
        generator,
        chunk_vae_cache: dict,
        profile: dict = None,
    ):
        available_end = min(start + num_frames, len(sample["faces"]))
        cache_key = (sample["video_key"], start, available_end, num_frames)
        if cache_key in chunk_vae_cache:
            if profile is not None:
                profile["cache"]["vae_hits"] += 1
            return chunk_vae_cache[cache_key]
        if profile is not None:
            profile["cache"]["vae_misses"] += 1

        inference_faces = self.pad_first_dim(sample["faces"][start:available_end], num_frames)
        ref_pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
            inference_faces, affine_transform=False
        )

        ref_flat = rearrange(ref_pixel_values, "f c h w -> f c h w").to(device=device, dtype=weight_dtype)
        masked_flat = rearrange(masked_pixel_values, "f c h w -> f c h w").to(device=device, dtype=weight_dtype)
        masks_small = torch.nn.functional.interpolate(masks, size=(latent_h, latent_w))
        mask_latents = rearrange(
            masks_small.to(device=device, dtype=weight_dtype), "f c h w -> 1 c f h w"
        )

        masked_image_latents = self.vae.encode(masked_flat).latent_dist.sample(generator=generator)
        masked_image_latents = (
            masked_image_latents - self.vae.config.shift_factor
        ) * self.vae.config.scaling_factor
        masked_image_latents = rearrange(
            masked_image_latents.to(device=device, dtype=weight_dtype), "f c h w -> 1 c f h w"
        )

        ref_latents = self.vae.encode(ref_flat).latent_dist.sample(generator=generator)
        ref_latents = (ref_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        ref_latents = rearrange(ref_latents.to(device=device, dtype=weight_dtype), "f c h w -> 1 c f h w")

        cached = {
            "ref_pixel_values": ref_pixel_values.cpu(),
            "masks": masks.cpu(),
            "mask_latents": mask_latents.cpu(),
            "masked_image_latents": masked_image_latents.cpu(),
            "ref_latents": ref_latents.cpu(),
        }
        chunk_vae_cache[cache_key] = cached

        del ref_flat, masked_flat, masks_small, mask_latents, masked_image_latents, ref_latents
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()

        return cached

    @torch.inference_mode()
    def __call__(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        num_frames: int = 16,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.5,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        mask_image_path: str = "latentsync/utils/mask.png",
        temp_dir: str = "temp",
        chunk_batch_size: int = 1,
        trace_unet_shapes: bool = False,
        trace_unet_shapes_path: Optional[str] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        is_train = self.unet.training
        self.unet.eval()
        previous_trace_unet_shapes = getattr(self.unet, "trace_unet_shapes", False)
        previous_trace_context = getattr(self.unet, "_trace_context", None)
        previous_trace_writer = getattr(self.unet, "_trace_shape_writer", None)
        trace_shape_writer = None
        if trace_unet_shapes_path:
            trace_dir = os.path.dirname(trace_unet_shapes_path)
            if trace_dir:
                os.makedirs(trace_dir, exist_ok=True)
            trace_shape_writer = open(trace_unet_shapes_path, "w", encoding="utf-8")
            trace_unet_shapes = True
        self.unet.trace_unet_shapes = trace_unet_shapes
        self.unet._trace_shape_writer = trace_shape_writer

        check_ffmpeg_installed()

        # 0. Define call parameters
        if chunk_batch_size < 1:
            raise ValueError("chunk_batch_size must be >= 1")
        device = self._execution_device
        profile_started_at = time.perf_counter()
        profile = self._new_profile(
            "single_chunk_microbatch" if chunk_batch_size > 1 else "single",
            chunk_batch_size,
            1,
            device,
        )
        profile["chunk_batch_size"] = chunk_batch_size
        phase_started_at = self._profile_start(device)
        mask_image = load_fixed_mask(height, mask_image_path)
        self.image_processor = ImageProcessor(height, device="cuda", mask_image=mask_image)
        self.set_progress_bar_config(desc=f"Sample frames: {num_frames}, chunk batch: {chunk_batch_size}")

        # 1. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Check inputs
        self.check_inputs(height, width, callback_steps)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        whisper_feature = self.audio_encoder.audio2feat(audio_path)
        whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)

        audio_samples = read_audio(audio_path)
        video_frames = read_video(video_path, use_decord=False)

        video_frames, faces, boxes, affine_matrices = self.loop_video(whisper_chunks, video_frames)
        profile["phases"]["preprocess"] += self._profile_elapsed(phase_started_at, device)

        synced_video_frames = []

        num_channels_latents = self.vae.config.latent_channels
        base_latent = self.prepare_chunk_latents(
            num_channels_latents,
            1,
            height,
            width,
            weight_dtype,
            device,
            generator,
        ).cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        num_inferences = math.ceil(len(whisper_chunks) / num_frames)
        group_starts = range(0, num_inferences, chunk_batch_size)
        for group_start in tqdm.tqdm(group_starts, desc=f"Doing chunk micro-batch x{chunk_batch_size}..."):
            group_chunk_ids = list(range(group_start, min(group_start + chunk_batch_size, num_inferences)))
            group_size = len(group_chunk_ids)

            batch_audio_embeds = []
            batch_ref_pixel_values = []
            batch_masks = []
            batch_mask_latents = []
            batch_masked_image_latents = []
            batch_ref_latents = []
            batch_latents = []
            batch_valid_lens = []

            for chunk_id in group_chunk_ids:
                start = chunk_id * num_frames
                end = min(start + num_frames, len(whisper_chunks))
                valid_len = end - start
                batch_valid_lens.append(valid_len)

                if self.unet.add_audio_layer:
                    audio_embeds = self.slice_or_pad_chunks(whisper_chunks, start, num_frames)
                    batch_audio_embeds.append(audio_embeds.to(dtype=weight_dtype).cpu())

                inference_faces = self.pad_first_dim(faces[start:end], num_frames)
                ref_pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
                    inference_faces, affine_transform=False
                )

                mask_latents, masked_image_latents = self.prepare_mask_latents(
                    masks,
                    masked_pixel_values,
                    height,
                    width,
                    weight_dtype,
                    device,
                    generator,
                    False,
                )
                ref_latents = self.prepare_image_latents(
                    ref_pixel_values,
                    device,
                    weight_dtype,
                    generator,
                    False,
                )

                latents = base_latent.repeat(1, 1, num_frames, 1, 1)

                batch_ref_pixel_values.append(ref_pixel_values.cpu())
                batch_masks.append(masks.cpu())
                batch_mask_latents.append(mask_latents.cpu())
                batch_masked_image_latents.append(masked_image_latents.cpu())
                batch_ref_latents.append(ref_latents.cpu())
                batch_latents.append(latents)

                del inference_faces, ref_pixel_values, masked_pixel_values, masks, mask_latents, masked_image_latents, ref_latents
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            if self.unet.add_audio_layer:
                audio_embeds = torch.stack(batch_audio_embeds, dim=0).to(device=device, dtype=weight_dtype)
                if do_classifier_free_guidance:
                    audio_embeds = torch.cat([torch.zeros_like(audio_embeds), audio_embeds], dim=0)
            else:
                audio_embeds = None

            ref_pixel_values = torch.stack(batch_ref_pixel_values, dim=0)
            masks = torch.stack(batch_masks, dim=0)
            mask_latents = torch.cat(batch_mask_latents, dim=0).to(device=device, dtype=weight_dtype)
            masked_image_latents = torch.cat(batch_masked_image_latents, dim=0).to(device=device, dtype=weight_dtype)
            ref_latents = torch.cat(batch_ref_latents, dim=0).to(device=device, dtype=weight_dtype)
            latents = torch.cat(batch_latents, dim=0).to(device=device, dtype=weight_dtype)

            if do_classifier_free_guidance:
                mask_latents = torch.cat([mask_latents, mask_latents], dim=0)
                masked_image_latents = torch.cat([masked_image_latents, masked_image_latents], dim=0)
                ref_latents = torch.cat([ref_latents, ref_latents], dim=0)

            # 9. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            denoise_started_at = self._profile_start(device)
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for j, t in enumerate(timesteps):
                    trace_timestep = t.item() if torch.is_tensor(t) else t
                    trace_this_step = self._should_trace_unet_step(trace_unet_shapes, group_start, j)
                    # expand the latents if we are doing classifier free guidance
                    unet_input = torch.cat([latents, latents], dim=0) if do_classifier_free_guidance else latents

                    unet_input = self.scheduler.scale_model_input(unet_input, t)
                    # concat latents, mask, masked_image_latents in the channel dimension
                    unet_input = torch.cat([unet_input, mask_latents, masked_image_latents, ref_latents], dim=1)
                    self._trace_pipeline_shape(
                        trace_this_step,
                        "chunk micro-batch UNet input after channel concat",
                        unet_input,
                        "channels = 4 noisy + 1 mask + 4 masked-image + 4 reference = 13",
                        chunk=group_start,
                        step=j,
                        timestep=trace_timestep,
                        writer=trace_shape_writer,
                    )
                    # predict the noise residual
                    self.unet.trace_unet_shapes = trace_this_step
                    self.unet._trace_context = {"chunk": group_start, "step": j, "timestep": trace_timestep}
                    try:
                        noise_pred = self.unet(unet_input, t, encoder_hidden_states=audio_embeds).sample
                    finally:
                        self.unet._trace_context = None
                    profile["num_unet_forward_calls"] += 1

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                    self._trace_pipeline_shape(
                        trace_this_step,
                        "scheduler.step prev_sample",
                        latents,
                        "updated latent x_(t-1), fed into the next denoise timestep",
                        chunk=group_start,
                        step=j,
                        timestep=trace_timestep,
                        writer=trace_shape_writer,
                    )

                    # call the callback, if provided
                    if j == len(timesteps) - 1 or ((j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and j % callback_steps == 0:
                            callback(j, t, latents)
            profile["phases"]["denoise"] += self._profile_elapsed(denoise_started_at, device)

            latents_cpu = latents.detach().cpu()
            ref_pixel_values_cpu = ref_pixel_values
            masks_cpu = masks
            del audio_embeds, mask_latents, masked_image_latents, ref_latents, ref_pixel_values, masks, latents
            if device.type == "cuda":
                torch.cuda.empty_cache()

            for local_index, valid_len in enumerate(batch_valid_lens):
                if valid_len <= 0:
                    continue
                chunk_latents = latents_cpu[local_index : local_index + 1].to(device=device, dtype=weight_dtype)
                decoded_latents = self.decode_latents(chunk_latents)
                ref_original = ref_pixel_values_cpu[local_index, :valid_len].to(device=device, dtype=weight_dtype)
                masks_original = masks_cpu[local_index, :valid_len].to(device=device, dtype=weight_dtype)
                pasted = decoded_latents[:valid_len] * (1 - masks_original)
                pasted = pasted + ref_original * masks_original
                synced_video_frames.append(pasted.detach().cpu())
                del chunk_latents, decoded_latents, ref_original, masks_original
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            del (
                batch_audio_embeds,
                batch_ref_pixel_values,
                batch_masks,
                batch_mask_latents,
                batch_masked_image_latents,
                batch_ref_latents,
                batch_latents,
                batch_valid_lens,
                latents_cpu,
                ref_pixel_values_cpu,
                masks_cpu,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

        synced_video_frames = self.restore_video(torch.cat(synced_video_frames), video_frames, boxes, affine_matrices)

        audio_samples_remain_length = int(synced_video_frames.shape[0] / video_fps * audio_sample_rate)
        audio_samples = audio_samples[:audio_samples_remain_length].cpu().numpy()

        if is_train:
            self.unet.train()
        self.unet.trace_unet_shapes = previous_trace_unet_shapes
        self.unet._trace_context = previous_trace_context
        self.unet._trace_shape_writer = previous_trace_writer
        if trace_shape_writer is not None:
            trace_shape_writer.close()

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        write_video(os.path.join(temp_dir, "video.mp4"), synced_video_frames, fps=video_fps)

        sf.write(os.path.join(temp_dir, "audio.wav"), audio_samples, audio_sample_rate)

        command = f"ffmpeg -y -loglevel error -nostdin -i {os.path.join(temp_dir, 'video.mp4')} -i {os.path.join(temp_dir, 'audio.wav')} -c:v libx264 -crf 18 -c:a aac -q:v 0 -q:a 0 {video_out_path}"
        result = subprocess.run(command, shell=True)
        if result.returncode == 0 and os.path.exists(video_out_path):
            profile["success"] = 1
            profile["outputs"].append(video_out_path)

        return self._finish_profile(profile, device, profile_started_at)

    @torch.inference_mode()
    def batch_inference(
        self,
        video_paths: list,
        audio_paths: list,
        video_out_paths: list,
        batch_size: int = 2,
        num_frames: int = 16,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.5,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        mask_image_path: str = "latentsync/utils/mask.png",
        temp_dir: str = "temp",
        trace_unet_shapes: bool = False,
        trace_unet_shapes_path: Optional[str] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback_steps: Optional[int] = 1,
    ):
        if not (len(video_paths) == len(audio_paths) == len(video_out_paths)):
            raise ValueError("video_paths, audio_paths, and video_out_paths must have the same length")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        is_train = self.unet.training
        self.unet.eval()
        previous_trace_unet_shapes = getattr(self.unet, "trace_unet_shapes", False)
        previous_trace_context = getattr(self.unet, "_trace_context", None)
        previous_trace_writer = getattr(self.unet, "_trace_shape_writer", None)
        trace_shape_writer = None
        if trace_unet_shapes_path:
            trace_dir = os.path.dirname(trace_unet_shapes_path)
            if trace_dir:
                os.makedirs(trace_dir, exist_ok=True)
            trace_shape_writer = open(trace_unet_shapes_path, "w", encoding="utf-8")
            trace_unet_shapes = True
        self.unet.trace_unet_shapes = trace_unet_shapes
        self.unet._trace_shape_writer = trace_shape_writer
        check_ffmpeg_installed()

        device = self._execution_device
        profile_started_at = time.perf_counter()
        profile = self._new_profile("gpu_batch", batch_size, len(video_out_paths), device)
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        self.check_inputs(height, width, callback_steps)

        mask_image = load_fixed_mask(height, mask_image_path)
        self.image_processor = ImageProcessor(height, device="cuda", mask_image=mask_image)
        self.set_progress_bar_config(desc=f"Batch sample frames: {num_frames}")

        do_classifier_free_guidance = guidance_scale > 1.0
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_channels_latents = self.vae.config.latent_channels
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor

        samples = []
        video_cache = {}
        chunk_vae_cache = {}
        phase_started_at = self._profile_start(device)
        for index, (video_path, audio_path) in enumerate(zip(video_paths, audio_paths)):
            whisper_feature = self.audio_encoder.audio2feat(audio_path)
            whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)
            audio_samples = read_audio(audio_path)
            cached_video = self.get_prepared_video_cache(video_path, video_cache, profile)
            video_frames, faces, boxes, affine_matrices = self.loop_prepared_video(len(whisper_chunks), cached_video)
            samples.append(
                {
                    "index": index,
                    "audio_path": audio_path,
                    "video_key": cached_video["video_key"],
                    "whisper_chunks": whisper_chunks,
                    "audio_samples": audio_samples,
                    "video_frames": video_frames,
                    "faces": faces,
                    "boxes": boxes,
                    "affine_matrices": affine_matrices,
                }
            )
        profile["phases"]["preprocess"] += self._profile_elapsed(phase_started_at, device)

        synced_by_index = {}
        groups = [samples[start : start + batch_size] for start in range(0, len(samples), batch_size)]
        for group in groups:
            sample_batch = len(group)
            max_chunks = max(len(sample["whisper_chunks"]) for sample in group)
            all_latents = self.prepare_batch_latents(
                sample_batch,
                max_chunks,
                num_channels_latents,
                height,
                width,
                weight_dtype,
                device,
                generator,
            )
            synced_chunks = [[] for _ in group]
            num_inferences = math.ceil(max_chunks / num_frames)

            for i in tqdm.tqdm(range(num_inferences), desc=f"Doing batch inference x{sample_batch}..."):
                start = i * num_frames
                end = min((i + 1) * num_frames, max_chunks)
                chunk_len = end - start

                batch_audio_embeds = []
                batch_ref_pixels = []
                batch_masks = []
                batch_mask_latents = []
                batch_masked_image_latents = []
                batch_ref_latents = []
                for sample in group:
                    if self.unet.add_audio_layer:
                        audio_embeds = self.slice_or_pad_chunks(sample["whisper_chunks"], start, num_frames)
                        batch_audio_embeds.append(audio_embeds.to(device, dtype=weight_dtype))
                    cached_chunk = self.get_chunk_vae_cache(
                        sample,
                        start,
                        num_frames,
                        latent_h,
                        latent_w,
                        weight_dtype,
                        device,
                        generator,
                        chunk_vae_cache,
                        profile,
                    )
                    batch_ref_pixels.append(cached_chunk["ref_pixel_values"])
                    batch_masks.append(cached_chunk["masks"])
                    batch_mask_latents.append(cached_chunk["mask_latents"])
                    batch_masked_image_latents.append(cached_chunk["masked_image_latents"])
                    batch_ref_latents.append(cached_chunk["ref_latents"])

                if self.unet.add_audio_layer:
                    audio_embeds = torch.stack(batch_audio_embeds, dim=0)
                    if do_classifier_free_guidance:
                        audio_embeds = torch.cat([torch.zeros_like(audio_embeds), audio_embeds], dim=0)
                else:
                    audio_embeds = None

                ref_pixel_values = torch.stack(batch_ref_pixels, dim=0)
                masks = torch.stack(batch_masks, dim=0)
                mask_latents = torch.cat(batch_mask_latents, dim=0).to(device=device, dtype=weight_dtype)
                masked_image_latents = torch.cat(batch_masked_image_latents, dim=0).to(
                    device=device, dtype=weight_dtype
                )
                ref_latents = torch.cat(batch_ref_latents, dim=0).to(device=device, dtype=weight_dtype)

                if do_classifier_free_guidance:
                    mask_latents = torch.cat([mask_latents, mask_latents], dim=0)
                    masked_image_latents = torch.cat([masked_image_latents, masked_image_latents], dim=0)
                    ref_latents = torch.cat([ref_latents, ref_latents], dim=0)

                latents = all_latents[:, :, start : start + num_frames]
                if latents.shape[2] < num_frames:
                    pad = torch.zeros(
                        (sample_batch, num_channels_latents, num_frames - latents.shape[2], latent_h, latent_w),
                        device=device,
                        dtype=weight_dtype,
                    )
                    latents = torch.cat([latents, pad], dim=2)

                denoise_started_at = self._profile_start(device)
                with self.progress_bar(total=num_inference_steps) as progress_bar:
                    for j, t in enumerate(timesteps):
                        trace_timestep = t.item() if torch.is_tensor(t) else t
                        trace_this_step = self._should_trace_unet_step(trace_unet_shapes, i, j)
                        unet_input = torch.cat([latents, latents], dim=0) if do_classifier_free_guidance else latents
                        unet_input = self.scheduler.scale_model_input(unet_input, t)
                        unet_input = torch.cat(
                            [unet_input, mask_latents, masked_image_latents, ref_latents], dim=1
                        )
                        self._trace_pipeline_shape(
                            trace_this_step,
                            "batch UNet input after channel concat",
                            unet_input,
                            "channels = 4 noisy + 1 mask + 4 masked-image + 4 reference = 13",
                            chunk=i,
                            step=j,
                            timestep=trace_timestep,
                            writer=trace_shape_writer,
                        )
                        self.unet.trace_unet_shapes = trace_this_step
                        self.unet._trace_context = {"chunk": i, "step": j, "timestep": trace_timestep}
                        try:
                            noise_pred = self.unet(unet_input, t, encoder_hidden_states=audio_embeds).sample
                        finally:
                            self.unet._trace_context = None
                        profile["num_unet_forward_calls"] += 1
                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)
                        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                        self._trace_pipeline_shape(
                            trace_this_step,
                            "batch scheduler.step prev_sample",
                            latents,
                            "updated latent x_(t-1), fed into the next denoise timestep",
                            chunk=i,
                            step=j,
                            timestep=trace_timestep,
                            writer=trace_shape_writer,
                        )
                        progress_bar.update()
                profile["phases"]["denoise"] += self._profile_elapsed(denoise_started_at, device)

                all_latents[:, :, start : start + chunk_len] = latents[:, :, :chunk_len]
                decoded_latents = self.decode_latents(latents[:, :, :chunk_len])
                decoded_latents = rearrange(decoded_latents, "(b f) c h w -> b f c h w", b=sample_batch)
                ref_original = ref_pixel_values[:, :chunk_len].to(device=device, dtype=weight_dtype)
                masks_original = masks[:, :chunk_len].to(device=device, dtype=weight_dtype)

                for local_index, sample in enumerate(group):
                    valid_len = max(0, min(chunk_len, len(sample["whisper_chunks"]) - start))
                    if valid_len <= 0:
                        continue
                    pasted = decoded_latents[local_index, :valid_len] * (1 - masks_original[local_index, :valid_len])
                    pasted = pasted + ref_original[local_index, :valid_len] * masks_original[local_index, :valid_len]
                    synced_chunks[local_index].append(pasted.detach().cpu())

                del (
                    batch_audio_embeds,
                    batch_ref_pixels,
                    batch_masks,
                    batch_mask_latents,
                    batch_masked_image_latents,
                    batch_ref_latents,
                    audio_embeds,
                    ref_pixel_values,
                    masks,
                    mask_latents,
                    masked_image_latents,
                    ref_latents,
                    latents,
                    decoded_latents,
                    ref_original,
                    masks_original,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            for local_index, sample in enumerate(group):
                synced_by_index[sample["index"]] = torch.cat(synced_chunks[local_index], dim=0)

        for sample, video_out_path in zip(samples, video_out_paths):
            synced_video_frames = self.restore_video(
                synced_by_index[sample["index"]],
                sample["video_frames"],
                sample["boxes"],
                sample["affine_matrices"],
            )
            audio_samples_remain_length = int(synced_video_frames.shape[0] / video_fps * audio_sample_rate)
            audio_samples = sample["audio_samples"][:audio_samples_remain_length].cpu().numpy()

            item_temp_dir = os.path.join(temp_dir, os.path.splitext(os.path.basename(video_out_path))[0])
            if os.path.exists(item_temp_dir):
                shutil.rmtree(item_temp_dir)
            os.makedirs(item_temp_dir, exist_ok=True)
            write_video(os.path.join(item_temp_dir, "video.mp4"), synced_video_frames, fps=video_fps)
            sf.write(os.path.join(item_temp_dir, "audio.wav"), audio_samples, audio_sample_rate)
            command = f"ffmpeg -y -loglevel error -nostdin -i {os.path.join(item_temp_dir, 'video.mp4')} -i {os.path.join(item_temp_dir, 'audio.wav')} -c:v libx264 -crf 18 -c:a aac -q:v 0 -q:a 0 {video_out_path}"
            result = subprocess.run(command, shell=True)
            if result.returncode == 0 and os.path.exists(video_out_path):
                profile["success"] += 1
                profile["outputs"].append(video_out_path)

        if is_train:
            self.unet.train()
        self.unet.trace_unet_shapes = previous_trace_unet_shapes
        self.unet._trace_context = previous_trace_context
        self.unet._trace_shape_writer = previous_trace_writer
        if trace_shape_writer is not None:
            trace_shape_writer.close()

        return self._finish_profile(profile, device, profile_started_at)
