###############################################################################
#  EchoMimicV3 digital human backend
#
#  This adapter keeps LiveTalking's TTS/session/output layers and uses the
#  official EchoMimicV3 Flash pipeline to generate complete video segments.
###############################################################################

import os
import sys
import time
import math
import glob
import json
import tempfile
import queue
from dataclasses import dataclass
from pathlib import Path
from threading import Thread, Event, Lock

import cv2
import numpy as np
import soundfile as sf
import torch
from PIL import Image

from avatars.base_avatar import BaseAvatar
from llm_prompt_deepseek import generate_echomimicv3_prompts
from registry import register
from utils.image import mirror_index
from utils.logger import logger


@dataclass
class EchoMimicV3AvatarData:
    avatar_id: str
    ref_image_path: str
    prompt: str
    negative_prompt: str
    description: str
    idle_frames: list


@dataclass
class EchoMimicV3AudioJob:
    audio: np.ndarray
    start_event: dict
    end_event: dict
    token: int
    speech_text: str
    prompt: str
    negative_prompt: str


def _resolve_path(path: str) -> str:
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    return os.path.abspath(path)


def _first_existing(paths):
    for path in paths:
        if path and os.path.exists(path):
            return path
    return paths[0] if paths else ""


class EchoMimicV3FlashEngine:
    def __init__(self, opt):
        self.opt = opt
        self.repo_path = _resolve_path(opt.echomimicv3_repo)
        self.model_root = _resolve_path(opt.echomimicv3_model_dir)
        self.config_path = _resolve_path(opt.echomimicv3_config_path) or os.path.join(
            self.repo_path, "config", "config.yaml"
        )
        self.base_model_dir = _resolve_path(opt.echomimicv3_base_model_dir) or _first_existing(
            [
                os.path.join(self.model_root, "Wan2.1-Fun-V1.1-1.3B-InP"),
                os.path.join(self.model_root, "Wan2.1-Fun-1.3B-InP"),
                os.path.abspath("Wan2.1-Fun-V1.1-1.3B-InP"),
                os.path.abspath("Wan2.1-Fun-1.3B-InP"),
            ]
        )
        self.wav2vec_dir = _resolve_path(opt.echomimicv3_wav2vec_dir) or _first_existing(
            [
                os.path.join(self.model_root, "chinese-wav2vec2-base"),
                os.path.join(self.model_root, "wav2vec2-base-960h"),
                os.path.abspath("chinese-wav2vec2-base"),
                os.path.abspath("wav2vec2-base-960h"),
            ]
        )
        self.transformer_path = _resolve_path(opt.echomimicv3_transformer_path) or _first_existing(
            [
                os.path.join(self.model_root, "echomimicv3-flash-pro", "diffusion_pytorch_model.safetensors"),
                os.path.join(self.model_root, "flash", "transformer", "diffusion_pytorch_model.safetensors"),
                os.path.join(self.model_root, "transformer", "diffusion_pytorch_model.safetensors"),
            ]
        )

        self._validate_paths()
        if self.repo_path not in sys.path:
            sys.path.insert(0, self.repo_path)

        self._load_dependencies()
        self._load_models()
        self._counter = 0
        self._lock = Lock()

    def _validate_paths(self):
        required = {
            "EchoMimicV3 repo": self.repo_path,
            "EchoMimicV3 config": self.config_path,
            "Wan2.1 base model": self.base_model_dir,
            "chinese-wav2vec2-base": self.wav2vec_dir,
            "EchoMimicV3 transformer": self.transformer_path,
        }
        missing = [f"{name}: {path}" for name, path in required.items() if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(
                "EchoMimicV3 is not ready. Missing required paths:\n"
                + "\n".join(missing)
                + "\nPlease download the official code plus Wan2.1-Fun-V1.1-1.3B-InP and chinese-wav2vec2-base."
            )
        expected_base_files = [
            "Wan2.1_VAE.pth",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        ]
        missing_base_files = [
            os.path.join(self.base_model_dir, name)
            for name in expected_base_files
            if not os.path.exists(os.path.join(self.base_model_dir, name))
        ]
        if missing_base_files:
            raise FileNotFoundError(
                "EchoMimicV3 base model directory exists but is incomplete. Missing files:\n"
                + "\n".join(missing_base_files)
                + "\nThis usually means the Wan2.1-Fun weights were not downloaded with Git LFS."
            )
        if "flash" in self.transformer_path.lower() and "chinese-wav2vec2-base" not in self.wav2vec_dir:
            logger.warning(
                "EchoMimicV3 Flash-Pro is using wav2vec_dir=%s. "
                "The official Flash-Pro setup expects chinese-wav2vec2-base; "
                "using wav2vec2-base-960h can noticeably hurt Chinese lip sync.",
                self.wav2vec_dir,
            )

    def _load_dependencies(self):
        from diffusers import FlowMatchEulerDiscreteScheduler
        from einops import rearrange
        from omegaconf import OmegaConf
        import pyloudnorm as pyln
        from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor

        from src.cache_utils import get_teacache_coefficients
        from src.fm_solvers import FlowDPMSolverMultistepScheduler
        from src.fm_solvers_unipc import FlowUniPCMultistepScheduler
        from src.pipeline_wan_fun_inpaint_audio_2512 import WanFunInpaintAudioPipeline
        from src.utils import filter_kwargs, get_image_to_video_latent2
        from src.wan_image_encoder import CLIPModel
        from src.wan_text_encoder import WanT5EncoderModel
        from src.wan_transformer3d_audio_2512 import WanTransformerAudioMask3DModel as WanTransformer
        from src.wan_vae import AutoencoderKLWan
        from src.wav2vec2 import Wav2Vec2Model

        self.FlowMatchEulerDiscreteScheduler = FlowMatchEulerDiscreteScheduler
        self.FlowDPMSolverMultistepScheduler = FlowDPMSolverMultistepScheduler
        self.FlowUniPCMultistepScheduler = FlowUniPCMultistepScheduler
        self.OmegaConf = OmegaConf
        self.AutoTokenizer = AutoTokenizer
        self.Wav2Vec2FeatureExtractor = Wav2Vec2FeatureExtractor
        self.Wav2Vec2Model = Wav2Vec2Model
        self.AutoencoderKLWan = AutoencoderKLWan
        self.CLIPModel = CLIPModel
        self.WanT5EncoderModel = WanT5EncoderModel
        self.WanTransformer = WanTransformer
        self.WanFunInpaintAudioPipeline = WanFunInpaintAudioPipeline
        self.filter_kwargs = filter_kwargs
        self.get_image_to_video_latent2 = get_image_to_video_latent2
        self.get_teacache_coefficients = get_teacache_coefficients
        try:
            from src.dist import set_multi_gpus_devices
            self.set_multi_gpus_devices = set_multi_gpus_devices
        except ModuleNotFoundError:
            self.set_multi_gpus_devices = self._set_single_device
        self.rearrange = rearrange
        self.pyln = pyln

    @staticmethod
    def _set_single_device(ulysses_degree=1, ring_degree=1):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        raise RuntimeError(
            "CUDA is not available. EchoMimicV3 Flash requires a working NVIDIA GPU/driver "
            "for practical LiveTalking inference."
        )

    def _load_models(self):
        logger.info("loading EchoMimicV3 Flash engine")
        opt = self.opt
        self.weight_dtype = torch.bfloat16 if opt.echomimicv3_weight_dtype == "bfloat16" else torch.float16
        if torch.cuda.is_available() and hasattr(torch.backends.cuda, "preferred_linalg_library"):
            try:
                torch.backends.cuda.preferred_linalg_library("magma")
                logger.info("Torch CUDA preferred linalg backend set to MAGMA for EchoMimicV3")
            except Exception:
                logger.exception("Failed to set Torch CUDA preferred linalg backend; continue with default")

        self.audio_encoder = self.Wav2Vec2Model.from_pretrained(
            self.wav2vec_dir, local_files_only=True
        ).to("cpu")
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.wav2vec_feature_extractor = self.Wav2Vec2FeatureExtractor.from_pretrained(
            self.wav2vec_dir, local_files_only=True
        )

        self.device = self.set_multi_gpus_devices(1, 1)
        self.config = self.OmegaConf.load(self.config_path)

        self.transformer = self.WanTransformer.from_pretrained(
            os.path.join(
                self.base_model_dir,
                self.config["transformer_additional_kwargs"].get("transformer_subpath", "transformer"),
            ),
            transformer_additional_kwargs=self.OmegaConf.to_container(
                self.config["transformer_additional_kwargs"]
            ),
            low_cpu_mem_usage=True,
            torch_dtype=self.weight_dtype,
        )

        from safetensors.torch import load_file

        state_dict = load_file(self.transformer_path)
        missing, unexpected = self.transformer.load_state_dict(state_dict, strict=False)
        logger.info("EchoMimicV3 transformer loaded, missing=%d unexpected=%d", len(missing), len(unexpected))

        self.vae = self.AutoencoderKLWan.from_pretrained(
            os.path.join(self.base_model_dir, self.config["vae_kwargs"].get("vae_subpath", "vae")),
            additional_kwargs=self.OmegaConf.to_container(self.config["vae_kwargs"]),
        ).to(self.weight_dtype)

        self.tokenizer = self.AutoTokenizer.from_pretrained(
            os.path.join(
                self.base_model_dir,
                self.config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"),
            )
        )
        self.text_encoder = self.WanT5EncoderModel.from_pretrained(
            os.path.join(
                self.base_model_dir,
                self.config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder"),
            ),
            additional_kwargs=self.OmegaConf.to_container(self.config["text_encoder_kwargs"]),
            low_cpu_mem_usage=True,
            torch_dtype=self.weight_dtype,
        ).eval()
        self.clip_image_encoder = self.CLIPModel.from_pretrained(
            os.path.join(
                self.base_model_dir,
                self.config["image_encoder_kwargs"].get("image_encoder_subpath", "image_encoder"),
            )
        ).to(self.weight_dtype).eval()

        scheduler_cls = {
            "Flow": self.FlowMatchEulerDiscreteScheduler,
            "Flow_Unipc": self.FlowUniPCMultistepScheduler,
            "Flow_DPM++": self.FlowDPMSolverMultistepScheduler,
        }["Flow_DPM++"]
        self.config["scheduler_kwargs"]["shift"] = 1
        scheduler = scheduler_cls(
            **self.filter_kwargs(scheduler_cls, self.OmegaConf.to_container(self.config["scheduler_kwargs"]))
        )

        self.pipeline = self.WanFunInpaintAudioPipeline(
            transformer=self.transformer,
            vae=self.vae,
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            scheduler=scheduler,
            clip_image_encoder=self.clip_image_encoder,
        )
        memory_mode = getattr(opt, "echomimicv3_gpu_memory_mode", "model_cpu_offload")
        self._configure_memory_mode(memory_mode)

        coefficients = self.get_teacache_coefficients(self.base_model_dir)
        if coefficients is not None:
            self.pipeline.transformer.enable_teacache(
                coefficients,
                opt.echomimicv3_num_steps,
                opt.echomimicv3_teacache_threshold,
                num_skip_start_steps=5,
                offload=bool(getattr(opt, "echomimicv3_teacache_offload", False)),
            )
        logger.info(
            "EchoMimicV3 Flash engine ready, memory_mode=%s, teacache_offload=%s",
            memory_mode,
            bool(getattr(opt, "echomimicv3_teacache_offload", False)),
        )

    def _configure_memory_mode(self, memory_mode: str):
        if memory_mode == "none":
            self.pipeline.to(device=self.device)
            return

        if memory_mode == "sequential_cpu_offload":
            try:
                self.pipeline.enable_sequential_cpu_offload(device=self.device)
                logger.info("EchoMimicV3 enabled sequential CPU offload")
                return
            except Exception:
                logger.exception("EchoMimicV3 sequential CPU offload failed; falling back to model CPU offload")

        try:
            self.pipeline.enable_model_cpu_offload(device=self.device)
            logger.info("EchoMimicV3 enabled model CPU offload")
        except Exception:
            logger.exception("EchoMimicV3 model CPU offload failed; falling back to full GPU mode")
            self.pipeline.to(device=self.device)

    def _get_sample_size(self, pil_img):
        sample_size = self.opt.echomimicv3_sample_size
        w, h = pil_img.size
        ori_a = w * h
        default_a = sample_size[0] * sample_size[1]
        if default_a < ori_a:
            ratio_a = math.sqrt(ori_a / sample_size[0] / sample_size[1])
            w = w / ratio_a // 16 * 16
            h = h / ratio_a // 16 * 16
        else:
            w = w // 16 * 16
            h = h // 16 * 16
        return int(h), int(w)

    def _loudness_norm(self, audio_array, sr=16000, lufs=-23):
        meter = self.pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio_array)
        if abs(loudness) > 100:
            return audio_array
        return self.pyln.normalize.loudness(audio_array, loudness, lufs)

    def _load_audio_16k(self, audio_path: str) -> np.ndarray:
        audio, sample_rate = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        if sample_rate != 16000 and audio.shape[0] > 0:
            import resampy

            audio = resampy.resample(audio, sample_rate, 16000)
        return np.asarray(audio, dtype=np.float32)

    def _get_audio_embed(self, mel_input, video_length, sr=16000):
        audio_feature = np.squeeze(self.wav2vec_feature_extractor(mel_input, sampling_rate=sr).input_values)
        audio_feature = torch.from_numpy(audio_feature).float().to(device="cpu").unsqueeze(0)
        with torch.no_grad():
            embeddings = self.audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)
        audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
        audio_emb = self.rearrange(audio_emb, "b s d -> s b d")
        return audio_emb.cpu().detach()

    def generate_frames(
        self,
        ref_image_path: str,
        audio: np.ndarray,
        prompt: str,
        negative_prompt: str = "",
        max_video_length: int = None,
    ):
        with self._lock:
            self._counter += 1
            seed = self.opt.echomimicv3_seed + self._counter
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                audio_path = tmp_audio.name
            try:
                sf.write(audio_path, audio, 16000)
                return self._generate_frames_from_audio_path(
                    ref_image_path,
                    audio_path,
                    prompt,
                    negative_prompt,
                    seed,
                    max_video_length=max_video_length,
                )
            finally:
                try:
                    os.remove(audio_path)
                except OSError:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    @torch.no_grad()
    def _generate_frames_from_audio_path(
        self,
        ref_image_path: str,
        audio_path: str,
        prompt: str,
        negative_prompt: str,
        seed: int,
        max_video_length: int = None,
    ):
        ref_image = Image.open(ref_image_path).convert("RGB")
        duration = sf.info(audio_path).frames / 16000.0
        max_length = int(max_video_length or self.opt.echomimicv3_video_length)
        video_length_actual = min(int(duration * self.opt.fps), max_length)
        if video_length_actual <= 0:
            return []
        ratio = self.vae.config.temporal_compression_ratio
        video_length_actual = int((video_length_actual - 1) // ratio * ratio) + 1

        mel_input = self._load_audio_16k(audio_path)
        mel_input = self._loudness_norm(mel_input, 16000)
        mel_input = mel_input[: int(video_length_actual / self.opt.fps * 16000)]
        audio_feature = self._get_audio_embed(mel_input, video_length_actual, sr=16000)
        audio_embeds = audio_feature.to(device=self.device, dtype=self.weight_dtype)

        indices = (torch.arange(5) - 2) * 1
        center_indices = torch.arange(0, video_length_actual, 1).unsqueeze(1) + indices.unsqueeze(0)
        center_indices = torch.clamp(center_indices, min=0, max=audio_embeds.shape[0] - 1)
        audio_embeds = audio_embeds[center_indices].unsqueeze(0).to(device=self.device)

        height, width = self._get_sample_size(ref_image)
        input_video, input_video_mask, clip_image = self.get_image_to_video_latent2(
            ref_image,
            None,
            video_length=video_length_actual,
            sample_size=[height, width],
        )

        generator = torch.Generator(device=self.device).manual_seed(seed)
        sample = self.pipeline(
            prompt,
            num_frames=video_length_actual,
            negative_prompt=negative_prompt or "",
            audio_embeds=audio_embeds,
            audio_scale=1.0,
            ip_mask=None,
            use_un_ip_mask=False,
            height=height,
            width=width,
            generator=generator,
            neg_scale=1.0,
            neg_steps=0,
            use_dynamic_cfg=False,
            use_dynamic_acfg=False,
            guidance_scale=self.opt.echomimicv3_guidance_scale,
            audio_guidance_scale=self.opt.echomimicv3_audio_guidance_scale,
            num_inference_steps=self.opt.echomimicv3_num_steps,
            video=input_video,
            mask_video=input_video_mask,
            clip_image=clip_image,
            cfg_skip_ratio=0.0,
            shift=5.0,
        ).videos

        frames = []
        sample = sample[:, :, :video_length_actual].detach().cpu()
        for frame_idx in range(video_length_actual):
            rgb = (sample[0, :, frame_idx].permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            frames.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return frames


def load_model(opt):
    return EchoMimicV3FlashEngine(opt)


def _read_prompt(path: str, fallback: str) -> str:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as file:
            prompt = file.read().strip()
            if prompt:
                return prompt
    return fallback


def _avatar_asset_candidates(avatar_id: str):
    assets_dir = os.path.abspath("assets/avatars")
    aliases = {
        "wav2lip256_avatar1": "avatar1",
        "wav2lip_avatar_1": "avatar1",
        "wav2lip_avatar_2": "avatar2",
        "wav2lip_avatar_3": "avatar3",
        "wav2lip_avatar_4": "avatar4",
        "wav2lip_avatar_5": "avatar5",
    }
    names = [avatar_id]
    alias = aliases.get(avatar_id)
    if alias:
        names.insert(0, alias)
    paths = []
    for name in names:
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            paths.append(os.path.join(assets_dir, f"{name}{ext}"))
    return paths


def _find_ref_image(avatar_id: str, avatar_path: str) -> str:
    candidates = []
    candidates.extend(_avatar_asset_candidates(avatar_id))
    echomimic_dir = os.path.join(avatar_path, "echomimicv3")
    for name in ("ref.png", "ref.jpg", "ref.jpeg", "reference.png", "reference.jpg"):
        candidates.append(os.path.join(echomimic_dir, name))
        candidates.append(os.path.join(avatar_path, name))
    candidates.extend(sorted(glob.glob(os.path.join(avatar_path, "full_imgs", "*.[jpJP][pnPN]*[gG]"))))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"EchoMimicV3 avatar needs a reference image under {echomimic_dir}/ref.png "
        f"or an existing full_imgs frame: {avatar_path}"
    )


def _load_idle_frames(avatar_path: str, ref_image_path: str):
    ref_frame = cv2.imread(ref_image_path)
    if ref_frame is None:
        raise FileNotFoundError(f"Cannot read reference image: {ref_image_path}")

    echomimic_dir = os.path.join(avatar_path, "echomimicv3")
    idle_dir_candidates = [
        os.path.join(echomimic_dir, "idle_frames_two_stage_silence"),
        os.path.join(echomimic_dir, "idle_frames"),
    ]
    image_paths = []
    idle_dir = idle_dir_candidates[-1]
    for candidate_dir in idle_dir_candidates:
        candidate_paths = []
        for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            candidate_paths.extend(glob.glob(os.path.join(candidate_dir, pattern)))
        candidate_paths = sorted(candidate_paths)
        if candidate_paths:
            idle_dir = candidate_dir
            image_paths = candidate_paths
            break

    if not image_paths:
        logger.info("EchoMimicV3 idle frames not found, fallback to reference image: %s", ref_image_path)
        return [ref_frame]

    frames = []
    for path in image_paths:
        frame = cv2.imread(path)
        if frame is None:
            logger.warning("skip unreadable EchoMimicV3 idle frame: %s", path)
            continue
        if not frames:
            target_h, target_w = frame.shape[:2]
        if frame.shape[:2] != (target_h, target_w):
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        frames.append(frame)

    if not frames:
        logger.warning("EchoMimicV3 idle frame directory has no readable images, fallback to reference image: %s", idle_dir)
        return [ref_frame]

    logger.info(
        "EchoMimicV3 idle video frames loaded: dir=%s frames=%d size=%dx%d",
        idle_dir,
        len(frames),
        target_w,
        target_h,
    )
    return frames


def load_avatar(avatar_id):
    avatar_path = f"./data/avatars/{avatar_id}"
    ref_image_path = _find_ref_image(avatar_id, avatar_path)
    prompt_path = os.path.join(avatar_path, "echomimicv3", "prompt.txt")
    negative_prompt_path = os.path.join(avatar_path, "echomimicv3", "negative_prompt.txt")
    config_path = os.path.join(avatar_path, "echomimicv3", "avatar_config.json")
    prompt = ""
    negative_prompt = ""
    description = ""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
            prompt = config.get("prompt", "")
            negative_prompt = config.get("negative_prompt", "")
            description = config.get("description", "")
    if not prompt:
        prompt = _read_prompt(prompt_path, "A person is speaking.")
    if not negative_prompt:
        negative_prompt = _read_prompt(
            negative_prompt_path,
            "blur, low quality, distorted face, bad hands, extra fingers, deformed body, strange movement, jitter, flicker",
        )
    if not description:
        description = f"{avatar_id}, a realistic front-facing digital human presenter."
    idle_frames = _load_idle_frames(avatar_path, ref_image_path)
    return EchoMimicV3AvatarData(
        avatar_id=avatar_id,
        ref_image_path=ref_image_path,
        prompt=prompt,
        negative_prompt=negative_prompt,
        description=description,
        idle_frames=idle_frames,
    )


def warm_up(opt, model, avatar):
    logger.info("EchoMimicV3 warm_up skipped; first generation will initialize runtime kernels.")


@register("avatar", "echomimicv3")
class EchoMimicV3Real(BaseAvatar):
    def __init__(self, opt, model: EchoMimicV3FlashEngine, avatar: EchoMimicV3AvatarData):
        super().__init__(opt)
        self.engine = model
        if getattr(opt, "echomimicv3_prompt", "") and opt.echomimicv3_prompt != "A person is speaking.":
            avatar.prompt = opt.echomimicv3_prompt
        self.avatar = avatar
        self.frame_list_cycle = avatar.idle_frames
        self._audio_jobs = queue.Queue()
        self._playback_frames = queue.Queue(maxsize=256)
        self._pending_audio = []
        self._pending_start_event = {}
        self._pending_lock = Lock()
        self._idle_index = 0
        self._state = "IDLE_STATIC"
        self._output_frame_size = self._resolve_output_frame_size()
        self.frame_list_cycle = [self._normalize_frame(frame) for frame in self.frame_list_cycle]
        self._idle_breath_enabled = bool(getattr(self.opt, "echomimicv3_idle_breath_enabled", True))
        self._idle_breath_cycle_seconds = float(getattr(self.opt, "echomimicv3_idle_breath_cycle_seconds", 4.8))
        self._idle_breath_scale = float(getattr(self.opt, "echomimicv3_idle_breath_scale", 0.0035))
        self._idle_breath_shift = float(getattr(self.opt, "echomimicv3_idle_breath_shift", 0.002))
        self._idle_breath_sway = float(getattr(self.opt, "echomimicv3_idle_breath_sway", 0.0))
        self._idle_render_index = 0

    def _resolve_output_frame_size(self):
        sample_size = getattr(self.opt, "echomimicv3_sample_size", None) or []
        if len(sample_size) >= 2:
            sample_h, sample_w = int(sample_size[0]), int(sample_size[1])
            ref_frame = cv2.imread(self.avatar.ref_image_path)
            if ref_frame is not None and sample_h > 0 and sample_w > 0:
                ref_h, ref_w = ref_frame.shape[:2]
                ref_area = ref_w * ref_h
                target_area = sample_w * sample_h
                if target_area < ref_area:
                    ratio = math.sqrt(ref_area / target_area)
                    width = int(ref_w / ratio // 16 * 16)
                    height = int(ref_h / ratio // 16 * 16)
                else:
                    width = int(ref_w // 16 * 16)
                    height = int(ref_h // 16 * 16)
                if width > 0 and height > 0:
                    return width, height
        first_frame = self.frame_list_cycle[0]
        height, width = first_frame.shape[:2]
        return width, height

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame)
        target_w, target_h = self._output_frame_size
        if frame.shape[:2] != (target_h, target_w):
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        if frame.dtype != np.uint8:
            frame = frame.clip(0, 255).astype(np.uint8)
        return frame

    def _apply_idle_breath(self, frame: np.ndarray) -> np.ndarray:
        if (
            not self._idle_breath_enabled
            or self._idle_breath_cycle_seconds <= 0
            or (
                abs(self._idle_breath_scale) <= 1e-6
                and abs(self._idle_breath_shift) <= 1e-6
                and abs(self._idle_breath_sway) <= 1e-6
            )
        ):
            return frame

        height, width = frame.shape[:2]
        cycle_frames = max(1.0, self._idle_breath_cycle_seconds * float(self.opt.fps))
        phase = 2.0 * math.pi * (self._idle_render_index % cycle_frames) / cycle_frames
        wave = 0.5 - 0.5 * math.cos(phase)
        sway = math.sin(phase)
        scale_x = 1.0 + self._idle_breath_scale * 0.35 * wave
        scale_y = 1.0 + self._idle_breath_scale * wave
        shift_x = width * self._idle_breath_sway * sway
        shift_y = -height * self._idle_breath_shift * wave
        center_x = width * 0.5
        center_y = height * 0.72
        matrix = np.array(
            [
                [scale_x, 0.0, center_x - scale_x * center_x],
                [0.0, scale_y, center_y - scale_y * center_y],
            ],
            dtype=np.float32,
        )
        matrix[0, 2] += shift_x
        matrix[1, 2] += shift_y
        return cv2.warpAffine(
            frame,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def put_audio_frame(self, audio_chunk: np.ndarray, datainfo: dict = None):
        datainfo = dict(datainfo or {})
        self._capture_choice_audio(audio_chunk, datainfo)
        chunk = np.asarray(audio_chunk, dtype=np.float32)
        status = datainfo.get("status")

        with self._pending_lock:
            if status == "start":
                self._pending_audio = []
                self._pending_start_event = dict(datainfo)
            if chunk.size > 0 and not np.allclose(chunk, 0.0):
                self._pending_audio.append(chunk.copy())
            if status == "end":
                if self._pending_audio:
                    audio = np.concatenate(self._pending_audio).astype(np.float32, copy=False)
                    self._enqueue_audio_jobs(audio, self._pending_start_event, datainfo)
                self._pending_audio = []
                self._pending_start_event = {}

    def _enqueue_audio_jobs(self, audio: np.ndarray, start_event: dict, end_event: dict):
        token = self.current_playback_token()
        speech_text = (start_event.get("text") or end_event.get("text") or "").strip()
        prompt_payload = self._build_generation_prompts(speech_text, start_event, end_event)
        job = EchoMimicV3AudioJob(
            audio=audio.copy(),
            start_event=dict(start_event),
            end_event=dict(end_event),
            token=token,
            speech_text=speech_text,
            prompt=prompt_payload["prompt"],
            negative_prompt=prompt_payload["negative_prompt"],
        )
        logger.info("EchoMimicV3 enqueue utterance, samples=%d", audio.shape[0])
        self._audio_jobs.put(job)

    def _build_generation_prompts(self, speech_text: str, start_event: dict, end_event: dict):
        prompt_override = start_event.get("prompt") or end_event.get("prompt")
        negative_override = start_event.get("negative_prompt") or end_event.get("negative_prompt")
        if prompt_override or negative_override:
            return {
                "prompt": prompt_override or self.avatar.prompt,
                "negative_prompt": negative_override or self.avatar.negative_prompt,
            }

        scene = start_event.get("scene") or end_event.get("scene") or ""
        action = start_event.get("action") or end_event.get("action") or ""
        payload = generate_echomimicv3_prompts(
            avatar_name=self.avatar.avatar_id,
            avatar_description=self.avatar.description,
            speech_text=speech_text,
            scene=scene,
            action=action,
        )
        return {
            "prompt": payload.get("prompt") or self.avatar.prompt,
            "negative_prompt": payload.get("negative_prompt") or self.avatar.negative_prompt,
        }

    def build_choice_cache_prompts(self, speech_text: str, metadata: dict = None):
        metadata = metadata or {}
        return self._build_generation_prompts(
            speech_text,
            {
                "text": speech_text,
                "scene": metadata.get("scene", ""),
                "action": metadata.get("action", ""),
                "prompt": metadata.get("prompt", ""),
                "negative_prompt": metadata.get("negative_prompt", ""),
            },
            {},
        )

    def play_cached_video_segment(
        self,
        frames: np.ndarray,
        audio: np.ndarray,
        datainfo: dict = None,
        playback_token: int = None,
    ):
        datainfo = dict(datainfo or {})
        if playback_token is None:
            playback_token = self.current_playback_token()
        if playback_token != self.current_playback_token():
            return

        frames = np.asarray(frames)
        audio = np.asarray(audio, dtype=np.float32)
        if frames.shape[0] <= 0 or audio.size <= 0:
            return

        expected_audio_samples = frames.shape[0] * 2 * self.chunk
        if audio.shape[0] != expected_audio_samples:
            logger.info(
                "EchoMimicV3 cached playback align: frames=%d video_duration=%.2fs "
                "original_audio_samples=%d aligned_audio_samples=%d",
                frames.shape[0],
                frames.shape[0] / self.opt.fps,
                audio.shape[0],
                expected_audio_samples,
            )
        aligned_audio = np.zeros(expected_audio_samples, dtype=np.float32)
        copy_samples = min(audio.shape[0], expected_audio_samples)
        aligned_audio[:copy_samples] = audio[:copy_samples]

        logger.info(
            "EchoMimicV3 cached segment playback enqueue, frames=%d audio_samples=%d aligned_audio_samples=%d",
            frames.shape[0],
            audio.shape[0],
            expected_audio_samples,
        )
        total_audio = aligned_audio.shape[0]
        for index, frame in enumerate(frames):
            if playback_token != self.current_playback_token():
                return
            chunks = []
            for sub in range(2):
                start = (index * 2 + sub) * self.chunk
                end = start + self.chunk
                chunk = np.zeros(self.chunk, dtype=np.float32)
                if start < total_audio:
                    source = aligned_audio[start:min(end, total_audio)]
                    chunk[:source.shape[0]] = source
                event = {}
                if index == 0 and sub == 0:
                    event.update(datainfo)
                    event["status"] = "start"
                if index == frames.shape[0] - 1 and sub == 1:
                    event.update(datainfo)
                    event["status"] = "end"
                chunks.append((chunk, event))
            self._playback_frames.put((self._normalize_frame(frame).copy(), chunks, playback_token))

    def flush_talk(self):
        super().flush_talk()
        self._clear_queue(self._audio_jobs)
        self._clear_queue(self._playback_frames)
        with self._pending_lock:
            self._pending_audio = []
            self._pending_start_event = {}

    def _next_idle_frame(self):
        if len(self.frame_list_cycle) <= 1:
            return self.frame_list_cycle[0]
        frame = self.frame_list_cycle[self._idle_index % len(self.frame_list_cycle)]
        self._idle_index += 1
        return frame

    def _pad_audio_chunk(self, chunk: np.ndarray):
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.shape[0] == self.chunk:
            return chunk
        padded = np.zeros(self.chunk, dtype=np.float32)
        if chunk.shape[0] > 0:
            padded[:min(self.chunk, chunk.shape[0])] = chunk[:self.chunk]
        return padded

    def _next_custom_frame_and_audio(self):
        audiotype = self.custom_audiotype
        if audiotype <= 1 or self.custom_audio_index.get(audiotype) is None:
            return None

        img_cycle = self.custom_img_cycle.get(audiotype)
        if img_cycle:
            frame_idx = mirror_index(len(img_cycle), self.custom_index.get(audiotype, 0))
            frame = img_cycle[frame_idx]
            self.custom_index[audiotype] = self.custom_index.get(audiotype, 0) + 1
        else:
            frame = self._normalize_frame(self._next_idle_frame())

        audio_chunks = []
        for _ in range(2):
            if self.custom_audiotype == audiotype:
                audio_chunks.append((self._pad_audio_chunk(self.get_custom_audio_stream(audiotype)), {}))
            else:
                audio_chunks.append((np.zeros(self.chunk, dtype=np.float32), {}))
        return frame, audio_chunks

    def _generation_loop(self, quit_event: Event):
        while not quit_event.is_set():
            try:
                job = self._audio_jobs.get(block=True, timeout=0.2)
            except queue.Empty:
                continue
            if job.token != self.current_playback_token():
                continue
            try:
                self._state = "GENERATING"
                logger.info(
                    "EchoMimicV3 utterance generation start, audio_samples=%d, prompt=%s, negative_prompt=%s",
                    job.audio.shape[0],
                    job.prompt,
                    job.negative_prompt,
                )
                frames = self._generate_utterance_frames(job, quit_event)
                logger.info("EchoMimicV3 utterance generation done, frames=%d", len(frames))
                if job.token != self.current_playback_token():
                    continue
                self._enqueue_playback(frames, job)
            except Exception:
                logger.exception("EchoMimicV3 generation failed")
                self._clear_queue(self._audio_jobs)
                self._clear_queue(self._playback_frames)
                self._enqueue_static_audio_playback(job)
            finally:
                if self._state == "GENERATING":
                    self._state = "IDLE_STATIC"

    def _generate_utterance_frames(self, job: EchoMimicV3AudioJob, quit_event: Event):
        duration = job.audio.shape[0] / self.sample_rate
        target_frames = max(1, int(math.ceil(duration * self.opt.fps)))
        logger.info(
            "EchoMimicV3 generation mode=full_utterance, duration=%.2fs, target_frames=%d",
            duration,
            target_frames,
        )
        if quit_event.is_set() or job.token != self.current_playback_token():
            return []
        frames = self.engine.generate_frames(
            self.avatar.ref_image_path,
            job.audio,
            job.prompt,
            job.negative_prompt,
            max_video_length=target_frames,
        )
        return frames[:target_frames]

    def _generate_overlap_utterance_frames(
        self,
        job: EchoMimicV3AudioJob,
        quit_event: Event,
        target_frames: int,
        segment_frames: int,
    ):
        segment_samples = max(self.chunk * 2, int(segment_frames / self.opt.fps * self.sample_rate))
        overlap_samples = int(max(0.0, getattr(self.opt, "echomimicv3_overlap_seconds", 0.5)) * self.sample_rate)
        overlap_samples = min(overlap_samples, max(0, segment_samples // 2))
        step_samples = max(self.chunk * 2, segment_samples - overlap_samples)
        total_segments = max(1, math.ceil(max(1, job.audio.shape[0] - overlap_samples) / step_samples))
        transition_frames = int(max(0, getattr(self.opt, "echomimicv3_transition_frames", 5)))
        overlap_frames = int(round(overlap_samples / self.sample_rate * self.opt.fps))

        frames = []
        offset = 0
        segment_index = 0
        while offset < job.audio.shape[0] and not quit_event.is_set():
            if job.token != self.current_playback_token():
                return []

            segment = job.audio[offset:offset + segment_samples]
            if segment.shape[0] <= 0:
                break

            logger.info(
                "EchoMimicV3 overlap segment generation start %d/%d, samples=%d, offset=%d",
                segment_index + 1,
                total_segments,
                segment.shape[0],
                offset,
            )
            segment_frames_out = self.engine.generate_frames(
                self.avatar.ref_image_path,
                segment,
                job.prompt,
                job.negative_prompt,
                max_video_length=segment_frames,
            )
            logger.info(
                "EchoMimicV3 overlap segment generation done %d/%d, frames=%d",
                segment_index + 1,
                total_segments,
                len(segment_frames_out),
            )

            if frames and overlap_frames > 0:
                segment_frames_out = segment_frames_out[min(overlap_frames, len(segment_frames_out)):]

            if frames and transition_frames > 0:
                segment_frames_out = self._blend_segment_head(frames, segment_frames_out, transition_frames)

            frames.extend(segment_frames_out)
            if len(frames) >= target_frames:
                return frames[:target_frames]

            offset += step_samples
            segment_index += 1

        return frames[:target_frames]

    @staticmethod
    def _blend_segment_head(previous_frames, next_frames, transition_frames: int):
        if not previous_frames or not next_frames or transition_frames <= 0:
            return next_frames

        blend_count = min(transition_frames, len(previous_frames), len(next_frames))
        if blend_count <= 0:
            return next_frames

        blended_next = list(next_frames)
        for i in range(blend_count):
            alpha = (i + 1) / (blend_count + 1)
            prev = previous_frames[-blend_count + i].astype(np.float32)
            nxt = blended_next[i].astype(np.float32)
            blended_next[i] = cv2.addWeighted(prev, 1.0 - alpha, nxt, alpha, 0).astype(np.uint8)
        return blended_next

    def _enqueue_playback(self, frames, job: EchoMimicV3AudioJob):
        if not frames:
            return
        expected_audio_samples = len(frames) * 2 * self.chunk
        if job.audio.shape[0] != expected_audio_samples:
            logger.info(
                "EchoMimicV3 playback align: generated_frames=%d video_duration=%.2fs "
                "original_audio_samples=%d aligned_audio_samples=%d",
                len(frames),
                len(frames) / self.opt.fps,
                job.audio.shape[0],
                expected_audio_samples,
            )
        audio = np.zeros(expected_audio_samples, dtype=np.float32)
        copy_samples = min(job.audio.shape[0], expected_audio_samples)
        if copy_samples > 0:
            audio[:copy_samples] = job.audio[:copy_samples]
        total_audio = audio.shape[0]
        for index, frame in enumerate(frames):
            if job.token != self.current_playback_token():
                return
            chunks = []
            for sub in range(2):
                start = (index * 2 + sub) * self.chunk
                end = start + self.chunk
                chunk = np.zeros(self.chunk, dtype=np.float32)
                if start < total_audio:
                    source = audio[start:min(end, total_audio)]
                    chunk[:source.shape[0]] = source
                event = {}
                if index == 0 and sub == 0:
                    event.update(job.start_event)
                if index == len(frames) - 1 and sub == 1:
                    event.update(job.end_event)
                chunks.append((chunk, event))
            self._playback_frames.put((self._normalize_frame(frame), chunks, job.token))

    def _enqueue_static_audio_playback(self, job: EchoMimicV3AudioJob):
        if job.token != self.current_playback_token() or job.audio.size == 0:
            return
        fallback_frames = max(1, int(math.ceil(job.audio.shape[0] / self.sample_rate * self.opt.fps)))
        idle_frame = self._next_idle_frame()
        logger.warning(
            "EchoMimicV3 falling back to static-frame audio playback, frames=%d audio_samples=%d",
            fallback_frames,
            job.audio.shape[0],
        )
        self._enqueue_playback([idle_frame] * fallback_frames, job)

    def render(self, quit_event):
        self.quit_event = quit_event
        self.init_customindex()
        self.tts.render(quit_event)
        self.output.start()

        generation_quit_event = Event()
        generation_thread = Thread(target=self._generation_loop, args=(generation_quit_event,))
        generation_thread.start()

        frame_interval = 1.0 / self.opt.fps
        try:
            while not quit_event.is_set():
                start = time.perf_counter()
                custom_payload = self._next_custom_frame_and_audio()
                idle_breath_frame = False
                if custom_payload is not None:
                    frame, audio_chunks = custom_payload
                    frame = self._normalize_frame(frame)
                    self.speaking = False
                else:
                    try:
                        frame, audio_chunks, token = self._playback_frames.get_nowait()
                        if token != self.current_playback_token():
                            continue
                        self._state = "PLAYING_GENERATED"
                        self.speaking = True
                    except queue.Empty:
                        if self._state == "PLAYING_GENERATED":
                            self._state = "RETURNING_STATIC"
                        frame = self._next_idle_frame()
                        audio_chunks = [
                            (np.zeros(self.chunk, dtype=np.float32), {}),
                            (np.zeros(self.chunk, dtype=np.float32), {}),
                        ]
                        self._state = "IDLE_STATIC"
                        self.speaking = False
                        idle_breath_frame = True

                frame = self._normalize_frame(frame).copy()
                if idle_breath_frame:
                    frame = self._apply_idle_breath(frame)
                    self._idle_render_index += 1
                cv2.putText(frame, "LiveTalking", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 128, 128), 1)
                self.output.push_video_frame(frame)
                self.record_video_data(frame)

                for audio_chunk, eventpoint in audio_chunks:
                    pcm = (audio_chunk * 32767).astype(np.int16)
                    self.output.push_audio_frame(pcm, eventpoint)
                    self.record_audio_data(pcm)

                buffer_size = self.output.get_buffer_size() if hasattr(self.output, "get_buffer_size") else 0
                elapsed = time.perf_counter() - start
                sleep_time = max(0.0, frame_interval - elapsed)
                if buffer_size >= 5:
                    sleep_time += 0.04 * buffer_size * 0.8
                time.sleep(sleep_time)
        finally:
            generation_quit_event.set()
            generation_thread.join()
            self.output.stop()
            logger.info("EchoMimicV3 render thread stop")
