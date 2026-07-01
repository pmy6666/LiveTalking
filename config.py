###############################################################################
#  配置解析 — CLI 参数 + YAML 配置
###############################################################################

import argparse
import json
import os
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent


def str_or_int(value):
    """尝试转换为 int，失败则返回 str"""
    try:
        return int(value)
    except ValueError:
        return value


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="LiveTalking Digital Human Server")

    # ─── 音频 ──────────────────────────────────────────────────────────
    parser.add_argument('--fps', type=int, default=25, help="video fps, must be 25")
    parser.add_argument('-l', type=int, default=10)
    parser.add_argument('-m', type=int, default=8)
    parser.add_argument('-r', type=int, default=10)

    # ─── 画面 ──────────────────────────────────────────────────────────
    # parser.add_argument('--W', type=int, default=450, help="GUI width")
    # parser.add_argument('--H', type=int, default=450, help="GUI height")

    # ─── 数字人模型 ────────────────────────────────────────────────────
    parser.add_argument('--model', type=str, default='wav2lip',
                        help="avatar model: musetalk/wav2lip/ultralight/echomimicv3/cached_media")
    parser.add_argument('--avatar_id', type=str, default='wav2lip256_avatar1',
                        help="avatar id in data/avatars")
    parser.add_argument('--batch_size', type=int, default=16, help="infer batch")
    parser.add_argument('--modelres', type=int, default=192)
    parser.add_argument('--modelfile', type=str, default='')

    # ─── EchoMimicV3 ──────────────────────────────────────────────────
    parser.add_argument('--echomimicv3_repo', type=str, default='third_party/echomimic_v3',
                        help="EchoMimicV3 official code path")
    parser.add_argument('--echomimicv3_model_dir', type=str, default='EchoMimicV3',
                        help="EchoMimicV3 local model root")
    parser.add_argument('--echomimicv3_base_model_dir', type=str, default='',
                        help="Wan2.1-Fun-V1.1-1.3B-InP path; defaults under echomimicv3_model_dir")
    parser.add_argument('--echomimicv3_wav2vec_dir', type=str, default='',
                        help="chinese-wav2vec2-base path; defaults under echomimicv3_model_dir")
    parser.add_argument('--echomimicv3_transformer_path', type=str, default='',
                        help="EchoMimicV3 Flash transformer safetensors path")
    parser.add_argument('--echomimicv3_config_path', type=str, default='',
                        help="EchoMimicV3 config.yaml path; defaults under echomimicv3_repo/config")
    parser.add_argument('--echomimicv3_sample_size', type=int, nargs=2, default=[768, 768],
                        help="EchoMimicV3 generated frame size, e.g. 512 512 or 768 768")
    parser.add_argument('--echomimicv3_video_length', type=int, default=81,
                        help="Max frames per EchoMimicV3 segment")
    parser.add_argument('--echomimicv3_full_audio_threshold', type=float, default=3.2,
                        help="Use one-shot full-audio generation when audio duration is within this many seconds")
    parser.add_argument('--echomimicv3_segment_frames', type=int, default=81,
                        help="Frames per overlap generation window for longer EchoMimicV3 utterances")
    parser.add_argument('--echomimicv3_overlap_seconds', type=float, default=0.5,
                        help="Audio overlap seconds between long-utterance generation windows")
    parser.add_argument('--echomimicv3_transition_frames', type=int, default=5,
                        help="Frames blended at segment joins for smoother EchoMimicV3 long utterances")
    parser.add_argument('--echomimicv3_num_steps', type=int, default=8,
                        help="EchoMimicV3 diffusion steps")
    parser.add_argument('--echomimicv3_guidance_scale', type=float, default=6.0)
    parser.add_argument('--echomimicv3_audio_guidance_scale', type=float, default=3.0)
    parser.add_argument('--echomimicv3_teacache_threshold', type=float, default=0.1)
    parser.add_argument('--echomimicv3_gpu_memory_mode', type=str, default='model_cpu_offload',
                        choices=['none', 'model_cpu_offload', 'sequential_cpu_offload'],
                        help="EchoMimicV3 GPU memory saving mode; sequential_cpu_offload saves most VRAM but is slowest")
    parser.add_argument('--echomimicv3_teacache_offload', action='store_true',
                        help="Offload EchoMimicV3 TeaCache tensors to CPU to reduce VRAM")
    parser.add_argument('--echomimicv3_weight_dtype', type=str, default='bfloat16',
                        choices=['float16', 'bfloat16'])
    parser.add_argument('--echomimicv3_seed', type=int, default=43)
    parser.add_argument('--echomimicv3_prompt', type=str, default='A person is speaking.')

    # ─── 自定义动作和多形象 ────────────────────────────────────────────
    parser.add_argument('--customvideo_config', type=str, default='',
                        help="custom action json")
    parser.add_argument('--choice_config', type=str, default='scripts/two_stage_pre/config.yaml',
                        help="choice-mode Talking Head config yaml")

    # ─── TTS ───────────────────────────────────────────────────────────
    parser.add_argument('--tts', type=str, default='edgetts',
                        help="tts plugin: none/edgetts/gpt-sovits/cosyvoice/fishtts/tencent/doubao/indextts2/azuretts/qwentts")
    parser.add_argument('--REF_FILE', type=str, default="zh-CN-YunxiaNeural",
                        help="参考文件名或语音模型ID")
    parser.add_argument('--REF_TEXT', type=str, default=None)
    parser.add_argument('--TTS_SERVER', type=str, default='http://127.0.0.1:9880')
    parser.add_argument('--TTS_TEXT_LANG', type=str, default='zh',
                        help="tts text language, e.g. zh/en/ja/ko")
    parser.add_argument('--TTS_PROMPT_LANG', type=str, default='zh',
                        help="tts prompt/reference language")
    parser.add_argument('--TTS_MEDIA_TYPE', type=str, default='wav',
                        help="tts response media type, e.g. ogg/wav/aac/raw")
    parser.add_argument('--GPT_SOVITS_STREAMING_MODE', type=int, default=2,
                        help="GPT-SoVITS api_v2 streaming mode: 0/1/2/3")
    parser.add_argument('--TTS_SPLIT_METHOD', type=str, default='cut5',
                        help="text split method for GPT-SoVITS style APIs")
    parser.add_argument('--TTS_SPEED_FACTOR', type=float, default=1.08,
                        help="GPT-SoVITS speech speed. Values above 1.0 are faster.")
    parser.add_argument('--TTS_FRAGMENT_INTERVAL', type=float, default=0.1,
                        help="GPT-SoVITS silence interval between generated fragments, in seconds.")
    parser.add_argument('--GPT_SOVITS_GPT_MODEL', type=str, default='',
                        help="optional remote GPT-SoVITS GPT/T2S weights path, e.g. s1v3.ckpt")
    parser.add_argument('--GPT_SOVITS_SOVITS_MODEL', type=str, default='',
                        help="optional remote GPT-SoVITS SoVITS/VITS weights path, e.g. s2Gv2ProPlus.pth")
    parser.add_argument('--GPT_SOVITS_S2D_MODEL', type=str, default='',
                        help="optional s2D checkpoint path; kept for bookkeeping, not used by api_v2 inference")

    # ─── 传输 ─────────────────────────────────────────────────────────
    parser.add_argument('--transport', type=str, default='webrtc',
                        help="output: rtcpush/webrtc/rtmp/virtualcam")
    parser.add_argument('--push_url', type=str,
                        default='http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream')
    parser.add_argument('--max_session', type=int, default=1)
    parser.add_argument('--listenport', type=int, default=8010,
                        help="web listen port")

    opt = parser.parse_args()

    if opt.choice_config:
        apply_choice_config(opt, opt.choice_config)

    # ─── 后处理 ────────────────────────────────────────────────────────
    opt.customopt = []
    if opt.customvideo_config:
        with open(opt.customvideo_config, 'r') as f:
            opt.customopt = json.load(f)

    return opt


def _resolve_choice_path(config_path: Path, value: str):
    if not value:
        return value
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return str(path)

    candidates = [
        config_path.parent / path,
        Path.cwd() / path,
        PROJECT_ROOT / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str((PROJECT_ROOT / path).resolve())


def apply_choice_config(opt, config_file: str):
    config_path = Path(config_file).expanduser()
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    if not config_path.exists():
        return

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        return

    opt.choice_config = str(config_path.resolve())
    choice = config.get("choice", {}) or {}
    opt.choice_tree_id = choice.get("tree_id", "default_choice_tree")
    opt.choice_video_cache_mode = choice.get("video_cache_mode", "two_stage")
    opt.choice_cache_local_only = opt.choice_video_cache_mode == "local_cache_only"

    runtime = config.get("runtime", {}) or {}
    for key, value in runtime.items():
        setattr(opt, key, value)

    cache_demo = config.get("cache_demo", {}) or {}
    opt.cache_demo = cache_demo
    opt.cache_demo_enabled = bool(cache_demo.get("enabled", False))
    if opt.cache_demo_enabled:
        opt.cache_demo_mode = cache_demo.get("mode", "local_only")
        opt.cache_demo_primary_cache = cache_demo.get("primary_cache", "choice_echomimicv3")
        opt.cache_demo_fallback_video_cache = cache_demo.get("fallback_video_cache", "choice_two_stage")
        opt.cache_demo_fallback_audio_cache = cache_demo.get("fallback_audio_cache", "choice_audio_wav")
        opt.cache_demo_voice = cache_demo.get("voice", "female")
        opt.choice_cache_local_only = True

    tts = config.get("tts", {}) or {}
    tts_map = {
        "name": "tts",
        "server": "TTS_SERVER",
        "media_type": "TTS_MEDIA_TYPE",
        "streaming_mode": "GPT_SOVITS_STREAMING_MODE",
        "text_lang": "TTS_TEXT_LANG",
        "prompt_lang": "TTS_PROMPT_LANG",
        "split_method": "TTS_SPLIT_METHOD",
        "speed_factor": "TTS_SPEED_FACTOR",
        "fragment_interval": "TTS_FRAGMENT_INTERVAL",
    }
    for key, attr in tts_map.items():
        if key in tts:
            setattr(opt, attr, tts[key])

    echomimic = config.get("echomimicv3", {}) or {}
    echomimic_map = {
        "sample_size": "echomimicv3_sample_size",
        "num_inference_steps": "echomimicv3_num_steps",
        "guidance_scale": "echomimicv3_guidance_scale",
        "audio_guidance_scale": "echomimicv3_audio_guidance_scale",
        "teacache_threshold": "echomimicv3_teacache_threshold",
        "gpu_memory_mode": "echomimicv3_gpu_memory_mode",
        "weight_dtype": "echomimicv3_weight_dtype",
        "seed": "echomimicv3_seed",
        "prompt": "echomimicv3_prompt",
    }
    for key, attr in echomimic_map.items():
        if key in echomimic:
            setattr(opt, attr, echomimic[key])
    if "fps" in echomimic:
        opt.fps = int(echomimic["fps"])

    paths = config.get("paths", {}) or {}
    path_map = {
        "echomimicv3_repo": "echomimicv3_repo",
        "echomimicv3_model_dir": "echomimicv3_model_dir",
        "echomimicv3_base_model_dir": "echomimicv3_base_model_dir",
        "echomimicv3_wav2vec_dir": "echomimicv3_wav2vec_dir",
        "echomimicv3_transformer_path": "echomimicv3_transformer_path",
        "echomimicv3_config_path": "echomimicv3_config_path",
    }
    for key, attr in path_map.items():
        if key in paths:
            setattr(opt, attr, _resolve_choice_path(config_path, paths[key]))
