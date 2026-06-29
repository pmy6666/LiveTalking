#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
import time
import wave
from pathlib import Path

import requests
import yaml
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "model_params.yaml"

DEFAULT_TTS_PARAMS = PROJECT_ROOT / "gpt_sovits_official_materials" / "current_tts_params.json"
DEFAULT_CONTENT = PROJECT_ROOT / "docs" / "notes" / "content.txt"
DEFAULT_REF_ID = "DongQing_6s"
DEFAULT_REF_AUDIO = PROJECT_ROOT / "bilibili_downloads" / "DongQing_6s.wav"
DEFAULT_STAGE1_VIDEO = PROJECT_ROOT / "LatentSync_test" / "api_stage1.mp4"
DEFAULT_LATENTSYNC_SCRIPT = PROJECT_ROOT / "LatentSync_test" / "run_latentsync_dongqing_batch.py"
DEFAULT_PROMPT = "请生成一句自然的中文数字人口播文案，长度控制在二十八到三十二个汉字，朗读时长大约七秒，只输出这一句话。"


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_nested(config: dict, dotted_key: str, default=None):
    value = config
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def set_nested(config: dict, dotted_key: str, value) -> None:
    target = config
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    target[keys[-1]] = value


def load_prompt_text(content_path: Path, ref_id: str) -> str:
    text = content_path.read_text(encoding="utf-8")
    pattern = re.compile(rf"^{re.escape(ref_id)}\s*[:：]\s*[\"“](.*?)[\"”]\s*$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Cannot find transcript for {ref_id!r} in {content_path}")
    return match.group(1).strip()


def wav_info(path: Path) -> dict:
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        sample_rate = wav.getframerate()
        return {
            "channels": wav.getnchannels(),
            "sample_rate": sample_rate,
            "sample_width": wav.getsampwidth(),
            "frames": frames,
            "duration_seconds": frames / float(sample_rate),
        }


def append_wav_silence(path: Path, seconds: float) -> dict:
    if seconds <= 0:
        return wav_info(path)

    with wave.open(str(path), "rb") as source:
        params = source.getparams()
        audio = source.readframes(source.getnframes())

    silence_frames = int(round(params.framerate * seconds))
    silence = b"\x00" * silence_frames * params.nchannels * params.sampwidth
    with wave.open(str(path), "wb") as target:
        target.setparams(params)
        target.writeframes(audio)
        target.writeframes(silence)

    info = wav_info(path)
    info["tail_silence_seconds"] = seconds
    info["tail_silence_frames"] = silence_frames
    return info


def generate_one_sentence(args: argparse.Namespace) -> str:
    api_key = args.deepseek_api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required. Export it before running this script.")

    client = OpenAI(api_key=api_key, base_url=args.deepseek_base_url)
    completion = client.chat.completions.create(
        model=args.deepseek_model,
        messages=[
            {
                "role": "system",
                "content": "你是一个简洁、自然、适合中文数字人口播的助手。只输出一句中文，不要解释。",
            },
            {"role": "user", "content": args.prompt},
        ],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    text = (completion.choices[0].message.content or "").strip()
    text = re.sub(r"\s+", "", text)
    if not text:
        raise RuntimeError("DeepSeek returned an empty sentence.")
    return text


def request_tts(args: argparse.Namespace, sentence: str, ref_audio: Path, prompt_text: str, out_path: Path) -> dict:
    params = load_json(args.tts_params)
    payload = {
        "text": sentence,
        "text_lang": args.tts_text_lang or params.get("text_lang", "zh"),
        "ref_audio_path": str(ref_audio.resolve()),
        "prompt_text": prompt_text,
        "prompt_lang": args.tts_prompt_lang or params.get("prompt_lang", "zh"),
        "text_split_method": args.tts_split_method or params.get("split_method", "cut5"),
        "media_type": args.tts_media_type or params.get("media_type", "wav"),
        "streaming_mode": args.tts_streaming_mode
        if args.tts_streaming_mode is not None
        else params.get("streaming_mode", 0),
        "speed_factor": args.tts_speed_factor
        if args.tts_speed_factor is not None
        else params.get("speed_factor", 1.08),
        "fragment_interval": args.tts_fragment_interval
        if args.tts_fragment_interval is not None
        else params.get("fragment_interval", 0.1),
    }

    response = requests.post(
        f"{args.tts_server.rstrip('/')}/tts",
        json=payload,
        timeout=args.tts_timeout,
    )
    if response.status_code != 200:
        raise RuntimeError(f"GPT-SoVITs request failed: {response.status_code} {response.text[:500]}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(response.content)
    result = {
        "server": args.tts_server,
        "params_file": str(args.tts_params.resolve()),
        "payload": payload,
        "output": str(out_path.resolve()),
        "bytes": len(response.content),
    }
    if payload["media_type"] == "wav":
        result["audio_info"] = append_wav_silence(out_path, args.tail_silence_seconds)
    return result


def run_latentsync(args: argparse.Namespace, audio_path: Path, out_dir: Path) -> dict:
    cmd = [
        sys.executable,
        str(args.latentsync_script),
        "--video",
        str(args.stage1_video),
        "--audio-dir",
        str(audio_path.parent),
        "--output-dir",
        str(out_dir),
        "--only",
        audio_path.stem,
        "--inference-steps",
        str(args.inference_steps),
        "--guidance-scale",
        str(args.guidance_scale),
        "--seed",
        str(args.seed),
    ]
    cmd.append("--enable-deepcache" if args.enable_deepcache else "--disable-deepcache")
    if args.skip_latentsync_dep_check:
        cmd.append("--skip-dep-check")

    start = time.time()
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    elapsed = time.time() - start

    output_video = out_dir / f"{audio_path.stem}.mp4"
    manifest = out_dir / "manifest.json"
    return {
        "command": cmd,
        "output_video": str(output_video.resolve()),
        "manifest": str(manifest.resolve()),
        "elapsed_seconds": round(elapsed, 3),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepSeek -> GPT-SoVITs -> LatentSync one-sentence end-to-end pipeline."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--deepseek-api-key", default="")
    parser.add_argument("--tts-speed-factor", type=float, default=None)
    parser.add_argument("--inference-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def build_runtime_args(cli_args: argparse.Namespace) -> argparse.Namespace:
    config = load_yaml(cli_args.config)
    if cli_args.prompt is not None:
        set_nested(config, "llm.prompt", cli_args.prompt)
    if cli_args.tts_speed_factor is not None:
        set_nested(config, "tts.speed_factor", cli_args.tts_speed_factor)
    if cli_args.inference_steps is not None:
        set_nested(config, "latentsync.inference_steps", cli_args.inference_steps)
    if cli_args.guidance_scale is not None:
        set_nested(config, "latentsync.guidance_scale", cli_args.guidance_scale)
    if cli_args.out_dir is not None:
        set_nested(config, "output.out_dir", str(cli_args.out_dir))

    return argparse.Namespace(
        config=cli_args.config,
        effective_config=config,
        prompt=get_nested(config, "llm.prompt", DEFAULT_PROMPT),
        deepseek_api_key=cli_args.deepseek_api_key,
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", get_nested(config, "llm.base_url", "https://api.deepseek.com")),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", get_nested(config, "llm.model", "deepseek-chat")),
        temperature=float(get_nested(config, "llm.temperature", 0.7)),
        max_tokens=int(get_nested(config, "llm.max_tokens", 80)),
        tts_server=os.getenv("GPT_SOVITS_SERVER", get_nested(config, "tts.server", "http://127.0.0.1:9880")),
        tts_params=project_path(get_nested(config, "tts.params_file", DEFAULT_TTS_PARAMS)),
        tts_timeout=int(get_nested(config, "tts.timeout", 180)),
        ref_id=get_nested(config, "tts.ref_id", DEFAULT_REF_ID),
        ref_audio=project_path(get_nested(config, "tts.ref_audio", DEFAULT_REF_AUDIO)),
        content=project_path(get_nested(config, "tts.transcript_file", DEFAULT_CONTENT)),
        tail_silence_seconds=float(get_nested(config, "tts.tail_silence_seconds", 1.0)),
        tts_speed_factor=float(get_nested(config, "tts.speed_factor", 1.08)),
        tts_fragment_interval=float(get_nested(config, "tts.fragment_interval", 0.1)),
        tts_streaming_mode=int(get_nested(config, "tts.streaming_mode", 0)),
        tts_media_type=get_nested(config, "tts.media_type", "wav"),
        tts_text_lang=get_nested(config, "tts.text_lang", "zh"),
        tts_prompt_lang=get_nested(config, "tts.prompt_lang", "zh"),
        tts_split_method=get_nested(config, "tts.split_method", "cut5"),
        stage1_video=project_path(get_nested(config, "latentsync.stage1_video", DEFAULT_STAGE1_VIDEO)),
        latentsync_script=project_path(get_nested(config, "latentsync.script", DEFAULT_LATENTSYNC_SCRIPT)),
        inference_steps=int(get_nested(config, "latentsync.inference_steps", 30)),
        guidance_scale=float(get_nested(config, "latentsync.guidance_scale", 2.2)),
        seed=int(get_nested(config, "latentsync.seed", 1247)),
        enable_deepcache=bool(get_nested(config, "latentsync.enable_deepcache", True)),
        skip_latentsync_dep_check=bool(get_nested(config, "latentsync.skip_dep_check", False)),
        out_dir=project_path(get_nested(config, "output.out_dir", SCRIPT_DIR / "outputs")),
    )


def main() -> int:
    args = build_runtime_args(parse_args())
    require_path(args.tts_params, "GPT-SoVITs params file")
    require_path(args.content, "reference transcript file")
    require_path(args.ref_audio, "TTS reference audio")
    require_path(args.stage1_video, "LatentSync stage1 video")
    require_path(args.latentsync_script, "LatentSync runner")

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir / run_id
    audio_dir = run_dir / "audio"
    video_dir = run_dir / "video"
    audio_path = audio_dir / "llm_tts.wav"
    timing = {}

    print("[1/3] DeepSeek generating one sentence...", flush=True)
    total_start = time.perf_counter()
    llm_start = total_start
    sentence = generate_one_sentence(args)
    timing["llm_seconds"] = round(time.perf_counter() - llm_start, 3)
    (run_dir / "llm_sentence.txt").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "llm_sentence.txt").write_text(sentence + "\n", encoding="utf-8")
    print(f"sentence: {sentence}", flush=True)
    print(f"llm_seconds: {timing['llm_seconds']}", flush=True)

    print("[2/3] GPT-SoVITs generating wav...", flush=True)
    prompt_text = load_prompt_text(args.content, args.ref_id)
    tts_start = time.perf_counter()
    tts_result = request_tts(args, sentence, args.ref_audio, prompt_text, audio_path)
    timing["tts_seconds"] = round(time.perf_counter() - tts_start, 3)
    print(f"audio: {audio_path}", flush=True)
    print(f"tts_seconds: {timing['tts_seconds']}", flush=True)

    print("[3/3] LatentSync generating video...", flush=True)
    latentsync_start = time.perf_counter()
    latentsync_result = run_latentsync(args, audio_path, video_dir)
    timing["latentsync_seconds"] = round(time.perf_counter() - latentsync_start, 3)
    timing["total_api_to_video_seconds"] = round(time.perf_counter() - total_start, 3)

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": str(args.config.resolve()),
        "run_dir": str(run_dir.resolve()),
        "timing": timing,
        "llm": {
            "base_url": args.deepseek_base_url,
            "model": args.deepseek_model,
            "prompt": args.prompt,
            "sentence": sentence,
        },
        "tts": tts_result,
        "latentsync": latentsync_result,
        "effective_config": args.effective_config,
    }
    manifest_path = run_dir / "pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("done")
    print(f"latentsync_seconds: {timing['latentsync_seconds']}")
    print(f"total_api_to_video_seconds: {timing['total_api_to_video_seconds']}")
    print(f"manifest: {manifest_path}")
    print(f"video: {latentsync_result['output_video']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
