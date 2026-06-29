#!/usr/bin/env python3
import argparse
import json
import time
import wave
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = PROJECT_ROOT / "test" / "ck_time" / "tts_batch_size_2_dongqing_20char"
DEFAULT_REF_AUDIO = PROJECT_ROOT / "bilibili_downloads" / "DongQing_6s.wav"
DEFAULT_PROMPT_TEXT = "那种快乐常常像一场梦，电影陪伴我们长大"

SAMPLES = [
    {
        "id": "dongqing_batch2_01",
        "text": "今天上午阳光很好，我们一起去公园慢慢散步吧。",
    },
    {
        "id": "dongqing_batch2_02",
        "text": "下午会议结束以后，请把今天测试结果整理清楚。",
    },
]


def wav_info(path: Path) -> dict:
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        sample_rate = wav.getframerate()
        return {
            "channels": wav.getnchannels(),
            "sample_rate": sample_rate,
            "sample_width": wav.getsampwidth(),
            "frames": frames,
            "duration_seconds": round(frames / float(sample_rate), 3),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DongQing TTS audios with request batch_size=2.")
    parser.add_argument("--server", default="http://127.0.0.1:9880")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--speed-factor", type=float, default=1.08)
    parser.add_argument("--fragment-interval", type=float, default=0.1)
    parser.add_argument("--text-split-method", default="cut5")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--ref-audio", default=str(DEFAULT_REF_AUDIO))
    parser.add_argument("--prompt-text", default=DEFAULT_PROMPT_TEXT)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def synthesize_one(args: argparse.Namespace, sample: dict, out_dir: Path, ref_audio: Path) -> dict:
    payload = {
        "text": sample["text"],
        "text_lang": "zh",
        "ref_audio_path": str(ref_audio),
        "prompt_text": args.prompt_text,
        "prompt_lang": "zh",
        "text_split_method": args.text_split_method,
        "batch_size": int(args.batch_size),
        "media_type": "wav",
        "streaming_mode": 0,
        "speed_factor": float(args.speed_factor),
        "fragment_interval": float(args.fragment_interval),
    }

    print(f"[dongqing-batch2] synthesize start: id={sample['id']} batch_size={args.batch_size}", flush=True)
    started_at = time.perf_counter()
    response = requests.post(f"{args.server.rstrip('/')}/tts", json=payload, timeout=args.timeout)
    elapsed = time.perf_counter() - started_at
    if response.status_code != 200:
        raise RuntimeError(f"GPT-SoVITS failed for {sample['id']}: {response.status_code} {response.text[:500]}")

    out_path = out_dir / f"{sample['id']}.wav"
    out_path.write_bytes(response.content)
    audio_info = wav_info(out_path)
    print(
        f"[dongqing-batch2] synthesize done: id={sample['id']} seconds={elapsed:.3f} "
        f"duration={audio_info['duration_seconds']:.3f}s wav={out_path}",
        flush=True,
    )

    return {
        "id": sample["id"],
        "text": sample["text"],
        "ref_audio": str(ref_audio),
        "prompt_text": args.prompt_text,
        "batch_size": payload["batch_size"],
        "seconds": round(elapsed, 3),
        "bytes": len(response.content),
        "wav_path": str(out_path),
        "audio_info": audio_info,
        "audio_duration_seconds": audio_info["duration_seconds"],
    }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_audio = Path(args.ref_audio).expanduser()
    if not ref_audio.is_absolute():
        ref_audio = PROJECT_ROOT / ref_audio
    if not ref_audio.exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_audio}")

    total_started_at = time.perf_counter()
    results = [synthesize_one(args, sample, out_dir, ref_audio) for sample in SAMPLES]
    total_seconds = round(time.perf_counter() - total_started_at, 3)

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "server": args.server,
        "batch_size": args.batch_size,
        "batch_semantics": "two sequential /tts requests; each request uses GPT-SoVITS batch_size=2",
        "speed_factor": args.speed_factor,
        "fragment_interval": args.fragment_interval,
        "text_split_method": args.text_split_method,
        "ref_audio": str(ref_audio),
        "prompt_text": args.prompt_text,
        "total_seconds": total_seconds,
        "items": results,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[dongqing-batch2] total_seconds={total_seconds:.3f}", flush=True)
    print(f"[dongqing-batch2] manifest={manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
