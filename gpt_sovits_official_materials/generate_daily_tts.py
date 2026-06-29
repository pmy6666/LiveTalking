#!/usr/bin/env python3
import argparse
import json
import time
import wave
from pathlib import Path

import requests


SCRIPT_DIR = Path(__file__).resolve().parent


def load_texts(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError(f"text file must contain a JSON list: {path}")
    items = []
    for index, item in enumerate(payload, start=1):
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        item_id = str(item.get("id") or f"text_{index:02d}").strip()
        items.append({"id": item_id, "text": text})
    return items


def wav_info(path: Path) -> dict:
    with wave.open(str(path), "rb") as wav:
        return {
            "channels": wav.getnchannels(),
            "sample_rate": wav.getframerate(),
            "sample_width": wav.getsampwidth(),
            "frames": wav.getnframes(),
            "duration_seconds": wav.getnframes() / float(wav.getframerate()),
        }


def request_tts(server: str, payload: dict, timeout: int) -> bytes:
    response = requests.post(f"{server.rstrip('/')}/tts", json=payload, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(f"GPT-SoVITS request failed: {response.status_code} {response.text[:500]}")
    return response.content


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate daily-life TTS samples with GPT-SoVITS.")
    parser.add_argument("--server", default="http://127.0.0.1:9880", help="GPT-SoVITS api_v2 server URL")
    parser.add_argument("--texts", default=str(SCRIPT_DIR / "daily_texts_zh.json"), help="JSON text list")
    parser.add_argument(
        "--ref_audio",
        default=str(SCRIPT_DIR / "reference_clips" / "official_ref_0s_8s.wav"),
        help="Reference audio path. Run make_reference_clip.sh first, or pass another clean clip.",
    )
    parser.add_argument(
        "--prompt_text",
        default="",
        help="Transcript for the reference audio. Accuracy matters for GPT-SoVITS clone quality.",
    )
    parser.add_argument("--text_lang", default="zh")
    parser.add_argument("--prompt_lang", default="zh")
    parser.add_argument("--split_method", default="cut5")
    parser.add_argument("--media_type", default="wav", choices=["wav", "raw", "ogg", "aac"])
    parser.add_argument("--streaming_mode", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument(
        "--speed_factor",
        type=float,
        default=1.0,
        help="Speech speed. Values above 1.0 are faster, for example 1.08 or 1.15.",
    )
    parser.add_argument(
        "--fragment_interval",
        type=float,
        default=0.08,
        help="Silence interval between generated fragments, in seconds. GPT-SoVITS default is usually 0.3.",
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--out_dir", default=str(SCRIPT_DIR / "generated_daily_tts"))
    args = parser.parse_args()

    texts_path = Path(args.texts)
    ref_audio = Path(args.ref_audio)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ref_audio.exists():
        raise FileNotFoundError(
            f"reference audio not found: {ref_audio}. "
            "Run ./make_reference_clip.sh first or pass --ref_audio."
        )
    if not args.prompt_text:
        print("[warn] --prompt_text is empty. This is usable for smoke tests, but voice quality may be unstable.")

    texts = load_texts(texts_path)
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "server": args.server,
        "ref_audio": str(ref_audio),
        "ref_audio_info": wav_info(ref_audio),
        "prompt_text": args.prompt_text,
        "text_lang": args.text_lang,
        "prompt_lang": args.prompt_lang,
        "split_method": args.split_method,
        "media_type": args.media_type,
        "streaming_mode": args.streaming_mode,
        "speed_factor": args.speed_factor,
        "fragment_interval": args.fragment_interval,
        "warnings": [],
        "items": [],
    }
    if not args.prompt_text:
        manifest["warnings"].append(
            "prompt_text is empty; GPT-SoVITS zero-shot voice quality is expected to be poor or unstable."
        )

    for index, item in enumerate(texts, start=1):
        output_path = out_dir / f"{index:02d}_{item['id']}.{args.media_type}"
        payload = {
            "text": item["text"],
            "text_lang": args.text_lang,
            "ref_audio_path": str(ref_audio),
            "prompt_text": args.prompt_text,
            "prompt_lang": args.prompt_lang,
            "text_split_method": args.split_method,
            "media_type": args.media_type,
            "streaming_mode": args.streaming_mode,
            "speed_factor": args.speed_factor,
            "fragment_interval": args.fragment_interval,
        }
        print(f"[{index}/{len(texts)}] generating {output_path.name}")
        audio_bytes = request_tts(args.server, payload, args.timeout)
        output_path.write_bytes(audio_bytes)
        entry = {
            "id": item["id"],
            "text": item["text"],
            "output": str(output_path),
            "bytes": len(audio_bytes),
        }
        if args.media_type == "wav":
            try:
                entry["audio_info"] = wav_info(output_path)
            except wave.Error as exc:
                entry["audio_info_error"] = str(exc)
        manifest["items"].append(entry)

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    print(f"done: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
