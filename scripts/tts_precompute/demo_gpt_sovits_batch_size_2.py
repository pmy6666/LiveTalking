#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = PROJECT_ROOT / "test" / "ck_time" / "tts_batch_size_2_demo"
DEFAULT_REFS_FILE = PROJECT_ROOT / "gpt_sovits_official_materials" / "voice_refs_bilibili.json"

DEMO_TEXTS = {
    "DongQing_6s": "清晨的阳光照进窗前，城市慢慢醒来。我们整理好心情，带着一点期待，继续走向新的故事。",
    "SaBeining": "今天的练习从一次小小的尝试开始。把参数固定下来，再观察耗时变化，结果就会更加清楚。",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a GPT-SoVITS batch_size=2 demo with two reference materials.")
    parser.add_argument("--server", default="http://127.0.0.1:9880")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--speed-factor", type=float, default=1.08)
    parser.add_argument("--fragment-interval", type=float, default=0.1)
    parser.add_argument("--text-split-method", default="cut5")
    parser.add_argument("--refs-file", default=str(DEFAULT_REFS_FILE))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def load_demo_refs(refs_file: str) -> list[dict]:
    refs_path = Path(refs_file).expanduser()
    if not refs_path.is_absolute():
        refs_path = PROJECT_ROOT / refs_path
    refs = json.loads(refs_path.read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in refs}
    missing = [ref_id for ref_id in DEMO_TEXTS if ref_id not in by_id]
    if missing:
        raise ValueError(f"Missing demo refs in {refs_path}: {missing}")
    return [by_id[ref_id] for ref_id in DEMO_TEXTS]


def synthesize_one(args: argparse.Namespace, sample: dict, out_dir: Path) -> dict:
    sample_id = sample["id"]
    ref_audio = Path(sample["ref_audio"]).expanduser()
    if not ref_audio.exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_audio}")

    payload = {
        "text": DEMO_TEXTS[sample_id],
        "text_lang": "zh",
        "ref_audio_path": str(ref_audio),
        "prompt_text": sample["prompt_text"],
        "prompt_lang": sample.get("prompt_lang", "zh"),
        "text_split_method": args.text_split_method,
        "batch_size": int(args.batch_size),
        "media_type": "wav",
        "streaming_mode": 0,
        "speed_factor": float(args.speed_factor),
        "fragment_interval": float(args.fragment_interval),
    }

    print(f"[batch2-demo] synthesize start: id={sample_id} batch_size={args.batch_size}", flush=True)
    started_at = time.perf_counter()
    response = requests.post(f"{args.server}/tts", json=payload, timeout=args.timeout)
    elapsed = time.perf_counter() - started_at
    if response.status_code != 200:
        raise RuntimeError(f"GPT-SoVITS failed for {sample_id}: {response.status_code} {response.text}")

    out_path = out_dir / f"{sample_id}_batch{args.batch_size}.wav"
    out_path.write_bytes(response.content)
    print(f"[batch2-demo] synthesize done: id={sample_id} seconds={elapsed:.3f} wav={out_path}", flush=True)

    return {
        "id": sample_id,
        "ref_audio": str(ref_audio),
        "prompt_text": sample["prompt_text"],
        "text": payload["text"],
        "batch_size": payload["batch_size"],
        "seconds": round(elapsed, 3),
        "bytes": len(response.content),
        "wav_path": str(out_path),
    }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total_started_at = time.perf_counter()
    for sample in load_demo_refs(args.refs_file):
        results.append(synthesize_one(args, sample, out_dir))

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "server": args.server,
        "batch_size": args.batch_size,
        "speed_factor": args.speed_factor,
        "fragment_interval": args.fragment_interval,
        "text_split_method": args.text_split_method,
        "total_seconds": round(time.perf_counter() - total_started_at, 3),
        "items": results,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[batch2-demo] manifest={manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
