#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def safe_name(value: str) -> str:
    value = value.strip() or "ref"
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "ref"


def load_refs(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        refs = json.load(file)
    if not isinstance(refs, list):
        raise ValueError(f"reference file must contain a JSON list: {path}")
    return refs


def main() -> int:
    parser = argparse.ArgumentParser(description="Run daily TTS generation for multiple bilibili reference voices.")
    parser.add_argument("--refs", default=str(SCRIPT_DIR / "voice_refs_bilibili.json"))
    parser.add_argument("--texts", default=str(SCRIPT_DIR / "daily_texts_zh.json"))
    parser.add_argument("--server", default="http://127.0.0.1:9880")
    parser.add_argument("--out_root", default=str(SCRIPT_DIR / "generated_bilibili_refs_tts"))
    parser.add_argument("--speed_factor", type=float, default=1.08)
    parser.add_argument("--fragment_interval", type=float, default=0.08)
    parser.add_argument("--media_type", default="wav", choices=["wav", "raw", "ogg", "aac"])
    parser.add_argument("--streaming_mode", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--only", default="", help="Comma-separated reference ids to run.")
    parser.add_argument("--continue_on_error", type=int, default=1)
    args = parser.parse_args()

    refs = load_refs(Path(args.refs))
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        refs = [ref for ref in refs if ref.get("id") in wanted]
    if not refs:
        raise RuntimeError("no references selected")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    batch_manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "server": args.server,
        "texts": args.texts,
        "refs": args.refs,
        "speed_factor": args.speed_factor,
        "fragment_interval": args.fragment_interval,
        "media_type": args.media_type,
        "streaming_mode": args.streaming_mode,
        "items": [],
    }

    generator = SCRIPT_DIR / "generate_daily_tts.py"
    for ref in refs:
        ref_id = safe_name(str(ref.get("id", "")))
        ref_audio = Path(str(ref.get("ref_audio", "")))
        prompt_text = str(ref.get("prompt_text", ""))
        prompt_lang = str(ref.get("prompt_lang", "zh") or "zh")
        out_dir = out_root / ref_id
        entry = {
            "id": ref_id,
            "ref_audio": str(ref_audio),
            "prompt_text": prompt_text,
            "out_dir": str(out_dir),
            "status": "pending",
        }

        if not ref_audio.exists():
            entry["status"] = "failed"
            entry["message"] = f"reference audio not found: {ref_audio}"
            batch_manifest["items"].append(entry)
            if not args.continue_on_error:
                break
            print(f"[skip] {ref_id}: {entry['message']}", file=sys.stderr)
            continue

        cmd = [
            sys.executable,
            str(generator),
            "--server",
            args.server,
            "--texts",
            args.texts,
            "--ref_audio",
            str(ref_audio),
            "--prompt_text",
            prompt_text,
            "--prompt_lang",
            prompt_lang,
            "--speed_factor",
            str(args.speed_factor),
            "--fragment_interval",
            str(args.fragment_interval),
            "--media_type",
            args.media_type,
            "--streaming_mode",
            str(args.streaming_mode),
            "--timeout",
            str(args.timeout),
            "--out_dir",
            str(out_dir),
        ]

        print(f"[ref] {ref_id} -> {out_dir}")
        started = time.perf_counter()
        result = subprocess.run(cmd, text=True)
        entry["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        entry["status"] = "ready" if result.returncode == 0 else "failed"
        entry["returncode"] = result.returncode
        batch_manifest["items"].append(entry)
        if result.returncode != 0 and not args.continue_on_error:
            break

    manifest_path = out_root / "batch_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(batch_manifest, file, ensure_ascii=False, indent=2)
    print(f"done: {manifest_path}")

    failed = [item for item in batch_manifest["items"] if item["status"] != "ready"]
    return 1 if failed and not args.continue_on_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
