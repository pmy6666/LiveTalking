#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TTS_MANIFEST = PROJECT_ROOT / "test" / "ck_time" / "tts_internal_batch_size_2_dongqing_20char" / "manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize true internal batch_size=2 TTS + Talking Head manifests.")
    parser.add_argument("--tts-manifest", default=str(DEFAULT_TTS_MANIFEST))
    parser.add_argument("--talking-head-manifest", required=True)
    parser.add_argument("--out", default="")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> int:
    args = parse_args()
    tts_manifest = Path(args.tts_manifest).expanduser()
    talking_manifest = Path(args.talking_head_manifest).expanduser()
    if not tts_manifest.is_absolute():
        tts_manifest = PROJECT_ROOT / tts_manifest
    if not talking_manifest.is_absolute():
        talking_manifest = PROJECT_ROOT / talking_manifest

    tts = load_json(tts_manifest)
    talking = load_json(talking_manifest)

    if not tts.get("is_true_model_batch"):
        raise RuntimeError(f"TTS manifest is not marked as true model batch: {tts_manifest}")
    if not talking.get("is_true_model_batch"):
        raise RuntimeError(f"Talking Head manifest is not marked as true model batch: {talking_manifest}")

    tts_seconds = float(tts["total_seconds"])
    talking_seconds = float(talking["total_seconds"])
    total_seconds = round(tts_seconds + talking_seconds, 3)
    avg_seconds = round(total_seconds / 2.0, 3)

    out_path = Path(args.out).expanduser() if args.out else talking_manifest.parent / "internal_batch_size_2_summary.md"
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path

    lines = [
        "# TTS + Talking Head 模型内部 batch_size=2 推理时间结果",
        "",
        f"运行时间：{time.strftime('%Y-%m-%d %H:%M:%S %z')}",
        "",
        "## 结论",
        "",
        "本次结果是否为模型内部 batch_size=2：是",
        "",
        f"TTS + Talking Head 内部 batch_size=2 总推理时间：{total_seconds:.3f} 秒",
        "",
        f"平均每条结果耗时：{avg_seconds:.3f} 秒",
        "",
        "## 核心结果",
        "",
        "| 指标 | 秒 |",
        "|---|---:|",
        f"| TTS internal batch_size=2 inference | {tts_seconds:.3f} |",
        f"| Talking Head internal batch_size=2 inference | {talking_seconds:.3f} |",
        f"| TTS + Talking Head internal batch_size=2 | {total_seconds:.3f} |",
        f"| Average per result | {avg_seconds:.3f} |",
        "",
        "## 内部 batch 证据",
        "",
        "| 阶段 | 证据 |",
        "|---|---|",
        f"| TTS | `batch_size={tts.get('batch_size')}`, `is_true_model_batch={tts.get('is_true_model_batch')}` |",
        f"| Talking Head | `observed_sample_batch={talking.get('evidence', {}).get('observed_sample_batch')}`, `observed_unet_batch={talking.get('evidence', {}).get('observed_unet_batch')}` |",
        "",
        "## 输出",
        "",
        "| id | audio duration | video duration | audio output | video output |",
        "|---|---:|---:|---|---|",
    ]

    talking_by_id = {item["id"]: item for item in talking.get("items", [])}
    for item in tts.get("items", []):
        pair = talking_by_id.get(item["id"], {})
        lines.append(
            "| {id} | {audio:.3f} | {video:.3f} | `{audio_out}` | `{video_out}` |".format(
                id=item["id"],
                audio=float(item.get("audio_duration_seconds", 0.0)),
                video=float(pair.get("video_duration_seconds", 0.0)),
                audio_out=item.get("wav_path", ""),
                video_out=pair.get("output", ""),
            )
        )

    lines.extend(
        [
            "",
            "## Manifest",
            "",
            f"- TTS: `{tts_manifest}`",
            f"- Talking Head: `{talking_manifest}`",
            "",
        ]
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"summary: {out_path}")
    print(f"tts_plus_talking_head_internal_batch2_seconds: {total_seconds:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
