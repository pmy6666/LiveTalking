#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path

import soundfile as sf
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GPT_SOVITS_ROOT = PROJECT_ROOT / "GPT-SoVITS"
HELPER_SCRIPT_DIR = PROJECT_ROOT / "scripts" / "tts_precompute"
DEFAULT_CONFIG = GPT_SOVITS_ROOT / "GPT_SoVITS" / "configs" / "tts_infer_livetalking_v2proplus.yaml"
DEFAULT_OUT_DIR = PROJECT_ROOT / "test" / "ck_time" / "tts_internal_batch_compare_dongqing_20char"
DEFAULT_REF_AUDIO = PROJECT_ROOT / "bilibili_downloads" / "DongQing_6s.wav"
DEFAULT_PROMPT_TEXT = "那种快乐常常像一场梦，电影陪伴我们长大"

SAMPLES = [
    {
        "id": "dongqing_01",
        "text": "今天上午阳光很好，我们一起去公园慢慢散步吧。",
    },
    {
        "id": "dongqing_02",
        "text": "下午会议结束以后，请把今天测试结果整理清楚。",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare GPT-SoVITS internal batch=2 vs batch=1 twice in one process.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--ref-audio", default=str(DEFAULT_REF_AUDIO))
    parser.add_argument("--prompt-text", default=DEFAULT_PROMPT_TEXT)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--speed-factor", type=float, default=1.0)
    parser.add_argument("--fragment-interval", type=float, default=0.1)
    parser.add_argument("--text-split-method", default="cut0")
    return parser.parse_args()


def tensor_shape(value) -> list[int] | str:
    if hasattr(value, "shape"):
        return list(value.shape)
    return type(value).__name__


def sync_if_cuda(device) -> None:
    if "cuda" in str(device):
        torch.cuda.synchronize()


def prepare_prompt_cache(tts, ref_audio: Path, prompt_text: str) -> float:
    started_at = time.perf_counter()
    tts.set_ref_audio(str(ref_audio))
    prompt_text = prompt_text.strip("\n")
    if prompt_text[-1] not in "。！？；：，,.!?;:":
        prompt_text += "。"
    phones, bert_features, norm_text = tts.text_preprocessor.segment_and_extract_feature_for_text(
        prompt_text, "zh", tts.configs.version
    )
    tts.prompt_cache["prompt_text"] = prompt_text
    tts.prompt_cache["prompt_lang"] = "zh"
    tts.prompt_cache["phones"] = phones
    tts.prompt_cache["bert_features"] = bert_features
    tts.prompt_cache["norm_text"] = norm_text
    sync_if_cuda(tts.configs.device)
    return round(time.perf_counter() - started_at, 3)


def get_reference_inputs(tts):
    refer_audio_spec = []
    sv_emb = [] if tts.is_v2pro else None
    for spec, audio_tensor in tts.prompt_cache["refer_spec"]:
        spec = spec.to(dtype=tts.precision, device=tts.configs.device)
        refer_audio_spec.append(spec)
        if tts.is_v2pro:
            sv_emb.append(tts.sv_model.compute_embedding3(audio_tensor))
    sync_if_cuda(tts.configs.device)
    return refer_audio_spec, sv_emb


def make_batches(tts, samples: list[dict], batch_size: int, text_split_method: str):
    combined_text = "\n".join(sample["text"] for sample in samples)
    data = tts.text_preprocessor.preprocess(combined_text, "zh", text_split_method, tts.configs.version)
    if len(data) != len(samples):
        raise RuntimeError(f"Expected {len(samples)} text samples after preprocessing, got {len(data)}")
    batches, batch_index_list = tts.to_batch(
        data,
        prompt_data=tts.prompt_cache,
        batch_size=batch_size,
        threshold=0.0,
        split_bucket=False,
        device=tts.configs.device,
        precision=tts.precision,
    )
    return batches, batch_index_list


def run_one_internal_batch(
    tts,
    batch,
    batch_index,
    refer_audio_spec,
    sv_emb,
    speed_factor: float,
    evidence: dict | None,
):
    from run_dongqing_internal_batch_size_2_20char import strict_vits_batch_decode

    batch_phones = batch["phones"]
    all_phoneme_ids = batch["all_phones"]
    all_phoneme_lens = batch["all_phones_len"]
    all_bert_features = batch["all_bert_features"]
    prompt = tts.prompt_cache["prompt_semantic"].expand(len(all_phoneme_ids), -1).to(tts.configs.device)

    if evidence is not None:
        evidence["semantic_model_calls"].append(
            {
                "method": "infer_panel_batch_infer",
                "batch_index": batch_index,
                "all_phoneme_ids_shape": tensor_shape(all_phoneme_ids),
                "all_phoneme_lens_shape": tensor_shape(all_phoneme_lens),
                "prompt_shape": tensor_shape(prompt),
                "all_bert_features_type": type(all_bert_features).__name__,
                "observed_batch_size": int(all_phoneme_lens.shape[0]),
            }
        )

    pred_semantic_list, idx_list = tts.t2s_model.model.infer_panel_batch_infer(
        all_phoneme_ids,
        all_phoneme_lens,
        prompt,
        all_bert_features,
        top_k=15,
        top_p=1,
        temperature=1,
        early_stop_num=tts.configs.hz * tts.configs.max_sec,
        max_len=batch["max_len"],
        repetition_penalty=1.35,
    )
    fragments = strict_vits_batch_decode(
        tts,
        pred_semantic_list,
        idx_list,
        batch_phones,
        refer_audio_spec,
        speed_factor,
        sv_emb,
        evidence,
    )
    sync_if_cuda(tts.configs.device)
    return fragments


def run_mode(
    tts,
    samples: list[dict],
    batch_size: int,
    text_split_method: str,
    refer_audio_spec,
    sv_emb,
    speed_factor: float,
    record_evidence: bool,
):
    evidence = None
    if record_evidence:
        evidence = {
            "requested_batch_size": batch_size,
            "semantic_model_calls": [],
            "vits_decode_calls": [],
            "captured_audio_batches": [],
        }

    batches, batch_index_list = make_batches(tts, samples, batch_size, text_split_method)
    fragments_by_input = [None] * len(samples)
    per_batch_seconds = []
    started_at = time.perf_counter()

    for batch_order, (batch, batch_index) in enumerate(zip(batches, batch_index_list), start=1):
        batch_started_at = time.perf_counter()
        batch_fragments = run_one_internal_batch(
            tts,
            batch,
            batch_index,
            refer_audio_spec,
            sv_emb,
            speed_factor,
            evidence,
        )
        batch_seconds = round(time.perf_counter() - batch_started_at, 3)
        per_batch_seconds.append(batch_seconds)
        if evidence is not None:
            evidence["captured_audio_batches"].append(
                {
                    "batch_order": batch_order,
                    "batch_seconds": batch_seconds,
                    "outer_batches": 1,
                    "inner_batch_sizes": [len(batch_fragments)],
                    "batch_index": batch_index,
                    "split_bucket": False,
                }
            )
        for local_idx, sample_idx in enumerate(batch_index):
            fragments_by_input[sample_idx] = batch_fragments[local_idx]

    sync_if_cuda(tts.configs.device)
    total_seconds = round(time.perf_counter() - started_at, 3)
    if any(fragment is None for fragment in fragments_by_input):
        raise RuntimeError(f"Missing output fragments for batch_size={batch_size}")

    return {
        "total_seconds": total_seconds,
        "per_batch_seconds": per_batch_seconds,
        "fragments": fragments_by_input,
        "batch_index_list": batch_index_list,
        "evidence": evidence,
    }


def write_outputs(out_dir: Path, mode_name: str, samples: list[dict], fragments, sr: int, fragment_interval: float):
    from run_dongqing_internal_batch_size_2_20char import tensor_to_int16, wav_info

    mode_dir = out_dir / mode_name
    mode_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for sample, fragment in zip(samples, fragments):
        wav_path = mode_dir / f"{sample['id']}.wav"
        sf.write(wav_path, tensor_to_int16(fragment, sr, fragment_interval), sr, format="wav")
        info = wav_info(wav_path)
        items.append(
            {
                "id": sample["id"],
                "text": sample["text"],
                "wav_path": str(wav_path),
                "audio_duration_seconds": info["duration_seconds"],
                "audio_info": info,
            }
        )
    return items


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    ref_audio = Path(args.ref_audio).expanduser()
    if not ref_audio.is_absolute():
        ref_audio = PROJECT_ROOT / ref_audio
    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    if not config_path.exists():
        raise FileNotFoundError(f"TTS config not found: {config_path}")
    if not ref_audio.exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_audio}")
    out_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(GPT_SOVITS_ROOT)
    sys.path.insert(0, str(HELPER_SCRIPT_DIR))
    sys.path.insert(0, str(GPT_SOVITS_ROOT))
    sys.path.insert(0, str(GPT_SOVITS_ROOT / "GPT_SoVITS"))

    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
    from module import commons

    tts = TTS(TTS_Config(str(config_path)))
    tts.commons = commons
    if tts.configs.use_vocoder:
        raise RuntimeError("This comparison script currently supports GPT-SoVITS VITS decode, not vocoder decode.")

    setup_seconds = prepare_prompt_cache(tts, ref_audio, args.prompt_text)
    refer_audio_spec, sv_emb = get_reference_inputs(tts)

    warmup = {"batch2_seconds": [], "batch1_twice_seconds": []}
    for _ in range(max(0, args.warmup_runs)):
        warmup_batch2 = run_mode(
            tts,
            SAMPLES,
            2,
            args.text_split_method,
            refer_audio_spec,
            sv_emb,
            args.speed_factor,
            record_evidence=False,
        )
        warmup["batch2_seconds"].append(warmup_batch2["total_seconds"])
        warmup_batch1 = run_mode(
            tts,
            SAMPLES,
            1,
            args.text_split_method,
            refer_audio_spec,
            sv_emb,
            args.speed_factor,
            record_evidence=False,
        )
        warmup["batch1_twice_seconds"].append(warmup_batch1["total_seconds"])

    batch2_result = run_mode(
        tts,
        SAMPLES,
        2,
        args.text_split_method,
        refer_audio_spec,
        sv_emb,
        args.speed_factor,
        record_evidence=True,
    )
    batch1_twice_result = run_mode(
        tts,
        SAMPLES,
        1,
        args.text_split_method,
        refer_audio_spec,
        sv_emb,
        args.speed_factor,
        record_evidence=True,
    )

    sr = tts.configs.sampling_rate
    batch2_items = write_outputs(out_dir, "batch2", SAMPLES, batch2_result["fragments"], sr, args.fragment_interval)
    batch1_items = write_outputs(out_dir, "batch1_twice", SAMPLES, batch1_twice_result["fragments"], sr, args.fragment_interval)

    batch2_semantic_ok = any(
        call.get("observed_batch_size") == 2 for call in batch2_result["evidence"]["semantic_model_calls"]
    )
    batch2_vits_ok = any(call.get("observed_batch_size") == 2 for call in batch2_result["evidence"]["vits_decode_calls"])
    batch1_semantic_ok = all(
        call.get("observed_batch_size") == 1 for call in batch1_twice_result["evidence"]["semantic_model_calls"]
    )
    batch1_vits_ok = all(
        call.get("observed_batch_size") == 1 for call in batch1_twice_result["evidence"]["vits_decode_calls"]
    )

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "stage": "tts",
        "timing_mode": "warm",
        "comparison": "internal_batch2_vs_internal_batch1_twice",
        "setup_seconds": setup_seconds,
        "warmup_runs": max(0, args.warmup_runs),
        "warmup": warmup,
        "sampling_rate": sr,
        "config": str(config_path),
        "ref_audio": str(ref_audio),
        "prompt_text": args.prompt_text,
        "text_split_method": args.text_split_method,
        "batch2": {
            "batch_size": 2,
            "total_seconds": batch2_result["total_seconds"],
            "per_batch_seconds": batch2_result["per_batch_seconds"],
            "batch_index_list": batch2_result["batch_index_list"],
            "semantic_model_batch_ok": batch2_semantic_ok,
            "acoustic_model_batch_ok": batch2_vits_ok,
            "is_true_model_batch": batch2_semantic_ok and batch2_vits_ok,
            "evidence": batch2_result["evidence"],
            "items": batch2_items,
        },
        "batch1_twice": {
            "batch_size": 1,
            "num_runs": 2,
            "total_seconds": batch1_twice_result["total_seconds"],
            "per_run_seconds": batch1_twice_result["per_batch_seconds"],
            "batch_index_list": batch1_twice_result["batch_index_list"],
            "semantic_model_batch_ok": batch1_semantic_ok,
            "acoustic_model_batch_ok": batch1_vits_ok,
            "evidence": batch1_twice_result["evidence"],
            "items": batch1_items,
        },
    }
    manifest["speedup"] = {
        "batch1_twice_over_batch2": round(
            manifest["batch1_twice"]["total_seconds"] / manifest["batch2"]["total_seconds"], 4
        )
        if manifest["batch2"]["total_seconds"] > 0
        else None,
        "seconds_saved_by_batch2": round(
            manifest["batch1_twice"]["total_seconds"] - manifest["batch2"]["total_seconds"], 3
        ),
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"tts_internal_compare_batch2_seconds: {manifest['batch2']['total_seconds']}")
    print(f"tts_internal_compare_batch1_twice_seconds: {manifest['batch1_twice']['total_seconds']}")
    print(f"tts_internal_compare_speedup_batch1_over_batch2: {manifest['speedup']['batch1_twice_over_batch2']}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
