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
DEFAULT_OUT_DIR = PROJECT_ROOT / "test" / "ck_time" / "tts_internal_batch_size_1_twice_dongqing_20char"
DEFAULT_REF_AUDIO = PROJECT_ROOT / "bilibili_downloads" / "DongQing_6s.wav"
DEFAULT_PROMPT_TEXT = "那种快乐常常像一场梦，电影陪伴我们长大"

SAMPLES = [
    {
        "id": "dongqing_batch1_01",
        "text": "今天上午阳光很好，我们一起去公园慢慢散步吧。",
    },
    {
        "id": "dongqing_batch1_02",
        "text": "下午会议结束以后，请把今天测试结果整理清楚。",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT-SoVITS internal batch_size=1 twice for latency baseline.")
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


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(HELPER_SCRIPT_DIR))
    from run_dongqing_internal_batch_size_2_20char import strict_vits_batch_decode, tensor_to_int16, wav_info

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
    sys.path.insert(0, str(GPT_SOVITS_ROOT))
    sys.path.insert(0, str(GPT_SOVITS_ROOT / "GPT_SoVITS"))

    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
    from module import commons

    tts = TTS(TTS_Config(str(config_path)))
    tts.commons = commons
    if tts.configs.use_vocoder:
        raise RuntimeError("This batch_size=1 baseline currently supports GPT-SoVITS VITS decode, not vocoder decode.")

    evidence = {
        "input_text_count": len(SAMPLES),
        "requested_batch_size": 1,
        "semantic_model_calls": [],
        "vits_decode_calls": [],
        "captured_audio_batches": [],
    }

    original_infer_panel_batch_infer = tts.t2s_model.model.infer_panel_batch_infer

    trace_enabled = True

    def traced_infer_panel_batch_infer(*infer_args, **infer_kwargs):
        all_phoneme_lens = infer_args[1] if len(infer_args) > 1 else None
        prompt = infer_args[2] if len(infer_args) > 2 else None
        if trace_enabled:
            evidence["semantic_model_calls"].append(
                {
                    "method": "infer_panel_batch_infer",
                    "all_phoneme_lens_shape": tensor_shape(all_phoneme_lens),
                    "prompt_shape": tensor_shape(prompt),
                    "observed_batch_size": int(all_phoneme_lens.shape[0]) if hasattr(all_phoneme_lens, "shape") else None,
                }
            )
        return original_infer_panel_batch_infer(*infer_args, **infer_kwargs)

    tts.t2s_model.model.infer_panel_batch_infer = traced_infer_panel_batch_infer

    setup_started_at = time.perf_counter()
    tts.set_ref_audio(str(ref_audio))
    prompt_text = args.prompt_text.strip("\n")
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
    setup_seconds = round(time.perf_counter() - setup_started_at, 3)

    refer_audio_spec = []
    sv_emb = [] if tts.is_v2pro else None
    for spec, audio_tensor in tts.prompt_cache["refer_spec"]:
        spec = spec.to(dtype=tts.precision, device=tts.configs.device)
        refer_audio_spec.append(spec)
        if tts.is_v2pro:
            sv_emb.append(tts.sv_model.compute_embedding3(audio_tensor))

    def run_one_sample(sample: dict, record_evidence: bool):
        nonlocal trace_enabled
        trace_enabled = record_evidence
        data = tts.text_preprocessor.preprocess(sample["text"], "zh", args.text_split_method, tts.configs.version)
        if len(data) != 1:
            raise RuntimeError(f"Expected one text sample for {sample['id']}, got {len(data)}")
        batches, batch_index_list = tts.to_batch(
            data,
            prompt_data=tts.prompt_cache,
            batch_size=1,
            threshold=0.0,
            split_bucket=False,
            device=tts.configs.device,
            precision=tts.precision,
        )
        if len(batches) != 1 or batch_index_list != [[0]]:
            raise RuntimeError(f"Expected one internal batch [[0]] for {sample['id']}, got {batch_index_list}")

        item = batches[0]
        prompt = tts.prompt_cache["prompt_semantic"].expand(len(item["all_phones"]), -1).to(tts.configs.device)
        pred_semantic_list, idx_list = tts.t2s_model.model.infer_panel_batch_infer(
            item["all_phones"],
            item["all_phones_len"],
            prompt,
            item["all_bert_features"],
            top_k=15,
            top_p=1,
            temperature=1,
            early_stop_num=tts.configs.hz * tts.configs.max_sec,
            max_len=item["max_len"],
            repetition_penalty=1.35,
        )
        fragments = strict_vits_batch_decode(
            tts,
            pred_semantic_list,
            idx_list,
            item["phones"],
            refer_audio_spec,
            args.speed_factor,
            sv_emb,
            evidence if record_evidence else None,
        )
        trace_enabled = True
        if len(fragments) != 1:
            raise RuntimeError(f"Expected one audio fragment for {sample['id']}, got {len(fragments)}")
        return fragments, batch_index_list

    warmup_seconds = []
    for _ in range(max(0, args.warmup_runs)):
        warmup_started_at = time.perf_counter()
        for sample in SAMPLES:
            _ = run_one_sample(sample, record_evidence=False)
        if "cuda" in str(tts.configs.device):
            torch.cuda.synchronize()
        warmup_seconds.append(round(time.perf_counter() - warmup_started_at, 3))

    total_started_at = time.perf_counter()
    items = []
    combined_fragments = []
    for run_index, sample in enumerate(SAMPLES, start=1):
        run_started_at = time.perf_counter()
        fragments, batch_index_list = run_one_sample(sample, record_evidence=True)
        if "cuda" in str(tts.configs.device):
            torch.cuda.synchronize()
        evidence["captured_audio_batches"].append(
            {
                "run_index": run_index,
                "outer_batches": 1,
                "inner_batch_sizes": [1],
                "batch_index_list": batch_index_list,
                "split_bucket": False,
            }
        )

        wav_path = out_dir / f"{sample['id']}.wav"
        sf.write(wav_path, tensor_to_int16(fragments[0], tts.configs.sampling_rate, args.fragment_interval), tts.configs.sampling_rate, format="wav")
        info = wav_info(wav_path)
        run_seconds = round(time.perf_counter() - run_started_at, 3)
        combined_fragments.extend(fragments)
        items.append(
            {
                "id": sample["id"],
                "text": sample["text"],
                "run_index": run_index,
                "run_seconds": run_seconds,
                "wav_path": str(wav_path),
                "audio_duration_seconds": info["duration_seconds"],
                "audio_info": info,
            }
        )

    sr, combined_audio = tts.audio_postprocess(
        [combined_fragments],
        tts.configs.sampling_rate,
        [[0, 1]],
        args.speed_factor,
        False,
        args.fragment_interval,
        False,
    )
    combined_path = out_dir / "combined_batch1_twice.wav"
    sf.write(combined_path, combined_audio, sr, format="wav")
    total_seconds = round(time.perf_counter() - total_started_at, 3)
    inference_seconds_sum = round(sum(item["run_seconds"] for item in items), 3)

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "stage": "tts",
        "batch_mode": "model_internal_batch1_twice",
        "batch_size": 1,
        "num_runs": 2,
        "total_seconds": total_seconds,
        "timing_mode": "warm",
        "warmup_runs": max(0, args.warmup_runs),
        "warmup_seconds": warmup_seconds,
        "setup_seconds": setup_seconds,
        "inference_seconds_sum": inference_seconds_sum,
        "sampling_rate": sr,
        "config": str(config_path),
        "ref_audio": str(ref_audio),
        "prompt_text": args.prompt_text,
        "combined_wav_path": str(combined_path),
        "evidence": evidence,
        "items": items,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"tts_internal_batch1_twice_seconds: {total_seconds}")
    print(f"tts_internal_batch1_twice_inference_sum_seconds: {inference_seconds_sum}")
    print(f"tts_internal_batch1_run_seconds: {[item['run_seconds'] for item in items]}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
