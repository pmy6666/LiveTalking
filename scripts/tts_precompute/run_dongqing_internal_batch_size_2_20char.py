#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GPT_SOVITS_ROOT = PROJECT_ROOT / "GPT-SoVITS"
DEFAULT_CONFIG = GPT_SOVITS_ROOT / "GPT_SoVITS" / "configs" / "tts_infer_livetalking_v2proplus.yaml"
DEFAULT_OUT_DIR = PROJECT_ROOT / "test" / "ck_time" / "tts_internal_batch_size_2_dongqing_20char"
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
    parser = argparse.ArgumentParser(description="Run GPT-SoVITS with true internal batch_size=2.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--ref-audio", default=str(DEFAULT_REF_AUDIO))
    parser.add_argument("--prompt-text", default=DEFAULT_PROMPT_TEXT)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--speed-factor", type=float, default=1.0)
    parser.add_argument("--fragment-interval", type=float, default=0.1)
    parser.add_argument(
        "--text-split-method",
        default="cut0",
        help="Use cut0 so the two newline-separated texts remain exactly two batch samples.",
    )
    return parser.parse_args()


def tensor_shape(value) -> list[int] | str:
    if hasattr(value, "shape"):
        return list(value.shape)
    return type(value).__name__


def tensor_to_int16(audio: torch.Tensor, sr: int, fragment_interval: float) -> np.ndarray:
    audio = audio.detach().float().cpu()
    max_audio = torch.abs(audio).max()
    if max_audio > 1:
        audio = audio / max_audio
    if fragment_interval > 0:
        audio = torch.cat([audio, torch.zeros(int(sr * fragment_interval), dtype=audio.dtype)])
    return (audio.numpy() * 32768).clip(-32768, 32767).astype(np.int16)


def get_batched_ge(tts, refer_audio_spec, sv_emb):
    def get_ge(refer, speaker_embedding):
        ge = None
        if refer is not None:
            refer_lengths = torch.LongTensor([refer.size(2)]).to(refer.device)
            refer_mask = torch.unsqueeze(tts.commons.sequence_mask(refer_lengths, refer.size(2)), 1).to(refer.dtype)
            if tts.vits_model.version == "v1":
                ge = tts.vits_model.ref_enc(refer * refer_mask, refer_mask)
            else:
                ge = tts.vits_model.ref_enc(refer[:, :704] * refer_mask, refer_mask)
            if tts.vits_model.is_v2pro:
                speaker_embedding = tts.vits_model.sv_emb(speaker_embedding)
                ge += speaker_embedding.unsqueeze(-1)
                ge = tts.vits_model.prelu(ge)
        return ge

    if isinstance(refer_audio_spec, list):
        ges = []
        for idx, ref_spec in enumerate(refer_audio_spec):
            ges.append(get_ge(ref_spec, sv_emb[idx] if tts.vits_model.is_v2pro else None))
        ge = torch.stack(ges, 0).mean(0)
    else:
        ge = get_ge(refer_audio_spec, sv_emb)
    return ge


@torch.no_grad()
def strict_vits_batch_decode(tts, pred_semantic_list, idx_list, batch_phones, refer_audio_spec, speed_factor, sv_emb, evidence):
    trimmed_semantics = [semantic[-idx:] for semantic, idx in zip(pred_semantic_list, idx_list)]
    semantic_lengths = torch.LongTensor([semantic.shape[0] for semantic in trimmed_semantics]).to(tts.configs.device)
    phone_lengths = torch.LongTensor([phones.shape[-1] for phones in batch_phones]).to(tts.configs.device)

    # ResidualVectorQuantizer.decode expects [n_quantizer, batch, time].
    codes = tts.batch_sequences(trimmed_semantics, axis=0, pad_value=0).unsqueeze(0).to(tts.configs.device)
    text = tts.batch_sequences(batch_phones, axis=0, pad_value=0).to(tts.configs.device)

    y_lengths = semantic_lengths * 2
    text_lengths = phone_lengths

    ge = get_batched_ge(tts, refer_audio_spec, sv_emb)
    if ge is not None and ge.shape[0] == 1 and codes.shape[1] > 1:
        ge = ge.expand(codes.shape[1], -1, -1).contiguous()
    ge_for_enc = tts.vits_model.ge_to512(ge.transpose(2, 1)).transpose(2, 1) if tts.vits_model.is_v2pro else ge

    if evidence is not None:
        evidence["vits_decode_calls"].append(
            {
                "codes_shape": tensor_shape(codes),
                "text_shape": tensor_shape(text),
                "y_lengths": [int(v) for v in y_lengths.detach().cpu().tolist()],
                "text_lengths": [int(v) for v in text_lengths.detach().cpu().tolist()],
                "ge_shape": tensor_shape(ge),
                "observed_batch_size": int(codes.shape[1]),
                "decode_mode": "strict_padded_batch",
            }
        )

    quantized = tts.vits_model.quantizer.decode(codes)
    if tts.vits_model.semantic_frame_rate == "25hz":
        quantized = torch.nn.functional.interpolate(quantized, size=int(quantized.shape[-1] * 2), mode="nearest")

    x, m_p, logs_p, y_mask, _, _ = tts.vits_model.enc_p(
        quantized,
        y_lengths,
        text,
        text_lengths,
        ge_for_enc,
        speed_factor,
    )
    z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * 0.5
    z = tts.vits_model.flow(z_p, y_mask, g=ge, reverse=True)
    decoded = tts.vits_model.dec((z * y_mask)[:, :, :], g=ge)

    upsample_rate = math.prod(tts.vits_model.upsample_rates)
    audio_lengths = (semantic_lengths * 2 * upsample_rate).detach().cpu().tolist()
    return [decoded[i, 0, : int(length)].detach() for i, length in enumerate(audio_lengths)]


def main() -> int:
    args = parse_args()
    if not GPT_SOVITS_ROOT.exists():
        raise FileNotFoundError(f"GPT-SoVITS repo not found: {GPT_SOVITS_ROOT}")

    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(
            f"TTS config not found: {config_path}. Start GPT-SoVITS once with start_gpt_sovits_v2proplus.sh first."
        )

    ref_audio = Path(args.ref_audio).expanduser()
    if not ref_audio.is_absolute():
        ref_audio = PROJECT_ROOT / ref_audio
    if not ref_audio.exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_audio}")

    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(GPT_SOVITS_ROOT)
    sys.path.insert(0, str(GPT_SOVITS_ROOT))
    sys.path.insert(0, str(GPT_SOVITS_ROOT / "GPT_SoVITS"))

    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
    from module import commons

    tts = TTS(TTS_Config(str(config_path)))
    tts.commons = commons
    if tts.configs.use_vocoder:
        raise RuntimeError("This strict internal batch script currently supports GPT-SoVITS VITS decode, not vocoder decode.")

    evidence = {
        "input_text_count": len(SAMPLES),
        "requested_batch_size": args.batch_size,
        "semantic_model_calls": [],
        "vits_decode_calls": [],
        "captured_audio_batches": [],
    }

    original_infer_panel_batch_infer = tts.t2s_model.model.infer_panel_batch_infer

    trace_enabled = True

    def traced_infer_panel_batch_infer(*infer_args, **infer_kwargs):
        all_phoneme_ids = infer_args[0] if len(infer_args) > 0 else None
        all_phoneme_lens = infer_args[1] if len(infer_args) > 1 else None
        prompt = infer_args[2] if len(infer_args) > 2 else None
        all_bert_features = infer_args[3] if len(infer_args) > 3 else None
        if trace_enabled:
            evidence["semantic_model_calls"].append(
                {
                    "method": "infer_panel_batch_infer",
                    "all_phoneme_ids_shape": tensor_shape(all_phoneme_ids),
                    "all_phoneme_lens_shape": tensor_shape(all_phoneme_lens),
                    "prompt_shape": tensor_shape(prompt),
                    "all_bert_features_type": type(all_bert_features).__name__,
                    "observed_batch_size": int(all_phoneme_lens.shape[0]) if hasattr(all_phoneme_lens, "shape") else None,
                }
            )
        return original_infer_panel_batch_infer(*infer_args, **infer_kwargs)

    tts.t2s_model.model.infer_panel_batch_infer = traced_infer_panel_batch_infer

    combined_text = "\n".join(sample["text"] for sample in SAMPLES)
    payload = {
        "text": combined_text,
        "text_lang": "zh",
        "ref_audio_path": str(ref_audio),
        "prompt_text": args.prompt_text,
        "prompt_lang": "zh",
        "top_k": 15,
        "top_p": 1,
        "temperature": 1,
        "text_split_method": args.text_split_method,
        "batch_size": int(args.batch_size),
        "batch_threshold": 0.0,
        "split_bucket": False,
        "media_type": "wav",
        "streaming_mode": 0,
        "parallel_infer": True,
        "speed_factor": float(args.speed_factor),
        "fragment_interval": float(args.fragment_interval),
        "repetition_penalty": 1.35,
    }

    setup_started_at = time.perf_counter()
    tts.set_ref_audio(str(ref_audio))
    prompt_text = args.prompt_text.strip("\n")
    if prompt_text[-1] not in "。！？；：，,.!?;:":
        prompt_text += "。"
    if tts.prompt_cache["prompt_text"] != prompt_text:
        phones, bert_features, norm_text = tts.text_preprocessor.segment_and_extract_feature_for_text(
            prompt_text, "zh", tts.configs.version
        )
        tts.prompt_cache["prompt_text"] = prompt_text
        tts.prompt_cache["prompt_lang"] = "zh"
        tts.prompt_cache["phones"] = phones
        tts.prompt_cache["bert_features"] = bert_features
        tts.prompt_cache["norm_text"] = norm_text
    setup_seconds = round(time.perf_counter() - setup_started_at, 3)

    data = tts.text_preprocessor.preprocess(combined_text, "zh", args.text_split_method, tts.configs.version)
    if len(data) != len(SAMPLES):
        raise RuntimeError(
            f"Expected {len(SAMPLES)} text samples after preprocessing, got {len(data)}. "
            "Use --text-split-method cut0 for this strict batch_size=2 benchmark."
        )
    batches, batch_index_list = tts.to_batch(
        data,
        prompt_data=tts.prompt_cache,
        batch_size=args.batch_size,
        threshold=0.0,
        split_bucket=False,
        device=tts.configs.device,
        precision=tts.precision,
    )
    if len(batches) != 1 or len(batch_index_list) != 1 or batch_index_list[0] != [0, 1]:
        raise RuntimeError(f"Expected exactly one internal batch [0, 1], got batch_index_list={batch_index_list}")

    item = batches[0]
    batch_phones = item["phones"]
    all_phoneme_ids = item["all_phones"]
    all_phoneme_lens = item["all_phones_len"]
    all_bert_features = item["all_bert_features"]
    max_len = item["max_len"]
    prompt = tts.prompt_cache["prompt_semantic"].expand(len(all_phoneme_ids), -1).to(tts.configs.device)

    refer_audio_spec = []
    sv_emb = [] if tts.is_v2pro else None
    for spec, audio_tensor in tts.prompt_cache["refer_spec"]:
        spec = spec.to(dtype=tts.precision, device=tts.configs.device)
        refer_audio_spec.append(spec)
        if tts.is_v2pro:
            sv_emb.append(tts.sv_model.compute_embedding3(audio_tensor))

    def run_internal_batch(record_evidence: bool):
        nonlocal trace_enabled
        trace_enabled = record_evidence
        pred_semantic_list, idx_list = tts.t2s_model.model.infer_panel_batch_infer(
            all_phoneme_ids,
            all_phoneme_lens,
            prompt,
            all_bert_features,
            top_k=payload["top_k"],
            top_p=payload["top_p"],
            temperature=payload["temperature"],
            early_stop_num=tts.configs.hz * tts.configs.max_sec,
            max_len=max_len,
            repetition_penalty=payload["repetition_penalty"],
        )
        fragments = strict_vits_batch_decode(
            tts,
            pred_semantic_list,
            idx_list,
            batch_phones,
            refer_audio_spec,
            args.speed_factor,
            sv_emb,
            evidence if record_evidence else None,
        )
        trace_enabled = True
        return fragments

    warmup_seconds = []
    for _ in range(max(0, args.warmup_runs)):
        warmup_started_at = time.perf_counter()
        _ = run_internal_batch(record_evidence=False)
        if "cuda" in str(tts.configs.device):
            torch.cuda.synchronize()
        warmup_seconds.append(round(time.perf_counter() - warmup_started_at, 3))

    started_at = time.perf_counter()
    captured_fragments = run_internal_batch(record_evidence=True)
    if "cuda" in str(tts.configs.device):
        torch.cuda.synchronize()
    evidence["captured_audio_batches"].append(
        {
            "outer_batches": 1,
            "inner_batch_sizes": [len(captured_fragments)],
            "batch_index_list": batch_index_list,
            "split_bucket": False,
        }
    )
    sr, combined_audio = tts.audio_postprocess(
        [captured_fragments],
        tts.configs.sampling_rate,
        batch_index_list,
        args.speed_factor,
        False,
        args.fragment_interval,
        False,
    )
    total_seconds = round(time.perf_counter() - started_at, 3)

    combined_path = out_dir / "combined_internal_batch2.wav"
    sf.write(combined_path, combined_audio, sr, format="wav")

    if len(captured_fragments) != len(SAMPLES):
        raise RuntimeError(
            f"Expected {len(SAMPLES)} captured audio fragments from one internal batch, got {len(captured_fragments)}. "
            f"This usually means text_split_method split the two input samples into extra fragments. "
            f"Use --text-split-method cut0 for this internal batch_size=2 benchmark."
        )

    items = []
    for sample, fragment in zip(SAMPLES, captured_fragments):
        wav_path = out_dir / f"{sample['id']}.wav"
        audio = tensor_to_int16(fragment, sr, args.fragment_interval)
        sf.write(wav_path, audio, sr, format="wav")
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

    observed_semantic_batch_sizes = [
        item.get("observed_batch_size")
        for item in evidence["semantic_model_calls"]
        if item.get("observed_batch_size") is not None
    ]
    observed_vits_batch_sizes = [
        item.get("observed_batch_size")
        for item in evidence["vits_decode_calls"]
        if item.get("observed_batch_size") is not None
    ]
    semantic_model_batch_ok = args.batch_size in observed_semantic_batch_sizes
    acoustic_model_batch_ok = args.batch_size in observed_vits_batch_sizes
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "stage": "tts",
        "batch_mode": "model_internal",
        "batch_size": args.batch_size,
        "semantic_model_batch_ok": semantic_model_batch_ok,
        "acoustic_model_batch_ok": acoustic_model_batch_ok,
        "is_true_model_batch": semantic_model_batch_ok and acoustic_model_batch_ok and len(captured_fragments) == len(SAMPLES),
        "total_seconds": total_seconds,
        "timing_mode": "warm",
        "warmup_runs": max(0, args.warmup_runs),
        "warmup_seconds": warmup_seconds,
        "setup_seconds": setup_seconds,
        "sampling_rate": sr,
        "config": str(config_path),
        "ref_audio": str(ref_audio),
        "prompt_text": args.prompt_text,
        "payload": payload,
        "combined_wav_path": str(combined_path),
        "evidence": evidence,
        "items": items,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"tts_internal_batch2_seconds: {total_seconds}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
