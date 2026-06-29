#!/usr/bin/env python3
"""Clean GPT-SoVITS reference wavs for clearer voice cloning prompts.

The script keeps the original files untouched by default and writes PCM wavs
with `_enhanced` appended to the filename. Use `--replace 1` to backup and
overwrite the source files after you have listened to the enhanced versions.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy import signal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILES = [
    PROJECT_ROOT / "bilibili_downloads" / "Female.wav",
    PROJECT_ROOT / "bilibili_downloads" / "SaBeining.wav",
    PROJECT_ROOT / "bilibili_downloads" / "DongQing_6s.wav",
]


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    return np.mean(audio, axis=1).astype(np.float32, copy=False)


def _resample(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return audio.astype(np.float32, copy=False)
    return librosa.resample(audio, orig_sr=source_sr, target_sr=target_sr).astype(np.float32, copy=False)


def _trim_and_pad(audio: np.ndarray, sr: int, top_db: float, pad_seconds: float) -> np.ndarray:
    if audio.size == 0:
        return audio
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    if trimmed.size == 0:
        trimmed = audio
    pad = np.zeros(int(pad_seconds * sr), dtype=np.float32)
    return np.concatenate([pad, trimmed.astype(np.float32, copy=False), pad])


def _butter_filter(audio: np.ndarray, sr: int, highpass: float, lowpass: float) -> np.ndarray:
    if audio.size == 0:
        return audio
    nyquist = sr / 2.0
    filtered = audio
    if 0 < highpass < nyquist:
        sos = signal.butter(4, highpass / nyquist, btype="highpass", output="sos")
        filtered = signal.sosfiltfilt(sos, filtered).astype(np.float32, copy=False)
    if 0 < lowpass < nyquist:
        sos = signal.butter(6, lowpass / nyquist, btype="lowpass", output="sos")
        filtered = signal.sosfiltfilt(sos, filtered).astype(np.float32, copy=False)
    return filtered.astype(np.float32, copy=False)


def _spectral_denoise(audio: np.ndarray, sr: int, strength: float) -> np.ndarray:
    if audio.size < sr or strength <= 0:
        return audio

    noise_len = max(int(0.2 * sr), min(int(0.6 * sr), audio.size // 8))
    edge_noise = np.concatenate([audio[:noise_len], audio[-noise_len:]])
    frame = min(1024, max(256, int(0.064 * sr)))
    hop = frame // 4

    _, _, noise_zxx = signal.stft(edge_noise, fs=sr, nperseg=frame, noverlap=frame - hop, boundary=None)
    noise_profile = np.median(np.abs(noise_zxx), axis=1, keepdims=True)

    _, _, zxx = signal.stft(audio, fs=sr, nperseg=frame, noverlap=frame - hop, boundary="zeros")
    magnitude = np.abs(zxx)
    phase = np.exp(1j * np.angle(zxx))

    floor = 0.12
    reduction = np.clip((magnitude - noise_profile * strength) / (magnitude + 1e-8), floor, 1.0)
    cleaned_zxx = magnitude * reduction * phase
    _, cleaned = signal.istft(cleaned_zxx, fs=sr, nperseg=frame, noverlap=frame - hop, input_onesided=True)

    if cleaned.size < audio.size:
        cleaned = np.pad(cleaned, (0, audio.size - cleaned.size))
    return cleaned[:audio.size].astype(np.float32, copy=False)


def _normalize_loudness(audio: np.ndarray, sr: int, target_lufs: float, peak_db: float) -> np.ndarray:
    if audio.size == 0:
        return audio
    audio = audio - float(np.mean(audio))
    meter = pyln.Meter(sr)
    try:
        loudness = meter.integrated_loudness(audio)
        if np.isfinite(loudness) and abs(loudness) < 100:
            audio = pyln.normalize.loudness(audio, loudness, target_lufs)
    except Exception:
        pass

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    peak_target = 10 ** (peak_db / 20.0)
    if peak > peak_target:
        audio = audio * (peak_target / peak)
    return np.clip(audio, -0.999, 0.999).astype(np.float32, copy=False)


def _fade_edges(audio: np.ndarray, sr: int, fade_ms: float) -> np.ndarray:
    fade_len = min(int(sr * fade_ms / 1000.0), audio.size // 2)
    if fade_len <= 1:
        return audio
    out = audio.copy()
    fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    out[:fade_len] *= fade_in
    out[-fade_len:] *= fade_in[::-1]
    return out


def _duration_control(audio: np.ndarray, sr: int, min_seconds: float, max_seconds: float) -> np.ndarray:
    if audio.size == 0:
        return audio
    max_samples = int(max_seconds * sr)
    min_samples = int(min_seconds * sr)
    if audio.size > max_samples:
        start = max(0, (audio.size - max_samples) // 2)
        audio = audio[start:start + max_samples]
    if audio.size < min_samples:
        audio = np.pad(audio, (0, min_samples - audio.size))
    return audio.astype(np.float32, copy=False)


def enhance_file(source: Path, output: Path, args: argparse.Namespace) -> dict:
    raw, source_sr = sf.read(source, always_2d=False)
    audio = _to_mono(np.asarray(raw))
    audio = _resample(audio, source_sr, args.sample_rate)
    audio = _trim_and_pad(audio, args.sample_rate, args.trim_top_db, args.edge_silence)
    audio = _butter_filter(audio, args.sample_rate, args.highpass, args.lowpass)
    audio = _spectral_denoise(audio, args.sample_rate, args.denoise_strength)
    audio = _normalize_loudness(audio, args.sample_rate, args.target_lufs, args.peak_db)
    audio = _duration_control(audio, args.sample_rate, args.min_seconds, args.max_seconds)
    audio = _fade_edges(audio, args.sample_rate, args.fade_ms)

    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, audio, args.sample_rate, subtype="PCM_16")
    return {
        "source_sr": source_sr,
        "target_sr": args.sample_rate,
        "source_seconds": len(raw) / source_sr,
        "target_seconds": audio.size / args.sample_rate,
        "peak": float(np.max(np.abs(audio))) if audio.size else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhance GPT-SoVITS reference wavs.")
    parser.add_argument("files", nargs="*", type=Path, default=DEFAULT_FILES)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--replace", type=int, default=0, help="backup originals and overwrite them")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--min-seconds", type=float, default=5.0)
    parser.add_argument("--max-seconds", type=float, default=10.0)
    parser.add_argument("--edge-silence", type=float, default=0.8)
    parser.add_argument("--trim-top-db", type=float, default=36.0)
    parser.add_argument("--highpass", type=float, default=70.0)
    parser.add_argument("--lowpass", type=float, default=7600.0)
    parser.add_argument("--denoise-strength", type=float, default=1.15)
    parser.add_argument("--target-lufs", type=float, default=-23.0)
    parser.add_argument("--peak-db", type=float, default=-1.0)
    parser.add_argument("--fade-ms", type=float, default=35.0)
    return parser.parse_args()


def main():
    args = parse_args()
    for source in args.files:
        source = source.expanduser().resolve()
        if not source.exists():
            print(f"missing: {source}")
            continue

        if args.output_dir is None:
            output = source.with_name(f"{source.stem}_enhanced.wav")
        else:
            output = args.output_dir.expanduser().resolve() / source.name

        report = enhance_file(source, output, args)
        print(
            f"enhanced: {source.name} -> {output} "
            f"sr {report['source_sr']}->{report['target_sr']} "
            f"duration {report['source_seconds']:.2f}s->{report['target_seconds']:.2f}s "
            f"peak={report['peak']:.3f}"
        )

        if args.replace:
            backup = source.with_name(f"{source.stem}.backup_before_enhance{source.suffix}")
            if not backup.exists():
                shutil.copy2(source, backup)
            shutil.copy2(output, source)
            print(f"replaced original, backup: {backup}")


if __name__ == "__main__":
    main()
