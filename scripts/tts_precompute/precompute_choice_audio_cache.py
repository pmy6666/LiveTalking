#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from choice.orchestrator import ChoiceAudioCache, ChoiceAudioSynthesizer, ChoiceOrchestrator, StaticChoiceTreeProvider
from choice.voice_profiles import load_voice_texts


VOICE_GROUPS = ("female", "male")
DEFAULT_FEMALE_REF = PROJECT_ROOT / "bilibili_downloads" / "DongQing_6s.wav"
DEFAULT_MALE_REF = PROJECT_ROOT / "bilibili_downloads" / "SaBeining.wav"


def status(message: str) -> None:
    print(f"[choice-audio-cache] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute LiveTalking choice-mode TTS audio caches.")
    parser.add_argument("--tree_id", default="default_choice_tree")
    parser.add_argument("--voices", nargs="*", choices=VOICE_GROUPS, default=None, help="Voice groups to cache. Defaults to female and male.")
    parser.add_argument("--female_ref", default=str(DEFAULT_FEMALE_REF))
    parser.add_argument("--male_ref", default=str(DEFAULT_MALE_REF))
    parser.add_argument("--female_ref_text", default=None)
    parser.add_argument("--male_ref_text", default=None)
    parser.add_argument("--tts_server", default="http://127.0.0.1:9880")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--speed_factor", type=float, default=1.08)
    parser.add_argument("--fragment_interval", type=float, default=0.1)
    parser.add_argument("--force", action="store_true", help="Regenerate cache files even when they already exist.")
    parser.add_argument("--export_wav", action="store_true", help="Also export playable WAV files for inspection.")
    parser.add_argument("--wav_dir", default=str(PROJECT_ROOT / "cache" / "choice_audio_wav"))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--timeout_note", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def normalize_ref_path(path: str) -> str:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return str(resolved.resolve())


def iter_nodes(tree_payload: dict):
    for node in tree_payload.get("nodes", []):
        yield node


def make_session(
    ref_file: str,
    ref_text: str,
    tts_server: str,
    batch_size: int,
    speed_factor: float,
    fragment_interval: float,
) -> SimpleNamespace:
    opt = SimpleNamespace(
        avatar_id="choice_audio_shared",
        tts="gpt-sovits",
        REF_FILE=ref_file,
        REF_TEXT=ref_text,
        TTS_SERVER=tts_server,
        TTS_TEXT_LANG="zh",
        TTS_PROMPT_LANG="zh",
        TTS_SPLIT_METHOD="cut5",
        TTS_BATCH_SIZE=batch_size,
        TTS_SPEED_FACTOR=speed_factor,
        TTS_FRAGMENT_INTERVAL=fragment_interval,
    )
    return SimpleNamespace(opt=opt, _choice_state={"tree_id": None})


def main() -> int:
    args = parse_args()
    voice_texts = load_voice_texts()
    female_ref = normalize_ref_path(args.female_ref)
    male_ref = normalize_ref_path(args.male_ref)
    female_ref_text = args.female_ref_text if args.female_ref_text is not None else voice_texts.get("DongQing_6s", "")
    male_ref_text = args.male_ref_text if args.male_ref_text is not None else voice_texts.get("SaBeining", "")

    for label, ref_file, ref_text in (
        ("female", female_ref, female_ref_text),
        ("male", male_ref, male_ref_text),
    ):
        if not Path(ref_file).exists():
            raise FileNotFoundError(f"{label} reference audio not found: {ref_file}")
        if not ref_text:
            raise ValueError(f"{label} reference text is empty; pass --{label}_ref_text")

    voices = args.voices or list(VOICE_GROUPS)
    provider = StaticChoiceTreeProvider(PROJECT_ROOT / "data" / "choice_trees")
    tree_payload = provider.load_tree(args.tree_id)
    orchestrator = ChoiceOrchestrator(str(PROJECT_ROOT))
    audio_cache = ChoiceAudioCache(PROJECT_ROOT / "cache" / "choice_audio")
    synthesizer = ChoiceAudioSynthesizer()

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tree_id": args.tree_id,
        "tts": "gpt-sovits",
        "tts_server": args.tts_server,
        "batch_size": args.batch_size,
        "speed_factor": args.speed_factor,
        "fragment_interval": args.fragment_interval,
        "female_ref_file": female_ref,
        "female_ref_text": female_ref_text,
        "male_ref_file": male_ref,
        "male_ref_text": male_ref_text,
        "export_wav": bool(args.export_wav),
        "wav_dir": str(Path(args.wav_dir).expanduser()),
        "items": [],
    }

    failed = 0
    ready = 0
    skipped = 0
    for voice in voices:
        ref_file = male_ref if voice == "male" else female_ref
        ref_text = male_ref_text if voice == "male" else female_ref_text
        avatar_session = make_session(
            ref_file,
            ref_text,
            args.tts_server,
            args.batch_size,
            args.speed_factor,
            args.fragment_interval,
        )
        avatar_session._choice_state["tree_id"] = args.tree_id
        tts_options = {"ref_file": ref_file, "ref_text": ref_text}

        for node in iter_nodes(tree_payload):
            tts_text = orchestrator._node_tts_text(node)
            if not tts_text.strip():
                continue
            cache_key = orchestrator._cache_key(avatar_session, node, tts_options)
            cache_path = audio_cache._file_path(cache_key)
            item = {
                "voice_group": voice,
                "node_id": node["node_id"],
                "cache_key": cache_key,
                "cache_path": str(cache_path),
                "ref_file": ref_file,
                "ref_text": ref_text,
                "batch_size": args.batch_size,
                "speed_factor": args.speed_factor,
                "fragment_interval": args.fragment_interval,
                "text": tts_text,
            }

            if cache_path.exists() and not args.force:
                if args.export_wav:
                    cached_audio = audio_cache.get(cache_key)
                    if cached_audio is not None and cached_audio.size > 0:
                        wav_path = export_wav(args.wav_dir, voice, node["node_id"], cached_audio)
                        item["wav_path"] = str(wav_path)
                skipped += 1
                item["status"] = "skipped_ready"
                manifest["items"].append(item)
                status(f"skip ready: voice={voice} node={node['node_id']}")
                continue

            if args.dry_run:
                item["status"] = "dry_run"
                manifest["items"].append(item)
                status(f"dry-run: voice={voice} node={node['node_id']} key={cache_key}")
                continue

            status(f"synthesize start: voice={voice} node={node['node_id']}")
            audio = synthesizer.synthesize(avatar_session, tts_text, tts_options)
            if audio is None or audio.size <= 0:
                failed += 1
                item["status"] = "failed"
                item["message"] = "TTS returned empty audio"
                manifest["items"].append(item)
                status(f"synthesize failed: voice={voice} node={node['node_id']}")
                continue

            audio_cache.set(cache_key, audio, persist=True)
            if args.export_wav:
                wav_path = export_wav(args.wav_dir, voice, node["node_id"], audio)
                item["wav_path"] = str(wav_path)
            ready += 1
            item["status"] = "ready"
            item["duration_seconds"] = round(float(audio.shape[0]) / 16000.0, 3)
            item["audio_samples"] = int(audio.shape[0])
            manifest["items"].append(item)
            status(
                f"synthesize done: voice={voice} node={node['node_id']} "
                f"duration={item['duration_seconds']}s"
            )

    manifest_path = PROJECT_ROOT / "cache" / "choice_audio" / f"{args.tree_id}_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    status(f"finished: ready={ready} skipped={skipped} failed={failed} manifest={manifest_path}")
    return 1 if failed else 0


def export_wav(wav_dir: str, avatar_id: str, node_id: str, audio) -> Path:
    root = Path(wav_dir).expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    target = root / avatar_id / f"{node_id}.wav"
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(target, audio, 16000)
    return target


if __name__ == "__main__":
    raise SystemExit(main())
