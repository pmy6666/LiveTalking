#!/usr/bin/env python3
import argparse
import importlib
import json
import math
import os
import socket
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import registry
from choice.echomimicv3_cache import ChoiceEchoMimicV3CacheStore, hash_choice_tree
from choice.orchestrator import ChoiceAudioSynthesizer, StaticChoiceTreeProvider
from choice.voice_profiles import apply_voice_profile
from utils.logger import logger


def status(message: str):
    print(f"[choice-cache] {message}", flush=True)
    logger.info(message)


def parse_script_args():
    parser = argparse.ArgumentParser(
        description="Precompute EchoMimicV3 video caches for LiveTalking choice trees",
        add_help=True,
    )
    parser.add_argument("--tree_id", default="default_choice_tree", help="choice tree id under data/choice_trees")
    parser.add_argument("--force", type=int, default=0, help="rebuild existing ready cache")
    parser.add_argument("--limit", type=int, default=0, help="only process first N nodes, 0 means all")
    parser.add_argument("--skip_root", type=int, default=0, help="skip root node")
    parser.add_argument("--dry_run", type=int, default=0, help="only print cache plan")
    script_args, server_args = parser.parse_known_args()
    return script_args, server_args


def parse_server_opt(server_args: list[str]):
    from config import parse_args

    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0]] + server_args
        return parse_args()
    finally:
        sys.argv = old_argv


def apply_echomimicv3_defaults(opt):
    opt.model = "echomimicv3"
    explicit_ref_file = bool(getattr(opt, "_explicit_ref_file_arg", False))
    explicit_ref_text = bool(getattr(opt, "_explicit_ref_text_arg", False))
    if explicit_ref_file:
        status(
            "keep explicit TTS reference: "
            f"avatar={opt.avatar_id} ref_file={getattr(opt, 'REF_FILE', '')}"
        )
    else:
        voice_profile = apply_voice_profile(opt, opt.avatar_id)
        if voice_profile:
            status(
                "apply avatar voice profile: "
                f"avatar={opt.avatar_id} profile={voice_profile['profile']} ref_file={voice_profile['ref_file']}"
            )
    if explicit_ref_file and not explicit_ref_text:
        status("explicit --REF_FILE was provided without --REF_TEXT; GPT-SoVITS clone quality may degrade")
    elif explicit_ref_file and not Path(getattr(opt, "REF_FILE", "")).exists():
        status(
            "explicit --REF_FILE does not exist: "
            f"{getattr(opt, 'REF_FILE', '')}"
        )
    wav2vec_dir = PROJECT_ROOT / "chinese-wav2vec2-base"
    if not wav2vec_dir.exists():
        raise FileNotFoundError(f"required chinese-wav2vec2-base not found: {wav2vec_dir}")

    configured_wav2vec_dir = getattr(opt, "echomimicv3_wav2vec_dir", "")
    if configured_wav2vec_dir:
        resolved = Path(configured_wav2vec_dir)
        if not resolved.is_absolute():
            resolved = (PROJECT_ROOT / resolved).resolve()
        if resolved != wav2vec_dir.resolve():
            status(
                "override --echomimicv3_wav2vec_dir "
                f"from {resolved} to {wav2vec_dir.resolve()} for EchoMimicV3 Flash"
            )
    opt.echomimicv3_wav2vec_dir = str(wav2vec_dir.resolve())

    for attr in ("echomimicv3_repo", "echomimicv3_model_dir", "echomimicv3_base_model_dir", "echomimicv3_transformer_path"):
        value = getattr(opt, attr, "")
        if value:
            path = Path(value)
            if not path.is_absolute():
                setattr(opt, attr, str((PROJECT_ROOT / path).resolve()))
    return opt


def load_echomimicv3_session(opt):
    status("importing avatars.echomimicv3_avatar")
    avatar_mod = importlib.import_module("avatars.echomimicv3_avatar")
    status("loading EchoMimicV3 model; this can take several minutes on first run")
    model = avatar_mod.load_model(opt)
    status("loading EchoMimicV3 avatar assets")
    avatar = avatar_mod.load_avatar(opt.avatar_id)
    avatar_mod.warm_up(opt, model, avatar)
    opt.sessionid = "precompute-choice-echomimicv3"
    status("creating precompute avatar session")
    return registry.create("avatar", "echomimicv3", opt=opt, model=model, avatar=avatar), model, avatar


def validate_tts_server(opt):
    if getattr(opt, "tts", "") != "gpt-sovits":
        return

    server_url = getattr(opt, "TTS_SERVER", "http://127.0.0.1:9880")
    parsed = urlparse(server_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    status(f"checking GPT-SoVITS server: {host}:{port}")
    try:
        with socket.create_connection((host, port), timeout=2):
            pass
    except OSError as exc:
        raise RuntimeError(
            f"GPT-SoVITS server is not reachable at {server_url}. "
            "Start ./start_gpt_sovits_v2proplus.sh first, or pass the correct --TTS_SERVER."
        ) from exc
    status("GPT-SoVITS server port is reachable")


def iter_nodes(tree_payload: dict, limit: int, skip_root: bool):
    root_node_id = tree_payload.get("root_node_id")
    count = 0
    for node in tree_payload.get("nodes", []):
        if skip_root and node.get("node_id") == root_node_id:
            continue
        if limit and count >= limit:
            break
        count += 1
        yield node


def _entry_key(item: dict) -> tuple[str, str, str, str]:
    return (
        item.get("avatar_id", ""),
        item.get("ref_file", ""),
        item.get("ref_text", ""),
        item.get("cache_key", ""),
    )


def write_manifest(cache_store, tree_id: str, tree_hash: str, avatar_id: str, results: list[dict], opt):
    manifest_path = cache_store.tree_cache_root(tree_id) / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except Exception:
            payload = {}
    else:
        payload = {}

    items = payload.get("items", {})
    for item in results:
        entry_status = "ready" if item["status"] == "skipped_ready" else item["status"]
        entry = {
            "cache_key": item.get("cache_key", ""),
            "status": entry_status,
            "duration_seconds": item.get("duration_seconds", 0),
            "frames": item.get("frames", 0),
            "audio_samples": item.get("audio_samples", 0),
            "message": item.get("message", ""),
            "avatar_id": avatar_id,
            "ref_file": getattr(opt, "REF_FILE", ""),
            "ref_text": getattr(opt, "REF_TEXT", ""),
        }
        existing = items.get(item["node_id"], [])
        entries = existing if isinstance(existing, list) else [existing]
        entries = [old for old in entries if _entry_key(old) != _entry_key(entry)]
        entries.append(entry)
        items[item["node_id"]] = entries

    payload.update({
        "schema_version": "choice_echomimicv3_v1",
        "tree_id": tree_id,
        "tree_hash": tree_hash,
        "avatar_id": avatar_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "items": items,
    })
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return manifest_path


def main():
    script_args, server_args = parse_script_args()
    opt = parse_server_opt(server_args)
    opt._explicit_ref_file_arg = any(arg == "--REF_FILE" or arg.startswith("--REF_FILE=") for arg in server_args)
    opt._explicit_ref_text_arg = any(arg == "--REF_TEXT" or arg.startswith("--REF_TEXT=") for arg in server_args)
    opt = apply_echomimicv3_defaults(opt)
    os.chdir(PROJECT_ROOT)

    provider = StaticChoiceTreeProvider(PROJECT_ROOT / "data" / "choice_trees")
    tree_payload = provider.load_tree(script_args.tree_id)
    tree_hash = hash_choice_tree(tree_payload)
    cache_store = ChoiceEchoMimicV3CacheStore(PROJECT_ROOT)

    status(f"project_root={PROJECT_ROOT}")
    status(f"tree_id={script_args.tree_id}, tree_hash={tree_hash}")
    status(f"avatar_id={opt.avatar_id}")
    status(f"ref_file={getattr(opt, 'REF_FILE', '')}")
    status(f"ref_text={getattr(opt, 'REF_TEXT', '')}")
    status(f"wav2vec_dir={opt.echomimicv3_wav2vec_dir}")
    status(f"base_model_dir={opt.echomimicv3_base_model_dir or opt.echomimicv3_model_dir}")
    status(f"transformer_path={opt.echomimicv3_transformer_path}")

    if script_args.dry_run:
        status("dry_run=1, skip EchoMimicV3 model loading")
        results = []
        for node in iter_nodes(tree_payload, script_args.limit, bool(script_args.skip_root)):
            tts_text = node.get("tts_text") or node.get("answer_text") or ""
            status(f"dry-run node={node['node_id']} text_len={len(tts_text)}")
            results.append({"node_id": node["node_id"], "status": "dry_run"})
        return 0

    validate_tts_server(opt)

    status("loading EchoMimicV3 session for choice cache precompute")
    avatar_session, model, avatar = load_echomimicv3_session(opt)
    audio_synth = ChoiceAudioSynthesizer()
    choice_orchestrator = None
    tts_options = {
        "ref_file": getattr(opt, "REF_FILE", ""),
        "ref_text": getattr(opt, "REF_TEXT", ""),
    }

    results = []
    started_at = time.perf_counter()
    for node in iter_nodes(tree_payload, script_args.limit, bool(script_args.skip_root)):
        node_id = node["node_id"]
        if choice_orchestrator is None:
            from choice.orchestrator import ChoiceOrchestrator

            choice_orchestrator = ChoiceOrchestrator(PROJECT_ROOT)
        tts_text = choice_orchestrator._node_tts_text(node)
        if not tts_text.strip():
            results.append({"node_id": node_id, "status": "skipped", "message": "empty tts text"})
            continue

        params = cache_store.build_params(avatar_session, script_args.tree_id, node, tts_text)
        cache_key = params["cache_key"]
        if not script_args.force and cache_store.get(script_args.tree_id, cache_key) is not None:
            logger.info("skip ready cache: %s cache_key=%s", node_id, cache_key)
            results.append({"node_id": node_id, "cache_key": cache_key, "status": "skipped_ready"})
            continue

        if script_args.dry_run:
            logger.info("dry-run cache target: %s cache_key=%s", node_id, cache_key)
            results.append({"node_id": node_id, "cache_key": cache_key, "status": "dry_run"})
            continue

        node_started_at = time.perf_counter()
        try:
            status(f"precompute node start: {node_id} cache_key={cache_key}")
            status(f"TTS synth start: {node_id}")
            audio = audio_synth.synthesize(avatar_session, tts_text, tts_options)
            if audio is None or audio.size <= 0:
                raise RuntimeError("TTS returned empty audio")
            status(f"TTS synth done: {node_id}, samples={audio.shape[0]}")

            duration = audio.shape[0] / 16000.0
            target_frames = max(1, int(math.ceil(duration * opt.fps)))
            status(f"EchoMimicV3 generate start: {node_id}, duration={duration:.2f}s, target_frames={target_frames}")
            frames = model.generate_frames(
                avatar.ref_image_path,
                audio,
                params["prompt"],
                params["negative_prompt"],
                max_video_length=target_frames,
            )
            if not frames:
                raise RuntimeError("EchoMimicV3 returned empty frames")
            status(f"EchoMimicV3 generate done: {node_id}, frames={len(frames)}")

            aligned_samples = len(frames) * 2 * avatar_session.chunk
            if audio.shape[0] != aligned_samples:
                aligned_audio = np.zeros(aligned_samples, dtype=np.float32)
                copy_samples = min(audio.shape[0], aligned_samples)
                aligned_audio[:copy_samples] = audio[:copy_samples]
                status(
                    f"EchoMimicV3 cache audio aligned: {node_id}, "
                    f"original_samples={audio.shape[0]}, aligned_samples={aligned_samples}"
                )
                audio = aligned_audio

            cache_store.set(
                script_args.tree_id,
                cache_key,
                params,
                audio,
                frames,
                source_tree_hash=tree_hash,
            )
            elapsed = time.perf_counter() - node_started_at
            status(f"precompute node done: {node_id}, frames={len(frames)}, elapsed={elapsed:.2f}s")
            results.append(
                {
                    "node_id": node_id,
                    "cache_key": cache_key,
                    "status": "ready",
                    "duration_seconds": duration,
                    "frames": len(frames),
                    "audio_samples": int(audio.shape[0]),
                }
            )
        except Exception as exc:
            logger.exception("precompute node failed: %s", node_id)
            results.append(
                {
                    "node_id": node_id,
                    "cache_key": cache_key,
                    "status": "failed",
                    "message": str(exc),
                }
            )

    manifest_path = write_manifest(cache_store, script_args.tree_id, tree_hash, opt.avatar_id, results, opt)
    elapsed = time.perf_counter() - started_at
    ready = sum(1 for item in results if item["status"] == "ready")
    skipped = sum(1 for item in results if item["status"].startswith("skipped"))
    failed = sum(1 for item in results if item["status"] == "failed")
    logger.info(
        "choice EchoMimicV3 precompute finished: ready=%d skipped=%d failed=%d elapsed=%.2fs manifest=%s",
        ready,
        skipped,
        failed,
        elapsed,
        manifest_path,
    )
    status(f"finished: ready={ready} skipped={skipped} failed={failed} elapsed={elapsed:.2f}s manifest={manifest_path}")
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
