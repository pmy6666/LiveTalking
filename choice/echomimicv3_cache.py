import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from utils.logger import logger


SCHEMA_VERSION = "choice_echomimicv3_v1"


def _sha1_file(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    digest = hashlib.sha1()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class ChoiceEchoMimicV3CacheStore:
    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root)
        self.cache_root = self.project_root / "cache" / "choice_echomimicv3"

    def tree_cache_root(self, tree_id: str) -> Path:
        return self.cache_root / tree_id

    def build_params(self, avatar_session, tree_id: str, node: dict[str, Any], tts_text: str) -> dict[str, Any]:
        opt = avatar_session.opt
        avatar = getattr(avatar_session, "avatar", None)
        ref_image_path = getattr(avatar, "ref_image_path", "")

        prompt_payload = self._prompt_payload(avatar_session, node, tts_text)
        ref_file = getattr(opt, "REF_FILE", "")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "tree_id": tree_id,
            "node_id": node["node_id"],
            "answer_text": tts_text,
            "avatar_id": getattr(opt, "avatar_id", ""),
            "model": getattr(opt, "model", ""),
            "ref_image_path": ref_image_path,
            "ref_image_sha1": _sha1_file(ref_image_path),
            "tts": getattr(opt, "tts", ""),
            "tts_ref_file": ref_file,
            "tts_ref_file_sha1": _sha1_file(ref_file),
            "tts_ref_text": getattr(opt, "REF_TEXT", ""),
            "prompt": prompt_payload.get("prompt", ""),
            "negative_prompt": prompt_payload.get("negative_prompt", ""),
            "sample_size": list(getattr(opt, "echomimicv3_sample_size", [])),
            "fps": getattr(opt, "fps", 25),
            "num_steps": getattr(opt, "echomimicv3_num_steps", None),
            "guidance_scale": getattr(opt, "echomimicv3_guidance_scale", None),
            "audio_guidance_scale": getattr(opt, "echomimicv3_audio_guidance_scale", None),
            "transformer_path": getattr(opt, "echomimicv3_transformer_path", ""),
            "weight_dtype": getattr(opt, "echomimicv3_weight_dtype", ""),
        }
        payload["cache_key"] = _json_hash(payload)
        return payload

    @staticmethod
    def _prompt_payload(avatar_session, node: dict[str, Any], tts_text: str) -> dict[str, str]:
        prompt_override = node.get("prompt")
        negative_override = node.get("negative_prompt")
        if prompt_override or negative_override:
            avatar = getattr(avatar_session, "avatar", None)
            return {
                "prompt": prompt_override or getattr(avatar, "prompt", ""),
                "negative_prompt": negative_override or getattr(avatar, "negative_prompt", ""),
            }

        if hasattr(avatar_session, "build_choice_cache_prompts"):
            return avatar_session.build_choice_cache_prompts(
                tts_text,
                {
                    "scene": node.get("scene") or node.get("metadata", {}).get("scene", ""),
                    "action": node.get("action") or node.get("metadata", {}).get("action", ""),
                },
            )

        avatar = getattr(avatar_session, "avatar", None)
        return {
            "prompt": getattr(avatar, "prompt", ""),
            "negative_prompt": getattr(avatar, "negative_prompt", ""),
        }

    def cache_dir(self, tree_id: str, cache_key: str) -> Path:
        return self.tree_cache_root(tree_id) / cache_key

    def manifest_path(self, tree_id: str) -> Path:
        return self.tree_cache_root(tree_id) / "manifest.json"

    def get_by_node(self, tree_id: str, node_id: str) -> Optional[dict[str, Any]]:
        candidates = self.get_candidates_by_node(tree_id, node_id)
        return candidates[0] if candidates else None

    def get_candidates_by_node(self, tree_id: str, node_id: str) -> list[dict[str, Any]]:
        candidate_metas = self.get_candidate_metas_by_node(tree_id, node_id)
        candidates = []
        for meta in candidate_metas:
            cache_key = meta.get("cache_key", "")
            payload = self.get(tree_id, cache_key)
            if payload is not None:
                candidates.append(payload)
        return candidates

    def get_candidate_metas_by_node(self, tree_id: str, node_id: str) -> list[dict[str, Any]]:
        manifest_path = self.manifest_path(tree_id)
        if not manifest_path.exists():
            return []
        try:
            with manifest_path.open("r", encoding="utf-8") as file:
                manifest = json.load(file)
            item = manifest.get("items", {}).get(node_id)
            if not item:
                return []
            entries = item if isinstance(item, list) else [item]
            candidate_metas = []
            for entry in entries:
                if not entry or entry.get("status") != "ready":
                    continue
                cache_key = entry.get("cache_key", "")
                if not cache_key:
                    continue
                meta = self.get_meta(tree_id, cache_key)
                if meta is not None:
                    candidate_metas.append(meta)
            return candidate_metas
        except Exception:
            logger.exception("load choice EchoMimicV3 manifest failed: %s", manifest_path)
            return []

    def get_meta(self, tree_id: str, cache_key: str) -> Optional[dict[str, Any]]:
        cache_dir = self.cache_dir(tree_id, cache_key)
        meta_path = cache_dir / "meta.json"
        audio_path = cache_dir / "audio.npy"
        frames_path = cache_dir / "frames.npz"
        if not meta_path.exists() or not audio_path.exists() or not frames_path.exists():
            return None

        try:
            with meta_path.open("r", encoding="utf-8") as file:
                meta = json.load(file)
            if meta.get("schema_version") != SCHEMA_VERSION or meta.get("cache_key") != cache_key:
                return None
            return meta
        except Exception:
            logger.exception("load choice EchoMimicV3 cache meta failed: %s", cache_dir)
            return None

    def get(self, tree_id: str, cache_key: str) -> Optional[dict[str, Any]]:
        cache_dir = self.cache_dir(tree_id, cache_key)
        meta_path = cache_dir / "meta.json"
        audio_path = cache_dir / "audio.npy"
        frames_path = cache_dir / "frames.npz"
        if not meta_path.exists() or not audio_path.exists() or not frames_path.exists():
            return None

        try:
            with meta_path.open("r", encoding="utf-8") as file:
                meta = json.load(file)
            if meta.get("schema_version") != SCHEMA_VERSION or meta.get("cache_key") != cache_key:
                return None
            audio = np.load(audio_path, allow_pickle=False).astype(np.float32, copy=False)
            with np.load(frames_path, allow_pickle=False) as frame_data:
                frames = frame_data["frames"]
            if audio.size <= 0 or frames.shape[0] <= 0:
                return None
            return {"meta": meta, "audio": audio, "frames": frames}
        except Exception:
            logger.exception("load choice EchoMimicV3 cache failed: %s", cache_dir)
            return None

    def is_compatible(self, avatar_session, node: dict[str, Any], tts_text: str, meta: dict[str, Any]) -> bool:
        opt = avatar_session.opt
        avatar = getattr(avatar_session, "avatar", None)
        ref_image_path = getattr(avatar, "ref_image_path", "")
        checks = {
            "schema_version": SCHEMA_VERSION,
            "node_id": node["node_id"],
            "answer_text": tts_text,
            "model": getattr(opt, "model", ""),
            "tts": getattr(opt, "tts", ""),
            "tts_ref_file": getattr(opt, "REF_FILE", ""),
            "tts_ref_file_sha1": _sha1_file(getattr(opt, "REF_FILE", "")),
            "tts_ref_text": getattr(opt, "REF_TEXT", ""),
            "sample_size": list(getattr(opt, "echomimicv3_sample_size", [])),
            "fps": getattr(opt, "fps", 25),
            "num_steps": getattr(opt, "echomimicv3_num_steps", None),
            "guidance_scale": getattr(opt, "echomimicv3_guidance_scale", None),
            "audio_guidance_scale": getattr(opt, "echomimicv3_audio_guidance_scale", None),
            "transformer_path": getattr(opt, "echomimicv3_transformer_path", ""),
            "weight_dtype": getattr(opt, "echomimicv3_weight_dtype", ""),
            "ref_image_path": ref_image_path,
            "ref_image_sha1": _sha1_file(ref_image_path),
        }
        for key, expected in checks.items():
            if meta.get(key) != expected:
                logger.info(
                    "choice EchoMimicV3 cache incompatible: node=%s field=%s cached=%r runtime=%r",
                    node["node_id"],
                    key,
                    meta.get(key),
                    expected,
                )
                return False
        cached_avatar_id = meta.get("avatar_id")
        runtime_avatar_id = getattr(opt, "avatar_id", "")
        if cached_avatar_id != runtime_avatar_id:
            logger.info(
                "choice EchoMimicV3 cache avatar_id alias accepted: node=%s cached=%r runtime=%r ref_image_sha1=%s",
                node["node_id"],
                cached_avatar_id,
                runtime_avatar_id,
                meta.get("ref_image_sha1"),
            )
        return True

    def exists(self, tree_id: str, cache_key: str) -> bool:
        return self.get(tree_id, cache_key) is not None

    def set(
        self,
        tree_id: str,
        cache_key: str,
        params: dict[str, Any],
        audio: np.ndarray,
        frames: list[np.ndarray] | np.ndarray,
        source_tree_hash: str = "",
    ) -> Path:
        tree_root = self.tree_cache_root(tree_id)
        tree_root.mkdir(parents=True, exist_ok=True)
        target_dir = self.cache_dir(tree_id, cache_key)
        tmp_dir = tree_root / f"{cache_key}.tmp-{os.getpid()}-{int(time.time() * 1000)}"
        tmp_dir.mkdir(parents=True, exist_ok=False)

        frame_array = np.asarray(frames)
        audio_array = np.asarray(audio, dtype=np.float32)
        meta = dict(params)
        meta.update(
            {
                "schema_version": SCHEMA_VERSION,
                "cache_key": cache_key,
                "audio_sample_rate": 16000,
                "audio_samples": int(audio_array.shape[0]),
                "frames": int(frame_array.shape[0]),
                "frame_shape": list(frame_array.shape[1:]),
                "duration_seconds": float(audio_array.shape[0] / 16000.0),
                "source_tree_hash": source_tree_hash,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
        )

        try:
            np.save(tmp_dir / "audio.npy", audio_array, allow_pickle=False)
            np.savez_compressed(tmp_dir / "frames.npz", frames=frame_array)
            if frame_array.shape[0] > 0:
                cv2.imwrite(str(tmp_dir / "preview.jpg"), frame_array[0])
            with (tmp_dir / "meta.json").open("w", encoding="utf-8") as file:
                json.dump(meta, file, ensure_ascii=False, indent=2)

            if target_dir.exists():
                self._remove_dir(target_dir)
            os.replace(tmp_dir, target_dir)
            return target_dir
        except Exception:
            self._remove_dir(tmp_dir)
            raise

    @staticmethod
    def _remove_dir(path: Path):
        if not path.exists():
            return
        for child in path.iterdir():
            if child.is_dir():
                ChoiceEchoMimicV3CacheStore._remove_dir(child)
            else:
                child.unlink()
        path.rmdir()


def hash_choice_tree(tree_payload: dict[str, Any]) -> str:
    stripped = {
        "tree_id": tree_payload.get("tree_id"),
        "root_node_id": tree_payload.get("root_node_id"),
        "nodes": tree_payload.get("nodes", []),
    }
    return _json_hash(stripped)
