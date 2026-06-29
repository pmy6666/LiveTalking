import asyncio
import hashlib
import json
import os
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock, Semaphore
from typing import Any, Dict, Optional

import numpy as np
import resampy
import requests
import soundfile as sf
from io import BytesIO
from requests.exceptions import ChunkedEncodingError

from choice.echomimicv3_cache import ChoiceEchoMimicV3CacheStore
from utils.logger import logger


def _request_without_env_proxy(method: str, url: str, **kwargs):
    session = requests.Session()
    session.trust_env = False
    try:
        return session.request(method, url, **kwargs)
    finally:
        session.close()


class StaticChoiceTreeProvider:
    def __init__(self, tree_root: Path):
        self.tree_root = tree_root
        self._tree_cache: dict[str, dict[str, Any]] = {}
        self._lock = Lock()

    def load_tree(self, tree_id: str) -> dict[str, Any]:
        with self._lock:
            cached = self._tree_cache.get(tree_id)
            if cached is not None:
                return cached

            tree_path = self.tree_root / f"{tree_id}.json"
            if not tree_path.exists():
                raise FileNotFoundError(f"choice tree not found: {tree_path}")

            with tree_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)

            nodes = payload.get("nodes", [])
            payload["node_map"] = {node["node_id"]: node for node in nodes}
            self._tree_cache[tree_id] = payload
            return payload

    def get_root(self, tree_id: str) -> dict[str, Any]:
        tree = self.load_tree(tree_id)
        root_id = tree.get("root_node_id")
        if not root_id:
            raise ValueError(f"choice tree {tree_id} missing root_node_id")
        return self.get_node(tree_id, root_id)

    def get_node(self, tree_id: str, node_id: str) -> dict[str, Any]:
        tree = self.load_tree(tree_id)
        node = tree["node_map"].get(node_id)
        if node is None:
            raise KeyError(f"choice node not found: {tree_id}/{node_id}")
        return node


class ChoiceAudioCache:
    def __init__(self, cache_root: Path, max_items: int = 64):
        self.cache_root = cache_root
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.max_items = max_items
        self._memory: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = Lock()

    def _file_path(self, cache_key: str) -> Path:
        return self.cache_root / f"{cache_key}.npy"

    def get(self, cache_key: str) -> Optional[np.ndarray]:
        with self._lock:
            cached = self._memory.get(cache_key)
            if cached is not None:
                self._memory.move_to_end(cache_key)
                return cached

        path = self._file_path(cache_key)
        if not path.exists():
            return None

        try:
            audio = np.load(path, allow_pickle=False)
        except Exception:
            logger.exception("load choice audio cache failed: %s", path)
            return None

        self.set(cache_key, audio, persist=False)
        return audio

    def set(self, cache_key: str, audio_stream: np.ndarray, persist: bool = True):
        audio_stream = np.asarray(audio_stream, dtype=np.float32)
        with self._lock:
            self._memory[cache_key] = audio_stream
            self._memory.move_to_end(cache_key)
            while len(self._memory) > self.max_items:
                self._memory.popitem(last=False)

        if persist:
            try:
                np.save(self._file_path(cache_key), audio_stream, allow_pickle=False)
            except Exception:
                logger.exception("persist choice audio cache failed: %s", cache_key)


class ChoiceAudioSynthesizer:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._gpt_sovits_lock = Semaphore(1)

    def synthesize(self, avatar_session, text: str, tts_options: dict) -> Optional[np.ndarray]:
        tts_name = getattr(avatar_session.opt, "tts", "")
        if tts_name == "gpt-sovits":
            return self._synthesize_gpt_sovits(avatar_session, text, tts_options)
        if tts_name == "edgetts":
            return self._synthesize_edge_tts(avatar_session, text, tts_options)
        logger.info("choice audio prefetch skipped for unsupported tts=%s", tts_name)
        return None

    def _read_audio_bytes(self, byte_stream: BytesIO) -> np.ndarray:
        stream, sample_rate = sf.read(byte_stream)
        stream = stream.astype(np.float32)
        if stream.ndim > 1:
            stream = stream[:, 0]
        if sample_rate != self.sample_rate and stream.shape[0] > 0:
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)
        return stream.astype(np.float32, copy=False)

    def _decode_pcm(self, pcm_bytes: bytes, sample_rate: int) -> np.ndarray:
        if not pcm_bytes:
            return np.zeros(0, dtype=np.float32)
        aligned_len = len(pcm_bytes) - (len(pcm_bytes) % 2)
        if aligned_len <= 0:
            return np.zeros(0, dtype=np.float32)
        stream = np.frombuffer(pcm_bytes[:aligned_len], dtype=np.int16).astype(np.float32) / 32768.0
        if sample_rate != self.sample_rate and stream.shape[0] > 0:
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)
        return stream.astype(np.float32, copy=False)

    def _synthesize_gpt_sovits(self, avatar_session, text: str, tts_options: dict) -> Optional[np.ndarray]:
        req = {
            "text": text,
            "text_lang": getattr(avatar_session.opt, "TTS_TEXT_LANG", "zh"),
            "ref_audio_path": tts_options.get("ref_file", avatar_session.opt.REF_FILE),
            "prompt_text": tts_options.get("ref_text", avatar_session.opt.REF_TEXT),
            "prompt_lang": getattr(avatar_session.opt, "TTS_PROMPT_LANG", "zh"),
            "text_split_method": getattr(avatar_session.opt, "TTS_SPLIT_METHOD", "cut5"),
            "media_type": "wav",
            "streaming_mode": 0,
            "batch_size": int(getattr(avatar_session.opt, "TTS_BATCH_SIZE", 1)),
            "speed_factor": float(getattr(avatar_session.opt, "TTS_SPEED_FACTOR", 1.08)),
            "fragment_interval": float(getattr(avatar_session.opt, "TTS_FRAGMENT_INTERVAL", 0.1)),
        }

        server_url = getattr(avatar_session.opt, "TTS_SERVER", "http://127.0.0.1:9880")
        with self._gpt_sovits_lock:
            try:
                res = _request_without_env_proxy(
                    "POST",
                    f"{server_url}/tts",
                    json=req,
                    timeout=(5, 60),
                )
                if res.status_code != 200:
                    logger.warning("choice gpt-sovits synth failed: %s %s", res.status_code, res.text)
                    return None
                return self._read_audio_bytes(BytesIO(res.content))
            except ChunkedEncodingError:
                logger.warning("choice gpt-sovits prefetch stream closed early; skip this node cache")
                return None
            except requests.RequestException:
                logger.exception("choice gpt-sovits prefetch request failed")
                return None

    def _synthesize_edge_tts(self, avatar_session, text: str, tts_options: dict) -> Optional[np.ndarray]:
        import edge_tts

        voice = tts_options.get("ref_file", avatar_session.opt.REF_FILE)

        async def _collect_audio() -> bytes:
            communicate = edge_tts.Communicate(text, voice)
            chunks = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    chunks.extend(chunk["data"])
            return bytes(chunks)

        audio_bytes = asyncio.new_event_loop().run_until_complete(_collect_audio())
        if not audio_bytes:
            return None
        return self._read_audio_bytes(BytesIO(audio_bytes))


class ChoiceOrchestrator:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.provider = StaticChoiceTreeProvider(self.project_root / "data" / "choice_trees")
        self.audio_cache = ChoiceAudioCache(self.project_root / "cache" / "choice_audio")
        self.echomimicv3_cache = ChoiceEchoMimicV3CacheStore(self.project_root)
        self.audio_synth = ChoiceAudioSynthesizer()
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="choice-prefetch")
        self.playback_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="choice-play")
        self._prefetching: set[str] = set()
        self._prefetch_lock = Lock()
        self.enable_prefetch = os.getenv("LIVETALKING_CHOICE_PREFETCH", "0") == "1"
        self._tts_term_map = {
            "WebRTC": "实时音视频",
            "webrtc": "实时音视频",
            "TTS": "语音合成",
            "tts": "语音合成",
            "ASR": "语音识别",
            "asr": "语音识别",
            "PCM": "音频数据",
            "pcm": "音频数据",
            "API": "接口",
            "api": "接口",
            "choice": "选项模式",
            "Choice": "选项模式",
            "init": "初始化",
            "select": "选择",
            "state": "状态",
            "reset": "重置",
            "human": "会话接口",
            "type": "类型字段",
        }

    def _ensure_state(self, avatar_session):
        state = getattr(avatar_session, "_choice_state", None)
        if state is None:
            state = {
                "mode": "choice",
                "tree_id": None,
                "current_node_id": None,
                "path": [],
                "last_choices": [],
            }
            avatar_session._choice_state = state
        return state

    @staticmethod
    def _build_choices(node: dict[str, Any]) -> list[dict[str, str]]:
        choices = []
        for item in node.get("choices", []):
            choices.append(
                {
                    "choice_id": item["choice_id"],
                    "choice_text": item["choice_text"],
                    "child_node_id": item["child_node_id"],
                }
            )
        return choices

    def _tts_options(self, avatar_session) -> dict[str, Any]:
        return {
            "ref_file": getattr(avatar_session.opt, "REF_FILE", ""),
            "ref_text": getattr(avatar_session.opt, "REF_TEXT", ""),
        }

    def _normalize_tts_text(self, text: str) -> str:
        normalized = text or ""
        for source, target in self._tts_term_map.items():
            normalized = normalized.replace(source, target)

        # 清掉残余的纯 ASCII 词，避免 GPT-SoVITS 走到英文文本清洗链路。
        normalized = re.sub(r"[A-Za-z_][A-Za-z0-9_./-]*", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        normalized = normalized.replace("、。", "、")
        normalized = normalized.replace("，，", "，")
        return normalized

    def _node_tts_text(self, node: dict[str, Any]) -> str:
        explicit_tts_text = node.get("tts_text")
        if explicit_tts_text:
            return explicit_tts_text
        return self._normalize_tts_text(node.get("answer_text", ""))

    def _cache_key(self, avatar_session, node: dict[str, Any], tts_options: dict[str, Any]) -> str:
        tts_text = self._node_tts_text(node)
        payload = json.dumps(
            {
                "tree_id": getattr(avatar_session, "_choice_state", {}).get("tree_id"),
                "node_id": node["node_id"],
                "tts": getattr(avatar_session.opt, "tts", ""),
                "ref_file": tts_options.get("ref_file", ""),
                "ref_text": tts_options.get("ref_text", ""),
                "batch_size": int(getattr(avatar_session.opt, "TTS_BATCH_SIZE", 1)),
                "speed_factor": float(getattr(avatar_session.opt, "TTS_SPEED_FACTOR", 1.08)),
                "fragment_interval": float(getattr(avatar_session.opt, "TTS_FRAGMENT_INTERVAL", 0.1)),
                "text": tts_text,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _serialize_node(self, avatar_session, node: dict[str, Any]) -> dict[str, Any]:
        tts_options = self._tts_options(avatar_session)
        cache_key = self._cache_key(avatar_session, node, tts_options)
        cache_hit = self.audio_cache.get(cache_key) is not None
        return {
            "node_id": node["node_id"],
            "answer_text": node.get("answer_text", ""),
            "display_text": node.get("display_text") or node.get("answer_text", ""),
            "tts_text": self._node_tts_text(node),
            "choices": self._build_choices(node),
            "audio_cache_hit": cache_hit,
        }

    def init_session(self, avatar_session, tree_id: str) -> dict[str, Any]:
        state = self._ensure_state(avatar_session)
        root = self.provider.get_root(tree_id)
        state["tree_id"] = tree_id
        state["current_node_id"] = root["node_id"]
        state["path"] = [root["node_id"]]
        state["last_choices"] = [choice["choice_id"] for choice in root.get("choices", [])]
        if self.enable_prefetch:
            self.prefetch_children(avatar_session, root)
        return {
            "mode": "choice",
            "tree_id": tree_id,
            "path": state["path"],
            "current": self._serialize_node(avatar_session, root),
        }

    def get_state(self, avatar_session) -> dict[str, Any]:
        state = self._ensure_state(avatar_session)
        if not state.get("tree_id") or not state.get("current_node_id"):
            return {"initialized": False}
        node = self.provider.get_node(state["tree_id"], state["current_node_id"])
        return {
            "initialized": True,
            "mode": "choice",
            "tree_id": state["tree_id"],
            "path": state["path"],
            "current": self._serialize_node(avatar_session, node),
        }

    def reset_session(self, avatar_session) -> dict[str, Any]:
        state = self._ensure_state(avatar_session)
        tree_id = state.get("tree_id") or "default_choice_tree"
        return self.init_session(avatar_session, tree_id)

    def select_choice(self, avatar_session, choice_id: str, interrupt: bool = True) -> dict[str, Any]:
        state = self._ensure_state(avatar_session)
        tree_id = state.get("tree_id")
        current_node_id = state.get("current_node_id")
        if not tree_id or not current_node_id:
            raise ValueError("choice mode not initialized")

        current_node = self.provider.get_node(tree_id, current_node_id)
        selected = None
        for choice in current_node.get("choices", []):
            if choice["choice_id"] == choice_id:
                selected = choice
                break
        if selected is None:
            raise ValueError("invalid choice_id for current node")

        next_node = self.provider.get_node(tree_id, selected["child_node_id"])
        state["current_node_id"] = next_node["node_id"]
        state["path"].append(next_node["node_id"])
        state["last_choices"] = [choice["choice_id"] for choice in next_node.get("choices", [])]

        if interrupt:
            avatar_session.flush_talk()

        playback_result = self._play_or_enqueue(avatar_session, next_node)
        if self.enable_prefetch:
            self.prefetch_children(avatar_session, next_node)

        return {
            "selected_choice_id": choice_id,
            "path": state["path"],
            "current": self._serialize_node(avatar_session, next_node),
            "audio_cache_hit": playback_result["audio_cache_hit"],
            "video_cache_hit": playback_result["video_cache_hit"],
            "cache_mode": playback_result["cache_mode"],
        }

    def _play_or_enqueue(self, avatar_session, node: dict[str, Any]) -> dict[str, Any]:
        tts_options = self._tts_options(avatar_session)
        cache_key = self._cache_key(avatar_session, node, tts_options)
        cached_audio = self.audio_cache.get(cache_key)
        tts_text = self._node_tts_text(node)
        datainfo = {
            "choice": {"node_id": node["node_id"]},
            "text": tts_text,
        }
        playback_token = avatar_session.current_playback_token()

        if getattr(avatar_session.opt, "model", "") == "echomimicv3":
            video_result = self._play_echomimicv3_cache_if_ready(
                avatar_session,
                node,
                tts_text,
                datainfo,
                playback_token,
            )
            if video_result is not None:
                return video_result

        if cached_audio is not None:
            self.playback_executor.submit(
                avatar_session.play_audio_stream,
                cached_audio,
                datainfo,
                playback_token,
            )
            return {
                "audio_cache_hit": True,
                "video_cache_hit": False,
                "cache_mode": "audio_cache",
            }

        self.playback_executor.submit(
            self._synthesize_and_play_node_audio,
            avatar_session,
            node,
            tts_options,
            cache_key,
            datainfo,
            tts_text,
            playback_token,
        )
        return {
            "audio_cache_hit": False,
            "video_cache_hit": False,
            "cache_mode": "realtime_fallback",
        }

    def _play_echomimicv3_cache_if_ready(
        self,
        avatar_session,
        node: dict[str, Any],
        tts_text: str,
        datainfo: dict[str, Any],
        playback_token: int,
    ) -> Optional[dict[str, Any]]:
        if not hasattr(avatar_session, "play_cached_video_segment"):
            return None

        tree_id = self._ensure_state(avatar_session).get("tree_id")
        if not tree_id:
            return None

        candidate_metas = self.echomimicv3_cache.get_candidate_metas_by_node(tree_id, node["node_id"])
        if not candidate_metas:
            logger.info("choice EchoMimicV3 video cache miss: %s/%s", tree_id, node["node_id"])
            return None

        cached_meta = None
        for meta in candidate_metas:
            if self.echomimicv3_cache.is_compatible(avatar_session, node, tts_text, meta):
                cached_meta = meta
                break
        if cached_meta is None:
            logger.info(
                "choice EchoMimicV3 video cache incompatible: %s/%s candidates=%d",
                tree_id,
                node["node_id"],
                len(candidate_metas),
            )
            return None

        cached_segment = self.echomimicv3_cache.get(tree_id, cached_meta.get("cache_key", ""))
        if cached_segment is None:
            logger.info(
                "choice EchoMimicV3 video cache payload missing: %s/%s cache_key=%s",
                tree_id,
                node["node_id"],
                cached_meta.get("cache_key", ""),
            )
            return None

        datainfo = dict(datainfo)
        datainfo["choice_video_cache"] = {
            "tree_id": tree_id,
            "node_id": node["node_id"],
            "cache_key": cached_segment["meta"].get("cache_key", ""),
        }
        self.playback_executor.submit(
            avatar_session.play_cached_video_segment,
            cached_segment["frames"],
            cached_segment["audio"],
            datainfo,
            playback_token,
        )
        logger.info("choice EchoMimicV3 video cache hit: %s/%s", tree_id, node["node_id"])
        return {
            "audio_cache_hit": True,
            "video_cache_hit": True,
            "cache_mode": "echomimicv3_precomputed",
        }

    def _synthesize_and_play_node_audio(
        self,
        avatar_session,
        node: dict[str, Any],
        tts_options: dict[str, Any],
        cache_key: str,
        datainfo: dict[str, Any],
        tts_text: str,
        playback_token: int,
    ):
        if playback_token != avatar_session.current_playback_token():
            return

        cached_audio = self.audio_cache.get(cache_key)
        if cached_audio is not None:
            avatar_session.play_audio_stream(cached_audio, datainfo, playback_token)
            return

        audio_stream = self.audio_synth.synthesize(avatar_session, tts_text, tts_options)
        if playback_token != avatar_session.current_playback_token():
            return

        if audio_stream is not None and audio_stream.size > 0:
            self.audio_cache.set(cache_key, audio_stream)
            avatar_session.play_audio_stream(audio_stream, datainfo, playback_token)
            return

        # 回退到原始流式 TTS，至少保证还能播报。
        datainfo = dict(datainfo)
        datainfo["choice_audio_cache"] = {
            "cache_key": cache_key,
            "capture": True,
            "store": self.audio_cache.set,
        }
        avatar_session.put_msg_txt(tts_text, datainfo)

    def prefetch_children(self, avatar_session, node: dict[str, Any]):
        tts_options = self._tts_options(avatar_session)
        tree_id = self._ensure_state(avatar_session)["tree_id"]
        for choice in node.get("choices", []):
            child_node = self.provider.get_node(tree_id, choice["child_node_id"])
            cache_key = self._cache_key(avatar_session, child_node, tts_options)
            if self.audio_cache.get(cache_key) is not None:
                continue
            with self._prefetch_lock:
                if cache_key in self._prefetching:
                    continue
                self._prefetching.add(cache_key)
            self.executor.submit(self._prefetch_child_audio, avatar_session, child_node, tts_options, cache_key)

    def _prefetch_child_audio(self, avatar_session, node: dict[str, Any], tts_options: dict[str, Any], cache_key: str):
        try:
            logger.info("prefetch choice audio start: %s", node["node_id"])
            audio_stream = self.audio_synth.synthesize(avatar_session, self._node_tts_text(node), tts_options)
            if audio_stream is not None and audio_stream.size > 0:
                self.audio_cache.set(cache_key, audio_stream)
                logger.info("prefetch choice audio done: %s", node["node_id"])
            else:
                logger.warning("prefetch choice audio skipped: %s", node["node_id"])
        except Exception:
            logger.exception("prefetch choice audio failed unexpectedly: %s", node["node_id"])
        finally:
            with self._prefetch_lock:
                self._prefetching.discard(cache_key)
