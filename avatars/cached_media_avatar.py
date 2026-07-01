import glob
import os
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import cv2
import numpy as np

from avatars.base_avatar import BaseAvatar
from registry import register
from utils.image import read_imgs, mirror_index
from utils.logger import logger


@dataclass
class CachedMediaAvatarData:
    avatar_id: str
    idle_frames: list[np.ndarray]


def load_model(opt=None):
    return None


def _avatar_asset_candidates(avatar_id: str) -> list[Path]:
    assets_dir = Path("assets") / "avatars"
    aliases = {
        "wav2lip256_avatar1": "avatar1",
        "wav2lip_avatar_1": "avatar1",
        "wav2lip_avatar_2": "avatar2",
        "wav2lip_avatar_3": "avatar3",
        "wav2lip_avatar_4": "avatar4",
        "wav2lip_avatar_5": "avatar5",
    }
    names = [avatar_id]
    alias = aliases.get(avatar_id)
    if alias:
        names.insert(0, alias)
    candidates = []
    for name in names:
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            candidates.append(assets_dir / f"{name}{ext}")
    return candidates


def _find_ref_image(avatar_id: str) -> Path:
    avatar_path = Path("data") / "avatars" / avatar_id
    candidates = _avatar_asset_candidates(avatar_id)
    candidates.extend(
        [
            avatar_path / "echomimicv3" / "ref.png",
            avatar_path / "echomimicv3" / "ref.jpg",
            avatar_path / "ref.png",
            avatar_path / "ref.jpg",
        ]
    )
    candidates.extend(Path(path) for path in sorted(glob.glob(str(avatar_path / "full_imgs" / "*.[jpJP][pnPN]*[gG]"))))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"cached_media avatar image not found for avatar_id={avatar_id}")


def _load_idle_frames(avatar_id: str, ref_image_path: Path) -> list[np.ndarray]:
    avatar_path = Path("data") / "avatars" / avatar_id
    image_paths: list[str] = []
    for idle_dir in (
        avatar_path / "echomimicv3" / "idle_frames_two_stage_silence",
        avatar_path / "echomimicv3" / "idle_frames",
        avatar_path / "full_imgs",
    ):
        for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            image_paths.extend(glob.glob(str(idle_dir / pattern)))
        if image_paths:
            break

    if image_paths:
        frames = read_imgs(sorted(image_paths))
        if frames:
            logger.info("cached_media idle frames loaded: frames=%d", len(frames))
            return frames

    frame = cv2.imread(str(ref_image_path))
    if frame is None:
        raise FileNotFoundError(f"cannot read cached_media avatar image: {ref_image_path}")
    logger.info("cached_media idle frame uses reference image: %s", ref_image_path)
    return [frame]


def load_avatar(avatar_id):
    ref_image_path = _find_ref_image(avatar_id)
    return CachedMediaAvatarData(
        avatar_id=avatar_id,
        idle_frames=_load_idle_frames(avatar_id, ref_image_path),
    )


def warm_up(opt=None, model=None, avatar=None):
    logger.info("cached_media warm_up skipped; no model is loaded.")


@register("avatar", "cached_media")
class CachedMediaAvatar(BaseAvatar):
    def __init__(self, opt, model, avatar: CachedMediaAvatarData):
        super().__init__(opt)
        self.avatar = avatar
        self.frame_list_cycle = [self._normalize_frame(frame) for frame in avatar.idle_frames]
        self._playback_frames = queue.Queue(maxsize=256)
        self._idle_index = 0
        self._state = "IDLE"

    def _resolve_output_size(self):
        frame = self.frame_list_cycle[0]
        height, width = frame.shape[:2]
        sample_size = getattr(self.opt, "echomimicv3_sample_size", None) or []
        if len(sample_size) >= 2 and int(sample_size[0]) > 0 and int(sample_size[1]) > 0:
            return int(sample_size[1]), int(sample_size[0])
        return width, height

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"cached_media frame must be HxWx3, got {frame.shape}")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if not hasattr(self, "_output_frame_size"):
            return frame
        target_w, target_h = self._output_frame_size
        if frame.shape[1] != target_w or frame.shape[0] != target_h:
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return frame

    def _next_idle_frame(self) -> np.ndarray:
        idx = mirror_index(len(self.frame_list_cycle), self._idle_index)
        self._idle_index += 1
        return self.frame_list_cycle[idx]

    def _audio_chunks_for_frame(self, audio: np.ndarray, index: int, total_frames: int, datainfo: dict) -> list[tuple[np.ndarray, dict]]:
        chunks = []
        for sub in range(2):
            start = (index * 2 + sub) * self.chunk
            end = start + self.chunk
            chunk = np.zeros(self.chunk, dtype=np.float32)
            source = audio[start:min(end, audio.shape[0])]
            if source.size > 0:
                chunk[:source.shape[0]] = source
            event = {}
            if index == 0 and sub == 0:
                event.update(datainfo)
                event["status"] = "start"
            if index == total_frames - 1 and sub == 1:
                event.update(datainfo)
                event["status"] = "end"
            chunks.append((chunk, event))
        return chunks

    def play_cached_video_segment(
        self,
        frames: np.ndarray,
        audio: np.ndarray,
        datainfo: dict = None,
        playback_token: int = None,
    ):
        datainfo = dict(datainfo or {})
        if playback_token is None:
            playback_token = self.current_playback_token()
        if playback_token != self.current_playback_token():
            return

        frames = np.asarray(frames)
        audio = np.asarray(audio, dtype=np.float32)
        if frames.shape[0] <= 0 or audio.size <= 0:
            return

        expected_audio_samples = frames.shape[0] * 2 * self.chunk
        aligned_audio = np.zeros(expected_audio_samples, dtype=np.float32)
        copy_samples = min(audio.shape[0], expected_audio_samples)
        aligned_audio[:copy_samples] = audio[:copy_samples]

        logger.info(
            "cached_media playback enqueue: frames=%d audio_samples=%d aligned_audio_samples=%d",
            frames.shape[0],
            audio.shape[0],
            expected_audio_samples,
        )
        for index, frame in enumerate(frames):
            if playback_token != self.current_playback_token():
                return
            audio_chunks = self._audio_chunks_for_frame(aligned_audio, index, frames.shape[0], datainfo)
            self._playback_frames.put((self._normalize_frame(frame).copy(), audio_chunks, playback_token))

    def flush_talk(self):
        super().flush_talk()
        self._clear_queue(self._playback_frames)
        self._state = "IDLE"

    def render(self, quit_event: Event):
        self.quit_event = quit_event
        self.init_customindex()
        self.output.start()
        self._output_frame_size = self._resolve_output_size()
        self.frame_list_cycle = [self._normalize_frame(frame) for frame in self.frame_list_cycle]

        frame_interval = 1.0 / self.opt.fps
        try:
            while not quit_event.is_set():
                started_at = time.perf_counter()
                try:
                    frame, audio_chunks, token = self._playback_frames.get_nowait()
                    if token != self.current_playback_token():
                        continue
                    self._state = "PLAYING"
                    self.speaking = True
                except queue.Empty:
                    frame = self._next_idle_frame()
                    audio_chunks = [
                        (np.zeros(self.chunk, dtype=np.float32), {}),
                        (np.zeros(self.chunk, dtype=np.float32), {}),
                    ]
                    self._state = "IDLE"
                    self.speaking = False

                frame = self._normalize_frame(frame).copy()
                cv2.putText(frame, "LiveTalking", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 128, 128), 1)
                self.output.push_video_frame(frame)
                self.record_video_data(frame)

                for audio_chunk, eventpoint in audio_chunks:
                    pcm = (np.clip(audio_chunk, -1.0, 1.0) * 32767).astype(np.int16)
                    self.output.push_audio_frame(pcm, eventpoint)
                    self.record_audio_data(pcm)

                buffer_size = self.output.get_buffer_size() if hasattr(self.output, "get_buffer_size") else 0
                sleep_time = max(0.0, frame_interval - (time.perf_counter() - started_at))
                if buffer_size >= 5:
                    sleep_time += 0.04 * buffer_size * 0.8
                time.sleep(sleep_time)
        finally:
            self.output.stop()
            logger.info("cached_media render thread stop")
