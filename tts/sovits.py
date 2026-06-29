import time
import os
import numpy as np
import resampy
import soundfile as sf
import requests
from io import BytesIO
from typing import Iterator
from requests.exceptions import ChunkedEncodingError

from utils.logger import logger
from utils.audio import pcm_to_float32
from .base_tts import BaseTTS, State
from registry import register


def _request_without_env_proxy(method: str, url: str, **kwargs):
    session = requests.Session()
    session.trust_env = False
    try:
        return session.request(method, url, **kwargs)
    finally:
        session.close()

@register("tts", "gpt-sovits")
class SovitsTTS(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.text_lang = getattr(opt, 'TTS_TEXT_LANG', 'zh')
        self.prompt_lang = getattr(opt, 'TTS_PROMPT_LANG', self.text_lang)
        self.media_type = getattr(opt, 'TTS_MEDIA_TYPE', 'ogg')
        self.streaming_mode = getattr(opt, 'GPT_SOVITS_STREAMING_MODE', 2)
        self.text_split_method = getattr(opt, 'TTS_SPLIT_METHOD', 'cut5')
        self.batch_size = int(getattr(opt, 'TTS_BATCH_SIZE', 1))
        self.speed_factor = float(getattr(opt, 'TTS_SPEED_FACTOR', 1.08))
        self.fragment_interval = float(getattr(opt, 'TTS_FRAGMENT_INTERVAL', 0.1))
        self.gpt_model_path = self._resolve_model_path(
            getattr(opt, 'GPT_SOVITS_GPT_MODEL', ''),
            's1v3.ckpt',
        )
        self.sovits_model_path = self._resolve_model_path(
            getattr(opt, 'GPT_SOVITS_SOVITS_MODEL', ''),
            's2Gv2ProPlus.pth',
        )
        self.s2d_model_path = self._resolve_model_path(
            getattr(opt, 'GPT_SOVITS_S2D_MODEL', ''),
            's2Dv2ProPlus.pth',
        )
        self._sync_remote_weights()

    def _resolve_model_path(self, configured_path: str, default_name: str) -> str:
        if configured_path:
            return configured_path
        for default_path in (
            os.path.join(self.project_root, 'models', 'gpt-sovits-v2proplus', default_name),
            os.path.join(self.project_root, default_name),
        ):
            if os.path.exists(default_path):
                return default_path
        return ''

    def txt_to_audio(self,msg:tuple[str, dict]): 
        text,textevent = msg
        ref_file = textevent.get('tts', {}).get('ref_file',self.opt.REF_FILE)
        ref_text = textevent.get('tts', {}).get('ref_text',self.opt.REF_TEXT)
        self.stream_tts(
            self.gpt_sovits(
                text=text,
                reffile=ref_file,
                reftext=ref_text,
                language=self.text_lang,
                server_url=self.opt.TTS_SERVER, #"http://127.0.0.1:5000", #args.server_url,
            ),
            msg
        )

    def _sync_remote_weights(self):
        if self.s2d_model_path:
            logger.info(
                "GPT-SoVITS s2D checkpoint provided but api_v2 inference does not load it directly: %s",
                self.s2d_model_path,
            )
        if self.gpt_model_path:
            self._set_remote_weight("/set_gpt_weights", self.gpt_model_path)
        if self.sovits_model_path:
            self._set_remote_weight("/set_sovits_weights", self.sovits_model_path)

    def _set_remote_weight(self, endpoint: str, weights_path: str):
        if not weights_path:
            return
        if not os.path.exists(weights_path):
            logger.warning("GPT-SoVITS weights path does not exist: %s", weights_path)
            return
        try:
            res = _request_without_env_proxy(
                "GET",
                f"{self.opt.TTS_SERVER}{endpoint}",
                params={"weights_path": weights_path},
                timeout=15,
            )
            if res.status_code != 200:
                logger.warning(
                    "Failed to switch GPT-SoVITS weights via %s: %s %s",
                    endpoint,
                    res.status_code,
                    res.text,
                )
                return
            logger.info("Switched GPT-SoVITS weights via %s -> %s", endpoint, weights_path)
        except Exception:
            logger.exception("Failed to connect GPT-SoVITS service when calling %s", endpoint)

    def gpt_sovits(self, text, reffile, reftext,language, server_url) -> Iterator[bytes]:
        start = time.perf_counter()
        received_any_chunk = False
        req={
            'text':text,
            'text_lang':language,
            'ref_audio_path':reffile,
            'prompt_text':reftext,
            'prompt_lang':self.prompt_lang,
            'text_split_method': self.text_split_method,
            'media_type': self.media_type,
            'streaming_mode': self.streaming_mode,
            'batch_size': self.batch_size,
            'speed_factor': self.speed_factor,
            'fragment_interval': self.fragment_interval,
        }
        # req["text"] = text
        # req["text_language"] = language
        # req["character"] = character
        # req["emotion"] = emotion
        # #req["stream_chunk_size"] = stream_chunk_size  # you can reduce it to get faster response, but degrade quality
        # req["streaming_mode"] = True
        try:
            res = _request_without_env_proxy(
                "POST",
                f"{server_url}/tts",
                json=req,
                stream=True,
            )
            end = time.perf_counter()
            logger.info(f"gpt_sovits Time to make POST: {end-start}s")

            if res.status_code != 200:
                logger.error("Error:%s", res.text)
                return
            first = True
        
            for chunk in res.iter_content(chunk_size=8192):
                logger.info('chunk len:%d',len(chunk))
                if first:
                    end = time.perf_counter()
                    logger.info(f"gpt_sovits Time to first chunk: {end-start}s")
                    first = False
                if chunk and self.state==State.RUNNING:
                    received_any_chunk = True
                    yield chunk
            #print("gpt_sovits response.elapsed:", res.elapsed)
        except ChunkedEncodingError:
            if received_any_chunk:
                logger.warning(
                    "sovits stream closed early after partial audio; tolerating server-side premature close"
                )
                return
            logger.exception(
                "sovits stream closed early before audio was delivered; this is usually a GPT-SoVITS server-side generation/streaming issue"
            )
        except Exception as e:
            logger.exception('sovits')

    def __create_bytes_stream(self, byte_stream):
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        logger.info(f'[INFO]tts audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0]>0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream

    def __streaming_returns_pcm(self) -> bool:
        streaming_mode = self.streaming_mode
        if isinstance(streaming_mode, bool):
            return streaming_mode and self.media_type in {'wav', 'raw'}
        return streaming_mode in {2, 3} and self.media_type in {'wav', 'raw'}

    def __decode_pcm_chunk(self, pcm_bytes: bytes, sample_rate: int) -> np.ndarray:
        if len(pcm_bytes) < 2:
            return np.zeros(0, dtype=np.float32)
        aligned_len = len(pcm_bytes) - (len(pcm_bytes) % 2)
        if aligned_len <= 0:
            return np.zeros(0, dtype=np.float32)
        stream = pcm_to_float32(pcm_bytes[:aligned_len], sample_width=2)
        if sample_rate != self.sample_rate and stream.shape[0] > 0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)
        return stream.astype(np.float32, copy=False)

    def __emit_audio_frames(self, stream: np.ndarray, text: str, textevent: dict, first: bool):
        if stream.size == 0:
            return first, stream

        idx = 0
        streamlen = stream.shape[0]
        while streamlen - idx >= self.chunk and self.state == State.RUNNING:
            eventpoint = {}
            if first:
                eventpoint = {'status': 'start', 'text': text}
                first = False
            eventpoint.update(**textevent)
            self.parent.put_audio_frame(stream[idx:idx + self.chunk], eventpoint)
            idx += self.chunk

        return first, stream[idx:]

    def stream_tts(self,audio_stream,msg:tuple[str, dict]):
        text,textevent = msg
        first = True
        pending_stream = np.zeros(0, dtype=np.float32)

        if self.__streaming_returns_pcm():
            sample_rate = 32000
            pcm_buffer = bytearray()
            expect_wav_header = self.media_type == 'wav'

            for chunk in audio_stream:
                if not chunk or self.state != State.RUNNING:
                    continue
                pcm_buffer.extend(chunk)

                if expect_wav_header:
                    if len(pcm_buffer) < 44:
                        continue
                    if pcm_buffer[:4] != b'RIFF' or pcm_buffer[8:12] != b'WAVE':
                        logger.warning('Unexpected GPT-SoVITS stream header, falling back to raw PCM decode.')
                    else:
                        sample_rate = int.from_bytes(pcm_buffer[24:28], byteorder='little', signed=False)
                        logger.info('Detected streaming WAV header from GPT-SoVITS, sample_rate=%d', sample_rate)
                    del pcm_buffer[:44]
                    expect_wav_header = False

                aligned_len = len(pcm_buffer) - (len(pcm_buffer) % 2)
                if aligned_len <= 0:
                    continue

                stream = self.__decode_pcm_chunk(bytes(pcm_buffer[:aligned_len]), sample_rate)
                del pcm_buffer[:aligned_len]
                if pending_stream.size > 0:
                    stream = np.concatenate((pending_stream, stream))
                first, pending_stream = self.__emit_audio_frames(stream, text, textevent, first)

            if self.state == State.RUNNING and pcm_buffer:
                stream = self.__decode_pcm_chunk(bytes(pcm_buffer), sample_rate)
                if pending_stream.size > 0:
                    stream = np.concatenate((pending_stream, stream))
                first, pending_stream = self.__emit_audio_frames(stream, text, textevent, first)
        else:
            buffer = bytearray()
            for chunk in audio_stream:
                if chunk is not None and len(chunk)>0 and self.state == State.RUNNING:
                    buffer.extend(chunk)

            if buffer and self.state == State.RUNNING:
                byte_stream = BytesIO(buffer)
                stream = self.__create_bytes_stream(byte_stream)
                first, pending_stream = self.__emit_audio_frames(stream, text, textevent, first)

        if self.state == State.RUNNING and pending_stream.size > 0:
            padded = np.zeros(self.chunk, dtype=np.float32)
            padded[:min(self.chunk, pending_stream.shape[0])] = pending_stream[:self.chunk]
            eventpoint = {}
            if first:
                eventpoint = {'status':'start','text':text}
                first = False
            eventpoint.update(**textevent)
            self.parent.put_audio_frame(padded, eventpoint)

        eventpoint={'status':'end','text':text}
        eventpoint.update(**textevent) 
        self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint)
