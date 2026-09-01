# -*- coding: utf-8 -*-
"""STT streaming abstraction for the SIP channel.

Providers:

* ``aliyun``           -- DashScope Paraformer streaming STT (legacy default)
* ``sherpa_zipformer`` -- Local sherpa-onnx streaming Zipformer ASR, with an
  optional KWS (keyword spotting) gate so audio is only decoded after the
  configured wake word is detected.
* ``openai``           -- OpenAI-compatible ``/v1/audio/transcriptions``
  (Whisper) API; utterances are cut by a lightweight energy VAD and
  transcribed one at a time.
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import tempfile
import threading
import time
import wave
from collections import deque
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Optional,
    Protocol,
    runtime_checkable,
)

import httpx

from qwenpaw.constant import MODELS_DIR

from ._audioop_compat import audioop

logger = logging.getLogger(__name__)

SUPPORTED_STT_PROVIDERS = ("aliyun", "sherpa_zipformer", "openai")

_STT_PROVIDER_ALIASES = {
    "local": "sherpa_zipformer",
    "local_zipformer": "sherpa_zipformer",
    "sherpa": "sherpa_zipformer",
    "sherpa-onnx": "sherpa_zipformer",
    "sherpa_onnx": "sherpa_zipformer",
    "zipformer": "sherpa_zipformer",
    "whisper": "openai",
    "whisper_api": "openai",
    "whisper-api": "openai",
    "openai_whisper": "openai",
    "openai-whisper": "openai",
    "openai_compatible": "openai",
}

_OPENAI_ASR_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_OPENAI_ASR_DEFAULT_MODEL = "whisper-1"
_OPENAI_ASR_SAMPLE_RATE = 16000
_OPENAI_ASR_SPEECH_THRESHOLD = 300
_OPENAI_ASR_MIN_SPEECH_SECONDS = 0.2

# Long-lived local-model cache, shared by channels that opt in via
# ``keep_model_loaded=True``. Keys cover every parameter that affects how
# the recognizer/KWS instances are constructed.
_zipformer_models_cache: dict[tuple, dict[str, Any]] = {}
_zipformer_models_lock = threading.Lock()

_ZIPFORMER_MODEL_NAME = (
    "sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-"
    "zh-en-punct-int8-2026-06-05"
)
_KWS_MODEL_NAME = (
    "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
)
_SAMPLE_RATE = 16000
_DEFAULT_PRE_ROLL_SECONDS = 0.5
_DEFAULT_WAKE_ACTIVE_TIMEOUT_SECONDS = 60.0

# Built-in wake words for the bundled zh-en KWS model. The token table is
# phone+ppinyin, so raw CJK characters must be converted to pinyin tokens.
_BUILTIN_KWS_TOKENIZED = {
    "小爱同学": "x iǎo ài t óng x ué",
    "小艺小艺": "x iǎo y ì x iǎo y ì",
    "小米小米": "x iǎo m ǐ x iǎo m ǐ",
}
_DEFAULT_KWS_KEYWORDS = tuple(_BUILTIN_KWS_TOKENIZED)

TranscriptCallback = Callable[[str], Awaitable[None]]
SpeechStartCallback = Callable[[], None]
WakeWordCallback = Callable[[str], None]


@runtime_checkable
class STTStreamEngine(Protocol):
    """Protocol for streaming STT engines."""

    on_transcript: Optional[TranscriptCallback]
    on_speech_start: Optional[SpeechStartCallback]
    on_wake_word: Optional[WakeWordCallback]

    async def start(self) -> None:
        ...

    async def feed_audio(self, chunk: bytes) -> None:
        ...

    async def stop(self) -> None:
        ...

    def mark_interaction(self) -> None:
        ...


def normalize_stt_provider(provider: str) -> str:
    """Normalize a user-supplied STT provider name."""
    key = (provider or "").strip().lower().replace(" ", "_")
    key = _STT_PROVIDER_ALIASES.get(key, key)
    if key not in SUPPORTED_STT_PROVIDERS:
        raise ValueError(
            f"Unsupported STT provider: {provider!r}. "
            f"Supported providers: {', '.join(SUPPORTED_STT_PROVIDERS)}",
        )
    return key


class AliyunSTTStream:
    """Streaming STT via DashScope Paraformer."""

    def __init__(
        self,
        *,
        api_key: str = "",
        language: str = "zh-CN",
    ) -> None:
        self._api_key = api_key
        self._language = language
        self._asr: Any = None
        self.on_transcript: Optional[TranscriptCallback] = None
        self.on_speech_start: Optional[SpeechStartCallback] = None
        self.on_wake_word: Optional[WakeWordCallback] = None
        self._speaking = False

    async def start(self) -> None:
        from dashscope_realtime import (
            DashScopeRealtimeASR,
        )
        from dashscope_realtime.asr import ASRConfig

        key = self._api_key or os.environ.get(
            "DASHSCOPE_API_KEY",
            "",
        )
        if not key:
            raise ValueError("DASHSCOPE_API_KEY not set")

        self._transcript_queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _on_sentence_end(text: str) -> None:
            logger.info("STT sentence_end: %s", text)
            self._speaking = False
            if text:
                loop.call_soon_threadsafe(
                    self._transcript_queue.put_nowait,
                    text,
                )

        def _on_error(e: Exception) -> None:
            logger.error("STT ASR error: %s", e)

        def _on_partial(text: str) -> None:
            if text:
                if not self._speaking:
                    self._speaking = True
                    if self.on_speech_start:
                        self.on_speech_start()
                logger.info("STT partial: %s", text)

        config = ASRConfig(
            sample_rate=16000,
            format="pcm",
            language_hints=["zh", "en"],
        )
        self._asr = DashScopeRealtimeASR(
            api_key=key,
            config=config,
            on_sentence_end=_on_sentence_end,
            on_error=_on_error,
            on_partial=_on_partial,
        )
        await self._asr.__aenter__()
        self._dispatch_task = asyncio.create_task(
            self._dispatch_transcripts(),
        )
        logger.info(
            "STT engine started (lang=%s)",
            self._language,
        )

    async def _dispatch_transcripts(self) -> None:
        """Dispatch transcripts to the callback."""
        try:
            while True:
                text = await self._transcript_queue.get()
                if text is None:
                    break
                if self.on_transcript:
                    asyncio.ensure_future(
                        self.on_transcript(text),
                    )
        except asyncio.CancelledError:
            pass

    async def feed_audio(self, chunk: bytes) -> None:
        if self._asr is not None:
            await self._asr.send_audio(chunk)

    def mark_interaction(self) -> None:
        """Aliyun STT has no wake-word session to extend."""

    async def stop(self) -> None:
        task = getattr(self, "_dispatch_task", None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if self._asr is not None:
            try:
                await self._asr.finish()
                await self._asr.__aexit__(
                    None,
                    None,
                    None,
                )
            except Exception:
                logger.debug(
                    "Error closing ASR",
                    exc_info=True,
                )
            self._asr = None


def resolve_openai_base_url(base_url: str) -> str:
    """Normalize an OpenAI-compatible API base URL.

    An empty value resolves to OpenAI's public endpoint. A trailing ``/v1``
    is appended when missing because OpenAI speech routes live under it.
    """
    value = (base_url or "").strip()
    if not value:
        value = _OPENAI_ASR_DEFAULT_BASE_URL
    value = value.rstrip("/")
    if not value.endswith("/v1"):
        value += "/v1"
    return value


def _openai_language_hint(language: str) -> str:
    """Map a channel language (e.g. ``zh-CN``) to a Whisper ISO-639-1 hint."""
    value = (language or "").strip().lower()
    if not value:
        return ""
    code = value.split("-")[0].split("_")[0]
    return code if len(code) == 2 else ""


def _pcm16_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap raw 16-bit mono PCM in a WAV container for transcription APIs."""
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm)
        return buffer.getvalue()


def _asr_response_text(payload: Any) -> str:
    """Extract transcription text from common OpenAI-compatible shapes."""
    if isinstance(payload, str):
        return payload.strip()
    if not isinstance(payload, dict):
        return ""
    value = payload.get("text")
    if isinstance(value, str):
        return value.strip()
    for key in ("data", "output"):
        nested = payload.get(key)
        if isinstance(nested, dict) and isinstance(nested.get("text"), str):
            return nested["text"].strip()
    return ""


class OpenAIWhisperSTT:
    """OpenAI-compatible ``/v1/audio/transcriptions`` STT adapter.

    Microphone PCM is segmented with a lightweight energy VAD: an utterance
    starts when the signal rises above the noise threshold and is finalized
    after the configured trailing silence (or the maximum utterance length).
    Each finalized utterance is uploaded as a 16 kHz WAV and transcribed
    through any OpenAI-compatible Whisper endpoint.

    The endpoint is not streaming, so partial transcripts and local KWS
    wake-word gating are unavailable. ``wake_word_enabled`` is accepted for
    channel-config compatibility but ignored (a warning is logged).
    """

    def __init__(
        self,
        *,
        api_key: str = "",
        base_url: str = "",
        model: str = "",
        language: str = "zh-CN",
        wake_word_enabled: bool = False,
        min_trailing_silence: float = 0.8,
        max_utterance_seconds: float = 15.0,
        min_speech_seconds: float = _OPENAI_ASR_MIN_SPEECH_SECONDS,
        speech_threshold: int = _OPENAI_ASR_SPEECH_THRESHOLD,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._base_url = resolve_openai_base_url(base_url)
        self._model = (model or "").strip() or _OPENAI_ASR_DEFAULT_MODEL
        self._language_hint = _openai_language_hint(language)
        self._wake_word_enabled = bool(wake_word_enabled)
        self._min_trailing_silence = max(
            0.0,
            float(min_trailing_silence),
        )
        self._max_utterance_seconds = max(
            1.0,
            float(max_utterance_seconds),
        )
        self._min_speech_seconds = max(
            0.0,
            float(min_speech_seconds),
        )
        self._speech_threshold = max(0, int(speech_threshold))

        self.on_transcript: Optional[TranscriptCallback] = None
        self.on_speech_start: Optional[SpeechStartCallback] = None
        self.on_wake_word: Optional[WakeWordCallback] = None

        self._client: Any = None
        self._worker_task: Optional[asyncio.Task] = None
        self._pending: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._started = False

        self._buffer = bytearray()
        self._silent_samples = 0
        self._speech_samples = 0
        self._speaking = False

    async def start(self) -> None:
        if self._wake_word_enabled:
            logger.warning(
                "OpenAI-compatible STT has no local wake-word gate; "
                "audio is transcribed directly. Disable wake_word_enabled "
                "to silence this warning.",
            )
        self._pending = asyncio.Queue()
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(30, read=60),
        )
        self._buffer.clear()
        self._silent_samples = 0
        self._speech_samples = 0
        self._speaking = False
        self._started = True
        self._worker_task = asyncio.create_task(self._transcribe_worker())
        logger.info(
            "OpenAI-compatible STT started (model=%s base=%s lang=%s)",
            self._model,
            self._base_url,
            self._language_hint or "-",
        )

    async def feed_audio(self, chunk: bytes) -> None:
        if not chunk or not self._started or self._worker_task is None:
            return
        if len(chunk) % 2:
            return
        rms = audioop.rms(chunk, 2)
        if not self._speaking:
            if rms < self._speech_threshold:
                return
            self._speaking = True
            self._buffer.clear()
            self._silent_samples = 0
            self._speech_samples = 0
            if self.on_speech_start:
                self.on_speech_start()

        self._buffer.extend(chunk)
        samples = len(chunk) // 2
        if rms > self._speech_threshold:
            self._silent_samples = 0
            self._speech_samples += samples
        else:
            self._silent_samples += samples

        trailing_samples = int(
            self._min_trailing_silence * _OPENAI_ASR_SAMPLE_RATE,
        )
        max_samples = int(
            self._max_utterance_seconds * _OPENAI_ASR_SAMPLE_RATE,
        )
        if (
            self._silent_samples >= trailing_samples
            or len(self._buffer) // 2 >= max_samples
        ):
            self._end_utterance()

    def _end_utterance(self) -> None:
        """Queue the buffered utterance for transcription, if any."""
        speech_samples = self._speech_samples
        self._speaking = False
        self._silent_samples = 0
        self._speech_samples = 0
        if not self._buffer:
            return
        if speech_samples < int(
            self._min_speech_seconds * _OPENAI_ASR_SAMPLE_RATE,
        ):
            self._buffer.clear()
            return
        pcm = bytes(self._buffer)
        self._buffer.clear()
        try:
            self._pending.put_nowait(pcm)
        except asyncio.QueueFull:  # pragma: no cover - unbounded queue
            logger.warning("OpenAI-compatible STT pending queue is full")

    async def _transcribe_worker(self) -> None:
        try:
            while True:
                pcm = await self._pending.get()
                if pcm is None:
                    break
                try:
                    text = await self._transcribe(pcm)
                except Exception:
                    logger.warning(
                        "OpenAI-compatible STT transcription failed",
                        exc_info=True,
                    )
                    continue
                if text and self.on_transcript:
                    asyncio.ensure_future(self.on_transcript(text))
        except asyncio.CancelledError:
            pass

    async def _transcribe(self, pcm: bytes) -> str:
        """Transcribe one 16 kHz mono PCM buffer over the configured API."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30, read=60),
            )
        wav = _pcm16_to_wav(pcm, _OPENAI_ASR_SAMPLE_RATE)
        data: dict[str, str] = {
            "model": self._model,
            "response_format": "json",
        }
        if self._language_hint:
            data["language"] = self._language_hint
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        response = await self._client.post(
            f"{self._base_url}/audio/transcriptions",
            headers=headers,
            data=data,
            files={"file": ("audio.wav", wav, "audio/wav")},
        )
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError:
            return response.text.strip()
        return _asr_response_text(payload)

    def mark_interaction(self) -> None:
        """No wake-word session exists for API STT."""

    async def stop(self) -> None:
        self._started = False
        task = self._worker_task
        self._worker_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                logger.debug(
                    "Error closing OpenAI-compatible STT client",
                    exc_info=True,
                )
            self._client = None
        self._buffer.clear()
        self._silent_samples = 0
        self._speech_samples = 0
        self._speaking = False


class SherpaZipformerSTT:
    """Local streaming Zipformer ASR with an optional KWS wake-word gate.

    Audio is always processed at 16 kHz mono 16-bit PCM, matching what
    ``SIPChannel._audio_reader`` produces.

    When ``wake_word_enabled`` is true the engine starts in ``sleeping``
    state: incoming audio only goes through the KWS model. On a keyword
    hit it replays a short pre-roll buffer into a fresh ASR stream and then
    stays active for 60 seconds after the most recent interaction. Further
    commands inside that window do not require the wake word. Once the window
    expires it returns to ``sleeping``. With ``wake_word_enabled`` false
    (default) audio goes straight to the Zipformer recognizer.
    """

    def __init__(
        self,
        *,
        model_dir: str = "",
        num_threads: int = 2,
        wake_word_enabled: bool = False,
        kws_model_dir: str = "",
        keywords_file: str = "",
        kws_num_threads: int = 1,
        pre_roll_seconds: float = _DEFAULT_PRE_ROLL_SECONDS,
        kws_score: float = 1.5,
        kws_threshold: float = 0.25,
        wake_active_timeout_seconds: float = (
            _DEFAULT_WAKE_ACTIVE_TIMEOUT_SECONDS
        ),
        rule1_min_trailing_silence: float = 0.8,
        rule2_min_trailing_silence: float = 0.4,
        rule3_min_utterance_length: float = 15.0,
        keep_model_loaded: bool = True,
    ) -> None:
        self._model_dir = model_dir
        self._num_threads = int(num_threads)
        self._keep_model_loaded = bool(keep_model_loaded)
        self._wake_word_enabled = bool(wake_word_enabled)
        self._kws_model_dir = kws_model_dir
        self._keywords_file = keywords_file
        self._kws_num_threads = int(kws_num_threads)
        self._kws_score = max(0.0, float(kws_score))
        self._kws_threshold = max(0.0, min(1.0, float(kws_threshold)))
        self._wake_active_timeout_seconds = max(
            0.0,
            float(wake_active_timeout_seconds),
        )
        self._rule1_min_trailing_silence = float(
            rule1_min_trailing_silence,
        )
        self._rule2_min_trailing_silence = float(
            rule2_min_trailing_silence,
        )
        self._rule3_min_utterance_length = float(
            rule3_min_utterance_length,
        )

        self.on_transcript: Optional[TranscriptCallback] = None
        self.on_speech_start: Optional[SpeechStartCallback] = None
        self.on_wake_word: Optional[WakeWordCallback] = None

        self._recognizer: Any = None
        self._kws: Any = None
        self._asr_stream: Any = None
        self._kws_stream: Any = None
        self._inline_keywords: Optional[str] = None
        self._temporary_keywords_file: Optional[Path] = None
        self._woken = not self._wake_word_enabled
        self._speaking = False
        self._last_wake_word = ""
        self._wake_expires_at = 0.0

        # Audio kept before a wake-word hit is replayed into the ASR stream
        # so a command spoken right after the keyword is not truncated.
        self._pre_roll: deque[Any] = deque()
        self._pre_roll_samples = 0
        self._pre_roll_limit = int(max(0.0, pre_roll_seconds) * _SAMPLE_RATE)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        zipformer_dir = resolve_zipformer_model_dir(self._model_dir)
        zipformer_model_files(zipformer_dir)

        kws_dir: Optional[Path] = None
        keywords_file = ""
        if self._wake_word_enabled:
            kws_dir = resolve_kws_model_dir(self._kws_model_dir)
            self._inline_keywords = None
            keyword_value = (self._keywords_file or "").strip()
            keyword_path = (
                Path(keyword_value).expanduser() if keyword_value else None
            )
            if keyword_path is not None and keyword_path.is_file():
                keywords_file = str(keyword_path)
            elif keyword_value and _looks_like_keywords_file_path(
                keyword_value,
            ):
                raise ValueError(
                    f"KWS keywords file not found: {keyword_path}",
                )
            else:
                if keyword_value:
                    logger.warning(
                        "kws_keywords_file %r is not an existing file; "
                        "treating it as a literal keyword",
                        keyword_value,
                    )
                    self._inline_keywords = _tokenize_kws_keyword(
                        keyword_value,
                    )
                default_file = kws_dir / "keywords.txt"
                if default_file.is_file():
                    keywords_file = str(default_file)
                else:
                    logger.warning(
                        "KWS model has no keywords.txt in %s; using "
                        "built-in default keywords",
                        kws_dir,
                    )
                    self._temporary_keywords_file = (
                        _write_temporary_keywords_file(
                            _DEFAULT_KWS_KEYWORDS,
                        )
                    )
                    keywords_file = str(self._temporary_keywords_file)

        try:
            models = await asyncio.to_thread(
                self._load_models,
                zipformer_dir,
                kws_dir,
                keywords_file,
            )
            self._recognizer = models["recognizer"]
            if self._wake_word_enabled:
                self._kws = models["kws"]
                self._kws_stream = self._create_kws_stream()
                self._woken = False
                self._last_wake_word = ""
                self._wake_expires_at = 0.0
            else:
                self._asr_stream = self._recognizer.create_stream()
                self._woken = True
        except Exception:
            self._cleanup_temporary_keywords_file()
            raise
        logger.info(
            "SherpaZipformerSTT started (wake_word=%s)",
            self._wake_word_enabled,
        )

    async def feed_audio(self, chunk: bytes) -> None:
        if not chunk or self._recognizer is None:
            return
        samples = _bytes_to_float32(chunk)

        if (
            self._wake_word_enabled
            and self._woken
            and self._wake_expires_at > 0
            and time.monotonic() >= self._wake_expires_at
        ):
            self._rearm_wake_word()

        if not self._woken:
            self._feed_kws(samples)
            if self._woken:
                # Replay only the audio before the current chunk; the
                # current chunk is fed to ASR once below.
                self._start_asr_stream()
            else:
                self._push_pre_roll(samples)

        if self._woken:
            self._feed_asr(samples)

    async def stop(self) -> None:
        if self._asr_stream is not None:
            try:
                self._asr_stream.input_finished()
            except Exception:
                logger.debug(
                    "Error finishing ASR stream",
                    exc_info=True,
                )
            self._asr_stream = None
        self._pre_roll.clear()
        self._pre_roll_samples = 0
        self._last_wake_word = ""
        self._wake_expires_at = 0.0
        if not self._keep_model_loaded:
            # Release local model instances; they were loaded fresh for
            # this engine and should not outlive the channel session.
            self._recognizer = None
            self._kws = None
            self._kws_stream = None
        self._cleanup_temporary_keywords_file()

    def mark_interaction(self) -> None:
        """Keep an active wake session alive after user/assistant activity."""
        if not self._wake_word_enabled or not self._woken:
            return
        self._wake_expires_at = (
            time.monotonic() + self._wake_active_timeout_seconds
        )

    def _rearm_wake_word(self) -> None:
        """Return to keyword spotting after the active-session timeout."""
        self._woken = False
        self._asr_stream = None
        self._speaking = False
        self._last_wake_word = ""
        self._wake_expires_at = 0.0
        self._pre_roll.clear()
        self._pre_roll_samples = 0
        if self._kws is not None:
            self._kws_stream = self._create_kws_stream()
        logger.info("Wake session expired; keyword gate re-armed")

    def _cleanup_temporary_keywords_file(self) -> None:
        """Remove a generated keywords file, if any."""
        if self._temporary_keywords_file is None:
            return
        try:
            self._temporary_keywords_file.unlink(missing_ok=True)
        except OSError:
            logger.debug(
                "Could not remove temporary KWS keywords file",
                exc_info=True,
            )
        self._temporary_keywords_file = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_models(
        self,
        zipformer_dir: Path,
        kws_dir: Optional[Path],
        keywords_file: str,
    ) -> dict[str, Any]:
        """Load the Zipformer recognizer and (optionally) KWS models.

        With ``keep_model_loaded`` the loaded instances are cached
        process-wide so a channel restart reuses them instead of paying the
        model-loading cost again.
        """
        tokens, encoder, decoder, joiner = zipformer_model_files(
            zipformer_dir,
        )
        kws_paths = kws_model_files(kws_dir) if kws_dir is not None else None
        # Temporary generated keyword files get a stable cache identity so
        # restarts reuse the cached KWS instance instead of leaking one entry
        # per generated file.
        cache_keywords = keywords_file
        if (
            self._temporary_keywords_file is not None
            and keywords_file == str(self._temporary_keywords_file)
        ):
            cache_keywords = _DEFAULT_KWS_KEYWORDS
        cache_key = (
            str(Path(tokens).resolve()),
            str(Path(encoder).resolve()),
            str(Path(decoder).resolve()),
            str(Path(joiner).resolve()),
            self._num_threads,
            self._rule1_min_trailing_silence,
            self._rule2_min_trailing_silence,
            self._rule3_min_utterance_length,
            None
            if kws_paths is None
            else tuple(str(Path(path).resolve()) for path in kws_paths),
            self._kws_num_threads,
            self._kws_score,
            self._kws_threshold,
            cache_keywords,
            self._inline_keywords or "",
        )

        with _zipformer_models_lock:
            cached = _zipformer_models_cache.get(cache_key)
            if cached is not None and self._keep_model_loaded:
                return cached
            if not self._keep_model_loaded:
                _zipformer_models_cache.pop(cache_key, None)

        models = self._load_models_uncached(
            tokens,
            encoder,
            decoder,
            joiner,
            kws_paths,
            keywords_file,
        )

        if self._keep_model_loaded:
            with _zipformer_models_lock:
                _zipformer_models_cache[cache_key] = models
        return models

    def _load_models_uncached(
        self,
        tokens: Path,
        encoder: Path,
        decoder: Path,
        joiner: Path,
        kws_paths: Optional[tuple[Path, Path, Path, Path]],
        keywords_file: str,
    ) -> dict[str, Any]:
        """Construct recognizer/KWS instances for one cache key."""
        import sherpa_onnx

        model_type = detect_transducer_model_type(encoder)
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(tokens),
            encoder=str(encoder),
            decoder=str(decoder),
            joiner=str(joiner),
            num_threads=self._num_threads,
            sample_rate=_SAMPLE_RATE,
            feature_dim=80,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=(
                self._rule1_min_trailing_silence
            ),
            rule2_min_trailing_silence=(
                self._rule2_min_trailing_silence
            ),
            rule3_min_utterance_length=(
                self._rule3_min_utterance_length
            ),
            decoding_method="greedy_search",
            model_type=model_type,
            provider="cpu",
        )
        models: dict[str, Any] = {"recognizer": recognizer}

        if kws_paths is None:
            return models

        kws_tokens, kws_encoder, kws_decoder, kws_joiner = kws_paths
        kws = sherpa_onnx.KeywordSpotter(
            tokens=str(kws_tokens),
            encoder=str(kws_encoder),
            decoder=str(kws_decoder),
            joiner=str(kws_joiner),
            keywords_file=keywords_file,
            num_threads=self._kws_num_threads,
            sample_rate=_SAMPLE_RATE,
            feature_dim=80,
            max_active_paths=4,
            keywords_score=self._kws_score,
            keywords_threshold=self._kws_threshold,
            provider="cpu",
        )
        models["kws"] = kws
        return models

    def _create_kws_stream(self) -> Any:
        """Create a KWS stream, adding a literal keyword when configured."""
        if self._kws is None:
            raise RuntimeError("KWS model is not loaded")
        if self._inline_keywords:
            return self._kws.create_stream(self._inline_keywords)
        return self._kws.create_stream()

    # ------------------------------------------------------------------
    # Audio pipeline
    # ------------------------------------------------------------------

    def _feed_kws(self, samples: Any) -> None:
        """Run one KWS decode pass; flip to ``woken`` on a keyword hit."""
        if self._kws is None or self._kws_stream is None:
            return
        self._kws_stream.accept_waveform(_SAMPLE_RATE, samples)
        while self._kws.is_ready(self._kws_stream):
            self._kws.decode_stream(self._kws_stream)
            keyword = self._kws.get_result(self._kws_stream)
            if not keyword:
                continue
            self._kws.reset_stream(self._kws_stream)
            self._woken = True
            self._last_wake_word = str(keyword).strip()
            self.mark_interaction()
            logger.info("Wake word detected: %s", keyword)
            if self.on_wake_word:
                self.on_wake_word(keyword)
            return

    def _push_pre_roll(self, samples: Any) -> None:
        """Keep a bounded pre-roll buffer of recent audio."""
        if self._pre_roll_limit <= 0:
            return
        self._pre_roll.append(samples)
        self._pre_roll_samples += len(samples)
        while (
            self._pre_roll
            and self._pre_roll_samples > self._pre_roll_limit
        ):
            dropped = self._pre_roll.popleft()
            self._pre_roll_samples -= len(dropped)

    def _start_asr_stream(self) -> None:
        """Create an ASR stream and replay buffered pre-roll audio."""
        if self._recognizer is None:
            return
        self._asr_stream = self._recognizer.create_stream()
        while self._pre_roll:
            buffered = self._pre_roll.popleft()
            self._asr_stream.accept_waveform(_SAMPLE_RATE, buffered)
            self._pre_roll_samples -= len(buffered)
        self._pre_roll_samples = 0

    def _feed_asr(self, samples: Any) -> None:
        """Decode samples and dispatch endpoint transcripts."""
        if self._recognizer is None or self._asr_stream is None:
            return
        self._asr_stream.accept_waveform(_SAMPLE_RATE, samples)
        while self._recognizer.is_ready(self._asr_stream):
            self._recognizer.decode_stream(self._asr_stream)
            partial = self._recognizer.get_result(self._asr_stream)
            if partial and not self._speaking:
                self._speaking = True
                self.mark_interaction()
                if self.on_speech_start:
                    self.on_speech_start()
                logger.info("Zipformer partial: %s", partial)

        if self._recognizer.is_endpoint(self._asr_stream):
            raw_text = self._recognizer.get_result(self._asr_stream).strip()
            text = strip_detected_wake_word(
                raw_text,
                self._last_wake_word,
            )
            self._recognizer.reset(self._asr_stream)
            self._speaking = False
            if text:
                self.mark_interaction()
                logger.info("Zipformer endpoint: %s", text)
                if self.on_transcript:
                    asyncio.ensure_future(self.on_transcript(text))


def strip_detected_wake_word(text: str, keyword: str) -> str:
    """Remove a detected wake word from the beginning of an ASR result."""
    result = (text or "").strip()
    compact_keyword = "".join((keyword or "").split())
    if not result or not compact_keyword:
        return result

    # ASR may insert spaces between letters/characters, so tolerate optional
    # whitespace inside the detected keyword while keeping the match anchored
    # at the beginning of the utterance.
    keyword_pattern = r"\s*".join(re.escape(char) for char in compact_keyword)
    match = re.match(
        rf"^\s*{keyword_pattern}",
        result,
        flags=re.IGNORECASE,
    )
    if match is None:
        return result
    return result[match.end():].lstrip(
        " \t\r\n,，。.!！?？:：;；、-—",
    )


def detect_transducer_model_type(encoder: Path) -> str:
    """Read a transducer encoder's exported model type.

    Older bilingual mobile assets identify as ``zipformer`` while the current
    X-ASR assets export ``zipformer2``.  The sherpa-onnx constructor needs the
    matching explicit type, so falling back to a filename or a hard-coded type
    would make the new downloads fail to load.
    """
    try:
        import onnxruntime as ort

        metadata = ort.InferenceSession(
            str(encoder),
            providers=["CPUExecutionProvider"],
        ).get_modelmeta().custom_metadata_map
    except Exception:
        logger.warning(
            "Could not read ONNX metadata from %s; assuming zipformer",
            encoder,
            exc_info=True,
        )
        return "zipformer"

    model_type = (metadata.get("model_type") or "").strip().lower()
    if model_type in {"zipformer", "zipformer2"}:
        return model_type
    logger.warning(
        "Unrecognized ONNX model_type %r in %s; assuming zipformer",
        model_type,
        encoder,
    )
    return "zipformer"


# ---------------------------------------------------------------------------
# Model directory / file helpers
# ---------------------------------------------------------------------------


def resolve_zipformer_model_dir(model_dir: str = "") -> Path:
    """Resolve the local streaming Zipformer model directory.

    Priority: explicit *model_dir*, ``ZIPFORMER_MODEL_DIR`` env var, then
    ``MODELS_DIR/sherpa-onnx-x-asr-480ms-streaming-...-int8-2026-06-05``.
    """
    raw = model_dir or os.environ.get("ZIPFORMER_MODEL_DIR", "")
    if raw:
        return Path(raw).expanduser()
    return MODELS_DIR / _ZIPFORMER_MODEL_NAME


def resolve_kws_model_dir(model_dir: str = "") -> Path:
    """Resolve the local KWS model directory."""
    raw = model_dir or os.environ.get("KWS_MODEL_DIR", "")
    if raw:
        return Path(raw).expanduser()
    return MODELS_DIR / _KWS_MODEL_NAME


def zipformer_model_files(model_dir: Path) -> tuple[Path, Path, Path, Path]:
    """Return ``(tokens, encoder, decoder, joiner)`` paths for Zipformer."""
    tokens = model_dir / "tokens.txt"
    if not tokens.is_file():
        raise ValueError(
            f"Zipformer model directory is incomplete: {model_dir}. "
            "Missing tokens.txt. Download "
            f"{_ZIPFORMER_MODEL_NAME} and set zipformer_model_dir or "
            "ZIPFORMER_MODEL_DIR.",
        )
    encoder = _pick_onnx(model_dir, "encoder", prefer_int8=True)
    decoder = _pick_onnx(model_dir, "decoder", prefer_int8=False)
    joiner = _pick_onnx(model_dir, "joiner", prefer_int8=True)
    return tokens, encoder, decoder, joiner


def kws_model_files(model_dir: Path) -> tuple[Path, Path, Path, Path]:
    """Return ``(tokens, encoder, decoder, joiner)`` paths for KWS."""
    tokens = model_dir / "tokens.txt"
    if not tokens.is_file():
        raise ValueError(
            f"KWS model directory is incomplete: {model_dir}. "
            "Missing tokens.txt. Download "
            f"{_KWS_MODEL_NAME} and set kws_model_dir or KWS_MODEL_DIR.",
        )
    encoder = _pick_onnx(model_dir, "encoder", prefer_int8=True)
    decoder = _pick_onnx(model_dir, "decoder", prefer_int8=False)
    joiner = _pick_onnx(model_dir, "joiner", prefer_int8=True)
    return tokens, encoder, decoder, joiner


def _looks_like_keywords_file_path(value: str) -> bool:
    """Return true for strings a user almost certainly meant as a file path."""
    value = value.strip()
    separators = {
        sep for sep in (os.sep, os.altsep, "/", "\\") if sep is not None
    }
    return (
        value.startswith("~")
        or value.lower().endswith(".txt")
        or any(separator in value for separator in separators)
    )


def validate_kws_keywords_value(keywords_file: str = "") -> str:
    """Validate a ``kws_keywords_file`` value before it is persisted.

    Empty means "use model defaults". An existing path is kept as-is. A
    path-like value that does not exist raises immediately, while any other
    non-empty value is treated as a literal wake word and must be
    convertible to the model's token format.
    """
    value = (keywords_file or "").strip()
    if not value:
        return value
    path = Path(value).expanduser()
    if path.is_file():
        return str(path)
    if _looks_like_keywords_file_path(value):
        raise ValueError(f"KWS keywords file not found: {path}")
    _tokenize_kws_keyword(value)
    return value


def _write_temporary_keywords_file(keywords: tuple[str, ...]) -> Path:
    """Write tokenized keywords to a temporary file for KeywordSpotter."""
    lines = []
    for keyword in keywords:
        tokenized = _tokenize_kws_keyword(keyword)
        if tokenized:
            lines.append(tokenized)
    if not lines:
        raise ValueError("No KWS keywords available")
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="qwenpaw-kws-keywords-",
        suffix=".txt",
        delete=False,
    ) as output:
        output.write("\n".join(lines))
        output.write("\n")
        return Path(output.name)


def _tokenize_kws_keyword(keyword: str) -> str:
    """Convert a literal wake word into the zh-en model's token format.

    The bundled zh-en KWS model uses ``phone+ppinyin`` tokens (English CMU
    phonemes plus Chinese initials/finals with tone), so raw CJK characters
    are converted to pinyin and the ``@keyword`` suffix keeps the original
    text as the reported result.
    """
    keyword = (keyword or "").strip()
    if not keyword:
        raise ValueError("KWS keyword is empty")
    tokenized = _BUILTIN_KWS_TOKENIZED.get(keyword)
    if tokenized is None:
        tokenized = _pinyin_tokens_for_cjk(keyword)
    return f"{tokenized} @{keyword}"


def _pinyin_tokens_for_cjk(keyword: str) -> str:
    """Tokenize CJK text into the model's partial-pinyin token sequence."""
    try:
        from pypinyin import pinyin
        from pypinyin.contrib.tone_convert import (
            to_finals_tone,
            to_initials,
        )
    except ImportError as exc:
        raise ValueError(
            f"Cannot tokenize custom wake word {keyword!r}: pypinyin is "
            "not installed. Install qwenpaw[sip] or set kws_keywords_file "
            "to an existing tokenized keywords.txt file.",
        ) from exc

    pieces: list[str] = []
    for item in pinyin(keyword, heteronym=False):
        syllable = item[0]
        initial = to_initials(syllable, strict=False)
        final = to_finals_tone(syllable, strict=False)
        if initial:
            pieces.append(initial)
        if final:
            pieces.append(final)
    if not pieces:
        raise ValueError(f"Cannot tokenize KWS keyword: {keyword!r}")
    return " ".join(pieces)


def resolve_kws_keywords_file(
    kws_model_dir: Path,
    keywords_file: str = "",
) -> str:
    """Resolve the KWS keywords file.

    A configured *keywords_file* wins. Otherwise the model's built-in
    ``keywords.txt`` is used. The recommended zh-en 3M model accepts
    tokenized Chinese and English wake words.
    """
    if keywords_file:
        path = Path(keywords_file).expanduser()
        if not path.is_file():
            raise ValueError(
                f"KWS keywords file not found: {path}",
            )
        return str(path)
    default = kws_model_dir / "keywords.txt"
    if not default.is_file():
        raise ValueError(
            f"KWS model has no keywords.txt: {kws_model_dir}. "
            "Set kws_keywords_file to a tokenized keywords file.",
        )
    return str(default)


def _pick_onnx(
    model_dir: Path,
    component: str,
    *,
    prefer_int8: bool,
) -> Path:
    """Pick a model file matching ``<component>*.onnx`` in *model_dir*."""
    candidates = sorted(
        path.name for path in model_dir.glob(f"{component}*.onnx")
    )
    if not candidates:
        raise ValueError(
            f"Missing {component}*.onnx in model directory: {model_dir}",
        )
    if prefer_int8:
        int8 = [name for name in candidates if "int8" in name.lower()]
        return model_dir / (int8[0] if int8 else candidates[0])
    normal = [name for name in candidates if "int8" not in name.lower()]
    return model_dir / (normal[0] if normal else candidates[0])


def _bytes_to_float32(data: bytes) -> Any:
    """Convert little-endian int16 PCM bytes to float32 in [-1, 1]."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        # numpy ships with onnxruntime, so this is defensive.
        raise ValueError(
            "numpy is required for local Zipformer ASR",
        ) from exc

    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    samples /= 32768.0
    return samples
