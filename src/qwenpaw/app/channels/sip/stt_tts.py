# -*- coding: utf-8 -*-
"""STT/TTS factory functions for the SIP voice channel.

TTS providers:

* ``aliyun``   -- DashScope CosyVoice streaming TTS (legacy default)
* ``edge_tts`` -- Microsoft Edge online TTS, decoded locally with ffmpeg
* ``kokoro``   -- Local Kokoro TTS through sherpa-onnx (zh/en, offline)
* ``openai``   -- OpenAI-compatible ``/v1/audio/speech`` API
* ``qwen3``    -- Qwen3-TTS 0.6B (DashScope API or local ``qwen_tts``)
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import mimetypes
import os
import queue as thread_queue
import re
import shutil
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator
from urllib.parse import urlparse
from urllib.request import url2pathname

import httpx

from qwenpaw.constant import MODELS_DIR

from ._audioop_compat import audioop
from .stt_engine import (
    AliyunSTTStream,
    OpenAIWhisperSTT,
    SherpaZipformerSTT,
    STTStreamEngine,
    normalize_stt_provider,
    resolve_openai_base_url,
)

logger = logging.getLogger(__name__)

# Enough for short scheduling jitter without allowing a whole response to sit
# in memory when the audio output blocks.
_TTS_QUEUE_MAX_CHUNKS = 64

SUPPORTED_TTS_PROVIDERS = (
    "aliyun",
    "edge_tts",
    "kokoro",
    "openai",
    "qwen3",
)

_TTS_PROVIDER_ALIASES = {
    "edge": "edge_tts",
    "edge-tts": "edge_tts",
    "edgetts": "edge_tts",
    "kokoro-onnx": "kokoro",
    "sherpa-kokoro": "kokoro",
    "openai_tts": "openai",
    "openai-tts": "openai",
    "openai_compatible": "openai",
    "qwen3_tts": "qwen3",
    "qwen3-tts": "qwen3",
    "qwen3-tts-flash": "qwen3",
}

_OPENAI_TTS_DEFAULT_MODEL = "tts-1"
_OPENAI_TTS_DEFAULT_VOICE = "alloy"

_QWEN3_DEFAULT_API_MODEL = "qwen3-tts-flash"
_QWEN3_DEFAULT_API_VOICE = "Cherry"
_QWEN3_DEFAULT_CLONE_MODEL = "qwen3-tts-vc-2026-01-22"
_QWEN3_DEFAULT_LOCAL_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

# Built-in DashScope Qwen3-TTS voices. A value not in this set is treated as
# a cloned ``voice_id``; cloned voices are enrolled against (and must be
# synthesized through) the dedicated VC model, not the base flash model.
_QWEN3_SYSTEM_VOICES = frozenset({
    "Cherry",
    "Serena",
    "Ethan",
    "Chelsie",
    "Dylan",
    "Jada",
    "Sunny",
    "Nofish",
    "Marcus",
    "Roy",
})
# DashScope models that only have built-in system voices. They cannot be used
# as enrollment targets for cloned voices.
_QWEN3_API_SYNTHESIS_MODELS = frozenset({
    "qwen3-tts-flash",
    "qwen3-tts-instruct-flash",
})
_QWEN3_LOCAL_MODEL_DIR_NAME = "Qwen3-TTS-12Hz-0.6B-Base"
_QWEN3_DASHSCOPE_UPLOAD_URL = "https://dashscope.aliyuncs.com/api/v1/uploads"
_QWEN3_CLONE_CACHE_DIR = MODELS_DIR / "qwen3-tts-clones"

_qwen3_local_tts_cache: dict[tuple, tuple[Any, Any]] = {}
_qwen3_local_tts_lock = threading.Lock()
# Serializes local Qwen3-TTS generation while a fixed seed is applied, so
# concurrent synthesis calls cannot steal/overwrite each other's RNG state.
_qwen3_generation_lock = threading.Lock()

_EDGE_TTS_DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"
_KOKORO_DEFAULT_VOICE = "zf_001"
_KOKORO_NATIVE_SAMPLE_RATE = 24000

# Legacy Kokoro v1.0 speaker map; numeric ids are accepted as well.
_KOKORO_SPEAKERS = {
    "af_alloy": 0,
    "af_aoede": 1,
    "af_bella": 2,
    "af_heart": 3,
    "af_jessica": 4,
    "af_kore": 5,
    "af_nicole": 6,
    "af_nova": 7,
    "af_river": 8,
    "af_sarah": 9,
    "af_sky": 10,
    "am_adam": 11,
    "am_echo": 12,
    "am_eric": 13,
    "am_fenrir": 14,
    "am_liam": 15,
    "am_michael": 16,
    "am_onyx": 17,
    "am_puck": 18,
    "am_santa": 19,
    "bf_alice": 20,
    "bf_emma": 21,
    "bf_isabella": 22,
    "bf_lily": 23,
    "bm_daniel": 24,
    "bm_fable": 25,
    "bm_george": 26,
    "bm_lewis": 27,
    "ef_dora": 28,
    "em_alex": 29,
    "ff_siwis": 30,
    "hf_alpha": 31,
    "hf_beta": 32,
    "hm_omega": 33,
    "hm_psi": 34,
    "if_sara": 35,
    "im_nicola": 36,
    "jf_alpha": 37,
    "jf_gongitsune": 38,
    "jf_nezumi": 39,
    "jf_tebukuro": 40,
    "jm_kumo": 41,
    "pf_dora": 42,
    "pm_alex": 43,
    "pm_santa": 44,
    "zf_xiaobei": 45,
    "zf_xiaoni": 46,
    "zf_xiaoxiao": 47,
    "zf_xiaoyi": 48,
    "zm_yunjian": 49,
    "zm_yunxi": 50,
    "zm_yunxia": 51,
    "zm_yunyang": 52,
}

_KOKORO_V11_BASE_SPEAKERS = {
    "af_maple": 0,
    "af_sol": 1,
    "bf_vale": 2,
}

_kokoro_tts_lock = threading.Lock()
_kokoro_tts_cache: Dict[tuple[str, int], tuple[Any, Any]] = {}


def _resolve_dashscope_key(api_key: str = "") -> str:
    """Return *api_key* if non-empty, else fall back to env var."""
    return api_key or os.environ.get("DASHSCOPE_API_KEY", "")


def _resolve_openai_key(api_key: str = "") -> str:
    """Return *api_key* if non-empty, else fall back to env var."""
    return api_key or os.environ.get("OPENAI_API_KEY", "")


def normalize_tts_provider(provider: str) -> str:
    """Normalize a user-supplied TTS provider name.

    ``edgetts`` / ``edge-tts`` / ``edge`` all map to ``edge_tts`` so old
    configs keep working. Raises ``ValueError`` for unknown providers.
    """
    key = (provider or "").strip().lower().replace(" ", "_")
    key = _TTS_PROVIDER_ALIASES.get(key, key)
    if key not in SUPPORTED_TTS_PROVIDERS:
        raise ValueError(
            f"Unsupported TTS provider: {provider!r}. "
            f"Supported providers: {', '.join(SUPPORTED_TTS_PROVIDERS)}",
        )
    return key


def normalize_qwen3_backend(backend: str) -> str:
    """Normalize ``api`` / ``local`` for the Qwen3-TTS provider."""
    key = (backend or "api").strip().lower().replace("-", "_")
    if key in {"local", "local_model", "offline"}:
        return "local"
    return "api"


_QWEN3_EXTRA_SYSTEM_VOICES_ENV = "QWENPAW_QWEN3_SYSTEM_VOICES"


def _extra_qwen3_system_voices() -> frozenset[str]:
    """Extra system voices supplied via the environment.

    DashScope may add built-in voices faster than this code is updated.
    Set ``QWENPAW_QWEN3_SYSTEM_VOICES`` to a comma-separated list of
    additional built-in voice names so they are not mistaken for cloned
    ``voice_id`` values.
    """
    raw = os.environ.get(_QWEN3_EXTRA_SYSTEM_VOICES_ENV, "")
    return frozenset(
        item.strip() for item in raw.split(",") if item.strip()
    )


def is_qwen3_system_voice(voice: str) -> bool:
    """Return whether *voice* is a built-in Qwen3-TTS system voice.

    DashScope's Qwen3 voice-cloning API binds each cloned voice to a
    dedicated VC model.  A non-system voice (a cloned ``voice_id``) must
    therefore be synthesized through that VC model instead of
    ``qwen3-tts-flash``.  Extra built-in voices can be configured through
    the ``QWENPAW_QWEN3_SYSTEM_VOICES`` environment variable.
    """
    value = (voice or "").strip()
    if not value:
        return True
    system_voices = {item.casefold() for item in _QWEN3_SYSTEM_VOICES}
    system_voices.update(
        item.casefold() for item in _extra_qwen3_system_voices()
    )
    return value.casefold() in system_voices


def resolve_qwen3_api_model(model: str, voice: str) -> str:
    """Choose the Qwen3-TTS model used for one DashScope API request.

    System voices use the configured synthesis model (``qwen3-tts-flash`` by
    default).  Cloned voices are enrolled against
    ``qwen3-tts-vc-2026-01-22`` and must use that same model during synthesis;
    otherwise the provider can silently fall back to a different timbre.
    """
    active_model = (model or "").strip() or _QWEN3_DEFAULT_API_MODEL
    if not is_qwen3_system_voice(voice):
        configured = (model or "").strip()
        if configured and configured not in _QWEN3_API_SYNTHESIS_MODELS:
            # A caller can explicitly select a newer/custom VC model.
            return configured
        return _QWEN3_DEFAULT_CLONE_MODEL
    return active_model


def _normalize_qwen3_prefix(name: str) -> str:
    """Provider-safe voice prefix for DashScope voice enrollment."""
    value = re.sub(r"[^a-z0-9]+", "", (name or "").strip().casefold())
    return (value or "qwenpawvoice")[:10]


def resolve_qwen3_model_dir(model_dir: str = "") -> str:
    """Resolve the local Qwen3-TTS model path.

    An explicit directory wins. An empty value uses the bundled
    ``MODELS_DIR/Qwen3-TTS-12Hz-0.6B-Base`` when it exists (the download
    manager target), otherwise the Hugging Face repo id.
    """
    if model_dir.strip():
        return model_dir
    local = MODELS_DIR / _QWEN3_LOCAL_MODEL_DIR_NAME
    if local.is_dir():
        return str(local)
    return _QWEN3_DEFAULT_LOCAL_MODEL


def create_stt_engine(
    provider: str,
    language: str,
    api_key: str = "",
    *,
    zipformer_model_dir: str = "",
    zipformer_num_threads: int = 2,
    wake_word_enabled: bool = False,
    kws_model_dir: str = "",
    kws_keywords_file: str = "",
    kws_num_threads: int = 1,
    kws_pre_roll_seconds: float = 0.5,
    kws_score: float = 1.5,
    kws_threshold: float = 0.25,
    wake_active_timeout_seconds: float = 60.0,
    asr_rule1_min_trailing_silence: float = 0.8,
    asr_rule2_min_trailing_silence: float = 0.4,
    asr_rule3_min_utterance_length: float = 15.0,
    openai_api_key: str = "",
    openai_base_url: str = "",
    openai_model: str = "whisper-1",
    keep_model_loaded: bool = True,
) -> STTStreamEngine:
    """Create a streaming STT engine for *provider*.

    ``provider`` accepts ``aliyun`` (DashScope Paraformer),
    ``sherpa_zipformer`` (local streaming Zipformer with optional KWS), or
    ``openai`` (any OpenAI-compatible ``/v1/audio/transcriptions`` endpoint).
    ``api_key`` is the DashScope key; OpenAI-compatible credentials come
    from ``openai_api_key`` / ``OPENAI_API_KEY``. ``keep_model_loaded`` only
    affects the local Zipformer provider: when enabled, loaded recognizer/KWS
    instances are cached process-wide and reused after channel restarts.
    """
    provider_key = normalize_stt_provider(provider)
    if provider_key == "aliyun":
        return AliyunSTTStream(
            api_key=_resolve_dashscope_key(api_key),
            language=language,
        )
    if provider_key == "openai":
        return OpenAIWhisperSTT(
            api_key=_resolve_openai_key(openai_api_key),
            base_url=openai_base_url,
            model=openai_model,
            language=language,
            wake_word_enabled=wake_word_enabled,
            min_trailing_silence=asr_rule1_min_trailing_silence,
            max_utterance_seconds=asr_rule3_min_utterance_length,
            min_speech_seconds=asr_rule2_min_trailing_silence,
        )
    if provider_key == "sherpa_zipformer":
        return SherpaZipformerSTT(
            model_dir=zipformer_model_dir,
            num_threads=zipformer_num_threads,
            wake_word_enabled=wake_word_enabled,
            kws_model_dir=kws_model_dir,
            keywords_file=kws_keywords_file,
            kws_num_threads=kws_num_threads,
            pre_roll_seconds=kws_pre_roll_seconds,
            kws_score=kws_score,
            kws_threshold=kws_threshold,
            wake_active_timeout_seconds=wake_active_timeout_seconds,
            rule1_min_trailing_silence=asr_rule1_min_trailing_silence,
            rule2_min_trailing_silence=asr_rule2_min_trailing_silence,
            rule3_min_utterance_length=asr_rule3_min_utterance_length,
            keep_model_loaded=keep_model_loaded,
        )
    raise ValueError(
        f"Unsupported STT provider: {provider}",
    )


# ----------------------------------------------------------
# Non-streaming TTS (legacy, kept for fallback / tests)
# ----------------------------------------------------------


async def synthesize_tts(
    provider: str,
    text: str,
    voice: str,
    api_key: str = "",
) -> bytes:
    """Synthesize *text* with the legacy Aliyun API and return WAV bytes."""
    if provider == "aliyun":
        return await _synthesize_aliyun(
            text,
            voice,
            _resolve_dashscope_key(api_key),
        )
    raise ValueError(
        f"Unsupported TTS provider: {provider}",
    )


async def _synthesize_aliyun(
    text: str,
    voice: str,
    api_key: str = "",
) -> bytes:
    from dashscope.audio.tts import SpeechSynthesizer

    response = await asyncio.to_thread(
        SpeechSynthesizer.call,
        model=voice or "sambert-zhichu-v1",
        text=text,
        sample_rate=8000,
        format="wav",
        api_key=api_key or None,
    )
    if response and hasattr(response, "get_audio_data"):
        return response.get_audio_data() or b""
    return b""


# ----------------------------------------------------------
# Streaming TTS providers
# ----------------------------------------------------------


async def synthesize_tts_stream(
    provider: str,
    text: str,
    voice: str,
    api_key: str = "",
    *,
    sample_rate: int = 8000,
    speed: float = 1.0,
    kokoro_model_dir: str = "",
    kokoro_model_variant: str = "float32",
    kokoro_num_threads: int = 2,
    kokoro_silence_scale: float = 0.2,
    openai_api_key: str = "",
    openai_base_url: str = "",
    openai_model: str = _OPENAI_TTS_DEFAULT_MODEL,
    qwen3_backend: str = "api",
    qwen3_model: str = "",
    qwen3_model_dir: str = "",
    qwen3_ref_audio: str = "",
    qwen3_ref_text: str = "",
    qwen3_device: str = "cpu",
    keep_model_loaded: bool = True,
) -> AsyncIterator[bytes]:
    """Yield raw 16-bit mono PCM chunks as they arrive from TTS.

    ``provider`` accepts ``aliyun``, ``edge_tts`` (or ``edgetts``),
    ``kokoro``, ``openai``, and ``qwen3`` (Qwen3-TTS 0.6B via DashScope
    API or local ``qwen_tts``). Kokoro, openai, and qwen3 synthesis are
    offline (the whole utterance is generated before playback starts),
    while aliyun and edge_tts stream over the network. ``api_key`` is the
    DashScope key; OpenAI-compatible credentials come from
    ``openai_api_key`` / ``OPENAI_API_KEY``.
    ``keep_model_loaded`` affects the local Kokoro and local Qwen3 models.
    ``speed`` is a global speech-rate multiplier (kokoro/openai/edge_tts);
    providers without native rate control ignore it.
    """
    provider_key = normalize_tts_provider(provider)

    if provider_key == "aliyun":
        async for chunk in _stream_aliyun(
            text,
            voice,
            _resolve_dashscope_key(api_key),
            sample_rate=sample_rate,
        ):
            yield chunk
        return

    if provider_key == "openai":
        async for chunk in _stream_openai_tts(
            text,
            voice,
            api_key=_resolve_openai_key(openai_api_key),
            sample_rate=sample_rate,
            base_url=openai_base_url,
            model=openai_model,
            speed=speed,
        ):
            yield chunk
        return

    if provider_key == "edge_tts":
        async for chunk in _stream_edge_tts(
            text,
            voice,
            sample_rate=sample_rate,
            speed=speed,
        ):
            yield chunk
        return

    if provider_key == "qwen3":
        async for chunk in _stream_qwen3(
            text,
            voice,
            api_key=_resolve_dashscope_key(api_key),
            sample_rate=sample_rate,
            backend=qwen3_backend,
            model=qwen3_model,
            model_dir=qwen3_model_dir,
            ref_audio=qwen3_ref_audio,
            ref_text=qwen3_ref_text,
            device=qwen3_device,
            keep_model_loaded=keep_model_loaded,
        ):
            yield chunk
        return

    if provider_key == "kokoro":
        async for chunk in _stream_kokoro(
            text,
            voice,
            sample_rate=sample_rate,
            speed=speed,
            silence_scale=kokoro_silence_scale,
            model_dir=kokoro_model_dir,
            model_variant=kokoro_model_variant,
            num_threads=kokoro_num_threads,
            keep_model_loaded=keep_model_loaded,
        ):
            yield chunk
        return

    # normalize_tts_provider() makes this unreachable.
    raise ValueError(  # pragma: no cover
        f"Unsupported TTS provider: {provider}",
    )


async def _stream_aliyun(
    text: str,
    voice: str,
    api_key: str = "",
    *,
    sample_rate: int = 8000,
) -> AsyncIterator[bytes]:
    from dashscope.audio.tts_v2 import (
        AudioFormat,
        ResultCallback,
        SpeechSynthesizer,
    )

    # tts_v2 SpeechSynthesizer doesn't accept api_key as __init__ arg;
    # set it via the module-level variable instead.
    if api_key:
        import dashscope

        dashscope.api_key = api_key

    fmt_map = {
        8000: AudioFormat.PCM_8000HZ_MONO_16BIT,
        16000: AudioFormat.PCM_16000HZ_MONO_16BIT,
        22050: AudioFormat.PCM_22050HZ_MONO_16BIT,
        24000: AudioFormat.PCM_24000HZ_MONO_16BIT,
    }
    audio_fmt = fmt_map.get(
        sample_rate,
        AudioFormat.PCM_8000HZ_MONO_16BIT,
    )

    queue: thread_queue.Queue[bytes | None] = thread_queue.Queue(
        maxsize=_TTS_QUEUE_MAX_CHUNKS,
    )
    stopped = threading.Event()

    def _enqueue(chunk: bytes | None) -> None:
        """Block the SDK callback until playback makes room or is cancelled."""
        while not stopped.is_set():
            try:
                queue.put(chunk, timeout=0.1)
                return
            except thread_queue.Full:
                pass

    class _Callback(ResultCallback):
        def on_data(self, data: bytes) -> None:
            _enqueue(data)

        def on_complete(self) -> None:
            _enqueue(None)

        def on_error(self, message: str) -> None:
            logger.error("TTS stream error: %s", message)
            _enqueue(None)

        def on_close(self) -> None:
            pass

        def on_open(self) -> None:
            pass

        def on_event(self, message) -> None:
            pass

    callback = _Callback()
    synthesizer = SpeechSynthesizer(
        model="cosyvoice-v1",
        voice=voice or "longxiaochun",
        format=audio_fmt,
        callback=callback,
    )

    # Run call() in a background thread; it blocks until synthesis
    # completes, but on_data callbacks fire in the WS thread and
    # push chunks into the queue in real time.
    synth_task = asyncio.get_running_loop().run_in_executor(
        None,
        synthesizer.call,
        text,
    )

    completed = False
    try:
        # ``Queue.get`` is synchronous because DashScope invokes callbacks
        # from its own thread. A timed get also lets us notice a producer that
        # exited without delivering its terminal callback.
        while True:
            try:
                chunk = await asyncio.to_thread(queue.get, True, 0.1)
            except thread_queue.Empty:
                if synth_task.done():
                    break
                continue
            if chunk is None:
                break
            yield chunk
        completed = True
    finally:
        stopped.set()

    # Surface synthesis failures after normal completion. On cancellation the
    # callback observes ``stopped`` and the executor future finishes itself.
    if completed:
        await synth_task


# ----------------------------------------------------------
# OpenAI-compatible TTS API
# ----------------------------------------------------------


async def _stream_openai_tts(
    text: str,
    voice: str,
    *,
    api_key: str = "",
    sample_rate: int = 8000,
    base_url: str = "",
    model: str = "",
    speed: float = 1.0,
) -> AsyncIterator[bytes]:
    """Synthesize with an OpenAI-compatible ``/v1/audio/speech`` endpoint.

    The endpoint returns the complete compressed audio payload (MP3 by
    default), so it is decoded to raw PCM with ffmpeg before playback
    starts -- the same approach used by the edge-tts provider.
    """
    endpoint_base = resolve_openai_base_url(base_url)
    active_model = (model or "").strip() or _OPENAI_TTS_DEFAULT_MODEL
    active_voice = (voice or "").strip() or _OPENAI_TTS_DEFAULT_VOICE
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": active_model,
        "input": text,
        "voice": active_voice,
        "response_format": "mp3",
        "speed": max(0.25, min(4.0, float(speed))),
    }
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30, read=120),
        ) as client:
            response = await client.post(
                f"{endpoint_base}/audio/speech",
                headers=headers,
                json=payload,
            )
            if response.status_code >= 400:
                detail = response.text[:400]
                raise ValueError(
                    f"OpenAI-compatible TTS request failed "
                    f"(HTTP {response.status_code}): {detail}",
                )
            audio = response.content
    except httpx.HTTPError as exc:
        raise ValueError(
            f"OpenAI-compatible TTS request failed: {exc}",
        ) from exc
    if not audio:
        raise ValueError(
            f"OpenAI-compatible TTS returned no audio for text: "
            f"{text[:80]!r}",
        )

    pcm = await asyncio.to_thread(
        _decode_mp3_to_pcm16,
        audio,
        sample_rate,
    )
    for piece in _chunk_bytes(pcm, _pcm_chunk_size(sample_rate)):
        yield piece


# ----------------------------------------------------------
# edge-tts (Microsoft Edge online TTS)
# ----------------------------------------------------------


async def _stream_edge_tts(
    text: str,
    voice: str,
    *,
    sample_rate: int = 8000,
    speed: float = 1.0,
) -> AsyncIterator[bytes]:
    """Synthesize with edge-tts and yield PCM chunks.

    edge-tts returns MP3 chunks, so the stream is buffered and decoded to
    raw PCM with ffmpeg before playback starts. This trades a little
    first-byte latency for a simple, dependency-light integration.
    """
    try:
        import edge_tts
    except ImportError as exc:
        raise ValueError(
            "edge_tts is not installed. "
            "Install it with: pip install 'qwenpaw[sip]'",
        ) from exc

    voice_name = voice or _EDGE_TTS_DEFAULT_VOICE
    # edge-tts rate is expressed as a percentage offset. Stay inside the
    # provider's safe -50%..+50% range (speed 0.5..1.5); values beyond it
    # are rejected or ignored by the underlying Azure service.
    safe_speed = max(0.5, min(1.5, float(speed)))
    rate_str = f"{int(round((safe_speed - 1.0) * 100)):+.0f}%"
    communicate = edge_tts.Communicate(text, voice_name, rate=rate_str)
    mp3 = bytearray()
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio":
            mp3.extend(chunk.get("data") or b"")
    if not mp3:
        raise ValueError(
            f"edge-tts returned no audio for text: {text[:80]!r}",
        )

    pcm = await asyncio.to_thread(
        _decode_mp3_to_pcm16,
        bytes(mp3),
        sample_rate,
    )
    for piece in _chunk_bytes(pcm, _pcm_chunk_size(sample_rate)):
        yield piece


def _decode_mp3_to_pcm16(mp3: bytes, sample_rate: int) -> bytes:
    """Decode MP3 bytes to raw 16-bit mono PCM at *sample_rate*.

    Looks for ffmpeg on PATH first, then honors an explicit
    ``FFMPEG_PATH`` environment variable (e.g. a manually downloaded
    ffmpeg.exe that has not been added to PATH).
    """
    ffmpeg = shutil.which("ffmpeg")
    explicit = os.environ.get("FFMPEG_PATH", "").strip()
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.is_file():
            ffmpeg = str(candidate)
        elif not ffmpeg:
            raise ValueError(
                f"FFMPEG_PATH does not point to an executable: {explicit}",
            )
    if not ffmpeg:
        raise ValueError(
            "ffmpeg is required on PATH to decode TTS audio. "
            "Install ffmpeg or switch to a provider that returns PCM.",
        )
    proc = subprocess.run(
        [
            ffmpeg,
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "pipe:1",
        ],
        input=mp3,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or b"").decode("utf-8", "replace").strip()
        raise ValueError(
            f"ffmpeg failed to decode edge-tts audio: {detail}",
        )
    return proc.stdout


# ----------------------------------------------------------
# Qwen3-TTS 0.6B (DashScope API / local qwen_tts)
# ----------------------------------------------------------


async def _stream_qwen3(
    text: str,
    voice: str,
    *,
    api_key: str = "",
    sample_rate: int = 8000,
    backend: str = "api",
    model: str = "",
    model_dir: str = "",
    ref_audio: str = "",
    ref_text: str = "",
    device: str = "cpu",
    keep_model_loaded: bool = True,
) -> AsyncIterator[bytes]:
    """Dispatch Qwen3-TTS to the DashScope API or a local model."""
    if normalize_qwen3_backend(backend) == "api":
        async for chunk in _stream_qwen3_api(
            text,
            voice,
            api_key=api_key,
            sample_rate=sample_rate,
            model=model,
        ):
            yield chunk
        return
    async for chunk in _stream_qwen3_local(
        text,
        sample_rate=sample_rate,
        model_dir=model_dir,
        ref_audio=ref_audio,
        ref_text=ref_text,
        device=device,
        keep_model_loaded=keep_model_loaded,
    ):
        yield chunk


async def _stream_qwen3_api(
    text: str,
    voice: str,
    *,
    api_key: str = "",
    sample_rate: int = 8000,
    model: str = "",
) -> AsyncIterator[bytes]:
    """Synthesize with DashScope Qwen3-TTS over HTTP.

    System voices use ``qwen3-tts-flash`` (or the configured model).
    Cloned voice ids are automatically routed to the dedicated VC model
    so the enrolled timbre is preserved.
    """
    from dashscope.audio.qwen_tts import SpeechSynthesizer

    active_model = resolve_qwen3_api_model(model, voice)
    active_voice = (voice or "").strip() or _QWEN3_DEFAULT_API_VOICE
    if not api_key:
        raise ValueError("Qwen3-TTS API requires a DashScope API key")

    response = await asyncio.to_thread(
        SpeechSynthesizer.call,
        model=active_model,
        text=text,
        voice=active_voice,
        api_key=api_key or None,
    )
    if getattr(response, "status_code", 0) != 200:
        detail = getattr(response, "message", "") or getattr(
            response,
            "code",
            "",
        )
        raise ValueError(
            f"Qwen3-TTS API request failed: {detail}",
        )
    audio_info = getattr(getattr(response, "output", None), "audio", None)
    audio_url = str(getattr(audio_info, "url", "") or "")
    audio_data = str(getattr(audio_info, "data", "") or "")
    if not audio_url and not audio_data:
        raise ValueError("Qwen3-TTS API returned no audio")
    audio = await asyncio.to_thread(
        _load_qwen3_api_audio,
        audio_url,
        audio_data,
    )
    if not audio:
        raise ValueError("Qwen3-TTS API returned an empty audio payload")

    pcm = await asyncio.to_thread(
        _decode_mp3_to_pcm16,
        audio,
        sample_rate,
    )
    for piece in _chunk_bytes(pcm, _pcm_chunk_size(sample_rate)):
        yield piece


def _load_qwen3_api_audio(url: str, data: str) -> bytes:
    """Return audio bytes from a Qwen3-TTS response URL or data field."""
    with httpx.Client(
        timeout=httpx.Timeout(30, read=120),
        follow_redirects=True,
    ) as client:
        if url:
            return client.get(url).raise_for_status().content
        value = data.strip()
        if value.startswith(("http://", "https://")):
            return client.get(value).raise_for_status().content
        if value.startswith("data:"):
            value = value.split(",", maxsplit=1)[1]
        return base64.b64decode(value)


def _qwen3_clone_fingerprint(
    model_path: str,
    ref_audio: str,
    ref_text: str,
) -> str:
    """Stable fingerprint for a (model, reference audio, transcript) tuple."""
    digest = hashlib.sha256()
    digest.update(model_path.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(ref_text.encode("utf-8"))
    digest.update(b"\x00")
    parsed = urlparse(ref_audio)
    if parsed.scheme in {"http", "https", "oss"} or (
        not parsed.scheme and not Path(ref_audio).is_file()
    ):
        # Remote URL or base64-style value: hash the value itself.
        digest.update(ref_audio.encode("utf-8"))
    else:
        path = (
            Path(url2pathname(parsed.path))
            if parsed.scheme == "file"
            else Path(ref_audio)
        ).expanduser()
        try:
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError:
            digest.update(str(path).encode("utf-8"))
    return digest.hexdigest()


def _qwen3_clone_prompt_file(fingerprint: str) -> Path:
    return _QWEN3_CLONE_CACHE_DIR / f"voice-clone-{fingerprint}.pt"


def _save_qwen3_clone_prompt(path: Path, prompt: dict) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prompt, path)


def _load_qwen3_clone_prompt(path: Path) -> dict | None:
    try:
        import torch
    except ImportError:
        return None
    if not path.is_file():
        return None
    try:
        prompt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        logger.warning(
            "Could not load cached Qwen3-TTS clone prompt: %s",
            path,
            exc_info=True,
        )
        return None
    if not isinstance(prompt, dict):
        return None
    return prompt


def _prompt_items_to_dict(items: Any) -> dict:
    """Convert ``VoiceClonePromptItem`` objects to a serializable dict."""
    return {
        "ref_code": [getattr(item, "ref_code", None) for item in items],
        "ref_spk_embedding": [
            getattr(item, "ref_spk_embedding", None) for item in items
        ],
        "x_vector_only_mode": [
            getattr(item, "x_vector_only_mode", False) for item in items
        ],
        "icl_mode": [getattr(item, "icl_mode", False) for item in items],
        "ref_text": [getattr(item, "ref_text", None) for item in items],
    }


def _dict_to_prompt_items(prompt: dict) -> Any | None:
    """Rebuild ``VoiceClonePromptItem`` objects from a cached dict.

    Passing the item list back to ``generate_voice_clone`` is important:
    the official implementation derives ``ref_ids`` from item ``ref_text``
    values and otherwise falls back to ``ref_ids=None``, which can produce
    ``'NoneType' object is not subscriptable`` on some qwen-tts versions.
    """
    required = {
        "ref_code",
        "ref_spk_embedding",
        "x_vector_only_mode",
        "icl_mode",
        "ref_text",
    }
    if not required.issubset(prompt):
        return None
    try:
        from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
    except ImportError:
        return None

    counts = {len(value) for value in prompt.values()}
    if len(counts) != 1:
        return None
    items = []
    for index in range(next(iter(counts))):
        items.append(
            VoiceClonePromptItem(
                ref_code=prompt["ref_code"][index],
                ref_spk_embedding=prompt["ref_spk_embedding"][index],
                x_vector_only_mode=bool(
                    prompt["x_vector_only_mode"][index],
                ),
                icl_mode=bool(prompt["icl_mode"][index]),
                ref_text=prompt["ref_text"][index],
            )
        )
    return items


def _load_or_create_qwen3_clone_prompt(
    model: Any,
    *,
    model_path: str,
    ref_audio: str,
    ref_text: str,
) -> Any:
    """Load a cached clone prompt or compute and persist it once."""
    fingerprint = _qwen3_clone_fingerprint(model_path, ref_audio, ref_text)
    prompt_file = _qwen3_clone_prompt_file(fingerprint)
    cached = _load_qwen3_clone_prompt(prompt_file)
    if cached is not None:
        items = _dict_to_prompt_items(cached)
        if items is not None:
            return items
    items = model.create_voice_clone_prompt(
        ref_audio=ref_audio,
        ref_text=ref_text,
    )
    prompt = _prompt_items_to_dict(items)
    try:
        _save_qwen3_clone_prompt(prompt_file, prompt)
    except Exception:
        logger.warning(
            "Could not persist Qwen3-TTS clone prompt to %s",
            prompt_file,
            exc_info=True,
        )
    return items


async def _stream_qwen3_local(
    text: str,
    *,
    sample_rate: int = 8000,
    model_dir: str = "",
    ref_audio: str = "",
    ref_text: str = "",
    device: str = "cpu",
    keep_model_loaded: bool = True,
) -> AsyncIterator[bytes]:
    """Synthesize with a local Qwen3-TTS-0.6B model and a cloned voice."""
    reference_audio = (ref_audio or "").strip()
    reference_text = (ref_text or "").strip()
    if not reference_audio or not reference_text:
        raise ValueError(
            "Qwen3-TTS local mode clones a voice from a reference sample; "
            "set qwen3_ref_audio and qwen3_ref_text",
        )
    model, voice_clone_prompt = await asyncio.to_thread(
        _get_qwen3_local_tts,
        model_dir,
        device,
        reference_audio,
        reference_text,
        keep_model_loaded,
    )

    def _generate() -> tuple[Any, int]:
        # Keep sampling enabled so synthesis terminates naturally and keeps
        # expressive prosody, but use a fixed seed for every request. The
        # subtalker is forced deterministic to avoid residual-codebook drift
        # that changes the perceived timbre between utterances.
        import torch

        with _qwen3_generation_lock:
            cpu_rng_state = torch.get_rng_state()
            cuda_rng_states = None
            if torch.cuda.is_available():
                cuda_rng_states = torch.cuda.get_rng_state_all()
            try:
                torch.manual_seed(0)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(0)
                wavs, sample_rate_out = model.generate_voice_clone(
                    text=text,
                    language="Auto",
                    voice_clone_prompt=voice_clone_prompt,
                    max_new_tokens=2048,
                    do_sample=True,
                    top_k=50,
                    top_p=1.0,
                    temperature=0.9,
                    repetition_penalty=1.05,
                    subtalker_dosample=False,
                    subtalker_top_k=1,
                    subtalker_top_p=1.0,
                    subtalker_temperature=1.0,
                )
            finally:
                # Restore the process-wide RNG state so other torch users in
                # this process are not affected by the fixed generation seed.
                torch.set_rng_state(cpu_rng_state)
                if cuda_rng_states is not None:
                    torch.cuda.set_rng_state_all(cuda_rng_states)
        return wavs, int(sample_rate_out)

    try:
        wavs, source_rate = await asyncio.to_thread(_generate)
    except RuntimeError as exc:
        if "no kernel image" in str(exc):
            raise ValueError(
                "The installed PyTorch CUDA build does not support this GPU. "
                "RTX 50 series (Blackwell) needs a CUDA 12.8+ PyTorch wheel. "
                "Set the PyTorch mirror to "
                "https://download.pytorch.org/whl/cu128, uninstall and "
                "reinstall torch_cuda, then restart QwenPaw.",
            ) from exc
        raise
    if isinstance(wavs, (list, tuple)):
        if not wavs:
            raise ValueError("Qwen3-TTS local generated no audio")
        samples = wavs[0]
    else:
        samples = wavs
    if samples is None or len(samples) == 0:
        raise ValueError("Qwen3-TTS local generated no audio")

    pcm = _samples_to_pcm16(samples)
    pcm = _resample_pcm16(pcm, source_rate, sample_rate)
    for piece in _chunk_bytes(pcm, _pcm_chunk_size(sample_rate)):
        yield piece


def _get_qwen3_local_tts(
    model_dir: str,
    device: str,
    ref_audio: str,
    ref_text: str,
    keep_model_loaded: bool,
) -> tuple[Any, Any]:
    """Load (and optionally cache) the Qwen3-TTS model plus clone prompt."""
    try:
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        raise ValueError(
            "qwen-tts is not installed. Use the Local Voice dependency "
            "installer in the settings page, or run: pip install qwen-tts",
        ) from exc
    except OSError as exc:
        raise ValueError(
            "qwen-tts is installed but its native audio dependencies could "
            "not be loaded. On Windows this is usually a torchaudio/torch "
            "version mismatch (for example torchaudio 2.11 with torch "
            "2.6.0+cu124) or a missing MSVC/CUDA runtime DLL. Reinstall the "
            "Local Voice dependencies so the installer can align torchaudio "
            "with the selected PyTorch build, or run: pip install "
            "torchaudio==<torch-version> --index-url "
            "https://download.pytorch.org/whl/cu128",
        ) from exc

    import torch

    requested_device = (device or "cpu").strip().lower() or "cpu"
    effective_device = requested_device
    if (
        requested_device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        logger.warning(
            "CUDA requested (%s) but torch has no CUDA support; "
            "falling back to cpu",
            requested_device,
        )
        effective_device = "cpu"

    path = resolve_qwen3_model_dir(model_dir)
    key = (path, effective_device, ref_audio, ref_text)
    with _qwen3_local_tts_lock:
        cached = _qwen3_local_tts_cache.get(key)
        if cached is not None and keep_model_loaded:
            return cached
        if not keep_model_loaded:
            _qwen3_local_tts_cache.pop(key, None)

        try:
            model = Qwen3TTSModel.from_pretrained(
                path,
                device_map=effective_device,
            )
        except RuntimeError as exc:
            if "Torch not compiled with CUDA" in str(exc):
                raise ValueError(
                    "This torch build has no CUDA support; select cpu as "
                    "the Qwen3-TTS device or install a CUDA-enabled torch.",
                ) from exc
            raise
        voice_clone_prompt = _load_or_create_qwen3_clone_prompt(
            model,
            model_path=path,
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
        cached = (model, voice_clone_prompt)
        if keep_model_loaded:
            _qwen3_local_tts_cache[key] = cached
        return cached


def _upload_qwen3_audio_to_dashscope_sync(
    path: Path,
    *,
    api_key: str,
    model: str,
) -> str:
    """Upload one local clone sample to DashScope model-bound temp OSS."""
    media_type = mimetypes.guess_type(path.name)[0] or "audio/wav"
    with httpx.Client(
        timeout=httpx.Timeout(30, read=300, write=3600),
        follow_redirects=True,
    ) as client:
        policy_response = client.get(
            _QWEN3_DASHSCOPE_UPLOAD_URL,
            params={"action": "getPolicy", "model": model},
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        policy_response.raise_for_status()
        payload = policy_response.json()
        policy = payload.get("data", payload)
        upload_dir = str(policy["upload_dir"]).rstrip("/")
        key = f"{upload_dir}/{uuid.uuid4().hex}-{path.name}"
        form = {
            "OSSAccessKeyId": str(policy["oss_access_key_id"]),
            "Signature": str(policy["signature"]),
            "policy": str(policy["policy"]),
            "x-oss-object-acl": str(policy.get("x_oss_object_acl", "private")),
            "x-oss-forbid-overwrite": str(
                policy.get("x_oss_forbid_overwrite", "true"),
            ),
            "key": key,
            "success_action_status": "200",
        }
        with path.open("rb") as handle:
            upload_response = client.post(
                str(policy["upload_host"]),
                data=form,
                files={"file": (path.name, handle, media_type)},
            )
        upload_response.raise_for_status()
        return f"oss://{key}"


async def _qwen3_clone_media_url(
    media: str,
    *,
    api_key: str,
    model: str,
) -> str:
    """Return an API-fetchable URL for a clone sample."""
    value = (media or "").strip()
    if not value:
        raise ValueError("Qwen3-TTS voice clone requires a reference audio")
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https", "oss"}:
        return value
    if parsed.scheme and parsed.scheme != "file":
        raise ValueError(
            "Qwen3-TTS clone sample must be a local file or HTTP(S)/OSS URL",
        )
    if parsed.scheme == "file":
        path = Path(url2pathname(parsed.path)).expanduser()
    else:
        path = Path(value).expanduser()
    if not path.is_file():
        raise ValueError(f"Qwen3-TTS clone sample not found: {path}")
    return await asyncio.to_thread(
        _upload_qwen3_audio_to_dashscope_sync,
        path,
        api_key=api_key,
        model=model,
    )


async def clone_qwen3_voice(
    media: str,
    *,
    preferred_name: str,
    api_key: str = "",
    model: str = "",
) -> str:
    """Clone a voice from a local/remote audio sample via DashScope.

    Returns the ``voice_id`` that should be used as the Qwen3-TTS voice.
    ``model`` is an optional dedicated VC model override; synthesis models
    such as ``qwen3-tts-flash`` are automatically mapped to the default VC
    model because cloned voices are bound to their enrollment model.
    """
    key = _resolve_dashscope_key(api_key)
    if not key:
        raise ValueError("Qwen3-TTS voice clone requires a DashScope API key")
    target_model = (model or "").strip() or _QWEN3_DEFAULT_CLONE_MODEL
    if target_model in _QWEN3_API_SYNTHESIS_MODELS:
        # A synthesis model cannot be used as an enrollment target. Cloned
        # voices are bound to the dedicated VC model; using flash here can
        # produce a voice_id that does not keep the same timbre in synthesis.
        target_model = _QWEN3_DEFAULT_CLONE_MODEL
    media_url = await _qwen3_clone_media_url(
        media,
        api_key=key,
        model=target_model,
    )

    def _clone() -> str:
        from dashscope.audio.tts_v2.enrollment import VoiceEnrollmentService

        service = VoiceEnrollmentService(api_key=key)
        return service.create_voice(
            target_model=target_model,
            prefix=_normalize_qwen3_prefix(preferred_name),
            url=media_url,
        )

    return await asyncio.to_thread(_clone)


# ----------------------------------------------------------
# Kokoro (local sherpa-onnx TTS)
# ----------------------------------------------------------


async def _stream_kokoro(
    text: str,
    voice: str,
    *,
    sample_rate: int = 8000,
    speed: float = 1.0,
    silence_scale: float = 0.2,
    model_dir: str = "",
    model_variant: str = "float32",
    num_threads: int = 2,
    keep_model_loaded: bool = True,
) -> AsyncIterator[bytes]:
    """Synthesize with a local Kokoro model and yield PCM chunks."""
    model_path = resolve_kokoro_model_dir(model_dir, model_variant)
    validate_kokoro_model_dir(model_path)
    tts, sherpa_onnx = await asyncio.to_thread(
        _get_kokoro_tts,
        model_path,
        num_threads,
        keep_model_loaded,
    )
    sid = resolve_kokoro_sid(voice, model_path)
    rate = max(0.5, min(4.0, float(speed)))
    pause = max(0.0, min(3.0, float(silence_scale)))

    def _generate() -> Any:
        gen_config = sherpa_onnx.GenerationConfig()
        gen_config.sid = sid
        gen_config.speed = rate
        gen_config.silence_scale = pause
        return tts.generate(text, gen_config)

    audio = await asyncio.to_thread(_generate)
    samples = getattr(audio, "samples", None)
    if samples is None or len(samples) == 0:
        raise ValueError(
            f"Kokoro generated no audio for text: {text[:80]!r}",
        )

    pcm = _samples_to_pcm16(samples)
    src_rate = int(
        getattr(audio, "sample_rate", 0) or _KOKORO_NATIVE_SAMPLE_RATE,
    )
    pcm = _resample_pcm16(pcm, src_rate, sample_rate)
    for piece in _chunk_bytes(pcm, _pcm_chunk_size(sample_rate)):
        yield piece


def resolve_kokoro_model_dir(
    model_dir: str = "",
    model_variant: str = "float32",
) -> Path:
    """Resolve the Kokoro model directory.

    Priority: explicit *model_dir*, ``KOKORO_MODEL_DIR`` env var, then the
    QwenPaw models directory. ``float32`` selects ``kokoro-multi-lang-v1_1``;
    ``int8`` selects ``kokoro-int8-multi-lang-v1_1`` for lower CPU use.
    """
    raw = model_dir or os.environ.get("KOKORO_MODEL_DIR", "")
    if raw:
        return Path(raw).expanduser()
    variant = (model_variant or "float32").strip().lower()
    if variant not in {"float32", "int8"}:
        raise ValueError("kokoro_model_variant must be 'float32' or 'int8'")
    name = "kokoro-int8-multi-lang-v1_1" if variant == "int8" else "kokoro-multi-lang-v1_1"
    return MODELS_DIR / name


def validate_kokoro_model_dir(model_dir: Path) -> None:
    """Check that *model_dir* contains a usable sherpa-onnx Kokoro model."""
    if not model_dir.is_dir():
        raise ValueError(
            f"Kokoro model directory not found: {model_dir}. "
            "Download a Kokoro v1.1 model and set kokoro_model_dir, or "
            "set KOKORO_MODEL_DIR.",
        )
    required = (
        "voices.bin",
        "tokens.txt",
        "espeak-ng-data",
    )
    missing = [name for name in required if not (model_dir / name).exists()]
    if not any((model_dir / name).is_file() for name in ("model.onnx", "model.int8.onnx")):
        missing.append("model.onnx or model.int8.onnx")
    if missing:
        raise ValueError(
            f"Kokoro model directory is incomplete: {model_dir}. "
            f"Missing: {', '.join(missing)}",
        )


def resolve_kokoro_sid(voice: str, model_dir: Path | None = None) -> int:
    """Resolve a Kokoro v1.0/v1.1 speaker name or numeric id.

    v1.1-zh starts with ``af_maple``, ``af_sol``, ``bf_vale``, followed by
    contiguous ``zf_001`` names. The v1.0 map remains available for existing
    model directories and saved configurations.
    """
    name = (voice or "").strip()
    if not name:
        # A persisted v1.0 custom model directory must retain its old default,
        # while new installations (which resolve to v1.1) use zf_001.
        if model_dir is None:
            return 3
        name = _KOKORO_DEFAULT_VOICE if _is_kokoro_v11(model_dir) else "zf_xiaobei"
    if name.isdigit():
        sid = int(name)
        if 0 <= sid <= 255:
            return sid
    if _is_kokoro_v11(model_dir):
        if name in _KOKORO_V11_BASE_SPEAKERS:
            return _KOKORO_V11_BASE_SPEAKERS[name]
        if name.startswith("zf_") and name[3:].isdigit():
            index = int(name[3:])
            if 1 <= index <= 99:
                return 2 + index
        raise ValueError(
            f"Unknown Kokoro v1.1 voice: {voice!r}. Use zf_001 or a numeric speaker id.",
        )
    if name == _KOKORO_DEFAULT_VOICE:
        # New default config remains usable when an existing v1.0 directory
        # is selected explicitly.
        return _KOKORO_SPEAKERS["zf_xiaobei"]
    if name in _KOKORO_SPEAKERS:
        return _KOKORO_SPEAKERS[name]
    known = ", ".join(
        sorted(_KOKORO_SPEAKERS),
    )
    raise ValueError(
        f"Unknown Kokoro voice: {voice!r}. "
        "Use a speaker id or one of: "
        f"{known}",
    )


def _get_kokoro_tts(
    model_dir: Path,
    num_threads: int,
    keep_model_loaded: bool = True,
) -> tuple[Any, Any]:
    """Load the Kokoro OfflineTts instance and sherpa_onnx.

    When *keep_model_loaded* is true the pair is cached process-wide;
    otherwise a fresh instance is created for the caller and any cached
    copy for the same key is dropped so memory can be reclaimed.
    """
    key = (str(model_dir.resolve()), int(num_threads))
    with _kokoro_tts_lock:
        cached = _kokoro_tts_cache.get(key)
        if cached is not None and keep_model_loaded:
            return cached
        if not keep_model_loaded:
            _kokoro_tts_cache.pop(key, None)

        try:
            import sherpa_onnx
        except ImportError as exc:
            raise ValueError(
                "sherpa-onnx is not installed. "
                "Install it with: pip install 'qwenpaw[sip]'",
            ) from exc

        lexicon = _existing_join(
            model_dir,
            "lexicon-us-en.txt",
            "lexicon-zh.txt",
        )
        rule_fsts = _existing_join(
            model_dir,
            "date-zh.fst",
            "phone-zh.fst",
            "number-zh.fst",
        )
        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                    model=str(_kokoro_model_file(model_dir)),
                    voices=str(model_dir / "voices.bin"),
                    tokens=str(model_dir / "tokens.txt"),
                    data_dir=str(model_dir / "espeak-ng-data"),
                    lexicon=lexicon,
                ),
                provider="cpu",
                debug=False,
                num_threads=int(num_threads),
            ),
            rule_fsts=rule_fsts,
            max_num_sentences=2,
        )
        if not tts_config.validate():
            raise ValueError(
                f"Invalid Kokoro model directory: {model_dir}",
            )
        tts = sherpa_onnx.OfflineTts(tts_config)
        cached = (tts, sherpa_onnx)
        if keep_model_loaded:
            _kokoro_tts_cache[key] = cached
        return cached


async def warmup_tts(
    provider: str,
    *,
    kokoro_model_dir: str = "",
    kokoro_model_variant: str = "float32",
    kokoro_num_threads: int = 2,
    qwen3_backend: str = "api",
    qwen3_model_dir: str = "",
    qwen3_ref_audio: str = "",
    qwen3_ref_text: str = "",
    qwen3_device: str = "cpu",
    keep_model_loaded: bool = True,
) -> None:
    """Load a local TTS model before the first interactive response.

    With *keep_model_loaded* disabled the warmup is skipped entirely: the
    model is loaded on demand for each synthesis and released afterwards.
    """
    if not keep_model_loaded:
        return
    provider_key = normalize_tts_provider(provider)
    if provider_key == "kokoro":
        model_path = resolve_kokoro_model_dir(
            kokoro_model_dir,
            kokoro_model_variant,
        )
        validate_kokoro_model_dir(model_path)
        await asyncio.to_thread(
            _get_kokoro_tts,
            model_path,
            kokoro_num_threads,
            keep_model_loaded,
        )
        return
    if (
        provider_key == "qwen3"
        and normalize_qwen3_backend(qwen3_backend) == "local"
    ):
        ref_audio = (qwen3_ref_audio or "").strip()
        ref_text = (qwen3_ref_text or "").strip()
        if not ref_audio or not ref_text:
            raise ValueError(
                "Qwen3-TTS local warmup requires qwen3_ref_audio and "
                "qwen3_ref_text",
            )
        await asyncio.to_thread(
            _get_qwen3_local_tts,
            qwen3_model_dir,
            qwen3_device,
            ref_audio,
            ref_text,
            keep_model_loaded,
        )


def _kokoro_model_file(model_dir: Path) -> Path:
    """Select float32 or int8 model file in a validated Kokoro directory."""
    for name in ("model.onnx", "model.int8.onnx"):
        candidate = model_dir / name
        if candidate.is_file():
            return candidate
    raise ValueError(f"Kokoro model file not found in {model_dir}")


def _is_kokoro_v11(model_dir: Path | None) -> bool:
    if model_dir is None:
        return False
    if "v1_1" in model_dir.name.lower():
        return True
    # Custom directories need not retain the archive name. The official v1.1
    # release includes a README identifying the version, which keeps named
    # speaker selection working after users relocate the extracted directory.
    readme = model_dir / "README.md"
    try:
        return "v1.1" in readme.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False


def _existing_join(model_dir: Path, *names: str) -> str:
    """Join existing files under *model_dir* as a comma-separated string."""
    return ",".join(
        str(model_dir / name) for name in names if (model_dir / name).is_file()
    )


def _samples_to_pcm16(samples: Any) -> bytes:
    """Convert float32 samples in [-1, 1] to little-endian int16 PCM."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        # numpy ships with onnxruntime, so this is defensive.
        raise ValueError("numpy is required for Kokoro TTS") from exc

    arr = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (arr * 32767.0).astype(np.int16)
    return pcm.tobytes()


def _resample_pcm16(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Resample raw 16-bit mono PCM with ``audioop``."""
    if src_rate == dst_rate:
        return pcm
    resampled, _ = audioop.ratecv(
        pcm,
        2,
        1,
        src_rate,
        dst_rate,
        None,
    )
    return resampled


def _pcm_chunk_size(sample_rate: int) -> int:
    """Return a ~40 ms PCM chunk size for *sample_rate*."""
    return max(320, sample_rate * 2 * 40 // 1000)


def _chunk_bytes(data: bytes, size: int) -> Iterator[bytes]:
    """Yield *data* in *size*-byte pieces."""
    for offset in range(0, len(data), size):
        yield data[offset : offset + size]
