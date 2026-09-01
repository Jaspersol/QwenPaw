# -*- coding: utf-8 -*-
"""Local voice channel.

Uses the computer's microphone and speakers directly:

    mic -> sounddevice InputStream (16 kHz mono int16)
        -> STT provider (sherpa_zipformer / aliyun / openai)
        -> Agent
        -> TTS provider (kokoro / edge_tts / aliyun / openai)
        -> sounddevice playback on the local output device

No SIP softphone or browser is involved, so TTS autoplay is unrestricted.

Voice feedback: a bare wake word is acknowledged with "在", a recognized
command with "收到，正在处理中", and ignored interruptions while a task is
running with "当前有任务正在处理中".
"""
from __future__ import annotations

import asyncio
import logging
import queue as thread_queue
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from qwenpaw.config.config import LocalVoiceChannelConfig
from qwenpaw.schemas import (
    AgentRequest,
    ContentType,
    Message,
    MessageType,
    Role,
    RunStatus,
    TextContent,
)

from ..base import BaseChannel, OnReplySent, ProcessHandler
from ..renderer import ChannelDisplayConfig
from ..sip._audioop_compat import audioop
from ..sip.stt_tts import (
    create_stt_engine,
    synthesize_tts_stream,
    warmup_tts,
)
from ..utils import extract_voice_message_text, is_final_voice_message

logger = logging.getLogger(__name__)

_LOCAL_USER_ID = "local_user"
_ROOT_SESSION_ID = "local_voice:local_user"
# Kept as an alias so the stable root session remains obvious at call sites.
_SESSION_ID = _ROOT_SESSION_ID
_SEGMENT_IDLE_TIMEOUT_SECONDS = 300.0
_SEGMENT_RECENT_RAW_TURNS = 2
_SEGMENT_SUMMARY_PREVIEW_CHARS = 500
_SEGMENT_MAX_TURNS = 50
_TARGET_SAMPLE_RATE = 16000
_FALLBACK_INPUT_SAMPLE_RATE = 48000
_TTS_SEGMENT_MAX_CHARS = 60
_TTS_SENTENCE_ENDINGS = frozenset("。！？!?；;\n")
_TTS_SOFT_BREAKS = frozenset("，,、：: ")

_WAKE_ACK_TEXT = "在"
_COMMAND_ACK_TEXT = "收到，正在处理中"
_BUSY_PROMPT_TEXT = "当前有任务正在处理中"
_WAKE_ACK_DELAY_SECONDS = 1.0
_TTS_DRAIN_TIMEOUT_SECONDS = 30.0

_EMOJI_RE = re.compile(
    "[\U00010000-\U0010ffff"
    "\U0000200d"
    "\U0000fe0f"
    "\U000023e9-\U000023fa"
    "\U00002702-\U000027b0"
    "\U0000fe00-\U0000fe0f"
    "]+",
    flags=re.UNICODE,
)


class LocalVoiceChannel(BaseChannel):
    """Always-on local voice assistant channel."""

    channel = "local_voice"
    uses_manager_queue = False

    def __init__(
        self,
        process: ProcessHandler,
        on_reply_sent: OnReplySent = None,
        display_config: ChannelDisplayConfig | None = None,
        no_text_debounce: bool = True,
    ) -> None:
        super().__init__(
            process,
            on_reply_sent,
            display_config=display_config,
            no_text_debounce=no_text_debounce,
        )
        self._config: Optional[LocalVoiceChannelConfig] = None
        self._stt: Any = None
        self._input_stream: Any = None
        self._input_rate = _TARGET_SAMPLE_RATE
        self._resample_state: Any = None

        self._in_queue: thread_queue.Queue[bytes | None] = thread_queue.Queue(
            maxsize=256,
        )
        self._speak_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._audio_task: Optional[asyncio.Task] = None
        self._speaker_task: Optional[asyncio.Task] = None
        self._tts_warmup_task: Optional[asyncio.Task] = None
        self._wake_prompt_task: Optional[asyncio.Task] = None

        # Voice-wake session segmentation.  A wake/speech burst maps to one
        # segment session; the root session only stores cross-segment summary.
        self._current_root_session_id = _ROOT_SESSION_ID
        self._current_segment_id: Optional[str] = None
        self._segment_chat_id: Optional[str] = None
        self._segment_started_at: Optional[float] = None
        self._segment_started_wall_at: Optional[str] = None
        self._segment_last_active_at: Optional[float] = None
        self._segment_messages: list[Dict[str, str]] = []
        self._segment_turn_count = 0
        self._segment_max_turns = _SEGMENT_MAX_TURNS
        self._segment_summary = ""
        self._segment_recent_raw: list[Dict[str, str]] = []
        self._segment_lock = asyncio.Lock()
        self._last_wake_at: Optional[float] = None
        self._last_wake_word = ""

        # While True the microphone is drained but not decoded, so agent
        # responses played through the speakers never trigger themselves.
        # The lock serializes the speak queue and fixed prompts (wake ack,
        # command ack, busy prompt) so two TTS streams cannot overlap.
        self._tts_lock = asyncio.Lock()
        self._tts_playing = asyncio.Event()
        self._wake_prompt_playing = asyncio.Event()
        self._processing = False
        self._last_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        process: ProcessHandler,
        config: LocalVoiceChannelConfig,
        on_reply_sent: OnReplySent = None,
        display_config: ChannelDisplayConfig | None = None,
        no_text_debounce: bool = True,
    ) -> "LocalVoiceChannel":
        instance = cls(
            process,
            on_reply_sent,
            display_config=display_config
            or ChannelDisplayConfig.from_config(config),
            no_text_debounce=no_text_debounce,
        )
        instance._config = config
        instance._segment_max_turns = max(1, config.segment_max_turns)
        return instance

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        cfg = self._config
        if not cfg or not cfg.enabled:
            logger.info("Local voice channel disabled, skip start")
            return

        self._last_error = None
        try:
            self._stt = create_stt_engine(
                cfg.stt_provider,
                cfg.language,
                cfg.dashscope_api_key,
                zipformer_model_dir=cfg.zipformer_model_dir,
                zipformer_num_threads=cfg.zipformer_num_threads,
                wake_word_enabled=cfg.wake_word_enabled,
                kws_model_dir=cfg.kws_model_dir,
                kws_keywords_file=cfg.kws_keywords_file,
                kws_num_threads=cfg.kws_num_threads,
                kws_pre_roll_seconds=cfg.kws_pre_roll_seconds,
                kws_score=cfg.kws_score,
                kws_threshold=cfg.kws_threshold,
                wake_active_timeout_seconds=cfg.wake_active_timeout_seconds,
                asr_rule1_min_trailing_silence=(
                    cfg.asr_rule1_min_trailing_silence
                ),
                asr_rule2_min_trailing_silence=(
                    cfg.asr_rule2_min_trailing_silence
                ),
                asr_rule3_min_utterance_length=(
                    cfg.asr_rule3_min_utterance_length
                ),
                openai_api_key=cfg.asr_api_key,
                openai_base_url=cfg.asr_api_base_url,
                openai_model=cfg.asr_model,
                keep_model_loaded=cfg.asr_keep_model_loaded,
            )
            self._stt.on_transcript = self._on_transcript
            self._stt.on_speech_start = self._on_speech_start
            self._stt.on_wake_word = self._on_wake_word

            # Loading local models can take a few seconds; errors surface
            # here instead of killing the audio loop later.
            await self._stt.start()

            self._start_input_stream()
            self._audio_task = asyncio.create_task(self._audio_loop())
            self._speaker_task = asyncio.create_task(self._speaker_loop())
            self._tts_warmup_task = asyncio.create_task(
                self._warmup_tts(),
            )

            if cfg.welcome_greeting:
                await self.send(
                    _LOCAL_USER_ID,
                    cfg.welcome_greeting,
                )
        except Exception as exc:
            self._last_error = str(exc) or exc.__class__.__name__
            logger.warning(
                "Local voice channel failed to start: %s",
                self._last_error,
            )
            try:
                await self.stop()
            except Exception:
                logger.debug(
                    "Error cleaning up after failed local voice start",
                    exc_info=True,
                )
            raise

        logger.info(
            "Local voice channel started (stt=%s tts=%s wake=%s)",
            cfg.stt_provider,
            cfg.tts_provider,
            cfg.wake_word_enabled,
        )

    async def stop(self) -> None:
        # Best-effort close the active segment so its summary is persisted
        # when the channel is deliberately stopped.
        try:
            async with self._segment_lock:
                await self._close_current_segment_locked()
        except Exception:
            logger.debug(
                "Local voice segment close during stop failed",
                exc_info=True,
            )

        if self._wake_prompt_task:
            self._wake_prompt_task.cancel()
            try:
                await self._wake_prompt_task
            except asyncio.CancelledError:
                pass
            self._wake_prompt_task = None
        self._wake_prompt_playing.clear()

        if self._tts_warmup_task:
            self._tts_warmup_task.cancel()
            try:
                await self._tts_warmup_task
            except asyncio.CancelledError:
                pass
            self._tts_warmup_task = None

        if self._audio_task:
            self._audio_task.cancel()
            try:
                await self._audio_task
            except asyncio.CancelledError:
                pass
            self._audio_task = None

        if self._input_stream is not None:
            try:
                self._input_stream.stop()
                self._input_stream.close()
            except Exception:
                logger.debug(
                    "Error closing input stream",
                    exc_info=True,
                )
            self._input_stream = None

        if self._stt is not None:
            try:
                await self._stt.stop()
            except Exception:
                logger.debug(
                    "Error stopping local STT",
                    exc_info=True,
                )
            self._stt = None

        if self._speaker_task:
            self._speaker_task.cancel()
            try:
                await self._speaker_task
            except asyncio.CancelledError:
                pass
            self._speaker_task = None
        try:
            import sounddevice as sd

            await asyncio.to_thread(sd.stop)
        except Exception:
            logger.debug(
                "Error stopping local audio playback",
                exc_info=True,
            )
        logger.info("Local voice channel stopped")

    async def health_check(self) -> Dict[str, Any]:
        running = self._audio_task is not None and not self._audio_task.done()
        if running:
            detail = "Microphone loop is running."
        elif self._last_error:
            detail = (
                "Microphone loop is not running. "
                f"Startup failed: {self._last_error}"
            )
        else:
            detail = "Microphone loop is not running."
        return {
            "channel": self.channel,
            "status": "healthy" if running else "unhealthy",
            "detail": detail,
        }

    # ------------------------------------------------------------------
    # Audio capture
    # ------------------------------------------------------------------

    def _start_input_stream(self) -> None:
        cfg = self._config
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise ValueError(
                "sounddevice is not installed. "
                "Install it with: pip install 'qwenpaw[sip]'",
            ) from exc

        def _callback(indata, frames, time_info, status) -> None:
            del frames, time_info
            if status:
                logger.debug("input stream status: %s", status)
            data = indata.tobytes()
            try:
                self._in_queue.put_nowait(data)
            except thread_queue.Full:
                try:
                    self._in_queue.get_nowait()
                except thread_queue.Empty:
                    pass
                try:
                    self._in_queue.put_nowait(data)
                except thread_queue.Full:
                    pass

        self._input_rate = int(cfg.input_sample_rate)
        try:
            self._input_stream = sd.InputStream(
                device=cfg.input_device,
                channels=1,
                samplerate=self._input_rate,
                dtype="int16",
                blocksize=int(
                    self._input_rate * cfg.block_duration_ms / 1000,
                ),
                callback=_callback,
            )
            self._input_stream.start()
        except Exception:
            logger.warning(
                "Failed to open input at %s Hz, falling back to %s Hz",
                self._input_rate,
                _FALLBACK_INPUT_SAMPLE_RATE,
                exc_info=True,
            )
            self._input_rate = _FALLBACK_INPUT_SAMPLE_RATE
            self._resample_state = None
            self._input_stream = sd.InputStream(
                device=cfg.input_device,
                channels=1,
                samplerate=self._input_rate,
                dtype="int16",
                blocksize=int(
                    self._input_rate * cfg.block_duration_ms / 1000,
                ),
                callback=_callback,
            )
            self._input_stream.start()

    async def _audio_loop(self) -> None:
        """Forward microphone PCM to STT unless TTS/agent is busy."""
        while True:
            try:
                data = await asyncio.to_thread(
                    self._in_queue.get,
                    True,
                    0.1,
                )
            except thread_queue.Empty:
                continue
            if data is None:
                break
            # Keep the microphone live while the Agent is thinking so an
            # attempted interruption can be acknowledged. It is still muted
            # while local audio is playing to avoid self-triggering.
            if (
                self._tts_playing.is_set()
                or self._wake_prompt_playing.is_set()
            ):
                continue

            if self._input_rate != _TARGET_SAMPLE_RATE:
                data, self._resample_state = audioop.ratecv(
                    data,
                    2,
                    1,
                    self._input_rate,
                    _TARGET_SAMPLE_RATE,
                    self._resample_state,
                )
            try:
                await self._stt.feed_audio(data)
            except Exception:
                logger.debug(
                    "local_voice feed_audio error",
                    exc_info=True,
                )

    # ------------------------------------------------------------------
    # STT callbacks
    # ------------------------------------------------------------------

    def _on_wake_word(self, keyword: str) -> None:
        logger.info("Local voice wake word detected: %s", keyword)
        self._last_wake_at = time.monotonic()
        self._last_wake_word = (keyword or "").strip()
        if self._processing:
            self._schedule_busy_prompt()
            return
        self._cancel_wake_prompt_task()
        self._wake_prompt_task = asyncio.create_task(
            self._play_wake_prompt_after_delay(),
        )

    def _cancel_wake_prompt_task(self) -> None:
        """Cancel a pending wake-only acknowledgment."""
        task = self._wake_prompt_task
        if task is not None and not task.done():
            task.cancel()
        self._wake_prompt_task = None

    async def _play_wake_prompt_after_delay(self) -> None:
        """Acknowledge a bare wake word, but let a follow-up command win."""
        try:
            await asyncio.sleep(_WAKE_ACK_DELAY_SECONDS)
            if self._processing:
                return
            await self._play_wake_prompt()
        except asyncio.CancelledError:
            pass

    async def _play_wake_prompt(self) -> None:
        """Play the fixed wake-only acknowledgment through TTS."""
        await self._play_fixed_prompt(
            _WAKE_ACK_TEXT,
            self._wake_prompt_playing,
        )

    async def _wait_tts_warmup(self) -> None:
        """Wait for the in-flight TTS warmup task and clear it once done."""
        task = self._tts_warmup_task
        if task is None:
            return
        try:
            await task
        finally:
            if self._tts_warmup_task is task:
                self._tts_warmup_task = None

    async def _play_fixed_prompt(
        self,
        text: str,
        playing_event: asyncio.Event,
    ) -> None:
        """Synthesize and play one short fixed prompt outside the TTS queue.

        The shared ``_tts_lock`` makes sure a fixed prompt never overlaps
        with the speak-queue playback: if the assistant is already speaking,
        the fixed prompt waits until the current utterance finishes.
        """
        if self._config is None:
            return
        async with self._tts_lock:
            playing_event.set()
            try:
                await self._synthesize_and_play(_clean_for_tts(text))
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "Local voice fixed prompt failed: %s",
                    text,
                    exc_info=True,
                )
            finally:
                playing_event.clear()

    def _schedule_busy_prompt(self) -> None:
        """Play the busy prompt immediately when no TTS is currently active.

        If the assistant is already speaking (the agent's reply or another
        fixed prompt), ignore the interruption entirely: do not enqueue the
        busy prompt and do not play it.
        """
        if self._tts_playing.is_set() or self._wake_prompt_playing.is_set():
            return
        try:
            self._speak_queue.put_nowait(_BUSY_PROMPT_TEXT)
        except asyncio.QueueFull:  # pragma: no cover - queue is unbounded
            logger.warning("Local voice speak queue is full")

    def _on_speech_start(self) -> None:
        logger.debug("Local voice speech started")
        if self._processing:
            self._schedule_busy_prompt()
            return
        self._cancel_wake_prompt_task()

    # ------------------------------------------------------------------
    # Voice segment lifecycle
    # ------------------------------------------------------------------

    async def _ensure_active_segment(self) -> str:
        """Return the active segment id, creating one when needed.

        A segment is created lazily on the first valid transcript after a
        wake/start; follow-up transcripts inside the timeout reuse it.
        """
        async with self._segment_lock:
            now = time.monotonic()
            if (
                self._current_segment_id
                and self._segment_last_active_at is not None
            ):
                if (
                    now - self._segment_last_active_at
                    <= _SEGMENT_IDLE_TIMEOUT_SECONDS
                ):
                    return self._current_segment_id
                await self._close_current_segment_locked()

            segment_id = self._new_segment_id()
            self._current_segment_id = segment_id
            self._segment_chat_id = None
            self._segment_started_at = now
            self._segment_started_wall_at = datetime.now().isoformat(
                timespec="seconds",
            )
            self._segment_last_active_at = now
            self._segment_messages = []
            self._segment_turn_count = 0
            self._segment_summary = ""
            self._segment_recent_raw = []

            await self._load_root_voice_context_locked()
            chat = await self._register_segment_chat_locked(segment_id)
            if chat is not None:
                self._segment_chat_id = chat.id

            logger.info(
                "Local voice segment started: %s",
                segment_id,
            )
            return segment_id

    async def _maybe_close_segment(self) -> None:
        """Close the current segment when it has been idle too long."""
        async with self._segment_lock:
            if not self._current_segment_id:
                return
            now = time.monotonic()
            if self._segment_last_active_at is not None and (
                now - self._segment_last_active_at
                > _SEGMENT_IDLE_TIMEOUT_SECONDS
            ):
                await self._close_current_segment_locked()

    def _new_segment_id(self) -> str:
        return (
            f"{self._current_root_session_id}:"
            f"seg_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
        )

    async def _register_segment_chat_locked(
        self,
        segment_id: str,
    ):
        if self._workspace is None:
            return None
        chat_manager = getattr(self._workspace, "chat_manager", None)
        if chat_manager is None:
            return None
        try:
            return await chat_manager.get_or_create_chat(
                session_id=segment_id,
                user_id=_LOCAL_USER_ID,
                channel=self.channel,
                name=f"语音 · {datetime.now():%m-%d %H:%M}",
                meta={
                    "voice": {
                        "root_session_id": self._current_root_session_id,
                        "segment_label": datetime.now().strftime(
                            "%m-%d %H:%M",
                        ),
                    },
                },
            )
        except Exception:
            logger.debug(
                "Local voice ChatSpec registration failed",
                exc_info=True,
            )
            return None

    async def _load_root_voice_context_locked(self) -> None:
        """Load cross-segment summary/recent turns from the root session."""
        if self._workspace is None:
            return
        session = getattr(self._workspace, "session", None)
        if session is None:
            return
        try:
            state = await session.get_session_state_dict(
                self._current_root_session_id,
                user_id=_LOCAL_USER_ID,
                channel=self.channel,
            )
            if not isinstance(state, dict):
                return
            voice_segments = state.get("voice_segments")
            if not isinstance(voice_segments, dict):
                voice_segments = {}
            self._segment_summary = str(
                voice_segments.get("latest_summary") or "",
            )
            recent_raw = voice_segments.get("recent_raw")
            if isinstance(recent_raw, list):
                self._segment_recent_raw = [
                    item
                    for item in recent_raw
                    if isinstance(item, dict)
                    and item.get("role") in {"user", "assistant"}
                ]
        except Exception:
            logger.debug(
                "Local voice root context load failed",
                exc_info=True,
            )

    async def _save_root_voice_context_locked(self) -> None:
        """Persist the cross-segment summary under the stable root session."""
        if self._workspace is None:
            return
        session = getattr(self._workspace, "session", None)
        if session is None:
            return
        try:
            state = await session.get_session_state_dict(
                self._current_root_session_id,
                user_id=_LOCAL_USER_ID,
                channel=self.channel,
            )
            if not isinstance(state, dict):
                state = {}
            voice_segments = state.get("voice_segments")
            if not isinstance(voice_segments, dict):
                voice_segments = {}
            voice_segments["latest_summary"] = self._build_segment_summary()
            voice_segments["recent_raw"] = self._segment_messages[
                -_SEGMENT_RECENT_RAW_TURNS * 2:
            ]
            if self._current_segment_id:
                voice_segments["last_segment_id"] = self._current_segment_id
            if self._segment_started_wall_at:
                voice_segments[
                    "last_segment_started_at"
                ] = self._segment_started_wall_at
            state["voice_segments"] = voice_segments
            await session.update_session_state(
                self._current_root_session_id,
                "voice_segments",
                voice_segments,
                user_id=_LOCAL_USER_ID,
                channel=self.channel,
            )
        except Exception:
            logger.debug(
                "Local voice root context save failed",
                exc_info=True,
            )

    async def _close_current_segment_locked(self) -> None:
        """Close the current segment and bridge its summary to the root."""
        if not self._current_segment_id:
            return
        segment_id = self._current_segment_id
        await self._save_root_voice_context_locked()
        chat_manager = (
            getattr(self._workspace, "chat_manager", None)
            if self._workspace is not None
            else None
        )
        if chat_manager is not None and self._segment_chat_id:
            try:
                await chat_manager.touch_chat(self._segment_chat_id)
            except Exception:
                logger.debug(
                    "Local voice segment chat touch failed",
                    exc_info=True,
                )
        logger.info("Local voice segment closed: %s", segment_id)
        self._current_segment_id = None
        self._segment_chat_id = None
        self._segment_started_at = None
        self._segment_started_wall_at = None
        self._segment_last_active_at = None
        self._segment_messages = []
        self._segment_turn_count = 0
        self._segment_summary = ""
        self._segment_recent_raw = []

    def _append_segment_message(self, role: str, text: str) -> None:
        """Append a message to the active segment and keep memory bounded.

        Only the most recent ``segment_max_turns`` turns are retained, where
        one turn normally maps to a user/assistant pair. Older messages are
        dropped from the in-memory list; the persisted summary and recent_raw
        already only need the latest few turns.
        """
        self._segment_messages.append({"role": role, "text": text})
        max_messages = max(1, self._segment_max_turns) * 2
        if len(self._segment_messages) > max_messages:
            del self._segment_messages[:len(self._segment_messages) - max_messages]

    def _build_segment_summary(self) -> str:
        """Build a lightweight deterministic summary for the closed segment.

        The current implementation is intentionally local and does not call
        an LLM; it can be replaced later by a call to the same summarizer used
        for Scroll continuation summaries.
        """
        if not self._segment_messages:
            return ""
        lines = []
        for item in self._segment_messages[-8:]:
            role = "用户" if item.get("role") == "user" else "助手"
            text = str(item.get("text") or "").strip()
            if text:
                lines.append(f"{role}: {text}")
        if not lines:
            return ""
        turns = self._segment_turn_count or max(
            1,
            len(self._segment_messages) // 2,
        )
        summary = (
            f"本轮语音会话共 {turns} 轮。\n最近对话：\n"
            + "\n".join(lines)
        )
        if len(summary) > _SEGMENT_SUMMARY_PREVIEW_CHARS:
            summary = summary[: _SEGMENT_SUMMARY_PREVIEW_CHARS] + "…"
        return summary

    async def _on_transcript(self, transcript: str) -> None:
        text = (transcript or "").strip()
        if not text:
            return
        self._cancel_wake_prompt_task()
        if self._processing:
            # A follow-up utterance is ignored while the current task is
            # running; acknowledge it without interrupting the Agent.
            logger.info(
                "Local voice transcript ignored while processing: %s",
                text[:80],
            )
            self._schedule_busy_prompt()
            return
        logger.info("Local voice transcript: %s", text)
        self._processing = True
        try:
            # Ensure this utterance belongs to a segment; after an idle
            # timeout the previous segment is closed and a new one starts.
            await self._ensure_active_segment()
            self._segment_last_active_at = time.monotonic()
            self._segment_turn_count += 1
            self._append_segment_message("user", text)

            # Fixed acknowledgment before the Agent starts processing.
            await self._play_fixed_prompt(
                _COMMAND_ACK_TEXT,
                self._wake_prompt_playing,
            )
            request = self.build_agent_request_from_native(
                {"transcript": text},
            )
            last_reply = ""
            last_response = None
            message_ids: set[str] = set()
            streamed_text = ""
            pending_tts = ""
            streamed_reply = False
            async for event in self._process(request):
                obj = getattr(event, "object", None)
                status = getattr(event, "status", None)
                event_type = getattr(event, "type", None)
                if (
                    obj == "message"
                    and status == RunStatus.InProgress
                    and event_type == MessageType.MESSAGE
                ):
                    message_id = str(getattr(event, "id", "") or "")
                    if message_id:
                        message_ids.add(message_id)
                elif (
                    obj == "content"
                    and getattr(event, "delta", False)
                    and str(getattr(event, "msg_id", "") or "") in message_ids
                ):
                    delta = str(getattr(event, "text", "") or "")
                    streamed_text += delta
                    pending_tts += delta
                    segments, pending_tts = _split_tts_segments(pending_tts)
                    for segment in segments:
                        await self.send(_LOCAL_USER_ID, segment)
                        streamed_reply = True
                elif is_final_voice_message(event):
                    reply = extract_voice_message_text(event)
                    if reply and streamed_text:
                        if reply.startswith(streamed_text):
                            suffix_start = len(streamed_text)
                            pending_tts += reply[suffix_start:]
                            streamed_text = reply
                        segments, pending_tts = _split_tts_segments(
                            pending_tts,
                            flush=True,
                        )
                        for segment in segments:
                            await self.send(_LOCAL_USER_ID, segment)
                            streamed_reply = True
                        last_reply = ""
                    elif reply:
                        last_reply = reply
                elif (
                    obj == "response"
                    and status == RunStatus.Completed
                ):
                    last_response = event
            if pending_tts:
                segments, pending_tts = _split_tts_segments(
                    pending_tts,
                    flush=True,
                )
                for segment in segments:
                    await self.send(_LOCAL_USER_ID, segment)
                    streamed_reply = True
            if (
                not streamed_reply
                and not last_reply
                and last_response is not None
            ):
                last_reply = self._response_to_text(last_response)
            if last_reply:
                logger.info(
                    "Local voice reply: %s",
                    last_reply[:80],
                )
                await self.send(_LOCAL_USER_ID, last_reply)
                self._append_segment_message("assistant", last_reply)
                self._segment_last_active_at = time.monotonic()
                if self._workspace is not None and self._segment_chat_id:
                    chat_manager = getattr(
                        self._workspace,
                        "chat_manager",
                        None,
                    )
                    if chat_manager is not None:
                        try:
                            await chat_manager.touch_chat(
                                self._segment_chat_id,
                            )
                        except Exception:
                            logger.debug(
                                "Local voice chat touch failed",
                                exc_info=True,
                            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "Local voice request failed",
            )
            await self.send(
                _LOCAL_USER_ID,
                "抱歉，处理刚才的请求时出错了。",
            )
        finally:
            # Wait for queued TTS to finish before accepting new audio, so
            # the assistant's own reply is never interpreted as a command.
            await self._drain_speak_queue()
            self._processing = False
            self._mark_stt_interaction()

    async def _drain_speak_queue(self) -> None:
        """Wait until the speaker queue is empty (or playback is idle)."""
        deadline = time.monotonic() + _TTS_DRAIN_TIMEOUT_SECONDS
        try:
            while not self._speak_queue.empty() or self._tts_playing.is_set():
                if time.monotonic() >= deadline:
                    logger.warning(
                        "Timed out draining local voice TTS queue (%d items "
                        "left); accepting new audio anyway",
                        self._speak_queue.qsize(),
                    )
                    return
                if (
                    self._speaker_task is None
                    or self._speaker_task.done()
                ):
                    logger.warning(
                        "Local voice speaker task stopped while draining "
                        "the TTS queue; clearing playback state",
                    )
                    self._tts_playing.clear()
                    return
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            raise

    # ------------------------------------------------------------------
    # Agent request / response
    # ------------------------------------------------------------------

    def _resolve_wake_word_label(self) -> str:
        """Return the wake word used for the current/local voice identity."""
        if self._last_wake_word:
            return self._last_wake_word
        if self._config is None:
            return "语音助手"
        value = (self._config.kws_keywords_file or "").strip()
        if not self._config.wake_word_enabled:
            return "语音助手"
        if not value:
            # Uses the model's default KWS keyword set; keep the identity
            # generic instead of guessing a possibly wrong keyword.
            return "语音助手"
        path = Path(value).expanduser()
        if path.is_file():
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and "@" in line:
                        return line.rsplit("@", 1)[-1].strip()
            except Exception:
                logger.debug(
                    "Failed to parse KWS keywords file for label",
                    exc_info=True,
                )
            return path.stem
        return value

    def build_agent_request_from_native(
        self,
        native_payload: Any,
    ) -> AgentRequest:
        text = native_payload.get("transcript", "")
        msg = Message(
            type=MessageType.MESSAGE,
            role=Role.USER,
            content=[TextContent(type=ContentType.TEXT, text=text)],
        )
        segment_id = self._current_segment_id or _ROOT_SESSION_ID
        root_session_id = self._current_root_session_id or _ROOT_SESSION_ID
        return AgentRequest(
            session_id=segment_id,
            user_id=_LOCAL_USER_ID,
            input=[msg],
            channel=self.channel,
            root_session_id=root_session_id,
            channel_meta={
                "root_session_id": root_session_id,
                "voice_wake_word": self._resolve_wake_word_label(),
            },
        )

    def resolve_session_id(
        self,
        sender_id: str,
        channel_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        del sender_id, channel_meta
        return self._current_segment_id or _ROOT_SESSION_ID

    async def send(
        self,
        to_handle: str,
        text: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        del to_handle, meta
        if text and text.strip():
            await self._speak_queue.put(text.strip())

    # ------------------------------------------------------------------
    # TTS playback
    # ------------------------------------------------------------------

    async def _speaker_loop(self) -> None:
        while True:
            text = await self._speak_queue.get()
            if text is None:
                self._tts_playing.clear()
                break
            text = _clean_for_tts(text)
            if not text:
                continue
            async with self._tts_lock:
                self._tts_playing.set()
                try:
                    await self._synthesize_and_play(text)
                except Exception:
                    logger.exception(
                        "Local voice TTS playback failed",
                    )
                finally:
                    # Keep echo suppression active between already queued
                    # sentences from the same streamed assistant response.
                    if self._speak_queue.empty():
                        self._tts_playing.clear()
                        self._mark_stt_interaction()

    async def _synthesize_and_play(self, text: str) -> None:
        cfg = self._config
        await self._wait_tts_warmup()
        stream = None
        try:
            async for chunk in synthesize_tts_stream(
                cfg.tts_provider,
                text,
                cfg.tts_voice,
                cfg.dashscope_api_key,
                sample_rate=cfg.tts_sample_rate,
                speed=getattr(cfg, "tts_speed", 1.0),
                kokoro_model_dir=cfg.kokoro_model_dir,
                kokoro_model_variant=cfg.kokoro_model_variant,
                kokoro_num_threads=cfg.kokoro_num_threads,
                kokoro_silence_scale=getattr(cfg, "kokoro_silence_scale", 0.2),
                openai_api_key=cfg.tts_api_key,
                openai_base_url=cfg.tts_api_base_url,
                openai_model=cfg.tts_model,
                qwen3_backend=cfg.qwen3_backend,
                qwen3_model=cfg.qwen3_model,
                qwen3_model_dir=cfg.qwen3_model_dir,
                qwen3_ref_audio=cfg.qwen3_ref_audio,
                qwen3_ref_text=cfg.qwen3_ref_text,
                qwen3_device=cfg.qwen3_device,
                keep_model_loaded=cfg.tts_keep_model_loaded,
            ):
                if not chunk:
                    continue
                if stream is None:
                    stream = await asyncio.to_thread(
                        self._open_output_stream,
                        cfg.tts_sample_rate,
                        cfg.output_device,
                    )
                await asyncio.to_thread(stream.write, chunk)
        finally:
            if stream is not None:
                await asyncio.to_thread(self._close_output_stream, stream)

    @staticmethod
    def _open_output_stream(
        sample_rate: int,
        output_device: Optional[int],
    ) -> Any:
        import sounddevice as sd

        stream = sd.RawOutputStream(
            samplerate=sample_rate,
            device=output_device,
            channels=1,
            dtype="int16",
        )
        stream.start()
        return stream

    @staticmethod
    def _close_output_stream(stream: Any) -> None:
        try:
            stream.stop()
        finally:
            stream.close()

    async def _warmup_tts(self) -> None:
        cfg = self._config
        try:
            await warmup_tts(
                cfg.tts_provider,
                kokoro_model_dir=cfg.kokoro_model_dir,
                kokoro_model_variant=cfg.kokoro_model_variant,
                kokoro_num_threads=cfg.kokoro_num_threads,
                qwen3_backend=cfg.qwen3_backend,
                qwen3_model_dir=cfg.qwen3_model_dir,
                qwen3_ref_audio=cfg.qwen3_ref_audio,
                qwen3_ref_text=cfg.qwen3_ref_text,
                qwen3_device=cfg.qwen3_device,
                keep_model_loaded=cfg.tts_keep_model_loaded,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Local voice TTS warmup failed", exc_info=True)

    def _mark_stt_interaction(self) -> None:
        """Extend the active wake session after a completed interaction."""
        callback = getattr(self._stt, "mark_interaction", None)
        if callable(callback):
            callback()

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    @property
    def config(self) -> Optional[LocalVoiceChannelConfig]:
        return self._config


def _clean_for_tts(text: str) -> str:
    """Remove characters that TTS providers commonly reject or read badly.

    Strips emoji, Markdown emphasis/lists/headings, code fences, links,
    and collapses whitespace so the engine only sees speakable prose.
    """
    text = _EMOJI_RE.sub("", text)
    # Collapse Markdown links to their label text: [label](url) -> label
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    # Bare URLs are read char-by-char; drop them entirely.
    text = re.sub(r"(?:https?://|www\.)\S+", "", text)
    # Code fences and inline code delimiters.
    text = re.sub(r"```[^`]*```", "", text, flags=re.DOTALL)
    text = text.replace("`", "")
    # Markdown emphasis / bold / strikethrough markers.
    text = text.replace("**", "").replace("__", "").replace("~~", "")
    # Headings and blockquote markers at line starts.
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)
    # List bullets (markdown and ordered) at line starts.
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.MULTILINE)
    # Horizontal rules.
    text = re.sub(
        r"^[ \t]*([-*_])([ \t]*\1){2,}[ \t]*$",
        "",
        text,
        flags=re.MULTILINE,
    )
    # Stray markdown/formatting symbols that survived. Replace with spaces
    # instead of deleting so content like "3*5" or "hello_world" keeps its
    # word/number boundaries; the whitespace collapse below normalizes them.
    text = text.replace("*", " ").replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_tts_segments(
    text: str,
    *,
    flush: bool = False,
    max_chars: int = _TTS_SEGMENT_MAX_CHARS,
) -> tuple[list[str], str]:
    """Return speakable sentences and the incomplete trailing fragment."""
    segments: list[str] = []
    remaining = text
    while remaining:
        sentence_end = next(
            (
                index + 1
                for index, char in enumerate(remaining)
                if char in _TTS_SENTENCE_ENDINGS
            ),
            None,
        )
        if sentence_end is not None:
            segment = remaining[:sentence_end].strip()
            remaining = remaining[sentence_end:].lstrip()
            if segment:
                segments.append(segment)
            continue
        if len(remaining) <= max_chars:
            break
        split_at = max_chars
        for index in range(max_chars - 1, max_chars // 2, -1):
            if remaining[index] in _TTS_SOFT_BREAKS:
                split_at = index + 1
                break
        segment = remaining[:split_at].strip()
        remaining = remaining[split_at:].lstrip()
        if segment:
            segments.append(segment)
    if flush and remaining.strip():
        segments.append(remaining.strip())
        remaining = ""
    return segments, remaining
