# -*- coding: utf-8 -*-
"""Unit tests for the local_voice (direct mic/speaker) channel."""
from __future__ import annotations

import asyncio
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from qwenpaw.app.channels import local_voice
from qwenpaw.app.channels.local_voice import (
    LocalVoiceChannel,
    _ROOT_SESSION_ID,
)
from qwenpaw.config.config import LocalVoiceChannelConfig
from qwenpaw.hooks.request_setup.local_voice_context import (
    LocalVoiceContextHook,
)
from qwenpaw.runtime.hooks import HookContext
from qwenpaw.schemas import ContentType, MessageType, RunStatus


class FakeSTT:
    def __init__(self) -> None:
        self.on_transcript = None
        self.on_speech_start = None
        self.on_wake_word = None
        self.fed = []
        self.started = False
        self.stopped = False
        self.interactions = 0

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def feed_audio(self, chunk: bytes) -> None:
        self.fed.append(chunk)

    def mark_interaction(self) -> None:
        self.interactions += 1


class FakeInputStream:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.closed = True


class FakeRawOutputStream(FakeInputStream):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.writes = []

    def write(self, chunk: bytes) -> None:
        self.writes.append(chunk)


@pytest.fixture
def no_fixed_prompts(monkeypatch):
    """Record fixed prompts instead of synthesizing real audio."""
    played = []

    async def _fake_synthesize_and_play(self, text):
        played.append(text)

    monkeypatch.setattr(
        LocalVoiceChannel,
        "_synthesize_and_play",
        _fake_synthesize_and_play,
    )
    return played


@pytest.fixture
def fake_sounddevice(monkeypatch):
    calls = {}

    def _fake_input_stream(**kwargs):
        calls["input"] = FakeInputStream(**kwargs)
        return calls["input"]

    def _fake_output_stream(**kwargs):
        calls["output"] = FakeRawOutputStream(**kwargs)
        return calls["output"]

    fake = SimpleNamespace(
        InputStream=_fake_input_stream,
        RawOutputStream=_fake_output_stream,
        stop=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    monkeypatch.setattr(local_voice, "sounddevice", fake, raising=False)
    return calls


def make_channel(config: LocalVoiceChannelConfig) -> LocalVoiceChannel:
    process = MagicMock()
    process.__call__ = MagicMock()
    return LocalVoiceChannel.from_config(process, config)


async def test_from_config_and_request_building():
    cfg = LocalVoiceChannelConfig(
        enabled=True,
        welcome_greeting="",
    )
    channel = make_channel(cfg)
    request = channel.build_agent_request_from_native(
        {"transcript": "你好"},
    )
    assert cfg.segment_max_turns == 50
    assert channel._segment_max_turns == 50
    assert channel.channel == "local_voice"
    assert request.channel == "local_voice"
    assert request.user_id == "local_user"
    assert request.input[0].content[0].text == "你好"


async def test_start_stop_lifecycle(fake_sounddevice, monkeypatch):
    stt = FakeSTT()
    monkeypatch.setattr(local_voice, "create_stt_engine", lambda *a, **k: stt)
    cfg = LocalVoiceChannelConfig(
        enabled=True,
        welcome_greeting="",
    )
    channel = make_channel(cfg)

    await channel.start()
    assert stt.started is True
    assert fake_sounddevice["input"].started is True
    assert channel._audio_task is not None

    await channel.stop()
    assert stt.stopped is True
    assert fake_sounddevice["input"].closed is True
    assert channel._audio_task is None


async def test_start_failure_is_reported_by_health(
    fake_sounddevice,
    monkeypatch,
):
    stt = FakeSTT()

    async def _fail_start() -> None:
        raise ValueError("KWS keywords file not found: 小爱同学")

    stt.start = _fail_start  # type: ignore[method-assign]
    monkeypatch.setattr(local_voice, "create_stt_engine", lambda *a, **k: stt)
    channel = make_channel(
        LocalVoiceChannelConfig(enabled=True, welcome_greeting=""),
    )

    with pytest.raises(ValueError, match="KWS keywords file not found"):
        await channel.start()

    health = await channel.health_check()
    assert health["status"] == "unhealthy"
    assert "KWS keywords file not found" in health["detail"]


async def test_audio_loop_ignores_mic_while_tts_playing():
    stt = FakeSTT()
    channel = make_channel(
        LocalVoiceChannelConfig(enabled=True, welcome_greeting=""),
    )
    channel._stt = stt
    channel._audio_task = asyncio.create_task(channel._audio_loop())

    channel._in_queue.put_nowait(b"\x01\x00")
    await asyncio.sleep(0.05)
    assert stt.fed == [b"\x01\x00"]

    channel._tts_playing.set()
    channel._in_queue.put_nowait(b"\x02\x00")
    await asyncio.sleep(0.05)
    assert stt.fed == [b"\x01\x00"]

    channel._tts_playing.clear()
    channel._wake_prompt_playing.set()
    channel._in_queue.put_nowait(b"\x03\x00")
    await asyncio.sleep(0.05)
    assert stt.fed == [b"\x01\x00"]

    channel._wake_prompt_playing.clear()
    channel._audio_task.cancel()
    try:
        await channel._audio_task
    except asyncio.CancelledError:
        pass


async def test_transcript_sends_reply_to_tts_queue(no_fixed_prompts):
    cfg = LocalVoiceChannelConfig(
        enabled=True,
        welcome_greeting="",
    )
    channel = make_channel(cfg)

    async def _fake_process(request):
        event = SimpleNamespace(
            object="message",
            type=MessageType.MESSAGE,
            status=RunStatus.Completed,
            content=[
                SimpleNamespace(
                    type=ContentType.TEXT,
                    text="你好，我是QwenPaw",
                ),
            ],
        )
        yield event

    channel._process = _fake_process  # type: ignore[method-assign]
    await channel._on_transcript("介绍一下你自己")
    assert no_fixed_prompts == ["收到，正在处理中"]
    assert channel._speak_queue.get_nowait() == "你好，我是QwenPaw"


async def test_transcript_skips_reasoning_and_tool_records(no_fixed_prompts):
    cfg = LocalVoiceChannelConfig(
        enabled=True,
        welcome_greeting="",
    )
    channel = make_channel(cfg)

    async def _fake_process(request):
        yield SimpleNamespace(
            object="message",
            type=MessageType.REASONING,
            status=RunStatus.Completed,
            content=[
                SimpleNamespace(
                    type=ContentType.TEXT,
                    text="我先思考一下",
                ),
            ],
        )
        yield SimpleNamespace(
            object="message",
            type=MessageType.PLUGIN_CALL,
            status=RunStatus.Completed,
            content=[
                SimpleNamespace(
                    type=ContentType.DATA,
                    data={"name": "web_search"},
                ),
            ],
        )
        yield SimpleNamespace(
            object="message",
            type=MessageType.PLUGIN_CALL_OUTPUT,
            status=RunStatus.Completed,
            content=[
                SimpleNamespace(
                    type=ContentType.DATA,
                    data={"output": "工具返回的内容"},
                ),
            ],
        )
        yield SimpleNamespace(
            object="message",
            type=MessageType.MESSAGE,
            status=RunStatus.Completed,
            content=[
                SimpleNamespace(
                    type=ContentType.TEXT,
                    text="今天天气不错",
                ),
            ],
        )

    channel._process = _fake_process  # type: ignore[method-assign]
    await channel._on_transcript("今天天气怎么样")
    assert no_fixed_prompts == ["收到，正在处理中"]
    assert channel._speak_queue.get_nowait() == "今天天气不错"
    assert channel._speak_queue.empty()


async def test_transcript_streams_complete_sentence_before_agent_finishes(
    no_fixed_prompts,
):
    channel = make_channel(
        LocalVoiceChannelConfig(enabled=True, welcome_greeting=""),
    )
    finish_agent = asyncio.Event()

    async def _fake_process(request):
        yield SimpleNamespace(
            object="message",
            id="msg-1",
            type=MessageType.MESSAGE,
            status=RunStatus.InProgress,
        )
        yield SimpleNamespace(
            object="content",
            msg_id="msg-1",
            delta=True,
            text="第一句。",
        )
        await finish_agent.wait()
        yield SimpleNamespace(
            object="content",
            msg_id="msg-1",
            delta=True,
            text="第二句",
        )
        yield SimpleNamespace(
            object="message",
            id="msg-1",
            type=MessageType.MESSAGE,
            status=RunStatus.Completed,
            content=[
                SimpleNamespace(
                    type=ContentType.TEXT,
                    text="第一句。第二句",
                ),
            ],
        )

    channel._process = _fake_process  # type: ignore[method-assign]
    task = asyncio.create_task(channel._on_transcript("测试流式回复"))
    first = await asyncio.wait_for(channel._speak_queue.get(), timeout=0.2)
    assert first == "第一句。"
    assert no_fixed_prompts == ["收到，正在处理中"]
    assert not task.done()

    finish_agent.set()
    await task
    assert channel._speak_queue.get_nowait() == "第二句"
    assert channel._speak_queue.empty()


async def test_transcript_ignores_reasoning_stream_deltas(no_fixed_prompts):
    channel = make_channel(
        LocalVoiceChannelConfig(enabled=True, welcome_greeting=""),
    )

    async def _fake_process(request):
        yield SimpleNamespace(
            object="message",
            id="reason-1",
            type=MessageType.REASONING,
            status=RunStatus.InProgress,
        )
        yield SimpleNamespace(
            object="content",
            msg_id="reason-1",
            delta=True,
            text="这段思考不能播报。",
        )
        yield SimpleNamespace(
            object="message",
            id="msg-1",
            type=MessageType.MESSAGE,
            status=RunStatus.Completed,
            content=[
                SimpleNamespace(type=ContentType.TEXT, text="最终回答"),
            ],
        )

    channel._process = _fake_process  # type: ignore[method-assign]
    await channel._on_transcript("测试")
    assert no_fixed_prompts == ["收到，正在处理中"]
    assert channel._speak_queue.get_nowait() == "最终回答"
    assert channel._speak_queue.empty()


async def test_send_enqueues_tts_and_speaker_plays(monkeypatch):
    cfg = LocalVoiceChannelConfig(
        enabled=True,
        welcome_greeting="",
    )
    channel = make_channel(cfg)
    stt = FakeSTT()
    channel._stt = stt
    played = []

    async def _fake_synthesize(*args, **kwargs):
        yield b"\x01\x00\x02\x00"

    async def _fake_play(self, text):
        played.append(text)

    monkeypatch.setattr(
        local_voice,
        "synthesize_tts_stream",
        _fake_synthesize,
    )
    monkeypatch.setattr(
        LocalVoiceChannel,
        "_synthesize_and_play",
        _fake_play,
    )
    channel._speaker_task = asyncio.create_task(channel._speaker_loop())

    await channel.send("local_user", "你好")
    await asyncio.sleep(0.05)
    assert played == ["你好"]
    assert stt.interactions == 1

    channel._speaker_task.cancel()
    try:
        await channel._speaker_task
    except asyncio.CancelledError:
        pass


async def test_wake_word_only_plays_fixed_ack(no_fixed_prompts):
    channel = make_channel(
        LocalVoiceChannelConfig(enabled=True, welcome_greeting=""),
    )
    channel._on_wake_word("小克小克")
    await asyncio.wait_for(channel._wake_prompt_task, timeout=2)

    assert no_fixed_prompts == ["在"]
    assert channel._wake_prompt_playing.is_set() is False


async def test_command_cancels_pending_wake_ack(no_fixed_prompts):
    channel = make_channel(
        LocalVoiceChannelConfig(enabled=True, welcome_greeting=""),
    )
    channel._on_wake_word("小克小克")
    task = channel._wake_prompt_task
    assert task is not None
    assert not task.done()

    channel._on_speech_start()
    try:
        await asyncio.wait_for(task, timeout=2)
    except asyncio.CancelledError:
        # The task can be cancelled before its first scheduling step.
        pass
    assert no_fixed_prompts == []
    assert channel._wake_prompt_playing.is_set() is False


async def test_speech_during_processing_queues_busy_prompt():
    channel = make_channel(
        LocalVoiceChannelConfig(enabled=True, welcome_greeting=""),
    )
    channel._processing = True

    channel._on_speech_start()
    await asyncio.sleep(0.05)
    assert channel._speak_queue.get_nowait() == "当前有任务正在处理中"
    assert channel._speak_queue.empty()


async def test_busy_prompt_is_skipped_while_tts_is_playing():
    channel = make_channel(
        LocalVoiceChannelConfig(enabled=True, welcome_greeting=""),
    )
    channel._processing = True
    channel._tts_playing.set()

    channel._on_speech_start()
    await asyncio.sleep(0.05)
    assert channel._speak_queue.empty()

    channel._tts_playing.clear()
    channel._on_speech_start()
    await asyncio.sleep(0.05)
    assert channel._speak_queue.get_nowait() == "当前有任务正在处理中"


async def test_busy_prompt_is_skipped_while_fixed_prompt_is_playing():
    channel = make_channel(
        LocalVoiceChannelConfig(enabled=True, welcome_greeting=""),
    )
    channel._processing = True
    channel._wake_prompt_playing.set()

    channel._on_speech_start()
    await asyncio.sleep(0.05)
    assert channel._speak_queue.empty()


async def test_busy_prompt_is_enqueued_whenever_tts_is_not_playing():
    channel = make_channel(
        LocalVoiceChannelConfig(enabled=True, welcome_greeting=""),
    )
    channel._processing = True

    channel._schedule_busy_prompt()
    channel._schedule_busy_prompt()
    assert channel._speak_queue.qsize() == 2
    assert channel._speak_queue.get_nowait() == "当前有任务正在处理中"
    assert channel._speak_queue.get_nowait() == "当前有任务正在处理中"
    assert channel._speak_queue.empty()


async def test_fixed_prompt_waits_for_speaker_queue(monkeypatch):
    channel = make_channel(
        LocalVoiceChannelConfig(enabled=True, welcome_greeting=""),
    )
    state = {
        "active": 0,
        "max_active": 0,
        "speaker_entered": asyncio.Event(),
        "release_speaker": asyncio.Event(),
        "fixed_entered": asyncio.Event(),
    }

    async def _fake_synthesize_and_play(self, text: str) -> None:
        state["active"] += 1
        state["max_active"] = max(state["max_active"], state["active"])
        try:
            if text == "回复":
                state["speaker_entered"].set()
                await state["release_speaker"].wait()
            elif text == "在":
                state["fixed_entered"].set()
        finally:
            state["active"] -= 1

    monkeypatch.setattr(
        LocalVoiceChannel,
        "_synthesize_and_play",
        _fake_synthesize_and_play,
    )
    channel._speaker_task = asyncio.create_task(channel._speaker_loop())

    await channel.send("local_user", "回复")
    await asyncio.wait_for(state["speaker_entered"].wait(), timeout=1)

    fixed_task = asyncio.create_task(
        channel._play_fixed_prompt("在", channel._wake_prompt_playing),
    )
    await asyncio.sleep(0.05)
    assert state["fixed_entered"].is_set() is False
    assert state["max_active"] == 1

    state["release_speaker"].set()
    await asyncio.wait_for(fixed_task, timeout=1)
    assert state["fixed_entered"].is_set() is True
    assert state["max_active"] == 1

    channel._speaker_task.cancel()
    try:
        await channel._speaker_task
    except asyncio.CancelledError:
        pass


async def test_transcript_during_processing_is_ignored_with_busy_prompt():
    channel = make_channel(
        LocalVoiceChannelConfig(enabled=True, welcome_greeting=""),
    )
    channel._processing = True

    await channel._on_transcript("插话内容")
    assert channel._speak_queue.get_nowait() == "当前有任务正在处理中"
    assert channel._speak_queue.empty()
    assert channel._processing is True


async def test_audio_loop_feeds_mic_while_agent_is_processing():
    stt = FakeSTT()
    channel = make_channel(
        LocalVoiceChannelConfig(enabled=True, welcome_greeting=""),
    )
    channel._stt = stt
    channel._processing = True
    channel._audio_task = asyncio.create_task(channel._audio_loop())

    channel._in_queue.put_nowait(b"\x01\x00")
    await asyncio.sleep(0.05)
    assert stt.fed == [b"\x01\x00"]

    channel._audio_task.cancel()
    try:
        await channel._audio_task
    except asyncio.CancelledError:
        pass


async def test_pcm_is_written_before_tts_generator_finishes(
    fake_sounddevice,
    monkeypatch,
):
    channel = make_channel(
        LocalVoiceChannelConfig(enabled=True, welcome_greeting=""),
    )
    release_second_chunk = asyncio.Event()

    async def _fake_synthesize(*args, **kwargs):
        yield b"\x01\x00"
        await release_second_chunk.wait()
        yield b"\x02\x00"

    monkeypatch.setattr(
        local_voice,
        "synthesize_tts_stream",
        _fake_synthesize,
    )
    task = asyncio.create_task(channel._synthesize_and_play("你好"))
    for _ in range(20):
        output = fake_sounddevice.get("output")
        if output is not None and output.writes:
            break
        await asyncio.sleep(0.01)

    assert fake_sounddevice["output"].writes == [b"\x01\x00"]
    assert not task.done()
    release_second_chunk.set()
    await task
    assert fake_sounddevice["output"].writes == [
        b"\x01\x00",
        b"\x02\x00",
    ]
    assert fake_sounddevice["output"].closed is True


def test_clean_for_tts():
    assert local_voice._clean_for_tts(" 你好\n世界 ") == "你好 世界"
    assert local_voice._clean_for_tts("3*5=15") == "3 5=15"
    assert local_voice._clean_for_tts("hello_world") == "hello world"


async def test_play_wake_prompt_uses_fixed_tts(monkeypatch):
    """A bare wake word must be acknowledged through the fixed TTS prompt."""
    channel = make_channel(
        LocalVoiceChannelConfig(enabled=True, welcome_greeting=""),
    )
    fixed_called = []

    async def _fake_play_fixed_prompt(self, text, playing_event):
        fixed_called.append(text)
        playing_event.clear()

    monkeypatch.setattr(
        LocalVoiceChannel,
        "_play_fixed_prompt",
        _fake_play_fixed_prompt,
    )

    await asyncio.wait_for(channel._play_wake_prompt(), timeout=1)
    assert fixed_called == ["在"]
    assert channel._wake_prompt_playing.is_set() is False


def test_split_tts_segments_uses_sentence_and_length_boundaries():
    segments, remaining = local_voice._split_tts_segments(
        "第一句。第二句还没结束",
    )
    assert segments == ["第一句。"]
    assert remaining == "第二句还没结束"

    segments, remaining = local_voice._split_tts_segments(
        "没有标点的尾句",
        flush=True,
    )
    assert segments == ["没有标点的尾句"]
    assert remaining == ""


def test_missing_sounddevice_error(fake_sounddevice, monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    cfg = LocalVoiceChannelConfig(enabled=True, welcome_greeting="")
    channel = make_channel(cfg)
    with pytest.raises(ValueError, match="sounddevice"):
        channel._start_input_stream()


# ---------------------------------------------------------------------------
# Voice session segmentation
# ---------------------------------------------------------------------------


def _fake_voice_workspace():
    """A minimal workspace exposing chat_manager for segment registration."""
    chat_manager = MagicMock()

    async def _get_or_create_chat(session_id, user_id, channel, name, meta):
        return SimpleNamespace(
            id=f"chat:{session_id}",
            session_id=session_id,
            user_id=user_id,
            channel=channel,
            name=name,
            meta=meta,
        )

    async def _touch_chat(chat_id):
        return None

    chat_manager.get_or_create_chat = _get_or_create_chat
    chat_manager.touch_chat = _touch_chat
    return SimpleNamespace(chat_manager=chat_manager, session=None)


def _reply_with(text: str):
    async def _fake_process(request):
        event = SimpleNamespace(
            object="message",
            type=MessageType.MESSAGE,
            status=RunStatus.Completed,
            content=[
                SimpleNamespace(type=ContentType.TEXT, text=text),
            ],
        )
        yield event

    return _fake_process


async def test_transcript_creates_segment_and_registers_chat(no_fixed_prompts):
    channel = make_channel(
        LocalVoiceChannelConfig(enabled=True, welcome_greeting=""),
    )
    channel._workspace = _fake_voice_workspace()
    channel._process = _reply_with("好的")  # type: ignore[method-assign]

    await channel._on_transcript("你好")
    assert channel._current_segment_id is not None
    assert channel._current_segment_id.startswith(
        f"{_ROOT_SESSION_ID}:seg_",
    )
    assert channel._segment_chat_id is not None
    assert channel._segment_messages[0] == {
        "role": "user",
        "text": "你好",
    }


async def test_followup_within_timeout_reuses_segment(no_fixed_prompts):
    channel = make_channel(
        LocalVoiceChannelConfig(enabled=True, welcome_greeting=""),
    )
    channel._workspace = _fake_voice_workspace()
    channel._process = _reply_with("好的")  # type: ignore[method-assign]

    await channel._on_transcript("第一问")
    first_segment_id = channel._current_segment_id
    assert first_segment_id

    await channel._on_transcript("追问")
    assert channel._current_segment_id == first_segment_id
    assert [m["role"] for m in channel._segment_messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_segment_messages_are_truncated_to_configured_turns():
    channel = make_channel(
        LocalVoiceChannelConfig(
            enabled=True,
            welcome_greeting="",
            segment_max_turns=2,
        ),
    )
    for index in range(5):
        channel._append_segment_message("user", f"u{index}")
        channel._append_segment_message("assistant", f"a{index}")

    assert channel._segment_max_turns == 2
    assert len(channel._segment_messages) == 4
    assert [m["role"] for m in channel._segment_messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert channel._segment_messages[0]["text"] == "u3"
    assert channel._segment_messages[-1]["text"] == "a4"


async def test_on_transcript_keeps_turn_count_after_truncation(no_fixed_prompts):
    channel = make_channel(
        LocalVoiceChannelConfig(
            enabled=True,
            welcome_greeting="",
            segment_max_turns=1,
        ),
    )
    channel._workspace = _fake_voice_workspace()
    channel._process = _reply_with("好的")  # type: ignore[method-assign]

    await channel._on_transcript("第一问")
    await channel._on_transcript("第二问")
    await channel._on_transcript("第三问")

    assert channel._segment_turn_count == 3
    assert len(channel._segment_messages) == 2
    assert channel._segment_messages[0] == {"role": "user", "text": "第三问"}
    assert channel._segment_messages[1] == {"role": "assistant", "text": "好的"}


async def test_idle_timeout_creates_new_segment(no_fixed_prompts):
    channel = make_channel(
        LocalVoiceChannelConfig(enabled=True, welcome_greeting=""),
    )
    channel._workspace = _fake_voice_workspace()
    channel._process = _reply_with("好的")  # type: ignore[method-assign]

    await channel._on_transcript("第一段")
    first_segment_id = channel._current_segment_id
    assert first_segment_id

    channel._segment_last_active_at = (
        time.monotonic() - local_voice._SEGMENT_IDLE_TIMEOUT_SECONDS - 1
    )
    await channel._on_transcript("新的一段")
    assert channel._current_segment_id is not None
    assert channel._current_segment_id != first_segment_id


async def test_build_request_includes_voice_prompt_meta(no_fixed_prompts):
    channel = make_channel(
        LocalVoiceChannelConfig(enabled=True, welcome_greeting=""),
    )
    channel._last_wake_word = "小克小克"
    request = channel.build_agent_request_from_native(
        {"transcript": "你好"},
    )

    assert request.channel_meta["voice_wake_word"] == "小克小克"
    assert "voice_summary" not in request.channel_meta
    assert "voice_recent_raw" not in request.channel_meta


async def test_local_voice_context_hook_injects_voice_style_without_history():
    hook = LocalVoiceContextHook()
    request = SimpleNamespace(
        channel="local_voice",
        channel_meta={
            "voice_wake_word": "小克小克",
            "voice_summary": "上一段聊了部署。",
            "voice_recent_raw": [
                {"role": "user", "text": "用 Docker 吗？"},
                {"role": "assistant", "text": "可以用 Docker。"},
            ],
        },
    )
    ctx = HookContext(
        request=request,
        session_id="seg-id",
        agent_id="default",
        root_session_id="local_voice:local_user",
        root_agent_id="default",
        workspace_dir=None,
        workspace=None,
        app_services=None,
    )

    result = await hook.run(ctx)

    assert result.action.value == "continue"
    assert len(ctx.context_injections) == 1
    content = ctx.context_injections[0]["content"]
    assert "小克小克" in content
    assert "口语化" in content
    assert "简短" in content
    assert "上一段聊了部署" not in content
    assert "用 Docker 吗" not in content
    assert "可以用 Docker" not in content
