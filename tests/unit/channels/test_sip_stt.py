# -*- coding: utf-8 -*-
"""Unit tests for local Sherpa Zipformer STT + KWS wake-word gate."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwenpaw.app.channels.sip import stt_engine
from qwenpaw.app.channels.sip.stt_engine import (
    OpenAIWhisperSTT,
    SherpaZipformerSTT,
)
from qwenpaw.app.channels.sip.stt_tts import create_stt_engine


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeAsrStream:
    def __init__(self, recognizer) -> None:
        self.recognizer = recognizer
        self.waveforms = []
        self.decoded = 0
        self.finished = False

    def accept_waveform(self, sample_rate, waveform) -> None:
        self.waveforms.append(waveform)
        self.recognizer.feed_count += 1
        self.decoded = 0

    def input_finished(self) -> None:
        self.finished = True


class FakeRecognizer:
    def __init__(
        self,
        *,
        partial_text: str,
        final_text: str,
        endpoint_after_feeds: int,
    ) -> None:
        self.partial_text = partial_text
        self.final_text = final_text
        self.endpoint_after_feeds = endpoint_after_feeds
        self.feed_count = 0
        self.streams = []

    def create_stream(self):
        stream = FakeAsrStream(self)
        self.streams.append(stream)
        return stream

    def is_ready(self, stream) -> bool:
        return stream.decoded == 0

    def decode_stream(self, stream) -> None:
        stream.decoded += 1

    def get_result(self, stream) -> str:
        if self.is_endpoint(stream):
            return self.final_text
        return self.partial_text

    def is_endpoint(self, stream) -> bool:
        return self.feed_count >= self.endpoint_after_feeds

    def reset(self, stream) -> bool:
        stream.waveforms.clear()
        return True


class FakeKwsStream:
    def __init__(self, spotter) -> None:
        self.spotter = spotter
        self.feed_count = 0
        self.decoded = 0

    def accept_waveform(self, sample_rate, waveform) -> None:
        self.feed_count += 1
        self.decoded = 0


class FakeKws:
    def __init__(
        self,
        keyword: str = "小爱同学",
        detect_on_feed: int = 1,
    ) -> None:
        self.keyword = keyword
        self.detect_on_feed = detect_on_feed
        self.streams = []

    def create_stream(self, keywords=None):
        stream = FakeKwsStream(self)
        stream.keywords = keywords
        self.streams.append(stream)
        return stream

    def is_ready(self, stream) -> bool:
        return stream.decoded == 0

    def decode_stream(self, stream) -> None:
        stream.decoded += 1

    def get_result(self, stream) -> str:
        if stream.feed_count >= self.detect_on_feed:
            return self.keyword
        return ""

    def reset_stream(self, stream) -> None:
        stream.decoded = 0


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------


def test_normalize_stt_provider():
    assert stt_engine.normalize_stt_provider("aliyun") == "aliyun"
    assert (
        stt_engine.normalize_stt_provider("sherpa_zipformer")
        == "sherpa_zipformer"
    )
    assert stt_engine.normalize_stt_provider("zipformer") == "sherpa_zipformer"
    assert stt_engine.normalize_stt_provider("local") == "sherpa_zipformer"
    assert stt_engine.normalize_stt_provider("openai") == "openai"
    assert stt_engine.normalize_stt_provider("whisper") == "openai"
    assert stt_engine.normalize_stt_provider("whisper_api") == "openai"
    assert stt_engine.normalize_stt_provider("openai_compatible") == "openai"


def test_normalize_stt_provider_rejects_unknown():
    with pytest.raises(ValueError, match="Unsupported STT provider"):
        stt_engine.normalize_stt_provider("unknown")


def test_create_stt_engine_returns_local_engine():
    engine = create_stt_engine("zipformer", "zh-CN")
    assert isinstance(engine, SherpaZipformerSTT)


def test_create_stt_engine_returns_openai_engine():
    engine = create_stt_engine(
        "whisper",
        "zh-CN",
        openai_api_key="sk-test",
        openai_base_url="https://example.test/v1",
        openai_model="whisper-test",
        asr_rule1_min_trailing_silence=0.5,
        asr_rule2_min_trailing_silence=0.2,
        asr_rule3_min_utterance_length=12.0,
    )
    assert isinstance(engine, OpenAIWhisperSTT)
    assert engine._base_url == "https://example.test/v1"
    assert engine._model == "whisper-test"
    assert engine._min_trailing_silence == 0.5
    assert engine._min_speech_seconds == 0.2
    assert engine._max_utterance_seconds == 12.0


def test_create_stt_engine_applies_low_latency_parameters():
    engine = create_stt_engine(
        "zipformer",
        "zh-CN",
        kws_pre_roll_seconds=0.25,
        asr_rule1_min_trailing_silence=0.7,
        asr_rule2_min_trailing_silence=0.3,
        asr_rule3_min_utterance_length=12.0,
        keep_model_loaded=False,
    )
    assert engine._pre_roll_limit == 4000
    assert engine._rule1_min_trailing_silence == 0.7
    assert engine._rule2_min_trailing_silence == 0.3
    assert engine._rule3_min_utterance_length == 12.0
    assert engine._keep_model_loaded is False


# ---------------------------------------------------------------------------
# Wake-word gate
# ---------------------------------------------------------------------------


async def test_wake_word_gate_feeds_asr_only_after_keyword():
    recognizer = FakeRecognizer(
        partial_text="今天",
        final_text="小爱同学，今天天气怎么样",
        endpoint_after_feeds=2,
    )
    kws = FakeKws()
    engine = SherpaZipformerSTT(
        wake_word_enabled=True,
        pre_roll_seconds=0,
    )
    engine._recognizer = recognizer
    engine._kws = kws
    engine._kws_stream = kws.create_stream()
    engine._woken = False

    transcripts = []
    wake_words = []
    speech_starts = 0

    async def _on_transcript(text: str) -> None:
        transcripts.append(text)

    engine.on_transcript = _on_transcript
    engine.on_wake_word = lambda keyword: wake_words.append(keyword)
    engine.on_speech_start = lambda: None

    def _speech_start() -> None:
        nonlocal speech_starts
        speech_starts += 1

    engine.on_speech_start = _speech_start

    # First chunk only runs KWS and must not reach ASR decoding yet.
    await engine.feed_audio(b"\x01\x00")
    assert wake_words == ["小爱同学"]
    assert engine._woken is True
    assert recognizer.feed_count == 1

    # Second chunk reaches ASR and triggers the endpoint.
    await engine.feed_audio(b"\x02\x00")
    await asyncio.sleep(0)
    assert transcripts == ["今天天气怎么样"]
    assert speech_starts >= 1
    assert engine._woken is True
    assert engine._last_wake_word == "小爱同学"
    assert len(kws.streams) == 1


async def test_wake_session_rearms_after_one_minute(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(stt_engine.time, "monotonic", lambda: clock[0])
    recognizer = FakeRecognizer(
        partial_text="",
        final_text="",
        endpoint_after_feeds=99,
    )
    kws = FakeKws()
    engine = SherpaZipformerSTT(
        wake_word_enabled=True,
        pre_roll_seconds=0,
        wake_active_timeout_seconds=60,
    )
    engine._recognizer = recognizer
    engine._kws = kws
    engine._kws_stream = kws.create_stream()
    engine._woken = False

    await engine.feed_audio(b"\x01\x00")
    assert engine._woken is True
    assert engine._wake_expires_at == 160.0

    # Once the active window expires, this audio is routed back through KWS.
    kws.detect_on_feed = 99
    clock[0] = 160.1
    await engine.feed_audio(b"\x02\x00")
    assert engine._woken is False
    assert engine._last_wake_word == ""
    assert len(kws.streams) == 2


def test_strip_detected_wake_word():
    strip = stt_engine.strip_detected_wake_word
    assert strip("小克小克，打开浏览器", "小克小克") == "打开浏览器"
    assert strip("小 克 小 克  打开浏览器", "小克小克") == "打开浏览器"
    assert strip("打开浏览器", "小克小克") == "打开浏览器"
    assert strip("小克小克", "小克小克") == ""


async def test_wake_word_replays_pre_roll_without_duplicate():
    recognizer = FakeRecognizer(
        partial_text="今天",
        final_text="今天天气怎么样",
        endpoint_after_feeds=99,
    )
    kws = FakeKws(detect_on_feed=2)
    engine = SherpaZipformerSTT(
        wake_word_enabled=True,
        pre_roll_seconds=1.0,
    )
    engine._recognizer = recognizer
    engine._kws = kws
    engine._kws_stream = kws.create_stream()
    engine._woken = False

    # No keyword yet: audio is buffered, ASR stays idle.
    await engine.feed_audio(b"\x01\x00")
    assert engine._woken is False
    assert recognizer.feed_count == 0

    # Keyword hits on this chunk; pre-roll is replayed once and the
    # current chunk is fed once (two waveforms total, no duplicate).
    await engine.feed_audio(b"\x02\x00")
    assert engine._woken is True
    assert recognizer.feed_count == 2


async def test_start_loads_local_models_without_wake(monkeypatch, tmp_path):
    recognizer = FakeRecognizer(
        partial_text="",
        final_text="",
        endpoint_after_feeds=99,
    )
    engine = SherpaZipformerSTT(wake_word_enabled=False)
    monkeypatch.setattr(
        stt_engine,
        "resolve_zipformer_model_dir",
        lambda model_dir="": tmp_path,
    )
    monkeypatch.setattr(
        stt_engine,
        "zipformer_model_files",
        lambda model_dir: (
            tmp_path / "tokens.txt",
            tmp_path / "encoder.onnx",
            tmp_path / "decoder.onnx",
            tmp_path / "joiner.onnx",
        ),
    )
    engine._load_models = lambda zipformer_dir, kws_dir, keywords_file: {
        "recognizer": recognizer,
    }

    await engine.start()
    assert engine._recognizer is recognizer
    assert engine._woken is True
    assert engine._asr_stream is not None


async def test_start_arms_kws_when_wake_enabled(monkeypatch, tmp_path):
    recognizer = FakeRecognizer(
        partial_text="",
        final_text="",
        endpoint_after_feeds=99,
    )
    kws = FakeKws()
    engine = SherpaZipformerSTT(wake_word_enabled=True)
    monkeypatch.setattr(
        stt_engine,
        "resolve_zipformer_model_dir",
        lambda model_dir="": tmp_path,
    )
    monkeypatch.setattr(
        stt_engine,
        "zipformer_model_files",
        lambda model_dir: (
            tmp_path / "tokens.txt",
            tmp_path / "encoder.onnx",
            tmp_path / "decoder.onnx",
            tmp_path / "joiner.onnx",
        ),
    )
    monkeypatch.setattr(
        stt_engine,
        "resolve_kws_model_dir",
        lambda model_dir="": tmp_path,
    )
    monkeypatch.setattr(
        stt_engine,
        "resolve_kws_keywords_file",
        lambda model_dir, keywords_file="": str(
            tmp_path / "keywords.txt",
        ),
    )
    (tmp_path / "keywords.txt").write_text(
        "x i 菐o m 菒 x i 菐o m 菒 @小米小米\n",
    )
    engine._load_models = lambda zipformer_dir, kws_dir, keywords_file: {
        "recognizer": recognizer,
        "kws": kws,
    }

    await engine.start()
    assert engine._woken is False
    assert engine._kws is kws
    assert engine._kws_stream is not None


async def test_start_accepts_literal_keyword_value(monkeypatch, tmp_path):
    recognizer = FakeRecognizer(
        partial_text="",
        final_text="",
        endpoint_after_feeds=99,
    )
    kws = FakeKws()
    engine = SherpaZipformerSTT(
        wake_word_enabled=True,
        keywords_file="小爱同学",
    )
    monkeypatch.setattr(
        stt_engine,
        "resolve_zipformer_model_dir",
        lambda model_dir="": tmp_path,
    )
    monkeypatch.setattr(
        stt_engine,
        "zipformer_model_files",
        lambda model_dir: (
            tmp_path / "tokens.txt",
            tmp_path / "encoder.onnx",
            tmp_path / "decoder.onnx",
            tmp_path / "joiner.onnx",
        ),
    )
    monkeypatch.setattr(
        stt_engine,
        "resolve_kws_model_dir",
        lambda model_dir="": tmp_path,
    )
    monkeypatch.setattr(
        stt_engine,
        "resolve_kws_keywords_file",
        lambda model_dir, keywords_file="": str(
            tmp_path / "keywords.txt",
        ),
    )
    engine._load_models = lambda zipformer_dir, kws_dir, keywords_file: {
        "recognizer": recognizer,
        "kws": kws,
    }

    await engine.start()

    assert engine._inline_keywords == "x iǎo ài t óng x ué @小爱同学"
    assert kws.streams[0].keywords == "x iǎo ài t óng x ué @小爱同学"


async def test_start_generates_keywords_when_model_has_none(
    monkeypatch,
    tmp_path,
):
    recognizer = FakeRecognizer(
        partial_text="",
        final_text="",
        endpoint_after_feeds=99,
    )
    kws = FakeKws()
    captured = {}

    def _load_models(zipformer_dir, kws_dir, keywords_file):
        captured["keywords_file"] = keywords_file
        return {"recognizer": recognizer, "kws": kws}

    engine = SherpaZipformerSTT(wake_word_enabled=True)
    engine._load_models = _load_models  # type: ignore[method-assign]
    monkeypatch.setattr(
        stt_engine,
        "resolve_zipformer_model_dir",
        lambda model_dir="": tmp_path,
    )
    monkeypatch.setattr(
        stt_engine,
        "zipformer_model_files",
        lambda model_dir: (
            tmp_path / "tokens.txt",
            tmp_path / "encoder.onnx",
            tmp_path / "decoder.onnx",
            tmp_path / "joiner.onnx",
        ),
    )
    monkeypatch.setattr(
        stt_engine,
        "resolve_kws_model_dir",
        lambda model_dir="": tmp_path,
    )

    await engine.start()

    generated = Path(captured["keywords_file"])
    assert generated.is_file()
    assert (
        "x iǎo ài t óng x ué @小爱同学"
        in generated.read_text(encoding="utf-8")
    )

    await engine.stop()
    assert not generated.exists()


def test_tokenize_kws_keyword():
    assert (
        stt_engine._tokenize_kws_keyword("小爱同学")
        == "x iǎo ài t óng x ué @小爱同学"
    )
    assert (
        stt_engine._tokenize_kws_keyword("小艺小艺")
        == "x iǎo y ì x iǎo y ì @小艺小艺"
    )


async def test_without_wake_word_audio_goes_directly_to_asr():
    recognizer = FakeRecognizer(
        partial_text="你好",
        final_text="你好",
        endpoint_after_feeds=99,
    )
    engine = SherpaZipformerSTT(wake_word_enabled=False)
    engine._recognizer = recognizer
    engine._asr_stream = recognizer.create_stream()
    engine._woken = True

    await engine.feed_audio(b"\x01\x00")
    assert recognizer.feed_count == 1
    assert engine._woken is True


# ---------------------------------------------------------------------------
# OpenAI-compatible ASR adapter
# ---------------------------------------------------------------------------


def test_resolve_openai_base_url_appends_v1():
    assert stt_engine.resolve_openai_base_url("") == (
        "https://api.openai.com/v1"
    )
    assert stt_engine.resolve_openai_base_url(
        "https://example.test/v1",
    ) == "https://example.test/v1"
    assert stt_engine.resolve_openai_base_url(
        "https://example.test/",
    ) == "https://example.test/v1"


def test_openai_language_hint_maps_channel_language():
    assert stt_engine._openai_language_hint("zh-CN") == "zh"
    assert stt_engine._openai_language_hint("en-US") == "en"
    assert stt_engine._openai_language_hint("") == ""
    assert stt_engine._openai_language_hint("fr-FR") == "fr"
    # Whisper language hints are ISO-639-1 two-letter codes.
    assert stt_engine._openai_language_hint("fil-PH") == ""


def test_pcm16_to_wav_has_valid_header():
    wav = stt_engine._pcm16_to_wav(b"\x01\x00\x02\x00", 16000)
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert len(wav) == 48


async def test_openai_stt_vad_queues_utterance_after_trailing_silence():
    engine = OpenAIWhisperSTT(
        min_trailing_silence=0.2,
        min_speech_seconds=0.0,
        speech_threshold=100,
    )
    engine._started = True
    engine._worker_task = asyncio.get_running_loop().create_future()
    starts = []
    engine.on_speech_start = lambda: starts.append(True)

    loud = b"\x00\x08" * 1600  # 0.1 s of clearly audible PCM
    silence = b"\x00\x00" * 3200  # 0.2 s trailing silence
    await engine.feed_audio(loud)
    await engine.feed_audio(silence)

    assert starts == [True]
    assert engine._speaking is False
    assert engine._pending.qsize() == 1
    assert engine._pending.get_nowait() == loud + silence


async def test_openai_stt_ignores_noise_and_short_utterances():
    engine = OpenAIWhisperSTT(
        min_trailing_silence=0.2,
        min_speech_seconds=0.2,
        speech_threshold=100,
    )
    engine._started = True
    engine._worker_task = asyncio.get_running_loop().create_future()

    await engine.feed_audio(b"\x00\x00" * 1600)
    assert engine._speaking is False
    assert engine._pending.qsize() == 0

    await engine.feed_audio(b"\x00\x08" * 1600)
    await engine.feed_audio(b"\x00\x00" * 3200)
    assert engine._pending.qsize() == 0


async def test_openai_stt_transcribes_with_multipart_wav():
    engine = OpenAIWhisperSTT(
        api_key="sk-test",
        base_url="https://example.test/v1",
        model="whisper-test",
        language="zh-CN",
    )

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self):
            return {"text": " 你好 "}

    class FakeClient:
        def __init__(self) -> None:
            self.calls = []

        async def post(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return FakeResponse()

    engine._client = FakeClient()
    text = await engine._transcribe(b"\x01\x00" * 160)

    assert text == "你好"
    args, kwargs = engine._client.calls[0]
    assert args == ("https://example.test/v1/audio/transcriptions",)
    assert kwargs["headers"]["Authorization"] == "Bearer sk-test"
    assert kwargs["data"]["model"] == "whisper-test"
    assert kwargs["data"]["language"] == "zh"
    assert kwargs["files"]["file"][2] == "audio/wav"


# ---------------------------------------------------------------------------
# Model file helpers
# ---------------------------------------------------------------------------


def test_pick_onnx_prefers_int8_for_encoder(tmp_path: Path):
    (tmp_path / "encoder-epoch-12-avg-2-chunk-16-left-64.onnx").write_bytes(
        b"fp32",
    )
    int8_path = tmp_path / "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
    int8_path.write_bytes(b"int8")
    assert stt_engine._pick_onnx(
        tmp_path,
        "encoder",
        prefer_int8=True,
    ).name.endswith(".int8.onnx")


def test_pick_onnx_prefers_fp32_for_decoder(tmp_path: Path):
    (tmp_path / "decoder.onnx").write_bytes(b"fp32")
    (tmp_path / "decoder.int8.onnx").write_bytes(b"int8")
    assert stt_engine._pick_onnx(
        tmp_path,
        "decoder",
        prefer_int8=False,
    ).name == "decoder.onnx"


def test_detect_transducer_model_type_reads_onnx_metadata(
    monkeypatch,
    tmp_path: Path,
):
    class FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        def get_modelmeta(self):
            return SimpleNamespace(
                custom_metadata_map={"model_type": "zipformer2"},
            )

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(InferenceSession=FakeSession),
    )
    assert stt_engine.detect_transducer_model_type(
        tmp_path / "encoder.onnx",
    ) == "zipformer2"


def test_zipformer_model_files_requires_tokens(tmp_path: Path):
    with pytest.raises(ValueError, match="Missing tokens.txt"):
        stt_engine.zipformer_model_files(tmp_path)


def test_resolve_model_dirs_use_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ZIPFORMER_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("KWS_MODEL_DIR", str(tmp_path))
    assert stt_engine.resolve_zipformer_model_dir("") == tmp_path
    assert stt_engine.resolve_kws_model_dir("") == tmp_path


def test_looks_like_keywords_file_path():
    assert stt_engine._looks_like_keywords_file_path("my_keywords.txt")
    assert stt_engine._looks_like_keywords_file_path("~/kw.txt")
    assert stt_engine._looks_like_keywords_file_path(
        str(Path("models") / "kw.txt"),
    )
    assert not stt_engine._looks_like_keywords_file_path("小爱同学")


def test_validate_kws_keywords_value(tmp_path: Path):
    assert stt_engine.validate_kws_keywords_value("") == ""
    keywords = tmp_path / "my_keywords.txt"
    keywords.write_text("x iǎo ài @小爱同学\n", encoding="utf-8")
    assert (
        stt_engine.validate_kws_keywords_value(str(keywords))
        == str(keywords)
    )
    with pytest.raises(ValueError, match="not found"):
        stt_engine.validate_kws_keywords_value(str(tmp_path / "missing.txt"))
    assert stt_engine.validate_kws_keywords_value("小爱同学") == "小爱同学"


def test_resolve_kws_keywords_file_custom(tmp_path: Path):
    keywords = tmp_path / "my_keywords.txt"
    keywords.write_text("x i 菐o m 菒 x i 菐o m 菒 @小米小米\n")
    assert stt_engine.resolve_kws_keywords_file(
        tmp_path,
        str(keywords),
    ) == str(keywords)


def test_bytes_to_float32():
    samples = stt_engine._bytes_to_float32(b"\x00\x80")
    assert float(samples[0]) == -1.0
