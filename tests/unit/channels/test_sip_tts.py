# -*- coding: utf-8 -*-
"""Unit tests for SIP STT/TTS provider factory."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwenpaw.app.channels.sip import stt_tts


# ---------------------------------------------------------------------------
# Provider normalization / dispatch
# ---------------------------------------------------------------------------


def test_normalize_tts_provider_aliases():
    assert stt_tts.normalize_tts_provider("aliyun") == "aliyun"
    assert stt_tts.normalize_tts_provider("edge_tts") == "edge_tts"
    assert stt_tts.normalize_tts_provider("edgetts") == "edge_tts"
    assert stt_tts.normalize_tts_provider("Edge-TTS") == "edge_tts"
    assert stt_tts.normalize_tts_provider("kokoro") == "kokoro"
    assert stt_tts.normalize_tts_provider("kokoro-onnx") == "kokoro"
    assert stt_tts.normalize_tts_provider("openai") == "openai"
    assert stt_tts.normalize_tts_provider("openai_tts") == "openai"
    assert stt_tts.normalize_tts_provider("openai_compatible") == "openai"
    assert stt_tts.normalize_tts_provider("qwen3") == "qwen3"
    assert stt_tts.normalize_tts_provider("qwen3-tts-flash") == "qwen3"
    assert stt_tts.normalize_qwen3_backend("local") == "local"
    assert stt_tts.normalize_qwen3_backend("offline") == "local"
    assert stt_tts.normalize_qwen3_backend("api") == "api"


def test_normalize_tts_provider_rejects_unknown():
    with pytest.raises(ValueError, match="Unsupported TTS provider"):
        stt_tts.normalize_tts_provider("unknown")


def test_is_qwen3_system_voice_recognizes_builtin_and_clone_ids():
    assert stt_tts.is_qwen3_system_voice("")
    assert stt_tts.is_qwen3_system_voice("Cherry")
    assert stt_tts.is_qwen3_system_voice("cherry")
    assert stt_tts.is_qwen3_system_voice("Serena")
    assert not stt_tts.is_qwen3_system_voice("voice-abc123")
    assert not stt_tts.is_qwen3_system_voice("qwen-tts-vc-my-voice")


def test_is_qwen3_system_voice_supports_env_extra_voices(monkeypatch):
    monkeypatch.setenv(
        stt_tts._QWEN3_EXTRA_SYSTEM_VOICES_ENV,
        "FutureVoice1, futurevoice2",
    )
    assert stt_tts.is_qwen3_system_voice("FutureVoice1")
    assert stt_tts.is_qwen3_system_voice("futurevoice2")
    assert not stt_tts.is_qwen3_system_voice("voice-abc123")


def test_resolve_qwen3_api_model_uses_vc_for_clone_voice():
    assert (
        stt_tts.resolve_qwen3_api_model("", "Cherry")
        == stt_tts._QWEN3_DEFAULT_API_MODEL
    )
    assert (
        stt_tts.resolve_qwen3_api_model("", "voice-abc123")
        == stt_tts._QWEN3_DEFAULT_CLONE_MODEL
    )
    # Explicitly configured VC model is preserved.
    vc = "qwen3-tts-vc-2026-01-22"
    assert stt_tts.resolve_qwen3_api_model(vc, "voice-abc123") == vc
    future_vc = "qwen3-tts-vc-2027-01-01"
    assert (
        stt_tts.resolve_qwen3_api_model(future_vc, "voice-abc123")
        == future_vc
    )
    # Passing a synthesis model for a cloned voice still routes to the
    # dedicated VC model.
    assert (
        stt_tts.resolve_qwen3_api_model("qwen3-tts-flash", "voice-abc123")
        == stt_tts._QWEN3_DEFAULT_CLONE_MODEL
    )


async def test_synthesize_tts_stream_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported TTS provider"):
        async for _ in stt_tts.synthesize_tts_stream(
            "unknown",
            "hello",
            "",
            sample_rate=8000,
        ):
            pass


async def test_synthesize_tts_stream_edge_tts(monkeypatch):
    class FakeCommunicate:
        def __init__(self, text, voice):
            self.text = text
            self.voice = voice

        async def stream(self):
            yield {"type": "audio", "data": b"fake-mp3"}

    monkeypatch.setitem(
        sys.modules,
        "edge_tts",
        SimpleNamespace(Communicate=FakeCommunicate),
    )
    monkeypatch.setattr(
        stt_tts,
        "_decode_mp3_to_pcm16",
        lambda mp3, sample_rate: b"\x01\x00\x02\x00",
    )

    chunks = [
        chunk
        async for chunk in stt_tts.synthesize_tts_stream(
            "edgetts",
            "你好",
            "",
            sample_rate=8000,
        )
    ]
    assert chunks == [b"\x01\x00\x02\x00"]


async def test_synthesize_tts_stream_openai(monkeypatch):
    class FakeResponse:
        status_code = 200
        content = b"fake-mp3"

        @property
        def text(self) -> str:
            return ""

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            self.calls = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            pass

        async def post(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return FakeResponse()

    monkeypatch.setattr(stt_tts.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(
        stt_tts,
        "_decode_mp3_to_pcm16",
        lambda audio, sample_rate: b"\x01\x00\x02\x00",
    )

    chunks = [
        chunk
        async for chunk in stt_tts.synthesize_tts_stream(
            "openai",
            "你好",
            "",
            sample_rate=8000,
            openai_api_key="sk-test",
            openai_base_url="https://example.test/v1",
            openai_model="tts-test",
        )
    ]
    assert chunks == [b"\x01\x00\x02\x00"]


def test_qwen3_prompt_dict_includes_ref_text():
    item = SimpleNamespace(
        ref_code="code",
        ref_spk_embedding="embedding",
        x_vector_only_mode=False,
        icl_mode=True,
        ref_text="hello",
    )
    prompt = stt_tts._prompt_items_to_dict([item])
    assert prompt["ref_text"] == ["hello"]
    assert prompt["ref_code"] == ["code"]


def test_qwen3_clone_fingerprint_is_content_based(tmp_path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio-bytes")
    first = stt_tts._qwen3_clone_fingerprint(
        "model",
        str(audio),
        "text",
    )
    second = stt_tts._qwen3_clone_fingerprint(
        "model",
        str(audio),
        "text",
    )
    assert first == second
    audio.write_bytes(b"different-bytes")
    assert stt_tts._qwen3_clone_fingerprint(
        "model",
        str(audio),
        "text",
    ) != first

def test_get_qwen3_local_tts_reports_native_load_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "qwen_tts" or name.startswith("qwen_tts."):
            raise OSError(127, "The specified procedure could not be found")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ValueError, match="torchaudio/torch"):
        stt_tts._get_qwen3_local_tts(
            "",
            "cpu",
            "sample.wav",
            "sample text",
            True,
        )

async def test_synthesize_tts_stream_qwen3_local(monkeypatch):
    class FakeSamples:
        def __len__(self):
            return 1

    class FakeModel:
        def __init__(self):
            self.captured = {}

        def generate_voice_clone(self, **kwargs):
            self.captured.update(kwargs)
            return ([FakeSamples()], 24000)

    fake_model = FakeModel()

    monkeypatch.setattr(
        stt_tts,
        "_get_qwen3_local_tts",
        lambda model_dir, device, ref_audio, ref_text, keep: (
            fake_model,
            object(),
        ),
    )
    monkeypatch.setattr(
        stt_tts,
        "_samples_to_pcm16",
        lambda samples: b"\x01\x00\x02\x00",
    )
    monkeypatch.setattr(
        stt_tts,
        "_resample_pcm16",
        lambda pcm, src_rate, dst_rate: pcm,
    )

    chunks = [
        chunk
        async for chunk in stt_tts.synthesize_tts_stream(
            "qwen3",
            "你好",
            "",
            sample_rate=8000,
            qwen3_backend="local",
            qwen3_ref_audio="sample.wav",
            qwen3_ref_text="sample text",
        )
    ]
    assert chunks == [b"\x01\x00\x02\x00"]
    assert fake_model.captured["do_sample"] is True
    assert fake_model.captured["subtalker_dosample"] is False
    assert fake_model.captured["top_k"] == 50
    assert fake_model.captured["subtalker_top_k"] == 1


async def test_synthesize_tts_stream_qwen3_local_reports_blackwell_cuda(monkeypatch):
    class FakeModel:
        def generate_voice_clone(self, **kwargs):
            raise RuntimeError(
                "CUDA error: no kernel image is available for execution on "
                "the device",
            )

    monkeypatch.setattr(
        stt_tts,
        "_get_qwen3_local_tts",
        lambda model_dir, device, ref_audio, ref_text, keep: (
            FakeModel(),
            object(),
        ),
    )

    with pytest.raises(ValueError, match="CUDA 12.8"):
        async for _ in stt_tts.synthesize_tts_stream(
            "qwen3",
            "你好",
            "",
            sample_rate=8000,
            qwen3_backend="local",
            qwen3_ref_audio="sample.wav",
            qwen3_ref_text="sample text",
        ):
            pass


async def test_clone_qwen3_voice_uploads_and_enrolls(monkeypatch):
    class FakeEnrollment:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def create_voice(self, **kwargs):
            self.kwargs.update(kwargs)
            return "voice-abc123"

    fake_enrollment = SimpleNamespace(VoiceEnrollmentService=FakeEnrollment)
    monkeypatch.setitem(
        sys.modules,
        "dashscope.audio.tts_v2.enrollment",
        fake_enrollment,
    )

    async def fake_media_url(media, *, api_key, model):
        return "oss://bucket/sample.wav"

    monkeypatch.setattr(stt_tts, "_qwen3_clone_media_url", fake_media_url)

    voice_id = await stt_tts.clone_qwen3_voice(
        "sample.wav",
        preferred_name="我的声音",
        api_key="sk-test",
        model="qwen3-tts-vc-test",
    )
    assert voice_id == "voice-abc123"


async def test_clone_qwen3_voice_defaults_to_vc_model(monkeypatch):
    created = []

    class FakeEnrollment:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def create_voice(self, **kwargs):
            created.append({**self.kwargs, **kwargs})
            return "voice-abc123"

    fake_enrollment = SimpleNamespace(VoiceEnrollmentService=FakeEnrollment)
    monkeypatch.setitem(
        sys.modules,
        "dashscope.audio.tts_v2.enrollment",
        fake_enrollment,
    )

    async def fake_media_url(media, *, api_key, model):
        return "oss://bucket/sample.wav"

    monkeypatch.setattr(stt_tts, "_qwen3_clone_media_url", fake_media_url)

    voice_id = await stt_tts.clone_qwen3_voice(
        "sample.wav",
        preferred_name="我的声音",
        api_key="sk-test",
    )
    assert voice_id == "voice-abc123"
    assert len(created) == 1
    assert created[0]["target_model"] == stt_tts._QWEN3_DEFAULT_CLONE_MODEL

    # Even if a caller accidentally passes the synthesis model as the clone
    # target, enrollment must still use the dedicated VC model.
    await stt_tts.clone_qwen3_voice(
        "sample.wav",
        preferred_name="我的声音",
        api_key="sk-test",
        model="qwen3-tts-flash",
    )
    assert len(created) == 2
    assert created[1]["target_model"] == stt_tts._QWEN3_DEFAULT_CLONE_MODEL


async def test_synthesize_tts_stream_kokoro(monkeypatch):
    class FakeGenerationConfig:
        pass

    class FakeSherpa:
        GenerationConfig = FakeGenerationConfig

    class FakeTts:
        def generate(self, text, gen_config):
            return SimpleNamespace(
                samples=[0.0, 0.5],
                sample_rate=24000,
            )

    monkeypatch.setattr(
        stt_tts,
        "resolve_kokoro_model_dir",
        lambda model_dir="", model_variant="float32": Path("/fake/kokoro"),
    )
    monkeypatch.setattr(
        stt_tts,
        "validate_kokoro_model_dir",
        lambda model_dir: None,
    )
    monkeypatch.setattr(
        stt_tts,
        "_get_kokoro_tts",
        lambda model_dir, num_threads, keep_model_loaded=True: (
            FakeTts(),
            FakeSherpa(),
        ),
    )
    monkeypatch.setattr(
        stt_tts,
        "resolve_kokoro_sid",
        lambda voice, model_dir=None: 45,
    )
    monkeypatch.setattr(
        stt_tts,
        "_samples_to_pcm16",
        lambda samples: b"\x01\x00\x02\x00\x03\x00",
    )
    monkeypatch.setattr(
        stt_tts,
        "_resample_pcm16",
        lambda pcm, src_rate, dst_rate: pcm,
    )

    chunks = [
        chunk
        async for chunk in stt_tts.synthesize_tts_stream(
            "kokoro",
            "你好",
            "zf_xiaobei",
            sample_rate=24000,
            kokoro_model_dir="/fake/kokoro",
            kokoro_num_threads=2,
        )
    ]
    assert chunks == [b"\x01\x00\x02\x00\x03\x00"]


async def test_warmup_tts_preloads_kokoro_only(monkeypatch):
    model_path = Path("/fake/kokoro")
    loaded = []
    monkeypatch.setattr(
        stt_tts,
        "resolve_kokoro_model_dir",
        lambda model_dir="", model_variant="float32": model_path,
    )
    monkeypatch.setattr(
        stt_tts,
        "validate_kokoro_model_dir",
        lambda path: None,
    )
    monkeypatch.setattr(
        stt_tts,
        "_get_kokoro_tts",
        lambda path, threads, keep_model_loaded=True: loaded.append(
            (path, threads),
        ),
    )

    await stt_tts.warmup_tts("kokoro", kokoro_num_threads=3)
    await stt_tts.warmup_tts("aliyun")
    await stt_tts.warmup_tts("openai")
    await stt_tts.warmup_tts(
        "kokoro",
        kokoro_num_threads=3,
        keep_model_loaded=False,
    )

    assert loaded == [(model_path, 3)]


# ---------------------------------------------------------------------------
# Kokoro helpers
# ---------------------------------------------------------------------------


def test_resolve_kokoro_sid_by_name_and_id():
    assert stt_tts.resolve_kokoro_sid("zf_xiaobei") == 45
    assert stt_tts.resolve_kokoro_sid("zm_yunxi") == 50
    assert stt_tts.resolve_kokoro_sid("45") == 45
    assert stt_tts.resolve_kokoro_sid("") == 3
    assert stt_tts.resolve_kokoro_sid(
        "zf_001",
        Path("kokoro-multi-lang-v1_1"),
    ) == 3


def test_resolve_kokoro_sid_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown Kokoro voice"):
        stt_tts.resolve_kokoro_sid("no_such_voice")


def test_validate_kokoro_model_dir_complete(tmp_path: Path):
    for name in ("model.onnx", "voices.bin", "tokens.txt"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "espeak-ng-data").mkdir()

    stt_tts.validate_kokoro_model_dir(tmp_path)


def test_validate_kokoro_model_dir_accepts_int8_model(tmp_path: Path):
    for name in ("model.int8.onnx", "voices.bin", "tokens.txt"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "espeak-ng-data").mkdir()

    stt_tts.validate_kokoro_model_dir(tmp_path)
    assert stt_tts._kokoro_model_file(tmp_path).name == "model.int8.onnx"


def test_resolve_kokoro_v11_voice_after_custom_directory_relocation(
    tmp_path: Path,
):
    (tmp_path / "README.md").write_text("Kokoro v1.1-zh")
    assert stt_tts.resolve_kokoro_sid("zf_001", tmp_path) == 3


def test_validate_kokoro_model_dir_missing_files(tmp_path: Path):
    with pytest.raises(ValueError, match="is incomplete"):
        stt_tts.validate_kokoro_model_dir(tmp_path)


def test_resolve_kokoro_model_dir_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("KOKORO_MODEL_DIR", str(tmp_path))
    assert stt_tts.resolve_kokoro_model_dir("") == tmp_path
    assert stt_tts.resolve_kokoro_model_dir("/explicit") == Path("/explicit")
    monkeypatch.delenv("KOKORO_MODEL_DIR")
    assert stt_tts.resolve_kokoro_model_dir("", "int8").name == (
        "kokoro-int8-multi-lang-v1_1"
    )


# ---------------------------------------------------------------------------
# edge-tts helpers
# ---------------------------------------------------------------------------


def test_decode_mp3_requires_ffmpeg(monkeypatch):
    monkeypatch.setattr(stt_tts.shutil, "which", lambda name: None)
    with pytest.raises(ValueError, match="ffmpeg"):
        stt_tts._decode_mp3_to_pcm16(b"mp3", 8000)


def test_decode_mp3_uses_ffmpeg_path_env(monkeypatch, tmp_path: Path):
    exe = tmp_path / "ffmpeg.exe"
    exe.write_bytes(b"")
    monkeypatch.setenv("FFMPEG_PATH", str(exe))
    monkeypatch.setattr(stt_tts.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        stt_tts.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=b"\x01\x00",
            stderr=b"",
        ),
    )

    assert stt_tts._decode_mp3_to_pcm16(b"mp3", 8000) == b"\x01\x00"


def test_pcm_chunk_size():
    assert stt_tts._pcm_chunk_size(8000) == 640
    assert stt_tts._pcm_chunk_size(24000) == 1920
