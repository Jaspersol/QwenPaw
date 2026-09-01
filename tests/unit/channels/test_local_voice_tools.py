# -*- coding: utf-8 -*-
"""Tests for Local Voice model visibility and LLM-free diagnostics."""
from __future__ import annotations

import subprocess

import pytest

from qwenpaw.app.channels.local_voice import tools
from qwenpaw.config.config import LocalVoiceChannelConfig


def test_local_kws_supported_only_for_zipformer_provider():
    assert tools.local_kws_supported("sherpa_zipformer") is True
    assert tools.local_kws_supported("zipformer") is True
    assert tools.local_kws_supported("openai") is False
    assert tools.local_kws_supported("aliyun") is False
    assert tools.local_kws_supported("unknown") is False


def test_voice_model_statuses_report_effective_custom_paths(tmp_path):
    zipformer = tmp_path / "zipformer"
    kws = tmp_path / "kws"
    kokoro = tmp_path / "kokoro"
    qwen3 = tmp_path / "qwen3"
    for directory in (zipformer, kws):
        directory.mkdir()
        (directory / "tokens.txt").write_text("tokens")
        for name in ("encoder.int8.onnx", "decoder.onnx", "joiner.int8.onnx"):
            (directory / name).write_bytes(b"onnx")
    (kws / "keywords.txt").write_text("hello")
    kokoro.mkdir()
    for name in ("model.onnx", "voices.bin", "tokens.txt", "espeak-ng-data"):
        path = kokoro / name
        if name == "espeak-ng-data":
            path.mkdir()
        else:
            path.write_bytes(b"model")
    qwen3.mkdir()
    (qwen3 / "config.json").write_text("{}")
    (qwen3 / "tokenizer.json").write_text("{}")
    (qwen3 / "model.safetensors").write_bytes(b"weights")

    statuses = tools.voice_model_statuses(
        LocalVoiceChannelConfig(
            zipformer_model_dir=str(zipformer),
            kws_model_dir=str(kws),
            kokoro_model_dir=str(kokoro),
            qwen3_model_dir=str(qwen3),
        )
    )

    assert [item.id for item in statuses] == [
        "zipformer",
        "kws",
        "kokoro",
        "kokoro_int8",
        "qwen3",
    ]
    assert all(item.complete for item in statuses[:3])
    assert statuses[3].complete is False
    assert statuses[4].complete is True
    assert {item.id: item.path for item in statuses}["kokoro"] == str(kokoro)
    assert {item.id: item.path for item in statuses}["qwen3"] == str(qwen3)


def test_download_rejects_existing_target(tmp_path):
    target = tmp_path / "already-there"
    target.mkdir()
    manager = tools.VoiceModelDownloadManager()
    config = LocalVoiceChannelConfig(zipformer_model_dir=str(target))

    with pytest.raises(ValueError, match="already exists"):
        manager.start("zipformer", config)


async def test_tts_diagnostic_does_not_create_agent_request(monkeypatch):
    manager = tools.LocalVoiceTestManager()
    called = []

    async def fake_tts(cfg, text):
        called.append((cfg, text))

    monkeypatch.setattr(manager, "_test_tts", fake_tts)
    await manager._run("tts", LocalVoiceChannelConfig(), "hello")

    assert called[0][1] == "hello"
    assert manager.status.status == "passed"
    assert manager.status.transcript is None


def test_optional_voice_dependencies_include_custom_keyword_tokenizer(
    monkeypatch,
):
    def fake_find_spec(package):
        return None if package == "pypinyin" else object()

    monkeypatch.setattr(tools.importlib.util, "find_spec", fake_find_spec)

    assert "pypinyin" in tools.optional_voice_dependencies()


def test_optional_voice_dependencies_include_qwen_tts(monkeypatch):
    def fake_find_spec(package):
        return None if package == "qwen_tts" else object()

    monkeypatch.setattr(tools.importlib.util, "find_spec", fake_find_spec)

    assert "qwen_tts" in tools.optional_voice_dependencies()


async def test_dependency_install_manager_reports_completion(monkeypatch):
    manager = tools.VoiceDependencyInstallManager()
    checks = iter([["pypinyin"], []])
    installed = []

    monkeypatch.setattr(
        tools,
        "optional_voice_dependencies",
        lambda: next(checks),
    )
    monkeypatch.setattr(
        tools,
        "_install_voice_dependencies",
        lambda packages, **kwargs: installed.extend(packages),
    )

    manager.start()
    await manager._task

    assert installed == ["pypinyin"]
    assert manager.status.status == "completed"
    assert manager.status.error is None


def test_dependency_installer_uses_fixed_allowlist(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(tools.sys, "frozen", False, raising=False)
    monkeypatch.setattr(tools.sys, "executable", "test-python")
    monkeypatch.setattr(tools.subprocess, "run", fake_run)

    tools._install_voice_dependencies(["pypinyin"])

    assert calls[0][0] == [
        "test-python",
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "pypinyin>=0.51.0,<1",
    ]


def test_dependency_installer_uses_cuda_index_for_torch(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(tools.sys, "frozen", False, raising=False)
    monkeypatch.setattr(tools.sys, "executable", "test-python")
    monkeypatch.setattr(tools.subprocess, "run", fake_run)

    tools._install_voice_dependencies(["torch_cuda"])

    assert calls[0] == [
        "test-python",
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "--force-reinstall",
        "--index-url",
        "https://download.pytorch.org/whl/cu128",
        "torch",
    ]

def test_dependency_installer_uses_configured_torch_index(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(tools.sys, "frozen", False, raising=False)
    monkeypatch.setattr(tools.sys, "executable", "test-python")
    monkeypatch.setattr(tools.subprocess, "run", fake_run)

    tools._install_voice_dependencies(
        ["torch_cuda"],
        torch_index_url="https://mirror.example/pytorch",
    )

    assert calls[0] == [
        "test-python",
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "--force-reinstall",
        "--index-url",
        "https://mirror.example/pytorch",
        "torch",
    ]


def test_dependency_installer_uses_configured_pip_index(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(tools.sys, "frozen", False, raising=False)
    monkeypatch.setattr(tools.sys, "executable", "test-python")
    monkeypatch.setattr(tools.subprocess, "run", fake_run)

    tools._install_voice_dependencies(
        ["pypinyin"],
        pip_index_url="https://pypi.tuna.tsinghua.edu.cn/simple",
    )

    assert calls[0] == [
        "test-python",
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "--index-url",
        "https://pypi.tuna.tsinghua.edu.cn/simple",
        "pypinyin>=0.51.0,<1",
    ]


async def test_install_manager_passes_index_urls_from_cfg(monkeypatch):
    manager = tools.VoiceDependencyInstallManager()
    captured = []

    def fake_install(packages, **kwargs):
        captured.append((packages, kwargs))

    monkeypatch.setattr(tools, "_install_voice_dependencies", fake_install)
    monkeypatch.setattr(tools, "dependency_installed", lambda name: True)
    cfg = LocalVoiceChannelConfig(
        pip_index_url="https://pypi.example/simple",
        torch_index_url="https://torch.example/cu124",
    )

    manager.start(["torch_cuda"], cfg=cfg)
    await manager._task

    assert captured == [(
        ["torch_cuda"],
        {
            "pip_index_url": "https://pypi.example/simple",
            "torch_index_url": "https://torch.example/cu124",
        },
    )]


def test_torchaudio_pin_uses_cuda_build(monkeypatch):
    monkeypatch.setattr(
        tools.importlib.metadata,
        "version",
        lambda name: "2.6.0+cu124",
    )

    assert tools._torchaudio_pin() == (
        "torchaudio==2.6.0+cu124",
        "https://download.pytorch.org/whl/cu124",
    )


def test_torchaudio_pin_uses_plain_version_for_cpu(monkeypatch):
    monkeypatch.setattr(
        tools.importlib.metadata,
        "version",
        lambda name: "2.6.0",
    )

    assert tools._torchaudio_pin() == ("torchaudio==2.6.0", "")


def test_torchaudio_pin_preserves_cpu_build_tag(monkeypatch):
    monkeypatch.setattr(
        tools.importlib.metadata,
        "version",
        lambda name: "2.6.0+cpu",
    )

    assert tools._torchaudio_pin() == ("torchaudio==2.6.0+cpu", "")


def test_torchaudio_matches_torch_uses_metadata(monkeypatch):
    versions = {
        "torch": "2.6.0+cu124",
        "torchaudio": "2.6.0+cu124",
    }
    monkeypatch.setattr(
        tools.importlib.metadata,
        "version",
        lambda name: versions[name],
    )
    assert tools._torchaudio_matches_torch() is True

    versions["torchaudio"] = "2.11.0"
    assert tools._torchaudio_matches_torch() is False


def test_maybe_align_torchaudio_installs_matching_build(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(
        tools.importlib.metadata,
        "version",
        lambda name: "2.6.0+cu124",
    )
    monkeypatch.setattr(
        tools,
        "dependency_installed",
        lambda name: name == "qwen_tts",
    )
    monkeypatch.setattr(tools, "_torchaudio_matches_torch", lambda: False)
    monkeypatch.setattr(tools.sys, "frozen", False, raising=False)
    monkeypatch.setattr(tools.sys, "executable", "test-python")
    monkeypatch.setattr(tools.subprocess, "run", fake_run)

    tools._maybe_align_torchaudio("torch_cuda")

    assert calls == [[
        "test-python",
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "--index-url",
        "https://download.pytorch.org/whl/cu124",
        "torchaudio==2.6.0+cu124",
    ]]
def test_maybe_align_torchaudio_uses_configured_torch_index(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(
        tools.importlib.metadata,
        "version",
        lambda name: "2.6.0+cu124",
    )
    monkeypatch.setattr(
        tools,
        "dependency_installed",
        lambda name: name == "qwen_tts",
    )
    monkeypatch.setattr(tools, "_torchaudio_matches_torch", lambda: False)
    monkeypatch.setattr(tools.sys, "frozen", False, raising=False)
    monkeypatch.setattr(tools.sys, "executable", "test-python")
    monkeypatch.setattr(tools.subprocess, "run", fake_run)

    tools._maybe_align_torchaudio(
        "torch_cuda",
        torch_index_url="https://mirror.example/pytorch",
    )

    assert calls == [[
        "test-python",
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "--index-url",
        "https://mirror.example/pytorch",
        "torchaudio==2.6.0+cu124",
    ]]


def test_maybe_align_torchaudio_skips_when_qwen_tts_not_installed(
    monkeypatch,
):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(
        tools,
        "dependency_installed",
        lambda name: False,
    )
    monkeypatch.setattr(tools.subprocess, "run", fake_run)

    tools._maybe_align_torchaudio("torch_cuda")

    assert calls == []


async def test_install_manager_only_installs_selected_packages(monkeypatch):
    manager = tools.VoiceDependencyInstallManager()
    installed = []

    monkeypatch.setattr(
        tools,
        "_install_voice_dependencies",
        lambda packages, **kwargs: installed.extend(packages),
    )
    monkeypatch.setattr(
        tools,
        "dependency_installed",
        lambda name: True,
    )

    manager.start(["torch_cuda"])
    assert manager.status.status == "installing"
    assert manager.status.packages == ["torch_cuda"]
    await manager._task
    assert installed == ["torch_cuda"]
    assert manager.status.status == "completed"


async def test_install_manager_filters_unknown_packages(monkeypatch):
    manager = tools.VoiceDependencyInstallManager()
    installed = []

    monkeypatch.setattr(
        tools,
        "_install_voice_dependencies",
        lambda packages, **kwargs: installed.extend(packages),
    )
    monkeypatch.setattr(
        tools,
        "dependency_installed",
        lambda name: True,
    )

    manager.start(["torch_cuda", "not-a-real-dependency"])
    await manager._task

    assert installed == ["torch_cuda"]


async def test_install_manager_reloads_torch_after_cuda_install(monkeypatch):
    manager = tools.VoiceDependencyInstallManager()
    forgotten = []

    monkeypatch.setattr(
        tools,
        "_install_voice_dependencies",
        lambda packages, **kwargs: None,
    )
    monkeypatch.setattr(
        tools,
        "dependency_installed",
        lambda name: True,
    )
    monkeypatch.setattr(
        tools,
        "_forget_imported_dependency",
        lambda name: forgotten.append(name),
    )

    manager.start(["torch_cuda"])
    await manager._task

    assert forgotten == ["torch_cuda"]
    assert manager.status.status == "completed"


def test_uninstall_torch_cuda_targets_pip_torch(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(tools.sys, "frozen", False, raising=False)
    monkeypatch.setattr(tools.sys, "executable", "test-python")
    monkeypatch.setattr(tools.subprocess, "run", fake_run)

    tools.uninstall_voice_dependencies(["torch_cuda"])

    assert calls[0] == [
        "test-python",
        "-m",
        "pip",
        "uninstall",
        "-y",
        "--disable-pip-version-check",
        "torch",
    ]


def test_uninstall_rejects_unknown_dependency():
    with pytest.raises(ValueError, match="Unknown Local Voice dependency"):
        tools.uninstall_voice_dependencies(["nope"])


def test_uninstall_forgets_imported_module(monkeypatch):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(tools.sys, "frozen", False, raising=False)
    monkeypatch.setattr(tools.sys, "executable", "test-python")
    monkeypatch.setattr(tools.subprocess, "run", fake_run)
    monkeypatch.setitem(tools.sys.modules, "pypinyin", object())
    monkeypatch.setitem(tools.sys.modules, "pypinyin.core", object())

    tools.uninstall_voice_dependencies(["pypinyin"])

    assert "pypinyin" not in tools.sys.modules
    assert "pypinyin.core" not in tools.sys.modules


def test_dependency_installed_torch_cuda_requires_cuda_build(monkeypatch):
    monkeypatch.setattr(
        tools.importlib.metadata,
        "version",
        lambda name: "2.6.0",
    )
    assert tools.dependency_installed("torch_cuda") is False

    monkeypatch.setattr(
        tools.importlib.metadata,
        "version",
        lambda name: "2.6.0+cu124",
    )
    assert tools.dependency_installed("torch_cuda") is True

def test_forget_torch_cuda_keeps_torch_in_sys_modules(monkeypatch):
    fake_torch = object()
    monkeypatch.setitem(tools.sys.modules, "torch", fake_torch)

    tools._forget_imported_dependency("torch_cuda")

    assert tools.sys.modules["torch"] is fake_torch
    monkeypatch.delitem(tools.sys.modules, "torch")

def test_dependency_installed_reflects_uninstall_of_imported_module(
    monkeypatch,
):
    monkeypatch.setitem(tools.sys.modules, "pypinyin", object())
    monkeypatch.setattr(
        tools.importlib.metadata,
        "distribution",
        lambda name: (_ for _ in ()).throw(
            tools.importlib.metadata.PackageNotFoundError(name)
        ),
    )

    assert tools.dependency_installed("pypinyin") is False
    monkeypatch.delitem(tools.sys.modules, "pypinyin")


def test_required_dependencies_include_torch_cuda_for_cuda_device():
    cfg = LocalVoiceChannelConfig(
        tts_provider="qwen3",
        qwen3_backend="local",
        qwen3_device="cuda:0",
    )
    assert "torch_cuda" in tools.required_voice_dependencies(cfg)


def test_required_dependencies_cpu_device_skips_torch_cuda():
    cfg = LocalVoiceChannelConfig(
        tts_provider="qwen3",
        qwen3_backend="local",
        qwen3_device="cpu",
    )
    assert "torch_cuda" not in tools.required_voice_dependencies(cfg)


def test_required_dependencies_for_local_asr_and_kokoro():
    cfg = LocalVoiceChannelConfig(
        stt_provider="sherpa_zipformer",
        tts_provider="kokoro",
    )
    required = tools.required_voice_dependencies(cfg)
    assert {"sherpa_onnx", "pypinyin", "sounddevice", "numpy"} <= required
    assert "torch_cuda" not in required


def test_required_dependencies_for_api_providers_only():
    cfg = LocalVoiceChannelConfig(
        stt_provider="openai",
        tts_provider="aliyun",
    )
    required = tools.required_voice_dependencies(cfg)
    assert required == {"sounddevice", "numpy"}


def test_dependency_items_flag_required_for_current_settings(monkeypatch):
    cfg = LocalVoiceChannelConfig(
        tts_provider="qwen3",
        qwen3_backend="local",
        qwen3_device="cuda:0",
    )
    # Installed detection imports torch in the real runtime; keep this unit
    # test focused on the required flags and avoid native torch loading.
    monkeypatch.setattr(
        tools,
        "dependency_installed",
        lambda name: name != "edge_tts",
    )
    items = tools.voice_dependency_items(cfg)
    by_id = {item.id: item for item in items}

    assert set(by_id) == set(tools._VOICE_DEPENDENCIES)
    assert by_id["torch_cuda"].required is True
    assert by_id["qwen_tts"].required is True
    assert by_id["huggingface_hub"].required is True
    assert by_id["edge_tts"].required is False


def test_list_qwen3_devices_without_torch(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    options = tools.list_qwen3_inference_devices()

    assert [item.value for item in options] == ["cpu", "cuda"]
    assert options[0].available is True
    assert options[1].available is False
    assert "torch_cuda" in options[1].label


def test_list_qwen3_devices_enumerates_each_gpu(monkeypatch):
    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 2

        @staticmethod
        def get_device_name(index):
            return f"Test GPU {index}"

    class FakeBackends:
        mps = None

    class FakeTorch:
        cuda = FakeCuda
        backends = FakeBackends

    monkeypatch.setitem(tools.sys.modules, "torch", FakeTorch)

    options = tools.list_qwen3_inference_devices()

    assert [item.value for item in options] == ["cpu", "cuda:0", "cuda:1"]
    assert options[1].label == "CUDA 0 · Test GPU 0"
    assert options[2].label == "CUDA 1 · Test GPU 1"
    monkeypatch.delitem(tools.sys.modules, "torch")


def test_list_qwen3_devices_offers_mps_on_macos(monkeypatch):
    class FakeMps:
        @staticmethod
        def is_available():
            return True

    class FakeBackends:
        mps = FakeMps

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    class FakeTorch:
        cuda = FakeCuda
        backends = FakeBackends

    monkeypatch.setitem(tools.sys.modules, "torch", FakeTorch)

    options = tools.list_qwen3_inference_devices()

    assert [item.value for item in options] == ["cpu", "cuda", "mps"]
    assert options[2].available is True
    monkeypatch.delitem(tools.sys.modules, "torch")
