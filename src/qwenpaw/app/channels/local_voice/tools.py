# -*- coding: utf-8 -*-
"""Model management and hardware diagnostics for the Local Voice channel.

The helpers in this module never construct an Agent request.  They are used by
the console to download the three optional offline voice models and to exercise
microphone/ASR/KWS/TTS independently from any LLM configuration.
"""
from __future__ import annotations

import asyncio
import importlib
import importlib.util
import logging
import os
import queue
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from qwenpaw.constant import MODELS_DIR
from qwenpaw.app.channels.sip._audioop_compat import audioop
from qwenpaw.app.channels.sip.stt_engine import (
    _KWS_MODEL_NAME,
    _ZIPFORMER_MODEL_NAME,
    normalize_stt_provider,
    resolve_kws_model_dir,
    resolve_zipformer_model_dir,
)
from qwenpaw.app.channels.sip.stt_tts import (
    resolve_kokoro_model_dir,
    synthesize_tts_stream,
)
from qwenpaw.config.config import LocalVoiceChannelConfig

logger = logging.getLogger(__name__)

VoiceAssetName = Literal[
    "zipformer",
    "kws",
    "kokoro",
    "kokoro_int8",
    "qwen3",
]
_QWEN3_LOCAL_MODEL_DIR_NAME = "Qwen3-TTS-12Hz-0.6B-Base"
VoiceTestName = Literal["wake_word", "asr", "tts"]

_VOICE_DEPENDENCIES: dict[str, dict[str, Any]] = {
    "sherpa_onnx": {
        "name": "sherpa-onnx",
        "spec": "sherpa-onnx>=1.13.0,<2",
        "module": "sherpa_onnx",
        "description": "Local Zipformer ASR / Kokoro TTS inference runtime.",
    },
    "sounddevice": {
        "name": "sounddevice",
        "spec": "sounddevice>=0.5.0,<1",
        "module": "sounddevice",
        "description": (
            "Microphone capture and speaker playback for Local Voice."
        ),
    },
    "numpy": {
        "name": "numpy",
        "spec": "numpy>=1.26,<3",
        "module": "numpy",
        "description": (
            "Audio sample conversion used by local models and tests."
        ),
    },
    "pypinyin": {
        "name": "pypinyin",
        "spec": "pypinyin>=0.51.0,<1",
        "module": "pypinyin",
        "description": (
            "Converts custom Chinese wake words to KWS pinyin tokens."
        ),
    },
    "edge_tts": {
        "name": "edge-tts",
        "spec": "edge-tts>=7.0.0,<8",
        "module": "edge_tts",
        "description": "Microsoft Edge online TTS provider.",
    },
    "huggingface_hub": {
        "name": "huggingface-hub",
        "spec": "huggingface-hub>=0.25.0,<2",
        "module": "huggingface_hub",
        "description": "Downloads the Qwen3-TTS 0.6B Hugging Face snapshot.",
    },
    "qwen_tts": {
        "name": "qwen-tts",
        "spec": "qwen-tts",
        "module": "qwen_tts",
        "description": "Local Qwen3-TTS 0.6B inference and voice cloning.",
    },
    "torch_cuda": {
        "name": "PyTorch (CUDA 12.8)",
        "spec": "torch",
        "module": "torch",
        "pip_name": "torch",
        "index_url": "https://download.pytorch.org/whl/cu128",
        "force_reinstall": True,
        "description": (
            "CUDA-enabled PyTorch used by local Qwen3-TTS. Install this "
            "instead of the CPU torch wheel when qwen3_device selects cuda."
        ),
    },
}
_DEPENDENCY_INSTALL_TIMEOUT_SECONDS = 900

_ASSETS: dict[str, dict[str, Any]] = {
    "zipformer": {
        "name": _ZIPFORMER_MODEL_NAME,
        "url": (
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "asr-models/"
            "sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-"
            "zh-en-punct-int8-2026-06-05.tar.bz2"
        ),
        "required": ("tokens.txt",),
    },
    "kws": {
        "name": _KWS_MODEL_NAME,
        "url": (
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "kws-models/"
            "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2"
        ),
        # The official zh-en 3M archive may not ship keywords.txt; the
        # engine falls back to built-in tokenized defaults in that case.
        "required": ("tokens.txt",),
    },
    "kokoro": {
        "name": "kokoro-multi-lang-v1_1",
        "url": (
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "tts-models/kokoro-multi-lang-v1_1.tar.bz2"
        ),
        "required": ("voices.bin", "tokens.txt", "espeak-ng-data"),
    },
    "kokoro_int8": {
        "name": "kokoro-int8-multi-lang-v1_1",
        "url": (
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "tts-models/kokoro-int8-multi-lang-v1_1.tar.bz2"
        ),
        "required": ("voices.bin", "tokens.txt", "espeak-ng-data"),
    },
    "qwen3": {
        "name": "Qwen3-TTS-12Hz-0.6B-Base",
        "url": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "kind": "huggingface",
        "required": ("config.json",),
    },
}


class VoiceModelStatus(BaseModel):
    id: VoiceAssetName
    name: str
    path: str
    installed: bool
    complete: bool
    size_bytes: int = 0
    missing_files: list[str] = Field(default_factory=list)
    download_url: str


class VoiceDownloadStatus(BaseModel):
    status: Literal["idle", "downloading", "completed", "failed"] = "idle"
    asset: VoiceAssetName | None = None
    downloaded_bytes: int = 0
    total_bytes: int | None = None
    error: str | None = None


class VoiceTestStatus(BaseModel):
    status: Literal["idle", "running", "passed", "failed", "timed_out"] = "idle"
    test: VoiceTestName | None = None
    message: str | None = None
    transcript: str | None = None
    keyword: str | None = None


class VoiceDependencyInstallStatus(BaseModel):
    """Progress exposed to the console for the fixed dependency installer."""

    status: Literal["idle", "installing", "completed", "failed"] = "idle"
    packages: list[str] = Field(default_factory=list)
    error: str | None = None


class VoiceDependencyItem(BaseModel):
    id: str
    name: str
    spec: str
    description: str
    installed: bool
    required: bool
    can_uninstall: bool = True


class VoiceDependencyStatus(BaseModel):
    items: list[VoiceDependencyItem]
    install: VoiceDependencyInstallStatus


class VoiceDeviceOption(BaseModel):
    value: str
    label: str
    available: bool
    description: str = ""


class VoiceDependencyInstallManager:
    """Install selected Local Voice dependencies in the background."""

    def __init__(self) -> None:
        self.status = VoiceDependencyInstallStatus()
        self._task: asyncio.Task[None] | None = None

    def start(
        self,
        packages: list[str] | None = None,
        *,
        cfg: LocalVoiceChannelConfig | None = None,
    ) -> None:
        if self._task and not self._task.done():
            raise RuntimeError("Local Voice dependencies are already installing")
        selected = packages or optional_voice_dependencies()
        selected = [name for name in selected if name in _VOICE_DEPENDENCIES]
        if not selected:
            self.status = VoiceDependencyInstallStatus(
                status="completed",
                packages=[],
            )
            return
        self.status = VoiceDependencyInstallStatus(
            status="installing",
            packages=selected,
        )
        self._task = asyncio.create_task(self._run(selected, cfg))

    async def _run(
        self,
        packages: list[str],
        cfg: LocalVoiceChannelConfig | None = None,
    ) -> None:
        try:
            await asyncio.to_thread(
                _install_voice_dependencies,
                packages,
                pip_index_url=cfg.pip_index_url if cfg else "",
                torch_index_url=cfg.torch_index_url if cfg else "",
            )
            importlib.invalidate_caches()
            if "torch_cuda" in packages:
                # The previous torch wheel may already be imported (for
                # example the CPU build); drop it so the freshly installed
                # CUDA wheel is loaded for verification and the device
                # selector.
                _forget_imported_dependency("torch_cuda")
            remaining = [
                name for name in packages if not dependency_installed(name)
            ]
            if remaining:
                raise RuntimeError(
                    "Installation finished but dependencies are still unavailable: "
                    + ", ".join(remaining),
                )
            self.status.status = "completed"
        except Exception as exc:
            logger.exception("Local Voice dependency installation failed")
            self.status.status = "failed"
            self.status.error = str(exc) or exc.__class__.__name__


def _dependency_metadata(name: str) -> dict[str, Any]:
    try:
        return _VOICE_DEPENDENCIES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown Local Voice dependency: {name}") from exc


def _pip_name(metadata: dict[str, Any]) -> str:
    """The pip distribution name for a dependency entry."""
    return str(metadata.get("pip_name") or metadata["name"])


def _dependency_index_url(
    name: str,
    metadata: dict[str, Any],
    *,
    pip_index_url: str = "",
    torch_index_url: str = "",
) -> str:
    """Resolve the pip index URL for a dependency.

    Torch/torchaudio use a dedicated PyTorch wheel index. If the user
    supplies ``torch_index_url`` (or ``QWENPAW_TORCH_INDEX_URL``) it wins;
    a generic PyPI mirror is intentionally *not* used for CUDA torch wheels.
    All other packages use ``pip_index_url`` / ``QWENPAW_PIP_INDEX_URL``.
    """
    if name == "torch_cuda":
        return (
            torch_index_url.strip()
            or os.environ.get("QWENPAW_TORCH_INDEX_URL", "").strip()
            or str(metadata.get("index_url") or "")
        )
    return (
        pip_index_url.strip()
        or os.environ.get("QWENPAW_PIP_INDEX_URL", "").strip()
        or str(metadata.get("index_url") or "")
    )


def _forget_imported_dependency(name: str) -> None:
    """Drop *name* from ``sys.modules`` after install/uninstall.

    ``importlib.invalidate_caches`` only clears finder caches; a module that
    was imported before pip replaced its files would otherwise keep the old
    code alive and make ``dependency_installed`` report a stale result.

    Torch/torchaudio are special: they cannot be reliably re-imported in the
    same process after being removed from ``sys.modules`` on Windows
    (``RuntimeError: function '_has_torch_function' already has a docstring``).
    For ``torch_cuda`` we therefore only invalidate finder caches and rely on
    package metadata for status; the user must restart QwenPaw to load the new
    CUDA build.
    """
    if name == "torch_cuda":
        importlib.invalidate_caches()
        return
    metadata = _dependency_metadata(name)
    prefixes = [name]
    if metadata.get("module"):
        prefixes.append(str(metadata["module"]))
    for key in list(sys.modules):
        if key in prefixes or any(
            key.startswith(f"{prefix}.") for prefix in prefixes
        ):
            sys.modules.pop(key, None)
    importlib.invalidate_caches()


def dependency_installed(name: str) -> bool:
    """Check one dependency id against the current interpreter."""
    metadata = _dependency_metadata(name)
    if name == "torch_cuda":
        try:
            torch_version = importlib.metadata.version("torch")
        except importlib.metadata.PackageNotFoundError:
            return False
        return "+cu" in torch_version
    module = metadata.get("module")
    if module and module not in sys.modules:
        return importlib.util.find_spec(module) is not None
    # The module is already imported in this process, so importlib cannot
    # tell whether the files were removed. Fall back to package metadata,
    # which reflects uninstalls performed by the dependency installer.
    try:
        return importlib.metadata.distribution(_pip_name(metadata)) is not None
    except importlib.metadata.PackageNotFoundError:
        return False


def _torch_version_parts(version: str) -> tuple[str, str]:
    """Split a torch/torchaudio version into ``(base, build_tag)``.

    ``2.6.0+cu124`` -> ``("2.6.0", "cu124")``; ``2.6.0`` -> ``("2.6.0", "")``.
    """
    base, separator, build = version.partition("+")
    return base.strip(), build.strip() if separator else ""


def _torchaudio_matches_torch() -> bool:
    """Return True when torchaudio's native ABI version matches torch.

    The check intentionally uses package metadata instead of importing
    torch/torchaudio: on a mismatched Windows installation the DLL import can
    fail (or emit a fatal loader message) before Python can handle it, and
    torch cannot always be re-imported cleanly after a pip replacement.
    """
    try:
        torch_version = importlib.metadata.version("torch")
        torchaudio_version = importlib.metadata.version("torchaudio")
    except Exception:
        # A broken torchaudio install is exactly the situation we need to
        # repair, so treat any failure as "not matching".
        return False
    torch_base, torch_build = _torch_version_parts(torch_version)
    audio_base, audio_build = _torch_version_parts(torchaudio_version)
    if torch_base != audio_base:
        return False
    # A CUDA torch wheel should be paired with the CUDA torchaudio wheel.
    if torch_build and torch_build != audio_build:
        return False
    return True


def _torchaudio_pin() -> tuple[str, str] | None:
    """Return ``(spec, index_url)`` matching the installed torch build."""
    try:
        torch_version = importlib.metadata.version("torch")
    except Exception:
        return None
    torch_base, torch_build = _torch_version_parts(torch_version)
    if not torch_base:
        return None
    if torch_build.startswith("cu"):
        if not torch_build[2:].isdigit():
            return None
        return (
            f"torchaudio=={torch_base}+{torch_build}",
            f"https://download.pytorch.org/whl/{torch_build}",
        )
    if torch_build:
        return f"torchaudio=={torch_base}+{torch_build}", ""
    return f"torchaudio=={torch_base}", ""


def _maybe_align_torchaudio(
    name: str,
    *,
    pip_index_url: str = "",
    torch_index_url: str = "",
) -> None:
    """Keep torchaudio ABI-compatible with torch after torch/qwen_tts installs.

    Qwen-TTS imports ``torchaudio`` at package import time.  ``qwen-tts``
    declares ``torchaudio`` without a version, so a plain install can grab a
    build for a different PyTorch major/minor version (e.g. torchaudio 2.11
    while the CUDA PyTorch wheel is 2.6.0+cu124). On Windows this commonly
    surfaces as ``OSError: [WinError 127]`` when loading ``_torchaudio.pyd``.
    """
    if name not in {"qwen_tts", "torch_cuda"}:
        return
    if name == "torch_cuda" and not dependency_installed("qwen_tts"):
        # No qwen_tts installed yet; the qwen_tts install step will align
        # torchaudio once torch is available.
        return
    if _torchaudio_matches_torch():
        return
    pin = _torchaudio_pin()
    if pin is None:
        return
    spec, index_url = pin
    torch_override = (
        torch_index_url.strip()
        or os.environ.get("QWENPAW_TORCH_INDEX_URL", "").strip()
    )
    if torch_override:
        index_url = torch_override
    elif not index_url:
        index_url = (
            pip_index_url.strip()
            or os.environ.get("QWENPAW_PIP_INDEX_URL", "").strip()
        )

    command: list[str]
    if getattr(sys, "frozen", False):
        from qwenpaw.plugins.loader import (
            _desktop_python,
            _ensure_plugin_site_on_path,
            _plugin_site_dir,
        )

        python = _desktop_python()
        if python is None:
            return
        command = [
            python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "--target",
            str(_plugin_site_dir()),
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
        ]
    if index_url:
        command += ["--index-url", index_url]
    command.append(spec)
    if getattr(sys, "frozen", False):
        _run_dependency_install(command)
        _ensure_plugin_site_on_path()
        return
    result = _run_dependency_install(command, check=False)
    if result.returncode != 0:
        output = "\n".join(
            part for part in (result.stdout, result.stderr) if part
        )
        if "No module named pip" not in output:
            raise RuntimeError(_dependency_install_error(output))
        uv = shutil.which("uv")
        if not uv:
            raise RuntimeError(
                "pip is unavailable and uv was not found; install qwenpaw[sip] "
                "manually.",
            )
        uv_command = [uv, "pip", "install", "--python", sys.executable]
        if index_url:
            uv_command += ["--index-url", index_url]
        uv_command.append(spec)
        _run_dependency_install(uv_command)


def _install_one_voice_dependency(
    name: str,
    *,
    pip_index_url: str = "",
    torch_index_url: str = "",
) -> None:
    metadata = _dependency_metadata(name)
    spec = str(metadata["spec"])
    index_url = _dependency_index_url(
        name,
        metadata,
        pip_index_url=pip_index_url,
        torch_index_url=torch_index_url,
    )

    # A frozen desktop executable is not a Python interpreter. Reuse the
    # bundled runtime and writable site directory already used by plugins.
    if getattr(sys, "frozen", False):
        from qwenpaw.plugins.loader import (
            _desktop_python,
            _ensure_plugin_site_on_path,
            _plugin_site_dir,
        )

        python = _desktop_python()
        if python is None:
            raise RuntimeError(
                "Bundled Python runtime is unavailable; reinstall QwenPaw "
                "Desktop or install qwenpaw[sip] manually.",
            )
        command = [
            python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "--target",
            str(_plugin_site_dir()),
        ]
        if metadata.get("force_reinstall"):
            command.append("--force-reinstall")
        if index_url:
            command += ["--index-url", index_url]
        command.append(spec)
        _run_dependency_install(command)
        _ensure_plugin_site_on_path()
        _maybe_align_torchaudio(
            name,
            pip_index_url=pip_index_url,
            torch_index_url=torch_index_url,
        )
        return

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
    ]
    if metadata.get("force_reinstall"):
        command.append("--force-reinstall")
    if index_url:
        command += ["--index-url", index_url]
    command.append(spec)
    result = _run_dependency_install(command, check=False)
    if result.returncode == 0:
        _maybe_align_torchaudio(
            name,
            pip_index_url=pip_index_url,
            torch_index_url=torch_index_url,
        )
        return

    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if "No module named pip" not in output:
        raise RuntimeError(_dependency_install_error(output))
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError(
            "pip is unavailable and uv was not found; install qwenpaw[sip] "
            "manually.",
        )
    uv_command = [uv, "pip", "install", "--python", sys.executable]
    if metadata.get("force_reinstall"):
        uv_command.append("--reinstall")
    if index_url:
        uv_command += ["--index-url", index_url]
    uv_command.append(spec)
    _run_dependency_install(uv_command)
    _maybe_align_torchaudio(
        name,
        pip_index_url=pip_index_url,
        torch_index_url=torch_index_url,
    )


def _install_voice_dependencies(
    packages: list[str],
    *,
    pip_index_url: str = "",
    torch_index_url: str = "",
) -> None:
    """Install selected dependency ids from the fixed allowlist."""
    for name in packages:
        _install_one_voice_dependency(
            name,
            pip_index_url=pip_index_url,
            torch_index_url=torch_index_url,
        )


def _uninstall_one_voice_dependency(name: str) -> None:
    """Uninstall one dependency id (best effort, reports failures)."""
    metadata = _dependency_metadata(name)
    target = _pip_name(metadata)

    if getattr(sys, "frozen", False):
        _uninstall_from_plugin_site(target)
        _forget_imported_dependency(name)
        return

    command = [
        sys.executable,
        "-m",
        "pip",
        "uninstall",
        "-y",
        "--disable-pip-version-check",
        target,
    ]
    result = _run_dependency_install(command, check=False)
    if result.returncode != 0:
        output = "\n".join(
            part for part in (result.stdout, result.stderr) if part
        )
        if "No module named pip" in output:
            uv = shutil.which("uv")
            if uv:
                _run_dependency_install(
                    [
                        uv,
                        "pip",
                        "uninstall",
                        "--python",
                        sys.executable,
                        target,
                    ],
                )
                _forget_imported_dependency(name)
                return
        raise RuntimeError(
            "Dependency uninstall failed for "
            f"{metadata['name']}"
            + (f": {output[-2000:]}" if output else ""),
        )
    _forget_imported_dependency(name)


def _uninstall_from_plugin_site(pip_name: str) -> None:
    """Remove a ``pip install --target`` distribution from the plugin site.

    A frozen desktop executable cannot run ``pip uninstall`` against the
    shared plugin site directory (pip has no ``--target`` mode for
    uninstall), so remove the recorded files directly instead.
    """
    from qwenpaw.plugins.loader import _plugin_site_dir

    site_dir = _plugin_site_dir().resolve()
    try:
        # pylint: disable-next=unexpected-keyword-arg
        distribution = importlib.metadata.distribution(
            pip_name,
            path=[str(site_dir)],
        )
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"{pip_name} was not found under the bundled plugin site: "
            f"{site_dir}"
        ) from exc
    for entry in distribution.files or []:
        target = (site_dir / entry.as_posix()).resolve()
        if site_dir not in target.parents:
            # Never touch anything outside the plugin site directory.
            continue
        try:
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            else:
                target.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning(
                "Could not remove %s during dependency uninstall: %s",
                target,
                exc,
            )
    # pylint: disable-next=protected-access
    dist_info = distribution._path  # noqa: SLF001 - stable metadata API
    if isinstance(dist_info, Path):
        try:
            shutil.rmtree(dist_info, ignore_errors=True)
        except OSError as exc:
            logger.warning(
                "Could not remove %s during dependency uninstall: %s",
                dist_info,
                exc,
            )


def uninstall_voice_dependencies(packages: list[str]) -> None:
    """Uninstall the requested dependency ids sequentially."""
    for name in packages:
        _uninstall_one_voice_dependency(name)


def _run_dependency_install(
    command: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_DEPENDENCY_INSTALL_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "Local Voice dependency installation timed out after 15 minutes",
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"Could not start dependency installer: {exc}",
        ) from exc
    if check and result.returncode != 0:
        output = "\n".join(
            part for part in (result.stdout, result.stderr) if part
        )
        raise RuntimeError(_dependency_install_error(output))
    return result


def _dependency_install_error(output: str) -> str:
    detail = output.strip()[-4000:]
    return "Local Voice dependency installation failed" + (
        f": {detail}" if detail else ""
    )


def _asset_path(asset: VoiceAssetName, cfg: LocalVoiceChannelConfig) -> Path:
    if asset == "zipformer":
        return resolve_zipformer_model_dir(cfg.zipformer_model_dir)
    if asset == "kws":
        return resolve_kws_model_dir(cfg.kws_model_dir)
    if asset == "qwen3":
        if cfg.qwen3_model_dir.strip():
            return Path(cfg.qwen3_model_dir).expanduser()
        return MODELS_DIR / _QWEN3_LOCAL_MODEL_DIR_NAME
    variant = "int8" if asset == "kokoro_int8" else "float32"
    configured_variant = cfg.kokoro_model_variant
    configured_dir = cfg.kokoro_model_dir if variant == configured_variant else ""
    return resolve_kokoro_model_dir(configured_dir, variant)


def _dir_size(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(
        entry.stat().st_size
        for entry in path.rglob("*")
        if entry.is_file()
    )


def voice_model_statuses(cfg: LocalVoiceChannelConfig) -> list[VoiceModelStatus]:
    """Return resolved paths and completeness for all offline voice models."""
    result: list[VoiceModelStatus] = []
    for asset, metadata in _ASSETS.items():
        path = _asset_path(asset, cfg)
        missing = _missing_model_files(asset, path, metadata["required"])
        result.append(
            VoiceModelStatus(
                id=asset,  # type: ignore[arg-type]
                name=metadata["name"],
                path=str(path),
                installed=path.is_dir(),
                complete=path.is_dir() and not missing,
                size_bytes=_dir_size(path),
                missing_files=missing,
                download_url=metadata["url"],
            )
        )
    return result


def _missing_model_files(
    asset: VoiceAssetName,
    path: Path,
    required: tuple[str, ...],
) -> list[str]:
    """Check the flexible ONNX names used by different sherpa releases."""
    missing = [name for name in required if not (path / name).exists()]
    if asset in ("zipformer", "kws"):
        for kind in ("encoder", "decoder", "joiner"):
            if not any(path.glob(f"{kind}*.onnx")):
                missing.append(f"{kind}*.onnx")
    elif asset in ("kokoro", "kokoro_int8"):
        expected = (
            "model.int8.onnx" if asset == "kokoro_int8" else "model.onnx"
        )
        if not (path / expected).is_file():
            missing.append(expected)
    elif asset == "qwen3":
        if not any(path.glob("*.safetensors")):
            missing.append("model weights (*.safetensors)")
        has_tokenizer = (
            (path / "tokenizer.json").is_file()
            or (path / "tokenizer_config.json").is_file()
        )
        if not has_tokenizer:
            missing.append("tokenizer.json or tokenizer_config.json")
    return missing


class VoiceModelDownloadManager:
    """Single-download manager for official model archives and HF snapshots."""

    def __init__(self) -> None:
        self.status = VoiceDownloadStatus()
        self._task: asyncio.Task[None] | None = None

    def start(self, asset: VoiceAssetName, cfg: LocalVoiceChannelConfig) -> None:
        if self._task and not self._task.done():
            raise RuntimeError("A voice model download is already in progress")
        target = _asset_path(asset, cfg)
        if target.exists():
            raise ValueError(
                f"Target model directory already exists: {target}. "
                "Remove or choose a different directory before downloading.",
            )
        self.status = VoiceDownloadStatus(status="downloading", asset=asset)
        self._task = asyncio.create_task(self._download(asset, target))

    async def _download(self, asset: VoiceAssetName, target: Path) -> None:
        try:
            await asyncio.to_thread(self._download_blocking, asset, target)
            self.status.status = "completed"
        except Exception as exc:  # keep error available to polling UI
            logger.exception("Voice model download failed: %s", asset)
            self.status.status = "failed"
            self.status.error = str(exc)

    def _download_blocking(self, asset: VoiceAssetName, target: Path) -> None:
        metadata = _ASSETS[asset]
        if metadata.get("kind") == "huggingface":
            self._download_huggingface(metadata["url"], target)
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        archive = target.parent / f".{asset}-{uuid.uuid4().hex}.tar.bz2"
        extraction_dir = Path(
            tempfile.mkdtemp(prefix=f".{asset}-extract-", dir=target.parent),
        )
        try:
            request = urllib.request.Request(
                metadata["url"], headers={"User-Agent": "QwenPaw/voice-models"},
            )
            with (
                urllib.request.urlopen(request, timeout=30) as response,
                archive.open("wb") as output,
            ):
                raw_size = response.headers.get("Content-Length")
                self.status.total_bytes = int(raw_size) if raw_size else None
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    self.status.downloaded_bytes += len(chunk)

            with tarfile.open(archive, "r:bz2") as package:
                root = extraction_dir.resolve()
                for member in package.getmembers():
                    if member.issym() or member.islnk():
                        raise ValueError("Model archive contains a link")
                    member_path = (extraction_dir / member.name).resolve()
                    if root not in member_path.parents and member_path != root:
                        raise ValueError("Model archive contains an unsafe path")
                package.extractall(extraction_dir)

            extracted = extraction_dir / metadata["name"]
            if not extracted.is_dir():
                raise ValueError("Downloaded archive has an unexpected layout")
            shutil.move(str(extracted), str(target))
        finally:
            archive.unlink(missing_ok=True)
            shutil.rmtree(extraction_dir, ignore_errors=True)

    def _download_huggingface(self, repo_id: str, target: Path) -> None:
        """Download a Hugging Face model snapshot into *target*."""
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ValueError(
                "huggingface-hub is not installed. "
                "Install qwenpaw[sip] or use the Local Voice dependency "
                "installer.",
            ) from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target),
            local_dir_use_symlinks=False,
        )
        self.status.downloaded_bytes = _dir_size(target)
        self.status.total_bytes = self.status.downloaded_bytes or None


def local_kws_supported(provider: str) -> bool:
    """True when *provider* is the local Zipformer engine with a KWS gate."""
    try:
        return normalize_stt_provider(provider) == "sherpa_zipformer"
    except ValueError:
        return False


class LocalVoiceTestManager:
    """Runs one microphone/speaker diagnostic at a time without an Agent."""

    def __init__(self) -> None:
        self.status = VoiceTestStatus()
        self._task: asyncio.Task[None] | None = None

    def start(
        self,
        test: VoiceTestName,
        cfg: LocalVoiceChannelConfig,
        text: str = "本地语音测试成功。",
    ) -> None:
        if self._task and not self._task.done():
            raise RuntimeError("A Local Voice test is already running")
        self.status = VoiceTestStatus(status="running", test=test)
        self._task = asyncio.create_task(self._run(test, cfg, text))

    async def _run(
        self,
        test: VoiceTestName,
        cfg: LocalVoiceChannelConfig,
        text: str,
    ) -> None:
        try:
            if test == "tts":
                await self._test_tts(cfg, text)
                self.status.message = "TTS audio played on the selected output device."
            else:
                await self._test_microphone(test, cfg)
            self.status.status = "passed"
        except TimeoutError:
            self.status.status = "timed_out"
            self.status.message = "No speech was detected before the test timed out."
        except Exception as exc:
            logger.exception("Local Voice %s test failed", test)
            self.status.status = "failed"
            self.status.message = str(exc)

    async def _test_tts(self, cfg: LocalVoiceChannelConfig, text: str) -> None:
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as exc:
            raise ValueError(
                "TTS test requires the Local Voice dependencies. "
                "Install qwenpaw[sip].",
            ) from exc
        pcm = bytearray()
        async for chunk in synthesize_tts_stream(
            cfg.tts_provider,
            text.strip() or "本地语音测试成功。",
            cfg.tts_voice,
            cfg.dashscope_api_key,
            sample_rate=cfg.tts_sample_rate,
            kokoro_model_dir=cfg.kokoro_model_dir,
            kokoro_model_variant=cfg.kokoro_model_variant,
            kokoro_num_threads=cfg.kokoro_num_threads,
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
            pcm.extend(chunk)
        if not pcm:
            raise ValueError("TTS provider returned no audio")
        samples = np.frombuffer(bytes(pcm), dtype=np.int16)
        await asyncio.to_thread(
            sd.play,
            samples,
            samplerate=cfg.tts_sample_rate,
            device=cfg.output_device,
            blocking=True,
        )

    async def _test_microphone(  # pylint: disable=too-many-statements
        self,
        test: VoiceTestName,
        cfg: LocalVoiceChannelConfig,
    ) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise ValueError(
                "Microphone tests require the Local Voice dependencies. "
                "Install qwenpaw[sip].",
            ) from exc
        from qwenpaw.app.channels.sip.stt_tts import create_stt_engine

        uses_local_kws = local_kws_supported(cfg.stt_provider)
        if test == "wake_word" and not uses_local_kws:
            # API STT providers have no local KWS gate; the microphone test
            # degrades to a regular ASR test and reports that in the UI.
            self.status.message = (
                "The selected ASR provider has no local wake-word gate; "
                "listening for any speech instead."
            )

        detected: asyncio.Future[str] = (
            asyncio.get_running_loop().create_future()
        )
        engine = create_stt_engine(
            cfg.stt_provider,
            cfg.language,
            cfg.dashscope_api_key,
            zipformer_model_dir=cfg.zipformer_model_dir,
            zipformer_num_threads=cfg.zipformer_num_threads,
            wake_word_enabled=test == "wake_word" and uses_local_kws,
            kws_model_dir=cfg.kws_model_dir,
            kws_keywords_file=cfg.kws_keywords_file,
            kws_num_threads=cfg.kws_num_threads,
            kws_pre_roll_seconds=cfg.kws_pre_roll_seconds,
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

        def _set_result(value: str, *, keyword: bool = False) -> None:
            if not detected.done():
                detected.set_result(value)
            if keyword:
                self.status.keyword = value
            else:
                self.status.transcript = value

        async def _on_transcript(value: str) -> None:
            _set_result(value)

        engine.on_transcript = _on_transcript
        engine.on_wake_word = lambda value: _set_result(value, keyword=True)
        samples: queue.Queue[bytes] = queue.Queue(maxsize=128)
        input_rate = int(cfg.input_sample_rate)
        resample_state: Any = None

        def _callback(indata, frames, time_info, status) -> None:
            del frames, time_info, status
            try:
                samples.put_nowait(indata.tobytes())
            except queue.Full:
                pass

        await engine.start()
        stream = None
        try:
            try:
                stream = sd.InputStream(
                    device=cfg.input_device,
                    channels=1,
                    samplerate=input_rate,
                    dtype="int16",
                    blocksize=int(input_rate * 0.02),
                    callback=_callback,
                )
                stream.start()
            except Exception:
                input_rate = 48000
                stream = sd.InputStream(
                    device=cfg.input_device,
                    channels=1,
                    samplerate=input_rate,
                    dtype="int16",
                    blocksize=int(input_rate * 0.02),
                    callback=_callback,
                )
                stream.start()
            deadline = time.monotonic() + (12 if test == "wake_word" else 10)
            while not detected.done() and time.monotonic() < deadline:
                try:
                    data = await asyncio.to_thread(samples.get, True, 0.1)
                except queue.Empty:
                    continue
                if input_rate != 16000:
                    data, resample_state = audioop.ratecv(
                        data,
                        2,
                        1,
                        input_rate,
                        16000,
                        resample_state,
                    )
                await engine.feed_audio(data)
            if not detected.done():
                raise TimeoutError
        finally:
            if stream is not None:
                await asyncio.to_thread(stream.stop)
                await asyncio.to_thread(stream.close)
            await engine.stop()


def optional_voice_dependencies() -> list[str]:
    """Ids of dependencies that are currently missing."""
    return [
        name for name in _VOICE_DEPENDENCIES if not dependency_installed(name)
    ]


def required_voice_dependencies(cfg: LocalVoiceChannelConfig) -> set[str]:
    """Dependency ids required by the current Local Voice settings."""
    required = {"sounddevice", "numpy"}
    try:
        stt_provider = normalize_stt_provider(cfg.stt_provider)
    except ValueError:
        stt_provider = ""
    tts_provider = (cfg.tts_provider or "").strip().lower()

    if stt_provider == "sherpa_zipformer":
        required.add("sherpa_onnx")
        required.add("pypinyin")
    if tts_provider == "edge_tts":
        required.add("edge_tts")
    if tts_provider == "kokoro":
        required.add("sherpa_onnx")
    if tts_provider == "qwen3" and cfg.qwen3_backend == "local":
        required.add("qwen_tts")
        required.add("huggingface_hub")
        if (cfg.qwen3_device or "cpu").strip().lower().startswith("cuda"):
            required.add("torch_cuda")
    return required


def voice_dependency_items(
    cfg: LocalVoiceChannelConfig,
) -> list[VoiceDependencyItem]:
    """All installable dependencies with installed/required flags."""
    required = required_voice_dependencies(cfg)
    return [
        VoiceDependencyItem(
            id=name,
            name=str(metadata["name"]),
            spec=str(metadata["spec"]),
            description=str(metadata.get("description") or ""),
            installed=dependency_installed(name),
            required=name in required,
        )
        for name, metadata in _VOICE_DEPENDENCIES.items()
    ]


def list_qwen3_inference_devices() -> list[VoiceDeviceOption]:
    """Devices selectable for local Qwen3-TTS inference."""
    options = [
        VoiceDeviceOption(
            value="cpu",
            label="CPU",
            available=True,
            description="Always available; slowest but works everywhere.",
        )
    ]
    try:
        import torch
    except ImportError:
        options.append(
            VoiceDeviceOption(
                value="cuda",
                label="CUDA (requires torch_cuda dependency)",
                available=False,
                description=(
                    "Install PyTorch (CUDA 12.8) from the dependency list."
                ),
            )
        )
        return options
    try:
        cuda_available = bool(torch.cuda.is_available())
        gpu_count = (
            int(torch.cuda.device_count()) if cuda_available else 0
        )
    except Exception:
        logger.warning(
            "Could not query torch CUDA devices",
            exc_info=True,
        )
        cuda_available = False
        gpu_count = 0
    if cuda_available:
        for index in range(gpu_count):
            try:
                name = torch.cuda.get_device_name(index)
            except Exception:
                name = f"GPU {index}"
            options.append(
                VoiceDeviceOption(
                    value=f"cuda:{index}",
                    label=f"CUDA {index} · {name}",
                    available=True,
                    description=(
                        "Select a specific GPU; never falls back to "
                        "integrated graphics implicitly."
                    ),
                )
            )
    else:
        options.append(
            VoiceDeviceOption(
                value="cuda",
                label="CUDA (unavailable)",
                available=False,
                description=(
                    "torch has no CUDA support. Install the torch_cuda "
                    "dependency or keep cpu."
                ),
            )
        )
    try:
        mps_available = (
            hasattr(torch.backends, "mps")
            and getattr(torch.backends.mps, "is_available", lambda: False)()
        )
    except Exception:
        mps_available = False
    if mps_available:
        options.append(
            VoiceDeviceOption(
                value="mps",
                label="Apple MPS",
                available=True,
                description="Apple Metal GPU acceleration on macOS.",
            )
        )
    return options
