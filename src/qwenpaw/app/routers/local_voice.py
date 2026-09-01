# -*- coding: utf-8 -*-
"""Console APIs for offline Local Voice assets and diagnostics."""
from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from qwenpaw.app.channels.local_voice.tools import (
    LocalVoiceTestManager,
    VoiceAssetName,
    VoiceDependencyInstallManager,
    VoiceDependencyInstallStatus,
    VoiceDependencyStatus,
    VoiceDeviceOption,
    VoiceDownloadStatus,
    VoiceModelDownloadManager,
    VoiceModelStatus,
    VoiceTestName,
    VoiceTestStatus,
    list_qwen3_inference_devices,
    optional_voice_dependencies,
    uninstall_voice_dependencies,
    voice_dependency_items,
    voice_model_statuses,
)
from qwenpaw.app.channels.sip.stt_tts import clone_qwen3_voice
from qwenpaw.config.config import LocalVoiceChannelConfig

router = APIRouter(prefix="/local-voice", tags=["local-voice"])


class LocalVoiceConfigBody(BaseModel):
    """Optional unsaved drawer values used for status and test actions."""

    config: dict = Field(default_factory=dict)


class StartDownloadBody(LocalVoiceConfigBody):
    asset: VoiceAssetName


class StartTestBody(LocalVoiceConfigBody):
    test: VoiceTestName
    text: str = Field(default="本地语音测试成功。", max_length=500)


class DependencyActionBody(LocalVoiceConfigBody):
    packages: list[str] = Field(default_factory=list)


class VoiceModelsResponse(BaseModel):
    models: list[VoiceModelStatus]
    missing_dependencies: list[str]


class ActionResponse(BaseModel):
    status: Literal["accepted"]
    message: str


class Qwen3VoiceCloneResponse(BaseModel):
    voice_id: str
    message: str = "Voice cloned successfully"


def _config_from_body(body: LocalVoiceConfigBody) -> LocalVoiceChannelConfig:
    # Pydantic filters display-only form values such as ``isBuiltin``.
    return LocalVoiceChannelConfig(**body.config)


def _download_manager(request: Request) -> VoiceModelDownloadManager:
    manager = getattr(request.app.state, "local_voice_model_downloads", None)
    if manager is None:
        manager = VoiceModelDownloadManager()
        request.app.state.local_voice_model_downloads = manager
    return manager


def _test_manager(request: Request) -> LocalVoiceTestManager:
    manager = getattr(request.app.state, "local_voice_test_manager", None)
    if manager is None:
        manager = LocalVoiceTestManager()
        request.app.state.local_voice_test_manager = manager
    return manager


def _dependency_install_manager(
    request: Request,
) -> VoiceDependencyInstallManager:
    manager = getattr(
        request.app.state,
        "local_voice_dependency_installer",
        None,
    )
    if manager is None:
        manager = VoiceDependencyInstallManager()
        request.app.state.local_voice_dependency_installer = manager
    return manager


@router.post("/models/status", response_model=VoiceModelsResponse)
async def get_voice_model_status(body: LocalVoiceConfigBody) -> VoiceModelsResponse:
    """Return model paths after applying configured and environment fallbacks."""
    cfg = _config_from_body(body)
    missing = await asyncio.to_thread(optional_voice_dependencies)
    return VoiceModelsResponse(
        models=voice_model_statuses(cfg),
        missing_dependencies=missing,
    )


@router.post("/models/download", response_model=ActionResponse)
async def start_voice_model_download(
    body: StartDownloadBody,
    request: Request,
) -> ActionResponse:
    """Download one fixed official sherpa-onnx model archive."""
    try:
        _download_manager(request).start(body.asset, _config_from_body(body))
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return ActionResponse(status="accepted", message="Voice model download started")


@router.post("/models/delete", response_model=ActionResponse)
async def delete_voice_model(body: StartDownloadBody) -> ActionResponse:
    """Delete one local ASR/TTS model directory."""
    cfg = _config_from_body(body)
    path = next(
        (
            Path(item.path)
            for item in voice_model_statuses(cfg)
            if item.id == body.asset
        ),
        None,
    )
    if path is None:
        raise HTTPException(
            status_code=404,
            detail="Unknown voice model asset",
        )
    resolved = path.expanduser().resolve()
    if (
        resolved == Path(resolved.anchor)
        or resolved == Path.home().resolve()
        or resolved == Path.cwd().resolve()
        or path.is_symlink()
    ):
        raise HTTPException(
            status_code=400,
            detail="Refusing to delete this path",
        )
    if not resolved.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Model directory does not exist: {path}",
        )
    await asyncio.to_thread(shutil.rmtree, resolved, ignore_errors=False)
    return ActionResponse(
        status="accepted",
        message=f"Deleted {body.asset}: {path}",
    )


@router.get("/models/download", response_model=VoiceDownloadStatus)
async def get_voice_model_download(request: Request) -> VoiceDownloadStatus:
    """Return the current single voice-model download progress."""
    return _download_manager(request).status


@router.post(
    "/dependencies/install",
    response_model=ActionResponse,
)
async def install_local_voice_dependencies(
    body: DependencyActionBody,
    request: Request,
) -> ActionResponse:
    """Start installing selected dependencies (or all missing ones)."""
    try:
        _dependency_install_manager(request).start(
            body.packages or None,
            cfg=_config_from_body(body),
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return ActionResponse(
        status="accepted",
        message="Local Voice dependency installation started",
    )


@router.get(
    "/dependencies/install",
    response_model=VoiceDependencyInstallStatus,
)
async def get_local_voice_dependency_install(
    request: Request,
) -> VoiceDependencyInstallStatus:
    """Return current Local Voice dependency installation progress."""
    return _dependency_install_manager(request).status


@router.post(
    "/dependencies/status",
    response_model=VoiceDependencyStatus,
)
async def get_local_voice_dependency_status(
    body: LocalVoiceConfigBody,
    request: Request,
) -> VoiceDependencyStatus:
    """List every installable dependency and mark required ones."""
    cfg = _config_from_body(body)
    items = await asyncio.to_thread(voice_dependency_items, cfg)
    return VoiceDependencyStatus(
        items=items,
        install=_dependency_install_manager(request).status,
    )


@router.post(
    "/dependencies/uninstall",
    response_model=ActionResponse,
)
async def uninstall_local_voice_dependencies(
    body: DependencyActionBody,
    request: Request,
) -> ActionResponse:
    """Uninstall the selected Local Voice dependencies."""
    if not body.packages:
        raise HTTPException(status_code=400, detail="No packages selected")
    manager = _dependency_install_manager(request)
    if manager.status.status == "installing":
        raise HTTPException(
            status_code=409,
            detail=(
                "Dependency installation is in progress; wait for it to "
                "finish before uninstalling."
            ),
        )
    try:
        await asyncio.to_thread(uninstall_voice_dependencies, body.packages)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ActionResponse(
        status="accepted",
        message=f"Uninstalled dependencies: {', '.join(body.packages)}",
    )


@router.get(
    "/qwen3/devices",
    response_model=list[VoiceDeviceOption],
)
async def get_qwen3_inference_devices() -> list[VoiceDeviceOption]:
    """Return selectable local Qwen3-TTS inference devices."""
    return await asyncio.to_thread(list_qwen3_inference_devices)


@router.post(
    "/qwen3/voice-clone",
    response_model=Qwen3VoiceCloneResponse,
)
async def clone_local_qwen3_voice(
    body: LocalVoiceConfigBody,
) -> Qwen3VoiceCloneResponse:
    """Clone a Qwen3-TTS voice from the configured reference audio.

    Uses the unsaved drawer values (same as model status and tests) so the
    button can run before the channel configuration is saved.
    """
    cfg = _config_from_body(body)
    if (
        (cfg.tts_provider or "").strip().lower() != "qwen3"
        or cfg.qwen3_backend != "api"
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Voice cloning is only available when tts_provider='qwen3' "
                "and qwen3_backend='api'"
            ),
        )
    try:
        reference = (cfg.qwen3_ref_audio or "").strip()
        preferred_name = (
            Path(reference).stem if reference and "://" not in reference
            else "qwenpaw"
        )
        voice_id = await clone_qwen3_voice(
            reference,
            preferred_name=preferred_name,
            api_key=cfg.dashscope_api_key,
            model=cfg.qwen3_model,
        )
    except Exception as exc:  # noqa: BLE001 - surface provider errors to UI
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Qwen3VoiceCloneResponse(voice_id=voice_id)


@router.post("/tests", response_model=ActionResponse)
async def start_local_voice_test(
    body: StartTestBody,
    request: Request,
) -> ActionResponse:
    """Start an ASR/KWS/TTS diagnostic without constructing an Agent request."""
    try:
        _test_manager(request).start(
            body.test,
            _config_from_body(body),
            body.text,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return ActionResponse(status="accepted", message="Local Voice test started")


@router.get("/tests", response_model=VoiceTestStatus)
async def get_local_voice_test(request: Request) -> VoiceTestStatus:
    """Return the current Local Voice test result."""
    return _test_manager(request).status
