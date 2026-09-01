# -*- coding: utf-8 -*-
"""Local voice context injection.

Injects only the voice-mode instructions needed for each wake segment.
Previous-session summaries and raw history are intentionally not included,
so the assistant starts each wake-up conversation fresh.
"""

from __future__ import annotations

import logging

from ..base import LifecycleHook
from ...runtime.hooks import HookContext, HookResult
from ...runtime.phases import Phase

logger = logging.getLogger(__name__)


class LocalVoiceContextHook(LifecycleHook):
    """Inject voice-mode instructions without cross-segment history."""

    phase = Phase.PRE_AGENT_BUILD
    name = "local_voice_context"
    priority = 20

    async def run(self, ctx: HookContext) -> HookResult:
        request = getattr(ctx, "request", None)
        channel = getattr(request, "channel", None) if request else None
        if channel != "local_voice":
            return HookResult()

        channel_meta = getattr(request, "channel_meta", None) or {}
        if not isinstance(channel_meta, dict):
            channel_meta = {}

        parts: list[str] = []

        wake_word = str(
            channel_meta.get("voice_wake_word") or "语音助手",
        ).strip()
        parts.append(
            f"你是语音助手“{wake_word}”",
        )
        parts.append(
            "当前是语音对话模式，回复请使用口语化、简洁、适合直接语音播报"
            "的语言；尽量简短，避免 Markdown、代码块、列表和长段落。",
        )
        parts.append(
            "语音播报口径：数字、日期、时间、金额、单位一律转成口语读法"
            "（例如年月日读作“某年某月某日”，“3.5”读作"
            "“三点五”）；不要输出网址、邮箱、代码或任何需要复制的符号。"
            "单次回复控制在 2~3 句话内，超过则简短分段。",
        )
        parts.append(
            "系统已经在你回答前播报了“收到，正在处理中”等固定提示音，"
            "你无需再次说“好的”“收到”之类的应答，直接切入正题即可。",
        )

        ctx.inject_context(
            "\n\n".join(parts),
            priority=20,
            source="local_voice_segment",
        )
        return HookResult()


__all__ = ["LocalVoiceContextHook"]
