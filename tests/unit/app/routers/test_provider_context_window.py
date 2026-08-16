# -*- coding: utf-8 -*-
"""Tests for active-model context-window metadata."""

from types import SimpleNamespace

from qwenpaw.app.routers.providers import _active_models_info
from qwenpaw.config.config import ModelSlotConfig
from qwenpaw.providers import fallback_chat_model


def _fallback_manager(chain):
    context_sizes = {
        slot.model: 1_000_000 + 100 * index
        for index, slot in enumerate(chain)
    }
    context_sizes["manual-model"] = 2_000_000
    provider = SimpleNamespace(
        get_context_size=lambda model_id: context_sizes.get(model_id, 0),
    )
    return SimpleNamespace(
        get_provider=lambda _provider_id: provider,
        get_fallback_models=lambda: chain,
    )


def test_active_models_info_uses_runtime_context_resolution():
    provider = SimpleNamespace(get_context_size=lambda _model_id: 1_000_000)
    manager = SimpleNamespace(get_provider=lambda _provider_id: provider)
    slot = ModelSlotConfig(provider_id="dashscope", model="qwen3.7-max")

    info = _active_models_info(manager, slot)

    assert info.active_llm == slot
    assert info.runtime_active_llm == slot
    assert info.effective_max_input_length == 1_000_000


def test_agent_model_outside_fallback_chain_is_runtime_model(monkeypatch):
    """An in-session switch outside the chain bypasses the chain."""
    chain = [
        ModelSlotConfig(provider_id="openai", model="chain-a"),
        ModelSlotConfig(provider_id="openai", model="chain-b"),
    ]
    manager = _fallback_manager(chain)
    monkeypatch.setattr(
        fallback_chat_model,
        "get_fallback_current_index",
        lambda: 0,
    )
    outside = ModelSlotConfig(provider_id="openai", model="manual-model")

    info = _active_models_info(
        manager,
        outside,
        configured_by_agent=True,
    )

    assert info.active_llm == outside
    assert info.runtime_active_llm == outside
    assert info.effective_max_input_length == 2_000_000


def test_agent_model_inside_fallback_chain_reports_pointer_slot(monkeypatch):
    """A chain member resolves to the slot the failover pointer selects."""
    chain = [
        ModelSlotConfig(provider_id="openai", model="chain-a"),
        ModelSlotConfig(provider_id="openai", model="chain-b"),
    ]
    manager = _fallback_manager(chain)
    monkeypatch.setattr(
        fallback_chat_model,
        "get_fallback_current_index",
        lambda: 1,
    )

    info = _active_models_info(
        manager,
        chain[0],
        configured_by_agent=True,
    )

    assert info.active_llm == chain[0]
    assert info.runtime_active_llm == chain[1]
    assert info.effective_max_input_length == 1_000_100


def test_global_model_outside_fallback_chain_still_uses_chain(monkeypatch):
    """Only an explicit agent-level outside model bypasses the chain."""
    chain = [
        ModelSlotConfig(provider_id="openai", model="chain-a"),
        ModelSlotConfig(provider_id="openai", model="chain-b"),
    ]
    manager = _fallback_manager(chain)
    monkeypatch.setattr(
        fallback_chat_model,
        "get_fallback_current_index",
        lambda: 0,
    )
    outside = ModelSlotConfig(provider_id="openai", model="manual-model")

    info = _active_models_info(
        manager,
        outside,
        configured_by_agent=False,
    )

    assert info.active_llm == outside
    assert info.runtime_active_llm == chain[0]
    assert info.effective_max_input_length == 1_000_000
