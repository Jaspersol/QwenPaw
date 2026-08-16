# -*- coding: utf-8 -*-
# pylint: disable=protected-access
"""Tests for the failover-chain build cache in model_factory."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from qwenpaw.agents import model_factory


def _fake_chain() -> SimpleNamespace:
    return SimpleNamespace(formatter="fmt", mark="chain")


def _fake_build(calls: list) -> object:
    def build(_manager, _fallback_models, **_kwargs):
        calls.append(1)
        chain = _fake_chain()
        return chain, "fmt"

    return build


@pytest.fixture(autouse=True)
def _clear_cache():
    model_factory._fallback_chain_cache.clear()
    yield
    model_factory._fallback_chain_cache.clear()


def _call(manager, slots):
    return model_factory._get_or_build_fallback_chain(
        manager,
        slots,
        retry_config=None,
        rate_limit_config=None,
        compact_threshold=None,
    )


def test_chain_cache_reuses_build_for_same_key(monkeypatch):
    calls: list = []
    monkeypatch.setattr(
        model_factory,
        "_build_fallback_chain",
        _fake_build(calls),
    )

    manager = SimpleNamespace(get_config_revision=lambda: 0)
    slots = [
        SimpleNamespace(provider_id="p", model="m"),
        SimpleNamespace(provider_id="q", model="n"),
    ]

    chain1, _fmt1 = _call(manager, slots)
    chain2, _fmt2 = _call(manager, slots)

    assert len(calls) == 1
    assert chain1 is chain2


def test_chain_cache_rebuilds_on_revision_change(monkeypatch):
    calls: list = []
    monkeypatch.setattr(
        model_factory,
        "_build_fallback_chain",
        _fake_build(calls),
    )

    manager = SimpleNamespace(get_config_revision=lambda: 0)
    slots = [SimpleNamespace(provider_id="p", model="m")]
    _call(manager, slots)

    # A provider config change bumps the revision -> rebuild.
    manager = SimpleNamespace(get_config_revision=lambda: 1)
    _call(manager, slots)
    assert len(calls) == 2


def test_chain_cache_rebuilds_on_slot_change(monkeypatch):
    calls: list = []
    monkeypatch.setattr(
        model_factory,
        "_build_fallback_chain",
        _fake_build(calls),
    )

    manager = SimpleNamespace(get_config_revision=lambda: 0)
    _call(manager, [SimpleNamespace(provider_id="p", model="m")])
    _call(manager, [SimpleNamespace(provider_id="p", model="m2")])
    assert len(calls) == 2


def test_chain_cache_rebuilds_when_agent_config_differs(monkeypatch):
    """Retry/rate-limit/compaction settings are per-agent and must be part
    of the cache key, otherwise one agent's settings leak to others."""
    calls: list = []
    monkeypatch.setattr(
        model_factory,
        "_build_fallback_chain",
        _fake_build(calls),
    )

    manager = SimpleNamespace(get_config_revision=lambda: 0)
    slots = [SimpleNamespace(provider_id="p", model="m")]

    model_factory._get_or_build_fallback_chain(
        manager,
        slots,
        retry_config="rc-a",
        rate_limit_config=None,
        compact_threshold=None,
    )
    # Different retry config -> rebuild.
    model_factory._get_or_build_fallback_chain(
        manager,
        slots,
        retry_config="rc-b",
        rate_limit_config=None,
        compact_threshold=None,
    )
    assert len(calls) == 2

    # Different compaction threshold -> rebuild again.
    model_factory._get_or_build_fallback_chain(
        manager,
        slots,
        retry_config="rc-b",
        rate_limit_config=None,
        compact_threshold=0.8,
    )
    assert len(calls) == 3
