# -*- coding: utf-8 -*-
# pylint: disable=protected-access,too-few-public-methods
"""Unit tests for the priority-chain failover wrapper."""

from __future__ import annotations

from typing import Any, AsyncGenerator

import pytest

from qwenpaw.config.config import ModelSlotConfig
from qwenpaw.exceptions import ModelFallbackExhaustedException
from qwenpaw.providers.fallback_chat_model import (
    FAILURE_WINDOW,
    RETIRE_FAILURE_THRESHOLD,
    PersistentFallbackChatModel,
    get_fallback_current_index,
    reset_fallback_state,
    set_fallback_current_index,
)


class _FakeModel:
    """Minimal duck-typed ChatModelBase stand-in."""

    stream = True
    context_size = 32768
    parameters = None
    credential = None
    _provider_id = "unit"

    def __init__(
        self,
        outcomes: list[bool] | None = None,
        *,
        name: str = "fake",
        outcome_fn: Any = None,
    ) -> None:
        # outcomes: per-call sequence; True = succeed, False = raise.
        # outcome_fn: callable(call_index_0based) -> bool; takes precedence.
        self.outcomes = list(outcomes) if outcomes is not None else [True]
        self.outcome_fn = outcome_fn
        self.calls = 0
        self.model = name

    def _next_outcome(self) -> bool:
        if self.outcome_fn is not None:
            return bool(self.outcome_fn(self.calls - 1))
        index = min(self.calls - 1, len(self.outcomes) - 1)
        return self.outcomes[index]

    async def __call__(self, *_args: Any, **_kwargs: Any) -> Any:
        self.calls += 1
        if not self._next_outcome():
            raise RuntimeError(f"boom:{self.model}")
        return f"ok:{self.model}"


async def _failing_stream() -> AsyncGenerator[Any, None]:
    if False:  # pragma: no cover - make the generator lazy
        yield None
    raise RuntimeError("stream-boom")


class _StreamFailModel(_FakeModel):
    """Model whose calls always return an immediately-failing stream."""

    async def __call__(self, *_args: Any, **_kwargs: Any) -> Any:
        self.calls += 1
        return _failing_stream()


def _slot(provider: str, model: str) -> ModelSlotConfig:
    return ModelSlotConfig(provider_id=provider, model=model)


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    reset_fallback_state()
    yield
    reset_fallback_state()


def _build(
    models: list[Any],
    slots: list[ModelSlotConfig],
) -> PersistentFallbackChatModel:
    for model, slot in zip(models, slots):
        model.model = f"{slot.provider_id}:{slot.model}"
    return PersistentFallbackChatModel(models, slots)


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------


async def test_all_healthy_uses_first_slot() -> None:
    m0, m1 = _FakeModel(), _FakeModel()
    chain = _build([m0, m1], [_slot("p0", "a"), _slot("p1", "b")])

    result = await chain()
    assert result == "ok:p0:a"
    assert m0.calls == 1
    assert m1.calls == 0
    assert get_fallback_current_index() == 0


async def test_success_resets_counter() -> None:
    # flaky primary: fail, fail, recover (one success), then fail forever
    flaky = _FakeModel(outcome_fn=lambda call: call == 2)
    healthy = _FakeModel()
    chain = _build(
        [flaky, healthy],
        [_slot("p0", "a"), _slot("p1", "b")],
    )

    # 1st failure -> probe backup succeeds (no error surfaced)
    assert await chain() == "ok:p1:b"
    # 2nd failure -> probe backup succeeds again
    assert await chain() == "ok:p1:b"
    # primary recovers -> success resets its counter
    assert await chain() == "ok:p0:a"
    assert get_fallback_current_index() == 0

    # three consecutive failures again -> retire primary
    assert await chain() == "ok:p1:b"
    assert await chain() == "ok:p1:b"
    assert await chain() == "ok:p1:b"  # 3rd consecutive failure retires p0
    assert get_fallback_current_index() == 1

    # now the pointer lives on the backup; primary is never called again
    flaky_before = flaky.calls
    assert await chain() == "ok:p1:b"
    assert flaky.calls == flaky_before


async def test_three_consecutive_failures_switch_permanently() -> None:
    dead = _FakeModel([False])
    healthy = _FakeModel()
    chain = _build([dead, healthy], [_slot("p0", "a"), _slot("p1", "b")])

    # failures 1 and 2 are absorbed by probing the backup
    assert await chain() == "ok:p1:b"
    assert await chain() == "ok:p1:b"
    # failure 3 retires the primary and moves the pointer to the backup
    assert await chain() == "ok:p1:b"
    assert get_fallback_current_index() == 1
    assert dead.calls == 3
    assert healthy.calls == 3

    # subsequent requests go straight to the backup
    assert await chain() == "ok:p1:b"
    assert dead.calls == 3
    assert healthy.calls == 4


async def test_chain_exhausted_stops() -> None:
    dead0 = _FakeModel([False])
    dead1 = _FakeModel([False])
    slots = [_slot("p0", "a"), _slot("p1", "b")]
    chain = _build([dead0, dead1], slots)

    # Round 1: both slots fail once each (raw error surfaces)
    with pytest.raises(RuntimeError):
        await chain()
    # Round 2: both fail twice
    with pytest.raises(RuntimeError):
        await chain()
    # Round 3: slot0 retires (3rd) then slot1 retires (3rd) -> exhausted
    with pytest.raises(ModelFallbackExhaustedException) as excinfo:
        await chain()
    assert excinfo.value.attempted == ["p0/a", "p1/b"]
    assert get_fallback_current_index() == 2

    # Chain is stopped: subsequent requests fail fast without calling models
    calls_before = (dead0.calls, dead1.calls)
    with pytest.raises(ModelFallbackExhaustedException):
        await chain()
    assert (dead0.calls, dead1.calls) == calls_before


async def test_exhausted_error_reports_last_error() -> None:
    dead = _FakeModel([False])
    slots = [_slot("p0", "a")]
    chain = _build([dead], slots)

    with pytest.raises(RuntimeError):
        await chain()
    with pytest.raises(RuntimeError):
        await chain()
    with pytest.raises(ModelFallbackExhaustedException) as excinfo:
        await chain()
    assert "boom:p0" in (excinfo.value.details or {}).get(
        "last_error",
        "",
    )


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


async def test_stream_failure_fails_over() -> None:
    streamy = _StreamFailModel()
    healthy = _FakeModel()
    chain = _build(
        [streamy, healthy],
        [_slot("p0", "a"), _slot("p1", "b")],
    )

    # Round 1-2: mid-stream failure is absorbed by probing the backup
    for _ in range(2):
        stream = await chain()
        chunks = [chunk async for chunk in stream]
        assert chunks == ["ok:p1:b"]
    assert get_fallback_current_index() == 0

    # Round 3: primary retires; pointer moves to the backup
    stream = await chain()
    chunks = [chunk async for chunk in stream]
    assert chunks == ["ok:p1:b"]
    assert get_fallback_current_index() == 1

    # Next request starts directly on the backup
    result = await chain()
    assert result == "ok:p1:b"
    assert streamy.calls == 3


async def test_stream_exhaustion_raises() -> None:
    streamy0 = _StreamFailModel()
    streamy1 = _StreamFailModel()
    chain = _build(
        [streamy0, streamy1],
        [_slot("p0", "a"), _slot("p1", "b")],
    )

    for _ in range(2):
        stream = await chain()
        with pytest.raises(RuntimeError):
            async for _chunk in stream:
                pass

    stream = await chain()
    with pytest.raises(ModelFallbackExhaustedException):
        async for _chunk in stream:
            pass
    assert get_fallback_current_index() == 2


# ---------------------------------------------------------------------------
# State resets
# ---------------------------------------------------------------------------


async def test_reset_state_restarts_chain() -> None:
    dead0 = _FakeModel([False])
    dead1 = _FakeModel([False])
    slots = [_slot("p0", "a"), _slot("p1", "b")]
    chain = _build([dead0, dead1], slots)

    for _ in range(3):
        with pytest.raises((RuntimeError, ModelFallbackExhaustedException)):
            await chain()
    assert get_fallback_current_index() == 2

    reset_fallback_state()
    assert get_fallback_current_index() == 0
    # A fresh call starts from the top again (slot0 retries from 0)
    with pytest.raises(RuntimeError):
        await chain()
    assert dead0.calls == 4


async def test_new_chain_identity_resets_state() -> None:
    dead0 = _FakeModel([False])
    dead1 = _FakeModel([False])
    chain = _build(
        [dead0, dead1],
        [_slot("p0", "a"), _slot("p1", "b")],
    )
    for _ in range(3):
        with pytest.raises((RuntimeError, ModelFallbackExhaustedException)):
            await chain()
    assert get_fallback_current_index() == 2

    # A different chain (different entries) starts over from the top
    other0 = _FakeModel()
    other1 = _FakeModel()
    chain2 = _build(
        [other0, other1],
        [_slot("p9", "x"), _slot("p8", "y")],
    )
    assert get_fallback_current_index() == 0
    assert await chain2() == "ok:p9:x"


async def test_intermittent_failures_retire_within_window() -> None:
    """An intermittently failing model is retired once its recent failure
    rate crosses the threshold, even though it never fails 3 times in a
    row (strict consecutive counting would never retire it and would keep
    retrying it first on every request)."""
    # 0-based call sequence: F, S, F, S, F, ...
    flaky = _FakeModel(outcome_fn=lambda call: call % 2 == 1)
    healthy = _FakeModel()
    chain = _build([flaky, healthy], [_slot("p0", "a"), _slot("p1", "b")])

    # Requests 1-4: failures probe the backup, successes serve directly.
    assert await chain() == "ok:p1:b"  # call0 = F -> probe
    assert await chain() == "ok:p0:a"  # call1 = S
    assert await chain() == "ok:p1:b"  # call2 = F -> probe
    assert await chain() == "ok:p0:a"  # call3 = S
    assert get_fallback_current_index() == 0

    # Request 5: call4 = F -> 3rd failure in the 5-attempt window -> retire.
    assert await chain() == "ok:p1:b"
    assert get_fallback_current_index() == 1

    # Subsequent requests go straight to the healthy backup.
    flaky_before = flaky.calls
    assert await chain() == "ok:p1:b"
    assert flaky.calls == flaky_before


def test_failure_rate_constants() -> None:
    assert FAILURE_WINDOW == 5
    assert RETIRE_FAILURE_THRESHOLD == 3
    # The threshold is strictly within the window so a model cannot be
    # retired until it has actually been attempted enough times.
    assert RETIRE_FAILURE_THRESHOLD <= FAILURE_WINDOW


async def test_set_fallback_current_index_moves_pointer() -> None:
    """The failover pointer can be moved to a specific chain slot."""
    _build(
        [_FakeModel(), _FakeModel(), _FakeModel()],
        [_slot("p0", "a"), _slot("p1", "b"), _slot("p2", "c")],
    )
    assert get_fallback_current_index() == 0

    set_fallback_current_index(1)
    assert get_fallback_current_index() == 1

    set_fallback_current_index(2)
    assert get_fallback_current_index() == 2

    # Out-of-range values are clamped.
    set_fallback_current_index(99)
    assert get_fallback_current_index() == 2
    set_fallback_current_index(-5)
    assert get_fallback_current_index() == 0
