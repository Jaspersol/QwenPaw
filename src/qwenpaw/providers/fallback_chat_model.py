# -*- coding: utf-8 -*-
"""Priority-chain chat model wrapper with failure-rate failover.

When the user configures an ordered list of default models ("fallback
models"), every agent request is served by a :class:`FallbackChatModel`
that wraps one inner model per chain slot.  The wrapper tracks a small
rolling window of recent outcomes per slot:

- On success the slot records a success in its window.
- On failure the request transparently tries the next slot in priority
  order, so the user does not see a raw error while any healthy backup
  remains.
- When a slot accumulates ``RETIRE_FAILURE_THRESHOLD`` failures within the
  last ``FAILURE_WINDOW`` attempts it is retired: the process-wide pointer
  permanently moves past it and future requests start at the next slot.
  Using a *window* rather than strict consecutive counting means an
  intermittently failing ("half-broken") model is also retired instead of
  being retried first on every request — which would otherwise double the
  latency/cost of every other request forever.
- When the *last* slot is also retired the chain is exhausted: the wrapper
  raises :class:`qwenpaw.exceptions.ModelFallbackExhaustedException` and
  stops trying (subsequent requests fail fast until the configuration
  changes or the process restarts).

The failover position is process-wide (shared by every ``FallbackChatModel``
instance) because the fallback list is a global setting.  Changing the list
through ``ProviderManager.set_fallback_models()`` calls
:func:`reset_fallback_state`, which starts the chain over from the top.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Any, AsyncGenerator

from agentscope.model import ChatModelBase
from agentscope.model._model_response import ChatResponse

from ..config.config import ModelSlotConfig
from ..exceptions import ModelFallbackExhaustedException

logger = logging.getLogger(__name__)

#: Sliding window (in attempts) over which a slot's failure rate is judged.
FAILURE_WINDOW = 5
#: Number of failures within ``FAILURE_WINDOW`` that retires a slot.  This
#: generalises "3 consecutive failures" (3 failures / 3 attempts) so that an
#: intermittently failing model is also retired instead of lingering at the
#: head of the chain and being retried first on every request.
RETIRE_FAILURE_THRESHOLD = 3

# ---------------------------------------------------------------------------
# Process-wide failover state
# ---------------------------------------------------------------------------


class _FallbackState:
    """Shared, process-wide failover position and per-slot recent outcomes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._fingerprint: str | None = None
        self._size = 0
        self._index = 0
        # Per-slot rolling window of recent outcomes (True=success,
        # False=failure), bounded to ``FAILURE_WINDOW`` entries.
        self._history: list[deque[bool]] = []

    def ensure(self, fingerprint: str, size: int) -> None:
        """Reset the state when the chain identity changed (or first use)."""
        with self._lock:
            if self._fingerprint != fingerprint:
                self._fingerprint = fingerprint
                self._size = size
                self._index = 0
                self._history = [
                    deque(maxlen=FAILURE_WINDOW) for _ in range(size)
                ]

    def reset(self) -> None:
        """Reset the failover position and all outcome history."""
        with self._lock:
            self._index = 0
            for history in self._history:
                history.clear()

    def index(self) -> int:
        with self._lock:
            return self._index

    def set_index(self, index: int) -> None:
        """Set the failover position to *index* (clamped into range)."""
        with self._lock:
            if self._size <= 0:
                self._index = 0
            else:
                self._index = max(0, min(index, self._size - 1))

    def _failures(self, index: int) -> int:
        """Number of failures in *index*'s window (lock must be held)."""
        if 0 <= index < len(self._history):
            return sum(1 for ok in self._history[index] if not ok)
        return 0

    def failure_count(self, index: int) -> int:
        with self._lock:
            return self._failures(index)

    def is_retired(self, index: int) -> bool:
        """Return *True* when *index* hit the failure threshold."""
        return self.failure_count(index) >= RETIRE_FAILURE_THRESHOLD

    def record_success(self, index: int) -> None:
        """Record a success for *index*."""
        with self._lock:
            if 0 <= index < len(self._history):
                self._history[index].append(True)

    def record_failure(self, index: int) -> int | None:
        """Record a failure for slot *index*.

        Only the *pointer* slot can advance the pointer.  Returns the new
        pointer index when that slot was retired (its failure rate crossed
        the threshold), or ``None`` otherwise.  The returned index may equal
        the chain size, signalling exhaustion.
        """
        with self._lock:
            if 0 <= index < len(self._history):
                self._history[index].append(False)
            if index != self._index:
                # Probed slot: no pointer movement.  Once its failure rate
                # crosses the threshold it is skipped by callers.
                return None
            if self._failures(index) < RETIRE_FAILURE_THRESHOLD:
                return None
            self._index += 1
            return self._index


_state = _FallbackState()


def reset_fallback_state() -> None:
    """Reset the process-wide failover position to the first slot.

    Called by :meth:`ProviderManager.set_fallback_models` after the chain
    is changed, and available for tests.
    """
    _state.reset()


def get_fallback_current_index() -> int:
    """Return the index of the slot currently in use (may equal size)."""
    return _state.index()


def set_fallback_current_index(index: int) -> None:
    """Move the process-wide failover position to *index* in the chain.

    Used by the model-switch endpoint so that switching a session to a model
    that is part of the chain continues the chain from that model.
    """
    _state.set_index(index)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


def _slot_label(slot: ModelSlotConfig) -> str:
    return f"{slot.provider_id}/{slot.model}"


def _innermost_formatter(model: Any) -> Any | None:
    """Resolve the innermost ``formatter`` through wrapper chains."""
    seen: set[int] = set()
    current = model
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        formatter = getattr(current, "formatter", None)
        if formatter is not None:
            return formatter
        current = getattr(current, "_inner", None) or getattr(
            current,
            "_model",
            None,
        )
    return None


def _build_exhausted_error(
    slots: list[ModelSlotConfig],
    last_exc: Exception | None,
) -> ModelFallbackExhaustedException:
    """Build the terminal error once every slot has failed."""
    attempted = [_slot_label(slot) for slot in slots]
    exc = ModelFallbackExhaustedException(attempted=attempted)
    if last_exc is not None:
        exc.details = {"last_error": str(last_exc)}
        exc.__cause__ = last_exc
        exc.__context__ = last_exc
    return exc


class FallbackChatModel(ChatModelBase):
    """Ordered multi-model wrapper with failure-rate failover.

    Args:
        models: One inner chat model per chain slot, already wrapped with
            retry/token-recording layers.  ``models[i]`` must correspond to
            ``slots[i]``.
        slots: The priority-ordered ``ModelSlotConfig`` entries that the
            inner models were built from.  Used for logging and for the
            exhausted-chain error.
    """

    def __init__(
        self,
        models: list[ChatModelBase],
        slots: list[ModelSlotConfig],
    ) -> None:
        if not models or not slots or len(models) != len(slots):
            raise ValueError(
                "FallbackChatModel requires a non-empty, parallel list of "
                "models and slots.",
            )
        first = models[0]
        super().__init__(
            credential=getattr(first, "credential", None),
            model=getattr(first, "model", "unknown"),
            parameters=getattr(first, "parameters", None)
            or ChatModelBase.Parameters(),
            stream=getattr(first, "stream", True),
            context_size=getattr(first, "context_size", 32768),
        )
        self._models = models
        self._slots = slots
        fingerprint = "\n".join(_slot_label(slot) for slot in slots)
        _state.ensure(fingerprint, len(models))

    # -- formatter transparency --------------------------------------------
    # The runtime inspects ``model.formatter`` to decide wire-format
    # behaviour (media stripping etc.).  Resolve it from the currently
    # active slot so the answer tracks failover switches.
    @property
    def formatter(self) -> Any | None:  # type: ignore[override]
        index = min(_state.index(), len(self._models) - 1)
        return _innermost_formatter(self._models[index])

    @formatter.setter
    def formatter(self, value: Any) -> None:
        # ChatModelBase subclasses may assign ``self.formatter`` during
        # construction; accept and ignore it — resolution is dynamic.
        del value

    # -- failover plumbing ---------------------------------------------------

    def _handle_failure(self, index: int, last_exc: Exception) -> int:
        """Record a failure at *index* and return the next slot to try.

        Retires the pointer slot when its failure rate crosses the
        threshold, skips slots already retired, and raises *last_exc* when
        the request has no healthy slot left.  Raises
        :class:`ModelFallbackExhaustedException` only when the *last* slot
        was just retired (the chain is permanently stopped).
        """
        new_index = _state.record_failure(index)
        if new_index is not None:
            if new_index >= len(self._models):
                raise _build_exhausted_error(self._slots, last_exc)
            self._log_switch(index, new_index)
            index = new_index
        else:
            # Failure rate below the threshold: keep the pointer, probe the
            # next slot for this request so the user's request still
            # succeeds when a healthy backup exists.
            index += 1
        while index < len(self._models) and _state.is_retired(index):
            index += 1
        if index >= len(self._models):
            # Probe walk ran past the end without a permanent switch: the
            # request simply fails with the last model's error.
            raise last_exc
        return index

    # -- core call logic -----------------------------------------------------

    async def _call_with_failover(
        self,
        args: tuple,
        kwargs: dict[str, Any],
    ) -> tuple[Any, int]:
        """Invoke the chain for one (non-streamed) request.

        Returns ``(result, index)`` of the slot that produced *result*.
        Raises the underlying exception only when every remaining slot
        failed, or :class:`ModelFallbackExhaustedException` once the whole
        chain has been retired.
        """
        last_exc: Exception | None = None
        index = _state.index()
        while True:
            if index >= len(self._models):
                raise _build_exhausted_error(self._slots, last_exc)
            if _state.is_retired(index):
                index += 1
                continue
            model = self._models[index]
            try:
                result = await model(*args, **kwargs)
            except Exception as exc:  # pylint: disable=broad-except
                last_exc = exc
                index = self._handle_failure(index, last_exc)
                continue
            if isinstance(result, AsyncGenerator):
                # A stream's real outcome is only known after it is fully
                # consumed; success/failure is accounted for in
                # ``_wrap_stream``, not here.
                return result, index
            _state.record_success(index)
            return result, index

    async def __call__(  # type: ignore[override]
        self,
        *args: Any,
        **kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        result, index = await self._call_with_failover(args, kwargs)
        if isinstance(result, AsyncGenerator):
            return self._wrap_stream(result, args, kwargs, index)
        return result

    async def _wrap_stream(
        self,
        stream: AsyncGenerator[ChatResponse, None],
        args: tuple,
        kwargs: dict[str, Any],
        index: int,
    ) -> AsyncGenerator[ChatResponse, None]:
        """Yield chunks; on mid-stream failure fail over to the next slot."""
        last_exc: Exception | None = None
        while True:
            try:
                async for chunk in stream:
                    yield chunk
                _state.record_success(index)
                return
            except Exception as exc:  # pylint: disable=broad-except
                last_exc = exc
                index = self._handle_failure(index, last_exc)
                model = self._models[index]
                try:
                    result = await model(*args, **kwargs)
                except Exception as exc2:  # pylint: disable=broad-except
                    last_exc = exc2
                    index = self._handle_failure(index, last_exc)
                    continue
                if isinstance(result, AsyncGenerator):
                    stream = result
                    continue
                _state.record_success(index)
                yield result
                return

    def _log_switch(self, retired_index: int, new_index: int) -> None:
        retired = self._slots[retired_index]
        switched_to = self._slots[new_index]
        logger.warning(
            "Model %s failed %d times within the last %d attempts; "
            "switching to next priority model %s (index %d).",
            _slot_label(retired),
            RETIRE_FAILURE_THRESHOLD,
            FAILURE_WINDOW,
            _slot_label(switched_to),
            new_index,
        )


__all__ = [
    "FAILURE_WINDOW",
    "RETIRE_FAILURE_THRESHOLD",
    "FallbackChatModel",
    "get_fallback_current_index",
    "reset_fallback_state",
    "set_fallback_current_index",
]
