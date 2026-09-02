# -*- coding: utf-8 -*-
"""Cross-model fallback wrapper for transient pre-output failures."""

from __future__ import annotations

import logging
import threading
from collections import deque
from contextvars import ContextVar, Token
from typing import Any, AsyncGenerator

from agentscope.model import ChatModelBase
from agentscope.model._model_response import ChatResponse

from ..config.config import ModelSlotConfig
from ..exceptions import ModelFallbackExhaustedException
from .model_error_policy import classify_model_error, is_fallback_eligible
from .stream_progress import has_meaningful_stream_content

logger = logging.getLogger(__name__)

_FALLBACK_NOTICE_SINK: ContextVar[dict[str, Any] | None] = ContextVar(
    "qwenpaw_fallback_notice_sink",
    default=None,
)


def install_fallback_notice_sink() -> dict[str, Any]:
    """Install a per-request sink for model-fallback transparency data.

    The pinned agentscope release drops ``ChatResponse.metadata`` when
    converting model output into agent events, so annotating responses
    alone never reaches the Console or channel notifiers.  The reply
    loop installs this sink before iterating events (same task context
    as the model call); ``FallbackChatModel`` publishes each fallback
    into it, and the agent re-attaches the data onto outgoing events.
    """
    sink: dict[str, Any] = {"events": [], "actual_model": None}
    _FALLBACK_NOTICE_SINK.set(sink)
    return sink


class FallbackChatModel(ChatModelBase):
    """Try configured models in order before any response becomes visible."""

    def __init__(self, models: list[ChatModelBase]) -> None:
        if not models:
            raise ValueError("FallbackChatModel requires at least one model")
        primary = models[0]
        self._active_model_var: ContextVar[ChatModelBase] = ContextVar(
            f"fallback_active_model_{id(self)}",
            default=primary,
        )
        self._default_model = getattr(primary, "model", "unknown")
        self._default_context_size = getattr(
            primary,
            "context_size",
            32_768,
        )
        super().__init__(
            credential=getattr(primary, "credential", None),
            model=getattr(primary, "model", "unknown"),
            parameters=getattr(primary, "parameters", None)
            or ChatModelBase.Parameters(),
            stream=getattr(primary, "stream", True),
            context_size=getattr(primary, "context_size", 32_768),
        )
        self._models = models
        self._activate_model(primary)

    @property
    def _active_model(self) -> ChatModelBase:
        """Return the model active in the current request context."""
        return self._active_model_var.get()

    @_active_model.setter
    def _active_model(self, model: ChatModelBase) -> None:
        self._active_model_var.set(model)

    @property
    def _inner(self) -> ChatModelBase:
        """Expose the request-local active model for wrapper traversal."""
        return self._active_model

    @_inner.setter
    def _inner(self, model: ChatModelBase) -> None:
        self._active_model = model

    @property
    def formatter(self) -> Any:
        """Expose the serving model's formatter to AgentScope.

        AgentScope reads media support and formats messages off the
        outermost model, which is this class once fallbacks are
        configured.  ``ChatModelBase`` defines no formatter of its own, so
        without this forwarding the attribute lookup raises.
        """
        active = getattr(self, "_active_model_var", None)
        if active is None:
            raise AttributeError("formatter")
        return active.get().formatter

    @formatter.setter
    def formatter(self, value: Any) -> None:
        """Route formatter installs down to the serving model."""
        self._active_model.formatter = value

    @property
    def model(self) -> str:
        """Return the current request's actual model name."""
        active = getattr(self, "_active_model_var", None)
        if active is not None:
            return str(getattr(active.get(), "model", self._default_model))
        return self._default_model

    @model.setter
    def model(self, value: str) -> None:
        self._default_model = value

    @property
    def context_size(self) -> int:
        """Return the current request's actual context window."""
        active = getattr(self, "_active_model_var", None)
        if active is not None:
            return int(
                getattr(
                    active.get(),
                    "context_size",
                    self._default_context_size,
                ),
            )
        return self._default_context_size

    @context_size.setter
    def context_size(self, value: int) -> None:
        self._default_context_size = value

    def _activate_model(self, model: ChatModelBase) -> None:
        """Expose routing metadata from the model handling the request."""
        self._active_model = model

    def _begin_request(self) -> Token:
        """Activate the primary model and snapshot the pre-request state.

        The returned token MUST be passed to :meth:`_end_request` once the
        request settles (response returned, stream exhausted, or error
        raised).  Without the reset, the last-served fallback would leak
        into the between-requests window where the compaction manager
        sizes the context budget and capability learning reads
        ``model_key`` -- both must see the primary model, because the
        next request always tries the primary first.
        """
        return self._active_model_var.set(self._models[0])

    def _end_request(self, token: Token) -> None:
        """Restore the pre-request active model.

        Ends by enforcing the invariant directly: between requests the
        context must expose the primary model.  Token reset alone is not
        enough -- an abandoned stream closed late resets out of order,
        and CPython then silently restores the token's stale snapshot
        instead of raising.
        """
        try:
            self._active_model_var.reset(token)
        except ValueError:
            # The stream was consumed in a different context than the
            # one that started the request; fall through and repair the
            # consumer's context below.
            pass
        if self._active_model_var.get() is not self._models[0]:
            self._active_model_var.set(self._models[0])

    @property
    def model_key(self) -> str:
        """Return the key for the model handling the current request."""
        key = getattr(self._active_model, "model_key", None)
        name = getattr(self._active_model, "model", None)
        return str(key or name or self.model)

    async def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        last_error: Exception | None = None
        fallback_events: list[dict[str, str]] = []
        token: Token | None = self._begin_request()
        try:
            for index, model in enumerate(self._models):
                self._activate_model(model)
                try:
                    response = await model(*args, **kwargs)
                except Exception as exc:
                    last_error = exc
                    if not self._can_try_next(index, exc):
                        raise
                    following = self._models[index + 1]
                    fallback_events.append(
                        self._record_fallback(model, following, exc),
                    )
                    continue
                if isinstance(response, AsyncGenerator):
                    stream_token = token
                    assert stream_token is not None
                    token = None  # the stream wrapper owns the reset now
                    return self._consume_with_fallback(
                        response,
                        index,
                        args,
                        kwargs,
                        fallback_events,
                        stream_token,
                    )
                return self._annotate_response(
                    response,
                    fallback_events,
                    model,
                )
            assert last_error is not None
            raise last_error
        finally:
            if token is not None:
                self._end_request(token)

    async def _consume_with_fallback(
        self,
        stream: AsyncGenerator[ChatResponse, None],
        index: int,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        fallback_events: list[dict[str, str]],
        reset_token: Token,
    ) -> AsyncGenerator[ChatResponse, None]:
        try:
            current = stream
            current_index = index
            current_model = self._models[index]
            emitted = False
            while True:
                fallback_error: Exception | None = None
                try:
                    async for chunk in current:
                        emitted = emitted or has_meaningful_stream_content(
                            chunk.content,
                        )
                        yield self._annotate_response(
                            chunk,
                            fallback_events,
                            current_model,
                        )
                        fallback_events = []
                    return
                except Exception as exc:
                    if emitted or not self._can_try_next(current_index, exc):
                        raise
                    fallback_error = exc
                finally:
                    await current.aclose()
                assert fallback_error is not None
                response, current_index = await self._start_fallback(
                    current_index,
                    fallback_error,
                    args,
                    kwargs,
                    fallback_events,
                )
                current_model = self._models[current_index]
                if not isinstance(response, AsyncGenerator):
                    yield self._annotate_response(
                        response,
                        fallback_events,
                        current_model,
                    )
                    return
                current = response
        finally:
            self._end_request(reset_token)

    async def _start_fallback(
        self,
        current_index: int,
        error: Exception,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        fallback_events: list[dict[str, str]],
    ) -> tuple[ChatResponse | AsyncGenerator[ChatResponse, None], int]:
        """Start the next usable fallback, skipping pre-stream failures."""
        last_error = error
        for next_index in range(current_index + 1, len(self._models)):
            current_model = self._models[next_index - 1]
            next_model = self._models[next_index]
            fallback_events.append(
                self._record_fallback(current_model, next_model, last_error),
            )
            self._activate_model(next_model)
            try:
                return await next_model(*args, **kwargs), next_index
            except Exception as exc:
                last_error = exc
                if not self._can_try_next(next_index, exc):
                    raise
        raise last_error

    def _can_try_next(self, index: int, exc: Exception) -> bool:
        if index + 1 >= len(self._models):
            return False
        # Only the primary model's error class decides whether fallback
        # engages at all.  Once the chain is running, a broken candidate
        # (revoked key, deleted model, ...) must not mask the healthy
        # candidates behind it, so its own error never stops the walk.
        return index > 0 or is_fallback_eligible(exc)

    def _record_fallback(
        self,
        current: ChatModelBase,
        following: ChatModelBase,
        exc: Exception,
    ) -> dict[str, str]:
        """Log one fallback hop and publish it to the request sink."""
        self._log_fallback(current, following, exc)
        event = self._fallback_event(current, following, exc)
        sink = _FALLBACK_NOTICE_SINK.get()
        if sink is not None:
            sink["events"].append(dict(event))
            sink["actual_model"] = self._actual_model_dict(following)
        return event

    @staticmethod
    def _model_identity(model: ChatModelBase) -> tuple[str, str]:
        key = str(getattr(model, "model_key", "") or "")
        name = str(getattr(model, "model", "unknown") or "unknown")
        if ":" not in key:
            provider_id = str(getattr(model, "_provider_id", "") or "")
            return provider_id, key or name
        provider_id, model_id = key.split(":", maxsplit=1)
        return provider_id, model_id

    @classmethod
    def _fallback_event(
        cls,
        current: ChatModelBase,
        following: ChatModelBase,
        exc: Exception,
    ) -> dict[str, str]:
        from_provider_id, from_model_id = cls._model_identity(current)
        to_provider_id, to_model_id = cls._model_identity(following)
        return {
            "type": "model_fallback",
            "from_provider_id": from_provider_id,
            "from_model_id": from_model_id,
            "to_provider_id": to_provider_id,
            "to_model_id": to_model_id,
            "reason_kind": classify_model_error(exc).kind,
        }

    @classmethod
    def _actual_model_dict(cls, active_model: ChatModelBase) -> dict[str, Any]:
        provider_id, model_id = cls._model_identity(active_model)
        return {
            "provider_id": provider_id,
            "model_id": model_id,
            "context_size": getattr(
                active_model,
                "context_size",
                32_768,
            ),
        }

    @staticmethod
    def _annotate_response(
        response: ChatResponse,
        events: list[dict[str, str]],
        active_model: ChatModelBase | None = None,
    ) -> ChatResponse:
        if not events and active_model is None:
            return response
        metadata = dict(getattr(response, "metadata", None) or {})
        if events:
            metadata["qwenpaw_model_fallbacks"] = list(events)
        if active_model is not None:
            metadata[
                "qwenpaw_actual_model"
            ] = FallbackChatModel._actual_model_dict(active_model)
        response.metadata = metadata
        return response

    @staticmethod
    def _log_fallback(
        current: ChatModelBase,
        following: ChatModelBase,
        exc: Exception,
    ) -> None:
        logger.warning(
            "Model %s failed before output; falling back to %s: %s",
            getattr(current, "model", "unknown"),
            getattr(following, "model", "unknown"),
            exc,
        )

    async def generate_structured_output(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        last_error: Exception | None = None
        fallback_events: list[dict[str, str]] = []
        token = self._begin_request()
        try:
            for index, model in enumerate(self._models):
                self._activate_model(model)
                try:
                    response = await model.generate_structured_output(
                        *args,
                        **kwargs,
                    )
                    return self._annotate_response(
                        response,
                        fallback_events,
                        model,
                    )
                except Exception as exc:
                    last_error = exc
                    if not self._can_try_next(index, exc):
                        raise
                    following = self._models[index + 1]
                    fallback_events.append(
                        self._record_fallback(model, following, exc),
                    )
            assert last_error is not None
            raise last_error
        finally:
            self._end_request(token)


# ---------------------------------------------------------------------------
# Persistent global fallback compatibility
# ---------------------------------------------------------------------------

# The main fallback wrapper above is request-local and always retries the
# primary model on the next request. The current branch also exposes a global
# priority chain whose pointer advances after repeated failures. Keeping that
# behavior in a separate wrapper preserves both routing contracts.
FAILURE_WINDOW = 5
RETIRE_FAILURE_THRESHOLD = 3


class _PersistentFallbackState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._fingerprint: str | None = None
        self._size = 0
        self._index = 0
        self._history: list[deque[bool]] = []

    def ensure(self, fingerprint: str, size: int) -> None:
        with self._lock:
            if self._fingerprint != fingerprint:
                self._fingerprint = fingerprint
                self._size = size
                self._index = 0
                self._history = [
                    deque(maxlen=FAILURE_WINDOW) for _ in range(size)
                ]

    def reset(self) -> None:
        with self._lock:
            self._index = 0
            for history in self._history:
                history.clear()

    def index(self) -> int:
        with self._lock:
            return self._index

    def set_index(self, index: int) -> None:
        with self._lock:
            if self._size <= 0:
                self._index = 0
            else:
                self._index = max(0, min(index, self._size - 1))

    def _failures(self, index: int) -> int:
        if 0 <= index < len(self._history):
            return sum(1 for succeeded in self._history[index] if not succeeded)
        return 0

    def failure_count(self, index: int) -> int:
        with self._lock:
            return self._failures(index)

    def is_retired(self, index: int) -> bool:
        return self.failure_count(index) >= RETIRE_FAILURE_THRESHOLD

    def record_success(self, index: int) -> None:
        with self._lock:
            if 0 <= index < len(self._history):
                self._history[index].append(True)

    def record_failure(self, index: int) -> int | None:
        with self._lock:
            if 0 <= index < len(self._history):
                self._history[index].append(False)
            if index != self._index:
                return None
            if self._failures(index) < RETIRE_FAILURE_THRESHOLD:
                return None
            self._index += 1
            return self._index


_persistent_state = _PersistentFallbackState()


def reset_fallback_state() -> None:
    """Reset the process-wide global fallback pointer and outcome history."""
    _persistent_state.reset()


def get_fallback_current_index() -> int:
    """Return the global fallback pointer (which may equal chain size)."""
    return _persistent_state.index()


def set_fallback_current_index(index: int) -> None:
    """Move the global fallback pointer to a configured chain slot."""
    _persistent_state.set_index(index)


def _slot_label(slot: ModelSlotConfig) -> str:
    return f"{slot.provider_id}/{slot.model}"


def _innermost_formatter(model: Any) -> Any | None:
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
    exc = ModelFallbackExhaustedException(
        attempted=[_slot_label(slot) for slot in slots],
    )
    if last_exc is not None:
        exc.details = {"last_error": str(last_exc)}
        exc.__cause__ = last_exc
        exc.__context__ = last_exc
    return exc


class PersistentFallbackChatModel(ChatModelBase):
    """Global priority chain that permanently retires repeatedly failing slots."""

    def __init__(
        self,
        models: list[ChatModelBase],
        slots: list[ModelSlotConfig],
    ) -> None:
        if not models or not slots or len(models) != len(slots):
            raise ValueError(
                "PersistentFallbackChatModel requires non-empty parallel "
                "model and slot lists.",
            )
        first = models[0]
        super().__init__(
            credential=getattr(first, "credential", None),
            model=getattr(first, "model", "unknown"),
            parameters=getattr(first, "parameters", None)
            or ChatModelBase.Parameters(),
            stream=getattr(first, "stream", True),
            context_size=getattr(first, "context_size", 32_768),
        )
        self._models = models
        self._slots = slots
        fingerprint = "\n".join(_slot_label(slot) for slot in slots)
        _persistent_state.ensure(fingerprint, len(models))

    @property
    def formatter(self) -> Any | None:  # type: ignore[override]
        index = min(_persistent_state.index(), len(self._models) - 1)
        return _innermost_formatter(self._models[index])

    @formatter.setter
    def formatter(self, value: Any) -> None:
        del value

    def _handle_failure(self, index: int, last_exc: Exception) -> int:
        new_index = _persistent_state.record_failure(index)
        if new_index is not None:
            if new_index >= len(self._models):
                raise _build_exhausted_error(self._slots, last_exc)
            self._log_switch(index, new_index)
            index = new_index
        else:
            index += 1
        while index < len(self._models) and _persistent_state.is_retired(index):
            index += 1
        if index >= len(self._models):
            raise last_exc
        return index

    async def _call_with_failover(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[Any, int]:
        last_exc: Exception | None = None
        index = _persistent_state.index()
        while True:
            if index >= len(self._models):
                raise _build_exhausted_error(self._slots, last_exc)
            if _persistent_state.is_retired(index):
                index += 1
                continue
            try:
                result = await self._models[index](*args, **kwargs)
            except Exception as exc:  # pylint: disable=broad-except
                last_exc = exc
                index = self._handle_failure(index, exc)
                continue
            if isinstance(result, AsyncGenerator):
                return result, index
            _persistent_state.record_success(index)
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
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        index: int,
    ) -> AsyncGenerator[ChatResponse, None]:
        while True:
            try:
                async for chunk in stream:
                    yield chunk
                _persistent_state.record_success(index)
                return
            except Exception as exc:  # pylint: disable=broad-except
                index = self._handle_failure(index, exc)
                try:
                    result = await self._models[index](*args, **kwargs)
                except Exception as next_exc:  # pylint: disable=broad-except
                    index = self._handle_failure(index, next_exc)
                    continue
                if isinstance(result, AsyncGenerator):
                    stream = result
                    continue
                _persistent_state.record_success(index)
                yield result
                return

    def _log_switch(self, retired_index: int, new_index: int) -> None:
        logger.warning(
            "Model %s failed %d times within the last %d attempts; "
            "switching to global priority model %s (index %d).",
            _slot_label(self._slots[retired_index]),
            RETIRE_FAILURE_THRESHOLD,
            FAILURE_WINDOW,
            _slot_label(self._slots[new_index]),
            new_index,
        )


__all__ = [
    "FAILURE_WINDOW",
    "RETIRE_FAILURE_THRESHOLD",
    "FallbackChatModel",
    "PersistentFallbackChatModel",
    "get_fallback_current_index",
    "install_fallback_notice_sink",
    "reset_fallback_state",
    "set_fallback_current_index",
]
