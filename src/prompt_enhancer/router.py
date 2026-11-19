# Used Opus 5.5 with High Effort, for the failover logic that tries one backend after another.
# prompt: Write a pure-Python failover router with no gradio import: try the preferred backend
#   first, then the others, skip unavailable backends with their reason, treat
#   empty output as a failure, call an on_event callback for every failed and successful attempt,
#   and raise AllBackendsFailed with all attempts. Later let Gemini try several models in order.
# Case Study 4: removed the local backend; the failover order is now derived from BACKEND_NAMES
#   (so a new backend is one entry), output tokens are capped by MAX_OUTPUT_TOKENS, and each
#   failover and the serving backend are logged.

"""Failover router. Pure Python — no gradio import, so it can be unit-tested directly."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence

from prompt_enhancer.backends.base import Attempt, Backend, BackendError, EnhanceResult
from prompt_enhancer.prompts import build_system_prompt, max_tokens_for

log = logging.getLogger(__name__)

GEMINI, OPENROUTER = "Gemini", "OpenRouter"
# Default failover order. Adding a backend (e.g. Vertex AI) = one entry here + one in app.build_backends.
BACKEND_NAMES = [GEMINI, OPENROUTER]


def try_order(preferred: str) -> list[str]:
    """The preferred backend first, then the others in BACKEND_NAMES order."""
    if preferred not in BACKEND_NAMES:
        raise ValueError(f"Unknown backend: {preferred!r}")
    return [preferred, *(name for name in BACKEND_NAMES if name != preferred)]

# on_event("failed", attempt, next_backend_label_or_None) / on_event("succeeded", attempt)
EventCallback = Callable[..., None]

# A backend name may map to several backends (e.g. one per Gemini model), tried in order.
BackendGroup = Backend | Sequence[Backend]

# Failures that apply to every model of a backend: don't try its other models.
GROUP_WIDE_REASONS = ("not configured", "auth failed")


class AllBackendsFailed(Exception):
    def __init__(self, attempts: list[Attempt]) -> None:
        self.attempts = attempts
        summary = "; ".join(f"{a.display}: {a.reason}" for a in attempts)
        super().__init__(f"All backends failed — {summary}")


def _log_attempt(attempt: Attempt, input_chars: int, output_chars: int) -> None:
    log.info(
        "attempt backend=%s model=%s status=%s reason=%r latency_s=%.2f input_chars=%d output_chars=%d",
        attempt.backend,
        attempt.model or "-",
        "ok" if attempt.ok else "failed",
        attempt.reason,
        attempt.latency_s,
        input_chars,
        output_chars,
    )


def _as_list(group: BackendGroup) -> list[Backend]:
    return [group] if isinstance(group, Backend) else list(group)


def _try_queue(preferred: str, backends: Mapping[str, BackendGroup]) -> list[tuple[str, Backend, str]]:
    """Flatten the failover order into (group name, backend, display label) entries."""
    queue = []
    for name in try_order(preferred):
        if name not in backends:
            continue
        group = _as_list(backends[name])
        for backend in group:
            label = f"{backend.name} ({backend.model_id})" if len(group) > 1 else backend.name
            queue.append((name, backend, label))
    return queue


def enhance(
    user_prompt: str,
    preferred: str,
    length: str,
    temperature: float,
    backends: Mapping[str, BackendGroup],
    on_event: EventCallback | None = None,
    max_output_tokens: int | None = None,
) -> EnhanceResult:
    """Try backends in failover order; return the first success or raise AllBackendsFailed.

    `max_output_tokens` caps the length preset's token budget (cost guard).
    """
    queue = _try_queue(preferred, backends)
    system_prompt = build_system_prompt(length)
    max_tokens = max_tokens_for(length)
    if max_output_tokens is not None:
        max_tokens = min(max_tokens, max_output_tokens)
    attempts: list[Attempt] = []

    i = 0
    while i < len(queue):
        name, backend, label = queue[i]
        start = time.perf_counter()
        text = ""
        available, reason = backend.availability()
        if available:
            try:
                text = backend.generate(
                    system_prompt,
                    user_prompt,
                    max_tokens,
                    temperature,
                )
                reason = "ok" if text.strip() else "empty response"
            except BackendError as exc:
                reason = exc.reason
            except Exception as exc:  # a buggy backend must not break failover
                log.exception("Unexpected error from backend %s", name)
                reason = f"error: {type(exc).__name__}"
            latency = time.perf_counter() - start
        else:
            latency = 0.0

        ok = reason == "ok"
        attempt = Attempt(backend.name, backend.model_id, ok, reason, latency, label)
        attempts.append(attempt)
        _log_attempt(attempt, len(user_prompt), len(text) if ok else 0)

        if ok:
            log.info("served backend=%s model=%s attempts=%d", backend.name, backend.model_id or "-", len(attempts))
            if on_event:
                on_event("succeeded", attempt)
            return EnhanceResult(text.strip(), backend.name, backend.model_id, latency, attempts)

        i += 1
        if reason.startswith(GROUP_WIDE_REASONS):
            while i < len(queue) and queue[i][0] == name:
                i += 1
        next_label = queue[i][2] if i < len(queue) else None
        if next_label:
            log.warning("failover from=%s to=%s reason=%r", label, next_label, reason)
        if on_event:
            on_event("failed", attempt, next_label)

    raise AllBackendsFailed(attempts)


def format_trace(attempts: list[Attempt]) -> str:
    """e.g. 'Gemini ❌ rate limited (429) → OpenRouter ✅ 3.1 s'."""
    parts = [
        f"{a.display} ✅ {a.latency_s:.1f} s" if a.ok else f"{a.display} ❌ {a.reason}"
        for a in attempts
    ]
    return " → ".join(parts)
