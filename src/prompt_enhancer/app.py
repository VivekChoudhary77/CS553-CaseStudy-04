# Used Opus 5.5 with High Effort, for creating the Gradio page and the entry point of the Prompt
#   Enhancer app.
# prompt: Build a gr.Blocks page with a status line, an input box, Backend, Length and Temperature
#   controls, Enhance and Clear buttons, a read-only output with a copy button, a 'Served by'
#   line, a fallback trace and three examples. Show a warning toast for each failed backend and a
#   success toast naming the one that answered, and bind to 0.0.0.0:7860.
# Case Study 4: removed the local backend, its loading timer and the Prometheus metrics; warn at
#   startup about unconfigured backends, show a clear error when none is configured, and pass
#   MAX_OUTPUT_TOKENS to the router.

"""Gradio UI and the `prompt-enhancer` entry point."""

from __future__ import annotations

import logging
from collections.abc import Mapping

import gradio as gr

from prompt_enhancer.backends.base import Attempt, Backend
from prompt_enhancer.backends.gemini import build_gemini_backends
from prompt_enhancer.backends.openrouter import OpenRouterBackend
from prompt_enhancer.config import Settings, load_settings, setup_logging
from prompt_enhancer.prompts import LENGTHS
from prompt_enhancer.router import (
    BACKEND_NAMES,
    GEMINI,
    OPENROUTER,
    AllBackendsFailed,
    BackendGroup,
    enhance,
    format_trace,
)

log = logging.getLogger(__name__)

MAX_INPUT_CHARS = 4000
EXAMPLES = ["write about dogs", "explain kubernetes", "email my boss about being late"]


def build_backends(settings: Settings) -> dict[str, BackendGroup]:
    return {
        GEMINI: build_gemini_backends(settings),
        OPENROUTER: OpenRouterBackend(settings),
    }


# What each backend needs; shown in the startup warning and the "nothing configured" error.
REQUIRED_ENV = {
    GEMINI: "GEMINI_API_KEY and GEMINI_MODEL",
    OPENROUTER: "OPENROUTER_API_KEY and OPENROUTER_MODEL",
}


def is_configured(group: BackendGroup) -> bool:
    members = [group] if isinstance(group, Backend) else list(group)
    return members[0].availability()[0]


def warn_unconfigured(backends: Mapping[str, BackendGroup]) -> None:
    for name in BACKEND_NAMES:
        if not is_configured(backends[name]):
            log.warning("%s backend not configured (set %s); it will be skipped", name, REQUIRED_ENV[name])
    if not any(is_configured(backends[name]) for name in BACKEND_NAMES):
        log.error("No backend is configured; every Enhance request will fail until keys are set")


def status_markdown(backends: Mapping[str, BackendGroup]) -> str:
    parts = []
    for name in BACKEND_NAMES:
        group = backends[name]
        members = [group] if isinstance(group, Backend) else list(group)
        available = is_configured(group)
        label = "configured" if available else "not configured"
        if available and len(members) > 1:
            label += f" ({len(members)} models)"
        parts.append(f"**{name}**: {'✅' if available else '❌'} {label}")
    return " · ".join(parts)


def success_message(attempts: list[Attempt]) -> str:
    winner = attempts[-1]
    msg = f"✅ Enhanced by {winner.backend} ({winner.model}) in {winner.latency_s:.1f} s"
    failed = [a.display for a in attempts if not a.ok]
    if failed:
        msg += f" after {', '.join(failed)} failed"
    return msg


def build_ui(backends: Mapping[str, BackendGroup], max_output_tokens: int | None = None) -> gr.Blocks:
    def on_enhance(prompt: str, preferred: str, length: str, temperature: float):
        prompt = (prompt or "").strip()
        if not prompt:
            gr.Warning("Enter a prompt first")
            return gr.skip(), gr.skip(), gr.skip(), ""
        if len(prompt) > MAX_INPUT_CHARS:
            gr.Warning(f"Prompt is too long ({len(prompt)} characters; max {MAX_INPUT_CHARS})")
            return gr.skip(), gr.skip(), gr.skip(), ""
        if not any(is_configured(backends[name]) for name in BACKEND_NAMES):
            needed = "; or ".join(REQUIRED_ENV[name] for name in BACKEND_NAMES)
            raise gr.Error(f"No backend is configured. Set {needed}.", duration=None)

        def on_event(kind: str, attempt: Attempt, next_backend: str | None = None) -> None:
            if kind == "failed" and next_backend:
                gr.Warning(f"⚠️ {attempt.display} failed ({attempt.reason}) — trying {next_backend}")

        try:
            result = enhance(
                prompt, preferred, length, float(temperature), backends, on_event, max_output_tokens
            )
        except AllBackendsFailed as exc:
            lines = "; ".join(f"{a.display}: {a.reason}" for a in exc.attempts)
            raise gr.Error(f"All backends failed — {lines}", duration=None) from None

        served_by = f"Served by **{result.backend}** · `{result.model}` · {result.latency_s:.1f} s"
        trace = f"Fallback trace: {format_trace(result.attempts)}"
        return result.text, served_by, trace, success_message(result.attempts)

    def show_success(message: str) -> None:
        if message:
            gr.Success(message)

    with gr.Blocks(title="Prompt Enhancer") as demo:
        gr.Markdown(
            "# Prompt Enhancer\n"
            "Paste a rough prompt and get back a clearer, more effective one. "
            "Uses Gemini or OpenRouter, with automatic failover."
        )
        gr.Markdown(status_markdown(backends))

        prompt = gr.Textbox(
            label="Your rough prompt",
            lines=5,
            placeholder="e.g. write a blog post about remote work",
        )
        with gr.Row():
            backend = gr.Radio(BACKEND_NAMES, value=GEMINI, label="Backend")
            length = gr.Radio(LENGTHS, value="Medium", label="Length")
            temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.1, label="Temperature")
        with gr.Row():
            enhance_btn = gr.Button("Enhance", variant="primary")
            clear_btn = gr.Button("Clear")

        output = gr.Textbox(
            label="Enhanced prompt", lines=10, interactive=False, buttons=["copy"]
        )
        served_by = gr.Markdown()
        trace = gr.Markdown()
        toast_msg = gr.State("")

        gr.Examples(examples=[[e] for e in EXAMPLES], inputs=[prompt])

        enhance_btn.click(
            on_enhance,
            inputs=[prompt, backend, length, temperature],
            outputs=[output, served_by, trace, toast_msg],
            api_name="enhance",
        ).success(show_success, inputs=[toast_msg], outputs=None, show_progress="hidden")

        clear_btn.click(
            lambda: ("", "", "", "", ""),
            outputs=[prompt, output, served_by, trace, toast_msg],
            show_progress="hidden",
            api_name=False,
        )
    return demo


def main() -> None:
    settings = load_settings()
    setup_logging(settings.log_level)
    log.info("Starting Prompt Enhancer: %r", settings)
    backends = build_backends(settings)
    warn_unconfigured(backends)
    demo = build_ui(backends, settings.max_output_tokens)
    demo.queue(default_concurrency_limit=2)
    demo.launch(server_name=settings.host, server_port=settings.port)


if __name__ == "__main__":
    main()
