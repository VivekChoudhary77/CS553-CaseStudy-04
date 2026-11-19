# Prompt Enhancer

A Gradio web app that rewrites a rough prompt into a clearer, more specific one. You can choose a short, medium or long rewrite.

It was built for WPI DS/CS553 (MLOps), Group 25. Case Study 4 deploys the **API-based product** as a single Docker container to three managed platforms: **GCP Cloud Run**, **Azure Container Apps** and **AWS ECS (Fargate)**.

The app calls two hosted LLM APIs: **Google Gemini** (AI Studio) and **OpenRouter**. You pick one in the UI. If it fails (rate limit, timeout, bad key), the app automatically tries the other one, and every attempt is shown in the result. Gemini can also be given several models, which are tried in order before falling over to OpenRouter.

```
Browser ─▶ Cloud Run / Container Apps / ECS ─▶ container: Gradio app ─┬─▶ Gemini API
                                                                      └─▶ OpenRouter API
```

## Environment variables

The app reads its settings only from environment variables. Locally they come from a `.env` file; in the cloud, set them as environment variables or secrets on the service. Copy `.env.example` to start.

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | (none) | Google AI Studio key. Without it, Gemini is skipped. |
| `GEMINI_MODEL` | (none) | One model id, or a comma-separated list tried in order |
| `GEMINI_REASONING_EFFORT` | `low` | `none` / `minimal` / `low` / `medium` / `high`, sent as a thinking budget; blank omits it |
| `GEMINI_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta` | Gemini REST endpoint |
| `OPENROUTER_API_KEY` | (none) | OpenRouter key. Without it, OpenRouter is skipped. |
| `OPENROUTER_MODEL` | (none) | Any OpenRouter model id, e.g. a `:free` model |
| `OPENROUTER_REASONING_EFFORT` | `none` | Sent as `reasoning.effort`; blank omits it |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter endpoint |
| `API_TIMEOUT_S` | `20` | Timeout per backend call, in seconds |
| `MAX_OUTPUT_TOKENS` | `1024` | Cost guard: upper limit on output tokens per request |
| `HOST` | `0.0.0.0` | Address the server binds to |
| `PORT` | `7860` | Port the server listens on. Cloud Run injects its own (`8080`). |
| `LOG_LEVEL` | `INFO` | Python log level |

A backend needs both its key and its model. If only one backend is configured, the app still starts and logs a warning for the other one. If none is configured, the UI shows a clear error on Enhance.

## Run locally with uv

You need [uv](https://docs.astral.sh/uv/) and Python 3.10–3.12.

```bash
uv sync                          # install dependencies
cp .env.example .env             # add your API keys and model ids
uv run prompt-enhancer           # open http://localhost:7860
```

Optional checks:

```bash
uv run pytest                                # unit tests (no network)
uv run python scripts/smoke_test.py          # call each configured backend once
```

## Build and run the container

The clouds run `linux/amd64`, so build for that platform (this matters on Apple Silicon):

```bash
docker buildx build --platform linux/amd64 -t prompt-enhancer:latest --load .

# Default port 7860 (Azure Container Apps, AWS ECS)
docker run --rm -p 7860:7860 --env-file .env prompt-enhancer:latest

# Cloud Run style: the platform sets PORT
docker run --rm -e PORT=8080 -p 8080:8080 --env-file .env prompt-enhancer:latest
```

The image contains no keys. They are passed at run time with `--env-file` locally, or as secrets on each cloud.

## Changes from CS3 for cloud deployment

Case Study 3 ran three products side by side on a class VM with Docker Compose: the app (with a local CPU model as a third backend), Prometheus, Grafana, an nginx gateway and an ngrok tunnel. Case Study 4 deploys only the API-based product to managed container platforms, so the project was cut down to one container:

- **Local LLM removed.** The `Qwen/Qwen2.5-0.5B-Instruct` backend, its model download script and the `torch` and `transformers` dependencies are gone. The failover chain is now Gemini ↔ OpenRouter. Without torch, the image is much smaller and starts faster, which matters for scale-to-zero platforms and free-tier memory limits.
- **One process, one port.** Cloud Run and Container Apps route traffic to a single port. The Prometheus node exporter (port 9100), the in-app `prometheus_client` metrics server (port 8000) and the `entrypoint.sh` that started them are gone. The container's command now runs the app directly.
- **Self-hosted monitoring and tunnelling removed.** `docker-compose.yml`, the Prometheus and Grafana configuration, the nginx gateway and ngrok are gone. Each cloud provides HTTPS, a public URL, logs and request metrics itself.
- **Port from the environment.** The server binds to `0.0.0.0` and reads `PORT` (default `7860`), because Cloud Run injects `PORT=8080`.
- **Secrets from the environment only.** Keys are read from environment variables, never baked into the image (`.dockerignore` excludes `.env*`) and never logged. A missing key disables that backend with a warning instead of crashing the app.
- **Cost guards.** `MAX_OUTPUT_TOKENS` caps the output length of every request, and `API_TIMEOUT_S` bounds every backend call.
- **Logging to stdout.** Cloud log collectors read stdout. Every request logs each attempt, every failover (`failover from=… to=… reason=…`) and the backend that served it (`served backend=…`), with prompt lengths only, never prompt text or keys.
- **Image hardening.** The container runs as a non-root user and is built for `linux/amd64`.
- **Extensible failover order.** The order is the chosen backend first, then the rest of `BACKEND_NAMES`, so another backend (e.g. Vertex AI) can be added with one entry.

## Project layout

```
src/prompt_enhancer/
  app.py           Gradio UI and entry point
  router.py        backend order and failover
  backends/        Gemini (native REST API), OpenRouter (OpenAI-compatible API)
  prompts.py       system prompt, length presets, output cleanup
  config.py        settings from environment / .env
tests/             unit tests
scripts/           smoke test for the configured backends
Dockerfile         app image (Python 3.12 slim + uv, non-root)
```
