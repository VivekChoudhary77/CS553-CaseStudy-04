# Used Opus 5.5 with High Effort, for the Docker image of the app (Case Study 3).
# prompt: Write a Dockerfile that installs the app with uv from uv.lock, keeps API keys out of the
#   image, caches the dependency layer separately from the source code, and serves on port 7860.
# Case Study 4: removed the node exporter, the model cache and entrypoint.sh; the app is the only
#   process, runs as a non-root user, and listens on $PORT (Cloud Run sets 8080; default 7860).
#   Build for the clouds with: docker buildx build --platform linux/amd64 ...

FROM python:3.12-slim

# uv, copied from its official image and pinned to the version used locally
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/

# Unprivileged user that runs the app (the files below stay owned by root: read-only for it)
RUN useradd --create-home --uid 1000 app

WORKDIR /app

# UV_COMPILE_BYTECODE: precompile .pyc files so the app starts faster
# UV_LINK_MODE=copy:    copy files out of the build cache (it's a separate mount, links can't cross it)
# UV_PYTHON_DOWNLOADS:  use the image's Python 3.12, never download another
# PATH:                 so `prompt-enhancer` from the virtualenv is found
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1

# 1) Dependencies only. This layer is rebuilt only when pyproject.toml or uv.lock change.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# 2) Our code. Editing src/ only re-runs this cheap step.
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Gradio UI (the platform may override PORT)
EXPOSE 7860

USER app
CMD ["prompt-enhancer"]
