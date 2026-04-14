FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
UV_LINK_MODE=copy \
UV_PYTHON_DOWNLOADS=never
RUN --mount=type=cache,target=/root/.cache/uv \
--mount=type=bind,source=uv.lock,target=uv.lock \
--mount=type=bind,source=pyproject.toml,target=pyproject.toml \
uv sync --frozen --no-install-project --no-dev
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
uv sync --frozen --no-dev

FROM python:3.13-slim-bookworm AS runtime
RUN apt-get update && \
apt-get install -y --no-install-recommends libgomp1 && \
rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONUNBUFFERED=1
COPY --from=builder /app /app
EXPOSE 8080
CMD [ "uvicorn","app:app","--host","0.0.0.0","--port","8080" ]