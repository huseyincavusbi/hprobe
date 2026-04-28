FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 && \
    rm -rf /var/lib/apt/lists/*

RUN uv pip install hprobes

RUN mkdir /data /results
WORKDIR /workspace

ENTRYPOINT ["hprobes"]
