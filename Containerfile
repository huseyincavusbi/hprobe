FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-venv && \
    rm -rf /var/lib/apt/lists/*

RUN uv venv /opt/venv

# Install torch with CUDA 12 support explicitly to avoid version detection bugs
RUN uv pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

WORKDIR /app
COPY . .

# Install local package
RUN uv pip install .

RUN mkdir /data /results
WORKDIR /workspace

ENTRYPOINT ["hprobes"]
