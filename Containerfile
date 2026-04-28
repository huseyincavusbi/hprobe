FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir hprobes==0.4.0

RUN mkdir /data /results
WORKDIR /workspace

ENTRYPOINT ["hprobes"]
