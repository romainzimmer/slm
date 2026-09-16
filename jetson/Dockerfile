ARG BASE_IMAGE=whitesscott/l4t-jetpack:r39.2.1
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PIP_BREAK_SYSTEM_PACKAGES=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
RUN mkdir -p src

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu132
RUN pip3 install --no-cache-dir --default-timeout=86400 \
    torch \
    --index-url ${TORCH_INDEX}
RUN pip3 install --no-cache-dir --default-timeout=86400 .

COPY src ./src
RUN pip3 install --no-cache-dir --no-deps .

CMD ["python3", "-m", "train", "--help"]
