FROM us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/m28-lai-sim@sha256:06e7b70a2484de19c73a18a6fc9ddefbfa1599e43cbcdcd60fd185495dfb9a19

USER root

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCHINDUCTOR_CACHE_DIR=/tmp/m39-torch-cache \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# The complete CUDA wheel closure is pinned by version and SHA-256. This base has
# no torch installation; the lock nevertheless requires the explicit +cu126 build.
COPY requirements.lock /opt/m39-image/requirements.lock
COPY verify_image.py /opt/m39-image/verify_image.py
RUN python3 -m pip install --no-cache-dir --index-url https://pypi.org/simple --no-deps \
      --require-hashes --only-binary=:all: \
      --requirement /opt/m39-image/requirements.lock \
    && python3 -m pip check \
    && python3 /opt/m39-image/verify_image.py

USER runner
WORKDIR /work

LABEL org.opencontainers.image.title="DNABR M39 ordered CUDA runtime" \
      org.opencontainers.image.version="2.12.1-cu126-py311"
