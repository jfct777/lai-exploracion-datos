FROM us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/m28-lai-sim@sha256:06e7b70a2484de19c73a18a6fc9ddefbfa1599e43cbcdcd60fd185495dfb9a19

USER root

RUN python3 -m pip install --no-cache-dir --disable-pip-version-check \
      torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu \
    && python3 -m pip check \
    && python3 -c "import torch; assert torch.__version__.startswith('2.12.1')"

USER runner
WORKDIR /work

LABEL org.opencontainers.image.title="DNABR M33 T0a CPU runtime" \
      org.opencontainers.image.version="2.12.1-cpu"
