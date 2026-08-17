FROM python:3.13-slim-bookworm@sha256:00faa2debb87529f9f0764e9491d8ba400a3678976616c3bd7cb193745ac20d1

ARG GNOMIX_COMMIT=e758f2c8108b660c6321c6a1b168b3a79e85a02d

RUN apt-get update \
    && apt-get install -y --no-install-recommends bcftools ca-certificates git tabix \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/AI-sandbox/gnomix.git /opt/gnomix \
    && git -C /opt/gnomix checkout --detach "${GNOMIX_COMMIT}" \
    && test "$(git -C /opt/gnomix rev-parse HEAD)" = "${GNOMIX_COMMIT}"

COPY containers/m28c-gnomix/requirements.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/requirements.lock \
    && cd /opt/gnomix \
    && python -c "import allel, sklearn, xgboost; from src.utils import read_vcf"

# Nextflow uses ps from procps to collect task-level CPU and memory metrics.
RUN apt-get update \
    && apt-get install -y --no-install-recommends procps \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=/opt/gnomix
WORKDIR /opt/gnomix

LABEL org.opencontainers.image.source="https://github.com/AI-sandbox/gnomix" \
      org.opencontainers.image.revision="e758f2c8108b660c6321c6a1b168b3a79e85a02d" \
      org.opencontainers.image.title="DNABR M28C Gnomix audit environment"
