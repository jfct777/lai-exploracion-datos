FROM python:3.13-slim-bookworm@sha256:00faa2debb87529f9f0764e9491d8ba400a3678976616c3bd7cb193745ac20d1

ARG GNOMIX_COMMIT=e758f2c8108b660c6321c6a1b168b3a79e85a02d
ARG GNOMIX_PATCH_SHA256=177c48acf7dc6838fae47131f91bc6e1444dbc062ac7ffa7f3e6ccc7cba03a95

RUN apt-get update \
    && apt-get install -y --no-install-recommends bcftools ca-certificates git tabix \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/AI-sandbox/gnomix.git /opt/gnomix \
    && git -C /opt/gnomix checkout --detach "${GNOMIX_COMMIT}" \
    && test "$(git -C /opt/gnomix rev-parse HEAD)" = "${GNOMIX_COMMIT}" \
    && git -C /opt/gnomix rev-parse HEAD > /opt/gnomix/GNOMIX_COMMIT \
    && chmod 0444 /opt/gnomix/GNOMIX_COMMIT

COPY containers/m28c-gnomix/requirements.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/requirements.lock \
    && cd /opt/gnomix \
    && python -c "import allel, sklearn, xgboost; from src.utils import read_vcf"

COPY containers/m28c-gnomix/patches/0001-guard-breakpoint-roundoff.patch /tmp/gnomix-breakpoint.patch
RUN echo "${GNOMIX_PATCH_SHA256}  /tmp/gnomix-breakpoint.patch" | sha256sum --check --strict \
    && git -C /opt/gnomix apply --check --unidiff-zero /tmp/gnomix-breakpoint.patch \
    && git -C /opt/gnomix apply --unidiff-zero /tmp/gnomix-breakpoint.patch \
    && printf '%s\n' "${GNOMIX_PATCH_SHA256}" > /opt/gnomix/GNOMIX_PATCH_SHA256 \
    && chmod 0444 /opt/gnomix/GNOMIX_PATCH_SHA256

COPY bin/audit_m28c_gnomix_runtime.py /tmp/audit_m28c_gnomix_runtime.py
RUN cd /opt/gnomix \
    && PYTHONPATH=/opt/gnomix python /tmp/audit_m28c_gnomix_runtime.py --output /opt/gnomix/GNOMIX_RUNTIME_AUDIT.json \
    && chmod 0444 /opt/gnomix/GNOMIX_RUNTIME_AUDIT.json

# Nextflow uses ps from procps to collect task-level CPU and memory metrics.
RUN apt-get update \
    && apt-get install -y --no-install-recommends procps \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=/opt/gnomix
WORKDIR /opt/gnomix

LABEL org.opencontainers.image.source="https://github.com/AI-sandbox/gnomix" \
      org.opencontainers.image.revision="e758f2c8108b660c6321c6a1b168b3a79e85a02d" \
      org.opencontainers.image.patch.sha256="177c48acf7dc6838fae47131f91bc6e1444dbc062ac7ffa7f3e6ccc7cba03a95" \
      org.opencontainers.image.title="DNABR M28C Gnomix audit environment" \
      org.opencontainers.image.sklearn="1.7.2"
