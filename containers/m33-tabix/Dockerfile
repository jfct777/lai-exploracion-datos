FROM python:3.13-slim-bookworm@sha256:00faa2debb87529f9f0764e9491d8ba400a3678976616c3bd7cb193745ac20d1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libhts3=1.16+ds-3 \
        procps=2:4.0.2-3 \
        tabix=1.16+ds-3 \
    && test "$(tabix --version | head -n 1)" = "tabix (htslib) 1.16" \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

LABEL org.opencontainers.image.title="DNABR M33 Tabix KAT runtime" \
      org.opencontainers.image.version="1.16" \
      org.opencontainers.image.base.digest="sha256:00faa2debb87529f9f0764e9491d8ba400a3678976616c3bd7cb193745ac20d1"
