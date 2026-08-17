FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends procps \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir --disable-pip-version-check \
      --requirement /tmp/requirements.lock \
    && python -m pip check

RUN useradd --create-home --uid 1000 runner
USER runner
WORKDIR /work
