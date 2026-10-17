# Build from the repository root, then freeze the produced image by digest.
# The base image must be supplied by immutable digest at build time.
ARG PYTHON_BASE
FROM ${PYTHON_BASE}
# Nextflow task monitoring requires ps inside the container, not just the host.
RUN apt-get update && apt-get install -y --no-install-recommends procps \
    && command -v ps
COPY containers/r02-haptools.requirements.txt /opt/r02/requirements.txt
RUN python3 -m pip install --no-cache-dir --only-binary=:all: -r /opt/r02/requirements.txt \
    && haptools --version \
    && python3 -m pip check
ENV PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKDIR /work
