FROM eclipse-temurin:17-jre-jammy@sha256:1e38389ecd9e5c444e40d4385be4a4a5f56a836a38a6e41c099810c32ec1c595

ARG FLARE_SHA256=8c804341b555f302591b12cd72e870b1ca7849055d1dcd2b5cfa09b725bd9420

LABEL org.opencontainers.image.title="DNABR M30 FLARE runtime" \
      org.opencontainers.image.description="Pinned Java 17 and Python 3 runtime for the M30 direct-FLARE baseline" \
      org.opencontainers.image.version="0.6.0"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY flare.jar /opt/flare/flare.jar

RUN echo "${FLARE_SHA256}  /opt/flare/flare.jar" | sha256sum --check --strict \
    && java -jar /opt/flare/flare.jar | grep -F "flare version 0.6.0 [616fcc9d4 03-Nov-2025]"

WORKDIR /work
CMD ["bash"]
