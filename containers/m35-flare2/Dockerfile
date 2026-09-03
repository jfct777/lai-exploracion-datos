FROM eclipse-temurin:17-jre-jammy@sha256:1e38389ecd9e5c444e40d4385be4a4a5f56a836a38a6e41c099810c32ec1c595

ARG FLARE_SHA256=8c804341b555f302591b12cd72e870b1ca7849055d1dcd2b5cfa09b725bd9420
ARG CREATE_MODEL_FILE_SHA256=299b58d1209759bf2ae68e745c74d05f8fd07d4e3caad34cf01e98d00ecf689e
ARG MODEL_WRAPPER_SHA256=37232ed1d5d6497ba91bcee0f0ee0880190902041a145b49fc5c6adcbe5b8854

LABEL org.opencontainers.image.title="DNABR M35 FLARE2 runtime" \
      org.opencontainers.image.description="Pinned FLARE 0.6 and model builder for paired M35 inference" \
      org.opencontainers.image.version="0.6.0"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         python3 python3-pip ca-certificates \
    && python3 -m pip install --no-cache-dir \
         numpy==1.26.4 scipy==1.13.1 scikit-learn==1.5.2 \
         joblib==1.4.2 threadpoolctl==3.5.0 \
    && rm -rf /var/lib/apt/lists/*

COPY flare.jar /opt/flare/flare.jar
COPY create_model_file.py /opt/flare/create_model_file.py
COPY m35_create_model_wrapper.py /opt/flare/m35_create_model_wrapper.py

RUN echo "${FLARE_SHA256}  /opt/flare/flare.jar" | sha256sum --check --strict \
    && echo "${CREATE_MODEL_FILE_SHA256}  /opt/flare/create_model_file.py" | sha256sum --check --strict \
    && echo "${MODEL_WRAPPER_SHA256}  /opt/flare/m35_create_model_wrapper.py" | sha256sum --check --strict \
    && java -jar /opt/flare/flare.jar | grep -F "flare version 0.6.0 [616fcc9d4 03-Nov-2025]" \
    && python3 -c "from sklearn.mixture import GaussianMixture; GaussianMixture(3, init_params='k-means++')"

WORKDIR /work
CMD ["bash"]
