FROM nextflow/nextflow@sha256:83bbf3dd9e84ecd4d53a86620d08fe0259bca00a8f59f6f31664eb90033f460f

ENV NXF_HOME=/.nextflow

RUN nextflow plugin install nf-google@1.27.3 \
    && test -d /.nextflow/plugins/nf-google-1.27.3 \
    && test "$(nextflow -version | awk '/version/{print $2; exit}')" = "26.04.6"

WORKDIR /opt/m33

COPY bin/m33_batch_postflight.py bin/m33_gcs_append_only.py bin/m33_infra_kat_contract.py \
     bin/m33_storage_policy.py bin/m33_tabix_kat.py bin/m33_tabix_kat_cloud_runner.py \
     bin/m33_tabix_kat_controller.py ./bin/
COPY conf/m33_infra_kat_authorization.json conf/m33_m0_materializer_contract.json \
     conf/m33_storage_namespace_policy.json conf/m33_tabix_kat.config \
     conf/m33_tabix_kat_source_auth.json ./conf/
COPY modules/33_TABIX_KAT.nf ./modules/
COPY workflows/m33_tabix_kat.nf ./workflows/
COPY containers/m33-controller/Dockerfile ./containers/m33-controller/Dockerfile
COPY containers/m33-tabix/Dockerfile ./containers/m33-tabix/Dockerfile
COPY conf/gcp/m33_batch_submitter_role.yaml ./conf/gcp/m33_batch_submitter_role.yaml

ENTRYPOINT ["python3", "/opt/m33/bin/m33_tabix_kat_controller.py"]
