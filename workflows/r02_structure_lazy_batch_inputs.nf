#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.package_root = null
params.requests = null
params.requests_sha256 = null
params.restore = null
params.restore_sha256 = null
params.authorization = null
params.authorization_sha256 = null
params.store_root = null
params.target_root = null
params.receipt_root = null
params.people = null
params.people_sha256 = null
params.run_root = null
params.attention_02_request_id = null
params.attention_10_request_id = null
params.measure_stage = 'FIT'
params.container = null
params.python_deps = null
params.recovered_contexts = null
params.reader_limits = null
params.reader_limits_sha256 = null

def need(name) {
    def value = params[name]
    if (value == null || value.toString().trim() == '') error "Missing --${name}"
    value
}

process PREPARE_SHARED_INDEX {
    tag '88_views_x_6_contexts_shared'
    cpus 1
    memory '2 GB'
    time '2h'
    cache 'deep'
    errorStrategy 'terminate'
    input:
    path requests
    path restore
    path authorization
    path people
    path recovered
    path reader_limits
    output:
    path 'prepared', emit: prepared
    script:
    """
    set -euo pipefail
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    PYRUN=(docker run --rm --user "\$(id -u):\$(id -g)" --cpus 1 --memory 2g \
      -v "\$PWD:\$PWD:rw" -w "\$PWD" \
      -v '${params.package_root}:${params.package_root}:ro' \
      -v '${params.python_deps}:${params.python_deps}:ro' \
      -v '${params.store_root}:${params.store_root}:ro' \
      -v '${params.target_root}:${params.target_root}:ro' \
      -v '${params.receipt_root}:${params.receipt_root}:ro' \
      -v '${params.requests}:${params.requests}:ro' \
      -v '${params.people}:${params.people}:ro' \
      -v '${params.recovered_contexts}:${params.recovered_contexts}:ro' \
      -v '${params.reader_limits}:${params.reader_limits}:ro' \
      -e 'PYTHONPATH=${params.python_deps}:${params.package_root}/bin' \
      -e USER=runner -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_runner \
      -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 \
      --entrypoint python3 '${params.container}')
    "\${PYRUN[@]}" '${params.package_root}/bin/r02_structure_lazy_batch_inputs.py' prepare \
      --requests '${params.requests}' --requests-sha256 '${params.requests_sha256}' \
      --restore '${params.restore}' --restore-sha256 '${params.restore_sha256}' \
      --authorization '${params.authorization}' --authorization-sha256 '${params.authorization_sha256}' \
      --store-root '${params.store_root}' --target-root '${params.target_root}' \
      --receipt-root '${params.receipt_root}' --people '${params.people}' \
      --people-sha256 '${params.people_sha256}' --reuse-contexts '${params.recovered_contexts}' \
      --reader-limits '${params.reader_limits}' --reader-limits-sha256 '${params.reader_limits_sha256}' \
      --output prepared
    """
}

process MATERIALIZE_REQUEST {
    tag request_id
    cpus 1
    memory '1 GB'
    time '30m'
    maxForks 1
    cache 'deep'
    errorStrategy 'terminate'
    stageInMode 'copy'
    input:
    tuple val(request_id), val(chromosome), val(core_bp), path(prepared)
    output:
    tuple val(request_id), val(chromosome), val(core_bp), path("materialized-${request_id}"), emit: entries
    script:
    """
    set -euo pipefail
    INDEX_SHA=\$(sha256sum '${prepared}/index.json' | cut -d' ' -f1)
    PYRUN=(docker run --rm --user "\$(id -u):\$(id -g)" --cpus 1 --memory 1g \
      -v "\$PWD:\$PWD:rw" -w "\$PWD" \
      -v '${params.package_root}:${params.package_root}:ro' \
      -v '${params.python_deps}:${params.python_deps}:ro' \
      -v '${params.store_root}:${params.store_root}:ro' \
      -v '${params.target_root}:${params.target_root}:ro' \
      -v '${params.people}:${params.people}:ro' \
      -e 'PYTHONPATH=${params.python_deps}:${params.package_root}/bin' \
      -e USER=runner -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_runner \
      --entrypoint python3 '${params.container}')
    "\${PYRUN[@]}" '${params.package_root}/bin/r02_structure_lazy_batch_inputs.py' materialize \
      --index '${prepared}/index.json' --index-sha256 "\$INDEX_SHA" \
      --request-id '${request_id}' --output 'materialized-${request_id}'
    """
}

process MEASURE_REAL_INTERVAL_PERSON_BATCH {
    tag request_id
    cpus 1
    memory '2 GB'
    time '30m'
    maxForks 1
    cache 'deep'
    errorStrategy 'terminate'
    stageInMode 'copy'
    input:
    tuple val(request_id), val(chromosome), val(core_bp), path(materialized), path(prepared)
    output:
    tuple val(request_id), path("${request_id}.measurement.json"), emit: measurements
    script:
    """
    set -euo pipefail
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    INDEX_SHA=\$(sha256sum '${prepared}/index.json' | cut -d' ' -f1)
    MATERIALIZED_SHA=\$(sha256sum '${materialized}/trainer-entries.json' | cut -d' ' -f1)
    PYRUN=(docker run --rm --user "\$(id -u):\$(id -g)" --cpus 1 --memory 2g \
      -v "\$PWD:\$PWD:rw" -w "\$PWD" \
      -v '${params.package_root}:${params.package_root}:ro' \
      -v '${params.python_deps}:${params.python_deps}:ro' \
      -v '${params.store_root}:${params.store_root}:ro' \
      -v '${params.target_root}:${params.target_root}:ro' \
      -e 'PYTHONPATH=${params.python_deps}:${params.package_root}/bin' \
      -e USER=runner -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_runner \
      -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 \
      --entrypoint python3 '${params.container}')
    "\${PYRUN[@]}" '${params.package_root}/bin/r02_structure_lazy_batch_inputs.py' measure \
      --index '${prepared}/index.json' --index-sha256 "\$INDEX_SHA" \
      --materialized '${materialized}/trainer-entries.json' --materialized-sha256 "\$MATERIALIZED_SHA" \
      --stage '${params.measure_stage}' --person-batch-index 0 --chromosome '${chromosome}' \
      --core-bp '${core_bp}' --output '${request_id}.measurement.json'
    """
}

process FINALIZE_PREPARATION {
    tag 'two_bounded_real_reader_intervals'
    cpus 1
    memory '512 MB'
    time '15m'
    cache 'deep'
    stageInMode 'copy'
    publishDir params.run_root, mode: 'copy', overwrite: false
    input:
    path prepared
    path materialized
    path measurements
    output:
    path 'FINAL.json'
    path 'prepared-final'
    path 'materialized-final'
    path 'measurements-final'
    script:
    """
    set -euo pipefail
    cp -a '${prepared}' prepared-final
    mkdir materialized-final measurements-final
    cp -a ${materialized} materialized-final/
    cp ${measurements} measurements-final/
    python3 '${params.package_root}/bin/r02_structure_finalize_preparation.py' staged \
      --prepared prepared-final --materialized materialized-final \
      --measurements measurements-final --published-root '${params.run_root}' \
      --output FINAL.json
    """
}

workflow {
    ['package_root','requests','requests_sha256','restore','restore_sha256','authorization',
     'authorization_sha256','store_root','target_root','receipt_root','people',
     'people_sha256','run_root','attention_02_request_id','attention_10_request_id',
     'container','python_deps','recovered_contexts','reader_limits','reader_limits_sha256'].each { need(it) }
    request_file = file(params.requests, checkIfExists:true)
    restore_file = file(params.restore, checkIfExists:true)
    authorization_file = file(params.authorization, checkIfExists:true)
    people_file = file(params.people, checkIfExists:true)
    recovered_file = file(params.recovered_contexts, checkIfExists:true)
    reader_limits_file = file(params.reader_limits, checkIfExists:true)
    PREPARE_SHARED_INDEX(request_file, restore_file, authorization_file, people_file, recovered_file,
        reader_limits_file)
    selected = Channel.fromList([[params.attention_02_request_id, '11', 46258094],
                                 [params.attention_10_request_id, '3', 86933252]])
        .combine(PREPARE_SHARED_INDEX.out.prepared)
    MATERIALIZE_REQUEST(selected)
    measurements_in = MATERIALIZE_REQUEST.out.entries
        .combine(PREPARE_SHARED_INDEX.out.prepared)
    MEASURE_REAL_INTERVAL_PERSON_BATCH(measurements_in)
    FINALIZE_PREPARATION(PREPARE_SHARED_INDEX.out.prepared,
        MATERIALIZE_REQUEST.out.entries.map{ it[3] }.collect(),
        MEASURE_REAL_INTERVAL_PERSON_BATCH.out.measurements.map{ it[1] }.collect())
}
