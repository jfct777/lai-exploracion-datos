#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.package_root = null
params.prepared_root = null
params.prepared_index_sha256 = null
params.people = null
params.run_root = null
params.container = null
params.python_deps = null

def need(name) {
    def value = params[name]
    if (value == null || value.toString().trim() == '') error "Missing --${name}"
    value
}

process MATERIALIZE_ALL_LAZY_TRAINER_ENTRIES {
    tag '528_requests_fit_select'
    cpus 1
    memory '1 GB'
    time '30m'
    cache 'deep'
    errorStrategy 'terminate'
    stageInMode 'copy'
    publishDir params.run_root, mode: 'copy', overwrite: false
    input:
    path prepared
    output:
    path 'all-entries'
    script:
    """
    set -euo pipefail
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    PYRUN=(docker run --rm --user "\$(id -u):\$(id -g)" --cpus 1 --memory 1g \
      -v "\$PWD:\$PWD:rw" -w "\$PWD" \
      -v '${params.package_root}:${params.package_root}:ro' \
      -v '${params.python_deps}:${params.python_deps}:ro' \
      -v '${params.people}:${params.people}:ro' \
      -e 'PYTHONPATH=${params.python_deps}:${params.package_root}/bin' \
      -e USER=runner -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_runner \
      -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 \
      --entrypoint python3 '${params.container}')
    "\${PYRUN[@]}" '${params.package_root}/bin/r02_structure_lazy_batch_inputs.py' materialize-all \
      --index '${prepared}/index.json' --index-sha256 '${params.prepared_index_sha256}' \
      --output all-entries --published-root '${params.run_root}/all-entries'
    """
}

workflow {
    ['package_root','prepared_root','prepared_index_sha256','people','run_root',
     'container','python_deps'].each { need(it) }
    MATERIALIZE_ALL_LAZY_TRAINER_ENTRIES(file(params.prepared_root, checkIfExists:true))
}
