nextflow.enable.dsl=2

params.request = null
params.request_sha256 = null
params.output_dir = null
params.source_bin = null

process VERIFY_NEURAL_TRAINING_ELIGIBILITY {
    tag 'structure-and-lai'
    cpus 1
    memory '1 GB'
    time '10m'
    cache 'deep'
    errorStrategy 'terminate'
    publishDir params.output_dir, mode: 'copy', overwrite: false

    input:
    path request
    path code

    output:
    path 'NEURAL_TRAINING_ELIGIBILITY.json'

    script:
    """
    set -euo pipefail
    PYTHONPATH=. python3 r02_neural_training_eligibility.py \
      --request '${request}' --request-sha256 '${params.request_sha256}' \
      --output NEURAL_TRAINING_ELIGIBILITY.json
    """
}

workflow {
    ['request','request_sha256','output_dir','source_bin'].each {
        if (!params[it]) error "Missing --${it}"
    }
    def request = file(params.request, checkIfExists:true)
    def code = ['r02_neural_training_eligibility.py','r02_lai_separation.py',
                'r02_lai_target_dependencies.py'].collect {
        file("${params.source_bin}/${it}", checkIfExists:true)
    }
    VERIFY_NEURAL_TRAINING_ELIGIBILITY(request, code)
}
