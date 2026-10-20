nextflow.enable.dsl=2

params.plan = null
params.script = null
params.outdir = null

process FIT_TARGET_VARIATION {
    cpus 1
    memory '3 GB'
    time '15m'
    errorStrategy 'terminate'
    publishDir params.outdir, mode: 'copy', overwrite: false

    input:
    path plan
    path script

    output:
    path 'FIT_TARGET_VARIATION.json'

    script:
    """
    python3 ${script} --plan ${plan} --output FIT_TARGET_VARIATION.json
    """
}

workflow {
    if (!params.plan || !params.script || !params.outdir) error 'Explicit plan/script/outdir required'
    FIT_TARGET_VARIATION(file(params.plan), file(params.script))
}
