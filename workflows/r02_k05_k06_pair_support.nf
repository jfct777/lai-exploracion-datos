nextflow.enable.dsl=2

process K05_K06_PAIR_SUPPORT {
    cpus 1
    memory '1 GB'
    time '10m'
    cache 'deep'
    publishDir params.output_dir, mode: 'copy', overwrite: false

    input:
    path plan
    val plan_sha256
    path code
    path inputs, stageAs: 'inputs/input????????'

    output:
    path 'result'

    script:
    """
    umask 077
    python3 '${code}' run --plan '${plan}' --plan-sha256 '${plan_sha256}' --output-dir result
    """
}

workflow {
    def safe = { value ->
        value instanceof String && value.startsWith('/') &&
        !value.tokenize('/').contains('..') && !value.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/)
    }
    if (![params.plan, params.source_bin, params.output_dir].every { safe.call(it) }) {
        error 'Safe absolute paths required'
    }
    def hash = { path ->
        java.security.MessageDigest.getInstance('SHA-256').digest(path.toFile().bytes).encodeHex().toString()
    }
    def plan = file(params.plan, checkIfExists: true)
    def parsed = new groovy.json.JsonSlurper().parse(plan.toFile())
    assert parsed.schema == 'r02_k05_k06_pair_support_plan_v1'
    def inputs = [file(parsed.people.path, checkIfExists: true),
                  file(parsed.community_summary.path, checkIfExists: true)]
    parsed.chromosomes.values().each { item ->
        inputs.add(file(item.contract.path, checkIfExists: true))
        inputs.add(file(item.intervals.path, checkIfExists: true))
    }
    K05_K06_PAIR_SUPPORT(
        plan,
        hash.call(plan),
        file(params.source_bin + '/r02_k05_k06_pair_support.py', checkIfExists: true),
        inputs.unique()
    )
}
