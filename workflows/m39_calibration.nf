nextflow.enable.dsl = 2

process M39_CALIBRATION_FIT_SELECT {
    publishDir params.m39cal_output_dir, mode: 'copy', overwrite: false
    container params.m39cal_image

    input:
    path development, stageAs: 'development.npz'
    path plan, stageAs: 'plan.json'
    path source_code, stageAs: 'm39_calibration.py'

    output:
    path 'fit', emit: fitted

    script:
    """
    python3 '${source_code}' fit-select \\
      --development '${development}' --plan '${plan}' --outdir fit
    test -f fit/lock.json
    """
}

process M39_CALIBRATION_SCORE {
    publishDir params.m39cal_output_dir, mode: 'copy', overwrite: false
    container params.m39cal_image

    input:
    path fitted, stageAs: 'fit'
    path score_partition, stageAs: 'score.npz'
    path comparators, stageAs: 'comparators'
    path source_code, stageAs: 'm39_calibration.py'

    output:
    path 'scored', emit: scored

    script:
    """
    python3 '${source_code}' score \\
      --lock '${fitted}/lock.json' --score '${score_partition}' \\
      --comparators-manifest '${comparators}/manifest.json' --outdir scored
    """
}

workflow {
    def required = [
        'm39cal_plan', 'm39cal_development', 'm39cal_score',
        'm39cal_comparators_dir', 'm39cal_output_dir', 'm39cal_image'
    ]
    required.each { name ->
        if (!params[name]) error "--${name} is required"
    }
    if (!(params.m39cal_image.toString() ==~ /.+@sha256:[a-f0-9]{64}/))
        error 'The calibration image must be pinned by SHA256 digest'
    if (file(params.m39cal_output_dir).exists() && !workflow.resume)
        error 'Use a fresh output directory'

    def sourceCode = file("${projectDir.resolve('..')}/bin/m39_calibration.py", checkIfExists: true)
    def development = file(params.m39cal_development, checkIfExists: true)
    def plan = file(params.m39cal_plan, checkIfExists: true)
    def scorePartition = file(params.m39cal_score, checkIfExists: true)
    def comparators = file(params.m39cal_comparators_dir, checkIfExists: true)
    if (!comparators.isDirectory()) error '--m39cal_comparators_dir must be a directory'
    if (!comparators.resolve('manifest.json').isFile())
        error 'The comparator directory must contain its frozen manifest.json'

    // Only these three files enter FIT; no whole binder, SCORE or M39 model sources.
    M39_CALIBRATION_FIT_SELECT(
        channel.value(development), channel.value(plan), channel.value(sourceCode)
    )
    // SCORE and immutable comparators are staged only after FIT writes its lock.
    // The launcher flattens and hashes the manifest before freezing the plan.
    M39_CALIBRATION_SCORE(
        M39_CALIBRATION_FIT_SELECT.out.fitted,
        channel.value(scorePartition), channel.value(comparators), channel.value(sourceCode)
    )
}
