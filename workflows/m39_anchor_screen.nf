nextflow.enable.dsl=2

process M39_ANCHOR_TRAIN {
    tag "${config.id}"
    publishDir "${params.m39_output_dir}/training", mode: 'copy', overwrite: false
    container params.m39_image
    cpus 2
    memory '3 GB'
    time '2h'
    maxForks 2
    input:
    tuple val(config), path(features)
    path development
    path sources
    output:
    path "${config.id}", emit: trained
    script:
    """
    PYTHONPATH=. python3 m39_anchor_screen.py --features '${features}' \\
      --development '${development}' --config-json '${groovy.json.JsonOutput.toJson(config)}' \\
      --outdir '${config.id}'
    """
}

process M39_ANCHOR_ADAPT {
    publishDir params.m39_output_dir, mode: 'copy', overwrite: false
    container params.m39_image
    input:
    path cases
    path sources
    output:
    path 'additional.plan.json'
    script:
    """
    PYTHONPATH=. python3 m39_anchor_plan.py --mode adapt \\
      --results ${cases.collect { "'${it}'" }.join(' ')} --output additional.plan.json
    """
}

process M39_ANCHOR_LOCK {
    publishDir params.m39_output_dir, mode: 'copy', overwrite: false
    container params.m39_image
    input:
    path cases
    path sources
    output:
    path 'selection.lock.json'
    script:
    """
    PYTHONPATH=. python3 m39_anchor_plan.py --mode lock \\
      --results ${cases.collect { "'${it}'" }.join(' ')} --output selection.lock.json
    """
}

process M39_ANCHOR_SCORE {
    publishDir params.m39_output_dir, mode: 'copy', overwrite: false
    container params.m39_image
    cpus 2
    memory '3 GB'
    time '45m'
    input:
    path selection_lock
    path cases
    path features
    path score_partition
    path sources
    output:
    path 'scored'
    script:
    """
    PYTHONPATH=. python3 m39_anchor_score.py --lock '${selection_lock}' \\
      --cases ${cases.collect { "'${it}'" }.join(' ')} \\
      --features ${features.collect { "'${it}/features.npz'" }.join(' ')} \\
      --score '${score_partition}' --outdir scored
    """
}

workflow INITIAL {
    take: configurations; development; sources
    main: M39_ANCHOR_TRAIN(configurations, development, sources)
    emit: trained = M39_ANCHOR_TRAIN.out.trained
}

workflow ADAPTIVE {
    take: configurations; development; sources
    main: M39_ANCHOR_TRAIN(configurations, development, sources)
    emit: trained = M39_ANCHOR_TRAIN.out.trained
}

workflow {
    if (!params.m39_output_dir || !params.m39_plan || !params.m39_binding_dir || !params.m39_feature_dir)
        error 'Output, plan, binding and feature directories are required'
    if (file(params.m39_output_dir).exists() && !workflow.resume) error 'Use a fresh output directory'
    def plan = new groovy.json.JsonSlurper().parseText(file(params.m39_plan).text)
    if (plan.cases.size() != (params.m39_profile_only ? 1 : 12)) error 'Unexpected initial case count'
    def sources = ['m39_carrier_models.py', 'm39_anchor_screen.py', 'm39_anchor_plan.py', 'm39_anchor_score.py'].collect {
        file("${projectDir.resolve('..')}/bin/${it}", checkIfExists: true)
    }
    def toCase = { config ->
        tuple(config, file("${params.m39_feature_dir}/radius_${config.radius_cm}cm/features.npz", checkIfExists: true))
    }
    def development = channel.value(file("${params.m39_binding_dir}/development.npz", checkIfExists: true))
    INITIAL(channel.fromList(plan.cases.collect(toCase)), development, channel.value(sources))
    if (!params.m39_profile_only) {
        def first = INITIAL.out.trained.collect()
        M39_ANCHOR_ADAPT(first, channel.value(sources))
        def extra = M39_ANCHOR_ADAPT.out.flatMap { path ->
            new groovy.json.JsonSlurper().parseText(path.text).cases.collect(toCase)
        }
        ADAPTIVE(extra, development, channel.value(sources))
        def allCases = INITIAL.out.trained.mix(ADAPTIVE.out.trained).collect()
        M39_ANCHOR_LOCK(allCases, channel.value(sources))
        def features = [.05, .2, .5].collect {
            file("${params.m39_feature_dir}/radius_${it}cm", checkIfExists: true)
        }
        // SCORE is staged only in this terminal process, after the immutable lock.
        M39_ANCHOR_SCORE(M39_ANCHOR_LOCK.out, allCases, channel.value(features),
            channel.value(file("${params.m39_binding_dir}/score.npz", checkIfExists: true)), channel.value(sources))
    }
}
