nextflow.enable.dsl=2

params.plan = null
params.plan_sha256 = null
params.code_dir = "${projectDir}/../bin"
params.output_dir = null
params.max_forks = 2
params.chromosomes = '1,22'

process PREPARE_ONE_REGIONAL_PREDICTOR {
    tag predictor_id
    container 'sha256:f8c26d3769b82ef68b07ca5b67c9a6acff862bcde397addc61dc014ce8dd2896'
    cpus 1
    memory '2 GB'
    time '30m'
    maxForks params.max_forks
    publishDir "${params.output_dir}/predictors", mode: 'copy', overwrite: false,
        saveAs: { name -> "${predictor_id}/${name}" }

    input:
    tuple val(predictor_id), path(request), val(request_sha256)
    path predictor_code

    output:
    tuple val(predictor_id), path(request), path('prepared/receipt.json'),
          path('prepared/regional-source.npz')

    script:
    """
    python3 r02_lai_regional_channel_bridge.py \
      --request '${request}' --request-sha256 '${request_sha256}' --output-dir prepared
    """
}

process LINK_FIVE_CONDITIONS_WITHOUT_PREDICTOR_COPY {
    tag link_id
    container 'sha256:f8c26d3769b82ef68b07ca5b67c9a6acff862bcde397addc61dc014ce8dd2896'
    cpus 1
    memory '2 GB'
    time '30m'
    maxForks params.max_forks
    publishDir "${params.output_dir}/links", mode: 'copy', overwrite: false,
        saveAs: { name -> "${link_id}/${name}" }

    input:
    tuple val(predictor_id), path(regional_request), path(regional_receipt), path(regional_batch),
          val(link_id), path(template), val(template_sha256)
    path link_code

    output:
    tuple val(link_id), path('link-request.json'), path('linked/receipt.json')

    script:
    """
    python3 r02_lai_factorized_regional_plan.py finalize-link \
      --template '${template}' --template-sha256 '${template_sha256}' \
      --regional-request '${regional_request}' --regional-receipt '${regional_receipt}' \
      --regional-batch '${regional_batch}' --request-output link-request.json \
      --output-dir linked
    test ! -e linked/batch.npz
    """
}

workflow {
    if (!params.plan || !params.plan_sha256 || !params.output_dir)
        error 'plan, plan_sha256, and output_dir are required'
    def planFile = file(params.plan, checkIfExists: true)
    def digest = java.security.MessageDigest.getInstance('SHA-256')
    if (digest.digest(planFile.bytes).encodeHex().toString() != params.plan_sha256)
        error 'factorized regional plan SHA-256 differs'
    def plan = new groovy.json.JsonSlurper().parse(planFile.toFile())
    if (plan.schema != 'r02_lai_factorized_regional_request_plan_v1' ||
        plan.status != 'ONE_PREDICTOR_PER_REGION_RECIPE_WITH_FIVE_METADATA_LINKS_PREPARE_ONLY' ||
        plan.predictor_count != 1152 || plan.link_count != 5760 ||
        plan.large_predictor_array_copies_per_condition != 0 ||
        plan.training_authorized != false || plan.selection_authorized != false ||
        plan.evaluate_opened != false)
        error 'factorized regional plan differs'
    def predictorIds = plan.predictors.collect { it.predictor_id }
    def linkIds = plan.links.collect { it.link_id }
    if (predictorIds.toSet().size() != 1152 || linkIds.toSet().size() != 5760 ||
        plan.links.groupBy { it.predictor_id }.values().any { rows ->
            rows.size() != 5 || rows.collect { it.condition }.toSet() != plan.conditions.toSet()
        })
        error 'predictor/link cardinality differs'

    def selectedChromosomes = params.chromosomes.toString().split(',').collect { it.trim() }.findAll { it }.toSet()
    if (selectedChromosomes.isEmpty() || !['1', '22'].containsAll(selectedChromosomes))
        error 'chromosomes must be a nonempty comma-separated subset of 1,22'
    def selectedPredictors = plan.predictors.findAll { selectedChromosomes.contains(it.chromosome) }
    def selectedLinks = plan.links.findAll { selectedChromosomes.contains(it.chromosome) }
    def expectedPredictors = 576 * selectedChromosomes.size()
    def expectedLinks = 5 * expectedPredictors
    if (selectedPredictors.size() != expectedPredictors || selectedLinks.size() != expectedLinks ||
        selectedLinks.groupBy { it.predictor_id }.keySet() != selectedPredictors.collect { it.predictor_id }.toSet() ||
        selectedLinks.groupBy { it.predictor_id }.values().any { rows ->
            rows.size() != 5 || rows.collect { it.condition }.toSet() != plan.conditions.toSet()
        })
        error 'selected chromosome predictor/link cardinality differs'

    predictor_rows = Channel.fromList(selectedPredictors).map { row ->
        tuple(row.predictor_id, file(row.request.path, checkIfExists: true), row.request.sha256)
    }
    link_rows = Channel.fromList(selectedLinks).map { row ->
        tuple(row.predictor_id, row.link_id,
              file(row.template.path, checkIfExists: true), row.template.sha256)
    }
    def predictorCode = [
        file(params.code_dir + '/r02_lai_regional_channel_bridge.py', checkIfExists: true),
        file(params.code_dir + '/r02_lai_regional_inputs.py', checkIfExists: true),
        file(params.code_dir + '/r02_lai_neural_channel_adapter.py', checkIfExists: true),
        file(params.code_dir + '/r02_lai_reference_features.py', checkIfExists: true),
        file(params.code_dir + '/r02_lai_event_annotations.py', checkIfExists: true),
        file(params.code_dir + '/r02_lai_cnn_config.py', checkIfExists: true)]
    PREPARE_ONE_REGIONAL_PREDICTOR(predictor_rows, predictorCode)
    // combine(by: 0) emits the Cartesian product for each predictor key.  Since
    // each selected predictor has exactly five distinct condition rows, it emits
    // exactly five link tasks.  join(by: 0) cannot represent duplicate keys.
    joined = PREPARE_ONE_REGIONAL_PREDICTOR.out.combine(link_rows, by: 0)
    def linkCode = predictorCode + [
        file(params.code_dir + '/r02_lai_factorized_regional_plan.py', checkIfExists: true),
        file(params.code_dir + '/r02_lai_neural_input.py', checkIfExists: true),
        file(params.code_dir + '/r02_lai_control_views.py', checkIfExists: true),
        file(params.code_dir + '/r02_lai_density_control.py', checkIfExists: true),
        file(params.code_dir + '/r02_lai_query_burden.py', checkIfExists: true)]
    LINK_FIVE_CONDITIONS_WITHOUT_PREDICTOR_COPY(joined, linkCode)
}
