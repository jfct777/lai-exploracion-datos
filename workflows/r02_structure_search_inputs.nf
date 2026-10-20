#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// Metadata-only planning. This workflow never materializes genotype batches or
// starts training; the generated manifest keeps both training and EVALUATE shut.
params.views_plan = null
params.views_plan_sha256 = null
params.numeric_decision = null
params.numeric_decision_sha256 = null
params.stores = null
params.stores_sha256 = null
params.real_input_link = null
params.real_input_link_sha256 = null
params.cache = null
params.cache_sha256 = null
params.target_catalogue = null
params.target_catalogue_sha256 = null
params.outdir = null
params.code_dir = "${projectDir}/../bin"
params.python = "python3"

process PLAN_STRUCTURE_SEARCH_INPUTS {
    tag '88-views_x_6-contexts'
    cpus 1
    memory '512 MB'
    cache 'deep'
    publishDir params.outdir, mode: 'copy', overwrite: false
    input:
    path views_plan
    path numeric_decision
    path stores
    path real_input_link
    path cache_receipt
    path target_catalogue
    path planner
    output:
    path 'STRUCTURE_INPUT_REQUESTS.json'
    script:
    """
    ${params.python} ${planner} \
      --views-plan ${views_plan} --views-plan-sha256 '${params.views_plan_sha256}' \
      --numeric-decision ${numeric_decision} --numeric-decision-sha256 '${params.numeric_decision_sha256}' \
      --stores ${stores} --stores-sha256 '${params.stores_sha256}' \
      --real-input-link ${real_input_link} --real-input-link-sha256 '${params.real_input_link_sha256}' \
      --cache ${cache_receipt} --cache-sha256 '${params.cache_sha256}' \
      --target-catalogue ${target_catalogue} --target-catalogue-sha256 '${params.target_catalogue_sha256}' \
      --output STRUCTURE_INPUT_REQUESTS.json
    """
}

workflow {
    ['views_plan', 'views_plan_sha256', 'numeric_decision', 'numeric_decision_sha256',
     'stores', 'stores_sha256', 'real_input_link', 'real_input_link_sha256',
     'cache', 'cache_sha256', 'target_catalogue', 'target_catalogue_sha256', 'outdir'].each {
        if (!params[it]) error "Missing --${it}"
    }
    def files = [params.views_plan, params.numeric_decision, params.stores,
                 params.real_input_link, params.cache, params.target_catalogue].collect {
        file(it, checkIfExists: true)
    }
    PLAN_STRUCTURE_SEARCH_INPUTS(
        Channel.value(files[0]), Channel.value(files[1]), Channel.value(files[2]),
        Channel.value(files[3]), Channel.value(files[4]), Channel.value(files[5]),
        Channel.value(file("${params.code_dir}/r02_structure_search_inputs.py", checkIfExists: true)))
}
