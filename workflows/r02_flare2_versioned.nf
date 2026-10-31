nextflow.enable.dsl=2


def positiveInteger(value, name) {
    def text = value == null ? '' : value.toString()
    if( !(text ==~ /[1-9][0-9]*/) ) {
        throw new IllegalStateException("${name} must be a positive integer")
    }
    return text.toInteger()
}


def memoryMiB(value, name) {
    def text = value == null ? '' : value.toString().trim()
    def match = text =~ /(?i)^([0-9]+(?:\.[0-9]+)?)\s*(B|KB|KiB|MB|MiB|GB|GiB)$/
    if( !match.matches() ) {
        throw new IllegalStateException("${name} must be a positive byte size with an explicit unit")
    }
    def amount = match[0][1].toDouble()
    if( amount <= 0 ) {
        throw new IllegalStateException("${name} must be positive")
    }
    def unit = match[0][2].toUpperCase()
    def bytesPerUnit = [B: 1d, KB: 1e3, KIB: 1024d, MB: 1e6, MIB: 1048576d,
                        GB: 1e9, GIB: 1073741824d][unit]
    return amount * bytesPerUnit / 1048576d
}


def positiveDuration(value, name) {
    def text = value == null ? '' : value.toString().trim()
    def match = text =~ /(?i)^([0-9]+(?:\.[0-9]+)?)\s*(ms|s|m|h|d)$/
    if( !match.matches() || match[0][1].toDouble() <= 0 ) {
        throw new IllegalStateException("${name} must be a positive duration with an explicit unit")
    }
}

/*
 * Recoverable FLARE2 entrypoint.  The panel-probability, model-building and
 * final-inference processes are separate cache units.  Direct FLARE 0.6.0 is
 * run only when explicitly requested; this workflow does not treat a previous
 * direct result as comparable without a later hash/axis authentication step.
 * Cache dependencies are explicit: panel probabilities bind every prepared
 * input and code file; model building binds the plan, panels, receipt,
 * builders and code; inference binds those model products plus both VCF/TBI
 * pairs and inference inputs.  The optional direct process binds the same
 * plan and exact VCF/TBI pair but has no dependency on clustered inference.
 */
include {
    R02_FLARE2_PANEL_PROBABILITIES;
    R02_FLARE2_BUILD_MODEL;
    R02_FLARE2_CLUSTERED_INFERENCE;
    R02_FLARE_0_6_0_DIRECT_REFERENCE
} from '../modules/R02_FLARE2_VERSIONED'


workflow {
    if( params.r02_flare2_execution_scope != 'FIT_ONLY_TRANSDUCTIVE_MODEL_BUILD_NOT_EVALUATION' ) {
        throw new IllegalStateException(
            'r02_flare2_execution_scope must remain FIT_ONLY_TRANSDUCTIVE_MODEL_BUILD_NOT_EVALUATION'
        )
    }
    def runDirectText = params.r02_flare2_run_direct.toString().toLowerCase()
    if( runDirectText !in ['true', 'false'] ) {
        throw new IllegalStateException('r02_flare2_run_direct must be boolean')
    }
    def runDirect = runDirectText == 'true'
    if( params.r02_flare2_prior_direct_receipt ) {
        throw new IllegalStateException(
            'This entrypoint does not compare a prior direct result: authenticate its receipt, people, sites and probability axes in a later scoring workflow'
        )
    }
    if( !params.r02_flare2_container || !params.r02_flare2_results
            || params.r02_flare2_chromosome == null ) {
        throw new IllegalStateException(
            'Set an exact container, results directory and autosomal chromosome; the template is not an admitted human run'
        )
    }
    def chromosome = params.r02_flare2_chromosome.toString().replaceFirst('(?i)^chr', '')
    if( !(chromosome ==~ /\d+/) || chromosome.toInteger() < 1 || chromosome.toInteger() > 22 ) {
        throw new IllegalStateException("r02_flare2_chromosome must be 1..22; received ${chromosome}")
    }
    def container = params.r02_flare2_container.toString()
    if( !(container ==~ /(?:sha256:[0-9a-f]{64}|.+@sha256:[0-9a-f]{64})/) ) {
        throw new IllegalStateException('r02_flare2_container must be an immutable SHA-256 image reference')
    }

    def required = [
        config           : params.r02_flare2_config,
        reference_vcf    : params.r02_flare2_reference_vcf,
        reference_index  : params.r02_flare2_reference_index,
        target_vcf       : params.r02_flare2_target_vcf,
        target_index     : params.r02_flare2_target_index,
        direct_panel_map : params.r02_flare2_direct_panel_map,
        fine_panel_map   : params.r02_flare2_fine_panel_map,
        panel_to_ancestry: params.r02_flare2_panel_to_ancestry,
        genetic_map      : params.r02_flare2_genetic_map,
        excludemarkers   : params.r02_flare2_excludemarkers,
        flare_jar        : params.r02_flare2_flare_jar,
        official_builder : params.r02_flare2_official_builder,
        seeded_builder   : params.r02_flare2_seeded_builder,
        roles            : params.r02_flare2_roles,
        launcher         : params.r02_flare2_launcher,
        model_io         : params.r02_flare2_model_io,
        m35_runner       : params.r02_flare2_m35_runner,
        m34_runner       : params.r02_flare2_m34_runner,
    ]
    required.each { name, value ->
        if( !value ) throw new IllegalStateException("Set --r02_flare2_${name}")
    }
    def inputs = required.collectEntries { name, value ->
        [(name): file(value, checkIfExists: true)]
    }
    def specification = new groovy.json.JsonSlurper().parse(inputs.config.toFile())
    def configuredChromosome = specification.chromosome.toString().replaceFirst('(?i)^chr', '')
    if( configuredChromosome != chromosome ) {
        throw new IllegalStateException(
            "r02_flare2_chromosome=${chromosome} differs from config chromosome=${configuredChromosome}"
        )
    }
    if( specification.evaluation_claim != params.r02_flare2_execution_scope ) {
        throw new IllegalStateException('config evaluation_claim differs from the FIT-only workflow scope')
    }
    def javaHeapMiB = positiveInteger(specification.java_heap_mib, 'config java_heap_mib')
    def javaThreads = positiveInteger(
        specification.shared_parameters?.nthreads, 'config shared_parameters.nthreads'
    )
    def processCpus = positiveInteger(params.r02_flare2_cpus, 'r02_flare2_cpus')
    positiveInteger(params.r02_flare2_max_forks, 'r02_flare2_max_forks')
    def reserveMiB = positiveInteger(
        params.r02_flare2_memory_reserve_mib, 'r02_flare2_memory_reserve_mib'
    )
    if( processCpus < javaThreads ) {
        throw new IllegalStateException(
            "r02_flare2_cpus=${processCpus} is smaller than FLARE nthreads=${javaThreads}"
        )
    }
    def javaProcessMemoryMiB = memoryMiB(params.r02_flare2_memory, 'r02_flare2_memory')
    if( javaProcessMemoryMiB < javaHeapMiB + reserveMiB ) {
        throw new IllegalStateException(
            "r02_flare2_memory provides ${javaProcessMemoryMiB} MiB; require at least " +
            "java_heap_mib ${javaHeapMiB} + reserve ${reserveMiB} MiB"
        )
    }
    memoryMiB(params.r02_flare2_model_memory, 'r02_flare2_model_memory')
    positiveDuration(params.r02_flare2_panel_time, 'r02_flare2_panel_time')
    positiveDuration(params.r02_flare2_model_time, 'r02_flare2_model_time')
    positiveDuration(params.r02_flare2_inference_time, 'r02_flare2_inference_time')

    def panel = R02_FLARE2_PANEL_PROBABILITIES(
        inputs.config,
        inputs.reference_vcf,
        inputs.reference_index,
        inputs.target_vcf,
        inputs.target_index,
        inputs.direct_panel_map,
        inputs.fine_panel_map,
        inputs.panel_to_ancestry,
        inputs.genetic_map,
        inputs.excludemarkers,
        inputs.flare_jar,
        inputs.official_builder,
        inputs.seeded_builder,
        inputs.roles,
        inputs.launcher,
        inputs.model_io,
        inputs.m35_runner,
        inputs.m34_runner,
    )

    def model = R02_FLARE2_BUILD_MODEL(
        panel.plan,
        panel.panels,
        panel.receipt,
        inputs.official_builder,
        inputs.seeded_builder,
        inputs.launcher,
        inputs.model_io,
        inputs.m35_runner,
        inputs.m34_runner,
    )

    R02_FLARE2_CLUSTERED_INFERENCE(
        panel.plan,
        model.model,
        model.receipt,
        model.mapping,
        inputs.reference_vcf,
        inputs.reference_index,
        inputs.target_vcf,
        inputs.target_index,
        inputs.fine_panel_map,
        inputs.genetic_map,
        inputs.excludemarkers,
        inputs.flare_jar,
        inputs.launcher,
        inputs.model_io,
        inputs.m35_runner,
        inputs.m34_runner,
    )

    if( runDirect ) {
        R02_FLARE_0_6_0_DIRECT_REFERENCE(
            panel.plan,
            inputs.reference_vcf,
            inputs.reference_index,
            inputs.target_vcf,
            inputs.target_index,
            inputs.direct_panel_map,
            inputs.genetic_map,
            inputs.excludemarkers,
            inputs.flare_jar,
            inputs.launcher,
            inputs.model_io,
            inputs.m35_runner,
            inputs.m34_runner,
        )
    }
}
