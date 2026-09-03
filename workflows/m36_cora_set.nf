nextflow.enable.dsl=2

include { M36_CORA_SET_PLAN; M36_CORA_SET_TRAIN; M36_CORA_MATERIALIZE; M36_CORA_CANONICAL_ADAPTER; M36_CORA_MATERIALIZE_TRAIN } from '../modules/36_CORA_SET'

workflow {
    if (!(params.m36_cora_mode in ['plan', 'smoke', 'materialize', 'materialize_train', 'train'])) error 'm36_cora_mode must be plan, smoke, materialize, materialize_train or train'
    if (!params.m36_cora_feature_chrom) error 'm36_cora_feature_chrom is required'
    if (!params.m36_cora_model_families) error 'm36_cora_model_families is required'
    if (params.m36_cora_mode != 'smoke' && params.m36_cora_synthetic_smoke) {
        error 'm36_cora_synthetic_smoke=true is permitted only with m36_cora_mode=smoke'
    }
    if (params.m36_cora_mode in ['materialize', 'materialize_train'] &&
        (!params.m36_cora_run_id || !(params.m36_cora_run_id ==~ /m36-cora-[a-z0-9][a-z0-9-]{2,50}/))) {
        error 'M36 materialization requires a unique m36_cora_run_id to prevent output collisions'
    }

    def repoDir = projectDir.resolve('..')
    def events = file(params.m36_cora_events, checkIfExists: true)
    def covariates = file(params.m36_cora_covariates, checkIfExists: true)
    def components = file(params.m36_cora_components, checkIfExists: true)
    def targets = file(params.m36_cora_targets, checkIfExists: true)
    def contract = file("${repoDir}/conf/m36_cora_set_preregistration.json", checkIfExists: true)
    def coraPy = file("${repoDir}/bin/m36_cora_set.py", checkIfExists: true)
    def modelsPy = file("${repoDir}/bin/m36_cora_models.py", checkIfExists: true)
    def trainPy = file("${repoDir}/bin/m36_cora_train.py", checkIfExists: true)
    def trainReceiptPy = file("${repoDir}/bin/m36_cora_train_receipt.py", checkIfExists: true)
    def materializePy = file("${repoDir}/bin/m36_cora_materialize.py", checkIfExists: true)
    def canonicalAdapterPy = file("${repoDir}/bin/m36_cora_canonical_adapter.py", checkIfExists: true)

    if (params.m36_cora_mode == 'plan') {
        M36_CORA_SET_PLAN(events, covariates, components, targets, contract, coraPy, modelsPy)
    } else if (params.m36_cora_mode in ['materialize', 'materialize_train']) {
        if (!params.m36_cora_rare_vcf || !params.m36_cora_genetic_map || !params.m36_cora_asibd_manifest || !params.m36_cora_asibd_segments) {
            error 'M36 materialize requires rare VCF, genetic map, asIBD manifest and segment files'
        }
        if (params.m36_cora_mode == 'materialize_train') {
            if (params.m36_cora_sample_metadata || params.m36_cora_pcrelate_components) {
                error 'M36 materialize_train derives both canonical adapter tables in-task; do not override one table'
            }
            M36_CORA_MATERIALIZE_TRAIN(
                file(params.m36_cora_rare_vcf, checkIfExists: true),
                file(params.m36_cora_locus_metadata ?: "${repoDir}/conf/m36_cora_no_locus_metadata.tab", checkIfExists: true),
                file(params.m36_cora_genetic_map, checkIfExists: true),
                file(params.m36_cora_canonical_metadata, checkIfExists: true),
                file(params.m36_cora_m20_feature_store, checkIfExists: true),
                file(params.m36_cora_modeling_master, checkIfExists: true),
                file(params.m36_cora_asibd_manifest, checkIfExists: true),
                channel.fromPath(params.m36_cora_asibd_segments, checkIfExists: true).collect(),
                canonicalAdapterPy, materializePy, trainPy, coraPy, modelsPy, trainReceiptPy
            )
        } else {
        def sampleMetadata
        def pcrelateComponents
        if (params.m36_cora_sample_metadata && params.m36_cora_pcrelate_components) {
            sampleMetadata = file(params.m36_cora_sample_metadata, checkIfExists: true)
            pcrelateComponents = file(params.m36_cora_pcrelate_components, checkIfExists: true)
        } else {
            def canonical = M36_CORA_CANONICAL_ADAPTER(
                file(params.m36_cora_canonical_metadata, checkIfExists: true),
                file(params.m36_cora_m20_feature_store, checkIfExists: true),
                file(params.m36_cora_modeling_master, checkIfExists: true), canonicalAdapterPy
            )
            sampleMetadata = canonical.sample_metadata
            pcrelateComponents = canonical.pcrelate_components
        }
        M36_CORA_MATERIALIZE(
            file(params.m36_cora_rare_vcf, checkIfExists: true),
            file(params.m36_cora_locus_metadata ?: "${repoDir}/conf/m36_cora_no_locus_metadata.tab", checkIfExists: true),
            file(params.m36_cora_genetic_map, checkIfExists: true),
            sampleMetadata,
            pcrelateComponents,
            file(params.m36_cora_asibd_manifest, checkIfExists: true),
            channel.fromPath(params.m36_cora_asibd_segments, checkIfExists: true).collect(), materializePy
        )
        }
    } else {
        if (params.m36_cora_mode == 'smoke' && !params.m36_cora_synthetic_smoke) {
            error 'M36 smoke requires m36_cora_synthetic_smoke=true'
        }
        if (params.m36_cora_mode == 'train' && !params.m36_cora_materialization_receipt) {
            error 'M36 train requires a M36_CORA_MATERIALIZE receipt'
        }
        if (params.m36_cora_mode == 'train' && (!params.m36_cora_loci || !params.m36_cora_carriers || !params.m36_cora_missing)) {
            error 'M36 train requires factorized loci, carrier and missing tables'
        }
        def receipt = params.m36_cora_mode == 'train' ? file(params.m36_cora_materialization_receipt, checkIfExists: true) : contract
        // Synthetic smoke generates its own inputs; these three distinct staged
        // fixtures only satisfy the isolated process interface without filename
        // collisions.  Train mode receives factorized materialized artifacts.
        def loci = params.m36_cora_mode == 'train' ? file(params.m36_cora_loci, checkIfExists: true) : events
        def carriers = params.m36_cora_mode == 'train' ? file(params.m36_cora_carriers, checkIfExists: true) : file("${repoDir}/tests/fixtures/m36_cora_factorized_loci.tab", checkIfExists: true)
        def missing = params.m36_cora_mode == 'train' ? file(params.m36_cora_missing, checkIfExists: true) : file("${repoDir}/tests/fixtures/m36_cora_factorized_metadata.tab", checkIfExists: true)
        M36_CORA_SET_TRAIN(params.m36_cora_mode, loci, carriers, missing, covariates, components, targets,
            receipt, trainPy, coraPy, modelsPy, trainReceiptPy)
    }
}
