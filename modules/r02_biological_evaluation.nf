// Shared consumer: both the original three-stage workflow and import-only entry
// use the same command. No annotation, study construction or training occurs here.
process R02_BIOLOGICAL_EVALUATION {
    tag 'matched_configuration_evidence'
    cache 'deep'
    cpus 1
    input:
    // Completed campaigns commonly use the same directory basename. Assign
    // unique staged names without copying or changing their immutable contents.
    path bundles, stageAs: 'segment??'
    path study, stageAs: 'study'
    path source_bin, stageAs: 'source_bin'
    path samples, stageAs: 'samples.ids'
    path kinship, stageAs: 'kinship.tsv'
    path import_contract, stageAs: 'import_contract.json'
    val import_options
    // Directory path inputs do not recursively content-hash their children in
    // Nextflow 26.04.6. Import-only callers also declare consumed files here.
    path content_inputs, stageAs: 'content_inputs/input????'
    output:
    path 'evaluation', emit: evaluation
    path 'import_receipt.json', optional: true, emit: import_receipt
    script:
    def manifests = bundles.collect { "--segment-manifest '${it}/manifest.json'" }.join(' ')
    def preflight = import_contract ? """
    python3 '${source_bin}/r02_biological_import.py' ${manifests} \
      --import-contract '${import_contract}' --expected-import-sha256 '${import_options.sha256}' \
      --chromosomes '${import_options.chromosomes}' --genome-build '${import_options.genome_build}' \
      --sample-ids '${samples}' --expected-samples ${params.expected_samples} \
      --expected-source-samples ${import_options.expected_source_samples} \
      --expected-configurations ${params.expected_configurations} \
      --pcrelate-file '${kinship}' --kinship-sha256 '${params.kinship_sha256}' \
      --study-contract '${study}/study_contract.json' \
      --thresholds '${params.kinship_thresholds}' --edge-thresholds-bp '${params.edge_thresholds_bp}' \
      --receipt import_receipt.json
    """ : ''
    """
    export PYTHONDONTWRITEBYTECODE=1
    export TMPDIR="\$PWD" SQLITE_TMPDIR="\$PWD"
    python3 '${source_bin}/preprocess_storage_guard.py' --directory "\$PWD" --minimum-free-gib ${params.min_free_gib}
    ${preflight}
    python3 '${source_bin}/r02_biological_evaluation.py' ${manifests} \
      --sample-ids '${samples}' --expected-samples ${params.expected_samples} \
      --expected-configurations ${params.expected_configurations} \
      --pcrelate-file '${kinship}' --expected-pcrelate-sha256 '${params.kinship_sha256}' \
      --study-contract '${study}/study_contract.json' \
      --thresholds '${params.kinship_thresholds}' --edge-thresholds-bp '${params.edge_thresholds_bp}' \
      --scratch-dir "\$PWD" \
      --max-database-mb ${params.max_database_mb} --min-free-disk-mb ${params.min_free_disk_mb} \
      --output-dir evaluation
    """
}
