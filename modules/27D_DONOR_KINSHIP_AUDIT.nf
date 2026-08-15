nextflow.enable.dsl=2

process WRITE_DONOR_KINSHIP_RUN_PROVENANCE {
    tag "donor_kinship_provenance"
    publishDir "${params.donor_kinship_results_dir}", mode: 'copy', overwrite: false
    cpus 1
    memory '1 GB'
    time '10m'

    input:
    val run_provenance_b64
    val phase
    val authorization_requested
    path preregistration
    path run_provenance_py

    output:
    path "run_provenance.json"

    script:
    """
    set -euo pipefail
    python3 ${run_provenance_py} \
      --base-b64 '${run_provenance_b64}' \
      --phase '${phase}' \
      --authorization-requested '${authorization_requested}' \
      --preregistration ${preregistration} \
      --out run_provenance.json
    """
}

process PREPARE_DONOR_KINSHIP_RESOURCES {
    tag "m27d_prepare_official_panel"
    publishDir "${params.donor_kinship_results_dir}/preparation", mode: 'copy', overwrite: false
    cpus params.donor_kinship_prepare_cpus
    memory params.donor_kinship_prepare_memory
    time params.donor_kinship_prepare_time

    input:
    path panel_vcfs
    path metadata
    path exclude_bed
    path preregistration
    path sample_strata_py
    path prepare_resources_r
    path bridge_py
    path manifest_py
    val provenance_b64

    output:
    path "m27d_sample_strata_summary.json", emit: strata_summary
    path "m27d_marker_preparation.json", emit: preparation_summary
    path "m27d_marker_qc.tsv", emit: marker_qc
    path "private/m27d_sample_strata.private.tsv", emit: private_strata
    path "private/m27d_official_panel_autosomes.gds", emit: private_gds
    path "private/m27d_ld_pruned_anchor_snp_ids.rds", emit: private_anchor
    path "private/m27d_ld_pruned_strict_snp_ids.rds", emit: private_strict
    path "m27d_marker_preparation.manifest.json", emit: manifest

    script:
    def inputArgs = panel_vcfs.collect { "--input ${it}" }.join(' \\\n      ')
    """
    set -euo pipefail
    mkdir -p private

    PYTHONPATH=. python3 ${sample_strata_py} \
      --panel-vcf ${panel_vcfs[0]} \
      --metadata ${metadata} \
      --private-out m27d_sample_strata.private.tsv \
      --summary-out m27d_sample_strata_summary.json \
      --suppress-below 5

    Rscript ${prepare_resources_r} \
      --panel-vcfs '${panel_vcfs.join(',')}' \
      --exclude-bed ${exclude_bed} \
      --preregistration ${preregistration} \
      --threads ${params.donor_kinship_prepare_cpus} \
      --outdir .

    mv m27d_sample_strata.private.tsv private/
    mv m27d_official_panel_autosomes.gds private/
    mv m27d_ld_pruned_anchor_snp_ids.rds private/
    mv m27d_ld_pruned_strict_snp_ids.rds private/

    python3 ${manifest_py} \
      --stage M27D_DONOR_KINSHIP_MARKER_PREPARATION \
      ${inputArgs} \
      --input ${metadata} \
      --input ${exclude_bed} \
      --input ${preregistration} \
      --input ${sample_strata_py} \
      --input ${prepare_resources_r} \
      --input ${bridge_py} \
      --output m27d_sample_strata_summary.json \
      --output m27d_marker_preparation.json \
      --output m27d_marker_qc.tsv \
      --output private/m27d_sample_strata.private.tsv \
      --output private/m27d_official_panel_autosomes.gds \
      --output private/m27d_ld_pruned_anchor_snp_ids.rds \
      --output private/m27d_ld_pruned_strict_snp_ids.rds \
      --provenance-b64 ${provenance_b64} \
      --params-json '{"scope":"m27d_marker_preparation","scientific_result":false}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref ../run_provenance.json \
      --out m27d_marker_preparation.manifest.json
    """
}

process BENCHMARK_DONOR_KINSHIP_RESOURCES {
    tag "m27d_pcrelate_resource_smoke"
    publishDir "${params.donor_kinship_results_dir}/resource_smoke", mode: 'copy', overwrite: false
    cpus params.donor_kinship_benchmark_cpus
    memory params.donor_kinship_benchmark_memory
    time params.donor_kinship_benchmark_time

    input:
    path prepared_gds
    path anchor_rds
    path strict_rds
    path metadata_strata
    path preparation_manifest
    val preparation_manifest_sha256
    path preregistration
    path pcrelate_smoke_r
    path verify_prepared_py
    path manifest_py
    val provenance_b64

    output:
    path "m27d_resource_smoke.json", emit: summary
    path "m27d_resource_smoke.tsv", emit: benchmark
    path "m27d_prepared_input_verification.json", emit: input_verification
    path "m27d_resource_smoke.manifest.json", emit: manifest

    script:
    """
    set -euo pipefail

    python3 ${verify_prepared_py} \
      --manifest ${preparation_manifest} \
      --expected-manifest-sha256 '${preparation_manifest_sha256}' \
      --gds ${prepared_gds} \
      --anchor-rds ${anchor_rds} \
      --strict-rds ${strict_rds} \
      --metadata-strata ${metadata_strata} \
      --out m27d_prepared_input_verification.json

    Rscript ${pcrelate_smoke_r} \
      --gds ${prepared_gds} \
      --anchor-rds ${anchor_rds} \
      --strict-rds ${strict_rds} \
      --metadata-strata ${metadata_strata} \
      --preregistration ${preregistration} \
      --thread-grid '${params.donor_kinship_thread_grid}' \
      --outdir .

    python3 ${manifest_py} \
      --stage M27D_DONOR_KINSHIP_RESOURCE_SMOKE \
      --input ${prepared_gds} \
      --input ${anchor_rds} \
      --input ${strict_rds} \
      --input ${metadata_strata} \
      --input ${preparation_manifest} \
      --input ${preregistration} \
      --input ${pcrelate_smoke_r} \
      --input ${verify_prepared_py} \
      --output m27d_prepared_input_verification.json \
      --output m27d_resource_smoke.json \
      --output m27d_resource_smoke.tsv \
      --provenance-b64 ${provenance_b64} \
      --params-json '{"scope":"m27d_pcrelate_resource_smoke_only","pcrelate":"without_KING","scientific_result":false}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref ../run_provenance.json \
      --out m27d_resource_smoke.manifest.json
    """
}

process VERIFY_DONOR_KINSHIP_PREPARED_INPUTS {
    tag "m27d_verify_prepared_inputs"
    publishDir "${params.donor_kinship_results_dir}/verification", mode: 'copy', overwrite: false
    cpus 1
    memory '4 GB'
    time '30m'

    input:
    path prepared_gds
    path anchor_rds
    path strict_rds
    path preparation_manifest
    val preparation_manifest_sha256
    path verify_prepared_py
    path reused_training_set
    val reused_training_set_sha256

    output:
    path "m27d_prepared_input_verification.json", emit: verification

    script:
    // The training-set arguments are added only when a phase actually reuses one. Passing
    // them always would make the phase that produces the training set verify a file it has
    // not written yet.
    def reuseArgs = reused_training_set_sha256
        ? "--training-set ${reused_training_set} --expected-training-set-sha256 '${reused_training_set_sha256}'"
        : ''
    """
    set -euo pipefail

    python3 ${verify_prepared_py} \
      --manifest ${preparation_manifest} \
      --expected-manifest-sha256 '${preparation_manifest_sha256}' \
      --gds ${prepared_gds} \
      --anchor-rds ${anchor_rds} \
      --strict-rds ${strict_rds} \
      ${reuseArgs} \
      --out m27d_prepared_input_verification.json
    """
}

process COMPARE_DONOR_KINSHIP_PC_COUNT {
    tag "m27d_pc_count_sensitivity"
    publishDir "${params.donor_kinship_results_dir}/pc_sensitivity", mode: 'copy', overwrite: false
    cpus 2
    memory '16 GB'
    time '1h'

    input:
    path configuration_pairs
    path configuration_inbreeding
    path configuration_summaries
    path strata
    path sample_universe
    path call_rates
    val reference_configuration
    path preregistration
    path comparison_py
    path kinship_graph_py
    path manifest_py
    val provenance_b64

    output:
    path "m27d_pc_count_sensitivity.json", emit: summary
    path "m27d_pc_count_sensitivity.tsv", emit: table
    path "m27d_pc_count_sensitivity.manifest.json", emit: manifest

    script:
    // The configuration identifier is recovered from each file name so the comparison can
    // never pair a table with the wrong arm's summary.
    def pairArgs = configuration_pairs.collect {
        "${it.name.replaceFirst(/^m27d_pcrelate_/, '').replaceFirst(/_pairs\.private\.tsv\.gz$/, '')}=${it}"
    }.join(' ')
    def fArgs = configuration_inbreeding.collect {
        "${it.name.replaceFirst(/^m27d_pcrelate_/, '').replaceFirst(/_inbreeding\.private\.tsv$/, '')}=${it}"
    }.join(' ')
    def summaryArgs = configuration_summaries.collect { it.toString() }.join(' ')
    """
    set -euo pipefail

    PYTHONPATH=. python3 ${comparison_py} \
      --pairs ${pairArgs} \
      --inbreeding ${fArgs} \
      --summaries ${summaryArgs} \
      --reference-configuration ${reference_configuration} \
      --samples ${sample_universe} \
      --call-rates ${call_rates} \
      --strata ${strata} \
      --preregistration ${preregistration} \
      --out-summary m27d_pc_count_sensitivity.json \
      --out-table m27d_pc_count_sensitivity.tsv

    python3 ${manifest_py} \
      --stage M27D_PC_COUNT_SENSITIVITY \
      --input ${strata} \
      --input ${sample_universe} \
      --input ${call_rates} \
      --input ${preregistration} \
      --input ${comparison_py} \
      --input ${kinship_graph_py} \
      ${configuration_pairs.collect { "--input ${it}" }.join(' \\\n      ')} \
      --output m27d_pc_count_sensitivity.json \
      --output m27d_pc_count_sensitivity.tsv \
      --provenance-b64 ${provenance_b64} \
      --params-json '{"scope":"m27d_pc_count_sensitivity","scientific_result":false,"king_executed":false,"one_factor":"n_pcs"}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref ../run_provenance.json \
      --out m27d_pc_count_sensitivity.manifest.json
    """
}

process RESOLVE_DONOR_KINSHIP_STRATA {
    tag "m27d_resolve_sample_strata"
    publishDir "${params.donor_kinship_results_dir}/strata", mode: 'copy', overwrite: false
    cpus 1
    memory '4 GB'
    time '20m'

    input:
    path panel_vcf
    path metadata
    path preregistration
    path sample_strata_py
    path bridge_py
    path manifest_py
    val provenance_b64

    output:
    path "m27d_sample_strata_summary.json", emit: summary
    path "private/m27d_sample_strata.private.tsv", emit: private_strata
    path "m27d_sample_strata.manifest.json", emit: manifest

    script:
    """
    set -euo pipefail
    mkdir -p private

    PYTHONPATH=. python3 ${sample_strata_py} \
      --panel-vcf ${panel_vcf} \
      --metadata ${metadata} \
      --private-out private/m27d_sample_strata.private.tsv \
      --summary-out m27d_sample_strata_summary.json \
      --suppress-below ${params.donor_kinship_suppress_below}

    python3 ${manifest_py} \
      --stage M27D_SAMPLE_STRATA_RESOLUTION \
      --input ${panel_vcf} \
      --input ${metadata} \
      --input ${preregistration} \
      --input ${sample_strata_py} \
      --input ${bridge_py} \
      --output m27d_sample_strata_summary.json \
      --output private/m27d_sample_strata.private.tsv \
      --provenance-b64 ${provenance_b64} \
      --params-json '{"scope":"m27d_sample_strata","scientific_result":false}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref ../run_provenance.json \
      --out m27d_sample_strata.manifest.json
    """
}

process AUDIT_BASELINE_DONOR_IDENTITY {
    tag "m27d_baseline_identity"
    publishDir "${params.donor_kinship_results_dir}/baseline_identity", mode: 'copy', overwrite: false
    cpus params.donor_kinship_pcrelate_cpus
    memory params.donor_kinship_pcrelate_memory
    time params.donor_kinship_baseline_time

    input:
    path prepared_gds
    path anchor_rds
    path strata
    path baseline_vcfs
    path preregistration
    path baseline_identity_r
    path common_r
    path manifest_py
    val provenance_b64

    output:
    path "m27d_baseline_identity.json", emit: summary
    path "private/m27d_baseline_identity.private.tsv", emit: private_table
    path "private/m27d_baseline_panel_identities.private.txt", emit: identities
    path "m27d_baseline_identity.manifest.json", emit: manifest

    script:
    """
    set -euo pipefail
    mkdir -p private bin
    cp ${common_r} bin/m27d_common.R
    cp ${baseline_identity_r} bin/\$(basename ${baseline_identity_r})

    Rscript bin/\$(basename ${baseline_identity_r}) \
      --panel-gds ${prepared_gds} \
      --baseline-vcfs '${baseline_vcfs.join(',')}' \
      --snp-rds ${anchor_rds} \
      --strata ${strata} \
      --preregistration ${preregistration} \
      --threads ${params.donor_kinship_pcrelate_threads} \
      --outdir .

    mv m27d_baseline_identity.private.tsv private/
    mv m27d_baseline_panel_identities.private.txt private/
    rm -f m27d_baseline_donors.gds

    python3 ${manifest_py} \
      --stage M27D_BASELINE_IDENTITY \
      --input ${prepared_gds} \
      --input ${anchor_rds} \
      --input ${strata} \
      --input ${preregistration} \
      --input ${baseline_identity_r} \
      --input ${common_r} \
      --output m27d_baseline_identity.json \
      --output private/m27d_baseline_identity.private.tsv \
      --output private/m27d_baseline_panel_identities.private.txt \
      --provenance-b64 ${provenance_b64} \
      --params-json '{"scope":"m27d_baseline_identity","king_executed":false,"scientific_result":false}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref ../run_provenance.json \
      --out m27d_baseline_identity.manifest.json
    """
}

process RUN_DONOR_KINSHIP_PASS0 {
    tag "m27d_pass0_pcrelate"
    publishDir "${params.donor_kinship_results_dir}/pass0", mode: 'copy', overwrite: false
    cpus params.donor_kinship_pcrelate_cpus
    memory params.donor_kinship_pcrelate_memory
    time params.donor_kinship_pass0_time

    input:
    path prepared_gds
    path anchor_rds
    path strata
    path preregistration
    path pass0_r
    path common_r
    path kinship_graph_py
    path manifest_py
    val provenance_b64

    output:
    path "m27d_pass0_pcrelate.json", emit: summary
    path "m27d_pass0_training_set.json", emit: training_summary
    path "private/m27d_pass0_training_set.private.txt", emit: training_set
    path "private/m27d_pass0_training_set_alternate_order.private.txt", emit: training_set_alternate
    path "private/m27d_pass0_sample_universe.private.txt", emit: sample_universe
    path "private/m27d_pass0_sample_call_rate.private.tsv", emit: call_rates
    path "private/m27d_pass0_related_pairs.private.tsv.gz", emit: pairs
    path "private/m27d_pass0_inbreeding.private.tsv", emit: inbreeding
    path "private/m27d_pass0_pca_scores.private.tsv.gz", emit: pca_scores
    path "m27d_pass0.manifest.json", emit: manifest

    script:
    """
    set -euo pipefail
    mkdir -p private bin
    cp ${common_r} bin/m27d_common.R
    cp ${pass0_r} bin/\$(basename ${pass0_r})

    Rscript bin/\$(basename ${pass0_r}) \
      --gds ${prepared_gds} \
      --snp-rds ${anchor_rds} \
      --strata ${strata} \
      --preregistration ${preregistration} \
      --threads ${params.donor_kinship_pcrelate_threads} \
      --outdir .

    python3 ${kinship_graph_py} \
      --pairs m27d_pass0_related_pairs.private.tsv.gz \
      --samples m27d_pass0_sample_universe.private.txt \
      --call-rates m27d_pass0_sample_call_rate.private.tsv \
      --strata ${strata} \
      --preregistration ${preregistration} \
      --stage M27D_PASS0_TRAINING_SET \
      --out-set m27d_pass0_training_set.private.txt \
      --out-alternate-set m27d_pass0_training_set_alternate_order.private.txt \
      --out-summary m27d_pass0_training_set.json

    mv m27d_pass0_inbreeding.private.tsv \
       m27d_pass0_related_pairs.private.tsv.gz \
       m27d_pass0_pca_scores.private.tsv.gz \
       m27d_pass0_sample_universe.private.txt \
       m27d_pass0_sample_call_rate.private.tsv \
       m27d_pass0_training_set.private.txt \
       m27d_pass0_training_set_alternate_order.private.txt private/

    python3 ${manifest_py} \
      --stage M27D_PASS0_PCRELATE \
      --input ${prepared_gds} \
      --input ${anchor_rds} \
      --input ${strata} \
      --input ${preregistration} \
      --input ${pass0_r} \
      --input ${common_r} \
      --input ${kinship_graph_py} \
      --output m27d_pass0_pcrelate.json \
      --output m27d_pass0_training_set.json \
      --output private/m27d_pass0_training_set.private.txt \
      --output private/m27d_pass0_training_set_alternate_order.private.txt \
      --output private/m27d_pass0_sample_universe.private.txt \
      --output private/m27d_pass0_sample_call_rate.private.tsv \
      --output private/m27d_pass0_related_pairs.private.tsv.gz \
      --output private/m27d_pass0_inbreeding.private.tsv \
      --output private/m27d_pass0_pca_scores.private.tsv.gz \
      --provenance-b64 ${provenance_b64} \
      --params-json '{"scope":"m27d_pass0","provisional":true,"king_executed":false,"scientific_result":false}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref ../run_provenance.json \
      --out m27d_pass0.manifest.json
    """
}

process FIT_DONOR_KINSHIP_PCA {
    tag "m27d_pca_${marker_set_id}"
    publishDir "${params.donor_kinship_results_dir}/pca", mode: 'copy', overwrite: false
    cpus params.donor_kinship_pcrelate_cpus
    memory params.donor_kinship_pcrelate_memory
    time params.donor_kinship_pca_time

    input:
    tuple val(marker_set_id), path(snp_rds)
    path prepared_gds
    path strata
    path training_set
    path preregistration
    path pca_r
    path common_r
    path manifest_py
    val provenance_b64

    output:
    tuple val(marker_set_id), path("private/m27d_pca_${marker_set_id}_scores.private.tsv.gz"), emit: scores
    path "m27d_pca_${marker_set_id}.json", emit: summary
    path "m27d_pca_${marker_set_id}.manifest.json", emit: manifest

    script:
    """
    set -euo pipefail
    mkdir -p private bin
    cp ${common_r} bin/m27d_common.R
    cp ${pca_r} bin/\$(basename ${pca_r})

    Rscript bin/\$(basename ${pca_r}) \
      --gds ${prepared_gds} \
      --snp-rds ${snp_rds} \
      --strata ${strata} \
      --training-set ${training_set} \
      --preregistration ${preregistration} \
      --marker-set-id ${marker_set_id} \
      --threads ${params.donor_kinship_pcrelate_threads} \
      --outdir .

    mv m27d_pca_${marker_set_id}_scores.private.tsv.gz private/

    python3 ${manifest_py} \
      --stage M27D_PCA_PROJECTION \
      --input ${prepared_gds} \
      --input ${snp_rds} \
      --input ${strata} \
      --input ${training_set} \
      --input ${preregistration} \
      --input ${pca_r} \
      --input ${common_r} \
      --output m27d_pca_${marker_set_id}.json \
      --output private/m27d_pca_${marker_set_id}_scores.private.tsv.gz \
      --provenance-b64 ${provenance_b64} \
      --params-json '{"scope":"m27d_pca","marker_set":"${marker_set_id}","fitted_on_training_set_only":true,"king_executed":false}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref ../run_provenance.json \
      --out m27d_pca_${marker_set_id}.manifest.json
    """
}

process RUN_DONOR_KINSHIP_CONFIGURATION {
    tag "m27d_pcrelate_${configuration_id}"
    publishDir "${params.donor_kinship_results_dir}/configurations", mode: 'copy', overwrite: false
    cpus params.donor_kinship_pcrelate_cpus
    memory params.donor_kinship_pcrelate_memory
    time params.donor_kinship_configuration_time

    input:
    tuple val(configuration_id), val(marker_set_id), path(snp_rds), path(pca_scores)
    path prepared_gds
    path strata
    path training_set
    path preregistration
    path configuration_r
    path common_r
    path manifest_py
    val provenance_b64

    output:
    path "private/m27d_pcrelate_${configuration_id}_pairs.private.tsv.gz", emit: pairs
    path "private/m27d_pcrelate_${configuration_id}_inbreeding.private.tsv", emit: inbreeding
    path "m27d_pcrelate_${configuration_id}.json", emit: summary
    path "m27d_pcrelate_${configuration_id}.manifest.json", emit: manifest

    script:
    """
    set -euo pipefail
    mkdir -p private bin
    cp ${common_r} bin/m27d_common.R
    cp ${configuration_r} bin/\$(basename ${configuration_r})

    zcat ${pca_scores} > pca_scores_${marker_set_id}.tsv

    Rscript bin/\$(basename ${configuration_r}) \
      --gds ${prepared_gds} \
      --snp-rds ${snp_rds} \
      --strata ${strata} \
      --training-set ${training_set} \
      --pca-scores pca_scores_${marker_set_id}.tsv \
      --preregistration ${preregistration} \
      --configuration-id ${configuration_id} \
      --marker-set-id ${marker_set_id} \
      --threads ${params.donor_kinship_pcrelate_threads} \
      --outdir .

    rm -f pca_scores_${marker_set_id}.tsv
    mv m27d_pcrelate_${configuration_id}_pairs.private.tsv.gz \
       m27d_pcrelate_${configuration_id}_inbreeding.private.tsv private/

    python3 ${manifest_py} \
      --stage M27D_PCRELATE_CONFIGURATION \
      --input ${prepared_gds} \
      --input ${snp_rds} \
      --input ${strata} \
      --input ${training_set} \
      --input ${pca_scores} \
      --input ${preregistration} \
      --input ${configuration_r} \
      --input ${common_r} \
      --output m27d_pcrelate_${configuration_id}.json \
      --output private/m27d_pcrelate_${configuration_id}_pairs.private.tsv.gz \
      --output private/m27d_pcrelate_${configuration_id}_inbreeding.private.tsv \
      --provenance-b64 ${provenance_b64} \
      --params-json '{"scope":"m27d_pcrelate","configuration":"${configuration_id}","marker_set":"${marker_set_id}","training_set_reused":true,"king_executed":false}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref ../run_provenance.json \
      --out m27d_pcrelate_${configuration_id}.manifest.json
    """
}

process SELECT_DONOR_KINSHIP_CANDIDATES {
    tag "m27d_candidate_selection"
    publishDir "${params.donor_kinship_results_dir}/candidates", mode: 'copy', overwrite: false
    cpus 2
    memory '16 GB'
    time '1h'

    input:
    path configuration_pairs
    path strata
    path sample_universe
    path call_rates
    path baseline_identities
    path stage_summaries
    path preregistration
    path selection_py
    path kinship_graph_py
    path manifest_py
    val provenance_b64

    output:
    path "m27d_candidate_selection.json", emit: summary
    path "m27d_candidate_counts.tsv", emit: public_counts
    path "m27d_gates.tsv", emit: gates
    path "private/m27d_candidate_selection.private.tsv", emit: private_table
    path "m27d_candidate_selection.manifest.json", emit: manifest

    script:
    def pairArgs = configuration_pairs.collect { it.toString() }.join(' ')
    def summaryArgs = stage_summaries.collect { it.toString() }.join(' ')
    """
    set -euo pipefail
    mkdir -p private

    PYTHONPATH=. python3 ${selection_py} \
      --pairs ${pairArgs} \
      --strata ${strata} \
      --samples ${sample_universe} \
      --call-rates ${call_rates} \
      --baseline-identities ${baseline_identities} \
      --stage-summaries ${summaryArgs} \
      --preregistration ${preregistration} \
      --suppress-below ${params.donor_kinship_suppress_below} \
      --out-private private/m27d_candidate_selection.private.tsv \
      --out-public m27d_candidate_counts.tsv \
      --out-gates m27d_gates.tsv \
      --out-summary m27d_candidate_selection.json

    python3 ${manifest_py} \
      --stage M27D_CANDIDATE_SELECTION \
      --input ${strata} \
      --input ${sample_universe} \
      --input ${call_rates} \
      --input ${baseline_identities} \
      --input ${preregistration} \
      ${configuration_pairs.collect { "--input ${it}" }.join(' \\\n      ')} \
      ${stage_summaries.collect { "--input ${it}" }.join(' \\\n      ')} \
      --input ${selection_py} \
      --input ${kinship_graph_py} \
      --output m27d_candidate_selection.json \
      --output m27d_candidate_counts.tsv \
      --output m27d_gates.tsv \
      --output private/m27d_candidate_selection.private.tsv \
      --provenance-b64 ${provenance_b64} \
      --params-json '{"scope":"m27d_candidate_selection","edge_rule":"union_across_configurations","king_executed":false}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref ../run_provenance.json \
      --out m27d_candidate_selection.manifest.json
    """
}
