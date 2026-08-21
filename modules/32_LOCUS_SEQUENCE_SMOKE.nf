nextflow.enable.dsl=2

process M32_LOCUS_SEQUENCE_SMOKE {
    tag { "m32_locus_sequence_${run_id}" }
    publishDir params.m32_smoke_results_dir, mode: 'copy', overwrite: false
    cpus params.m32_smoke_cpus
    memory params.m32_smoke_memory
    time params.m32_smoke_time

    input:
    val run_id
    val seed
    path preregistration
    path contract_py
    path tensor_py
    path occupancy_py
    path smoke_py
    path config_nf
    path module_nf
    path workflow_nf
    val git_commit
    val repository_root

    output:
    tuple val(run_id),
        path("${run_id}/m32_locus_sequence.occupancy_and_invariants.json"),
        path("${run_id}/m32_locus_sequence.provenance.json"),
        path("${run_id}/m32_locus_sequence.manifest.json"),
        path("${run_id}/m32_locus_sequence.receipt.json"), emit: receipt

    script:
    """
    set -euo pipefail
    python3 ${smoke_py} \
      --preregistration ${preregistration} \
      --run-id ${run_id} --seed ${seed} \
      --git-commit ${git_commit} --repository-root ${repository_root} \
      --source bin/m32_locus_contract.py=${contract_py} \
      --source bin/m32_locus_tensor.py=${tensor_py} \
      --source bin/m32_locus_occupancy.py=${occupancy_py} \
      --source bin/m32_locus_smoke.py=${smoke_py} \
      --source conf/m32_locus_sequence_smoke_preregistration.json=${preregistration} \
      --source conf/m32_locus_sequence_smoke.config=${config_nf} \
      --source modules/32_LOCUS_SEQUENCE_SMOKE.nf=${module_nf} \
      --source workflows/m32_locus_sequence_smoke.nf=${workflow_nf} \
      --outdir .
    """
}
