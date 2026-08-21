nextflow.enable.dsl=2

process M32_MATERIALIZE_COORDINATES {
    tag { "m32_coordinates_${root_label}" }
    publishDir { "${params.m32_occ_results_dir}/${params.m32_occ_run_id}" }, mode: 'copy', overwrite: false
    cpus params.m32_occ_cpus
    memory params.m32_occ_memory
    time params.m32_occ_time

    input:
    tuple val(root_label), val(root_seed), path(genetic_map), path(rare_sites), path(flare_vcf)
    path preregistration
    path prepare_py
    path contract_py
    path smoke_py
    path tensor_py
    path occupancy_py

    output:
    tuple val(root_label),
        path("coordinates-${root_label}/flare_grid.coordinates.tsv"),
        path("coordinates-${root_label}/rare_loci.coordinates.tsv"),
        path("coordinates-${root_label}/coordinate_materialization.json"), emit: coordinates

    script:
    """
    set -euo pipefail
    python3 ${prepare_py} \
      --preregistration ${preregistration} --root-label ${root_label} --root-seed ${root_seed} \
      --genetic-map ${genetic_map} --rare-sites ${rare_sites} --flare-vcf ${flare_vcf} \
      --outdir coordinates-${root_label}
    """
}

process M32_REAL_OCCUPANCY_SCREEN {
    tag { "m32_real_occupancy_${root_label}" }
    publishDir { "${params.m32_occ_results_dir}/${params.m32_occ_run_id}" }, mode: 'copy', overwrite: false
    cpus params.m32_occ_cpus
    memory params.m32_occ_memory
    time params.m32_occ_time

    input:
    tuple val(root_label), path(grid), path(rare), path(materialization)
    path preregistration
    path real_py
    path prepare_py
    path contract_py
    path occupancy_py
    path smoke_py
    path tensor_py
    path config_nf
    path module_nf
    path workflow_nf
    val git_commit
    val repository_root
    val nextflow_version

    output:
    tuple val(root_label),
        path("occupancy-${root_label}/m32_real_occupancy.json"),
        path("occupancy-${root_label}/m32_real_occupancy.provenance.json"),
        path("occupancy-${root_label}/m32_real_occupancy.manifest.json"),
        path("occupancy-${root_label}/m32_real_occupancy.receipt.json"), emit: reports

    script:
    """
    set -euo pipefail
    python3 ${real_py} \
      --preregistration ${preregistration} --root-label ${root_label} \
      --grid ${grid} --rare ${rare} --materialization ${materialization} \
      --git-commit ${git_commit} --repository-root ${repository_root} \
      --source bin/m32_locus_contract.py=${contract_py} \
      --source bin/m32_locus_occupancy.py=${occupancy_py} \
      --source bin/m32_locus_smoke.py=${smoke_py} \
      --source bin/m32_locus_tensor.py=${tensor_py} \
      --source bin/m32_prepare_coordinates.py=${prepare_py} \
      --source bin/m32_real_occupancy.py=${real_py} \
      --source conf/m32_real_occupancy_preregistration.json=${preregistration} \
      --source conf/m32_real_occupancy.config=${config_nf} \
      --source modules/32_REAL_OCCUPANCY.nf=${module_nf} \
      --source workflows/m32_real_occupancy.nf=${workflow_nf} \
      --nextflow-version ${nextflow_version} --outdir occupancy-${root_label}
    """
}
