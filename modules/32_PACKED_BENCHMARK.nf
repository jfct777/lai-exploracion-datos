nextflow.enable.dsl=2

process M32_AUTHENTICATE_PACKED_SOURCES {
    tag 'm32_source_auth'
    cpus 1
    memory '256 MB'
    time '5m'

    input:
    path source_auth_py
    path benchmark_py
    path ordered_linear_py
    path contract_py
    path smoke_py
    path preregistration
    path config_nf
    path module_nf
    path workflow_nf
    val git_commit
    val repository_root

    output:
    path 'm32_source_auth.json', emit: auth

    script:
    """
    set -euo pipefail
    python3 ${source_auth_py} \
      --repository-root ${repository_root} --git-commit ${git_commit} \
      --source bin/m32_source_auth.py=${source_auth_py} \
      --source bin/m32_packed_benchmark.py=${benchmark_py} \
      --source bin/m31_ordered_linear.py=${ordered_linear_py} \
      --source bin/m32_locus_contract.py=${contract_py} \
      --source bin/m32_locus_smoke.py=${smoke_py} \
      --source conf/m32_packed_benchmark_preregistration.json=${preregistration} \
      --source conf/m32_packed_benchmark.config=${config_nf} \
      --source modules/32_PACKED_BENCHMARK.nf=${module_nf} \
      --source workflows/m32_packed_benchmark.nf=${workflow_nf} \
      --output m32_source_auth.json
    """
}

process M32_MATERIALIZE_PACKED_TENSOR {
    tag { "m32_tensor_${root_label}" }
    container params.m32_pack_container_image
    containerOptions params.m32_pack_container_options
    cpus params.m32_pack_cpus
    memory params.m32_pack_memory
    time params.m32_pack_time

    input:
    tuple val(root_label), val(root_seed), path(grid), path(rare), path(sites), path(target), path(tree), path(pools), path(flare_vcf)
    path genetic_map
    path preregistration
    path benchmark_py
    path ordered_linear_py
    path contract_py
    path smoke_py
    path config_nf
    path module_nf
    path workflow_nf
    path source_auth_py
    path source_auth
    val git_commit
    val nextflow_version
    val container_image_id
    val expected_memory_bytes

    output:
    tuple val(root_label), val(root_seed),
        path('m32_tensor.private.npz'),
        path('m32_tensor_materialization.json'),
        path('m32_tensor_materialization.provenance.json'),
        path('m32_source_auth.json'), emit: tensor

    script:
    """
    set -euo pipefail
    python3 ${benchmark_py} materialize \
      --preregistration ${preregistration} --root-label ${root_label} --root-seed ${root_seed} \
      --genetic-map ${genetic_map} --grid-coordinates ${grid} --rare-coordinates ${rare} \
      --sites ${sites} --target ${target} --tree ${tree} --pools ${pools} --flare-vcf ${flare_vcf} \
      --git-commit ${git_commit} --source-auth ${source_auth} \
      --source bin/m32_source_auth.py=${source_auth_py} \
      --source bin/m32_packed_benchmark.py=${benchmark_py} \
      --source bin/m31_ordered_linear.py=${ordered_linear_py} \
      --source bin/m32_locus_contract.py=${contract_py} \
      --source bin/m32_locus_smoke.py=${smoke_py} \
      --source conf/m32_packed_benchmark_preregistration.json=${preregistration} \
      --source conf/m32_packed_benchmark.config=${config_nf} \
      --source modules/32_PACKED_BENCHMARK.nf=${module_nf} \
      --source workflows/m32_packed_benchmark.nf=${workflow_nf} \
      --nextflow-version ${nextflow_version} \
      --container-image-id ${container_image_id} --expected-memory-bytes ${expected_memory_bytes} \
      --tensor-out m32_tensor.private.npz \
      --materialization-audit m32_tensor_materialization.json \
      --materialization-provenance m32_tensor_materialization.provenance.json
    """
}

process M32_BENCHMARK_PACKED_TENSOR {
    tag { "m32_benchmark_${root_label}" }
    publishDir { "${params.m32_pack_results_dir}/${params.m32_pack_run_id}/${root_label}" }, mode:'copy', overwrite:false
    container params.m32_pack_container_image
    containerOptions params.m32_pack_container_options
    cpus params.m32_pack_cpus
    memory params.m32_pack_memory
    time params.m32_pack_time

    input:
    tuple val(root_label), val(root_seed), path(tensor), path(materialization_audit), path(materialization_provenance), path(source_auth)
    path preregistration
    path benchmark_py
    path ordered_linear_py
    path contract_py
    path smoke_py
    path config_nf
    path module_nf
    path workflow_nf
    path source_auth_py
    val git_commit
    val nextflow_version
    val container_image_id
    val expected_memory_bytes

    output:
    tuple val(root_label),
        path('m32_packed_benchmark.json'),
        path('m32_packed_benchmark.provenance.json'),
        path('m32_packed_benchmark.manifest.json'),
        path('m32_packed_benchmark.receipt.json'),
        path('m32_tensor_materialization.json'),
        path('m32_tensor_materialization.provenance.json'),
        path('m32_source_auth.json'), emit: reports

    script:
    """
    set -euo pipefail
    python3 ${benchmark_py} benchmark \
      --preregistration ${preregistration} --root-label ${root_label} --root-seed ${root_seed} \
      --tensor ${tensor} --materialization-audit ${materialization_audit} \
      --materialization-provenance ${materialization_provenance} \
      --git-commit ${git_commit} --source-auth ${source_auth} \
      --source bin/m32_source_auth.py=${source_auth_py} \
      --source bin/m32_packed_benchmark.py=${benchmark_py} \
      --source bin/m31_ordered_linear.py=${ordered_linear_py} \
      --source bin/m32_locus_contract.py=${contract_py} \
      --source bin/m32_locus_smoke.py=${smoke_py} \
      --source conf/m32_packed_benchmark_preregistration.json=${preregistration} \
      --source conf/m32_packed_benchmark.config=${config_nf} \
      --source modules/32_PACKED_BENCHMARK.nf=${module_nf} \
      --source workflows/m32_packed_benchmark.nf=${workflow_nf} \
      --nextflow-version ${nextflow_version} \
      --container-image-id ${container_image_id} --expected-memory-bytes ${expected_memory_bytes} \
      --report m32_packed_benchmark.json \
      --provenance m32_packed_benchmark.provenance.json \
      --manifest m32_packed_benchmark.manifest.json \
      --receipt m32_packed_benchmark.receipt.json
    """
}
