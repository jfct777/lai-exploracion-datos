nextflow.enable.dsl=2

process M31_PRE2_VERIFY_AUTHORIZATION {
    tag 'm31_pre2_external_run_authorization'
    publishDir "${params.m31_pre2_results_dir}/authorization", mode: 'copy', overwrite: false
    container params.m31_pre2_container_image
    containerOptions params.m31_pre2_pregate_container_options
    cpus 1
    memory '2 GB'
    time '30m'

    input:
    path authorization
    path contract
    path pipeline_py
    path contract_validator
    path runner_py
    path core_py
    path receipt_py
    val run_id
    val expected_git_commit
    val container_digest
    val expected_execution_source_sha256_json
    val max_cost_usd

    output:
    path 'm31_pre2.execution_authorization.json', emit: report

    script:
    """
    set -euo pipefail
    python3 ${pipeline_py} verify-authorization \
      --authorization ${authorization} --contract ${contract} \
      --run-id '${run_id}' --expected-git-commit '${expected_git_commit}' \
      --container-digest '${container_digest}' --max-cost-usd '${max_cost_usd}' \
      --expected-execution-source-sha256-json '${expected_execution_source_sha256_json}' \
      --output m31_pre2.execution_authorization.json
    """
}

process M31_PRE2_VERIFY_TECHNICAL {
    tag 'm31_pre2_known_answers_and_pre1_c'
    publishDir "${params.m31_pre2_results_dir}/technical", mode: 'copy', overwrite: false
    container params.m31_pre2_container_image
    containerOptions params.m31_pre2_pregate_container_options
    cpus 2
    memory '4 GB'
    time '1h'

    input:
    path contract
    path contract_validator
    path pipeline_py
    path runner_py
    path core_py
    path receipt_py
    path pre1_c_checkpoint
    path pre1_c_prediction
    val expected_pre1_c_checkpoint_sha256
    val expected_pre1_c_prediction_sha256
    val expected_pre1_c_metrics_json

    output:
    path 'technical/m31_pre2.contract.json', emit: contract_report
    path 'technical/m31_pre2.known_answer.json', emit: known_answer
    path 'technical/m31_pre2.technical_evidence.json', emit: evidence

    script:
    """
    set -euo pipefail
    mkdir -p technical
    python3 ${contract_validator} --contract ${contract} \
      --output technical/m31_pre2.contract.json
    python3 ${pipeline_py} known-answer --contract ${contract} \
      --output technical/m31_pre2.known_answer.json
    python3 ${pipeline_py} verify-technical \
      --known-answer technical/m31_pre2.known_answer.json \
      --pre1-c-checkpoint ${pre1_c_checkpoint} \
      --pre1-c-prediction ${pre1_c_prediction} \
      --expected-pre1-c-checkpoint-sha256 '${expected_pre1_c_checkpoint_sha256}' \
      --expected-pre1-c-prediction-sha256 '${expected_pre1_c_prediction_sha256}' \
      --expected-pre1-c-metrics-json '${expected_pre1_c_metrics_json}' \
      --output technical/m31_pre2.technical_evidence.json
    """
}


process M31_PRE2_FIT_PREDICT {
    tag "m31_pre2_workers_${workers}"
    publishDir "${params.m31_pre2_results_dir}/workers", mode: 'copy', overwrite: false
    container params.m31_pre2_container_image
    containerOptions params.m31_pre2_pregate_container_options
    cpus { workers }
    memory params.m31_pre2_fit_memory
    time params.m31_pre2_fit_time
    maxForks params.m31_pre2_fit_max_forks

    input:
    val workers
    path contract
    path genetic_map
    tuple val(root17_label), val(root17_seed),
        path(root17_sites, stageAs: 'root17/preflight/*'),
        path(root17_target, stageAs: 'root17/preflight/*'),
        path(root17_tree, stageAs: 'root17/m28/*'),
        path(root17_pools, stageAs: 'root17/m28/*'),
        path(root17_truth, stageAs: 'root17/m28/*'),
        path(root17_flare_vcf, stageAs: 'root17/m30/*'),
        path(root17_flare_audit, stageAs: 'root17/m30/*')
    tuple val(root18_label), val(root18_seed),
        path(root18_sites, stageAs: 'root18/preflight/*'),
        path(root18_target, stageAs: 'root18/preflight/*'),
        path(root18_tree, stageAs: 'root18/m28/*'),
        path(root18_pools, stageAs: 'root18/m28/*'),
        path(root18_flare_vcf, stageAs: 'root18/m30/*'),
        path(root18_flare_audit, stageAs: 'root18/m30/*')
    path pipeline_py
    path contract_validator
    path runner_py
    path core_py
    path receipt_py
    path module_nf
    path workflow_nf
    path config_nf
    path execution_authorization
    val run_id
    val expected_contract_sha256
    val expected_runner_sha256
    val expected_core_sha256
    val expected_git_commit
    val expected_execution_source_sha256_json
    val container_digest

    output:
    tuple val(workers), path("worker-${workers}"), emit: worker_bundle

    script:
    """
    set -euo pipefail
    test '${root17_label}' = 'root17'
    test '${root17_seed}' = '20260817'
    test '${root18_label}' = 'root18'
    test '${root18_seed}' = '20260818'
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export BLIS_NUM_THREADS=1
    python3 ${pipeline_py} fit-predict \
      --contract ${contract} --genetic-map ${genetic_map} \
      --expected-contract-sha256 '${expected_contract_sha256}' \
      --expected-runner-sha256 '${expected_runner_sha256}' \
      --expected-core-sha256 '${expected_core_sha256}' \
      --expected-git-commit '${expected_git_commit}' \
      --expected-execution-source-sha256-json '${expected_execution_source_sha256_json}' \
      --container-digest '${container_digest}' --workers ${workers} \
      --run-id '${run_id}' \
      --execution-source ${contract_validator} --execution-source ${receipt_py} \
      --execution-source ${module_nf} \
      --execution-source ${workflow_nf} \
      --execution-source ${config_nf} \
      --execution-authorization ${execution_authorization} \
      --train-root17-sites ${root17_sites} --train-root17-target ${root17_target} \
      --train-root17-tree ${root17_tree} --train-root17-pools ${root17_pools} \
      --train-root17-truth ${root17_truth} \
      --train-root17-flare-vcf ${root17_flare_vcf} \
      --train-root17-flare-audit ${root17_flare_audit} \
      --eval-root18-sites ${root18_sites} --eval-root18-target ${root18_target} \
      --eval-root18-tree ${root18_tree} --eval-root18-pools ${root18_pools} \
      --eval-root18-flare-vcf ${root18_flare_vcf} \
      --eval-root18-flare-audit ${root18_flare_audit} \
      --outdir worker-${workers}
    """
}


process M31_PRE2_VERIFY_WORKERS {
    tag 'm31_pre2_workers_1_4_8_exact'
    publishDir "${params.m31_pre2_results_dir}/gate", mode: 'copy', overwrite: false
    container params.m31_pre2_container_image
    containerOptions params.m31_pre2_pregate_container_options
    cpus 2
    memory '4 GB'
    time '1h'

    input:
    path worker_dirs
    path pipeline_py
    path contract_validator
    path runner_py
    path core_py
    path receipt_py

    output:
    path 'm31_pre2.worker_screen.json', emit: screen

    script:
    """
    set -euo pipefail
    python3 ${pipeline_py} verify-workers \
      --worker-dir worker-1 --worker-dir worker-4 --worker-dir worker-8 \
      --output m31_pre2.worker_screen.json
    """
}


process M31_PRE2_ROOT17_GATE {
    tag 'm31_pre2_root17_gate'
    publishDir "${params.m31_pre2_results_dir}/gate", mode: 'copy', overwrite: false
    container params.m31_pre2_container_image
    containerOptions params.m31_pre2_pregate_container_options
    cpus 2
    memory '4 GB'
    time '1h'

    input:
    path contract
    path worker4_dir
    path worker_screen
    path technical_evidence
    path pipeline_py
    path contract_validator
    path runner_py
    path core_py
    path receipt_py
    path module_nf
    path workflow_nf
    path config_nf
    path execution_authorization
    val run_id
    val execution_deadline_utc
    val root18_reserve_seconds

    output:
    path 'm31_pre2.root17.metrics.json', emit: metrics
    path 'm31_pre2.root17.receipt.json', emit: receipt
    path 'm31_pre2.runtime_budget.json', emit: runtime_budget
    path 'm31_pre2.OPEN_ROOT18.json', optional: true, emit: open_token

    script:
    """
    set -euo pipefail
    python3 ${pipeline_py} gate \
      --contract ${contract} --runner ${runner_py} --core ${core_py} \
      --contract-code ${contract_validator} --receipt-code ${receipt_py} \
      --module ${module_nf} \
      --workflow ${workflow_nf} --config ${config_nf} \
      --execution-authorization ${execution_authorization} \
      --worker4-dir ${worker4_dir} --worker-screen ${worker_screen} \
      --technical-evidence ${technical_evidence} \
      --root17-metrics m31_pre2.root17.metrics.json \
      --run-id '${run_id}' --output m31_pre2.root17.receipt.json \
      --execution-deadline-utc '${execution_deadline_utc}' \
      --root18-reserve-seconds '${root18_reserve_seconds}' \
      --runtime-budget-output m31_pre2.runtime_budget.json \
      --open-token m31_pre2.OPEN_ROOT18.json
    """
}


process M31_PRE2_SCORE_ROOT18 {
    tag 'm31_pre2_root18_one_way_score'
    publishDir "${params.m31_pre2_results_dir}/score", mode: 'copy', overwrite: false
    container params.m31_pre2_container_image
    containerOptions params.m31_pre2_scorer_container_options
    cpus 4
    memory params.m31_pre2_score_memory
    time params.m31_pre2_score_time
    cache false
    maxRetries 0
    maxForks 1

    input:
    path open_token
    path runtime_budget
    path receipt
    path root17_metrics
    path technical_evidence
    path worker_screen
    path worker4_dir
    path contract
    path genetic_map
    tuple val(root18_label), val(root18_seed),
        path(root18_sites, stageAs: 'root18/preflight/*'),
        path(root18_target, stageAs: 'root18/preflight/*'),
        path(root18_tree, stageAs: 'root18/m28/*'),
        path(root18_pools, stageAs: 'root18/m28/*'),
        path(root18_flare_vcf, stageAs: 'root18/m30/*'),
        path(root18_flare_audit, stageAs: 'root18/m30/*')
    path pipeline_py
    path contract_validator
    path runner_py
    path core_py
    path receipt_py
    path module_nf
    path workflow_nf
    path config_nf
    path execution_authorization
    val run_id
    val execution_deadline_utc
    val min_score_remaining_seconds
    val root18_truth_source
    val opening_ledger

    output:
    path 'root18_score/m31_pre2.root18.result.json', emit: result

    script:
    """
    set -euo pipefail
    test '${root18_label}' = 'root18'
    test '${root18_seed}' = '20260818'
    python3 ${pipeline_py} score \
      --contract ${contract} --runner ${runner_py} --core ${core_py} \
      --contract-code ${contract_validator} --receipt-code ${receipt_py} \
      --module ${module_nf} \
      --workflow ${workflow_nf} --config ${config_nf} \
      --execution-authorization ${execution_authorization} \
      --worker4-dir ${worker4_dir} --receipt ${receipt} \
      --open-token ${open_token} --technical-evidence ${technical_evidence} \
      --runtime-budget ${runtime_budget} \
      --execution-deadline-utc '${execution_deadline_utc}' \
      --min-score-remaining-seconds '${min_score_remaining_seconds}' \
      --worker-screen ${worker_screen} \
      --root17-metrics ${root17_metrics} --opening-ledger '${opening_ledger}' \
      --run-id '${run_id}' --root18-truth-source '${root18_truth_source}' \
      --genetic-map ${genetic_map} --eval-root18-sites ${root18_sites} \
      --eval-root18-target ${root18_target} --eval-root18-tree ${root18_tree} \
      --eval-root18-pools ${root18_pools} --eval-root18-flare-vcf ${root18_flare_vcf} \
      --eval-root18-flare-audit ${root18_flare_audit} --outdir root18_score
    """
}
