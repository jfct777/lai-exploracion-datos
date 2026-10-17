nextflow.enable.dsl=2

process R02_STRUCTURE_INCREMENT {
    tag "${executionMode}:paired_B0_B1_B2"
    cpus 1
    memory "${params.memory_gb} GB"
    cache 'deep'
    publishDir params.output_dir, mode:'copy', overwrite:false
    input:
    path bundle, stageAs:'prepared_features'
    path code, stageAs:'r02_structure_increment.py'
    val contractHash
    path evaluationContract, stageAs:'evaluation_contract.json'
    val evaluationHash
    val executionMode
    val codeHash
    path selectionCheckpoint, stageAs:'selection_checkpoint'
    val selectionHash
    path review, stageAs:'selection_review.json'
    val reviewHash
    path numericalContract, stageAs:'numerical_contract.json'
    val numericalHash
    path pairOperator, stageAs:'r02_pair_operator.py'
    val pairOperatorHash
    path pairPreconditioner, stageAs:'r02_pair_preconditioner.py'
    val pairPreconditionerHash
    val resultName
    output:
    path "${resultName}", emit: evaluated
    script:
    def evaluationArgs = evaluationHash ? "--evaluation-contract evaluation_contract.json --evaluation-contract-sha256 ${evaluationHash}" : ''
    def reviewedArgs = executionMode == 'evaluate' ? "--selection-checkpoint selection_checkpoint --selection-checkpoint-sha256 ${selectionHash} --review selection_review.json --review-sha256 ${reviewHash}" : ''
    def numericalArgs = numericalHash ? "--numerical-contract numerical_contract.json --numerical-contract-sha256 ${numericalHash} --pair-operator-code r02_pair_operator.py --pair-operator-code-sha256 ${pairOperatorHash} --pair-preconditioner-code r02_pair_preconditioner.py --pair-preconditioner-code-sha256 ${pairPreconditionerHash}" : ''
    def codeCheck = codeHash ? "test \"\$(sha256sum r02_structure_increment.py | cut -d ' ' -f1)\" = '${codeHash}'" : ''
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    test "\$(sha256sum prepared_features/contract.json | cut -d ' ' -f1)" = '${contractHash}'
    ${codeCheck}
    python3 r02_structure_increment.py --bundle prepared_features \
      --contract prepared_features/contract.json --output-dir ${resultName} --mode ${executionMode} ${evaluationArgs} ${reviewedArgs} ${numericalArgs}
    """
}

workflow {
    def safe = { p -> p instanceof String && p.startsWith('/') && !p.tokenize('/').contains('..') && !p.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/) }
    if (!safe(params.bundle) || !safe(params.code) || !safe(params.output_dir)) error 'Absolute safe paths required'
    if (!(params.contract_sha256 ==~ /[0-9a-f]{64}/)) error 'Contract SHA256 required'
    if (!(params.memory_gb?.toString() ==~ /[1-9][0-9]*/)) error 'Positive memory_gb required'
    def evaluationPath = params.evaluation_contract ?: null
    def evaluationHash = params.evaluation_contract_sha256 ?: ''
    if (evaluationPath && (!safe(evaluationPath) || !(evaluationHash ==~ /[0-9a-f]{64}/))) error 'Separate evaluation contract requires absolute safe path and SHA256'
    if (!evaluationPath && evaluationHash) error 'Evaluation SHA256 requires its configuration file'
    def mode = params.mode ?: 'legacy'
    if (!(mode in ['legacy','select','evaluate'])) error 'Mode must be legacy, select or evaluate'
    def codeHash = params.code_sha256 ?: ''
    if ((mode != 'legacy' || codeHash) && !(codeHash ==~ /[0-9a-f]{64}/)) error 'Pinned evaluator code SHA256 required'
    if (mode != 'legacy' && !evaluationPath) error 'Separated v4 execution requires evaluation configuration'
    def checkpoint = params.selection_checkpoint ?: null
    def selectionHash = params.selection_checkpoint_sha256 ?: ''
    def review = params.review ?: null
    def reviewHash = params.review_sha256 ?: ''
    def numerical = params.numerical_contract ?: null
    def numericalHash = params.numerical_contract_sha256 ?: ''
    def pairOperator = params.pair_operator_code ?: null
    def pairOperatorHash = params.pair_operator_code_sha256 ?: ''
    def pairPreconditioner = params.pair_preconditioner_code ?: null
    def pairPreconditionerHash = params.pair_preconditioner_code_sha256 ?: ''
    if (numerical || numericalHash || pairOperator || pairOperatorHash || pairPreconditioner || pairPreconditionerHash) {
        if (!safe(numerical) || !safe(pairOperator) || !safe(pairPreconditioner) || !(numericalHash ==~ /[0-9a-f]{64}/) || !(pairOperatorHash ==~ /[0-9a-f]{64}/) || !(pairPreconditionerHash ==~ /[0-9a-f]{64}/)) error 'Pair backend requires safe numerical/code paths and SHA256 values'
        if (mode == 'legacy') error 'Pair backend applies only to separated v4 execution'
    }
    if (mode == 'evaluate') {
        if (!safe(checkpoint) || !safe(review) || !(selectionHash ==~ /[0-9a-f]{64}/) || !(reviewHash ==~ /[0-9a-f]{64}/)) error 'EVALUATE requires SELECT checkpoint and explicit review with SHA256'
    } else if (checkpoint || selectionHash || review || reviewHash) error 'Checkpoint/review arguments apply only to EVALUATE'
    // Legacy schemas stage their original contract as an unused sentinel. V4
    // requires a separately reviewed configuration; Python rejects its absence.
    R02_STRUCTURE_INCREMENT(file(params.bundle,checkIfExists:true),file(params.code,checkIfExists:true),params.contract_sha256,
        file(evaluationPath ?: "${params.bundle}/contract.json",checkIfExists:true),evaluationHash,mode,codeHash,
        file(checkpoint ?: "${params.bundle}/contract.json",checkIfExists:true),selectionHash,
        file(review ?: "${params.bundle}/contract.json",checkIfExists:true),reviewHash,
        file(numerical ?: "${params.bundle}/contract.json",checkIfExists:true),numericalHash,
        file(pairOperator ?: params.code,checkIfExists:true),pairOperatorHash,
        file(pairPreconditioner ?: params.code,checkIfExists:true),pairPreconditionerHash,
        mode == 'select' ? 'selection' : 'evaluation')
}
