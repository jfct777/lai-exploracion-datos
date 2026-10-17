nextflow.enable.dsl=2

/* Assemble and score all 88 K02/K07 views. One view is one resumable numerical
 * checkpoint; two reducers choose alpha from all 44 SELECT blocks per scope.
 * EVALUATE has no process in this workflow.
 */
process STAGE_PREPARED_VIEW {
  tag { viewSpec.view_id }
  cpus 1; memory '2 GB'; disk '3 GB'; maxForks params.max_forks; cache 'deep'
  input:
  val viewSpec
  path plan; path commonObjects; path reusedCommon; path evaluationBlock; path evaluationWhole; path stager
  output: tuple val(viewSpec), path('staged/prepared'), path('staged/common.npz'), path('staged/evaluation.json')
  script:
  """
  umask 077
  python3 ${stager} --plan ${plan} --common-objects ${commonObjects} --reused-common ${reusedCommon} \
    --view-id '${viewSpec.view_id}' --evaluation-block ${evaluationBlock} --evaluation-whole ${evaluationWhole} \
    --out staged
  """
}

process STAGE_SHARD_MANIFEST {
  tag { viewSpec.view_id }
  cpus 1; memory '1 GB'; disk '1 GB'; maxForks params.max_forks; cache 'deep'
  input:
  val viewSpec
  path plan; path manifestObjects; path manifestStager
  output: tuple val(viewSpec), path('manifest.json')
  script:
  """
  umask 077
  python3 ${manifestStager} --plan ${plan} --inventory ${manifestObjects} \
    --view-id '${viewSpec.view_id}' --output manifest.json
  """
}

process ASSEMBLE_FULLCOMMON_V4 {
  tag { viewSpec.view_id }
  cpus 1; memory '4 GB'; disk '4 GB'; maxForks params.max_forks; cache 'deep'
  container params.container; stageInMode 'copy'
  input: tuple val(viewSpec), path(prepared), path(common), path(evaluation), path(shardManifest); path assembler; path amendment
  output: tuple val(viewSpec), path('assembled')
  script:
  """
  umask 077
  export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  test ! -e prepared/shards/${viewSpec.block_id}/manifest.json
  cp ${shardManifest} prepared/shards/${viewSpec.block_id}/manifest.json
  python3 ${assembler} --prepared-root prepared --common-npz ${common} --common-key ${viewSpec.common_key} \
    --block-id ${viewSpec.block_id} --evaluation ${evaluation} --run-scope-amendment ${amendment} \
    --output assembled
  owner="\$(stat -c '%u:%g' .)"
  chown -R "\$owner" assembled
  chmod -R u=rwX,go= assembled
  """
}

process SCORE_VIEW_WITH_PAIR_OPERATOR {
  tag { "${viewSpec.view_id}:${arm}" }
  cpus params.select_cpus; memory "${params.select_memory_gb} GB"; disk "${params.select_disk_gb} GB"
  maxForks params.max_forks; cache 'deep'; container params.container; stageInMode 'copy'
  containerOptions { params.solver_checkpoint_dir ? "-v ${params.solver_checkpoint_dir}:${params.solver_checkpoint_dir}" : '' }
  publishDir { "${params.output}/scores/${viewSpec.view_id}/${arm}" }, mode:'copy', overwrite:false
  input:
  tuple val(viewSpec), val(arm), path(bundle)
  path evaluator, stageAs:'r02_structure_increment.py'
  path pairOperator, stageAs:'r02_pair_operator.py'
  path pairPreconditioner, stageAs:'r02_pair_preconditioner.py'
  path numerics, stageAs:'pair-numerics.json'
  output: tuple val(viewSpec.exclusion_scope), path('score-*')
  script:
  def checkpointArgs=params.solver_checkpoint_dir ? "--solver-checkpoint-dir '${params.solver_checkpoint_dir}'" : ''
  """
  umask 077
  export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=${params.select_cpus} OPENBLAS_NUM_THREADS=${params.select_cpus} MKL_NUM_THREADS=${params.select_cpus}
  test "\$(sha256sum r02_structure_increment.py | cut -d ' ' -f1)" = '${params.evaluator_sha256}'
  test "\$(sha256sum r02_pair_operator.py | cut -d ' ' -f1)" = '${params.pair_operator_sha256}'
  test "\$(sha256sum r02_pair_preconditioner.py | cut -d ' ' -f1)" = '${params.pair_preconditioner_sha256}'
  test "\$(sha256sum pair-numerics.json | cut -d ' ' -f1)" = '${params.numerical_contract_sha256}'
  python3 r02_structure_increment.py --bundle ${bundle} --contract ${bundle}/contract.json \
    --evaluation-contract ${bundle}/evaluation.json --evaluation-contract-sha256 "\$(sha256sum ${bundle}/evaluation.json | cut -d ' ' -f1)" \
    --output-dir score-${viewSpec.view_id}-${arm} --mode score --score-arm '${arm}' --numerical-contract pair-numerics.json \
    --numerical-contract-sha256 '${params.numerical_contract_sha256}' \
    --pair-operator-code r02_pair_operator.py --pair-operator-code-sha256 '${params.pair_operator_sha256}' \
    --pair-preconditioner-code r02_pair_preconditioner.py --pair-preconditioner-code-sha256 '${params.pair_preconditioner_sha256}' \
    ${checkpointArgs}
  owner="\$(stat -c '%u:%g' .)"
  chown -R "\$owner" score-${viewSpec.view_id}-${arm}
  chmod -R u=rwX,go= score-${viewSpec.view_id}-${arm}
  """
}

process REDUCE_SCOPE_SELECTION {
  tag { scope }
  cpus 1; memory '2 GB'; disk '2 GB'; maxForks 2; cache 'deep'; container params.container
  publishDir { "${params.output}/selection/${scope}" }, mode:'copy', overwrite:false
  input: tuple val(scope), path(scoreDirs); path plan; path reducer
  output: tuple val(scope), path('selection-*'), emit:selected
  script:
  def inputs=scoreDirs.collect { "--score-dir '${it}'" }.join(' ')
  """
  umask 077
  test "\$(sha256sum ${reducer} | cut -d ' ' -f1)" = '${params.reducer_sha256}'
  python3 ${reducer} --plan ${plan} --scope '${scope}' ${inputs} --output selection-${scope}
  owner="\$(stat -c '%u:%g' .)"
  chown -R "\$owner" selection-${scope}
  chmod -R u=rwX,go= selection-${scope}
  """
}

workflow {
  def required=[params.tasks,params.common_objects,params.shard_manifest_objects,params.reused_common,params.evaluation_block,
                params.evaluation_whole,params.stager,params.manifest_stager,params.assembler,params.evaluator,params.reducer,
                params.pair_operator,params.pair_preconditioner,params.numerical_contract,
                params.run_scope_amendment,
                params.admission,params.output]
  if (!required.every { p -> p instanceof String && p.startsWith('/') &&
      !p.tokenize('/').contains('..') && !p.find(/[\n\r'"`\$\\;|&<>*?\[\]{}]/) } || !params.container)
    error 'Absolute safe paths and pinned container required'
  def checkpointPath=params.solver_checkpoint_dir ?: null
  def checkpointInvalid=checkpointPath && (!(checkpointPath instanceof String) ||
      !checkpointPath.startsWith('/') || checkpointPath.tokenize('/').contains('..') ||
      checkpointPath.find(/[\n\r'"`\$\\;|&<>*?\[\]{}]/) ||
      !file(checkpointPath).toFile().isDirectory())
  if (checkpointInvalid) error 'Solver checkpoint directory must be an existing absolute safe directory'
  def hashes=[params.workflow_sha256,params.stager_sha256,params.manifest_stager_sha256,
              params.shard_manifest_objects_sha256,params.assembler_sha256,params.common_objects_sha256,
              params.reused_common_sha256,params.evaluation_block_sha256,params.evaluation_whole_sha256,
              params.evaluator_sha256,params.reducer_sha256,params.pair_operator_sha256,
              params.pair_preconditioner_sha256,params.numerical_contract_sha256,params.admission_sha256]
  hashes << params.run_scope_amendment_sha256
  if (!hashes.every { it ==~ /[0-9a-f]{64}/ }) error 'Exact source/config/admission SHA256 values required'
  [params.max_forks,params.select_cpus,params.select_memory_gb,params.select_disk_gb].each { v ->
    if (!(v?.toString() ==~ /[1-9][0-9]*/)) error 'Positive explicit concurrency/resources required'
  }
  def admissionDigest=java.security.MessageDigest.getInstance('SHA-256').digest(file(params.admission).bytes).encodeHex().toString()
  def planDigest=java.security.MessageDigest.getInstance('SHA-256').digest(file(params.tasks).bytes).encodeHex().toString()
  if (admissionDigest!=params.admission_sha256) error 'Admission SHA256 differs'
  def pinnedFiles=[[params.stager,params.stager_sha256],[params.manifest_stager,params.manifest_stager_sha256],
    [params.shard_manifest_objects,params.shard_manifest_objects_sha256],[params.assembler,params.assembler_sha256],
    [params.common_objects,params.common_objects_sha256],[params.reused_common,params.reused_common_sha256],
    [params.evaluation_block,params.evaluation_block_sha256],[params.evaluation_whole,params.evaluation_whole_sha256],
    [params.evaluator,params.evaluator_sha256],[params.reducer,params.reducer_sha256],
    [params.pair_operator,params.pair_operator_sha256],[params.pair_preconditioner,params.pair_preconditioner_sha256],
    [params.numerical_contract,params.numerical_contract_sha256],
    [params.run_scope_amendment,params.run_scope_amendment_sha256]]
  if (!pinnedFiles.every { item -> java.security.MessageDigest.getInstance('SHA-256').digest(file(item[0]).bytes).encodeHex().toString()==item[1] })
    error 'A packaged source/configuration SHA256 differs'
  def plan=new groovy.json.JsonSlurper().parse(file(params.tasks,checkIfExists:true))
  if (plan.tasks.size()!=88 || plan.evaluate_allowed!=false) error 'Exact 88-view plan with EVALUATE closed required'
  def admission=new groovy.json.JsonSlurper().parse(file(params.admission,checkIfExists:true))
  if (admission.schema!='r02_k02_k07_pair_select_admission_v1' || admission.decision!='ADMIT_FIT_SELECT' ||
      admission.evaluate_allowed!=false || admission.view_count!=88 || admission.score_checkpoint_count!=264 ||
      admission.global_selection_count!=2 || admission.machine_type!='e2-standard-4' ||
      admission.disk_type!='pd-balanced' || admission.disk_gib!=280 || admission.max_forks!=1 ||
      admission.executor_queue_size!=2 || admission.max_runtime_hours!=24 || admission.team_label!='frank' ||
      admission.assembly_plan_sha256!=planDigest || admission.workflow_sha256!=params.workflow_sha256 ||
      admission.evaluator_sha256!=params.evaluator_sha256 ||
      admission.stager_sha256!=params.stager_sha256 ||
      admission.manifest_stager_sha256!=params.manifest_stager_sha256 ||
      admission.shard_manifest_objects_sha256!=params.shard_manifest_objects_sha256 ||
      admission.assembler_sha256!=params.assembler_sha256 ||
      admission.common_objects_sha256!=params.common_objects_sha256 ||
      admission.reused_common_sha256!=params.reused_common_sha256 ||
      admission.evaluation_block_sha256!=params.evaluation_block_sha256 ||
      admission.evaluation_whole_sha256!=params.evaluation_whole_sha256 ||
      admission.reducer_sha256!=params.reducer_sha256 || admission.pair_operator_sha256!=params.pair_operator_sha256 ||
      admission.pair_preconditioner_sha256!=params.pair_preconditioner_sha256 ||
      admission.run_scope_amendment_sha256!=params.run_scope_amendment_sha256 ||
      admission.numerical_contract_sha256!=params.numerical_contract_sha256)
    error 'Root admission is absent or differs from the exact campaign'
  def inputs=Channel.fromList(plan.tasks)
  STAGE_PREPARED_VIEW(inputs,Channel.value(file(params.tasks,checkIfExists:true)),
      Channel.value(file(params.common_objects,checkIfExists:true)),
      Channel.value(file(params.reused_common,checkIfExists:true)),
      Channel.value(file(params.evaluation_block,checkIfExists:true)),
      Channel.value(file(params.evaluation_whole,checkIfExists:true)),
      Channel.value(file(params.stager,checkIfExists:true)))
  STAGE_SHARD_MANIFEST(inputs,Channel.value(file(params.tasks,checkIfExists:true)),
      Channel.value(file(params.shard_manifest_objects,checkIfExists:true)),
      Channel.value(file(params.manifest_stager,checkIfExists:true)))
  STAGE_PREPARED_VIEW.out.join(STAGE_SHARD_MANIFEST.out).set { assemblyInputs }
  ASSEMBLE_FULLCOMMON_V4(assemblyInputs,Channel.value(file(params.assembler,checkIfExists:true)),
      Channel.value(file(params.run_scope_amendment,checkIfExists:true)))
  ASSEMBLE_FULLCOMMON_V4.out.flatMap { viewSpec,bundle -> ['B0','B1','B2'].collect { arm -> tuple(viewSpec,arm,bundle) } }.set { scoringInputs }
  SCORE_VIEW_WITH_PAIR_OPERATOR(scoringInputs,
      Channel.value(file(params.evaluator,checkIfExists:true)),
      Channel.value(file(params.pair_operator,checkIfExists:true)),
      Channel.value(file(params.pair_preconditioner,checkIfExists:true)),
      Channel.value(file(params.numerical_contract,checkIfExists:true)))
  SCORE_VIEW_WITH_PAIR_OPERATOR.out.groupTuple().map { scope,dirs -> tuple(scope,dirs) }.set { groupedScores }
  REDUCE_SCOPE_SELECTION(groupedScores,Channel.value(file(params.tasks,checkIfExists:true)),
      Channel.value(file(params.reducer,checkIfExists:true)))
}
