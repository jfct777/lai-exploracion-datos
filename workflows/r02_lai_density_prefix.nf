nextflow.enable.dsl = 2
params.plan=null; params.plan_sha256=null; params.code_dir=null; params.output_dir=null; params.max_forks=2

process PREPARE_LAI_DENSITY_PREFIX {
    tag density_task_id
    container 'sha256:f8c26d3769b82ef68b07ca5b67c9a6acff862bcde397addc61dc014ce8dd2896'
    cpus 1; memory '2 GB'; time '30 min'; maxForks params.max_forks; cache 'deep'
    publishDir { params.output_dir+'/'+density_task_id }, mode:'copy', overwrite:false
    input:
    tuple val(density_task_id), path(density_batch, stageAs:'density-control-views.npz'),
          path(density_receipt, stageAs:'density-receipt.json'),
          path(event_axis, stageAs:'event-axis.npz'),
          path(event_axis_receipt, stageAs:'axis-receipt.json')
    path prefix_code
    output:
    tuple val(density_task_id), path('prepared/event-burden-prefix.npz'), path('prepared/receipt.json')
    script:
    """
    python3 '${prefix_code}' build-prefix --event-conditions '${density_batch}' \
      --control-receipt '${density_receipt}' --event-axis '${event_axis}' \
      --event-axis-receipt '${event_axis_receipt}' --condition DENSITY_MATCHED --output-dir prepared
    """
}

workflow {
    if(!params.plan||!params.plan_sha256||!params.code_dir||!params.output_dir) error 'required parameters absent'
    def p=file(params.plan,checkIfExists:true);def md=java.security.MessageDigest.getInstance('SHA-256')
    if(md.digest(p.bytes).encodeHex().toString()!=params.plan_sha256) error 'density prefix plan SHA-256 differs'
    def plan=new groovy.json.JsonSlurper().parse(p.toFile())
    if(plan.schema!='r02_lai_density_prefix_plan_v1'||plan.status!='THIRTY_SIX_DENSITY_PREFIX_REQUESTS'||
       plan.tasks?.size()!=36||plan.radius_embedded_in_prefix!=false||plan.training_authorized!=false)
       error 'exact 36-task PREPARE prefix plan required'
    def rows=plan.tasks.collect{t->tuple(t.density_task_id,file(t.density_batch.path,checkIfExists:true),
      file(t.density_receipt.path,checkIfExists:true),file(t.event_axis.path,checkIfExists:true),
      file(t.event_axis_receipt.path,checkIfExists:true))}
    PREPARE_LAI_DENSITY_PREFIX(Channel.fromList(rows),
      file(params.code_dir+'/r02_lai_query_burden.py',checkIfExists:true))
}
