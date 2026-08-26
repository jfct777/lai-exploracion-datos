nextflow.enable.dsl=2

process M34_NAM_BUILD_TRIAGE_PLAN {
    tag 'm34_triage_plan_R0'
    publishDir {
        "${params.m34_inputs_results_dir}/${params.m34_inputs_run_id}/triage_plan"
    }, mode: 'copy', overwrite: false
    container params.m34_inputs_pytorch_image
    containerOptions { "--network none --user ${params.m34_inputs_container_user}" }
    cpus { params.m34_inputs_plan_cpus }
    memory { params.m34_inputs_plan_memory }
    time { params.m34_inputs_plan_time }

    input:
    path adaptiveContract
    path adaptiveSweepPy

    output:
    path 'm34_triage.plan.json', emit: plan

    script:
    """
    set -euo pipefail
    python3 ${adaptiveSweepPy} \
      --contract ${adaptiveContract} \
      --stage triage \
      --output m34_triage.plan.json
    """

    stub:
    """
    set -euo pipefail
    python3 -c "import json; tasks=[{'family':'local_linear','config_id':'linear_r0','seed':1103,'rotation':'R0','arm':a,'radius_cM':0.2,'sweep_stage':'triage','maximum_updates':300,'learning_rate':0.0003,'weight_decay':0.0001} for a in ('RD','RE')]; open('m34_triage.plan.json','x').write(json.dumps({'schema_version':'1.0.0','stage':'M34_TRIAGE_PLAN','status':'PLAN_ONLY_NO_EXECUTION','task_count':2,'tasks':tasks}))"
    """
}
