nextflow.enable.dsl=2

// One authenticated analysis command, invoked by the R02 checkpoint supervisor.
// Scientific implementation remains in the existing/module-specific tools.
process R02_ANALYSIS_TASK {
    tag params.r02_stage
    cache false

    input:
    path command_json
    path execution_tool
    path frozen_bin

    output:
    path 'task_receipt.json'

    script:
    """
    python3 '${execution_tool}' --command '${command_json}' --source-bin '${frozen_bin}' --receipt task_receipt.json
    """
}

workflow {
    R02_ANALYSIS_TASK(file(params.r02_command_file, checkIfExists:true),
                      file(params.r02_exec_tool, checkIfExists:true),
                      file(params.r02_source_bin, checkIfExists:true))
}
