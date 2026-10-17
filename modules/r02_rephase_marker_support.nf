nextflow.enable.dsl=2

process R02_REPHASE_RARE_AXIS {
    tag "chr${chromosome}"
    cpus 1
    memory '1 GB'
    time '20 min'
    maxForks params.max_forks
    publishDir { "${params.output_dir}/chr${chromosome}/axis" }, mode: 'copy', overwrite: false
    input:
    tuple val(chromosome), path(specification)
    output:
    tuple val(chromosome), path('axis-metadata'), path('complete.json'), path(specification)
    script:
    """
    python3 -B '${params.source_bin}/r02_rephase_marker_support_task.py' axis \\
      --spec "\${PWD}/${specification}" --deadline-epoch '${params.deadline_epoch}'
    """
}

process R02_REPHASE_MARKER_SUPPORT {
    tag "chr${chromosome}"
    cpus 1
    memory '1 GB'
    time '20 min'
    maxForks params.max_forks
    publishDir { "${params.output_dir}/chr${chromosome}/support" }, mode: 'copy', overwrite: false
    input:
    tuple val(chromosome), path(axis), path(specification)
    output:
    tuple val(chromosome), path('result'), path('complete.json')
    script:
    """
    python3 -B '${params.source_bin}/r02_rephase_marker_support_task.py' stream \\
      --spec "\${PWD}/${specification}" --axis "\${PWD}/${axis}" \\
      --deadline-epoch '${params.deadline_epoch}'
    """
}
