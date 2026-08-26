nextflow.enable.dsl=2

process M33_DEVELOPMENT_SCREEN_TRAIN {
    tag { "${rotation}_${family}_r${radius}_q${share}_${arm}" }
    publishDir { "${params.m33_dev_screen_results_dir}/${params.m33_dev_screen_run_id}/training" },
               mode: 'copy', overwrite: false
    cpus params.m33_dev_screen_cpus
    memory params.m33_dev_screen_memory
    time params.m33_dev_screen_time
    maxForks params.m33_dev_screen_max_forks

    input:
    tuple val(rotation), val(family), val(radius), val(share), val(beta), val(seed), val(arm)
    path source_files
    path pre4_contract
    path amendment

    output:
    tuple val(rotation), val(family), val(radius), val(share), val(beta), val(seed), val(arm),
          path("${rotation}.${family}.r${radius}.q${share}.${arm}"), emit: bundle

    script:
    def outputName = "${rotation}.${family}.r${radius}.q${share}.${arm}"
    """
    set -euo pipefail
    mkdir -p staged/bin
    cp ${source_files} staged/bin/
    PYTHONPATH=staged/bin python3 staged/bin/m33_train_development.py \
      --runtime /m33-runtime \
      --pre4 ${pre4_contract} \
      --rotation '${rotation}' --family '${family}' --radius '${radius}' \
      --beta '${beta}' --seed '${seed}' --arm '${arm}' \
      --max-updates 200 --validation-start 200 --validation-every 200 \
      --skip-prediction --outdir '${outputName}'
    """
}

process M33_DEVELOPMENT_SCREEN_PREDICT {
    tag { "${rotation}_${family}_r${radius}_q${share}_${arm}" }
    publishDir { "${params.m33_dev_predict_results_dir}/${params.m33_dev_predict_run_id}" },
               mode: 'copy', overwrite: false
    cpus params.m33_dev_predict_cpus
    memory params.m33_dev_predict_memory
    time params.m33_dev_predict_time
    maxForks params.m33_dev_predict_max_forks

    input:
    tuple val(rotation), val(family), val(radius), val(share), val(arm), path(checkpoint)
    path source_files
    path pre4_contract

    output:
    tuple val(rotation), val(family), val(radius), val(share), val(arm),
          path("${rotation}.${family}.r${radius}.q${share}.${arm}.prediction"), emit: bundle

    script:
    def outputName = "${rotation}.${family}.r${radius}.q${share}.${arm}.prediction"
    """
    set -euo pipefail
    mkdir -p staged/bin '${outputName}'
    cp ${source_files} staged/bin/
    PYTHONPATH=staged/bin python3 staged/bin/m33_predict_development.py \
      --runtime /m33-runtime --pre4 ${pre4_contract} \
      --rotation '${rotation}' --family '${family}' --radius '${radius}' --arm '${arm}' \
      --checkpoint ${checkpoint} \
      --output '${outputName}/score.prediction.npz' \
      --receipt '${outputName}/prediction.receipt.json'
    """
}

process M33_DEVELOPMENT_SCREEN_SCORE {
    tag { "${rotation}_${family}_r${radius}_q${share}_${arm}" }
    publishDir { "${params.m33_dev_score_results_dir}/${params.m33_dev_score_run_id}" },
               mode: 'copy', overwrite: false
    cpus 1
    memory '4 GB'
    time '10m'
    maxForks 2

    input:
    tuple val(rotation), val(family), val(radius), val(share), val(arm), path(prediction)
    path target_vcf
    path truth
    path genetic_map
    path source_files

    output:
    path "${rotation}.${family}.r${radius}.q${share}.${arm}.metrics.json", emit: metrics

    script:
    def outputName = "${rotation}.${family}.r${radius}.q${share}.${arm}.metrics.json"
    """
    set -euo pipefail
    mkdir -p staged/bin
    cp ${source_files} staged/bin/
    PYTHONPATH=staged/bin python3 staged/bin/m33_score_development.py score \
      --prediction ${prediction} --target-vcf ${target_vcf} \
      --truth ${truth} --map ${genetic_map} --output '${outputName}'
    """
}
