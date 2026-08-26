nextflow.enable.dsl=2

process M34_NAM_PACK_BASELINE {
    tag 'm34_pack_flare_f0_valid'
    publishDir {
        "${params.m34_inputs_results_dir}/${params.m34_inputs_run_id}/baseline"
    }, mode: 'copy', overwrite: false
    container params.m34_inputs_pytorch_image
    containerOptions { "--network none --user ${params.m34_inputs_container_user}" }
    cpus { params.m34_inputs_score_cpus }
    memory { params.m34_inputs_score_memory }
    time { params.m34_inputs_score_time }

    input:
    tuple val(split), path(f0Dir)
    path packBaselinePy
    path bridgeCorePy

    output:
    tuple val('FLARE'), val('F0'), val('F0'),
          path('m34_f0.valid.prediction.npz'), emit: prediction

    script:
    """
    set -euo pipefail
    test '${split}' = VALID
    mkdir -p staged/bin
    cp ${packBaselinePy} staged/bin/m34_pack_f0_prediction.py
    cp ${bridgeCorePy} staged/bin/m33_safe_bridge_core.py
    PYTHONPATH=staged/bin python3 staged/bin/m34_pack_f0_prediction.py \
      --f0 ${f0Dir}/m34_f0.npz \
      --marker-cm ${f0Dir}/marker_cM.npz \
      --output m34_f0.valid.prediction.npz
    """

    stub:
    """
    set -euo pipefail
    touch m34_f0.valid.prediction.npz
    """
}
