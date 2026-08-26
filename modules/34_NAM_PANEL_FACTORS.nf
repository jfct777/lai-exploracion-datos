nextflow.enable.dsl=2

process M34_NAM_PREPARE_PANEL_FACTORS {
    tag { "m34_panel_factors_${split}" }
    publishDir {
        "${params.m34_inputs_results_dir}/${params.m34_inputs_run_id}/${split.toLowerCase()}/bridge"
    }, mode: 'copy', overwrite: false
    container params.m34_inputs_pytorch_image
    containerOptions { "--network none --user ${params.m34_inputs_container_user}" }
    cpus { params.m34_inputs_bridge_cpus }
    memory { params.m34_inputs_bridge_memory }
    time { params.m34_inputs_bridge_time }
    maxForks params.m34_inputs_bridge_max_forks

    input:
    tuple val(split), val(mosaicDonorRole), path(mosaicVcf)
    path panelVcf
    path splitTsv
    path geneticMap
    path bridgePy
    path mosaicPy
    path bridgeCorePy

    output:
    tuple val(split),
          path("m34_${split.toLowerCase()}_bridge/m34_ref_train.chr22.vcf.gz"),
          path("m34_${split.toLowerCase()}_bridge/m34_target.chr22.vcf.gz"),
          path("m34_${split.toLowerCase()}_bridge/m34_ref_train.sample_map.tsv"),
          path("m34_${split.toLowerCase()}_bridge/m34_selected_loci.npz"),
          path("m34_${split.toLowerCase()}_bridge/m34_target_rare_diploid.npz"),
          path("m34_${split.toLowerCase()}_bridge/m34_reference_rare_summary.npz"),
          path("m34_${split.toLowerCase()}_bridge/m34_panel_factors.receipt.json"),
          emit: factors

    script:
    """
    set -euo pipefail
    mkdir -p staged/bin
    cp ${bridgePy} staged/bin/m34_prepare_panel_factors.py
    cp ${mosaicPy} staged/bin/m34_generate_mosaics.py
    cp ${bridgeCorePy} staged/bin/m33_safe_bridge_core.py
    PYTHONPATH=staged/bin python3 staged/bin/m34_prepare_panel_factors.py \
      --panel-vcf ${panelVcf} \
      --mosaic-vcf ${mosaicVcf} \
      --split-tsv ${splitTsv} \
      --genetic-map ${geneticMap} \
      --outdir m34_${split.toLowerCase()}_bridge \
      --chromosome 22 \
      --min-mac 2 \
      --max-maf-exclusive 0.01 \
      --mosaic-donor-role ${mosaicDonorRole}
    """

    stub:
    """
    set -euo pipefail
    mkdir -p m34_${split.toLowerCase()}_bridge
    touch \
      m34_${split.toLowerCase()}_bridge/m34_ref_train.chr22.vcf.gz \
      m34_${split.toLowerCase()}_bridge/m34_target.chr22.vcf.gz \
      m34_${split.toLowerCase()}_bridge/m34_ref_train.sample_map.tsv \
      m34_${split.toLowerCase()}_bridge/m34_selected_loci.npz \
      m34_${split.toLowerCase()}_bridge/m34_target_rare_diploid.npz \
      m34_${split.toLowerCase()}_bridge/m34_reference_rare_summary.npz \
      m34_${split.toLowerCase()}_bridge/m34_panel_factors.receipt.json
    """
}
