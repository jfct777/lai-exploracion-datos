nextflow.enable.dsl=2

process M34_NAM_GENERATE_MOSAICS {
    tag { "m34_mosaics_${split}" }
    publishDir {
        "${params.m34_inputs_results_dir}/${params.m34_inputs_run_id}/${split.toLowerCase()}/mosaics"
    }, mode: 'copy', overwrite: false
    container params.m34_inputs_pytorch_image
    containerOptions { "--network none --user ${params.m34_inputs_container_user}" }
    cpus { params.m34_inputs_mosaic_cpus }
    memory { params.m34_inputs_mosaic_memory }
    time { params.m34_inputs_mosaic_time }
    maxForks params.m34_inputs_mosaic_max_forks

    input:
    tuple val(split), val(donorRole), val(forbiddenRole), val(donorPartition),
          val(seed), val(targetIndividuals), val(targetPrefix),
          val(mixtureProportions), val(admixtureGenerations)
    path phasedVcf
    path splitTsv
    path geneticMap
    path mosaicPy

    output:
    tuple val(split), val(donorRole),
          path("m34_${split.toLowerCase()}_mosaic/m34_target.chr22.vcf.gz"),
          path("m34_${split.toLowerCase()}_mosaic/m34_truth.chr22.tsv.gz"),
          path("m34_${split.toLowerCase()}_mosaic/m34_donor_audit.private.tsv"),
          path("m34_${split.toLowerCase()}_mosaic/m34_mosaic.receipt.json"),
          emit: mosaics

    script:
    """
    set -euo pipefail
    python3 ${mosaicPy} \
      --phased-vcf ${phasedVcf} \
      --split-tsv ${splitTsv} \
      --genetic-map ${geneticMap} \
      --outdir m34_${split.toLowerCase()}_mosaic \
      --chromosome 22 \
      --donor-role ${donorRole} \
      --forbidden-role REF_TRAIN \
      --forbidden-role ${forbiddenRole} \
      --donor-unit-partition ${donorPartition} \
      --seed ${seed} \
      --target-individuals ${targetIndividuals} \
      --target-prefix ${targetPrefix} \
      --mixture-proportions '${mixtureProportions}' \
      --admixture-generations ${admixtureGenerations}
    """

    stub:
    """
    set -euo pipefail
    mkdir -p m34_${split.toLowerCase()}_mosaic
    touch \
      m34_${split.toLowerCase()}_mosaic/m34_target.chr22.vcf.gz \
      m34_${split.toLowerCase()}_mosaic/m34_truth.chr22.tsv.gz \
      m34_${split.toLowerCase()}_mosaic/m34_donor_audit.private.tsv \
      m34_${split.toLowerCase()}_mosaic/m34_mosaic.receipt.json
    """
}
