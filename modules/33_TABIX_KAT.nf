nextflow.enable.dsl=2

process M33_TABIX_BUILD_ONE {
    tag { "m33_tabix_kat_${replica}" }
    cache false
    cpus 1
    memory '512 MB'
    time '5m'

    input:
    path kat_script
    val replica

    output:
    tuple val(replica), path("fixture.${replica}.vcf.gz"),
        path("fixture.${replica}.vcf.gz.tbi"), path("build.${replica}.json")

    script:
    """
    set -euo pipefail
    python3 ${kat_script} build \
      --replica ${replica} \
      --task-hash ${task.hash} \
      --task-work-uri ${task.workDir} \
      --expected-work-prefix gs://teams-usp/frank/lai-exploracion-datos/work/nextflow/${params.m33_tabix_kat_run_id}/ \
      --expected-runtime-service-account dnabr-m33-frank@uspbr-242713.iam.gserviceaccount.com \
      --output-dir .
    """
}

process M33_TABIX_PUBLISH {
    tag 'm33_tabix_kat_publish'
    cache false
    cpus 1
    memory '256 MB'
    time '5m'

    input:
    path kat_receipt
    path controller_receipt
    path storage_policy
    path storage_validator
    path publisher

    output:
    path 'publication.receipt.json'

    script:
    """
    set -euo pipefail
    python3 ${publisher} \
      --run-id ${params.m33_tabix_kat_run_id} \
      --kat-receipt ${kat_receipt} \
      --controller-receipt ${controller_receipt} \
      --storage-policy ${storage_policy} \
      --storage-validator ${storage_validator} \
      --output publication.receipt.json
    """
}

process M33_TABIX_COMPARE {
    tag 'm33_tabix_kat_compare'
    cache false
    cpus 1
    memory '256 MB'
    time '5m'

    input:
    tuple val(replica_a), path(vcf_a), path(tbi_a), path(receipt_a)
    tuple val(replica_b), path(vcf_b), path(tbi_b), path(receipt_b)
    path kat_script

    output:
    path 'm33_tabix_kat.receipt.json'
    path 'LOCAL_CANDIDATE_READY'

    script:
    """
    set -euo pipefail
    test "${replica_a}" != "${replica_b}"
    python3 ${kat_script} compare \
      --receipt-a ${receipt_a} \
      --receipt-b ${receipt_b} \
      --vcf-a ${vcf_a} \
      --tbi-a ${tbi_a} \
      --vcf-b ${vcf_b} \
      --tbi-b ${tbi_b} \
      --output m33_tabix_kat.receipt.json \
      --local-candidate-ready LOCAL_CANDIDATE_READY \
      --require-cloud
    """
}
