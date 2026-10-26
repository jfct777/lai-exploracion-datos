nextflow.enable.dsl=2

/*
 * One chr22 structure worker: denominator audit, then six paired FIT jobs.
 * SELECT/EVALUATE are deliberately absent. Checkpoints live in output_root,
 * outside a Nextflow task directory, so a retried task can resume them.
 */

process AUDIT_MASKED_CHR22_DENOMINATORS {
    tag 'chr22-365-update-denominator-audit'
    cpus 1
    memory '2 GB'
    time '4h'
    maxForks 1
    cache 'deep'
    errorStrategy 'terminate'

    input:
    path campaign
    path pre_receipt
    path requests

    output:
    path 'denominator-audit-receipt.json', emit: receipt

    script:
    def requestArgs = requests.collect { "--all-request ${it}" }.join(' ')
    """
    python3 ${params.code_root}/bin/r02_masked_training_request.py audit-denominators \
      --campaign ${campaign} \
      --pre-receipt ${pre_receipt} \
      ${requestArgs} \
      --output ${params.output_root}/denominator-audit-final
    cp ${params.output_root}/denominator-audit-final/DENOMINATOR_AUDIT.json \
      denominator-audit-receipt.json
    """
}

process AUDIT_MASKED_CHR22_RARE_IDENTITY {
    tag 'chr22-rare-identity-two-family-audit'
    cpus 1
    memory '2 GB'
    time '4h'
    maxForks 1
    cache 'deep'
    errorStrategy 'terminate'

    input:
    path campaign
    path pre_receipt
    path requests

    output:
    path 'rare-identity-audit-receipt.json', emit: receipt

    script:
    def requestArgs = requests.collect { "--all-request ${it}" }.join(' ')
    """
    python3 ${params.code_root}/bin/r02_masked_training_request.py audit-rare-identity \
      --campaign ${campaign} \
      --pre-receipt ${pre_receipt} \
      ${requestArgs} \
      --output ${params.output_root}/rare-identity-audit
    cp ${params.output_root}/rare-identity-audit/RARE_IDENTITY_AUDIT.json \
      rare-identity-audit-receipt.json
    """
}


process TRAIN_MASKED_CHR22_FIT {
    tag { fit_id }
    cpus 1
    memory '8 GB'
    // The Python fit has an 8 h cumulative ceiling.  One extra hour lets it
    // persist the final checkpoint and receipt before Nextflow's hard stop.
    time '9h'
    maxForks 1
    cache 'deep'
    errorStrategy 'terminate'

    input:
    tuple val(fit_id), path(request), path(campaign), path(pre_receipt), path(requests), path(audit_receipt), path(rare_audit_receipt)

    output:
    tuple val(fit_id), path('training-receipt.json'), emit: receipts

    script:
    def requestArgs = requests.collect { "--all-request ${it}" }.join(' ')
    """
    python3 ${params.code_root}/bin/r02_masked_training_request.py run \
      --campaign ${campaign} \
      --request ${request} \
      --pre-receipt ${pre_receipt} \
      --denominator-audit ${params.output_root}/denominator-audit-final/DENOMINATOR_AUDIT.json \
      --rare-identity-audit ${params.output_root}/rare-identity-audit/RARE_IDENTITY_AUDIT.json \
      ${requestArgs} \
      --output ${params.output_root}/fits/${fit_id}
    cp ${params.output_root}/fits/${fit_id}/TRAINING_RECEIPT.json training-receipt.json
    """
}


workflow {
    if (!(params.campaign && params.pre_receipt && params.output_root && params.repo_root && params.code_root)) {
        throw new IllegalArgumentException('campaign, pre_receipt, output_root, repo_root and code_root are required')
    }
    def ids = [
        'local_deepsets__no_rare', 'local_deepsets__count_only', 'local_deepsets__full',
        'local_attention__no_rare', 'local_attention__count_only', 'local_attention__full',
        'local_deepsets__full__rare_identity', 'local_attention__full__rare_identity'
    ]
    def requestPaths = ids.collect { file("${params.request_root}/REQUEST_${it.toUpperCase()}.json", checkIfExists: true) }
    def campaignFile = file(params.campaign, checkIfExists: true)
    def preFile = file(params.pre_receipt, checkIfExists: true)
    AUDIT_MASKED_CHR22_DENOMINATORS(campaignFile, preFile, requestPaths)
    AUDIT_MASKED_CHR22_RARE_IDENTITY(campaignFile, preFile, requestPaths)
    def fitEntries = ids.withIndex().collect { id, index -> [id, requestPaths[index]] }
    def fits = Channel.fromList(fitEntries)
        .combine(AUDIT_MASKED_CHR22_DENOMINATORS.out.receipt)
        .combine(AUDIT_MASKED_CHR22_RARE_IDENTITY.out.receipt)
        .map { id, request, audit, rareAudit ->
        tuple(id, request, campaignFile, preFile, requestPaths, audit, rareAudit)
    }
    TRAIN_MASKED_CHR22_FIT(fits)
}
