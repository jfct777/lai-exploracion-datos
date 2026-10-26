nextflow.enable.dsl=2

/*
 * FIT-only two-COMMON-flank diagnostic.  `count` is independent of trained
 * checkpoints.  `readiness` records a one-shot wait state without polling.
 * `score` is launched only after both exact FULL FIT receipts exist.
 */

process COUNT_MASKED_CHR22_TWO_COMMON_FLANKS {
    tag 'chr22-730-fragment-two-common-flank-counts'
    cpus 1
    memory '2 GB'
    time '20h'
    maxForks 1
    cache 'deep'
    errorStrategy 'terminate'

    input:
    path campaign
    path control_request
    path pre_receipt
    path control_post
    path requests

    output:
    path 'flank-counts-receipt.json', emit: receipt

    script:
    def requestArgs = requests.collect { "--all-request ${it}" }.join(' ')
    """
    PYTHONPATH=${params.core_code_root}/bin python3 ${params.flank_code_root}/bin/r02_masked_flank_diagnostic_v2.py count \
      --campaign ${campaign} \
      --control-request ${control_request} \
      --pre-receipt ${pre_receipt} \
      --control-post ${control_post} \
      ${requestArgs} \
      --output ${params.output_root}/two-common-flanks-v2/counts
    cp ${params.output_root}/two-common-flanks-v2/counts/FLANK_COUNTS.json \
      flank-counts-receipt.json
    """
}


process CHECK_MASKED_CHR22_FLANK_SCORE_READINESS {
    tag 'chr22-two-common-flank-score-readiness'
    cpus 1
    memory '256 MB'
    time '5m'
    cache false
    errorStrategy 'terminate'

    input:
    val campaign_path
    val deepsets_receipt_path
    val attention_receipt_path

    output:
    path 'flank-score-readiness.json', emit: receipt

    script:
    def deepsets = deepsets_receipt_path ? "--local-deepsets-receipt ${deepsets_receipt_path}" : ''
    def attention = attention_receipt_path ? "--local-attention-receipt ${attention_receipt_path}" : ''
    """
    PYTHONPATH=${params.core_code_root}/bin python3 ${params.flank_code_root}/bin/r02_masked_flank_diagnostic_v2.py readiness \
      --campaign ${campaign_path} \
      ${deepsets} ${attention} \
      --output flank-score-readiness.json
    """
}


process SCORE_MASKED_CHR22_TWO_COMMON_FLANKS {
    tag { "chr22-two-common-flanks-${family}" }
    cpus 1
    memory '8 GB'
    time '4h'
    maxForks 1
    cache 'deep'
    errorStrategy 'terminate'

    input:
    tuple val(family), path(training_receipt), path(campaign),
          path(control_request), path(pre_receipt), path(control_post),
          path(requests), path(count_receipt)

    output:
    tuple val(family), path('flank-score-receipt.json'), emit: receipts

    script:
    def requestArgs = requests.collect { "--all-request ${it}" }.join(' ')
    """
    PYTHONPATH=${params.core_code_root}/bin python3 ${params.flank_code_root}/bin/r02_masked_flank_diagnostic_v2.py score \
      --campaign ${campaign} \
      --control-request ${control_request} \
      --pre-receipt ${pre_receipt} \
      --control-post ${control_post} \
      --count-receipt ${count_receipt} \
      --training-receipt ${training_receipt} \
      --family ${family} \
      ${requestArgs} \
      --output ${params.output_root}/two-common-flanks-v2/scores/${family}
    cp ${params.output_root}/two-common-flanks-v2/scores/${family}/FLANK_SCORE.json \
      flank-score-receipt.json
    """
}


workflow {
    if (!(params.mode in ['count', 'readiness', 'score'])) {
        throw new IllegalArgumentException('mode must be count, readiness or score')
    }
    if (!(params.campaign && params.flank_code_root && params.core_code_root && params.output_root)) {
        throw new IllegalArgumentException('campaign, flank_code_root, core_code_root and output_root are required')
    }
    if (params.mode == 'readiness') {
        CHECK_MASKED_CHR22_FLANK_SCORE_READINESS(
            params.campaign as String,
            params.local_deepsets_receipt ?: '',
            params.local_attention_receipt ?: '')
    } else {
        if (!(params.request_root && params.pre_receipt && params.control_request &&
              params.control_post)) {
            throw new IllegalArgumentException(
                'request_root, pre_receipt, control_request and control_post are required')
        }
        def ids = [
            'local_deepsets__no_rare', 'local_deepsets__count_only', 'local_deepsets__full',
            'local_attention__no_rare', 'local_attention__count_only', 'local_attention__full',
            'local_deepsets__full__rare_identity', 'local_attention__full__rare_identity'
        ]
        def requestPaths = ids.collect {
            file("${params.request_root}/REQUEST_${it.toUpperCase()}.json", checkIfExists: true)
        }
        def campaignFile = file(params.campaign, checkIfExists: true)
        def controlFile = file(params.control_request, checkIfExists: true)
        def preFile = file(params.pre_receipt, checkIfExists: true)
        def postFile = file(params.control_post, checkIfExists: true)
        if (params.mode == 'count') {
            COUNT_MASKED_CHR22_TWO_COMMON_FLANKS(
                campaignFile, controlFile, preFile, postFile, requestPaths)
        } else {
            if (!(params.count_receipt && params.local_deepsets_receipt &&
                  params.local_attention_receipt)) {
                throw new IllegalArgumentException(
                    'score requires count_receipt and both FULL FIT receipts')
            }
            def countFile = file(params.count_receipt, checkIfExists: true)
            def fits = Channel.of(
                tuple('local_deepsets', file(params.local_deepsets_receipt,
                                              checkIfExists: true)),
                tuple('local_attention', file(params.local_attention_receipt,
                                               checkIfExists: true)))
            def entries = fits.map { family, receipt -> tuple(family, receipt,
                campaignFile, controlFile, preFile, postFile, requestPaths, countFile) }
            SCORE_MASKED_CHR22_TWO_COMMON_FLANKS(entries)
        }
    }
}
