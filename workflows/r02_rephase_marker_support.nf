nextflow.enable.dsl=2
include { R02_REPHASE_RARE_AXIS; R02_REPHASE_MARKER_SUPPORT } from '../modules/r02_rephase_marker_support.nf'

params.campaign = null
params.campaign_sha256 = null
params.source_bin = null
params.output_dir = null
params.deadline_epoch = null
params.max_forks = 4

workflow {
    if (!params.campaign || !params.source_bin || !params.output_dir || !params.deadline_epoch) error('Explicit campaign, source, output and absolute deadline required')
    if (!(params.max_forks in [1,2,3,4])) error('At most four admitted tasks')
    def campaignFile = file(params.campaign, checkIfExists: true)
    def actualSha = java.security.MessageDigest.getInstance('SHA-256').digest(campaignFile.bytes).encodeHex().toString()
    if (actualSha != params.campaign_sha256) error('Campaign SHA differs')
    def plan = new groovy.json.JsonSlurper().parse(campaignFile.toFile())
    if (plan.schema != 'r02_rephase_marker_campaign_v1' || !plan.jobs || plan.genotype_fields_parsed != false) error('Wrong metadata-only campaign')
    def names = plan.jobs.collect { it.chromosome }
    if (names.unique(false).size() != names.size() || !names.every { it in (1..22).collect { n -> n.toString() } }) error('Repeated or invalid chromosomes')
    def jobs = Channel.fromList(plan.jobs).map { row ->
        def input = file(row.specification.path, checkIfExists: true)
        if (java.security.MessageDigest.getInstance('SHA-256').digest(input.bytes).encodeHex().toString() != row.specification.sha256) error('Input specification SHA differs')
        tuple(row.chromosome, input)
    }
    R02_REPHASE_RARE_AXIS(jobs)
    R02_REPHASE_MARKER_SUPPORT(R02_REPHASE_RARE_AXIS.out.map { chromosome, axis, receipt, specification -> tuple(chromosome, axis, specification) })
}
