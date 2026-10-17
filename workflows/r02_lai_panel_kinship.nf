nextflow.enable.dsl=2

process PANEL_PREPARE {
    tag "chr${chrom}"
    cpus 2
    memory '4 GB'
    maxForks 3
    time '2h'
    cache 'deep'
    container params.preparation_image
    publishDir "${params.output_dir}/chr${chrom}", mode:'copy', overwrite:false
    input:
    tuple val(chrom), path(contract), val(digest), path(deps,stageAs:'deps/input????????')
    path code
    output:
    tuple val(chrom), path('prepared'), emit: preparations
    script:
    """
    umask 077
    python3 r02_lai_panel_prepare.py --contract '${contract}' --contract-sha256 '${digest}' --outdir prepared
    """
}

process PANEL_MERGE {
    cpus 4
    memory '8 GB'
    time '1h'
    cache 'deep'
    container params.preparation_image
    publishDir "${params.output_dir}/merge", mode:'copy', overwrite:false
    input:
    path preparations,stageAs:'preparations/chr??'
    path campaign
    val digest
    path roles
    path code
    output:
    path 'merged', emit: merged
    script:
    def list=preparations instanceof List ? preparations:[preparations]
    def options=list.collect{p->"--preparation '${p}'"}.join(' ')
    """
    umask 077
    python3 r02_lai_panel_workflow.py --contract '${campaign}' --contract-sha256 '${digest}' ${options} --outdir merged
    """
}

process PANEL_GDS {
    cpus 4
    memory '12 GB'
    time '2h'
    cache 'deep'
    container params.r_image
    publishDir "${params.output_dir}/gds", mode:'copy', overwrite:false
    input:
    path merged
    path code
    output:
    path 'gds', emit: gds
    script:
    """
    umask 077
    digest=\$(sha256sum '${merged}/gds_contract.json' | cut -d' ' -f1)
    Rscript r02_lai_panel_gds.R --contract '${merged}/gds_contract.json' --contract-sha256 "\$digest" --outdir gds
    """
}

process PANEL_PCAIR {
    cpus 4
    memory '24 GB'
    time '2h'
    cache 'deep'
    container params.r_image
    publishDir "${params.output_dir}/pcair", mode:'copy', overwrite:false
    input:
    path gds
    path code
    output:
    tuple path(gds), path('pcair'), emit: checkpoint
    script:
    """
    umask 077
    digest=\$(sha256sum '${gds}/kinship_contract.json' | cut -d' ' -f1)
    Rscript r02_lai_panel_kinship.R --stage pcair --contract '${gds}/kinship_contract.json' --contract-sha256 "\$digest" --outdir pcair
    """
}

process PANEL_PCRELATE {
    tag "PC${pcs}"
    cpus 4
    memory '24 GB'
    maxForks 1
    time '2h'
    cache 'deep'
    container params.r_image
    publishDir "${params.output_dir}/pc${pcs}", mode:'copy', overwrite:false
    input:
    tuple path(gds),path(pcair),val(pcs)
    path code
    output:
    tuple val(pcs),path('pcrelate'),emit: estimates
    script:
    """
    umask 077
    digest=\$(sha256sum '${gds}/kinship_contract.json' | cut -d' ' -f1)
    checkpoint=\$(sha256sum '${pcair}/pcair.checkpoint.rds' | cut -d' ' -f1)
    Rscript r02_lai_panel_kinship.R --stage pcrelate --contract '${gds}/kinship_contract.json' --contract-sha256 "\$digest" \
      --checkpoint '${pcair}/pcair.checkpoint.rds' --checkpoint-sha256 "\$checkpoint" --n-pcs ${pcs} --outdir pcrelate
    """
}

workflow {
    def safe={x -> x instanceof String && x.startsWith('/') && !x.tokenize('/').contains('..') && !x.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/)}
    if (![params.campaign,params.source_bin,params.output_dir].every{safe(it)} || !(params.campaign_sha256==~/[0-9a-f]{64}/)) error 'Explicit canonical paths/hash required'
    def descriptor={d-> if(!safe(d.path) || !(d.sha256==~/[0-9a-f]{64}/))error 'Invalid descriptor';file(d.path,checkIfExists:true)}
    def parse={p-> new groovy.json.JsonSlurper().parse(file(p,checkIfExists:true).toFile())}
    def campaign=parse(params.campaign)
    if(campaign.schema!='r02_lai_panel_campaign_v1' || campaign.chromosomes.toSet()!=(1..22).collect{it.toString()}.toSet() || campaign.jobs.size()!=22)error 'All22autosomes required'
    def jobs=campaign.jobs.collect{j->
       def f=descriptor(j.contract);def c=parse(j.contract.path)
       if(c.chromosome!=j.chromosome || c.schema!='r02_lai_panel_prepare_input_v1')error 'Mismatched chromosome preparation'
       def deps=[descriptor(c.roles),descriptor(c.genetic_map),descriptor(c.long_ld_bed)]
       c.projections.each{k,v-> ['vcf','index','projection_manifest','build_audit','reference_identity'].each{n->deps<<descriptor(v[n])}}
       tuple(j.chromosome,f,j.contract.sha256,deps)
    }
    if(jobs.collect{it[0]}.toSet()!=campaign.chromosomes.toSet())error 'Missing/duplicate job'
    def code=file("${params.source_bin}/*",checkIfExists:true)
    PANEL_PREPARE(Channel.fromList(jobs),code)
    PANEL_MERGE(PANEL_PREPARE.out.preparations.map{c,p->p}.collect(),file(params.campaign),params.campaign_sha256,descriptor(campaign.kinship_roles),code)
    PANEL_GDS(PANEL_MERGE.out.merged,code)
    PANEL_PCAIR(PANEL_GDS.out.gds,code)
    PANEL_PCRELATE(PANEL_PCAIR.out.checkpoint.flatMap{g,p->campaign.kinship.pc_counts.collect{n->tuple(g,p,n)}},code)
}
