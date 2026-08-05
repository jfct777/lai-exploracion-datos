nextflow.enable.dsl=2

import groovy.json.JsonOutput

include {
    INVENTORY_RARE_ALLELE_ORIENTATION;
    AGGREGATE_RARE_ALLELE_ORIENTATION_INVENTORY;
    WRITE_ALLELE_ORIENTATION_INVENTORY_RUN_PROVENANCE
} from '../modules/24_RARE_ALLELE_ORIENTATION_AUDIT'


def normalizeInventoryChromosomes(value) {
    def items = value instanceof List ? value : value.toString().split(',') as List
    def chromosomes = items
        .collect { it?.toString()?.trim() }
        .findAll { it }
        .collect { it.toLowerCase().startsWith('chr') ? it.substring(3) : it }
    if( chromosomes.isEmpty() || chromosomes.size() != chromosomes.toSet().size() ) {
        throw new IllegalStateException("M24 inventory: chromosome list is empty or contains duplicates")
    }
    def invalid = chromosomes.findAll { !(it ==~ /\d+/) || it.toInteger() < 1 || it.toInteger() > 22 }
    if( invalid ) {
        throw new IllegalStateException("M24 inventory accepts autosomes 1..22 only; invalid: ${invalid}")
    }
    return chromosomes
}


def resolveInventoryGitCommit(dir) {
    try {
        def head = new File("${dir}/.git/HEAD").text.trim()
        if( !head.startsWith('ref:') ) return head
        def ref = head.substring(4).trim()
        def refFile = new File("${dir}/.git/${ref}")
        if( refFile.exists() ) return refFile.text.trim()
        def packed = new File("${dir}/.git/packed-refs")
        if( packed.exists() )
            for( line in packed.readLines() )
                if( line.endsWith(" ${ref}")) return line.split(' ')[0]
    } catch( ignored ) { }
    return 'unknown'
}


workflow RARE_ALLELE_ORIENTATION_INVENTORY {
    take:
    rare_vcfs
    inventory_py
    orientation_py
    aggregate_py
    manifest_py

    main:
    def chromosomes = normalizeInventoryChromosomes(params.allele_orientation_inventory_chromosomes)
    def chromosome_set = chromosomes as Set
    def full_autosome_inventory = chromosome_set == ((1..22).collect { it.toString() } as Set)
    def canonical_dir = params.allele_orientation_inventory_canonical_m14_per_chr_dir
    if( !canonical_dir )
        throw new IllegalStateException("M24 inventory: missing canonical M14 per-chromosome directory")
    if( !params.allele_orientation_inventory_split_manifest )
        throw new IllegalStateException("M24 inventory: missing split manifest")

    def split_manifest = file(params.allele_orientation_inventory_split_manifest)
    def reference_fasta = file(params.ref_fasta)
    def reference_fai = file("${params.ref_fasta}.fai")
    def container_path = params.m14_analysis_container_image ?: params.container_image
    def container_sha = params.m14_analysis_container_digest ?: params.container_digest ?: 'unavailable'
    def provenance = [
        git_commit       : resolveInventoryGitCommit(projectDir.toString()),
        nextflow_version : workflow.nextflow.version.toString(),
        container_path   : container_path,
        container_sha256 : container_sha,
    ]
    def provenance_b64 = JsonOutput.toJson(provenance).bytes.encodeBase64().toString()
    def run_provenance = provenance + [
        nextflow_command : workflow.commandLine,
        launch_dir       : workflow.launchDir.toString(),
        project_dir      : projectDir.toString(),
        chromosomes      : chromosomes,
        orientation_universes: ['filter_cohort_2723', 'm14_subset_2619'],
        scientific_scope : 'allele orientation and additive burden only; no topology, training or evaluation',
    ]
    def run_provenance_b64 = JsonOutput.prettyPrint(
        JsonOutput.toJson(run_provenance)
    ).bytes.encodeBase64().toString()
    WRITE_ALLELE_ORIENTATION_INVENTORY_RUN_PROVENANCE(channel.value(run_provenance_b64))

    def inventory_inputs = rare_vcfs
        .filter { chr, _vcf, _tbi ->
            chromosome_set.contains(chr.toString().replaceFirst('(?i)^chr', ''))
        }
        .map { chr, vcf, vcf_tbi ->
            def normalized_chr = chr.toString().replaceFirst('(?i)^chr', '')
            def prefix = "dnabr.hg38.2723.chr${normalized_chr}"
            def canonical_summary = file("${canonical_dir}/${prefix}.sharing_scan.summary.json")
            tuple(
                normalized_chr, vcf, vcf_tbi,
                reference_fasta, reference_fai,
                canonical_summary, split_manifest,
                inventory_py, orientation_py, manifest_py,
            )
        }
    INVENTORY_RARE_ALLELE_ORIENTATION(inventory_inputs, channel.value(provenance_b64))

    if( full_autosome_inventory ) {
        AGGREGATE_RARE_ALLELE_ORIENTATION_INVENTORY(
            INVENTORY_RARE_ALLELE_ORIENTATION.out.summaries.collect(),
            INVENTORY_RARE_ALLELE_ORIENTATION.out.burdens.collect(),
            INVENTORY_RARE_ALLELE_ORIENTATION.out.manifests.collect(),
            channel.value(aggregate_py),
            channel.value(manifest_py),
            channel.value(provenance_b64),
        )
    } else {
        log.info "[M24 inventory] Partial run (${chromosomes.join(',')}): aggregate intentionally disabled"
    }

    emit:
    summaries = INVENTORY_RARE_ALLELE_ORIENTATION.out.summaries
    burdens = INVENTORY_RARE_ALLELE_ORIENTATION.out.burdens
    manifests = INVENTORY_RARE_ALLELE_ORIENTATION.out.manifests
}
