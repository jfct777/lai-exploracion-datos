nextflow.enable.dsl=2

import groovy.json.JsonOutput

include {
    WRITE_M14_RUN_PROVENANCE;
    ANALYZE_RARE_ALLELE_SHARING;
    AGGREGATE_RARE_ALLELE_SHARING;
    COMPARE_M14_ORIENTATION
} from '../modules/14_RARE_ALLELE_SHARING_PAINTER'


workflow {
    if( params.painting_carrier_allele_mode != 'minor_allele' ) {
        throw new IllegalStateException('m14_minor.nf requires --painting_carrier_allele_mode minor_allele')
    }
    if( params.painting_sample_ids_file || params.painting_max_samples != null ) {
        throw new IllegalStateException('M14-minor obtains its cohort only from canonical summaries')
    }

    def chromosomes = params.painting_chromosomes instanceof Collection \
        ? params.painting_chromosomes.collect { it.toString().replaceFirst('(?i)^chr', '') } \
        : params.painting_chromosomes.toString().split(',').collect {
            it.trim().replaceFirst('(?i)^chr', '')
        }
    if( chromosomes.isEmpty() || chromosomes.any { !(it ==~ /\d+/) || it.toInteger() < 1 || it.toInteger() > 22 } ) {
        throw new IllegalStateException("M14-minor accepts autosomes 1..22; received ${chromosomes}")
    }
    if( chromosomes.size() != chromosomes.unique().size() ) {
        throw new IllegalStateException("M14-minor chromosome list contains duplicates: ${chromosomes}")
    }

    def gitCommit = System.getenv('DNABR_GIT_COMMIT') ?: 'unknown'
    def provenance = [
        git_commit       : gitCommit,
        nextflow_version : workflow.nextflow.version.toString(),
        container_path   : params.m14_analysis_container_image,
        container_sha256 : params.m14_analysis_container_digest,
        run_id           : params.cloud_run_id,
    ]
    def provenanceB64 = JsonOutput.toJson(provenance).bytes.encodeBase64().toString()
    def runProvenance = provenance + [
        nextflow_command : workflow.commandLine,
        launch_dir       : workflow.launchDir.toString(),
        project_dir      : projectDir.toString(),
        scientific_scope : 'M14 minor-allele reconstruction and deterministic ALT comparison; no clustering or TEST',
        chromosomes      : chromosomes,
    ]
    def runProvenanceB64 = JsonOutput.prettyPrint(JsonOutput.toJson(runProvenance))
        .bytes.encodeBase64().toString()
    WRITE_M14_RUN_PROVENANCE(channel.value(runProvenanceB64))

    def repoDir = projectDir.resolve('..')
    def painterPy = file("${repoDir}/bin/rare_allele_sharing_painter.py")
    def orientationPy = file("${repoDir}/bin/rare_allele_orientation.py")
    def manifestPy = file("${repoDir}/bin/write_stage_manifest.py")
    def comparePy = file("${repoDir}/bin/compare_m14_orientation.py")
    def auditPy = file("${repoDir}/bin/audit_rare_allele_orientation.py")

    def reVcf = ~/dnabr\.hg38\.2723\.chr(\d+)\.rare\.vcf\.gz$/
    def chVcfs = channel
        .fromPath("${params.painting_input_dir}/${params.painting_input_glob}")
        .filter { value -> value.getName() ==~ reVcf }
        .map { vcf ->
            def matcher = (vcf.getName() =~ reVcf)
            def chrom = matcher[0][1]
            def tbi = vcf.resolveSibling("${vcf.getName()}.tbi")
            if( !tbi.exists() ) throw new IllegalStateException("Missing index for ${vcf}")
            tuple(chrom, vcf, tbi)
        }
        .filter { chrom, _vcf, _tbi -> chromosomes.contains(chrom) }

    def reSummary = ~/dnabr\.hg38\.2723\.chr(\d+)\.sharing_scan\.summary\.json$/
    def chCanonical = channel
        .fromPath("${params.painting_canonical_m14_per_chr_dir}/*.sharing_scan.summary.json")
        .filter { value -> value.getName() ==~ reSummary }
        .map { summary ->
            def matcher = (summary.getName() =~ reSummary)
            tuple(matcher[0][1], summary)
        }
        .filter { chrom, _summary -> chromosomes.contains(chrom) }

    def chScanInput = chVcfs.join(chCanonical).map { chrom, vcf, tbi, canonical ->
        tuple(chrom, vcf, tbi, canonical, painterPy, orientationPy, manifestPy)
    }
    def scan = ANALYZE_RARE_ALLELE_SHARING(
        chScanInput, channel.value(''), channel.value(provenanceB64)
    )

    def chSegments = scan.pairwise_segments.map { _chrom, value -> value }.collect()
    def chSummaries = scan.scan_summaries.map { _chrom, value -> value }.collect()
    def chManifests = scan.manifests.map { _chrom, value -> value }.collect()
    def chAggregate = chSegments
        .map { values -> tuple('aggregate', values) }
        .join(chSummaries.map { values -> tuple('aggregate', values) })
        .join(chManifests.map { values -> tuple('aggregate', values) })
        .map { _key, segments, summaries, manifests ->
            tuple(segments, summaries, manifests, painterPy, orientationPy, manifestPy)
        }
    AGGREGATE_RARE_ALLELE_SHARING(chAggregate, channel.value(provenanceB64))

    if( params.painting_enable_orientation_comparison ) {
        def historicalFiles = chromosomes.collectMany { chrom ->
            def base = "${params.painting_canonical_m14_per_chr_dir}/dnabr.hg38.2723.chr${chrom}"
            [
                file("${base}.sharing_windows.tsv.gz"),
                file("${base}.pairwise_segments.tsv.gz"),
                file("${base}.sharing_scan.summary.json"),
            ]
        }
        def chCurrentFiles = scan.sharing_windows.map { _chrom, value -> value }
            .concat(scan.pairwise_segments.map { _chrom, value -> value })
            .concat(scan.scan_summaries.map { _chrom, value -> value })
            .collect()
        COMPARE_M14_ORIENTATION(
            channel.value(historicalFiles),
            chCurrentFiles,
            comparePy,
            auditPy,
            orientationPy,
            painterPy,
            manifestPy,
            channel.value(chromosomes.join(',')),
            channel.value(provenanceB64),
        )
    }
}
