nextflow.enable.dsl=2

// Separate, reusable prerequisite: never re-audit the FASTA in every support
// task. Deep input hashing plus the two producer hashes incur real FASTA I/O.
process R02_LAI_BUILD_VERIFICATION {
    tag "chr${params.chromosome}_REF_FASTA_audit"
    cache 'deep'
    cpus 1
    memory { "${params.audit_memory_mb} MB" }
    time { params.audit_time }
    input:
    path source_files, stageAs: 'audit_code/*'
    path vcf
    path fasta
    path fai
    path reference_contract
    path contig_map
    output:
    path 'reference_verification.json', emit: receipt
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 TMPDIR="\$PWD"
    python3 audit_code/preprocess_storage_guard.py --directory "\$PWD" --minimum-free-gib ${params.min_free_gib}
    python3 audit_code/r02_lai_build_verification.py \\
      --vcf '${vcf}' --expected-vcf-sha256 '${params.expected_vcf_sha256}' \\
      --fasta '${fasta}' --reference-contract '${reference_contract}' \\
      --expected-reference-contract-sha256 '${params.expected_reference_contract_sha256}' \\
      --contig-map '${contig_map}' --expected-contig-map-sha256 '${params.expected_contig_map_sha256}' \\
      --chromosome '${params.chromosome}' --reference-contig '${params.reference_contig}' \\
      --max-records ${params.max_records} --max-line-bytes ${params.max_line_bytes} \\
      --timeout-seconds ${params.timeout_seconds} --min-free-gib ${params.min_free_gib} \\
      --out reference_verification.json
    """
}

workflow {
    def paths = ['source_bin','vcf','fasta','fai','reference_contract','contig_map']
    def integers = ['max_records','max_line_bytes','timeout_seconds','audit_memory_mb']
    def hashes = ['expected_vcf_sha256','expected_reference_contract_sha256','expected_contig_map_sha256']
    (paths + integers + hashes + ['chromosome','reference_contig','min_free_gib','audit_time']).each { key ->
        if (params[key] == null || params[key].toString().find(/[\n\r'"`$\\;|&<>]/)) error "Missing or unsafe parameter: ${key}"
    }
    integers.each { key -> if (!(params[key].toString() ==~ /[1-9][0-9]*/)) error "Positive integer required: ${key}" }
    hashes.each { key -> if (!(params[key].toString() ==~ /[0-9a-f]{64}/)) error "Invalid SHA256: ${key}" }
    if (!(params.chromosome.toString() ==~ /([1-9]|1[0-9]|2[0-2])/)) error 'Autosome 1-22 required'
    if (!(params.reference_contig.toString() ==~ /[A-Za-z0-9_.:-]+/)) error 'Unsafe reference contig'
    if (!(params.min_free_gib.toString() ==~ /[0-9]+(\.[0-9]+)?/)) error 'Nonnegative reserve required'
    if (!(params.audit_time.toString() ==~ /[1-9][0-9]*(s|m|h)/)) error 'Explicit audit_time with s/m/h unit required'
    def fasta = file(params.fasta,checkIfExists:true)
    def fai = file(params.fai,checkIfExists:true)
    if (fai.name != fasta.name + '.fai') error 'FASTA index must have the adjacent FASTA .fai name'
    def sourceFiles = ['r02_lai_build_verification.py','r02_lai_allele_support.py',
                       'r02_genomic_pair_evidence.py','preprocess_storage_guard.py'].collect { name ->
        file("${params.source_bin}/${name}",checkIfExists:true)
    }
    R02_LAI_BUILD_VERIFICATION(sourceFiles,file(params.vcf,checkIfExists:true),fasta,fai,
        file(params.reference_contract,checkIfExists:true),file(params.contig_map,checkIfExists:true))
}
