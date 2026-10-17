nextflow.enable.dsl=2

// Export one explicitly authorized role per chromosome, with candidate counts.
// This neither assigns roles nor certifies build, phase, or allele history.
process R02_LAI_PLINK_EXPORT {
    tag "chr${chrom}_${role}"
    cache 'deep'
    cpus 1
    input:
    tuple val(chrom), val(role), path(contract), path(catalogue_manifest), path(catalogue_table), path(bim_table), path(fasta), path(fai), path(reference_contract), path(contig_map)
    path bed
    path bim
    path fam
    path roles
    path crosswalk
    path source_scripts, stageAs: 'source_bin/*'
    output:
    tuple val(chrom), val(role), path('role_export'), emit: exports
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 TMPDIR="\$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 source_bin/r02_lai_plink_export.py \\
      --bed '${bed}' --bim '${bim}' --fam '${fam}' --roles '${roles}' --crosswalk '${crosswalk}' \\
      --catalogue-manifest '${catalogue_manifest}' --catalogue-table '${catalogue_table}' --bim-table '${bim_table}' \\
      --fasta '${fasta}' --fai '${fai}' --reference-contract '${reference_contract}' --contig-map '${contig_map}' \\
      --contract '${contract}' --role '${role}' --chromosome '${chrom}' --outdir role_export
    """
}

workflow {
    def commonNames = ['bed','bim','fam','roles','crosswalk']
    def jobPaths = ['contract','catalogue_manifest','catalogue_table','bim_table','fasta','fai','reference_contract','contig_map']
    def jobFields = (['chromosome','role'] + jobPaths).toSet()
    def sourceNames = ['r02_lai_plink_export.py','r02_lai_plink_catalogue.py','r02_lai_allele_support.py',
                       'r02_lai_build_verification.py','r02_genomic_pair_evidence.py','r02_lai_genotype_support.py']
    // Reject shell metacharacters and globs before file() can expand a path.
    def checkedFile = { value, label ->
        if (!(value instanceof CharSequence) || !value.toString().trim() ||
            value.toString().find(/[\n\r'"`$\\;|&<>*?\[\]{}]/)) error "Missing or unsafe input path: ${label}"
        def resolved = file(value.toString(), checkIfExists:true)
        if (resolved instanceof Collection) error "Expected one explicit file: ${label}"
        return resolved
    }
    def common = commonNames.collect { name -> checkedFile(params[name],name) }
    if (!(params.source_scripts instanceof List) || params.source_scripts.size() != sourceNames.size())
        error 'Explicit six-file source closure required'
    def scripts = params.source_scripts.collect { item -> checkedFile(item,'source_scripts') }
    if (scripts.collect { it.name }.sort() != sourceNames.sort()) error 'Unexpected source closure'
    if (!(params.jobs instanceof List) || params.jobs.isEmpty()) error 'Nonempty explicit jobs list required'
    def seen = [] as Set
    def jobs = params.jobs.collect { job ->
        if (!(job instanceof Map) || job.keySet() != jobFields) error 'Unexpected or missing export job fields'
        def chrom = job.chromosome?.toString()
        def role = job.role?.toString()
        if (!(chrom ==~ /([1-9]|1[0-9]|2[0-2])/) || !(role in ['REF_TRAIN','SOURCE_VALID']))
            error 'Unauthorized chromosome or panel role'
        if (!seen.add("${chrom}:${role}".toString())) error 'Duplicate chromosome/role export job'
        def paths = jobPaths.collect { name -> checkedFile(job[name],name) }
        def fasta = paths[4]
        def fai = paths[5]
        if (fai.toString() != fasta.toString()+'.fai') error 'FASTA requires its explicitly authenticated adjacent FAI'
        def stagedNames = (common + scripts + paths).collect { it.name }
        if (stagedNames.toSet().size() != stagedNames.size()) error 'Duplicate staged input names'
        if ('source_bin' in stagedNames || 'role_export' in stagedNames) error 'Input name collides with task output or code directory'
        tuple(chrom,role,*paths)
    }
    R02_LAI_PLINK_EXPORT(Channel.fromList(jobs),common[0],common[1],common[2],common[3],common[4],scripts)
}
