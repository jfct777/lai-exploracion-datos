// Proceso ligero para verificar que Slurm + Singularity funcionan.
// No lee ni escribe datos reales del pipeline.
process SMOKE_TEST {
    cpus   1
    memory '1 GB'
    time   '10m'
    publishDir "${params.outdir}/00_smoke_test", mode: 'copy'

    output:
    path "smoke_test.txt"

    script:
    """
    set -euo pipefail
    echo "=== SMOKE TEST ==="       >  smoke_test.txt
    echo "hostname: \$(hostname)"   >> smoke_test.txt
    echo "date:     \$(date -Iseconds)" >> smoke_test.txt
    echo "user:     \$(whoami)"     >> smoke_test.txt
    echo "pwd:      \$(pwd)"        >> smoke_test.txt
    echo "python3:  \$(python3 --version 2>&1 || echo 'not found')" >> smoke_test.txt
    echo "slurm_job_id: \${SLURM_JOB_ID:-none}" >> smoke_test.txt
    echo "OK"                       >> smoke_test.txt
    """
}
