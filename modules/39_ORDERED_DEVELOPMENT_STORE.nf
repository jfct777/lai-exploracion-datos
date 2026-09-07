nextflow.enable.dsl=2

process M39_ORDERED_DEVELOPMENT_STORE {
    tag "chr22-ordered-${roles}"
    publishDir params.m39_output_dir, mode: 'copy', overwrite: false
    container params.m39_image
    cpus 2
    memory '6 GB'
    time '1h'
    maxForks 1
    input:
    path inputs
    path bridge_contract
    path profile_contract
    path sources
    val roles
    output:
    path 'ordered-development', emit: materialized
    script:
    """
    OPENBLAS_NUM_THREADS=${task.cpus} OMP_NUM_THREADS=${task.cpus} PYTHONPATH=. \\
      python3 m39_ordered_training_store.py --bridge-config '${bridge_contract}' \\
      --profile-config '${profile_contract}' --input-dir . --roles ${roles} --outdir ordered-development
    """
    stub:
    """
    mkdir ordered-development
    python3 -c 'import json; from pathlib import Path; Path("ordered-development/stub.json").write_text(json.dumps({"scope": "STUB_NO_DATA_PROCESSED", "roles": "${roles}"}))'
    """
}
