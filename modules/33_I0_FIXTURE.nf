nextflow.enable.dsl=2

process M33_I0_MAKE_FIXTURE {
    tag 'm33_i0_make_fixture'
    cache false
    cpus 1
    memory '256 MB'
    time '5m'

    input:
    path i0_script
    path authorization
    path contract
    path source_auth
    val repo_root

    output:
    tuple path('flare.anc.vcf.gz'), path('fixture_source.receipt.json')

    script:
    """
    set -euo pipefail
    python3 ${i0_script} make-fixture \
      --output-vcf flare.anc.vcf.gz \
      --manifest fixture_source.receipt.json \
      --authorization ${authorization} \
      --contract ${contract} \
      --source-auth ${source_auth} \
      --repo-root ${repo_root} \
      --container-image '${task.container}'
    """
}

process M33_I0_DERIVE_FIXTURE_INDEX {
    tag 'm33_i0_derive_fixture_index'
    cache false
    cpus 1
    memory '512 MB'
    time '5m'

    input:
    tuple path(source_vcf), path(source_manifest)
    path i0_script
    path authorization
    path contract
    path source_auth
    val repo_root

    output:
    path 'flare.anc.vcf.gz.tbi'
    path 'i0_fixture.receipt.json'
    path 'I0_FIXTURE_PASS'

    script:
    """
    set -euo pipefail
    chmod a-w ${source_vcf}
    python3 ${i0_script} derive \
      --source ${source_vcf} \
      --source-manifest ${source_manifest} \
      --authorization ${authorization} \
      --contract ${contract} \
      --source-auth ${source_auth} \
      --repo-root ${repo_root} \
      --output-tbi flare.anc.vcf.gz.tbi \
      --receipt i0_fixture.receipt.json \
      --marker I0_FIXTURE_PASS \
      --run-id ${params.m33_i0_fixture_run_id} \
      --container-image '${task.container}'
    """
}
