process SAFE_BRIDGE_KAT {
    tag 'm33-safe-bridge-kat'

    cpus 1
    memory '2 GB'
    time '10m'
    cache false
    maxRetries 0
    errorStrategy 'terminate'

    input:
    path fixture
    path contract
    path base_contract
    path runner
    path core

    output:
    path 'safe_bridge_kat_output'

    script:
    """
    PYTHONPATH=. python3 ${runner} \\
      --fixture ${fixture} \\
      --contract ${contract} \\
      --base-contract ${base_contract} \\
      --output-dir safe_bridge_kat_output
    test ! -e safe_bridge_kat_output/READY
    """
}
