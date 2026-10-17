nextflow.enable.dsl=2

params.repo = null
params.outdir = null
params.image = 'dnabr-r02-real-reader-trainer:20261007t'

process RUN_FP64_RECOVERY_FIXTURE {
    tag 'synthetic-fp64-recovery'
    cpus 1
    memory '4 GB'
    time '4m'
    cache false
    errorStrategy 'terminate'
    publishDir params.outdir, mode: 'copy', overwrite: false

    output:
    path 'MANIFEST.json'
    path 'COMMANDS_AND_OUTPUT.log'
    path 'ELAPSED_SECONDS.txt'
    path 'retained'

    script:
    """
    set -euo pipefail
    exec > >(tee COMMANDS_AND_OUTPUT.log) 2>&1
    START=\$(date +%s.%N)
    REPO='${params.repo}'
    IMAGE='${params.image}'
    PYRUN=(docker run --rm --user "\$(id -u):\$(id -g)" --cpus 1 --memory 4g \
      -v "\$REPO:/repo:ro" -v "\$PWD:\$PWD:rw" -w "\$PWD" \
      -e PYTHONPATH=/repo/bin:/repo/.claude/tests --entrypoint python3 "\$IMAGE")

    echo 'STEP prepare synthetic 22-store fixture'
    "\${PYRUN[@]}" /repo/tests/r02_neural_structure_container_fixture.py \
      --output "\$PWD/fixture-result"

    echo 'STEP build authenticated person batches'
    SPEC_SHA=\$(sha256sum fixture-result/build-spec.json | cut -d' ' -f1)
    "\${PYRUN[@]}" /repo/.claude/runs/r02-corrections-20261006j/parallel-launch-20261008ac/person_batching/build_human_autosomal_person_batches.py \
      --spec "\$PWD/fixture-result/build-spec.json" --spec-sha256 "\$SPEC_SHA" \
      --output "\$PWD/fixture-result/person-batches"

    echo 'STEP finalize synthetic-only admission'
    PACKET_SHA=\$(sha256sum fixture-result/packet.json | cut -d' ' -f1)
    "\${PYRUN[@]}" /repo/bin/r02_neural_structure_admission.py finalize \
      --packet "\$PWD/fixture-result/packet.json" --packet-sha256 "\$PACKET_SHA" \
      --build-manifest "\$PWD/fixture-result/person-batches/integration-manifest.json" \
      --output "\$PWD/fixture-result/final" --synthetic-code-test

    echo 'STEP run wrapper with forced SIGTERM recovery and explicit FP64 Adam state'
    CONTRACT_SHA=\$(sha256sum fixture-result/final/contract.json | cut -d' ' -f1)
    RESOLVED_SHA=\$(sha256sum fixture-result/resolved.json | cut -d' ' -f1)
    "\${PYRUN[@]}" /repo/bin/r02_neural_structure_run.py \
      --contract "\$PWD/fixture-result/final/contract.json" --contract-sha256 "\$CONTRACT_SHA" \
      --resolved "\$PWD/fixture-result/resolved.json" --resolved-sha256 "\$RESOLVED_SHA" \
      --arm RARE --seed 1701 --output "\$PWD/fixture-result/training/RARE" \
      --checkpoint-dir "\$PWD/fixture-result/checkpoints/RARE" \
      --measurement "\$PWD/fixture-result/measurement.json" \
      --adam-state-precision fp64 --force-recovery-step1 --synthetic-code-test \
      --checkpoint-timeout-seconds 180 --poll-seconds 0.05

    END=\$(date +%s.%N)
    awk -v start="\$START" -v end="\$END" 'BEGIN { printf "%.6f\\n", end-start }' > ELAPSED_SECONDS.txt
    echo 'STEP verify checkpoint dtype, identity and recovery evidence'
    "\${PYRUN[@]}" /repo/tests/r02_neural_fp64_fixture_verify.py \
      --root "\$PWD/fixture-result" --elapsed "\$PWD/ELAPSED_SECONDS.txt" \
      --output "\$PWD/MANIFEST.json"

    mkdir retained
    cp fixture-result/measurement.json retained/
    cp fixture-result/training/RARE/training.receipt.json retained/
    cp fixture-result/final/contract.json retained/
    cp fixture-result/checkpoints/RARE/CURRENT retained/
    cp fixture-result/checkpoints/RARE/FORCED_STEP1.json retained/
    CURRENT_MANIFEST=\$(python3 -c 'import json; print(json.load(open("fixture-result/checkpoints/RARE/CURRENT"))["manifest"])')
    cp "fixture-result/checkpoints/RARE/\$CURRENT_MANIFEST" retained/
    CURRENT_PAYLOAD=\$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["payload"])' "fixture-result/checkpoints/RARE/\$CURRENT_MANIFEST")
    cp "fixture-result/checkpoints/RARE/\$CURRENT_PAYLOAD" retained/
    """
}

workflow {
    if (!params.repo) error 'Missing --repo'
    if (!params.outdir) error 'Missing --outdir'
    RUN_FP64_RECOVERY_FIXTURE()
}
