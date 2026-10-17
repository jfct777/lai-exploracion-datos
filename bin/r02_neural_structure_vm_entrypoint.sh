#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

REPO=/home/jose.tantalean/projects/lai-exploracion-datos
STAGE="$REPO/.claude/runs/r02-corrections-20261006j/parallel-launch-20261009ap/neural-inputs/runtime/stage"
RUN_ROOT="$STAGE/operator-run"
CONTROL=/opt/r02-neural-structure
PACKET="$REPO/.claude/runs/r02-corrections-20261006j/parallel-integral-20261009au/neural-structure/packet/PACKET.json"
INDEX="$REPO/.claude/runs/r02-corrections-20261006j/parallel-launch-20261009ap/neural-inputs/package-v5/STORE_REQUESTS.json"
PRIOR="$REPO/.claude/runs/r02-corrections-20261006j/parallel-launch-20261009ap/neural-inputs/runtime/COMPLETE.json"
CATALOGUE="$REPO/.claude/runs/r02-corrections-20261006j/parallel-launch-20261009ap/neural-inputs/package-v5/structure-targets.json"
TARGET=/home/jose.tantalean/gcs-dnabr/refined/DNABR_QC/presentacion/biologico/R02_20260930/r02-common-88-preparation-20261008ak/products/neural-targets/neural_chr01_part01__BLOCK_AND_LD_MARGIN
CONTAINER=us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/m33-t0a@sha256:c03864a9ed0c56b00fd1a234daee2d17ddfa57d4c426628bd59cd9daf351ee99
IDENTITY="$CONTROL/LIVE_IDENTITY.json"
FROZEN="$REPO/.claude/runs/r02-corrections-20261006j/parallel-integral-20261009au/neural-structure/deployment/FROZEN_MANIFEST.json"

test -f "$IDENTITY"
IDENTITY_SHA=$(sha256sum "$IDENTITY" | cut -d' ' -f1)
FROZEN_SHA=$(sha256sum "$FROZEN" | cut -d' ' -f1)
export CLOUDSDK_CONFIG="$CONTROL/gcloud" NXF_OFFLINE=true
exec python3 "$REPO/bin/r02_neural_structure_operator.py" \
  --identity "$IDENTITY" --identity-sha256 "$IDENTITY_SHA" \
  --repo "$REPO" --packet "$PACKET" --request-index "$INDEX" \
  --prior-full22-gate "$PRIOR" \
  --prior-full22-gate-sha256 de6539bc0196d7a0586e19817634991e28d3c98310125af6cd4275a1730d7a17 \
  --target-catalogue "$CATALOGUE" \
  --target-catalogue-sha256 cb75e43b24de0bcef27bdfa6734045b05a95987e64223c17c005315b8c45f1c8 \
  --target-root "$TARGET" --training-container "$CONTAINER" \
  --frozen-manifest "$FROZEN" --frozen-manifest-sha256 "$FROZEN_SHA" \
  --run-root "$RUN_ROOT" --work-dir "$STAGE/nf-work" \
  --gcloud-config "$CLOUDSDK_CONFIG" \
  --publication-uri gs://projects-usp/dnaBr-lai/datalake/refined/DNABR_QC/presentacion/biologico/R02_20260930/r02-neural-structure-instrumentation-20261009au/
