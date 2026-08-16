#!/usr/bin/env python3
"""The exact order of M27D stages, written once, run inside the pinned container.

Two callers need to run the audit end to end on a synthetic cohort: the integration test,
which checks that every stage still does what it claims, and the coancestry screen, which
needs the same stages over many cohorts and only cares about the kinship output.  Written
twice, the two would drift, and the screen would end up validating a pipeline that is not
the one the test guards.

The stages are named blocks so a caller can stop where it needs to.  The screen stops
after the configurations: it has no baseline to reconcile and certifies no donor, so
running those stages would cost time and prove nothing about its question.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

IMAGE = (
    "us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/dnabr-qc@sha256:"
    "3a4661e41f7e397e986472bb8039671f85b1e8f7b86fc26af83a9837ef83d954"
)

PREAMBLE = r"""
set -euo pipefail
cd /out
export PYTHONPATH=/repo/bin
PREREG=/fx/prereg.json
M27D_THREADS=__M27D_THREADS__
PCA_TRAINING_SET_FILE=__M27D_PCA_TRAINING_SET__
PCRELATE_TRAINING_SET_FILE=__M27D_PCRELATE_TRAINING_SET__
export M27D_CONFIGURATION_IDS=__M27D_CONFIGURATION_IDS__
"""

STRATA = r"""
python3 /repo/bin/m27d_prepare_sample_strata.py --panel-vcf /fx/panel/panel.1.vcf \
  --metadata /fx/metadata.tsv --private-out strata.private.tsv \
  --summary-out strata_summary.json --suppress-below 1
"""

PREPARE = r"""
PANEL=$(for c in $(seq 1 22); do printf "%s," "/fx/panel/panel.$c.vcf"; done | sed 's/,$//')
Rscript /repo/bin/m27d_prepare_genotype_resources.R --panel-vcfs "$PANEL" \
  --exclude-bed /fx/exclude.bed --preregistration "$PREREG" --threads "$M27D_THREADS" --outdir . >/dev/null
cp /repo/bin/m27d_common.R .
"""

PASS0 = r"""
Rscript /repo/bin/m27d_pass0_pcrelate.R --gds m27d_official_panel_autosomes.gds \
  --snp-rds m27d_ld_pruned_anchor_snp_ids.rds --strata strata.private.tsv \
  --preregistration "$PREREG" --threads "$M27D_THREADS" --outdir .
"""

TRAINING_SET = r"""
python3 /repo/bin/m27d_kinship_graph.py --pairs m27d_pass0_related_pairs.private.tsv.gz \
  --samples m27d_pass0_sample_universe.private.txt \
  --call-rates m27d_pass0_sample_call_rate.private.tsv --strata strata.private.tsv \
  --preregistration "$PREREG" \
  --stage M27D_PASS0_TRAINING_SET --out-set training_set.txt \
  --out-alternate-set training_set_alt.txt --out-summary training_set.json
"""

BASELINE_IDENTITY = r"""
BASE=$(for c in $(seq 1 22); do printf "%s," "/fx/baseline/baseline.chr$c.vcf"; done | sed 's/,$//')
Rscript /repo/bin/m27d_baseline_identity.R --panel-gds m27d_official_panel_autosomes.gds \
  --baseline-vcfs "$BASE" --snp-rds m27d_ld_pruned_anchor_snp_ids.rds \
  --strata strata.private.tsv --preregistration "$PREREG" --threads "$M27D_THREADS" --outdir . >/dev/null
"""

PCA_REFIT = r"""
for pair in anchor:m27d_ld_pruned_anchor_snp_ids.rds strict:m27d_ld_pruned_strict_snp_ids.rds; do
  id="${pair%%:*}"; rds="${pair##*:}"
  Rscript /repo/bin/m27d_pca_projection.R --gds m27d_official_panel_autosomes.gds \
    --snp-rds "$rds" --strata strata.private.tsv --training-set "$PCA_TRAINING_SET_FILE" \
    --preregistration "$PREREG" --marker-set-id "$id" --threads "$M27D_THREADS" --outdir .
done
"""

CONFIGURATIONS = r"""
python3 - "$PREREG" > configs.txt <<'PY'
import json, os, sys
selected = {value for value in os.environ["M27D_CONFIGURATION_IDS"].split(",") if value}
for c in json.load(open(sys.argv[1]))["configurations"]:
    if selected and c["id"] not in selected:
        continue
    print(c["id"], "anchor" if abs(float(c["ld_r2_max"]) - 0.2) < 1e-9 else "strict")
PY

while read -r cid marker; do
  zcat "m27d_pca_${marker}_scores.private.tsv.gz" > "pca_scores_${marker}.tsv"
  Rscript /repo/bin/m27d_pcrelate_configuration.R --gds m27d_official_panel_autosomes.gds \
    --snp-rds "m27d_ld_pruned_${marker}_snp_ids.rds" --strata strata.private.tsv \
    --training-set "$PCRELATE_TRAINING_SET_FILE" --pca-scores "pca_scores_${marker}.tsv" \
    --preregistration "$PREREG" --configuration-id "$cid" --marker-set-id "$marker" \
    --threads "$M27D_THREADS" --outdir .
  rm -f "pca_scores_${marker}.tsv"
done < configs.txt
"""

SELECTION = r"""
python3 /repo/bin/m27d_candidate_selection.py --pairs m27d_pcrelate_*_pairs.private.tsv.gz \
  --strata strata.private.tsv --samples m27d_pass0_sample_universe.private.txt \
  --call-rates m27d_pass0_sample_call_rate.private.tsv \
  --baseline-identities m27d_baseline_panel_identities.private.txt \
  --stage-summaries m27d_baseline_identity.json m27d_pass0_pcrelate.json \
                    m27d_pca_anchor.json m27d_pca_strict.json \
  --preregistration "$PREREG" --suppress-below 1 \
  --out-private candidates.private.tsv --out-public candidate_counts.tsv \
  --out-gates gates.tsv --out-summary candidate_selection.json
"""

# The order is the design: pass0 estimates with everyone in the fitting set, the graph it
# produces chooses the training set, the PCA is refitted on that set alone, and only then
# does each configuration run.  Reordering any of it changes what is being tested.
FULL_AUDIT = (STRATA, PREPARE, PASS0, TRAINING_SET, BASELINE_IDENTITY, PCA_REFIT,
              CONFIGURATIONS, SELECTION)
THROUGH_CONFIGURATIONS = (STRATA, PREPARE, PASS0, TRAINING_SET, PCA_REFIT, CONFIGURATIONS)
THROUGH_TRAINING_SET = (STRATA, PREPARE, PASS0, TRAINING_SET)
REFIT_CONFIGURATIONS = (PCA_REFIT, CONFIGURATIONS)


def _safe_training_filename(value: str) -> str:
    path = Path(value)
    if path.name != value or not value or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for character in value
    ):
        raise ValueError(f"Unsafe training-set filename: {value!r}")
    return value


def chain(
    stages=FULL_AUDIT,
    threads: int = 2,
    pca_training_set: str = "training_set.txt",
    pcrelate_training_set: str = "training_set.txt",
    configuration_ids: tuple[str, ...] | None = None,
) -> str:
    if threads < 1:
        raise ValueError("threads must be positive")
    selected = configuration_ids or ()
    for value in selected:
        _safe_training_filename(value)
    replacements = {
        "__M27D_THREADS__": str(threads),
        "__M27D_PCA_TRAINING_SET__": _safe_training_filename(pca_training_set),
        "__M27D_PCRELATE_TRAINING_SET__": _safe_training_filename(pcrelate_training_set),
        "__M27D_CONFIGURATION_IDS__": ",".join(selected),
    }
    rendered = PREAMBLE + "".join(stages)
    for token, value in replacements.items():
        if rendered.count(token) != 1:
            raise ValueError(f"Expected exactly one {token} placeholder")
        rendered = rendered.replace(token, value)
    if "__M27D_" in rendered:
        raise ValueError("Unresolved M27D chain placeholder")
    return rendered


def image_available() -> bool:
    if shutil.which("docker") is None:
        return False
    probe = subprocess.run(
        ["docker", "image", "inspect", IMAGE], capture_output=True, check=False
    )
    return probe.returncode == 0


def run(
    fixture: Path,
    outdir: Path,
    repo: Path,
    stages=FULL_AUDIT,
    threads: int = 2,
    timeout: int | None = None,
    pca_training_set: str = "training_set.txt",
    pcrelate_training_set: str = "training_set.txt",
    configuration_ids: tuple[str, ...] | None = None,
) -> subprocess.CompletedProcess:
    """Run the stages in the pinned container. Nothing here reaches the network."""
    outdir.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        [
            "docker", "run", "--rm",
            "--network", "none",
            "-v", f"{repo}:/repo:ro",
            "-v", f"{fixture}:/fx:ro",
            "-v", f"{outdir}:/out",
            "-w", "/out",
            IMAGE,
            "bash",
            "-lc",
            chain(
                stages,
                threads,
                pca_training_set,
                pcrelate_training_set,
                configuration_ids,
            ),
        ],
        capture_output=True, text=True, check=False, timeout=timeout,
    )
