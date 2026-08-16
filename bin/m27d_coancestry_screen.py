#!/usr/bin/env python3
"""Run the whole M27D design over synthetic cohorts where the answer is known.

For every point of a small screen over drift intensity, and for every seed, this builds a
cohort, runs the real stages inside the pinned container, and scores the result against
the truth the cohort was built from.  Nothing here reimplements PC-Relate or the
selection: it drives the same scripts the production path drives.

The screen exists because a single drift value would answer a single question and look
like an answer to all of them.  Three points — none, intermediate, strong — say whether
the behaviour changes with the thing being varied or is flat, and a flat result is as
informative as a sloped one.

Two deliberate limits.  The component counts are compared *only here*, because only here
is there a truth to be right about; and the run has a declared wall-clock ceiling, so a
screen that grows past its budget stops instead of quietly becoming an overnight job.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m27d_fixture_scoring import (  # noqa: E402
    delta_versus_pass0,
    read_marker_counts,
    read_pairs,
    read_truth,
    representation,
    score_pass,
)
from m27d_pipeline_chain import (  # noqa: E402
    THROUGH_CONFIGURATIONS,
    image_available,
    run as run_chain,
)
from m27d_synthetic_cohort import CohortLayout, Scenario, build  # noqa: E402

# The screen, declared here rather than buried in a call site so the values and their
# reasons are readable together.  These are drift parameters of the generative model, not
# targets for the estimate: what phi they produce after conditioning is the measurement.
SCENARIOS = (
    Scenario(
        name="panmictic_pure",
        f_background=0.0, f_intermediate=0.0, f_deme=0.0,
        rationale=(
            "No structure anywhere. Every pair's truth is exactly its pedigree value, so "
            "this isolates the estimator's own bias from any labelling question. If a "
            "parent-offspring pair is not near 0.25 here, nothing measured in the other "
            "scenarios can be attributed to coancestry."
        ),
    ),
    Scenario(
        name="null_demes",
        f_background=0.05, f_intermediate=0.0, f_deme=0.0,
        rationale=(
            "Background structure exists, the demes are labels with no genetic content. "
            "Any deme pair retained here is a false positive of the estimator rather than "
            "of the biology, so this is the floor the other points are read against."
        ),
    ),
    Scenario(
        name="below_graph_threshold",
        f_background=0.05, f_intermediate=0.01, f_deme=0.02,
        rationale=(
            "Within-deme coancestry lands near 0.030, below the 0.0442 the graph cuts at. "
            "The deme should not become a clique in pass0, so the training set should keep "
            "it and the refit should have something to fit the deme axis with."
        ),
    ),
    Scenario(
        name="above_graph_threshold",
        f_background=0.05, f_intermediate=0.02, f_deme=0.06,
        rationale=(
            "Within-deme coancestry lands near 0.079, above the graph cut. The prediction "
            "under test is that the deme becomes a clique in pass0 and collapses to one "
            "member of the training set, which is the failure the real panel shows."
        ),
    ),
    Scenario(
        name="isolate",
        f_background=0.05, f_intermediate=0.04, f_deme=0.12,
        rationale=(
            "Small-isolate drift: within-deme near 0.155, between-deme 0.04. Chosen to sit "
            "above the transition rather than to make the estimate resemble the panel; "
            "calibrating drift until the output matches observed values would be fitting "
            "the generator to the answer."
        ),
    ),
)


def detectability_margin(scenario: Scenario, n_deme: int, n_samples: int, n_markers: int) -> float:
    """How far the deme sits above the noise edge of the covariance spectrum.

    A deme is only separable by the components when its block in the relationship matrix
    exceeds the bulk edge, roughly 2*F*n_deme against sqrt(n/M).  Below one, the axis does
    not exist to be conditioned on and the method cannot be blamed for failing to use it;
    above one, it does.  Publishing the margin keeps a null result from being read as a
    verdict on the method when it is a verdict on the cohort.
    """
    edge = (n_samples / n_markers) ** 0.5
    return round(2.0 * scenario.within_deme_coancestry * n_deme / edge, 3) if edge else None


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def members(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def score_run(
    outdir: Path,
    fixture_dir: Path,
    contract: dict,
    pca_training_set_path: Path | None = None,
    pcrelate_training_set_path: Path | None = None,
) -> dict[str, object]:
    truth_rows = read_truth(fixture_dir / "truth_pairs.tsv")
    truth_json = load(fixture_dir / "truth.json")
    threshold = float(contract["pcrelate"]["primary_phi_threshold"])
    report_threshold = min(float(v) for v in contract["pcrelate"]["descriptive_phi_thresholds"])

    pass0_path = outdir / "m27d_pass0_related_pairs.private.tsv.gz"
    pass0_pairs = read_pairs(pass0_path)
    pass0_counts = read_marker_counts(pass0_path)
    strict_training = members(outdir / "training_set.txt")
    pca_training = members(pca_training_set_path or outdir / "training_set.txt")
    pcrelate_training = members(pcrelate_training_set_path or outdir / "training_set.txt")
    alternate = members(outdir / "training_set_alt.txt")
    pca_is_strict = pca_training == strict_training
    pca_alternate = alternate if pca_is_strict else None
    universe = [
        line.strip()
        for line in (outdir / "m27d_pass0_sample_universe.private.txt")
        .read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    pass0_edges = {
        pair for pair, value in pass0_pairs.items() if value >= threshold
    }
    pass0_scores = outdir / "m27d_pass0_pca_scores.private.tsv.gz"

    # Whether each deme is a clique in the pass0 graph is the causal step between drift and
    # the training set collapsing, so it is measured rather than inferred from the outcome.
    deme_structure = {}
    for deme, member_ids in truth_json["demes"].items():
        inside = set(member_ids)
        internal = [p for p in pass0_edges if p[0] in inside and p[1] in inside]
        possible = len(inside) * (len(inside) - 1) // 2
        deme_structure[deme] = {
            "n_members": len(inside),
            "n_internal_edges_pass0": len(internal),
            "possible_pairs": possible,
            "is_complete_clique_in_pass0": bool(possible) and len(internal) == possible,
            "n_in_training_set": len(inside & strict_training),
            "survivor_is_a_pedigree_member": sorted(
                sample for sample in inside & strict_training
                if any(sample in (u["child"], u["half_sibling"], u["father"], u["mother"],
                                  u["second_mother"])
                       for u in truth_json["pedigree_units"])
            ),
        }

    configurations: dict[str, object] = {}
    for config in contract["configurations"]:
        cid = config["id"]
        pairs_path = outdir / f"m27d_pcrelate_{cid}_pairs.private.tsv.gz"
        if not pairs_path.exists():
            configurations[cid] = {"status": "MISSING_OUTPUT"}
            continue
        pairs = read_pairs(pairs_path)
        marker_set = "anchor" if abs(float(config["ld_r2_max"]) - 0.2) < 1e-9 else "strict"
        configurations[cid] = {
            "n_pcs": config["n_pcs"],
            "ld_r2_max": config["ld_r2_max"],
            "marker_set": marker_set,
            "scored": score_pass(
                pairs, truth_rows, threshold, report_threshold,
                read_marker_counts(pairs_path), pcrelate_training,
            ),
            "delta_vs_pass0": delta_versus_pass0(pass0_pairs, pairs, truth_rows, threshold),
            "representation": representation(
                truth_json, pca_training, pca_alternate,
                load(outdir / f"m27d_pca_{marker_set}.json"),
                pass0_scores, universe, pass0_edges,
                training_is_pass0_independent_set=pca_is_strict,
            ),
        }
    return {
        "truth": truth_json,
        "pass0": {
            "scored": score_pass(
                pass0_pairs, truth_rows, threshold, report_threshold, pass0_counts, strict_training
            ),
            "summary": load(outdir / "m27d_pass0_pcrelate.json"),
            "deme_structure": deme_structure,
        },
        "training_set": load(outdir / "training_set.json"),
        "training_sets": {
            "n_strict": len(strict_training),
            "n_pca": len(pca_training),
            "n_pcrelate": len(pcrelate_training),
            "pca_equals_strict": pca_training == strict_training,
            "pcrelate_equals_strict": pcrelate_training == strict_training,
            "pca_equals_pcrelate": pca_training == pcrelate_training,
        },
        "configurations": configurations,
    }


def run_point(scenario: Scenario, seed: int, layout: CohortLayout, repo: Path,
              preregistration: Path, threads: int, keep: Path | None,
              point_timeout: int) -> dict[str, object]:
    workspace = Path(tempfile.mkdtemp(prefix=f"m27d-screen-{scenario.name}-{seed}-"))
    fixture_dir, outdir = workspace / "fx", workspace / "out"
    started = time.monotonic()
    try:
        build(fixture_dir, preregistration, scenario, layout, seed)
        completed = run_chain(
            fixture_dir, outdir, repo, THROUGH_CONFIGURATIONS, threads,
            timeout=point_timeout,
        )
        elapsed = time.monotonic() - started
        if completed.returncode != 0:
            return {
                "scenario": scenario.name,
                "seed": seed,
                "status": "CHAIN_FAILED",
                "elapsed_seconds": round(elapsed, 1),
                "stderr_tail": completed.stderr[-2000:],
            }
        result = score_run(outdir, fixture_dir, load(fixture_dir / "prereg.json"))
        result.update({
            "scenario": scenario.name,
            "seed": seed,
            "status": "OK",
            "elapsed_seconds": round(elapsed, 1),
            # Published, not merely computed: below one the deme is not separable by the
            # components at all, so a null result there is a verdict on the cohort and not
            # on the method. It was defined and never called, which is the same as absent.
            "detectability_margin": detectability_margin(
                scenario, layout.n_deme_members,
                result["truth"]["n_samples"], result["truth"]["n_markers"],
            ),
        })
        if keep is not None:
            destination = keep / f"{scenario.name}-seed{seed}"
            destination.mkdir(parents=True, exist_ok=True)
            for name in ("truth.json", "truth_pairs.tsv"):
                shutil.copy2(fixture_dir / name, destination / name)
        return result
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def run(args: argparse.Namespace) -> dict[str, object]:
    if not image_available():
        raise SystemExit("The pinned analysis container is not available; nothing was run")
    layout = CohortLayout(
        n_background_per_group=args.background_per_group,
        n_deme_members=args.deme_members,
        n_markers_per_chromosome=args.markers_per_chromosome,
    )
    selected = [s for s in SCENARIOS if not args.scenarios or s.name in args.scenarios]
    if not selected:
        raise SystemExit(f"No scenario matches {args.scenarios}")

    results: list[dict[str, object]] = []
    started = time.monotonic()
    stopped_early = None
    for scenario in selected:
        for seed in args.seeds:
            elapsed = time.monotonic() - started
            if elapsed > args.max_wall_seconds:
                # The budget is declared before the run and enforced here rather than
                # trusted: a screen that overruns silently is how a local job becomes an
                # overnight one, and a truncated screen must say so in its own output.
                stopped_early = {
                    "reason": "WALL_CLOCK_BUDGET_EXHAUSTED",
                    "budget_seconds": args.max_wall_seconds,
                    "elapsed_seconds": round(elapsed, 1),
                    "completed_points": len(results),
                    "planned_points": len(selected) * len(args.seeds),
                }
                break
            results.append(
                run_point(scenario, seed, layout, args.repo, args.preregistration,
                          args.threads, args.keep_truth, args.point_timeout_seconds)
            )
            print(
                f"[{results[-1]['status']}] {scenario.name} seed={seed} "
                f"{results[-1].get('elapsed_seconds')}s",
                flush=True,
            )
        if stopped_early:
            break

    summary = {
        "stage": "M27D_COANCESTRY_SCREEN",
        "scientific_result": False,
        "synthetic_cohort_only": True,
        "king_executed": False,
        "pcair_used": False,
        "cloud_executed": False,
        "container": "pinned by digest; run with --network none",
        "layout": {
            "n_background_per_group": layout.n_background_per_group,
            "n_deme_members": layout.n_deme_members,
            "n_markers_per_chromosome": layout.n_markers_per_chromosome,
            "n_chromosomes": layout.n_chromosomes,
        },
        "seeds": args.seeds,
        "scenarios": [
            {"name": s.name, "f_background": s.f_background, "f_intermediate": s.f_intermediate,
             "f_deme": s.f_deme, "within_deme_coancestry": round(s.within_deme_coancestry, 6),
             "between_deme_coancestry": round(s.between_deme_coancestry, 6),
             "rationale": s.rationale}
            for s in selected
        ],
        "wall_clock_budget_seconds": args.max_wall_seconds,
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "stopped_early": stopped_early,
        "results": results,
        "interpretation_limits": [
            "Truth here is the generative model, not a pedigree record. A conclusion "
            "transfers to the panel only in so far as the model resembles it.",
            "Component counts are compared inside this fixture because it has known "
            "answers. Nothing here licenses choosing a component count on the real panel.",
            "The demes are drawn as independent individuals from a drifted frequency, so "
            "they carry no runs of homozygosity and no chromosomal correlation. A real "
            "isolate has both, and neither is tested.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument(
        "--preregistration", type=Path,
        default=repo / "conf" / "m27d_donor_kinship_preregistration.json",
    )
    parser.add_argument("--scenarios", nargs="*", default=[])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37])
    parser.add_argument("--background-per-group", type=int, default=50)
    parser.add_argument("--deme-members", type=int, default=6)
    parser.add_argument("--markers-per-chromosome", type=int, default=700)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-wall-seconds", type=int, default=3600)
    # The whole-run budget is only checked between points, so a hung container would
    # never reach it. This bounds each point on its own.
    parser.add_argument("--point-timeout-seconds", type=int, default=900)
    parser.add_argument("--keep-truth", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
