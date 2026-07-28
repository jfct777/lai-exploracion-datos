#!/usr/bin/env python3
"""Module 16.5 — IBD Community Detection Enhanced (biological interpretability).

Fork of Module 16 specialised for the DNABR biological questions:
  1. Macro-structure — do rare-variant co-sharing communities reflect Brazilian regions/UFs?
  2. Cryptic kinship — are small dense communities extended families?
  3. Founder effects — isolated subpopulations with high intra/low inter?

Compared to M16, this module adds (implemented across Sprints 2-5):
  * Laplacian normalisation of the sharing matrix before Sym-NMF
    (fixes the "dominant red component" degree-bias artefact).
  * Cophenetic correlation for K selection (Brunet et al. 2004 PNAS).
  * ARI between Leiden seeds as partition-stability metric.
  * Per-sample assignment confidence via the consensus co-assignment.
  * Silhouette per community on the sharing-based distance.
  * UMAP over the spectral embedding for the network layout (replaces
    Fruchterman-Reingold, which collapses large rare-variant co-sharing graphs into a
    hairball + noise ring).
  * Hierarchical within-community ordering in the ordered heatmap.
  * Optional metadata overlay: Fisher enrichment tests and sidebars.
  * Bio-detectors: cryptic kinship + founder-effect candidate tables.

Consumes Module 14 aggregate outputs (no M15 dependency), exactly like
M16.  Two complementary views of population substructure:

  * **Discrete communities** via the Leiden algorithm (``leidenalg``)
    scanned across several resolution parameters; multiple random seeds
    per resolution are summarised into a consensus co-assignment matrix.
  * **Soft ancestry-like memberships** via Symmetric NMF (Wang et al.
    2011): ``S ~= H H^T`` with ``H >= 0`` giving overlap-aware mixtures
    that behave like ADMIXTURE proportions when rows are normalised.

Pipeline stages (dispatched by ``--mode``):

    build-graph : segments TSV -> aggregated pair weights -> sparse
                  symmetric N x N matrix + igraph graph (C backend)
    leiden      : multi-resolution Leiden + consensus across seeds
    symnmf      : Sym-NMF for multiple K, NNDSVD-initialised
    validate    : intra- vs inter-community sharing (Mann-Whitney)
    plot        : all figures (network, heatmap, structure, sankey,
                  modularity curve, validation boxplot)
    report      : HTML report embedding the plots + numerical summary
    all         : chain every stage in order

Scalability notes (validated mentally for N ~= 3000, edge density ~= 30 %):

  * All N x N quantities use scipy.sparse CSR when density < 25 % so
    chr12 / N=3000 stays under ~100 MB instead of ~72 GB dense.
  * igraph is used for Leiden and the Fruchterman-Reingold layout; both
    are native C.  networkx is only imported for legacy format helpers
    (never for layout or algorithms).
  * Pair-wise segment loading is performed with ``pandas.read_csv`` in
    chunks when the TSV exceeds ``--segments-chunk-bytes`` so we never
    materialise 10^8 segment rows in RAM.
  * All plots cap their effective rendered size (down-sample nodes,
    raster heatmaps, tick sub-sampling) so a single figure stays well
    below 1 GB of RGBA buffer even at N = 10k.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import re
import string
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

# Defensive sys.path entry for project-local Python extras (e.g. colorcet,
# distinctipy) that aren't baked into dnabr-qc-hpc.sif.  Singularity
# mounts /home by default, so this directory is reachable inside the
# container.  Falls back silently when the directory doesn't exist —
# important for standalone debugging outside the project tree.
_EXTRAS_PATH = str(Path.home() / "python_extras")
if os.path.isdir(_EXTRAS_PATH) and _EXTRAS_PATH not in sys.path:
    # Prepend so updated wheels (colorcet, distinctipy, AND a newer
    # umap-learn>=0.5.11) win over the container's bundled versions.
    # Background: the container ships sklearn 1.8 where the keyword
    # ``force_all_finite`` was removed in favour of ``ensure_all_finite``
    # — but its own umap-learn 0.5.7 still passes the old name, so
    # UMAP.fit_transform() always raises TypeError.  We patch it by
    # installing umap-learn>=0.5.11 into ``python_extras`` and pointing
    # sys.path at it first.  numpy from python_extras is also 2.x and
    # ABI-compatible with the container's scipy/sklearn.
    sys.path.insert(0, _EXTRAS_PATH)

# Numba (imported transitively by umap-learn) tries to write its AOT
# JIT cache next to its source files — a read-only path inside a
# Singularity image.  Redirect it to the CWD before umap imports numba.
# The Nextflow wrapper also sets NUMBA_CACHE_DIR; this block is just a
# defensive fallback for direct script invocations.
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path.cwd() / ".numba_cache"))
try:
    Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
except OSError:  # pragma: no cover
    pass

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import ListedColormap, to_rgba  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.cluster.hierarchy import cophenet, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402
from scipy.stats import fisher_exact, mannwhitneyu  # noqa: E402

try:
    from sklearn.metrics import adjusted_rand_score, silhouette_samples  # noqa: E402
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False

try:
    from scipy.sparse.linalg import eigsh  # noqa: E402
    _HAS_EIGSH = True
except ImportError:  # pragma: no cover
    _HAS_EIGSH = False

try:
    import umap  # noqa: E402
    _HAS_UMAP = True
except ImportError:  # pragma: no cover
    _HAS_UMAP = False

try:
    from adjustText import adjust_text as _adjust_text  # noqa: E402
    _HAS_ADJUSTTEXT = True
except ImportError:  # pragma: no cover
    _HAS_ADJUSTTEXT = False

# Optional heavy deps: fail fast with a clear message if the sif is wrong.
try:
    import igraph as ig  # noqa: E402
    import leidenalg as la  # noqa: E402
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(
        "[FATAL] Module 16.5 requires `igraph` and `leidenalg` in the Singularity "
        "image. Missing: %s\n" % exc
    )
    raise

try:
    import seaborn as sns  # noqa: E402
    _HAS_SEABORN = True
except ImportError:  # pragma: no cover
    _HAS_SEABORN = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NOISE_LABEL = -1

# Default Leiden sweep includes γ=0.8 and 1.2 (finer around the expected
# peak) and drops γ=5 (empirically produces hundreds of micro-clusters
# without biological meaning at the DNABR scale).
DEFAULT_LEIDEN_RESOLUTIONS = (0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0)
# Default K sweep spans 2-12: Brazil classically has 3-4 ancestral
# components (EUR/AFR/NAT/+Asian); starting at K=2 allows macro-structure
# to emerge; K>12 typically overfits at N~10^3.
DEFAULT_NMF_K = (2, 3, 4, 5, 6, 8, 10, 12)
# Number of random restarts per K for cophenetic stability (Brunet 2004).
DEFAULT_NMF_INITS = 30

# Colour palette used for community plotting.  Up to ~50 Leiden communities
# at γ≥2 require a categorical palette where neighbouring labels remain
# visually distinct.  Three named palettes are supported via
# ``--plot-palette`` (see ``_resolve_palette``):
#
#   * 'journal'  — 24 hand-picked hues, matches Module 14/15 for visual
#                  consistency.  Cycles past 24; some hues (fabebe, fffac8)
#                  are pale and can vanish against the white background —
#                  kept as default only to preserve historical plots.
#   * 'tab20'    — Matplotlib categorical, 20 alternating dark/light hues.
#                  More uniform contrast than journal, recommended for
#                  reports where ≤20 communities matter.
#   * 'husl'     — seaborn HUSL generator: arbitrary K perceptually uniform
#                  hues equispaced in the HUSL colour space (max contrast
#                  between adjacent labels).  Best for >20 communities.
#                  Generated on demand for the exact label count, so no
#                  recycling.
_PALETTE_JOURNAL = (
    "#4363d8", "#e6194b", "#3cb44b", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080",
    "#e6beff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#808080", "#ff69b4",
    "#00ced1", "#8b4513", "#2e8b57", "#daa520",
)
_NOISE_COLOR = "#cccccc"
_BACKGROUND_COLOR = "#fafafa"

# Module-level palette state.  Resolved once at run-start via
# ``_resolve_palette`` from CLI arg, then read by ``_community_color`` and
# everywhere a community-indexed colour is needed.  This avoids threading a
# palette name through every plotting function.
_COMMUNITY_PALETTE: tuple[str, ...] = _PALETTE_JOURNAL
_PALETTE_NAME: str = "journal"


def _golden_shuffle(seq: Sequence[Any]) -> list[Any]:
    """Reorder ``seq`` so that adjacent indices land far apart in the
    original ordering.  Uses the golden-ratio quasi-random rule: position
    ``i`` of the output picks element ``round(i * phi * n) % n`` of the
    input, where phi=0.618... is the golden-ratio conjugate.  This is the
    standard low-discrepancy permutation in categorical viz (Roberts 2018
    "Unreasonable Effectiveness of Quasirandom Sequences", Glasbey 2007).

    With this shuffle applied to an equispaced hue ramp, communities
    labelled 0, 1, 2, … get hues ~222° apart instead of consecutive ones —
    the maximum perceptual contrast possible for any sequential index
    assignment without prior knowledge of cluster adjacency.
    """
    n = len(seq)
    if n <= 2:
        return list(seq)
    PHI = 0.6180339887498949
    perm: list[int] = []
    seen = set()
    for i in range(n):
        idx = int(round(i * PHI * n)) % n
        # Linear probing on collisions keeps the permutation a bijection
        # without changing the quasi-random spread meaningfully.
        while idx in seen:
            idx = (idx + 1) % n
        perm.append(idx)
        seen.add(idx)
    return [seq[i] for i in perm]


def _resolve_palette(name: str, n_colors: int) -> tuple[str, ...]:
    """Return a palette of length ≥ ``n_colors`` for the named scheme.

    Choices
    -------
    * ``journal``     — 24 hand-picked hues; historical M14/M15 default.
                         Kept only for backwards compatibility — pale
                         hues vanish on white background.
    * ``tab20``       — Matplotlib's 20 alternating dark/light bins;
                         max contrast for ≤20 communities by design.
    * ``glasbey``     — Glasbey, van der Heijden, Toh, Gout 2007 categorical
                         palette as packaged by ``colorcet.glasbey_category10``.
                         256 colours, CIELAB-distance-optimised so adjacent
                         indices have maximum perceptual contrast.  The
                         canonical state-of-the-art categorical palette
                         used by Bokeh and HoloViews.  Recommended default
                         for >10 communities.
    * ``distinctipy`` — Roberts 2020 (`distinctipy` package) generates
                         exactly ``n_colors`` colours with maximised pairwise
                         CIELAB distance.  Slower (combinatorial fit) but
                         optimal for very small N (<8) or when you want
                         the palette tuned to the exact community count.
    * ``husl``        — Perceptually-uniform HUSL ramp (seaborn) + golden-
                         ratio shuffle so labels 0,1,2,… land ~222° apart
                         in hue.  Useful when ``glasbey`` is not available.

    Unknown names raise ValueError to surface config typos at run start
    rather than silently falling back.
    """
    key = (name or "glasbey").lower().strip()
    if key in ("journal", "default"):
        return _PALETTE_JOURNAL
    if key == "tab20":
        cmap = plt.get_cmap("tab20")
        return tuple(matplotlib.colors.to_hex(cmap(i)) for i in range(cmap.N))
    if key == "glasbey":
        import colorcet
        # glasbey_category10 starts with the Tableau-10 base (max contrast
        # for the first 10 categories) and continues with 246 additional
        # CIELAB-optimised colours.  colorcet 3.2 returns [r,g,b] floats;
        # normalise to hex strings for consistency with the other paths.
        return tuple(matplotlib.colors.to_hex(c)
                     for c in colorcet.glasbey_category10)
    if key == "distinctipy":
        import distinctipy
        n = max(int(n_colors), 8)
        # distinctipy returns RGB tuples in [0,1]; convert to hex.  The
        # ``pastel_factor=0.0`` gives saturated colours (more visible on
        # white) and ``rng=42`` keeps the palette reproducible across runs.
        rgbs = distinctipy.get_colors(n, pastel_factor=0.0, rng=42)
        return tuple(matplotlib.colors.to_hex(c) for c in rgbs)
    if key == "husl":
        try:
            import seaborn as sns
        except ImportError as exc:
            raise RuntimeError(
                "--plot-palette husl requested but seaborn is missing; use "
                "--plot-palette glasbey/distinctipy/tab20/journal instead."
            ) from exc
        n = max(int(n_colors), 12)
        base = [matplotlib.colors.to_hex(c) for c in sns.color_palette("husl", n)]
        return tuple(_golden_shuffle(base))
    raise ValueError(
        f"Unknown --plot-palette '{name}'.  "
        f"Supported: journal, tab20, glasbey, distinctipy, husl."
    )

# File name conventions (consumed by Module 16 stages and the report).
NAME_GRAPH_EDGES = "graph_edges.tsv.gz"
NAME_GRAPH_NODES = "graph_nodes.tsv"
NAME_GRAPH_SUMMARY = "graph_summary.json"
NAME_GRAPH_MATRIX = "graph_sharing_matrix.npz"   # scipy sparse
NAME_LEIDEN_ASSIGN = "leiden_assignments.tsv"
NAME_LEIDEN_MOD = "leiden_modularity.tsv"
NAME_LEIDEN_CONSENSUS_TPL = "leiden_consensus_res_{r}.tsv.gz"
NAME_NMF_SOFT_TPL = "nmf_soft_memberships_k{k}.tsv"
NAME_NMF_ERR = "nmf_reconstruction_error.tsv"
NAME_VALIDATION = "validation_intra_vs_inter.tsv"
NAME_GLOBAL_SUMMARY = "global_community_summary.json"
NAME_HTML_REPORT = "report.html"
PLOTS_DIR = "plots"
# M16.5-specific outputs (sprints 2-3)
NAME_LEIDEN_ARI = "leiden_ari_by_resolution.tsv"
NAME_COMMUNITY_SILHOUETTE = "community_silhouette.tsv"
NAME_NMF_COPHENETIC = "nmf_cophenetic_by_k.tsv"
NAME_KINSHIP_CANDIDATES = "cryptic_kinship_candidates.tsv"
NAME_FOUNDER_CANDIDATES = "founder_effect_candidates.tsv"
NAME_ENRICHMENT = "community_metadata_enrichment.tsv"
NAME_COMMUNITY_LABELS = "community_labels.tsv"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logger(log_path: Path | None = None) -> logging.Logger:
    logger = logging.getLogger("m16_5")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if log_path is not None:
        fh = logging.FileHandler(log_path, mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


LOG = _setup_logger()


def _fail(msg: str, code: int = 1) -> None:
    LOG.error(msg)
    sys.exit(code)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_float_list(s: str) -> list[float]:
    return [float(x) for x in re.split(r"[,\s]+", s.strip()) if x]


def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in re.split(r"[,\s]+", s.strip()) if x]


def _parse_bool(s: str | bool) -> bool:
    if isinstance(s, bool):
        return s
    return str(s).lower() in {"1", "true", "yes", "y", "on"}


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos para todas las etapas del módulo."""
    p = argparse.ArgumentParser(
        prog="ibd_community_enhanced",
        description="Module 16.5: IBD community detection enhanced "
                    "(Laplacian-normalised Sym-NMF, cophenetic K, ARI "
                    "multi-seed, UMAP network, optional metadata overlay).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", required=True,
                   choices=("build-graph", "leiden", "symnmf", "validate",
                            "plot", "report", "all"),
                   help="Pipeline stage to run.")
    p.add_argument("--input-dir", required=True,
                   help="Directory with Module 14 aggregate outputs "
                        "(all_pairwise_segments.tsv.gz, pair_sharing_summary.tsv, "
                        "individual_sharing_summary.tsv).")
    p.add_argument("--output-dir", required=True,
                   help="Directory where Module 16.5 writes its outputs.")
    p.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1),
                   help="Number of threads for BLAS / numpy operations.")
    p.add_argument("--seed", type=int, default=42,
                   help="Base random seed (used for Leiden seeds and NMF init).")
    # Graph construction
    p.add_argument("--min-edge-bp", type=int, default=100_000,
                   help="Drop pair edges whose total_shared_bp falls below "
                        "this threshold.")
    p.add_argument("--min-max-segment-bp", type=int, default=0,
                   help="Drop pairs whose LONGEST single IBD segment is "
                        "below this threshold (0 = disabled).  Biologically "
                        "robust kinship filter: expected length of a real "
                        "IBD segment from a common ancestor within the last "
                        "~10 generations is >= 1 Mb (~ 1 cM).  Using this "
                        "flag forces streaming of the segments TSV even "
                        "when the edge-weight-transform would not require "
                        "it.")
    p.add_argument("--edge-weight-transform",
                   choices=("log1p", "raw", "n_shared_variants",
                            "mean_jaccard_weighted"),
                   default="log1p",
                   help="How to convert per-pair sharing statistics into a "
                        "scalar edge weight.")
    p.add_argument("--segments-chunk-rows", type=int, default=5_000_000,
                   help="Chunk size (rows) when streaming the segments TSV.")
    # Leiden
    p.add_argument("--leiden-resolutions", type=_parse_float_list,
                   default=list(DEFAULT_LEIDEN_RESOLUTIONS),
                   help="Comma-separated list of resolution parameters.")
    p.add_argument("--leiden-n-seeds", type=int, default=25,
                   help="Number of random restarts per resolution.")
    p.add_argument("--leiden-min-community-size", type=int, default=5,
                   help="Communities smaller than this become NOISE "
                        f"(label = {NOISE_LABEL}).")
    p.add_argument("--leiden-consensus-resolution", type=float, default=1.0,
                   help="Resolution whose consensus matrix is persisted.")
    # Sym-NMF
    p.add_argument("--nmf-k-values", type=_parse_int_list,
                   default=list(DEFAULT_NMF_K),
                   help="Comma-separated list of K for Sym-NMF.")
    p.add_argument("--nmf-inits", type=int, default=DEFAULT_NMF_INITS,
                   help="Number of restarts per K.  Used to "
                        "compute the cophenetic correlation (Brunet et al. "
                        "2004) which picks the K whose solutions are most "
                        "stable across restarts — the standard way to "
                        "select K in NMF-based ancestry analyses.")
    p.add_argument("--nmf-init-mode",
                   choices=("nndsvd-fast", "random-cophenetic"),
                   default="nndsvd-fast",
                   help="Initialisation strategy for Sym-NMF.  "
                        "'nndsvd-fast' (default): deterministic NNDSVD "
                        "seeded by the top-k eigenpairs of S + tiny noise; "
                        "fast and reproducible but converges to the same "
                        "optimum across restarts (cophenetic ≈ 1 always, "
                        "K-selection by Brunet 2004 is non-informative). "
                        "'random-cophenetic': non-negative uniform random "
                        "init per restart (genuine stochasticity); enables "
                        "Brunet's cophenetic K-selection.  Use this mode "
                        "whenever --nmf-inits > 1 and you need data-driven "
                        "K selection.")
    p.add_argument("--nmf-max-iter", type=int, default=500)
    p.add_argument("--nmf-tol", type=float, default=1e-5)
    p.add_argument("--nmf-operational-k", type=int, default=0,
                   help="Operational K override.  When >0 and present in "
                        "--nmf-k-values, this K is marked as recommended "
                        "regardless of cophenetic/dispersion analysis.  "
                        "Useful when a biologically-motivated K (e.g. K=8 "
                        "for ~8 ancestral components in admixed Latin "
                        "American cohorts) is preferred over a data-driven "
                        "selection that would always pick max-K under "
                        "asymptotically-flat cophenetic curves.  Default 0 "
                        "= use data-driven dispersion-elbow selection.")
    p.add_argument("--nmf-dispersion-threshold", type=float, default=0.85,
                   help="Minimum dispersion_index (Kim & Park 2007) for a K "
                        "to be considered as a recommended candidate.  "
                        "Higher = require more bimodal consensus matrix.  "
                        "Default 0.85 balances stringency (≥0.9 too strict "
                        "for sparse rare-variant co-sharing graphs) against permissiveness.")
    p.add_argument("--nmf-cophenetic-floor", type=float, default=0.90,
                   help="Minimum cophenetic correlation (Brunet et al. "
                        "2004) for a K to be eligible.  Standard threshold "
                        "is 0.90 — solutions below are unstable across "
                        "restarts.")
    p.add_argument("--nmf-min-marginal-coph", type=float, default=0.005,
                   help="Marginal cophenetic gain (coph[K] − coph[K-1]) "
                        "below which the curve is considered flat and the "
                        "elbow has been reached.  Default 0.005 = require "
                        "at least 0.5%% improvement to keep increasing K.")
    p.add_argument("--laplacian-normalize", type=_parse_bool, default=True,
                   help="Normalise S as D^{-1/2} S D^{-1/2} before Sym-NMF. "
                        "Equivalent to the symmetric normalised Laplacian "
                        "kernel (Shi & Malik 2000).  Corrects the "
                        "degree-bias artefact that collapses most of the "
                        "soft-membership mass into a single component when "
                        "sampling across regions/UFs is unbalanced.")
    # Bio-detector thresholds (sprint 3)
    p.add_argument("--kinship-segment-mb", type=float, default=10.0,
                   help="max_segment_bp threshold (in Mb) for pair-level "
                        "kinship candidates.  ≥1 Mb ~ 1 cM → common "
                        "ancestor <10 generations (Browning 2012).  10 Mb "
                        "is a conservative default for close kin.")
    p.add_argument("--kinship-max-size", type=int, default=15,
                   help="Community size cap for family-like candidate "
                        "communities.  Communities above this cap are "
                        "scored only at pair level, not as a single family.")
    p.add_argument("--founder-intra-inter-ratio", type=float, default=3.0,
                   help="Minimum intra/inter median sharing ratio for a "
                        "community to be flagged as a founder-effect "
                        "candidate.  Default 3.0 reflects the empirical "
                        "regime in admixed continental populations: "
                        "genuine bottlenecks/founder events show ratios "
                        "≥3 (Browning & Browning 2015), while the 10.0 "
                        "threshold inherited from European isolates "
                        "(Hutterites etc.) is unreachable in admixed "
                        "Brazilian cohorts.")
    p.add_argument("--founder-min-silhouette", type=float, default=0.0,
                   help="Minimum median silhouette for a community to be "
                        "flagged as a founder-effect candidate.  Default "
                        "0.0 disables this filter (recommended for "
                        "admixed cohorts where silhouettes are inherently "
                        "<0.05 due to continuous admixture gradients).  "
                        "Set to >0 only for cohorts with discrete "
                        "subpopulations where Rousseeuw's interpretation "
                        "(>0.3 strong, >0.5 dense) is biologically valid.")
    p.add_argument("--founder-min-size", type=int, default=10,
                   help="Minimum community size to be COMPUTED in the "
                        "founder-effect candidate analysis (everything "
                        "below this size is dropped before metrics are "
                        "evaluated).")
    p.add_argument("--founder-min-size-for-report", type=int, default=10,
                   help="Minimum community size required to APPEAR in the "
                        "final reported founder candidates table "
                        "(``is_founder_candidate=True``).  Use this to "
                        "exclude family-trio artefacts (n=3) and small "
                        "kinship blobs that mathematically pass the "
                        "intra/inter ratio test but are not biologically "
                        "founder events.  Defaults to 10, matching "
                        "fineSTRUCTURE's empirical threshold for valid "
                        "haplotype clusters in admixed cohorts (Lawson "
                        "2012, Nunes 2025 Brazil A–R).")
    # Validation / plots
    p.add_argument("--validation-resolution", type=float, default=1.0,
                   help="Resolution at which intra/inter validation runs.")
    p.add_argument("--top-samples-per-community", type=int, default=3,
                   help="Reported centroids per community in the summary JSON.")
    p.add_argument("--metadata-file", default=None,
                   help="Optional TSV with sample_id column for colouring.")
    p.add_argument("--plot-color-by", default=None,
                   help="Column of --metadata-file to overlay on plots.")
    p.add_argument("--plot-dpi", type=int, default=200)
    p.add_argument("--plot-width-inches", type=float, default=12.0)
    p.add_argument("--plot-height-inches", type=float, default=8.0)
    p.add_argument("--plot-export-pdf", type=_parse_bool, default=False)
    p.add_argument("--plot-export-svg", type=_parse_bool, default=False,
                   help="When True, also write the 2D network UMAP as an "
                        "SVG vector ('network_umap_res{r}.svg') alongside "
                        "the PNG.  Vector format: infinite zoom, editable "
                        "in Inkscape/Illustrator, renders in any browser "
                        "without WebGL.  Only applies to the 2D network "
                        "plot (heatmap/kinship stay PNG).")
    p.add_argument("--plot-network-max-nodes", type=int, default=2000,
                   help="Cap the node count rendered in the network plot "
                        "(top-degree subset is retained).  For DNABR-scale "
                        "cohorts (N≈3k) set to ≥3500 to avoid losing small "
                        "communities visually.")
    p.add_argument("--plot-heatmap-max-nodes", type=int, default=3000,
                   help="Cap the row/col count of the ordered sharing "
                        "heatmap.  Same caveat as --plot-network-max-nodes.")
    p.add_argument("--plot-cluster-label-min-size", type=int, default=10,
                   help="Minimum community size to draw a centroid label "
                        "(C{id} or paper-style annotation) on the UMAP "
                        "panel.  Set 0 to disable; low (3) labels every "
                        "cluster including family trios; high (20) labels "
                        "only major communities.  Default 10 matches the "
                        "biological threshold for valid haplotype clusters "
                        "in admixed cohorts (Lawson 2012).")
    p.add_argument("--plot-community-annotations-file", type=str, default="",
                   help="Optional TSV with columns 'community_id' and "
                        "'annotation' overriding the default 'C{id}' "
                        "centroid labels on the UMAP.  Highest priority — "
                        "always wins over auto-annotation.  Useful when "
                        "you have hand-curated labels (e.g. quilombola "
                        "names) that aren't derivable from the metadata.")
    p.add_argument("--plot-auto-annotate-by", type=str, default="",
                   help="Primary metadata column from which to AUTO-derive "
                        "UMAP centroid labels (e.g. 'finestructure_clusters' "
                        "for paper-Nunes mappings, 'Region' for geo, "
                        "'MTDNA_MAIN_HAPLOGROUP' for uniparental).  Each "
                        "community of size ≥ --plot-cluster-label-min-size "
                        "gets the MODAL value of this column as its label.  "
                        "Communities that collapse to the same modal value "
                        "are disambiguated deterministically as "
                        "'<base>·subA/B/C…' ordered by descending size.  "
                        "The full mapping (with size, purity, manual_label "
                        "column for editing) is persisted as "
                        "'community_auto_annotations.tsv'; edits to the "
                        "manual_label column survive subsequent re-runs.  "
                        "Empty (default) = no auto-annotation.")
    p.add_argument("--plot-auto-annotate-secondary", type=str, default="",
                   help="Optional second metadata column composed into the "
                        "auto label as '<primary>·<secondary>' before "
                        "collision disambiguation (e.g. primary="
                        "'finestructure_clusters', secondary="
                        "'finestructure_bigclusters' yields labels like "
                        "'BrazilA·AM_1·subA').  Useful when a single "
                        "primary column maps multiple substructures to the "
                        "same value but a coarser column resolves them.  "
                        "Empty (default) = primary only.")
    p.add_argument("--plot-adjust-labels", type=_parse_bool, default=False,
                   help="When True, run adjustText on centroid labels of "
                        "the static PNG so labels of communities packed "
                        "close in UMAP space repulse each other (force-"
                        "directed) instead of overlapping.  Adds thin grey "
                        "leader lines from the displaced label back to its "
                        "centroid.  Requires the adjustText package.")
    p.add_argument("--plot-palette", type=str, default="glasbey",
                   choices=["journal", "tab20", "glasbey",
                            "distinctipy", "husl"],
                   help="Categorical palette for community colours.  "
                        "'glasbey' (default; Glasbey 2007 / "
                        "colorcet.glasbey_category10, CIELAB-distance "
                        "optimised, 256 colours, max perceptual contrast "
                        "between adjacent indices — state-of-the-art "
                        "categorical palette used by Bokeh/HoloViews); "
                        "'distinctipy' (Roberts 2020, generates exactly "
                        "n colours with maximised pairwise distance); "
                        "'tab20' (20 alternating bins); 'journal' (legacy "
                        "24 hand-picked); 'husl' (HUSL ramp + golden "
                        "shuffle).  See ``_resolve_palette`` for details.")
    p.add_argument("--plot-umap-3d", type=_parse_bool, default=False,
                   help="When True, additionally render a 4-panel static "
                        "3D PNG ('network_umap_3d_res{r}.png': XY top-"
                        "down, XZ side, YZ front, perspective view) of a "
                        "3D UMAP fit on the same spectral embedding as "
                        "the 2D plot.  Publication-grade orthogonal-"
                        "projection multiview (Patterson 2006 EigenAnalysis, "
                        "Lawson 2012 fineSTRUCTURE) — legible offline, no "
                        "browser/WebGL required.  Cost: ~2× UMAP fit time "
                        "per resolution.")
    p.add_argument("--extra-color-columns", type=str, default="",
                   help="Comma-separated metadata columns whose values "
                        "drive the REPLOT fan-out (one task per column, "
                        "set at the Nextflow workflow layer via "
                        "--ibd_enhanced_extra_color_columns).  Each "
                        "column produces an additional set of network "
                        "PNGs coloured by that column.")
    return p


# ---------------------------------------------------------------------------
# I/O layer
# ---------------------------------------------------------------------------

@dataclass
class M14Paths:
    """Agrupa las rutas de entrada publicadas por el módulo 14."""
    segments: Path
    pair_summary: Path
    individual_summary: Path
    global_summary: Path | None = None

    @classmethod
    def from_dir(cls, input_dir: Path) -> "M14Paths":
        """Resuelve y valida las salidas requeridas dentro de un directorio de M14."""
        seg = input_dir / "all_pairwise_segments.tsv.gz"
        ps = input_dir / "pair_sharing_summary.tsv"
        ind = input_dir / "individual_sharing_summary.tsv"
        gs = input_dir / "global_sharing_summary.json"
        for p in (seg, ps, ind):
            if not p.exists():
                _fail(f"Missing required Module 14 output: {p}")
        return cls(segments=seg, pair_summary=ps, individual_summary=ind,
                   global_summary=gs if gs.exists() else None)


def load_individuals(path: Path) -> list[str]:
    """Canonical ordered sample list (keeps isolated samples as graph nodes)."""
    df = pd.read_csv(path, sep="\t", dtype={"sample_id": str})
    if "sample_id" not in df.columns:
        _fail(f"{path}: expected a 'sample_id' column")
    samples = df["sample_id"].astype(str).tolist()
    if len(samples) != len(set(samples)):
        _fail(f"{path}: sample_id column contains duplicates")
    LOG.info("Loaded %d individuals from %s", len(samples), path.name)
    return samples


def load_pair_summary(path: Path) -> pd.DataFrame:
    """Carga el resumen por pares y valida sus columnas principales."""
    df = pd.read_csv(path, sep="\t", dtype={"sample_a": str, "sample_b": str})
    expected = {"sample_a", "sample_b", "n_segments", "total_shared_bp",
                "mean_jaccard"}
    missing = expected - set(df.columns)
    if missing:
        _fail(f"{path}: missing columns {missing}")
    LOG.info("Loaded %d pair-summary rows from %s", len(df), path.name)
    return df


def load_segments_aggregated(path: Path, chunk_rows: int) -> pd.DataFrame:
    """Stream ``all_pairwise_segments.tsv.gz`` in chunks and aggregate by
    pair to avoid materialising every segment row in RAM.

    Returns a DataFrame with columns (sample_a, sample_b, n_segments,
    total_shared_bp, n_shared_variants_total, mean_jaccard,
    max_segment_bp).  ``max_segment_bp`` is the length of the longest
    single IBD-sharing segment for that pair, useful as a kinship proxy
    that is robust to liberal M14 filtering (the longest segment length
    approximately tracks the age of the most recent common ancestor).
    """
    LOG.info("Streaming segments from %s (chunk=%d rows)", path.name, chunk_rows)
    agg: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"n": 0, "bp": 0, "nsv": 0, "jsum": 0.0, "maxbp": 0}
    )
    dtype = {"sample_a": str, "sample_b": str, "length_bp": np.int64,
             "n_shared_variants": np.int64, "jaccard": np.float32}
    usecols = ["sample_a", "sample_b", "length_bp",
               "n_shared_variants", "jaccard"]
    t0 = time.time()
    total_rows = 0
    for chunk in pd.read_csv(path, sep="\t", compression="infer",
                              usecols=usecols, dtype=dtype,
                              chunksize=chunk_rows):
        total_rows += len(chunk)
        # Fully vectorised groupby inside each chunk; final merge happens
        # in Python but on *chunked* aggregates so the key space is bounded
        # by the unique pair count, not by the segment count.
        g = chunk.groupby(["sample_a", "sample_b"], sort=False)
        summary = g.agg(
            n=("length_bp", "size"),
            bp=("length_bp", "sum"),
            maxbp=("length_bp", "max"),
            nsv=("n_shared_variants", "sum"),
            jsum=("jaccard", "sum"),
        ).reset_index()
        for row in summary.itertuples(index=False):
            key = (row.sample_a, row.sample_b)
            slot = agg[key]
            slot["n"] += int(row.n)
            slot["bp"] += int(row.bp)
            slot["nsv"] += int(row.nsv)
            slot["jsum"] += float(row.jsum)
            m = int(row.maxbp)
            if m > slot["maxbp"]:
                slot["maxbp"] = m
        LOG.info("  %d chunk rows processed (pairs so far: %d, elapsed=%.1fs)",
                 total_rows, len(agg), time.time() - t0)

    if not agg:
        return pd.DataFrame(columns=[
            "sample_a", "sample_b", "n_segments", "total_shared_bp",
            "n_shared_variants_total", "mean_jaccard", "max_segment_bp",
        ])

    keys = np.fromiter(agg.keys(), dtype=object, count=len(agg))
    out = pd.DataFrame({
        "sample_a": [k[0] for k in keys],
        "sample_b": [k[1] for k in keys],
        "n_segments": [agg[k]["n"] for k in keys],
        "total_shared_bp": [agg[k]["bp"] for k in keys],
        "n_shared_variants_total": [agg[k]["nsv"] for k in keys],
        "mean_jaccard": [agg[k]["jsum"] / agg[k]["n"] for k in keys],
        "max_segment_bp": [agg[k]["maxbp"] for k in keys],
    })
    LOG.info("Aggregated %d segment rows into %d unique pairs in %.1fs "
             "(max_segment_bp: median=%.0f, p95=%.0f, max=%.0f)",
             total_rows, len(out), time.time() - t0,
             float(out["max_segment_bp"].median()),
             float(out["max_segment_bp"].quantile(0.95)),
             float(out["max_segment_bp"].max()))
    return out


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _compute_edge_weight(df: pd.DataFrame, transform: str) -> np.ndarray:
    if transform == "raw":
        w = df["total_shared_bp"].to_numpy(dtype=np.float64)
    elif transform == "log1p":
        w = np.log1p(df["total_shared_bp"].to_numpy(dtype=np.float64))
    elif transform == "n_shared_variants":
        if "n_shared_variants_total" in df.columns:
            col = "n_shared_variants_total"
        else:
            _fail("--edge-weight-transform=n_shared_variants requires the "
                  "column 'n_shared_variants_total' (produced by --mode "
                  "build-graph when streaming from segments).")
        w = df[col].to_numpy(dtype=np.float64)
    elif transform == "mean_jaccard_weighted":
        w = (df["mean_jaccard"].to_numpy(dtype=np.float64)
             * df["n_segments"].to_numpy(dtype=np.float64))
    else:  # pragma: no cover — argparse enforces choices
        _fail(f"Unknown weight transform: {transform}")
    return w


def aggregate_pair_weights(pair_df: pd.DataFrame,
                           segments_summary: pd.DataFrame | None,
                           weight_transform: str,
                           min_max_segment_bp: int = 0) -> pd.DataFrame:
    """Produce a unified pair-weight DataFrame.

    We prefer the segments-derived summary when available because it also
    carries ``n_shared_variants_total`` (useful for alternative weight
    transforms).  Otherwise we fall back to the ready-made pair summary
    shipped by Module 14.

    If ``min_max_segment_bp > 0`` and the source DataFrame carries a
    ``max_segment_bp`` column (produced by the segments streaming
    aggregator), pairs whose longest single IBD segment is below the
    threshold are dropped *before* weight computation.  This is the
    biologically most robust filter against background-ancestry noise:
    a real IBD segment from a common ancestor <= ~10 generations back
    has expected length >= 1 Mb (~ 1 cM), while noise stacks of many
    short segments can inflate ``total_shared_bp`` without any single
    segment being long.
    """
    src = segments_summary if segments_summary is not None else pair_df
    n_input = len(src)
    if min_max_segment_bp > 0:
        if "max_segment_bp" not in src.columns:
            _fail("--min-max-segment-bp requires the 'max_segment_bp' "
                  "column, which is only produced when segments are "
                  "streamed.  Do not set --min-max-segment-bp when the "
                  "pair summary is used directly.")
        keep_seg = src["max_segment_bp"].to_numpy() >= int(min_max_segment_bp)
        src = src.loc[keep_seg].reset_index(drop=True)
        LOG.info("max_segment_bp filter: kept %d / %d pairs "
                 "(threshold=%d bp)", len(src), n_input, min_max_segment_bp)
    w = _compute_edge_weight(src, weight_transform)
    keep = w > 0
    cols = ["sample_a", "sample_b", "total_shared_bp",
            "n_segments", "mean_jaccard"]
    if "max_segment_bp" in src.columns:
        cols = cols + ["max_segment_bp"]
    out = src.loc[keep, cols].copy()
    if "n_shared_variants_total" in src.columns:
        out["n_shared_variants_total"] = (
            src.loc[keep, "n_shared_variants_total"].to_numpy()
        )
    out["weight"] = w[keep]
    LOG.info("Pair-weight table: %d edges after positive-weight filter",
             len(out))
    return out


def build_sparse_matrix(pair_df: pd.DataFrame,
                        samples: Sequence[str],
                        min_edge_bp: int,
                        min_weight: float | None = None) -> tuple[sp.csr_matrix, np.ndarray]:
    """Return a symmetric CSR sharing matrix aligned to ``samples``.

    Returns:
        S    : scipy.sparse.csr_matrix, shape (N, N), dtype float64
        kept : boolean mask over pair_df rows that were kept
    """
    idx = {s: i for i, s in enumerate(samples)}
    n = len(samples)
    a_raw = pair_df["sample_a"].astype(str).to_numpy()
    b_raw = pair_df["sample_b"].astype(str).to_numpy()
    bp = pair_df["total_shared_bp"].to_numpy(dtype=np.int64)
    w = pair_df["weight"].to_numpy(dtype=np.float64)
    keep = bp >= int(min_edge_bp)
    if min_weight is not None:
        keep &= w >= float(min_weight)
    # Also drop pairs whose samples are not in the canonical list.
    in_a = np.array([s in idx for s in a_raw])
    in_b = np.array([s in idx for s in b_raw])
    keep &= in_a & in_b
    n_dropped = int((~keep).sum())
    if n_dropped:
        LOG.info("Dropped %d pair rows (bp<%d or sample missing)",
                 n_dropped, min_edge_bp)

    a_idx = np.fromiter((idx[s] for s in a_raw[keep]), dtype=np.int64,
                        count=int(keep.sum()))
    b_idx = np.fromiter((idx[s] for s in b_raw[keep]), dtype=np.int64,
                        count=int(keep.sum()))
    ww = w[keep]
    # Symmetrise: stack (a,b) and (b,a) with identical weights.
    rows = np.concatenate([a_idx, b_idx])
    cols = np.concatenate([b_idx, a_idx])
    data = np.concatenate([ww, ww])
    S = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    S.setdiag(0.0)
    S.eliminate_zeros()
    LOG.info("Sharing matrix: N=%d, nnz=%d, density=%.2e",
             n, S.nnz, S.nnz / max(1, n * n))
    return S, keep


def sparse_to_igraph(S: sp.csr_matrix, samples: Sequence[str]) -> ig.Graph:
    """Build an undirected weighted igraph from the upper triangle of S."""
    triu = sp.triu(S, k=1).tocoo()
    edges = list(zip(triu.row.tolist(), triu.col.tolist()))
    g = ig.Graph(n=len(samples), edges=edges, directed=False,
                 vertex_attrs={"name": list(samples)},
                 edge_attrs={"weight": triu.data.astype(np.float64).tolist()})
    LOG.info("igraph: |V|=%d  |E|=%d  (isolated=%d)",
             g.vcount(), g.ecount(),
             sum(1 for d in g.degree() if d == 0))
    return g


def save_graph(out_dir: Path, S: sp.csr_matrix, g: ig.Graph,
               samples: Sequence[str],
               pair_df_with_weight: pd.DataFrame,
               min_edge_bp: int, weight_transform: str,
               min_max_segment_bp: int = 0) -> None:
    """Guarda la matriz dispersa, el grafo y su información de procedencia."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Sparse matrix (npz, binary, O(nnz) IO).
    sp.save_npz(out_dir / NAME_GRAPH_MATRIX, S)
    # Edge list: human-readable TSV for downstream inspection.
    triu = sp.triu(S, k=1).tocoo()
    edges_df = pd.DataFrame({
        "sample_a": [samples[i] for i in triu.row],
        "sample_b": [samples[j] for j in triu.col],
        "weight": triu.data.astype(np.float64),
    })
    edges_df.to_csv(out_dir / NAME_GRAPH_EDGES, sep="\t",
                    index=False, compression="gzip")
    # Node list with degree statistics.
    wdeg = np.asarray(S.sum(axis=1)).ravel()
    deg = np.diff(S.indptr) if sp.isspmatrix_csr(S) else (S != 0).sum(axis=1)
    deg = np.asarray(deg).ravel().astype(np.int64)
    nodes_df = pd.DataFrame({
        "node_id": np.arange(len(samples), dtype=np.int64),
        "sample_id": samples,
        "degree": deg,
        "weighted_degree": wdeg,
    })
    nodes_df.to_csv(out_dir / NAME_GRAPH_NODES, sep="\t", index=False)
    summary = {
        "n_nodes": len(samples),
        "n_edges": int(g.ecount()),
        "n_isolated": int((deg == 0).sum()),
        "min_edge_bp": int(min_edge_bp),
        "min_max_segment_bp": int(min_max_segment_bp),
        "weight_transform": weight_transform,
        "density": float(g.ecount() / max(1, g.vcount() * (g.vcount() - 1) / 2)),
        "weighted_degree_summary": {
            "mean": float(wdeg.mean() if wdeg.size else 0.0),
            "median": float(np.median(wdeg) if wdeg.size else 0.0),
            "max": float(wdeg.max() if wdeg.size else 0.0),
            "min": float(wdeg.min() if wdeg.size else 0.0),
        },
    }
    if "max_segment_bp" in pair_df_with_weight.columns:
        ms = pair_df_with_weight["max_segment_bp"].to_numpy()
        if ms.size:
            summary["max_segment_bp_summary"] = {
                "median": float(np.median(ms)),
                "p95":    float(np.quantile(ms, 0.95)),
                "max":    float(ms.max()),
                "min":    float(ms.min()),
            }
    (out_dir / NAME_GRAPH_SUMMARY).write_text(json.dumps(summary, indent=2))
    LOG.info("Saved graph artefacts to %s", out_dir)


def load_graph(out_dir: Path) -> tuple[sp.csr_matrix, ig.Graph, list[str]]:
    """Recupera un grafo publicado junto con su matriz y orden de muestras."""
    nodes_df = pd.read_csv(out_dir / NAME_GRAPH_NODES, sep="\t",
                           dtype={"sample_id": str})
    samples = nodes_df["sample_id"].tolist()
    S = sp.load_npz(out_dir / NAME_GRAPH_MATRIX).tocsr()
    g = sparse_to_igraph(S, samples)
    return S, g, samples


# ---------------------------------------------------------------------------
# Leiden (multi-resolution + consensus)
# ---------------------------------------------------------------------------

def _relabel_small_communities(membership: np.ndarray,
                               min_size: int) -> np.ndarray:
    """Return membership with communities of size < min_size relabelled as
    NOISE_LABEL and the remaining labels compacted to ``0..K-1``.
    """
    m = np.asarray(membership, dtype=np.int64).copy()
    labels, counts = np.unique(m, return_counts=True)
    small = set(int(l) for l, c in zip(labels, counts) if c < min_size)
    if small:
        m[np.isin(m, list(small))] = NOISE_LABEL
    # Compact non-noise labels so they become 0, 1, 2 ...
    non_noise = m != NOISE_LABEL
    if non_noise.any():
        unique = np.unique(m[non_noise])
        remap = {old: new for new, old in enumerate(unique.tolist())}
        out = np.full_like(m, NOISE_LABEL)
        out[non_noise] = np.vectorize(remap.get, otypes=[np.int64])(
            m[non_noise]
        )
        return out
    return m


def _leiden_single(g: ig.Graph, resolution: float,
                    seed: int) -> tuple[np.ndarray, float]:
    part = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=float(resolution),
        seed=int(seed),
    )
    return np.asarray(part.membership, dtype=np.int64), float(part.modularity)


def run_leiden_multiresolution(g: ig.Graph,
                                resolutions: Sequence[float],
                                n_seeds: int,
                                min_community_size: int,
                                base_seed: int,
                                consensus_resolution: float,
                                ) -> tuple[pd.DataFrame, pd.DataFrame,
                                           sp.csr_matrix | None, float | None,
                                           dict[float, list[np.ndarray]]]:
    """Run Leiden at several resolutions with multiple seeds.

    For each resolution we retain the partition with the highest modularity;
    for the ``consensus_resolution`` we additionally accumulate a
    co-occurrence matrix ``C[i,j] = (# seeds where i,j in same cluster) /
    n_seeds`` which is exposed as a sparse matrix.  All per-seed
    memberships are also retained so that ARI-based stability can be
    computed downstream by ``compute_ari_multi_seed``.

    Returns
    -------
    assignments : DataFrame indexed by node with one column per resolution
                  (``community_res_{r}``).
    modularity  : DataFrame with one row per (resolution, seed) pair.
    consensus   : sparse CSR matrix at ``consensus_resolution`` (or None).
    best_modularity_consensus : modularity of the representative partition
                                at ``consensus_resolution`` (or None).
    memberships_by_res : dict[res] -> list of raw per-seed membership
                         arrays before the small-community filter, used
                         by ARI stability diagnostics.
    """
    n_nodes = g.vcount()
    rng = np.random.default_rng(base_seed)
    seeds = rng.integers(low=1, high=2**31 - 1, size=n_seeds).tolist()

    mod_rows: list[dict[str, Any]] = []
    assignments: dict[str, np.ndarray] = {}
    memberships_by_res: dict[float, list[np.ndarray]] = {}
    best_membership_consensus: np.ndarray | None = None
    best_mod_consensus: float | None = None

    # The consensus matrix is accumulated as counts in a dense (small N) or
    # LIL (large N) container; we convert to CSR only once at the end.
    accumulate_consensus = any(
        abs(r - consensus_resolution) < 1e-9 for r in resolutions
    )
    consensus_counts: np.ndarray | None = None
    if accumulate_consensus:
        if n_nodes <= 5000:
            consensus_counts = np.zeros((n_nodes, n_nodes), dtype=np.int32)
        else:
            # For very large cohorts we keep the counts sparse to cap memory.
            LOG.info("Consensus matrix will accumulate sparsely "
                     "(n_nodes=%d > 5000).", n_nodes)
            consensus_counts = None

    sparse_consensus = None
    if accumulate_consensus and consensus_counts is None:
        sparse_consensus = sp.lil_matrix((n_nodes, n_nodes), dtype=np.float32)

    for res in resolutions:
        LOG.info("Leiden: resolution=%.3f  seeds=%d", res, n_seeds)
        best_mod = -np.inf
        best_membership = None
        memberships_by_res[float(res)] = []
        for sd in seeds:
            t0 = time.time()
            memb, mod = _leiden_single(g, res, sd)
            dt = time.time() - t0
            # Raw partition size before the noise filter — useful in the CSV.
            n_raw = int(np.unique(memb).size)
            mod_rows.append({
                "resolution": float(res),
                "seed": int(sd),
                "modularity": float(mod),
                "n_communities_raw": n_raw,
                "time_sec": round(dt, 3),
            })
            memberships_by_res[float(res)].append(memb.copy())
            if mod > best_mod:
                best_mod = mod
                best_membership = memb
            if (accumulate_consensus
                    and abs(res - consensus_resolution) < 1e-9):
                if consensus_counts is not None:
                    # Broadcast equality — O(N^2) but only for N <= 5000.
                    eq = memb[:, None] == memb[None, :]
                    np.add.at(consensus_counts, np.where(eq), 1)
                else:
                    # Sparse path: fill one block per cluster.
                    labels = np.unique(memb)
                    for lbl in labels:
                        idx = np.flatnonzero(memb == lbl)
                        if idx.size < 2:
                            continue
                        rr, cc = np.meshgrid(idx, idx, indexing="ij")
                        sparse_consensus[rr.ravel(), cc.ravel()] += 1.0

        # Apply the small-community filter on the best partition.
        filtered = _relabel_small_communities(best_membership,
                                               min_community_size)
        assignments[f"community_res_{res:g}"] = filtered
        n_comm = int((np.unique(filtered) != NOISE_LABEL).sum())
        n_noise = int((filtered == NOISE_LABEL).sum())
        LOG.info("  best modularity=%.4f, communities=%d, noise=%d",
                 best_mod, n_comm, n_noise)
        if abs(res - consensus_resolution) < 1e-9:
            best_membership_consensus = filtered
            best_mod_consensus = best_mod

    if accumulate_consensus:
        if consensus_counts is not None:
            consensus = sp.csr_matrix(consensus_counts / float(n_seeds))
        else:
            consensus = sparse_consensus.tocsr() / float(n_seeds)
    else:
        consensus = None

    modularity_df = pd.DataFrame(mod_rows)
    assignments_df = pd.DataFrame(assignments)
    return (assignments_df, modularity_df, consensus, best_mod_consensus,
            memberships_by_res)


def save_leiden(out_dir: Path, samples: Sequence[str],
                assignments_df: pd.DataFrame,
                modularity_df: pd.DataFrame,
                consensus: sp.csr_matrix | None,
                consensus_resolution: float,
                ari_df: pd.DataFrame | None = None,
                confidence: np.ndarray | None = None) -> None:
    """Persist Leiden artefacts plus M16.5-specific ARI and confidence.

    ``ari_df``    : output of compute_ari_multi_seed (optional)
    ``confidence``: per-sample assignment confidence at the consensus
                    resolution (optional; aligned to ``samples`` order)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if "sample_id" not in assignments_df.columns:
        assignments_df.insert(0, "sample_id", list(samples))
    if confidence is not None:
        assignments_df = assignments_df.copy()
        assignments_df["assignment_confidence"] = confidence.astype(np.float64)
    assignments_df.to_csv(out_dir / NAME_LEIDEN_ASSIGN, sep="\t", index=False,
                           float_format="%.4f")
    modularity_df.to_csv(out_dir / NAME_LEIDEN_MOD, sep="\t", index=False)
    if consensus is not None:
        triu = sp.triu(consensus, k=1).tocoo()
        cdf = pd.DataFrame({
            "sample_a": [samples[i] for i in triu.row],
            "sample_b": [samples[j] for j in triu.col],
            "consensus_frequency": triu.data.astype(np.float32),
        })
        out = out_dir / NAME_LEIDEN_CONSENSUS_TPL.format(
            r=f"{consensus_resolution:g}"
        )
        cdf.to_csv(out, sep="\t", index=False, compression="gzip")
    if ari_df is not None and not ari_df.empty:
        ari_df.to_csv(out_dir / NAME_LEIDEN_ARI, sep="\t", index=False,
                       float_format="%.4f")
    LOG.info("Saved Leiden artefacts to %s", out_dir)


def load_leiden_assignments(out_dir: Path) -> pd.DataFrame:
    """Carga las asignaciones Leiden publicadas por una ejecución anterior."""
    return pd.read_csv(out_dir / NAME_LEIDEN_ASSIGN, sep="\t",
                       dtype={"sample_id": str})


# ---------------------------------------------------------------------------
# Symmetric NMF
# ---------------------------------------------------------------------------

def _nndsvd_init(S: np.ndarray, k: int, seed: int) -> np.ndarray:
    """NNDSVD-flavoured initialisation for Sym-NMF.

    Because the target is ``S ~= H H^T``, we seed H with the top-k positive
    eigenvectors of the symmetrised ``S`` scaled by ``sqrt(lambda)``.  Tiny
    positive noise is added so the multiplicative update (which is fixed on
    exact zeros) can explore.

    Esta inicialización es prácticamente determinista; la semilla solo
    perturbs the noise term (O(1e-4)).  Multiple restarts converge to the
    same optimum, so cophenetic correlation (Brunet 2004) is uninformative
    in this mode.  Use ``_random_nonneg_init`` for genuine cophenetic
    K-selection.
    """
    rng = np.random.default_rng(seed)
    n = S.shape[0]
    try:
        # eigsh on the dense matrix is fine up to N ~ 10^4 (O(N^2 k)).
        vals, vecs = np.linalg.eigh(S)
    except np.linalg.LinAlgError:  # pragma: no cover
        H = rng.random((n, k), dtype=np.float64) * 1e-3
        return H
    # Retain the top-k largest eigenpairs.
    idx = np.argsort(vals)[::-1][:k]
    vals = np.clip(vals[idx], a_min=0.0, a_max=None)
    vecs = vecs[:, idx]
    H = np.abs(vecs) * np.sqrt(vals)[None, :]
    # Add tiny noise (avoid zeros frozen by multiplicative updates).
    H += rng.random(H.shape) * (H.max() if H.size else 1.0) * 1e-4 + 1e-8
    return H


def _random_nonneg_init(S: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Uniform random non-negative initialisation for Sym-NMF.

    Produces genuine inter-restart variability so that Brunet et al. (2004)
    cophenetic K-selection is informative.  Scale matches NNDSVD
    (~sqrt(S.max() / k)) so the multiplicative update converges in a
    similar number of iterations to the deterministic seed.
    """
    rng = np.random.default_rng(seed)
    n = S.shape[0]
    s_max = float(S.max()) if S.size else 1.0
    scale = float(np.sqrt(max(s_max, 1e-12) / max(k, 1)))
    # Uniform on [0, scale]; ε floor avoids frozen zeros under the
    # multiplicative update.
    H = rng.random((n, k), dtype=np.float64) * scale
    H += 1e-8
    return H


def symnmf(S: np.ndarray, k: int, max_iter: int, tol: float,
           seed: int, init_mode: str = "nndsvd-fast",
           ) -> tuple[np.ndarray, list[float]]:
    """Symmetric NMF: ``min || S - H H^T ||_F^2, H >= 0``.

    Multiplicative update (Ding et al. 2005):
        H <- H * (S H) / (H H^T H + eps)

    ``init_mode``:
      * ``nndsvd-fast`` (default): NNDSVD-style spectral seed (deterministic
        across restarts up to O(1e-4) noise; fast convergence).
      * ``random-cophenetic``: uniform non-negative random init per restart;
        required for Brunet et al. (2004) cophenetic K-selection.
    """
    if init_mode == "random-cophenetic":
        H = _random_nonneg_init(S, k, seed)
    else:
        H = _nndsvd_init(S, k, seed)
    eps = 1e-12
    errors: list[float] = []
    prev_err = np.inf
    for it in range(1, max_iter + 1):
        SH = S @ H
        HtH = H.T @ H
        denom = H @ HtH + eps
        # Element-wise multiplicative update with half-step damping for
        # numerical stability on large condition numbers.
        H_new = H * np.sqrt(SH / denom)
        H = H_new
        # Error every 10 iterations (||S - HH^T||_F is O(N^2) and would
        # dominate the cost if computed every step).
        if it % 10 == 0 or it == max_iter:
            err = float(np.linalg.norm(S - H @ H.T, ord="fro"))
            errors.append(err)
            if abs(prev_err - err) / max(prev_err, 1e-12) < tol:
                LOG.info("Sym-NMF k=%d converged at iter=%d (err=%.3e)",
                         k, it, err)
                break
            prev_err = err
    else:
        LOG.info("Sym-NMF k=%d reached max_iter=%d (err=%.3e)",
                 k, max_iter, errors[-1] if errors else float("nan"))
    return H, errors


def run_symnmf_multi_k(S: sp.spmatrix, k_values: Sequence[int],
                       max_iter: int, tol: float, seed: int,
                       init_mode: str = "nndsvd-fast",
                       ) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    """Run Sym-NMF for each K and collect reconstruction-error curves."""
    S_dense = S.toarray().astype(np.float64, copy=False)
    # Normalise scale for numerical stability; restore after decomposition
    # is irrelevant for soft-membership interpretation (we row-normalise H).
    scale = S_dense.max()
    if scale > 0:
        S_dense = S_dense / scale
    out: dict[int, np.ndarray] = {}
    err_rows: list[dict[str, Any]] = []
    for k in k_values:
        LOG.info("Sym-NMF: k=%d  max_iter=%d  tol=%.1e  init=%s",
                 k, max_iter, tol, init_mode)
        H, errs = symnmf(S_dense, k=int(k), max_iter=int(max_iter),
                         tol=float(tol), seed=int(seed),
                         init_mode=init_mode)
        # Row-normalise so H[i, :] sums to 1 -> ADMIXTURE-like proportions.
        row_sums = H.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        H_norm = H / row_sums
        out[int(k)] = H_norm
        for step, e in enumerate(errs, start=1):
            err_rows.append({"k": int(k),
                             "eval_step": int(step * 10),
                             "frobenius_error": float(e)})
    err_df = pd.DataFrame(err_rows)
    return out, err_df


def save_symnmf(out_dir: Path, samples: Sequence[str],
                H_by_k: dict[int, np.ndarray], err_df: pd.DataFrame) -> None:
    """Guarda las matrices de membresía SymNMF y sus errores de reconstrucción."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, H in H_by_k.items():
        cols = [f"component_{i+1}" for i in range(H.shape[1])]
        df = pd.DataFrame(H, columns=cols)
        df.insert(0, "sample_id", samples)
        df.to_csv(out_dir / NAME_NMF_SOFT_TPL.format(k=k),
                  sep="\t", index=False, float_format="%.6g")
    err_df.to_csv(out_dir / NAME_NMF_ERR, sep="\t", index=False,
                  float_format="%.6g")
    LOG.info("Saved Sym-NMF artefacts to %s", out_dir)


def load_symnmf(out_dir: Path, k: int) -> pd.DataFrame | None:
    """Carga las membresías SymNMF para un valor de k si están disponibles."""
    p = out_dir / NAME_NMF_SOFT_TPL.format(k=k)
    if not p.exists():
        return None
    return pd.read_csv(p, sep="\t", dtype={"sample_id": str})


# ---------------------------------------------------------------------------
# M16.5 — Sprint 2: robustness metrics (ARI, confidence, silhouette,
# Laplacian normalisation, cophenetic NMF)
# ---------------------------------------------------------------------------

def laplacian_normalize(S: sp.spmatrix) -> sp.csr_matrix:
    """Return the symmetric normalised Laplacian kernel of S.

    Definition (Shi & Malik 2000; Ng, Jordan & Weiss 2002):
        S_norm = D^{-1/2} S D^{-1/2}       where D = diag(sum_j S_ij)

    In the rare-variant co-sharing context, the raw sharing matrix S is biased toward
    samples with many neighbours — e.g. individuals from oversampled
    Brazilian regions accumulate artificially higher total_shared_bp
    simply because more comparison partners exist in the cohort.  A
    subsequent Sym-NMF on raw S therefore places most of the mass in
    the component whose spectral direction aligns with "high degree",
    which empirically presented as the single dominant red component in
    the M16 STRUCTURE plot.

    Normalising by D^{-1/2} converts S into a kernel whose eigen-
    decomposition decouples modular structure from raw degree, i.e.
    Sym-NMF now recovers real sub-population substructure rather than
    sampling artefacts.
    """
    d = np.asarray(S.sum(axis=1)).ravel()
    # Guard against zero-degree nodes (isolated samples survive as-is).
    with np.errstate(divide="ignore"):
        inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = sp.diags(inv_sqrt)
    return (D_inv_sqrt @ S @ D_inv_sqrt).tocsr()


def compute_ari_multi_seed(memberships_by_res: dict[float, list[np.ndarray]],
                            ) -> pd.DataFrame:
    """Pairwise Adjusted Rand Index between Leiden seeds at each resolution.

    ARI (Hubert & Arabie 1985) is the standard chance-corrected measure
    of partition agreement.  High median ARI across seeds at a given γ
    means the partition is reproducible (biological signal); low ARI
    means the algorithm is chasing stochastic noise.  We summarise
    (min, q25, median, q75, max) so that a single γ can be selected as
    "robust" — the γ that maximises both modularity AND ARI.
    """
    rows: list[dict[str, Any]] = []
    if not _HAS_SKLEARN:
        LOG.warning("scikit-learn not available; skipping ARI computation.")
        return pd.DataFrame(rows)
    for res, memberships in memberships_by_res.items():
        if len(memberships) < 2:
            continue
        ari_values: list[float] = []
        for i in range(len(memberships)):
            for j in range(i + 1, len(memberships)):
                ari_values.append(
                    float(adjusted_rand_score(memberships[i], memberships[j]))
                )
        if not ari_values:
            continue
        arr = np.asarray(ari_values, dtype=np.float64)
        rows.append({
            "resolution": float(res),
            "n_pairs": int(arr.size),
            "median_ari": float(np.median(arr)),
            "q25_ari": float(np.quantile(arr, 0.25)),
            "q75_ari": float(np.quantile(arr, 0.75)),
            "min_ari": float(arr.min()),
            "max_ari": float(arr.max()),
        })
    return pd.DataFrame(rows)


def compute_assignment_confidence(
    consensus: sp.csr_matrix,
    membership: np.ndarray,
) -> np.ndarray:
    """Per-sample confidence: median co-assignment frequency with peers.

    For sample i assigned to community c, confidence[i] = median over
    j in community c of C[i, j], where C is the consensus matrix
    (fraction of Leiden seeds that placed i and j in the same cluster).

    Samples with confidence < ~0.5 are "border samples" — candidates to
    recent admixture between two groups or ambiguous kinship.  Noise
    samples get confidence = NaN because their community has size = 1.
    """
    n = len(membership)
    out = np.full(n, np.nan, dtype=np.float64)
    if consensus is None:
        return out
    C = consensus.tocsr()
    for comm_label in np.unique(membership):
        if int(comm_label) == NOISE_LABEL:
            continue
        idx = np.flatnonzero(membership == comm_label)
        if idx.size < 2:
            continue
        # Slice the consensus block and compute per-row median (over
        # other members of the same community).
        sub = C[idx][:, idx].toarray()
        # Ignore self (diagonal) when computing the median.
        np.fill_diagonal(sub, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            out[idx] = np.nanmedian(sub, axis=1)
    return out


def compute_silhouette_per_community(
    S: sp.spmatrix,
    membership: np.ndarray,
) -> pd.DataFrame:
    """Silhouette per community on the sharing-based distance matrix.

    Rousseeuw (1987): silhouette quantifies whether each sample is
    closer to its own cluster than to any other.  We build the
    distance from normalised sharing:
        D[i, j] = 1 - S_row_normalised[i, j]
    so that strong sharing implies short distance.  Samples with
    silhouette > 0.3 form a well-separated cluster (biologically
    meaningful founder or endogamous group); silhouette < 0 indicates
    misclassification or admixture between two groups.
    """
    if not _HAS_SKLEARN:
        LOG.warning("scikit-learn not available; skipping silhouette.")
        return pd.DataFrame()
    non_noise = membership != NOISE_LABEL
    if non_noise.sum() < 2:
        return pd.DataFrame()
    # Row-normalise sharing so each row sums to 1 (similarity); then
    # convert to a symmetric distance matrix in [0, 1].
    dense = S.toarray().astype(np.float64)
    row_sums = dense.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    sim = dense / row_sums
    # Symmetrise the similarity (average with transpose) and convert to
    # distance.  Zero similarity → distance 1; self-similarity → 0.
    sim = 0.5 * (sim + sim.T)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    # Silhouette requires at least 2 distinct labels among non-noise.
    labels = membership[non_noise]
    if np.unique(labels).size < 2:
        return pd.DataFrame()
    sil = silhouette_samples(
        dist[non_noise][:, non_noise], labels, metric="precomputed"
    )
    df = pd.DataFrame({
        "community": labels,
        "silhouette": sil,
    })
    agg = df.groupby("community").agg(
        n_samples=("silhouette", "size"),
        mean_silhouette=("silhouette", "mean"),
        median_silhouette=("silhouette", "median"),
        p25_silhouette=("silhouette", lambda s: float(np.quantile(s, 0.25))),
        p75_silhouette=("silhouette", lambda s: float(np.quantile(s, 0.75))),
    ).reset_index()
    return agg


def _symnmf_dominant_components(H: np.ndarray) -> np.ndarray:
    """Assign each sample to its top-loading component (tie-break: lowest idx)."""
    return np.argmax(H, axis=1).astype(np.int64)


def run_symnmf_cophenetic(
    S: sp.spmatrix,
    k_values: Sequence[int],
    n_inits: int,
    max_iter: int,
    tol: float,
    base_seed: int,
    laplacian: bool,
    init_mode: str = "nndsvd-fast",
    k_operational: int = 0,
    dispersion_threshold: float = 0.85,
    cophenetic_floor: float = 0.90,
    min_marginal_coph: float = 0.005,
) -> tuple[dict[int, np.ndarray], pd.DataFrame, pd.DataFrame]:
    """Sym-NMF with cophenetic correlation for K selection.

    Pipeline:
      1. Optionally Laplacian-normalise S (fixes degree bias, see
         ``laplacian_normalize``).
      2. For each K, run ``n_inits`` NNDSVD-initialised Sym-NMF
         restarts with distinct seeds.
      3. Build a consensus matrix C[i, j] = fraction of restarts in
         which samples i and j share the same dominant component.
      4. Linkage(1 - C, 'average') → cophenetic correlation (Brunet et
         al. 2004 PNAS): Pearson correlation between 1-C and the
         cophenetic distances of the UPGMA dendrogram.
      5. Keep the restart with the lowest reconstruction error as the
         representative H for that K.

    K selection rule of thumb (Brunet 2004): pick the largest K whose
    cophenetic correlation exceeds ~0.90 — that K captures the most
    substructure without overfitting.

    Returns
    -------
    H_by_k : dict[K, H_row_normalised]
    err_df : per-K reconstruction-error curves (long format)
    coph_df : per-K cophenetic correlation + consensus diagnostics
    """
    dense = S.toarray().astype(np.float64, copy=False)
    if laplacian:
        LOG.info("Sym-NMF: applying Laplacian normalisation "
                 "D^{-1/2} S D^{-1/2} (fixes degree-bias artefact).")
        dense = laplacian_normalize(S).toarray().astype(np.float64, copy=False)
    scale = dense.max()
    if scale > 0:
        dense = dense / scale
    rng = np.random.default_rng(base_seed)
    H_by_k: dict[int, np.ndarray] = {}
    err_rows: list[dict[str, Any]] = []
    coph_rows: list[dict[str, Any]] = []
    n = dense.shape[0]
    if init_mode == "nndsvd-fast" and n_inits > 1:
        LOG.warning(
            "run_symnmf_cophenetic: init_mode='nndsvd-fast' with "
            "n_inits=%d.  NNDSVD is essentially deterministic (only "
            "O(1e-4) noise varies between restarts), so cophenetic "
            "correlation will be ≈1 for all K and is uninformative for "
            "K-selection.  Use --nmf-init-mode random-cophenetic for "
            "genuine Brunet 2004 K-selection, or set --nmf-inits 1 "
            "to skip the redundant restarts.", n_inits)
    for k in k_values:
        k_int = int(k)
        LOG.info("Sym-NMF: k=%d  n_inits=%d  max_iter=%d  init=%s",
                 k_int, n_inits, max_iter, init_mode)
        seeds = rng.integers(low=1, high=2**31 - 1, size=n_inits).tolist()
        # Consensus counter: how often do i,j land in the same top component.
        C = np.zeros((n, n), dtype=np.float64)
        best_err = np.inf
        best_H = None
        rec_err_final: list[float] = []
        for s in seeds:
            H, errs = symnmf(dense, k=k_int, max_iter=max_iter, tol=tol,
                              seed=int(s), init_mode=init_mode)
            final_err = errs[-1] if errs else float("nan")
            rec_err_final.append(final_err)
            if final_err < best_err:
                best_err = final_err
                best_H = H
            dom = _symnmf_dominant_components(H)
            # eq[i,j] = 1 iff dom[i]==dom[j]
            eq = (dom[:, None] == dom[None, :]).astype(np.float64)
            C += eq
        C /= float(n_inits)
        # Cophenetic correlation (Brunet 2004) on consensus.
        # Use 1-C as a dissimilarity; UPGMA linkage.
        coph = float("nan")
        dispersion = float("nan")
        if n >= 3:
            try:
                dmat = 1.0 - C
                np.fill_diagonal(dmat, 0.0)
                # squareform requires strict symmetry and zero diagonal.
                dmat = 0.5 * (dmat + dmat.T)
                condensed = squareform(dmat, checks=False)
                Z = linkage(condensed, method="average")
                coph_val, _ = cophenet(Z, condensed)
                coph = float(coph_val)
                # Dispersion index (Kim & Park 2007): sum over (i,j) of
                # 4*(C[i,j]-0.5)^2 / (N*(N-1)).  = 1 when C is perfectly
                # bimodal (0 or 1), < 1 when entries are in between.
                off = C[np.triu_indices(n, k=1)]
                dispersion = float(np.mean(4.0 * (off - 0.5) ** 2))
            except Exception as exc:  # pragma: no cover
                LOG.warning("Cophenetic failed for k=%d: %s", k_int, exc)
        # Per-init diagnostic stats (genuine variability across restarts).
        err_arr = np.asarray(rec_err_final, dtype=np.float64)
        err_min = float(err_arr.min()) if err_arr.size else float("nan")
        err_max = float(err_arr.max()) if err_arr.size else float("nan")
        err_mean = float(err_arr.mean()) if err_arr.size else float("nan")
        err_std = float(err_arr.std(ddof=1)) if err_arr.size > 1 else 0.0
        err_cv = (err_std / err_mean) if (err_mean and np.isfinite(err_mean)) else float("nan")
        # Detect degenerate solutions: an init with err << median is suspect
        # (likely H≈0 trivial solution — multiplicative update fixed-point).
        # Flag inits whose error is more than 5×CV below the median.
        n_degenerate = 0
        if err_arr.size >= 3 and err_std > 0:
            med = float(np.median(err_arr))
            n_degenerate = int(np.sum(err_arr < (med - 5.0 * err_std)))
            if n_degenerate > 0:
                LOG.warning("k=%d: %d/%d inits flagged as potentially "
                            "degenerate (err << median by >5σ).  "
                            "Consider filtering these from cophenetic "
                            "computation.", k_int, n_degenerate, n_inits)
        LOG.info("k=%d  inits=%d  err_mean=%.4f  err_std=%.6f  err_cv=%.2e  "
                 "coph=%.4f  disp=%.4f",
                 k_int, n_inits, err_mean, err_std, err_cv, coph, dispersion)
        coph_rows.append({
            "k": k_int,
            "n_inits": int(n_inits),
            "mean_reconstruction_error": err_mean,
            "min_reconstruction_error": err_min,
            "max_reconstruction_error": err_max,
            "std_reconstruction_error": err_std,
            "cv_reconstruction_error": err_cv,
            "n_degenerate_inits": n_degenerate,
            "cophenetic_correlation": coph,
            "dispersion_index": dispersion,
        })
        # Emit the best H (lowest reconstruction error), row-normalised.
        if best_H is None:  # pragma: no cover — shouldn't happen
            best_H = np.zeros((n, k_int))
        row_sums = best_H.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        H_by_k[k_int] = best_H / row_sums
        # Log all per-init error trajectories for diagnostic purposes.
        for i, e in enumerate(rec_err_final):
            err_rows.append({
                "k": k_int,
                "init_idx": int(i),
                "frobenius_error": float(e),
            })
    err_df = pd.DataFrame(err_rows)
    coph_df = pd.DataFrame(coph_rows)
    if not coph_df.empty:
        coph_df = coph_df.sort_values("k").reset_index(drop=True)
        coph_df["is_recommended_k"] = _select_recommended_k(
            coph_df,
            operational_k=k_operational,
            dispersion_threshold=dispersion_threshold,
            cophenetic_floor=cophenetic_floor,
            min_marginal_coph=min_marginal_coph,
        )
    return H_by_k, err_df, coph_df


# ---------------------------------------------------------------------------
# K selection — dispersion-elbow with optional operational override
# ---------------------------------------------------------------------------

def _select_recommended_k(coph_df: pd.DataFrame,
                          *,
                          operational_k: int,
                          dispersion_threshold: float,
                          cophenetic_floor: float,
                          min_marginal_coph: float) -> pd.Series:
    """Pick the recommended K with biological + statistical criteria.

    Decision rule (in priority order):

      1. **Operational override.** If ``operational_k`` > 0 and present in the
         tested K grid, use it directly.  This lets the user lock K=8 or K=12
         based on biological priors (e.g. expected number of ancestral
         components) without running into the asymptotic-cophenetic problem.

      2. **Dispersion-elbow** (Kim & Park 2007).  Among Ks whose cophenetic
         correlation ≥ ``cophenetic_floor`` (default 0.90, Brunet 2004),
         find the smallest K such that:
            (a) ``dispersion_index ≥ dispersion_threshold`` AND
            (b) marginal cophenetic gain over K-1 is < ``min_marginal_coph``
                (the curve has flattened).
         This avoids the failure mode of "always pick max-K" when
         cophenetic is asymptotically flat.

      3. **Fallback.** Highest dispersion_index among Ks meeting the
         cophenetic floor; if none, simply max cophenetic.
    """
    # 1. Operational override.
    if operational_k and operational_k > 0:
        if int(operational_k) in coph_df["k"].astype(int).tolist():
            LOG.info("K-selection: operational override → K = %d.",
                     int(operational_k))
            return coph_df["k"].astype(int) == int(operational_k)
        LOG.warning("K-selection: operational K=%d not in tested grid %s; "
                    "falling back to data-driven selection.",
                    operational_k, coph_df["k"].tolist())

    # Filter to Ks meeting the cophenetic floor.
    eligible = coph_df[coph_df["cophenetic_correlation"] >= cophenetic_floor]
    if eligible.empty:
        # Nothing meets the floor — pick max-cophenetic as a graceful fallback.
        k_opt = int(coph_df.sort_values(
            "cophenetic_correlation", ascending=False).iloc[0]["k"])
        LOG.warning("K-selection: no K reaches cophenetic floor %.2f; "
                    "fallback to max-cophenetic K = %d.",
                    cophenetic_floor, k_opt)
        return coph_df["k"].astype(int) == k_opt

    # 2. Dispersion-elbow with marginal-cophenetic criterion.
    eligible = eligible.sort_values("k").reset_index(drop=True)
    eligible["coph_gain"] = eligible["cophenetic_correlation"].diff().fillna(np.inf)
    dispersion_ok = eligible["dispersion_index"] >= dispersion_threshold
    elbow_ok = eligible["coph_gain"] < min_marginal_coph
    candidates = eligible[dispersion_ok & elbow_ok]
    if not candidates.empty:
        k_opt = int(candidates.iloc[0]["k"])
        LOG.info("K-selection: dispersion-elbow → K = %d "
                 "(dispersion ≥ %.2f, marginal coph gain < %.4f).",
                 k_opt, dispersion_threshold, min_marginal_coph)
        return coph_df["k"].astype(int) == k_opt

    # 3. Fallback — among eligible (coph ≥ floor), pick max dispersion.
    k_opt = int(eligible.sort_values(
        "dispersion_index", ascending=False).iloc[0]["k"])
    LOG.info("K-selection: dispersion-elbow not met for any K; "
             "fallback to max-dispersion (coph ≥ %.2f) → K = %d.",
             cophenetic_floor, k_opt)
    return coph_df["k"].astype(int) == k_opt


# ---------------------------------------------------------------------------
# M16.5 — Sprint 3: biological-question detectors
# ---------------------------------------------------------------------------

def detect_cryptic_kinship(
    pair_with_seg: pd.DataFrame,
    assignments_df: pd.DataFrame,
    resolution: float,
    kinship_segment_bp: int,
    max_community_size: int,
) -> pd.DataFrame:
    """Identify candidate close-kin pairs and small extended-family communities.

    Biological grounding:
      An IBD segment of length ≥ 1 cM (~ 1 Mb) is expected from a common
      ancestor within ~10 generations (Browning & Browning 2012; Henn
      et al. 2012).  Segments of length ≥ 10 Mb imply a first- to
      fourth-degree relative — biologically relevant for the "cryptic
      kinship" question #2 in DNABR.

    Criteria:
      * pair-level: max_segment_bp ≥ ``kinship_segment_bp`` (default 10 Mb)
      * community-level: community size ≤ ``max_community_size`` AND
        median max_segment_bp across its internal pairs ≥ threshold

    Returns one long DataFrame with a ``candidate_type`` column so both
    views are preserved in a single TSV.
    """
    rows: list[dict[str, Any]] = []
    if "max_segment_bp" not in pair_with_seg.columns:
        LOG.warning("detect_cryptic_kinship: pair DataFrame lacks "
                    "max_segment_bp; skipping kinship detector.")
        return pd.DataFrame(rows)

    col = f"community_res_{resolution:g}"
    if col not in assignments_df.columns:
        LOG.warning("detect_cryptic_kinship: missing %s in assignments; "
                    "skipping.", col)
        return pd.DataFrame(rows)
    lookup = dict(zip(assignments_df["sample_id"].astype(str),
                       assignments_df[col].astype(int)))

    sa = pair_with_seg["sample_a"].astype(str).to_numpy()
    sb = pair_with_seg["sample_b"].astype(str).to_numpy()
    ms = pair_with_seg["max_segment_bp"].to_numpy(dtype=np.int64)
    tb = pair_with_seg["total_shared_bp"].to_numpy(dtype=np.int64)
    ns = pair_with_seg["n_segments"].to_numpy(dtype=np.int64)

    # 1) Pair-level candidates.
    hot = ms >= int(kinship_segment_bp)
    for i in np.flatnonzero(hot):
        a = sa[i]; b = sb[i]
        rows.append({
            "candidate_type": "pair",
            "community": -99,
            "size": 2,
            "sample_a": a,
            "sample_b": b,
            "max_segment_bp": int(ms[i]),
            "total_shared_bp": int(tb[i]),
            "n_segments": int(ns[i]),
            "community_a": int(lookup.get(a, NOISE_LABEL)),
            "community_b": int(lookup.get(b, NOISE_LABEL)),
        })

    # 2) Community-level candidates: small dense communities whose internal
    #    pairs show consistently long segments (extended families).
    #    For each community c, restrict pairs to both-in-c and compute the
    #    median max_segment_bp; also count members.
    comm_a = np.fromiter(
        (lookup.get(s, NOISE_LABEL) for s in sa), dtype=np.int64, count=len(sa)
    )
    comm_b = np.fromiter(
        (lookup.get(s, NOISE_LABEL) for s in sb), dtype=np.int64, count=len(sb)
    )
    intra_mask = (comm_a == comm_b) & (comm_a != NOISE_LABEL)
    if intra_mask.any():
        tmp = pd.DataFrame({
            "community": comm_a[intra_mask],
            "max_segment_bp": ms[intra_mask],
        })
        grouped = tmp.groupby("community")["max_segment_bp"].agg(
            median="median", count="size"
        ).reset_index()
        member_sizes = (
            assignments_df[col].astype(int).value_counts().to_dict()
        )
        for _, row in grouped.iterrows():
            c = int(row["community"])
            size = int(member_sizes.get(c, 0))
            if size == 0 or size > int(max_community_size):
                continue
            if row["median"] < int(kinship_segment_bp):
                continue
            members = assignments_df.loc[
                assignments_df[col].astype(int) == c, "sample_id"
            ].tolist()
            rows.append({
                "candidate_type": "community",
                "community": c,
                "size": size,
                "sample_a": ",".join(str(m) for m in members[:20]),
                "sample_b": "",
                "max_segment_bp": float(row["median"]),
                "total_shared_bp": int(0),
                "n_segments": int(row["count"]),
                "community_a": c,
                "community_b": c,
            })
    return pd.DataFrame(rows)


def detect_founder_effects(
    pair_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    silhouette_df: pd.DataFrame,
    resolution: float,
    min_ratio: float,
    min_silhouette: float,
    min_size: int,
    min_size_for_report: int = 0,
) -> pd.DataFrame:
    """Identify communities behaving like isolated subpopulations / founders.

    Biological grounding:
      An endogamous or bottlenecked population shows markedly higher
      sharing among its members than across the rest of the cohort
      (intra/inter ratio).  In *discrete* subpopulations its members
      also form a compact silhouette cluster — but in *admixed*
      cohorts (Brazilian DNABR, Hispanic populations) the silhouette
      criterion fails because admixture gradients produce silhouettes
      <0.05 even for genuine bottlenecks (Browning & Browning 2015).
      The intra/inter sharing ratio is the more biologically robust
      signal across both regimes.

    Criteria (all must hold):
      * community size >= ``min_size``
      * median intra/inter total_shared_bp ratio >= ``min_ratio``
      * median silhouette >= ``min_silhouette``  (skipped when
        ``min_silhouette`` <= 0; recommended for admixed cohorts)
    """
    rows: list[dict[str, Any]] = []
    col = f"community_res_{resolution:g}"
    if col not in assignments_df.columns:
        LOG.warning("detect_founder_effects: missing %s; skipping.", col)
        return pd.DataFrame(rows)

    lookup = dict(zip(assignments_df["sample_id"].astype(str),
                       assignments_df[col].astype(int)))
    sa = pair_df["sample_a"].astype(str).to_numpy()
    sb = pair_df["sample_b"].astype(str).to_numpy()
    bp = pair_df["total_shared_bp"].to_numpy(dtype=np.float64)
    comm_a = np.fromiter(
        (lookup.get(s, NOISE_LABEL) for s in sa), dtype=np.int64, count=len(sa)
    )
    comm_b = np.fromiter(
        (lookup.get(s, NOISE_LABEL) for s in sb), dtype=np.int64, count=len(sb)
    )

    sil_by_comm: dict[int, float] = {}
    if silhouette_df is not None and not silhouette_df.empty:
        sil_by_comm = dict(zip(
            silhouette_df["community"].astype(int),
            silhouette_df["median_silhouette"].astype(float),
        ))

    sizes = assignments_df[col].astype(int).value_counts().to_dict()
    for c, size in sizes.items():
        if int(c) == NOISE_LABEL or size < int(min_size):
            continue
        intra = bp[(comm_a == c) & (comm_b == c)]
        inter = bp[((comm_a == c) ^ (comm_b == c))
                    & (comm_a != NOISE_LABEL) & (comm_b != NOISE_LABEL)]
        if intra.size == 0 or inter.size == 0:
            continue
        med_intra = float(np.median(intra))
        med_inter = float(np.median(inter))
        ratio = med_intra / med_inter if med_inter > 0 else float("inf")
        sil = sil_by_comm.get(int(c), float("nan"))
        # When min_silhouette <= 0, the silhouette filter is disabled
        # (recommended default for admixed cohorts where silhouettes are
        # inherently <0.05).  Otherwise both criteria must hold.
        sil_ok = (
            float(min_silhouette) <= 0.0
            or (np.isfinite(sil) and sil >= float(min_silhouette))
        )
        # Size filter for the *report* (separate from the compute filter
        # ``min_size`` above).  Communities passing min_size are still
        # computed and emitted in the table, but only those at or above
        # ``min_size_for_report`` get ``is_founder_candidate=True`` so
        # downstream pipelines (paper figures, biological interpretation)
        # don't inherit family-trio artefacts.
        size_ok = int(size) >= int(min_size_for_report)
        is_candidate = (ratio >= float(min_ratio)) and sil_ok and size_ok
        rows.append({
            "community": int(c),
            "size": int(size),
            "median_intra_bp": med_intra,
            "median_inter_bp": med_inter,
            "intra_inter_ratio": ratio,
            "median_silhouette": float(sil) if np.isfinite(sil) else float("nan"),
            "is_founder_candidate": bool(is_candidate),
        })
    return pd.DataFrame(rows).sort_values(
        "intra_inter_ratio", ascending=False
    ).reset_index(drop=True) if rows else pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# M16.5 — Sprint 5: optional metadata layer
# ---------------------------------------------------------------------------

_METADATA_ID_CANDIDATES = (
    "sample_id", "IID", "Sample_ID", "SampleID", "id", "ID"
)

# Schema (and order) of community_auto_annotations.tsv.  ``manual_label`` is
# user-editable; everything else is regenerated on each run.  See
# ``resolve_community_annotations`` for the merge semantics.
_AUTO_ANN_COLS = (
    "community_id", "size",
    "primary_modal", "primary_purity",
    "secondary_modal", "secondary_purity",
    "auto_label", "manual_label",
)


def _excel_letters(idx: int) -> str:
    """0→'A', 1→'B', ..., 25→'Z', 26→'AA', 27→'AB' (Excel column scheme).

    Used to suffix communities that collapse to the same composed label
    (e.g. three communities all modal 'BrazilA' become 'BrazilA·subA',
    'BrazilA·subB', 'BrazilA·subC' ordered by descending community size).
    """
    if idx < 0:
        raise ValueError(f"_excel_letters: idx must be >=0, got {idx}")
    out: list[str] = []
    n = idx
    while True:
        out.append(string.ascii_uppercase[n % 26])
        n = n // 26 - 1
        if n < 0:
            break
    return "".join(reversed(out))


def resolve_community_annotations(
    args,
    assignments_df: pd.DataFrame,
    out_dir: Path,
) -> dict[int, str] | None:
    """Resolve UMAP centroid labels with three-tier precedence.

    Tiers (highest first):
      1. ``--plot-community-annotations-file <ruta>`` — archivo explícito
         file with ``community_id`` + (``manual_label`` | ``annotation``).
         Use when the user has hand-curated labels external to the run
         directory (e.g. a project-wide canonical mapping).
      2. ``--plot-auto-annotate-by <columna>`` y la opción
         ``--plot-auto-annotate-secondary <columna>`` — etiquetas automáticas
         from the modal value of one or two metadata columns per
         community.  Collisions ('BrazilA' appearing in 3 distinct
         communities) are disambiguated deterministically as
         '<base>·sub<A|B|C…>' by descending community size.  When a
         secondary column is provided, the composed base label is
         '<primary>·<secondary>' before disambiguation, so three
         communities that all map to ('BrazilA', 'AM_1') become
         'BrazilA·AM_1·subA/B/C', while a single 'BrazilA·AM_2' keeps no
         suffix.

         Output: ``<out_dir>/community_auto_annotations.tsv`` with one
         row per labelled community and columns:

             community_id, size,
             primary_modal, primary_purity,
             secondary_modal, secondary_purity,
             auto_label, manual_label

         The ``manual_label`` column is preserved across re-runs (read
         in before the new auto labels are written), so the user can
         edit that file in place — refining 'BrazilA·AM_1·subA' to a
         project-specific identifier such as 'BrazilA·quilombola_pal' —
         and a subsequent ``-resume`` (or REPLOT fan-out) run will pick
         the manual_label up automatically.  No separate file path
         required for that flow.
      3. Fallback — ``None`` is returned and the network plot uses
         'C{id}' centroid labels.

    Returns: mapping ``{community_id: final_label}`` or ``None``.
    """
    # ----- Tier 2: auto-derived labels --------------------------------
    auto_col = (getattr(args, "plot_auto_annotate_by", "") or "").strip()
    sec_col = (
        getattr(args, "plot_auto_annotate_secondary", "") or ""
    ).strip()
    md_path = getattr(args, "metadata_file", None)
    md_path = Path(str(md_path)) if md_path else None
    min_lbl_size = int(getattr(args, "plot_cluster_label_min_size", 10))

    auto_label_map: dict[int, str] = {}  # cid → composed auto label
    rows: list[dict] = []  # → community_auto_annotations.tsv

    if (auto_col and md_path is not None and md_path.exists()
            and md_path.stat().st_size > 0):
        try:
            md_df = pd.read_csv(md_path, sep="\t",
                                dtype={"sample_id": str})
            id_col = next(
                (c for c in _METADATA_ID_CANDIDATES if c in md_df.columns),
                None,
            )
            asg_col = f"community_res_{args.validation_resolution:g}"
            errors: list[str] = []
            if id_col is None:
                errors.append(
                    f"metadata {md_path} lacks an ID column from "
                    f"{list(_METADATA_ID_CANDIDATES)}")
            if auto_col not in md_df.columns:
                errors.append(
                    f"primary column '{auto_col}' not in metadata")
            if sec_col and sec_col not in md_df.columns:
                errors.append(
                    f"secondary column '{sec_col}' not in metadata")
            if asg_col not in assignments_df.columns:
                errors.append(
                    f"assignments missing {asg_col}")
            if errors:
                LOG.warning("Auto-annotate disabled: %s",
                            "; ".join(errors))
            else:
                cols_to_use = [id_col, auto_col]
                if sec_col:
                    cols_to_use.append(sec_col)
                md_subset = md_df[cols_to_use].rename(
                    columns={id_col: "sample_id"})
                merged = assignments_df[["sample_id", asg_col]].merge(
                    md_subset, on="sample_id", how="left")

                # Per-community modals + purity for primary (and secondary).
                # Step 1: compute the composed *base* label without
                #         disambiguation suffix.
                per_cid: list[dict] = []  # {cid, size, p_mod, p_pur, ...}
                base_to_cids: dict[str, list[int]] = defaultdict(list)
                for cid, grp in merged.groupby(asg_col):
                    cid_int = int(cid)
                    if cid_int == NOISE_LABEL:
                        continue
                    full_size = int(
                        (assignments_df[asg_col] == cid_int).sum())
                    if full_size < min_lbl_size:
                        continue
                    p_grp = grp.dropna(subset=[auto_col])
                    if p_grp.empty:
                        continue
                    p_modal_series = p_grp[auto_col].mode()
                    if p_modal_series.empty:
                        continue
                    p_modal = str(p_modal_series.iloc[0])
                    p_count = int((p_grp[auto_col] == p_modal).sum())
                    p_purity = float(p_count) / float(len(p_grp))

                    s_modal: str | None = None
                    s_purity: float | None = None
                    if sec_col:
                        s_grp = grp.dropna(subset=[sec_col])
                        if not s_grp.empty:
                            s_modal_series = s_grp[sec_col].mode()
                            if not s_modal_series.empty:
                                s_modal = str(s_modal_series.iloc[0])
                                s_count = int(
                                    (s_grp[sec_col] == s_modal).sum())
                                s_purity = (
                                    float(s_count) / float(len(s_grp))
                                )

                    base = (f"{p_modal}·{s_modal}"
                            if s_modal else p_modal)
                    per_cid.append({
                        "cid": cid_int,
                        "size": full_size,
                        "p_mod": p_modal,
                        "p_pur": p_purity,
                        "s_mod": s_modal,
                        "s_pur": s_purity,
                        "base": base,
                    })
                    base_to_cids[base].append(cid_int)

                # Step 2: disambiguate collisions deterministically by
                # descending community size.  Largest community keeps
                # ·subA (i.e. 'BrazilA·subA'), next ·subB, etc.  When a
                # base appears only once it stays as-is.
                cid_to_auto: dict[int, str] = {}
                size_lookup = {r["cid"]: r["size"] for r in per_cid}
                for base, cids in base_to_cids.items():
                    if len(cids) == 1:
                        cid_to_auto[cids[0]] = base
                        continue
                    sorted_cids = sorted(
                        cids, key=lambda c: (-size_lookup[c], c))
                    for idx, c in enumerate(sorted_cids):
                        cid_to_auto[c] = f"{base}·sub{_excel_letters(idx)}"

                # Step 3: compose the persisted rows in canonical order.
                for r in sorted(per_cid, key=lambda r: r["cid"]):
                    rows.append({
                        "community_id": r["cid"],
                        "size": r["size"],
                        "primary_modal": r["p_mod"],
                        "primary_purity": round(r["p_pur"], 4),
                        "secondary_modal": (r["s_mod"]
                                            if r["s_mod"] is not None
                                            else ""),
                        "secondary_purity": (round(r["s_pur"], 4)
                                             if r["s_pur"] is not None
                                             else ""),
                        "auto_label": cid_to_auto[r["cid"]],
                        "manual_label": "",
                    })
                auto_label_map = dict(cid_to_auto)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Auto-annotation failed (column='%s'): %s",
                        auto_col, exc)

    # ----- Persist & merge in any pre-existing manual_label edits -----
    ann_path = out_dir / "community_auto_annotations.tsv"
    manual_from_tsv: dict[int, str] = {}
    if rows:
        if ann_path.exists() and ann_path.stat().st_size > 0:
            try:
                prev = pd.read_csv(ann_path, sep="\t")
                if ("community_id" in prev.columns
                        and "manual_label" in prev.columns):
                    for _, r in prev.iterrows():
                        try:
                            cid = int(r["community_id"])
                        except (TypeError, ValueError):
                            continue
                        v = r["manual_label"]
                        if pd.notna(v) and str(v).strip():
                            manual_from_tsv[cid] = str(v).strip()
            except Exception as exc:  # noqa: BLE001
                LOG.warning("Could not read prior %s: %s; "
                            "manual_label edits will be discarded.",
                            ann_path, exc)
        # Merge: keep edits in the new TSV.
        for r in rows:
            v = manual_from_tsv.get(r["community_id"])
            if v:
                r["manual_label"] = v
        ann_df = pd.DataFrame(rows, columns=list(_AUTO_ANN_COLS))
        out_dir.mkdir(parents=True, exist_ok=True)
        ann_df.to_csv(ann_path, sep="\t", index=False)
        LOG.info(
            "Auto-annotated %d communities (primary='%s'%s); persisted %s",
            len(rows), auto_col,
            f", secondary='{sec_col}'" if sec_col else "",
            ann_path,
        )

    # ----- Tier 1: explicit override file (highest priority) ----------
    final: dict[int, str] = {}
    for r in rows:
        final[int(r["community_id"])] = (
            r["manual_label"] if r["manual_label"] else r["auto_label"]
        )

    annot_path = getattr(args, "plot_community_annotations_file", None)
    if annot_path and str(annot_path).strip():
        annot_path_p = Path(str(annot_path))
        if annot_path_p.exists() and annot_path_p.stat().st_size > 0:
            try:
                ext = pd.read_csv(annot_path_p, sep="\t")
                id_c = next(
                    (c for c in ("community_id", "community", "id")
                     if c in ext.columns), None)
                # Prefer the new manual_label column, then legacy aliases.
                lbl_c = next(
                    (c for c in ("manual_label", "annotation",
                                 "label", "name")
                     if c in ext.columns), None)
                if id_c and lbl_c:
                    n_over = 0
                    for _, r in ext.iterrows():
                        if pd.isna(r[id_c]) or pd.isna(r[lbl_c]):
                            continue
                        v = str(r[lbl_c]).strip()
                        if not v:
                            continue
                        try:
                            cid = int(r[id_c])
                        except (TypeError, ValueError):
                            continue
                        final[cid] = v
                        n_over += 1
                    LOG.info(
                        "Manual annotations file %s applied: %d entries "
                        "override auto labels.", annot_path_p, n_over)
                else:
                    LOG.warning(
                        "Annotations file %s lacks recognised columns "
                        "(community_id + manual_label/annotation); "
                        "ignoring.", annot_path_p)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("Failed to load annotations from %s: %s",
                            annot_path_p, exc)

    return final or None


def load_metadata_safe(path: Path | None,
                        color_by_col: str | None,
                        samples: Sequence[str],
                        ) -> tuple[np.ndarray | None, str | None,
                                   list[str]]:
    """Load an optional metadata TSV and align to the cohort sample list.

    Returns (values_aligned, column_name, warnings).  The returned
    ``values_aligned`` is a string array (len == len(samples)) where
    unknown samples become "NA".  If anything fails — path missing,
    empty placeholder (conf/empty.txt), bad column, <2 levels, or <3
    members per level — we return (None, None, warnings) and the
    downstream plots/tests degrade silently.
    """
    warnings_list: list[str] = []
    if path is None:
        return None, None, warnings_list
    p = Path(path)
    if p.name == "empty.txt" or (p.exists() and p.stat().st_size == 0):
        return None, None, warnings_list
    if not p.exists():
        warnings_list.append(f"metadata file not found: {p}")
        return None, None, warnings_list
    try:
        df = pd.read_csv(p, sep="\t", dtype=str)
    except Exception as exc:
        warnings_list.append(f"failed to read metadata {p}: {exc}")
        return None, None, warnings_list
    # Locate sample-id column.
    id_col = next(
        (c for c in _METADATA_ID_CANDIDATES if c in df.columns), None
    )
    if id_col is None:
        warnings_list.append(
            f"metadata {p} has no recognised sample-id column "
            f"(expected one of {_METADATA_ID_CANDIDATES})"
        )
        return None, None, warnings_list
    if color_by_col is None:
        warnings_list.append(
            "metadata provided but --plot-color-by is empty; "
            "skipping annotation"
        )
        return None, None, warnings_list
    if color_by_col not in df.columns:
        warnings_list.append(
            f"metadata {p} lacks column '{color_by_col}'"
        )
        return None, None, warnings_list
    lookup = dict(zip(df[id_col].astype(str), df[color_by_col].astype(str)))
    aligned = np.array([lookup.get(str(s), "NA") for s in samples],
                        dtype=object)
    # Validate enough levels / size.
    levels, counts = np.unique(aligned, return_counts=True)
    real_levels = [(l, c) for l, c in zip(levels, counts) if l != "NA"]
    if len(real_levels) < 2:
        warnings_list.append(
            f"metadata column '{color_by_col}' has fewer than 2 levels; "
            "skipping annotation"
        )
        return None, None, warnings_list
    small = [l for l, c in real_levels if c < 3]
    if small:
        warnings_list.append(
            f"metadata levels with <3 samples will be shown but not "
            f"Fisher-tested: {small}"
        )
    LOG.info("Metadata loaded: %d samples, column '%s' with levels %s",
             int((aligned != "NA").sum()), color_by_col,
             sorted({l for l, _ in real_levels}))
    return aligned, color_by_col, warnings_list


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg (1995) step-up FDR correction (no external dep)."""
    p = np.asarray(pvals, dtype=np.float64)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    # Enforce monotonicity from the top.
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty_like(p)
    q[order] = np.clip(ranked, 0, 1)
    return q


def fisher_enrichment_community_vs_metadata(
    assignments_df: pd.DataFrame,
    resolution: float,
    metadata_values: np.ndarray,
    metadata_name: str,
) -> pd.DataFrame:
    """Fisher's exact test per (community × metadata level) 2x2 table.

    For each non-noise community c and each metadata level l, build
    the contingency table [in_c_and_l, in_c_not_l; not_c_and_l,
    not_c_not_l] and compute Fisher's exact (Agresti 1992) p-value
    plus odds ratio.  P-values are BH-corrected (FDR) across ALL
    (c, l) tests jointly — cleaner than per-community correction when
    the number of communities is ~10-30 and the number of levels is
    small.

    This is the direct statistical test of biological question #1:
    "does community C correspond to Brazilian region R?"  Cells with
    q < 0.01 and log2 OR > 1 mark a real geographic signal.
    """
    col = f"community_res_{resolution:g}"
    if col not in assignments_df.columns:
        return pd.DataFrame()
    membership = assignments_df[col].astype(int).to_numpy()
    # Align metadata (assumed already aligned to same order).
    if len(metadata_values) != len(membership):
        LOG.warning("Fisher enrichment: metadata length mismatch "
                    "(%d vs %d); skipping.",
                    len(metadata_values), len(membership))
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    levels = [l for l in np.unique(metadata_values) if l != "NA"]
    comms = [c for c in np.unique(membership) if c != NOISE_LABEL]
    n_total = int((membership != NOISE_LABEL).sum())
    if n_total == 0 or not levels or not comms:
        return pd.DataFrame()
    for c in comms:
        in_c = (membership == c) & (membership != NOISE_LABEL)
        not_c = (~in_c) & (membership != NOISE_LABEL)
        for l in levels:
            in_l = (metadata_values == l)
            a = int((in_c & in_l).sum())
            b = int((in_c & ~in_l).sum())
            cc = int((not_c & in_l).sum())
            d = int((not_c & ~in_l).sum())
            if (a + b) == 0 or (a + cc) == 0:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    odds, pval = fisher_exact(
                        [[a, b], [cc, d]], alternative="greater"
                    )
            except Exception:  # pragma: no cover
                odds, pval = float("nan"), float("nan")
            rows.append({
                "community": int(c),
                "metadata_column": str(metadata_name),
                "level": str(l),
                "n_in_community_and_level": a,
                "n_in_community_not_level": b,
                "n_out_of_community_in_level": cc,
                "n_out_of_community_not_level": d,
                "odds_ratio": float(odds),
                "log2_odds_ratio": float(np.log2(odds)) if odds > 0
                                    else float("-inf"),
                "p_value": float(pval),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["q_value"] = _bh_fdr(df["p_value"].to_numpy())
    df["neg_log10_q"] = -np.log10(np.clip(df["q_value"], 1e-300, 1.0))
    df = df.sort_values(["community", "q_value"]).reset_index(drop=True)
    return df


def compute_community_labels(
    assignments_df: pd.DataFrame,
    resolution: float,
    metadata_values: np.ndarray | None,
    metadata_name: str | None,
) -> pd.DataFrame:
    """Derive per-community descriptive labels.

    Each community gets (size, modal metadata level, its fraction,
    median assignment confidence).  Used for plot annotations such as
    "C3 → NE (73%)" and for the HTML report summary table.
    """
    col = f"community_res_{resolution:g}"
    if col not in assignments_df.columns:
        return pd.DataFrame()
    membership = assignments_df[col].astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    for c in np.unique(membership):
        idx = np.flatnonzero(membership == c)
        size = int(idx.size)
        median_conf = float("nan")
        if "assignment_confidence" in assignments_df.columns:
            conf_vals = assignments_df.loc[idx, "assignment_confidence"]
            if len(conf_vals):
                median_conf = float(np.nanmedian(conf_vals.to_numpy(
                    dtype=np.float64
                )))
        label = "Noise" if int(c) == NOISE_LABEL else f"C{int(c)}"
        modal_level = None
        modal_fraction = float("nan")
        if metadata_values is not None and metadata_name is not None:
            vals = np.asarray(metadata_values)[idx]
            real = vals[vals != "NA"]
            if real.size:
                levels, counts = np.unique(real, return_counts=True)
                top = int(np.argmax(counts))
                modal_level = str(levels[top])
                modal_fraction = float(counts[top] / real.size)
                if int(c) != NOISE_LABEL:
                    label = (f"C{int(c)}·{modal_level} "
                             f"({int(modal_fraction * 100)}%)")
        rows.append({
            "community": int(c),
            "label": label,
            "size": size,
            "median_confidence": median_conf,
            "modal_metadata_level": modal_level or "",
            "modal_metadata_fraction": modal_fraction,
        })
    return pd.DataFrame(rows).sort_values(
        ["size"], ascending=False
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_intra_vs_inter(pair_df: pd.DataFrame,
                            assignments_df: pd.DataFrame,
                            resolutions: Sequence[float],
                            ) -> pd.DataFrame:
    """Statistical comparison of ``total_shared_bp`` intra vs inter per
    resolution.

    Reports four complementary signals per resolution:

    * **Median ratio** ``median_intra_bp / median_inter_bp`` — primary
      effect-size metric for the thesis (corrida_C γ=1 = 2.59).  Intuitive
      and unit-meaningful (kb of co-shared sequence).
    * **Mann-Whitney U** + ``mann_whitney_p`` — significance under the
      alternative ``intra > inter``.  At N=2619 with ~57k inter pairs the
      p-value is ~0 for any non-degenerate effect, so it confirms
      directionality but is not informative for ranking resolutions.
    * **Cliff's δ** (Romano et al. 2006, ``cliff_delta``) — non-parametric
      effect size on [-1, 1], robust to the heavy-tailed sharing
      distribution.  Romano benchmarks: ``|δ| < 0.147`` small,
      ``0.147–0.33`` medium, ``0.33–0.474`` large, ``≥ 0.474`` very large.
      Computed from U via Vargha-Delaney's identity (no extra O(N²) pass).
    * **AUC / probability of superiority** (Vargha-Delaney 2000,
      ``auc_intra_over_inter``) — P(``X_intra > X_inter``) on [0, 1].
      Reads as "what fraction of intra-pairs share more than a random
      inter-pair".  0.5 = no separation; 1.0 = full separation.

    Identity used to derive δ and AUC from U
    (``mannwhitneyu(intra, inter, alternative='greater')``):

        AUC = U / (n_intra * n_inter)      # Vargha-Delaney A
        δ   = 2 * AUC - 1                  # Romano 2006

    The pair DataFrame is expected to carry 'sample_a', 'sample_b',
    'total_shared_bp'.  Assignments df must carry 'sample_id' plus
    'community_res_{r}' columns.
    """
    comm_by_sample_per_res: dict[float, dict[str, int]] = {}
    for res in resolutions:
        col = f"community_res_{res:g}"
        if col not in assignments_df.columns:
            continue
        comm_by_sample_per_res[res] = dict(
            zip(assignments_df["sample_id"], assignments_df[col].astype(int))
        )

    out_rows: list[dict[str, Any]] = []
    bp = pair_df["total_shared_bp"].to_numpy(dtype=np.float64)
    sa = pair_df["sample_a"].astype(str).to_numpy()
    sb = pair_df["sample_b"].astype(str).to_numpy()

    for res, mp in comm_by_sample_per_res.items():
        get = mp.get
        ca = np.fromiter((get(s, -99) for s in sa), dtype=np.int64,
                         count=len(sa))
        cb = np.fromiter((get(s, -99) for s in sb), dtype=np.int64,
                         count=len(sb))
        # Valid pairs: both samples mapped and neither is noise.
        valid = (ca != -99) & (cb != -99) & (ca != NOISE_LABEL) & (cb != NOISE_LABEL)
        if not valid.any():
            out_rows.append({
                "resolution": float(res), "n_intra": 0, "n_inter": 0,
                "median_intra_bp": float("nan"),
                "median_inter_bp": float("nan"),
                "ratio_median_intra_inter": float("nan"),
                "mann_whitney_u": float("nan"),
                "mann_whitney_p": float("nan"),
                "cliff_delta": float("nan"),
                "auc_intra_over_inter": float("nan"),
            })
            continue
        intra = valid & (ca == cb)
        inter = valid & (ca != cb)
        intra_bp = bp[intra]
        inter_bp = bp[inter]
        if intra_bp.size == 0 or inter_bp.size == 0:
            u, p = float("nan"), float("nan")
            auc, delta = float("nan"), float("nan")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res_stat = mannwhitneyu(intra_bp, inter_bp,
                                         alternative="greater")
            u, p = float(res_stat.statistic), float(res_stat.pvalue)
            # Vargha-Delaney AUC and Cliff's δ derived from U (see docstring).
            # Guard against pathological cases (n_intra * n_inter == 0 is
            # caught above; this protects against floating-point degeneracy).
            denom = float(intra_bp.size) * float(inter_bp.size)
            auc = u / denom if denom > 0 else float("nan")
            delta = (2.0 * auc - 1.0) if np.isfinite(auc) else float("nan")
        med_intra = float(np.median(intra_bp)) if intra_bp.size else float("nan")
        med_inter = float(np.median(inter_bp)) if inter_bp.size else float("nan")
        ratio = (med_intra / med_inter) if (med_inter and np.isfinite(med_inter)
                                            and med_inter > 0) else float("inf")
        out_rows.append({
            "resolution": float(res),
            "n_intra": int(intra_bp.size),
            "n_inter": int(inter_bp.size),
            "median_intra_bp": med_intra,
            "median_inter_bp": med_inter,
            "ratio_median_intra_inter": float(ratio),
            "mann_whitney_u": u,
            "mann_whitney_p": p,
            "cliff_delta": float(delta),
            "auc_intra_over_inter": float(auc),
        })
    return pd.DataFrame(out_rows)


def save_validation(out_dir: Path, df: pd.DataFrame) -> None:
    """Guarda las comparaciones intra e intercomunidad."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / NAME_VALIDATION, sep="\t", index=False,
              float_format="%.6g")
    LOG.info("Saved validation to %s", out_dir / NAME_VALIDATION)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _community_color(label: int) -> str:
    if label == NOISE_LABEL:
        return _NOISE_COLOR
    return _COMMUNITY_PALETTE[int(label) % len(_COMMUNITY_PALETTE)]


def _community_cmap(max_label: int) -> tuple[ListedColormap, dict[int, int]]:
    """Build a ListedColormap indexed by ``label_to_code``.  Code 0 is
    reserved for noise; real communities start at code 1.
    """
    labels = [-1] + list(range(max_label + 1))
    colors = [_NOISE_COLOR] + [_community_color(i) for i in range(max_label + 1)]
    cmap = ListedColormap(colors, name="m16_community")
    label_to_code = {lbl: i for i, lbl in enumerate(labels)}
    return cmap, label_to_code


def _save_fig(fig: plt.Figure, path: Path, dpi: int,
              export_pdf: bool, export_svg: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    if export_pdf:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight",
                    facecolor="white")
    if export_svg:
        fig.savefig(path.with_suffix(".svg"), bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)


# --- Network plot ----------------------------------------------------------

def plot_network(g: ig.Graph, membership: np.ndarray, out_path: Path,
                  *, max_nodes: int, dpi: int, width_in: float,
                  height_in: float, export_pdf: bool,
                  layout_seed: int) -> None:
    """Spring-layout network plot using igraph's C-backed FR layout.

    For N > max_nodes we retain the top-degree subset (stratified per
    community so every community stays visible).
    """
    n = g.vcount()
    if n == 0:
        LOG.warning("Empty graph: skipping network plot.")
        return

    wdeg = np.array(g.strength(weights="weight"), dtype=np.float64)
    keep_mask = np.ones(n, dtype=bool)
    if n > max_nodes:
        # Stratified top-degree subset: sort within community and take the
        # top k_c nodes where k_c is proportional to community size.
        order_in = np.argsort(-wdeg)
        comm_order: dict[int, list[int]] = defaultdict(list)
        for node in order_in:
            comm_order[int(membership[node])].append(int(node))
        keep_mask = np.zeros(n, dtype=bool)
        sizes = {c: len(v) for c, v in comm_order.items()}
        total = float(sum(sizes.values()))
        for c, nodes in comm_order.items():
            take = max(1, int(round(max_nodes * sizes[c] / total)))
            for node in nodes[:take]:
                keep_mask[node] = True
        LOG.info("Network plot: down-sampled %d/%d nodes (stratified by "
                 "community).", int(keep_mask.sum()), n)

    sub = g.subgraph(np.flatnonzero(keep_mask).tolist())
    sub_members = membership[keep_mask]
    sub_wdeg = wdeg[keep_mask]

    LOG.info("Network plot: Fruchterman-Reingold layout on %d nodes / %d edges",
             sub.vcount(), sub.ecount())
    # igraph's FR layout takes a matrix in ``seed``, not an RNG seed.
    # Control determinism via the igraph RNG; wrap in a ``Random`` instance
    # so we don't mutate the process-wide state.
    import random as _random
    ig.set_random_number_generator(_random.Random(layout_seed))
    layout = sub.layout_fruchterman_reingold(
        weights="weight", niter=500,
    ) if sub.ecount() > 0 else sub.layout_random()
    coords = np.asarray(layout.coords)

    fig, ax = plt.subplots(figsize=(width_in, height_in))
    # Edges: LineCollection is O(E) matplotlib-side and avoids per-edge axes
    # calls (which OOM at E > 1e5).
    if sub.ecount() > 0:
        edge_list = sub.get_edgelist()
        segs = np.array([[coords[i], coords[j]] for i, j in edge_list])
        w = np.asarray(sub.es["weight"], dtype=np.float64)
        if w.max() > 0:
            alpha = 0.05 + 0.35 * (w / w.max())
        else:
            alpha = np.full_like(w, 0.15)
        lc = LineCollection(segs, colors=[(0.4, 0.4, 0.4, a) for a in alpha],
                            linewidths=0.3, zorder=1)
        ax.add_collection(lc)
    # Nodes: scatter; size scales with sqrt(weighted degree) so that
    # visually extreme hubs don't crowd the rest.
    size_scale = 10 + 120 * np.sqrt(
        sub_wdeg / (sub_wdeg.max() if sub_wdeg.max() > 0 else 1.0)
    )
    colors = [_community_color(int(m)) for m in sub_members]
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=size_scale,
                edgecolors="white", linewidths=0.3, zorder=2)
    ax.set_axis_off()
    ax.set_title("Rare-variant co-sharing network — nodes coloured by Leiden community",
                  loc="left", fontsize=12, fontweight="bold")
    # Legend: one swatch per community (up to 20).
    uniq_comms = np.unique(sub_members)
    uniq_comms = np.concatenate([
        uniq_comms[uniq_comms == NOISE_LABEL],
        uniq_comms[uniq_comms != NOISE_LABEL][:20],
    ])
    handles = [
        Patch(facecolor=_community_color(int(c)), edgecolor="white",
              label=f"Noise ({int((membership == NOISE_LABEL).sum())})"
              if c == NOISE_LABEL
              else f"C{int(c)} ({int((membership == c).sum())})")
        for c in uniq_comms
    ]
    ax.legend(handles=handles, loc="center left",
               bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False,
               title="Community (size)", title_fontsize=9)
    fig.tight_layout()
    _save_fig(fig, out_path, dpi=dpi, export_pdf=export_pdf)


# --- Modularity curve ------------------------------------------------------

def plot_modularity_vs_resolution(mod_df: pd.DataFrame, out_path: Path,
                                    *, dpi: int, width_in: float,
                                    height_in: float, export_pdf: bool) -> None:
    """Grafica la modularidad de Leiden a lo largo de las resoluciones."""
    if mod_df.empty:
        return
    fig, ax = plt.subplots(figsize=(width_in, height_in * 0.6))
    by_res = mod_df.groupby("resolution")["modularity"]
    res = sorted(by_res.groups.keys())
    med = [by_res.get_group(r).median() for r in res]
    q25 = [by_res.get_group(r).quantile(0.25) for r in res]
    q75 = [by_res.get_group(r).quantile(0.75) for r in res]
    ax.fill_between(res, q25, q75, alpha=0.25, color="#4363d8",
                     label="IQR across seeds")
    ax.plot(res, med, "-o", color="#4363d8", label="median modularity")
    ax.set_xlabel("Leiden resolution parameter γ")
    ax.set_ylabel("Modularity")
    ax.set_title("Modularity vs resolution (Leiden multi-seed)",
                  loc="left", fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save_fig(fig, out_path, dpi=dpi, export_pdf=export_pdf)


# --- Sharing heatmap ordered by community ---------------------------------

def plot_sharing_heatmap(S: sp.csr_matrix, membership: np.ndarray,
                          samples: Sequence[str], out_path: Path,
                          *, max_nodes: int, dpi: int,
                          width_in: float, height_in: float,
                          export_pdf: bool) -> None:
    """Grafica la matriz de sharing ordenada por comunidad."""
    n = S.shape[0]
    if n == 0:
        return
    order = np.lexsort((np.arange(n), membership))
    if n > max_nodes:
        # Take the first max_nodes nodes of the ordering.
        LOG.info("Heatmap: down-sampling %d -> %d nodes", n, max_nodes)
        order = order[:max_nodes]
    ordered = S[order, :][:, order].toarray()
    # log1p scaling keeps the dynamic range readable when weights span
    # several orders of magnitude.
    M = np.log1p(ordered)

    fig, ax = plt.subplots(figsize=(width_in, width_in))
    im = ax.imshow(M, cmap="magma", interpolation="nearest", aspect="equal",
                    origin="upper")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("log1p(shared weight)")

    # Draw community boundaries.
    ordered_members = membership[order]
    boundaries = np.where(np.diff(ordered_members) != 0)[0] + 1
    for b in boundaries:
        ax.axhline(b - 0.5, color="white", linewidth=0.4, alpha=0.7)
        ax.axvline(b - 0.5, color="white", linewidth=0.4, alpha=0.7)

    # Community label bar (left): a thin strip of community colours.
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Pairwise sharing matrix ordered by community "
                 f"({ordered.shape[0]} samples)",
                 loc="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    _save_fig(fig, out_path, dpi=dpi, export_pdf=export_pdf)


# --- NMF structure plot ----------------------------------------------------

def plot_nmf_structure(H: np.ndarray, membership: np.ndarray,
                        out_path: Path, *, k: int, dpi: int,
                        width_in: float, height_in: float,
                        export_pdf: bool) -> None:
    """STRUCTURE-style stacked bars of H rows ordered by dominant component
    within each community.
    """
    n, kk = H.shape
    if kk != k:
        LOG.warning("Sym-NMF H shape %s does not match k=%d", H.shape, k)
    dom = np.argmax(H, axis=1)
    order = np.lexsort((dom, membership))
    H_ord = H[order]
    member_ord = membership[order]

    fig, (ax_bar, ax_strip) = plt.subplots(
        2, 1, figsize=(width_in, height_in * 0.8),
        gridspec_kw={"height_ratios": [20, 1], "hspace": 0.02},
        sharex=True,
    )
    # Stacked bars via numpy cumulative sum of rows (fully vectorised).
    x = np.arange(n)
    bottom = np.zeros(n, dtype=np.float64)
    for comp in range(kk):
        ax_bar.bar(x, H_ord[:, comp], bottom=bottom, width=1.0,
                   color=_COMMUNITY_PALETTE[comp % len(_COMMUNITY_PALETTE)],
                   linewidth=0, align="edge")
        bottom += H_ord[:, comp]
    ax_bar.set_xlim(0, n)
    ax_bar.set_ylim(0, 1.0)
    ax_bar.set_ylabel(f"Soft membership (K={k})")
    ax_bar.set_xticks([])
    ax_bar.set_title("Sym-NMF soft memberships (samples ordered by Leiden "
                     "community then dominant component)",
                     loc="left", fontweight="bold", fontsize=11)

    # Community strip beneath the structure plot.
    strip = member_ord.astype(np.int32).reshape(1, -1)
    # Remap noise to a dedicated code so it renders in grey.
    strip_display = strip.copy()
    strip_display[strip == NOISE_LABEL] = -1
    uniq = sorted(int(c) for c in np.unique(strip_display))
    lut = {c: _community_color(c) for c in uniq}
    codes = np.zeros_like(strip_display)
    for i, c in enumerate(uniq):
        codes[strip_display == c] = i
    cmap = ListedColormap([lut[c] for c in uniq])
    ax_strip.imshow(codes, cmap=cmap, aspect="auto",
                     interpolation="nearest")
    ax_strip.set_yticks([])
    ax_strip.set_xticks([])
    ax_strip.set_xlabel("Samples")
    fig.tight_layout()
    _save_fig(fig, out_path, dpi=dpi, export_pdf=export_pdf)


# --- Sankey multi-resolution ------------------------------------------------

def plot_sankey_multiresolution(assign_df: pd.DataFrame, out_path: Path,
                                  *, resolutions: Sequence[float],
                                  dpi: int, width_in: float,
                                  height_in: float,
                                  export_pdf: bool) -> None:
    """Flow-between-resolutions using stacked bars connected by polygons.

    We draw one vertical bar per resolution and fill quadrilaterals
    proportional to the |community_a ∩ community_b| count between
    consecutive resolutions.  Pure matplotlib — no plotly dependency.
    """
    cols = [f"community_res_{r:g}" for r in resolutions
            if f"community_res_{r:g}" in assign_df.columns]
    if len(cols) < 2:
        LOG.warning("Not enough resolution columns for sankey plot.")
        return
    n = len(assign_df)
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    # Compute stacked positions for each column.
    stacks: list[dict[int, tuple[float, float]]] = []
    for col in cols:
        vals, counts = np.unique(assign_df[col].to_numpy(), return_counts=True)
        order = np.argsort(-counts)
        pos = {}
        y = 0
        for v, c in zip(vals[order], counts[order]):
            pos[int(v)] = (y, y + c)
            y += c
        stacks.append(pos)

    x_positions = np.linspace(0, 1, len(cols))
    # Bars.
    bar_width = 0.03
    for i, (x, col, stack) in enumerate(zip(x_positions, cols, stacks)):
        for lbl, (y0, y1) in stack.items():
            ax.fill_between([x - bar_width / 2, x + bar_width / 2],
                             y0, y1, color=_community_color(lbl),
                             edgecolor="white", linewidth=0.4)
            ax.text(x, (y0 + y1) / 2, str(lbl) if lbl != NOISE_LABEL
                    else "N",
                    ha="center", va="center", fontsize=6,
                    color="white", fontweight="bold")

    # Flows between consecutive stacks.
    for i in range(len(cols) - 1):
        left_col, right_col = cols[i], cols[i + 1]
        pair = assign_df[[left_col, right_col]].to_numpy()
        lut = defaultdict(int)
        for l, r in pair:
            lut[(int(l), int(r))] += 1
        # For each left label, iterate right labels in order of current
        # stack; fill polygon using updated cursors.
        left_cursor = {k: v[0] for k, v in stacks[i].items()}
        right_cursor = {k: v[0] for k, v in stacks[i + 1].items()}
        x_l = x_positions[i] + bar_width / 2
        x_r = x_positions[i + 1] - bar_width / 2
        for (l, r), c in sorted(lut.items()):
            y_l0 = left_cursor[l]
            y_l1 = y_l0 + c
            y_r0 = right_cursor[r]
            y_r1 = y_r0 + c
            poly = np.array([
                [x_l, y_l0], [x_r, y_r0],
                [x_r, y_r1], [x_l, y_l1],
            ])
            ax.fill(poly[:, 0], poly[:, 1],
                    color=_community_color(l),
                    alpha=0.35, edgecolor="none")
            left_cursor[l] = y_l1
            right_cursor[r] = y_r1

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"γ={r}" for r in resolutions[: len(cols)]],
                        fontsize=9)
    ax.set_yticks([])
    ax.set_ylim(0, n)
    ax.set_title("Community assignment flows across Leiden resolutions",
                 loc="left", fontweight="bold", fontsize=11)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    _save_fig(fig, out_path, dpi=dpi, export_pdf=export_pdf)


# --- Validation boxplot -----------------------------------------------------

def plot_validation_box(pair_df: pd.DataFrame, membership: np.ndarray,
                         samples: Sequence[str], out_path: Path,
                         *, dpi: int, width_in: float,
                         height_in: float, export_pdf: bool) -> None:
    """Compara gráficamente el sharing dentro y entre comunidades."""
    idx = {s: i for i, s in enumerate(samples)}
    sa = pair_df["sample_a"].astype(str).to_numpy()
    sb = pair_df["sample_b"].astype(str).to_numpy()
    bp = pair_df["total_shared_bp"].to_numpy(dtype=np.float64)
    ca = np.fromiter((idx.get(s, -1) for s in sa), dtype=np.int64, count=len(sa))
    cb = np.fromiter((idx.get(s, -1) for s in sb), dtype=np.int64, count=len(sb))
    valid = (ca >= 0) & (cb >= 0)
    ca, cb, bp = ca[valid], cb[valid], bp[valid]
    la_ = membership[ca]
    lb_ = membership[cb]
    non_noise = (la_ != NOISE_LABEL) & (lb_ != NOISE_LABEL)
    la_, lb_, bp = la_[non_noise], lb_[non_noise], bp[non_noise]
    intra = bp[la_ == lb_]
    inter = bp[la_ != lb_]
    if intra.size == 0 or inter.size == 0:
        LOG.warning("Validation boxplot: empty intra or inter; skipping.")
        return

    fig, ax = plt.subplots(figsize=(width_in * 0.55, height_in * 0.7))
    bp_log = lambda arr: np.log10(np.clip(arr, 1, None))
    ax.boxplot([bp_log(intra), bp_log(inter)],
                labels=[f"Intra-community\n(n={intra.size})",
                         f"Inter-community\n(n={inter.size})"],
                widths=0.5, patch_artist=True,
                boxprops=dict(facecolor="#4363d8", alpha=0.6),
                medianprops=dict(color="black"),
                showfliers=False)
    ax.set_ylabel("log₁₀(total_shared_bp)")
    ax.set_title("Intra- vs inter-community sharing",
                  loc="left", fontweight="bold", fontsize=11)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    _save_fig(fig, out_path, dpi=dpi, export_pdf=export_pdf)


# --- Network plot: UMAP over spectral embedding (M16.5) --------------------

def _spectral_embedding(S: sp.csr_matrix, n_components: int,
                         laplacian: bool = True) -> np.ndarray:
    """Return top-k eigenvectors of the sharing matrix as a low-dim embedding.

    Why spectral (vs Fruchterman-Reingold):
      FR minimises an energy function that treats all edges as springs;
      at N~10^3 with moderate density it collapses into a hairball + a
      ring of disconnected nodes (exactly what you observed in the M16
      network plots).  Spectral embeddings instead resolve modular
      structure along the leading eigenvectors of the graph Laplacian —
      the same mathematical objects that Leiden optimises.  Projecting
      those eigenvectors via UMAP preserves local neighbourhoods while
      flattening to 2D, which is the modern best practice for
      cell-community visualisation (Traag 2019; McInnes 2018).
    """
    n = S.shape[0]
    k = min(int(n_components), max(2, n - 2))
    if not _HAS_EIGSH or k >= n - 1:
        # Dense fallback — acceptable up to N ~= 5000.
        M = S.toarray().astype(np.float64)
        if laplacian:
            d = M.sum(axis=1)
            inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
            M = (inv_sqrt[:, None] * M) * inv_sqrt[None, :]
        vals, vecs = np.linalg.eigh(M)
        idx = np.argsort(-vals)[:k]
        return vecs[:, idx].astype(np.float64)
    if laplacian:
        M = laplacian_normalize(S).astype(np.float64)
    else:
        M = S.astype(np.float64)
    # eigsh with which='LA' for largest algebraic eigenvalues of the
    # symmetric normalised sharing — these correspond to the slowest-
    # mixing diffusion modes (Coifman & Lafon 2006 diffusion maps).
    vals, vecs = eigsh(M, k=k, which="LA")
    order = np.argsort(-vals)
    return vecs[:, order].astype(np.float64)


def plot_network_umap(S: sp.csr_matrix, membership: np.ndarray,
                       out_path: Path,
                       *, confidence: np.ndarray | None = None,
                       metadata_values: np.ndarray | None = None,
                       metadata_name: str | None = None,
                       max_nodes: int, dpi: int, width_in: float,
                       height_in: float, export_pdf: bool,
                       export_svg: bool = False,
                       layout_seed: int,
                       n_spectral: int = 15,
                       label_min_size: int = 10,
                       community_annotations: dict[int, str] | None = None,
                       adjust_labels: bool = False,
                       static_3d_png_path: Path | None = None,
                       ) -> None:
    """UMAP-over-spectral-embedding network plot (M16.5).

    Renders a static PNG (and optional PDF) via matplotlib.  When
    ``static_3d_png_path`` is provided, also writes a 4-panel 3D
    multiview PNG (XY / XZ / YZ orthogonal projections + perspective
    view at 600 DPI) — the publication-grade alternative to interactive
    HTML, legible offline and without WebGL.
    """
    n = S.shape[0]
    if n == 0:
        LOG.warning("Empty graph: skipping UMAP network plot.")
        return

    # ---- 1. Stratified down-sampling to fit max_nodes -------------------
    wdeg = np.asarray(S.sum(axis=1)).ravel()
    keep_mask = np.ones(n, dtype=bool)
    if n > max_nodes:
        order_in = np.argsort(-wdeg)
        comm_order: dict[int, list[int]] = defaultdict(list)
        for node in order_in:
            comm_order[int(membership[node])].append(int(node))
        keep_mask = np.zeros(n, dtype=bool)
        sizes = {c: len(v) for c, v in comm_order.items()}
        total = float(sum(sizes.values()))
        for c, nodes in comm_order.items():
            take = max(1, int(round(max_nodes * sizes[c] / total)))
            for node in nodes[:take]:
                keep_mask[node] = True
        LOG.info("UMAP network: down-sampled %d/%d nodes (stratified).",
                 int(keep_mask.sum()), n)

    sub_idx = np.flatnonzero(keep_mask)
    sub_S = S[sub_idx][:, sub_idx]
    sub_memb = membership[sub_idx]
    sub_wdeg = wdeg[sub_idx]

    # ---- 2. Spectral embedding → UMAP (computed once, reused) -----------
    LOG.info("UMAP network: computing spectral embedding (%d components) + "
             "UMAP on %d nodes.", n_spectral, sub_S.shape[0])
    emb = _spectral_embedding(sub_S, n_components=n_spectral, laplacian=True)
    if _HAS_UMAP and sub_S.shape[0] >= 4:
        try:
            reducer = umap.UMAP(
                n_components=2, random_state=int(layout_seed),
                n_neighbors=min(15, max(2, sub_S.shape[0] - 1)),
                min_dist=0.3, metric="euclidean",
            )
            coords = reducer.fit_transform(emb)
        except Exception as exc:  # pragma: no cover
            LOG.warning("UMAP failed (%s); falling back to raw 2D spectral.",
                         exc)
            coords = emb[:, :2]
    else:
        if not _HAS_UMAP:
            LOG.warning("umap-learn not available; using raw 2D spectral.")
        coords = emb[:, :2]

    # ---- 3a. Static matplotlib path (PNG + optional PDF/SVG) ------------
    _plot_network_static(
        coords=coords, sub_idx=sub_idx, sub_memb=sub_memb,
        sub_wdeg=sub_wdeg, membership=membership,
        out_path=out_path,
        confidence=confidence,
        metadata_values=metadata_values, metadata_name=metadata_name,
        dpi=dpi, width_in=width_in, height_in=height_in,
        export_pdf=export_pdf, export_svg=export_svg,
        label_min_size=label_min_size,
        community_annotations=community_annotations,
        adjust_labels=adjust_labels,
    )

    # ---- 3b. Optional 3D static multiview PNG ---------------------------
    # Replaces the previous Plotly Scatter3d HTML, which (a) failed to
    # use WebGL acceleration → laggy on >1k points, (b) collapsed all
    # communities into a single visual hue at typical viewing distance,
    # and (c) shipped 5 MB HTMLs that the user couldn't manipulate
    # smoothly.  A static 4-panel PNG at 600 DPI is the population-genetics
    # publication standard (Patterson 2006 EigenAnalysis, Lawson 2012
    # fineSTRUCTURE, Browning 2018 IBDNe): orthogonal projections + a
    # perspective view convey 3D structure without interactivity.
    if static_3d_png_path is not None:
        if not _HAS_UMAP or sub_S.shape[0] < 4:
            LOG.warning(
                "umap-learn unavailable or N=%d < 4; cannot fit 3D UMAP, "
                "skipping %s.", sub_S.shape[0], static_3d_png_path,
            )
        else:
            try:
                reducer3d = umap.UMAP(
                    n_components=3, random_state=int(layout_seed),
                    n_neighbors=min(15, max(2, sub_S.shape[0] - 1)),
                    min_dist=0.3, metric="euclidean",
                )
                coords3d = reducer3d.fit_transform(emb)
            except Exception as exc:  # pragma: no cover
                LOG.warning("3D UMAP failed (%s); falling back to first "
                            "3 spectral components.", exc)
                coords3d = emb[:, :3] if emb.shape[1] >= 3 else np.column_stack(
                    [emb[:, 0], emb[:, min(1, emb.shape[1]-1)],
                     np.zeros(emb.shape[0])])
            sub_meta = (
                np.asarray(metadata_values)[sub_idx]
                if metadata_values is not None else None
            )
            _plot_network_3d_multiview(
                coords3d=coords3d, sub_idx=sub_idx, sub_memb=sub_memb,
                sub_wdeg=sub_wdeg, membership=membership,
                community_annotations=community_annotations,
                metadata_values=sub_meta, metadata_name=metadata_name,
                out_path=static_3d_png_path,
                dpi=dpi, width_in=width_in, height_in=height_in,
            )


def _plot_network_static(*, coords: np.ndarray, sub_idx: np.ndarray,
                          sub_memb: np.ndarray, sub_wdeg: np.ndarray,
                          membership: np.ndarray, out_path: Path,
                          confidence: np.ndarray | None,
                          metadata_values: np.ndarray | None,
                          metadata_name: str | None,
                          dpi: int, width_in: float, height_in: float,
                          export_pdf: bool, export_svg: bool,
                          label_min_size: int,
                          community_annotations: dict[int, str] | None,
                          adjust_labels: bool) -> None:
    """Matplotlib path for plot_network_umap (PNG + optional PDF/SVG).

    Splits out the static rendering so the parent function can also feed
    the same coords to the Plotly path without duplicating the UMAP
    layout cost.  When ``adjust_labels`` is True, centroid labels are
    laid out by adjustText (force-directed repulsion) instead of plain
    placement — this fixes overlap between communities packed close in
    UMAP space.
    """
    has_meta = (metadata_values is not None and metadata_name is not None)
    ncols = 2 if has_meta else 1
    fig, axes = plt.subplots(
        1, ncols, figsize=(width_in * (1.5 if has_meta else 1.0), height_in),
        squeeze=False,
    )

    def _scatter(ax: plt.Axes, colors: list[str], title: str,
                  legend_title: str, handles: list[Patch]) -> None:
        size_scale = 10 + 120 * np.sqrt(
            sub_wdeg / (sub_wdeg.max() if sub_wdeg.max() > 0 else 1.0)
        )
        if confidence is not None:
            sub_conf = confidence[sub_idx]
            sub_conf = np.where(np.isfinite(sub_conf), sub_conf, 0.0)
            edge_alpha = np.clip(0.25 + 0.75 * sub_conf, 0.25, 1.0)
        else:
            edge_alpha = np.full(coords.shape[0], 1.0)
        is_noise = sub_memb == NOISE_LABEL
        ax.scatter(coords[is_noise, 0], coords[is_noise, 1],
                    c=[_NOISE_COLOR] * int(is_noise.sum()),
                    s=size_scale[is_noise] * 0.4, alpha=0.35,
                    linewidths=0, zorder=1)
        ax.scatter(coords[~is_noise, 0], coords[~is_noise, 1],
                    c=[colors[i] for i in np.flatnonzero(~is_noise)],
                    s=size_scale[~is_noise],
                    edgecolors="white",
                    linewidths=0.3 * edge_alpha[~is_noise],
                    zorder=2)
        ax.set_axis_off()
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        if handles:
            ax.legend(handles=handles, loc="center left",
                       bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=False,
                       title=legend_title, title_fontsize=8)
        # Centroid labels removed (2026-05-07).  At N≈3k with 50+
        # communities, label-on-plot + adjustText leaders produce a
        # caotic mosaic of arrows that obscures the actual data —
        # unsuitable for publication figures.  All identity now lives
        # in the legend ('C9 — BrazilA·subA (n=40)'), keeping the
        # scatter clean.  This matches the convention in Lawson 2012,
        # Nunes 2025 and most Nature/Cell population-genetics figures.
        # ``adjust_labels`` is accepted for backwards CLI compatibility
        # but has no effect; if/when an alternative layered label
        # overlay is needed, this is the hook to extend.

    def _community_legend_label(c: int) -> str:
        """Compose 'C{id} — annotation (size)' for the legend handles.

        The annotation comes from ``community_annotations`` (e.g.
        'BrazilA·subA' from the auto-annotation table).  When absent,
        falls back to plain 'C{id} (size)'.  Size is the FULL community
        count (``membership``), not the down-sampled view, so legend
        figures are biologically accurate even when the scatter is
        rendered with max_nodes < N.
        """
        full_size = int((membership == c).sum())
        if int(c) == NOISE_LABEL:
            return f"Noise (n={full_size})"
        ann = (community_annotations or {}).get(int(c))
        if ann:
            return f"C{int(c)} — {ann} (n={full_size})"
        return f"C{int(c)} (n={full_size})"

    # Panel A: community-coloured.  Legend keeps noise + the top-30
    # communities by size (ordered descending), with annotations
    # appended.  At ~50 communities the cap drops the long tail of
    # 3-5-individual kin clusters that would saturate the legend
    # without adding biological insight.
    colors_comm = [_community_color(int(m)) for m in sub_memb]
    non_noise_comms = np.unique(sub_memb[sub_memb != NOISE_LABEL])
    sizes_full = {int(c): int((membership == c).sum())
                   for c in non_noise_comms}
    top_n = 30
    top_comms = sorted(non_noise_comms,
                        key=lambda c: -sizes_full[int(c)])[:top_n]
    uniq_comms = np.concatenate([
        np.array([NOISE_LABEL])
        if (membership == NOISE_LABEL).any() else np.array([], dtype=int),
        np.asarray(top_comms, dtype=int),
    ])
    handles_comm = [
        Patch(facecolor=_community_color(int(c)), edgecolor="white",
              label=_community_legend_label(int(c)))
        for c in uniq_comms
    ]
    _scatter(axes[0, 0], colors_comm,
             "Rare-variant co-sharing — UMAP over spectral embedding "
             "(colour = Leiden community)",
             "Community (size)", handles_comm)

    # Panel B: metadata-coloured (only when metadata supplied).
    if has_meta:
        sub_meta = np.asarray(metadata_values)[sub_idx]
        levels, inverse = np.unique(sub_meta.astype(str), return_inverse=True)
        palette = _COMMUNITY_PALETTE
        lvl_colors = [palette[i % len(palette)] for i in range(len(levels))]
        colors_meta = [lvl_colors[i] for i in inverse]
        handles_meta = [
            Patch(facecolor=lvl_colors[i], edgecolor="white",
                  label=f"{lvl} ({int((sub_meta == lvl).sum())})")
            for i, lvl in enumerate(levels)
        ]
        _scatter(axes[0, 1], colors_meta,
                 f"Same layout coloured by {metadata_name}",
                 metadata_name, handles_meta)

    fig.tight_layout()
    _save_fig(fig, out_path, dpi=dpi, export_pdf=export_pdf,
              export_svg=export_svg)


def _compute_node_visuals(
    sub_memb: np.ndarray,
    sub_wdeg: np.ndarray,
    membership: np.ndarray,
    community_annotations: dict[int, str] | None,
    *,
    confidence_sub: np.ndarray | None = None,
    metadata_values: np.ndarray | None = None,
    metadata_name: str | None = None,
) -> dict[str, Any]:
    """Single source of truth for node visuals (community or metadata mode).

    Sizes/edge-alpha are always community/wdeg-driven.  Colours + legend
    depend on whether ``metadata_values`` is supplied:

      * **community mode** (metadata_values is None) — colours come from
        the categorical palette indexed by community id; ``is_noise`` is
        ``sub_memb == NOISE_LABEL``; legend title is ``"Community (n)"``.
      * **metadata mode** — colours come from the same palette indexed
        by the alphabetical order of the unique levels of
        ``metadata_values`` (so the same level always gets the same hue
        across resolutions); samples with missing metadata are coloured
        as ``_NOISE_COLOR`` and grouped under ``"(missing)"`` in the
        legend; legend title becomes ``metadata_name``.
    """
    # Size: sqrt-scaled wdeg so very-high-sharing samples stand out without
    # crushing the distribution.
    wdeg_max = float(sub_wdeg.max()) if sub_wdeg.size and sub_wdeg.max() > 0 else 1.0
    mpl_sizes = 10.0 + 120.0 * np.sqrt(np.maximum(sub_wdeg, 0.0) / wdeg_max)

    if confidence_sub is not None:
        conf = np.where(np.isfinite(confidence_sub), confidence_sub, 0.0)
        edge_alpha = np.clip(0.25 + 0.75 * conf, 0.25, 1.0)
    else:
        edge_alpha = np.full(sub_memb.shape[0], 1.0)

    if metadata_values is not None and metadata_name is not None:
        meta = np.asarray(metadata_values, dtype=object)
        valid = np.array([
            m is not None and not (isinstance(m, float) and np.isnan(m))
            and str(m).strip() not in ("", "nan", "NaN", "None")
            for m in meta
        ])
        meta_str = np.array([str(m) if v else "" for m, v in zip(meta, valid)])
        levels = sorted(set(meta_str[valid].tolist()))
        level_to_idx = {lvl: i for i, lvl in enumerate(levels)}
        colors = [_NOISE_COLOR] * meta.shape[0]
        for i in np.flatnonzero(valid):
            colors[i] = _COMMUNITY_PALETTE[
                level_to_idx[meta_str[i]] % len(_COMMUNITY_PALETTE)
            ]
        full_counts = {lvl: int((meta_str == lvl).sum()) for lvl in levels}
        legend_handles = [
            Patch(
                facecolor=_COMMUNITY_PALETTE[level_to_idx[lvl] % len(_COMMUNITY_PALETTE)],
                edgecolor="white",
                label=f"{lvl} (n={full_counts[lvl]})",
            )
            for lvl in levels
        ]
        n_missing = int((~valid).sum())
        if n_missing:
            legend_handles.append(
                Patch(facecolor=_NOISE_COLOR, edgecolor="white",
                      label=f"(missing) (n={n_missing})")
            )
        return {
            "mpl_sizes": mpl_sizes,
            "colors": colors,
            "edge_alpha": edge_alpha,
            "is_noise": ~valid,  # "noise" = missing-metadata in this mode
            "legend_handles": legend_handles,
            "legend_label_by_c": {},  # unused in metadata mode
            "legend_title": metadata_name,
        }

    # --- Community mode (default) ----------------------------------------
    is_noise = sub_memb == NOISE_LABEL
    colors = [_community_color(int(m)) for m in sub_memb]

    non_noise = np.unique(sub_memb[~is_noise])
    full_sizes = {int(c): int((membership == c).sum()) for c in non_noise}
    legend_label_by_c: dict[int, str] = {}
    for c in non_noise:
        cid = int(c)
        ann = (community_annotations or {}).get(cid)
        if ann:
            legend_label_by_c[cid] = f"C{cid} — {ann} (n={full_sizes[cid]})"
        else:
            legend_label_by_c[cid] = f"C{cid} (n={full_sizes[cid]})"
    legend_label_by_c[NOISE_LABEL] = (
        f"Noise (n={int((membership == NOISE_LABEL).sum())})"
    )

    top_n = 30
    top_comms = sorted(non_noise, key=lambda c: -full_sizes[int(c)])[:top_n]
    uniq = np.concatenate([
        np.array([NOISE_LABEL])
        if (membership == NOISE_LABEL).any() else np.array([], dtype=int),
        np.asarray(top_comms, dtype=int),
    ])
    legend_handles = [
        Patch(facecolor=_community_color(int(c)), edgecolor="white",
              label=legend_label_by_c[int(c)])
        for c in uniq
    ]
    return {
        "mpl_sizes": mpl_sizes,
        "colors": colors,
        "edge_alpha": edge_alpha,
        "is_noise": is_noise,
        "legend_handles": legend_handles,
        "legend_label_by_c": legend_label_by_c,
        "legend_title": "Community (n)",
    }


def _plot_network_3d_multiview(*, coords3d: np.ndarray,
                                sub_idx: np.ndarray,
                                sub_memb: np.ndarray,
                                sub_wdeg: np.ndarray,
                                membership: np.ndarray,
                                community_annotations: dict[int, str] | None,
                                metadata_values: np.ndarray | None,
                                metadata_name: str | None,
                                out_path: Path,
                                dpi: int, width_in: float,
                                height_in: float) -> None:
    """4-panel static 3D UMAP plot: XY · XZ · YZ · perspective.

    Replaces the previous Plotly Scatter3d HTML.  Rationale: Scatter3d
    does not use WebGL acceleration; on N>1k it lags badly, and at the
    default camera distance individual community hues compress into a
    visually uniform haze.  A static publication-grade plot is the
    population-genetics standard (Patterson 2006, Lawson 2012, Browning
    2018) — three orthogonal projections expose structure that any
    single 2D view collapses, plus a perspective panel for context.

    Panel layout
    ------------
    Top row    : XY (top-down, mirrors the 2D network plot) · XZ (side)
    Bottom row : YZ (front) · perspective view at elev=22°, azim=-60°
    Legend bar : shared across the figure, on the right margin

    All panels share colour mapping, marker scaling, and axis limits so
    the reader can cross-reference: a tight cluster in XY that disperses
    in XZ is genuinely 3D structure, not a 2D artefact.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — register projection
    n = coords3d.shape[0]
    if n == 0:
        LOG.warning("Empty graph: skipping 3D multiview plot.")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    visuals = _compute_node_visuals(
        sub_memb=sub_memb, sub_wdeg=sub_wdeg, membership=membership,
        community_annotations=community_annotations,
        metadata_values=metadata_values, metadata_name=metadata_name,
    )
    mpl_sizes = visuals["mpl_sizes"]
    colors = visuals["colors"]
    is_noise = visuals["is_noise"]
    legend_handles = visuals["legend_handles"]
    legend_title = visuals["legend_title"]

    # Reserve ~20% of figure width for the shared legend on the right.
    # Slightly taller than 2D since two rows of panels stack vertically.
    fig = plt.figure(figsize=(width_in * 1.25, height_in * 1.15))
    gs = fig.add_gridspec(
        2, 3, width_ratios=[1.0, 1.0, 0.45],
        wspace=0.18, hspace=0.18,
        left=0.04, right=0.99, top=0.94, bottom=0.04,
    )
    ax_xy = fig.add_subplot(gs[0, 0])
    ax_xz = fig.add_subplot(gs[0, 1])
    ax_yz = fig.add_subplot(gs[1, 0])
    ax_3d = fig.add_subplot(gs[1, 1], projection="3d")
    ax_legend = fig.add_subplot(gs[:, 2])
    ax_legend.set_axis_off()

    def _scatter_2d(ax: plt.Axes, xs: np.ndarray, ys: np.ndarray,
                     xlabel: str, ylabel: str, title: str) -> None:
        # Noise behind communities; transparent halo so we can see overlap.
        ax.scatter(xs[is_noise], ys[is_noise],
                    c=[_NOISE_COLOR] * int(is_noise.sum()),
                    s=mpl_sizes[is_noise] * 0.4, alpha=0.30,
                    linewidths=0, zorder=1)
        ax.scatter(xs[~is_noise], ys[~is_noise],
                    c=[colors[i] for i in np.flatnonzero(~is_noise)],
                    s=mpl_sizes[~is_noise],
                    edgecolors="white", linewidths=0.3,
                    alpha=0.92, zorder=2)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.set_facecolor(_BACKGROUND_COLOR)
        ax.grid(True, alpha=0.25, linewidth=0.4)

    _scatter_2d(ax_xy, coords3d[:, 0], coords3d[:, 1],
                 "UMAP-1", "UMAP-2", "A. Top view (UMAP-1 × UMAP-2)")
    _scatter_2d(ax_xz, coords3d[:, 0], coords3d[:, 2],
                 "UMAP-1", "UMAP-3", "B. Side view (UMAP-1 × UMAP-3)")
    _scatter_2d(ax_yz, coords3d[:, 1], coords3d[:, 2],
                 "UMAP-2", "UMAP-3", "C. Front view (UMAP-2 × UMAP-3)")

    # 3D perspective.  elev=22°, azim=-60° matches the matplotlib
    # default-but-rotated camera; gives good depth cue without occluding
    # the densest cluster.  Sizes shrink slightly because perspective
    # inflates near-points.
    ax_3d.scatter(
        coords3d[is_noise, 0], coords3d[is_noise, 1], coords3d[is_noise, 2],
        c=[_NOISE_COLOR] * int(is_noise.sum()),
        s=mpl_sizes[is_noise] * 0.35, alpha=0.25, depthshade=False,
        linewidths=0,
    )
    ax_3d.scatter(
        coords3d[~is_noise, 0], coords3d[~is_noise, 1], coords3d[~is_noise, 2],
        c=[colors[i] for i in np.flatnonzero(~is_noise)],
        s=mpl_sizes[~is_noise] * 0.75, alpha=0.92, depthshade=True,
        edgecolors="white", linewidths=0.25,
    )
    ax_3d.view_init(elev=22, azim=-60)
    ax_3d.set_xlabel("UMAP-1", fontsize=9)
    ax_3d.set_ylabel("UMAP-2", fontsize=9)
    ax_3d.set_zlabel("UMAP-3", fontsize=9)
    ax_3d.set_title("D. 3D perspective (elev=22°, azim=−60°)",
                     loc="left", fontsize=10, fontweight="bold")
    ax_3d.tick_params(labelsize=6)

    # Shared legend column.  Title reflects the active colouring scheme
    # (community by default, or the metadata column when REPLOT supplied
    # one).  Top-30 cap inside _compute_node_visuals keeps the box tight.
    ax_legend.legend(
        handles=legend_handles, loc="center left", frameon=False,
        fontsize=7, title=legend_title, title_fontsize=9,
        labelspacing=0.4, handletextpad=0.6, borderaxespad=0.0,
    )

    suptitle_extra = (
        f" — coloured by {metadata_name}"
        if (metadata_values is not None and metadata_name) else ""
    )
    fig.suptitle(
        f"{out_path.stem}{suptitle_extra}  (palette: {_PALETTE_NAME})",
        fontsize=12, fontweight="bold", x=0.04, y=0.99, ha="left",
    )

    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    LOG.info("Wrote 3D multiview PNG %s (%d nodes, %d communities).",
              out_path, n, len(legend_handles))


# --- Sharing heatmap with hierarchical within-community ordering (M16.5) ---

def _within_community_hierarchical_order(S: sp.csr_matrix,
                                          membership: np.ndarray) -> np.ndarray:
    """Produce a global ordering: community block sorted by size, then
    UPGMA leaf order within each community (so bloated diagonals reveal
    fine-scale substructure instead of arbitrary sample index).
    """
    order_parts: list[np.ndarray] = []
    # Noise first so it appears as an empty top-left block.
    noise_idx = np.flatnonzero(membership == NOISE_LABEL)
    if noise_idx.size:
        order_parts.append(noise_idx)
    # Remaining communities sorted descending by size.
    non_noise = np.unique(membership[membership != NOISE_LABEL])
    sizes = {int(c): int((membership == c).sum()) for c in non_noise}
    for c in sorted(non_noise, key=lambda x: -sizes[int(x)]):
        idx = np.flatnonzero(membership == c)
        if idx.size < 3:
            order_parts.append(idx)
            continue
        sub = S[idx][:, idx].toarray().astype(np.float64)
        # Similarity → distance; row-normalise then symmetrise.
        row_sums = sub.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        sim = sub / row_sums
        sim = 0.5 * (sim + sim.T)
        dmat = 1.0 - sim
        np.fill_diagonal(dmat, 0.0)
        try:
            condensed = squareform(dmat, checks=False)
            Z = linkage(condensed, method="average")
            from scipy.cluster.hierarchy import leaves_list
            leaf_order = leaves_list(Z)
            order_parts.append(idx[leaf_order])
        except Exception as exc:  # pragma: no cover
            LOG.warning("UPGMA ordering failed for community %d (%s); "
                         "using unsorted.", int(c), exc)
            order_parts.append(idx)
    return np.concatenate(order_parts) if order_parts else np.arange(S.shape[0])


def plot_sharing_heatmap_enhanced(
    S: sp.csr_matrix,
    membership: np.ndarray,
    out_path: Path,
    *, confidence: np.ndarray | None = None,
    metadata_values: np.ndarray | None = None,
    metadata_name: str | None = None,
    max_nodes: int, dpi: int, width_in: float,
    height_in: float, export_pdf: bool) -> None:
    """Sharing heatmap with within-community hierarchical ordering + sidebars.

    Visual improvements over M16:
      * Samples inside each community are UPGMA-ordered on the
        row-normalised sharing distance, so sub-communities and
        cryptic family blocks emerge inside each diagonal tile instead
        of being randomly permuted.
      * A left sidebar stacks annotation strips (always: community;
        when available: confidence; when metadata supplied: metadata).
      * Boundaries between communities are thin white lines.
    """
    n = S.shape[0]
    if n == 0:
        return
    order = _within_community_hierarchical_order(S, membership)
    if n > max_nodes:
        LOG.info("Heatmap (enhanced): down-sampling %d -> %d.", n, max_nodes)
        order = order[:max_nodes]

    ordered = S[order, :][:, order].toarray()
    M = np.log1p(ordered)

    n_plot = ordered.shape[0]
    # Number of sidebar strips to draw.
    strip_names = ["Community"]
    if confidence is not None:
        strip_names.append("Confidence")
    if metadata_values is not None and metadata_name is not None:
        strip_names.append(metadata_name)
    n_strips = len(strip_names)
    # Left side: 1 gridspec column per strip + the main heatmap.
    strip_w = 0.04
    fig = plt.figure(figsize=(width_in, width_in))
    gs = fig.add_gridspec(
        1, n_strips + 2,
        width_ratios=[strip_w] * n_strips + [1.0, 0.04],
        wspace=0.02,
    )
    strip_axes = [fig.add_subplot(gs[0, i]) for i in range(n_strips)]
    ax = fig.add_subplot(gs[0, n_strips])
    cax = fig.add_subplot(gs[0, n_strips + 1])

    im = ax.imshow(M, cmap="magma", interpolation="nearest", aspect="equal",
                    origin="upper")
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("log1p(shared weight)")

    # Boundaries between communities.
    ordered_members = membership[order]
    boundaries = np.where(np.diff(ordered_members) != 0)[0] + 1
    for b in boundaries:
        ax.axhline(b - 0.5, color="white", linewidth=0.4, alpha=0.7)
        ax.axvline(b - 0.5, color="white", linewidth=0.4, alpha=0.7)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Pairwise sharing ordered by community, UPGMA within "
                 f"({n_plot} samples)",
                 loc="left", fontweight="bold", fontsize=11)

    # Sidebar strips.
    def _strip(ax_strip: plt.Axes, values: np.ndarray, label: str,
                cmap_name: str | None, categorical: bool) -> None:
        if categorical:
            # Unique codes → colour lookup.
            levels, inverse = np.unique(
                values.astype(str), return_inverse=True
            )
            palette = [_community_color(i - 1) if levels[i] == str(NOISE_LABEL)
                        else _COMMUNITY_PALETTE[i % len(_COMMUNITY_PALETTE)]
                        for i in range(len(levels))]
            cmap = ListedColormap(palette)
            strip = inverse.reshape(-1, 1)
            ax_strip.imshow(strip, cmap=cmap, aspect="auto",
                             interpolation="nearest")
        else:
            strip = np.asarray(values, dtype=np.float64).reshape(-1, 1)
            strip = np.where(np.isfinite(strip), strip, 0.0)
            ax_strip.imshow(
                strip, cmap=cmap_name or "viridis", aspect="auto",
                vmin=0.0, vmax=1.0, interpolation="nearest",
            )
        ax_strip.set_xticks([])
        ax_strip.set_yticks([])
        ax_strip.set_xlabel(label, rotation=90, fontsize=7, va="top")

    _strip(strip_axes[0], ordered_members.astype(str), "Community",
            None, True)
    idx_strip = 1
    if confidence is not None:
        conf_ordered = np.asarray(confidence)[order]
        _strip(strip_axes[idx_strip], conf_ordered, "Confidence",
                "viridis", False)
        idx_strip += 1
    if metadata_values is not None and metadata_name is not None:
        meta_ordered = np.asarray(metadata_values)[order]
        _strip(strip_axes[idx_strip], meta_ordered, metadata_name,
                None, True)
        idx_strip += 1

    _save_fig(fig, out_path, dpi=dpi, export_pdf=export_pdf)


# --- Kinship diagnostic scatter (M16.5) ------------------------------------

def plot_kinship_scatter(pair_with_seg: pd.DataFrame,
                          membership: np.ndarray,
                          samples: Sequence[str],
                          out_path: Path,
                          *, kinship_segment_bp: int,
                          dpi: int, width_in: float,
                          height_in: float, export_pdf: bool) -> None:
    """Scatter of mean_segment_bp vs total_shared_bp per pair, log-log.

    Biological reading (Browning 2012; Henn 2012):
      * Upper-right corner: first- to fourth-degree relatives
        (long segments, high total bp).  The ``kinship_segment_bp``
        threshold is drawn as a horizontal dashed line.
      * Lower-left: distant / shared-ancestry background.
      * Upper-left: few very long segments but small total → possibly
        partial IBD tracts from a specific chromosomal region.
      * Right side but short segments: many short segments → same
        ancestry background accumulated across chromosomes.
    Colour indicates whether both members of the pair belong to the
    same Leiden community (intra) or different ones (inter).
    """
    if pair_with_seg.empty or "max_segment_bp" not in pair_with_seg.columns:
        LOG.warning("Kinship scatter: need max_segment_bp column; skipping.")
        return
    idx = {s: i for i, s in enumerate(samples)}
    sa = pair_with_seg["sample_a"].astype(str).to_numpy()
    sb = pair_with_seg["sample_b"].astype(str).to_numpy()
    valid = np.array([s in idx for s in sa]) & np.array([s in idx for s in sb])
    if not valid.any():
        return
    ms = pair_with_seg["max_segment_bp"].to_numpy(dtype=np.float64)[valid]
    tb = pair_with_seg["total_shared_bp"].to_numpy(dtype=np.float64)[valid]
    ns = pair_with_seg["n_segments"].to_numpy(dtype=np.float64)[valid]
    mean_seg = tb / np.clip(ns, 1, None)

    ca = np.fromiter((membership[idx[s]] for s in sa[valid]),
                     dtype=np.int64, count=int(valid.sum()))
    cb = np.fromiter((membership[idx[s]] for s in sb[valid]),
                     dtype=np.int64, count=int(valid.sum()))
    intra = (ca == cb) & (ca != NOISE_LABEL)

    # log-log scale needs strict positives.
    keep = (mean_seg > 0) & (tb > 0)
    mean_seg, tb, intra, ms = mean_seg[keep], tb[keep], intra[keep], ms[keep]

    fig, ax = plt.subplots(figsize=(width_in * 0.85, height_in * 0.8))
    ax.scatter(mean_seg[~intra], tb[~intra], s=4, c="#9aa0a6", alpha=0.35,
                linewidths=0, label=f"Inter-community (n={int((~intra).sum())})")
    ax.scatter(mean_seg[intra], tb[intra], s=6, c="#4363d8", alpha=0.5,
                linewidths=0, label=f"Intra-community (n={int(intra.sum())})")
    # Highlight kinship candidates (max_segment_bp crosses threshold).
    hot = ms >= float(kinship_segment_bp)
    if hot.any():
        ax.scatter(mean_seg[hot], tb[hot], s=22, facecolor="none",
                    edgecolors="#e6194b", linewidths=0.8,
                    label=f"Kinship candidates (n={int(hot.sum())}, "
                          f"max_seg ≥ {kinship_segment_bp / 1e6:.0f} Mb)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mean segment length per pair (bp, log scale)")
    ax.set_ylabel("total shared bp per pair (log scale)")
    ax.axhline(float(kinship_segment_bp), color="#e6194b",
                linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_title("Kinship diagnostic — per-pair IBD length vs total sharing",
                  loc="left", fontweight="bold", fontsize=11)
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    _save_fig(fig, out_path, dpi=dpi, export_pdf=export_pdf)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _embed_png(path: Path) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;height:auto;"/>'


def _load_tsv_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        try:
            return pd.read_csv(path, sep="\t")
        except Exception:  # pragma: no cover
            return None
    return None


def build_html_report(out_dir: Path,
                      summary: dict,
                      graph_summary: dict | None,
                      validation_df: pd.DataFrame | None,
                      modularity_df: pd.DataFrame | None,
                      resolutions: Sequence[float],
                      metadata_available: bool = False,
                      metadata_name: str | None = None,
                      warnings_list: list[str] | None = None) -> None:
    """M16.5 integrated HTML report: 6 sections mapping to bio questions.

    Sections:
      1. Dataset summary (N, config, warnings if any)
      2. Leiden stability: modularity + ARI + confidence
      3. Sym-NMF: cophenetic K + STRUCTURE plot at the recommended K
      4. Bio Q#1 — Macro-structure: enrichment dotplot (with metadata) OR
         silhouette + intra/inter (no metadata)
      5. Bio Q#2 — Cryptic kinship: scatter + table
      6. Bio Q#3 — Founder effects + Noise characterisation

    Each section carries an automatic interpretation line so the HTML
    is readable without opening the TSVs side by side.
    """
    plots_dir = out_dir / PLOTS_DIR
    sections: list[str] = []
    sections.append("<h1>Module 16.5 — IBD Community Detection Enhanced</h1>")
    sections.append(f"<p><i>Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}</i></p>")
    if warnings_list:
        sections.append('<div class="warn"><b>Warnings</b><ul>' +
                         "".join(f"<li>{w}</li>" for w in warnings_list) +
                         "</ul></div>")

    ari_df = _load_tsv_if_exists(out_dir / NAME_LEIDEN_ARI)
    sil_df = _load_tsv_if_exists(out_dir / NAME_COMMUNITY_SILHOUETTE)
    coph_df = _load_tsv_if_exists(out_dir / NAME_NMF_COPHENETIC)
    kin_df = _load_tsv_if_exists(out_dir / NAME_KINSHIP_CANDIDATES)
    founder_df = _load_tsv_if_exists(out_dir / NAME_FOUNDER_CANDIDATES)
    enrich_df = _load_tsv_if_exists(out_dir / NAME_ENRICHMENT)
    labels_df = _load_tsv_if_exists(out_dir / NAME_COMMUNITY_LABELS)

    # Section 1: Dataset summary.
    sections.append("<h2>1. Dataset &amp; run summary</h2>")
    if labels_df is not None and not labels_df.empty:
        sections.append("<h3>Communities at the validation resolution</h3>")
        sections.append(labels_df.to_html(index=False, float_format="%.3g",
                                           classes="table"))
    sections.append("<details><summary>Run configuration</summary>"
                    "<pre>" + json.dumps(summary, indent=2)
                    + "</pre></details>")
    if graph_summary is not None:
        sections.append("<details><summary>Graph statistics</summary>"
                        "<pre>" + json.dumps(graph_summary, indent=2)
                        + "</pre></details>")

    # Section 2: Leiden stability.
    sections.append("<h2>2. Leiden stability</h2>")
    interp_bits = []
    if ari_df is not None and not ari_df.empty:
        mid = ari_df[ari_df["resolution"] == 1.0]
        if not mid.empty:
            interp_bits.append(f"ARI median at γ=1.0 = "
                               f"{float(mid['median_ari'].iloc[0]):.3f}")
        best = ari_df.loc[ari_df["median_ari"].idxmax()]
        interp_bits.append(
            f"most stable γ = {float(best['resolution']):.2f} "
            f"(median ARI = {float(best['median_ari']):.3f})"
        )
    if modularity_df is not None and not modularity_df.empty:
        top = modularity_df.groupby("resolution")["modularity"].median()
        if not top.empty:
            best_g = float(top.idxmax())
            interp_bits.append(
                f"highest median modularity at γ = {best_g:.2f} "
                f"(Q = {float(top.max()):.3f})"
            )
    if interp_bits:
        sections.append("<p class='interp'>" + "; ".join(interp_bits) + "</p>")
    if ari_df is not None and not ari_df.empty:
        sections.append("<h3>ARI across Leiden seeds</h3>")
        sections.append(ari_df.to_html(index=False, float_format="%.4f",
                                        classes="table"))
    if modularity_df is not None and not modularity_df.empty:
        mdf = modularity_df.groupby("resolution")["modularity"].agg(
            ["median", "std", "count"]
        ).reset_index()
        sections.append("<h3>Modularity per resolution (across seeds)</h3>")
        sections.append(mdf.to_html(index=False, float_format="%.4f",
                                     classes="table"))

    # Section 3: Sym-NMF.
    sections.append("<h2>3. Sym-NMF soft memberships</h2>")
    if coph_df is not None and not coph_df.empty:
        k_opt_row = coph_df[coph_df.get("is_recommended_k", False) == True]  # noqa: E712
        if not k_opt_row.empty:
            k_opt = int(k_opt_row["k"].iloc[0])
            sections.append(
                f"<p class='interp'>Recommended K (largest with "
                f"cophenetic ≥ 0.90, Brunet 2004) = <b>K = {k_opt}</b>.</p>"
            )
        sections.append(coph_df.to_html(index=False, float_format="%.4f",
                                         classes="table"))
    for structure in sorted(plots_dir.glob("nmf_structure_k*.png")):
        m = re.search(r"nmf_structure_k(\d+)", structure.name)
        if m:
            sections.append(f"<h3>K = {m.group(1)}</h3>")
        sections.append(_embed_png(structure))

    # Section 4: Biological question #1 — Macro-structure.
    sections.append("<h2>4. Biological question #1 — Macro-structure</h2>")
    # Network UMAP plots (always).
    for r in resolutions:
        p = plots_dir / f"network_umap_res{r:g}.png"
        if p.exists():
            sections.append(f"<h3>Network UMAP — γ = {r}</h3>")
            sections.append(_embed_png(p))
    heat = plots_dir / "sharing_heatmap_enhanced.png"
    if heat.exists():
        sections.append("<h3>Sharing heatmap (UPGMA within community)</h3>")
        sections.append(_embed_png(heat))
    if metadata_available and enrich_df is not None and not enrich_df.empty:
        sig = enrich_df[(enrich_df["q_value"] < 0.01)
                         & (enrich_df["odds_ratio"] > 1)]
        if not sig.empty:
            sections.append(
                f"<p class='interp'>{len(sig)} "
                f"(community × {metadata_name}) pairs pass q &lt; 0.01 "
                f"with OR &gt; 1 → geography-driven communities.</p>"
            )
        sections.append("<h3>Fisher enrichment community × "
                         f"{metadata_name}</h3>")
        sections.append(enrich_df.head(50).to_html(
            index=False, float_format="%.4g", classes="table"
        ))
    elif metadata_available:
        sections.append(
            "<p class='interp'>Metadata provided but no significant "
            "enrichments detected at q &lt; 0.01.</p>"
        )
    else:
        sections.append(
            "<p class='interp'>No metadata supplied: question #1 is "
            "answered only structurally.  With region/UF TSV, this "
            "section would add a Fisher enrichment dotplot.</p>"
        )
    if sil_df is not None and not sil_df.empty:
        sections.append("<h3>Silhouette per community</h3>")
        sections.append(sil_df.to_html(index=False, float_format="%.4f",
                                        classes="table"))

    # Section 5: Biological question #2 — Cryptic kinship.
    sections.append("<h2>5. Biological question #2 — Cryptic kinship</h2>")
    scatter_path = plots_dir / "kinship_scatter.png"
    if scatter_path.exists():
        sections.append(_embed_png(scatter_path))
    if kin_df is not None and not kin_df.empty:
        n_pair = int((kin_df["candidate_type"] == "pair").sum())
        n_comm = int((kin_df["candidate_type"] == "community").sum())
        sections.append(
            f"<p class='interp'>{n_pair} candidate close-kin pairs and "
            f"{n_comm} candidate family-like communities detected.</p>"
        )
        sections.append(kin_df.head(50).to_html(
            index=False, float_format="%.0f", classes="table"
        ))
    else:
        sections.append("<p class='interp'>No kinship candidates above "
                         "threshold.</p>")

    # Section 6: Biological question #3 — Founder effects + Noise.
    sections.append("<h2>6. Biological question #3 — Founder effects</h2>")
    if founder_df is not None and not founder_df.empty:
        candidates = founder_df[founder_df["is_founder_candidate"] == True]  # noqa: E712
        if not candidates.empty:
            comm_list = ", ".join(f"C{int(c)}" for c
                                   in candidates["community"].tolist())
            sections.append(
                f"<p class='interp'>Founder-effect candidate "
                f"communities (size ≥ threshold, intra/inter ≥ ratio, "
                f"silhouette ≥ cutoff): <b>{comm_list}</b>.</p>"
            )
        sections.append(founder_df.to_html(
            index=False, float_format="%.4g", classes="table"
        ))
    else:
        sections.append("<p class='interp'>No founder candidates "
                         "above threshold.</p>")
    if validation_df is not None and not validation_df.empty:
        sections.append("<h3>Mann-Whitney intra vs inter sharing</h3>")
        sections.append(validation_df.to_html(
            index=False, float_format="%.4g", classes="table"
        ))
        box = plots_dir / "validation_intra_vs_inter.png"
        if box.exists():
            sections.append(_embed_png(box))

    style = """
    <style>
        body { font-family: -apple-system, 'Segoe UI', Helvetica, sans-serif;
               max-width: 1200px; margin: 2em auto; color: #222;
               line-height: 1.5; }
        h1, h2, h3 { color: #111; }
        h1 { border-bottom: 2px solid #4363d8; padding-bottom: 0.2em; }
        h2 { border-bottom: 1px solid #ddd; padding-bottom: 0.2em;
             margin-top: 2em; }
        table.table { border-collapse: collapse; margin-bottom: 1em; }
        table.table th, table.table td { border: 1px solid #ddd;
                                          padding: 4px 8px; font-size: 0.88em; }
        table.table th { background: #f0f0f0; }
        pre { background: #f7f7f7; padding: 1em; overflow-x: auto;
              border-left: 3px solid #4363d8; font-size: 0.85em; }
        img { border: 1px solid #ddd; margin: 0.5em 0; max-width: 100%;
              height: auto; }
        .interp { background: #eef4ff; padding: 0.6em 1em;
                  border-left: 3px solid #4363d8; margin: 0.6em 0;
                  font-style: italic; }
        .warn { background: #fff4e5; padding: 0.6em 1em;
                border-left: 3px solid #f58231; margin: 0.6em 0;
                font-size: 0.9em; }
        details summary { cursor: pointer; color: #4363d8;
                          font-size: 0.9em; }
    </style>
    """
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Module 16.5 — IBD Community Detection Enhanced</title>"
        f"{style}</head><body>" + "\n".join(sections) + "</body></html>"
    )
    (out_dir / NAME_HTML_REPORT).write_text(html, encoding="utf-8")
    LOG.info("Wrote HTML report to %s", out_dir / NAME_HTML_REPORT)


# ---------------------------------------------------------------------------
# Stage drivers
# ---------------------------------------------------------------------------

def _configure_threads(nthreads: int) -> None:
    for env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(env, str(nthreads))


def do_build_graph(args, inputs: M14Paths, out_dir: Path
                    ) -> tuple[sp.csr_matrix, ig.Graph, list[str],
                               pd.DataFrame]:
    """Construye el grafo ponderado a partir de las salidas de M14."""
    samples = load_individuals(inputs.individual_summary)
    pair_summary = load_pair_summary(inputs.pair_summary)
    seg_summary = None
    # Segments streaming is required whenever we need per-pair statistics
    # that are not pre-computed in Module 14's pair_sharing_summary.tsv:
    #  - n_shared_variants_total  (for --edge-weight-transform=n_shared_variants)
    #  - max_segment_bp           (for --min-max-segment-bp)
    needs_stream = (
        args.edge_weight_transform in ("n_shared_variants",)
        or int(getattr(args, "min_max_segment_bp", 0)) > 0
    )
    if needs_stream:
        seg_summary = load_segments_aggregated(
            inputs.segments, chunk_rows=args.segments_chunk_rows,
        )
    pair_w = aggregate_pair_weights(
        pair_summary, seg_summary,
        args.edge_weight_transform,
        min_max_segment_bp=int(getattr(args, "min_max_segment_bp", 0)),
    )
    S, _ = build_sparse_matrix(pair_w, samples,
                                min_edge_bp=args.min_edge_bp)
    g = sparse_to_igraph(S, samples)
    save_graph(out_dir, S=S, g=g, samples=samples,
                pair_df_with_weight=pair_w,
                min_edge_bp=args.min_edge_bp,
                weight_transform=args.edge_weight_transform,
                min_max_segment_bp=int(getattr(args, "min_max_segment_bp", 0)))
    return S, g, samples, pair_summary


def do_leiden(args, g: ig.Graph, samples: Sequence[str], out_dir: Path
               ) -> tuple[pd.DataFrame, pd.DataFrame, sp.csr_matrix | None]:
    """Ejecuta Leiden en varias resoluciones y guarda estabilidad y consenso."""
    (assignments_df, mod_df, consensus, _,
     memberships_by_res) = run_leiden_multiresolution(
        g,
        resolutions=args.leiden_resolutions,
        n_seeds=args.leiden_n_seeds,
        min_community_size=args.leiden_min_community_size,
        base_seed=args.seed,
        consensus_resolution=args.leiden_consensus_resolution,
    )
    # M16.5 additions: ARI stability + per-sample confidence.
    ari_df = compute_ari_multi_seed(memberships_by_res)
    confidence = None
    cons_col = f"community_res_{args.leiden_consensus_resolution:g}"
    if consensus is not None and cons_col in assignments_df.columns:
        confidence = compute_assignment_confidence(
            consensus, assignments_df[cons_col].to_numpy(dtype=np.int64)
        )
    save_leiden(out_dir, samples=samples,
                 assignments_df=assignments_df, modularity_df=mod_df,
                 consensus=consensus,
                 consensus_resolution=args.leiden_consensus_resolution,
                 ari_df=ari_df, confidence=confidence)
    return assignments_df, mod_df, consensus


def do_symnmf(args, S: sp.csr_matrix, samples: Sequence[str],
               out_dir: Path) -> dict[int, np.ndarray]:
    """Run Sym-NMF with cophenetic K-selection (M16.5 enhancement).

    Uses ``run_symnmf_cophenetic`` which:
      * Applies Laplacian normalisation if --laplacian-normalize is set
        (default True), fixing the degree-bias artefact that made the
        M16 soft memberships dominated by a single component.
      * Runs ``--nmf-inits`` restarts per K and builds a consensus
        matrix for cophenetic K-selection (Brunet et al. 2004).
    """
    H_by_k, err_df, coph_df = run_symnmf_cophenetic(
        S,
        k_values=args.nmf_k_values,
        n_inits=args.nmf_inits,
        max_iter=args.nmf_max_iter,
        tol=args.nmf_tol,
        base_seed=args.seed,
        laplacian=bool(args.laplacian_normalize),
        init_mode=str(args.nmf_init_mode),
        k_operational=int(args.nmf_operational_k),
        dispersion_threshold=float(args.nmf_dispersion_threshold),
        cophenetic_floor=float(args.nmf_cophenetic_floor),
        min_marginal_coph=float(args.nmf_min_marginal_coph),
    )
    save_symnmf(out_dir, samples=samples, H_by_k=H_by_k, err_df=err_df)
    if coph_df is not None and not coph_df.empty:
        coph_df.to_csv(out_dir / NAME_NMF_COPHENETIC, sep="\t", index=False,
                        float_format="%.4f")
        LOG.info("Saved cophenetic K diagnostics to %s",
                 out_dir / NAME_NMF_COPHENETIC)
    return H_by_k


def do_validate(args, pair_summary: pd.DataFrame,
                 assignments_df: pd.DataFrame,
                 out_dir: Path) -> pd.DataFrame:
    """Valida la separación intra e intercomunidad para las resoluciones elegidas."""
    vdf = validate_intra_vs_inter(pair_summary, assignments_df,
                                   args.leiden_resolutions)
    save_validation(out_dir, vdf)
    return vdf


def _assignments_to_map(assign_df: pd.DataFrame, col: str) -> np.ndarray:
    return assign_df[col].to_numpy(dtype=np.int64)


def do_plot(args, S: sp.csr_matrix, g: ig.Graph, samples: Sequence[str],
             assignments_df: pd.DataFrame,
             modularity_df: pd.DataFrame | None,
             H_by_k: dict[int, np.ndarray] | None,
             pair_summary: pd.DataFrame,
             out_dir: Path,
             seg_summary: pd.DataFrame | None = None,
             metadata_values: np.ndarray | None = None,
             metadata_name: str | None = None) -> None:
    """Produce the M16.5 minimal plot set (4 core figures):
      * UMAP-over-spectral network per resolution
      * Enhanced sharing heatmap (UPGMA within community + sidebars)
      * Sym-NMF STRUCTURE plot per K (reuses the M16 function — data
        is already improved by the Laplacian normalisation in
        ``run_symnmf_cophenetic``)
      * Kinship diagnostic scatter

    All other M16 plots (FR network, modularity curve, sankey,
    validation boxplot) are derivable from the numerical outputs and
    are therefore omitted to keep the report focussed on the biological
    questions.  They can be resurrected via the helper functions
    preserved above for reproducibility.
    """
    plots_dir = out_dir / PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Read confidence column if present (written by save_leiden when
    # available; loaded verbatim by load_leiden_assignments).
    confidence = None
    if "assignment_confidence" in assignments_df.columns:
        confidence = assignments_df["assignment_confidence"].to_numpy(
            dtype=np.float64
        )

    # Community annotations — see ``resolve_community_annotations`` for
    # the full three-tier precedence (explicit file > auto-derived with
    # disambiguation > 'C{id}' fallback) and the persisted TSV schema
    # users edit in place.
    community_annotations = resolve_community_annotations(
        args, assignments_df, out_dir)

    umap_3d = bool(getattr(args, "plot_umap_3d", False))
    adjust_labels = bool(getattr(args, "plot_adjust_labels", False))

    # 1) UMAP-over-spectral network plot per resolution.
    for res in args.leiden_resolutions:
        col = f"community_res_{res:g}"
        if col not in assignments_df.columns:
            continue
        memb = _assignments_to_map(assignments_df, col)
        png_3d_path = (plots_dir / f"network_umap_3d_res{res:g}.png"
                       if umap_3d else None)
        plot_network_umap(
            S, memb,
            plots_dir / f"network_umap_res{res:g}.png",
            confidence=confidence,
            metadata_values=metadata_values,
            metadata_name=metadata_name,
            max_nodes=args.plot_network_max_nodes,
            dpi=args.plot_dpi,
            width_in=args.plot_width_inches,
            height_in=args.plot_height_inches,
            export_pdf=args.plot_export_pdf,
            export_svg=bool(getattr(args, "plot_export_svg", False)),
            layout_seed=args.seed,
            label_min_size=int(args.plot_cluster_label_min_size),
            community_annotations=community_annotations,
            adjust_labels=adjust_labels,
            static_3d_png_path=png_3d_path,
        )

    # 2) Enhanced heatmap + NMF STRUCTURE at the validation resolution.
    col = f"community_res_{args.validation_resolution:g}"
    if col in assignments_df.columns:
        memb = _assignments_to_map(assignments_df, col)
        plot_sharing_heatmap_enhanced(
            S, memb,
            plots_dir / "sharing_heatmap_enhanced.png",
            confidence=confidence,
            metadata_values=metadata_values,
            metadata_name=metadata_name,
            max_nodes=args.plot_heatmap_max_nodes,
            dpi=args.plot_dpi, width_in=args.plot_width_inches,
            height_in=args.plot_height_inches,
            export_pdf=args.plot_export_pdf,
        )
        # 3) Sym-NMF structure plots (Laplacian-normalised upstream).
        if H_by_k:
            for k, H in H_by_k.items():
                plot_nmf_structure(
                    H, memb, plots_dir / f"nmf_structure_k{k}.png",
                    k=k, dpi=args.plot_dpi,
                    width_in=args.plot_width_inches,
                    height_in=args.plot_height_inches,
                    export_pdf=args.plot_export_pdf,
                )
        # Validation boxplot (kept — trivial cost, complements scatter).
        plot_validation_box(
            pair_summary, memb, samples,
            plots_dir / "validation_intra_vs_inter.png",
            dpi=args.plot_dpi, width_in=args.plot_width_inches,
            height_in=args.plot_height_inches,
            export_pdf=args.plot_export_pdf,
        )
    # M16.5 — kinship diagnostic scatter (requires segments summary).
    val_col = f"community_res_{args.validation_resolution:g}"
    if (seg_summary is not None and not seg_summary.empty
            and val_col in assignments_df.columns):
        memb_val = assignments_df[val_col].to_numpy(dtype=np.int64)
        plot_kinship_scatter(
            seg_summary, memb_val, samples,
            plots_dir / "kinship_scatter.png",
            kinship_segment_bp=int(args.kinship_segment_mb * 1_000_000),
            dpi=args.plot_dpi,
            width_in=args.plot_width_inches,
            height_in=args.plot_height_inches,
            export_pdf=args.plot_export_pdf,
        )


# ---------------------------------------------------------------------------
# Top-level main
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    """Coordina las etapas solicitadas y sus artefactos de salida."""
    args = build_parser().parse_args(argv)
    _configure_threads(args.threads)
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / PLOTS_DIR).mkdir(exist_ok=True)
    log_path = out_dir / "m16_5.log"
    _setup_logger(log_path)
    # Resolve the community-colour palette once at run start.  All plotting
    # helpers read the module-level _COMMUNITY_PALETTE.  64 hues comfortably
    # covers γ=3 Leiden runs (~60 communities) without recycling for HUSL.
    global _COMMUNITY_PALETTE, _PALETTE_NAME
    _COMMUNITY_PALETTE = _resolve_palette(args.plot_palette, n_colors=64)
    _PALETTE_NAME = str(args.plot_palette)
    LOG.info("Community palette: '%s' (%d colours)",
             _PALETTE_NAME, len(_COMMUNITY_PALETTE))
    LOG.info("Module 16.5 — mode=%s", args.mode)
    LOG.info("Input:  %s", in_dir)
    LOG.info("Output: %s", out_dir)

    inputs = M14Paths.from_dir(in_dir)
    summary: dict[str, Any] = {
        "mode": args.mode,
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "resolutions": list(args.leiden_resolutions),
        "nmf_k_values": list(args.nmf_k_values),
        "nmf_inits": int(args.nmf_inits),
        "nmf_init_mode": str(args.nmf_init_mode),
        "laplacian_normalize": bool(args.laplacian_normalize),
        "seed": int(args.seed),
        "min_edge_bp": int(args.min_edge_bp),
        "min_max_segment_bp": int(getattr(args, "min_max_segment_bp", 0)),
        "edge_weight_transform": args.edge_weight_transform,
        "kinship_segment_mb": float(args.kinship_segment_mb),
        "founder_intra_inter_ratio": float(args.founder_intra_inter_ratio),
        "founder_min_silhouette": float(args.founder_min_silhouette),
    }
    # Metadata may be absent; load_metadata_safe degrades silently.
    metadata_values: np.ndarray | None = None
    metadata_name: str | None = None
    metadata_warnings: list[str] = []
    # We need the sample order to align metadata → samples; this is
    # available after _ensure_graph.  Defer the actual load.

    S: sp.csr_matrix | None = None
    g: ig.Graph | None = None
    samples: list[str] | None = None
    pair_summary: pd.DataFrame | None = None
    assignments_df: pd.DataFrame | None = None
    modularity_df: pd.DataFrame | None = None
    H_by_k: dict[int, np.ndarray] | None = None
    validation_df: pd.DataFrame | None = None
    # Mutable holder so the `plot` stage can reuse the segments
    # aggregated DataFrame produced by the `validate` stage, avoiding a
    # second stream over all_pairwise_segments.tsv.gz.
    nonlocal_seg_summary: dict[str, pd.DataFrame] = {}

    def _ensure_graph() -> None:
        nonlocal S, g, samples, pair_summary, metadata_values, metadata_name
        if S is None:
            if (out_dir / NAME_GRAPH_MATRIX).exists():
                LOG.info("Loading cached graph from %s", out_dir)
                S, g, samples = load_graph(out_dir)
                pair_summary = load_pair_summary(inputs.pair_summary)
            else:
                S, g, samples, pair_summary = do_build_graph(args, inputs, out_dir)
        # Attach metadata as soon as samples are known (idempotent).
        if samples is not None and metadata_values is None:
            vals, col, wmsgs = load_metadata_safe(
                Path(args.metadata_file) if args.metadata_file else None,
                args.plot_color_by, samples,
            )
            metadata_values = vals
            metadata_name = col
            metadata_warnings.extend(wmsgs)

    def _ensure_assignments() -> None:
        nonlocal assignments_df, modularity_df
        if assignments_df is None:
            path = out_dir / NAME_LEIDEN_ASSIGN
            if path.exists():
                LOG.info("Loading cached Leiden assignments from %s", path)
                assignments_df = load_leiden_assignments(out_dir)
                mpath = out_dir / NAME_LEIDEN_MOD
                modularity_df = pd.read_csv(mpath, sep="\t") if mpath.exists() else None
            else:
                _ensure_graph()
                assignments_df, modularity_df, _ = do_leiden(
                    args, g, samples, out_dir
                )

    def _ensure_symnmf() -> None:
        nonlocal H_by_k
        if H_by_k is None:
            cached: dict[int, np.ndarray] = {}
            all_cached = True
            for k in args.nmf_k_values:
                df = load_symnmf(out_dir, k)
                if df is None:
                    all_cached = False
                    break
                cached[k] = df.drop(columns=["sample_id"]).to_numpy(
                    dtype=np.float64
                )
            if all_cached and cached:
                H_by_k = cached
            else:
                _ensure_graph()
                H_by_k = do_symnmf(args, S, samples, out_dir)

    try:
        if args.mode in ("build-graph", "all"):
            _ensure_graph()

        if args.mode in ("leiden", "all"):
            _ensure_graph()
            assignments_df, modularity_df, _ = do_leiden(
                args, g, samples, out_dir
            )

        if args.mode in ("symnmf", "all"):
            _ensure_graph()
            H_by_k = do_symnmf(args, S, samples, out_dir)

        if args.mode in ("validate", "all"):
            _ensure_graph()
            _ensure_assignments()
            validation_df = do_validate(args, pair_summary, assignments_df,
                                         out_dir)
            # Silhouette per community at the validation resolution.
            val_col = f"community_res_{args.validation_resolution:g}"
            sil_df = pd.DataFrame()
            if val_col in assignments_df.columns:
                memb_val = assignments_df[val_col].to_numpy(dtype=np.int64)
                sil_df = compute_silhouette_per_community(S, memb_val)
                if not sil_df.empty:
                    sil_df.to_csv(out_dir / NAME_COMMUNITY_SILHOUETTE,
                                   sep="\t", index=False,
                                   float_format="%.4f")
                    LOG.info("Saved silhouette per community to %s",
                             out_dir / NAME_COMMUNITY_SILHOUETTE)
            # Bio-detectors: cryptic kinship + founder effects.  Kinship
            # requires ``max_segment_bp``, which needs segments streaming.
            seg_summary = load_segments_aggregated(
                inputs.segments, chunk_rows=args.segments_chunk_rows,
            )
            if val_col in assignments_df.columns and not seg_summary.empty:
                kin_df = detect_cryptic_kinship(
                    seg_summary, assignments_df,
                    resolution=float(args.validation_resolution),
                    kinship_segment_bp=int(
                        args.kinship_segment_mb * 1_000_000
                    ),
                    max_community_size=int(args.kinship_max_size),
                )
                if not kin_df.empty:
                    kin_df.to_csv(out_dir / NAME_KINSHIP_CANDIDATES,
                                   sep="\t", index=False,
                                   float_format="%.0f")
                    LOG.info("Saved cryptic kinship candidates to %s "
                             "(%d rows)",
                             out_dir / NAME_KINSHIP_CANDIDATES, len(kin_df))
                founder_df = detect_founder_effects(
                    pair_summary, assignments_df, sil_df,
                    resolution=float(args.validation_resolution),
                    min_ratio=float(args.founder_intra_inter_ratio),
                    min_silhouette=float(args.founder_min_silhouette),
                    min_size=int(args.founder_min_size),
                    min_size_for_report=int(args.founder_min_size_for_report),
                )
                if not founder_df.empty:
                    founder_df.to_csv(out_dir / NAME_FOUNDER_CANDIDATES,
                                       sep="\t", index=False,
                                       float_format="%.4f")
                    LOG.info("Saved founder-effect candidates to %s "
                             "(%d communities scored)",
                             out_dir / NAME_FOUNDER_CANDIDATES,
                             len(founder_df))
                # Store segments summary on the outer frame so the plot
                # stage can reuse it without re-streaming.
                nonlocal_seg_summary["df"] = seg_summary
            # Optional metadata-driven outputs.
            if metadata_values is not None and metadata_name is not None:
                enrich_df = fisher_enrichment_community_vs_metadata(
                    assignments_df,
                    resolution=float(args.validation_resolution),
                    metadata_values=metadata_values,
                    metadata_name=metadata_name,
                )
                if not enrich_df.empty:
                    enrich_df.to_csv(out_dir / NAME_ENRICHMENT,
                                      sep="\t", index=False,
                                      float_format="%.4g")
                    LOG.info("Saved Fisher enrichment to %s (%d rows)",
                             out_dir / NAME_ENRICHMENT, len(enrich_df))
            labels_df = compute_community_labels(
                assignments_df,
                resolution=float(args.validation_resolution),
                metadata_values=metadata_values,
                metadata_name=metadata_name,
            )
            if not labels_df.empty:
                labels_df.to_csv(out_dir / NAME_COMMUNITY_LABELS,
                                  sep="\t", index=False,
                                  float_format="%.4f")
                LOG.info("Saved community labels to %s",
                         out_dir / NAME_COMMUNITY_LABELS)

        if args.mode in ("plot", "all"):
            _ensure_graph()
            _ensure_assignments()
            _ensure_symnmf()
            do_plot(args, S, g, samples, assignments_df, modularity_df,
                     H_by_k, pair_summary, out_dir,
                     seg_summary=nonlocal_seg_summary.get("df"),
                     metadata_values=metadata_values,
                     metadata_name=metadata_name)

        if args.mode in ("report", "all"):
            _ensure_graph()
            _ensure_assignments()
            if validation_df is None and (out_dir / NAME_VALIDATION).exists():
                validation_df = pd.read_csv(out_dir / NAME_VALIDATION, sep="\t")
            graph_summary = None
            if (out_dir / NAME_GRAPH_SUMMARY).exists():
                graph_summary = json.loads(
                    (out_dir / NAME_GRAPH_SUMMARY).read_text()
                )
            build_html_report(out_dir,
                               summary=summary,
                               graph_summary=graph_summary,
                               validation_df=validation_df,
                               modularity_df=modularity_df,
                               resolutions=args.leiden_resolutions,
                               metadata_available=(metadata_values is not None),
                               metadata_name=metadata_name,
                               warnings_list=metadata_warnings)

        # Final: write the global summary JSON for any 'all' / intermediate run.
        if assignments_df is not None:
            counts_by_res = {}
            for res in args.leiden_resolutions:
                col = f"community_res_{res:g}"
                if col in assignments_df.columns:
                    counts = assignments_df[col].astype(int).value_counts().to_dict()
                    counts_by_res[col] = {int(k): int(v) for k, v in counts.items()}
            summary["communities_per_resolution"] = counts_by_res
        if validation_df is not None:
            summary["validation"] = validation_df.to_dict(orient="records")
        (out_dir / NAME_GLOBAL_SUMMARY).write_text(
            json.dumps(summary, indent=2, default=str)
        )
    except Exception as exc:
        LOG.exception("Module 16.5 failed: %s", exc)
        raise
    LOG.info("Module 16.5 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
