#!/usr/bin/env python3
"""Fail-closed preflight for a continuous M14-minor structure analysis.

This program does not fit embeddings or predictive models.  It only asks whether
the planned grouped, out-of-fold analysis is identifiable with the frozen data.
Fold 3 is reserved and is never used as an endpoint or emitted in analytical
tables.  The upstream M14 matrix remains cohort-transductive; this preflight
does not claim prospective validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


DEV_FOLDS = (0, 1, 2, 4)
RESERVED_FOLD = 3
PRIMARY_PHI = 0.0442
SENSITIVITY_PHI = 0.0884
EXPECTED_SAMPLES = 2619
EXPECTED_PAIRS = 54522
EXPECTED_MIN_SEGMENT_BP = 1_000_000


class ContractError(RuntimeError):
    """Raised when an input violates a frozen preflight invariant."""


class DisjointSet:
    def __init__(self, nodes: Iterable[str]):
        self.parent = {str(node): str(node) for node in nodes}
        self.size = {str(node): 1 for node in nodes}

    def find(self, node: str) -> str:
        node = str(node)
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != node:
            nxt = self.parent[node]
            self.parent[node] = root
            node = nxt
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, low_memory=False)


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ContractError(f"{label}: missing columns {missing}")


def normalize_id_column(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    result = frame.copy()
    result[column] = result[column].astype(str).str.strip()
    if result[column].eq("").any() or result[column].eq("nan").any():
        raise ContractError(f"blank identifier in {column}")
    return result


def deduplicate_identical(frame: pd.DataFrame, key: str, label: str) -> pd.DataFrame:
    duplicates = frame[frame.duplicated(key, keep=False)]
    for value, group in duplicates.groupby(key, sort=False):
        if len(group.drop_duplicates()) != 1:
            raise ContractError(f"{label}: conflicting duplicate for {value}")
    return frame.drop_duplicates(key, keep="first").copy()


def load_and_validate_global(path: Path, pair_count: int) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "carrier_allele_mode_minor": payload.get("carrier_allele_mode") == "minor_allele",
        "n_samples_2619": int(payload.get("n_samples", -1)) == EXPECTED_SAMPLES,
        "summary_pairs_54522": int(payload.get("total_sharing_pairs", -1)) == EXPECTED_PAIRS,
        "file_pairs_match_summary": pair_count == int(payload.get("total_sharing_pairs", -1)),
        "min_segment_bp_1m": int(payload.get("parameters_used", {}).get("min_segment_bp", -1))
        == EXPECTED_MIN_SEGMENT_BP,
    }
    if not all(checks.values()):
        raise ContractError(f"M14-minor global contract failed: {checks}")
    return {"payload": payload, "checks": checks}


def component_labels(nodes: Iterable[str], kin: pd.DataFrame, threshold: float) -> dict[str, str]:
    node_set = {str(node) for node in nodes}
    dsu = DisjointSet(node_set)
    relevant = kin.loc[
        kin["ID1"].isin(node_set)
        & kin["ID2"].isin(node_set)
        & (pd.to_numeric(kin["kin"], errors="coerce") >= threshold)
    ]
    for row in relevant.itertuples(index=False):
        dsu.union(str(row.ID1), str(row.ID2))
    roots = {node: dsu.find(node) for node in sorted(node_set)}
    ordered_roots = {root: f"c{idx:05d}" for idx, root in enumerate(sorted(set(roots.values())))}
    return {node: ordered_roots[root] for node, root in roots.items()}


def projection_summary(
    pairs: pd.DataFrame,
    sample_table: pd.DataFrame,
    analysis_ids: set[str],
    primary_regions: list[str],
    anchor_scope: str,
) -> pd.DataFrame:
    """Measure EVAL connectivity using only EVAL-to-TRAIN edges.

    No EVAL-to-EVAL edge can make a sample projectable.  ``all_dnabr`` is the
    preregistered primary anchor scope; ``rht_eur`` is a reported sensitivity.
    """
    fold_by_id = sample_table.set_index("sample_id")["fold"].astype(int).to_dict()
    if RESERVED_FOLD in {fold_by_id.get(x) for x in analysis_ids}:
        raise ContractError("reserved fold leaked into analysis_ids")
    dev_ids = set(fold_by_id)
    if anchor_scope == "rht_eur":
        anchor_universe = analysis_ids
    elif anchor_scope == "all_dnabr":
        anchor_universe = dev_ids
    else:
        raise ValueError(anchor_scope)

    pair_nodes = pairs[["sample_a", "sample_b"]]
    records: list[dict] = []
    analysis = sample_table[sample_table["sample_id"].isin(analysis_ids)].copy()
    for fold in DEV_FOLDS:
        eval_table = analysis[analysis["fold"].astype(int).eq(fold)]
        eval_ids = set(eval_table["sample_id"])
        train_ids = {sid for sid in anchor_universe if fold_by_id.get(sid) != fold}
        connected: set[str] = set()
        left = pair_nodes["sample_a"].isin(eval_ids) & pair_nodes["sample_b"].isin(train_ids)
        right = pair_nodes["sample_b"].isin(eval_ids) & pair_nodes["sample_a"].isin(train_ids)
        connected.update(pair_nodes.loc[left, "sample_a"])
        connected.update(pair_nodes.loc[right, "sample_b"])
        for region in ["__ALL__", *primary_regions]:
            region_ids = eval_ids if region == "__ALL__" else set(
                eval_table.loc[eval_table["region"].eq(region), "sample_id"]
            )
            n_eval = len(region_ids)
            n_connected = len(region_ids & connected)
            records.append(
                {
                    "anchor_scope": anchor_scope,
                    "fold": fold,
                    "region": region,
                    "n_eval": n_eval,
                    "n_connected": n_connected,
                    "n_unprojectable": n_eval - n_connected,
                    "unprojectable_fraction": (n_eval - n_connected) / n_eval if n_eval else None,
                }
            )
    return pd.DataFrame.from_records(records)


def build_preflight(args: argparse.Namespace) -> tuple[dict, dict[str, pd.DataFrame]]:
    if args.primary_phi != PRIMARY_PHI or args.sensitivity_phi != SENSITIVITY_PHI:
        raise ContractError(
            "phi values are frozen because component columns were precomputed at 0.0442 and 0.0884"
        )
    paths = {
        "pairs": Path(args.pairs),
        "global_summary": Path(args.global_summary),
        "metadata": Path(args.metadata),
        "burden": Path(args.burden),
        "feature_store": Path(args.feature_store),
        "pcrelate": Path(args.pcrelate),
        "modeling_master": Path(args.modeling_master),
        "split_manifest": Path(args.split_manifest),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise ContractError(f"{label}: file not found: {path}")

    pairs = normalize_id_column(read_tsv(paths["pairs"]), "sample_a")
    pairs = normalize_id_column(pairs, "sample_b")
    require_columns(
        pairs,
        ["sample_a", "sample_b", "n_shared_variants_total", "total_shared_bp"],
        "pairs",
    )
    if pairs[["sample_a", "sample_b"]].duplicated().any():
        raise ContractError("pairs contains duplicated directed rows")
    canonical_pairs = pairs.apply(
        lambda row: tuple(sorted((row["sample_a"], row["sample_b"]))), axis=1
    )
    if canonical_pairs.duplicated().any():
        raise ContractError("pairs contains duplicated undirected rows")
    global_contract = load_and_validate_global(paths["global_summary"], len(pairs))
    expected_sample_ids = set(map(str, global_contract["payload"]["ordered_samples"]))
    if len(expected_sample_ids) != EXPECTED_SAMPLES:
        raise ContractError("global ordered_samples is not an exact unique 2619-sample set")
    if not (set(pairs["sample_a"]) | set(pairs["sample_b"])).issubset(expected_sample_ids):
        raise ContractError("pair endpoint is outside the frozen 2619-sample universe")

    metadata = normalize_id_column(read_tsv(paths["metadata"]), "ID")
    require_columns(
        metadata,
        ["ID", "Cohort", "Region", "State", "Autosomes_European_anc"],
        "metadata",
    )
    metadata_duplicate_ids = set(metadata.loc[metadata.duplicated("ID", keep=False), "ID"])
    if metadata_duplicate_ids != {"BB-COVL-397"}:
        raise ContractError(
            f"unexpected metadata duplicate set: {sorted(metadata_duplicate_ids)}"
        )
    metadata = deduplicate_identical(metadata, "ID", "metadata")
    metadata = metadata.rename(
        columns={
            "ID": "sample_id",
            "Cohort": "metadata_cohort",
            "Region": "metadata_region",
            "State": "metadata_state",
            "Autosomes_European_anc": "q_eur",
        }
    )

    burden = normalize_id_column(read_tsv(paths["burden"]), "sample_id")
    require_columns(
        burden,
        ["sample_id", "minor_m14_subset_carrier_rate", "minor_m14_subset_callable_sites"],
        "burden",
    )
    burden = deduplicate_identical(burden, "sample_id", "burden")
    if set(burden["sample_id"]) != expected_sample_ids:
        raise ContractError("burden sample universe differs from frozen M14-minor universe")

    feature = normalize_id_column(read_tsv(paths["feature_store"]), "sample_id")
    require_columns(feature, ["sample_id", "rare_gt_nonmissing_sites", "rare_missing_sites"], "feature")
    feature = deduplicate_identical(feature, "sample_id", "feature_store")
    if set(feature["sample_id"]) != expected_sample_ids:
        raise ContractError("feature-store sample universe differs from frozen M14-minor universe")

    kin = normalize_id_column(read_tsv(paths["pcrelate"]), "ID1")
    kin = normalize_id_column(kin, "ID2")
    require_columns(kin, ["ID1", "ID2", "kin"], "pcrelate")

    modeling = normalize_id_column(read_tsv(paths["modeling_master"]), "sample_id")
    require_columns(
        modeling,
        ["sample_id", "kinship_group_id_phi0442", "kinship_group_id_phi0884"],
        "modeling_master",
    )
    modeling = deduplicate_identical(modeling, "sample_id", "modeling_master")
    if set(modeling["sample_id"]) != expected_sample_ids:
        raise ContractError("modeling-master sample universe differs from frozen M14-minor universe")

    split = normalize_id_column(read_tsv(paths["split_manifest"]), "sample_id")
    require_columns(
        split, ["sample_id", "eligible", "fold", "split", "split_group_key"], "split_manifest"
    )
    split = deduplicate_identical(split, "sample_id", "split_manifest")
    if set(split["sample_id"]) != expected_sample_ids:
        raise ContractError("split-manifest sample universe differs from frozen M14-minor universe")
    split["fold_numeric"] = pd.to_numeric(split["fold"], errors="coerce")
    eligible = split["eligible"].str.lower().isin({"true", "1", "yes"})
    eligible_train = eligible & split["split"].eq("TRAIN")
    if split.loc[eligible_train, "fold_numeric"].isna().any():
        raise ContractError("eligible TRAIN row has missing or invalid fold")
    split.loc[split["fold_numeric"].notna(), "fold"] = split.loc[
        split["fold_numeric"].notna(), "fold_numeric"
    ].astype(int)
    dev = split.loc[eligible_train & split["fold_numeric"].isin(DEV_FOLDS)].copy()
    dev["fold"] = dev["fold_numeric"].astype(int)
    if dev["fold"].eq(RESERVED_FOLD).any():
        raise ContractError("reserved fold present in development rows")

    samples = (
        dev[["sample_id", "fold"]]
        .merge(metadata, on="sample_id", how="left", validate="one_to_one")
        .merge(
            burden[["sample_id", "minor_m14_subset_carrier_rate", "minor_m14_subset_callable_sites"]],
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            feature[["sample_id", "rare_gt_nonmissing_sites", "rare_missing_sites"]],
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            modeling[["sample_id", "kinship_group_id_phi0442", "kinship_group_id_phi0884"]],
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
    )
    samples["q_eur"] = pd.to_numeric(samples["q_eur"], errors="coerce")
    samples["region"] = samples["metadata_region"].fillna("UNKNOWN").str.strip().str.upper()
    samples.loc[samples["region"].eq(""), "region"] = "UNKNOWN"
    samples["cohort"] = samples["metadata_cohort"].fillna("").str.strip().str.upper()

    join_columns = [
        "metadata_cohort",
        "q_eur",
        "minor_m14_subset_carrier_rate",
        "minor_m14_subset_callable_sites",
        "rare_gt_nonmissing_sites",
        "kinship_group_id_phi0442",
        "kinship_group_id_phi0884",
    ]
    join_missing = {column: int(samples[column].isna().sum()) for column in join_columns}
    joins_pass = all(value == 0 for value in join_missing.values())

    pair_nodes = set(pairs["sample_a"]) | set(pairs["sample_b"])
    analysis_mask = samples["cohort"].eq("RHT") & samples["q_eur"].gt(args.q_eur_threshold)
    analysis = samples.loc[analysis_mask].copy()
    analysis["observed_m14"] = analysis["sample_id"].isin(pair_nodes)
    analysis_ids = set(analysis["sample_id"])

    tables: dict[str, pd.DataFrame] = {}
    region_summaries: list[pd.DataFrame] = []
    component_columns = {
        args.primary_phi: "kinship_group_id_phi0442",
        args.sensitivity_phi: "kinship_group_id_phi0884",
    }
    for phi, component_column in component_columns.items():
        temp = analysis.copy()
        temp["component_id"] = temp[component_column].astype(str)
        grouped = temp.groupby("region", dropna=False)
        summary = grouped.agg(
            n_people=("sample_id", "size"),
            n_components=("component_id", "nunique"),
            n_observed_m14=("observed_m14", "sum"),
        ).reset_index()
        largest = (
            temp.groupby(["region", "component_id"]).size().groupby(level=0).max().rename("largest_component")
        )
        summary = summary.merge(largest, on="region", how="left")
        summary["largest_component_fraction"] = summary["largest_component"] / summary["n_people"]
        summary["phi"] = phi
        region_summaries.append(summary)
    region_summary = pd.concat(region_summaries, ignore_index=True)
    tables["region_summary"] = region_summary

    primary_base = region_summary.loc[
        region_summary["phi"].eq(args.primary_phi)
        & ~region_summary["region"].isin({"UNKNOWN", "NAN"})
        & region_summary["n_people"].ge(args.min_region_people)
    ]
    primary_regions = sorted(primary_base["region"].tolist())

    fold_region = (
        analysis[analysis["region"].isin(primary_regions)]
        .groupby(["fold", "region"])
        .size()
        .rename("n_people")
        .reset_index()
    )
    tables["fold_region_counts"] = fold_region

    projection_all = projection_summary(pairs, samples, analysis_ids, primary_regions, "all_dnabr")
    projection_rht = projection_summary(pairs, samples, analysis_ids, primary_regions, "rht_eur")
    projection = pd.concat([projection_all, projection_rht], ignore_index=True)
    tables["projection_summary"] = projection

    primary_rows = region_summary[
        region_summary["region"].isin(primary_regions)
        & region_summary["phi"].isin([args.primary_phi, args.sensitivity_phi])
    ]
    region_component_support = bool(
        len(primary_regions) >= 3
        and primary_rows.groupby("phi")["region"].nunique().eq(len(primary_regions)).all()
        and primary_rows["n_components"].ge(args.min_region_components).all()
    )
    component_concentration = bool(
        not primary_rows.empty
        and primary_rows["largest_component_fraction"].le(args.max_component_class_fraction).all()
    )
    all_folds_have_classes = bool(
        fold_region.groupby("fold")["region"].nunique().reindex(DEV_FOLDS, fill_value=0).eq(len(primary_regions)).all()
    )
    primary_projection_rows = projection_all[projection_all["region"].eq("__ALL__")]
    projectable_overall = bool(
        len(primary_projection_rows) == len(DEV_FOLDS)
        and primary_projection_rows["unprojectable_fraction"].notna().all()
        and primary_projection_rows["unprojectable_fraction"].le(args.max_unprojectable_fraction).all()
    )
    class_projection_rows = projection_all[projection_all["region"].isin(primary_regions)]
    projectable_each_class_diagnostic = bool(
        len(class_projection_rows) == len(DEV_FOLDS) * len(primary_regions)
        and class_projection_rows["unprojectable_fraction"].notna().all()
        and class_projection_rows["unprojectable_fraction"].le(args.max_unprojectable_fraction).all()
    )
    projection_oof_region = (
        class_projection_rows.groupby("region", as_index=False)
        .agg(n_eval=("n_eval", "sum"), n_connected=("n_connected", "sum"), n_unprojectable=("n_unprojectable", "sum"))
    )
    projection_oof_region["unprojectable_fraction"] = (
        projection_oof_region["n_unprojectable"] / projection_oof_region["n_eval"]
    )
    tables["projection_oof_region"] = projection_oof_region

    used_pair_mask = pairs["sample_a"].isin(set(samples["sample_id"])) & pairs["sample_b"].isin(
        set(samples["sample_id"])
    )
    used_pairs = pairs.loc[used_pair_mask]
    reserved_ids = set(split.loc[split["fold_numeric"].eq(RESERVED_FOLD), "sample_id"])
    reserved_endpoints_used = int(
        used_pairs["sample_a"].isin(reserved_ids).sum() + used_pairs["sample_b"].isin(reserved_ids).sum()
    )

    split_groups = dev[["sample_id", "split_group_key"]].merge(
        modeling[["sample_id", "kinship_group_id_phi0442"]], on="sample_id", validate="one_to_one"
    )
    # Component labels may use different canonical roots; compare partitions,
    # not the literal identifiers.
    split_to_model = split_groups.groupby("split_group_key")["kinship_group_id_phi0442"].nunique()
    model_to_split = split_groups.groupby("kinship_group_id_phi0442")["split_group_key"].nunique()
    frozen_groups_match = bool(split_to_model.le(1).all() and model_to_split.le(1).all())
    fold_by_id = samples.set_index("sample_id")["fold"].astype(int).to_dict()
    kin_numeric = pd.to_numeric(kin["kin"], errors="coerce")
    kin_dev = kin.loc[kin["ID1"].isin(fold_by_id) & kin["ID2"].isin(fold_by_id)].copy()
    kin_dev["kin_numeric"] = kin_numeric.loc[kin_dev.index]
    kin_dev["fold1"] = kin_dev["ID1"].map(fold_by_id)
    kin_dev["fold2"] = kin_dev["ID2"].map(fold_by_id)
    pcrelate_crossfold_edges = {
        str(phi): int(
            (kin_dev["kin_numeric"].ge(phi) & kin_dev["fold1"].ne(kin_dev["fold2"])).sum()
        )
        for phi in (args.primary_phi, args.sensitivity_phi)
    }

    gates = {
        "global_m14_minor_contract": all(global_contract["checks"].values()),
        "joins_complete": joins_pass,
        "frozen_primary_groups_match_split": frozen_groups_match,
        "pcrelate_no_crossfold_edges_both_phi": all(
            value == 0 for value in pcrelate_crossfold_edges.values()
        ),
        "at_least_three_primary_regions": len(primary_regions) >= 3,
        "region_component_support_both_phi": region_component_support,
        "largest_component_at_most_half_class": component_concentration,
        "all_dev_folds_contain_primary_regions": all_folds_have_classes,
        "all_dnabr_anchor_unprojectable_at_most_20pct_overall": projectable_overall,
    }
    decision = "PASS_PREFLIGHT_IDENTIFIABLE" if all(gates.values()) else "STOP_PREFLIGHT_NOT_IDENTIFIABLE"

    report = {
        "schema_version": "p1a-preflight-v1",
        "code_commit": args.code_commit,
        "decision": decision,
        "scope": {
            "cohort": "RHT",
            "q_eur_strictly_greater_than": args.q_eur_threshold,
            "development_folds": list(DEV_FOLDS),
            "reserved_fold": RESERVED_FOLD,
            "primary_anchor_scope": "all_dnabr_train_other_folds",
            "sensitivity_anchor_scope": "rht_qeur_train_other_folds",
            "interpretation": "descriptive_DEV_not_prospective_validation",
        },
        "thresholds": {
            "primary_phi": args.primary_phi,
            "sensitivity_phi": args.sensitivity_phi,
            "min_region_people": args.min_region_people,
            "min_region_components": args.min_region_components,
            "max_component_class_fraction": args.max_component_class_fraction,
            "max_unprojectable_fraction": args.max_unprojectable_fraction,
        },
        "counts": {
            "development_samples": int(len(samples)),
            "analysis_samples": int(len(analysis)),
            "analysis_observed_m14": int(analysis["observed_m14"].sum()),
            "primary_regions": primary_regions,
            "reserved_endpoints_used": reserved_endpoints_used,
            "pcrelate_crossfold_edges": pcrelate_crossfold_edges,
        },
        "joins_missing": join_missing,
        "gates": gates,
        "diagnostics_not_load_bearing": {
            "all_dnabr_anchor_unprojectable_at_most_20pct_each_region": (
                projectable_each_class_diagnostic
            ),
            "reserved_fold_endpoints_emitted": reserved_endpoints_used,
            "note": (
                "The per-Region 20% rule was examined only after the global preflight and is "
                "reported as a post-hoc diagnostic, not used to change the decision."
            ),
        },
        "input_sha256": {label: sha256_file(path) for label, path in paths.items()},
        "upstream_caveat": (
            "M14-minor orientation and segment construction used the full 2619-person cohort. "
            "Excluding fold 3 endpoints prevents downstream reuse but does not make the upstream "
            "representation prospectively inductive."
        ),
        "callability_caveat": (
            "Available callable-site counts are marginal per individual, not exact pairwise callable "
            "intersections; no pairwise callability correction is claimed in this preflight."
        ),
    }
    return report, tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--global-summary", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--burden", required=True)
    parser.add_argument("--feature-store", required=True)
    parser.add_argument("--pcrelate", required=True)
    parser.add_argument("--modeling-master", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--primary-phi", type=float, default=PRIMARY_PHI)
    parser.add_argument("--sensitivity-phi", type=float, default=SENSITIVITY_PHI)
    parser.add_argument("--q-eur-threshold", type=float, default=0.50)
    parser.add_argument("--min-region-people", type=int, default=30)
    parser.add_argument("--min-region-components", type=int, default=5)
    parser.add_argument("--max-component-class-fraction", type=float, default=0.50)
    parser.add_argument("--max-unprojectable-fraction", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    report, tables = build_preflight(args)
    (output_dir / "preflight_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
