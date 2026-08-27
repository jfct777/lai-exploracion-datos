#!/usr/bin/env python3
"""Build an immutable FIT/VALID factor manifest for one M34 mosaic root."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ANCESTRIES = ("AFR", "EUR", "NAM")
ROOT_SEEDS = {
    "R0": {"FIT": 1439610605, "VALID": 1702577247},
    "R1": {"FIT": 667875703, "VALID": 513710823},
    "R2": {"FIT": 348301061, "VALID": 1179260632},
}
FACTOR_NAMES = ("selected_variant", "target", "reference", "f0", "marker_cm", "truth")
SELECTED_MEMBERS = {"locus_id", "chrom", "pos", "ref", "alt", "cM"}
TARGET_MEMBERS = {"sample_key_sha256", "locus_id", "minor_dosage", "observed_mask"}
REFERENCE_MEMBERS = {
    "ancestry", "locus_id", "minor_ac", "callable_an", "minor_af",
    "observed_mask", "no_support",
}
F0_MEMBERS = {
    "sample_key_sha256", "marker_chrom", "marker_pos", "marker_ref", "marker_alt", "F0",
}
TRUTH_MEMBERS = {"sample_key_sha256", "marker_pos", "labels"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def serialized_sha256(value: Any) -> str:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid receipt: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"receipt is not an object: {path}")
    return payload


def descriptor(path: Path, base: Path | None = None) -> dict[str, Any]:
    result = {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if base is not None:
        relative = Path(os.path.relpath(path.resolve(), base.resolve()))
        require(not relative.is_absolute() and ".." not in relative.parts,
                f"input is outside the co-staged manifest bundle: {path.name}")
        result["relative_path"] = relative.as_posix()
    return result


def _output_by_name(receipt: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    outputs = receipt.get("outputs")
    require(isinstance(outputs, (list, dict)), "receipt outputs are missing")
    if isinstance(outputs, dict):
        require(name in outputs and isinstance(outputs[name], dict),
                f"receipt output is missing: {name}")
        return outputs[name]
    matches = [row for row in outputs
               if isinstance(row, dict) and row.get("name") == name]
    require(len(matches) == 1, f"receipt output is missing or duplicated: {name}")
    return matches[0]


def _same_descriptor(observed: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    require(observed.get("sha256") == expected.get("sha256"), f"{label} SHA-256 differs")
    if "bytes" in observed and "bytes" in expected:
        require(observed["bytes"] == expected["bytes"], f"{label} byte count differs")


def _load_npz(path: Path, members: set[str],
              excluded: set[str] | None = None) -> dict[str, np.ndarray]:
    require(path.is_file() and not path.is_symlink(), f"invalid factor: {path}")
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == members, f"NPZ member inventory differs: {path.name}")
        return {
            name: np.ascontiguousarray(archive[name])
            for name in members - (excluded or set())
        }


def _npy_shape_in_npz(path: Path, member: str) -> tuple[tuple[int, ...], np.dtype]:
    """Read one NPY header without loading its potentially large tensor."""
    with zipfile.ZipFile(path, "r") as archive:
        member_name = f"{member}.npy"
        require(member_name in archive.namelist(), f"NPZ member is missing: {member}")
        with archive.open(member_name, "r") as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
            else:
                raise ValueError(f"unsupported NPY header version for {member}: {version}")
    require(not fortran, f"Fortran-ordered NPZ member is forbidden: {member}")
    return tuple(shape), np.dtype(dtype)


@dataclass(frozen=True)
class SplitPaths:
    selected_variant: Path
    target: Path
    reference: Path
    f0: Path
    marker_cm: Path
    truth: Path
    mosaic_receipt: Path
    bridge_receipt: Path
    flare_receipt: Path


@dataclass(frozen=True)
class SplitAudit:
    split: str
    paths: Mapping[str, Path]
    sample_keys: np.ndarray
    selected_axis_sha256: str
    reference_sha256: str
    marker_axis_sha256: str
    selected_loci: int
    markers: int
    people: int
    provenance: Mapping[str, Any]
    source_sha256: Mapping[str, str]


def _split_contract(root: str, fit_people: int, valid_people: int) -> dict[str, dict[str, Any]]:
    require(root in ROOT_SEEDS, "M34 manifest root is not declared")
    require(fit_people > 0 and valid_people > 0, "M34 split sizes must be positive")
    return {
        "FIT": {
            "donor_role": "SOURCE_VALID", "people": fit_people,
            "prefix": f"M34_{root}_FIT", "seed": ROOT_SEEDS[root]["FIT"],
        },
        "VALID": {
            "donor_role": "SOURCE_TEST", "people": valid_people,
            "prefix": f"M34_{root}_VALID", "seed": ROOT_SEEDS[root]["VALID"],
        },
    }


def _validate_mosaic(
    split: str, receipt: Mapping[str, Any], split_contract: Mapping[str, Mapping[str, Any]],
) -> None:
    expected = split_contract[split]
    opposite_role = "SOURCE_TEST" if split == "FIT" else "SOURCE_VALID"
    require(receipt.get("stage") == "M34_NAM_EXPLORATORY_MOSAICS" and
            receipt.get("decision") == "PASS_EXPLORATORY_MOSAICS_WITH_LOCAL_TRUTH",
            f"{split} mosaic receipt identity differs")
    scope = receipt.get("scope", {})
    require(scope.get("exploratory_only") is True and
            scope.get("confirmatory_validation") is False and
            scope.get("generalizes_to_dnabr") is False,
            f"{split} mosaic claim level differs")
    parameters = receipt.get("parameters", {})
    mixture = parameters.get("mixture_proportions", {})
    require(str(parameters.get("chromosome")).removeprefix("chr") == "22",
            f"{split} mosaic chromosome differs")
    require(parameters.get("rotation") == 0 and
            parameters.get("target_prefix") == expected["prefix"],
            f"{split} mosaic root or donor partition differs")
    require(parameters.get("target_individuals") == expected["people"],
            f"{split} mosaic person count differs")
    require(parameters.get("transition_parameterization") == "pulse_generations" and
            math.isclose(float(parameters.get("transitions_per_morgan", -1)), 12.0,
                         rel_tol=0.0, abs_tol=1e-12),
            f"{split} mosaic generation parameter differs")
    require(tuple(mixture) == ANCESTRIES and all(math.isclose(
        float(mixture[name]), value, rel_tol=0.0, abs_tol=1e-12,
    ) for name, value in zip(ANCESTRIES, (0.25, 0.60, 0.15))),
            f"{split} mosaic proportions differ")
    roles = receipt.get("role_audit", {})
    require(roles.get("donor_role") == expected["donor_role"] and
            set(roles.get("forbidden_roles", ())) == {"REF_TRAIN", opposite_role} and
            roles.get("atomic_units_crossing_forbidden_roles") == 0 and
            roles.get("ref_train_used_as_donor") is False and
            roles.get("source_test_used_as_donor") is (split == "VALID") and
            roles.get("unit_partition") == "all" and
            roles.get("unit_partition_rotation") == 0 and
            roles.get("selected_atomic_units") == 15,
            f"{split} mosaic donor role differs")
    expected_people_by_ancestry = (
        {"AFR": 141, "EUR": 151, "NAM": 14}
        if split == "FIT" else {"AFR": 135, "EUR": 151, "NAM": 11}
    )
    require(roles.get("selected_people") == sum(expected_people_by_ancestry.values()) and
            roles.get("donor_people_by_ancestry") == expected_people_by_ancestry and
            roles.get("donor_atomic_units_by_ancestry") == {"AFR": 5, "EUR": 8, "NAM": 2},
            f"{split} donor unit audit differs")
    require(receipt.get("counts", {}).get("target_individuals") == expected["people"],
            f"{split} mosaic count audit differs")
    require(parameters.get("seed") == expected["seed"],
            f"{split} mosaic seed differs from the selected root")


def _validate_bridge(split: str, mosaic: Mapping[str, Any], bridge: Mapping[str, Any],
                     files: Mapping[str, Path],
                     split_contract: Mapping[str, Mapping[str, Any]]) -> None:
    role = split_contract[split]["donor_role"]
    require(bridge.get("schema_version") == "m34_panel_factors_receipt_v1" and
            bridge.get("stage") == "M34_EXPLORATORY_VCF_TO_FACTORS_BRIDGE" and
            bridge.get("decision") == f"PASS_EXPLORATORY_PANEL_FACTORS_{role}_MOSAICS",
            f"{split} bridge receipt identity differs")
    scope = bridge.get("scope", {})
    require(scope.get("exploratory_only") is True and
            scope.get("confirmatory_validation") is False and
            scope.get("generalizes_to_dnabr") is False,
            f"{split} bridge claim level differs")
    roles = bridge.get("roles", {})
    require(roles.get("reference_role") == "REF_TRAIN" and
            roles.get("frequency_role") == "REF_TRAIN" and
            roles.get("mosaic_donor_role_upstream") == role and
            roles.get("source_valid_panel_genotypes_opened") is False and
            roles.get("source_test_panel_genotypes_opened") is False and
            roles.get("source_test_open") is (split == "VALID") and
            roles.get("source_test_mosaic_donors_upstream") is (split == "VALID"),
            f"{split} bridge role firewall differs")
    counts = bridge.get("counts", {})
    require(counts.get("reference_samples") == 753 and
            counts.get("reference_samples_by_ancestry") ==
            {"AFR": 341, "EUR": 387, "NAM": 25} and
            counts.get("split_biological_roles") ==
            {"REF_TRAIN": 753, "SOURCE_VALID": 306,
             "SOURCE_TEST": 297, "DISCOVERY": 149},
            f"{split} bridge role counts differ")
    mosaic_output = _output_by_name(mosaic, "m34_target.chr22.vcf.gz")
    _same_descriptor(bridge.get("inputs", {}).get("mosaic_vcf", {}), mosaic_output,
                     f"{split} mosaic-to-bridge binding")
    expected_names = {
        "selected_variant": "m34_selected_loci.npz",
        "target": "m34_target_rare_diploid.npz",
        "reference": "m34_reference_rare_summary.npz",
    }
    for logical, output_name in expected_names.items():
        _same_descriptor(descriptor(files[logical]), _output_by_name(bridge, output_name),
                         f"{split} bridge {logical}")
    require(counts.get("target_samples") == split_contract[split]["people"],
            f"{split} bridge target count differs")

    mosaic_inputs = mosaic.get("inputs", {})
    bridge_inputs = bridge.get("inputs", {})
    for mosaic_name, bridge_name in (
        ("phased_vcf", "panel_vcf"),
        ("split_tsv", "split_tsv"),
        ("genetic_map", "genetic_map"),
    ):
        require(mosaic_inputs.get(mosaic_name, {}).get("sha256") ==
                bridge_inputs.get(bridge_name, {}).get("sha256"),
                f"{split} mosaic-to-bridge {bridge_name} source hash differs")


def _validate_flare(split: str, bridge: Mapping[str, Any], flare: Mapping[str, Any],
                    marker_count: int,
                    split_contract: Mapping[str, Mapping[str, Any]]) -> None:
    require(flare.get("schema_version") == "1.0.0" and
            flare.get("stage") == "M34_AFR_EUR_NAM_FLARE" and
            flare.get("status") == "PASS_TRUTH_BLIND_FLARE" and
            flare.get("claim_level") == "exploratory",
            f"{split} FLARE receipt identity differs")
    require(str(flare.get("chromosome")).removeprefix("chr") == "22" and
            tuple(flare.get("ancestry_names", ())) == ANCESTRIES,
            f"{split} FLARE axes differ")
    require(flare.get("truth_argument_available") is False and
            flare.get("truth_accessed") is False and
            flare.get("scoring_performed") is False and
            flare.get("preflight_only") is False,
            f"{split} FLARE run was not truth-blind")
    expected_parameters = {
        "array": False, "probs": True, "em": True,
        "min-mac": 1, "min-maf": 0.0, "gen": 12.0,
        "update-p": False, "panel-probs": False,
        "seed": 3401103, "nthreads": 4,
    }
    require(flare.get("parameters") == expected_parameters,
            f"{split} FLARE parameters differ from the frozen contract")
    shape = flare.get("shape", {})
    require(shape.get("target_sample_count") == split_contract[split]["people"] and
            shape.get("marker_count") == marker_count,
            f"{split} FLARE dimensions differ")
    observed = flare.get("input_sha256", {})
    for flare_name, bridge_name in (
        ("reference_vcf", "m34_ref_train.chr22.vcf.gz"),
        ("target_vcf", "m34_target.chr22.vcf.gz"),
        ("sample_map", "m34_ref_train.sample_map.tsv"),
    ):
        require(observed.get(flare_name) == _output_by_name(bridge, bridge_name).get("sha256"),
                f"{split} bridge-to-FLARE {flare_name} SHA-256 differs")
    require(observed.get("genetic_map") ==
            bridge.get("inputs", {}).get("genetic_map", {}).get("sha256"),
            f"{split} bridge-to-FLARE genetic map SHA-256 differs")
    ancestry_audit = flare.get("ancestry_vcf_audit")
    require(isinstance(ancestry_audit, dict) and
            ancestry_audit.get("sample_count") == split_contract[split]["people"] and
            ancestry_audit.get("marker_count") == marker_count and
            isinstance(ancestry_audit.get("sha256"), str) and
            len(ancestry_audit["sha256"]) == 64,
            f"{split} FLARE ancestry output audit differs")


def validate_split(
    split: str, inputs: SplitPaths,
    split_contract: Mapping[str, Mapping[str, Any]],
) -> SplitAudit:
    expected = split_contract[split]
    all_paths = {name: getattr(inputs, name) for name in (*FACTOR_NAMES,
                 "mosaic_receipt", "bridge_receipt", "flare_receipt")}
    require(all(path.is_file() and not path.is_symlink() for path in all_paths.values()),
            f"{split} has a missing or symbolic input")
    require(len(set(path.resolve() for path in all_paths.values())) == len(all_paths),
            f"{split} logical inputs must be distinct files")

    selected = _load_npz(inputs.selected_variant, SELECTED_MEMBERS)
    target = _load_npz(inputs.target, TARGET_MEMBERS)
    reference = _load_npz(inputs.reference, REFERENCE_MEMBERS)
    f0 = _load_npz(inputs.f0, F0_MEMBERS, {"F0"})
    f0_shape, f0_dtype = _npy_shape_in_npz(inputs.f0, "F0")
    marker = _load_npz(inputs.marker_cm, {"marker_cM"})["marker_cM"]
    truth = _load_npz(inputs.truth, TRUTH_MEMBERS)

    people = expected["people"]
    loci = len(selected["locus_id"])
    markers = len(f0["marker_pos"])
    require(people > 0 and loci > 0 and markers > 0, f"{split} factor axes are empty")
    require(selected["locus_id"].shape == (loci,) and
            all(selected[name].shape == (loci,) for name in ("chrom", "pos", "ref", "alt", "cM")) and
            np.all(selected["chrom"] == 22) and np.all(selected["pos"][:-1] < selected["pos"][1:]),
            f"{split} selected-locus axis differs")
    require(target["sample_key_sha256"].shape == (people,) and
            target["sample_key_sha256"].dtype == np.dtype("|S64") and
            len(set(target["sample_key_sha256"].tolist())) == people and
            np.array_equal(target["locus_id"], selected["locus_id"]) and
            target["minor_dosage"].shape == target["observed_mask"].shape == (people, loci),
            f"{split} target factor axes differ")
    require(tuple(value.decode("ascii") for value in reference["ancestry"]) == ANCESTRIES and
            np.array_equal(reference["locus_id"], selected["locus_id"]) and
            all(reference[name].shape == (3, loci) for name in
                ("minor_ac", "callable_an", "minor_af", "observed_mask", "no_support")),
            f"{split} reference factor axes differ")
    require(f0["sample_key_sha256"].shape == (people,) and
            np.array_equal(f0["sample_key_sha256"], target["sample_key_sha256"]) and
            f0_shape == (people, 2, markers, 3) and f0_dtype == np.dtype("<f4") and
            all(f0[name].shape == (markers,) for name in
                ("marker_chrom", "marker_pos", "marker_ref", "marker_alt")) and
            np.all(f0["marker_chrom"] == 22) and np.all(f0["marker_pos"][:-1] < f0["marker_pos"][1:]),
            f"{split} F0 axes differ")
    require(marker.dtype == np.dtype("<f8") and marker.shape == (markers,) and
            np.all(np.isfinite(marker)) and np.all(marker[:-1] <= marker[1:]),
            f"{split} marker cM axis differs")
    require(np.array_equal(truth["sample_key_sha256"], target["sample_key_sha256"]) and
            np.array_equal(truth["marker_pos"], f0["marker_pos"]) and
            truth["labels"].shape == (people, 2, markers) and
            np.all((truth["labels"] >= 0) & (truth["labels"] < 3)),
            f"{split} truth axes differ")

    mosaic_receipt = strict_json(inputs.mosaic_receipt)
    bridge_receipt = strict_json(inputs.bridge_receipt)
    flare_receipt = strict_json(inputs.flare_receipt)
    _validate_mosaic(split, mosaic_receipt, split_contract)
    _validate_bridge(split, mosaic_receipt, bridge_receipt, all_paths, split_contract)
    _validate_flare(split, bridge_receipt, flare_receipt, markers, split_contract)
    source_sha256 = {
        "phased_panel": mosaic_receipt["inputs"]["phased_vcf"]["sha256"],
        "role_split": mosaic_receipt["inputs"]["split_tsv"]["sha256"],
        "genetic_map": mosaic_receipt["inputs"]["genetic_map"]["sha256"],
    }

    marker_axis = np.rec.fromarrays(
        [f0["marker_chrom"], f0["marker_pos"], f0["marker_ref"], f0["marker_alt"], marker],
        names="chrom,pos,ref,alt,cM",
    )
    return SplitAudit(
        split=split,
        paths=all_paths,
        sample_keys=target["sample_key_sha256"],
        selected_axis_sha256=array_sha256(np.rec.fromarrays(
            [selected["locus_id"], selected["chrom"], selected["pos"],
             selected["ref"], selected["alt"], selected["cM"]],
            names="locus_id,chrom,pos,ref,alt,cM",
        )),
        reference_sha256=sha256_file(inputs.reference),
        marker_axis_sha256=array_sha256(marker_axis),
        selected_loci=loci,
        markers=markers,
        people=people,
        provenance={
            "mosaic_receipt": descriptor(inputs.mosaic_receipt),
            "bridge_receipt": descriptor(inputs.bridge_receipt),
            "flare_receipt": descriptor(inputs.flare_receipt),
            "flare_ancestry_vcf_sha256": flare_receipt["ancestry_vcf_audit"]["sha256"],
        },
        source_sha256=source_sha256,
    )


def build(
    fit: SplitPaths, valid: SplitPaths, manifest_directory: Path,
    root: str = "R0", fit_people: int = 24, valid_people: int = 8,
) -> tuple[dict[str, Any], dict[str, Any]]:
    split_contract = _split_contract(root, fit_people, valid_people)
    audits = {
        "FIT": validate_split("FIT", fit, split_contract),
        "VALID": validate_split("VALID", valid, split_contract),
    }
    require(set(audits["FIT"].sample_keys.tolist()).isdisjoint(
        audits["VALID"].sample_keys.tolist()), "FIT and VALID sample axes overlap")
    require(audits["FIT"].selected_axis_sha256 == audits["VALID"].selected_axis_sha256,
            "FIT and VALID selected-locus axes differ")
    require(audits["FIT"].reference_sha256 == audits["VALID"].reference_sha256,
            "FIT and VALID reference factors differ")
    require(audits["FIT"].marker_axis_sha256 == audits["VALID"].marker_axis_sha256,
            "FIT and VALID FLARE marker axes differ")
    require(audits["FIT"].source_sha256 == audits["VALID"].source_sha256,
            "FIT and VALID panel, split or genetic-map sources differ")

    def relative_path(path: Path) -> str:
        relative = Path(os.path.relpath(path.resolve(), manifest_directory.resolve()))
        require(not relative.is_absolute() and ".." not in relative.parts,
                f"factor is outside the co-staged manifest bundle: {path.name}")
        return relative.as_posix()

    manifest = {
        "schema_version": "1.0.0",
        "ancestry_names": list(ANCESTRIES),
        "haplotypes": 2,
        "rotation": root,
        "splits": {
            split: [{name: relative_path(audits[split].paths[name]) for name in FACTOR_NAMES}]
            for split in ("FIT", "VALID")
        },
    }
    receipt = {
        "schema_version": "1.0.0",
        "stage": "M34_BUILD_FACTORIZED_MANIFEST",
        "status": "PASS_EXPLORATORY_FACTORIZED_MANIFEST",
        "claim_level": "exploratory",
        "chromosome": "22",
        "root": root,
        "ancestry_names": list(ANCESTRIES),
        "haplotypes": 2,
        "split_people": {split: audits[split].people for split in ("FIT", "VALID")},
        "split_donor_roles": {
            split: split_contract[split]["donor_role"] for split in ("FIT", "VALID")
        },
        "mosaic_parameters": {
            "admixture_generations": 12,
            "mixture_proportions": {"AFR": 0.25, "EUR": 0.60, "NAM": 0.15},
        },
        "axes": {
            "selected_loci": audits["FIT"].selected_loci,
            "markers": audits["FIT"].markers,
            "fit_valid_samples_disjoint": True,
            "selected_axis_sha256": audits["FIT"].selected_axis_sha256,
            "reference_factor_sha256": audits["FIT"].reference_sha256,
            "marker_axis_sha256": audits["FIT"].marker_axis_sha256,
        },
        "splits": {
            split: {
                "factors": {
                    name: descriptor(audits[split].paths[name], manifest_directory)
                    for name in FACTOR_NAMES
                },
                "provenance": {
                    name: descriptor(audits[split].paths[name], manifest_directory)
                    for name in ("mosaic_receipt", "bridge_receipt", "flare_receipt")
                } | {
                    "flare_ancestry_vcf_sha256":
                        audits[split].provenance["flare_ancestry_vcf_sha256"],
                },
            }
            for split in ("FIT", "VALID")
        },
        "manifest_sha256": serialized_sha256(manifest),
        "scientific_evidence": False,
        "training_performed": False,
    }
    receipt["semantic_sha256"] = canonical_sha256(receipt)
    return manifest, receipt


def write_outputs(manifest: Mapping[str, Any], receipt: Mapping[str, Any],
                  manifest_path: Path, receipt_path: Path) -> None:
    require(manifest_path.resolve() != receipt_path.resolve(), "output paths must differ")
    require(not manifest_path.exists() and not receipt_path.exists(), "refusing to overwrite outputs")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        require(sha256_file(manifest_path) == receipt["manifest_sha256"],
                "written manifest hash differs")
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8")
    except BaseException:
        manifest_path.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("fit", "valid"):
        for name in (*FACTOR_NAMES, "mosaic_receipt", "bridge_receipt", "flare_receipt"):
            parser.add_argument(f"--{split}-{name.replace('_', '-')}",
                                dest=f"{split}_{name}", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--root", choices=tuple(ROOT_SEEDS), default="R0")
    parser.add_argument("--fit-people", type=int, default=24)
    parser.add_argument("--valid-people", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_paths = {
        split.upper(): SplitPaths(**{
            name: getattr(args, f"{split}_{name}")
            for name in (*FACTOR_NAMES, "mosaic_receipt", "bridge_receipt", "flare_receipt")
        }) for split in ("fit", "valid")
    }
    manifest, receipt = build(
        split_paths["FIT"], split_paths["VALID"], args.manifest.parent,
        args.root, args.fit_people, args.valid_people,
    )
    write_outputs(manifest, receipt, args.manifest, args.receipt)
    print(json.dumps({"status": receipt["status"], "manifest_sha256": receipt["manifest_sha256"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
