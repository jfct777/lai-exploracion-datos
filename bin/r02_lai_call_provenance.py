#!/usr/bin/env python3
"""Authenticate sparse pre-phase missingness for exact role-panel cells.

The sparse mask records cells whose genotype was missing before phasing and
complete afterwards.  Absence from that mask means ``ORIGINAL_OBSERVED`` only
inside the proved Cartesian product of the listed loci and the exact ordered
samples of the role VCF.  This module does not filter cells, certify phase,
admit training/EVALUATE, or establish allele representation outside a listed
literal REF/ALT key.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import gzip
from pathlib import Path
import re

import r02_lai_allele_support as base


MANIFEST_SCHEMA = "r02_lai_call_provenance_v1"
MANIFEST_STATUS = "AUTHENTICATED_SPARSE_MISSING_BEFORE_CALLED_AFTER"
PROOF_SCHEMA = "r02_lai_role_call_concordance_v1"
PROOF_STATUS = "COMPLETE_EXACT_ROLE_PANEL_CALL_CONCORDANCE"
TRANSITION = "MISSING_BEFORE_PHASE_CALLED_AFTER_NOT_ORIGINAL_OBSERVATION"
LOCUS_COLUMNS = ("chromosome", "position_bp", "REF", "ALT")
MASK_COLUMNS = (*LOCUS_COLUMNS, "sample_id", "role", "norm_GT", "phased_GT", "status")
LIMITATIONS = {
    "audited_loci": "REF_IDENTITY_ONLY_AT_LISTED_LITERAL_REF_ALT_LOCI",
    "alt_scope": "ALT_NOT_GLOBALLY_AUDITED",
    "completed_calls": "NOT_ORIGINAL_OBSERVATIONS",
    "filtering": "ANNOTATION_ONLY_NO_EXCLUSION",
    "phase": "NOT_PHASE_ACCURACY_EVIDENCE",
    "scientific_use": "NO_TRAINING_EVALUATE_OR_TRUTH_ADMISSION",
}
MANIFEST_FIELDS = {
    "schema", "status", "chromosome", "panel_role", "panel_vcf_sha256",
    "panel_sample_order_sha256", "panel_sample_members_sha256",
    "panel_sample_count", "domain", "sparse_mask", "concordance_proof",
    "limitations",
}
DOMAIN_FIELDS = {"coverage", "loci_path", "loci_sha256", "locus_count", "locus_columns"}
MASK_FIELDS = {"path", "sha256", "row_count", "columns", "transition"}
DESCRIPTOR_FIELDS = {"path", "sha256"}
PROOF_FIELDS = {
    "schema", "status", "chromosome", "panel_role", "panel_vcf_sha256",
    "panel_sample_order_sha256", "panel_sample_members_sha256",
    "panel_sample_count", "domain_loci_sha256", "domain_locus_count",
    "sparse_mask_sha256", "sparse_mask_rows", "total_cells",
    "concordant_original_cells", "missing_before_called_after_cells",
    "discordant_cells", "ambiguous_cells",
}
SHA256 = re.compile(r"[0-9a-f]{64}")


def _require(condition, message):
    base.require(condition, message)


def _resolve(manifest_path, value):
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


def _descriptor(manifest_path, descriptor, label):
    _require(isinstance(descriptor, dict) and set(descriptor) == DESCRIPTOR_FIELDS,
             f"Invalid {label} descriptor")
    _require(isinstance(descriptor["sha256"], str) and SHA256.fullmatch(descriptor["sha256"]),
             f"Invalid {label} SHA256")
    path = _resolve(manifest_path, descriptor["path"])
    _require(path.is_file() and base.sha256(path) == descriptor["sha256"],
             f"{label} hash mismatch")
    return path


def _open_table(path):
    with path.open("rb") as handle:
        compressed = handle.read(2) == b"\x1f\x8b"
    return gzip.open(path, "rt", newline="") if compressed else path.open(newline="")


def _locus(row, chromosome):
    _require(set(row) == set(LOCUS_COLUMNS), "Unexpected locus-domain columns")
    try:
        position = int(row["position_bp"])
    except (TypeError, ValueError):
        raise ValueError("Invalid locus position") from None
    key = (row["chromosome"], position, row["REF"], row["ALT"])
    _require(key[0] == chromosome and position > 0, "Locus outside declared chromosome")
    _require(all(isinstance(a, str) and len(a) == 1 and a in "ACGT" for a in key[2:])
             and key[2] != key[3], "Provenance locus must be a literal biallelic SNV key")
    return key


def _parse_called_gt(value):
    _require(isinstance(value, str) and value.count("|") == 1 and "/" not in value,
             "Sparse transition requires a phased diploid genotype")
    fields = value.split("|")
    _require(all(field in ("0", "1") for field in fields),
             "Sparse transition genotype is not a biallelic fully called genotype")
    return tuple(int(field) for field in fields)


@dataclass(frozen=True)
class CallProvenance:
    chromosome: str
    panel_role: str
    panel_vcf_sha256: str
    samples: frozenset[str]
    loci: frozenset[tuple[str, int, str, str]]
    completed: dict[tuple[tuple[str, int, str, str], str], tuple[int, int]]
    receipts: dict
    manifest: dict

    def covers(self, key):
        """Return whether the exact literal locus belongs to the proved domain."""
        return tuple(key) in self.loci

    def proves_literal_key(self, key, record_alleles=None):
        """Support REF identity only for the exact audited biallelic row."""
        key = tuple(key)
        return (key in self.loci and
                (record_alleles is None or tuple(record_alleles) == (key[2], key[3])))

    def classify(self, key, sample_id, post_gt, phased):
        """Classify one complete runtime call; reject any uncertain axis mapping."""
        key = tuple(key)
        _require(key in self.loci, "Cell provenance requested outside authenticated locus domain")
        _require(sample_id in self.samples, "Cell provenance requested outside authenticated sample domain")
        _require(isinstance(post_gt, tuple) and len(post_gt) == 2
                 and all(type(a) is int and a in (0, 1) for a in post_gt),
                 "Cell provenance requires a complete diploid runtime genotype")
        expected = self.completed.get((key, sample_id))
        if expected is None:
            return "ORIGINAL_OBSERVED"
        _require(phased and post_gt == expected,
                 "Runtime genotype differs from authenticated missing-before/called-after cell")
        return "MISSING_BEFORE_CALLED_AFTER"


def load_call_provenance(manifest_path, *, panel_vcf_sha256, panel_role, chromosome, samples):
    """Load and fully authenticate one exact role-panel provenance manifest."""
    manifest_path = Path(manifest_path)
    data = base.read_json(manifest_path)
    _require(set(data) == MANIFEST_FIELDS, "Unexpected or missing call-provenance manifest fields")
    _require(data["schema"] == MANIFEST_SCHEMA and data["status"] == MANIFEST_STATUS,
             "Unknown or incomplete call-provenance manifest")
    _require(data["chromosome"] == chromosome and data["panel_role"] == panel_role
             and data["panel_vcf_sha256"] == panel_vcf_sha256,
             "Call provenance does not bind this chromosome, role and panel VCF")
    _require(isinstance(panel_role, str) and re.fullmatch(r"[A-Z][A-Z0-9_]*", panel_role)
             and SHA256.fullmatch(panel_vcf_sha256), "Invalid runtime role or panel SHA256")
    samples = list(samples)
    _require(type(data["panel_sample_count"]) is int and data["panel_sample_count"] == len(samples)
             and data["panel_sample_order_sha256"] == base.sample_hash(samples)
             and data["panel_sample_members_sha256"] == base.sample_hash(sorted(samples)),
             "Call provenance sample axis differs from the runtime panel")
    _require(len(samples) == len(set(samples)) and all(samples), "Runtime panel samples are invalid")
    _require(data["limitations"] == LIMITATIONS, "Call-provenance limitations differ from the supported scope")

    domain = data["domain"]
    _require(isinstance(domain, dict) and set(domain) == DOMAIN_FIELDS,
             "Unexpected or missing provenance-domain fields")
    _require(domain["coverage"] == "FULL_CARTESIAN_PRODUCT_OF_LISTED_LOCI_AND_PANEL_SAMPLES"
             and domain["locus_columns"] == list(LOCUS_COLUMNS),
             "Sparse absence lacks a proved full Cartesian domain")
    loci_path = _descriptor(manifest_path, {"path": domain["loci_path"], "sha256": domain["loci_sha256"]},
                            "locus-domain file")
    loci = set()
    with _open_table(loci_path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        _require(tuple(reader.fieldnames or ()) == LOCUS_COLUMNS, "Unexpected locus-domain header")
        for row in reader:
            key = _locus(row, chromosome)
            _require(key not in loci, "Duplicate locus in provenance domain")
            loci.add(key)
    _require(type(domain["locus_count"]) is int and domain["locus_count"] > 0
             and len(loci) == domain["locus_count"], "Provenance locus-domain count differs")

    mask = data["sparse_mask"]
    _require(isinstance(mask, dict) and set(mask) == MASK_FIELDS,
             "Unexpected or missing sparse-mask fields")
    _require(mask["columns"] == list(MASK_COLUMNS) and mask["transition"] == TRANSITION,
             "Unsupported sparse-mask columns or transition")
    mask_path = _descriptor(manifest_path, {"path": mask["path"], "sha256": mask["sha256"]},
                            "sparse-mask file")
    completed = {}
    sample_set = frozenset(samples)
    with _open_table(mask_path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        _require(tuple(reader.fieldnames or ()) == MASK_COLUMNS, "Unexpected sparse-mask header")
        for row in reader:
            key = _locus({name: row[name] for name in LOCUS_COLUMNS}, chromosome)
            sample = row["sample_id"]
            _require(key in loci and sample in sample_set, "Sparse-mask cell is outside the proved domain")
            _require(row["norm_GT"] in ("./.", ".|.") and row["status"] == TRANSITION,
                     "Sparse mask contains an invalid pre/post transition")
            _require(row["role"] == panel_role, "Sparse-mask role differs from the exact runtime role")
            cell = (key, sample)
            _require(cell not in completed, "Duplicate sparse-mask cell")
            completed[cell] = _parse_called_gt(row["phased_GT"])
    _require(type(mask["row_count"]) is int and mask["row_count"] == len(completed),
             "Sparse-mask row count differs")

    proof_descriptor = data["concordance_proof"]
    proof_path = _descriptor(manifest_path, proof_descriptor, "role-panel concordance proof")
    proof = base.read_json(proof_path)
    _require(set(proof) == PROOF_FIELDS and proof["schema"] == PROOF_SCHEMA
             and proof["status"] == PROOF_STATUS, "Unknown or malformed role-panel concordance proof")
    numeric = ("panel_sample_count", "domain_locus_count", "sparse_mask_rows", "total_cells",
               "concordant_original_cells", "missing_before_called_after_cells",
               "discordant_cells", "ambiguous_cells")
    _require(all(type(proof[name]) is int and proof[name] >= 0 for name in numeric),
             "Role-panel concordance proof contains invalid counts")
    binding = (
        proof["chromosome"] == chromosome,
        proof["panel_role"] == panel_role,
        proof["panel_vcf_sha256"] == panel_vcf_sha256,
        proof["panel_sample_order_sha256"] == data["panel_sample_order_sha256"],
        proof["panel_sample_members_sha256"] == data["panel_sample_members_sha256"],
        proof["panel_sample_count"] == len(samples),
        proof["domain_loci_sha256"] == domain["loci_sha256"],
        proof["domain_locus_count"] == len(loci),
        proof["sparse_mask_sha256"] == mask["sha256"],
        proof["sparse_mask_rows"] == len(completed),
    )
    _require(all(binding), "Role-panel concordance proof does not bind the runtime axes and files")
    total = len(samples) * len(loci)
    _require(proof["total_cells"] == total
             and proof["concordant_original_cells"] + proof["missing_before_called_after_cells"] == total
             and proof["missing_before_called_after_cells"] == len(completed)
             and proof["discordant_cells"] == 0 and proof["ambiguous_cells"] == 0,
             "Role-panel concordance proof is incomplete, discordant or arithmetically inconsistent")

    receipts = {
        "manifest": {"path": str(manifest_path.resolve()), "sha256": base.sha256(manifest_path),
                     "size_bytes": manifest_path.stat().st_size},
        "locus_domain": {"path": str(loci_path.resolve()), "sha256": domain["loci_sha256"],
                         "size_bytes": loci_path.stat().st_size},
        "sparse_mask": {"path": str(mask_path.resolve()), "sha256": mask["sha256"],
                        "size_bytes": mask_path.stat().st_size},
        "concordance_proof": {"path": str(proof_path.resolve()), "sha256": proof_descriptor["sha256"],
                              "size_bytes": proof_path.stat().st_size},
    }
    return CallProvenance(chromosome, panel_role, panel_vcf_sha256, sample_set, frozenset(loci), completed,
                          receipts, data)
