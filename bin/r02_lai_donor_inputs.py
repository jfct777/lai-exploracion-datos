#!/usr/bin/env python3
"""Assemble authenticated regional inputs for ``AutonomousDonorContextLAI``.

The model receives diploid ALT dosage and observation masks.  Simulated truth
is returned in a separate object and is never used to choose people, loci,
reference donors, panels, regions, or query support.  FLARE arrays and prior
neural weights are not accepted by this interface.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

from r02_lai_donor_context import DonorContextBatch, STATE_NAMES
from r02_lai_donor_eligibility import partition_reference_mask
from r02_lai_individual_reference import IndividualReferenceArchive
from r02_lai_source_rare_catalogue import (
    CONDITIONS, canonical_locus_axis_sha256, catalogue_coverage,
    load_source_rare_axis_projection, load_source_rare_catalogue,
)


PANEL_TO_INDEX = {"African": 0, "European": 1, "Native_American": 2}
INDEX_FIELDS = (
    "person_id",
    "common_source_index", "common_chromosome", "common_position_bp",
    "common_ref", "common_alt", "common_position_cm",
    "event_source_index", "event_chromosome", "event_position_bp",
    "event_ref", "event_alt", "event_position_cm",
)
LOCUS_INDEX_FIELDS = tuple(field for field in INDEX_FIELDS if field != "person_id")
OVERLAY_FIELDS = (
    "person_id", "role", "query_cm", "query_chrom", "query_valid",
    "query_score_weight_cm", "truth_state", "truth_observed",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def region_plan_sha256(region_ids: Sequence[str], owner) -> str:
    """Digest exact ordered region names and the contiguous ownership matrix."""
    import numpy as np

    value = np.ascontiguousarray(owner)
    require(np.issubdtype(value.dtype, np.integer), "region owner must be integer")
    digest = hashlib.sha256()
    digest.update(json.dumps(list(region_ids), ensure_ascii=False,
                             separators=(",", ":")).encode())
    digest.update(b"\0" + str(value.dtype).encode() + b"\0")
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class FilePin:
    path: str | Path
    sha256: str

    def authenticate(self, name: str) -> Path:
        path = Path(self.path)
        require(path.is_file(), name + " is absent")
        require(len(self.sha256) == 64 and sha256(path) == self.sha256,
                name + " SHA-256 differs")
        return path


@dataclass(frozen=True)
class PinnedDonorSources:
    factorized_receipt: FilePin
    factorized_indices: FilePin
    target_manifest: FilePin
    target_loci: FilePin
    target_people: FilePin
    target_alt_dosage: FilePin
    target_observed: FilePin
    query_overlay: FilePin
    query_overlay_receipt: FilePin
    reference_archive: FilePin
    reference_index: FilePin
    original_roles: FilePin
    fold_roles: FilePin
    source_rare_catalogue: FilePin | None
    expected_build: str
    expected_chromosome: str
    source_pool_split: str
    related_unit_pairs: tuple[tuple[str, str], ...]
    source_rare_projection_receipt: FilePin | None = None
    source_rare_projection_table: FilePin | None = None


@dataclass(frozen=True)
class RegionOwnership:
    """Prospective query assignment; ``owner`` is [person, query], -1 if invalid."""

    region_ids: tuple[str, ...]
    owner: object
    plan_sha256: str


@dataclass(frozen=True)
class DonorRegionTargets:
    truth_state: object
    truth_observed: object
    query_score_weight_cm: object
    query_global_index: object
    full_case_denominator_by_person: object
    state_names: tuple[str, ...] = STATE_NAMES


@dataclass(frozen=True)
class DonorRegionInputs:
    batch: DonorContextBatch
    targets: DonorRegionTargets
    region_id: str
    condition: str
    effective_halo_cm: float
    locus_keys: tuple[tuple[str, int, str, str], ...]
    lineage: dict


class DonorInputReader:
    """Authenticate one case once and reuse open archive/memmaps across regions."""

    def __init__(self, sources: PinnedDonorSources):
        self.sources = sources
        self.paths = None
        self.reference = None
        self._permission = None

    def __enter__(self):
        import numpy as np

        self.paths = _authenticate_sources(self.sources)
        self.receipt = json.loads(self.paths["factorized_receipt"].read_text())
        self.manifest = json.loads(self.paths["target_manifest"].read_text())
        self.overlay_receipt = json.loads(self.paths["query_overlay_receipt"].read_text())
        with np.load(self.paths["factorized_indices"], allow_pickle=False) as stored:
            self.index = _load_npz_fields(stored, INDEX_FIELDS, "factorized indices")
        with np.load(self.paths["query_overlay"], allow_pickle=False) as stored:
            self.query = _load_npz_fields(stored, OVERLAY_FIELDS, "query/truth overlay")
        self.loci_fields, self.loci_rows = _rows(self.paths["target_loci"])
        self.people_fields, self.people_rows = _rows(self.paths["target_people"])
        self.role_tables = (_rows(self.paths["original_roles"]),
                            _rows(self.paths["fold_roles"]))
        if self.sources.source_rare_projection_receipt is not None:
            self.source_rare_catalogue = load_source_rare_axis_projection(
                self.paths["source_rare_projection_receipt"],
                self.sources.source_rare_projection_receipt.sha256,
                self.paths["source_rare_projection_table"],
                self.sources.source_rare_projection_table.sha256)
        else:
            self.source_rare_catalogue = load_source_rare_catalogue(
                self.paths["source_rare_catalogue"],
                self.sources.source_rare_catalogue.sha256)
        require(self.source_rare_catalogue.build == self.sources.expected_build
                and self.source_rare_catalogue.chromosome == self.sources.expected_chromosome
                and str(self.receipt.get("chromosome")) == self.sources.expected_chromosome
                and self.manifest.get("genome_build") == self.source_rare_catalogue.build
                and str(self.manifest.get("chromosome")) == self.source_rare_catalogue.chromosome,
                "catalogue and input-spec build or chromosome binding differs")
        channel_keys = {
            channel: _indexed_channel_keys(self.index, channel, self.loci_rows, np)
            for channel in ("common", "event")
        }
        material_keys = channel_keys["common"] + channel_keys["event"]
        if self.sources.source_rare_projection_receipt is not None:
            material_axis_sha256 = canonical_locus_axis_sha256(material_keys)
            require(material_axis_sha256 == self.source_rare_catalogue.projection_axis_sha256,
                    "target material axis differs from authenticated reference projection axis")
            require(self.source_rare_catalogue.loci <= set(material_keys),
                    "source-rare projection contains a key outside the exact material axis")
        self.catalogue_coverage = catalogue_coverage(
            self.source_rare_catalogue, self.loci_rows,
            channel_keys["common"], channel_keys["event"])
        self.reference = IndividualReferenceArchive(
            self.paths["reference_archive"], self.sources.reference_archive.sha256,
            1, {str(self.receipt["chromosome"]): self.paths["reference_index"]})
        try:
            if self.sources.source_rare_projection_receipt is not None:
                chromosome = self.sources.expected_chromosome
                reference_manifest = self.reference._manifest(chromosome)
                require(self.reference._inventory[chromosome]["manifest_sha256"]
                        == self.source_rare_catalogue.reference_manifest_sha256
                        and reference_manifest.get("source_vcf_sha256")
                        == self.source_rare_catalogue.reference_source_vcf_sha256,
                        "current reference manifest/source differs from projection parent")
                self.catalogue_coverage["classified_material_axis_sha256"] = material_axis_sha256
            # Large genotype arrays are opened only after catalogue/target/reference
            # axes have been authenticated.
            self.target_alt = np.load(self.paths["target_alt_dosage"], mmap_mode="r",
                                      allow_pickle=False)
            self.target_seen = np.load(self.paths["target_observed"], mmap_mode="r",
                                       allow_pickle=False)
        except Exception:
            self.close()
            raise
        return self

    def set_reference_cell_limit(self, value: int):
        require(type(value) is int and value > 0,
                "positive explicit reference-cell limit required")
        self.reference.max_output_genotypes = value

    def close(self):
        if self.reference is not None:
            self.reference.close()
            self.reference = None

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def assemble_region(self, ownership: RegionOwnership, region_id: str, **parameters):
        return assemble_region(self, ownership, region_id, **parameters)


def _rows(path: Path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        rows = list(reader)
    require(reader.fieldnames is not None, "TSV header is absent: " + str(path))
    return reader.fieldnames, rows


def _load_npz_fields(stored, fields, name):
    """Load only the explicit model/target fields, excluding incidental arrays."""
    missing = set(fields).difference(stored.files)
    require(not missing, name + " lacks: " + ",".join(sorted(missing)))
    return {field: stored[field] for field in fields}


def _indexed_channel_keys(index, channel, loci_rows, np):
    source = np.asarray(index[channel + "_source_index"], dtype=np.int64)
    chrom = index[channel + "_chromosome"].astype(str)
    bp = index[channel + "_position_bp"].astype(int)
    ref = index[channel + "_ref"].astype(str)
    alt = index[channel + "_alt"].astype(str)
    cm = index[channel + "_position_cm"].astype(float)
    require(all(len(value) == len(source) for value in (chrom, bp, ref, alt, cm))
            and np.isfinite(cm).all(), channel + " index axes differ")
    require(((0 <= source) & (source < len(loci_rows))).all()
            and len(set(source.tolist())) == len(source),
            channel + " source indices are invalid or duplicated")
    keys = []
    for i, source_row in enumerate(source.tolist()):
        row = loci_rows[source_row]
        key = (str(int(chrom[i])), int(bp[i]), ref[i], alt[i])
        require((str(int(row["chromosome"])), int(row["position_bp"]),
                 row["REF"], row["ALT"]) == key,
                "index/target exact locus key differs")
        keys.append(key)
    require(len(keys) == len(set(keys)), channel + " locus keys are duplicated")
    return tuple(keys)


def classify_pinned_locus_metadata(source_rare_catalogue: FilePin,
                                   factorized_indices: FilePin,
                                   target_loci: FilePin) -> dict[str, object]:
    """Classify R against V without reading people, genotypes, truth or references."""
    import numpy as np

    catalogue_path = source_rare_catalogue.authenticate("source_rare_catalogue")
    index_path = factorized_indices.authenticate("factorized_indices")
    loci_path = target_loci.authenticate("target_loci")
    catalogue = load_source_rare_catalogue(catalogue_path,
                                            source_rare_catalogue.sha256)
    fields, rows = _rows(loci_path)
    require(fields == ["chromosome", "position_bp", "REF", "ALT", "counted_allele"],
            "target material table columns differ")
    with np.load(index_path, allow_pickle=False) as stored:
        index = _load_npz_fields(stored, LOCUS_INDEX_FIELDS, "factorized locus indices")
    common = _indexed_channel_keys(index, "common", rows, np)
    event = _indexed_channel_keys(index, "event", rows, np)
    report = catalogue_coverage(catalogue, rows, common, event)
    return {**report,
            "source_rare_catalogue_manifest_sha256": source_rare_catalogue.sha256,
            "source_rare_table_sha256": catalogue.table_sha256,
            "original_source_catalogue_sha256": catalogue.source_catalogue_sha256,
            "factorized_indices_sha256": factorized_indices.sha256,
            "target_loci_sha256": target_loci.sha256,
            "people_read": False, "genotypes_read": False, "truth_read": False,
            "references_read": False, "training_performed": False}


def classify_pinned_projection_metadata(projection_receipt: FilePin,
                                        projection_table: FilePin,
                                        factorized_indices: FilePin,
                                        target_loci: FilePin) -> dict[str, object]:
    """Bind authenticated R-intersect-V rows to the exact target/index axis only."""
    import numpy as np

    receipt_path = projection_receipt.authenticate("source_rare_projection_receipt")
    table_path = projection_table.authenticate("source_rare_projection_table")
    index_path = factorized_indices.authenticate("factorized_indices")
    loci_path = target_loci.authenticate("target_loci")
    projection = load_source_rare_axis_projection(
        receipt_path, projection_receipt.sha256, table_path, projection_table.sha256)
    fields, rows = _rows(loci_path)
    require(fields == ["chromosome", "position_bp", "REF", "ALT", "counted_allele"],
            "target material table columns differ")
    with np.load(index_path, allow_pickle=False) as stored:
        index = _load_npz_fields(stored, LOCUS_INDEX_FIELDS, "factorized locus indices")
    common = _indexed_channel_keys(index, "common", rows, np)
    event = _indexed_channel_keys(index, "event", rows, np)
    material = common + event
    axis_sha256 = canonical_locus_axis_sha256(material)
    require(axis_sha256 == projection.projection_axis_sha256,
            "target material axis differs from authenticated reference projection axis")
    require(projection.loci <= set(material),
            "source-rare projection contains a key outside the exact material axis")
    report = catalogue_coverage(projection, rows, common, event)
    return {**report,
            "source_rare_projection_receipt_sha256": projection_receipt.sha256,
            "source_rare_projection_table_sha256": projection_table.sha256,
            "original_source_catalogue_sha256": projection.source_catalogue_sha256,
            "factorized_indices_sha256": factorized_indices.sha256,
            "target_loci_sha256": target_loci.sha256,
            "classified_material_axis_sha256": axis_sha256,
            "reference_manifest_sha256": projection.reference_manifest_sha256,
            "reference_source_vcf_sha256": projection.reference_source_vcf_sha256,
            "people_read": False, "genotypes_read": False, "truth_read": False,
            "reference_genotypes_read": False, "training_performed": False}


def sources_from_spec(path: str | Path) -> PinnedDonorSources:
    """Construct the pinned FIT sources recorded by ``INPUT_SPEC.json``."""
    spec = json.loads(Path(path).read_text())
    require(spec.get("schema") in {"r02_lai_donor_input_spec_v2",
                                   "r02_lai_donor_input_spec_v3"},
            "input specification schema differs")
    data = spec["sources"]
    case = spec.get("case")
    require(isinstance(case, dict) and case.get("genome_build") == "GRCh38"
            and str(case.get("chromosome")) in {str(value) for value in range(1, 23)},
            "input specification must declare GRCh38 and one autosome")
    material = data["target_material"]
    material_root = Path(material["directory"])
    overlay = data["query_truth_overlay"]
    reference = data["reference_archive"]
    roles = data["roles_by_pool"]
    pool = roles["source_pool"].split("/", 1)
    require(pool[0] == "SOURCE_VALID" and len(pool) == 2 and pool[1],
            "source_pool must identify SOURCE_VALID/<split>")
    pairs = roles.get("related_unit_pairs")
    require(isinstance(pairs, list)
            and all(isinstance(pair, list) and len(pair) == 2 for pair in pairs),
            "related_unit_pairs must be an explicit list of pairs")
    projection = data.get("source_rare_axis_projection")
    require((spec["schema"] == "r02_lai_donor_input_spec_v3") == (projection is not None),
            "v3 input specification requires one source-rare axis projection")
    if projection is not None:
        require(isinstance(projection, dict)
                and set(projection) == {"receipt", "table"},
                "source-rare projection descriptor fields differ")
    return PinnedDonorSources(
        factorized_receipt=FilePin(**data["factorized_receipt"]),
        factorized_indices=FilePin(**data["factorized_indices"]),
        target_manifest=FilePin(material_root / "manifest.json",
                                material["manifest_sha256"]),
        target_loci=FilePin(material_root / "loci.tsv", material["loci_tsv_sha256"]),
        target_people=FilePin(material_root / "persons.tsv",
                              material["persons_tsv_sha256"]),
        target_alt_dosage=FilePin(material_root / "alt_dosage.npy",
                                  material["alt_dosage_npy_sha256"]),
        target_observed=FilePin(material_root / "genotype_observed.npy",
                                material["genotype_observed_npy_sha256"]),
        query_overlay=FilePin(overlay["path"], overlay["sha256"]),
        query_overlay_receipt=FilePin(overlay["receipt_path"],
                                      overlay["receipt_sha256"]),
        reference_archive=FilePin(reference["path"], reference["sha256"]),
        reference_index=FilePin(reference["chr22_index_path"],
                                reference["chr22_index_sha256"]),
        original_roles=FilePin(roles["original_roles_path"],
                               roles["original_roles_sha256"]),
        fold_roles=FilePin(roles["fold_roles_path"], roles["fold_roles_sha256"]),
        source_rare_catalogue=(None if projection is not None
                               else FilePin(**data["source_rare_catalogue"])),
        expected_build=case["genome_build"],
        expected_chromosome=str(case["chromosome"]),
        source_pool_split=pool[1],
        related_unit_pairs=tuple(tuple(str(value) for value in pair) for pair in pairs),
        source_rare_projection_receipt=(FilePin(**projection["receipt"])
                                        if projection is not None else None),
        source_rare_projection_table=(FilePin(**projection["table"])
                                      if projection is not None else None),
    )


def _authenticate_sources(sources: PinnedDonorSources):
    pins = (
        ("factorized_receipt", sources.factorized_receipt),
        ("factorized_indices", sources.factorized_indices),
        ("target_manifest", sources.target_manifest),
        ("target_loci", sources.target_loci),
        ("target_people", sources.target_people),
        ("target_alt_dosage", sources.target_alt_dosage),
        ("target_observed", sources.target_observed),
        ("query_overlay", sources.query_overlay),
        ("query_overlay_receipt", sources.query_overlay_receipt),
        ("reference_archive", sources.reference_archive),
        ("reference_index", sources.reference_index),
        ("original_roles", sources.original_roles),
        ("fold_roles", sources.fold_roles),
    )
    source_pins = (("source_rare_projection_receipt", sources.source_rare_projection_receipt),
                   ("source_rare_projection_table", sources.source_rare_projection_table)) \
        if sources.source_rare_projection_receipt is not None \
        else (("source_rare_catalogue", sources.source_rare_catalogue),)
    result = {}
    for name, pin in pins + source_pins:
        if name == "reference_archive":
            path = Path(pin.path)
            require(path.is_file() and len(pin.sha256) == 64,
                    "reference_archive is absent or has invalid SHA-256")
            result[name] = path  # IndividualReferenceArchive authenticates it once.
        else:
            result[name] = pin.authenticate(name)
    return result


def _validate_ownership(ownership, query_valid, np):
    require(isinstance(ownership.region_ids, tuple) and ownership.region_ids
            and len(set(ownership.region_ids)) == len(ownership.region_ids),
            "unique nonempty region IDs required")
    owner = np.asarray(ownership.owner)
    require(owner.shape == query_valid.shape and np.issubdtype(owner.dtype, np.integer),
            "region ownership must be an integer [person,query] matrix")
    require(np.array_equal(owner >= 0, query_valid),
            "every valid query must have exactly one owner and invalid queries none")
    require((owner[query_valid] < len(ownership.region_ids)).all(),
            "region owner index is outside region_ids")
    require(ownership.plan_sha256 == region_plan_sha256(ownership.region_ids, owner),
            "region ownership plan SHA-256 differs")
    return owner.astype(np.int64, copy=False)


def _pack(values, masks, fill, np):
    maximum = max((int(mask.sum()) for mask in masks), default=0)
    require(maximum > 0, "selected region has no query for a selected person")
    shape = (len(masks), maximum) + values.shape[2:]
    result = np.full(shape, fill, dtype=values.dtype)
    index = np.full((len(masks), maximum), -1, dtype=np.int64)
    valid = np.zeros((len(masks), maximum), dtype=np.bool_)
    for person, mask in enumerate(masks):
        selected = np.flatnonzero(mask)
        result[person, :len(selected)] = values[person, selected]
        index[person, :len(selected)] = selected
        valid[person, :len(selected)] = True
    return result, index, valid


def _eligible_loci(locus_chrom, locus_cm, owned_chrom, owned_cm, halo, np):
    keep = np.zeros(len(locus_cm), dtype=np.bool_)
    for chrom in np.unique(owned_chrom):
        q = np.unique(owned_cm[owned_chrom == chrom])
        require(np.isfinite(q).all(), "owned query cM must be finite")
        li = np.flatnonzero(locus_chrom == chrom)
        if not len(li):
            continue
        cm = locus_cm[li]
        pos = np.searchsorted(q, cm)
        left = np.maximum(pos - 1, 0)
        right = np.minimum(pos, len(q) - 1)
        distance = np.minimum(np.abs(cm - q[left]), np.abs(cm - q[right]))
        keep[li] = distance <= halo
    return keep


def _pool_permission(reference_ids, reference_panels, sources, paths, role_tables=None):
    if role_tables is None:
        original_fields, original = _rows(paths["original_roles"])
        fold_fields, folded = _rows(paths["fold_roles"])
    else:
        (original_fields, original), (fold_fields, folded) = role_tables
    require({"sample_id", "atomic_unit_id", "role"} <= set(original_fields),
            "original role table lacks original units")
    require({"sample_id", "role", "ancestry", "split"} <= set(fold_fields),
            "fold role table columns differ")
    original_by_id = {row["sample_id"]: row for row in original}
    require(len(original_by_id) == len(original), "duplicate original-role sample")
    folded_by_id = {row["sample_id"]: row for row in folded}
    require(len(folded_by_id) == len(folded), "duplicate fold-role sample")
    reference_units = []
    for identity, panel in zip(reference_ids, reference_panels):
        require(identity in original_by_id and identity in folded_by_id,
                "reference is absent from role tables")
        require(original_by_id[identity]["role"] == "REF_TRAIN"
                and folded_by_id[identity]["role"] == "REF_TRAIN"
                and folded_by_id[identity]["ancestry"] == panel,
                "reference role or panel differs")
        reference_units.append(original_by_id[identity]["atomic_unit_id"])
    source_ids = [row["sample_id"] for row in folded
                  if row["role"] == "SOURCE_VALID"
                  and row["split"] == sources.source_pool_split]
    require(source_ids and len(source_ids) == len(set(source_ids)),
            "source pool is empty or duplicated")
    source_units = []
    for identity in source_ids:
        require(identity in original_by_id
                and original_by_id[identity]["role"] == "SOURCE_VALID",
                "source pool original role differs")
        source_units.append(original_by_id[identity]["atomic_unit_id"])
    result = partition_reference_mask(reference_ids, reference_units,
                                      reference_panels, sorted(set(source_units)),
                                      sources.related_unit_pairs)
    return result, len(source_ids), len(set(source_units))


def assemble_region(sources: PinnedDonorSources | DonorInputReader, ownership: RegionOwnership,
                    region_id: str, *, condition: str, depth: int,
                    locus_radius_cm: float, query_radius_cm: float,
                    max_output_genotypes: int,
                    person_indices: Sequence[int]) -> DonorRegionInputs:
    """Read one metadata-chosen region without training or FLARE inputs."""
    import numpy as np
    import torch

    if isinstance(sources, PinnedDonorSources):
        with DonorInputReader(sources) as reader:
            return assemble_region(reader, ownership, region_id, condition=condition,
                                   depth=depth, locus_radius_cm=locus_radius_cm,
                                   query_radius_cm=query_radius_cm,
                                   max_output_genotypes=max_output_genotypes,
                                   person_indices=person_indices)
    require(isinstance(sources, DonorInputReader) and sources.paths is not None,
            "an entered DonorInputReader or PinnedDonorSources is required")
    reader = sources
    pins = reader.sources
    require(condition in CONDITIONS,
            "condition must use complete source-rare catalogue membership")
    require(type(depth) is int and depth > 0, "positive explicit depth required")
    require(all(type(x) in (int, float) and math.isfinite(x) and x > 0
                for x in (locus_radius_cm, query_radius_cm)),
            "positive finite explicit cM radii required")
    require(type(max_output_genotypes) is int and max_output_genotypes > 0,
            "positive explicit reference-cell limit required")
    paths = reader.paths
    receipt, manifest, overlay_receipt = reader.receipt, reader.manifest, reader.overlay_receipt
    require(receipt.get("schema") == "r02_lai_factorized_predictors_v1"
            and receipt.get("status") == "PREDICTOR_INDEX_COMPLETE_NOT_TRAINING",
            "factorized receipt differs")
    require(receipt.get("role") == "FIT" and receipt.get("truth_arrays_loaded") is False
            and receipt.get("training_authorized") is False,
            "only truth-blind FIT inputs are admitted")
    require(receipt.get("outputs_sha256", {}).get("indices.npz") == pins.factorized_indices.sha256,
            "factorized receipt does not bind indices")
    require(manifest.get("schema") == "r02_lai_genotype_material_v1"
            and manifest.get("case_id") == receipt.get("case_id")
            and str(manifest.get("chromosome")) == str(receipt.get("chromosome"))
            and manifest.get("donor_split") == pins.source_pool_split,
            "target material identity differs")
    require(overlay_receipt.get("schema") == "r02_lai_truthblind_context_receipt_v2"
            and overlay_receipt.get("output_sha256") == pins.query_overlay.sha256
            and overlay_receipt.get("truth_read_for_query_geometry") is False,
            "query overlay receipt differs")
    require(overlay_receipt.get("roles") == ["FIT"], "overlay role is not FIT")
    manifest_outputs = manifest.get("outputs_sha256", {})
    expected_target = {
        "loci.tsv": pins.target_loci.sha256,
        "persons.tsv": pins.target_people.sha256,
        "alt_dosage.npy": pins.target_alt_dosage.sha256,
        "genotype_observed.npy": pins.target_observed.sha256,
    }
    require(all(manifest_outputs.get(name) == value for name, value in expected_target.items()),
            "target manifest does not bind target files")
    receipt_inputs = set(receipt.get("inputs_sha256", {}).values())
    require(set(expected_target.values()) <= receipt_inputs
            and pins.query_overlay.sha256 in receipt_inputs
            and pins.target_manifest.sha256 in receipt_inputs,
            "factorized receipt does not bind material or overlay")

    index, query = reader.index, reader.query
    required_query = {"person_id", "query_cm", "query_chrom", "query_valid",
                      "query_score_weight_cm", "truth_state", "truth_observed"}
    require(required_query <= set(query), "query/truth overlay arrays differ")
    require(np.array_equal(index["person_id"].astype(str), query["person_id"].astype(str)),
            "index and overlay person order differs")
    query_valid = np.asarray(query["query_valid"], dtype=np.bool_)
    truth_observed = np.asarray(query["truth_observed"], dtype=np.bool_)
    truth_state = np.asarray(query["truth_state"])
    require(query_valid.shape == truth_observed.shape == truth_state.shape
            and not (truth_observed & ~query_valid).any()
            and np.isin(truth_state[truth_observed], np.arange(6)).all(),
            "query masks or six-state truth differ")
    require(set(query.get("role", np.asarray([])).astype(str).tolist()) == {"FIT"},
            "serialized query role is not FIT")
    owner = _validate_ownership(ownership, query_valid, np)
    require(region_id in ownership.region_ids, "unknown region ID")
    region_number = ownership.region_ids.index(region_id)

    selected_people = np.asarray(tuple(person_indices), dtype=np.int64)
    require(selected_people.ndim == 1 and len(selected_people)
            and len(np.unique(selected_people)) == len(selected_people)
            and ((0 <= selected_people) & (selected_people < len(query_valid))).all(),
            "person_indices must be unique valid indices")

    loci_fields, loci_rows = reader.loci_fields, reader.loci_rows
    people_fields, people_rows = reader.people_fields, reader.people_rows
    require(loci_fields == ["chromosome", "position_bp", "REF", "ALT", "counted_allele"]
            and people_fields == ["sample_key_sha256", "donor_fold", "use"],
            "target material table columns differ")
    require([row["sample_key_sha256"] for row in people_rows]
            == query["person_id"].astype(str).tolist(), "target person order differs")
    require({row["donor_fold"] for row in people_rows} == {pins.source_pool_split},
            "target source-pool split differs")
    require({row["use"] for row in people_rows} == {"CONDITIONAL_DEVELOPMENT"},
            "target people use is not conditional development")

    channel_names = ["common", "event"]
    source_index, keys, locus_cm, rare = [], [], [], []
    for channel in channel_names:
        indices = np.asarray(index[channel + "_source_index"], dtype=np.int64)
        chrom = index[channel + "_chromosome"].astype(str)
        bp = index[channel + "_position_bp"].astype(int)
        ref = index[channel + "_ref"].astype(str)
        alt = index[channel + "_alt"].astype(str)
        cm = index[channel + "_position_cm"].astype(float)
        require(len(indices) == len(cm) and np.isfinite(cm).all(), channel + " cM differs")
        for i, source in enumerate(indices.tolist()):
            row = loci_rows[source]
            key = (str(int(chrom[i])), int(bp[i]), ref[i], alt[i])
            require((str(int(row["chromosome"])), int(row["position_bp"]), row["REF"], row["ALT"]) == key,
                    "index/target exact locus key differs")
            require(row["counted_allele"] == "ALT", "target material must retain raw ALT dosage")
            source_index.append(source); keys.append(key); locus_cm.append(float(cm[i]))
            rare.append(key in reader.source_rare_catalogue.counted_allele_by_locus)
    require(len(keys) == len(set(keys)) == len(set(source_index)),
            "COMMON/EVENT locus axes overlap or duplicate")
    locus_chrom = np.asarray([int(key[0]) for key in keys], dtype=np.int64)
    locus_cm = np.asarray(locus_cm, dtype=np.float64)
    rare = np.asarray(rare, dtype=np.bool_)
    if condition == "NO_SOURCE_RARE":
        condition_keep = ~rare
        source_index = np.asarray(source_index, dtype=np.int64)[condition_keep]
        keys = tuple(key for key, use in zip(keys, condition_keep.tolist()) if use)
        locus_cm = locus_cm[condition_keep]
        locus_chrom = locus_chrom[condition_keep]
        rare = rare[condition_keep]
    else:
        source_index = np.asarray(source_index, dtype=np.int64)
    owned_all = owner == region_number
    owned_chrom = np.asarray(query["query_chrom"])[owned_all].astype(np.int64)
    owned_cm = np.asarray(query["query_cm"])[owned_all].astype(np.float64)
    halo = depth * float(locus_radius_cm) + float(query_radius_cm)
    keep = _eligible_loci(locus_chrom, locus_cm, owned_chrom, owned_cm, halo, np)
    source_index = source_index[keep]
    keys = tuple(key for key, use in zip(keys, keep.tolist()) if use)
    locus_cm = locus_cm[keep]; locus_chrom = locus_chrom[keep]
    rare = rare[keep]
    real_loci = len(keys)
    expected_rare = np.asarray([
        key in reader.source_rare_catalogue.counted_allele_by_locus for key in keys
    ], dtype=np.bool_)
    require(np.array_equal(rare, expected_rare),
            "regional rare flags differ from complete source catalogue membership")
    if condition == "NO_SOURCE_RARE":
        require(not rare.any(), "NO_SOURCE_RARE retained a source-rare locus")
    allele_code = {"A": 0, "C": 1, "G": 2, "T": 3}
    require(all(len(ref) == len(alt) == 1 and ref in allele_code and alt in allele_code
                for _, _, ref, alt in keys), "donor context currently requires SNV REF/ALT alleles")
    alleles = np.asarray([[allele_code[ref], allele_code[alt]] for _, _, ref, alt in keys],
                         dtype=np.int64).reshape((-1, 2))

    target_alt, target_seen = reader.target_alt, reader.target_seen
    require(target_alt.shape == target_seen.shape == (len(people_rows), len(loci_rows))
            and target_seen.dtype == np.bool_, "target dosage axes differ")
    target_alt = np.asarray(target_alt[np.ix_(selected_people, source_index)], dtype=np.int8)
    target_seen = np.asarray(target_seen[np.ix_(selected_people, source_index)], dtype=np.bool_)
    require(np.isin(target_alt[target_seen], [0, 1, 2]).all(), "target observed ALT dosage differs")

    reader.set_reference_cell_limit(max_output_genotypes)
    if keys:
        reference = reader.reference.read(keys)
        reference_ids, reference_panels = reference.reference_ids, reference.reference_panels
        reference_dosage, reference_seen = reference.dosage.T, reference.observed.T
    else:
        # Authenticate only the people/panel manifest.  One explicitly masked
        # numerical row keeps tensor ranks valid but is not a genomic locus.
        reader.reference._manifest(str(receipt["chromosome"]))
        reference_ids = reader.reference._reference_ids or ()
        reference_panels = reader.reference._reference_panels or ()
        target_alt = np.zeros((len(selected_people), 1), dtype=np.int8)
        target_seen = np.zeros((len(selected_people), 1), dtype=np.bool_)
        reference_dosage = np.zeros((len(reference_ids), 1), dtype=np.int8)
        reference_seen = np.zeros((len(reference_ids), 1), dtype=np.bool_)
        alleles = np.asarray([[0, 1]], dtype=np.int64)
        rare = np.asarray([False], dtype=np.bool_)
        locus_cm = np.asarray([0.0], dtype=np.float64)
        locus_chrom = np.asarray([int(receipt["chromosome"])], dtype=np.int64)
    permission, pool_people, pool_units = _pool_permission(
        reference_ids, reference_panels, pins, paths, reader.role_tables)
    allowed = np.broadcast_to(np.asarray(permission["allowed"], dtype=np.bool_),
                              (len(selected_people), len(reference_ids))).copy()

    selected_owned = owned_all[selected_people]
    query_cm, global_index, packed_valid = _pack(
        np.asarray(query["query_cm"])[selected_people], selected_owned, 0.0, np)
    query_chrom, check_index, _ = _pack(
        np.asarray(query["query_chrom"])[selected_people], selected_owned, 0, np)
    require(np.array_equal(global_index, check_index), "packed query axes differ")
    state, check_index, _ = _pack(truth_state[selected_people], selected_owned, 0, np)
    require(np.array_equal(global_index, check_index), "packed truth axis differs")
    observed, _, _ = _pack(truth_observed[selected_people], selected_owned, False, np)
    score_weight, _, _ = _pack(
        np.asarray(query["query_score_weight_cm"])[selected_people], selected_owned, 0.0, np)
    denominators = (query_valid & truth_observed).sum(axis=1).astype(np.int64)[selected_people]
    require((denominators > 0).all(), "full-case per-person denominator is zero")

    batch = DonorContextBatch(
        target_alt_dosage=torch.from_numpy(target_alt.astype(np.float32, copy=True)),
        reference_alt_dosage=torch.from_numpy(reference_dosage.astype(np.float32, copy=True)),
        target_observed=torch.from_numpy(target_seen.copy()),
        reference_observed=torch.from_numpy(reference_seen.copy()),
        ref_alt_alleles=torch.from_numpy(alleles),
        rare_catalog=torch.from_numpy(rare),
        locus_cm=torch.from_numpy(locus_cm),
        locus_chrom=torch.from_numpy(locus_chrom),
        query_cm=torch.from_numpy(query_cm),
        query_chrom=torch.from_numpy(query_chrom.astype(np.int64)),
        query_valid=torch.from_numpy(packed_valid),
        reference_panel=torch.tensor([PANEL_TO_INDEX[value]
                                      for value in reference_panels], dtype=torch.long),
        donor_allowed=torch.from_numpy(allowed))
    targets = DonorRegionTargets(
        truth_state=torch.from_numpy(state.astype(np.int64)),
        truth_observed=torch.from_numpy(observed),
        query_score_weight_cm=torch.from_numpy(score_weight),
        query_global_index=torch.from_numpy(global_index),
        full_case_denominator_by_person=torch.from_numpy(denominators))
    lineage = {
        "case_id": receipt["case_id"], "task_id": receipt["task_id"],
        "chromosome": str(receipt["chromosome"]), "region_plan_sha256": ownership.plan_sha256,
        "factorized_indices_sha256": pins.factorized_indices.sha256,
        "source_rare_catalogue_manifest_sha256": (
            pins.source_rare_catalogue.sha256 if pins.source_rare_catalogue is not None else None),
        "source_rare_projection_receipt_sha256": (
            pins.source_rare_projection_receipt.sha256
            if pins.source_rare_projection_receipt is not None else None),
        "source_rare_table_sha256": reader.source_rare_catalogue.table_sha256,
        "original_source_catalogue_sha256":
            reader.source_rare_catalogue.source_catalogue_sha256,
        "source_build": reader.source_rare_catalogue.build,
        "build_binding_scope":
            "INPUT_SPEC_DECLARATION_MATCHES_CATALOGUE;MATERIAL_HAS_NO_EMBEDDED_BUILD",
        "query_overlay_sha256": pins.query_overlay.sha256,
        "reference_archive_sha256": pins.reference_archive.sha256,
        "source_pool_split": pins.source_pool_split,
        "source_pool_people": pool_people, "source_pool_original_units": pool_units,
        "reference_people": len(reference_ids),
        "reference_counts_by_panel": permission["reference_samples_before_by_panel"],
        "reference_counts_allowed_by_panel": permission["reference_samples_after_by_panel"],
        "permission_partition_fixed": permission["partition_fixed"],
        "truth_used_for_input_selection": False, "flare_read": False,
        "fv_weights_read": False, "training_performed": False,
        "real_loci": real_loci, "tensor_loci": len(locus_cm),
        "condition": condition,
        "region_source_rare_loci": int(rare.sum()),
        "empty_region_padding": real_loci == 0,
        "selected_people": len(selected_people),
        "owned_queries": int(packed_valid.sum()),
        "full_case_denominators_fixed": True,
        "source_rare_coverage": dict(reader.catalogue_coverage),
    }
    return DonorRegionInputs(batch, targets, region_id, condition, halo, keys, lineage)
