"""Factorized, truth-free ordered common context for the M39 technical bridge.

This module does not select neighbours, train models, infer rare phase or load
ancestry truth. Common matrices remain shared; one query/anchor/site chunk is
expanded at a time. Candidate rank is local to an anchor, never a donor identity.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Iterator, Mapping, Sequence

import numpy as np

from m33_safe_bridge_core import require, write_exclusive_json
from m34_prepare_panel_factors import _locus_id, sha256_file

SCHEMA = "m39-ordered-context-factorized-v1"
SCOPE = "technical_only_common_ordered_diploid_rare_no_truth"
CHANNEL_NAMES = ("match", "mismatch", "query_missing", "reference_missing", "joint_called")
MAX_CANDIDATES = 8  # Authorized ceiling for this technical step, not an optimum.
MANIFEST_NAME = "manifest.json"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_DATA_FIELDS = frozenset((
    "selected", "reference_ancestry", "reference_sample_key_sha256", "sample_key_sha256",
    "minor_is_alt", "common_ref", "common_target", "common_cm", "ref_dosage",
    "ref_observed", "query_dosage", "query_observed", "counts",
    "reference_hap_original_index",
    "common_pos", "common_ref_allele", "common_alt_allele", "common_locus_id",
))
_CANDIDATE_FIELDS = frozenset((
    "common_context", "candidate_ref_index", "candidate_hap", "candidate_mask", "ref_dosage",
    "rare_ref_observed", "query_dosage", "query_observed", "pooled_summary",
    "reference_sample_key_sha256", "sample_key_sha256", "reference_ancestry", "ancestry",
    "anchor_indices", "minor_is_alt", "common_feature_names", "rare_semantics", "radius_cm",
    "locus_id", "chrom", "pos", "ref", "alt", "cM",
))
_ARRAY_FIELDS = frozenset((
    "common_ref", "common_target", "common_cm", "reference_sample_key_sha256",
    "sample_key_sha256", "reference_ancestry", "reference_hap_original_index",
    "reference_hap_canonicalized", "candidate_ref_index", "candidate_hap", "candidate_mask",
    "anchor_indices", "locus_id", "chrom", "pos", "ref", "alt", "cM", "minor_is_alt",
    "radius_cm", "ref_dosage", "ref_observed", "query_dosage", "query_observed",
    "common_pos", "common_ref_allele", "common_alt_allele", "common_locus_id",
))


def _array(value, name: str) -> np.ndarray:
    require(isinstance(value, np.ndarray) and value.dtype.kind != "O", f"{name}: numeric/string array required")
    return value


def _integer(value: np.ndarray, name: str) -> None:
    require(value.dtype.kind in "iu", f"{name}: integer dtype required")


def _binary(value: np.ndarray, name: str) -> None:
    require(value.dtype.kind in "biu" and bool(np.all((value == 0) | (value == 1))),
            f"{name}: binary mask required")


def _keys(value: np.ndarray, name: str) -> None:
    require(value.ndim == 1 and value.dtype == np.dtype("S64"), f"{name}: S64 axis required")
    keys = [bytes(x).decode("ascii", errors="replace") for x in value]
    require(len(set(keys)) == len(keys) and all(_SHA.fullmatch(x) for x in keys),
            f"{name}: duplicate/invalid hashed sample keys")


def _common_matrix(value: np.ndarray, name: str, people: int) -> None:
    require(value.dtype == np.dtype("int8") and value.ndim == 3 and value.shape[1:] == (people, 2),
            f"{name}: expected int8 [sites,people,2]")
    # Validate in blocks: an all-sites temporary is not needed even here.
    for begin in range(0, len(value), 4096):
        block = value[begin:begin + 4096]
        require(bool(np.all((block >= -1) & (block <= 1))), f"{name}: alleles must be -1/0/1")


def canonicalize_reference_homologs(data: Mapping) -> dict:
    """Copy only common REF alleles into a source-label-independent lane order.

    A lane is ordered by SHA256(person key, complete common sequence), including
    missing values and site order. Rare values are never inspected. The mapping
    records canonical lane -> original common lane. Identical common lanes are
    interchangeable; this operation does not identify a rare haplotype.
    """
    require(set(data) <= _DATA_FIELDS, "undeclared data fields")
    common = _array(data["common_ref"], "common_ref")
    keys = _array(data["reference_sample_key_sha256"], "reference_sample_key_sha256")
    _keys(keys, "reference_sample_key_sha256")
    _common_matrix(common, "common_ref", len(keys))
    previous = np.asarray(data.get("reference_hap_original_index",
                                 np.tile(np.arange(2, dtype=np.int8), (len(keys), 1))))
    require(previous.shape == (len(keys), 2) and previous.dtype.kind in "iu"
            and np.all(np.sort(previous, axis=1) == [0, 1]), "invalid original haplotype mapping")
    canonical = np.empty_like(common)
    mapping = np.empty((len(keys), 2), dtype=np.int8)
    for person, key in enumerate(keys):
        digests = []
        for hap in range(2):
            digest = hashlib.sha256(b"M39_ORDERED_COMMON_LANE_V1|" + bytes(key) + b"|")
            digest.update(np.ascontiguousarray(common[:, person, hap]).tobytes())
            digests.append(digest.digest())
        order = np.asarray(sorted(range(2), key=lambda h: digests[h]))
        canonical[:, person, :] = common[:, person, :][:, order]
        mapping[person] = previous[person, order]
    result = dict(data)
    result["common_ref"] = canonical
    result["reference_hap_original_index"] = mapping
    return result


def _validate_arrays(a: Mapping[str, np.ndarray]) -> None:
    require(set(a) == _ARRAY_FIELDS, "factorized array inventory differs")
    for name, value in a.items():
        _array(value, name)
    for name in ("sample_key_sha256", "reference_sample_key_sha256"):
        _keys(a[name], name)
    nq, nr = len(a["sample_key_sha256"]), len(a["reference_sample_key_sha256"])
    require(nq > 0 and nr > 0, "query/reference axes must be nonempty")
    require(not (set(a["sample_key_sha256"]) & set(a["reference_sample_key_sha256"])),
            "query/reference sample roles overlap")
    _common_matrix(a["common_ref"], "common_ref", nr)
    _common_matrix(a["common_target"], "common_target", nq)
    cm = a["common_cm"]
    require(cm.dtype == np.dtype("float64") and cm.shape == (len(a["common_ref"]),)
            and len(cm) == len(a["common_target"]) and np.all(np.isfinite(cm))
            and np.all(cm[1:] >= cm[:-1]), "common cM axis invalid or unsorted")
    for name in ("common_pos", "common_ref_allele", "common_alt_allele", "common_locus_id"):
        require(a[name].shape == cm.shape, f"{name}: common locus axis differs")
    _integer(a["common_pos"], "common_pos")
    _integer(a["common_locus_id"], "common_locus_id")
    require(np.all(a["common_pos"] > 0) and a["common_ref_allele"].dtype.kind == "S"
            and a["common_alt_allele"].dtype.kind == "S", "common locus allele types differ")
    common_order = np.lexsort((a["common_alt_allele"], a["common_ref_allele"], a["common_pos"]))
    require(np.array_equal(common_order, np.arange(len(cm))), "common genomic allele order differs")
    require(len(np.unique(a["common_locus_id"])) == len(cm), "duplicate common locus identity")
    for bp, ref, alt, locus in zip(a["common_pos"], a["common_ref_allele"], a["common_alt_allele"], a["common_locus_id"]):
        require(ref in (b"A", b"C", b"G", b"T") and alt in (b"A", b"C", b"G", b"T") and ref != alt,
                "invalid common allele")
        require(int(locus) == _locus_id("22", int(bp), ref.decode("ascii"), alt.decode("ascii")),
                "common canonical locus identity differs")
    ancestry = a["reference_ancestry"]
    _integer(ancestry, "reference_ancestry")
    require(ancestry.shape == (nr,) and np.all((ancestry >= 0) & (ancestry < 3)), "reference ancestry invalid")
    mapping = a["reference_hap_original_index"]
    _integer(mapping, "reference_hap_original_index")
    require(mapping.shape == (nr, 2) and np.all(np.sort(mapping, axis=1) == [0, 1]),
            "invalid original haplotype mapping")
    _binary(a["reference_hap_canonicalized"], "reference_hap_canonicalized")
    require(a["reference_hap_canonicalized"].shape == (1,), "canonicalization flag shape differs")
    anchors = a["anchor_indices"]
    _integer(anchors, "anchor_indices")
    require(anchors.ndim == 1 and len(anchors) > 0 and np.all(anchors >= 0)
            and np.all(anchors[1:] > anchors[:-1]), "anchors must be strictly increasing unique indices")
    nj = len(anchors)
    for name in ("locus_id", "chrom", "pos", "ref", "alt", "cM", "minor_is_alt"):
        require(a[name].shape == (nj,), f"{name}: anchor axis differs")
    for name in ("locus_id", "chrom", "pos"):
        _integer(a[name], name)
    require(np.all(a["chrom"] == 22) and np.all(a["pos"] > 0), "only chromosome22 positive positions allowed")
    require(len(np.unique(a["locus_id"])) == nj and np.all(np.isfinite(a["cM"]))
            and a["cM"].dtype == np.dtype("float64"), "invalid locus identity or cM")
    order = np.lexsort((a["locus_id"], a["pos"], a["cM"]))
    require(np.array_equal(order, np.arange(nj)), "anchor genomic order differs")
    require(a["ref"].dtype.kind == a["alt"].dtype.kind == "S"
            and all(x in (b"A", b"C", b"G", b"T") for x in a["ref"])
            and all(x in (b"A", b"C", b"G", b"T") for x in a["alt"])
            and np.all(a["ref"] != a["alt"]), "invalid anchor alleles")
    for bp, ref, alt, locus in zip(a["pos"], a["ref"], a["alt"], a["locus_id"]):
        require(int(locus) == _locus_id("22", int(bp), ref.decode("ascii"), alt.decode("ascii")),
                "anchor canonical locus identity differs")
    require(not np.intersect1d(a["common_locus_id"], a["locus_id"]).size,
            "common scaffold overlaps rare anchors")
    _binary(a["minor_is_alt"], "minor_is_alt")
    radius = a["radius_cm"]
    require(radius.dtype == np.dtype("float64") and radius.shape == (1,)
            and math.isfinite(float(radius[0])) and 0 < radius[0] <= 1,
            "radius must match authorized (0,1] cM technical scope")
    candidate, hap, mask = (a[k] for k in ("candidate_ref_index", "candidate_hap", "candidate_mask"))
    _integer(candidate, "candidate_ref_index")
    _integer(hap, "candidate_hap")
    _binary(mask, "candidate_mask")
    require(candidate.ndim == 5 and candidate.shape[:4] == (nq, nj, 2, 3)
            and 0 < candidate.shape[4] <= MAX_CANDIDATES and hap.shape == mask.shape == candidate.shape,
            "candidate axes or K ceiling invalid")
    active = mask.astype(bool)
    require(np.all((candidate[active] >= 0) & (candidate[active] < nr))
            and np.all((hap[active] >= 0) & (hap[active] < 2)), "candidate person/haplotype index invalid")
    require(np.all(candidate[~active] == -1) and np.all(hap[~active] == -1), "padded candidates must use -1")
    # No duplicated common haplotype within an ancestry's K lanes.
    for q, j, h, anc in np.ndindex(nq, nj, 2, 3):
        ok = active[q, j, h, anc]
        people, lanes = candidate[q, j, h, anc][ok], hap[q, j, h, anc][ok]
        require(np.all(ancestry[people] == anc), "candidate ancestry differs from reference")
        require(len(set(zip(people.tolist(), lanes.tolist()))) == len(people), "duplicated candidate haplotype")
    for dosage_name, mask_name, shape in (("ref_dosage", "ref_observed", (nj, nr)),
                                           ("query_dosage", "query_observed", (nq, nj))):
        dosage, observed = a[dosage_name], a[mask_name]
        _integer(dosage, dosage_name)
        _binary(observed, mask_name)
        require(dosage.shape == observed.shape == shape, f"{dosage_name}: diploid axes differ")
        require(np.all((dosage >= 0) & (dosage <= 2)) and np.all(dosage[observed == 0] == 0),
                f"{dosage_name}: invalid diploid dosage or missing encoding")


def build_factorized_store(data: Mapping, candidates: Mapping) -> "OrderedContextStore":
    """Bind authenticated upstream factors without reconstructing retrieval.

    The caller authenticates VCF/factor sources and subsets query roles by key
    before calling materialize_radius. Inputs are never mutated. Common arrays
    are shared read-only views in memory; the caller must not mutate their backing
    arrays until save() completes. Persistence creates independent NPY files.
    """
    require(set(data) <= _DATA_FIELDS and set(candidates) <= _CANDIDATE_FIELDS, "undeclared input fields")
    anchors = _array(candidates["anchor_indices"], "anchor_indices")
    _integer(anchors, "anchor_indices")
    require(anchors.ndim == 1 and len(anchors) > 0 and np.all(anchors >= 0)
            and np.all(anchors < len(data["selected"]["locus_id"])), "anchor index out of range")
    for name in ("sample_key_sha256", "reference_sample_key_sha256", "reference_ancestry"):
        require(np.array_equal(data[name], candidates[name]), f"{name}: upstream axes differ")
    require(np.array_equal(candidates["ancestry"], np.asarray(["AFR", "EUR", "NAM"], dtype="S4")),
            "ancestry labels/order differ")
    require(np.array_equal(candidates["rare_semantics"],
                           np.asarray(["diploid_reference_feature_not_phased_allele"], dtype="S64")),
            "rare dosage semantics differ")
    arrays = {k: data[k] for k in ("common_ref", "common_target", "common_cm", "reference_ancestry",
                                  "common_pos", "common_ref_allele", "common_alt_allele", "common_locus_id",
                                  "sample_key_sha256", "reference_sample_key_sha256")}
    arrays.update({k: candidates[k] for k in ("candidate_ref_index", "candidate_hap", "candidate_mask",
                                             "anchor_indices", "radius_cm")})
    for key in ("locus_id", "chrom", "pos", "ref", "alt", "cM"):
        expected = data["selected"][key][anchors]
        require(np.array_equal(candidates[key], expected), f"{key}: selected locus identity differs")
        arrays[key] = expected
    arrays["minor_is_alt"] = data["minor_is_alt"][anchors]
    require(np.array_equal(arrays["minor_is_alt"], candidates["minor_is_alt"]), "minor orientation differs")
    nr = len(data["reference_sample_key_sha256"])
    arrays["reference_hap_original_index"] = data.get(
        "reference_hap_original_index", np.tile(np.arange(2, dtype=np.int8), (nr, 1)))
    arrays["reference_hap_canonicalized"] = np.asarray(
        ["reference_hap_original_index" in data], dtype=np.uint8)
    arrays["ref_dosage"] = data["ref_dosage"][anchors]
    arrays["ref_observed"] = data["ref_observed"][anchors]
    arrays["query_dosage"] = data["query_dosage"][:, anchors]
    arrays["query_observed"] = data["query_observed"][:, anchors]
    _validate_arrays(arrays)
    for name in ("query_dosage", "query_observed"):
        require(np.array_equal(arrays[name], candidates[name]), f"{name}: attached values differ")
    for q, j in np.ndindex(len(arrays["sample_key_sha256"]), len(anchors)):
        mask = arrays["candidate_mask"][q, j].astype(bool)
        people = arrays["candidate_ref_index"][q, j][mask]
        for source, attached in (("ref_dosage", "ref_dosage"), ("ref_observed", "rare_ref_observed")):
            require(candidates[attached].shape == arrays["candidate_mask"].shape,
                    f"{attached}: attachment axes differ")
            require(np.array_equal(candidates[attached][q, j][mask], arrays[source][j, people]),
                    f"{attached}: attached diploid values differ")
            require(np.all(candidates[attached][q, j][~mask] == 0), f"{attached}: padding differs")
    return OrderedContextStore(arrays)


@dataclass(frozen=True)
class OrderedWindow:
    """One query/anchor block; channels are uint8 [2,3,K,sites,5]."""
    query_index: int
    anchor_index: int
    anchor_source_index: int
    site_offset: int
    window_site_count: int
    common_site_index: np.ndarray
    delta_cm: np.ndarray
    channels: np.ndarray
    candidate_ref_index: np.ndarray
    candidate_hap: np.ndarray
    candidate_original_hap: np.ndarray
    candidate_mask: np.ndarray
    ref_dosage: np.ndarray
    ref_observed: np.ndarray
    query_dosage: int
    query_observed: bool

    @property
    def nbytes(self) -> int:
        return sum(value.nbytes for value in self.__dict__.values() if isinstance(value, np.ndarray))


class OrderedContextStore:
    """Read-only factors with hash-authenticated mmap persistence."""

    def __init__(self, arrays: Mapping[str, np.ndarray]):
        _validate_arrays(arrays)
        views = {}
        for name, value in arrays.items():
            view = value.view()
            view.flags.writeable = False
            views[name] = view
        self.arrays = MappingProxyType(views)

    @property
    def factorized_nbytes(self) -> int:
        return sum(value.nbytes for value in self.arrays.values())

    @property
    def shape(self) -> tuple[int, int, int]:
        shape = self.arrays["candidate_mask"].shape
        return shape[0], shape[1], shape[-1]

    def window_bounds(self, anchor_index: int) -> tuple[int, int]:
        require(type(anchor_index) in (int, np.int32, np.int64) and 0 <= anchor_index < self.shape[1],
                "anchor index out of range")
        cm, center = self.arrays["common_cm"], float(self.arrays["cM"][anchor_index])
        radius = float(self.arrays["radius_cm"][0])
        # Match upstream abs(cm-center)<=radius, including floating boundary ties.
        lo = int(np.searchsorted(cm, np.nextafter(center - radius, -np.inf), side="left"))
        hi = int(np.searchsorted(cm, np.nextafter(center + radius, np.inf), side="right"))
        while lo < hi and abs(float(cm[lo]) - center) > radius:
            lo += 1
        while hi > lo and abs(float(cm[hi - 1]) - center) > radius:
            hi -= 1
        return lo, hi

    def window_working_bytes(self, site_count: int) -> int:
        """Conservative per-window array allocation ceiling, excluding shared mmap.

        Includes returned channels/identities plus comparison/gather temporaries;
        not a bound on total interpreter RSS or unrelated callers' allocations.
        """
        require(type(site_count) is int and site_count >= 0, "invalid site count")
        slots = 6 * self.shape[-1]
        return 4096 + slots * 64 + site_count * (slots * 32 + 64)

    def read_window(self, query_index: int, anchor_index: int, *, max_window_bytes: int,
                    site_start: int = 0, site_stop: int | None = None) -> OrderedWindow:
        require(type(query_index) in (int, np.int32, np.int64) and 0 <= query_index < self.shape[0],
                "query index out of range")
        require(type(max_window_bytes) is int and max_window_bytes > 0, "invalid window byte ceiling")
        lo, hi = self.window_bounds(anchor_index)
        total = hi - lo
        stop = total if site_stop is None else site_stop
        require(type(site_start) is int and type(stop) is int and 0 <= site_start <= stop <= total,
                "site chunk bounds invalid")
        count = stop - site_start
        require(self.window_working_bytes(count) <= max_window_bytes, "window exceeds byte ceiling; use site chunks")
        a = self.arrays
        people = a["candidate_ref_index"][query_index, anchor_index]
        haps = a["candidate_hap"][query_index, anchor_index]
        valid = a["candidate_mask"][query_index, anchor_index].astype(bool)
        safe_people, safe_haps = np.where(valid, people, 0), np.where(valid, haps, 0)
        sites = np.arange(lo + site_start, lo + stop, dtype=np.int64)
        refs = a["common_ref"][sites[:, None, None, None], safe_people[None], safe_haps[None]]
        refs = np.moveaxis(refs, 0, -1)
        query = a["common_target"][sites, query_index, :].T[:, None, None, :]
        available = valid[..., None]
        joint = (refs >= 0) & (query >= 0) & available
        channels = np.zeros((*valid.shape, count, len(CHANNEL_NAMES)), dtype=np.uint8)
        channels[..., 0] = (refs == query) & joint
        channels[..., 1] = (refs != query) & joint
        channels[..., 2] = (query < 0) & available
        channels[..., 3] = (refs < 0) & available
        channels[..., 4] = joint
        original_haps = np.where(valid, a["reference_hap_original_index"][safe_people, safe_haps], -1).astype(np.int8)
        dosage = np.where(valid, a["ref_dosage"][anchor_index, safe_people], 0).astype(np.int8)
        observed = valid & a["ref_observed"][anchor_index, safe_people].astype(bool)
        result = OrderedWindow(
            int(query_index), int(anchor_index), int(a["anchor_indices"][anchor_index]), site_start, total,
            sites, a["common_cm"][sites] - float(a["cM"][anchor_index]), channels,
            people, haps, original_haps, valid, dosage, observed,
            int(a["query_dosage"][query_index, anchor_index]), bool(a["query_observed"][query_index, anchor_index]))
        for value in result.__dict__.values():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
        return result

    def iter_windows(self, *, max_window_bytes: int, site_chunk_size: int,
                     query_indices: Sequence[int] | None = None,
                     anchor_indices: Sequence[int] | None = None) -> Iterator[OrderedWindow]:
        """Yield stable query-major/anchor-major chunks, including empty windows.

        Consumers must release each yielded chunk rather than collecting all of
        them. site_chunk_size is a ceiling; the byte budget can lower it further.
        """
        require(type(site_chunk_size) is int and site_chunk_size > 0, "invalid site chunk size")
        require(type(max_window_bytes) is int and max_window_bytes >= self.window_working_bytes(0),
                "window byte ceiling below fixed metadata")
        per_site = self.window_working_bytes(1) - self.window_working_bytes(0)
        affordable = (max_window_bytes - self.window_working_bytes(0)) // per_site
        queries = range(self.shape[0]) if query_indices is None else query_indices
        anchors = tuple(range(self.shape[1]) if anchor_indices is None else anchor_indices)
        for q in queries:
            for j in anchors:
                lo, hi = self.window_bounds(j)
                count = hi - lo
                if count == 0:
                    yield self.read_window(q, j, max_window_bytes=max_window_bytes)
                    continue
                require(affordable > 0, "window byte ceiling cannot hold one site")
                step = min(site_chunk_size, affordable)
                for offset in range(0, count, step):
                    yield self.read_window(q, j, max_window_bytes=max_window_bytes,
                                           site_start=offset, site_stop=min(count, offset + step))

    def save(self, directory: Path, *, source_hashes: Mapping[str, str]) -> dict:
        """Write an exclusive store; manifest is the final completion marker.

        On failure a partial directory is retained without a valid completion
        manifest. Nothing is overwritten or removed. Source authentication is
        the caller's responsibility; supplied digests are bound into the manifest.
        """
        directory = Path(directory)
        require(not directory.exists() and not directory.is_symlink(), "store output already exists")
        require(source_hashes and all(isinstance(k, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", k)
                                     and isinstance(v, str) and _SHA.fullmatch(v)
                                     for k, v in source_hashes.items()), "invalid source hashes")
        directory.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory.mkdir(mode=0o700, parents=False, exist_ok=False)
        descriptors = {}
        for name, value in sorted(self.arrays.items()):
            path = directory / f"{name}.npy"
            with path.open("xb") as handle:
                np.save(handle, value, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(path, 0o400)
            descriptors[name] = {"file": path.name, "sha256": sha256_file(path), "size": path.stat().st_size,
                                 "dtype": value.dtype.str, "shape": list(value.shape), "nbytes": value.nbytes}
        manifest = {"schema_version": SCHEMA, "scope": SCOPE, "channel_names": list(CHANNEL_NAMES),
                    "rare_phase": "discarded_diploid_dosage_only", "source_sha256": dict(source_hashes),
                    "arrays": descriptors}
        write_exclusive_json(directory / MANIFEST_NAME, manifest)
        return {"manifest_sha256": sha256_file(directory / MANIFEST_NAME),
                "factorized_nbytes": self.factorized_nbytes,
                "file_bytes": sum(x["size"] for x in descriptors.values()),
                "array_count": len(descriptors)}

    @classmethod
    def open(cls, directory: Path, *, expected_manifest_sha256: str) -> "OrderedContextStore":
        """Authenticate every file before np.load, then mmap it read-only."""
        directory = Path(directory)
        require(directory.is_dir() and not directory.is_symlink(), "store directory missing or symlinked")
        require(isinstance(expected_manifest_sha256, str) and _SHA.fullmatch(expected_manifest_sha256),
                "invalid expected manifest hash")
        manifest_path = directory / MANIFEST_NAME
        require(manifest_path.is_file() and not manifest_path.is_symlink()
                and sha256_file(manifest_path) == expected_manifest_sha256, "manifest hash mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        require(set(manifest) == {"schema_version", "scope", "channel_names", "rare_phase", "source_sha256", "arrays"}
                and manifest["schema_version"] == SCHEMA and manifest["scope"] == SCOPE
                and manifest["channel_names"] == list(CHANNEL_NAMES)
                and manifest["rare_phase"] == "discarded_diploid_dosage_only", "store manifest contract differs")
        require(set(manifest["arrays"]) == _ARRAY_FIELDS, "manifest array inventory differs")
        require(manifest["source_sha256"] and all(isinstance(v, str) and _SHA.fullmatch(v)
                                                  for v in manifest["source_sha256"].values()), "invalid source hashes")
        expected_files = {MANIFEST_NAME}
        for name, descriptor in manifest["arrays"].items():
            require(set(descriptor) == {"file", "sha256", "size", "dtype", "shape", "nbytes"}
                    and descriptor["file"] == f"{name}.npy", "invalid array descriptor/path")
            path = directory / descriptor["file"]
            expected_files.add(path.name)
            require(path.is_file() and not path.is_symlink() and path.stat().st_size == descriptor["size"]
                    and sha256_file(path) == descriptor["sha256"], f"array hash/size mismatch: {name}")
        require({p.name for p in directory.iterdir()} == expected_files, "store file inventory differs")
        arrays = {}
        for name, descriptor in manifest["arrays"].items():
            path = directory / descriptor["file"]
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            require(value.dtype.str == descriptor["dtype"] and list(value.shape) == descriptor["shape"]
                    and value.nbytes == descriptor["nbytes"], f"array metadata mismatch: {name}")
            arrays[name] = value
        result = cls(arrays)
        # Detect concurrent replacement during authentication/loading.
        require(sha256_file(manifest_path) == expected_manifest_sha256, "manifest changed during load")
        for name, descriptor in manifest["arrays"].items():
            require(sha256_file(directory / descriptor["file"]) == descriptor["sha256"],
                    f"array changed during load: {name}")
        return result
