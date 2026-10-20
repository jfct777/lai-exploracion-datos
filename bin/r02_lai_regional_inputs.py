#!/usr/bin/env python3
"""Prepare and read exact regional LAI predictor views without reading truth.

The preparation step consumes one authenticated ``r02_lai_factorized_predictors_v1``
index.  It writes only geometry and source references.  Every query column belongs
to exactly one core.  CNN views add the exact neighboring query columns needed by
the configured convolution depth.  Attention views close EVENT-to-EVENT edges for
all configured layers before adding COMMON context.

The reader returns counted dosage, observation masks, fixed coordinates, person
roles, query predictors, FLARE probabilities, and the current twelve explicit
COMMON and EVENT predictors consumed by ``TaskModel``.  It does not construct
the five LAI controls, load simulated truth, train a model or select a radius.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r02_lai_event_annotations as event_annotations


SCHEMA = "r02_lai_regional_layout_v1"
RECEIPT_SCHEMA = "r02_lai_regional_layout_receipt_v1"
DEFAULT_SCALES_CM = (0.05, 0.2, 1.0)
FAMILIES = ("lai_cnn", "lai_local_attention")
GRCH38_AUTOSOME_LENGTH_BP = {
    1: 248956422, 2: 242193529, 3: 198295559, 4: 190214555,
    5: 181538259, 6: 170805979, 7: 159345973, 8: 145138636,
    9: 138394717, 10: 133797422, 11: 135086622, 12: 133275309,
    13: 114364328, 14: 107043718, 15: 101991189, 16: 90338345,
    17: 83257441, 18: 80373285, 19: 58617616, 20: 64444167,
    21: 46709983, 22: 50818468,
}
COMMON_FEATURE_NAMES = ("alt_dose_fraction", "fit_alt_af", "ref_A", "ref_C", "ref_G", "ref_T",
                        "alt_A", "alt_C", "alt_G", "alt_T", "bp_fraction", "map_fraction")
EVENT_METADATA_NAMES = ("source_minor_af", "counted_is_alt", "ref_A", "ref_C", "ref_G", "ref_T",
                        "alt_A", "alt_C", "alt_G", "alt_T", "bp_fraction", "map_fraction")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_positive(value, name):
    require(type(value) in (int, float) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0, "invalid " + name)


def balanced_cores(size, chunks):
    """Half-open contiguous cores covering ``range(size)`` once."""
    require(type(size) is int and size > 0, "positive query count required")
    require(type(chunks) is int and 0 < chunks <= size, "invalid core chunk count")
    return [(i * size // chunks, (i + 1) * size // chunks) for i in range(chunks)]


def indices_to_spans(indices):
    values = sorted(set(int(value) for value in indices))
    if not values:
        return []
    spans, start, previous = [], values[0], values[0]
    for value in values[1:]:
        require(value >= 0, "negative array index")
        if value != previous + 1:
            spans.append([start, previous + 1])
            start = value
        previous = value
    spans.append([start, previous + 1])
    return spans


def spans_to_indices(spans, size=None):
    values, previous = [], 0
    for span in spans:
        require(isinstance(span, list) and len(span) == 2
                and all(type(x) is int for x in span), "invalid half-open span")
        start, end = span
        require(0 <= start < end and start >= previous, "overlapping/unsorted span")
        if size is not None:
            require(end <= size, "span exceeds array")
        values.extend(range(start, end)); previous = end
    return values


def _merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def radius_spans(source_cm, target_cm, radius_cm):
    """Indices of sorted ``source_cm`` within radius of any target."""
    _finite_positive(radius_cm, "radius_cm")
    require(all(math.isfinite(float(x)) for x in source_cm), "nonfinite source cM")
    require(all(float(source_cm[i]) <= float(source_cm[i + 1])
                for i in range(len(source_cm) - 1)), "source cM must be sorted")
    intervals = []
    for target in target_cm:
        require(math.isfinite(float(target)), "nonfinite target cM")
        intervals.append([bisect_left(source_cm, float(target) - radius_cm),
                          bisect_right(source_cm, float(target) + radius_cm)])
    return _merge_intervals(intervals)


def _located(cm, chrom, index):
    return math.isfinite(float(cm[index])) and 1 <= int(chrom[index]) <= 22


def query_blocks(cm, chrom, valid, max_gap_cm):
    """Valid query blocks matching ``ContextualModel._convolve_queries``."""
    _finite_positive(max_gap_cm, "query_max_gap_cm")
    require(len(cm) == len(chrom) == len(valid), "query axis lengths differ")
    blocks, block, previous = [], [], None
    for index in range(len(cm)):
        if not _located(cm, chrom, index):
            continue
        if not bool(valid[index]):
            if block:
                blocks.append(block); block = []
            previous = None
            continue
        split = previous is not None and (int(chrom[index]) != int(chrom[previous])
                or float(cm[index]) - float(cm[previous]) > max_gap_cm
                + 2 * 1.1920928955078125e-7 * max(abs(float(cm[index])),
                                                    abs(float(cm[previous])), max_gap_cm))
        if split and block:
            blocks.append(block); block = []
        block.append(index); previous = index
    if block:
        blocks.append(block)
    return blocks


def cnn_query_read_indices(core_start, core_end, cm, chrom, valid_rows,
                           depth, max_gap_cm, blocks_by_person=None):
    """Union exact per-person CNN neighbors with the complete core rows."""
    require(type(depth) is int and depth >= 0, "invalid CNN depth")
    selected = set(range(core_start, core_end))
    blocks_by_person = (blocks_by_person if blocks_by_person is not None else
                        [query_blocks(cm, chrom, valid, max_gap_cm) for valid in valid_rows])
    for blocks in blocks_by_person:
        for block in blocks:
            owned = [offset for offset, index in enumerate(block)
                     if core_start <= index < core_end]
            if owned:
                selected.update(block[max(0, min(owned) - depth):
                                      min(len(block), max(owned) + depth + 1)])
    return sorted(selected)


def attention_event_spans(query_cm, event_cm, radius_cm, depth):
    """Query-to-EVENT edges followed by ``depth`` EVENT attention closures."""
    require(type(depth) is int and depth >= 0, "invalid attention depth")
    selected = spans_to_indices(radius_spans(event_cm, query_cm, radius_cm))
    for _ in range(depth):
        expanded = spans_to_indices(radius_spans(event_cm,
                                                  [event_cm[i] for i in selected],
                                                  radius_cm)) if selected else []
        selected = sorted(set(selected) | set(expanded))
    return indices_to_spans(selected)


def _geometry(family, core, query_cm, query_chrom, valid_rows,
              common_cm, event_cm, scale_cm, depth, query_max_gap_cm,
              valid_any=None, cnn_query_read=None):
    start, end = core
    valid_any = (valid_any if valid_any is not None else
                 [any(bool(row[i]) for row in valid_rows) for i in range(len(query_cm))])
    core_valid = [i for i in range(start, end) if valid_any[i]
                  and _located(query_cm, query_chrom, i)]
    if family == "lai_cnn":
        query_read = (cnn_query_read if cnn_query_read is not None else
                      cnn_query_read_indices(start, end, query_cm, query_chrom,
                                             valid_rows, depth, query_max_gap_cm))
        query_context = [i for i in query_read if valid_any[i]
                         and _located(query_cm, query_chrom, i)]
        event = radius_spans(event_cm, [query_cm[i] for i in query_context], scale_cm)
    elif family == "lai_local_attention":
        query_read = list(range(start, end))
        query_context = core_valid
        event = attention_event_spans([query_cm[i] for i in query_context],
                                      event_cm, scale_cm, depth)
    else:
        raise ValueError("unknown LAI family")
    event_indices = spans_to_indices(event)
    common_targets = [query_cm[i] for i in query_context]
    common_targets.extend(event_cm[i] for i in event_indices)
    common = radius_spans(common_cm, common_targets, scale_cm)
    return {
        "query_core": [start, end],
        "query_read": indices_to_spans(query_read),
        "event_read": event,
        "common_read": common,
        "counts": {"query_core_rows": end - start,
                   "query_read_rows": len(query_read),
                   "event_rows": len(event_indices),
                   "common_rows": len(spans_to_indices(common))},
    }


def _same_axis(array, np, name, equal_nan=False):
    require(array.ndim == 2 and array.shape[0] > 0 and array.shape[1] > 0,
            name + " must be [person,query]")
    expected = np.broadcast_to(array[:1], array.shape)
    require(np.array_equal(array, expected, equal_nan=equal_nan),
            name + " differs between people")
    return array[0]


def prepare_case(index_dir, output_dir, scales=DEFAULT_SCALES_CM, chunks=16,
                 depth=2, query_max_gap_cm=0.01):
    """Build one truth-blind regional layout from an existing predictor index."""
    import numpy as np

    index_dir, output_dir = Path(index_dir), Path(output_dir)
    receipt_path, index_path = index_dir / "receipt.json", index_dir / "indices.npz"
    require(receipt_path.is_file() and index_path.is_file(), "index files absent")
    receipt = json.loads(receipt_path.read_text())
    require(receipt.get("schema") == "r02_lai_factorized_predictors_v1"
            and receipt.get("status") == "PREDICTOR_INDEX_COMPLETE_NOT_TRAINING",
            "factorized index receipt differs")
    require(receipt.get("role") in {"FIT", "SELECT"}
            and receipt.get("evaluate_opened") is False
            and receipt.get("truth_arrays_loaded") is False,
            "only truth-blind FIT/SELECT indices are accepted")
    index_sha256 = sha256(index_path)
    require(index_sha256 == receipt["outputs_sha256"]["indices.npz"],
            "factorized index hash differs")
    for value in scales:
        _finite_positive(value, "scale_cm")
    require(len(set(float(x) for x in scales)) == len(scales), "duplicate scales")
    require(type(depth) is int and depth > 0, "positive depth required")
    _finite_positive(query_max_gap_cm, "query_max_gap_cm")

    with np.load(index_path, allow_pickle=False) as index:
        people = index["person_id"].astype(str)
        common_cm = index["common_position_cm"].astype(float).tolist()
        event_cm = index["event_position_cm"].astype(float).tolist()
        common_chrom = index["common_chromosome"].astype(int)
        event_chrom = index["event_chromosome"].astype(int)
        require(len(common_cm) == receipt["loci"]["common"]
                and len(event_cm) == receipt["loci"]["event"], "catalogue counts differ")
        require(np.all(common_chrom == int(receipt["chromosome"]))
                and np.all(event_chrom == int(receipt["chromosome"])),
                "catalogue chromosome differs")
        receipt_annotation_schema = receipt.get("event_annotation_schema", event_annotations.LEGACY_SCHEMA)
        index_annotation_schema = (str(index["event_annotation_schema"].item())
                                   if "event_annotation_schema" in index.files
                                   else event_annotations.LEGACY_SCHEMA)
        require(receipt_annotation_schema == index_annotation_schema,
                "EVENT annotation schema differs between receipt and index")
        if index_annotation_schema == event_annotations.SCHEMA:
            require(set(event_annotations.TYPED_ARRAY_FIELDS) <= set(index.files),
                    "Versioned EVENT annotation arrays incomplete")
        else:
            require(index_annotation_schema == event_annotations.LEGACY_SCHEMA,
                    "Unknown EVENT annotation schema")
    overlay_desc, flare = receipt["query_overlay"], receipt["flare"]
    require(sha256(overlay_desc["path"]) == overlay_desc["sha256"], "overlay hash differs")
    require(sha256(flare["local_path"]) == flare["sha256"], "FLARE hash differs")
    with np.load(overlay_desc["path"], allow_pickle=False) as query:
        # Deliberate allowlist: truth_state and truth_observed are never read.
        query_people = query["person_id"].astype(str)
        role = query["role"].astype(str)
        components = query["component_id"].astype(str)
        query_cm_array = query["query_cm"]
        query_chrom_array = query["query_chrom"]
        query_bp_array = query["query_pos_bp"]
        query_valid = query["query_valid"]
        query_features = query["query_features"]
    require(np.array_equal(people, query_people), "index/overlay person order differs")
    require(set(role.tolist()) == {receipt["role"]} and len(components) == len(people),
            "overlay roles/components differ")
    require(query_valid.dtype == np.bool_ and query_features.shape[:2] == query_valid.shape,
            "query masks/features differ")
    query_cm = _same_axis(query_cm_array, np, "query_cm", equal_nan=True).astype(float).tolist()
    query_chrom = _same_axis(query_chrom_array, np, "query_chrom").astype(int).tolist()
    _same_axis(query_bp_array, np, "query_pos_bp")
    require(query_valid.shape == (len(people), receipt["queries_per_person"]),
            "query dimensions differ")

    cases = []
    cores = balanced_cores(len(query_cm), chunks)
    valid_rows = query_valid.tolist()
    # Query masks and convolution blocks do not depend on radius.  Computing
    # them once avoids repeating the 128-person query scan for every region and
    # recipe while retaining the individual masks in the eventual reader.
    valid_any = query_valid.any(axis=0).tolist()
    blocks_by_person = [query_blocks(query_cm, query_chrom, row, query_max_gap_cm)
                        for row in valid_rows]
    cnn_reads = [cnn_query_read_indices(start, end, query_cm, query_chrom,
                                        valid_rows, depth, query_max_gap_cm,
                                        blocks_by_person=blocks_by_person)
                 for start, end in cores]
    for family in FAMILIES:
        for scale in sorted(float(x) for x in scales):
            recipe = {"family": family, "common_radius_cm": scale,
                      "rare_radius_cm": scale, "depth": depth,
                      "query_max_gap_cm": query_max_gap_cm if family == "lai_cnn" else None,
                      "geometry_source": "CY_CONTEXT_DESIGN_EXISTING_CANDIDATE_NOT_SELECTED"}
            regions = []
            for number, core in enumerate(cores):
                item = _geometry(family, core, query_cm, query_chrom, valid_rows,
                                 common_cm, event_cm, scale, depth, query_max_gap_cm,
                                 valid_any=valid_any,
                                 cnn_query_read=cnn_reads[number] if family == "lai_cnn" else None)
                item["region_id"] = f"region-{number:02d}"
                regions.append(item)
            cases.append({"recipe": recipe, "regions": regions})

    sources = {
        # Reuse the digest checked above.  The compressed index can be large;
        # rereading it would add I/O without adding an independent check.
        "factorized_index": {"path": str(index_path.resolve()), "sha256": index_sha256},
        "factorized_receipt": {"path": str(receipt_path.resolve()), "sha256": sha256(receipt_path)},
        "query_overlay": overlay_desc,
        "flare_probabilities": {"path": str(Path(flare["local_path"]).resolve()),
                                "sha256": flare["sha256"]},
        "alt_dosage": {"path": str(Path(receipt["genotype_directory"]) / "alt_dosage.npy"),
                       "sha256": receipt["inputs_sha256"][str(Path(receipt["genotype_directory"]) / "alt_dosage.npy")]},
        "genotype_observed": {"path": str(Path(receipt["genotype_directory"]) / "genotype_observed.npy"),
                              "sha256": receipt["inputs_sha256"][str(Path(receipt["genotype_directory"]) / "genotype_observed.npy")]},
    }
    layout = {
        "schema": SCHEMA, "status": "REGIONAL_GEOMETRY_COMPLETE_NO_TRAINING_NO_SELECTION",
        "task_id": receipt["task_id"], "case_id": receipt["case_id"],
        "direction": receipt["direction"], "role": receipt["role"],
        "chromosome": str(receipt["chromosome"]), "people": len(people),
        "queries": len(query_cm), "common_loci": len(common_cm), "event_loci": len(event_cm),
        "query_cores": chunks, "query_ownership": "EVERY_QUERY_ROW_EXACTLY_ONE_CORE",
        "coordinates": "GRCh38_BP_AND_PINNED_GENETIC_CM_NOT_REGION_RENORMALIZED",
        "chromosome_length_bp": GRCH38_AUTOSOME_LENGTH_BP[int(receipt["chromosome"])],
        "map_start_cm": min(float(x) for x in query_cm if math.isfinite(float(x))),
        "map_end_cm": max(float(x) for x in query_cm if math.isfinite(float(x))),
        "missingness": "OBSERVED_MASK_SEPARATE_FROM_ZERO_DOSAGE",
        "truth_arrays_loaded": False, "training_authorized": False,
        "selection_performed": False, "evaluate_opened": False,
        "catalogue_revision_policy": "SOURCE_ARRAYS_AND_HASHES_ARE_PER_LAYOUT;_REBUILD_LAYOUT_FOR_REVISED_CATALOGUE",
        "current_catalogue_limit": receipt.get("catalogue_scope"),
        "feature_policy": "CURRENT_TWELVE_EXPLICIT_PREDICTORS;L3_REFERENCE_FEATURES_AND_FIVE_CONTROLS_REQUIRE_SEPARATE_ADMISSION",
        "event_annotations": {
            "schema": receipt_annotation_schema,
            "policy": "LOCUS_LEVEL_NOT_CONCATENATED_WITH_TWELVE_EVENT_METADATA_FEATURES",
            "unknown_counts": "NEGATIVE_ONE_WITH_EXPLICIT_KNOWN_MASK",
            "known_zero_coverage": "AC_ZERO_AN_ZERO_RETAINED_KNOWN_FREQUENCY_UNDEFINED",
        },
        "network_arrays": {"common_features": list(COMMON_FEATURE_NAMES),
                           "event_metadata": list(EVENT_METADATA_NAMES),
                           "consumer": "bin/r02_context_train.py:TaskModel.forward"},
        "sources": sources, "layouts": cases,
    }
    require(not output_dir.exists() and output_dir.parent.is_dir(), "fresh output directory required")
    output_dir.mkdir(mode=0o700)
    layout_path = output_dir / "layout.json"
    layout_path.write_text(json.dumps(layout, indent=2, sort_keys=True, allow_nan=False) + "\n")
    result = {"schema": RECEIPT_SCHEMA,
              "status": "REGIONAL_LAYOUT_PREPARED_NOT_TRAINED_OR_SELECTED",
              "task_id": receipt["task_id"], "layout_sha256": sha256(layout_path),
              "families": list(FAMILIES), "scales_cm": sorted(float(x) for x in scales),
              "regions_per_recipe": chunks, "recipes": len(cases),
              "truth_arrays_loaded": False, "evaluate_opened": False,
              "outputs": {"layout.json": sha256(layout_path)}}
    (output_dir / "receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


class RegionalReader:
    """Read one prepared raw region; NumPy is imported only when instantiated."""
    def __init__(self, layout_path, expected_sha256=None, authenticate=True):
        import numpy as np
        self.np = np; self.path = Path(layout_path)
        if expected_sha256 is not None:
            require(sha256(self.path) == expected_sha256, "layout hash differs")
        self.layout = json.loads(self.path.read_text())
        require(self.layout.get("schema") == SCHEMA, "layout schema differs")
        if authenticate:
            for source in self.layout["sources"].values():
                require(sha256(source["path"]) == source["sha256"], "layout source hash differs")
        self._source_cache = None

    def _sources(self):
        """Open each pinned predictor array once per reader instance.

        The current twelve-feature LAI fields and, for a revised catalogue,
        versioned locus annotations are retained separately.  Simulated truth
        is absent from this allowlist.  Large dosage, missingness and FLARE
        arrays stay memory-mapped.
        """
        if self._source_cache is not None:
            return self._source_cache
        np = self.np; sources = self.layout["sources"]
        index_fields = ("person_id", "common_source_index", "common_is_alt",
            "common_chromosome", "common_position_bp", "common_position_cm",
            "common_ref", "common_alt", "common_fit_alt_ac", "common_fit_an",
            "event_source_index", "event_is_alt", "event_chromosome",
            "event_position_bp", "event_position_cm", "event_ref", "event_alt",
            "event_source_minor_ac", "event_source_an")
        with np.load(sources["factorized_index"]["path"], allow_pickle=False) as source:
            require(set(index_fields) <= set(source.files), "factorized network fields incomplete")
            index = {name: source[name] for name in index_fields}
            schema = self.layout.get("event_annotations", {}).get(
                "schema", event_annotations.LEGACY_SCHEMA)
            if schema == event_annotations.SCHEMA:
                require(set(event_annotations.TYPED_ARRAY_FIELDS) <= set(source.files),
                        "versioned EVENT annotation fields incomplete")
                index.update({name: source[name] for name in event_annotations.TYPED_ARRAY_FIELDS})
                event_annotations.validate_index_arrays(index, np)
            else:
                require(schema == event_annotations.LEGACY_SCHEMA,
                        "unknown layout EVENT annotation schema")
        query_fields = ("person_id", "role", "component_id", "query_features",
                        "query_cm", "query_chrom", "query_pos_bp", "query_valid")
        with np.load(sources["query_overlay"]["path"], allow_pickle=False) as source:
            require(set(query_fields) <= set(source.files), "query predictor fields incomplete")
            query = {name: source[name] for name in query_fields}
        # truth_state/truth_observed are deliberately never requested above.
        self._source_cache = {
            "index": index, "query": query,
            "alt": np.load(sources["alt_dosage"]["path"], mmap_mode="r", allow_pickle=False),
            "observed": np.load(sources["genotype_observed"]["path"], mmap_mode="r", allow_pickle=False),
            "flare": np.load(sources["flare_probabilities"]["path"], mmap_mode="r", allow_pickle=False),
        }
        return self._source_cache

    def _recipe(self, family, scale_cm):
        found = [x for x in self.layout["layouts"]
                 if x["recipe"]["family"] == family
                 and float(x["recipe"]["common_radius_cm"]) == float(scale_cm)]
        require(len(found) == 1, "layout recipe absent or duplicated")
        return found[0]

    def load_raw(self, family, scale_cm, region_id, person_indices=None):
        """Return raw arrays aligned for a later admitted LAI feature adapter."""
        np = self.np; recipe = self._recipe(family, scale_cm)
        found = [x for x in recipe["regions"] if x["region_id"] == region_id]
        require(len(found) == 1, "region absent or duplicated"); region = found[0]
        pids = list(range(self.layout["people"])) if person_indices is None else [int(x) for x in person_indices]
        require(pids and len(set(pids)) == len(pids)
                and min(pids) >= 0 and max(pids) < self.layout["people"], "invalid person indices")
        qids = spans_to_indices(region["query_read"], self.layout["queries"])
        cids = spans_to_indices(region["common_read"], self.layout["common_loci"])
        eids = spans_to_indices(region["event_read"], self.layout["event_loci"])
        cached = self._sources(); idx = cached["index"]
        person = idx["person_id"][pids].astype(str)
        common_source = idx["common_source_index"][cids]
        event_source = idx["event_source_index"][eids]
        fixed = {}
        for prefix, ids in (("common", cids), ("event", eids)):
            for field in ("is_alt", "chromosome", "position_bp", "position_cm", "ref", "alt"):
                fixed[prefix + "_" + field] = idx[prefix + "_" + field][ids]
        for field in ("common_fit_alt_ac", "common_fit_an"):
            fixed[field] = idx[field][cids]
        for field in ("event_source_minor_ac", "event_source_an"):
            fixed[field] = idx[field][eids]
        alt, observed = cached["alt"], cached["observed"]
        common_alt = np.asarray(alt[np.ix_(pids, common_source)])
        event_alt = np.asarray(alt[np.ix_(pids, event_source)])
        common_valid = np.asarray(observed[np.ix_(pids, common_source)], dtype=bool)
        event_valid = np.asarray(observed[np.ix_(pids, event_source)], dtype=bool)
        common_counted = np.where(fixed["common_is_alt"][None, :], common_alt, 2 - common_alt)
        event_counted = np.where(fixed["event_is_alt"][None, :], event_alt, 2 - event_alt)
        common_counted = np.where(common_valid, common_counted, 0).astype(np.int8)
        event_counted = np.where(event_valid, event_counted, 0).astype(np.int8)
        query = cached["query"]
        result = {"person_id": person, "role": query["role"][pids].astype(str),
                  "component_id": query["component_id"][pids].astype(str),
                  "query_features": query["query_features"][np.ix_(pids, qids, range(query["query_features"].shape[2]))],
                  "query_cm": query["query_cm"][np.ix_(pids, qids)],
                  "query_chrom": query["query_chrom"][np.ix_(pids, qids)],
                  "query_pos_bp": query["query_pos_bp"][np.ix_(pids, qids)],
                  "query_valid": query["query_valid"][np.ix_(pids, qids)]}
        flare = cached["flare"]
        result["baseline_probabilities"] = np.asarray(flare[np.ix_(pids, qids, range(6))])
        start, end = region["query_core"]
        result["query_core"] = np.broadcast_to(np.asarray([start <= i < end for i in qids], bool),
                                                (len(pids), len(qids))).copy()
        bases = "ACGT"
        chrom_length = self.layout["chromosome_length_bp"]
        map_start, map_end = self.layout["map_start_cm"], self.layout["map_end_cm"]
        require(map_end > map_start, "layout map normalization range differs")
        for prefix, ids, dosage, valid in (("common", cids, common_counted, common_valid),
                                            ("event", eids, event_counted, event_valid)):
            result[prefix + "_global_index"] = np.asarray(ids, dtype=np.int64)
            result[prefix + "_dosage"] = dosage; result[prefix + "_valid"] = valid
            for field in ("chromosome", "position_bp", "position_cm", "ref", "alt", "is_alt"):
                result[prefix + "_" + field] = fixed[prefix + "_" + field]
            cm = fixed[prefix + "_position_cm"].astype(float)
            bp = fixed[prefix + "_position_bp"].astype(np.int64)
            ref = fixed[prefix + "_ref"].astype(str); alt_allele = fixed[prefix + "_alt"].astype(str)
            one_hot = np.column_stack(([[int(x == base) for base in bases] for x in ref],
                                       [[int(x == base) for base in bases] for x in alt_allele]))
            geometry = np.column_stack((bp / chrom_length,
                                        (cm - map_start) / (map_end - map_start)))
            coordinates = {
                prefix + "_cm": np.broadcast_to(cm, valid.shape).copy(),
                prefix + "_chrom": np.full(valid.shape, int(self.layout["chromosome"]), np.int64),
                prefix + "_pos_bp": np.broadcast_to(bp, valid.shape).copy(),
                prefix + "_padding": np.zeros(valid.shape, bool),
            }
            result.update(coordinates)
            core_cm = [float(result["query_cm"][0, offset]) for offset, global_index in enumerate(qids)
                       if start <= global_index < end
                       and math.isfinite(float(result["query_cm"][0, offset]))]
            core_site = ((cm >= min(core_cm)) & (cm <= max(core_cm))) if core_cm else np.zeros(len(cm), bool)
            result[prefix + "_core"] = np.broadcast_to(core_site, valid.shape).copy()
        if cids:
            fit_ac = fixed["common_fit_alt_ac"].astype(float)
            fit_an = fixed["common_fit_an"].astype(float)
            require(bool((fit_an > 0).all()) and bool(((fit_ac >= 0) & (fit_ac <= fit_an)).all()),
                    "COMMON FIT allele counts differ")
            cref, calt = fixed["common_ref"].astype(str), fixed["common_alt"].astype(str)
            common_one_hot = np.column_stack(([[int(x == base) for base in bases] for x in cref],
                                               [[int(x == base) for base in bases] for x in calt]))
            cbp = fixed["common_position_bp"].astype(np.int64)
            ccm = fixed["common_position_cm"].astype(float)
            common_meta = np.column_stack((fit_ac / fit_an, common_one_hot,
                                           cbp / chrom_length,
                                           (ccm - map_start) / (map_end - map_start))).astype(np.float32)
            common_alt_fraction = np.where(common_valid, common_alt, 0).astype(np.float32)[..., None] / 2
            result["common_features"] = np.concatenate((common_alt_fraction,
                np.broadcast_to(common_meta, (len(pids), *common_meta.shape))), axis=2)
        else:
            result["common_features"] = np.zeros((len(pids), 0, len(COMMON_FEATURE_NAMES)), np.float32)
        if eids:
            source_ac = fixed["event_source_minor_ac"].astype(float)
            source_an = fixed["event_source_an"].astype(float)
            require(bool((source_an > 0).all()) and bool(((source_ac > 0) & (source_ac <= source_an / 2)).all()),
                    "EVENT source allele counts differ")
            eref, ealt = fixed["event_ref"].astype(str), fixed["event_alt"].astype(str)
            event_one_hot = np.column_stack(([[int(x == base) for base in bases] for x in eref],
                                              [[int(x == base) for base in bases] for x in ealt]))
            ebp = fixed["event_position_bp"].astype(np.int64)
            ecm = fixed["event_position_cm"].astype(float)
            event_meta = np.column_stack((source_ac / source_an,
                                          fixed["event_is_alt"].astype(float), event_one_hot,
                                          ebp / chrom_length,
                                          (ecm - map_start) / (map_end - map_start))).astype(np.float32)
            result["event_metadata"] = np.broadcast_to(event_meta,
                                                        (len(pids), *event_meta.shape)).copy()
        else:
            result["event_metadata"] = np.zeros((len(pids), 0, len(EVENT_METADATA_NAMES)), np.float32)
        # ContextualModel requires at least one tensor column.  An empty
        # neighborhood therefore receives one explicitly invalid padding cell;
        # it never becomes an observed event or COMMON marker.
        for prefix, width in (("common", len(COMMON_FEATURE_NAMES)),
                              ("event", len(EVENT_METADATA_NAMES))):
            if result[prefix + "_valid"].shape[1] == 0:
                result[prefix + "_valid"] = np.zeros((len(pids), 1), bool)
                result[prefix + "_dosage"] = np.zeros((len(pids), 1), np.int8)
                result[prefix + "_cm"] = np.full((len(pids), 1), map_start, np.float64)
                result[prefix + "_chrom"] = np.full((len(pids), 1), int(self.layout["chromosome"]), np.int64)
                result[prefix + "_pos_bp"] = np.ones((len(pids), 1), np.int64)
                result[prefix + "_padding"] = np.ones((len(pids), 1), bool)
                result[prefix + "_core"] = np.zeros((len(pids), 1), bool)
                name = "common_features" if prefix == "common" else "event_metadata"
                result[name] = np.zeros((len(pids), 1, width), np.float32)
        if self.layout.get("event_annotations", {}).get("schema") == event_annotations.SCHEMA:
            result.update(event_annotations.region_arrays(
                idx, eids, np, pad_empty=True, validated=True))
        result["query_global_index"] = np.asarray(qids, dtype=np.int64)
        result["layout_recipe"] = recipe["recipe"]
        result["region_id"] = region_id
        result["raw_catalogue_fields"] = fixed
        return result

    @staticmethod
    def core_predictions(region, predictions):
        """Return only valid, uniquely owned query predictions for loss/scoring."""
        mask = region["query_core"] & region["query_valid"]
        require(tuple(predictions.shape[:2]) == tuple(mask.shape), "prediction/query axes differ")
        indices = region["query_global_index"]
        return predictions[mask], indices[None, :].repeat(mask.shape[0], axis=0)[mask]


def _parse_scales(value):
    result = tuple(float(x) for x in value.split(",") if x.strip())
    require(result, "at least one scale required")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scales-cm", default=",".join(map(str, DEFAULT_SCALES_CM)))
    parser.add_argument("--core-chunks", type=int, default=16)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--query-max-gap-cm", type=float, default=0.01)
    args = parser.parse_args(); os.umask(0o077)
    value = prepare_case(args.index_dir, args.output_dir, _parse_scales(args.scales_cm),
                         args.core_chunks, args.depth, args.query_max_gap_cm)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
