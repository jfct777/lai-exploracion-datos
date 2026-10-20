"""Versioned within-person dose relocation; a mechanistic ablation, not a null test.

Generate one chromosome map before window iteration. The source factory must
yield the same authenticated, excluded 1-D position/dosage/observed chunks twice.
Only nonzero destinations are stored. No source frequencies are recalculated.
"""
from dataclasses import dataclass
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np

SCHEMA = "r02_structure_dose_relocation_v1"


def validate_selected_control(spec, contract, resolved_sha256, arm, seed):
    """Refuse to turn the ablation into a fourth primary-search condition."""
    require(set(spec) == {"schema", "frozen_selection", "primary_contract", "maps_by_request"} and
            spec["schema"] == "r02_structure_post_selection_relocation_v1" and arm == "RARE"
            and contract["objective"] == "structure", "only post-selection RARE relocation is admitted")
    def authenticated(record):
        path = Path(record["path"])
        require(hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"],
                "post-selection evidence hash differs")
        return json.loads(path.read_text())
    selected = authenticated(spec["frozen_selection"])
    primary = authenticated(spec["primary_contract"])
    require(not primary.get("dose_relocation") and all(primary.get(key) == contract.get(key) for key in
            ("objective", "scope", "batches", "persons", "steps", "target_definition", "count_scope",
             "feature_definition", "target_contract_sha256", "dataset_mode", "pair_loss_policy")),
            "control and primary fit must have the same people/data/target/budget")
    require(all(primary.get(key) == contract.get(key) for key in
            ("excluded_intervals", "event_metadata_features", "common_features", "limits")),
            "control and primary features/exclusions/resource limits differ")
    require(selected.get("schema") == "r02_structure_frozen_recipe_for_ablation_v1"
            and selected.get("status") == "PRIMARY_SEARCH_COMPLETE_RECIPE_FROZEN"
            and selected.get("selection_role") == "SELECT"
            and selected.get("resolved_sha256") == resolved_sha256
            and selected.get("steps") == contract["steps"] and selected.get("seed") == seed
            and selected.get("rare_effect_selection") == "report_only"
            and selected.get("comparison_use") == "DEVELOPMENT_NOT_INDEPENDENT_CONFIRMATION",
            "complete frozen primary recipe required; no tuning by relocation effect")
    receipts = [authenticated(record) for record in selected.get("primary_training_receipts", [])]
    require(len(receipts) == 3 and {r.get("arm") for r in receipts} == {"NONE", "COUNTS", "RARE"},
            "three completed primary conditions required")
    for receipt in receipts:
        require(receipt.get("status") == "COMPLETE" and receipt.get("score_role") == "SELECT"
                and not receipt.get("dose_relocation") and receipt.get("steps") == contract["steps"]
                and receipt.get("seed") == seed and receipt.get("resolved_sha256") == resolved_sha256
                and receipt.get("contract_sha256") == spec["primary_contract"]["sha256"],
                "primary condition incomplete or not the same recipe/seed/budget")
    expected = {entry["autosomal_request_sha256"] for entries in contract["batches"].values() for entry in entries}
    require(set(spec["maps_by_request"]) == expected, "all FIT/SELECT input maps required")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sample_indices(n, k, rng):
    """Floyd's exact uniform subset algorithm, O(k) extra memory, no N-array."""
    require(type(n) is int and type(k) is int and 0 <= k <= n, "invalid subset size")
    chosen = set()
    for upper in range(n - k, n):
        draw = int(rng.integers(0, upper + 1))
        chosen.add(upper if draw in chosen else draw)
    return np.asarray(sorted(chosen), dtype=np.int64)


def _chunks(factory, excluded_bp=None):
    previous = 0
    for position, dose, observed in factory():
        position, dose, observed = map(np.asarray, (position, dose, observed))
        require(position.ndim == 1 and dose.shape == observed.shape == position.shape,
                "one person and one ordered chromosome required")
        require(np.issubdtype(position.dtype, np.integer) and observed.dtype == np.bool_,
                "position/mask type differs")
        require(np.all(position > previous) and np.all(np.diff(position) > 0),
                "positions must be positive, unique and globally ordered")
        require(excluded_bp is None or not np.any((position >= excluded_bp[0]) & (position <= excluded_bp[1])),
                "excluded locus reached relocation source, including missing calls")
        require(np.isin(dose[observed], (0, 1, 2)).all(), "observed dosage must be 0/1/2")
        if len(position):
            previous = int(position[-1])
        yield position[observed].astype("<i8"), dose[observed].astype(np.uint8)


@dataclass(frozen=True)
class SparseDoseMap:
    positions: np.ndarray
    doses: np.ndarray
    receipt: dict

    def apply(self, position, observed):
        """Return 0/1/2 on observed loci, zero placeholders elsewhere."""
        position, observed = np.asarray(position), np.asarray(observed)
        require(position.shape == observed.shape and observed.dtype == np.bool_,
                "window position/mask shape differs")
        out = np.zeros(position.shape, dtype=np.uint8)
        if len(self.positions):
            at = np.searchsorted(self.positions, position)
            inside = at < len(self.positions)
            hit = inside & (self.positions[np.minimum(at, len(self.positions)-1)] == position)
            require(not np.any(hit & ~observed), "relocated dose reached a missing/padded site")
            out[hit & observed] = self.doses[at[hit & observed]]
        return out

    def save(self, path):
        """Atomic, idempotent write; existing different content is never replaced."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            other = self.load(path)
            require(other.receipt == self.receipt and np.array_equal(other.positions, self.positions)
                    and np.array_equal(other.doses, self.doses), "existing dose map differs")
            return
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        try:
            with temporary.open("xb") as handle:
                np.savez(handle, positions=self.positions, doses=self.doses,
                         receipt=np.asarray(json.dumps(self.receipt, sort_keys=True)))
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)  # atomic no-clobber publication
            except FileExistsError:
                self.save(path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path, expected_sha256=None):
        path = Path(path)
        if expected_sha256 is not None:
            require(hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256,
                    "saved dose-map hash differs")
        with np.load(path, allow_pickle=False) as data:
            require(set(data.files) == {"positions", "doses", "receipt"}, "dose-map fields differ")
            positions, doses = data["positions"].copy(), data["doses"].copy()
            receipt = json.loads(str(data["receipt"]))
        require(receipt["schema"] == SCHEMA and positions.ndim == 1 and doses.shape == positions.shape
                and np.issubdtype(positions.dtype, np.integer) and np.all(positions > 0)
                and np.all(np.diff(positions) > 0) and np.isin(doses, (1, 2)).all(),
                "invalid sparse dose map")
        require(receipt["map_sha256"] == _map_hash(positions, doses), "dose-map content differs")
        return cls(positions, doses, receipt)


def _map_hash(positions, doses):
    return hashlib.sha256(np.asarray(positions, dtype="<i8").tobytes()
                          + np.asarray(doses, dtype=np.uint8).tobytes()).hexdigest()


def _publish_text(path, text):
    """Atomic no-clobber text publication with idempotent equality check."""
    path = Path(path)
    if path.exists():
        require(path.read_text() == text, "existing published text differs")
        return
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".pending-", dir=path.parent)
    try:
        with os.fdopen(handle, "w") as stream:
            stream.write(text); stream.flush(); os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            require(path.read_text() == text, "concurrent published text differs")
    finally:
        Path(temporary).unlink(missing_ok=True)


def build_map(factory, *, person_id, chromosome, stage, source_sha256, seed=20260930,
              max_nonzero, excluded_bp=None):
    """Two bounded source scans; source identity must come from its verified manifest."""
    require(stage in ("FIT", "SELECT") and str(chromosome) in {str(x) for x in range(1, 23)},
            "only explicit development autosomes allowed")
    require(person_id and len(source_sha256) == 64 and
            all(c in "0123456789abcdef" for c in source_sha256), "verified source identity required")
    require(type(seed) is int and seed >= 0 and type(max_nonzero) is int and max_nonzero >= 0,
            "explicit seed/storage bound required")
    counts = np.zeros(3, dtype=np.int64)
    observed_hash = hashlib.sha256()
    dose_hash = hashlib.sha256()
    require(excluded_bp is None or (len(excluded_bp) == 2 and
            all(type(x) is int for x in excluded_bp) and 0 < excluded_bp[0] <= excluded_bp[1]),
            "invalid exclusion interval")
    for pos, dose in _chunks(factory, excluded_bp):
        observed_hash.update(pos.tobytes())
        dose_hash.update(dose.tobytes())
        counts += np.bincount(dose, minlength=3)
        require(int(counts[1] + counts[2]) <= max_nonzero, "sparse-map resource bound exceeded")
    n, k, n2 = int(counts.sum()), int(counts[1] + counts[2]), int(counts[2])
    identity = dict(schema=SCHEMA, stage=stage, person_id=str(person_id), chromosome=str(chromosome),
                    source_sha256=source_sha256, allowed_observed_sha256=observed_hash.hexdigest(), seed=seed)
    seed_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    rng = np.random.Generator(np.random.PCG64(int(seed_hash, 16)))
    ranks = sample_indices(n, k, rng)
    doses = np.ones(k, dtype=np.uint8)
    doses[sample_indices(k, n2, rng)] = 2
    destinations = np.empty(k, dtype=np.int64)
    old_positions, old_doses, offset = [], [], 0
    second_hash = hashlib.sha256()
    second_dose_hash = hashlib.sha256()
    second_counts = np.zeros(3, dtype=np.int64)
    for pos, dose in _chunks(factory, excluded_bp):
        second_hash.update(pos.tobytes())
        second_dose_hash.update(dose.tobytes())
        second_counts += np.bincount(dose, minlength=3)
        lo, hi = np.searchsorted(ranks, (offset, offset + len(pos)))
        destinations[lo:hi] = pos[ranks[lo:hi] - offset]
        positive = dose > 0
        old_positions.append(pos[positive]); old_doses.append(dose[positive])
        offset += len(pos)
    require(second_hash.hexdigest() == observed_hash.hexdigest() and
            second_dose_hash.hexdigest() == dose_hash.hexdigest() and np.array_equal(counts, second_counts),
            "source changed between the two scans")
    old_positions = np.concatenate(old_positions) if old_positions else np.empty(0, np.int64)
    old_doses = np.concatenate(old_doses) if old_doses else np.empty(0, np.uint8)
    union = np.union1d(old_positions, destinations)
    old = np.zeros(len(union), np.uint8); new = old.copy()
    old[np.searchsorted(union, old_positions)] = old_doses
    new[np.searchsorted(union, destinations)] = doses
    receipt = dict(identity, seed_sha256=seed_hash, numpy_version=np.__version__, rng="PCG64",
                   source_observed_dose_sha256=dose_hash.hexdigest(), excluded_bp=excluded_bp,
                   n_observed=n, dosage_counts=counts.tolist(), n_nonzero=k,
                   changed_loci=int(np.count_nonzero(old != new)), nonzero_union_loci=len(union),
                   alterable=bool(np.count_nonzero(counts) > 1),
                   map_sha256=_map_hash(destinations, doses),
                   evidence_level="MECHANISTIC_ABLATION_NOT_CONDITIONAL_NULL",
                   training_admitted=False)
    return SparseDoseMap(destinations, doses, receipt)


def relocate_view(view, maps):
    """Apply whole-chromosome maps identically to each core and repeated halo."""
    require(set(maps) == set(map(str, view["person_id"])), "exact person maps required")
    result = dict(view)
    result["events"] = dict(view["events"])
    features = view["events"]["features"].copy()
    for index, person in enumerate(map(str, view["person_id"])):
        item = maps[person]
        require(item.receipt["person_id"] == person and
                item.receipt["chromosome"] == str(view["chromosome"]) and
                item.receipt["stage"] == str(view["role"][index]) and
                item.receipt["excluded_bp"] == view["excluded_bp"] and
                item.receipt["source_sha256"] == view["input_lineage"]["chromosome_manifest_sha256"],
                "person/chromosome/stage/source of dose map differs")
        features[index, :, 0] = item.apply(view["events_pos_bp"][index],
                                         view["events"]["valid"][index]) / 2
    result["events"]["features"] = features
    mask = result["events_core"] & result["events"]["valid"]
    dose = features[:, :, 0] * 2
    result["rare_core_counts"] = np.column_stack(((mask & (dose > 0)).sum(1), mask.sum(1),
                                                  np.where(mask, dose, 0).sum(1)))
    result["dose_relocation"] = {person: item.receipt["map_sha256"] for person, item in maps.items()}
    return result


class RelocatedAutosomalDataset:
    """Apply maps from a frozen manifest to an authenticated source dataset."""
    def __init__(self, dataset, request_sha256, manifest_path, manifest_sha256):
        self.dataset = dataset
        self.map_provider = FrozenMapProvider(dataset, request_sha256, manifest_path, manifest_sha256)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def iter_person_batch(self, batch_index):
        for view in self.dataset.iter_person_batch(batch_index):
            maps = self.map_provider(str(view["chromosome"]), tuple(map(str, view["person_id"])),
                                     view["excluded_bp"])
            yield relocate_view(view, maps)

    def __iter__(self):
        for index in range(self.n_person_batches):
            yield from self.iter_person_batch(index)


class FrozenMapProvider:
    """Read exact precomputed maps; never construct a map from a partial window."""
    def __init__(self, dataset, request_sha256, manifest_path, manifest_sha256):
        path = Path(manifest_path).resolve()
        require(dataset.request_sha256 == request_sha256, "dataset/request identity differs")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == manifest_sha256,
                "relocation manifest hash differs")
        manifest = json.loads(path.read_text())
        require(set(manifest) == {"schema", "request_sha256", "seed", "records", "max_cached_bytes"}
                and manifest["schema"] == "r02_structure_dose_maps_v1"
                and manifest["request_sha256"] == request_sha256
                and type(manifest["seed"]) is int and manifest["seed"] >= 0
                and type(manifest["max_cached_bytes"]) is int and manifest["max_cached_bytes"] > 0,
                "explicit whole-request relocation manifest required")
        self.root, self.manifest, self.dataset = path.parent, manifest, dataset
        self.records, self.cache = {}, OrderedDict()
        for row in manifest["records"]:
            require(set(row) == {"person_id", "chromosome", "path", "sha256", "bytes", "excluded_bp"},
                    "relocation record fields differ")
            key = (str(row["chromosome"]), str(row["person_id"]))
            require(key not in self.records, "duplicate person/chromosome map")
            self.records[key] = row
        require(set(self.records) == {(str(c), p) for c in range(1, 23) for p in dataset.ids},
                "relocation maps must cover all people and 22 autosomes")

    def __call__(self, chromosome, person_ids, excluded_bp):
        # Hold only the current person batch/chromosome in memory. The returned
        # maps survive in the caller until that window has been transformed.
        wanted = {(chromosome, person) for person in person_ids}
        for key in list(self.cache):
            if key not in wanted:
                del self.cache[key]
        result, total = {}, 0
        for person in person_ids:
            key = (chromosome, person)
            row = self.records[key]
            require(row["excluded_bp"] == excluded_bp, "map territory differs from input exclusion")
            path = (self.root / row["path"]).resolve()
            require(path.is_relative_to(self.root) and path.is_file() and
                    path.stat().st_size == row["bytes"], "map path/size differs")
            # save() uses ZIP_STORED. Still bound decoded memory for a supplied
            # compressed NPZ: include arrays, temporary copies and receipt cost.
            with zipfile.ZipFile(path) as archive:
                resident_bound = 4 * sum(info.file_size for info in archive.infolist()) + 65536
            total += resident_bound
            require(total <= self.manifest["max_cached_bytes"], "map batch exceeds memory budget")
            stat = path.stat()
            signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            if key not in self.cache:
                item = SparseDoseMap.load(path, row["sha256"])
                require(item.receipt["seed"] == self.manifest["seed"] and
                        item.receipt["person_id"] == person and item.receipt["chromosome"] == chromosome
                        and item.receipt["stage"] == self.dataset.request["stage"] and
                        item.receipt["source_sha256"] == self.dataset.stores[chromosome]["manifest_sha256"]
                        and item.receipt["excluded_bp"] == excluded_bp,
                        "map source/person/stage/seed/territory differs")
                self.cache[key] = (signature, item)
            require(self.cache[key][0] == signature, "map changed after authentication")
            result[person] = self.cache[key][1]
        return result


def generate_dataset_maps(dataset, request_sha256, output, *, seed, max_nonzero,
                          max_output_bytes, max_cached_bytes):
    """Recoverable preparation over authenticated chunks; no targets or fitting.

    Run this in a separate bounded Nextflow task before the selected recipe's
    relocated-dose fit. Existing maps are authenticated/reused, never overwritten.
    """
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    require(dataset.request_sha256 == request_sha256, "dataset/request identity differs")
    require(type(max_output_bytes) is int and max_output_bytes > 0 and
            type(max_cached_bytes) is int and max_cached_bytes > 0, "explicit storage budgets required")
    records = []
    total = sum(p.stat().st_size for p in root.iterdir() if p.is_file())
    require(total <= max_output_bytes, "existing map directory exceeds output budget")
    for chromosome in map(str, range(1, 23)):
        reader = dataset._reader(chromosome)
        block = next(b for b in reader.blocks if b["block_id"] == dataset.request["block_id"])
        excluded = None
        if block["chromosome"] == chromosome:
            excluded = ([1, reader.length] if dataset.request["exclusion_scope"] == "WHOLE_CHROMOSOME"
                        else [int(block["excluded_start_bp"]), int(block["excluded_end_bp"])])
        for person in dataset.ids:
            key = dict(schema=SCHEMA, numpy_version=np.__version__, rng="PCG64",
                       person_id=person, chromosome=chromosome, stage=dataset.request["stage"],
                       source_sha256=reader.manifest_sha256, excluded_bp=excluded, seed=seed)
            key_hash = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
            path = root / f"{key_hash}.npz"
            if path.exists():
                # Its sidecar pins the file, including original-source hashes.
                sidecar = path.with_suffix(".sha256")
                item = SparseDoseMap.load(path, sidecar.read_text().strip() if sidecar.is_file() else None)
                require(all(item.receipt.get(k) == v for k, v in key.items()), "saved map source differs")
                require(item.receipt["n_nonzero"] <= max_nonzero, "saved map exceeds new sparse-map resource bound")
                if not sidecar.is_file():
                    require(total + 65 <= max_output_bytes, "sidecar exceeds output budget")
                    _publish_text(sidecar, hashlib.sha256(path.read_bytes()).hexdigest() + "\n")
                    total += 65
            else:
                def chunks():
                    for chunk in reader.manifest["channels"]["rare"]:
                        values = reader.read("rare", [reader.people[person]], chunk["start_bp"],
                                             chunk["end_bp"], excluded)
                        yield (values["position_bp"].astype(np.int64), values["dosage"][0].astype(np.uint8),
                               values["observed"][0].astype(bool))
                item = build_map(chunks, person_id=person, chromosome=chromosome,
                                 stage=dataset.request["stage"], source_sha256=reader.manifest_sha256,
                                 seed=seed, max_nonzero=max_nonzero, excluded_bp=excluded)
                require(total + item.positions.nbytes + item.doses.nbytes + 16384 <= max_output_bytes,
                        "dose-map output budget exceeded before publication")
                item.save(path)
                _publish_text(path.with_suffix(".sha256"), hashlib.sha256(path.read_bytes()).hexdigest() + "\n")
                total += path.stat().st_size + 65
            size = path.stat().st_size
            require(total <= max_output_bytes, "dose-map output budget exceeded")
            records.append(dict(person_id=person, chromosome=chromosome, path=path.name,
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=size,
                                excluded_bp=excluded))
    manifest = dict(schema="r02_structure_dose_maps_v1", request_sha256=request_sha256, seed=seed,
                    max_cached_bytes=max_cached_bytes, records=records)
    # Different radii/tiles/views reuse the same source/person/exclusion maps;
    # only the small request manifest differs. Do not copy 22 chromosome maps
    # into a new directory merely because a context radius changed.
    path = root / f"manifest-{request_sha256}.json"
    text = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    if path.exists():
        require(path.read_text() == text, "existing relocation manifest differs")
    else:
        require(total + len(text.encode()) <= max_output_bytes, "final manifest exceeds output budget")
        _publish_text(path, text)
    return manifest
