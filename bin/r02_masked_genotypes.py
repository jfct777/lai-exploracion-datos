"""Hide person-by-cell genotypes BEFORE constructing neural features.

This is a new objective, not an adapter for historical pair-regression batches.
Selection is supplied independently of genotypes. This module selects no masking
rate, radius, population, or learning budget and does not authorize training.
"""
from dataclasses import dataclass
import hashlib
import numpy as np

from r02_genomic_event_inputs import ARRAY_FIELDS


def require(ok, message):
    if not ok:
        raise ValueError(message)


@dataclass(frozen=True)
class CellMaskPlan:
    person_ids: tuple[str, ...]
    chromosome: int
    edges_cm: np.ndarray
    selected: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "person_ids", tuple(self.person_ids))
        require(self.person_ids and len(set(self.person_ids)) == len(self.person_ids)
                and all(isinstance(p, str) and p for p in self.person_ids),
                "unique nonempty person identifiers required")
        require(type(self.chromosome) is int and 1 <= self.chromosome <= 22,
                "autosomal chromosome required")
        edges = np.array(self.edges_cm, dtype=np.float64, copy=True)
        selected = np.array(self.selected, copy=True)
        require(edges.ndim == 2 and edges.shape[1] == 2 and len(edges) > 0
                and np.isfinite(edges).all(), "finite cell edges required")
        require(np.all(edges[:, 1] > edges[:, 0])
                and np.array_equal(edges[1:, 0], edges[:-1, 1]),
                "cell edges must be contiguous, ordered, nonempty intervals")
        require(selected.dtype == np.bool_
                and selected.shape == (len(self.person_ids), len(edges)),
                "selection must be boolean person by cell")
        edges.setflags(write=False); selected.setflags(write=False)
        object.__setattr__(self, "edges_cm", edges)
        object.__setattr__(self, "selected", selected)

    @property
    def identity(self):
        import json
        digest = hashlib.sha256(json.dumps({"persons": self.person_ids,
            "chromosome": self.chromosome, "schema": "r02_cell_mask_plan_v1"},
            sort_keys=True).encode())
        digest.update(self.edges_cm.astype("<f8").tobytes())
        digest.update(self.selected.tobytes())
        return digest.hexdigest()

    def indices(self, positions_cm):
        cm = np.asarray(positions_cm)
        require(cm.ndim == 1 and np.issubdtype(cm.dtype, np.number)
                and np.isfinite(cm).all(), "one finite genetic coordinate per locus required")
        require(np.all((cm >= self.edges_cm[0, 0]) & (cm <= self.edges_cm[-1, 1])),
                "locus outside the declared cell axis")
        index = np.searchsorted(self.edges_cm[:, 1], cm, side="right")
        return np.minimum(index, len(self.edges_cm) - 1)


@dataclass(frozen=True)
class MaskedChannel:
    # inputs contains no raw dosage, raw observation mask or unmasked burden.
    inputs: dict
    targets: dict
    catalogue_statistics: dict
    plan_identity: str
    channel: str


def mask_raw_channel(arrays, *, channel, chromosome, person_ids, plan):
    """Return fresh visible arrays and SEPARATE labels; never pass labels to model.

    Alleles stay unchanged. Source AC/AN/frequency are provenance in a SEPARATE
    dictionary, not automatically neural inputs: a future assembler must admit
    any frequency feature under its FIT/validation policy. Mask validity does not
    erase locus metadata; genotype_visible and target_scored have different jobs.
    Only a verified raw-store reader should call this function: shape checks do
    not authenticate source files, roles, family components or cell provenance.
    """
    require(isinstance(plan, CellMaskPlan), "CellMaskPlan required")
    require(tuple(person_ids) == plan.person_ids and chromosome == plan.chromosome,
            "person order or chromosome differs from mask plan")
    require(channel in {"common", "rare"}, "common or rare channel required")
    require(set(arrays) == ARRAY_FIELDS, "raw store fields differ; precomputed features forbidden")
    dose, observed = np.asarray(arrays["dosage"]), np.asarray(arrays["observed"])
    require(dose.ndim == 2 and dose.shape[0] == len(person_ids)
            and dose.dtype == np.uint8 and observed.dtype == np.bool_
            and observed.shape == dose.shape, "raw dosage/observation shape or type differs")
    require(np.all(dose[observed] <= 2) and np.all(dose[~observed] == 255),
            "observed dosage must be 0/1/2; missing dosage must be 255")
    n = dose.shape[1]
    for name in ARRAY_FIELDS - {"dosage", "observed"}:
        require(np.asarray(arrays[name]).shape == (n,), "locus axis differs: " + name)
    for name in ("ref", "alt", "counted_allele", "position_bp", "allele_count", "allele_number"):
        require(np.asarray(arrays[name]).dtype.kind in "iu", "integer metadata required: " + name)
    require(np.all((arrays["ref"] >= 0) & (arrays["ref"] < 4))
            and np.all((arrays["alt"] >= 0) & (arrays["alt"] < 4))
            and np.all(arrays["ref"] != arrays["alt"]), "nucleotide identity differs")
    require(np.all((arrays["counted_allele"] >= 0) & (arrays["counted_allele"] <= 1)),
            "counted allele differs")
    require(np.all(arrays["position_bp"] > 0)
            and np.all(arrays["position_bp"][1:] > arrays["position_bp"][:-1]),
            "positive increasing base-pair coordinates required")
    require(np.all(arrays["allele_number"] > 0) and np.all(arrays["allele_count"] >= 0)
            and np.all(arrays["allele_count"] <= arrays["allele_number"])
            and np.asarray(arrays["frequency"]).dtype.kind in "fiu"
            and np.isfinite(arrays["frequency"]).all()
            and np.allclose(arrays["frequency"], arrays["allele_count"] / arrays["allele_number"],
                            rtol=0, atol=1e-12), "source frequency/count metadata differs")
    if channel == "common":
        require(np.all(arrays["counted_allele"] == 1), "COMMON must count ALT")
    cell_indices = plan.indices(arrays["position_cm"])
    hidden = plan.selected[:, cell_indices]
    visible = observed & ~hidden
    scored = observed & hidden
    statistic_names = {"allele_count", "allele_number", "frequency"}
    statistics = {name: np.array(arrays[name], copy=True) for name in statistic_names}
    inputs = {name: np.array(arrays[name], copy=True)
              for name in ARRAY_FIELDS - {"dosage", "observed"} - statistic_names}
    inputs.update(dosage=np.where(visible, dose, 0).astype(np.uint8),
                  genotype_visible=visible, locus_valid=np.ones(n, dtype=bool),
                  cell_index=cell_indices, chromosome=np.full(n, chromosome, dtype=np.int8))
    # -1 is never a real genotype. Only target_scored entries enter the loss.
    targets = {"dosage": np.where(scored, dose.astype(np.int64), -1),
               "target_scored": scored}
    return MaskedChannel(inputs, targets, statistics, plan.identity, channel)


def mask_both_channels(common, rare, *, chromosome, person_ids, plan):
    """Use the exact same selection in both channels; preserve each allele rule."""
    return {name: mask_raw_channel(data, channel=name, chromosome=chromosome,
                                  person_ids=person_ids, plan=plan)
            for name, data in (("common", common), ("rare", rare))}


def visible_burden(masked):
    """Counts over supplied visible loci, not an implicit whole-genome total."""
    dose = masked.inputs["dosage"]
    visible = masked.inputs["genotype_visible"]
    return np.stack(((dose > 0).sum(1), dose.sum(1, dtype=np.int64),
                     visible.sum(1)), axis=-1).astype(np.int64)
