#!/usr/bin/env python3
"""A synthetic cohort where endogamy and recent pedigree are separate, known facts.

The fixture M27D already had validates one regime: a large population with families
inside it.  That is the biobank case PC-AiR and PC-Relate were designed for, and the
audit passes it.  It says nothing about the regime that actually dominates the panel —
a deme of six people, all similar to one another because their ancestors were few, and
nobody in it a recent relative of anybody else.

Those two causes produce the same estimate from a moment estimator, and the whole design
of M27D rests on the claim that conditioning on principal components removes the second
while keeping the first.  Nothing has ever tested that claim, so this builds a cohort
where the answer is known before the pipeline runs.

Generative model, hierarchical Balding-Nichols:

    ancestral p ~ U(0.05, 0.95)
      |-- background group 1, group 2      drift F_background, unrelated individuals
      `-- intermediate branch              drift F_intermediate
            |-- deme A                     drift F_deme, unrelated individuals
            |-- deme B                     drift F_deme, unrelated individuals
            `-- deme C                     drift F_deme, and the pedigree lives here

Inside a deme, allele frequencies are drawn once and individuals are then independent
draws from them.  That is endogamy without genealogy: two members are similar because
their deme drifted, not because they share a recent ancestor.  Deme C additionally
carries a parent-offspring pair and a half-sibling pair built by explicit haplotype
transmission, so a real pedigree sits inside an endogamous deme — the hard case, and the
one the panel actually contains.

The truth labels distinguish the pedigree component from the coancestry component of
every pair, because the question is not whether phi is estimated accurately but whether
the two causes end up on opposite sides of the threshold.  That classification survives
even if the mapping between the drift parameter and the estimand carries a factor this
module got wrong, which is why the decision rests on it rather than on absolute error.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

ALLELES = (("A", "G"), ("C", "T"), ("G", "A"), ("T", "C"))
UNRELATED = "unrelated"
PARENT_OFFSPRING = "parent_offspring"
HALF_SIBLING = "half_sibling"
FIRST_COUSIN = "first_cousin"
# Expected pedigree kinship, before any coancestry the deme contributes on top.
PEDIGREE_PHI = {
    PARENT_OFFSPRING: 0.25,
    HALF_SIBLING: 0.125,
    FIRST_COUSIN: 0.0625,
    UNRELATED: 0.0,
}


@dataclass(frozen=True)
class Scenario:
    """One point of the coancestry screen.

    ``f_background`` separates the two large groups so the leading components have real
    continental-scale structure to find.  ``f_intermediate`` is shared by the three demes
    and is what makes two different demes look related to each other.  ``f_deme`` is each
    deme's own drift on top of that, and is what makes two members of one deme look
    related.  Setting both deme terms to zero turns the demes into labels with no genetic
    content, which is the null the screen needs at one end.
    """

    name: str
    f_background: float
    f_intermediate: float
    f_deme: float
    rationale: str = ""

    @property
    def within_deme_coancestry(self) -> float:
        return 1.0 - (1.0 - self.f_intermediate) * (1.0 - self.f_deme)

    @property
    def between_deme_coancestry(self) -> float:
        return self.f_intermediate

    @property
    def within_background_group_coancestry(self) -> float:
        """Two members of one background group share its drift, and that is not zero.

        Labelling them zero buried the best available positive control: a group of fifty
        with the same drift the method is asked to remove from a group of six. The two are
        the same physical quantity, and the only difference is how representable the group
        is — which is exactly what is under test, so calling one of them "nothing" made the
        comparison circular.
        """
        return self.f_background


# One pedigree unit is father, mother, a second mother, and two children: four
# first-degree pairs and one second-degree pair, from five people.  Units are the
# denominator of every sensitivity number, so they are counted rather than assumed.
FIRST_DEGREE_PER_UNIT = 4
SECOND_DEGREE_PER_UNIT = 1
PEOPLE_PER_UNIT = 5


@dataclass
class CohortLayout:
    n_background_per_group: int = 50
    # The screen variable. It sizes the two pure-coancestry demes only: the bias a small
    # deme induces goes as -1/(2*n), so a fixture fixed at six would answer for six and
    # say nothing about the n=2 case the real panel actually contains.
    n_deme_members: int = 6
    # The pedigree deme keeps its own size so the small-deme screen does not silently
    # remove the pedigree it is supposed to carry.
    n_pedigree_deme_members: int = 6
    # Sensitivity is counted over true pairs, not over seeds. One unit per replicate gives
    # four first-degree pairs and one second-degree pair, which is too thin to support any
    # statement, so the background carries several and the deme carries the hard case.
    n_pedigree_units_in_deme: int = 1
    n_pedigree_units_in_background: int = 6
    # Optional third-degree controls.  Their ancestors are latent: only the two first
    # cousins are observed, which lets a cross-fitted analysis exclude both endpoints
    # without leaking their pedigree through sampled parents or grandparents.
    n_first_cousin_pairs_in_deme: int = 0
    n_first_cousin_pairs_in_background: int = 0
    n_markers_per_chromosome: int = 700
    n_chromosomes: int = 22
    n_baseline_shared: int = 7
    demes: tuple[str, ...] = ("DEME_A", "DEME_B", "DEME_C")
    pedigree_deme: str = "DEME_C"
    # Populated by build(); kept here so callers can read the layout without re-deriving.
    samples: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        needed = (
            self.n_pedigree_units_in_deme * PEOPLE_PER_UNIT
            + self.n_first_cousin_pairs_in_deme * 2
        )
        if self.n_pedigree_deme_members < needed:
            raise ValueError(
                f"The pedigree deme holds {self.n_pedigree_deme_members} people but "
                f"{needed} are needed for {self.n_pedigree_units_in_deme} pedigree unit(s)"
            )
        background_needed = (
            self.n_pedigree_units_in_background * PEOPLE_PER_UNIT
            + self.n_first_cousin_pairs_in_background * 2
        )
        if self.n_background_per_group < background_needed:
            raise ValueError(
                "One background group is too small to hold its pedigree units"
            )


def beta_frequency(rng: random.Random, ancestral: float, drift: float) -> float:
    """Balding-Nichols draw: a descendant frequency with variance drift*p*(1-p)."""
    if drift <= 0.0:
        return ancestral
    if drift >= 1.0:
        return 1.0 if rng.random() < ancestral else 0.0
    scale = (1.0 - drift) / drift
    alpha = max(ancestral * scale, 1e-6)
    beta = max((1.0 - ancestral) * scale, 1e-6)
    return rng.betavariate(alpha, beta)


def background_ids(layout: CohortLayout) -> list[str]:
    return [f"BG{index:03d}" for index in range(2 * layout.n_background_per_group)]


def deme_ids(layout: CohortLayout, deme: str) -> list[str]:
    prefix = deme.split("_")[-1]
    size = (
        layout.n_pedigree_deme_members if deme == layout.pedigree_deme
        else layout.n_deme_members
    )
    return [f"D{prefix}{index:02d}" for index in range(size)]


def pedigree_units(layout: CohortLayout) -> list[dict[str, str]]:
    """Every pedigree unit, and where it sits.

    Units in the background are the reference: a panmictic surround is the easy case, and
    if a relative is lost there the problem is the estimator, not the deme.  Units inside
    the endogamous deme are the hard case the real panel contains, and the contrast
    between the two locations is what separates "the method fails" from "the method fails
    where the deme is small".
    """
    units: list[dict[str, str]] = []
    members = deme_ids(layout, layout.pedigree_deme)
    for index in range(layout.n_pedigree_units_in_deme):
        block = members[index * PEOPLE_PER_UNIT: (index + 1) * PEOPLE_PER_UNIT]
        units.append(_unit(block, "deme", layout.pedigree_deme))
    background = background_ids(layout)
    for index in range(layout.n_pedigree_units_in_background):
        block = background[len(background) - (index + 1) * PEOPLE_PER_UNIT:
                           len(background) - index * PEOPLE_PER_UNIT]
        units.append(_unit(block, "background", "BACKGROUND"))
    return units


def first_cousin_units(layout: CohortLayout) -> list[dict[str, str]]:
    """Observed first-cousin pairs whose connecting pedigree remains latent.

    Keeping the ancestors outside the cohort prevents their genotypes from entering a
    PCA or PC-Relate training set.  Each pair is generated from its own grandparents,
    sibling parents and unrelated spouses, so different units are independent conditional
    on the population frequency at a marker.
    """
    units: list[dict[str, str]] = []
    deme_members = deme_ids(layout, layout.pedigree_deme)
    deme_start = layout.n_pedigree_units_in_deme * PEOPLE_PER_UNIT
    for index in range(layout.n_first_cousin_pairs_in_deme):
        block = deme_members[deme_start + 2 * index: deme_start + 2 * (index + 1)]
        units.append(_first_cousin_unit(block, "deme", layout.pedigree_deme))

    background = background_ids(layout)
    nuclear_start = len(background) - layout.n_pedigree_units_in_background * PEOPLE_PER_UNIT
    cousin_start = nuclear_start - 2 * layout.n_first_cousin_pairs_in_background
    for index in range(layout.n_first_cousin_pairs_in_background):
        block = background[cousin_start + 2 * index: cousin_start + 2 * (index + 1)]
        units.append(
            _first_cousin_unit(block, "background", background_group(layout, block[0]))
        )
    return units


def _first_cousin_unit(block: list[str], location: str, group: str) -> dict[str, str]:
    if len(block) != 2:
        raise ValueError(f"A first-cousin unit needs two observed people, got {block}")
    return {
        "cousin_1": block[0],
        "cousin_2": block[1],
        "location": location,
        "group": group,
    }


def _unit(block: list[str], location: str, group: str) -> dict[str, str]:
    father, mother, second_mother, child, half_sibling = block
    return {
        "father": father,
        "mother": mother,
        "second_mother": second_mother,
        "child": child,
        "half_sibling": half_sibling,
        "location": location,
        "group": group,
    }


def offspring_of(layout: CohortLayout) -> dict[str, tuple[str, str]]:
    """Children mapped to their two parents, which is who inherits from whom."""
    mapping: dict[str, tuple[str, str]] = {}
    for unit in pedigree_units(layout):
        mapping[unit["child"]] = (unit["father"], unit["mother"])
        mapping[unit["half_sibling"]] = (unit["father"], unit["second_mother"])
    return mapping


def all_samples(layout: CohortLayout) -> list[str]:
    samples = background_ids(layout)
    for deme in layout.demes:
        samples.extend(deme_ids(layout, deme))
    return samples


def background_group(layout: CohortLayout, sample: str) -> str:
    """Which of the two background groups a sample belongs to, or none."""
    if not sample.startswith("BG"):
        return "NONE"
    return "BG1" if int(sample[2:]) < layout.n_background_per_group else "BG2"


def deme_of(layout: CohortLayout, sample: str) -> str:
    for deme in layout.demes:
        if sample in deme_ids(layout, deme):
            return deme
    return "BACKGROUND"


def build_marker(
    rng: random.Random, layout: CohortLayout, scenario: Scenario, samples: list[str]
) -> list[int]:
    """Genotype dosages for one marker, drawn down the hierarchy."""
    ancestral = rng.uniform(0.05, 0.95)
    group_frequency = [
        beta_frequency(rng, ancestral, scenario.f_background),
        beta_frequency(rng, ancestral, scenario.f_background),
    ]
    intermediate = beta_frequency(rng, ancestral, scenario.f_intermediate)
    deme_frequency = {
        deme: beta_frequency(rng, intermediate, scenario.f_deme) for deme in layout.demes
    }

    offspring = offspring_of(layout)
    cousin_units = first_cousin_units(layout)
    cousin_members = {
        unit[key]
        for unit in cousin_units
        for key in ("cousin_1", "cousin_2")
    }
    haplotypes: dict[str, tuple[int, int]] = {}
    for position, sample in enumerate(background_ids(layout)):
        if sample in offspring or sample in cousin_members:
            continue
        frequency = group_frequency[0 if position < layout.n_background_per_group else 1]
        haplotypes[sample] = (int(rng.random() < frequency), int(rng.random() < frequency))
    for deme in layout.demes:
        frequency = deme_frequency[deme]
        for sample in deme_ids(layout, deme):
            if sample in offspring or sample in cousin_members:
                continue
            haplotypes[sample] = (
                int(rng.random() < frequency),
                int(rng.random() < frequency),
            )
    # Children inherit one haplotype from each parent. That is what makes their kinship a
    # pedigree fact rather than a consequence of the frequency their group drifted to.
    for child, (father, mother) in offspring.items():
        haplotypes[child] = (
            haplotypes[father][rng.randrange(2)],
            haplotypes[mother][rng.randrange(2)],
        )

    for unit in cousin_units:
        if unit["location"] == "background":
            frequency = group_frequency[0 if unit["group"] == "BG1" else 1]
        else:
            frequency = deme_frequency[unit["group"]]
        first, second = _draw_first_cousins(rng, frequency)
        haplotypes[unit["cousin_1"]] = first
        haplotypes[unit["cousin_2"]] = second

    index_of = {sample: position for position, sample in enumerate(samples)}
    row = [0] * len(samples)
    for sample, (left, right) in haplotypes.items():
        row[index_of[sample]] = left + right
    return row


def _draw_diploid(rng: random.Random, frequency: float) -> tuple[int, int]:
    return int(rng.random() < frequency), int(rng.random() < frequency)


def _inherit(
    rng: random.Random, parent_1: tuple[int, int], parent_2: tuple[int, int]
) -> tuple[int, int]:
    return parent_1[rng.randrange(2)], parent_2[rng.randrange(2)]


def _draw_first_cousins(
    rng: random.Random, frequency: float
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Draw two first cousins by explicit Mendelian transmission at one marker."""
    grandparent_1 = _draw_diploid(rng, frequency)
    grandparent_2 = _draw_diploid(rng, frequency)
    sibling_parent_1 = _inherit(rng, grandparent_1, grandparent_2)
    sibling_parent_2 = _inherit(rng, grandparent_1, grandparent_2)
    spouse_1 = _draw_diploid(rng, frequency)
    spouse_2 = _draw_diploid(rng, frequency)
    return (
        _inherit(rng, sibling_parent_1, spouse_1),
        _inherit(rng, sibling_parent_2, spouse_2),
    )


def compose(coancestry: float, pedigree_phi: float) -> float:
    """Total kinship of a pedigree pair sitting on a background of coancestry.

    Two alleles are IBD through the pedigree with probability ``pedigree_phi``; when they
    are not, they are two draws from the group's pool and are IBD with probability
    ``coancestry``.  Adding the two terms instead of composing them overstates the truth
    by ``coancestry * pedigree_phi``, which at a coancestry of 0.20 is 0.05 — the same
    size as the contrast the fixture exists to resolve.
    """
    return coancestry + pedigree_phi * (1.0 - coancestry)


def truth_pairs(layout: CohortLayout, scenario: Scenario) -> list[dict[str, object]]:
    """Every pair, with its pedigree component and its coancestry component apart.

    Keeping the two components separate is the point.  A single expected phi would fold
    them together, and the fixture would then only be able to say whether the estimate is
    close, never whether the method put the two causes on different sides of the line.
    """
    samples = all_samples(layout)
    pedigree: dict[tuple[str, str], tuple[str, str]] = {}
    for unit in pedigree_units(layout):
        for parent in (unit["father"], unit["mother"]):
            pedigree[tuple(sorted((unit["child"], parent)))] = (PARENT_OFFSPRING, unit["location"])
        for parent in (unit["father"], unit["second_mother"]):
            pedigree[tuple(sorted((unit["half_sibling"], parent)))] = (
                PARENT_OFFSPRING, unit["location"]
            )
        pedigree[tuple(sorted((unit["child"], unit["half_sibling"])))] = (
            HALF_SIBLING, unit["location"]
        )
    for unit in first_cousin_units(layout):
        pedigree[tuple(sorted((unit["cousin_1"], unit["cousin_2"])))] = (
            FIRST_COUSIN,
            unit["location"],
        )

    rows: list[dict[str, object]] = []
    for left_index, left in enumerate(samples):
        for right in samples[left_index + 1:]:
            key = tuple(sorted((left, right)))
            relationship, location = pedigree.get(key, (UNRELATED, "none"))
            left_deme, right_deme = deme_of(layout, left), deme_of(layout, right)
            if left_deme == right_deme == "BACKGROUND":
                same_group = background_group(layout, left) == background_group(layout, right)
                if same_group:
                    coancestry_class = "within_background_group"
                    coancestry = scenario.within_background_group_coancestry
                else:
                    coancestry_class, coancestry = "between_background_groups", 0.0
            elif left_deme == right_deme:
                coancestry_class = "within_deme"
                coancestry = scenario.within_deme_coancestry
            elif "BACKGROUND" in (left_deme, right_deme):
                coancestry_class, coancestry = "none", 0.0
            else:
                coancestry_class = "between_demes"
                coancestry = scenario.between_deme_coancestry
            pedigree_phi = PEDIGREE_PHI[relationship]
            rows.append(
                {
                    "ID1": key[0],
                    "ID2": key[1],
                    "true_relationship": relationship,
                    "true_degree": {
                        PARENT_OFFSPRING: 1,
                        HALF_SIBLING: 2,
                        FIRST_COUSIN: 3,
                        UNRELATED: 0,
                    }[relationship],
                    "pedigree_location": location,
                    "pedigree_phi": pedigree_phi,
                    "coancestry_class": coancestry_class,
                    "coancestry_phi": round(coancestry, 6),
                    "total_phi": round(compose(coancestry, pedigree_phi), 6),
                    "deme_1": left_deme,
                    "deme_2": right_deme,
                    "has_recent_kinship": relationship != UNRELATED,
                }
            )
    return rows


def write_vcf(path: Path, chromosome: int, samples: list[str], dosages: list[list[int]]) -> None:
    """Shared by every synthetic cohort in the module, so the format lives in one place."""
    calls = ("0/0", "0/1", "1/1")
    with path.open("w", encoding="utf-8") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write(f"##contig=<ID=chr{chromosome},length=100000000>\n")
        handle.write(
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples) + "\n"
        )
        for marker, row in enumerate(dosages, start=1):
            ref, alt = ALLELES[(marker + chromosome) % len(ALLELES)]
            position = marker * 40000
            handle.write(
                f"chr{chromosome}\t{position}\tchr{chromosome}:{position}\t{ref}\t{alt}\t"
                ".\tPASS\t.\tGT\t" + "\t".join(calls[value] for value in row) + "\n"
            )


def metadata_rows(layout: CohortLayout) -> list[dict[str, str]]:
    rows = []
    for sample in all_samples(layout):
        deme = deme_of(layout, sample)
        background = deme == "BACKGROUND"
        group_a = background and int(sample[2:]) < layout.n_background_per_group
        rows.append(
            {
                "IID": sample,
                "Sample_ID(Aliases)": sample,
                "Illumina_ID": sample,
                "original_IID": sample,
                "Exclude": "FALSE",
                "N_genotypes": "4400",
                "Source": "SOURCE_BG" if background else "SOURCE_DEME",
                "Ancestry": ("African" if group_a else "European") if background
                else "Native_American",
                "Population": ("POP_BG1" if group_a else "POP_BG2") if background else deme,
                "Country": "Kenya" if background else "Brazil",
                "Maximum_unrelated_dataset": "TRUE",
            }
        )
    return rows


def contract_for(layout: CohortLayout, base_preregistration: Path, n_samples: int) -> dict:
    """The preregistration with the fixture's own absolute checkpoints.

    ``pass0_checkpoint`` pins absolute numbers on purpose, so a fixture that inherited the
    panel's would refuse to run.  Everything scientific — thresholds, configurations, the
    axis contract — is left exactly as the real contract declares it.
    """
    contract = json.loads(base_preregistration.read_text(encoding="utf-8"))
    contract["scope"]["official_panel_samples_expected"] = n_samples
    contract["scope"]["baseline_samples_expected"] = layout.n_baseline_shared + 1
    contract["identity_contract"]["expected_shared_baseline_identities"] = layout.n_baseline_shared
    contract["identity_contract"]["joint_autosomal_genotypes_min_per_shared_identity"] = 100
    contract["pass0_checkpoint"] = {
        "expected_pairs": n_samples * (n_samples - 1) // 2,
        "expected_eligible_samples": n_samples,
        "max_wall_minutes": 90,
        "max_peak_rss_gib": 22.4,
    }
    return contract


def build(outdir: Path, base_preregistration: Path, scenario: Scenario,
          layout: CohortLayout, seed: int) -> dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)
    panel_dir = outdir / "panel"
    baseline_dir = outdir / "baseline"
    panel_dir.mkdir(exist_ok=True)
    baseline_dir.mkdir(exist_ok=True)

    samples = all_samples(layout)
    for chromosome in range(1, layout.n_chromosomes + 1):
        # The seed enters per chromosome so a rerun with the same seed is identical and a
        # different seed changes every marker rather than only the first chromosome.
        rng = random.Random(seed * 1000 + chromosome)
        dosages = [
            build_marker(rng, layout, scenario, samples)
            for _ in range(layout.n_markers_per_chromosome)
        ]
        write_vcf(panel_dir / f"panel.{chromosome}.vcf", chromosome, samples, dosages)

        shared = samples[: layout.n_baseline_shared]
        baseline_samples = [f"REF_{name}" for name in shared] + ["REF_ABSENT"]
        extra = random.Random(seed * 2000 + chromosome)
        baseline_rows = [
            [row[samples.index(name)] for name in shared] + [extra.randrange(3)]
            for row in dosages
        ]
        write_vcf(
            baseline_dir / f"baseline.chr{chromosome}.vcf",
            chromosome, baseline_samples, baseline_rows,
        )

    rows = metadata_rows(layout)
    with (outdir / "metadata.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    (outdir / "exclude.bed").write_text("chr6\t25000000\t35000000\n", encoding="utf-8")
    (outdir / "prereg.json").write_text(
        json.dumps(contract_for(layout, base_preregistration, len(samples)),
                   indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    pairs = truth_pairs(layout, scenario)
    with (outdir / "truth_pairs.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(pairs)

    truth = {
        "scenario": {
            "name": scenario.name,
            "f_background": scenario.f_background,
            "f_intermediate": scenario.f_intermediate,
            "f_deme": scenario.f_deme,
            "within_deme_coancestry": round(scenario.within_deme_coancestry, 6),
            "between_deme_coancestry": round(scenario.between_deme_coancestry, 6),
            "rationale": scenario.rationale,
        },
        "seed": seed,
        "n_samples": len(samples),
        "n_pairs": len(pairs),
        "n_markers": layout.n_chromosomes * layout.n_markers_per_chromosome,
        "demes": {deme: deme_ids(layout, deme) for deme in layout.demes},
        "pedigree_units": pedigree_units(layout),
        "first_cousin_units": first_cousin_units(layout),
        "always_excluded_from_training": sorted(
            unit[key]
            for unit in first_cousin_units(layout)
            for key in ("cousin_1", "cousin_2")
        ),
        "n_pedigree_units_in_deme": layout.n_pedigree_units_in_deme,
        "n_pedigree_units_in_background": layout.n_pedigree_units_in_background,
        "n_first_cousin_pairs_in_deme": layout.n_first_cousin_pairs_in_deme,
        "n_first_cousin_pairs_in_background": layout.n_first_cousin_pairs_in_background,
        "n_pairs_by_class": {
            name: sum(1 for row in pairs if row["coancestry_class"] == name)
            for name in ("none", "within_deme", "between_demes")
        },
        "n_pairs_with_recent_kinship": sum(1 for row in pairs if row["has_recent_kinship"]),
        "n_first_degree": sum(1 for row in pairs if row["true_degree"] == 1),
        "n_second_degree": sum(1 for row in pairs if row["true_degree"] == 2),
        "n_true_pairs_by_degree_and_location": {
            f"degree{degree}_{location}": sum(
                1 for row in pairs
                if row["true_degree"] == degree and row["pedigree_location"] == location
            )
            for degree in (1, 2, 3) for location in ("deme", "background")
        },
        "n_baseline_shared": layout.n_baseline_shared,
    }
    (outdir / "truth.json").write_text(
        json.dumps(truth, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return truth
