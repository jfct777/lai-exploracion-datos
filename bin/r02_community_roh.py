#!/usr/bin/env python3
"""Reuse authenticated ROH calls to describe fixed community assignments.

This is not a ROH caller, a population test, or a founder-event estimator.
Individual burdens retain the historical detection limit. Nested least-squares
comparisons are in-sample descriptions, with identical support in both models.
No physical interval is joined to another assembly and no callable genome
denominator is invented. Small-cell and individual outputs remain private.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import json
import math
from pathlib import Path
import re
import resource
import shutil
import time

import numpy as np

from r02_community_levels import ANCESTRY, PHIS, require, sha256
from r02_community_ibd import Ledger, load_people, save_json, strict_json, table, write_table

SCHEMA = "r02_community_roh_v1"
STATUS = "COMPLETE_DESCRIPTIVE_ROH_COMPARISON_NOT_FOUNDER_IDENTIFICATION"
OUTCOMES = ("NSEG", "KB")
LIMITATIONS = [
    "Reuses historical PLINK calls above the recorded 1500 kb minimum; no new ROH inference.",
    "ROH burdens describe homozygosity, not a uniquely identified founder event or event age.",
    "No callable autosomal denominator is authenticated: KB is not F_ROH.",
    "Historical assembly is not certified by the ROH log; no coordinate cross-dataset join is performed.",
    "Fixed transductive communities: partial R2 is in-sample association, not held-out prediction or population validation.",
    "Component balancing changes the target average; connected components are not certified independent families.",
    "Global ancestry is an estimated covariate; collection is not sequencing batch.",
    "Rare-catalogue genotype availability is only a measurement proxy, not ROH callability, DP or GQ.",
    "No p-values, confidence intervals, winner selection, rare increment or demographic model comparison.",
    "Zero ROH is retained only for an explicit zero record in the authenticated individual table.",
    "No new length classes or conversion from physical length to genealogical age.",
    "Individual identifiers, small groups and detailed results remain private.",
]


def whitespace_rows(path, required):
    with Path(path).open() as handle:
        header = handle.readline().split()
        require(header and len(set(header)) == len(header) and set(required) <= set(header),
                "Invalid ROH header")
        for line in handle:
            if not line.strip():
                continue
            values = line.split()
            require(len(values) == len(header), "Ragged ROH record")
            yield dict(zip(header, values))


def finite(value, name):
    result = float(value)
    require(math.isfinite(result) and result >= 0, "Invalid nonnegative " + name)
    return result


def integer(value, name):
    require(re.fullmatch(r"[0-9]+", str(value)) is not None, "Invalid integer " + name)
    return int(value)


def print_tolerance(text):
    """Half the least printed decimal unit, not a biological tolerance."""
    value = Decimal(text)
    require(value.is_finite(), "Nonfinite printed number")
    return float(Decimal(10) ** value.as_tuple().exponent / 2) + 1e-8


def load_roh(indiv_path, hom_path, log_path, samples, chromosomes, minimum_kb):
    require(minimum_kb == 1500, "This reuse contract preserves the historical 1500 kb threshold")
    log = Path(log_path).read_text()
    expected_options = {"homozyg-kb": "1500", "homozyg-snp": "50", "homozyg-density": "50",
                        "homozyg-gap": "1000", "homozyg-window-snp": "50",
                        "homozyg-window-het": "1", "homozyg-window-missing": "5"}
    require("PLINK v1.9.0-b.8" in log and "--homozyg\n" in log, "Unexpected ROH caller/log")
    for name, value in expected_options.items():
        require(re.search(r"^\s*--" + name + r"\s+" + value + r"\s*$", log, re.M),
                "Historical ROH parameter differs: " + name)
    require(re.search(r"^" + str(len(samples)) + r" people \(", log, re.M), "ROH log/cohort mismatch")
    scan = re.search(r"--homozyg: Scan complete, found ([0-9]+) ROH\.", log)
    require(scan is not None, "ROH log has no completion count")
    individuals = {}
    for row in whitespace_rows(indiv_path, ("IID", "NSEG", "KB", "KBAVG")):
        sample = row["IID"]
        require(sample not in individuals, "Duplicate ROH individual")
        individuals[sample] = dict(NSEG=integer(row["NSEG"], "NSEG"), KB=finite(row["KB"], "KB"),
                                   tolerance=print_tolerance(row["KB"]))
    require(set(individuals) == set(samples), "ROH individual universe differs; no missing-to-zero imputation")
    segments, totals, seen_chromosomes, row_keys = defaultdict(list), defaultdict(float), set(), set()
    for row in whitespace_rows(hom_path, ("IID", "CHR", "POS1", "POS2", "KB", "NSNP")):
        sample, chrom = row["IID"], row["CHR"]
        require(sample in individuals and chrom in chromosomes, "Unknown ROH person/chromosome")
        start, end = integer(row["POS1"], "POS1"), integer(row["POS2"], "POS2")
        require(1 <= start <= end, "Invalid ROH coordinates")
        key = sample, chrom, start, end
        require(key not in row_keys, "Duplicated ROH segment")
        row_keys.add(key)
        kb = finite(row["KB"], "KB")
        # PLINK writes inclusive physical lengths, with rounded decimal display.
        require(abs(kb - (end-start+1)/1000) <= print_tolerance(row["KB"]), "ROH kb/bp disagreement")
        require(kb + print_tolerance(row["KB"]) >= minimum_kb and integer(row["NSNP"], "NSNP") >= 50,
                "ROH falls below historical length/marker threshold")
        segments[sample].append((chrom, start, end, kb))
        totals[sample] += print_tolerance(row["KB"])
        seen_chromosomes.add(chrom)
    require(seen_chromosomes == set(chromosomes), "Incomplete ROH chromosome coverage")
    require(len(row_keys) == int(scan[1]), "ROH count/log disagreement")
    results = []
    for sample in samples:
        records = sorted(segments[sample], key=lambda x: (int(x[0]), x[1], x[2]))
        for a, b in zip(records, records[1:]):
            require(a[0] != b[0] or a[2] < b[1], "Overlapping ROH within one person/chromosome")
        row = individuals[sample]
        require(row["NSEG"] == len(records), "Individual/segment NSEG mismatch")
        sum_kb = math.fsum(r[3] for r in records)
        require(abs(row["KB"]-sum_kb) <= totals[sample]+row["tolerance"], "Individual/segment KB mismatch")
        require(row["NSEG"] != 0 or row["KB"] == 0, "Zero segments with nonzero KB")
        lengths = [r[3] for r in records]
        results.append(dict(sample_id=sample, NSEG=row["NSEG"], KB=row["KB"],
                            segment_bp_sum=sum(r[2]-r[1]+1 for r in records),
                            longest_reported_kb=max(lengths) if lengths else 0,
                            median_reported_kb=float(np.median(lengths)) if lengths else None))
    return results, dict(n_samples=len(samples), n_segments=len(row_keys),
        zero_roh_people=sum(r["NSEG"] == 0 for r in results), chromosomes=list(chromosomes),
        caller="PLINK v1.9.0-b.8", historical_options=expected_options,
        genome_build="NOT_CERTIFIED_FROM_LOG_NO_COORDINATE_CROSS_JOIN",
        kb_semantics="Historical printed total validated against inclusive segment coordinates and decimal rounding")


def load_availability(spec, ledger, people, chromosomes):
    manifest_path = ledger.add(spec["manifest"])
    manifest = strict_json(manifest_path)
    path = ledger.add(spec["npz"])
    require(manifest["schema"] == "r02_pair_evidence_v1" and manifest["kind"] == "aggregate" and
            manifest["n_samples"] == len(people["samples"]) and manifest["samples_sha256"] == people["sample_hash"] and
            manifest["chromosomes"] == chromosomes and manifest["output_npz_sha256"] == sha256(path),
            "Availability manifest differs")
    with np.load(path, allow_pickle=False) as data:
        require(data["samples"].tolist() == people["samples"] and data["chromosomes"].tolist() == chromosomes,
                "Availability people/chromosomes differ")
        n_sites = int(data["n_sites"])
        called = data["individual_callable"].astype(float)
    require(n_sites > 0 and called.shape == (len(people["samples"]),) and np.isfinite(called).all() and
            np.all((called >= 0) & (called <= n_sites)), "Invalid callable rare-site vector")
    return called/n_sites, n_sites


def one_hot(values):
    levels = sorted(set(values))
    return np.asarray([[float(v == level) for level in levels[1:]] for v in values], dtype=float).reshape(len(values), -1)


def nested_comparison(y, baseline, labels, weights):
    """Two outcomes share exactly the same people, weights and fitted subspaces."""
    y, x, w = np.asarray(y, float), np.asarray(baseline, float), np.asarray(weights, float)
    require(y.ndim == 2 and y.shape[1] == 2 and x.ndim == 2 and x.shape[0] == y.shape[0] and
            len(labels) == len(y) and w.shape == (len(y),), "Nested model dimensions differ")
    require(np.isfinite(y).all() and np.isfinite(x).all() and np.isfinite(w).all() and np.all(w > 0),
            "Nested model has unknown/nonfinite values")
    root = np.sqrt(w)[:, None]
    full = np.column_stack((x, one_hot(labels)))
    results = []
    for matrix in (x, full):
        weighted = matrix * root
        # Unit-norm columns avoid arbitrary kb/proxy scale affecting SVD rank.
        norms = np.linalg.norm(weighted, axis=0)
        weighted = weighted[:, norms > 0] / norms[norms > 0]
        beta, _, rank, _ = np.linalg.lstsq(weighted, y*root, rcond=None)
        residual = y*root - weighted@beta
        results.append((int(rank), np.sum(residual**2, axis=0)))
    rank0, sse0 = results[0]; rank1, sse1 = results[1]
    require(rank1 >= rank0 and np.all(sse1 <= sse0 + 1e-8*np.maximum(1, sse0)), "Nested fit is not nested")
    records = []
    for j, outcome in enumerate(OUTCOMES):
        status = "DESCRIPTIVE_IN_SAMPLE_PARTIAL_R2"
        if len(y) <= rank1:
            status = "NO_RESIDUAL_DEGREES_OF_FREEDOM"
        elif rank1 == rank0:
            status = "COMMUNITY_NOT_SEPARABLE_FROM_BASELINE"
        elif sse0[j] <= 1e-20 * max(1., float(np.sum(w*y[:, j]**2))):
            status = "NO_BASELINE_RESIDUAL_VARIATION"
        value = None if status != "DESCRIPTIVE_IN_SAMPLE_PARTIAL_R2" else float(np.clip(1-sse1[j]/sse0[j], 0, 1))
        records.append(dict(outcome=outcome, n_people=len(y), n_communities=len(set(labels)),
                            baseline_rank=rank0, full_rank=rank1, rank_gain=rank1-rank0,
                            residual_df_nominal=len(y)-rank1, baseline_sse=float(sse0[j]),
                            full_sse=float(sse1[j]), partial_r2=value, model_status=status))
    return records


def guard(directory, max_memory_mb, min_free_mb):
    require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024 <= max_memory_mb, "RSS guard exceeded")
    require(shutil.disk_usage(directory).free >= min_free_mb*2**20, "Disk reserve exhausted")


def run(contract_path, expected_sha, output, max_memory_mb, min_free_mb):
    started = time.monotonic()
    path, output = Path(contract_path).resolve(strict=True), Path(output)
    require(not output.exists(), "Output exists: never overwrite analysis")
    output.parent.mkdir(parents=True, exist_ok=True)
    guard(output.parent, max_memory_mb, min_free_mb)
    ledger = Ledger(path.parent)
    ledger.add(dict(path=str(path), sha256=expected_sha))
    contract = strict_json(path)
    require(set(contract) == {"schema", "expected_samples", "expected_partitions", "chromosomes",
            "community_evidence", "roh_individual", "roh_segments", "roh_log", "minimum_kb", "availability"},
            "Unexpected ROH contract keys")
    require(contract["schema"] == SCHEMA and contract["chromosomes"] == [str(i) for i in range(1, 23)],
            "Expected authenticated 22-autosome contract")
    people = load_people(contract, ledger)
    require(people["phis"] == list(PHIS), "Both agreed PC-Relate thresholds required")
    community_path = ledger.add(contract["community_evidence"])
    community_manifest = strict_json(community_path)
    community_rows = list(table(ledger.adjacent(community_path, community_manifest, "individuals.private.tsv")))
    roh, audit = load_roh(ledger.add(contract["roh_individual"]), ledger.add(contract["roh_segments"]),
        ledger.add(contract["roh_log"]), people["samples"], contract["chromosomes"], contract["minimum_kb"])
    availability, n_sites = load_availability(contract["availability"], ledger, people, contract["chromosomes"])
    common = [i for i in range(len(roh)) if all(people["labels"][p["partition_id"]][i] >= 0 for p in people["partitions"])]
    eligible = [i for i, (q, row) in enumerate(zip(people["ancestry"], community_rows))
                if q is not None and row["Cohort"] not in ("", "NA")]
    for i in eligible:
        require(abs(sum(people["ancestry"][i])-1) <= .001, "Ancestry composition not normalized")
    eligible_set = set(eligible)
    common = [i for i in common if i in eligible_set]
    comparison_rows, group_rows = [], []
    for p in people["partitions"]:
        guard(output.parent, max_memory_mb, min_free_mb)
        labels = people["labels"][p["partition_id"]]
        assigned = [i for i in eligible if labels[i] >= 0]
        for support, initial in (("assigned_partition", assigned), ("common_all_partitions", common)):
            for phi in (None, *PHIS):
                indices = initial if phi is None else [i for i in initial if people["components"][phi][i] is not None]
                require(indices, "No complete support for declared comparison")
                counts = None if phi is None else Counter(people["components"][phi][i] for i in indices)
                weights = np.ones(len(indices)) if counts is None else np.asarray([1/counts[people["components"][phi][i]] for i in indices])
                target = np.asarray([[roh[i][v] for v in OUTCOMES] for i in indices])
                q = np.asarray([people["ancestry"][i] for i in indices])
                cohort = [community_rows[i]["Cohort"] for i in indices]
                base = np.column_stack((np.ones(len(indices)), q, one_hot(cohort)))
                selected_labels = [labels[i] for i in indices]
                context = dict(partition_id=p["partition_id"], graph_id=p["graph_id"], family=p["family"], resolution=p["resolution"],
                    support=support, weighting="person" if phi is None else "component_balanced", phi=phi,
                    n_before_component_filter=len(initial), n_unknown_component=len(initial)-len(indices),
                    n_balanced_components=None if counts is None else len(counts), sum_weights=float(sum(weights)))
                for model, matrix in (("Q_and_collection", base),
                        ("Q_collection_and_rare_availability", np.column_stack((base, availability[indices])))):
                    comparison_rows.extend(dict(**context, baseline=model, **r) for r in
                                           nested_comparison(target, matrix, selected_labels, weights))
                for label in sorted(set(selected_labels)):
                    loc = [j for j, value in enumerate(selected_labels) if value == label]
                    w = weights[loc]
                    group_rows.append(dict(**context, community=label, n_people=len(loc),
                        n_zero_roh=sum(target[j, 0] == 0 for j in loc),
                        mean_NSEG=float(np.average(target[loc, 0], weights=w)),
                        mean_KB=float(np.average(target[loc, 1], weights=w)),
                        median_NSEG_person=float(np.median(target[loc, 0])), median_KB_person=float(np.median(target[loc, 1])),
                        max_KB=float(np.max(target[loc, 1])), n_collections=len({cohort[j] for j in loc}),
                        n_components=None if phi is None else len({people["components"][phi][indices[j]] for j in loc}),
                        max_component_fraction=None if phi is None else
                            max(Counter(people["components"][phi][indices[j]] for j in loc).values())/len(loc)))
    ledger.recheck()
    output.mkdir()
    write_table(output/"persons.private.tsv", [dict(**r, rare_catalogue_available_fraction=float(availability[i])) for i, r in enumerate(roh)])
    write_table(output/"models.tsv", comparison_rows)
    write_table(output/"communities.private.tsv", group_rows)
    summary = dict(schema=SCHEMA, status=STATUS, roh_audit=audit, n_partitions=len(people["partitions"]),
        n_common_support=len(common), n_complete_ancestry_collection=len(eligible), n_model_rows=len(comparison_rows),
        availability_catalogue_sites=n_sites, no_winner_selected=True, no_pvalues=True, no_confidence_intervals=True,
        founder_event_identified=False, kinship_thresholds=list(PHIS), limitations=LIMITATIONS,
        model_status_counts=dict(Counter(r["model_status"] for r in comparison_rows)), seconds=time.monotonic()-started)
    save_json(output/"summary.json", summary)
    with (output/"INFORME.md").open("x") as handle:
        handle.write("# ROH en las comunidades guardadas: evidencia complementaria\n\n")
        handle.write(f"Se reutilizaron {audit['n_segments']} tramos ROH de {len(roh)} personas y {len(people['partitions'])} particiones; "
                     f"{audit['zero_roh_people']} personas tienen cero tramos explícitamente registrados. No se ejecutó de nuevo PLINK.\n\n")
        handle.write("ROH es un tramo de homocigosidad dentro de una persona. NSEG cuenta tramos; KB suma kilobases; "
                     "no se calculó F_ROH ni edad de un supuesto fundador. El mínimo histórico es 1500 kb.\n\n")
        handle.write("Cada comparación usa las mismas personas y pesos en dos modelos de mínimos cuadrados: "
                     "ancestría global y colección, frente a esos mismos predictores más comunidad. "
                     "El R² parcial mide qué fracción de variación residual de NSEG o KB describe añadir comunidad. "
                     "Es un ajuste dentro de esta cohorte: no demuestra población nueva ni causa demográfica. "
                     "El modelo de sensibilidad añade a ambos lados disponibilidad del catálogo raro, no calidad ROH certificada.\n\n")
        handle.write(f"Se distinguen asignados por partición y soporte común de {len(common)} personas. "
                     "Los pesos inversos al tamaño del componente cambian la media hacia componentes observados; "
                     "no hacen independientes a las personas ni borran familias. La mediana de cada grupo sigue siendo por persona.\n\n")
        handle.write("No se eligió configuración ganadora. Una diferencia ROH es compatible con varias historias "
                     "y debe confrontarse con parentesco, mezcla, medición e IBD; ausencia de diferencia no descarta una historia fundadora.\n\n")
        handle.write("## Límites\n\n"+"\n".join("- "+v for v in LIMITATIONS)+"\n")
    manifest = dict(schema=SCHEMA, status=STATUS, inputs=ledger.records, samples_sha256=people["sample_hash"],
        parameters=dict(minimum_kb=contract["minimum_kb"], max_memory_mb=max_memory_mb, min_free_mb=min_free_mb),
        contains_individual_identifiers=True, public_distribution_allowed=False,
        code_sha256={name:sha256(Path(__file__).parent/name) for name in
                     ("r02_community_roh.py", "r02_community_ibd.py", "r02_community_levels.py")},
        numpy_version=np.__version__, outputs_sha256={p.name:sha256(p) for p in output.iterdir() if p.is_file()})
    save_json(output/"manifest.json", manifest)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-memory-mb", type=int, required=True)
    parser.add_argument("--min-free-mb", type=int, required=True)
    args = parser.parse_args()
    require(re.fullmatch(r"[0-9a-f]{64}", args.contract_sha256) and 0 < args.max_memory_mb <= 2048 and args.min_free_mb >= 0,
            "Invalid explicit resource/hash parameters")
    print(json.dumps(run(args.contract, args.contract_sha256, args.output_dir, args.max_memory_mb, args.min_free_mb), allow_nan=False))


if __name__ == "__main__":
    main()
