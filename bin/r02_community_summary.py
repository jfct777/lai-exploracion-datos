#!/usr/bin/env python3
"""Spanish, private descriptive report of an authenticated saved community bundle.

No individual data, re-clustering, kinship estimation, winner selection or tests.
Every source partition and community is retained; unknowns remain unknowns.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import resource

import r02_community_evidence as evidence


SCHEMA = "r02_community_summary_v1"
STATUS = "COMPLETE_DESCRIPTIVE_COMMUNITY_SUMMARY_NOT_VALIDATED"
PHIS = (.0221, .0442)
SCOPES = ("cohort", "assigned", "unassigned", "active_unassigned", "isolated")
ANCESTRY_NAMES = dict(zip(evidence.ANCESTRY, ("Africana", "Europea", "Indígena americana", "Asiática oriental")))
CATEGORY_NAMES = dict(zip(evidence.CATEGORICAL, ("fineSTRUCTURE", "Región", "Estado", "Cohorte", "Fuente", "Origen")))
BASE = {"graph_id", "family", "resolution", "scope", "community_local", "n_samples", "n_cohort", "n_assigned",
        "fraction_cohort", "fraction_assigned"}
COMPONENT = {"kinship_threshold", "n_components", "n_component_known", "n_component_missing", "max_component_n",
             "max_observed_component_fraction_all", "max_component_fraction_known", "component_status"}
ANCESTRY_FIELDS = {"field", "observation_scope", "n_observed", "n_unavailable_for_scope", "field_n_missing",
                  "n_excluded_incomplete_vector", "n_complete_vector", "mean_fraction", "std_fraction", "std_ddof", "unit"}
CATEGORY_FIELDS = {"field", "category", "is_missing", "n", "n_observed", "n_missing", "fraction_all", "fraction_observed"}
COMPARE_FIELDS = {"left_graph", "left_resolution", "right_graph", "right_resolution", "n_cohort", "n_both_assigned",
                  "fraction_cohort_both_assigned", "n_left_only_assigned", "n_right_only_assigned", "n_neither_assigned",
                  "left_n_communities_common", "right_n_communities_common", "ari", "status"}
require, sha256 = evidence.require, evidence.sha256


def integer(value):
    require(isinstance(value, str) and value.isascii() and value.isdigit(), "Expected nonnegative integer")
    return int(value)


def number(value, optional=False):
    if optional and value == "NA":
        return None
    result = float(value)
    require(math.isfinite(result), "Nonfinite table number")
    return result


def same_number(actual, expected):
    value = number(actual, optional=True)
    require((value is None and expected is None) or
            (value is not None and expected is not None and math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-12)),
            "Inconsistent numerical denominator/value")


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def row_key(row):
    scope = row["scope"]
    require(scope in (*SCOPES, "community"), "Unknown scope")
    label = integer(row["community_local"]) if scope == "community" else None
    require(scope == "community" or row["community_local"] == "", "Non-community has local label")
    return row["graph_id"], number(row["resolution"]), scope, label


def read_table(inputs, manifest_path, manifest, name, columns):
    path = inputs.adjacent(manifest_path, manifest, name + ".tsv")
    rows = evidence.kinship_base.read_table(path)
    require(len(rows) == manifest["table_rows"].get(name), "Table row count differs from manifest: " + name)
    require(all(set(row) == columns for row in rows), "Unsupported table columns: " + name)
    inputs.check()
    return rows


def authenticate(manifest_path, expected_sha256, max_memory_mb):
    path = Path(manifest_path).resolve()
    inputs = evidence.Inputs(path.parent, max_memory_mb)
    inputs.add(dict(path=str(path), sha256=expected_sha256))
    manifest = evidence.strict_json(path)
    require(manifest.get("schema") == "r02_community_evidence_v1" and manifest.get("status") == evidence.STATUS,
            "Unsupported community bundle version/status")
    require(manifest.get("chromosomes") == evidence.AUTOSOMES and type(manifest.get("n_cohort")) is int
            and manifest["n_cohort"] > 0, "Invalid source cohort/autosomes")
    require(isinstance(manifest.get("sample_ids_sha256"), str)
            and evidence.re.fullmatch(r"[0-9a-f]{64}", manifest["sample_ids_sha256"]), "Invalid cohort identity hash")
    require(all(manifest.get(key) is True for key in
                ("no_reclustering", "no_new_kinship", "no_new_target", "no_pvalues", "no_population_validation"))
            and manifest.get("incremental_rare_utility") == "NOT_ESTIMATED"
            and manifest.get("contains_individual_identifiers") is False
            and manifest.get("public_distribution_allowed") is False, "Unsupported source interpretation/privacy")
    ledger, graph_ids = {}, set()
    for graph in manifest["graph_ledger"]:
        gid, family = graph["graph_id"], graph["family"]
        require(isinstance(gid, str) and evidence.re.fullmatch(r"[A-Za-z0-9_.:/-]+", gid)
                and gid not in graph_ids and family in {"H", "R", "C", "R_plus_C"}, "Invalid/duplicate graph ledger")
        graph_ids.add(gid)
        require(isinstance(graph["resolutions"], list) and graph["resolutions"], "Empty/invalid resolution ledger")
        for gamma in graph["resolutions"]:
            require(type(gamma) in (int, float) and math.isfinite(gamma) and gamma > 0, "Invalid resolution")
            require((gid, float(gamma)) not in ledger, "Duplicate partition")
            ledger[gid, float(gamma)] = family
    require(len(graph_ids) == manifest["n_graphs"] and len(ledger) == manifest["n_partitions"] and ledger,
            "Incomplete graph/partition ledger")
    audit = manifest["metadata_audit"]
    require(audit.get("output_units") == "fraction" and audit.get("ancestry_columns") == list(evidence.ANCESTRY)
            and audit.get("categorical_columns") == list(evidence.CATEGORICAL), "Unsupported annotation schema/units")
    tables = {name: read_table(inputs, path, manifest, name, fields) for name, fields in
              (("scopes", BASE), ("components", BASE | COMPONENT), ("ancestry", BASE | ANCESTRY_FIELDS),
               ("categorical", BASE | CATEGORY_FIELDS))}
    tables["comparisons"] = (read_table(inputs, path, manifest, "comparisons", COMPARE_FIELDS)
                             if manifest["table_rows"].get("comparisons", 0) else [])
    return path, inputs, manifest, ledger, tables


def validate_groups(manifest, ledger, tables):
    """Validate all scopes and annotations, including groups of zero persons."""
    groups, base_rows = {}, {}
    n = manifest["n_cohort"]
    for row in tables["scopes"]:
        key = row_key(row)
        require(key not in groups and key[:2] in ledger and row["family"] == ledger[key[:2]], "Unknown/duplicate scope")
        size, assigned = integer(row["n_samples"]), integer(row["n_assigned"])
        require(integer(row["n_cohort"]) == n and 0 <= size <= n and 0 <= assigned <= n, "Cohort universe differs")
        same_number(row["fraction_cohort"], size / n)
        same_number(row["fraction_assigned"], fraction(size, assigned) if key[2] == "community" else None)
        groups[key] = dict(scope=key[2], community_local=key[3], n_samples=size, n_cohort=n, n_assigned=assigned,
                           components={}, ancestry={}, categorical={})
        base_rows[key] = {field: row[field] for field in BASE}
    for partition in ledger:
        for scope in SCOPES:
            require((*partition, scope, None) in groups, "Missing required scope")
        p = {scope: groups[*partition, scope, None]["n_samples"] for scope in SCOPES}
        communities = [g for key, g in groups.items() if key[:2] == partition and key[2] == "community"]
        require(p["cohort"] == n and p["assigned"] + p["unassigned"] == n
                and p["active_unassigned"] + p["isolated"] == p["unassigned"]
                and sum(g["n_samples"] for g in communities) == p["assigned"]
                and all(g["n_samples"] > 0 for g in communities), "Partition denominator decomposition differs")
        require(all(g["n_assigned"] == p["assigned"] for key, g in groups.items() if key[:2] == partition),
                "Inconsistent assigned denominator")

    def checked(row):
        key = row_key(row)
        require(key in groups and all(row[field] == base_rows[key][field] for field in BASE),
                "Annotation scope/universe differs")
        return groups[key]

    for row in tables["components"]:
        group, phi = checked(row), number(row["kinship_threshold"])
        require(phi in PHIS and phi not in group["components"], "Unknown/duplicate kinship cut-off")
        known, missing, count = (integer(row[field]) for field in ("n_component_known", "n_component_missing", "n_components"))
        maximum = None if row["max_component_n"] == "NA" else integer(row["max_component_n"])
        require(known + missing == group["n_samples"] and 0 <= count <= known, "Invalid component counts")
        require((known == 0 and count == 0 and maximum is None) or
                (known > 0 and count > 0 and maximum is not None and math.ceil(known / count) <= maximum <= known-count+1),
                "Invalid observed maximum component")
        same_number(row["max_observed_component_fraction_all"], fraction(maximum, group["n_samples"]) if known else None)
        same_number(row["max_component_fraction_known"], fraction(maximum, known) if known else None)
        expected_status = "NO_EVALUABLE" if not known else "DESCRIPTIVE_PARTIAL" if missing else "DESCRIPTIVE"
        require(row["component_status"] == expected_status, "Component missingness status differs")
        group["components"][phi] = dict(n_known=known, n_missing=missing, n_components=count, maximum_observed_n=maximum,
            maximum_fraction_all=fraction(maximum, group["n_samples"]) if known else None,
            maximum_fraction_known=fraction(maximum, known) if known else None, status=expected_status)
    for row in tables["ancestry"]:
        group, field, scope = checked(row), row["field"], row["observation_scope"]
        require(field in evidence.ANCESTRY and scope in {"field_observed", "complete_vector"}, "Unknown ancestry field/scope")
        target = group["ancestry"].setdefault(field, {})
        require(scope not in target and row["unit"] == "fraction" and row["std_ddof"] == "0", "Ancestry duplicate/units/std differ")
        values = {field: integer(row[field]) for field in ("n_observed", "n_unavailable_for_scope", "field_n_missing",
                                                          "n_excluded_incomplete_vector", "n_complete_vector")}
        size = group["n_samples"]
        require(all(value <= size for value in values.values())
                and values["n_observed"] + values["n_unavailable_for_scope"] == size
                and values["n_complete_vector"] <= size-values["field_n_missing"], "Invalid ancestry denominators")
        require((scope == "field_observed" and values["n_observed"] == size-values["field_n_missing"]
                 and values["n_excluded_incomplete_vector"] == 0) or
                (scope == "complete_vector" and values["n_observed"] == values["n_complete_vector"]
                 and values["n_excluded_incomplete_vector"] == size-values["n_complete_vector"]-values["field_n_missing"]),
                "Field missingness confused with incomplete-vector exclusion")
        mean, std = number(row["mean_fraction"], True), number(row["std_fraction"], True)
        require((not values["n_observed"] and mean is None and std is None) or
                (values["n_observed"] > 0 and mean is not None and std is not None and 0 <= mean <= 1 and 0 <= std <= .5 + 1e-12),
                "Invalid ancestry mean/std or missingness")
        target[scope] = dict(**values, mean_fraction=mean, std_fraction=std, std_ddof=0, unit="fraction")
    for row in tables["categorical"]:
        group, field = checked(row), row["field"]
        require(field in evidence.CATEGORICAL and row["is_missing"] in {"True", "False"}, "Unknown categorical field/missingness")
        missing_flag = row["is_missing"] == "True"
        require(missing_flag == (row["category"] == evidence.metadata_base.MISSING), "Missing-category label mismatch")
        observed, missing, count = (integer(row[field]) for field in ("n_observed", "n_missing", "n"))
        require(observed + missing == group["n_samples"] and count <= (missing if missing_flag else observed),
                "Invalid category denominators")
        target = group["categorical"].setdefault(field, dict(n_observed=observed, n_missing=missing, counts={}))
        require((target["n_observed"], target["n_missing"]) == (observed, missing) and row["category"] not in target["counts"],
                "Duplicate/inconsistent categorical row")
        same_number(row["fraction_all"], fraction(count, group["n_samples"]))
        same_number(row["fraction_observed"], fraction(count, observed) if not missing_flag else None)
        target["counts"][row["category"]] = count
    for group in groups.values():
        require(set(group["components"]) == set(PHIS) and set(group["ancestry"]) == set(evidence.ANCESTRY)
                and set(group["categorical"]) == set(evidence.CATEGORICAL), "Incomplete annotation grid")
        complete_counts = set()
        for field, scopes in group["ancestry"].items():
            require(set(scopes) == {"field_observed", "complete_vector"}, "Incomplete ancestry observation scopes")
            require(scopes["field_observed"]["field_n_missing"] == scopes["complete_vector"]["field_n_missing"],
                    "Ancestry field missingness inconsistent")
            complete_counts.update(item["n_complete_vector"] for item in scopes.values())
        require(len(complete_counts) == 1, "Complete-vector universes differ among ancestry fields")
        for target in group["categorical"].values():
            counts = target["counts"]
            require(sum(counts.values()) == group["n_samples"]
                    and counts.get(evidence.metadata_base.MISSING) == target["n_missing"], "Category counts do not cover scope")
            observed = {key: value for key, value in counts.items() if key != evidence.metadata_base.MISSING}
            maximum = max(observed.values(), default=None)
            target.update(dominant_observed_n=maximum,
                          dominant_observed_categories=sorted(key for key, value in observed.items() if value == maximum))
    for partition in ledger:
        scopes = {scope: groups[*partition, scope, None] for scope in SCOPES}
        communities = [g for key, g in groups.items() if key[:2] == partition and key[2] == "community"]
        decompositions = ((scopes["cohort"], [scopes["assigned"], scopes["unassigned"]]),
                          (scopes["unassigned"], [scopes["active_unassigned"], scopes["isolated"]]),
                          (scopes["assigned"], communities))
        for parent, children in decompositions:
            for phi in PHIS:
                require(parent["components"][phi]["n_known"] == sum(child["components"][phi]["n_known"] for child in children),
                        "Component coverage does not match disjoint scopes")
            for field in evidence.CATEGORICAL:
                counts = evidence.Counter()
                for child in children:
                    counts.update(child["categorical"][field]["counts"])
                require(all(counts[category] == value for category, value in parent["categorical"][field]["counts"].items())
                        and all(parent["categorical"][field]["counts"].get(category, 0) == value for category, value in counts.items()),
                        "Category counts do not match disjoint scopes")
            for field in evidence.ANCESTRY:
                for scope in ("field_observed", "complete_vector"):
                    for count_field in ("n_observed", "field_n_missing", "n_complete_vector", "n_excluded_incomplete_vector"):
                        require(parent["ancestry"][field][scope][count_field] ==
                                sum(child["ancestry"][field][scope][count_field] for child in children),
                                "Ancestry counts do not match disjoint scopes")
    return groups


def summarize(manifest, ledger, groups, comparisons):
    partitions = []
    for (gid, gamma), family in ledger.items():
        scopes = {scope: groups[gid, gamma, scope, None] for scope in SCOPES}
        communities = sorted((group for key, group in groups.items() if key[:2] == (gid, gamma) and key[2] == "community"),
                             key=lambda group: group["community_local"])
        partition = dict(graph_id=gid, family=family, resolution=gamma, n_cohort=manifest["n_cohort"],
                         n_communities=len(communities), scopes=scopes, communities=communities, concentration={})
        for phi in PHIS:
            known = sum(group["components"][phi]["n_known"] for group in communities)
            assigned = scopes["assigned"]["n_samples"]
            require(known == scopes["assigned"]["components"][phi]["n_known"], "Community component coverage differs from assigned scope")
            maximum_sum = sum(group["components"][phi]["maximum_observed_n"] or 0 for group in communities) if known else None
            partition["concentration"][phi] = dict(sum_community_maximum_observed_n=maximum_sum,
                n_assigned=assigned, n_component_known=known, n_component_missing=assigned-known,
                sum_maximum_fraction_assigned=fraction(maximum_sum, assigned) if known else None,
                sum_maximum_fraction_known=fraction(maximum_sum, known) if known else None,
                status="NO_EVALUABLE" if not known else "DESCRIPTIVE_PARTIAL" if known < assigned else "DESCRIPTIVE")
        partitions.append(partition)
    converted, seen = [], set()
    for row in comparisons:
        left = row["left_graph"], number(row["left_resolution"])
        right = row["right_graph"], number(row["right_resolution"])
        key = tuple(sorted((left, right)))
        require(left in ledger and right in ledger and left != right and key not in seen, "Unknown/duplicate comparison")
        seen.add(key)
        counts = {field: integer(row[field]) for field in COMPARE_FIELDS if field.startswith("n_") or "_n_communities_common" in field}
        n = manifest["n_cohort"]
        require(counts["n_cohort"] == n and sum(counts[field] for field in
            ("n_both_assigned", "n_left_only_assigned", "n_right_only_assigned", "n_neither_assigned")) == n,
            "Comparison cohort denominators differ")
        require(counts["n_both_assigned"] + counts["n_left_only_assigned"] == groups[*left, "assigned", None]["n_samples"]
                and counts["n_both_assigned"] + counts["n_right_only_assigned"] == groups[*right, "assigned", None]["n_samples"],
                "Comparison assigned support differs")
        same_number(row["fraction_cohort_both_assigned"], counts["n_both_assigned"] / n)
        ari = number(row["ari"], True)
        usable = counts["n_both_assigned"] >= 2 and counts["left_n_communities_common"] >= 2 and counts["right_n_communities_common"] >= 2
        require((usable and ari is not None and -1 <= ari <= 1 and row["status"] == "DESCRIPTIVE_COMMON_ASSIGNED") or
                (not usable and ari is None and row["status"] == "NO_EVALUABLE_DEGENERATE_OR_INSUFFICIENT"), "ARI interpretation differs")
        converted.append(dict(left_graph=left[0], left_resolution=left[1], right_graph=right[0], right_resolution=right[1],
                              **counts, ari=ari, status=row["status"]))
    return dict(schema=SCHEMA, status=STATUS, n_cohort=manifest["n_cohort"], n_graphs=manifest["n_graphs"],
                n_partitions=len(partitions), partitions=partitions, comparisons=converted)


def flat_rows(summary):
    for p in summary["partitions"]:
        common = {key: p[key] for key in ("graph_id", "family", "resolution", "n_cohort")}
        sizes = {"n_" + scope: value["n_samples"] for scope, value in p["scopes"].items() if scope != "cohort"}
        for phi, concentration in p["concentration"].items():
            yield "partitions", dict(**common, **sizes, n_communities=p["n_communities"], kinship_threshold=phi,
                fraction_assigned=p["scopes"]["assigned"]["n_samples"] / p["n_cohort"],
                **{key: value for key, value in concentration.items() if key != "n_assigned"})
        for group in p["communities"]:
            for phi, components in group["components"].items():
                yield "community_components", dict(**common, community_local=group["community_local"],
                    n_samples=group["n_samples"], n_assigned=group["n_assigned"], kinship_threshold=phi, **components)


def md(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ").replace("<", "&lt;").replace(">", "&gt;")


def display(value):
    return "no evaluable" if value is None else f"{value:.4g}"


def report_lines(summary, source_sha):
    yield "# Resumen descriptivo de comunidades Ronda 2\n\n"
    yield (f"Entrada autenticada: manifiesto SHA256 `{source_sha}`. Cohorte: {summary['n_cohort']} personas; "
           f"{summary['n_graphs']} grafos y **todas las {summary['n_partitions']} particiones** del registro. "
           "No se selecciona una partición ganadora. No se reconstruyen parentescos ni comunidades.\n\n")
    yield "## Qué permiten y qué no permiten estas tablas\n\n"
    yield ("Permiten describir cobertura, composición ancestral y de procedencia, y concentración de componentes "
           "de dependencia dentro de las comunidades. Los componentes PC-Relate son conexiones transitivas al corte "
           "phi declarado, no familias genealógicas certificadas ni poblaciones. Varios componentes en una comunidad "
           "no demuestran independencia o estructura fina. La utilidad rara incremental permanece **NO ESTIMADA**.\n\n"
           "H utiliza longitudes de segmentos raros; R, Jaccard raro global; C, semejanza común; R+C, su combinación declarada. "
           "Sus pesos, resoluciones y etiquetas locales no son escalas biológicas intercambiables.\n\n"
           "La suma de máximos toma, en cada comunidad, el tamaño de su mayor componente OBSERVADO y suma esos tamaños. "
           "Se divide por todas las personas asignadas y, por separado, por las asignadas con componente conocido. "
           "No es una tasa familiar; los faltantes no valen cero. Una comunidad sin componente conocido aporta cero "
           "al recuento OBSERVADO si otras sí tienen datos, no un máximo poblacional estimado; si ninguna tiene datos, "
           "la suma y las fracciones son no evaluables. Leer siempre junto a conocidas/asignadas. "
           "La tabla conserva ambos cortes phi sin elegir uno.\n\n"
           "**La suma depende del número y tamaño de comunidades.** Al subdividir una partición puede aumentar "
           "sin cambiar ningún parentesco; en el límite de comunidades individuales sería uno entre conocidas. "
           "No usarla para ordenar resoluciones, comparar éxito poblacional ni seleccionar una partición.\n\n")
    yield "## Cobertura y concentración de cada partición\n\n"
    yield "| Grafo | Resolución | Asignadas / cohorte | Sin comunidad | Comunidades | phi | Suma máximos / asignadas | Suma máximos / conocidas | Componente conocido | Componente faltante |\n"
    yield "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for p in summary["partitions"]:
        a, u = p["scopes"]["assigned"]["n_samples"], p["scopes"]["unassigned"]["n_samples"]
        for phi, c in p["concentration"].items():
            yield f"| {md(p['graph_id'])} | {p['resolution']:g} | {a}/{p['n_cohort']} | {u} | {p['n_communities']} | {phi:g} | {display(c['sum_maximum_fraction_assigned'])} | {display(c['sum_maximum_fraction_known'])} | {c['n_component_known']}/{a} | {c['n_component_missing']}/{a} |\n"
    yield "\n## Concordancia sobre personas asignadas en ambas particiones\n\n"
    yield "El índice de Rand ajustado (ARI) describe concordancia, no aporte raro ni verdad poblacional. `-1` no es una comunidad.\n\n"
    for row in summary["comparisons"]:
        yield (f"- {md(row['left_graph'])} (resolución {row['left_resolution']:g}) frente a {md(row['right_graph'])} "
               f"({row['right_resolution']:g}): ARI {display(row['ari'])}; soporte común {row['n_both_assigned']}/{row['n_cohort']}; "
               f"sólo izquierda {row['n_left_only_assigned']}, sólo derecha {row['n_right_only_assigned']}, ninguna {row['n_neither_assigned']}.\n")
    yield "\n## Composición de todas las comunidades\n\n"
    yield ("Las cuatro ancestrías son estimaciones preexistentes, no verdad local ni nuevas inferencias. "
           "Se muestran media ± desviación descriptiva (divisor n), en fracción, sobre vectores completos. "
           "`summary.json` conserva además medias por campo, sus faltantes, exclusiones y todos los recuentos categóricos, "
           "incluidas personas sin comunidad. Las categorías dominantes conservan todos los empates; no son etiquetas poblacionales.\n\n")
    for p in summary["partitions"]:
        yield f"### {md(p['graph_id'])}, resolución {p['resolution']:g}\n\n"
        yield (f"Personas asignadas: {p['scopes']['assigned']['n_samples']}/{p['n_cohort']}; sin comunidad: "
               f"{p['scopes']['unassigned']['n_samples']}/{p['n_cohort']}, de ellas activas sin asignar "
               f"{p['scopes']['active_unassigned']['n_samples']} y aisladas {p['scopes']['isolated']['n_samples']}.\n\n")
        if not p["communities"]:
            yield "No hay comunidades asignadas; las características de la cohorte y los no asignados permanecen en `summary.json`.\n\n"
        for group in p["communities"]:
            yield f"**Comunidad local {group['community_local']}**: {group['n_samples']}/{group['n_assigned']} personas asignadas; {group['n_samples']}/{p['n_cohort']} de la cohorte.\n\n"
            for phi, c in group["components"].items():
                yield (f"- phi {phi:g}: {c['n_components']} componentes observados; máximo {display(c['maximum_observed_n'])}/{group['n_samples']} "
                       f"personas; conocidas {c['n_known']}, faltantes {c['n_missing']}.\n")
            yield "\n"
            for field, scopes in group["ancestry"].items():
                item = scopes["complete_vector"]
                yield (f"- Ancestría {ANCESTRY_NAMES[field]}: {display(item['mean_fraction'])} ± {display(item['std_fraction'])}; "
                       f"n={item['n_observed']}/{group['n_samples']}; faltantes del campo {item['field_n_missing']}, "
                       f"excluidas por otro campo faltante {item['n_excluded_incomplete_vector']}.\n")
            yield "\n"
            for field, item in group["categorical"].items():
                category = "; ".join(md(value) for value in item["dominant_observed_categories"]) or "no evaluable"
                yield (f"- {CATEGORY_NAMES[field]}: categoría(s) más frecuente(s) {category}; "
                       f"cada una {display(item['dominant_observed_n'])}/{item['n_observed']} observadas; "
                       f"faltantes {item['n_missing']}/{group['n_samples']}.\n")
            yield "\n"
    yield "## Límites y siguiente interpretación\n\n"
    yield ("No hay p-valores, intervalos inferenciales, poblaciones confirmadas ni criterios nuevos de éxito. "
           "Cohorte, Fuente y Origen describen procedencia, no lote técnico. Región y colección pueden estar confundidas. "
           "No se dispone aquí de profundidad de lectura (DP), calidad genotípica (GQ), lote técnico ni llamabilidad local. "
           "City, campos clínicos e identificadores individuales no se consumen. Agregados pequeños siguen siendo privados.\n\n"
           "La lectura útil combina simultáneamente cobertura, dispersión entre componentes y composición ancestral/procedencia: "
           "un cambio entre R, C y R+C puede ser cambio de personas representadas, familiares o mezcla; la discordancia sola "
           "no atribuye señal adicional a raras. La evaluación independiente del aporte incremental necesita un contraste "
           "y diseño separados, no otra etiqueta para estas mismas tablas.\n")


def run(manifest_path, expected_sha256, output_dir, max_memory_mb):
    output = Path(output_dir).absolute()
    require(not output.exists() and not output.is_symlink(), "Output already exists; no overwrite")
    modules = (evidence, evidence.metadata_base, evidence.study_base, evidence.biological,
               evidence.kinship_base, evidence.evidence, evidence.biological.diagnostics,
               evidence.metadata_base.saved, evidence.kinship_base.historical, evidence.biological.diagnostics.sweep)
    source_paths = [Path(__file__), *(Path(module.__file__) for module in modules)]
    sources = {path.name: sha256(path) for path in source_paths}
    path, inputs, manifest, ledger, tables = authenticate(manifest_path, expected_sha256, max_memory_mb)
    groups = validate_groups(manifest, ledger, tables)
    summary = summarize(manifest, ledger, groups, tables["comparisons"])
    del tables
    inputs.recheck()
    require(all(not Path(p).is_relative_to(output) for p in inputs.records), "Output would contain an input")
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    counts = evidence.write_rows(output, flat_rows(summary), inputs)
    with (output / "summary.json").open("x") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    with (output / "report.md").open("x") as handle:
        for line in report_lines(summary, expected_sha256):
            handle.write(line)
    inputs.recheck()
    require(all(sha256(path) == sources[path.name] for path in source_paths), "Source changed during summary")
    result = dict(schema=SCHEMA, status=STATUS, n_cohort=manifest["n_cohort"], sample_ids_sha256=manifest["sample_ids_sha256"],
        n_graphs=manifest["n_graphs"], n_partitions=manifest["n_partitions"], table_rows=counts,
        source_manifest_sha256=expected_sha256, source_manifest_path=str(path), input_files=inputs.records,
        source_sha256=sources, source_bundle_code_sha256=manifest["source_sha256"],
        source_bundle_package_versions=manifest["package_versions"],
        package_versions={name: evidence.importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")},
        no_reclustering=True, no_new_kinship=True,
        no_pvalues=True, no_population_validation=True, no_winner_selection=True,
        incremental_rare_utility="NOT_ESTIMATED", contains_individual_identifiers=False, public_distribution_allowed=False,
        formulas={"sum_community_maximum_observed_n": "sum over disjoint communities c of max observed component count within c",
                  "sum_maximum_fraction_assigned": "sum_community_maximum_observed_n / n_assigned; NA when no known component",
                  "sum_maximum_fraction_known": "sum_community_maximum_observed_n / n_component_known; NA when zero",
                  "dominant_observed_categories": "all nonmissing categories tied for the largest count; no tie-breaking"},
        memory=dict(max_memory_mb=max_memory_mb, peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                    external_hard_limit_required=True),
        limitations=["All saved partitions retained; not independent replications", "Components are not certified families or independent populations",
                     "Sum of within-community maxima depends on partition granularity and coverage; never rank resolutions or population success by it",
                     "Different ancestry observation scopes use different denominators", "No technical-batch/DP/GQ/local-callability adjustment",
                     "ARI and composition do not identify incremental rare utility", "Private aggregates; no authorization for public disclosure"],
        outputs_sha256={p.name: sha256(p) for p in output.iterdir() if p.is_file()})
    with (output / "manifest.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-memory-mb", required=True, type=int)
    args = parser.parse_args()
    import os
    os.umask(0o077)
    result = run(args.manifest, args.expected_manifest_sha256, args.output_dir, args.max_memory_mb)
    print(json.dumps({field: result[field] for field in ("status", "n_cohort", "n_graphs", "n_partitions")}))


if __name__ == "__main__":
    main()
