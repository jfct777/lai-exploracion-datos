#!/usr/bin/env python3
"""Private descriptive metadata summaries of 42 authenticated saved partitions.

No clustering, genotypes, ancestry estimation, kinship fitting or significance
tests. Identifiers are used for an exact in-memory join and never exported.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import importlib.metadata
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
import m165_sweep_figures as saved


CATEGORICAL = ('finestructure_clusters', 'Region', 'State', 'Cohort')
ANCESTRY = ('Autosomes_African_anc', 'Autosomes_European_anc',
            'Autosomes_Indigenous_anc', 'Autosomes_EastAsian_anc')
REFERENCE = 'finestructure_clusters'
MISSING = '__MISSING__'
COMMUNITY_COLUMNS = ('config_id', 'resolution', 'community_local', 'n_samples',
                     'fraction_cohort', 'fraction_assigned', 'n_reference_observed',
                     'n_reference_missing', 'dominant_finestructure', 'dominant_n',
                     'purity_observed', 'small_n')
DEFAULTS = dict(expected_samples=2619, categorical_columns=list(CATEGORICAL),
                ancestry_columns=list(ANCESTRY), disclose_min_n=5,
                missing_tokens=['', 'NA', 'N/A', 'nan', 'none', 'null', '.', 'Unknown', 'Sin dato'],
                ancestry_sum_tolerance=0.001)


def settings_from_json(text=None):
    supplied = {} if text is None else json.loads(text)
    saved.require(isinstance(supplied, dict) and not set(supplied) - set(DEFAULTS),
                  'Unknown metadata-summary setting')
    settings = {**DEFAULTS, **supplied}
    for key in ('expected_samples', 'disclose_min_n'):
        saved.require(type(settings[key]) is int and settings[key] >= 1,
                      'Positive integer metadata-summary setting required')
    columns = settings['categorical_columns']
    saved.require(isinstance(columns, list) and all(isinstance(c, str) for c in columns)
                  and len(columns) == len(set(columns)) and REFERENCE in columns
                  and set(columns) <= set(CATEGORICAL), 'Categorical columns outside authorized scope')
    saved.require(settings['ancestry_columns'] == list(ANCESTRY),
                  'All four autosomal ancestry components must retain their explicit identities and order')
    tokens = settings['missing_tokens']
    saved.require(isinstance(tokens, list) and all(isinstance(t, str) for t in tokens)
                  and '' in tokens and 'unknown' in {t.casefold() for t in tokens},
                  'Missing tokens must explicitly include empty and Unknown')
    tolerance = settings['ancestry_sum_tolerance']
    saved.require(type(tolerance) in (int, float) and math.isfinite(tolerance)
                  and 0 <= tolerance <= .01, 'Invalid four-component rounding tolerance')
    return settings


def read_metadata(path, sample_column, samples, settings):
    """Keep complete rows only to detect conflicting duplicates, not for export."""
    path = Path(path)
    saved.require(sample_column and len(samples) == len(set(samples)) and all(samples),
                  'Invalid explicit sample key or node universe')
    required = [sample_column, *settings['categorical_columns'], *settings['ancestry_columns']]
    saved.require(len(required) == len(set(required)), 'Sample key conflicts with an annotation column')
    missing_tokens = {value.strip().casefold() for value in settings['missing_tokens']}
    lookup, full_rows = {}, {}
    n_rows = n_duplicates = 0
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        saved.require(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames))
                      and set(required) <= set(reader.fieldnames), 'Missing or duplicate metadata header')
        for row in reader:
            n_rows += 1
            saved.require(None not in row and all(value is not None for value in row.values()),
                          'Malformed metadata row')
            sample = row[sample_column]
            saved.require(sample and sample == sample.strip(), 'Empty or whitespace-padded metadata identifier')
            if sample in full_rows:
                saved.require(row == full_rows[sample], 'Conflicting duplicate metadata identifier')
                n_duplicates += 1
                continue
            full_rows[sample] = row
            annotations = {}
            for column in settings['categorical_columns']:
                value = row[column].strip()
                saved.require(value != MISSING, 'Reserved missing-category label occurs in metadata')
                annotations[column] = None if value.casefold() in missing_tokens else value
            for column in settings['ancestry_columns']:
                value = row[column].strip()
                if value.casefold() in missing_tokens:
                    annotations[column] = None
                else:
                    try:
                        number = float(value)
                    except ValueError:
                        raise ValueError('Nonnumeric ancestry fraction in authorized column') from None
                    saved.require(math.isfinite(number) and 0 <= number <= 1,
                                  'Ancestry fractions must be finite values in [0,1]')
                    annotations[column] = number
            fractions = [annotations[c] for c in settings['ancestry_columns']]
            if all(number is not None for number in fractions):
                saved.require(abs(sum(fractions) - 1.) <= settings['ancestry_sum_tolerance'],
                              'Four ancestry fractions do not sum to one within the declared rounding tolerance')
            lookup[sample] = annotations
    saved.require(all(sample in lookup for sample in samples), 'Metadata does not cover the full saved cohort')
    aligned = [lookup[sample] for sample in samples]
    audit = dict(n_metadata_rows=n_rows, n_unique_metadata_ids=len(lookup),
                 n_identical_duplicates_collapsed=n_duplicates,
                 n_extra_metadata_ids=len(set(lookup) - set(samples)), n_cohort_matched=len(aligned),
                 sample_column=sample_column, duplicate_policy='collapse_only_identical_complete_rows',
                 join='exact identifiers, reordered to authenticated graph_nodes node_id',
                 no_cohort_exclusions=True, fields={})
    for column in settings['categorical_columns']:
        counts = Counter(row[column] for row in aligned)
        audit['fields'][column] = dict(n_missing=counts.get(None, 0),
            n_observed=len(aligned) - counts.get(None, 0), n_levels_observed=len(set(counts) - {None}),
            counts={MISSING if key is None else key: value for key, value in
                    sorted(counts.items(), key=lambda kv: (kv[0] is None, kv[0] or ''))})
    for column in settings['ancestry_columns']:
        missing = sum(row[column] is None for row in aligned)
        audit['fields'][column] = dict(n_missing=missing, n_observed=len(aligned) - missing)
    return aligned, audit


def reference_scores(labels, references):
    """Assigned-only descriptive agreement; missing reference is not a class."""
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score
    saved.require(len(labels) == len(references), 'Reference and partition denominators differ')
    pairs = [(label, reference) for label, reference in zip(labels, references)
             if label >= 0 and reference is not None]
    by_community = {}
    for label, reference in pairs:
        by_community.setdefault(label, Counter())[reference] += 1
    counts = Counter(reference for _, reference in pairs)
    largest = max(counts.values(), default=0)
    result = dict(n_assigned_with_reference=len(pairs),
                  n_assigned_missing_reference=sum(label >= 0 and ref is None for label, ref in zip(labels, references)),
                  n_reference_levels_assigned=len(counts),
                  dominant_finestructure_assigned=';'.join(sorted(k for k, v in counts.items() if v == largest)),
                  dominant_finestructure_n=largest,
                  dominant_reference_share_assigned=largest / len(pairs) if pairs else None,
                  weighted_purity_assigned=sum(max(c.values()) for c in by_community.values()) / len(pairs) if pairs else None,
                  reference_ari=None, reference_nmi=None, reference_ami=None,
                  agreement_status='INSUFFICIENT_ASSIGNED_WITH_REFERENCE',
                  degenerate_reference_or_partition=len(counts) < 2 or len(by_community) < 2)
    if len(pairs) >= 2:
        predicted, reference = zip(*pairs)
        result.update(reference_ari=float(adjusted_rand_score(reference, predicted)),
                      reference_nmi=float(normalized_mutual_info_score(reference, predicted, average_method='arithmetic')),
                      reference_ami=float(adjusted_mutual_info_score(reference, predicted, average_method='arithmetic')),
                      agreement_status='DESCRIPTIVE_DEGENERATE' if result['degenerate_reference_or_partition'] else 'DESCRIPTIVE')
    return result


def summarize(data, metadata, settings):
    import numpy as np
    saved.require(len(metadata) == data['n_cohort'] == settings['expected_samples'],
                  'Metadata and saved cohort denominators differ')
    sizes, distributions, ancestry, partitions = [], [], [], []
    references = [row[REFERENCE] for row in metadata]
    for graph in data['graphs']:
        for gamma in saved.GAMMAS:
            labels = graph['labels'][gamma]
            saved.require(len(labels) == len(metadata), 'Saved partition length differs from metadata')
            communities = sorted(set(labels) - {-1})
            assigned = [i for i, label in enumerate(labels) if label >= 0]
            unassigned = [i for i, label in enumerate(labels) if label == -1]
            scopes = [('cohort', '', list(range(len(labels)))), ('assigned', '', assigned),
                      ('unassigned', '', unassigned),
                      ('active_unassigned', '', [i for i in unassigned if graph['active'][i]]),
                      ('isolated', '', [i for i, active in enumerate(graph['active']) if not active])]
            community_indices = {label: [i for i, value in enumerate(labels) if value == label]
                                 for label in communities}
            scopes.extend(('community', label, indices) for label, indices in community_indices.items())
            common = dict(config_id=graph['config_id'], resolution=gamma)
            for label, indices in community_indices.items():
                known = Counter(references[i] for i in indices if references[i] is not None)
                n_known = sum(known.values()); dominant_n = max(known.values(), default=0)
                sizes.append(dict(**common, community_local=label, n_samples=len(indices),
                    fraction_cohort=len(indices)/len(labels), fraction_assigned=len(indices)/len(assigned),
                    n_reference_observed=n_known, n_reference_missing=len(indices)-n_known,
                    dominant_finestructure=';'.join(sorted(k for k, v in known.items() if v == dominant_n)),
                    dominant_n=dominant_n, purity_observed=dominant_n/n_known if n_known else None,
                    small_n=len(indices) < settings['disclose_min_n']))
            scores = reference_scores(labels, references)
            largest = max(map(len, community_indices.values()), default=0)
            partitions.append(dict(**common, n_cohort=len(labels), n_active=graph['n_active'],
                n_isolated=len(labels)-graph['n_active'], n_assigned=len(assigned), n_unassigned=len(unassigned),
                n_active_unassigned=graph['n_active']-len(assigned), n_communities=len(communities),
                largest_community=largest, largest_community_share_assigned=largest/len(assigned) if assigned else None,
                **scores))
            for scope, community, indices in scopes:
                base = dict(**common, scope=scope, community_local=community, n_samples=len(indices))
                for column in settings['categorical_columns']:
                    counts = Counter(metadata[i][column] for i in indices)
                    missing = counts.get(None, 0); observed = len(indices)-missing
                    # Always emit a missing row, including empty scopes, so zero is explicit.
                    for value in sorted(set(counts)-{None}) + [None]:
                        count = counts.get(value, 0)
                        distributions.append(dict(**base, field=column, category=MISSING if value is None else value,
                            is_missing=value is None, n=count, denominator_all=len(indices),
                            denominator_observed=observed, n_missing=missing,
                            fraction_all=count/len(indices) if indices else None,
                            fraction_observed=count/observed if value is not None and observed else None,
                            small_cell=0 < count < settings['disclose_min_n']))
                for column in settings['ancestry_columns']:
                    values = [metadata[i][column] for i in indices if metadata[i][column] is not None]
                    suppressed = len(values) < settings['disclose_min_n']
                    stats = dict(mean=None, median=None, q25=None, q75=None, iqr=None)
                    if not suppressed:
                        q25, median, q75 = map(float, np.quantile(values, [.25, .5, .75], method='linear'))
                        stats.update(mean=float(np.mean(values)), median=median, q25=q25, q75=q75, iqr=q75-q25)
                    ancestry.append(dict(**base, field=column, n_observed=len(values),
                        n_missing=len(indices)-len(values), suppressed_small_n=suppressed,
                        disclose_min_n=settings['disclose_min_n'], **stats))
    saved.require(len(partitions) == 42, 'Expected all 42 saved graph-resolution cells')
    return {'community_sizes.tsv': sizes, 'categorical_distributions.tsv': distributions,
            'ancestry_summary.tsv': ancestry, 'partition_metadata_summary.tsv': partitions}


def readme(settings):
    return f'''# Resumen descriptivo de metadata sobre las 42 particiones guardadas

Identificadores usados sólo en memoria; unión exacta y reordenación al grafo.
No se modifican cohortes, conexiones, pesos, asignaciones ni filtros. No hay
genotipos, reclustering, PC-Relate nuevo, pruebas de hipótesis ni elección óptima.

Las comunidades se identifican localmente por configuración y gamma: C0 no es
una identidad común entre particiones. Cohorte, asignados y no asignados tienen
denominadores explícitos; no asignados incluye aislados y activos en grupos <3.
Las cinco clases de scope no son todas disjuntas: no sumar cohort/assigned con
community. Missing cuenta vacíos, NA y Unknown, sin imputación. Exclude de otra
columna de la fuente no elimina personas. AM_1/AM_2 y los nombres de cohortes
se conservan literales, sin inventar región de nacimiento o reclutamiento.

Autosomes_African/European/Indigenous/EastAsian_anc son cuatro fracciones de la
metadata de referencia, no ADMIXTURE de-novo K=3 ni ancestría local. Se resumen
por separado sin renormalización: media, mediana, cuartiles lineales e IQR=q75-q25.
Las estadísticas numéricas quedan vacías cuando N observado < {settings['disclose_min_n']};
el N, los faltantes y la supresión siguen explícitos. Los conteos categóricos
pequeños no se suprimen en este paquete privado y se marcan small_cell.

ARI=índice de Rand ajustado; NMI=información mutua normalizada;
AMI=información mutua ajustada. Comparan fineSTRUCTURE y las etiquetas existentes
únicamente en asignados con referencia observada; promedio arithmetic para NMI
y AMI. Se registra el denominador en cada celda y la degeneración de particiones.
No confundir reference_ari con el ARI entre semillas del barrido original.
La pureza ponderada suma la categoría mayoritaria de cada comunidad sobre los
asignados con referencia. Aumentar el número de grupos puede elevar pureza sin
mejorar validez; se informa además la fracción de referencia dominante y del
grupo mayor. No se debe elegir gamma maximizando estos números retrospectivos.

fineSTRUCTURE es anotación genética, no geografía, pedigrí o validación
independiente certificada. Las siete resoluciones reutilizan las mismas personas;
no son siete muestras biológicas independientes. Parentesco, ancestría global,
densidad y reclutamiento pueden explicar concordancia. No p-valores ni población
confirmada. Ciudades, datos clínicos y sexo quedan fuera del contrato.

Los resultados agregados siguen siendo sensibles, sobre todo grupos pequeños.
Mantener el paquete privado; la ausencia de IDs no garantiza anonimato. Para
difusión externa se requiere una revisión específica de divulgación y tamaños.
'''


def run(results_dir, metadata_file, output_dir, sample_column='ID', settings_json_text=None):
    settings = settings_from_json(settings_json_text)
    root, metadata_path, output = Path(results_dir), Path(metadata_file), Path(output_dir)
    saved.require(not output.exists() and not output.is_symlink(), 'Output directory exists; no overwrite')
    saved.require(not output.resolve().is_relative_to(root.resolve()), 'Output may not be inside the source results')
    source_hash = saved.sha256(__file__); validator_hash = saved.sha256(saved.__file__)
    metadata_hash = saved.sha256(metadata_path)
    data = saved.load_results(root)
    first = data['graphs'][0]['config_id']
    samples = [row['sample_id'] for row in saved.table(root/first/'graph_nodes.tsv')]
    annotations, audit = read_metadata(metadata_path, sample_column, samples, settings)
    tables = summarize(data, annotations, settings)
    after = {name: saved.sha256(root/name) for name in data['input_sha256']}
    saved.require(after == data['input_sha256'] and saved.sha256(metadata_path) == metadata_hash,
                  'Input metadata or saved graph changed during summary')
    saved.require(saved.sha256(__file__) == source_hash and saved.sha256(saved.__file__) == validator_hash,
                  'Source code changed during summary')
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in tables.items():
        # Empty community table remains a schema-bearing file for a fully empty graph.
        if rows:
            saved.write_table(output/name, rows)
        else:
            saved.require(name == 'community_sizes.tsv', 'Unexpected empty summary table')
            with (output/name).open('x', newline='', encoding='utf-8') as handle:
                csv.writer(handle, delimiter='\t', lineterminator='\n').writerow(COMMUNITY_COLUMNS)
    audit.update(source_path=str(metadata_path.resolve()), source_sha256=metadata_hash,
                 source_bytes=metadata_path.stat().st_size, settings=settings)
    (output/'metadata_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    (output/'README.md').write_text(readme(settings), encoding='utf-8')
    manifest = dict(status='COMPLETE_DESCRIPTIVE_METADATA_SUMMARY', n_cohort=data['n_cohort'],
        n_graphs=len(data['graphs']), n_cells=len(tables['partition_metadata_summary.tsv']),
        no_reclustering=True, no_new_genotypes=True, no_cohort_exclusions=True,
        no_kinship_computation=True, contains_individual_identifiers=False,
        public_distribution_allowed=False, biological_validation=False, originals_changed=0,
        settings=settings, sample_column=sample_column, metadata_sha256=metadata_hash,
        input_sha256_before=data['input_sha256'], input_sha256_after=after,
        source_sha256=source_hash, validator_sha256=validator_hash, source_core_sha256=data['core_sha256'],
        source_input_sha256=data['source_sha256'],
        n_output_rows={name: len(rows) for name, rows in tables.items()},
        versions={package: importlib.metadata.version(package) for package in ('numpy', 'scikit-learn')},
        outputs_sha256={p.name: saved.sha256(p) for p in sorted(output.iterdir())})
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--metadata-file', required=True)
    parser.add_argument('--sample-column', default='ID')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--settings-json-text')
    args = parser.parse_args()
    result = run(args.results_dir, args.metadata_file, args.output_dir, args.sample_column, args.settings_json_text)
    print(json.dumps({key: result[key] for key in ('status', 'n_cohort', 'n_graphs', 'n_cells')}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Third-party parser exceptions can embed a raw cell; never print them.
        print('Metadata summary stopped ('+type(error).__name__+'); inputs or contract did not pass validation', file=sys.stderr)
        raise SystemExit(1) from None
