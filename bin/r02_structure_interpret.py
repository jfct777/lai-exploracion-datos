#!/usr/bin/env python3
"""Interpret an authenticated descriptive M14.2/IBD package without reanalysis.

The unit is a configuration and accumulated pair-length threshold, not a person
or independent experiment. All source cases are retained. No hypothesis test,
population assignment, optimization or new biological filter is performed.
"""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

SCHEMA = 'r02_structure_interpretation_v1'
STATUS = 'COMPLETE_DESCRIPTIVE_INTERPRETATION_NOT_POPULATION_VALIDATION'
INTEGRATION_FILES = {'configuration_cards.json', 'configuration_chromosomes.tsv',
    'configuration_components.tsv', 'configuration_kinship.tsv', 'configuration_metadata_categories.private.tsv',
    'configuration_metadata_coverage.tsv', 'configuration_metrics.tsv', 'manifest.json'}
IBD_FILES = {'configuration_ibd_geometry.tsv', 'manifest.json'}
MISSING = {'', 'NA', 'None', 'null'}
LIMITATIONS = [
    'Las 370 filas son configuraciones y umbrales de los mismos datos, no 370 experimentos independientes.',
    'Personas, parejas, segmentos y posiciones repetidas no son réplicas biológicas independientes.',
    'El Jaccard local resume intervalos seleccionados por M14: no es Jaccard de todo el genoma ni validación independiente.',
    'La suma de bp de parejas cuenta una misma coordenada de nuevo cuando pertenece a otra pareja; no es territorio genómico único.',
    'La unión elimina solapamientos dentro de cada pareja y cromosoma; no distingue dosis IBD1/IBD2 ni copias haplotípicas.',
    'IBD es una inferencia complementaria. Su territorio observado no equivale al territorio donde podría haberse detectado IBD.',
    'Los cortes PC-Relate son sensibilidades heredadas; estar bajo un corte o entre componentes no certifica independencia.',
    'La concentración por cromosoma no está ajustada por longitud, recombinación, densidad de marcadores ni territorio evaluable.',
    'La disponibilidad de metadatos no demuestra control de lote, procedencia o calidad técnica.',
    'No se evaluaron aquí las comunidades M16.5, el aporte incremental B0/B1/B2, estructura ancestral fina ni efectos fundadores.',
    'No se calcularon p-valores, intervalos de confianza ni una configuración ganadora; un cero observado no demuestra ausencia biológica.',
]
DEFINITIONS = {
    'active_people_fraction': 'Personas con al menos una pareja retenida / todas las personas de la cohorte.',
    'retained_pairs_fraction': 'Parejas retenidas / [n_cohorte*(n_cohorte-1)/2], no entre las personas activas solamente.',
    'rare_local_J': 'Suma I / suma U en intervalos M14; I=alelos raros portados por ambas personas, U=portados por al menos una, sólo genotipos evaluables en ambas.',
    'rare_joint_call_fraction': 'Suma Q / suma sitios del catálogo examinados en las parejas e intervalos seleccionados; observaciones pareja×intervalo×sitio, no posiciones genómicas únicas.',
    'rare_missing_GT_fraction': 'Llamadas genotípicas desconocidas / (2×sitios del catálogo examinados en parejas e intervalos); dos personas por pareja.',
    'kinship': 'Conteos, bp y pesos por corte PC-Relate; se conservan denominadores de todas las parejas y de las parejas con parentesco observado.',
    'components': 'Componentes transitivos del grafo de parentesco observado, no comunidades M16.5 ni pedigrí certificado. Un peso de arista es log(1+bp acumulados por pareja).',
    'component_incident_fraction': 'Mayor suma de pesos incidentes en un componente / suma de pesos de todas las aristas. Una arista entre componentes incide en ambos; no sumar incidencias como si fueran una partición.',
    'max_chromosome_pair_bp_fraction': 'Mayor suma de bp de parejas en un cromosoma / suma de bp de parejas en los cromosomas analizados.',
    'reported_ibd_overlap_over_m14': 'bp de intersección M14∩IBD / unión M14, sumadas por pareja y cromosoma.',
    'reported_ibd_overlap_over_selected_pairs': 'Misma intersección / IBD informado en las parejas que M14 retuvo; el denominador cambia al seleccionar parejas.',
    'reported_ibd_overlap_over_all_pairs': 'Misma intersección / IBD informado en todas las parejas de la cohorte; denominador fijo del paquete, no territorio de detectabilidad conocida.',
    'metadata_coverage': 'Personas con un campo registrado / personas del estrato representado o excluido. No se exportan nombres de personas ni categorías privadas.',
    'sum_span_union': 'Longitud de un intervalo es su extremo final menos inicial según el contrato fuente. Suma de registros puede duplicar solapamientos; unión por pareja/cromosoma no. El span entre extremos exteriores incluye posibles huecos. Aquí se reutiliza la unión autenticada, no se recalculan intervalos.',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digests(path):
    h, m, size = hashlib.sha256(), hashlib.md5(), 0
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block); m.update(block); size += len(block)
    return dict(sha256=h.hexdigest(), bytes=size, md5_base64=base64.b64encode(m.digest()).decode())


def read_json(path):
    def unique(pairs):
        result = {}
        for k, v in pairs:
            require(k not in result, 'Duplicate JSON key: ' + k)
            result[k] = v
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON value')))


def write_json(path, value):
    with Path(path).open('x') as out:
        json.dump(value, out, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        out.write('\n')


def same(a, b, label):
    require((a is None and b is None) or (a is not None and b is not None and
            math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-10)), 'Inconsistent ' + label)


def numeric(value, *, integer=False, optional=False):
    if optional and (value is None or str(value) in MISSING):
        return None
    require(not isinstance(value, bool), 'Boolean is not a measurement')
    value = str(value)
    if integer:
        require(re.fullmatch(r'[0-9]+', value) is not None, 'Invalid nonnegative integer')
        return int(value)
    n = float(value)
    require(math.isfinite(n) and n >= 0, 'Negative/nonfinite measurement')
    return n


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def key(row, edge_name='min_edge_bp'):
    name = row['config_id']
    require(re.fullmatch(r'L[1-9][0-9]*_G[1-9][0-9]*_N[1-9][0-9]*', name) is not None, 'Invalid configuration ID')
    return name, numeric(row[edge_name], integer=True)


def rows(path, max_rows):
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames), 'Duplicate/missing TSV columns')
        for n, row in enumerate(reader, 1):
            require(n <= max_rows and None not in row and None not in row.values(), 'Oversized or malformed TSV')
            yield row


def indexed(path, keys, max_rows):
    result = {}
    for row in rows(path, max_rows):
        k = keys(row)
        require(k not in result, 'Duplicate table key: ' + str(k))
        result[k] = row
    return result


def authenticate(args):
    require(re.fullmatch(r'[a-f0-9]{64}', args.package_sha256) is not None, 'Invalid package SHA256')
    require(args.max_input_mib > 0 and args.max_table_rows > 0, 'Positive resource limits required')
    paths = [Path(p) for p in (args.package, args.package_publication, args.integration_receipt, args.ibd_receipt)]
    budget = args.max_input_mib * 1024 * 1024
    require(all(p.is_file() and p.stat().st_size <= budget for p in paths), 'Missing/oversized metadata input')
    package = read_json(args.package)
    require(digests(args.package)['sha256'] == args.package_sha256, 'Package SHA256 mismatch')
    publication = read_json(args.package_publication)
    require(all(publication[k] == digests(args.package)[k] for k in ('sha256', 'bytes', 'md5_base64')),
            'Package publication receipt mismatch')
    require(package['schema'] == 'r02_descriptive_structure_package_v1' and
            package['status'] == 'COMPLETE_DESCRIPTIVE_PACKAGE_NOT_POPULATION_VALIDATION' and
            package['full_descriptive_package_complete'] is True, 'Incomplete or wrong package')
    require(package['chromosomes'] == list(map(str, range(1, 23))) and package['configurations'] > 0 and
            package['n_samples'] >= 2 and re.fullmatch(r'[a-f0-9]{64}', package['cohort_sha256']), 'Wrong autosomal/cohort scope')
    require(package['K02_K03_K07_evaluated'] is False and package['ancestry_specific_rare_evidence_evaluated'] is False,
            'Unexpected scientific scope; this report is descriptive only')
    manifests, files, evidence = {}, {}, []
    total = sum(p.stat().st_size for p in paths)
    for group, receipt_path, directory, expected in (
        ('descriptive_integration', args.integration_receipt, args.integration_dir, INTEGRATION_FILES),
        ('reported_ibd_geometry', args.ibd_receipt, args.ibd_dir, IBD_FILES)):
        spec = package[group]
        require(digests(receipt_path)['sha256'] == spec['receipt_sha256'], 'Source receipt SHA256 mismatch')
        receipt = read_json(receipt_path)
        require(receipt['status'] == 'PUBLISHED_VERIFIED' and receipt['manifest_sha256'] == spec['manifest_sha256'],
                'Source publication incomplete or changed')
        require(len(receipt['files']) == len(expected) and {r['name'] for r in receipt['files']} == expected,
                'Incomplete source publication inventory')
        require(len(spec['files']) == len(expected) and {r['name'] for r in spec['files']} == expected,
                'Incomplete package inventory')
        package_records = {r['name']: r for r in spec['files']}
        files[group] = {}
        for record in receipt['files']:
            name = record['name']
            require(all(record[k] == package_records[name][k] for k in ('sha256', 'bytes', 'md5_base64', 'uri', 'generation')),
                    'Package/source receipt disagreement: ' + name)
            local = Path(directory) / name if directory else Path(record['path'])
            require(local.is_file(), 'Missing source: ' + name)
            total += local.stat().st_size
            require(total <= budget and local.stat().st_size == record['bytes'], 'Input size or resource bound exceeded')
            require(digests(local) == {k:record[k] for k in ('sha256', 'bytes', 'md5_base64')}, 'Source bytes changed: ' + name)
            files[group][name] = local
            evidence.append(dict(group=group, name=name, path=str(local.resolve()),
                sha256=record['sha256'], bytes=record['bytes'], uri=record['uri'], generation=record['generation']))
        manifest = read_json(files[group]['manifest.json'])
        expected_hashes = {r['name']:r['sha256'] for r in receipt['files'] if r['name'] != 'manifest.json'}
        declared = {**manifest['outputs_sha256'], **manifest.get('operational_outputs_sha256', {})}
        require(declared == expected_hashes, 'Manifest output inventory mismatch')
        require(manifest['chromosomes'] == package['chromosomes'], 'Chromosome scope mismatch')
        manifests[group] = manifest
    legacy, ibd = manifests['descriptive_integration'], manifests['reported_ibd_geometry']
    require(legacy['schema'] == 'r02_biological_evaluation_v1' and legacy['status'] == 'BIOLOGICAL_DESCRIPTIVE_NOT_VALIDATED' and
            legacy['parallel_stage'] == 'COLLECTED_ALL_G', 'Unexpected descriptive producer')
    require(ibd['schema'] == 'r02_checkpoint_ibd_aggregate_v1' and ibd['status'] == 'COMPLETE_DESCRIPTIVE_REPORTED_GEOMETRY',
            'Unexpected IBD producer')
    require(legacy['sample_ids_sha256'] == ibd['cohort_sha256'] == package['cohort_sha256'] and
            legacy['n_samples'] == package['n_samples'] and
            legacy['n_configurations'] == ibd['configurations'] == package['configurations'] and
            legacy['n_cards'] == package['configuration_threshold_cases'], 'Cohort/configuration mismatch')
    require(legacy['parameters']['edge_thresholds_bp'] == ibd['thresholds_bp'] and
            package['configuration_threshold_cases'] == package['configurations'] * len(ibd['thresholds_bp']), 'Threshold grid mismatch')
    require(ibd['callability'] == 'UNKNOWN' and ibd['independent_validation'] is False and
            ibd['global_threshold_applied_after_autosomal_sum'] is True and
            all(legacy[k] is True for k in ('no_pvalues', 'no_winner_selected', 'no_population_labels')), 'Evidence scope changed')
    return package, legacy, ibd, files, evidence


def interpret(package, legacy, files, max_rows):
    src = files['descriptive_integration']
    cards_doc = read_json(src['configuration_cards.json'])
    cards = {}
    for card in cards_doc['cards']:
        k = key(card)
        require(k not in cards, 'Duplicate configuration card')
        require(card['decision_status'] == 'DESCRIPTIVE_ONLY_NO_SELECTION', 'Card was promoted to selection')
        cards[k] = card
    thresholds = legacy['parameters']['edge_thresholds_bp']
    configs = sorted({k[0] for k in cards})
    expected = {(c, t) for c in configs for t in thresholds}
    require(len(configs) == package['configurations'] and set(cards) == expected and
            len(cards) == package['configuration_threshold_cases'], 'Incomplete card grid')
    metrics = indexed(src['configuration_metrics.tsv'], lambda r: (*key(r), r['metric']), max_rows)
    kin = indexed(src['configuration_kinship.tsv'], lambda r: (*key(r), numeric(r['kinship_threshold'])), max_rows)
    components = indexed(src['configuration_components.tsv'], lambda r: (*key(r), numeric(r['kinship_threshold'])), max_rows)
    chroms = indexed(src['configuration_chromosomes.tsv'], lambda r: (*key(r), r['chrom']), max_rows)
    metadata = indexed(src['configuration_metadata_coverage.tsv'], lambda r: (*key(r), r['field'], r['stratum']), max_rows)
    ibd = indexed(files['reported_ibd_geometry']['configuration_ibd_geometry.tsv'], lambda r:key(r,'min_autosomal_m14_bp'), max_rows)
    phis = legacy['parameters']['kinship_thresholds']
    require(set(kin) == set(components) == {(*k, phi) for k in expected for phi in phis}, 'Incomplete kinship/component grid')
    require(set(ibd) == expected, 'Incomplete IBD grid')
    metric_names = set(next(iter(cards.values()))['answer']['measurements'])
    require(set(metrics) == {(*k, name) for k in expected for name in metric_names}, 'Incomplete metric grid')
    require(all(k[:2] in expected and k[2] in package['chromosomes'] for k in chroms), 'Foreign chromosome/configuration')
    metadata_axes = {k[2:] for k in metadata}
    require(metadata_axes and set(metadata) == {(*k, *axis) for k in expected for axis in metadata_axes}, 'Incomplete metadata grid')
    output = []
    n = package['n_samples']; possible = n * (n - 1) // 2
    baseline = None
    for k in sorted(expected):
        card = cards[k]; answer = card['answer']; first = kin[(*k, phis[0])]
        require(answer['n_people'] == n and numeric(first['n_people'], integer=True) == n and
                numeric(first['n_possible_pairs'], integer=True) == possible, 'Incorrect cohort denominator')
        active, pairs, segments, bp = (numeric(answer[x],integer=True) for x in
            ('n_active','n_pairs','n_segments','summed_disjoint_pair_bp'))
        require(active <= n and pairs <= possible, 'Impossible coverage')
        result = dict(config_id=k[0], min_edge_bp=k[1], n_people=n, n_possible_pairs=possible,
            n_active=active, n_isolated=n-active, n_pairs=pairs, n_segments=segments,
            summed_disjoint_pair_bp=bp, active_people_fraction=ratio(active,n), retained_pairs_fraction=ratio(pairs,possible))
        m = re.fullmatch(r'L([0-9]+)_G([0-9]+)_N([0-9]+)', k[0])
        result.update(min_length_bp=int(m[1]), max_gap_bp=int(m[2]), min_shared_effective=int(m[3]))
        for name in sorted(metric_names):
            row = metrics[(*k,name)]; entry = answer['measurements'][name]
            for field in ('numerator','denominator','value'):
                value = numeric(row[field],optional=True)
                same(value, entry[field], name+'.'+field)
                result[name+'__'+field] = value
            require(row['status'] == entry['status'] and row['reason'] == entry['reason'], 'Card/metric missing-state differs')
            num, den, value = (result[name+'__'+f] for f in ('numerator','denominator','value'))
            same(value, ratio(num,den) if num is not None and den is not None else None, name+' ratio')
            require(numeric(row['n_segments_total'],integer=True) == segments and
                    numeric(row['n_segments_evaluable'],integer=True) <= segments, 'Metric segment denominator differs')
            for field in ('status','reason','unit','scope','n_segments_evaluable','n_segments_total'):
                result[name+'__'+field] = row[field]
        same(result['people_represented_fraction__numerator'],active,'represented numerator')
        same(result['people_represented_fraction__denominator'],n,'represented denominator')
        same(result['possible_pairs_retained_fraction__numerator'],pairs,'pair numerator')
        same(result['possible_pairs_retained_fraction__denominator'],possible,'pair denominator')
        for phi in phis:
            row, component = kin[(*k,phi)], components[(*k,phi)]
            for field, expected_value in [('n_people',n),('n_possible_pairs',possible),('n_active',active),
                 ('n_isolated',n-active),('n_pairs',pairs),('n_segments',segments),('total_shared_bp',bp),
                 ('max_gap_bp',int(m[2])),('min_length_bp',int(m[1])),('min_shared_effective',int(m[3]))]:
                same(numeric(row[field]), expected_value, 'kinship '+field)
            observed, missing, above, below = (numeric(row[x],integer=True) for x in
                ('n_observed_kinship','n_missing_kinship','n_kin_ge_threshold','n_kin_lt_threshold'))
            require(observed+missing == pairs and above+below == observed, 'Kinship count denominator inconsistent')
            observed_bp, missing_bp, above_bp, below_bp = (numeric(row[x]) for x in
                ('bp_observed_kinship','bp_missing_kinship','bp_kin_ge_threshold','bp_kin_lt_threshold'))
            same(observed_bp+missing_bp,bp,'kinship bp denominator');same(above_bp+below_bp,observed_bp,'observed bp')
            for field, num, den in [('fraction_n_kin_ge_all_pairs',above,pairs),('fraction_n_kin_ge_observed_pairs',above,observed),
                ('fraction_n_missing_all_pairs',missing,pairs),('fraction_bp_kin_ge_all_pairs',above_bp,bp),
                ('fraction_bp_kin_ge_observed_pairs',above_bp,observed_bp),('fraction_bp_missing_all_pairs',missing_bp,bp)]:
                same(numeric(row[field],optional=True),ratio(num,den),field)
            prefix = 'phi_'+format(phi,'.8g')+'__'
            for field,value in row.items():
                if field not in ('config_id','min_edge_bp','kinship_threshold'):
                    result[prefix+field] = value
            for field,value in component.items():
                if field not in ('config_id','min_edge_bp','kinship_threshold'):
                    result[prefix+'component_'+field] = value
            component_pairs = sum(numeric(component[x],integer=True) for x in
                ('n_within_component_pairs','n_between_component_pairs','n_unassigned_component_pairs'))
            require(component_pairs == pairs, 'Component pair denominator inconsistent')
            same(numeric(component['max_component_incident_weight_fraction'],optional=True),
                 ratio(numeric(component['max_component_incident_weight'],optional=True),numeric(component['total_edge_weight']))
                 if component['max_component_incident_weight'] not in MISSING else None,'component concentration')
        chromosome_rows = [r for ck,r in chroms.items() if ck[:2] == k]
        chromosome_bp = {r['chrom']:numeric(r['pair_bp'],integer=True) for r in chromosome_rows}
        require(sum(chromosome_bp.values()) == bp and sum(numeric(r['n_segments'],integer=True) for r in chromosome_rows) == segments,
                'Chromosome totals disagree')
        for row in chromosome_rows:
            same(numeric(row['total_pair_bp']),bp,'chromosome denominator')
            same(numeric(row['fraction_pair_bp'],optional=True),ratio(numeric(row['pair_bp']),bp),'chromosome fraction')
        maximum = max(chromosome_bp.values(), default=0)
        result.update(n_chromosomes_with_segments=sum(x>0 for x in chromosome_bp.values()),
            max_chromosome_pair_bp=maximum, max_chromosome_pair_bp_fraction=ratio(maximum,bp),
            max_chromosomes=';'.join(c for c in package['chromosomes'] if chromosome_bp.get(c,0)==maximum and maximum>0))
        for chrom in package['chromosomes']:
            result['chr'+chrom+'__pair_bp']=chromosome_bp.get(chrom,0)
        g = ibd[k]
        require(g['status'] == 'DESCRIPTIVE_REPORTED_GEOMETRY_NOT_POPULATION_VALIDATION', 'Unexpected IBD evidence status')
        values = {name:numeric(g[name],integer=True) for name in ('selected_pairs','pairs_with_reported_ibd','m14_union_bp',
            'ibd_reported_overlap_bp','ibd_reported_bp_in_selected_pairs','ibd_reported_bp_all_pairs')}
        require(values['selected_pairs'] == pairs and values['m14_union_bp'] == bp and values['pairs_with_reported_ibd'] <= pairs,
                'M14/IBD selected territory or pair count differs')
        overlap, selected, all_bp = (values[x] for x in ('ibd_reported_overlap_bp','ibd_reported_bp_in_selected_pairs','ibd_reported_bp_all_pairs'))
        require(overlap <= min(bp,selected) and selected <= all_bp, 'Impossible reported IBD intersection')
        require(baseline is None or baseline == all_bp, 'All-pairs IBD denominator must remain fixed')
        baseline = all_bp
        for field,den in [('fraction_m14_overlapping_reported_ibd',bp),
            ('fraction_reported_ibd_of_selected_pairs_overlapping_m14',selected),
            ('fraction_all_reported_ibd_overlapping_selected_m14',all_bp)]:
            same(numeric(g[field],optional=True),ratio(overlap,den),field)
            result['ibd__'+field+'__numerator']=overlap
            result['ibd__'+field+'__denominator']=den
            result['ibd__'+field+'__value']=ratio(overlap,den)
        result.update({'ibd__'+field:value for field,value in values.items()})
        result['ibd__callability']='UNKNOWN'
        for field,stratum in sorted(metadata_axes):
            row=metadata[(*k,field,stratum)]
            require(stratum in ('represented','excluded_no_retained_edges'), 'Unknown metadata stratum')
            people=numeric(row['n_people'],integer=True); known=numeric(row['n_measured'],integer=True)
            require(people == (active if stratum=='represented' else n-active) and known <= people, 'Metadata denominator differs')
            same(numeric(row['fraction_measured'],optional=True),ratio(known,people),'metadata fraction')
            prefix='metadata__'+field+'__'+stratum+'__'
            result.update({prefix+'n_measured':known,prefix+'n_people':people,prefix+'fraction':ratio(known,people),
                           prefix+'status':row['status']})
        output.append(result)
    return output


def write_tsv(path, rows):
    require(rows, 'No output rows')
    fields = list(rows[0])
    require(all(set(row)==set(fields) for row in rows), 'Output schemas differ between cases')
    with Path(path).open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,delimiter='\t',lineterminator='\n');writer.writeheader()
        writer.writerows({k:'NA' if v is None else v for k,v in row.items()} for row in rows)


def describe(output, package, legacy):
    measures=['n_active','n_pairs','summed_disjoint_pair_bp','rare_local_J__value','max_chromosome_pair_bp_fraction',
              'ibd__fraction_m14_overlapping_reported_ibd__value']
    phis=legacy['parameters']['kinship_thresholds']
    measures += ['phi_'+format(phi,'.8g')+'__fraction_bp_kin_ge_observed_pairs' for phi in phis]
    summaries=[]
    for threshold in legacy['parameters']['edge_thresholds_bp']:
        selected=[r for r in output if r['min_edge_bp']==threshold]
        for measure in measures:
            values=[float(r[measure]) for r in selected if r[measure] is not None and str(r[measure]) not in MISSING]
            summaries.append(dict(min_edge_bp=threshold,measure=measure,n_configurations=len(selected),n_evaluable=len(values),
                minimum=min(values) if values else None,median=statistics.median(values) if values else None,maximum=max(values) if values else None))
    by_config=defaultdict(list)
    for row in output:by_config[row['config_id']].append(row)
    changes=[]
    for config, cases in sorted(by_config.items()):
        cases.sort(key=lambda r:r['min_edge_bp'])
        for before,after in zip(cases,cases[1:]):
            row=dict(config_id=config,from_min_edge_bp=before['min_edge_bp'],to_min_edge_bp=after['min_edge_bp'])
            for measure in measures:
                a,b=before[measure],after[measure]
                row[measure+'__before']=a;row[measure+'__after']=b
                row[measure+'__delta']=float(b)-float(a) if a is not None and b is not None and str(a) not in MISSING and str(b) not in MISSING else None
            require(after['n_active']<=before['n_active'] and after['n_pairs']<=before['n_pairs'] and
                    after['summed_disjoint_pair_bp']<=before['summed_disjoint_pair_bp'], 'Higher minimum cannot add retained pairs/people/bp')
            changes.append(row)
    lines=['# Interpretación descriptiva de configuraciones M14.2 e IBD','',
        f"Se conservaron **{len(output)} casos = {package['configurations']} configuraciones × {len(legacy['parameters']['edge_thresholds_bp'])} mínimos de longitud acumulada por pareja**, en {package['n_samples']} personas y 22 autosomas.",
        '', 'No se vuelve a ejecutar M14, M14.2 o M16.5. No se selecciona una configuración ganadora ni se declaran poblaciones.','',
        '## Qué se comparó','',
        'L es la longitud mínima de cada segmento (bp); G es la distancia máxima entre coincidencias consecutivas (bp); N es el mínimo efectivo de coincidencias. El mínimo de arista se aplica después de sumar las longitudes de los 22 cromosomas de cada pareja.','',
        'Cada configuración selecciona sus propios segmentos: comparar sus Jaccard locales no aísla un cambio sobre territorio idéntico. Las diferencias entre umbrales se emparejan dentro de la misma configuración.','',
        '## Alcance y contraprestaciones observadas','',
        '| Mínimo acumulado por pareja (bp) | Personas representadas: mínimo–mediana–máximo | Parejas retenidas: mínimo–mediana–máximo | Casos |',
        '|---:|---:|---:|---:|']
    def formatted(values):return '–'.join(format(x,'.6g') for x in values)
    for threshold in legacy['parameters']['edge_thresholds_bp']:
        a=next(x for x in summaries if x['min_edge_bp']==threshold and x['measure']=='n_active')
        b=next(x for x in summaries if x['min_edge_bp']==threshold and x['measure']=='n_pairs')
        lines.append(f"| {threshold} | {formatted([a['minimum'],a['median'],a['maximum']])} / {package['n_samples']} | {formatted([b['minimum'],b['median'],b['maximum']])} / {package['n_samples']*(package['n_samples']-1)//2} | {a['n_configurations']} |")
    lines += ['', 'Los rangos anteriores describen configuraciones del mismo conjunto de datos: no son intervalos de confianza. `threshold_summary.tsv` conserva los rangos de Jaccard, concentración por cromosoma, parentesco e IBD. `threshold_changes.tsv` conserva cada cambio pareado y su signo; no constituye una prueba de superioridad.', '',
        '## Compartición rara, parentesco y concentración','',
        '| Medida | Mínimo–mediana–máximo entre casos | Casos evaluables | Denominador |',
        '|---|---:|---:|---|']
    described=[('rare_local_J__value','Jaccard raro local','suma U de observaciones pareja×intervalo×sitio'),
        ('max_chromosome_pair_bp_fraction','Fracción del cromosoma con mayor longitud compartida','bp acumulados de todas las parejas seleccionadas'),
        ('ibd__fraction_m14_overlapping_reported_ibd__value','Fracción M14 que coincide con IBD informado','unión M14 sumada por pareja/cromosoma')]
    described += [('phi_'+format(phi,'.8g')+'__fraction_bp_kin_ge_observed_pairs',
        'Fracción de bp en parejas con PC-Relate ≥ '+format(phi,'.8g'),
        'bp de parejas retenidas con PC-Relate observado') for phi in phis]
    for field,label,denominator in described:
        values=[float(r[field]) for r in output if r[field] is not None and str(r[field]) not in MISSING]
        display=formatted([min(values),statistics.median(values),max(values)]) if values else 'NA'
        lines.append(f'| {label} | {display} | {len(values)}/{len(output)} | {denominator} |')
    lines += ['', 'Un valor mayor de Jaccard o de coincidencia IBD puede acompañarse de menos personas representadas o mayor concentración de parentesco. Esta tabla conserva esas dimensiones separadas: no las suma en una puntuación de calidad ni afirma que su variación tenga una única causa.', '',
        '## Qué está medido y qué falta','']
    for name in sorted({key[:-8] for key in output[0] if key.endswith('__status') and not key.startswith(('phi_','metadata__'))}):
        available=sum(r.get(name+'__value') is not None for r in output)
        reasons=sorted({str(r.get(name+'__reason','')) for r in output if r.get(name+'__value') is None})
        lines.append(f"- `{name}`: {available}/{len(output)} casos con valor; {'sin faltantes' if available==len(output) else 'razón registrada: '+ '; '.join(reasons)}.")
    lines += ['', 'El suplemento IBD aporta geometría informada incluso cuando la anotación original M14.2 decía que IBD no era evaluable. No reemplaza esa ausencia por falsa cobertura de genotipos: `callability=UNKNOWN` se conserva.', '',
        '## Definiciones y denominadores','']
    for term,definition in DEFINITIONS.items():lines.append(f'- **{term}:** {definition}')
    lines += ['', '## Límites de conclusión',''] + ['- '+x for x in LIMITATIONS]
    lines += ['', '## Decisión que estos resultados permiten','',
        'Identificar qué alcance y qué concentración familiar/cromosómica tienen las configuraciones ya ejecutadas, y qué medidas siguen sin soporte. Permiten justificar los controles de evaluaciones posteriores; por sí solos no demuestran estructura adicional rara ni sustituyen la evaluación independiente de comunidades o predicciones.', '']
    return summaries,changes,'\n'.join(lines)


def run(args):
    package,legacy,ibd,files,evidence=authenticate(args)
    output=interpret(package,legacy,files,args.max_table_rows)
    summaries,changes,markdown=describe(output,package,legacy)
    # Recheck the bytes actually interpreted before publishing a complete marker.
    for record in evidence:
        require(digests(record['path'])['sha256']==record['sha256'], 'Source changed during interpretation')
    out=Path(args.output)
    require(not out.exists(), 'Output directory already exists; never overwrite')
    out.mkdir(parents=True,mode=0o700)
    write_tsv(out/'configuration_interpretation.tsv',output)
    write_tsv(out/'threshold_summary.tsv',summaries)
    if changes:write_tsv(out/'threshold_changes.tsv',changes)
    (out/'RESUMEN.md').write_text(markdown)
    report=dict(schema=SCHEMA,status=STATUS,n_cases=len(output),n_samples=package['n_samples'],
        cohort_sha256=package['cohort_sha256'],chromosomes=package['chromosomes'],
        definitions=DEFINITIONS,limitations=LIMITATIONS,threshold_summary=summaries,
        no_pvalues=True,no_winner_selected=True,no_population_labels=True,source_package_sha256=args.package_sha256,
        source_publication_verification='Prior authenticated receipts; no new GCS API request by this report',
        source_files=evidence,rows=output)
    write_json(out/'interpretation.json',report)
    outputs={p.name:digests(p)['sha256'] for p in out.iterdir() if p.is_file()}
    manifest=dict(schema=SCHEMA,status=STATUS,n_cases=len(output),n_samples=package['n_samples'],
        cohort_sha256=package['cohort_sha256'],source_package_sha256=args.package_sha256,
        outputs_sha256=outputs,code_sha256=digests(__file__)['sha256'],
        resources=dict(max_input_mib=args.max_input_mib,max_table_rows=args.max_table_rows),
        no_pvalues=True,no_winner_selected=True,no_population_labels=True,private_categories_exported=False)
    write_json(out/'manifest.json',manifest)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('package','package-publication','integration-receipt','ibd-receipt','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--package-sha256',required=True)
    p.add_argument('--integration-dir',type=Path);p.add_argument('--ibd-dir',type=Path)
    p.add_argument('--max-input-mib',type=int,default=64);p.add_argument('--max-table-rows',type=int,default=50000)
    args=p.parse_args()
    print(json.dumps(run(args),sort_keys=True))


if __name__=='__main__':main()
