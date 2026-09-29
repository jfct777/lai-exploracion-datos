#!/usr/bin/env python3
"""Private annotation companions to saved M16.5 figures, without redrawing them."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import importlib.metadata
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import textwrap

sys.path.insert(0, str(Path(__file__).parent))
import m165_sweep_figures as saved

DEFAULTS = dict(dpi=300, rows_per_page=14, max_lines_per_page=28, top_communities=5,
                width_inches=14.0, height_inches=10.0, original_figure_link_prefix='.')
MISSING = '__MISSING__'
ANCESTRIES = (('Autosomes_African_anc', 'Africana'), ('Autosomes_European_anc', 'Europea'),
              ('Autosomes_Indigenous_anc', 'Indígena'), ('Autosomes_EastAsian_anc', 'Este asiático'))
EXPECTED_CELLS = {(f'L{length}_G50000_N20_T{threshold}_U0', gamma)
                  for length, threshold in saved.GRID for gamma in saved.GAMMAS}


def settings_from_json(text=None):
    supplied = {} if text is None else json.loads(text)
    saved.require(isinstance(supplied, dict) and not set(supplied)-set(DEFAULTS), 'Unknown presentation setting')
    settings = {**DEFAULTS, **supplied}
    for name in ('dpi', 'rows_per_page', 'max_lines_per_page', 'top_communities'):
        saved.require(type(settings[name]) is int and settings[name] > 0, 'Positive integer presentation setting required')
    saved.require(72 <= settings['dpi'] <= 600 and settings['rows_per_page'] <= 20
                  and 12 <= settings['max_lines_per_page'] <= 32 and settings['top_communities'] <= 10,
                  'Presentation density exceeds the readable bounds')
    saved.require(12 <= settings['width_inches'] <= 18 and 9 <= settings['height_inches'] <= 14,
                  'Invalid page dimensions')
    prefix = settings['original_figure_link_prefix']
    saved.require(isinstance(prefix, str) and re.fullmatch(r'[A-Za-z0-9_./-]+', prefix)
                  and not PurePosixPath(prefix).is_absolute(), 'Original figure link prefix must be a relative path')
    return settings


def safe_file(root, relative):
    saved.require(isinstance(relative, str) and relative and '\\' not in relative
                  and not any(ord(c)<32 for c in relative), 'Invalid artifact path')
    path = PurePosixPath(relative)
    saved.require(not path.is_absolute() and all(p not in ('', '.', '..') for p in relative.split('/')),
                  'Absolute or traversing artifact path')
    candidate = Path(root).joinpath(*path.parts)
    saved.require(candidate.is_file() and candidate.resolve().is_relative_to(Path(root).resolve()),
                  'Artifact is missing or escapes its declared bundle')
    return candidate


def load_bundle(root, status):
    root = Path(root)
    manifest_path = safe_file(root, 'manifest.json')
    manifest = json.loads(manifest_path.read_text())
    saved.require(manifest.get('status') == status and manifest.get('no_reclustering') is True,
                  'Unexpected or incomplete upstream manifest')
    outputs = manifest.get('outputs_sha256')
    saved.require(isinstance(outputs, dict) and outputs, 'Missing upstream output digests')
    hashes = {'manifest.json': saved.sha256(manifest_path)}
    for name, expected in outputs.items():
        saved.require(isinstance(expected, str) and re.fullmatch(r'[0-9a-f]{64}', expected), 'Malformed artifact digest')
        observed = saved.sha256(safe_file(root, name))
        saved.require(observed == expected, 'Upstream output checksum mismatch')
        hashes[name] = observed
    return dict(root=root, manifest=manifest, hashes=hashes)


def bundle_table(bundle, name):
    saved.require(name in bundle['manifest']['outputs_sha256'], 'Uncertified summary table')
    return saved.table(safe_file(bundle['root'], name))


def cell_key(row):
    key = (row['config_id'], float(row['resolution']))
    saved.require(key in EXPECTED_CELLS, 'Unrecognized graph-resolution cell')
    return key


def indexed_cells(rows):
    keyed = {cell_key(row): row for row in rows}
    saved.require(len(keyed) == len(rows) == 42 and set(keyed) == EXPECTED_CELLS,
                  'Missing or duplicated saved graph-resolution cell')
    return keyed


def integer(row, key):
    value = int(row[key]); saved.require(value >= 0, 'Negative aggregate count')
    return value


def load_inputs(figures_dir, metadata_dir, kinship_dir):
    bundles = {
        'figures': load_bundle(figures_dir, 'COMPLETE_HISTORICAL_VISUAL_REPRODUCTION'),
        'metadata': load_bundle(metadata_dir, 'COMPLETE_DESCRIPTIVE_METADATA_SUMMARY'),
        'kinship': load_bundle(kinship_dir, 'COMPLETE_DESCRIPTIVE_EDGE_KINSHIP')}
    figures, metadata, kinship = (bundles[name]['manifest'] for name in ('figures', 'metadata', 'kinship'))
    anchors = metadata['input_sha256_before']
    saved.require(anchors and anchors == metadata['input_sha256_after'] == kinship['input_sha256'],
                  'Metadata and kinship summaries do not share unchanged graph inputs')
    saved.require(figures['input_sha256_before'] == figures['input_sha256_after'] and
                  all(figures['input_sha256_before'].get(k) == v for k, v in anchors.items()),
                  'Figures and summaries do not share unchanged source graphs')
    saved.require(len({m['source_core_sha256'] for m in (figures, metadata, kinship)}) == 1,
                  'Source clustering core differs between bundles')
    saved.require(figures['metadata']['sha256'] == metadata['metadata_sha256'], 'Figure annotations use different metadata')
    saved.require(metadata['contains_individual_identifiers'] is False and kinship['contains_individual_identifiers'] is False,
                  'Only aggregate summaries are eligible')
    saved.require(figures['n_cells'] == figures['n_png'] == metadata['n_cells'] == kinship['n_partitions'] == 42,
                  'Expected all 42 saved cells')
    records = indexed_cells(figures['figures'])
    prefix = figures['parameters']['prefix']
    invariants = indexed_cells(bundle_table(bundles['figures'], prefix+'_invariants.tsv'))
    partitions = indexed_cells(bundle_table(bundles['metadata'], 'partition_metadata_summary.tsv'))
    kin_partitions = indexed_cells(bundle_table(bundles['kinship'], 'partition_kinship_summary.tsv'))
    graph_rows = bundle_table(bundles['kinship'], 'graph_kinship_summary.tsv')
    graph_kinship = {row['config_id']: row for row in graph_rows}
    saved.require(len(graph_rows) == len(graph_kinship) == 6, 'Missing or duplicated graph kinship summary')
    sizes = defaultdict(list); categories = defaultdict(list); ancestry = defaultdict(list)
    for row in bundle_table(bundles['metadata'], 'community_sizes.tsv'): sizes[cell_key(row)].append(row)
    for row in bundle_table(bundles['metadata'], 'categorical_distributions.tsv'): categories[cell_key(row)].append(row)
    for row in bundle_table(bundles['metadata'], 'ancestry_summary.tsv'): ancestry[cell_key(row)].append(row)
    correspondences = []; inverse_correspondences = []
    for key in sorted(EXPECTED_CELLS):
        record, inv, part, kin = records[key], invariants[key], partitions[key], kin_partitions[key]
        expected_stem = f'{prefix}_{key[0]}_gamma{key[1]:g}'
        saved.require(record['stem'] == expected_stem and re.fullmatch(r'[A-Za-z0-9_.-]+', expected_stem),
                      'Unexpected figure filename')
        saved.require(expected_stem+'.png' in figures['outputs_sha256'], 'Primary PNG is not certified')
        for fkey, pkey in (('n_nodes', 'n_cohort'), ('n_active', 'n_active'), ('n_isolated', 'n_isolated'),
                           ('n_assigned', 'n_assigned'), ('n_communities', 'n_communities')):
            saved.require(integer(inv, fkey) == int(record[fkey]) == integer(part, pkey) == integer(kin, pkey),
                          'Figure, metadata and kinship denominators disagree')
        saved.require(integer(inv, 'n_nodes') == metadata['n_cohort'] and
                      integer(inv, 'n_edges') == integer(kin, 'all_n_edges') == integer(graph_kinship[key[0]], 'n_edges'),
                      'Graph edge or cohort denominators disagree')
        labels = {int(row['community_local']): row for row in sizes[key]}
        saved.require(len(labels) == len(sizes[key]) == integer(part, 'n_communities') and
                      sum(integer(r, 'n_samples') for r in sizes[key]) == integer(part, 'n_assigned'),
                      'Community size table does not reproduce the partition')
        reference = [row for row in categories[key] if row['scope'] == 'community' and row['field'] == 'finestructure_clusters']
        seen = set(); totals = Counter()
        for row in reference:
            label = int(row['community_local']); pair = (label, row['category'])
            saved.require(pair not in seen and label in labels, 'Repeated or unknown community/reference category')
            seen.add(pair); n = integer(row, 'n'); denominator = integer(row, 'denominator_all')
            saved.require(denominator == integer(labels[label], 'n_samples') and n <= denominator,
                          'Category denominator differs from its community')
            totals[label] += n
            correspondences.append(dict(config_id=key[0], resolution=key[1], community_local=label,
                fineSTRUCTURE=row['category'], is_missing=row['is_missing'], n=n, denominator=denominator,
                percent=100*n/denominator if denominator else None,
                small_cell=row['small_cell'], original_figure=expected_stem+'.png'))
        saved.require(all(totals[label] == integer(row, 'n_samples') for label, row in labels.items()),
                      'FineSTRUCTURE categories do not sum to every community')
        inverse_correspondences.extend(inverse_reference_rows(
            categories[key], key, integer(part, 'n_cohort'), integer(part, 'n_unassigned'), expected_stem+'.png'))
    return dict(bundles=bundles, records=records, invariants=invariants, partitions=partitions,
                kinship=kin_partitions, graph_kinship=graph_kinship, sizes=sizes,
                categories=categories, ancestry=ancestry, correspondences=correspondences,
                inverse_correspondences=inverse_correspondences)


def inverse_reference_rows(rows, key, n_cohort, n_unassigned, original_figure):
    """Invert aggregate membership with the whole-cohort category denominator."""
    reference = [r for r in rows if r['field'] == 'finestructure_clusters']
    scopes = {}
    for scope, expected in (('cohort', n_cohort), ('unassigned', n_unassigned)):
        selected = [r for r in reference if r['scope'] == scope]
        values = {r['category']: r for r in selected}
        saved.require(len(values) == len(selected) and values and
                      all(integer(r, 'denominator_all') == expected for r in selected) and
                      sum(integer(r, 'n') for r in selected) == expected,
                      'FineSTRUCTURE cohort/unassigned categories do not conserve their denominator')
        scopes[scope] = values
    communities = defaultdict(list)
    for row in reference:
        if row['scope'] == 'community' and integer(row, 'n'):
            communities[row['category']].append(row)
    saved.require(set(communities) <= set(scopes['cohort']) and
                  set(scopes['unassigned']) <= set(scopes['cohort']), 'Reference category missing from cohort')
    output = []
    for category, source in sorted(scopes['cohort'].items()):
        denominator = integer(source, 'n')
        if not denominator: continue
        assigned = sorted(communities[category], key=lambda r: int(r['community_local']))
        unassigned = integer(scopes['unassigned'].get(category, {'n': 0}), 'n')
        saved.require(sum(integer(r, 'n') for r in assigned)+unassigned == denominator,
                      'FineSTRUCTURE inverse correspondence does not conserve whole-cohort category')
        for label, n in [(int(r['community_local']), integer(r, 'n')) for r in assigned]+[('sin_asignar', unassigned)]:
            output.append(dict(config_id=key[0], resolution=key[1], fineSTRUCTURE=category,
                is_missing=source['is_missing'], community_local=label, n=n,
                denominator_finestructure_cohort=denominator, percent_of_finestructure_cohort=100*n/denominator,
                small_cell=n<5, original_figure=original_figure))
    return output


def inverse_markdown(rows):
    groups = defaultdict(list)
    for row in rows: groups[row['fineSTRUCTURE']].append(row)
    lines = ['## fineSTRUCTURE → comunidades y personas sin asignar', '',
        'Lectura inversa: N es el TOTAL de esa categoría en la cohorte completa, no sólo las personas asignadas. '
        'Cada fila suma N entre comunidades y sin asignar (incluye aislados y activos no asignados; gris del original). '
        'Las comunidades con cero personas de esa categoría se omiten; sin asignar se muestra incluso si es cero. '
        'Es una correspondencia muchos a muchos, no una jerarquía ni una equivalencia poblacional.', '',
        '| Categoría fineSTRUCTURE | N en la cohorte | Distribución hacia comunidades: n/N (%) |',
        '|---|---:|---|']
    for category, values in sorted(groups.items()):
        parts = []
        for row in values:
            label = 'sin asignar' if row['community_local'] == 'sin_asignar' else f"C{row['community_local']}"
            denominator = row['denominator_finestructure_cohort']
            parts.append(f"{label}: {row['n']}/{denominator} ({percent(row['n'], denominator)})")
        title = 'Sin dato de fineSTRUCTURE' if category == MISSING else plain(category)
        lines.append(f"| {title} | {values[0]['denominator_finestructure_cohort']} | {'; '.join(parts)} |")
    lines.extend(['', 'Conteos completos: `finestructure_a_comunidades.tsv`; porcentajes redondeados sólo para lectura.', ''])
    return lines


def percent(n, denominator):
    return f'{100*n/denominator:.1f} %' if denominator else 'no definido'


def plain(value):
    return str(value).replace('\n', ' ').replace('\r', ' ').replace('|', '\\|').replace('`', "'")


def number(value, digits=3):
    return 'no disponible' if value in ('', 'NA', None) else f'{float(value):.{digits}f}'


def correspondence_blocks(rows):
    groups = defaultdict(list)
    for row in rows: groups[int(row['community_local'])].append(row)
    blocks = []
    for label, values in sorted(groups.items(), key=lambda item: (-int(item[1][0]['denominator']), item[0])):
        parts = []
        for row in sorted(values, key=lambda r: (-int(r['n']), r['fineSTRUCTURE'])):
            if not int(row['n']): continue
            category = 'Sin dato' if row['fineSTRUCTURE'] == MISSING else row['fineSTRUCTURE']
            parts.append(f"{category}: {row['n']}/{row['denominator']} ({percent(int(row['n']), int(row['denominator']))})")
        lines = textwrap.wrap('   ·   '.join(parts), width=103, break_long_words=False, break_on_hyphens=False)
        blocks.append(dict(community=label, n=int(values[0]['denominator']), lines=lines or ['Sin observaciones']))
    return blocks


def paginate(blocks, settings):
    pages, page, used = [], [], 0
    for block in blocks:
        cost = len(block['lines'])+1
        saved.require(cost <= settings['max_lines_per_page'], 'One correspondence exceeds page density limit')
        if page and (len(page) >= settings['rows_per_page'] or used+cost > settings['max_lines_per_page']):
            pages.append(page); page=[]; used=0
        page.append(block); used += cost
    if page: pages.append(page)
    return pages or [[]]


def render_correspondence(rows, output, stem, title, settings):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    pages = paginate(correspondence_blocks(rows), settings)
    pngs = []
    with PdfPages(output/(stem+'.correspondencia.pdf')) as book:
        for index, page in enumerate(pages, 1):
            fig = plt.figure(figsize=(settings['width_inches'], settings['height_inches']), facecolor='white')
            ax = fig.add_axes([.025, .15, .95, .69]); ax.set_xlim(0,1)
            units = max(sum(len(block['lines'])+1 for block in page), 8)
            ax.set_ylim(units+.5, -.5); ax.axis('off')
            cursor = 0
            for block in page:
                bottom = cursor+len(block['lines'])-1; middle=(cursor+bottom)/2
                ax.text(.005,middle,f"C{block['community']}\nN={block['n']}",ha='left',va='center',fontsize=11,color='#202020')
                ax.annotate('',xy=(.165,middle),xytext=(.105,middle),arrowprops=dict(arrowstyle='->',color='#24577A',lw=1))
                ax.plot([.19,.17,.17,.19],[cursor-.28,cursor-.28,bottom+.28,bottom+.28],color='#24577A',lw=1.2)
                for offset,line in enumerate(block['lines']):
                    ax.text(.205,cursor+offset,line,ha='left',va='center',fontsize=10,color='#202020')
                cursor += len(block['lines'])+1
            if not page: ax.text(.1,1,'Sin comunidades asignadas; no se omiten personas de la cohorte.',fontsize=11)
            fig.text(.03,.965,'Comunidad → composición fineSTRUCTURE',fontsize=17,fontweight='bold',va='top')
            fig.text(.03,.92,title+f'\nComplemento de leyendas · página {index}/{len(pages)} · todas las categorías presentes, sin agrupar',fontsize=11,va='top')
            fig.text(.03,.10,'Cada corchete reúne categorías de una comunidad; n/N usa TODAS sus personas, incluidos faltantes.\n'
                     'Una categoría puede aparecer en varias comunidades: correspondencia muchos a muchos, no equivalencia.\n'
                     'C# es local a configuración y gamma; no representa ancestría. El gris del original también marca no asignados.\n'
                     'No se cambian puntos ni geometría del original. Conteos agregados privados; no validación biológica.',fontsize=10,va='top')
            name=stem+'.correspondencia'+('' if index==1 else f'.p{index:03d}')+'.png'
            fig.savefig(output/name,dpi=settings['dpi'],facecolor='white');book.savefig(fig,dpi=settings['dpi']);plt.close(fig)
            pngs.append(name)
    return dict(pngs=pngs, pdf=stem+'.correspondencia.pdf', n_pages=len(pages), n_communities=len({r['community_local'] for r in rows}))


def category_text(rows, field, scope='assigned', community=''):
    selected=[r for r in rows if r['field']==field and r['scope']==scope and str(r['community_local'])==str(community)]
    if not selected: return 'no disponible'
    denominator=int(selected[0]['denominator_all']); missing=int(selected[0]['n_missing'])
    ranked=sorted((r for r in selected if r['is_missing']=='False' and int(r['n'])>0),key=lambda r:(-int(r['n']),r['category']))
    shown=ranked[:3];other=sum(int(r['n']) for r in ranked[3:])
    text='; '.join(f"{plain(r['category'])}: {r['n']}/{denominator} ({percent(int(r['n']),denominator)})" for r in shown)
    if other: text+=f'; otras categorías: {other}/{denominator}'
    return (text+'; ' if text else '')+f'faltantes: {missing}/{denominator}'


def interpretation(data, key, settings):
    inv, part, kin = data['invariants'][key], data['partitions'][key], data['kinship'][key]
    stem=data['records'][key]['stem']; n=int(part['n_cohort']); assigned=int(part['n_assigned'])
    original_link=str(PurePosixPath(settings['original_figure_link_prefix'])/(stem+'.png'))
    text=[f'# Interpretación descriptiva · {stem}', '',
          f"Figura original: [{stem}.png]({original_link}). El nombre identifica el original en el directorio de figuras, no una copia modificada.",
          f"Complemento: [{stem}.correspondencia.pdf]({stem}.correspondencia.pdf); todas las categorías y comunidades figuran también en `correspondencias_completas.tsv`.", '',
          '## Cobertura y configuración', '',
          f"Chr22; L={int(inv['length_bp'])/1000:g} kb por segmento, T={int(inv['threshold_bp'])/1e6:g} Mb acumulados por pareja, G=50 kb, N=20 sitios, U=0; gamma={key[1]:g}.",
          f"Cohorte: {n}. Activos: {part['n_active']}; aislados: {part['n_isolated']}; asignados: {assigned}/{n} ({percent(assigned,n)}); activos no asignados: {part['n_active_unassigned']}. {part['n_communities']} comunidades ≥3 personas; la mayor reúne {part['largest_community']}/{assigned} asignados ({percent(int(part['largest_community']),assigned)}).", '',
          '## Estabilidad y correspondencia, métricas distintas', '',
          f"ARI entre semillas: mediana {number(inv['median_ari'])}, cuartiles [{number(inv['q25_ari'])}, {number(inv['q75_ari'])}]. Son 300 comparaciones dependientes entre 25 semillas sobre nodos activos antes del filtro de tamaño: no réplicas biológicas ni intervalo de confianza.",
          f"Comparación con fineSTRUCTURE entre asignados con referencia observada (N={part['n_assigned_with_reference']}; faltantes={part['n_assigned_missing_reference']}): ARI={number(part['reference_ari'])}, NMI={number(part['reference_nmi'])}, AMI={number(part['reference_ami'])}; estado={part['agreement_status']}. ARI ajusta coincidencia de pares; NMI/AMI resumen información compartida normalizada/ajustada.",
          f"Pureza ponderada={number(part['weighted_purity_assigned'])}; fracción de la categoría de referencia dominante={number(part['dominant_reference_share_assigned'])}. Pureza puede subir al subdividir más: no identifica un gamma óptimo. Ninguna métrica convierte las comunidades en poblaciones confirmadas.", '',
          '## Composición y ancestría global de referencia', '',
          'Cuatro componentes autosómicos de metadata: no ADMIXTURE de-novo K=3, ancestría local, origen de un segmento ni porcentaje de parentesco. Mediana y rango intercuartílico (IQR) describen dispersión, no incertidumbre inferencial. Son resúmenes marginales por componente: las cuatro medianas no tienen por qué sumar 100 %, aunque las fracciones de cada persona sí formen una composición.', '']
    sizes=sorted(data['sizes'][key],key=lambda r:(-int(r['n_samples']),int(r['community_local'])))
    selected=[('assigned','', 'Todos los asignados'),('unassigned','', 'Todos los no asignados')]
    selected.extend(('community',r['community_local'],f"C{r['community_local']} (N={r['n_samples']})") for r in sizes[:settings['top_communities']])
    for scope,label,title in selected:
        text.append(f'### {title}');text.append('')
        text.append('fineSTRUCTURE: '+category_text(data['categories'][key],'finestructure_clusters',scope,label)+'.')
        text.append('')
        for field,human in ANCESTRIES:
            matches=[r for r in data['ancestry'][key] if r['scope']==scope and str(r['community_local'])==str(label) and r['field']==field]
            saved.require(len(matches)==1,'Missing or duplicate ancestry summary')
            row=matches[0]
            if row['suppressed_small_n']=='True':
                text.append(f"- {human}: N={row['n_observed']}, faltantes={row['n_missing']}; mediana/IQR suprimidos por N<{row['disclose_min_n']}.")
            else:
                text.append(f"- {human}: mediana {100*float(row['median']):.1f} %; Q25–Q75 [{100*float(row['q25']):.1f}, {100*float(row['q75']):.1f}] %; IQR={100*float(row['iqr']):.1f} puntos porcentuales; N={row['n_observed']}, faltantes={row['n_missing']}.")
        text.extend(['', 'Region: '+category_text(data['categories'][key],'Region',scope,label)+'.',
                     'State: '+category_text(data['categories'][key],'State',scope,label)+'.',
                     'Cohort: '+category_text(data['categories'][key],'Cohort',scope,label)+'.', ''])
    text.extend(inverse_markdown([r for r in data['inverse_correspondences']
                                  if (r['config_id'], r['resolution']) == key]))
    text.extend(['## Contraste descriptivo con PC-Relate', '',
        f"Umbral operativo φ≥{kin['kinship_threshold']}; coeficiente estimado, no familia certificada. Entre las {kin['all_n_edges']} conexiones retenidas: {kin['all_n_kin_ge_threshold']} sobre el corte, {kin['all_n_kin_lt_threshold']} por debajo y {kin['all_n_missing_kinship']} sin valor finito/evaluable. Fracción sobre el corte entre todas las conexiones={number(kin['all_fraction_kin_ge_all_edges'])}; entre observadas={number(kin['all_fraction_kin_ge_observed_edges'])}.", '',
        '| Conexiones retenidas | Total | φ sobre corte | Observadas | Ausentes/no finitas |', '|---|---:|---:|---:|---:|'])
    for name,label in (('within_assigned','Dentro de comunidad asignada'),('between_assigned','Entre comunidades asignadas'),('with_unassigned','Algún extremo no asignado')):
        text.append(f"| {label} | {kin[name+'_n_edges']} | {kin[name+'_n_kin_ge_threshold']} | {kin[name+'_n_observed_kinship']} | {kin[name+'_n_missing_kinship']} |")
    text.extend(['', 'Los denominadores son conexiones del grafo, NO todos los pares posibles dentro de cada comunidad. Estas clases son disjuntas; el parentesco global del grafo no cambia al variar gamma. Una fracción menor puede reflejar un denominador mayor de conexiones, no menos parejas sobre el corte: comparar también conteos absolutos. PC-Relate preexistente usa autosomas e incluye chr22: contraste complementario, no validación genéticamente independiente. Por debajo del corte no significa personas independientes.', '',
        '## Límites de interpretación', '',
        'Los corchetes muestran pertenencias muchos a muchos, no traducciones de C# a poblaciones. C# y colores son locales a cada gamma/configuración; no equiparar etiquetas entre figuras. En el panel fineSTRUCTURE el gris también atenúa no asignados Leiden, no indica falta de metadata. La metadata actual tiene sus propios niveles: no se presupone equivalencia exacta con una figura publicada.',
        'La geometría espectral→UMAP conserva coordenadas por grafo entre resoluciones; no alinea grafos diferentes. Islas, distancias y posiciones de aislados no prueban separación poblacional. Concordancia puede reflejar ancestría global, parentesco, densidad, reclutamiento o selección de personas. Region/State se presentan como etiquetas registradas, no se infiere lugar de nacimiento ni migración.',
        'Las 42 celdas reutilizan cohortes y variantes: no son réplicas independientes. No p-valores, selección automática óptima, genealogías, ancestrías nuevas ni confirmación biológica. Resumen privado: los grupos pequeños pueden seguir siendo identificables aun sin IDs.', ''])
    return '\n'.join(text)


def run(figures_dir, metadata_dir, kinship_dir, output_dir, settings_json_text=None):
    settings=settings_from_json(settings_json_text); output=Path(output_dir)
    saved.require(not output.exists() and not output.is_symlink(),'Output directory exists; no overwrite')
    for root in (figures_dir,metadata_dir,kinship_dir):
        saved.require(not output.resolve().is_relative_to(Path(root).resolve()),'Output must be outside input bundles')
    source_hash=saved.sha256(__file__);validator_hash=saved.sha256(saved.__file__)
    data=load_inputs(figures_dir,metadata_dir,kinship_dir)
    output.mkdir(parents=True,exist_ok=False)
    saved.write_table(output/'correspondencias_completas.tsv',data['correspondences'])
    saved.write_table(output/'finestructure_a_comunidades.tsv',data['inverse_correspondences'])
    records=[]
    for key in sorted(EXPECTED_CELLS):
        stem=data['records'][key]['stem']; inv=data['invariants'][key]
        rows=[row for row in data['correspondences'] if (row['config_id'],row['resolution'])==key]
        title=f"Chr22 · L={int(inv['length_bp'])/1000:g} kb · T={int(inv['threshold_bp'])/1e6:g} Mb · gamma={key[1]:g}"
        rendered=render_correspondence(rows,output,stem,title,settings)
        (output/(stem+'.interpretacion.md')).write_text(interpretation(data,key,settings),encoding='utf-8')
        records.append(dict(config_id=key[0],resolution=key[1],original_figure=stem+'.png',
            interpretation=stem+'.interpretacion.md',**rendered))
    hashes={}
    for kind,bundle in data['bundles'].items():
        after={name:saved.sha256(safe_file(bundle['root'],name)) for name in bundle['hashes']}
        saved.require(after==bundle['hashes'],'An original input changed during presentation')
        hashes[kind]=dict(path=str(bundle['root'].resolve()),before=bundle['hashes'],after=after)
    saved.require(saved.sha256(__file__)==source_hash and saved.sha256(saved.__file__)==validator_hash,'Presentation source changed')
    (output/'README.md').write_text('''# Complementos de correspondencia para 42 figuras M16.5

Los originales no se copian ni modifican. Cada interpretación identifica su PNG
fuente por nombre; los directorios de origen y hashes constan en manifest.json.
La distribución final coloca estos complementos junto a los PNG originales.
Por defecto los enlaces asumen esa ubicación adyacente: en el directorio local
de interpretación separado pueden no resolverse hasta preparar la publicación.
original_figure_link_prefix permite declarar otra ruta relativa de presentación.
Cada PDF de correspondencia reúne todas sus páginas; PNG sin sufijo es página 1
y .p002, .p003... son continuaciones. Los corchetes cubren todas las comunidades
y categorías presentes, sin combinar categorías minoritarias. El TSV conserva
los conteos exactos, incluidos faltantes y categorías de conteo cero registradas.
Los porcentajes de cada corchete usan todas las personas de su comunidad.
Cada interpretación también incluye la dirección inversa: fineSTRUCTURE hacia
comunidades y sin asignar. Su N incluye TODAS las personas de esa categoría en
la cohorte, incluidos aislados y activos no asignados. La tabla completa está en
finestructure_a_comunidades.tsv. Ambas direcciones son muchos a muchos.

Las notas describen los agregados autenticados. No ajustan agrupamientos, no
atribuyen causas, no confirman poblaciones y no eligen parámetros óptimos.
No se exportan IDs, genotipos, posiciones individuales ni metadata clínica.
Artefactos privados: conteos pequeños no garantizan anonimato.
''',encoding='utf-8')
    manifest=dict(status='COMPLETE_DESCRIPTIVE_METADATA_PRESENTATION',n_cells=42,n_interpretations=42,
        n_correspondence_pdfs=len(records),n_correspondence_pngs=sum(len(r['pngs']) for r in records),
        records=records,settings=settings,input_bundles=hashes,source_sha256=source_hash,validator_sha256=validator_hash,
        no_reclustering=True,no_new_genotypes=True,no_new_kinship=True,no_original_geometry_changes=True,
        originals_changed=0,contains_individual_identifiers=False,public_distribution_allowed=False,biological_validation=False,
        versions={p:importlib.metadata.version(p) for p in ('matplotlib','numpy')},
        outputs_sha256={p.name:saved.sha256(p) for p in sorted(output.iterdir())})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('figures-dir','metadata-dir','kinship-dir','output-dir'): parser.add_argument('--'+flag,required=True)
    parser.add_argument('--settings-json-text')
    result=run(**vars(parser.parse_args()))
    print(json.dumps({k:result[k] for k in ('status','n_cells','n_correspondence_pngs')}))


if __name__=='__main__':
    try: main()
    except Exception as error:
        print('Metadata presentation stopped ('+type(error).__name__+'); input contract or rendering check failed',file=sys.stderr)
        raise SystemExit(1) from None
