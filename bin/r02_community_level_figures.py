#!/usr/bin/env python3
"""Render a private, auditable presentation from saved community-level metrics.

Selection here is descriptive and retrospective. It does not identify a true
population, tune the upstream graph, or replace held-out genomic evaluation.
Only aggregate tables enter the presentation bundle; individual IDs stay in the
upstream private evidence product. Small aggregate cells are still private.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import textwrap

import r02_community_levels as base

SCHEMA = 'r02_community_level_presentation_v1'
STATUS = 'COMPLETE_DESCRIPTIVE_PRESENTATION_NOT_POPULATION_VALIDATION'
COLORS = {'H': '#777777', 'R': '#0072B2', 'C': '#D55E00', 'R_plus_C': '#009E73'}
NAMES = {'H': 'Longitudes (H)', 'R': 'Raras (R)', 'C': 'Comunes (C)', 'R_plus_C': 'Raras + comunes (R+C)'}
QCOLORS = ['#0072B2', '#E69F00', '#009E73', '#CC79A7']
QNAMES = ['AFR', 'EUR', 'NAM', 'EAS']

def rows(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))

def num(value):
    if value in ('NA', '', None):
        return None
    result = float(value)
    base.require(math.isfinite(result), 'Nonfinite metric')
    return result

def label(row):
    return f"{row['graph_id']} / γ={float(row['resolution']):g}"

def representative(configs, graph):
    """Select an illustrative partition only among fully covered alternatives.

    Highest DF-adjusted ancestry association; then fewer groups, then lower
    resolution. It is an explicit plotting rule, not a biological optimum.
    """
    eligible = [r for r in configs if r['graph_id'] == graph
                and int(r['n_assigned']) == int(r['n_cohort']) and int(r['n_communities']) > 1
                and num(r['eta2_adjusted']) is not None]
    base.require(eligible, 'No full-coverage nondegenerate presentation candidate: ' + graph)
    return max(eligible, key=lambda r: (num(r['eta2_adjusted']), -int(r['n_communities']), -float(r['resolution'])))

def frontier(configs, shared):
    """Retain tradeoffs on same-support eta, coverage, K and seed stability."""
    def point(r):
        score = num(shared[r['partition_id']]['eta2'])
        return None if score is None else (score, int(r['n_assigned']), -int(r['n_communities']), num(r['median_ari']))
    points = {r['partition_id']: point(r) for r in configs}
    selected = set()
    for pid, p in points.items():
        if p is not None and not any(q is not None and all(a >= b for a, b in zip(q, p))
                                    and any(a > b for a, b in zip(q, p)) for q in points.values()):
            selected.add(pid)
    return selected

def write_table(path, values):
    base.require(bool(values), 'Cannot write empty table')
    with path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(values)

def savefig(fig, out, name, caption):
    import matplotlib.pyplot as plt
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(out / f'{name}.{extension}', dpi=300, facecolor='white')
    with (out / f'{name}.descripcion.md').open('x', encoding='utf-8') as handle:
        handle.write(caption + '\n')
    plt.close(fig)

def render(evidence, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    evidence = evidence.resolve()
    manifest = json.loads((evidence / 'manifest.json').read_text())
    base.require(manifest['schema'] == base.SCHEMA and manifest['status'] == base.STATUS, 'Wrong evidence schema/status')
    required = {'config_metrics.tsv', 'community_metrics.tsv', 'summary.json', 'kinship_reused.json'}
    base.require(required <= set(manifest['outputs_sha256']), 'Missing authenticated presentation inputs')
    for filename, checksum in manifest['outputs_sha256'].items():
        base.require(Path(filename).name == filename and base.sha256(evidence / filename) == checksum, 'Evidence checksum mismatch')
    summary = json.loads((evidence / 'summary.json').read_text())
    metrics, communities = rows(evidence / 'config_metrics.tsv'), rows(evidence / 'community_metrics.tsv')
    configs = [r for r in metrics if r['support'] == 'assigned_complete' and r['weighting'] == 'person']
    common_rows = [r for r in metrics if r['support'] == 'common_all_partitions' and r['weighting'] == 'person']
    shared = {r['partition_id']: r for r in common_rows}
    base.require(len(configs) == summary['n_partitions'] == len(shared), 'Partition inventory incomplete')
    base.require(len(common_rows) == len(shared) and len({r['partition_id'] for r in configs}) == len(configs)
                 and {r['partition_id'] for r in configs} == set(shared), 'Duplicate or mismatched partition IDs')
    base.require(not out.exists(), 'No overwrite of figures')
    out.mkdir(mode=0o700)
    for filename in ('config_metrics.tsv', 'community_metrics.tsv', 'summary.json', 'kinship_reused.json'):
        shutil.copy2(evidence / filename, out / filename)
    # Do not copy individuals.private.tsv into the presentation bundle.
    representatives = [representative(configs, graph) for graph in ('primary/R', 'primary/C', 'primary/R_plus_C')]
    nondominated = frontier(configs, shared)
    selection = [dict(graph_id=r['graph_id'], resolution=r['resolution'], partition_id=r['partition_id'],
                      role='ILLUSTRATIVE_ANCESTRY_DESCRIPTION', rule='MAX_ADJUSTED_ETA_FULL_COVERAGE_WITHIN_GRAPH',
                      n_assigned=r['n_assigned'], n_communities=r['n_communities'], eta2=r['eta2'],
                      eta2_adjusted=r['eta2_adjusted'], biological_optimum=False) for r in representatives]
    write_table(out / 'seleccion_figuras.tsv', selection)
    write_table(out / 'comparacion_57.tsv', [dict(**r, eta2_common_support=shared[r['partition_id']]['eta2'],
        on_descriptive_frontier=r['partition_id'] in nondominated) for r in configs])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
    for family, color in COLORS.items():
        data = [r for r in configs if r['family'] == family]
        for ax, key, title in zip(axes, ('eta2', 'fraction_assigned', 'median_ari'),
            ('A. Asociación ancestral en personas comunes', 'B. Personas representadas', 'C. Estabilidad entre semillas')):
            y = [num(shared[r['partition_id']]['eta2']) if key == 'eta2' else num(r[key]) for r in data]
            ax.scatter([int(r['n_communities']) for r in data], y, color=color, alpha=.72, s=42,
                       label=NAMES[family], edgecolors='white', linewidths=.4)
            ax.set(xscale='log', xlabel='Número de comunidades (escala log)', title=title, ylim=(-.03, 1.04))
    axes[0].set_ylabel('Fracción de variación ancestral descrita')
    axes[1].set_ylabel(f"Asignados / {summary['n_cohort']} personas")
    axes[2].set_ylabel('ARI mediano; 1 = coincidencia de etiquetas')
    axes[0].legend(loc='lower right', fontsize=9)
    fig.suptitle('57 particiones: más grupos no equivalen a más respaldo biológico', fontsize=16)
    fig.text(.05, .055, f"Panel A: las mismas {summary['n_common_assigned_complete']} personas asignadas en las 57 particiones. "
             'ARI usa los nodos activos de cada grafo.\nLas diferencias de cobertura y complejidad se muestran; no hay una puntuación de validación.', fontsize=10)
    fig.subplots_adjust(left=.06, right=.98, bottom=.25, top=.8, wspace=.35)
    savefig(fig, out, '00_comparacion_57', '# Comparación de las 57 particiones\n\nCada punto es una partición guardada. '
            'No se recalculó el agrupamiento. La asociación usa una intersección común de personas; cobertura y ARI '
            'conservan sus denominadores originales. El ARI mide concordancia entre semillas, no verdad biológica. '
            'No hay intervalos ni contrastes confirmatorios. La frontera no dominada en comparacion_57.tsv sólo conserva compromisos descriptivos.')

    fig, axes = plt.subplots(1, 3, figsize=(15, 7))
    for ax, chosen in zip(axes, representatives):
        data = [r for r in communities if r['partition_id'] == chosen['partition_id']
                and r['scope'] == 'community' and float(r['kinship_threshold']) == .0221]
        data.sort(key=lambda r: int(r['community_local']))
        y, left = np.arange(len(data)), np.zeros(len(data))
        for field, name, color in zip(base.ANCESTRY, QNAMES, QCOLORS):
            values = np.array([num(r[field + '_mean']) for r in data])
            ax.barh(y, values, left=left, color=color, label=name, height=.72)
            left += values
        ax.set_yticks(y, [f"C{r['community_local']} · n={r['n_samples']}" for r in data])
        ax.invert_yaxis()
        ax.set(xlim=(0, 1), xlabel='Proporción ancestral media estimada',
               title=f"{NAMES[chosen['family']]}\nγ={float(chosen['resolution']):g} · {chosen['n_assigned']}/{chosen['n_cohort']} personas\n"
               f"η²={float(chosen['eta2']):.3f}; {chosen['n_communities']} comunidades")
    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc='lower center', bbox_to_anchor=(.5, .10), ncol=4)
    fig.suptitle('Nivel amplio: comunidades con distinta composición ancestral', fontsize=17)
    fig.text(.055, .045, 'Medias de estimaciones existentes, no ancestrías puras ni poblaciones verdaderas. '
             'C0 de un panel no es C0 de otro.\nSelección descriptiva por cobertura completa y asociación ajustada dentro de cada grafo; '
             'no demuestra aporte exclusivo de las raras.', fontsize=10)
    fig.subplots_adjust(left=.1, right=.98, bottom=.23, top=.76, wspace=.66)
    savefig(fig, out, '01_ancestria_amplia', '# K01: composición ancestral amplia\n\n'
            'Las barras son medias AFR/EUR/NAM/EAS por comunidad; n es el número de personas. '
            'Se conserva la numeración local original. Dentro de cada grafo principal se representa la partición '
            'no degenerada con todas las personas asignadas y mayor asociación ajustada por número de comunidades; '
            'empates favorecen menos comunidades y menor resolución. Es una selección de presentación retrospectiva, '
            'no un óptimo biológico ni generalización. Las 57 alternativas y sus sensibilidades permanecen en las tablas.')

    rare = representatives[0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.8), gridspec_kw={'width_ratios': [1.35, 1]})
    for offset, phi, color in ((-.17, .0221, '#0072B2'), (.17, .0442, '#D55E00')):
        data = sorted([r for r in communities if r['partition_id'] == rare['partition_id']
                       and r['scope'] == 'community' and float(r['kinship_threshold']) == phi],
                      key=lambda r: int(r['community_local']))
        y = np.arange(len(data))
        values = [float(r['maximum_component_fraction_all']) for r in data]
        axes[0].barh(y+offset, values, height=.3, color=color, label=f'φ = {phi}')
    sensitivity = [r for r in metrics if r['partition_id'] == rare['partition_id'] and r['support'] == 'assigned_complete']
    write_table(out / 'sensibilidad_ancestral_representantes.tsv', [r for r in metrics if r['partition_id'] in
                {s['partition_id'] for s in representatives} and r['support'] == 'assigned_complete'])
    for shift, phi, color in ((-.18, None, '#777777'), (0, .0221, '#0072B2'), (.18, .0442, '#D55E00')):
        record = next(r for r in sensitivity if (r['weighting'] == 'person' if phi is None
                      else r['weighting'] == 'component_balanced' and float(r['kinship_threshold']) == phi))
        axes[1].scatter(np.arange(4)+shift, [num(record['eta2_'+q]) for q in QNAMES], color=color, s=52,
                        label='Personas' if phi is None else f'Componentes φ={phi}')
    axes[0].set_yticks(y, [f"C{r['community_local']} · n={r['n_samples']}" for r in data])
    axes[1].set_xticks(np.arange(4), QNAMES)
    axes[0].set(xlim=(0, 1.03), xlabel='Personas del componente mayor / personas de la comunidad', title='A. Concentración en un componente')
    axes[1].set(ylim=(0, 1), ylabel='Fracción de variación ancestral descrita (η²)',
                title='B. Sensibilidad de la asociación ancestral')
    axes[0].invert_yaxis()
    handles, labels_legend = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc='lower center', bbox_to_anchor=(.5, .125), ncol=3, fontsize=10)
    fig.suptitle(f"Parentesco: interpretar las {rare['n_communities']} comunidades de raras (γ={float(rare['resolution']):g})", fontsize=16)
    fig.text(.08, .05, 'Componente = personas conectadas directa o transitivamente por PC-Relate; no es un pedigrí confirmado.\n'
             'Panel B: las comunidades no cambian. Dar igual peso a cada componente cambia qué conjunto se resume.', fontsize=10)
    fig.subplots_adjust(left=.13, right=.98, bottom=.23, top=.80, wspace=.38)
    savefig(fig, out, '02_parentesco_comunidades_raras', '# K05: concentración de parentesco\n\n'
            'Se muestran los dos cortes ya establecidos, sin escoger el favorable. El denominador del panel A es '
            'el número de personas de cada comunidad; los desconocidos están declarados en community_metrics.tsv. '
            'El panel B compara la asociación ancestral con peso igual por persona y peso igual por componente, '
            'conservando las comunidades. No equivale a volver a agrupar sin familiares ni a una prueba de independencia. '
            'Un grupo con ancestría distintiva y concentración de parentesco puede reflejar ambos fenómenos.')

    levels = [
        ('K01', 'Composición ancestral amplia', 'DESCRIPCIÓN CUANTIFICADA', '01_ancestria_amplia', 'Asociación con proporciones globales; no poblaciones nuevas.'),
        ('K05', 'Concentración de parentesco', 'DESCRIPCIÓN CUANTIFICADA', '02_parentesco_comunidades_raras', 'Identifica concentración, no pedigrí ni independencia.'),
        ('K02', 'Estructura más fina', 'NO IDENTIFICADA AQUÍ', '', 'Requiere incremento retenido y controles ancestral/familiar/técnico.'),
        ('K03', 'Grupos discretos frente a continuidad', 'NO IDENTIFICADO AQUÍ', '', 'Falta comparar con modelo continuo y evaluar personas no usadas al agrupar.'),
        ('K04', 'Estructura dentro de AFR/EUR/NAM', 'NO IDENTIFICADA AQUÍ', '', 'Necesita asignación local de copias, fase e incertidumbre.'),
        ('K06', 'Fundador o endogamia', 'NO IDENTIFICADO AQUÍ', '', 'Necesita evidencia entre familias e historia demográfica alternativa.'),
        ('K07', 'Aporte adicional de raras', 'NO ESTIMADO AQUÍ', '', 'K02/K07 predictivo es otro análisis; estas etiquetas no lo sustituyen.'),
    ]
    write_table(out / 'conclusiones_por_nivel.tsv', [dict(codigo=a, nivel=b, estado=c, figura=d, conclusion=e) for a,b,c,d,e in levels])
    fig, ax = plt.subplots(figsize=(15, 5.6))
    ax.axis('off')
    table = ax.table(cellText=[[a+' · '+b,c,textwrap.fill(e, 78)] for a,b,c,d,e in levels],
                     colLabels=['Interpretación', 'Resultado de esta comparación', 'Alcance'],
                     colWidths=[.29,.26,.45], cellLoc='left', bbox=[0,.07,1,.80])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (r,c), cell in table.get_celld().items():
        cell.set_edgecolor('white')
        cell.set_facecolor('#E8F0F2' if r == 0 else '#EDF7F1' if r <= 2 else '#F2F2F2')
    ax.set_title('Qué podemos explicar y qué no está demostrado con estas comunidades', fontsize=16)
    fig.text(.04,.035,'«No identificado aquí» no significa ausencia de estructura: especifica lo que estos datos y esta comparación permiten concluir.', fontsize=11)
    savefig(fig, out, '03_alcance_por_nivel', '# Alcance por nivel\n\nLas categorías son interpretaciones diferentes, no una escala de validación. '
            'La tabla distingue evidencias calculadas en esta entrega de cuestiones que necesitan otros datos o diseños. '
            'No se selecciona una figura ganadora para un nivel que este análisis no identifica.')

    report(out, summary, representatives, metrics, communities, len(nondominated))
    result = dict(schema=SCHEMA, status=STATUS, n_cohort=summary['n_cohort'], n_partitions=summary['n_partitions'],
                  evidence_manifest_sha256=base.sha256(evidence/'manifest.json'), source_manifest_path=str(evidence/'manifest.json'),
                  source_sha256={Path(__file__).name: base.sha256(__file__), 'r02_community_levels.py': base.sha256(base.__file__)},
                  no_reclustering=True, no_population_validation=True, public_distribution_allowed=False,
                  contains_individual_identifiers=False, contains_private_small_cells=True,
                  retrospective_illustrative_selection=True, selections=selection,
                  outputs_sha256={p.name: base.sha256(p) for p in sorted(out.iterdir()) if p.is_file()})
    base.save_json(out/'manifest.json', result)
    return result

def report(out, summary, representatives, metrics, communities, frontier_size):
    lines = ['# M16.5: interpretación por niveles, Ronda 2', '',
        f"Se compararon **{summary['n_partitions']} particiones**, de {summary['n_graphs']} grafos, en **{summary['n_cohort']} personas**. "
        'No se repitieron M14 ni M16.5. Se calcularon medidas nuevas sobre sus asignaciones guardadas.', '',
        '## Resultado que sí puede presentarse', '',
        'Las comunidades describen diferencias de composición ancestral y algunas concentran parentesco. '
        'Ambas explicaciones pueden coexistir. No se concluye que cada comunidad sea una población, '
        'ni que las raras aporten estructura fina adicional frente a las comunes.', '',
        '| Figura | Pregunta que responde |', '|---|---|',
        '| 01_ancestria_amplia | ¿Qué composición ancestral tienen las comunidades de raras, comunes y combinación? |',
        '| 02_parentesco_comunidades_raras | ¿Cuánto concentra cada comunidad de raras a personas conectadas por parentesco? |',
        '| 00_comparacion_57 | ¿Cómo cambian asociación, cobertura y estabilidad al cambiar configuración? |',
        '| 03_alcance_por_nivel | ¿Qué interpretaciones quedan respaldadas descriptivamente y cuáles no se han identificado? |', '',
        '## Por qué se muestran estas particiones', '',
        'Dentro de cada grafo principal R, C y R+C se retuvieron candidatos con todas las personas asignadas y más de una comunidad. '
        'La figura presenta el de mayor asociación ancestral ajustada por número de comunidades; empates favorecen menos comunidades y menor resolución. '
        'Esta regla retrospectiva sirve para ilustrar K01: **no identifica parámetros biológicos óptimos ni generalización**. '
        'No se fuerza una comparación por γ idéntico: γ controla la resolución del algoritmo y no representa una escala biológica común entre grafos.', '',
        '| Representante | Comunidades | Asignados | η² | η² ajustada | η² equilibrando componentes φ=0,0221 / 0,0442 |', '|---|---:|---:|---:|---:|---:|']
    for r in representatives:
        balanced = [next(x for x in metrics if x['partition_id'] == r['partition_id'] and x['support'] == 'assigned_complete'
                         and x['weighting'] == 'component_balanced' and float(x['kinship_threshold']) == phi) for phi in base.PHIS]
        lines.append(f"| {label(r)} | {r['n_communities']} | {r['n_assigned']}/{r['n_cohort']} | {float(r['eta2']):.4f} | "
                     f"{float(r['eta2_adjusted']):.4f} | {float(balanced[0]['eta2']):.4f} / {float(balanced[1]['eta2']):.4f} |")
    rare = representatives[0]
    lines += ['', '### Las comunidades de raras: lectura conjunta', '',
              '| Comunidad local | Personas | AFR media | EUR media | NAM media | Mayor componente φ=0,0221 |', '|---|---:|---:|---:|---:|---:|']
    data = sorted([x for x in communities if x['partition_id'] == rare['partition_id'] and x['scope'] == 'community'
                   and float(x['kinship_threshold']) == .0221], key=lambda r:int(r['community_local']))
    for r in data:
        q = [100*float(r[f+'_mean']) for f in base.ANCESTRY[:3]]
        lines.append(f"| C{r['community_local']} | {r['n_samples']} | {q[0]:.1f}% | {q[1]:.1f}% | {q[2]:.1f}% | "
                     f"{r['maximum_component_n']}/{r['n_samples']} |")
    rare_sensitivity = [r for r in metrics if r['partition_id'] == rare['partition_id'] and r['support'] == 'assigned_complete']
    nam_person = num(next(r for r in rare_sensitivity if r['weighting'] == 'person')['eta2_NAM'])
    nam_balanced = [num(next(r for r in rare_sensitivity if r['weighting'] == 'component_balanced'
                              and float(r['kinship_threshold']) == phi)['eta2_NAM']) for phi in base.PHIS]
    fmt = lambda value: 'NA (no evaluable)' if value is None else f'{value:.4f}'
    lines += ['', f"**Sensibilidad NAM:** la fracción de variación NAM descrita pasa de {fmt(nam_person)} con igual peso por persona "
              f"a {fmt(nam_balanced[0])} / {fmt(nam_balanced[1])} equilibrando componentes a φ=0,0221 / 0,0442. "
              'La asociación NAM de esta partición depende mucho de la representación de componentes de parentesco; '
              'la medida conjunta de cuatro ancestrías oculta parte de esa sensibilidad. '
              'No se afirma que ponderar elimine todo parentesco ni que NAM carezca de estructura.']
    lines += ['', 'Una comunidad pequeña concentrada en un componente no debe presentarse como una población independiente. '
              'Una diferencia de medias ancestrales tampoco convierte a sus integrantes en personas de una ancestría pura.', '',
              '## Matemática y denominadores', '',
              'qᵢ es el vector de cuatro proporciones ancestrales de la persona i. g(i) es su comunidad. '
              'μ es la media global y μg la media de la comunidad, sobre las mismas personas y con los mismos pesos.', '',
              '`η² = 1 − Σᵢ wᵢ ||qᵢ − μg(i)||² / Σᵢ wᵢ ||qᵢ − μ||²`.', '',
              'El numerador mide diferencias dentro de comunidades; el denominador mide todas las diferencias observadas. '
              'La norma suma las diferencias cuadradas de las cuatro proporciones: es geometría euclídea declarada, '
              'no cuatro contrastes independientes ni una distancia genealógica. η²=0 significa que esta partición no describe '
              'variación entre medias; η² alto significa asociación descriptiva, no descubrimiento causal. Si el denominador es cero, se informa NA.', '',
              '`η² ajustada = 1 − (1 − η²)(n − 1)/(n − K)` para pesos individuales iguales y n>K. '
              'n es el número de personas usadas; K el número de comunidades representadas. Es un diagnóstico de complejidad, '
              'no una corrección completa por parentesco ni por seleccionar entre 57 alternativas. No se truncan valores negativos.', '',
              'En la sensibilidad, `wᵢ=1/nc`, donde nc cuenta miembros del componente de parentesco dentro del soporte analizado. '
              'Todos los componentes pesan igual en conjunto. Esto cambia el estimando: describe un componente típico, '
              'no demuestra que todos los integrantes sean independientes. Las comunidades NO se vuelven a aprender al ponderar.', '',
              'La diversidad efectiva de componentes de una comunidad es `1/Σc pc²`, con pc la fracción de personas de ese componente '
              'entre quienes tienen componente conocido. Es una medida de concentración, no tamaño de muestra independiente.', '',
              f"La comparación común usa {summary['n_common_assigned_complete']}/{summary['n_cohort']} personas asignadas en las 57 particiones. "
              'También se publica cada partición con su propio soporte; no se confunden esos denominadores. '
              f"La frontera descriptiva conserva {frontier_size} compromisos no dominados entre asociación en soporte común, cobertura, "
              'menos comunidades y estabilidad. No es una clasificación biológica.', '',
              '## Qué falta para otras conclusiones', '',
              'K02/K07: evaluación genética retenida del aporte raro frente a comunes adecuados, cantidad de raras y medición. '
              'K03: comparación discreta frente a continua con agrupamiento aprendido sólo en ajuste y asignación de personas nuevas. '
              'K04: correspondencia de alelos y copias con ancestría local, fase y territorio evaluable. '
              'K06: evidencia demográfica entre familias que distinga efecto fundador, endogamia, mezcla y parentesco reciente. '
              'Los datos de esta entrega no sustituyen esas evaluaciones. No identificado no significa que el fenómeno no exista.', '',
              '## Límites de esta entrega', '',
              'Las proporciones ancestrales son estimaciones existentes sin incertidumbre individual propagada. '
              'No hay ajuste nuevo por lote de secuenciación, profundidad, calidad de genotipo o genotipabilidad. '
              'Colección y región pueden confundirse; no son sustitutos de lote. La asociación de raras con ancestría no demuestra '
              'ventaja incremental, especialmente porque las comunes también la representan. No se crean p-valores ni intervalos '
              'con personas o parejas falsamente independientes. Las etiquetas se aprendieron en estas mismas 2.619 personas.', '',
              '## Fuentes', '',
              'Resultados primarios: r02-autosomes-20261001b, r02-study-20261002a y r02-community-evidence-20261003a; '
              'los manifiestos conservan rutas, SHA256, cohortes y parámetros. Se han autenticado de nuevo las entradas.', '',
              '[Lawson, van Dorp y Falush, 2018](https://doi.org/10.1038/s41467-018-05257-7) distingue agrupamiento e interpretación histórica. '
              '[Bradburd, Coop y Ralph, 2018](https://doi.org/10.1534/genetics.118.301333) aborda modelos continuos y discretos y su evaluación. '
              'No prescriben nuestros representantes ni demuestran que una comunidad DNABR sea una población.', '',
              'La revisión metodológica es descriptiva; no constituye aval confirmatorio.', '',
              'Entrega privada. Contiene celdas pequeñas: ausencia de nombres no implica autorización de publicación externa.']
    with (out/'INFORME.md').open('x', encoding='utf-8') as handle:
        handle.write('\n'.join(lines)+'\n')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    result = render(args.evidence, args.output_dir.absolute())
    print(json.dumps({k:result[k] for k in ('status','n_cohort','n_partitions')}))

if __name__ == '__main__':
    main()
