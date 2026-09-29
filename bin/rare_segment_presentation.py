#!/usr/bin/env python3
"""Present one saved M14 configuration and its sweep, without detecting segments.

The candidate is selected from the saved pre-filter chains. Counts and total bp
must reproduce the independent configuration summary before anything is drawn.
All pair intervals are preserved in the overview and pseudonymous ledger. The
detail view is an explicitly deterministic subset, not a biological ranking.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection

BLUE = '#286D93'
ORANGE = '#BA5D18'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def geometry_minimum(length, gap):
    if length < 1 or gap < 1:
        raise ValueError('Length and gap must be positive')
    return math.ceil((length - 1) / gap) + 1


def configuration(table, length, gap, count):
    effective = max(count, geometry_minimum(length, gap))
    rows = table[(table.min_length_bp == length) & (table.max_gap_bp == gap)
                 & (table.min_shared_effective == effective)]
    if len(rows) != 1:
        raise ValueError(f'Expected one measured configuration: {length}/{gap}/{effective}')
    return rows.iloc[0]


def select_chains(path, target, chrom, sample_ids):
    """Stream the saved chains; never count rows from another gap twice."""
    selected = []
    samples = set(sample_ids)
    with gzip.open(path, 'rt') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            if int(row['max_gap_bp']) != int(target.max_gap_bp):
                continue
            if int(row['length_bp']) < int(target.min_length_bp) or int(row['n_shared_variants']) < int(target.min_shared_effective):
                continue
            if row['chrom'].removeprefix('chr') != str(chrom):
                raise ValueError('Mixed chromosomes in candidate chains')
            a, b = sorted((row['sample_a'], row['sample_b']))
            if a == b or not {a, b} <= samples:
                raise ValueError('Self-pair or sample outside the declared cohort')
            values = {k: int(row[k]) for k in ('start_pos', 'end_pos', 'length_bp', 'n_shared_variants')}
            if values['start_pos'] < 1 or values['length_bp'] != values['end_pos'] - values['start_pos'] + 1:
                raise ValueError('Invalid inclusive coordinates')
            selected.append(dict(sample_a=a, sample_b=b, **values))
    selected.sort(key=lambda r: (r['sample_a'], r['sample_b'], r['start_pos'], r['end_pos']))
    keys = {(r['sample_a'], r['sample_b'], r['start_pos'], r['end_pos']) for r in selected}
    pairs = sorted({(r['sample_a'], r['sample_b']) for r in selected})
    observed = dict(n_segments=len(selected), n_pairs=len(pairs),
                    total_shared_bp=sum(r['length_bp'] for r in selected),
                    n_shared_variants_total=sum(r['n_shared_variants'] for r in selected))
    if len(keys) != len(selected) or any(observed[k] != int(target[k]) for k in observed):
        raise ValueError('Extracted intervals do not reproduce the saved configuration')
    labels = {sample: f'S{i+1:04d}' for i, sample in enumerate(sorted(sample_ids))}
    rank = {pair: i + 1 for i, pair in enumerate(pairs)}
    for row in selected:
        row['pair_rank'] = rank[(row['sample_a'], row['sample_b'])]
        row['sample_a'], row['sample_b'] = labels[row['sample_a']], labels[row['sample_b']]
    return selected, observed


def detail_ranks(n_pairs, count):
    if count < 1 or n_pairs < 1:
        raise ValueError('Detail requires at least one pair and one requested row')
    return np.unique(np.linspace(1, n_pairs, min(count, n_pairs), dtype=int))


def canvas(size):
    fig = Figure(figsize=size, facecolor='white')
    FigureCanvasAgg(fig)
    return fig


def intervals_figure(rows, chrom, target, detail_count=None):
    n_pairs = max(row['pair_rank'] for row in rows)
    chosen = detail_ranks(n_pairs, detail_count) if detail_count else np.arange(1, n_pairs + 1)
    ranks = {int(rank): i + 1 for i, rank in enumerate(chosen)}
    bars = [r for r in rows if r['pair_rank'] in ranks]
    fig = canvas((14, 10 if detail_count is None else max(8, .19 * len(chosen) + 2.3)))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=.15 if detail_count else .1, right=.97, bottom=.13, top=.86)
    segments = [[((r['start_pos'] - .5) / 1e6, ranks[r['pair_rank']]),
                 ((r['end_pos'] + .5) / 1e6, ranks[r['pair_rank']])] for r in bars]
    ax.add_collection(LineCollection(segments, colors=BLUE, linewidths=2 if detail_count else .15))
    ax.set_xlim(0, max(r['end_pos'] for r in rows) / 1e6)
    ax.set_ylim(len(chosen) + 1, 0)
    ax.set_xlabel(f'Posición en chr{chrom} (Mb; 1 Mb = 1.000.000 bases)', fontsize=11)
    if detail_count:
        labels = {r['pair_rank']: f"{r['sample_a']}–{r['sample_b']}" for r in bars}
        ax.set_yticks(range(1, len(chosen) + 1), [labels[int(r)] for r in chosen], fontsize=8)
        ax.set_ylabel('Pareja (identificadores sustituidos)', fontsize=10)
    else:
        ax.set_ylabel('Número de fila de la pareja; no indica parentesco ni grupo', fontsize=10)
    ax.grid(axis='x', alpha=.2)
    title = 'Detalle de parejas seleccionadas por posición en la lista' if detail_count else 'Todos los segmentos de la configuración candidata'
    fig.suptitle(f'M14 · chr{chrom} · {title}\n'
                 f"L ≥ {int(target.min_length_bp)/1000:g} kb | G ≤ {int(target.max_gap_bp)/1000:g} kb | N ≥ {int(target.min_shared_effective)}",
                 fontsize=15, y=.975)
    ax.set_title(f'{len(bars):,} intervalos dibujados · {len(chosen):,} de {n_pairs:,} parejas'.replace(',', '.'), fontsize=11, pad=12)
    note = ('Todas las filas están incluidas; la imagen comprimida no permite resolver cada pareja. Consulte el PDF y el TSV.\n'
            if not detail_count else 'Subconjunto determinista espaciado uniformemente en la lista; no representa un grupo biológico ni una muestra aleatoria.\n')
    fig.text(.1, .035, note + 'Azul = segmento compartido detectado por M14; no implica ancestría ni IBD confirmado.\n'
             'Orden lexicográfico de participantes, igual que 02A. Eje X termina en el último segmento, no en el extremo del cromosoma.', fontsize=9)
    return fig, bars


def sweep_figure(table, kinship, target, n_samples, threshold, nominal_count):
    kin = kinship[(kinship.kinship_threshold == threshold) & (kinship.kinship_group == 'kin_ge_threshold')].set_index('config_id')
    if not kin.index.is_unique or set(kin.index) != set(table.config_id):
        raise ValueError('Kinship and sweep configurations must agree one-to-one')
    for _, measured in table.iterrows():
        related = kin.loc[measured.config_id]
        for field in ('n_pairs', 'n_segments', 'total_shared_bp'):
            if int(measured[field]) != int(related[field + '_total']):
                raise ValueError('Kinship and sweep totals disagree')
    length = int(target.min_length_bp)
    gap = int(target.max_gap_bp)
    n = nominal_count
    lengths = sorted(table.min_length_bp.unique())
    sweep_l = [configuration(table, int(l), gap, n) for l in lengths]
    fig = canvas((15, 10))
    axes = fig.subplots(2, 2)
    fig.subplots_adjust(left=.075, right=.97, top=.86, bottom=.12, hspace=.47, wspace=.25)
    ax = axes[0, 0]
    x = np.arange(len(sweep_l))
    ax.plot(x, [r.n_pairs for r in sweep_l], 'o-', color=BLUE)
    ax.set_yscale('log')
    for i, r in enumerate(sweep_l):
        ax.annotate(f'{int(r.n_pairs):,}'.replace(',', '.'), (i, r.n_pairs), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=9)
    idx = lengths.index(length)
    ax.scatter([idx], [target.n_pairs], s=130, facecolors='none', edgecolors=ORANGE, linewidths=2, zorder=5)
    ax.set_xticks(x, [f'{l/1000:g}' for l in lengths])
    ax.set_xlabel(f'Longitud mínima L (kb); G={gap/1000:g} kb, N nominal={n}')
    ax.set_ylabel('Parejas con segmentos (escala logarítmica)')
    ax.set_title('A · Exigir más longitud conserva menos parejas', loc='left', fontsize=11)
    ax = axes[0, 1]
    for r in sweep_l:
        k = kin.loc[r.config_id]
        coverage = 100 * k.n_ids_total / n_samples
        fraction = 100 * k.pair_fraction
        ax.scatter(coverage, fraction, color=ORANGE if r.config_id == target.config_id else BLUE, s=55)
        ax.annotate(f'{int(r.min_length_bp)/1000:g} kb', (coverage, fraction), xytext=(4, 7), textcoords='offset points', fontsize=9)
    ax.set_xlim(-3, 104)
    ax.set_ylim(-3, 83)
    ax.set_xlabel(f'Personas con segmentos / {n_samples} personas (%)')
    ax.set_ylabel(f'Parejas con coeficiente PC-Relate ≥ {threshold} (%)')
    ax.set_title('B · Los tramos largos concentran parentesco elevado', loc='left', fontsize=11)
    ax = axes[1, 0]
    rows = table[(table.min_length_bp == length) & (table.max_gap_bp == gap)].sort_values('min_shared_effective')
    x = np.arange(len(rows))
    ax.bar(x, rows.n_pairs, color=[ORANGE if c == target.config_id else BLUE for c in rows.config_id])
    for i, value in enumerate(rows.n_pairs):
        ax.text(i, value + rows.n_pairs.max() * .02, f'{int(value):,}'.replace(',', '.'), ha='center', fontsize=9)
    ax.set_ylim(0, rows.n_pairs.max() * 1.25)
    ax.set_xticks(x, rows.min_shared_effective.astype(str))
    ax.set_xlabel(f'Mínimo efectivo de sitios compartidos N; L={length/1000:g} kb, G={gap/1000:g} kb')
    ax.set_ylabel('Parejas con segmentos')
    ax.set_title('C · Más sitios no equivalen automáticamente a más calidad', loc='left', fontsize=11)
    ax = axes[1, 1]
    gaps = sorted(table.max_gap_bp.unique())
    rows = [configuration(table, length, int(g), n) for g in gaps]
    ax.bar(range(len(rows)), [r.n_pairs for r in rows], color=[ORANGE if r.config_id == target.config_id else BLUE for r in rows])
    for i, r in enumerate(rows):
        k = kin.loc[r.config_id]
        ax.text(i, r.n_pairs + max(q.n_pairs for q in rows) * .025,
                f'{int(r.n_pairs):,} parejas\n{k.pair_fraction*100:.2f}% con parentesco ≥ {threshold}'.replace(',', '.'),
                ha='center', fontsize=9)
    ax.set_xticks(range(len(rows)), [f'{g/1000:g}' for g in gaps])
    ax.set_ylim(0, max(r.n_pairs for r in rows) * 1.35)
    ax.set_xlabel(f'Separación máxima G (kb); L={length/1000:g} kb, N nominal={n}')
    ax.set_ylabel('Parejas con segmentos')
    ax.set_title('D · Permitir huecos mayores aumenta conexiones', loc='left', fontsize=11)
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
    fig.suptitle('M14 · chr22 · Qué muestra el barrido y por qué conservar una candidata', fontsize=17, y=.97)
    fig.text(.075, .915, f'Naranja: {length/1000:g} kb / {gap/1000:g} kb / {n} sitios. Selección provisional de desarrollo, no óptimo biológico demostrado.', fontsize=11)
    fig.text(.075, .035, 'L = longitud del tramo; G = separación entre dos sitios compartidos consecutivos; N = número de sitios compartidos.\n'
             'Cada pareja se cuenta una vez por configuración. PC-Relate mide parentesco; estar por debajo del corte no certifica independencia.\n'
             'Los paneles son cortes del barrido completo (74 configuraciones); no validan estructura poblacional ni identidad por descendencia.', fontsize=9)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('chains', 'configuration-summary', 'kinship-summary', 'sample-ids', 'output-dir', 'prefix', 'chr'):
        parser.add_argument('--' + name, required=True)
    for name, default in [('length', 250000), ('gap', 50000), ('min-shared', 20), ('detail-pairs', 40), ('dpi', 300)]:
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--kinship-threshold', type=float, default=.0221)
    args = parser.parse_args(argv)
    if not args.prefix or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.prefix):
        parser.error('Prefix must be a filename stem')
    output = Path(args.output_dir)
    if output.exists():
        parser.error('Output directory already exists; choose a new destination')
    ids = Path(args.sample_ids).read_text().splitlines()
    if not ids or len(ids) != len(set(ids)) or any(not x for x in ids):
        parser.error('Sample universe must be nonempty and unique')
    table = pd.read_csv(args.configuration_summary, sep='\t')
    kinship = pd.read_csv(args.kinship_summary, sep='\t')
    target = configuration(table, args.length, args.gap, args.min_shared)
    rows, observed = select_chains(args.chains, target, args.chr, ids)
    if not rows:
        parser.error('Selected configuration has no segments')
    people = len({r[k] for r in rows for k in ('sample_a', 'sample_b')})
    k = kinship[(kinship.config_id == target.config_id) & (kinship.kinship_threshold == args.kinship_threshold) & (kinship.kinship_group == 'kin_ge_threshold')]
    if len(k) != 1 or int(k.iloc[0].n_ids_total) != people:
        parser.error('Active participants do not match kinship summary')
    output.mkdir(parents=True)
    figures = []
    ledger = output / f'{args.prefix}__06B_segmentos_candidata.tsv.gz'
    pd.DataFrame(rows).to_csv(ledger, sep='\t', index=False, compression={'method': 'gzip', 'mtime': 0})
    descriptors = {}
    for code, detail in [('06B_segmentos_candidata_completos', None), ('06C_segmentos_candidata_detalle', args.detail_pairs)]:
        fig, bars = intervals_figure(rows, args.chr, target, detail)
        stem = f'{args.prefix}__{code}'
        figures.append((stem, fig))
        descriptors[stem] = {'n_bars': len(bars), 'n_pairs': len({r['pair_rank'] for r in bars}), 'all_intervals': detail is None}
    figures.insert(0, (f'{args.prefix}__06A_resumen_decision_M14', sweep_figure(table, kinship, target, len(ids), args.kinship_threshold, args.min_shared)))
    for stem, fig in figures:
        for extension in ('png', 'pdf'):
            fig.savefig(output / f'{stem}.{extension}', dpi=args.dpi, metadata={'Creator': 'rare_segment_presentation', 'CreationDate': None, 'ModDate': None} if extension == 'pdf' else {'Software': 'rare_segment_presentation'})
        fig.clear()
    manifest = dict(config_id=target.config_id, n_people=people, cohort_size=len(ids), **observed,
                    inputs={name: {'filename': Path(getattr(args, name)).name, 'sha256': digest(getattr(args, name))} for name in ('chains', 'configuration_summary', 'kinship_summary', 'sample_ids')},
                    renderer_sha256=digest(__file__), parameters=vars(args), views=descriptors,
                    interpretation='Observed rare-sharing segments; not validated IBD or population groups',
                    figure_hashes={p.name: digest(p) for p in sorted(output.glob('*'))})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({**observed, 'n_people': people, 'n_figures': len(figures)}))


if __name__ == '__main__':
    main()
