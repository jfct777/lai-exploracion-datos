#!/usr/bin/env python3
"""Reproducible descriptive figures from saved M16.5 partitions; no clustering."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import re
import sys


GRID = ((250000, 250000), (250000, 500000), (250000, 1000000),
        (500000, 500000), (500000, 750000), (500000, 1000000))
GAMMAS = (0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0)
DEFAULT_NETWORKS = ("L250000_G50000_N20_T250000_U0", "L250000_G50000_N20_T500000_U0",
                    "L500000_G50000_N20_T500000_U0")
REQUIRED = ("graph_nodes.tsv", "graph_edges.tsv.gz", "graph_summary.json",
            "leiden_assignments.tsv", "resolution_summary.tsv",
            "leiden_ari_by_resolution.tsv", "leiden_modularity.tsv")
SCOPE = ("Resultado descriptivo de chr22; comunidades del grafo, no poblaciones "
         "validadas ni ancestrías. Las semillas miden variabilidad algorítmica.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def table(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def keyed_resolution(rows, name):
    result = {float(r["resolution"]): r for r in rows}
    require(len(result) == len(rows) == 7 and set(result) == set(GAMMAS),
            f"Missing or duplicate resolution in {name}")
    return result


def load_results(results_dir):
    """Authenticate files and reconcile graph, assignments and 42 plot rows.

    IDs are kept in memory only. Errors do not expose endpoint/sample values.
    No communities, distances, ARI or graph thresholds are recomputed.
    """
    root = Path(results_dir)
    graphs, plot_rows, hashes = [], [], {}
    canonical_samples, source_hashes, core_hash = None, None, None
    for length, threshold in GRID:
        config_id = f"L{length}_G50000_N20_T{threshold}_U0"
        directory = root / config_id
        require(directory.is_dir(), f"Missing graph directory: {config_id}")
        manifest = json.loads((directory / "manifest.json").read_text())
        hashes[f"{config_id}/manifest.json"] = sha256(directory / "manifest.json")
        require(manifest["status"] == "COMPLETE_DESCRIPTIVE" and manifest["chromosome"] == "22",
                "Input graph is not a completed descriptive chr22 result")
        plan = manifest["configuration"]
        require(plan == dict(length_bp=length, min_edge_bp=threshold, gap_bp=50000,
                            min_shared=20, min_shared_effective=20, min_max_segment_bp=0,
                            source_config_id=f"L{length}_G50000_N20", config_id=config_id),
                "Input graph configuration differs from the six-graph figure contract")
        settings = manifest["parameters"]
        require(tuple(settings["resolutions"]) == GAMMAS and settings["min_community_size"] == 3
                and settings["consensus_resolution"] == 1.0 and settings["n_seeds"] == 25,
                "Unexpected saved partition settings")
        require(manifest["no_nmf"] and manifest["no_founder_classification"],
                "Input scope differs from the descriptive sweep")
        if source_hashes is None:
            source_hashes, core_hash = manifest["source_sha256"], manifest["core_sha256"]
        require(manifest["source_sha256"] == source_hashes and manifest["core_sha256"] == core_hash,
                "Graphs do not share source data and core version")
        for name in REQUIRED:
            require(name in manifest["outputs_sha256"], f"Uncertified input artifact: {name}")
            actual = sha256(directory / name)
            require(actual == manifest["outputs_sha256"][name], f"Input hash mismatch: {name}")
            hashes[f"{config_id}/{name}"] = actual
        nodes = table(directory / "graph_nodes.tsv")
        samples = [r["sample_id"] for r in nodes]
        require(samples and len(samples) == len(set(samples)) and all(samples), "Invalid node universe")
        require([int(r["node_id"]) for r in nodes] == list(range(len(samples))), "Invalid node order")
        if canonical_samples is None:
            canonical_samples = samples
        require(samples == canonical_samples, "Graphs differ in sample order or cohort")
        n = len(samples)
        require(n == manifest["n_cohort"] == settings["expected_samples"], "Cohort denominator mismatch")
        index = {sample: i for i, sample in enumerate(samples)}
        edges, seen, degrees = [], set(), [0] * n
        for row in table(directory / "graph_edges.tsv.gz"):
            require(row["sample_a"] in index and row["sample_b"] in index, "Edge outside node universe")
            a, b = sorted((index[row["sample_a"]], index[row["sample_b"]]))
            weight = float(row["weight"])
            require(a != b and (a, b) not in seen and math.isfinite(weight) and weight > 0,
                    "Duplicate, self or invalid weighted edge")
            seen.add((a, b)); edges.append((a, b)); degrees[a] += 1; degrees[b] += 1
        require(degrees == [int(r["degree"]) for r in nodes], "Edge and node degrees differ")
        active = [degree > 0 for degree in degrees]
        graph_summary = json.loads((directory / "graph_summary.json").read_text())
        require(graph_summary["n_nodes"] == n and graph_summary["n_edges"] == len(edges)
                == manifest["n_edges"] and sum(active) == manifest["n_active"]
                and graph_summary["n_isolated"] == n - sum(active)
                and graph_summary["weight_transform"] == "log1p"
                and graph_summary["min_edge_bp"] == threshold
                and graph_summary["min_max_segment_bp"] == 0, "Graph metadata parity failed")
        assignments = table(directory / "leiden_assignments.tsv")
        require([r["sample_id"] for r in assignments] == samples, "Assignment order or cohort mismatch")
        summaries = keyed_resolution(table(directory / "resolution_summary.tsv"), "summary")
        aris = keyed_resolution(table(directory / "leiden_ari_by_resolution.tsv"), "ARI")
        scores = table(directory / "leiden_modularity.tsv")
        labels_by_gamma = {}
        for gamma in GAMMAS:
            labels = [int(r[f"community_res_{gamma:g}"]) for r in assignments]
            require(all(label >= -1 for label in labels), "Invalid community label")
            require(all(label == -1 for label, supported in zip(labels, active) if not supported),
                    "An isolated node has an assignment")
            sizes = Counter(label for label in labels if label >= 0)
            require(all(size >= 3 for size in sizes.values()), "Reported community below minimum size")
            assigned = sum(sizes.values())
            expected = dict(n_cohort=n, n_active=sum(active), n_isolated=n-sum(active),
                            n_assigned=assigned, n_active_unassigned=sum(active)-assigned,
                            n_communities=len(sizes), largest_community=max(sizes.values(), default=0))
            row, ari = summaries[gamma], aris[gamma]
            require(row["config_id"] == config_id and all(int(row[k]) == v for k, v in expected.items()),
                    "Saved assignments do not reproduce resolution counts")
            require(int(ari["n_nodes"]) == sum(active) and int(ari["n_pairs"]) == 300,
                    "ARI denominator differs from 25 seeds on supported nodes")
            quantiles = [float(ari[k]) for k in ("min_ari", "q25_ari", "median_ari", "q75_ari", "max_ari")]
            require(all(math.isfinite(x) and -1 <= x <= 1 for x in quantiles)
                    and quantiles == sorted(quantiles), "Invalid ARI quantiles")
            subset = [r for r in scores if float(r["resolution"]) == gamma]
            representatives = [r for r in subset if r["is_representative"] == "True"]
            require(len(subset) == 25 and len({r["seed"] for r in subset}) == 25
                    and len(representatives) == 1, "Missing or duplicate representative/seed")
            require(float(representatives[0]["rb_quality"]) == max(float(r["rb_quality"]) for r in subset),
                    "Representative is not the RB-quality maximum")
            plot_rows.append(dict(config_id=config_id, length_bp=length, min_edge_bp=threshold,
                                  gap_bp=50000, min_shared=20, resolution=gamma, **expected,
                                  percent_assigned=100*assigned/n,
                                  **{k: float(ari[k]) for k in ("median_ari", "q25_ari", "q75_ari", "min_ari", "max_ari")}))
            labels_by_gamma[gamma] = labels
        graphs.append(dict(config_id=config_id, length_bp=length, threshold_bp=threshold,
                           n_cohort=n, n_active=sum(active), edges=edges, active=active,
                           labels=labels_by_gamma))
    return dict(graphs=graphs, rows=plot_rows, n_cohort=len(canonical_samples),
                input_sha256=hashes, source_sha256=source_hashes, core_sha256=core_hash)


def graph_label(graph):
    return f"L={graph['length_bp']/1000:g} kb · T={graph['threshold_bp']/1e6:g} Mb"


def shared_layout(data, seed, iterations):
    """One unweighted union-graph layout; no clustering or genetic distances."""
    import igraph as ig
    import numpy as np
    union = sorted({edge for graph in data["graphs"] for edge in graph["edges"]})
    shown = sorted({node for edge in union for node in edge})
    require(shown, "Cannot lay out a union graph without edges")
    index = {node: i for i, node in enumerate(shown)}
    graph = ig.Graph(n=len(shown), edges=[(index[a], index[b]) for a, b in union], directed=False)
    initial = np.random.default_rng(seed).uniform(-1, 1, (len(shown), 2))
    ig.set_random_number_generator(random.Random(seed))
    try:
        points = np.asarray(graph.layout_fruchterman_reingold(
            seed=initial.tolist(), niter=iterations, grid="grid").coords, dtype=float)
    finally:
        ig.set_random_number_generator(None)
    require(np.isfinite(points).all(), "Layout coordinates are not finite")
    positions = np.full((data["n_cohort"], 2), np.nan)
    positions[shown] = points
    span = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 1.)
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    limits = (center[0]-.55*span, center[0]+.55*span, center[1]-.55*span, center[1]+.55*span)
    return positions, shown, limits


def describe(output, stem, text):
    (output / f"{stem}.descripcion.md").write_text(
        f"# {stem}\n\n{text}\n\n{SCOPE}\n\n"
        "Fuente: etapa 08_M16_5_barrido_de_03, resultados guardados y hashes en manifest.json. "
        "Sin genotipos nuevos, reclustering, inferencia de ancestría ni selección de óptimo.\n", encoding="utf-8")


def save_figure(fig, output, stem, dpi, description, book=None):
    fig.savefig(output / f"{stem}.png", dpi=dpi, facecolor="white")
    fig.savefig(output / f"{stem}.pdf", dpi=dpi, facecolor="white")
    if book is not None:
        book.savefig(fig, dpi=dpi, facecolor="white")
    describe(output, stem, description)
    import matplotlib.pyplot as plt
    plt.close(fig)


def summary_figures(data, output, prefix, dpi):
    import numpy as np
    import matplotlib.pyplot as plt
    labels = [graph_label(graph) for graph in data["graphs"]]
    rows = data["rows"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), constrained_layout=True)
    for ax, metric, title, limit, formatting in zip(axes,
            ("n_communities", "percent_assigned", "median_ari"),
            ("Número de comunidades (tamaño ≥3)", "% de la cohorte asignado", "ARI mediano entre semillas (nodos activos)"),
            (max(r["n_communities"] for r in rows), 100, 1), ("d", ".1f", ".3f")):
        values = np.array([r[metric] for r in rows]).reshape(6, 7)
        lower = min(0., float(values.min())) if metric == "median_ari" else 0
        im = ax.imshow(values, aspect="auto", cmap="cividis", vmin=lower, vmax=limit)
        ax.set_xticks(range(7), [f"{g:g}" for g in GAMMAS]); ax.set_xlabel("Resolución gamma")
        ax.set_yticks(range(6), labels); ax.set_title(title, loc="left", fontweight="bold")
        for i in range(6):
            for j in range(7):
                value = int(values[i,j]) if formatting == "d" else values[i,j]
                ax.text(j, i, format(value, formatting), ha="center", va="center", fontsize=9,
                        color="white" if values[i,j] < (lower+limit)/2 else "#202020")
        fig.colorbar(im, ax=ax, fraction=.022, pad=.015)
    fig.suptitle(f"M16.5 · resultado descriptivo de chr22 · N={data['n_cohort']}\n"
                 "G=50 kb · N mínimo=20 · U=0 · no selección de óptimo", fontsize=15)
    save_figure(fig, output, prefix+"A_resumen_42", dpi,
                "Cada celda es un grafo/resolución observado. Filas: longitud mínima M14 L y suma mínima por pareja T. "
                "Columnas: gamma. Color secuencial y cifras muestran grupos≥3, porcentaje asignado/cohorte y ARI. "
                "ARI alto puede reflejar particiones gruesas o grafos fragmentados; no valida poblaciones.")
    print("Figura A completada", file=sys.stderr, flush=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True, constrained_layout=True)
    ari_lower = min(0., min(r["min_ari"] for r in rows))
    for i, (ax, graph) in enumerate(zip(axes.flat, data["graphs"])):
        selected = rows[i*7:(i+1)*7]
        ax.fill_between(GAMMAS, [r["q25_ari"] for r in selected], [r["q75_ari"] for r in selected],
                        color="#5B8DB8", alpha=.27, label="IQR (25–75 %)")
        ax.plot(GAMMAS, [r["median_ari"] for r in selected], "o-", color="#24577A", label="Mediana")
        ax.set_title(graph_label(graph), fontsize=11)
        ax.set_ylim(ari_lower,1.04); ax.set_xticks(GAMMAS); ax.grid(axis="y", color="#dddddd", linewidth=.5)
        ax.set_xlabel("Gamma"); ax.set_ylabel("ARI en nodos activos")
        ax.text(.02,.03,f"Activos: {graph['n_active']}/{graph['n_cohort']}\n25 semillas; 300 comparaciones dependientes",
                transform=ax.transAxes, fontsize=8)
    axes.flat[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Estabilidad algorítmica · chr22\nLa banda es IQR entre comparaciones, no intervalo de confianza biológico", fontsize=14)
    save_figure(fig, output, prefix+"B_estabilidad", dpi,
                "Seis paneles con escalas iguales. Punto/línea: ARI mediano; banda: cuartiles25–75 de las "
                "300 comparaciones entre25 semillas sobre nodos activos, antes de filtrar grupos<3. "
                "Comparaciones dependientes; banda NO es intervalo de confianza ni replicación biológica.")
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    selected = [r for r in rows if r["resolution"] == 1.]
    left = np.zeros(6)
    for key, title, color, hatch in (("n_assigned", "Asignados (grupos ≥3)", "#24577A", None),
            ("n_active_unassigned", "Activos en grupos <3", "#DDAA55", "///"),
            ("n_isolated", "Aislados (grado 0)", "#D4D4D4", "..")):
        values = np.array([r[key] for r in selected])
        ax.barh(range(6), values, left=left, color=color, hatch=hatch, edgecolor="white", label=title)
        for i,v in enumerate(values):
            if v: ax.text(left[i]+v/2, i, str(v), ha="center", va="center", fontsize=9,
                          color="white" if key=="n_assigned" else "#222222")
        left += values
    ax.set_yticks(range(6), labels); ax.invert_yaxis(); ax.set_xlim(0,data["n_cohort"])
    ax.set_xlabel(f"Personas; denominador común = {data['n_cohort']}")
    ax.set_title("Cobertura de la cohorte · gamma=1 · resultado descriptivo de chr22", loc="left", fontsize=14)
    ax.legend(loc="upper center", bbox_to_anchor=(.5,-.13), ncol=3, fontsize=9)
    save_figure(fig, output, prefix+"C_cobertura_gamma1", dpi,
                "Barras apiladas de conteos, todas con el mismo denominador. Azul: asignados a grupos≥3; "
                "ocre rayado: personas activas excluidas por tamaño<3; gris punteado: grado0. "
                "Las tres clases son disjuntas y suman toda la cohorte. Gamma1 es referencia, no ganador.")


def select_networks(config_ids=None, resolution=1.0, all_networks=False):
    known = [f"L{length}_G50000_N20_T{threshold}_U0" for length, threshold in GRID]
    require(resolution in GAMMAS, "Network resolution was not evaluated")
    if all_networks:
        require(config_ids is None, "Do not combine explicit network IDs and --all-networks")
        return [(config, gamma) for config in known for gamma in GAMMAS]
    selected = list(DEFAULT_NETWORKS) if config_ids is None else (
        config_ids.split(",") if isinstance(config_ids,str) else list(config_ids))
    require(selected and len(set(selected))==len(selected) and all(c in known for c in selected),
            "Network IDs must be unique, explicit evaluated configurations")
    return [(config,resolution) for config in selected]


def network_figures(data, output, prefix, dpi, positions, shown, limits, selection):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.collections import LineCollection
    from matplotlib.colors import hsv_to_rgb
    from matplotlib.lines import Line2D
    shown = np.array(shown, dtype=int)
    sizes_rows = []
    # A common count axis permits direct comparison without changing the layout.
    max_group_size = max((count for graph in data["graphs"] for gamma in GAMMAS
                         if (graph["config_id"], gamma) in selection
                         for label, count in Counter(graph["labels"][gamma]).items()
                         if label >= 0), default=1)
    with PdfPages(output / f"{prefix}D_comparacion_redes.pdf") as book:
        for graph in data["graphs"]:
            short = f"L{graph['length_bp']//1000}_T{graph['threshold_bp']//1000}"
            for gamma in GAMMAS:
                if (graph["config_id"],gamma) not in selection:
                    continue
                labels = np.asarray(graph["labels"][gamma]); active = np.asarray(graph["active"])
                sizes = Counter(labels[labels>=0].tolist())
                ranked = sorted(sizes, key=lambda label:(-sizes[label],label))
                palette = {label:hsv_to_rgb(((i*.61803398875)%1, .58, .77)) for i,label in enumerate(ranked)}
                fig, (ax, ax_sizes) = plt.subplots(1, 2, figsize=(14,9),
                                                  gridspec_kw={"width_ratios":[7,3]})
                fig.subplots_adjust(left=.025,right=.96,top=.82,bottom=.19,wspace=.12)
                segments = positions[np.asarray(graph["edges"],dtype=int)]
                ax.add_collection(LineCollection(segments, colors="#333333", linewidths=.20,
                                                 alpha=.10, rasterized=True, zorder=1))
                inactive = shown[~active[shown]]
                small = shown[active[shown] & (labels[shown]<0)]
                assigned = shown[labels[shown]>=0]
                if inactive.size: ax.scatter(*positions[inactive].T,s=7,c="#BBBBBB",marker="x",linewidths=.45,zorder=2,rasterized=True)
                if small.size: ax.scatter(*positions[small].T,s=10,c="#444444",marker="x",linewidths=.65,zorder=3,rasterized=True)
                if assigned.size: ax.scatter(*positions[assigned].T,s=8,c=[palette[x] for x in labels[assigned]],
                                             linewidths=0,alpha=.87,zorder=4,rasterized=True)
                for rank,label in enumerate(ranked):
                    sizes_rows.append(dict(config_id=graph["config_id"],resolution=gamma,
                                           community_local=int(label),n_samples=sizes[label],
                                           annotated=False,shown_in_size_panel=True))
                # All communities remain visible; no labels obscure network nodes.
                ax_sizes.barh(range(len(ranked)), [sizes[label] for label in ranked],
                              color=[palette[label] for label in ranked], height=.72)
                ax_sizes.set_yticks(range(len(ranked)), [f"C{label}" for label in ranked], fontsize=7)
                ax_sizes.invert_yaxis()
                for rank,label in enumerate(ranked):
                    ax_sizes.text(sizes[label]+max_group_size*.015,rank,str(sizes[label]),
                                  va="center",ha="left",fontsize=7)
                ax_sizes.set_xlim(0,max_group_size*1.17)
                ax_sizes.set_xlabel("Personas",fontsize=10)
                ax_sizes.set_title("Todos los grupos asignados\nOrdenados por tamaño",fontsize=10,pad=10)
                ax_sizes.tick_params(axis="x",labelsize=8)
                ax_sizes.tick_params(axis="y",length=0)
                ax_sizes.spines["left"].set_visible(False)
                ax.set_xlim(limits[:2]); ax.set_ylim(limits[2:]); ax.set_aspect("equal"); ax.axis("off")
                omitted = data["n_cohort"]-len(shown)
                fig.suptitle(f"Resultado descriptivo de chr22 · {graph_label(graph)} · gamma={gamma:g}\n"
                             f"G=50 kb · N mínimo=20 · U=0 · cohorte={graph['n_cohort']} · activos={graph['n_active']} · asignados={assigned.size}\n"
                             f"{len(sizes)} comunidades ≥3 · {len(graph['edges'])} aristas dibujadas completas",
                             fontsize=12,y=.975)
                fig.legend(handles=[Line2D([],[],marker="o",ls="",color="#557A91",label="Círculo: asignado; color local"),
                    Line2D([],[],marker="x",ls="",color="#444444",label="Cruz oscura: grupo <3"),
                    Line2D([],[],marker="x",ls="",color="#BBBBBB",label="Cruz gris: aislado en este grafo")],
                    loc="lower center",bbox_to_anchor=(.5,.115),ncol=3,fontsize=8,frameon=False)
                fig.text(.5,.035,f"Layout común fijo de la unión; distancias sin significado genómico/geográfico. Aislados en toda la unión, sólo contados: {omitted}.\n"
                         "Barras: personas en cada grupo, todos incluidos. Colores y C# locales: no son ancestrías ni equivalencias entre paneles.\n"
                         "Representante con máxima calidad RB entre 25 semillas de este grafo y gamma; no es consenso ni validación biológica.",
                         ha="center",fontsize=8,linespacing=1.5)
                stem=f"{prefix}D_{short}_gamma{gamma:g}"
                save_figure(fig,output,stem,dpi,
                    f"{graph_label(graph)}; G = 50 kb; N mínimo = 20; U = 0; gamma = {gamma:g}. Cohorte: {graph['n_cohort']} personas; "
                    f"activos: {graph['n_active']}; asignados: {assigned.size}; comunidades de tamaño ≥ 3: {len(sizes)}. "
                    f"Se dibujan TODAS las {len(graph['edges'])} aristas guardadas, sin submuestreo, con transparencia 0,10 y ancho uniforme; "
                    "su peso científico sigue siendo log1p(bp), pero grosor/longitud gráfica no lo codifican. "
                    "Círculos: asignados; cruces oscuras: activos no asignados; cruces grises: aislados en este grafo. "
                    f"Los {omitted} aislados de toda la unión se incluyen en el denominador, no se colocan artificialmente. "
                    "Layout Fruchterman–Reingold NO ponderado, calculado una vez sobre unión de aristas; posiciones/escala/orden fijos. "
                    "No es reclustering, mapa geográfico, cromosómico, ni medida de distancia genética. Paleta local determinista; "
                    "el color/C# no alinea comunidades entre paneles. No hay etiquetas sobre los nodos. "
                    "El panel derecho muestra TODOS los grupos con los mismos colores que sus nodos: barras horizontales de personas, "
                    "orden descendente por tamaño y desempate por C#, conteo al extremo y escala común entre las redes seleccionadas. "
                    "Sus tamaños completos también aparecen en tamaños_comunidades.tsv. "
                    "Partición seleccionada por máxima calidad RB entre 25 semillas dentro de este mismo grafo y gamma; "
                    "no es una partición consensuada. La calidad RB es la función que optimiza Leiden, no una medida "
                    "de validez biológica. "
                    "Proximidad o solapamiento de puntos no constituye evidencia biológica.",book=book)
                print(f"Red completada: {short}, gamma={gamma:g}",file=sys.stderr,flush=True)
    return sizes_rows


def write_table(path, rows):
    require(rows, "Refusing to write an empty figure table")
    with path.open("x",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),delimiter="\t",lineterminator="\n")
        writer.writeheader();writer.writerows(rows)


def run(results_dir, output_dir, prefix="M16_5_chr22__A01__09", dpi=300,
        layout_seed=42, layout_iterations=200, network_config_ids=None,
        network_resolution=1.0, all_networks=False):
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}",prefix), "Unsafe output prefix")
    require(72<=dpi<=600 and layout_seed>=0 and 1<=layout_iterations<=2000, "Invalid rendering settings")
    output=Path(output_dir)
    require(not output.exists(), "Output directory exists; no overwrite")
    selection=select_networks(network_config_ids,network_resolution,all_networks)
    data=load_results(results_dir)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                         "axes.spines.right":False,"pdf.fonttype":42})
    output.mkdir(parents=True,exist_ok=False)
    write_table(output/f"{prefix}_tabla_42.tsv",data["rows"])
    summary_figures(data,output,prefix,dpi)
    positions,shown,limits=shared_layout(data,layout_seed,layout_iterations)
    sizes=network_figures(data,output,prefix,dpi,positions,shown,limits,selection)
    write_table(output/f"{prefix}_tamaños_comunidades.tsv",sizes)
    # Detect source mutation during rendering before issuing a success receipt.
    for name,expected in data["input_sha256"].items():
        require(sha256(Path(results_dir)/name)==expected,"Source changed during rendering")
    versions={name:importlib.metadata.version(name) for name in ("numpy","matplotlib","igraph")}
    result=dict(status="COMPLETE_FIGURES_FROM_SAVED_PARTITIONS",source_stage="08_M16_5_barrido_de_03",
        n_graphs=6,n_resolutions=7,n_cells=42,n_network_figures=len(selection),
        n_png=3+len(selection),n_pdf=4+len(selection),n_descriptions=3+len(selection),
        n_cohort=data["n_cohort"],n_union_active=len(shown),n_union_isolated=data["n_cohort"]-len(shown),
        settings=dict(prefix=prefix,dpi=dpi,layout_seed=layout_seed,layout_iterations=layout_iterations,
                      network_selection=[dict(config_id=c,resolution=g) for c,g in selection],
                      all_networks=all_networks),
        layout=dict(algorithm="igraph Fruchterman-Reingold unweighted union, grid=grid",computed_once=True,
                    shared_positions=True,shared_axis_limits=list(limits),
                    coordinates_sha256=hashlib.sha256(positions[shown].astype("<f8").tobytes()).hexdigest()),
        no_reclustering=True,no_new_genotypes=True,contains_individual_identifiers=False,
        privacy="No sample IDs or node-to-ID map exported; graph structure remains sensitive, keep private.",
        scope=SCOPE,input_sha256=data["input_sha256"],source_sha256=data["source_sha256"],
        core_sha256=data["core_sha256"],renderer_sha256=sha256(__file__),package_versions=versions,
        outputs_sha256={p.name:sha256(p) for p in sorted(output.iterdir()) if p.is_file()})
    require(len(list(output.glob("*.png")))==result["n_png"]
            and len(list(output.glob("*.pdf")))==result["n_pdf"]
            and len(list(output.glob("*.descripcion.md")))==result["n_descriptions"],"Incomplete figure inventory")
    with (output/"manifest.json").open("x",encoding="utf-8") as handle:
        json.dump(result,handle,indent=2,ensure_ascii=False,allow_nan=False);handle.write("\n")
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir",required=True)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--prefix",default="M16_5_chr22__A01__09")
    parser.add_argument("--dpi",type=int,default=300)
    parser.add_argument("--layout-seed",type=int,default=42)
    parser.add_argument("--layout-iterations",type=int,default=200)
    parser.add_argument("--network-config-ids",default=None,
                        help="Explicit comma-separated evaluated graph IDs; default three reference graphs")
    parser.add_argument("--network-resolution",type=float,default=1.0)
    parser.add_argument("--all-networks",action="store_true",
                        help="Optional exhaustive networks; not the default presentation")
    args=parser.parse_args(argv)
    result=run(**vars(args))
    print(json.dumps({k:result[k] for k in ("status","n_cells","n_png","n_pdf")}))


if __name__=="__main__":
    main()
