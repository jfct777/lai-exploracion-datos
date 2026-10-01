#!/usr/bin/env python3
"""Descriptive R/C communities on authenticated, jointly evaluable autosomal pairs.

R is masked whole-catalogue rare Jaccard; C is the positive part of the common
GRM. Neither matrix is a physical length or a kinship probability. Existing
M16.5 Leiden and spectral/UMAP routines are reused without invented bp fields.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

import r02_genomic_pair_evidence as evidence

AUTOSOMES = [str(c) for c in range(1, 23)]
DEFAULTS = dict(expected_samples=2619, resolutions=[0.5, 1.0, 2.0], n_seeds=25,
                seed=42, min_community_size=3, include_combination=True, include_rare_graph=True,
                n_spectral=15, n_neighbors=15, min_dist=0.3, dpi=300)


def settings_from(path):
    supplied = json.loads(Path(path).read_text())
    evidence.require(isinstance(supplied, dict) and not set(supplied) - set(DEFAULTS),
                     "Unknown weighted-community setting")
    result = dict(DEFAULTS, **supplied)
    for key in ("expected_samples", "n_seeds", "seed", "min_community_size", "n_spectral", "n_neighbors", "dpi"):
        evidence.require(type(result[key]) is int, "Integer setting required")
    evidence.require(result["expected_samples"] >= 4 and result["n_seeds"] >= 2
                     and result["seed"] >= 0 and result["min_community_size"] >= 1,
                     "Invalid cohort or Leiden settings")
    evidence.require(result["resolutions"] == [0.5, 1.0, 2.0],
                     "This campaign evaluates exactly gamma 0.5, 1 and 2")
    evidence.require(type(result["include_combination"]) is bool
                     and type(result["include_rare_graph"]) is bool
                     and 72 <= result["dpi"] <= 600 and result["n_spectral"] == 15
                     and result["n_neighbors"] == 15 and result["min_dist"] == 0.3,
                     "Unexpected projection or combination settings")
    return result


def matrices_from(arrays, n_samples, include_combination):
    """No missing pair is turned into a zero edge; insufficient input fails."""
    required = {"I", "U", "Q", "J", "K", "common_N", "common_numerator", "pair_eligible_R_C"}
    evidence.require(required <= set(arrays), "Missing pair-evidence fields")
    shape = (n_samples, n_samples)
    for name in required:
        evidence.require(arrays[name].shape == shape, "Evidence dimensions differ from cohort")
    evidence.validate_rare_counts(arrays["I"], arrays["U"], arrays["Q"])
    common_n = arrays["common_N"]
    evidence.require(common_n.dtype.kind in "iu" and np.all(common_n >= 0)
                     and np.array_equal(common_n, common_n.T),
                     "Common denominators must be symmetric nonnegative integers")
    evidence.require(np.isfinite(arrays["common_numerator"]).all()
                     and np.allclose(arrays["common_numerator"], arrays["common_numerator"].T,
                                     rtol=1e-12, atol=1e-14),
                     "Invalid common numerator")
    offdiag = ~np.eye(n_samples, dtype=bool)
    eligible = (arrays["U"] > 0) & (arrays["Q"] > 0) & (arrays["common_N"] > 0)
    evidence.require(np.array_equal(eligible, arrays["pair_eligible_R_C"]),
                     "Saved pair eligibility disagrees with denominators")
    evidence.require(np.all(eligible[offdiag]),
                     "Some off-diagonal pairs are unevaluable; no zero imputation or automatic cohort removal allowed")
    expected_j = evidence.masked_jaccard(arrays["I"], arrays["U"])
    expected_k = np.full(shape, np.nan)
    np.divide(arrays["common_numerator"], arrays["common_N"], out=expected_k, where=arrays["common_N"] > 0)
    evidence.require(np.allclose(arrays["J"], expected_j, rtol=1e-12, atol=1e-14, equal_nan=True)
                     and np.allclose(arrays["K"], expected_k, rtol=1e-12, atol=1e-14, equal_nan=True),
                     "Saved similarity does not reproduce its sufficient statistics")
    for name in ("J", "K"):
        values = arrays[name]
        evidence.require(np.isfinite(values[offdiag]).all()
                         and np.allclose(values, values.T, rtol=1e-12, atol=1e-14, equal_nan=True),
                         "Non-finite or asymmetric observed similarity")
    rare = arrays["J"].copy()
    common = np.maximum(arrays["K"], 0.0)
    np.fill_diagonal(rare, 0.0)
    np.fill_diagonal(common, 0.0)
    evidence.require(np.isfinite(rare).all() and ((rare >= 0) & (rare <= 1)).all()
                     and np.isfinite(common).all(), "Invalid graph weights")
    graphs = {"R": rare, "C": common}
    scales = {name: float(matrix[matrix > 0].mean()) if np.any(matrix > 0) else None
              for name, matrix in graphs.items()}
    combination_status = "NOT_REQUESTED"
    if include_combination:
        if all(scales[name] is not None for name in ("R", "C")):
            graphs["R_plus_C"] = 0.5 * rare / scales["R"] + 0.5 * common / scales["C"]
            combination_status = "DEFINED_ALPHA_0_5"
        else:
            combination_status = "NOT_ESTIMABLE_NO_POSITIVE_WEIGHTS"
    denominator_summary = {}
    for name in ("U", "Q", "common_N"):
        values = arrays[name][np.triu(offdiag, 1)]
        denominator_summary[name] = dict(minimum=int(values.min()), median=float(np.median(values)), maximum=int(values.max()))
    return graphs, dict(n_possible_pairs=n_samples * (n_samples - 1) // 2,
                        n_unevaluable_pairs=0, denominator_summary=denominator_summary,
                        common_negative_pairs=int(np.count_nonzero(np.triu((arrays["K"] < 0) & offdiag, 1))),
                        positive_weight_scales=scales, combination_status=combination_status,
                        normalization_scope="all eligible pairs in this descriptive transductive cohort; not a training/evaluation split")


def write_json(path, payload):
    with Path(path).open("x") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False, ensure_ascii=False)
        handle.write("\n")


def draw_graph(core, matrix, assignments, name, settings, directory, metadata=None, metadata_name=None):
    import m165_spectral_figures as projection
    import matplotlib.pyplot as plt
    coordinates, diagnostic = None, dict(status="NO_EDGES_NO_INFORMATIVE_PROJECTION")
    if matrix.nnz:
        coordinates, embedding, eigenvalues, diagnostic = projection.project_matrix(matrix, settings, core=core)
        np.savez_compressed(directory / "coordinates.private.npz", coordinates=coordinates,
                            spectral_embedding=embedding, eigenvalues=eigenvalues,
                            isolated=np.asarray(matrix.sum(axis=1)).ravel() == 0)
    for gamma in settings["resolutions"]:
        labels = assignments[f"community_res_{gamma:g}"].to_numpy(dtype=int)
        stem = directory / f"{name}_autosomes22_gamma{gamma:g}"
        if coordinates is None:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, "No hay pesos positivos: todas las personas quedan sin asignación.\nNo existe una proyección genética informativa.",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            fig.savefig(str(stem) + ".png", dpi=settings["dpi"])
            fig.savefig(str(stem) + ".pdf")
            plt.close(fig)
        else:
            core._plot_network_static(coords=coordinates, sub_idx=np.arange(matrix.shape[0]), sub_memb=labels,
                sub_wdeg=np.asarray(matrix.sum(axis=1)).ravel(), membership=labels,
                out_path=Path(str(stem) + ".png"), confidence=None,
                metadata_values=None if metadata is None else np.asarray(metadata), metadata_name=metadata_name,
                dpi=settings["dpi"], width_in=12., height_in=8., export_pdf=True, export_svg=False,
                label_min_size=10, community_annotations=None, adjust_labels=False,
                figure_title=f"{name} · 22 autosomas (chr1–chr22) · gamma={gamma:g} · N={matrix.shape[0]}",
                figure_note="Comunidades descriptivas, no poblaciones confirmadas. Pesos sin unidades de longitud.\n"
                            "Mismas coordenadas para las tres resoluciones del grafo. No alineadas entre R y C.\n"
                            "Gris = sin asignación. Las posiciones de personas aisladas no contienen información genética.")
        Path(str(stem) + ".descripcion.md").write_text(
            f"# {name}: 22 autosomas, resolución {gamma:g}\n\n"
            "Se reutilizan Leiden ponderado y la proyección espectral/UMAP de M16.5. "
            "La partición representativa maximiza calidad RB dentro del mismo grafo y resolución; "
            "no se escoge la resolución por calidad ni por apariencia. Las 25 semillas planificadas "
            "miden variabilidad algorítmica, no réplicas biológicas; el número efectivo consta en el manifiesto. "
            "R=I/U sobre todo el catálogo raro conjuntamente evaluable. C=max(GRM,0). "
            "R_plus_C=0,5 R/sR+0,5 C/sC, donde cada escala es la media de pesos positivos fuera de la diagonal. "
            "No se aplican filtros de longitud, Jaccard mínimo, RC ni selección de intervalos M14. "
            "Los metadatos sólo colorean; no entrenan la proyección ni Leiden. "
            "Las coordenadas se conservan; semilla fija no garantiza igualdad bit a bit en subespacios espectrales degenerados.\n")
    return diagnostic


def run(evidence_path, samples_path, settings_path, output, metadata_path=None,
        metadata_sample_column=None, metadata_color_column=None):
    import ibd_community_enhanced as core
    import m165_spectral_figures as projection
    settings = settings_from(settings_path)
    samples = evidence.read_samples(samples_path, settings["expected_samples"])
    output = Path(output)
    evidence.require(not output.exists(), "Output exists; no overwrite")
    arrays, source_manifest = evidence.load_bundle(evidence_path, samples, "aggregate")
    evidence.require(source_manifest["chromosomes"] == AUTOSOMES, "All 22 autosomes required for both R and C")
    inputs = {str(Path(p)): evidence.sha256(p) for p in
              (evidence_path, Path(evidence_path).with_suffix(".manifest.json"), samples_path, settings_path)}
    matrices, measurement = matrices_from(arrays, len(samples), settings["include_combination"])
    if not settings["include_rare_graph"]:
        # A common-LD sensitivity reuses the same rare definition, but the
        # combination still needs its original normalization and eligibility.
        del matrices["R"]
        measurement["standalone_rare_graph"] = "NOT_REQUESTED; R still defines R_plus_C when requested"
    else:
        measurement["standalone_rare_graph"] = "INCLUDED"
    metadata, metadata_info = None, None
    if metadata_path is not None:
        metadata, metadata_info = projection.read_metadata(metadata_path, metadata_sample_column, metadata_color_column, samples)
        inputs[str(Path(metadata_path))] = evidence.sha256(metadata_path)
    else:
        evidence.require(metadata_sample_column is None and metadata_color_column is None, "Metadata columns require a metadata file")
    output.mkdir(parents=True, exist_ok=False)
    graph_records = []
    for name, weights in matrices.items():
        directory = output / name
        directory.mkdir()
        matrix = sparse.csr_matrix(weights)
        matrix.eliminate_zeros()
        graph = core.sparse_to_igraph(matrix, samples)
        sparse.save_npz(directory / "weights.npz", matrix)
        active = np.asarray(graph.degree()) > 0
        pd.DataFrame(dict(sample_id=samples, degree=graph.degree(), weighted_degree=np.asarray(matrix.sum(axis=1)).ravel(),
                          active=active)).to_csv(directory / "nodes.private.tsv", sep="\t", index=False)
        assignments, qualities, consensus, _, memberships = core.run_leiden_multiresolution(
            graph, settings["resolutions"], settings["n_seeds"], settings["min_community_size"], settings["seed"], 1.0)
        ari = core.compute_ari_multi_seed(memberships)
        core.save_leiden(directory, samples, assignments, qualities, consensus, 1.0, ari_df=ari)
        with gzip.open(directory / "seed_memberships.private.tsv.gz", "wt", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(("resolution", "seed", "sample_id", "community_raw", "active"))
            for gamma, vectors in memberships.items():
                seeds = qualities.loc[qualities.resolution == gamma, "seed"].tolist()
                for seed, vector in zip(seeds, vectors):
                    evidence.require(np.all(vector[~active] == -1), "Isolated node assigned to a community")
                    writer.writerows((gamma, seed, sample, int(label), int(available))
                                     for sample, label, available in zip(samples, vector, active))
        summary = []
        for gamma in settings["resolutions"]:
            labels = assignments[f"community_res_{gamma:g}"].to_numpy()
            assigned = labels >= 0
            _, sizes = np.unique(labels[assigned], return_counts=True)
            summary.append(dict(graph=name, resolution=gamma, n_cohort=len(samples), n_active=int(active.sum()),
                                n_isolated=int((~active).sum()), n_assigned=int(assigned.sum()),
                                n_communities=len(sizes), largest_community=int(sizes.max()) if sizes.size else 0))
        pd.DataFrame(summary).to_csv(directory / "resolution_summary.tsv", sep="\t", index=False)
        diagnostic = draw_graph(core, matrix, assignments, name, settings, directory, metadata, metadata_color_column)
        write_json(directory / "projection_diagnostic.json", diagnostic)
        record = dict(graph=name, status="COMPLETE_DESCRIPTIVE" if matrix.nnz else "COMPLETE_NO_POSITIVE_EDGES",
                      chromosomes=AUTOSOMES, n_edges=graph.ecount(), n_active=int(active.sum()),
                      n_cohort=len(samples), weight_units="dimensionless", settings=settings,
                      outputs_sha256={p.name: evidence.sha256(p) for p in directory.iterdir() if p.is_file()})
        write_json(directory / "manifest.json", record)
        graph_records.append(record)
    for path, expected in inputs.items():
        evidence.require(evidence.sha256(path) == expected, "Input changed during communities/projection")
    result = dict(status="COMPLETE_DESCRIPTIVE_WEIGHTED_COMMUNITIES", chromosomes=AUTOSOMES,
                  n_cohort=len(samples), graphs=graph_records, measurement=measurement, metadata=metadata_info,
                  parameters=settings, input_sha256=inputs, source_sha256={str(Path(p).name): evidence.sha256(p)
                    for p in (__file__, evidence.__file__, core.__file__, projection.__file__)},
                  scope="transductive description, not biological confirmation or supervised independent evaluation",
                  no_bp_fields=True, no_M14_interval_selection=True, no_RC_filter=True,
                  no_missing_pairs_imputed_zero=True, contains_individual_identifiers=True,
                  public_distribution_allowed=False)
    write_json(output / "manifest.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("evidence", "samples", "settings", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--metadata")
    parser.add_argument("--metadata-sample-column")
    parser.add_argument("--metadata-color-column")
    args = parser.parse_args(argv)
    result = run(args.evidence, args.samples, args.settings, args.output, args.metadata,
                 args.metadata_sample_column, args.metadata_color_column)
    print(json.dumps({"status": result["status"], "graphs": [g["graph"] for g in result["graphs"]]}))


if __name__ == "__main__":
    main()
