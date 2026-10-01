#!/usr/bin/env python3
"""Historical spectral/UMAP figures from authenticated saved graphs; no clustering."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).parent))
import m165_sweep_figures as saved


DEFAULTS = dict(prefix="M16_5_chr22__A01__10", dpi=300, seed=42,
                n_spectral=15, n_neighbors=15, min_dist=0.3, resolution=1.0,
                config_ids=list(saved.DEFAULT_NETWORKS), expected_samples=2619,
                width_inches=12.0, height_inches=8.0, resolutions=None, combined_pdf=True,
                inline_metadata_legend=False, legend_wrap_chars=56)
EIGENVALUE_TOLERANCE = 1e-8
KNOWN_CONFIGS = tuple(f"L{length}_G50000_N20_T{threshold}_U0" for length,threshold in saved.GRID)


def settings_from_json(path=None, text=None):
    saved.require((path is None) != (text is None), "Provide exactly one parameters JSON source")
    supplied = json.loads(Path(path).read_text(encoding="utf-8") if path is not None else text)
    saved.require(isinstance(supplied, dict) and not set(supplied)-set(DEFAULTS),
                  "Unknown figure parameter")
    settings = dict(DEFAULTS, **supplied)
    saved.require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", settings["prefix"]), "Unsafe prefix")
    for key in ("dpi", "seed", "n_spectral", "n_neighbors", "expected_samples"):
        saved.require(type(settings[key]) is int, "Integer figure parameter required")
    saved.require(72 <= settings["dpi"] <= 600 and settings["expected_samples"] >= 4
                  and 6 <= settings["width_inches"] <= 24 and 6 <= settings["height_inches"] <= 18,
                  "Invalid figure size or cohort")
    saved.require(settings["seed"] == 42 and settings["n_spectral"] == 15
                  and settings["n_neighbors"] == 15 and settings["min_dist"] == 0.3,
                  "Historical projection settings must remain unchanged")
    configs=settings["config_ids"]
    saved.require(isinstance(configs,list) and configs and all(isinstance(c,str) for c in configs)
                  and len(set(configs))==len(configs) and set(configs)<=set(KNOWN_CONFIGS),
                  "Select unique authenticated stage08 graph configurations")
    saved.require(type(settings["resolution"]) in (int,float) and settings["resolution"] in saved.GAMMAS,
                  "Scalar resolution was not evaluated")
    resolutions=selected_resolutions(settings)
    saved.require(isinstance(resolutions,list) and resolutions
                  and all(type(g) in (int,float) and g in saved.GAMMAS for g in resolutions)
                  and len(set(resolutions))==len(resolutions), "Select unique saved resolutions")
    saved.require(type(settings["combined_pdf"]) is bool,"combined_pdf must be a boolean")
    saved.require(type(settings["inline_metadata_legend"]) is bool,
                  "inline_metadata_legend must be a boolean")
    saved.require(type(settings["legend_wrap_chars"]) is int
                  and 32 <= settings["legend_wrap_chars"] <= 100,
                  "legend_wrap_chars must be an integer from 32 to 100")
    return settings


def selected_resolutions(settings):
    """An explicit list overrides the backward-compatible scalar fallback."""
    return [settings["resolution"]] if settings["resolutions"] is None else settings["resolutions"]


def numeric_hash(array, dtype):
    import numpy as np
    return hashlib.sha256(np.asarray(array, dtype=dtype).tobytes()).hexdigest()


def matrix_hash(matrix):
    digest = hashlib.sha256()
    for array, dtype in ((matrix.shape, "<i8"), (matrix.indptr, "<i8"),
                         (matrix.indices, "<i8"), (matrix.data, "<f8")):
        digest.update(bytes.fromhex(numeric_hash(array, dtype)))
    return digest.hexdigest()


def authenticated_matrix(directory, graph):
    """Cross-check original CSR against authenticated edge/node tables, without filtering."""
    import numpy as np
    import scipy.sparse as sp
    manifest = json.loads((directory/"manifest.json").read_text())
    name = "graph_sharing_matrix.npz"
    saved.require(name in manifest["outputs_sha256"], "Uncertified original matrix")
    actual = saved.sha256(directory/name)
    saved.require(actual == manifest["outputs_sha256"][name], "Original matrix hash mismatch")
    matrix = sp.load_npz(directory/name).tocsr(copy=True)
    matrix.sum_duplicates(); matrix.sort_indices()
    n = graph["n_cohort"]
    saved.require(matrix.shape == (n,n) and np.isfinite(matrix.data).all()
                  and (matrix.data > 0).all() and not matrix.diagonal().any()
                  and (matrix != matrix.T).nnz == 0, "Invalid original weighted matrix")
    nodes = saved.table(directory/"graph_nodes.tsv")
    index = {row["sample_id"]: i for i,row in enumerate(nodes)}
    a,b,w = [],[],[]
    for row in saved.table(directory/"graph_edges.tsv.gz"):
        a.append(index[row["sample_a"]]); b.append(index[row["sample_b"]]); w.append(float(row["weight"]))
    reconstructed = sp.csr_matrix((w+w,(a+b,b+a)), shape=(n,n))
    delta = matrix-reconstructed
    saved.require(matrix.nnz == 2*len(graph["edges"])
                  and (not delta.nnz or np.max(np.abs(delta.data)) <= 1e-10),
                  "Original matrix disagrees with saved edges")
    weighted_degree = np.asarray(matrix.sum(axis=1)).ravel()
    saved.require(np.allclose(weighted_degree, [float(r["weighted_degree"]) for r in nodes],
                              rtol=1e-7, atol=1e-10), "Matrix/node weighted-degree mismatch")
    return matrix, [row["sample_id"] for row in nodes], actual


def read_metadata(path, sample_column, color_column, sample_ids):
    """Exact ID join kept in memory; no sample identifiers enter output tables."""
    saved.require(sample_column and color_column and sample_column != color_column,
                  "Metadata requires distinct explicit sample and colour columns")
    delimiter = "\t" if Path(path).suffix.lower() in (".tsv", ".txt") else ","
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        saved.require(reader.fieldnames and sample_column in reader.fieldnames
                      and color_column in reader.fieldnames, "Required metadata columns missing")
        values, full_rows = {}, {}
        n_rows, duplicates = 0, 0
        for row in reader:
            n_rows += 1
            sample = row[sample_column]
            saved.require(sample, "Empty metadata sample key")
            if sample in values:
                saved.require(row == full_rows[sample], "Conflicting duplicate metadata sample key")
                duplicates += 1
                continue
            full_rows[sample] = row
            values[sample] = row[color_column].strip() or "Sin dato"
    saved.require(all(sample in values for sample in sample_ids), "Metadata does not cover the full cohort")
    ordered = [values[sample] for sample in sample_ids]
    return ordered, dict(n_metadata_rows=n_rows, n_unique_metadata_ids=len(values),
                         n_identical_duplicates_collapsed=duplicates,
                         n_extra_metadata_ids=len(set(values)-set(sample_ids)), n_cohort_matched=len(ordered),
                         n_missing_colour=ordered.count("Sin dato"), n_levels=len(set(ordered)),
                         sample_column=sample_column, color_column=color_column,
                         sha256=saved.sha256(path))


def project_matrix(matrix, settings, core=None, umap_factory=None):
    """Literal historical mathematics, all nodes, no labels and no fallback."""
    import numpy as np
    from scipy.sparse.csgraph import connected_components
    if core is None:
        import ibd_community_enhanced as core
    if umap_factory is None:
        import umap
        umap_factory = umap.UMAP
    before = matrix_hash(matrix)
    embedding = core._spectral_embedding(matrix, n_components=settings["n_spectral"], laplacian=True)
    saved.require(embedding.shape[0] == matrix.shape[0] and np.isfinite(embedding).all(),
                  "Invalid spectral embedding")
    normalized = core.laplacian_normalize(matrix).astype(np.float64)
    product = normalized @ embedding
    eigenvalues = np.sum(embedding*product, axis=0)/np.sum(embedding*embedding, axis=0)
    residual = np.linalg.norm(product-embedding*eigenvalues, axis=0)
    active = np.asarray(matrix.sum(axis=1)).ravel() > 0
    n_components, component_labels = connected_components(matrix[active][:,active], directed=False)
    component_sizes = np.bincount(component_labels)
    neighbors = min(settings["n_neighbors"], max(2, matrix.shape[0]-1))
    reducer = umap_factory(n_components=2, random_state=settings["seed"],
                           n_neighbors=neighbors, min_dist=settings["min_dist"],
                           metric="euclidean", n_jobs=1)
    try:
        coordinates = np.asarray(reducer.fit_transform(embedding))
    except Exception as exc:
        raise RuntimeError("UMAP failed; no spectral fallback or success receipt permitted") from exc
    saved.require(coordinates.shape == (matrix.shape[0],2) and np.isfinite(coordinates).all(),
                  "UMAP returned invalid coordinates")
    saved.require(matrix_hash(matrix) == before, "Projection mutated input matrix")
    diagnostic = dict(n_nodes=matrix.shape[0], n_active=int(active.sum()), n_isolated=int((~active).sum()),
        n_connected_components_active=int(n_components), largest_component=int(max(component_sizes,default=0)),
        n_stationary_candidates=int(n_components), n_spectral_actual=embedding.shape[1],
        n_modes_lambda_near_one=int(np.isclose(eigenvalues,1,rtol=0,atol=EIGENVALUE_TOLERANCE).sum()),
        eigenvalue_tolerance=EIGENVALUE_TOLERANCE, eigenvalues=eigenvalues.tolist(),
        eigen_residual_max=float(residual.max()), n_neighbors_actual=neighbors,
        isolates_included_in_projection=True, isolated_positions_informative=False,
        stationary_subspace_may_dominate=bool(n_components >= embedding.shape[1]),
        matrix_numeric_sha256_before=before, matrix_numeric_sha256_after=matrix_hash(matrix),
        coordinates_sha256=numeric_hash(coordinates,"<f8"), umap_parameters=reducer.get_params())
    return coordinates, embedding, eigenvalues, diagnostic


def load_coordinate_source(directory, expected_sha256, settings, spectral_hash):
    """Anchor a previous coordinate package to an explicitly supplied manifest hash."""
    if directory is None:
        saved.require(expected_sha256 is None,"Coordinate source hash without directory")
        return None
    saved.require(isinstance(expected_sha256,str) and re.fullmatch(r"[0-9a-f]{64}",expected_sha256),
                  "Coordinate source requires an explicit manifest SHA256")
    root=Path(directory); path=root/'manifest.json'
    saved.require(saved.sha256(path)==expected_sha256,"Coordinate source manifest hash mismatch")
    manifest=json.loads(path.read_text())
    saved.require(manifest['status']=='COMPLETE_HISTORICAL_VISUAL_REPRODUCTION'
                  and manifest['no_reclustering'] and manifest['no_threshold_changes']
                  and not manifest['projection_uses_labels']
                  and manifest['input_sha256_before']==manifest['input_sha256_after'],
                  "Coordinate source is not an authenticated unchanged graph projection")
    for key in ('seed','n_spectral','n_neighbors','min_dist','expected_samples'):
        saved.require(manifest['parameters'][key]==settings[key],"Coordinate projection settings differ")
    saved.require(manifest['spectral_function_sha256']==spectral_hash,"Coordinate spectral function differs")
    diagnostics={d['config_id']:d for d in manifest['diagnostics']}
    saved.require(len(diagnostics)==len(manifest['diagnostics']) and set(diagnostics)<=set(KNOWN_CONFIGS)
                  and set(diagnostics)==set(manifest['parameters']['config_ids']),
                  "Coordinate source has duplicate or mislabeled configurations")
    return dict(root=root,manifest=manifest,diagnostics=diagnostics,
                hashes={'manifest.json':expected_sha256},reused=[])


def reuse_projection(source, graph, matrix, settings, current_hashes):
    """Return certified coordinates or None only if the graph was never in that source."""
    import numpy as np
    if source is None or graph['config_id'] not in source['diagnostics']:
        return None
    config=graph['config_id']; manifest=source['manifest']; diagnostic=dict(source['diagnostics'][config])
    for name,digest in current_hashes.items():
        if name.startswith(config+'/'):
            saved.require(manifest['input_sha256_before'].get(name)==digest,
                          "Coordinate source graph inputs differ")
    saved.require(diagnostic['n_nodes']==graph['n_cohort']
                  and diagnostic['n_active']==graph['n_active']
                  and diagnostic['n_edges']==len(graph['edges'])
                  and diagnostic['node_order_sha256']==current_hashes[f'{config}/graph_nodes.tsv']
                  and diagnostic['matrix_file_sha256']==current_hashes[f'{config}/graph_sharing_matrix.npz']
                  and diagnostic['matrix_numeric_sha256_before']==matrix_hash(matrix)
                  ==diagnostic['matrix_numeric_sha256_after'],"Coordinate graph/node-order mismatch")
    prefix=manifest['parameters']['prefix']
    saved.require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}",prefix),"Unsafe coordinate-source prefix")
    allowed={f'private/{prefix}_{config}.coordinates.npz'}
    allowed.update(f'private/{prefix}_{config}_gamma{gamma:g}.coordinates.npz' for gamma in saved.GAMMAS)
    relative=diagnostic.get('coordinates_file')
    if relative is None:  # Original stage10 scalar-resolution receipt.
        gamma=diagnostic['resolution']
        saved.require(gamma in saved.GAMMAS,"Unrecognized source coordinate resolution")
        relative=f'private/{prefix}_{config}_gamma{gamma:g}.coordinates.npz'
    saved.require(relative in allowed and relative in manifest['outputs_sha256'],
                  "Uncertified or unsafe coordinate artifact path")
    path=source['root']/relative; digest=saved.sha256(path)
    saved.require(digest==manifest['outputs_sha256'][relative],"Coordinate artifact hash mismatch")
    with np.load(path,allow_pickle=False) as archive:
        saved.require(set(archive.files)=={'node_index','coordinates','spectral_embedding','eigenvalues','isolated'},
                      "Unexpected coordinate artifact schema")
        arrays={key:archive[key].copy() for key in archive.files}
    n=graph['n_cohort']; k=diagnostic['n_spectral_actual']
    saved.require(np.array_equal(arrays['node_index'],np.arange(n))
                  and arrays['coordinates'].shape==(n,2)
                  and arrays['spectral_embedding'].shape==(n,k)
                  and arrays['eigenvalues'].shape==(k,)
                  and np.array_equal(arrays['isolated'],np.asarray(matrix.sum(axis=1)).ravel()==0)
                  and all(np.isfinite(arrays[key]).all() for key in ('coordinates','spectral_embedding','eigenvalues')),
                  "Coordinate dimensions/order/isolate mask mismatch")
    saved.require(numeric_hash(arrays['coordinates'],'<f8')==diagnostic['coordinates_sha256']
                  and np.array_equal(arrays['eigenvalues'],np.asarray(diagnostic['eigenvalues'])),
                  "Coordinate values disagree with source receipt")
    source['hashes'][relative]=digest;source['reused'].append(config)
    diagnostic['coordinate_origin']='authenticated_saved_coordinates'
    diagnostic['coordinate_source_manifest_sha256']=source['hashes']['manifest.json']
    diagnostic['coordinate_source_artifact_sha256']=digest
    return arrays['coordinates'],arrays['spectral_embedding'],arrays['eigenvalues'],diagnostic


def render_projection(core, matrix, coordinates, labels, graph, diagnostic, settings,
                      output, stem, metadata=None, metadata_name=None):
    import numpy as np
    n = len(labels); assigned = int((labels >= 0).sum())
    gamma=diagnostic['resolution']
    title = (f"Resultado descriptivo de {saved.chromosome_label(graph)} · {saved.graph_label(graph)} · gamma = {gamma:g}\n"
             f"G = 50 kb · N = 20 · U = 0 · cohorte: {n} · activos: {graph['n_active']} · asignados: {assigned}")
    note = ("Proyección descriptiva: 15 dimensiones espectrales → UMAP; 15 vecinos, min_dist = 0,3, semilla = 42.\n"
            f"Componentes con aristas: {diagnostic['n_connected_components_active']}; modos calculados λ≈1: {diagnostic['n_modes_lambda_near_one']}/{diagnostic['n_spectral_actual']}. "
            f"Aislados: {diagnostic['n_isolated']}, incluidos pero sin posición informativa.\n"
            "Mismas coordenadas entre resoluciones de un grafo, independientes entre grafos. Distancias entre islas no son distancias genéticas.\n"
            "Partición guardada de máxima calidad RB entre 25 semillas; no consenso ni reclustering.\n"
            "Todos los nodos dibujados, sin aristas; leyenda hasta 30 grupos. Gris = no asignado Leiden, también en metadatos; no significa dato faltante.")
    if settings['inline_metadata_legend']:
        note = note.replace('leyenda hasta 30 grupos', 'leyenda con todos los grupos')
        note += ("\nLeyenda integrada: cada C# enumera las categorías fineSTRUCTURE y sus cantidades; "
                 "los conteos suman n. La correspondencia es muchos a muchos, no equivalencia.")
    core._plot_network_static(coords=coordinates, sub_idx=np.arange(n), sub_memb=labels,
        sub_wdeg=np.asarray(matrix.sum(axis=1)).ravel(), membership=labels,
        out_path=output/f"{stem}.png", confidence=None,
        metadata_values=None if metadata is None else np.asarray(metadata), metadata_name=metadata_name,
        dpi=settings["dpi"], width_in=settings["width_inches"], height_in=settings["height_inches"],
        export_pdf=True, export_svg=False, label_min_size=10,
        community_annotations=None, adjust_labels=False, figure_title=title, figure_note=note,
        inline_metadata_legend=settings['inline_metadata_legend'],
        legend_wrap_chars=settings['legend_wrap_chars'])
    description = (title+"\n\n"+note+"\n\n"
        f"L = {graph['length_bp']/1000:g} kb: longitud mínima de una cadena M14; "
        f"T = {graph['threshold_bp']/1e6:g} Mb: longitud acumulada mínima por pareja para retener la arista; "
        "G = 50 kb: salto máximo entre posiciones compartidas; N = 20: mínimo de posiciones compartidas por cadena; "
        f"U = 0: sin filtro adicional por longitud del mayor segmento; gamma = {gamma:g}: resolución Leiden ya ejecutada. "
        "Se reutilizan _spectral_embedding y _plot_network_static del programa M16.5. "
        "El embedding conserva literalmente los autovectores mayores de D^(-1/2) W D^(-1/2), "
        "incluidos los estacionarios, sin ponderarlos por autovalores ni retirar aislados. W es el peso log1p(bp) guardado. "
        "La multiplicidad de λ = 1 en componentes desconectadas limita la interpretación del subespacio truncado; "
        "islas visuales no demuestran poblaciones. UMAP no recibe etiquetas ni metadatos. Sus vecinos internos no son nuevas aristas científicas. "
        "Todos los nodos se dibujan; tamaño = 10 + 120 sqrt(grado ponderado / máximo). "
        "Los no asignados se atenúan en gris (tamaño × 0,4, alpha 0,35), incluidos los aislados. "
        "La leyenda histórica enumera las 30 comunidades mayores y los no asignados; no se eliminan los puntos de grupos restantes. "
        "Si hay panel de metadatos usa las mismas coordenadas, sin intervenir en proyección ni asignación. "
        "En ese panel el gris también marca personas no asignadas por Leiden; no indica ausencia de etiqueta fineSTRUCTURE. "
        "La posición de las personas aisladas carece de información de co-sharing. "
        "La geometría se ajusta o se recupera una sola vez por grafo y se conserva idéntica en todas sus resoluciones; "
        "no está alineada entre grafos. Los colores/C# son etiquetas locales, no ancestrías ni equivalencias entre resoluciones. "
        "La semilla fija no garantiza reproducción bit a bit: las bases de subespacios degenerados y la "
        "inicialización espectral interna de UMAP pueden variar numéricamente. Los NPZ guardan las posiciones "
        "efectivamente usadas y permiten reutilizarlas sin volver a estimarlas. "
        "Los NPZ privados contienen índices, no IDs; conservan información individual y no se publican automáticamente.")
    if settings['inline_metadata_legend']:
        description = description.replace(
            "La leyenda histórica enumera las 30 comunidades mayores y los no asignados; no se eliminan los puntos de grupos restantes. ",
            "La leyenda integrada enumera TODAS las comunidades y los no asignados, con su composición completa en fineSTRUCTURE. "
            "Cada número tras una categoría es la cantidad de personas de esa categoría dentro del grupo; todos suman n del grupo, incluidos los datos faltantes si los hubiera. "
            "Una categoría puede pertenecer a varias comunidades. No representa ancestría ni asignación uno a uno. "
            "Sólo se amplía el espacio de leyenda y el lienzo cuando hace falta; no se recalculan ni alteran las coordenadas. ")
    description = f"Alcance: {saved.chromosome_label(graph)}. " + description
    (output/f"{stem}.descripcion.md").write_text(description+"\n",encoding="utf-8")


def append_raster_pages(path, settings, books):
    """Concatenate literal historical PNGs without repositioning any marks."""
    import matplotlib.pyplot as plt
    if not books:
        return
    raster=plt.imread(path)
    fig=plt.figure(figsize=(raster.shape[1]/settings['dpi'],raster.shape[0]/settings['dpi']))
    ax=fig.add_axes([0,0,1,1]);ax.imshow(raster);ax.axis('off')
    for book in books:
        book.savefig(fig,dpi=settings['dpi'])
    plt.close(fig)


def run(results_dir, output_dir, parameters_json=None, metadata_file=None,
        metadata_sample_column=None, metadata_color_column="finestructure_clusters", parameters_json_text=None,
        coordinates_source_dir=None, coordinates_source_manifest_sha256=None):
    settings = settings_from_json(parameters_json, parameters_json_text)
    output = Path(output_dir); root = Path(results_dir)
    saved.require(not output.exists(), "Output directory exists; no overwrite")
    saved.require(metadata_file is not None or metadata_sample_column is None, "Metadata column without file")
    data = saved.load_results(root)
    saved.require(data["n_cohort"] == settings["expected_samples"], "Unexpected full-cohort denominator")
    import numpy as np
    import ibd_community_enhanced as core
    from matplotlib.backends.backend_pdf import PdfPages
    before = dict(data["input_sha256"])
    core_hash = saved.sha256(core.__file__)
    validator_hash = saved.sha256(saved.__file__)
    parameters_hash = (saved.sha256(parameters_json) if parameters_json is not None
                       else hashlib.sha256(parameters_json_text.encode()).hexdigest())
    spectral_hash=hashlib.sha256(inspect.getsource(core._spectral_embedding).encode()).hexdigest()
    coordinate_source=load_coordinate_source(coordinates_source_dir,coordinates_source_manifest_sha256,
                                             settings,spectral_hash)
    if settings['inline_metadata_legend']:
        saved.require(metadata_file is not None and coordinate_source is not None,
                      "Legend-only revision requires metadata and authenticated saved coordinates")
        saved.require(metadata_color_column == 'finestructure_clusters',
                      "Integrated fineSTRUCTURE legend requires finestructure_clusters metadata")
    resolutions=selected_resolutions(settings)
    metadata_record = None
    prepared = []
    for config_id in settings["config_ids"]:
        graph = next(g for g in data["graphs"] if g["config_id"] == config_id)
        matrix,samples,npz_hash = authenticated_matrix(root/config_id,graph)
        before[f"{config_id}/graph_sharing_matrix.npz"] = npz_hash
        label_hashes={gamma:numeric_hash(graph['labels'][gamma],'<i8') for gamma in resolutions}
        metadata = None
        if metadata_file is not None:
            metadata,record = read_metadata(metadata_file,metadata_sample_column,metadata_color_column,samples)
            saved.require(metadata_record is None or metadata_record == record, "Metadata changed during load")
            metadata_record = record
        projection=reuse_projection(coordinate_source,graph,matrix,settings,before)
        saved.require(not settings['inline_metadata_legend'] or projection is not None,
                      "Legend-only revision cannot compute a new projection")
        if projection is None:
            projection=project_matrix(matrix,settings,core=core)
            projection[3]['coordinate_origin']='computed_once_for_graph'
        coordinates,embedding,eigenvalues,diagnostic=projection
        saved.require(all(numeric_hash(graph['labels'][g],'<i8')==h for g,h in label_hashes.items()),
                      "Assignments changed during projection")
        reference_labels=np.asarray(graph['labels'][resolutions[0]])
        coordinate_stem=f"{settings['prefix']}_{config_id}"
        if len(resolutions)==1:
            coordinate_stem+=f'_gamma{resolutions[0]:g}'
        diagnostic.update(config_id=config_id, n_edges=len(graph["edges"]),
            node_order_sha256=before[f"{config_id}/graph_nodes.tsv"],
            assignment_values_sha256=label_hashes[resolutions[0]], matrix_file_sha256=npz_hash,
            n_assigned=int((reference_labels>=0).sum()), n_communities=len(set(reference_labels[reference_labels>=0])),
            resolution=resolutions[0], resolutions_rendered=resolutions,
            coordinates_file=f'private/{coordinate_stem}.coordinates.npz')
        prepared.append((graph,matrix,label_hashes,coordinates,embedding,eigenvalues,diagnostic,metadata))
        print(f"Geometría preparada: {config_id}; nodos={graph['n_cohort']}; origen={diagnostic['coordinate_origin']}",
              file=sys.stderr,flush=True)
    # A UMAP failure above never creates an apparently complete output folder.
    output.mkdir(parents=True,exist_ok=False)
    (output/"private").mkdir()
    diagnostics,invariants,figure_records,pdf_pages=[],[],[],{}
    source_rows={(r['config_id'],r['resolution']):r for r in data['rows']}
    with ExitStack() as stack:
        full_book=None
        if settings['combined_pdf']:
            filename=f"{settings['prefix']}_comparacion.pdf"
            full_book=stack.enter_context(PdfPages(output/filename));pdf_pages[filename]=0
        for graph,matrix,label_hashes,coords,emb,eigenvalues,diag,metadata in prepared:
            np.savez_compressed(output/diag['coordinates_file'],node_index=np.arange(graph['n_cohort']),
                                coordinates=coords,spectral_embedding=emb,eigenvalues=eigenvalues,
                                isolated=np.asarray(matrix.sum(axis=1)).ravel()==0)
            with ExitStack() as graph_stack:
                graph_book=None
                if len(resolutions)>1:
                    graph_pdf=f"{settings['prefix']}_{graph['config_id']}_resoluciones.pdf"
                    graph_book=graph_stack.enter_context(PdfPages(output/graph_pdf));pdf_pages[graph_pdf]=0
                for gamma in resolutions:
                    labels=np.asarray(graph['labels'][gamma],dtype=np.int64)
                    cell=dict(diag,resolution=gamma,assignment_values_sha256=label_hashes[gamma],
                              n_assigned=int((labels>=0).sum()),n_communities=len(set(labels[labels>=0])))
                    row=source_rows[(graph['config_id'],gamma)]
                    stem=f"{settings['prefix']}_{graph['config_id']}_gamma{gamma:g}"
                    render_projection(core,matrix,coords,labels,graph,cell,settings,output,stem,
                                      metadata,metadata_color_column if metadata is not None else None)
                    append_raster_pages(output/f'{stem}.png',settings,[b for b in (full_book,graph_book) if b is not None])
                    pdf_pages[f'{stem}.pdf']=1
                    if full_book is not None:pdf_pages[filename]+=1
                    if graph_book is not None:pdf_pages[graph_pdf]+=1
                    saved.require(numeric_hash(labels,'<i8')==label_hashes[gamma]
                                  and matrix_hash(matrix)==diag['matrix_numeric_sha256_before']
                                  and numeric_hash(coords,'<f8')==diag['coordinates_sha256'],
                                  "Rendering mutated matrix, assignments or coordinates")
                    invariant=dict(config_id=graph['config_id'],length_bp=graph['length_bp'],
                        threshold_bp=graph['threshold_bp'],gap_bp=50000,min_shared=20,min_max_segment_bp=0,
                        **{k:cell[k] for k in ('resolution','n_nodes','n_active','n_isolated','n_edges',
                            'n_assigned','n_communities','n_connected_components_active','n_modes_lambda_near_one',
                            'node_order_sha256','assignment_values_sha256','matrix_file_sha256','coordinates_sha256')},
                        **{k:row[k] for k in ('n_active_unassigned','largest_community','percent_assigned',
                            'median_ari','q25_ari','q75_ari','min_ari','max_ari')},
                        ari_denominator='active_nodes_before_minimum_group_size_filter',ari_seed_pairs=300,
                        unchanged_graph=True,unchanged_partition=True)
                    invariants.append(invariant)
                    figure_records.append(dict(invariant,stem=stem,coordinates_file=diag['coordinates_file'],
                                               coordinate_origin=diag['coordinate_origin']))
            diagnostics.append(diag)
    saved.write_table(output/f"{settings['prefix']}_invariants.tsv",invariants)
    for name,expected in before.items():
        saved.require(saved.sha256(root/name)==expected,"Input changed during rendering")
    saved.require((parameters_json is None or saved.sha256(parameters_json)==parameters_hash)
                  and saved.sha256(core.__file__)==core_hash
                  and saved.sha256(saved.__file__)==validator_hash,"Source code/settings changed during rendering")
    if metadata_record:
        saved.require(saved.sha256(metadata_file)==metadata_record["sha256"],"Metadata changed during rendering")
    coordinate_receipt=None
    if coordinate_source is not None:
        for name,expected in coordinate_source['hashes'].items():
            saved.require(saved.sha256(coordinate_source['root']/name)==expected,"Coordinate source changed during rendering")
        coordinate_receipt=dict(manifest_sha256=coordinates_source_manifest_sha256,
            input_sha256_before=coordinate_source['hashes'],
            input_sha256_after={k:saved.sha256(coordinate_source['root']/k) for k in coordinate_source['hashes']},
            reused_config_ids=coordinate_source['reused'])
    (output/"diagnostics.json").write_text(json.dumps(diagnostics,indent=2,allow_nan=False)+"\n")
    manifest=dict(status="COMPLETE_HISTORICAL_VISUAL_REPRODUCTION",parameters=settings,
        chromosomes=data["chromosomes"],
        status_semantics="Historical spectral/UMAP procedure reused on the chromosomes explicitly listed; not a claim that input results are historical",
        no_reclustering=True,no_new_genotypes=True,no_threshold_changes=True,no_subsampling=True,no_umap_fallback=True,
        bitwise_reproducibility_claimed=False,coordinate_artifacts_persisted=True,
        reproducibility_note="Fixed seed does not uniquely fix degenerate spectral bases or UMAP spectral initialization; saved private coordinates preserve the realized layout.",
        projection_uses_labels=False,metadata=metadata_record,diagnostics=diagnostics,figures=figure_records,
        same_coordinates_within_graph=True,independent_geometry_between_graphs=True,
        coordinates_source=coordinate_receipt,pdf_pages=pdf_pages,
        n_graphs=len(prepared),n_resolutions=len(resolutions),n_cells=len(invariants),
        n_png=len(invariants),n_pdf=len(pdf_pages),n_descriptions=len(invariants),n_private_npz=len(prepared),
        n_projections_computed=sum(d['coordinate_origin']=='computed_once_for_graph' for d in diagnostics),
        n_coordinate_sets_reused=sum(d['coordinate_origin']=='authenticated_saved_coordinates' for d in diagnostics),
        input_sha256_before=before,input_sha256_after={k:saved.sha256(root/k) for k in before},
        parameters_sha256=parameters_hash,source_core_sha256=data["core_sha256"],
        renderer_core_sha256=core_hash,validator_sha256=validator_hash,renderer_sha256=saved.sha256(__file__),
        spectral_function_sha256=spectral_hash,
        static_renderer_function_sha256=hashlib.sha256(inspect.getsource(core._plot_network_static).encode()).hexdigest(),
        versions={p:importlib.metadata.version(p) for p in
                  ("numpy","scipy","matplotlib","umap-learn","scikit-learn","igraph")},
        privacy="No sample IDs exported; private coordinates preserve individual graph information.",
        comparison_pdf="Raster concatenation of unchanged historical-style PNG exports; graph books hold selected resolutions in requested order.",
        outputs_sha256={str(p.relative_to(output)):saved.sha256(p) for p in sorted(output.rglob('*')) if p.is_file()})
    saved.require(len(list(output.glob('*.png')))==manifest['n_png'] and len(list(output.glob('*.pdf')))==manifest['n_pdf']
                  and len(list(output.glob('*.descripcion.md')))==manifest['n_descriptions']
                  and len(list((output/'private').glob('*.npz')))==manifest['n_private_npz'],"Incomplete output inventory")
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',required=True)
    parser.add_argument('--output-dir',required=True)
    parameters=parser.add_mutually_exclusive_group(required=True)
    parameters.add_argument('--parameters-json')
    parameters.add_argument('--parameters-json-text')
    parser.add_argument('--metadata-file')
    parser.add_argument('--metadata-sample-column')
    parser.add_argument('--metadata-color-column',default='finestructure_clusters')
    parser.add_argument('--coordinates-source-dir')
    parser.add_argument('--coordinates-source-manifest-sha256')
    result=run(**vars(parser.parse_args(argv)))
    print(json.dumps({k:result[k] for k in ('status','n_graphs','n_png','n_pdf')}))


if __name__=='__main__':
    main()
