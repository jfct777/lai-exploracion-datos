#!/usr/bin/env python3
"""Funciones compartidas para calcular hashes y validar la reanudación del módulo 23.

No tiene dependencias externas. Los procesos de Nextflow lo reciben como archivo de entrada para que
pueda importarse desde el directorio de trabajo; las ejecuciones directas usan la ruta relativa.
"""
import hashlib
import json
import os
import platform
from importlib import metadata
from pathlib import Path

# Valores de procedencia que significan "no pude calcularlo": el preflight fail-closed los RECHAZA.
_MISSING = {"", "unavailable", "unknown", None}


def sha256_file(path):
    """sha256 de un archivo por chunks (idéntico al de write_stage_manifest.py; fuente única aquí)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def lib_version(dist):
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


_THREAD_LIMITER = None  # mantiene vivo el limite de threadpoolctl mientras el proceso corre


_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def pin_env():
    """FUERZA las env vars de 1-hilo (no `setdefault`: un entorno con OMP=4 heredado romperia el
    determinismo). Es el pin de IMPORT-TIME: OpenBLAS/MKL leen estas vars al cargarse, asi que DEBE
    llamarse ANTES de importar numpy/sklearn. No afecta el n_jobs de joblib (paralelismo de PROCESOS)."""
    for var in _THREAD_VARS:
        os.environ[var] = "1"


def pin_threadpools():
    """Pin REAL en runtime via threadpoolctl (independiente de las env vars y de que BLAS ya se haya
    cargado): limita TODOS los pools nativos (OpenBLAS/MKL/OpenMP) a 1 hilo. Debe llamarse DESPUES de
    importar numpy/sklearn. Guarda el limitador en una global para que no se recolecte. FAIL-CLOSED: si
    threadpoolctl no esta, el pin real NO se puede garantizar -> aborta (el determinismo numerico es
    requisito, no opcional). Devuelve el estado de hilos, que ENTRA a la huella fail-closed."""
    global _THREAD_LIMITER
    try:
        import threadpoolctl
    except Exception as exc:
        raise SystemExit(f"[pin_threadpools] threadpoolctl ausente ({exc}) -> fail-closed: no se puede "
                         "garantizar el pin real de hilos (determinismo numerico).")
    _THREAD_LIMITER = threadpoolctl.threadpool_limits(limits=1)
    return {**{v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "threadpoolctl": "limits=1", "threadpoolctl_version": threadpoolctl.__version__}


def _canonical(obj):
    """Serialización canónica (claves ordenadas) para que el hash de config sea estable."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def compute_fingerprint(code_files, input_files, config, container_sha256, thread_env=None,
                        input_hashes=None):
    """Huella científica fail-closed que namespacea los checkpoints y valida la reanudación.

    Comprueba cuatro elementos para evitar que una reanudación mezcle resultados incompatibles:
      - CÓDIGO: sha256 de rare_bench_cv.py + _common.py (edición durante desarrollo no cambia el
        contenedor → sin esto se mezclarían script-v1 y script-v2).
      - DATOS: sha256 de las 4 entradas (matrix/samples/split/modeling_master). SOLO EL PREFLIGHT las
        HASHEA (única lectura del .npz para calcular su sha256); las tareas HEREDAN esos hashes vía
        `input_hashes` → nadie re-HASHEA el .npz por tarea. Las tareas D/E SÍ cargan
        la matriz para entrenar; lo que no repiten es el hash completo.
      - CONFIG científica: hash canónico del dict `config` (deny-by-default: el llamador excluye
        n_jobs/outdir/checkpoint_dir/mode/set/fold — los operativos).
      - ENTORNO numérico: container_sha256 (subsume versión de librerías) + pin real de hilos BLAS.

    FAIL-CLOSED: si no puede calcular CUALQUIER hash o si container_sha256 es un placeholder de
    'no disponible', aborta. Nunca produce una huella con campos 'unavailable'.
    """
    if container_sha256 in _MISSING:
        raise SystemExit(f"[fingerprint] container_sha256 ausente/placeholder ({container_sha256!r}) "
                         "-> fail-closed: no se puede garantizar el stack numérico.")
    try:
        code = {Path(p).name: sha256_file(p) for p in code_files}
        expected = set(input_files)  # ROLES (claves del dict): estables ante renombres de archivo
        if input_hashes is not None:
            # Tareas: heredan el hash del preflight; NO re-HASHEAN la matriz (aunque la cargan para
            # entrenar). Validar completitud (fail-closed).
            if set(input_hashes) != expected or any(v in _MISSING for v in input_hashes.values()):
                raise SystemExit(f"[fingerprint] input_hashes heredados incompletos/invalidos "
                                 f"({sorted(input_hashes)} vs {sorted(expected)}) -> fail-closed")
            inputs = dict(input_hashes)
        else:
            # Preflight (o monolitico): unico que HASHEA las 4 entradas (lee el .npz para sha256), por ROL.
            inputs = {role: sha256_file(path) for role, path in input_files.items()}
    except OSError as exc:
        raise SystemExit(f"[fingerprint] no se pudo hashear una entrada/código -> fail-closed: {exc}")

    config_hash = sha256_text(_canonical(config))
    fp = {
        "config_hash": config_hash,
        "config": config,
        "code_sha256": code,
        "input_sha256": inputs,
        "container_sha256": container_sha256,
        "thread_env": thread_env or {v: os.environ.get(v) for v in
                                     ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "versions": {
            "python": platform.python_version(),
            "numpy": lib_version("numpy"),
            "scipy": lib_version("scipy"),
            "scikit_learn": lib_version("scikit-learn"),
        },
    }
    # ID compacto que namespacea el checkpoint: cubre config + datos + código + entorno.
    # thread_env ENTRA al hash (no depende del contenedor: es del entorno de lanzamiento; sin el pin
    # BLAS correcto hay no-determinismo numérico). versions queda como provenance legible pero fuera
    # del hash: el container_sha256 ya las subsume.
    fp["fingerprint_id"] = sha256_text(_canonical({
        "config_hash": config_hash,
        "code": code,
        "inputs": inputs,
        "container": container_sha256,
        "thread_env": fp["thread_env"],
    }))
    return fp


def assert_fingerprint_matches(reference, current, context=""):
    """Fail-closed: aborta si dos huellas no coinciden en su fingerprint_id. Cada tarea (abc/fold)
    re-computa su huella y la contrasta con la del preflight para cazar drift entre preflight y tarea."""
    ref_id = reference.get("fingerprint_id")
    cur_id = current.get("fingerprint_id")
    if ref_id != cur_id:
        diffs = []
        for k in ("config_hash", "code_sha256", "input_sha256", "container_sha256"):
            if reference.get(k) != current.get(k):
                diffs.append(k)
        raise SystemExit(f"[fingerprint] huella no coincide {context} -> fail-closed. "
                         f"Difieren: {diffs or ['fingerprint_id']}. "
                         "Un cambio de código/datos/config/entorno invalida la reutilización.")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    """Escritura simple. En el modelo Nextflow-particionado la atomicidad la da el orquestador: una
    tarea que muere NO publica su output, así que el agregador nunca ve un JSON parcial. No se
    reimplementa el checkpoint atómico (esa era la vía descartada)."""
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False))
