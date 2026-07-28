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

# Valores que indican que la procedencia no pudo calcularse; preflight los rechaza.
_MISSING = {"", "unavailable", "unknown", None}


def sha256_file(path):
    """sha256 de un archivo por chunks (idéntico al de write_stage_manifest.py; fuente única aquí)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    """Calcula sha256 sobre texto codificado en UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def lib_version(dist):
    """Devuelve la versión instalada de una distribución o None si no existe."""
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


_THREAD_LIMITER = None  # mantiene vivo el limite de threadpoolctl mientras el proceso corre


_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def pin_env():
    """Fija las variables de entorno para que cada biblioteca use un hilo.

    No usa `setdefault` porque un valor heredado como OMP=4 rompería el determinismo. OpenBLAS y MKL
    leen estas variables al cargarse, así que debe llamarse antes de importar numpy y sklearn. No
    modifica n_jobs de joblib ni el paralelismo entre procesos.
    """
    for var in _THREAD_VARS:
        os.environ[var] = "1"


def pin_threadpools():
    """Aplica en runtime el límite de hilos mediante threadpoolctl.

    Limita los pools nativos de OpenBLAS, MKL y OpenMP a un hilo aunque BLAS ya esté cargado.
    Debe llamarse después de importar numpy y sklearn. El limitador se conserva en una variable
    global para evitar que el recolector lo libere. Si threadpoolctl no está disponible, el proceso
    termina porque no puede garantizar el determinismo numérico.
    """
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
    """Calcula la huella científica que identifica los checkpoints de una ejecución.

    Comprueba cuatro elementos para evitar que una reanudación mezcle resultados incompatibles:
      - Código: sha256 de rare_bench_cv.py y _common.py. Una edición durante el desarrollo no cambia
        el contenedor; sin esta huella podrían mezclarse dos versiones de los scripts.
      - Datos: sha256 de las cuatro entradas. Preflight calcula las huellas una vez y las tareas las
        reciben mediante `input_hashes`. Las tareas D y E cargan la matriz para entrenar, pero no
        vuelven a leerla solo para calcular su hash.
      - Configuración científica: hash canónico del diccionario `config`. El llamador excluye los
        parámetros operativos n_jobs, outdir, checkpoint_dir, mode, set y fold.
      - Entorno numérico: sha256 del contenedor y límite efectivo de hilos de BLAS.

    Si algún hash no puede calcularse o el sha256 del contenedor no está disponible, la ejecución
    termina sin producir una huella incompleta.
    """
    if container_sha256 in _MISSING:
        raise SystemExit(f"[fingerprint] container_sha256 ausente/placeholder ({container_sha256!r}) "
                         "-> fail-closed: no se puede garantizar el stack numérico.")
    try:
        code = {Path(p).name: sha256_file(p) for p in code_files}
        expected = set(input_files)  # Las claves describen la función del archivo y no dependen de su nombre.
        if input_hashes is not None:
            # Las tareas heredan las huellas de preflight y no vuelven a calcularlas.
            if set(input_hashes) != expected or any(v in _MISSING for v in input_hashes.values()):
                raise SystemExit(f"[fingerprint] input_hashes heredados incompletos/invalidos "
                                 f"({sorted(input_hashes)} vs {sorted(expected)}) -> fail-closed")
            inputs = dict(input_hashes)
        else:
            # Preflight y el modo monolítico calculan las huellas de las cuatro entradas por función.
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
    # thread_env forma parte del hash porque depende del entorno de lanzamiento; sin el límite
    # correcto de BLAS habría variación numérica. versions queda como procedencia legible, pero fuera
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
    """Carga un archivo JSON desde la ruta indicada."""
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    """Escritura simple. En el modelo Nextflow-particionado la atomicidad la da el orquestador: una
    una tarea que termina con error no publica su salida, así que el agregador no recibe JSON parciales. No se
    reimplementa el checkpoint atómico (esa era la vía descartada)."""
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False))
