#!/usr/bin/env python3
"""Genera el manifest.json reproducible de una etapa.

Cada manifiesto contiene: etapa, timestamp, commit git, contenedor (path + sha256), versiones
(python + librerias del contenedor + nextflow), parametros efectivos, y sha256 de ENTRADAS y
SALIDAS, mas un puntero a run_provenance.json. NO recomputa ni reabre nada: solo hashea los
archivos que se le pasan (que el proceso ya produjo/consumio en el work dir).

La procedencia de ARTEFACTO (commit, version Nextflow, path + sha256 del contenedor) se computa
UNA sola vez en el head node (main.nf) porque el .sif NO es visible desde dentro del contenedor,
y llega aqui como JSON en base64 (--provenance-b64) para evitar problemas de quoting. Las versiones
de librerias y de python SI se leen aqui dentro, porque este script corre dentro del mismo
contenedor que produjo los artefactos.

El comando de Nextflow se guarda en run_provenance.json porque cambia al usar `-resume`. Dejarlo
fuera de las entradas de cada etapa permite reutilizar correctamente los procesos almacenados.
"""
import argparse
import base64
import hashlib
import json
import platform
from importlib import metadata
from pathlib import Path


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _lib_version(dist):
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, help="nombre de la etapa (proceso Nextflow)")
    ap.add_argument("--input", action="append", default=[], type=Path,
                    help="archivo de ENTRADA a hashear (repetible)")
    ap.add_argument("--output", action="append", default=[], type=Path,
                    help="archivo de SALIDA a hashear (repetible)")
    ap.add_argument("--params-json", default="{}", help="params efectivos de la etapa, como JSON")
    ap.add_argument("--provenance-b64", default="",
                    help="JSON en base64 con {git_commit, nextflow_command, nextflow_version, "
                         "container_path, container_sha256}, computado en el head node")
    ap.add_argument("--stamp", default="", help="timestamp inyectado por el proceso (TZ del host)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    try:
        params = json.loads(args.params_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--params-json no es JSON valido: {exc}")

    prov = {}
    if args.provenance_b64:
        try:
            prov = json.loads(base64.b64decode(args.provenance_b64).decode("utf-8"))
        except (ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"--provenance-b64 invalido: {exc}")

    manifest = {
        "stage": args.stage,
        "generated": args.stamp,
        "git_commit": prov.get("git_commit"),
        "container": {
            "path": prov.get("container_path"),
            "sha256": prov.get("container_sha256"),
        },
        "versions": {
            "python": platform.python_version(),
            "nextflow": prov.get("nextflow_version"),
            "pandas": _lib_version("pandas"),
            "numpy": _lib_version("numpy"),
            "scikit_learn": _lib_version("scikit-learn"),
        },
        "params": params,
        "inputs": {p.name: _sha256(p) for p in args.input},
        "sha256": {p.name: _sha256(p) for p in args.output},
        # El comando de Nextflow cambia con -resume y se guarda fuera del cache de cada etapa.
        "run_provenance_file": "../run_provenance.json",
    }
    args.out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
