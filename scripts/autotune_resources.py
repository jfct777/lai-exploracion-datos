#!/usr/bin/env python3
"""Ajusta los recursos del pipeline Nextflow de control de calidad de DNABR.

Detecta si la ejecución usa laptop, WSL o Slurm y genera una configuración adaptada al hardware
disponible.

Referencias:
- bcftools: https://samtools.github.io/bcftools/howtos/scaling.html
- plink2: https://www.cog-genomics.org/plink/2.0/parallel
"""

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    """Define y devuelve los argumentos de línea de comandos."""
    p = argparse.ArgumentParser(
        description="Ajusta la configuración de recursos de Nextflow para DNABR"
    )
    p.add_argument(
        "--mode",
        required=True,
        choices=["laptop", "slurm"],
        help="Entorno de ejecución: laptop local o slurm en HPC",
    )
    p.add_argument(
        "--out",
        default="conf/auto_resources.config",
        help="Archivo de configuración de salida",
    )
    p.add_argument(
        "--safety-margin",
        type=float,
        default=0.8,
        help="Margen de seguridad para asignar recursos, entre 0.0 y 1.0",
    )
    p.add_argument(
        "--max-forks-heavy",
        type=int,
        default=None,
        help="Máximo de procesos pesados en paralelo; por defecto se calcula automáticamente",
    )
    p.add_argument(
        "--partition",
        default="cpu",
        help="Partición de Slurm que se consulta con sinfo",
    )
    p.add_argument(
        "--cpus",
        type=int,
        default=None,
        help="Sobrescribe la cantidad de CPU detectada",
    )
    p.add_argument(
        "--mem-gb",
        type=float,
        default=None,
        help="Sobrescribe la memoria total detectada en GB",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Muestra los recursos calculados sin escribir el archivo de configuración",
    )
    return p.parse_args()


def _sinfo_query(partition):
    """Consulta el mínimo de CPU y memoria entre los nodos de una partición.

    Usa el mínimo para que la configuración sea válida en cualquier nodo. Devuelve
    (cpus, mem_mb) o (None, None) si la consulta falla.
    """
    try:
        # %c indica CPU por nodo y %m memoria por nodo en MB.
        result = subprocess.run(
            ["sinfo", "--noheader", "-p", partition, "-o", "%c %m"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=10,
        )
        if result.returncode != 0:
            return None, None

        min_cpus, min_mem_mb = None, None
        for line in result.stdout.strip().splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            # sinfo puede añadir '+' para indicar "o más".
            cpus = int(parts[0].rstrip("+"))
            mem_mb = int(parts[1].rstrip("+"))
            if min_cpus is None or cpus < min_cpus:
                min_cpus = cpus
            if min_mem_mb is None or mem_mb < min_mem_mb:
                min_mem_mb = mem_mb
        return min_cpus, min_mem_mb
    except Exception:
        return None, None


def get_available_cpus(partition=None):
    """Detecta la cantidad de CPU disponibles.

    Prioridad:
    1. sinfo si se indicó una partición.
    2. SLURM_CPUS_PER_TASK o SLURM_JOB_CPUS_PER_NODE dentro de un job.
    3. os.cpu_count() como alternativa local.
    """
    if partition:
        cpus, _ = _sinfo_query(partition)
        if cpus is not None:
            return cpus

    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK") or os.getenv("SLURM_JOB_CPUS_PER_NODE")
    if slurm_cpus:
        return int(slurm_cpus)

    try:
        return os.cpu_count() or 4
    except Exception:
        return 4


def get_total_memory_gb(partition=None):
    """Detecta la memoria total disponible en GB.

    Prioridad:
    1. sinfo, si se indicó una partición.
    2. SLURM_MEM_PER_NODE o SLURM_MEM_PER_CPU dentro de un job.
    3. MemTotal de /proc/meminfo como alternativa local.
    """
    if partition:
        _, mem_mb = _sinfo_query(partition)
        if mem_mb is not None:
            return mem_mb / 1024.0

    slurm_mem_per_node = os.getenv("SLURM_MEM_PER_NODE")
    slurm_mem_per_cpu = os.getenv("SLURM_MEM_PER_CPU")

    if slurm_mem_per_node:
        return int(slurm_mem_per_node) / 1024.0

    if slurm_mem_per_cpu:
        cpus = get_available_cpus()
        return (int(slurm_mem_per_cpu) * cpus) / 1024.0

    # Como alternativa local, usa MemTotal de /proc/meminfo.
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_kb = int(line.split()[1])
                    return mem_kb / (1024.0 * 1024.0)
    except Exception:
        pass

    # Valor conservador si no fue posible detectar la memoria.
    return 8.0


def _ceil_mem(val):
    """Redondea la memoria hacia arriba para evitar una asignación insuficiente."""
    return math.ceil(val)


def compute_module_resources(total_cpus, total_mem_gb, safety_margin, mode):
    """
    Calcula los recursos de cada módulo según el hardware disponible.
    
    Estrategia:
    - bcftools usa entre dos y cuatro hilos porque está limitado por I/O.
    - plink2 escala con las CPU disponibles.
    - La cantidad de hilos nunca supera las CPU asignadas.
    - La memoria depende del costo de cada módulo y se redondea hacia arriba.
    - maxForks limita los procesos pesados en laptops.
    
    Devuelve un diccionario con esta estructura:
    {
        'module_name': {
            'cpus': int,
            'memory': str (e.g., '4 GB'),
            'threads': int,
            'maxForks': int or None
        }
    }
    """
    safe_cpus = max(1, int(total_cpus * safety_margin))
    safe_mem_gb = total_mem_gb * safety_margin
    
    # En laptop se limita el paralelismo de forma conservadora.
    if mode == "laptop":
        # bcftools está limitado por I/O y usa como máximo cuatro hilos.
        bcftools_cpus = min(4, max(2, safe_cpus // 4))
        bcftools_mem = max(2, safe_mem_gb / 8)

        # plink2 puede usar hasta seis CPU en una laptop de 16 núcleos.
        plink_cpus = min(6, max(2, safe_cpus // 2))
        plink_mem = max(4, safe_mem_gb / 4)
        plink_mem_heavy = max(6, safe_mem_gb / 3)  # El cálculo de LD requiere más memoria.

        python_cpus = 1
        python_mem = max(2, safe_mem_gb / 8)
        heavy_maxforks = 2
    else:  # En Slurm se dimensiona para unas 2700 muestras WGS por cromosoma.
        # bcftools procesa registros VCF anchos y sus buffers de compresión consumen memoria.
        bcftools_cpus = min(4, max(2, safe_cpus // 4))
        bcftools_mem = max(4, min(16, safe_mem_gb / 8))  # cap at 16 GB

        # La matriz de genotipos de chr1 ocupa cerca de 5 GB y requiere memoria de trabajo adicional.
        plink_cpus = min(8, max(2, safe_cpus // 2))
        plink_mem = max(12, min(32, safe_mem_gb / 5))  # cap at 32 GB
        plink_mem_heavy = max(24, min(64, safe_mem_gb / 3))  # LD --r2-unphased, cap 64 GB
        plink_mem_tag = max(16, min(48, safe_mem_gb / 4))  # indep-pairwise, cap 48 GB

        # Los scripts de Python procesan millones de variantes con pandas.
        python_cpus = 1
        python_mem = max(4, min(16, safe_mem_gb / 8))  # cap at 16 GB
        heavy_maxforks = None  # En HPC no se fija un límite adicional.
    
    # Los hilos nunca deben superar las CPU asignadas.
    bcftools_threads = min(bcftools_cpus, 4, max(1, safe_cpus // 4))
    plink_threads = plink_cpus  # threads == cpus for plink2
    
    resources = {
        "preprocess_norm_leftalign": {
            "cpus": bcftools_cpus,
            "memory": f"{_ceil_mem(bcftools_mem)} GB",
            "threads": bcftools_threads,
            "maxForks": None,
        },
        "preprocess_filter_snv_biallelic_pass": {
            "cpus": bcftools_cpus,
            "memory": f"{_ceil_mem(bcftools_mem)} GB",
            "threads": bcftools_threads,
            "maxForks": None,
        },
        "qc_bcftools_stats": {
            "cpus": bcftools_cpus,
            "memory": f"{_ceil_mem(bcftools_mem)} GB",
            "threads": bcftools_threads,
            "maxForks": None,
        },
        "qc_plink_make_pgen": {
            "cpus": plink_cpus,
            "memory": f"{_ceil_mem(plink_mem)} GB",
            "threads": plink_threads,
            "maxForks": None,
        },
        "qc_plink_missing_het": {
            "cpus": plink_cpus,
            "memory": f"{_ceil_mem(plink_mem)} GB",
            "threads": plink_threads,
            "maxForks": heavy_maxforks,
        },
        "qc_aggregate_report_py": {
            "cpus": 1,
            "memory": f"{max(2, min(8, _ceil_mem(safe_mem_gb / 12)))} GB",
            "threads": 1,
            "maxForks": 1,
        },
        # Módulos 07 a 11 de genética de poblaciones.
        "sfs_from_filtered_vcf": {
            "cpus": python_cpus,
            "memory": f"{_ceil_mem(python_mem)} GB",
            "threads": 1,
            "maxForks": None,
        },
        "build_ancestral_tsv_from_fasta": {
            "cpus": python_cpus,
            "memory": f"{_ceil_mem(python_mem)} GB",
            "threads": 1,
            "maxForks": None,
        },
        "daf_dsfs_from_ancestral_tsv": {
            "cpus": python_cpus,
            "memory": f"{_ceil_mem(python_mem)} GB",
            "threads": 1,
            "maxForks": None,
        },
        "ld_decay_from_pgen": {
            "cpus": plink_cpus,
            "memory": f"{_ceil_mem(plink_mem_heavy)} GB",
            "threads": plink_threads,
            "maxForks": heavy_maxforks,
        },
        "tag_snps_from_pgen": {
            "cpus": plink_cpus,
            "memory": f"{_ceil_mem(plink_mem_tag if mode == 'slurm' else plink_mem)} GB",
            "threads": plink_threads,
            "maxForks": heavy_maxforks,
        },
    }
    
    return resources


def generate_nextflow_config(resources, out_path):
    """Genera el archivo de Nextflow con los recursos calculados."""
    process_names = {
        "preprocess_norm_leftalign": "PREPROCESS_NORM_LEFTALIGN",
        "preprocess_filter_snv_biallelic_pass": "PREPROCESS_FILTER_SNV_BIALLELIC_PASS",
        "qc_bcftools_stats": "QC_BCFTOOLS_STATS",
        "qc_plink_make_pgen": "QC_PLINK_MAKE_PGEN",
        "qc_plink_missing_het": "QC_PLINK_MISSING_HET",
        "qc_aggregate_report_py": "QC_AGGREGATE_REPORT_PY",
        # Módulos 07 a 11.
        "sfs_from_filtered_vcf": "SFS_FROM_FILTERED_VCF",
        "build_ancestral_tsv_from_fasta": "BUILD_ANCESTRAL_TSV_FROM_FASTA",
        "daf_dsfs_from_ancestral_tsv": "DAF_DSFS_FROM_ANCESTRAL_TSV",
        "ld_decay_from_pgen": "LD_DECAY_FROM_PGEN",
        "tag_snps_from_pgen": "TAG_SNPS_FROM_PGEN",
    }
    
    lines = [
        "// Configuración de recursos generada automáticamente",
        "// Este archivo se actualiza con scripts/autotune_resources.py",
        "//",
        "// Uso:",
        "//   nextflow run main.nf -profile local_docker -c conf/auto_resources.config ...",
        "",
        "params {",
        "  resources {",
    ]
    
    for module_key, res in resources.items():
        lines.append(f"    {module_key} {{")
        lines.append(f"      cpus = {res['cpus']}")
        lines.append(f"      memory = '{res['memory']}'")
        lines.append(f"      threads = {res['threads']}")
        if res['maxForks'] is not None:
            lines.append(f"      maxForks = {res['maxForks']}")
        else:
            lines.append(f"      maxForks = null")
        lines.append(f"    }}")
        lines.append("")
    
    lines.append("  }")
    lines.append("}")
    lines.append("")
    lines.append("process {")
    
    for module_key, proc_name in process_names.items():
        res = resources[module_key]
        lines.append(f"  withName: '{proc_name}' {{")
        lines.append(f"    cpus = params.resources.{module_key}.cpus")
        lines.append(f"    memory = params.resources.{module_key}.memory")
        if res['maxForks'] is not None:
            lines.append(f"    maxForks = params.resources.{module_key}.maxForks")
        lines.append(f"  }}")
        lines.append("")
    
    lines.append("}")
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"✓ Configuración de recursos generada: {out_path}")


# Módulos pesados de plink2 cuyo maxForks se ajusta en conjunto.
HEAVY_MODULES = ["qc_plink_missing_het", "ld_decay_from_pgen", "tag_snps_from_pgen"]


def main():
    """Estima recursos desde trazas y actualiza la configuración generada."""
    args = parse_args()
    
    partition = args.partition if args.mode == "slurm" else None
    total_cpus = args.cpus if args.cpus is not None else get_available_cpus(partition)
    total_mem_gb = args.mem_gb if args.mem_gb is not None else get_total_memory_gb(partition)
    source = "sinfo" if (partition and args.cpus is None) else ("override" if args.cpus is not None else "local")
    
    print(f"Entorno detectado: {args.mode}")
    print(f"  CPU totales: {total_cpus} (fuente: {source})")
    print(f"  Memoria total: {total_mem_gb:.1f} GB (fuente: {source})")
    print(f"  Margen de seguridad: {args.safety_margin * 100:.0f}%")
    print()
    
    resources = compute_module_resources(
        total_cpus, total_mem_gb, args.safety_margin, args.mode
    )
    
    # Aplica --max-forks-heavy a los tres módulos pesados.
    if args.max_forks_heavy is not None:
        for mod in HEAVY_MODULES:
            resources[mod]["maxForks"] = args.max_forks_heavy
    
    print("Recursos calculados:")
    for module_key, res in resources.items():
        print(f"  {module_key}:")
        print(f"    cpus={res['cpus']}, memory={res['memory']}, threads={res['threads']}, maxForks={res['maxForks']}")
    print()
    
    if args.dry_run:
        print("--dry-run: no se escribió la configuración.")
    else:
        generate_nextflow_config(resources, args.out)
        print()
        print("Uso:")
        print(f"  nextflow run main.nf -profile slurm_singularity -c {args.out} ...")


if __name__ == "__main__":
    main()
