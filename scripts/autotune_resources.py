#!/usr/bin/env python3
"""
Auto-tune resource allocation for DNABR QC Nextflow pipeline.

Detects environment (laptop/WSL or HPC/Slurm) and generates optimized
resource configuration for each module based on available hardware.

References:
- bcftools threading: https://samtools.github.io/bcftools/howtos/scaling.html
  (--threads only helps with I/O compression, not processing)
- plink2 threading: https://www.cog-genomics.org/plink/2.0/parallel
  (--threads provides real parallelization for computation)
"""

import argparse
import math
import os
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Auto-tune Nextflow resource configuration for DNABR QC pipeline"
    )
    p.add_argument(
        "--mode",
        required=True,
        choices=["laptop", "slurm"],
        help="Environment mode: laptop (local) or slurm (HPC)",
    )
    p.add_argument(
        "--out",
        default="conf/auto_resources.config",
        help="Output Nextflow config file (default: conf/auto_resources.config)",
    )
    p.add_argument(
        "--safety-margin",
        type=float,
        default=0.8,
        help="Safety margin for resource allocation (0.0-1.0, default: 0.8)",
    )
    p.add_argument(
        "--max-forks-heavy",
        type=int,
        default=None,
        help="Max parallel forks for heavy processes (default: auto)",
    )
    p.add_argument(
        "--cpus",
        type=int,
        default=None,
        help="Override detected CPU count",
    )
    p.add_argument(
        "--mem-gb",
        type=float,
        default=None,
        help="Override detected total memory (GB)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print computed resources without writing config file",
    )
    return p.parse_args()


def get_available_cpus():
    """Detect available CPUs from environment or system."""
    # Check Slurm environment variables first
    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK") or os.getenv("SLURM_JOB_CPUS_PER_NODE")
    if slurm_cpus:
        return int(slurm_cpus)
    
    # Fallback to system detection
    try:
        return os.cpu_count() or 4
    except Exception:
        return 4


def get_available_memory_gb():
    """Detect available memory in GB from environment or system."""
    # Check Slurm environment variables first
    slurm_mem_per_node = os.getenv("SLURM_MEM_PER_NODE")
    slurm_mem_per_cpu = os.getenv("SLURM_MEM_PER_CPU")
    
    if slurm_mem_per_node:
        # Usually in MB
        return int(slurm_mem_per_node) / 1024.0
    
    if slurm_mem_per_cpu:
        cpus = get_available_cpus()
        return (int(slurm_mem_per_cpu) * cpus) / 1024.0
    
    # Use MemTotal for deterministic results (safety_margin handles OS overhead)
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_kb = int(line.split()[1])
                    return mem_kb / (1024.0 * 1024.0)
    except Exception:
        pass
    
    # Conservative default
    return 8.0


def compute_module_resources(total_cpus, total_mem_gb, safety_margin, mode):
    """
    Compute resource allocation for each module based on available hardware.
    
    Strategy:
    - bcftools modules: threads capped at 2-4 (I/O bound, not CPU bound)
    - plink2 modules: threads scale with available CPUs (real parallelization)
    - Memory: proportional to module intensity
    - maxForks: limit heavy processes on laptops
    
    Returns dict with structure:
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
    
    # bcftools threading: --threads only helps bgzf I/O, cap at 2-3
    bcftools_threads = min(3, max(1, safe_cpus // 4))
    
    # plink2 threading: real parallelization, scale with cores
    plink_threads = min(16, max(2, safe_cpus // 2))
    
    if mode == "laptop":
        # bcftools: I/O bound → low cpus, let Nextflow schedule multiple tasks
        bcftools_cpus = min(2, safe_cpus)
        bcftools_mem = max(2, math.ceil(safe_mem_gb / 6))
        # plink2: CPU bound → give more threads per task
        plink_cpus = max(2, min(8, safe_cpus // 2))
        plink_threads = plink_cpus  # match cpus for plink
        plink_mem = max(4, math.ceil(safe_mem_gb / 4))
        plink_mem_heavy = max(6, math.ceil(safe_mem_gb / 3))
        # python scripts: lightweight
        python_cpus = 1
        python_mem = max(2, math.ceil(safe_mem_gb / 8))
        # heavy forks: ensure cpus*forks <= safe_cpus
        heavy_maxforks = max(1, safe_cpus // plink_cpus)
    else:  # slurm mode: more aggressive
        bcftools_cpus = min(4, max(1, safe_cpus // 4))
        bcftools_mem = max(4, math.ceil(safe_mem_gb / 6))
        plink_cpus = min(8, max(2, safe_cpus // 2))
        plink_threads = plink_cpus
        plink_mem = max(8, math.ceil(safe_mem_gb / 3))
        plink_mem_heavy = max(16, math.ceil(safe_mem_gb / 2))
        python_cpus = min(2, max(1, safe_cpus // 4))
        python_mem = max(4, math.ceil(safe_mem_gb / 6))
        heavy_maxforks = None  # no limit on HPC
    
    resources = {
        "preprocess_norm_leftalign": {
            "cpus": bcftools_cpus,
            "memory": f"{bcftools_mem} GB",
            "threads": bcftools_threads,
            "maxForks": None,
        },
        "preprocess_filter_snv_biallelic_pass": {
            "cpus": bcftools_cpus,
            "memory": f"{bcftools_mem} GB",
            "threads": bcftools_threads,
            "maxForks": None,
        },
        "qc_bcftools_stats": {
            "cpus": bcftools_cpus,
            "memory": f"{bcftools_mem} GB",
            "threads": bcftools_threads,
            "maxForks": None,
        },
        "qc_plink_make_pgen": {
            "cpus": plink_cpus,
            "memory": f"{plink_mem} GB",
            "threads": plink_threads,
            "maxForks": None,
        },
        "qc_plink_missing_het": {
            "cpus": plink_cpus,
            "memory": f"{plink_mem} GB",
            "threads": plink_threads,
            "maxForks": heavy_maxforks,
        },
        "qc_aggregate_report_py": {
            "cpus": 1,
            "memory": f"{max(2, math.ceil(safe_mem_gb / 8))} GB",
            "threads": 1,
            "maxForks": 1,
        },
        # Modules 07-11: downstream population-genetics
        "sfs_from_filtered_vcf": {
            "cpus": python_cpus,
            "memory": f"{python_mem} GB",
            "threads": 1,
            "maxForks": None,
        },
        "build_ancestral_tsv_from_fasta": {
            "cpus": python_cpus,
            "memory": f"{python_mem} GB",
            "threads": 1,
            "maxForks": None,
        },
        "daf_dsfs_from_ancestral_tsv": {
            "cpus": python_cpus,
            "memory": f"{python_mem} GB",
            "threads": 1,
            "maxForks": None,
        },
        "ld_decay_from_pgen": {
            "cpus": plink_cpus,
            "memory": f"{plink_mem_heavy} GB",
            "threads": plink_threads,
            "maxForks": heavy_maxforks,
        },
        "tag_snps_from_pgen": {
            "cpus": plink_cpus,
            "memory": f"{plink_mem} GB",
            "threads": plink_threads,
            "maxForks": heavy_maxforks,
        },
    }
    
    return resources


def generate_nextflow_config(resources, out_path):
    """Generate Nextflow config file with computed resources."""
    process_names = {
        "preprocess_norm_leftalign": "PREPROCESS_NORM_LEFTALIGN",
        "preprocess_filter_snv_biallelic_pass": "PREPROCESS_FILTER_SNV_BIALLELIC_PASS",
        "qc_bcftools_stats": "QC_BCFTOOLS_STATS",
        "qc_plink_make_pgen": "QC_PLINK_MAKE_PGEN",
        "qc_plink_missing_het": "QC_PLINK_MISSING_HET",
        "qc_aggregate_report_py": "QC_AGGREGATE_REPORT_PY",
        # Modules 07-11
        "sfs_from_filtered_vcf": "SFS_FROM_FILTERED_VCF",
        "build_ancestral_tsv_from_fasta": "BUILD_ANCESTRAL_TSV_FROM_FASTA",
        "daf_dsfs_from_ancestral_tsv": "DAF_DSFS_FROM_ANCESTRAL_TSV",
        "ld_decay_from_pgen": "LD_DECAY_FROM_PGEN",
        "tag_snps_from_pgen": "TAG_SNPS_FROM_PGEN",
    }
    
    lines = [
        "// Auto-generated resource configuration",
        "// Este archivo se actualiza con scripts/autotune_resources.py",
        "//",
        "// To use this config:",
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
    
    print(f"✓ Generated resource config: {out_path}")


def main():
    args = parse_args()
    
    total_cpus = args.cpus if args.cpus else get_available_cpus()
    total_mem_gb = args.mem_gb if args.mem_gb else get_available_memory_gb()
    
    print(f"Detected environment: {args.mode}")
    print(f"  Total CPUs: {total_cpus}{' (override)' if args.cpus else ''}")
    print(f"  Total Memory: {total_mem_gb:.1f} GB{' (override)' if args.mem_gb else ''}")
    print(f"  Safe CPUs: {max(1, int(total_cpus * args.safety_margin))}")
    print(f"  Safe Memory: {total_mem_gb * args.safety_margin:.1f} GB")
    print(f"  Safety margin: {args.safety_margin * 100:.0f}%")
    print()
    
    resources = compute_module_resources(
        total_cpus, total_mem_gb, args.safety_margin, args.mode
    )
    
    if args.max_forks_heavy is not None:
        for key in ("qc_plink_missing_het", "ld_decay_from_pgen", "tag_snps_from_pgen"):
            if key in resources:
                resources[key]["maxForks"] = args.max_forks_heavy
    
    print("Computed resource allocation:")
    for module_key, res in resources.items():
        print(f"  {module_key}:")
        print(f"    cpus={res['cpus']}, memory={res['memory']}, threads={res['threads']}, maxForks={res['maxForks']}")
    print()
    
    if args.dry_run:
        print("(dry-run mode — no config file written)")
        return
    
    generate_nextflow_config(resources, args.out)
    print()
    print("Usage:")
    print(f"  nextflow run main.nf -profile local_docker -c {args.out} ...")


if __name__ == "__main__":
    main()
