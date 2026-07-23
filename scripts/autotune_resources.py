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
import subprocess
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
        help="Max parallel forks for heavy processes: qc_plink_missing_het, "
             "ld_decay_from_pgen, tag_snps_from_pgen (default: auto)",
    )
    p.add_argument(
        "--partition",
        default="cpu",
        help="Slurm partition to query with sinfo (default: cpu). "
             "Only used in slurm mode.",
    )
    p.add_argument(
        "--cpus",
        type=int,
        default=None,
        help="Override detected CPU count (useful for testing or non-standard environments)",
    )
    p.add_argument(
        "--mem-gb",
        type=float,
        default=None,
        help="Override detected total memory in GB",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview computed resources without writing the config file",
    )
    return p.parse_args()


def _sinfo_query(partition):
    """Query sinfo for the minimum CPUs and memory across nodes in a partition.

    Uses the *minimum* across nodes so the config is safe for every node in
    the partition.  Returns (cpus, mem_mb) or (None, None) on failure.
    """
    try:
        # %c = CPUs per node,  %m = memory per node in MB
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
            # sinfo may append '+' to indicate "or more"
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
    """Detect available CPUs.

    Priority:
    1. sinfo (if partition given — works from login nodes)
    2. SLURM_CPUS_PER_TASK / SLURM_JOB_CPUS_PER_NODE (inside a job)
    3. os.cpu_count() (local fallback)
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
    """Detect total memory in GB.

    Priority:
    1. sinfo (if partition given — works from login nodes)
    2. SLURM_MEM_PER_NODE / SLURM_MEM_PER_CPU (inside a job)
    3. /proc/meminfo MemTotal (local fallback)
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

    # Fallback to /proc/meminfo (Linux/WSL) — use MemTotal for determinism
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


def _ceil_mem(val):
    """Round memory up to next integer GB using math.ceil to avoid under-allocation."""
    return math.ceil(val)


def compute_module_resources(total_cpus, total_mem_gb, safety_margin, mode):
    """
    Compute resource allocation for each module based on available hardware.
    
    Strategy:
    - bcftools modules: threads capped at 2-4 (I/O bound, not CPU bound)
    - plink2 modules: threads scale with available CPUs (real parallelization)
    - threads always <= cpus to prevent Slurm oversubscription
    - Memory: proportional to module intensity, rounded up with math.ceil
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
    
    # Laptop mode: more conservative, limit concurrency
    if mode == "laptop":
        # bcftools: I/O-bound, cap at 4 threads (diminishing returns)
        bcftools_cpus = min(4, max(2, safe_cpus // 4))
        bcftools_mem = max(2, safe_mem_gb / 8)

        # plink2: real parallelization — scale up to 6 cpus for 16-core laptops
        plink_cpus = min(6, max(2, safe_cpus // 2))
        plink_mem = max(4, safe_mem_gb / 4)
        plink_mem_heavy = max(6, safe_mem_gb / 3)  # LD decay needs more memory

        python_cpus = 1
        python_mem = max(2, safe_mem_gb / 8)
        heavy_maxforks = 2
    else:  # slurm mode — sized for ~2700 WGS samples per chromosome
        # bcftools: streaming I/O but wide VCF records (one GT per sample);
        # multi-threaded compression buffers add up.  Cap 16 GB.
        bcftools_cpus = min(4, max(2, safe_cpus // 4))
        bcftools_mem = max(4, min(16, safe_mem_gb / 8))  # cap at 16 GB

        # plink2: genotype matrix ~5 GB for chr1 (7M var × 2723 samples)
        # plus working memory.  Cap at 32 GB standard, 64 GB for LD.
        plink_cpus = min(8, max(2, safe_cpus // 2))
        plink_mem = max(12, min(32, safe_mem_gb / 5))  # cap at 32 GB
        plink_mem_heavy = max(24, min(64, safe_mem_gb / 3))  # LD --r2-unphased, cap 64 GB
        plink_mem_tag = max(16, min(48, safe_mem_gb / 4))  # indep-pairwise, cap 48 GB

        # Python scripts: pandas on millions of variants × 2723 samples
        python_cpus = 1
        python_mem = max(4, min(16, safe_mem_gb / 8))  # cap at 16 GB
        heavy_maxforks = None  # no limit on HPC
    
    # INVARIANT: threads must always be <= cpus to avoid oversubscription
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
        # Modules 07-11: downstream population-genetics
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


# Heavy plink2 modules whose maxForks should be overridden together
HEAVY_MODULES = ["qc_plink_missing_het", "ld_decay_from_pgen", "tag_snps_from_pgen"]


def main():
    args = parse_args()
    
    partition = args.partition if args.mode == "slurm" else None
    total_cpus = args.cpus if args.cpus is not None else get_available_cpus(partition)
    total_mem_gb = args.mem_gb if args.mem_gb is not None else get_total_memory_gb(partition)
    source = "sinfo" if (partition and args.cpus is None) else ("override" if args.cpus is not None else "local")
    
    print(f"Detected environment: {args.mode}")
    print(f"  Total CPUs: {total_cpus} (source: {source})")
    print(f"  Total Memory: {total_mem_gb:.1f} GB (source: {source})")
    print(f"  Safety margin: {args.safety_margin * 100:.0f}%")
    print()
    
    resources = compute_module_resources(
        total_cpus, total_mem_gb, args.safety_margin, args.mode
    )
    
    # Apply --max-forks-heavy to all 3 heavy modules
    if args.max_forks_heavy is not None:
        for mod in HEAVY_MODULES:
            resources[mod]["maxForks"] = args.max_forks_heavy
    
    print("Computed resource allocation:")
    for module_key, res in resources.items():
        print(f"  {module_key}:")
        print(f"    cpus={res['cpus']}, memory={res['memory']}, threads={res['threads']}, maxForks={res['maxForks']}")
    print()
    
    if args.dry_run:
        print("--dry-run: config NOT written.")
    else:
        generate_nextflow_config(resources, args.out)
        print()
        print("Usage:")
        print(f"  nextflow run main.nf -profile slurm_singularity -c {args.out} ...")


if __name__ == "__main__":
    main()
