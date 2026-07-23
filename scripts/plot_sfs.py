#!/usr/bin/env python3
"""
Visualize Site Frequency Spectrum (SFS) from DNABR QC pipeline.

Usage:
    python scripts/plot_sfs.py --sfs stats/dnabr_qc/07_sfs/*.sfs.tsv \
                                --rare_tail stats/dnabr_qc/07_sfs/*.rare_tail_ac.tsv \
                                --out_prefix plots/sfs
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sfs", required=True, help="SFS TSV file(s)", nargs="+")
    p.add_argument("--rare_tail", help="Rare tail AC TSV file(s)", nargs="+")
    p.add_argument("--out_prefix", default="sfs", help="Output file prefix")
    p.add_argument("--dpi", type=int, default=300, help="Figure DPI")
    p.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"], help="Output format")
    return p.parse_args()


def plot_sfs_standard(df, out_path, dpi=300):
    """Standard SFS plot: linear scale, raw counts."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(df))
    width = 0.8
    
    bars = ax.bar(x, df['n_sites'], width=width, color='steelblue', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Allele Frequency Bin', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Sites', fontsize=12, fontweight='bold')
    ax.set_title('Site Frequency Spectrum (SFS)', fontsize=14, fontweight='bold')
    
    labels = [f"{row['af_bin_lo']:.3f}-{row['af_bin_hi']:.3f}" for _, row in df.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, df['n_sites'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count):,}',
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_sfs_log_scale(df, out_path, dpi=300):
    """SFS with log scale on Y-axis (better for visualizing wide range)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(df))
    width = 0.8
    
    bars = ax.bar(x, df['n_sites'], width=width, color='darkorange', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Allele Frequency Bin', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Sites (log₁₀)', fontsize=12, fontweight='bold')
    ax.set_title('Site Frequency Spectrum (Log Scale)', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    
    labels = [f"{row['af_bin_lo']:.3f}-{row['af_bin_hi']:.3f}" for _, row in df.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_sfs_proportions(df, out_path, dpi=300):
    """SFS as proportions (normalized to 1)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df = df.copy()
    df['proportion'] = df['n_sites'] / df['n_sites'].sum()
    df['cumulative'] = df['proportion'].cumsum()
    
    x = np.arange(len(df))
    width = 0.8
    
    bars = ax.bar(x, df['proportion'], width=width, color='seagreen', alpha=0.8, edgecolor='black')
    
    # Add cumulative line
    ax2 = ax.twinx()
    ax2.plot(x, df['cumulative'], color='red', marker='o', linewidth=2, markersize=6, label='Cumulative')
    ax2.set_ylabel('Cumulative Proportion', fontsize=12, fontweight='bold', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim([0, 1.05])
    ax2.legend(loc='upper left')
    
    ax.set_xlabel('Allele Frequency Bin', fontsize=12, fontweight='bold')
    ax.set_ylabel('Proportion of Sites', fontsize=12, fontweight='bold')
    ax.set_title('Site Frequency Spectrum (Proportions)', fontsize=14, fontweight='bold')
    
    labels = [f"{row['af_bin_lo']:.3f}-{row['af_bin_hi']:.3f}" for _, row in df.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Add percentage labels on bars
    for i, (bar, prop) in enumerate(zip(bars, df['proportion'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{prop*100:.1f}%',
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_rare_tail(df, out_path, dpi=300, max_ac=20):
    """Plot rare tail distribution (singleton, doubleton, etc.)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df = df[df['AC'] <= max_ac].copy()
    
    bars = ax.bar(df['AC'], df['n_sites'], color='darkred', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Allele Count (AC)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Sites', fontsize=12, fontweight='bold')
    ax.set_title(f'Rare Variant Tail (AC ≤ {max_ac})', fontsize=14, fontweight='bold')
    ax.set_xticks(df['AC'])
    
    # Add count labels
    for bar, count in zip(bars, df['n_sites']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count):,}',
                ha='center', va='bottom', fontsize=9, rotation=0)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_summary_panel(df_sfs, df_rare, out_path, dpi=300):
    """Multi-panel summary figure."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Panel 1: Standard SFS
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(df_sfs))
    ax1.bar(x, df_sfs['n_sites'], color='steelblue', alpha=0.8, edgecolor='black')
    ax1.set_xlabel('AF Bin')
    ax1.set_ylabel('N Sites')
    ax1.set_title('A) Standard SFS', fontweight='bold')
    labels = [f"{row['af_bin_lo']:.3f}-{row['af_bin_hi']:.3f}" for _, row in df_sfs.iterrows()]
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    
    # Panel 2: Log scale SFS
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(x, df_sfs['n_sites'], color='darkorange', alpha=0.8, edgecolor='black')
    ax2.set_xlabel('AF Bin')
    ax2.set_ylabel('N Sites (log₁₀)')
    ax2.set_title('B) Log Scale SFS', fontweight='bold')
    ax2.set_yscale('log')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    
    # Panel 3: Proportions
    ax3 = fig.add_subplot(gs[1, 0])
    df_sfs_copy = df_sfs.copy()
    df_sfs_copy['proportion'] = df_sfs_copy['n_sites'] / df_sfs_copy['n_sites'].sum()
    ax3.bar(x, df_sfs_copy['proportion'], color='seagreen', alpha=0.8, edgecolor='black')
    ax3.set_xlabel('AF Bin')
    ax3.set_ylabel('Proportion')
    ax3.set_title('C) Proportions', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    
    # Panel 4: Rare tail
    ax4 = fig.add_subplot(gs[1, 1])
    df_rare_sub = df_rare[df_rare['AC'] <= 20].copy()
    ax4.bar(df_rare_sub['AC'], df_rare_sub['n_sites'], color='darkred', alpha=0.8, edgecolor='black')
    ax4.set_xlabel('Allele Count (AC)')
    ax4.set_ylabel('N Sites')
    ax4.set_title('D) Rare Tail (AC ≤ 20)', fontweight='bold')
    ax4.set_xticks(df_rare_sub['AC'])
    
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def main():
    args = parse_args()
    
    out_dir = Path(args.out_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load SFS data
    dfs_sfs = []
    for sfs_file in args.sfs:
        df = pd.read_csv(sfs_file, sep='\t')
        dfs_sfs.append(df)
    
    # Concatenate if multiple chromosomes
    df_sfs = pd.concat(dfs_sfs, ignore_index=True) if len(dfs_sfs) > 1 else dfs_sfs[0]
    
    # If multiple chromosomes, aggregate by bin
    if 'chr' in df_sfs.columns and df_sfs['chr'].nunique() > 1:
        df_sfs = df_sfs.groupby(['af_bin_lo', 'af_bin_hi'], as_index=False).agg({'n_sites': 'sum'})
    
    # Load rare tail if provided
    df_rare = None
    if args.rare_tail:
        dfs_rare = []
        for rare_file in args.rare_tail:
            df = pd.read_csv(rare_file, sep='\t')
            dfs_rare.append(df)
        df_rare = pd.concat(dfs_rare, ignore_index=True) if len(dfs_rare) > 1 else dfs_rare[0]
        if 'chr' in df_rare.columns and df_rare['chr'].nunique() > 1:
            df_rare = df_rare.groupby('AC', as_index=False).agg({'n_sites': 'sum'})
    
    # Generate plots
    fmt = args.format
    plot_sfs_standard(df_sfs, f"{args.out_prefix}_standard.{fmt}", dpi=args.dpi)
    plot_sfs_log_scale(df_sfs, f"{args.out_prefix}_log.{fmt}", dpi=args.dpi)
    plot_sfs_proportions(df_sfs, f"{args.out_prefix}_proportions.{fmt}", dpi=args.dpi)
    
    if df_rare is not None:
        plot_rare_tail(df_rare, f"{args.out_prefix}_rare_tail.{fmt}", dpi=args.dpi)
        plot_summary_panel(df_sfs, df_rare, f"{args.out_prefix}_summary.{fmt}", dpi=args.dpi)
    
    print("\nSummary statistics:")
    print(f"Total sites: {df_sfs['n_sites'].sum():,}")
    print(f"Sites AF<0.05 (rare): {df_sfs[df_sfs['af_bin_hi'] <= 0.05]['n_sites'].sum():,} ({df_sfs[df_sfs['af_bin_hi'] <= 0.05]['n_sites'].sum() / df_sfs['n_sites'].sum() * 100:.1f}%)")
    if df_rare is not None:
        print(f"Singletons (AC=1): {df_rare[df_rare['AC'] == 1]['n_sites'].values[0]:,}")
        print(f"Doubletons (AC=2): {df_rare[df_rare['AC'] == 2]['n_sites'].values[0]:,}")


if __name__ == "__main__":
    main()
