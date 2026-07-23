#!/usr/bin/env python3
"""
Visualize LD decay from DNABR QC pipeline.

Usage:
    python scripts/plot_ld_decay.py --ld_decay stats/dnabr_qc/10_ld_decay/*.ld_decay.tsv \
                                     --out_prefix plots/ld_decay
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    pass  # seaborn optional, use default matplotlib style


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ld_decay", required=True, help="LD decay TSV file(s)", nargs="+")
    p.add_argument("--out_prefix", default="ld_decay", help="Output file prefix")
    p.add_argument("--max_dist_kb", type=int, default=1000, help="Max distance to plot (kb)")
    p.add_argument("--dpi", type=int, default=300, help="Figure DPI")
    p.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"], help="Output format")
    p.add_argument("--include_sex_chr", action="store_true", default=False,
                   help="Include sex chromosomes (X, Y). Default: exclude for population genetics accuracy")
    return p.parse_args()


def plot_ld_decay_standard(df, out_path, max_dist_kb=1000, dpi=300):
    """Standard LD decay plot: mean r² vs distance."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    df = df[df['bin'] <= max_dist_kb * 1000].copy()
    df['dist_kb'] = df['bin'] / 1000
    
    # Plot mean and median
    ax.plot(df['dist_kb'], df['mean_r2'], 'o-', color='steelblue', linewidth=2, 
            markersize=4, label='Mean r²', alpha=0.8)
    ax.plot(df['dist_kb'], df['median_r2'], 's-', color='darkorange', linewidth=2, 
            markersize=4, label='Median r²', alpha=0.8)
    
    # Add reference lines
    ax.axhline(y=0.2, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='r² = 0.2 (LD threshold)')
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Physical Distance (kb)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Linkage Disequilibrium (r²)', fontsize=12, fontweight='bold')
    ax.set_title('LD Decay', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_ld_decay_log_distance(df, out_path, max_dist_kb=1000, dpi=300):
    """LD decay with log scale on X-axis (emphasizes short-range LD)."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    df = df[(df['bin'] > 0) & (df['bin'] <= max_dist_kb * 1000)].copy()
    df['dist_kb'] = df['bin'] / 1000
    
    ax.plot(df['dist_kb'], df['mean_r2'], 'o-', color='steelblue', linewidth=2, 
            markersize=4, label='Mean r²', alpha=0.8)
    ax.plot(df['dist_kb'], df['median_r2'], 's-', color='darkorange', linewidth=2, 
            markersize=4, label='Median r²', alpha=0.8)
    
    ax.axhline(y=0.2, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='r² = 0.2')
    
    ax.set_xlabel('Physical Distance (kb, log scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Linkage Disequilibrium (r²)', fontsize=12, fontweight='bold')
    ax.set_title('LD Decay (Log Distance)', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_ld_decay_with_n_pairs(df, out_path, max_dist_kb=1000, dpi=300):
    """LD decay with n_pairs on secondary axis."""
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    df = df[df['bin'] <= max_dist_kb * 1000].copy()
    df['dist_kb'] = df['bin'] / 1000
    
    # Primary axis: r²
    ax1.plot(df['dist_kb'], df['mean_r2'], 'o-', color='steelblue', linewidth=2, 
             markersize=4, label='Mean r²', alpha=0.8)
    ax1.axhline(y=0.2, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='r² = 0.2')
    ax1.set_xlabel('Physical Distance (kb)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Linkage Disequilibrium (r²)', fontsize=12, fontweight='bold', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_ylim([0, 1.05])
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Secondary axis: n_pairs
    ax2 = ax1.twinx()
    ax2.bar(df['dist_kb'], df['n_pairs'], width=df['dist_kb'].diff().median(), 
            color='gray', alpha=0.3, label='N pairs')
    ax2.set_ylabel('Number of Variant Pairs', fontsize=12, fontweight='bold', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.legend(loc='upper right', fontsize=10)
    
    ax1.set_title('LD Decay with Sample Size', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_ld_half_decay(df, out_path, dpi=300):
    """Calculate and annotate LD half-decay distance."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    df = df[df['bin'] > 0].copy()
    df['dist_kb'] = df['bin'] / 1000
    
    # Find half-decay distance (where r² drops to 50% of initial)
    r2_initial = df.iloc[0]['mean_r2']
    r2_half = r2_initial / 2
    
    # Interpolate to find distance at r² = r2_half
    idx_below = df[df['mean_r2'] <= r2_half].index
    if len(idx_below) > 0:
        idx = idx_below[0]
        if idx > 0:
            # Linear interpolation
            x0, y0 = df.loc[idx-1, ['dist_kb', 'mean_r2']]
            x1, y1 = df.loc[idx, ['dist_kb', 'mean_r2']]
            half_decay_kb = x0 + (r2_half - y0) * (x1 - x0) / (y1 - y0)
        else:
            half_decay_kb = df.loc[idx, 'dist_kb']
    else:
        half_decay_kb = None
    
    ax.plot(df['dist_kb'], df['mean_r2'], 'o-', color='steelblue', linewidth=2, 
            markersize=4, label='Mean r²', alpha=0.8)
    
    # Annotate half-decay
    if half_decay_kb is not None:
        ax.axhline(y=r2_half, color='green', linestyle='--', linewidth=1.5, alpha=0.6)
        ax.axvline(x=half_decay_kb, color='green', linestyle='--', linewidth=1.5, alpha=0.6)
        ax.plot(half_decay_kb, r2_half, 'go', markersize=10)
        ax.text(half_decay_kb, r2_half + 0.05, 
                f'Half-decay\n{half_decay_kb:.1f} kb\n(r² = {r2_half:.3f})',
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Physical Distance (kb)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Linkage Disequilibrium (r²)', fontsize=12, fontweight='bold')
    ax.set_title('LD Half-Decay Distance', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")
    
    return half_decay_kb


def plot_summary_panel(df, out_path, max_dist_kb=1000, dpi=300):
    """Multi-panel summary figure."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    df = df[df['bin'] <= max_dist_kb * 1000].copy()
    df['dist_kb'] = df['bin'] / 1000
    
    # Panel A: Standard LD decay
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df['dist_kb'], df['mean_r2'], 'o-', color='steelblue', linewidth=2, markersize=3)
    ax1.plot(df['dist_kb'], df['median_r2'], 's-', color='darkorange', linewidth=2, markersize=3)
    ax1.axhline(y=0.2, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
    ax1.set_xlabel('Distance (kb)')
    ax1.set_ylabel('r²')
    ax1.set_title('A) LD Decay', fontweight='bold')
    ax1.legend(['Mean', 'Median', 'r²=0.2'], fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Panel B: Log scale
    ax2 = fig.add_subplot(gs[0, 1])
    df_pos = df[df['bin'] > 0]
    ax2.plot(df_pos['dist_kb'], df_pos['mean_r2'], 'o-', color='steelblue', linewidth=2, markersize=3)
    ax2.axhline(y=0.2, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
    ax2.set_xlabel('Distance (kb, log)')
    ax2.set_ylabel('r²')
    ax2.set_title('B) LD Decay (Log Distance)', fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    
    # Panel C: With n_pairs
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(df['dist_kb'], df['mean_r2'], 'o-', color='steelblue', linewidth=2, markersize=3)
    ax3.set_xlabel('Distance (kb)')
    ax3.set_ylabel('r²', color='steelblue')
    ax3.tick_params(axis='y', labelcolor='steelblue')
    ax3.set_title('C) LD Decay with Sample Size', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3_twin = ax3.twinx()
    ax3_twin.bar(df['dist_kb'], df['n_pairs'], width=df['dist_kb'].diff().median(), 
                 color='gray', alpha=0.3)
    ax3_twin.set_ylabel('N pairs', color='gray')
    ax3_twin.tick_params(axis='y', labelcolor='gray')
    
    # Panel D: Statistics table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Calculate statistics
    r2_10kb = df[df['dist_kb'] <= 10]['mean_r2'].mean()
    r2_50kb = df[(df['dist_kb'] > 10) & (df['dist_kb'] <= 50)]['mean_r2'].mean()
    r2_100kb = df[(df['dist_kb'] > 50) & (df['dist_kb'] <= 100)]['mean_r2'].mean()
    total_pairs = df['n_pairs'].sum()
    
    # Calculate distance at r²=0.2 (handle case where it never drops that low)
    df_below_02 = df[df['mean_r2'] <= 0.2]
    dist_at_02_str = f"~{df_below_02.iloc[0]['dist_kb']:.0f} kb" if len(df_below_02) > 0 else ">1000 kb (does not decay to 0.2)"
    
    stats_text = f"""
    D) Summary Statistics
    
    Total variant pairs: {total_pairs:,}
    
    Mean r² by distance:
      0-10 kb:    {r2_10kb:.3f}
      10-50 kb:   {r2_50kb:.3f}
      50-100 kb:  {r2_100kb:.3f}
    
    Initial r² (0-1kb): {df.iloc[0]['mean_r2']:.3f}
    
    Distance at r²=0.2:
      {dist_at_02_str}
    """
    
    ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def main():
    args = parse_args()
    
    out_dir = Path(args.out_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load LD decay data
    dfs = []
    for ld_file in args.ld_decay:
        df = pd.read_csv(ld_file, sep='\t')
        dfs.append(df)
    
    # Concatenate if multiple chromosomes
    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    
    # Exclude sex chromosomes by default (LD estimates are biased due to hemizygosity)
    if not args.include_sex_chr and 'chr' in df.columns:
        sex_chr = ['X', 'Y', 'x', 'y', 'chrX', 'chrY', 'CHRX', 'CHRY']
        df = df[~df['chr'].astype(str).isin(sex_chr)]
        print(f"Excluded sex chromosomes. Using {df['chr'].nunique()} autosomes.")
    
    # If multiple chromosomes, aggregate by bin
    if 'chr' in df.columns and df['chr'].nunique() > 1:
        df = df.groupby('bin', as_index=False).agg({
            'n_pairs': 'sum',
            'mean_r2': 'mean',
            'median_r2': 'mean'  # Average of medians across chromosomes
        })
    
    # Sort by distance
    df = df.sort_values('bin').reset_index(drop=True)
    
    # Generate plots
    fmt = args.format
    plot_ld_decay_standard(df, f"{args.out_prefix}_standard.{fmt}", max_dist_kb=args.max_dist_kb, dpi=args.dpi)
    plot_ld_decay_log_distance(df, f"{args.out_prefix}_log.{fmt}", max_dist_kb=args.max_dist_kb, dpi=args.dpi)
    plot_ld_decay_with_n_pairs(df, f"{args.out_prefix}_with_n_pairs.{fmt}", max_dist_kb=args.max_dist_kb, dpi=args.dpi)
    half_decay_kb = plot_ld_half_decay(df, f"{args.out_prefix}_half_decay.{fmt}", dpi=args.dpi)
    plot_summary_panel(df, f"{args.out_prefix}_summary.{fmt}", max_dist_kb=args.max_dist_kb, dpi=args.dpi)
    
    print("\nSummary statistics:")
    print(f"Total variant pairs: {df['n_pairs'].sum():,}")
    print(f"Initial r² (0-1kb): {df.iloc[0]['mean_r2']:.3f}")
    if half_decay_kb:
        print(f"Half-decay distance: {half_decay_kb:.1f} kb")
    
    # Find distance at r² = 0.2
    df_below_02 = df[df['mean_r2'] <= 0.2]
    if len(df_below_02) > 0:
        dist_at_02 = df_below_02.iloc[0]['bin'] / 1000
        print(f"Distance at r² = 0.2: ~{dist_at_02:.0f} kb")


if __name__ == "__main__":
    main()
