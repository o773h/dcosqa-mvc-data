#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
import hashlib
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
BUNDLE = Path(__file__).resolve().parents[1]
ROOT = BUNDLE.parents[1]
RAW = BUNDLE.parents[1] / 'data/raw/shuffle20_control_n150/raw_solutions.csv'
DERIVED = BUNDLE / 'derived'
CURATED = BUNDLE / 'curated'
ANALYSIS_READY = BUNDLE / 'analysis_ready'
for d in [DERIVED, CURATED, ANALYSIS_READY]:
    d.mkdir(exist_ok=True)
OUT_AR = DERIVED / 'multishuffle_n150_analysis_ready_recomputed_from_control20_raw.csv'
OUT_BY = CURATED / 'multishuffle_n150_by_instance.csv'
OUT_PCT = CURATED / 'multishuffle_n150_percentile_summary.csv'

def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda : handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def family_count(bitstrings, n, tau_frac=0.1):
    b = sorted(set(map(str, bitstrings)))
    if len(b) <= 1:
        return len(b)
    mat = np.array([[1 if ch == '1' else 0 for ch in s] for s in b], dtype=np.uint8)
    dist = pdist(mat, metric='hamming') * n
    z = linkage(dist, method='complete')
    return int(pd.Series(fcluster(z, t=tau_frac * n, criterion='distance')).nunique())
if not RAW.exists():
    raise FileNotFoundError(RAW)
df = pd.read_csv(RAW, low_memory=False)
required = ['Nodes', 'Instance', 'Solver_Canonical', 'Bitstring', 'IsValid', 'Size', 'Occurrences', 'K_exact', 'Shuffle_ID']
missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f'missing required columns in {RAW}: {missing}')
df['Nodes'] = pd.to_numeric(df['Nodes'], errors='coerce').astype('Int64')
df['Instance'] = pd.to_numeric(df['Instance'], errors='coerce').astype('Int64')
df['Size'] = pd.to_numeric(df['Size'], errors='coerce')
df['K_exact'] = pd.to_numeric(df['K_exact'], errors='coerce')
df['Occurrences'] = pd.to_numeric(df['Occurrences'], errors='coerce').fillna(0)
df['Shuffle_ID'] = pd.to_numeric(df['Shuffle_ID'], errors='coerce').astype('Int64')
if df['IsValid'].dtype != bool:
    df['IsValid'] = df['IsValid'].astype(str).str.lower().isin(['true', '1', 'yes'])
cert = df[df['IsValid'].eq(True) & df['Size'].eq(df['K_exact'])].copy()
keep_solver = cert['Solver_Canonical'].astype(str).isin(['DC-OSQA', 'Shuffled-DC(s=0.1)'])
keep_shuffle = cert['Solver_Canonical'].astype(str).eq('DC-OSQA') & cert['Shuffle_ID'].eq(-1) | cert['Solver_Canonical'].astype(str).eq('Shuffled-DC(s=0.1)') & cert['Shuffle_ID'].between(0, 19)
cert = cert[keep_solver & keep_shuffle].copy()
group_cols = ['Nodes', 'Instance', 'Solver_Canonical', 'Shuffle_ID']
counts = cert.groupby(group_cols, dropna=False).agg(Unique_strict=('Bitstring', 'nunique'), Strict_occurrences=('Occurrences', 'sum')).reset_index()
fam_rows = []
for (gk, gsub) in cert.groupby(group_cols, dropna=False):
    fam_rows.append({'Nodes': gk[0], 'Instance': gk[1], 'Solver_Canonical': gk[2], 'Shuffle_ID': gk[3], 'Families_complete_linkage': family_count(gsub['Bitstring'].astype(str).unique(), int(gk[0]))})
counts = counts.merge(pd.DataFrame(fam_rows), on=group_cols, how='left')
instances = sorted(pd.to_numeric(df['Instance'], errors='coerce').dropna().astype(int).unique())
key_rows = []
for inst in instances:
    key_rows.append({'Nodes': 150, 'Instance': inst, 'Solver_Canonical': 'DC-OSQA', 'Shuffle_ID': -1})
    for sid in range(20):
        key_rows.append({'Nodes': 150, 'Instance': inst, 'Solver_Canonical': 'Shuffled-DC(s=0.1)', 'Shuffle_ID': sid})
keyspace = pd.DataFrame(key_rows)
ar = keyspace.merge(counts, on=group_cols, how='left')
ar['Unique_strict'] = ar['Unique_strict'].fillna(0).astype(int)
ar['Strict_occurrences'] = ar['Strict_occurrences'].fillna(0).astype(float)
ar['Families_complete_linkage'] = ar['Families_complete_linkage'].fillna(0).astype(int)
ar['zero_yield'] = ar['Unique_strict'].eq(0).astype(int)
ar['source_dataset'] = 'qa_shuffle20_n150'
ar['metric'] = 'Unique certified candidates'
ar.to_csv(OUT_AR, index=False)
dc = ar[ar['Solver_Canonical'].eq('DC-OSQA')][['Instance', 'Unique_strict', 'Strict_occurrences', 'Families_complete_linkage']].rename(columns={'Unique_strict': 'dc_unique', 'Strict_occurrences': 'dc_occurrences', 'Families_complete_linkage': 'dc_families'})
sh = ar[ar['Solver_Canonical'].eq('Shuffled-DC(s=0.1)')].copy()
rows = []
for (metric, dc_col, sh_col) in [('Unique certified candidates', 'dc_unique', 'Unique_strict'), ('Certified occurrences', 'dc_occurrences', 'Strict_occurrences'), ('Complete-linkage families', 'dc_families', 'Families_complete_linkage')]:
    tmp = sh.merge(dc[['Instance', dc_col]], on='Instance', how='left')
    tmp[sh_col] = pd.to_numeric(tmp[sh_col], errors='coerce')
    tmp[dc_col] = pd.to_numeric(tmp[dc_col], errors='coerce')
    g = tmp.groupby('Instance', dropna=False)
    by = g.agg(shuffle_mean=(sh_col, 'mean'), shuffle_median=(sh_col, 'median'), shuffle_min=(sh_col, 'min'), shuffle_max=(sh_col, 'max'), n_shuffles=(sh_col, 'count'), dc_value=(dc_col, 'first')).reset_index()
    by['metric'] = metric
    by['dc_minus_shuffle_mean'] = by['dc_value'] - by['shuffle_mean']
    by['dc_minus_shuffle_median'] = by['dc_value'] - by['shuffle_median']
    by['dc_above_shuffle_median'] = by['dc_value'] > by['shuffle_median']
    by['dc_above_shuffle_max'] = by['dc_value'] > by['shuffle_max']
    pct = []
    for (inst, sub) in tmp.groupby('Instance', dropna=False):
        dc_val = sub[dc_col].iloc[0]
        pct.append((inst, float((sub[sh_col] <= dc_val).mean() * 100.0)))
    pct = pd.DataFrame(pct, columns=['Instance', 'dc_percentile_vs_20_shuffles'])
    by = by.merge(pct, on='Instance', how='left')
    rows.append(by)
by_instance = pd.concat(rows, ignore_index=True)
by_instance = by_instance[['metric', 'Instance', 'dc_value', 'shuffle_mean', 'shuffle_median', 'shuffle_min', 'shuffle_max', 'n_shuffles', 'dc_minus_shuffle_mean', 'dc_minus_shuffle_median', 'dc_above_shuffle_median', 'dc_above_shuffle_max', 'dc_percentile_vs_20_shuffles']].rename(columns={'Instance': 'instance'})
by_instance.to_csv(OUT_BY, index=False)
summary = by_instance.groupby('metric', dropna=False).agg(mean_dc_percentile=('dc_percentile_vs_20_shuffles', 'mean'), median_dc_percentile=('dc_percentile_vs_20_shuffles', 'median'), instances_dc_above_shuffle_median=('dc_above_shuffle_median', 'sum'), instances_dc_above_90th_percentile=('dc_percentile_vs_20_shuffles', lambda s: int((s >= 90).sum())), instances_dc_equal_or_above_best=('dc_above_shuffle_max', 'sum'), n_instances=('instance', 'count'), mean_dc_minus_shuffle_mean=('dc_minus_shuffle_mean', 'mean'), median_dc_minus_shuffle_mean=('dc_minus_shuffle_mean', 'median')).reset_index()
summary.to_csv(OUT_PCT, index=False)
status = 'PASS' if len(ar) == 20 * 21 and set(ar.loc[ar['Solver_Canonical'].eq('Shuffled-DC(s=0.1)'), 'Shuffle_ID'].dropna().astype(int)) == set(range(20)) else 'CHECK'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'code/supp_shuffle20_control'
DER = BASE / 'derived'
CUR = BASE / 'curated'
INPUT = DER / 'multishuffle_n150_analysis_ready_recomputed_from_control20_raw.csv'
OUT_PDF = CUR / 'shuffle20_control.pdf'
OUT_PNG = CUR / 'shuffle20_control.png'
SHUFFLE_COLOR = '#8FAFC1'
DC_COLOR = '#4F6F8F'
GRID = '#B0B0B0'
CUR.mkdir(parents=True, exist_ok=True)

def pick(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f'missing columns among {candidates}; available={list(df.columns)}')
    return None

def clean_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', color=GRID, alpha=0.2, linewidth=0.75)
    ax.set_axisbelow(True)

def main():
    df = pd.read_csv(INPUT)
    instance_col = pick(df, ['instance', 'Instance'])
    solver_col = pick(df, ['solver_canonical', 'Solver_Canonical', 'solver', 'Solver', 'condition'])
    unique_col = pick(df, ['Unique_strict', 'unique_strict', 'unique'])
    occ_col = pick(df, ['Strict_occurrences', 'strict_occurrences', 'occurrences'])
    fam_col = pick(df, ['Families_complete_linkage', 'families_complete_linkage', 'families'])
    df[instance_col] = pd.to_numeric(df[instance_col], errors='coerce')
    df[unique_col] = pd.to_numeric(df[unique_col], errors='coerce')
    df[occ_col] = pd.to_numeric(df[occ_col], errors='coerce')
    labels = df[solver_col].astype(str)
    dc_mask = labels.str.contains('DC-OSQA|DC-Offset', case=False, regex=True)
    shuffle_mask = labels.str.contains('shuffle', case=False, regex=False)
    if dc_mask.sum() == 0:
        raise SystemExit('no DC rows detected')
    if shuffle_mask.sum() == 0:
        raise SystemExit('no shuffle rows detected')
    dc = df[dc_mask].copy()
    sh = df[shuffle_mask].copy()
    dc_inst = dc.groupby(instance_col, as_index=False).agg(dc_unique=(unique_col, 'mean'), dc_occ=(occ_col, 'mean'))
    sh_inst = sh.groupby(instance_col, as_index=False).agg(shuffle_unique_mean=(unique_col, 'mean'), shuffle_occ_mean=(occ_col, 'mean'))
    order_df = dc_inst.merge(sh_inst, on=instance_col, how='inner')
    order_df['ordering_gain'] = order_df['dc_unique'] - order_df['shuffle_unique_mean']
    order_df = order_df.sort_values(instance_col, ascending=True).reset_index(drop=True)
    order_map = {inst: i for (i, inst) in enumerate(order_df[instance_col].tolist())}
    df = df[df[instance_col].isin(order_map)].copy()
    df['plot_x'] = df[instance_col].map(order_map)
    ninst = len(order_df)
    (fig, axes) = plt.subplots(2, 1, figsize=(13.5, 8.0), sharex=True, constrained_layout=False)
    specs = [(axes[0], unique_col, 'Unique certified candidates'), (axes[1], fam_col, 'Complete-linkage Hamming families ($\\tau=0.1N$)')]
    rng = np.random.default_rng(20260710)
    for (label, (ax, metric, ylabel)) in zip(['(a)', '(b)'], specs):
        for i in range(ninst):
            if i % 2 == 0:
                ax.axvspan(i - 0.5, i + 0.5, color='#F2F4F7', zorder=0)
        for b in range(ninst - 1):
            ax.axvline(b + 0.5, color='#D9DEE5', linewidth=0.6, zorder=1)
        for (inst, sub) in df[shuffle_mask].groupby(instance_col):
            x0 = order_map[inst]
            jitter = rng.uniform(-0.22, 0.22, len(sub))
            ax.scatter(x0 + jitter, sub[metric], s=20, color=SHUFFLE_COLOR, edgecolor='none', alpha=0.6, zorder=2)
        sh_median = df[shuffle_mask].groupby(instance_col)[metric].median()
        for (inst, medv) in sh_median.items():
            ax.scatter(order_map[inst], medv, s=58, marker='o', facecolor='none', edgecolor='#2F4353', linewidth=1.3, zorder=3)
        dc_metric = df[dc_mask].groupby(instance_col, as_index=False)[metric].mean()
        dc_metric['plot_x'] = dc_metric[instance_col].map(order_map)
        dc_metric = dc_metric.sort_values('plot_x')
        ax.scatter(dc_metric['plot_x'], dc_metric[metric], s=62, marker='s', color=DC_COLOR, edgecolor='white', linewidth=0.8, zorder=4)
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.6, ninst - 0.4)
        ax.minorticks_off()
        clean_axis(ax)
        ax.text(0.0, 1.02, label, transform=ax.transAxes, fontsize=14, fontweight='bold', ha='left', va='bottom')
    tick_pos = list(range(ninst))
    axes[1].set_xticks(tick_pos)
    axes[1].set_xticklabels([str(int(order_df.iloc[i][instance_col])) for i in tick_pos], fontsize=8.5)
    axes[1].set_xlabel('Graph instance')
    handles = [Line2D([0], [0], marker='o', linestyle='none', markerfacecolor=SHUFFLE_COLOR, markeredgecolor='none', markersize=6, label='20 shuffled controls'), Line2D([0], [0], marker='o', linestyle='none', markerfacecolor='none', markeredgecolor='#2F4353', markeredgewidth=1.3, markersize=7, label='Shuffle median'), Line2D([0], [0], marker='s', linestyle='none', markerfacecolor=DC_COLOR, markeredgecolor='white', markersize=7, label='Degree-aligned DC-OSQA')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.005), ncol=3, frameon=False, fontsize=10)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.93, bottom=0.085, hspace=0.13)
    fig.savefig(OUT_PDF, bbox_inches='tight')
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    plt.close(fig)
    order_df.to_csv(CUR / 'shuffle20_instance_plot_order.csv', index=False)
    audit = ['status: PASS', f'input: {INPUT.relative_to(ROOT)}', f'input_rows: {len(df)}', f'input_sha256: {sha256(INPUT)}', f'instances: {len(order_df)}', f'shuffle_rows: {int(shuffle_mask.sum())}', f'dc_rows: {int(dc_mask.sum())}', f'output_pdf: {OUT_PDF.relative_to(ROOT)}', f'output_png: {OUT_PNG.relative_to(ROOT)}', f'output_pdf_sha256: {sha256(OUT_PDF)}']
if __name__ == '__main__':
    main()
