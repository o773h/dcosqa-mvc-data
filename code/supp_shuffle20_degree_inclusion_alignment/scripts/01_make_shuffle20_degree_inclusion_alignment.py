#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
BUNDLE = Path(__file__).resolve().parents[1]
RAW = BUNDLE.parents[1] / 'data/raw/shuffle20_control_n150/raw_solutions.csv'
DERIVED = BUNDLE / 'derived'
DERIVED.mkdir(exist_ok=True)
OUT = DERIVED / 'shuffle20_n150_certified_counts_recomputed_from_raw.csv'
raw = pd.read_csv(RAW, low_memory=False)
required = ['Nodes', 'Instance', 'Graph_Type', 'Graph_M', 'Graph_Seed', 'Solver', 'Solver_Canonical', 'Assignment_Type', 'Shuffle_ID', 'Shuffle_Seed', 'Shuffle_Hash', 'OffsetScale', 'Num_Reads', 'source_dataset', 'Embedding_Source', 'Embedding_Status', 'clean_dataset', 'Bitstring', 'Occurrences', 'IsValid', 'Size', 'K_exact']
missing = [c for c in required if c not in raw.columns]
if missing:
    raise SystemExit(f'Missing raw columns: {missing}')
group_cols = ['Nodes', 'Instance', 'Graph_Type', 'Graph_M', 'Graph_Seed', 'Solver', 'Solver_Canonical', 'Assignment_Type', 'Shuffle_ID', 'Shuffle_Seed', 'Shuffle_Hash', 'OffsetScale', 'Num_Reads', 'source_dataset', 'Embedding_Source', 'Embedding_Status', 'clean_dataset', 'K_exact']
r = raw.copy()
if r['IsValid'].dtype != bool:
    r['IsValid'] = r['IsValid'].astype(str).str.lower().isin(['true', '1', 'yes'])
r['Occurrences'] = pd.to_numeric(r['Occurrences'], errors='coerce').fillna(0)
r['Size'] = pd.to_numeric(r['Size'], errors='coerce')
r['K_exact'] = pd.to_numeric(r['K_exact'], errors='coerce')
cert = r['IsValid'].eq(True) & r['Size'].eq(r['K_exact'])

def nunique_cert_bitstrings(x):
    idx = x.index
    return r.loc[idx[cert.loc[idx]], 'Bitstring'].nunique()

def sum_cert_occ(x):
    idx = x.index
    return r.loc[idx[cert.loc[idx]], 'Occurrences'].sum()
agg = r.groupby(group_cols, dropna=False).agg(RawOccurrences=('Occurrences', 'sum'), RawUniqueBitstrings=('Bitstring', 'nunique'), ValidOccurrences=('Occurrences', lambda x: x[r.loc[x.index, 'IsValid']].sum()), ValidUniqueBitstrings=('Bitstring', lambda x: r.loc[x.index[r.loc[x.index, 'IsValid']], 'Bitstring'].nunique()), CertifiedOccurrences=('Occurrences', sum_cert_occ), UniqueCertifiedCandidates=('Bitstring', nunique_cert_bitstrings)).reset_index()
agg.insert(0, 'scope', 'shuffle20_n150')
value_cols = ['RawOccurrences', 'RawUniqueBitstrings', 'ValidOccurrences', 'ValidUniqueBitstrings', 'CertifiedOccurrences', 'UniqueCertifiedCandidates']
for c in value_cols:
    pass
BUNDLE = Path(__file__).resolve().parents[1]
RAW = BUNDLE.parents[1] / 'data/raw/shuffle20_control_n150/raw_solutions.csv'
DERIVED = BUNDLE / 'derived'
DERIVED.mkdir(exist_ok=True)
OUT = DERIVED / 'qa_shuffle20_degree_inclusion_probabilities_recomputed_from_raw.csv'
raw = pd.read_csv(RAW, low_memory=False)
r = raw.copy()
if r['IsValid'].dtype != bool:
    r['IsValid'] = r['IsValid'].astype(str).str.lower().isin(['true', '1', 'yes'])
r['Size'] = pd.to_numeric(r['Size'], errors='coerce')
r['K_exact'] = pd.to_numeric(r['K_exact'], errors='coerce')
r['Occurrences'] = pd.to_numeric(r['Occurrences'], errors='coerce').fillna(0)
cert = r[r['IsValid'].eq(True) & r['Size'].eq(r['K_exact'])].copy()
_degree_rows = []
for _graph in r[['Instance', 'Nodes', 'Graph_M', 'Graph_Seed']].drop_duplicates().itertuples(index=False):
    _network = nx.barabasi_albert_graph(int(_graph.Nodes), int(_graph.Graph_M), seed=int(_graph.Graph_Seed))
    _degree = dict(_network.degree())
    _maximum = max(_degree.values())
    for (_vertex, _value) in sorted(_degree.items()):
        _degree_rows.append({'Instance': int(_graph.Instance), 'vertex': int(_vertex), 'degree': int(_value), 'degree_norm': float(_value) / float(_maximum)})
degree_meta = pd.DataFrame(_degree_rows).sort_values(['Instance', 'vertex']).reset_index(drop=True)
rows = []
group_cols = ['Instance', 'Assignment_Type', 'Shuffle_ID']
for ((inst, atype, sid), g) in cert.groupby(group_cols, dropna=False, sort=True):
    gtype = atype
    certified_occ = float(g['Occurrences'].sum())
    if certified_occ == 0:
        continue
    meta = degree_meta[degree_meta['Instance'].eq(inst)].copy()
    if len(meta) == 0:
        raise SystemExit(f'No degree metadata for Instance={inst}')
    inc = np.zeros(len(meta), dtype=float)
    vertices = meta['vertex'].to_numpy(dtype=int)
    for (bitstring, occ) in zip(g['Bitstring'].astype(str), g['Occurrences'].to_numpy(dtype=float)):
        arr = np.fromiter((1 if bitstring[v] == '1' else 0 for v in vertices), dtype=np.int8, count=len(vertices))
        inc += occ * arr
    meta['GroupType'] = gtype
    meta['Shuffle_ID'] = int(sid)
    meta['p_inclusion'] = inc / certified_occ
    meta['certified_occ'] = int(certified_occ) if certified_occ.is_integer() else certified_occ
    rows.append(meta[['Instance', 'GroupType', 'Shuffle_ID', 'vertex', 'degree', 'degree_norm', 'p_inclusion', 'certified_occ']])
recomputed = pd.concat(rows, ignore_index=True)
recomputed = recomputed.sort_values(['Instance', 'GroupType', 'Shuffle_ID', 'vertex'], kind='mergesort').reset_index(drop=True)
recomputed.to_csv(OUT, index=False)
cmp_cols = ['degree', 'degree_norm', 'p_inclusion', 'certified_occ']
for c in cmp_cols:
    pass
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
BASE = Path(__file__).resolve().parents[1]
INPUT = BASE / 'derived' / 'qa_shuffle20_degree_inclusion_probabilities_recomputed_from_raw.csv'
OUT_DIR = BASE / 'curated'
OUT_PDF = OUT_DIR / 'shuffle20_degree_inclusion_alignment.pdf'
OUT_PNG = OUT_DIR / 'shuffle20_degree_inclusion_alignment.png'
OUT_VERTEX = BASE / 'derived' / 'shuffle20_original_minus_shuffled_vertex_shift.csv'
OUT_CORR = BASE / 'derived' / 'shuffle20_degree_shift_correlations.csv'
OUT_BINS = BASE / 'derived' / 'shuffle20_degree_shift_degree_groups.csv'
MEAN_COLOR = '#2F6F8F'
POINT_COLOR = '#858585'
ZERO_COLOR = '#555555'

def sem(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors='coerce').dropna()
    if len(values) <= 1:
        return float('nan')
    return float(values.std(ddof=1) / math.sqrt(len(values)))

def clean_axis(ax: plt.Axes) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out')

def main() -> None:
    df = pd.read_csv(INPUT)
    required = {'Instance', 'GroupType', 'Shuffle_ID', 'vertex', 'degree', 'p_inclusion'}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f'missing columns: {missing}')
    original = df[df['GroupType'].eq('original')].groupby(['Instance', 'vertex'], as_index=False).agg(degree=('degree', 'first'), p_original=('p_inclusion', 'first'))
    shuffled = df[df['GroupType'].eq('shuffled')].groupby(['Instance', 'vertex'], as_index=False).agg(p_shuffled_mean=('p_inclusion', 'mean'), p_shuffled_sd=('p_inclusion', 'std'), n_shuffles=('Shuffle_ID', 'nunique'))
    vertex = original.merge(shuffled, on=['Instance', 'vertex'], how='inner', validate='one_to_one')
    vertex['delta_p'] = vertex['p_original'] - vertex['p_shuffled_mean']
    vertex.to_csv(OUT_VERTEX, index=False)
    corr_rows = []
    bin_rows = []
    for (instance, group) in vertex.groupby('Instance'):
        rho = group['degree'].corr(group['delta_p'], method='spearman')
        corr_rows.append({'Instance': int(instance), 'spearman_rho': float(rho), 'vertices': len(group), 'mean_delta': float(group['delta_p'].mean())})
        # Degree groups retain all tied-degree vertices together. The cut points
        # separate the BA(m=2) minimum/modal degree (d <= 2), the range around
        # the ensemble mean degree (3 <= d <= 4), and the above-mean tail
        # (d >= 5). All three groups are present in every N=150 instance.
        group = group.copy()
        group['degree_group'] = np.select(
            [group['degree'].le(2), group['degree'].le(4)],
            ['Low', 'Middle'],
            default='High',
        )
        observed_groups = set(group['degree_group'])
        expected_groups = {'Low', 'Middle', 'High'}
        if len(group) != 150 or observed_groups != expected_groups:
            raise RuntimeError(f'Instance {instance}: unexpected degree groups {sorted(observed_groups)}')
        for (degree_group, b) in group.groupby('degree_group'):
            bin_rows.append({'Instance': int(instance), 'degree_group': str(degree_group), 'mean_degree': float(b['degree'].mean()), 'mean_delta_p': float(b['delta_p'].mean()), 'vertices': len(b)})
    corr_df = pd.DataFrame(corr_rows).sort_values('Instance')
    per_instance_bins = pd.DataFrame(bin_rows)
    group_order = ['Low', 'Middle', 'High']
    per_instance_bins['degree_group'] = pd.Categorical(per_instance_bins['degree_group'], categories=group_order, ordered=True)
    pooled_bins = per_instance_bins.groupby('degree_group', observed=False, as_index=False).agg(mean_degree=('mean_degree', 'mean'), mean_delta_p=('mean_delta_p', 'mean'), sem_delta_p=('mean_delta_p', sem), instances=('Instance', 'nunique'), vertices=('vertices', 'sum'))
    corr_df.to_csv(OUT_CORR, index=False)
    pooled_bins.to_csv(OUT_BINS, index=False)
    (fig, axes) = plt.subplots(1, 2, figsize=(9.2, 3.75), constrained_layout=True)
    ax = axes[0]
    x = np.arange(len(corr_df), dtype=float)
    ax.axhline(0, color=ZERO_COLOR, linestyle='--', linewidth=0.9, zorder=0)
    ax.scatter(x, corr_df['spearman_rho'], s=28, color=POINT_COLOR, edgecolor='white', linewidth=0.45, zorder=3)
    mean_rho = float(corr_df['spearman_rho'].mean())
    ax.axhline(mean_rho, color=MEAN_COLOR, linewidth=2.0, zorder=2, label=f'Mean ρ = {mean_rho:.2f}')
    tick_positions = np.arange(0, len(corr_df), 2)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(int(corr_df.iloc[i]['Instance'])) for i in tick_positions])
    ax.set_xlim(-0.6, len(corr_df) - 0.4)
    ax.set_xlabel('Graph instance')
    ax.set_ylabel('Degree-shift\nSpearman correlation, ρ')
    ax.legend(frameon=False, fontsize=8.5, loc='lower right')
    ax.text(-0.14, 1.04, '(a)', transform=ax.transAxes, fontsize=12, fontweight='bold')
    clean_axis(ax)
    ax = axes[1]
    bx = np.arange(len(pooled_bins), dtype=float)
    y = pooled_bins['mean_delta_p'].to_numpy(float)
    e = pooled_bins['sem_delta_p'].to_numpy(float)
    ax.axhline(0, color=ZERO_COLOR, linestyle='--', linewidth=0.9, zorder=0)
    for (degree_group, group) in per_instance_bins.groupby('degree_group', observed=False):
        group_index = group_order.index(str(degree_group))
        jitter = np.linspace(-0.14, 0.14, len(group))
        ax.scatter(np.full(len(group), group_index, dtype=float) + jitter, group['mean_delta_p'], s=14, color=POINT_COLOR, alpha=0.48, edgecolor='none', zorder=2)
    ax.errorbar(bx, y, yerr=e, fmt='D', markersize=6.5, color=MEAN_COLOR, markerfacecolor=MEAN_COLOR, markeredgecolor='#222222', markeredgewidth=0.7, linewidth=1.25, capsize=3.5, zorder=5)
    ax.set_xticks(bx)
    ax.set_xticklabels(['Low\n($d\\leq2$)', 'Middle\n($3\\leq d\\leq4$)', 'High\n($d\\geq5$)'])
    ax.set_xlabel('Degree group')
    ax.set_ylabel('Vertex-inclusion shift, Δp')
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.text(-0.14, 1.04, '(b)', transform=ax.transAxes, fontsize=12, fontweight='bold')
    clean_axis(ax)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches='tight')
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    plt.close(fig)
    positive = int((corr_df['spearman_rho'] > 0).sum())
if __name__ == '__main__':
    main()
