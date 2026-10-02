#!/usr/bin/env python3
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
B = Path(__file__).resolve().parents[1]
CERTIFIED = ROOT / 'code/main_qabase_vs_dcosqa' / 'derived/fig1_fig2_certified_bitstrings_recomputed_from_raw.csv'
DER = B / 'derived'
DER.mkdir(parents=True, exist_ok=True)
OUT = DER / 'degree_bin_vertex_inclusion_shift_by_instance_strict_raw_derived.csv'
NS = [70, 80, 90, 100, 120, 150]
SOLVERS = ['QA-Base', 'DC-OSQA']
BINS = ['Low', 'Middle', 'High']
WEIGHTINGS = {'Unique-certified-candidate inclusion': 'unique', 'Occurrence-weighted certified inclusion': 'occurrence'}
EXPECTED_ALL_KEYS = 120
EXPECTED_MATCHED_KEYS = 118
EXPECTED_ROWS = EXPECTED_MATCHED_KEYS * len(WEIGHTINGS) * len(BINS)
EXPECTED_MISSING_QA = {(120, 4), (150, 15)}

def degree_bins(n: int, instance: int) -> pd.DataFrame:
    n = int(n)
    instance = int(instance)
    seed = int(n * 1000 + 42 + instance)
    graph = nx.barabasi_albert_graph(n, 2, seed=seed)
    if graph.number_of_nodes() != n:
        raise RuntimeError(f'node-count mismatch for N={n}, instance={instance}')
    out = pd.DataFrame({'vertex': np.arange(n, dtype=int), 'degree': [graph.degree(v) for v in range(n)]})
    # Degree groups retain tied-degree vertices together. The cut points isolate
    # the BA(m=2) minimum/modal degree (d <= 2), the range around the ensemble
    # mean degree (3 <= d <= 4), and the above-mean tail (d >= 5).
    degree = out['degree'].to_numpy(dtype=int)
    labels = np.where(degree <= 2, BINS[0], np.where(degree <= 4, BINS[1], BINS[2]))
    out['degree_bin'] = pd.Categorical(labels, categories=BINS, ordered=True)
    if set(out['degree_bin'].astype(str)) != set(BINS):
        raise RuntimeError(f'missing degree group for N={n}, instance={instance}')
    return out

def inclusion_by_vertex(group: pd.DataFrame, n: int, mode: str) -> np.ndarray:
    bitstrings = group['Bitstring'].astype(str)
    bad_lengths = bitstrings.str.len().ne(n)
    if bad_lengths.any():
        vals = sorted(bitstrings[bad_lengths].str.len().unique())
        raise RuntimeError(f'bitstring length mismatch for N={n}: {vals}')
    matrix = np.asarray([[1 if ch == '1' else 0 for ch in bitstring] for bitstring in bitstrings], dtype=float)
    if mode == 'unique':
        weights = np.ones(len(group), dtype=float)
    elif mode == 'occurrence':
        weights = pd.to_numeric(group['Occurrences'], errors='raise').to_numpy(dtype=float)
    else:
        raise RuntimeError(f'unknown weighting mode: {mode}')
    if len(weights) == 0 or weights.sum() <= 0:
        raise RuntimeError(f'undefined inclusion probability for N={n}')
    return np.average(matrix, axis=0, weights=weights)
if not CERTIFIED.exists():
    raise SystemExit(f'missing certified input: {CERTIFIED}')
df = pd.read_csv(CERTIFIED, dtype={'Bitstring': 'string'}, low_memory=False)
required = ['N', 'instance', 'solver_canonical', 'Bitstring', 'Occurrences']
missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f'missing columns in {CERTIFIED}: {missing}')
df = df[df['N'].isin(NS) & df['solver_canonical'].isin(SOLVERS)].copy()
df['N'] = pd.to_numeric(df['N'], errors='raise').astype(int)
df['instance'] = pd.to_numeric(df['instance'], errors='raise').astype(int)
df['Occurrences'] = pd.to_numeric(df['Occurrences'], errors='raise')
all_keys = {(n, i) for n in NS for i in range(20)}
solver_keys = {solver: set(map(tuple, df.loc[df['solver_canonical'].eq(solver), ['N', 'instance']].drop_duplicates().to_numpy())) for solver in SOLVERS}
matched_keys = solver_keys['QA-Base'] & solver_keys['DC-OSQA']
missing_qa = solver_keys['DC-OSQA'] - solver_keys['QA-Base']
missing_dc = solver_keys['QA-Base'] - solver_keys['DC-OSQA']
if len(all_keys) != EXPECTED_ALL_KEYS:
    raise RuntimeError(f'unexpected all-key count: {len(all_keys)}')
if missing_qa != EXPECTED_MISSING_QA:
    raise RuntimeError(f'unexpected QA-zero-yield keys: {sorted(missing_qa)}')
if missing_dc:
    raise RuntimeError(f'unexpected DC-zero-yield keys: {sorted(missing_dc)}')
if len(matched_keys) != EXPECTED_MATCHED_KEYS:
    raise RuntimeError(f'expected {EXPECTED_MATCHED_KEYS} matched keys, found {len(matched_keys)}')
rows = []
for (n, instance) in sorted(matched_keys):
    bins = degree_bins(n, instance)
    solver_groups = {}
    for solver in SOLVERS:
        g = df[df['N'].eq(n) & df['instance'].eq(instance) & df['solver_canonical'].eq(solver)].copy()
        if g.empty:
            raise RuntimeError(f'missing solver group: N={n}, instance={instance}, solver={solver}')
        solver_groups[solver] = g
    for (weighting_label, mode) in WEIGHTINGS.items():
        qa_p = inclusion_by_vertex(solver_groups['QA-Base'], n, mode)
        dc_p = inclusion_by_vertex(solver_groups['DC-OSQA'], n, mode)
        vertex = bins.copy()
        vertex['delta_p_vertex'] = dc_p - qa_p
        grouped = vertex.groupby('degree_bin', observed=False)['delta_p_vertex'].mean().reindex(BINS)
        if grouped.isna().any():
            raise RuntimeError(f'missing degree bin for N={n}, instance={instance}')
        for (degree_bin, delta_p) in grouped.items():
            rows.append({'N': n, 'instance': instance, 'weighting': weighting_label, 'degree_bin': str(degree_bin), 'delta_p': float(delta_p)})
out = pd.DataFrame(rows)
if len(out) != EXPECTED_ROWS:
    raise RuntimeError(f'expected {EXPECTED_ROWS} rows, found {len(out)}')
dup = out.duplicated(['N', 'instance', 'weighting', 'degree_bin'])
if dup.any():
    raise RuntimeError('duplicate output keys:\n' + out.loc[dup].to_string(index=False))
if set(out['degree_bin']) != set(BINS):
    raise RuntimeError(f"unexpected bins: {sorted(out['degree_bin'].unique())}")
if set(out['weighting']) != set(WEIGHTINGS):
    raise RuntimeError(f"unexpected weighting labels: {sorted(out['weighting'].unique())}")
out = out.sort_values(['weighting', 'N', 'instance', 'degree_bin'], kind='stable').reset_index(drop=True)
out.to_csv(OUT, index=False)
summary = out.groupby(['weighting', 'degree_bin'], observed=False)['delta_p'].agg(['count', 'mean', 'std']).reset_index()
summary['sem'] = summary['std'] / np.sqrt(summary['count'])
expected_means = {
    ('Unique-certified-candidate inclusion', 'Low'): -0.039736145918026135,
    ('Unique-certified-candidate inclusion', 'Middle'): 0.04830169902279132,
    ('Unique-certified-candidate inclusion', 'High'): 0.023469650906594915,
    ('Occurrence-weighted certified inclusion', 'Low'): -0.04978197387152038,
    ('Occurrence-weighted certified inclusion', 'Middle'): 0.06253991251717783,
    ('Occurrence-weighted certified inclusion', 'High'): 0.026501613771954195,
}
observed_means = summary.set_index(['weighting', 'degree_bin'])['mean'].to_dict()
for key, expected in expected_means.items():
    observed = observed_means.get(key)
    if observed is None or not np.isclose(observed, expected, rtol=0.0, atol=1e-09):
        raise RuntimeError(f'degree-bin manuscript-claim check failed for {key}: {observed} != {expected}')
summary.to_csv(DER / 'degree_bin_vertex_inclusion_shift_summary.csv', index=False)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'code/supp_degree_bin_vertex_inclusion_shift'
CUR = BASE / 'curated'
DER = BASE / 'derived'
INPUT = DER / 'degree_bin_vertex_inclusion_shift_by_instance_strict_raw_derived.csv'
OUT_PDF = CUR / 'degree_bin_vertex_inclusion_shift.pdf'
OUT_PNG = CUR / 'degree_bin_vertex_inclusion_shift.png'
ORDER = ['Low', 'Middle', 'High']
PANELS = [('Unique-certified-candidate inclusion', '(a)', 'Unique-certified candidates'), ('Occurrence-weighted certified inclusion', '(b)', 'Occurrence-weighted candidates')]
POINT_COLOR = '#8A8A8A'
MEAN_COLOR = '#2F6F8F'
LINE_COLOR = '#222222'

def sem(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))

def clean_axis(ax: plt.Axes) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.2)
    ax.set_axisbelow(True)
    ax.tick_params(direction='out')

def main() -> None:
    if not INPUT.is_file():
        raise FileNotFoundError(f'Missing derived input: {INPUT}\nRun scripts/00_build_degree_bin_shift_N70_150_strict.py first.')
    df = pd.read_csv(INPUT)
    required = {'N', 'instance', 'weighting', 'degree_bin', 'delta_p'}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f'Missing columns: {sorted(missing)}')
    if len(df) != 708:
        raise SystemExit(f'Expected 708 input rows; got {len(df)}')
    expected_weightings = {panel[0] for panel in PANELS}
    observed_weightings = set(df['weighting'].astype(str).unique())
    if observed_weightings != expected_weightings:
        raise SystemExit(f'Unexpected weightings: {sorted(observed_weightings)}')
    CUR.mkdir(parents=True, exist_ok=True)
    (fig, axes) = plt.subplots(1, 2, figsize=(10.2, 4.8), sharey=True, constrained_layout=True)
    summaries = []
    for (ax, (weighting, panel_label, xlabel)) in zip(axes, PANELS):
        sub = df[df['weighting'].eq(weighting)].copy()
        pivot = sub.pivot(index=['N', 'instance'], columns='degree_bin', values='delta_p')[ORDER].sort_index()
        if pivot.shape != (118, 3):
            raise SystemExit(f'{weighting}: expected 118 x 3 table; got {pivot.shape}')
        if pivot.isna().any().any():
            raise SystemExit(f'{weighting}: missing degree-bin values')
        x = np.arange(3, dtype=float)
        means = []
        sems = []
        for (i, degree_bin) in enumerate(ORDER):
            values = pivot[degree_bin].to_numpy(dtype=float)
            mean = float(np.mean(values))
            error = sem(values)
            means.append(mean)
            sems.append(error)
            jitter = np.linspace(-0.16, 0.16, len(values))
            ax.scatter(np.full(len(values), x[i]) + jitter, values, s=17, color=POINT_COLOR, alpha=0.38, edgecolors='none', zorder=2)
            summaries.append({'weighting': weighting, 'degree_bin': degree_bin, 'pairs': len(values), 'mean': mean, 'sem': error})
        means = np.asarray(means)
        sems = np.asarray(sems)
        ax.errorbar(x, means, yerr=sems, fmt='D', markersize=6.5, color=MEAN_COLOR, markerfacecolor=MEAN_COLOR, markeredgecolor=LINE_COLOR, markeredgewidth=0.7, linewidth=1.25, capsize=3.5, zorder=5)
        ax.axhline(0, color='#555555', linestyle='--', linewidth=0.9, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(['Low\n($d\\leq2$)', 'Middle\n($3\\leq d\\leq4$)', 'High\n($d\\geq5$)'])
        ax.set_xlabel(xlabel)
        ax.text(0.01, 1.04, panel_label, transform=ax.transAxes, ha='left', va='bottom', fontsize=11, fontweight='bold', clip_on=False)
        clean_axis(ax)
    axes[0].set_ylabel('Change in certified inclusion probability\n(DC-OSQA − QA-Base)')
    fig.savefig(OUT_PDF, bbox_inches='tight')
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    plt.close(fig)
    summary = pd.DataFrame(summaries)
    if len(summary) != 6 or not summary['pairs'].eq(118).all():
        raise SystemExit('Degree-bin summary contract failed')
if __name__ == '__main__':
    main()
