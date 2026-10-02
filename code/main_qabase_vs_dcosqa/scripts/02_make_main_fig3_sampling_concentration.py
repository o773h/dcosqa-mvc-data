#!/usr/bin/env python3
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
B = ROOT / 'code/main_qabase_vs_dcosqa'
DER = B / 'derived'
DER.mkdir(parents=True, exist_ok=True)
SRC = DER / 'fig1_fig2_certified_bitstrings_recomputed_from_raw.csv'
OUT = DER / 'main_fig3_sampling_concentration_paired_strict.csv'
NS = [70, 80, 90, 100, 120, 150]
SOLVERS = ['QA-Base', 'DC-OSQA']
EXPECTED_MATCHED = 118

def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda : handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def concentration_metrics(counts: np.ndarray) -> dict:
    counts = np.asarray(counts, dtype=float)
    counts = counts[np.isfinite(counts) & (counts > 0)]
    if counts.size == 0:
        raise RuntimeError('empty positive occurrence vector')
    p = counts / counts.sum()
    k = len(p)
    entropy = float(-(p * np.log(p)).sum())
    normalized_entropy = entropy / np.log(k) if k > 1 else 0.0
    ess = float(1.0 / np.sum(p ** 2))
    ess_fraction = ess / k
    descending = np.sort(p)[::-1]
    top_decile_count = max(1, int(np.ceil(0.1 * k)))
    top_decile_share = float(descending[:top_decile_count].sum())
    sorted_p = np.sort(p)
    index = np.arange(1, k + 1, dtype=float)
    gini = float(2.0 * np.sum(index * sorted_p) / (k * sorted_p.sum()) - (k + 1.0) / k)
    return {'NormalizedEntropy': normalized_entropy, 'ESSFraction': ess_fraction, 'TopDecileShare': top_decile_share, 'GiniCoefficient': gini, 'UniqueCertified': int(k), 'CertifiedOccurrences': float(counts.sum())}
if not SRC.exists():
    raise SystemExit(f'missing source: {SRC}')
df = pd.read_csv(SRC, low_memory=False)
required = {'N', 'instance', 'solver_canonical', 'Bitstring', 'Occurrences'}
missing = sorted(required - set(df.columns))
if missing:
    raise SystemExit(f'missing columns: {missing}')
df = df[df['N'].isin(NS) & df['solver_canonical'].isin(SOLVERS)].copy()
df['N'] = pd.to_numeric(df['N'], errors='raise').astype(int)
df['instance'] = pd.to_numeric(df['instance'], errors='raise').astype(int)
df['Occurrences'] = pd.to_numeric(df['Occurrences'], errors='raise')
rows = []
for ((n, instance, solver), g) in df.groupby(['N', 'instance', 'solver_canonical'], dropna=False):
    counts = g.groupby('Bitstring', dropna=False)['Occurrences'].sum().to_numpy(dtype=float)
    metrics = concentration_metrics(counts)
    rows.append({'N': int(n), 'instance': int(instance), 'solver_canonical': str(solver), **metrics})
out = pd.DataFrame(rows)
qa_keys = set(map(tuple, out.loc[out['solver_canonical'].eq('QA-Base'), ['N', 'instance']].to_numpy()))
dc_keys = set(map(tuple, out.loc[out['solver_canonical'].eq('DC-OSQA'), ['N', 'instance']].to_numpy()))
matched = qa_keys & dc_keys
missing_qa = sorted(((int(n), int(i)) for (n, i) in dc_keys - qa_keys))
missing_dc = sorted(((int(n), int(i)) for (n, i) in qa_keys - dc_keys))
if len(matched) != EXPECTED_MATCHED:
    raise RuntimeError(f'expected {EXPECTED_MATCHED} matched keys, found {len(matched)}')
if missing_qa != [(120, 4), (150, 15)]:
    raise RuntimeError(f'unexpected missing QA keys: {missing_qa}')
if missing_dc:
    raise RuntimeError(f'unexpected missing DC keys: {missing_dc}')
out = out[out.apply(lambda r: (int(r['N']), int(r['instance'])) in matched, axis=1)].copy()
out = out.sort_values(['N', 'instance', 'solver_canonical'], kind='stable').reset_index(drop=True)
if len(out) != EXPECTED_MATCHED * 2:
    raise RuntimeError(f'expected {EXPECTED_MATCHED * 2} rows, found {len(out)}')
out.to_csv(OUT, index=False)
manifest = {'status': 'PASS_STRICT_CERTIFIED_TO_CONCENTRATION', 'source': str(SRC.relative_to(ROOT)), 'source_sha256': sha256(SRC), 'selected_source_rows': int(len(df)), 'matched_graph_instances': int(len(matched)), 'missing_qa_keys': missing_qa, 'missing_dc_keys': missing_dc, 'output': str(OUT.relative_to(ROOT)), 'output_rows': int(len(out)), 'output_sha256': sha256(OUT), 'definitions': {'NormalizedEntropy': 'Shannon entropy divided by log(number of unique certified candidates)', 'ESSFraction': 'inverse Simpson effective sample size divided by number of unique certified candidates', 'TopDecileShare': 'probability mass of the most frequent ceil(10% of unique certified candidates)', 'GiniCoefficient': 'Gini coefficient of certified-candidate probability masses'}}
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[3]
SOURCE_BASE = ROOT / 'code/main_qabase_vs_dcosqa'
INPUT = SOURCE_BASE / 'derived' / 'main_fig3_sampling_concentration_paired_strict.csv'
OUT_BASE = ROOT / 'code/main_qabase_vs_dcosqa'
OUT_DIR = OUT_BASE / 'curated'
OUT_PDF = OUT_DIR / 'Fig3_sampling_concentration_paired.pdf'
OUT_PNG = OUT_DIR / 'Fig3_sampling_concentration_paired.png'
OUT_SUMMARY = OUT_BASE / 'derived/Fig3_sampling_concentration_summary.csv'
QA_COLOR = '#5F5F5F'
DC_COLOR = '#2F6F8F'
PAIR_COLOR = '#B5B5B5'

def sem(values: np.ndarray) -> float:
    if len(values) <= 1:
        return float('nan')
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))

def clean_axis(ax: plt.Axes) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.18)
    ax.set_axisbelow(True)
    ax.tick_params(direction='out')

def main() -> None:
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)
    df = pd.read_csv(INPUT)
    source_rows = len(df)
    source_n = set(pd.to_numeric(df['N'], errors='raise').astype(int).unique())
    source_solvers = set(df['solver_canonical'].dropna().astype(str).unique())
    required = {'N', 'instance', 'solver_canonical', 'NormalizedEntropy', 'ESSFraction', 'TopDecileShare', 'GiniCoefficient'}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f'missing columns: {sorted(missing)}')
    df = df[df['solver_canonical'].isin(['QA-Base', 'DC-OSQA'])].copy()
    metrics = [('NormalizedEntropy', 'Normalized entropy', 'Higher = less concentrated'), ('ESSFraction', 'Effective-support fraction', 'Higher = broader effective support'), ('TopDecileShare', 'Top-decile occurrence share', 'Lower = less concentrated'), ('GiniCoefficient', 'Gini coefficient', 'Lower = more even')]
    (fig, axes) = plt.subplots(2, 2, figsize=(9.4, 7.2), constrained_layout=True)
    summary_rows = []
    for (idx, (ax, (metric, ylabel, subtitle))) in enumerate(zip(axes.flat, metrics)):
        pivot = df.pivot_table(index=['N', 'instance'], columns='solver_canonical', values=metric, aggfunc='first').dropna(subset=['QA-Base', 'DC-OSQA']).sort_index()
        qa = pivot['QA-Base'].to_numpy(float)
        dc = pivot['DC-OSQA'].to_numpy(float)
        for (q, d) in zip(qa, dc):
            ax.plot([0, 1], [q, d], color=PAIR_COLOR, linewidth=0.55, alpha=0.5, zorder=1)
        ax.scatter(np.zeros(len(qa)), qa, s=16, color=QA_COLOR, alpha=0.7, edgecolor='white', linewidth=0.3, zorder=3)
        ax.scatter(np.ones(len(dc)), dc, s=19, marker='s', color=DC_COLOR, alpha=0.82, edgecolor='white', linewidth=0.35, zorder=4)
        means = [np.mean(qa), np.mean(dc)]
        ax.scatter([0, 1], means, s=82, marker='D', color=['#333333', '#174E6B'], edgecolor='white', linewidth=0.8, zorder=6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['QA-Base', 'DC-OSQA'])
        ax.set_xlim(-0.28, 1.28)
        ax.set_ylabel(ylabel)
        ax.text(0.01, 1.05, f'({chr(97 + idx)})', transform=ax.transAxes, fontsize=12.5, fontweight='bold', ha='left', va='bottom', clip_on=False, zorder=20)
        clean_axis(ax)
        for (solver, values) in [('QA-Base', qa), ('DC-OSQA', dc)]:
            summary_rows.append({'metric': metric, 'solver': solver, 'pairs': len(values), 'mean': float(np.mean(values)), 'sem': sem(values)})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False)
    fig.savefig(OUT_PDF, bbox_inches='tight')
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    plt.close(fig)
    expected_n = {70, 80, 90, 100, 120, 150}
    expected_source_solvers = {'QA-Base', 'DC-OSQA'}
    expected_metrics = {'NormalizedEntropy', 'ESSFraction', 'TopDecileShare', 'GiniCoefficient'}
    plotted_n = set(pd.to_numeric(df['N'], errors='raise').astype(int).unique())
    plotted_solvers = set(df['solver_canonical'].dropna().astype(str).unique())
    summary_df = pd.DataFrame(summary_rows)
    observed_summary_solvers = set(summary_df['solver'].astype(str).unique())
    observed_summary_metrics = set(summary_df['metric'].astype(str).unique())
    ok = OUT_PDF.is_file() and OUT_PNG.is_file() and (OUT_PDF.stat().st_size > 0) and (OUT_PNG.stat().st_size > 0) and (source_rows == 236) and (source_n == expected_n) and (source_solvers == expected_source_solvers) and (len(df) == 236) and (plotted_n == expected_n) and (plotted_solvers == {'QA-Base', 'DC-OSQA'}) and (observed_summary_solvers == {'QA-Base', 'DC-OSQA'}) and (observed_summary_metrics == expected_metrics) and (len(summary_df) == 8) and summary_df['pairs'].eq(118).all()
    if not ok:
        raise SystemExit(1)
if __name__ == '__main__':
    main()
