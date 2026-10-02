#!/usr/bin/env python3
from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import wilcoxon
BASE = Path(__file__).resolve().parents[1]
RAW = BASE / 'derived' / 'verified_candidate_stream.csv.gz'
REFERENCE_META = BASE / 'reference' / 'qa_bam34_graph_metadata_recorded_kexact.csv'
OUTDIR = BASE / 'derived'
OUT_META = OUTDIR / 'bam34_recorded_graph_metadata_and_kexact_used.csv'
OUT_COUNTS = OUTDIR / 'bam34_certified_candidate_counts_by_graph_solver.csv'
OUT_PAIRED = OUTDIR / 'bam34_paired_statistics_vs_qabase_by_offset_scale.csv'
OUT_BEST = OUTDIR / 'bam34_best_offset_scale_by_graph_size_and_metric.csv'
OUT_BYN = OUTDIR / 'bam34_paired_gain_plot_summary_from_raw.csv'
OUTDIR.mkdir(parents=True, exist_ok=True)

def sem(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors='coerce').dropna()
    if len(values) <= 1:
        return float('nan')
    return float(values.std(ddof=1) / math.sqrt(len(values)))

def bit_to_arr(value: str) -> np.ndarray:
    return np.fromiter((1 if c == '1' else 0 for c in str(value)), dtype=np.uint8)

def complete_linkage_family_count(bitstrings, n: int) -> int:
    bits = sorted(set(map(str, bitstrings)))
    count = len(bits)
    if count == 0:
        return 0
    if count == 1:
        return 1
    matrix = np.vstack([bit_to_arr(bit) for bit in bits])
    if matrix.shape[1] != n:
        raise RuntimeError(f'bitstring length {matrix.shape[1]} does not match N={n}')
    distances = pdist(matrix, metric='hamming') * n
    hierarchy = linkage(distances, method='complete')
    labels = fcluster(hierarchy, t=0.1 * n, criterion='distance')
    return int(pd.Series(labels).nunique())

def canonical_solver(raw_solver: str) -> str:
    value = str(raw_solver)
    if value == 'QA-Base':
        return 'QA-Base'
    if value.startswith('DC-Offset'):
        return 'DC-OSQA'
    return value
raw = pd.read_csv(RAW, dtype={'Bitstring': 'string'}, low_memory=False)
required = {'Nodes', 'Instance', 'Solver', 'Bitstring', 'Size', 'IsValid', 'Occurrences', 'Embedding_Source'}
missing = sorted(required - set(raw.columns))
if missing:
    raise RuntimeError(f'raw file missing columns: {missing}')
raw['Nodes'] = pd.to_numeric(raw['Nodes'], errors='raise').astype(int)
raw['Instance'] = pd.to_numeric(raw['Instance'], errors='raise').astype(int)
raw['Size'] = pd.to_numeric(raw['Size'], errors='raise').astype(int)
raw['Occurrences'] = pd.to_numeric(raw['Occurrences'], errors='raise').astype(int)
raw['IsValid_bool'] = raw['IsValid'].astype(str).str.lower().isin(['true', '1', 'yes'])
if 'Graph_M' not in raw.columns:
    extracted = raw['Embedding_Source'].astype(str).str.extract('_m(\\d+)_N', expand=False)
    if extracted.isna().any():
        bad = raw.loc[extracted.isna(), 'Embedding_Source'].drop_duplicates()
        raise RuntimeError('could not recover Graph_M from Embedding_Source:\n' + bad.to_string(index=False))
    raw['Graph_M'] = extracted.astype(int)
else:
    raw['Graph_M'] = pd.to_numeric(raw['Graph_M'], errors='raise').astype(int)
if 'Solver_Canonical' not in raw.columns:
    raw['Solver_Canonical'] = raw['Solver'].map(canonical_solver)
meta = pd.read_csv(REFERENCE_META)
required_meta = {'Graph_M', 'Nodes', 'Instance', 'K_exact', 'Graph_Seed', 'Edges'}
missing_meta = sorted(required_meta - set(meta.columns))
if missing_meta:
    raise RuntimeError(f'recorded graph metadata missing columns: {missing_meta}')
meta = meta[['Graph_M', 'Nodes', 'Instance', 'K_exact', 'Graph_Seed', 'Edges']].drop_duplicates().sort_values(['Graph_M', 'Nodes', 'Instance']).reset_index(drop=True)
for column in ['Graph_M', 'Nodes', 'Instance', 'K_exact', 'Graph_Seed', 'Edges']:
    meta[column] = pd.to_numeric(meta[column], errors='raise').astype(int)
if len(meta) != 80:
    raise RuntimeError(f'expected 80 recorded BAM34 graph metadata rows, got {len(meta)}')
if meta[['Graph_M', 'Nodes', 'Instance']].duplicated().any():
    raise RuntimeError('duplicate recorded BAM34 graph metadata keys')
expected_graphs = raw[['Graph_M', 'Nodes', 'Instance']].drop_duplicates().sort_values(['Graph_M', 'Nodes', 'Instance']).reset_index(drop=True)
metadata_graphs = meta[['Graph_M', 'Nodes', 'Instance']].sort_values(['Graph_M', 'Nodes', 'Instance']).reset_index(drop=True)
if not expected_graphs.equals(metadata_graphs):
    missing_from_meta = expected_graphs.merge(metadata_graphs, on=['Graph_M', 'Nodes', 'Instance'], how='left', indicator=True)
    missing_from_meta = missing_from_meta.loc[missing_from_meta['_merge'].eq('left_only')]
    extra_in_meta = metadata_graphs.merge(expected_graphs, on=['Graph_M', 'Nodes', 'Instance'], how='left', indicator=True)
    extra_in_meta = extra_in_meta.loc[extra_in_meta['_merge'].eq('left_only')]
    raise RuntimeError(f'raw/metadata graph-key mismatch\nmissing from metadata:\n{missing_from_meta}\nextra in metadata:\n{extra_in_meta}')
meta['K_exact_method'] = 'graph-level K_exact recorded by original BAM34 experiment runner and independently verified by scipy.optimize.milp/HiGHS'
meta.to_csv(OUT_META, index=False)
required_verified_columns = {'Graph_M', 'Nodes', 'Instance', 'Graph_Seed', 'Edges', 'graph_hash_sha256', 'K_exact', 'K_exact_highs', 'strict_exact_verified'}
missing_verified_columns = sorted(required_verified_columns - set(raw.columns))
if missing_verified_columns:
    raise RuntimeError(f'verified stream missing columns: {missing_verified_columns}')
raw['K_exact'] = pd.to_numeric(raw['K_exact'], errors='raise').astype(int)
raw['K_exact_highs'] = pd.to_numeric(raw['K_exact_highs'], errors='raise').astype(int)
if not raw['K_exact'].eq(raw['K_exact_highs']).all():
    mismatch = raw.loc[~raw['K_exact'].eq(raw['K_exact_highs']), ['Graph_M', 'Nodes', 'Instance', 'K_exact', 'K_exact_highs']].drop_duplicates().head(30)
    raise RuntimeError('recorded K_exact differs from HiGHS K_exact\\n' + mismatch.to_string(index=False))
df = raw.copy()
verified_exact_mask = df['strict_exact_verified'] if pd.api.types.is_bool_dtype(df['strict_exact_verified']) else df['strict_exact_verified'].astype(str).str.strip().str.lower().isin(['true', '1', 'yes', 'y', 't'])
recomputed_exact_mask = df['IsValid_bool'] & df['Size'].eq(df['K_exact'])
if not verified_exact_mask.eq(recomputed_exact_mask).all():
    raise RuntimeError('strict_exact_verified differs from IsValid == True AND Size == K_exact')
strict = df.loc[verified_exact_mask].copy()
all_groups = df[['Graph_M', 'Nodes', 'Instance', 'Solver', 'Solver_Canonical', 'OffsetScale']].drop_duplicates().sort_values(['Graph_M', 'Nodes', 'Instance', 'Solver'], na_position='last')
rows = []
for group in all_groups.itertuples(index=False):
    mask = strict['Graph_M'].eq(group.Graph_M) & strict['Nodes'].eq(group.Nodes) & strict['Instance'].eq(group.Instance) & strict['Solver'].eq(group.Solver)
    subset = strict.loc[mask]
    if subset.empty:
        unique_strict = 0
        strict_occurrences = 0
        families = 0
    else:
        unique_strict = int(subset['Bitstring'].astype(str).nunique())
        strict_occurrences = int(subset['Occurrences'].sum())
        families = complete_linkage_family_count(subset['Bitstring'].astype(str).unique(), int(group.Nodes))
    rows.append({'Graph_M': int(group.Graph_M), 'Nodes': int(group.Nodes), 'Instance': int(group.Instance), 'Solver': group.Solver, 'Solver_Canonical': group.Solver_Canonical, 'OffsetScale': group.OffsetScale, 'K_exact': int(meta.loc[meta['Graph_M'].eq(group.Graph_M) & meta['Nodes'].eq(group.Nodes) & meta['Instance'].eq(group.Instance), 'K_exact'].iloc[0]), 'UniqueStrict': unique_strict, 'StrictOccurrences': strict_occurrences, 'FamiliesComplete': families})
out = pd.DataFrame(rows).sort_values(['Graph_M', 'Nodes', 'Instance', 'Solver'])
out.to_csv(OUT_COUNTS, index=False)
paired_rows = []
for ((graph_m, nodes), subset) in out.groupby(['Graph_M', 'Nodes']):
    baseline = subset.loc[subset['Solver'].eq('QA-Base'), ['Instance', 'UniqueStrict', 'StrictOccurrences', 'FamiliesComplete']].rename(columns={'UniqueStrict': 'QA_UniqueStrict', 'StrictOccurrences': 'QA_StrictOccurrences', 'FamiliesComplete': 'QA_FamiliesComplete'})
    for solver in sorted((value for value in subset['Solver'].dropna().unique() if value != 'QA-Base')):
        dc = subset.loc[subset['Solver'].eq(solver)].copy()
        merged = dc.merge(baseline, on='Instance', how='inner', validate='one_to_one')
        if merged.empty:
            continue
        alpha = pd.to_numeric(merged['OffsetScale'], errors='coerce').dropna()
        row = {'Graph_M': int(graph_m), 'Nodes': int(nodes), 'Solver': solver, 'OffsetScale': float(alpha.iloc[0]) if len(alpha) else np.nan, 'Pairs': len(merged)}
        for metric in ['UniqueStrict', 'FamiliesComplete', 'StrictOccurrences']:
            qa_col = 'QA_' + metric
            difference = merged[metric] - merged[qa_col]
            row[f'{metric}_DC_mean'] = float(merged[metric].mean())
            row[f'{metric}_QA_mean'] = float(merged[qa_col].mean())
            row[f'{metric}_MeanGain'] = float(difference.mean())
            row[f'{metric}_Wins'] = int(difference.gt(0).sum())
            row[f'{metric}_Losses'] = int(difference.lt(0).sum())
            row[f'{metric}_Ties'] = int(difference.eq(0).sum())
            try:
                row[f'{metric}_WilcoxonP'] = float(wilcoxon(merged[metric], merged[qa_col], zero_method='wilcox').pvalue)
            except Exception:
                row[f'{metric}_WilcoxonP'] = np.nan
        paired_rows.append(row)
paired = pd.DataFrame(paired_rows).sort_values(['Graph_M', 'Nodes', 'OffsetScale'])
paired.to_csv(OUT_PAIRED, index=False)
best_rows = []
for ((graph_m, nodes), subset) in paired.groupby(['Graph_M', 'Nodes']):
    for metric in ['UniqueStrict', 'FamiliesComplete']:
        gain_col = f'{metric}_MeanGain'
        best_index = subset[gain_col].idxmax()
        best = subset.loc[best_index]
        alpha_01 = subset.loc[np.isclose(pd.to_numeric(subset['OffsetScale'], errors='coerce'), 0.1), gain_col]
        best_rows.append({'Graph_M': int(graph_m), 'Nodes': int(nodes), 'Metric': metric, 'BestAlpha': best['OffsetScale'], 'BestMeanGain': best[gain_col], 'BestWins': best[f'{metric}_Wins'], 'BestLosses': best[f'{metric}_Losses'], 'BestP': best[f'{metric}_WilcoxonP'], 'Alpha0p1Gain': float(alpha_01.iloc[0]) if len(alpha_01) else np.nan})
best_table = pd.DataFrame(best_rows)
best_table.to_csv(OUT_BEST, index=False)
summary_rows = []
for (keys, subset) in out.groupby(['Graph_M', 'Nodes', 'Solver', 'Solver_Canonical', 'OffsetScale'], dropna=False):
    (graph_m, nodes, solver, solver_canonical, alpha) = keys
    summary_rows.append({'Graph_M': int(graph_m), 'Nodes': int(nodes), 'Solver': solver, 'Solver_Canonical': solver_canonical, 'OffsetScale': alpha, 'Instances': int(subset['Instance'].nunique()), 'UniqueStrict_mean': float(subset['UniqueStrict'].mean()), 'UniqueStrict_sem': sem(subset['UniqueStrict']), 'FamiliesComplete_mean': float(subset['FamiliesComplete'].mean()), 'FamiliesComplete_sem': sem(subset['FamiliesComplete']), 'StrictOccurrences_mean': float(subset['StrictOccurrences'].mean()), 'StrictOccurrences_sem': sem(subset['StrictOccurrences'])})
byn = pd.DataFrame(summary_rows).sort_values(['Graph_M', 'Nodes', 'Solver'], na_position='last')
byn.to_csv(OUT_BYN, index=False)
_validation_counts = pd.read_csv(OUT_COUNTS)
_validation_required = {'Graph_M', 'Nodes', 'Instance', 'Solver', 'OffsetScale', 'UniqueStrict', 'FamiliesComplete'}
_validation_missing = sorted(_validation_required - set(_validation_counts.columns))
if _validation_missing:
    raise RuntimeError(f'strict-count table missing columns required for paired-gain validation: {_validation_missing}')
_validation_counts['Graph_M'] = pd.to_numeric(_validation_counts['Graph_M'], errors='raise').astype(int)
_validation_counts['Nodes'] = pd.to_numeric(_validation_counts['Nodes'], errors='raise').astype(int)
_validation_counts['Instance'] = pd.to_numeric(_validation_counts['Instance'], errors='raise').astype(int)
_validation_counts['OffsetScale'] = pd.to_numeric(_validation_counts['OffsetScale'], errors='coerce')
_validation_base = _validation_counts.loc[_validation_counts['Solver'].astype(str).eq('QA-Base'), ['Graph_M', 'Nodes', 'Instance', 'UniqueStrict', 'FamiliesComplete']].drop_duplicates(['Graph_M', 'Nodes', 'Instance'], keep='first').rename(columns={'UniqueStrict': 'QA_UniqueStrict', 'FamiliesComplete': 'QA_FamiliesComplete'})
_validation_dc = _validation_counts.loc[~_validation_counts['Solver'].astype(str).eq('QA-Base') & _validation_counts['OffsetScale'].notna(), ['Graph_M', 'Nodes', 'Instance', 'OffsetScale', 'UniqueStrict', 'FamiliesComplete']].copy()
_validation_pairs = _validation_dc.merge(_validation_base, on=['Graph_M', 'Nodes', 'Instance'], how='inner', validate='many_to_one')
_validation_pairs['UniqueGain'] = _validation_pairs['UniqueStrict'] - _validation_pairs['QA_UniqueStrict']
_validation_pairs['FamilyGain'] = _validation_pairs['FamiliesComplete'] - _validation_pairs['QA_FamiliesComplete']
byn = _validation_pairs.groupby(['Graph_M', 'Nodes', 'OffsetScale'], as_index=False).agg(UniqueStrict_mean=('UniqueGain', 'mean'), UniqueStrict_sem=('UniqueGain', sem), FamiliesComplete_mean=('FamilyGain', 'mean'), FamiliesComplete_sem=('FamilyGain', sem), Instances=('Instance', 'nunique')).sort_values(['Graph_M', 'Nodes', 'OffsetScale']).reset_index(drop=True)
if len(byn) != 32:
    raise RuntimeError(f'expected 32 paired-gain validation rows (2 m × 4 N × 4 alpha), got {len(byn)}')
if not byn['Instances'].eq(10).all():
    raise RuntimeError('paired-gain validation does not contain 10 instances for every (Graph_M, Nodes, OffsetScale) condition')
byn.to_csv(OUT_BYN, index=False)
import matplotlib.pyplot as plt
BASE = Path(__file__).resolve().parents[1]
CUR = BASE / 'curated'
CUR.mkdir(parents=True, exist_ok=True)
DER = BASE / 'derived'
SRC = DER / 'bam34_paired_gain_plot_summary_from_raw.csv'
OUT_PDF = CUR / 'bam34_alpha_controls.pdf'
OUT_PNG = CUR / 'bam34_alpha_controls.png'
df = pd.read_csv(SRC)
column_aliases = {'UniqueStrict_mean': 'UniqueMean', 'UniqueStrict_sem': 'UniqueSEM', 'FamiliesComplete_mean': 'FamilyMean', 'FamiliesComplete_sem': 'FamilySEM'}
df = df.rename(columns={old: new for (old, new) in column_aliases.items() if old in df.columns and new not in df.columns})
required = {'Graph_M', 'Nodes', 'OffsetScale', 'UniqueMean', 'UniqueSEM', 'FamilyMean', 'FamilySEM', 'Instances'}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f'missing required columns: {sorted(missing)}')
df = df.copy()
df['Graph_M'] = df['Graph_M'].astype(int)
df['Nodes'] = df['Nodes'].astype(int)
df['OffsetScale'] = df['OffsetScale'].astype(float)
df = df[df['Graph_M'].isin([3, 4])]
if len(df) != 32:
    raise SystemExit(f'expected 32 rows for m=3/4 x N=70,90,120,150 x 4 alpha; got {len(df)}')
expected_N = {70, 90, 120, 150}
expected_alpha = {0.01, 0.025, 0.05, 0.1}
for m in [3, 4]:
    sub = df[df['Graph_M'].eq(m)]
    if set(sub['Nodes']) != expected_N:
        raise SystemExit(f"Graph_M={m} unexpected N set: {sorted(sub['Nodes'].unique())}")
    if set((round(x, 3) for x in sub['OffsetScale'])) != expected_alpha:
        raise SystemExit(f"Graph_M={m} unexpected alpha set: {sorted(sub['OffsetScale'].unique())}")
checks = [(3, 90, 0.05, 'UniqueMean', 53.4), (3, 90, 0.05, 'FamilyMean', 8.1), (3, 120, 0.05, 'UniqueMean', 95.6), (3, 120, 0.1, 'FamilyMean', 9.9), (4, 120, 0.1, 'UniqueMean', -37.4), (4, 120, 0.1, 'FamilyMean', -3.5)]
for (m, n, a, col, exp) in checks:
    val = float(df[df.Graph_M.eq(m) & df.Nodes.eq(n) & df.OffsetScale.round(3).eq(round(a, 3))][col].iloc[0])
    if round(val, 1) != exp:
        raise SystemExit(f'claim check failed: m={m} N={n} alpha={a} {col} got {val}, expected {exp}')
alpha_order = [0.01, 0.025, 0.05, 0.1]
labels = {0.01: '0.01', 0.025: '0.025', 0.05: '0.05', 0.1: '0.10'}
series_styles = {
    0.01: {'color': '#0072B2', 'marker': 'o', 'linestyle': '-'},
    0.025: {'color': '#E69F00', 'marker': 's', 'linestyle': (0, (6, 2))},
    0.05: {'color': '#000000', 'marker': '^', 'linestyle': (0, (4, 1.5, 1, 1.5))},
    0.1: {'color': '#CC79A7', 'marker': 'D', 'linestyle': (0, (1.5, 1.5))},
}
(fig, axes) = plt.subplots(2, 2, figsize=(10.8, 7.2), sharex=True)
metric_info = [('UniqueMean', 'UniqueSEM', 'Unique certified candidates'), ('FamilyMean', 'FamilySEM', 'Complete-linkage families')]
for (r, m) in enumerate([3, 4]):
    for (c, (mean_col, sem_col, title)) in enumerate(metric_info):
        ax = axes[r, c]
        sub_m = df[df['Graph_M'].eq(m)]
        for a in alpha_order:
            sub = sub_m[sub_m['OffsetScale'].round(3).eq(round(a, 3))].sort_values('Nodes')
            x = sub['Nodes'].to_numpy()
            y = sub[mean_col].to_numpy()
            e = sub[sem_col].to_numpy()
            style = series_styles[a]
            ax.plot(x, y, color=style['color'], marker=style['marker'], linestyle=style['linestyle'], linewidth=2.0, markersize=6, markeredgecolor='white', markeredgewidth=0.5, label=f'$\\alpha={labels[a]}$')
            ax.fill_between(x, y - e, y + e, color=style['color'], alpha=0.10)
        ax.axhline(0, color='#888888', linestyle='--', linewidth=1)
        ax.set_title(('BA $m=3$' if m == 3 else 'BA $m=4$') + ' — ' + title)
        ax.set_xticks(sorted(expected_N))
        ax.set_xlabel('Graph size N')
        ax.set_ylabel('Mean paired gain vs QA-Base')
        ax.grid(True, alpha=0.25)
(handles, lab) = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, lab, loc='upper center', ncol=4, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT_PDF, bbox_inches='tight')
fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
plt.close(fig)
df.to_csv(CUR / 'bam34_alpha_controls_plot_input_used.csv', index=False)
