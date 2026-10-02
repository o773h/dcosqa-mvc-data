#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
BUNDLE = Path(__file__).resolve().parents[1]
ROOT = BUNDLE.parents[1]
RAW = ROOT / 'data/raw' / 'qa_base_repr_n200' / 'raw_solutions.csv'
TARGET = BUNDLE / 'analysis_ready' / 'n200_qabase_extension_analysis_ready.csv'
ACTIVE_INPUT = BUNDLE / 'analysis_ready' / 'n200_qabase_scale_extension_input.csv'
DERIVED = BUNDLE / 'derived'
AUDIT = BUNDLE / 'audit'
DERIVED.mkdir(exist_ok=True)
OUT = DERIVED / 'n200_qabase_extension_analysis_ready_recomputed_from_raw.csv'
DETAIL = AUDIT / 'n200_qabase_extension_qabase_raw_to_analysis_ready_detail.csv'
MISMATCH = AUDIT / 'n200_qabase_extension_qabase_raw_to_analysis_ready_mismatches.csv'
ACTIVE_DETAIL = AUDIT / 'n200_qabase_extension_qabase_raw_to_active_input_detail.csv'
ACTIVE_MISMATCH = AUDIT / 'n200_qabase_extension_qabase_raw_to_active_input_mismatches.csv'
raw = pd.read_csv(RAW, low_memory=False)
target = pd.read_csv(TARGET, low_memory=False)
active = pd.read_csv(ACTIVE_INPUT, low_memory=False)
r = raw.copy()
if r['IsValid'].dtype != bool:
    r['IsValid'] = r['IsValid'].astype(str).str.lower().isin(['true', '1', 'yes'])
r['Size'] = pd.to_numeric(r['Size'], errors='coerce')
r['Occurrences'] = pd.to_numeric(r['Occurrences'], errors='coerce').fillna(0)
kmap = target[['instance', 'K_exact']].drop_duplicates()
if kmap.groupby('instance')['K_exact'].nunique().max() != 1:
    raise SystemExit('K_exact is not unique by instance in target')
kmap = dict(zip(kmap['instance'], kmap['K_exact']))
r['K_exact'] = r['Instance'].map(kmap)
r['K_exact'] = pd.to_numeric(r['K_exact'], errors='coerce')
cert = r[r['IsValid'].eq(True) & r['Size'].eq(r['K_exact'])].copy()
agg = cert.groupby('Instance', dropna=False).agg(Unique_strict=('Bitstring', 'nunique'), Strict_occurrences=('Occurrences', 'sum')).reset_index().rename(columns={'Instance': 'instance'})
meta_cols = [c for c in target.columns if c not in ['Unique_strict', 'Strict_occurrences']]
recomputed = target[meta_cols].merge(agg, on='instance', how='left')
recomputed['Unique_strict'] = recomputed['Unique_strict'].fillna(0).astype(int)
recomputed['Strict_occurrences'] = recomputed['Strict_occurrences'].fillna(0).astype(float)
recomputed = recomputed[target.columns]
recomputed.to_csv(OUT, index=False)
TARGET.parent.mkdir(parents=True, exist_ok=True)
recomputed.to_csv(TARGET, index=False)

def compare_table(left, right, key, value_cols, detail_path, mismatch_path):
    l = left[key + value_cols].copy()
    rr = right[key + value_cols].copy()
    merged = l.merge(rr, on=key, how='outer', suffixes=('_target', '_recomputed'), indicator=True)
    mismatch = merged['_merge'].ne('both')
    for c in value_cols:
        a = pd.to_numeric(merged[f'{c}_target'], errors='coerce')
        b = pd.to_numeric(merged[f'{c}_recomputed'], errors='coerce')
        mismatch = mismatch | ~a.fillna(-999999999).eq(b.fillna(-999999999))
    merged['mismatch'] = mismatch
    mism = merged[merged['mismatch']].copy()
    return len(mism)
value_cols = ['Unique_strict', 'Strict_occurrences']
m1 = compare_table(target, recomputed, ['instance'], value_cols, DETAIL, MISMATCH)
active_qb = active[active['solver_canonical'].eq('QA-Base')].copy()
m2 = compare_table(active_qb, recomputed, ['instance'], value_cols, ACTIVE_DETAIL, ACTIVE_MISMATCH)
BUNDLE = Path(__file__).resolve().parents[1]
ROOT = BUNDLE.parents[1]
RAWROOT = ROOT / 'data/raw'
TARGET = BUNDLE / 'analysis_ready' / 'n200_extension_analysis_ready.csv'
ACTIVE_INPUT = BUNDLE / 'analysis_ready' / 'n200_qabase_scale_extension_input.csv'
DERIVED = BUNDLE / 'derived'
AUDIT = BUNDLE / 'audit'
DERIVED.mkdir(exist_ok=True)
OUT = DERIVED / 'n200_extension_analysis_ready_recomputed_from_raw_strict_counts.csv'
ACTIVE_OUT = DERIVED / 'n200_qabase_scale_extension_input_recomputed_from_raw_strict_counts.csv'
DETAIL = AUDIT / 'n200_qabase_extension_full_raw_to_analysis_ready_strict_counts_detail.csv'
MISMATCH = AUDIT / 'n200_qabase_extension_full_raw_to_analysis_ready_strict_counts_mismatches.csv'
ACTIVE_DETAIL = AUDIT / 'n200_qabase_extension_full_raw_to_active_input_strict_counts_detail.csv'
ACTIVE_MISMATCH = AUDIT / 'n200_qabase_extension_full_raw_to_active_input_strict_counts_mismatches.csv'
target = pd.read_csv(TARGET, low_memory=False)
active = pd.read_csv(ACTIVE_INPUT, low_memory=False)
strict_cols = ['Unique_strict', 'Strict_occurrences', 'zero_yield']
kmap = target[['instance', 'K_exact']].drop_duplicates()
if kmap.groupby('instance')['K_exact'].nunique().max() != 1:
    raise SystemExit('K_exact is not unique by instance')
kmap = dict(zip(kmap['instance'], kmap['K_exact']))
solver_map = {'QA-Base': 'QA-Base', 'BC-Offset(s=0.1)': 'BC-OSQA', 'DC-Offset(s=0.1)': 'DC-OSQA', 'SA-Max': 'SA-Max'}
rows = []
raw_stats = []
for sd in sorted(target['source_dataset'].dropna().astype(str).unique()):
    raw_path = RAWROOT / sd / 'raw_solutions.csv'
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    raw = pd.read_csv(raw_path, low_memory=False)
    r = raw.copy()
    if r['IsValid'].dtype != bool:
        r['IsValid'] = r['IsValid'].astype(str).str.lower().isin(['true', '1', 'yes'])
    r['Size'] = pd.to_numeric(r['Size'], errors='coerce')
    r['Occurrences'] = pd.to_numeric(r['Occurrences'], errors='coerce').fillna(0)
    r['Instance'] = pd.to_numeric(r['Instance'], errors='coerce').astype('Int64')
    if 'Solver_Canonical' not in r.columns:
        r['Solver_Canonical'] = r['Solver'].map(solver_map).fillna(r['Solver'].astype(str))
    expected_solvers = set(target.loc[target['source_dataset'].astype(str).eq(sd), 'solver_canonical'].dropna().astype(str).unique().tolist())
    r = r[r['Solver_Canonical'].astype(str).isin(expected_solvers)].copy()
    r['K_exact'] = r['Instance'].map(kmap)
    r['K_exact'] = pd.to_numeric(r['K_exact'], errors='coerce')
    cert = r[r['IsValid'].eq(True) & r['Size'].eq(r['K_exact'])].copy()
    agg = cert.groupby(['Instance', 'Solver_Canonical'], dropna=False).agg(Unique_strict=('Bitstring', 'nunique'), Strict_occurrences=('Occurrences', 'sum')).reset_index().rename(columns={'Instance': 'instance', 'Solver_Canonical': 'solver_canonical'})
    keyspace = target[target['source_dataset'].astype(str).eq(sd)][['source_dataset', 'instance', 'solver_canonical']].drop_duplicates()
    tmp = keyspace.merge(agg, on=['instance', 'solver_canonical'], how='left')
    tmp['Unique_strict'] = tmp['Unique_strict'].fillna(0).astype(int)
    tmp['Strict_occurrences'] = tmp['Strict_occurrences'].fillna(0).astype(float)
    tmp['zero_yield'] = tmp['Unique_strict'].eq(0).astype(int)
    rows.append(tmp)
    raw_stats.append({'source_dataset': sd, 'raw_rows': len(raw), 'used_raw_rows_after_solver_filter': len(r), 'certified_raw_rows': len(cert), 'raw_path': raw_path.relative_to(ROOT).as_posix()})
raw_counts = pd.concat(rows, ignore_index=True)
raw_counts = raw_counts.drop_duplicates(subset=['source_dataset', 'instance', 'solver_canonical'], keep='last')
key = ['source_dataset', 'instance', 'solver_canonical']
recomputed = target.copy()
recomputed = recomputed.drop(columns=[c for c in strict_cols if c in recomputed.columns])
recomputed = recomputed.merge(raw_counts[key + strict_cols], on=key, how='left')
for c in strict_cols:
    recomputed[c] = recomputed[c].fillna(0)
recomputed = recomputed[target.columns]
recomputed.to_csv(OUT, index=False)
TARGET.parent.mkdir(parents=True, exist_ok=True)
recomputed.to_csv(TARGET, index=False)

def compare(left, right, key_cols, value_cols, detail_path, mismatch_path):
    l = left[key_cols + value_cols].copy()
    rr = right[key_cols + value_cols].copy()
    for df in [l, rr]:
        for c in key_cols:
            df[c] = df[c].astype('string').fillna('<NA>')
    m = l.merge(rr, on=key_cols, how='outer', suffixes=('_target', '_recomputed'), indicator=True)
    bad = m['_merge'].ne('both')
    for c in value_cols:
        a = pd.to_numeric(m[f'{c}_target'], errors='coerce')
        b = pd.to_numeric(m[f'{c}_recomputed'], errors='coerce')
        bad = bad | ~np.isclose(a.fillna(-999999999), b.fillna(-999999999), rtol=1e-12, atol=1e-12)
    m['mismatch'] = bad
    mism = m[m['mismatch']].copy()
    return len(mism)
m1 = compare(target, recomputed, key, strict_cols, DETAIL, MISMATCH)
active_recomputed = active.copy()
active_recomputed = active_recomputed.drop(columns=[c for c in strict_cols if c in active_recomputed.columns])
active_recomputed = active_recomputed.merge(raw_counts[key + strict_cols], on=key, how='left')
for c in strict_cols:
    active_recomputed[c] = active_recomputed[c].fillna(0)
active_recomputed = active_recomputed[active.columns]
active_recomputed.to_csv(ACTIVE_OUT, index=False)
ACTIVE_INPUT.parent.mkdir(parents=True, exist_ok=True)
active_recomputed.to_csv(ACTIVE_INPUT, index=False)
m2 = compare(active, active_recomputed, key, strict_cols, ACTIVE_DETAIL, ACTIVE_MISMATCH)
raw_stats_df = pd.DataFrame(raw_stats)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE = Path(__file__).resolve().parents[1]
INPUT = BASE / 'derived' / 'n200_qabase_scale_extension_input_recomputed_from_raw_strict_counts.csv'
OUT_DIR = BASE / 'curated'
OUT_PDF = OUT_DIR / 'n200_qabase_extension.pdf'
OUT_PNG = OUT_DIR / 'n200_qabase_extension.png'
OUT_SUMMARY = BASE / 'derived' / 'n200_qabase_extension_summary.csv'
ORDER = ['QA-Base', 'DC-OSQA', 'BC-OSQA', 'SA-Max']
STYLE = {'QA-Base': {'color': '#666666', 'marker': 'o'}, 'DC-OSQA': {'color': '#2F6F8F', 'marker': 's'}, 'BC-OSQA': {'color': '#6F8E72', 'marker': '^'}, 'SA-Max': {'color': '#B07A4F', 'marker': 'D'}}

def clean_axis(ax: plt.Axes) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out')

def main() -> None:
    df = pd.read_csv(INPUT)
    required = {'instance', 'solver_canonical', 'Unique_strict', 'Families_complete_linkage'}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f'missing columns: {missing}')
    metrics = [('Unique_strict', 'Unique certified candidates'), ('Families_complete_linkage', 'Complete-linkage families')]
    summary_rows = []
    (fig, axes) = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
    x = np.arange(len(ORDER), dtype=float)
    for (panel_idx, (ax, (metric, ylabel))) in enumerate(zip(axes, metrics)):
        for (xpos, solver) in enumerate(ORDER):
            values = pd.to_numeric(df.loc[df['solver_canonical'].eq(solver), metric], errors='coerce').dropna()
            jitter = np.linspace(-0.11, 0.11, len(values))
            ax.scatter(np.full(len(values), xpos, dtype=float) + jitter, values, s=20, color=STYLE[solver]['color'], alpha=0.5, edgecolor='none', zorder=2)
            mean_value = float(values.mean())
            ax.scatter([xpos], [mean_value], s=82, color=STYLE[solver]['color'], marker=STYLE[solver]['marker'], edgecolor='white', linewidth=0.9, zorder=5)
            summary_rows.append({'metric': metric, 'solver': solver, 'instances': len(values), 'mean': mean_value, 'median': float(values.median()), 'zero_count': int((values == 0).sum())})
        ax.set_xticks(x)
        ax.set_xticklabels(ORDER, rotation=15)
        ax.set_xlim(-0.4, len(ORDER) - 0.6)
        ax.set_ylabel(ylabel)
        ax.text(-0.14, 1.04, f'({chr(97 + panel_idx)})', transform=ax.transAxes, fontsize=12, fontweight='bold')
        clean_axis(ax)
    pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches='tight')
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    plt.close(fig)
if __name__ == '__main__':
    main()
