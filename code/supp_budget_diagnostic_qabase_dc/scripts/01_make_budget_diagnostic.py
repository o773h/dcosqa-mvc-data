#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
BUNDLE = Path(__file__).resolve().parents[1]
ROOT = BUNDLE.parents[1]
RAWROOT = ROOT / 'data/raw'
TARGET_REL = 'analysis_ready/budget_n150_analysis_ready.csv'
TARGET = BUNDLE / TARGET_REL
DERIVED = BUNDLE / 'derived'
DERIVED.mkdir(exist_ok=True)
target = pd.read_csv(TARGET, low_memory=False)
strict_cols = [c for c in ['Unique_strict', 'Strict_occurrences', 'zero_yield'] if c in target.columns]
if 'Unique_strict' not in strict_cols:
    raise SystemExit('target lacks Unique_strict')
required = ['instance', 'solver_canonical', 'source_datasets', 'budget_label', 'K_exact']
missing = [c for c in required if c not in target.columns]
if missing:
    raise SystemExit('missing cols: ' + ','.join(missing))

def split_sources(x):
    return [s.strip() for s in str(x).split(',') if s.strip()]

def infer_solver_raw(row):
    solver = str(row['solver_canonical'])
    if solver == 'QA-Base':
        return 'QA-Base'
    if solver == 'DC-OSQA':
        return 'DC-Offset(s=0.1)'
    if solver == 'BC-OSQA':
        return 'BC-Offset(s=0.1)'
    return solver

def load_raw(sd):
    p = RAWROOT / sd / 'raw_solutions.csv'
    if not p.exists():
        return (None, p)
    d = pd.read_csv(p, low_memory=False)
    for c in ['Nodes', 'Instance', 'Size', 'Occurrences']:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors='coerce')
    if 'Occurrences' not in d.columns:
        d['Occurrences'] = 1
    if d['IsValid'].dtype != bool:
        d['IsValid'] = d['IsValid'].astype(str).str.lower().isin(['true', '1', 'yes'])
    return (d, p)
raw_cache = {}
rows = []
raw_used = []
for (_, tr) in target[required].drop_duplicates().iterrows():
    inst = int(tr['instance'])
    k = float(tr['K_exact'])
    solver_canonical = str(tr['solver_canonical'])
    solver_raw = infer_solver_raw(tr)
    budget_label = str(tr['budget_label'])
    bits = {}
    occ = 0.0
    raw_rows = 0
    cert_rows = 0
    source_status = []
    for sd in split_sources(tr['source_datasets']):
        if sd not in raw_cache:
            raw_cache[sd] = load_raw(sd)
        (d, p) = raw_cache[sd]
        if d is None:
            source_status.append(f'{sd}:missing')
            raw_used.append({'source_dataset': sd, 'status': 'missing', 'path': p.as_posix()})
            continue
        source_status.append(f'{sd}:used')
        raw_used.append({'source_dataset': sd, 'status': 'used', 'path': p.relative_to(ROOT).as_posix(), 'rows': len(d)})
        sub = d[d['Instance'].eq(inst) & d['Solver'].astype(str).eq(solver_raw)].copy()
        if 'Nodes' in sub.columns:
            sub = sub[sub['Nodes'].eq(150)]
        raw_rows += len(sub)
        cert = sub[sub['IsValid'].eq(True) & pd.to_numeric(sub['Size'], errors='coerce').eq(k)].copy()
        cert_rows += len(cert)
        for (_, r) in cert.iterrows():
            b = str(r['Bitstring'])
            o = float(r['Occurrences']) if not pd.isna(r['Occurrences']) else 0.0
            bits[b] = bits.get(b, 0.0) + o
            occ += o
    u = len(bits)
    rows.append({'instance': inst, 'solver_canonical': solver_canonical, 'source_datasets': tr['source_datasets'], 'budget_label': budget_label, 'Unique_strict': int(u), 'Strict_occurrences': float(occ), 'zero_yield': bool(u == 0), 'solver_raw_inferred': solver_raw, 'raw_rows': raw_rows, 'cert_rows': cert_rows, 'source_status': ';'.join(source_status)})
raw_counts = pd.DataFrame(rows)
key = ['source_datasets', 'instance', 'solver_canonical', 'budget_label']
recomputed = target.copy()
recomputed = recomputed.drop(columns=[c for c in strict_cols if c in recomputed.columns])
recomputed = recomputed.merge(raw_counts[key + strict_cols], on=key, how='left')
for c in strict_cols:
    if c == 'zero_yield':
        recomputed[c] = recomputed[c].fillna(pd.to_numeric(recomputed['Unique_strict'], errors='coerce').fillna(0).eq(0))
    else:
        recomputed[c] = recomputed[c].fillna(0)
recomputed = recomputed[target.columns]
out = DERIVED / 'budget_n150_analysis_ready_recomputed_from_raw_strict_counts.csv'
recomputed.to_csv(out, index=False)
TARGET.parent.mkdir(parents=True, exist_ok=True)
recomputed.to_csv(TARGET, index=False)
m = target[key + strict_cols].merge(recomputed[key + strict_cols], on=key, how='outer', suffixes=('_target', '_recomputed'), indicator=True)
bad = m['_merge'].ne('both')
for c in strict_cols:
    if c == 'zero_yield':
        a = m[f'{c}_target'].astype(str).str.lower().isin(['true', '1'])
        b = m[f'{c}_recomputed'].astype(str).str.lower().isin(['true', '1'])
        bad = bad | (a != b)
    else:
        a = pd.to_numeric(m[f'{c}_target'], errors='coerce')
        b = pd.to_numeric(m[f'{c}_recomputed'], errors='coerce')
        bad = bad | ~np.isclose(a.fillna(-999999999), b.fillna(-999999999), rtol=1e-12, atol=1e-12)
m['mismatch'] = bad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE = Path(__file__).resolve().parents[1]
INPUT = BASE / 'derived' / 'budget_n150_analysis_ready_recomputed_from_raw_strict_counts.csv'
OUT_DIR = BASE / 'curated'
OUT_PDF = OUT_DIR / 'budget_diagnostic_qabase_dc.pdf'
OUT_PNG = OUT_DIR / 'budget_diagnostic_qabase_dc.png'
OUT_SUMMARY = BASE / 'derived' / 'budget_diagnostic_summary.csv'
ORDER = ['QA-Base 10k', 'DC-OSQA 10k', 'QA-Base 50k']
STYLE = {'QA-Base 10k': {'color': '#707070', 'marker': 'o'}, 'DC-OSQA 10k': {'color': '#2F6F8F', 'marker': 's'}, 'QA-Base 50k': {'color': '#A67A4F', 'marker': '^'}}

def clean_axis(ax: plt.Axes) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out')

def main() -> None:
    df = pd.read_csv(INPUT)
    required = {'instance', 'budget_label', 'Unique_strict', 'Families_complete_linkage'}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f'missing columns: {missing}')
    metrics = [('Unique_strict', 'Unique certified candidates'), ('Families_complete_linkage', 'Complete-linkage families')]
    summary_rows = []
    (fig, axes) = plt.subplots(1, 2, figsize=(8.8, 3.8), constrained_layout=True)
    x = np.arange(len(ORDER), dtype=float)
    for (panel_idx, (ax, (metric, ylabel))) in enumerate(zip(axes, metrics)):
        for (xpos, label) in enumerate(ORDER):
            values = pd.to_numeric(df.loc[df['budget_label'].eq(label), metric], errors='coerce').dropna()
            jitter = np.linspace(-0.1, 0.1, len(values))
            ax.scatter(np.full(len(values), xpos, dtype=float) + jitter, values, s=20, color=STYLE[label]['color'], alpha=0.5, edgecolor='none', zorder=2)
            mean_value = float(values.mean())
            ax.scatter([xpos], [mean_value], s=82, color=STYLE[label]['color'], marker=STYLE[label]['marker'], edgecolor='white', linewidth=0.9, zorder=5)
            summary_rows.append({'metric': metric, 'budget_label': label, 'instances': len(values), 'mean': mean_value, 'median': float(values.median()), 'min': float(values.min()), 'max': float(values.max())})
        ax.set_xticks(x)
        ax.set_xticklabels(['QA 10k', 'DC 10k', 'QA 50k'])
        ax.set_xlim(-0.35, len(ORDER) - 0.65)
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
