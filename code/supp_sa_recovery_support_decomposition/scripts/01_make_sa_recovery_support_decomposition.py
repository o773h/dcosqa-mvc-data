#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
BUNDLE = Path(__file__).resolve().parents[1]
ROOT = BUNDLE.parents[1]
RAWROOT = ROOT / 'data/raw'
analysis_ready = pd.read_csv(ROOT / 'data/metadata/sa_recovery_n150_source_map.csv', low_memory=False)
COMPARISON_SPEC = ROOT / 'data/metadata/sa_recovery_n150_comparison_spec.csv'
comparison_spec = pd.read_csv(COMPARISON_SPEC, low_memory=False)
DERIVED = BUNDLE / 'derived'
DERIVED.mkdir(exist_ok=True)
required_ar = ['N', 'instance', 'solver_label', 'source_dataset', 'solver_raw', 'K_exact']
missing = [c for c in required_ar if c not in analysis_ready.columns]
if missing:
    raise SystemExit('analysis_ready missing required cols: ' + ','.join(missing))

def load_raw(sd):
    p = RAWROOT / sd / 'raw_solutions.csv'
    if not p.exists():
        raise FileNotFoundError(f'missing raw source: {sd}')
    d = pd.read_csv(p, low_memory=False)
    for c in ['Nodes', 'Instance', 'Size', 'Occurrences']:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors='coerce')
    if d['IsValid'].dtype != bool:
        d['IsValid'] = d['IsValid'].astype(str).str.lower().isin(['true', '1', 'yes'])
    return d
raw_cache = {}
sets = {}
count_rows = []
for (_, r) in analysis_ready[required_ar].drop_duplicates().iterrows():
    sd = str(r['source_dataset'])
    solver_raw = str(r['solver_raw'])
    label = str(r['solver_label'])
    n = int(r['N'])
    inst = int(r['instance'])
    k = float(r['K_exact'])
    if sd not in raw_cache:
        raw_cache[sd] = load_raw(sd)
    d = raw_cache[sd]
    sub = d[d['Nodes'].eq(n) & d['Instance'].eq(inst) & d['Solver'].astype(str).eq(solver_raw)].copy()
    cert = sub[sub['IsValid'].eq(True) & pd.to_numeric(sub['Size'], errors='coerce').eq(k)].copy()
    bitset = set(cert['Bitstring'].dropna().astype(str).unique())
    sets[inst, label] = bitset
    count_rows.append({'instance': inst, 'solver_label': label, 'source_dataset': sd, 'solver_raw': solver_raw, 'raw_rows': len(sub), 'cert_rows': len(cert), 'Unique_strict_from_raw': len(bitset), 'Strict_occurrences_from_raw': pd.to_numeric(cert['Occurrences'], errors='coerce').fillna(0).sum() if len(cert) else 0})
raw_counts = pd.DataFrame(count_rows)
rows = []
for (_, r) in comparison_spec.iterrows():
    inst = int(r['instance'])
    A = str(r['A'])
    B = str(r['B'])
    Aset = sets.get((inst, A), set())
    Bset = sets.get((inst, B), set())
    inter = Aset & Bset
    union = Aset | Bset
    au = len(Aset)
    bu = len(Bset)
    iu = len(inter)
    uu = len(union)
    rows.append({'instance': inst, 'A': A, 'B': B, 'A_unique': au, 'B_unique': bu, 'intersection': iu, 'union': uu, 'jaccard': iu / uu if uu else 0.0, 'overlap_over_min': iu / min(au, bu) if min(au, bu) else 0.0, 'union_gain_over_max': uu / max(au, bu) if max(au, bu) else 0.0, 'A_only': au - iu, 'B_only': bu - iu})
recomputed = pd.DataFrame(rows)
out_name = 'sa_recovery_n150_overlap_pairs_recomputed_from_raw_bitstrings.csv'
recomputed.to_csv(DERIVED / out_name, index=False)
BASE = Path(__file__).resolve().parents[1]
INPUT = BASE / 'derived/sa_recovery_n150_overlap_pairs_recomputed_from_raw_bitstrings.csv'
OUT_DIR = BASE / 'curated'
OUT_PDF = OUT_DIR / 'sa_recovery_support_decomposition.pdf'
OUT_PNG = OUT_DIR / 'sa_recovery_support_decomposition.png'
OUT_SUMMARY = BASE / 'derived' / 'sa_recovery_support_decomposition_summary.csv'
POINT_COLOR = '#8A8A8A'
MEAN_COLOR = '#2F6F8F'

def clean_axis(ax: plt.Axes) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out')

def short_label(a: str, b: str) -> str:
    repl = {'SA-Recovery 10k': 'SA-R 10k', 'SA-Recovery 50k': 'SA-R 50k', 'DC-OSQA': 'DC', 'SA-Max': 'SA'}
    return f'{repl.get(a, a)}\nvs\n{repl.get(b, b)}'

def main() -> None:
    df = pd.read_csv(INPUT)
    required = {'instance', 'A', 'B', 'jaccard', 'union_gain_over_max', 'A_only', 'B_only'}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f'missing columns: {missing}')
    df['comparison'] = df['A'].astype(str) + ' vs ' + df['B'].astype(str)
    comparisons = df[['A', 'B', 'comparison']].drop_duplicates().reset_index(drop=True)
    summary_rows = []
    metrics = [('jaccard', 'Jaccard overlap'), ('union_gain_over_max', 'Union support / larger individual support')]
    (fig, axes) = plt.subplots(1, 2, figsize=(10.4, 4.0), constrained_layout=True)
    x = np.arange(len(comparisons), dtype=float)
    for (panel_idx, (ax, (metric, ylabel))) in enumerate(zip(axes, metrics)):
        for (xpos, row) in comparisons.iterrows():
            mask = df['comparison'].eq(row['comparison'])
            values = pd.to_numeric(df.loc[mask, metric], errors='coerce').dropna()
            jitter = np.linspace(-0.13, 0.13, len(values))
            ax.scatter(np.full(len(values), xpos, dtype=float) + jitter, values, s=18, color=POINT_COLOR, alpha=0.55, edgecolor='none', zorder=2)
            mean_value = float(values.mean())
            ax.scatter([xpos], [mean_value], s=76, color=MEAN_COLOR, marker='D', edgecolor='white', linewidth=0.9, zorder=5)
            summary_rows.append({'metric': metric, 'A': row['A'], 'B': row['B'], 'pairs': len(values), 'mean': mean_value, 'median': float(values.median())})
        ax.set_xticks(x)
        ax.set_xticklabels([short_label(row.A, row.B) for row in comparisons.itertuples()], fontsize=8)
        ax.set_xlim(-0.45, len(comparisons) - 0.55)
        ax.set_ylabel(ylabel)
        ax.text(-0.12, 1.04, f'({chr(97 + panel_idx)})', transform=ax.transAxes, fontsize=12, fontweight='bold')
        clean_axis(ax)
    pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches='tight')
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    plt.close(fig)
if __name__ == '__main__':
    main()
