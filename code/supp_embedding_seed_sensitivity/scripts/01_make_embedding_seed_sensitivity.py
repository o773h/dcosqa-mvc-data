#!/usr/bin/env python3
from pathlib import Path
import hashlib, re
import pandas as pd
import matplotlib
matplotlib.rcParams.update({'font.size':8.5,'axes.labelsize':8.5,'xtick.labelsize':7.5,'ytick.labelsize':7.5})
ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / 'code/supp_embedding_seed_sensitivity/raw/raw_solutions.csv'
AUD = Path(__file__).resolve().parents[1] / 'audit'
LABEL = 'embedding_seed_sensitivity_n150_v2'
KEYWORDS = ['embedding', 'seed', 'sensitivity']

def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda : handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def bitstring_cols(cols):
    out = []
    for c in cols:
        x = c.lower()
        if x in {'bitstring', 'solution', 'sample', 'state'} or 'bitstring' in x or 'solution' in x:
            out.append(c)
    return out

def cert_cols(cols):
    low = {c.lower(): c for c in cols}
    wanted = ['isvalid', 'valid', 'size', 'k_exact', 'k', 'occurrences', 'count', 'energy', 'objective']
    return [low[w] for w in wanted if w in low]

def bitstring_looks_valid(series):
    vals = series.dropna().astype(str).head(200).tolist()
    ok = 0
    for v in vals:
        vv = v.strip().replace(' ', '')
        if re.fullmatch('[01]{10,}', vv) or re.fullmatch('[\\[\\(]?[01,\\s]+[\\]\\)]?', v.strip()):
            ok += 1
    return (ok, len(vals))
if not RAW.exists():
    raise SystemExit(f'missing raw source: {RAW}')
raw_head = pd.read_csv(RAW, nrows=0, low_memory=False)
raw_cols = list(raw_head.columns)
bcols = bitstring_cols(raw_cols)
ccols = cert_cols(raw_cols)
if not bcols:
    raise SystemExit('FAIL no bitstring/solution column in raw source')
if not ccols:
    raise SystemExit('FAIL no certification/count/energy-like columns in raw source')
sample = pd.read_csv(RAW, nrows=1000, low_memory=False)
valid_counts = []
for c in bcols:
    (ok, total) = bitstring_looks_valid(sample[c])
    valid_counts.append((c, ok, total))
if not any((ok > 0 for (_, ok, _) in valid_counts)):
    raise SystemExit('FAIL bitstring columns exist but sampled values do not look like bitstrings')
keylike_cols = []
for c in raw_cols:
    cl = c.lower()
    if any((k in cl for k in ['n', 'instance', 'graph', 'solver', 'source', 'embedding', 'seed', 'schedule', 'topology', 'control', 'alpha', 'rep'])):
        keylike_cols.append(c)
keyword_hits = []
hay = (' '.join(raw_cols) + ' ' + str(RAW)).lower()
sample_text = sample.head(50).astype(str).to_string().lower()
for k in KEYWORDS:
    if k.lower() in hay or k.lower() in sample_text:
        keyword_hits.append(k)
out = AUD / 'candidate_bitstring_raw_source_audit.csv'
import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import wilcoxon
ROOT = Path(__file__).resolve().parents[3]
B = Path(__file__).resolve().parents[1]
RAW = B / 'raw/raw_solutions.csv'
KMAP = ROOT / 'data/metadata/ba_m2_n150_kexact_reference.csv'
DER = B / 'derived'
AUD = B / 'audit'
DER.mkdir(exist_ok=True)
LABEL = 'embedding_seed_sensitivity_n150_v2'
TAU_FRAC = 0.1

def nb(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(['true', '1', 'yes', 'y'])

def solver_norm(x):
    s = str(x)
    sl = s.lower()
    if sl in ['qa-base', 'qabase', 'qa_base', 'qa base']:
        return 'QA-Base'
    if sl in ['dc-osqa', 'dcosqa', 'dc_osqa', 'dc osqa']:
        return 'DC-OSQA'
    if 'qa-base' in sl or 'qabase' in sl:
        return 'QA-Base'
    if 'dc' in sl:
        return 'DC-OSQA'
    return s

def count_complete_linkage(bitstrings, n):
    vals = sorted(set(map(str, bitstrings)))
    if len(vals) == 0:
        return 0
    if len(vals) == 1:
        return 1
    x = np.array([[1 if ch == '1' else 0 for ch in b] for b in vals], dtype=np.uint8)
    d = pdist(x, metric='hamming') * int(n)
    z = linkage(d, method='complete')
    labels = fcluster(z, t=TAU_FRAC * int(n), criterion='distance')
    return int(len(set(labels)))

def safe_wilcoxon(d, q, alt):
    diff = np.asarray(d, dtype=float) - np.asarray(q, dtype=float)
    if len(diff) == 0 or np.allclose(diff, 0):
        return np.nan
    return float(wilcoxon(diff, alternative=alt, zero_method='wilcox').pvalue)

def paired_stats(pair, unit):
    rows = []
    for metric in ['Unique_strict', 'Strict_occurrences', 'Families_complete_linkage']:
        q = pair[f'QA-Base__{metric}'].astype(float).to_numpy()
        d = pair[f'DC-OSQA__{metric}'].astype(float).to_numpy()
        gain = d - q
        rows.append({'unit': unit, 'metric': metric, 'n_units': int(len(pair)), 'wins': int((gain > 0).sum()), 'losses': int((gain < 0).sum()), 'ties': int((gain == 0).sum()), 'qa_mean': float(np.mean(q)) if len(q) else np.nan, 'dc_mean': float(np.mean(d)) if len(d) else np.nan, 'mean_gain': float(np.mean(gain)) if len(gain) else np.nan, 'median_gain': float(np.median(gain)) if len(gain) else np.nan, 'wilcoxon_two_sided_p': safe_wilcoxon(d, q, 'two-sided'), 'wilcoxon_greater_p': safe_wilcoxon(d, q, 'greater')})
    return rows
if not RAW.exists():
    raise SystemExit(f'missing raw file: {RAW}')
if not KMAP.exists():
    raise SystemExit(f'missing K_exact reference: {KMAP}')
df = pd.read_csv(RAW, low_memory=False)
kmap = pd.read_csv(KMAP)
kmap = kmap[kmap['N'].eq(150)][['instance', 'K_exact']].rename(
    columns={'instance': 'Instance', 'K_exact': 'K_exact_inferred'}
)
df['IsValid_bool'] = nb(df['IsValid'])
df['Solver2'] = df['Solver_Canonical'].map(solver_norm)
df = df.merge(kmap, on='Instance', how='left')
exact = df[df['IsValid_bool'] & df['Size'].eq(df['K_exact_inferred'])].copy()
key = ['Nodes', 'Instance', 'Solver2', 'EmbeddingRep', 'EmbeddingSeed']
registry = df.groupby(key, dropna=False).agg(
    K_exact=('K_exact_inferred', 'first')
).reset_index()
counts = exact.groupby(key, dropna=False).agg(
    Exact_rows=('Bitstring', 'size'),
    Unique_strict=('Bitstring', pd.Series.nunique),
    Strict_occurrences=('Occurrences', 'sum')
).reset_index()
fam_rows = []
for (key_vals, sub) in exact.groupby(key, dropna=False):
    rec = dict(zip(key, key_vals))
    n = int(sub['Nodes'].iloc[0])
    rec['Families_complete_linkage'] = count_complete_linkage(sub['Bitstring'], n)
    rec['Exact_unique_bitstrings_for_families'] = int(sub['Bitstring'].nunique())
    fam_rows.append(rec)
families = pd.DataFrame(fam_rows)
by_solver = registry.merge(counts, on=key, how='left').merge(families, on=key, how='left')
zero_cols = [
    'Exact_rows', 'Unique_strict', 'Strict_occurrences',
    'Families_complete_linkage', 'Exact_unique_bitstrings_for_families'
]
by_solver[zero_cols] = by_solver[zero_cols].fillna(0)
by_solver[zero_cols] = by_solver[zero_cols].astype(int)
by_solver = by_solver.rename(columns={'Solver2': 'Solver_Canonical'})
by_solver_path = DER / f'{LABEL}_exact_by_solver_embedding.csv'
by_solver.to_csv(by_solver_path, index=False)
pair = by_solver.pivot_table(index=['Instance', 'EmbeddingRep', 'EmbeddingSeed'], columns='Solver_Canonical', values=['Unique_strict', 'Strict_occurrences', 'Families_complete_linkage'], aggfunc='first')
pair.columns = [f'{solver}__{metric}' for (metric, solver) in pair.columns]
pair = pair.reset_index()
required = [f'{solver}__{metric}' for solver in ['QA-Base', 'DC-OSQA'] for metric in ['Unique_strict', 'Strict_occurrences', 'Families_complete_linkage']]
pair = pair.dropna(subset=required).copy()
pair_path = DER / f'{LABEL}_exact_pair_table.csv'
pair.to_csv(pair_path, index=False)
stats = pd.DataFrame(paired_stats(pair, 'embedding_instance') + paired_stats(pair.groupby('Instance', dropna=False)[required].mean().reset_index(), 'graph_mean_over_5_embeddings'))
stats_path = DER / f'{LABEL}_exact_paired_statistics.csv'
stats.to_csv(stats_path, index=False)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'code/supp_embedding_seed_sensitivity'
DER = BASE / 'derived'
CUR = BASE / 'curated'
AUD = BASE / 'audit'
PAIR = DER / 'embedding_seed_sensitivity_n150_v2_exact_pair_table.csv'
OUT_PDF = CUR / 'embedding_seed_sensitivity.pdf'
OUT_PNG = CUR / 'embedding_seed_sensitivity.png'
CUR.mkdir(parents=True, exist_ok=True)
GRAPH_MEAN_COLOR = '#666666'
REP_COLORS = {1: '#0072B2', 2: '#E69F00', 3: '#009E73', 4: '#D55E00', 5: '#CC79A7'}
REP_MARKERS = {1: 'o', 2: 'o', 3: 'o', 4: 'o', 5: 'o'}
ZERO = '#555555'
GRID = '#B0B0B0'

def clean_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', color=GRID, alpha=0.24, linewidth=0.8)
    ax.set_axisbelow(True)

def main():
    df = pd.read_csv(PAIR)
    required = {'Instance', 'EmbeddingRep', 'DC-OSQA__Unique_strict', 'QA-Base__Unique_strict', 'DC-OSQA__Families_complete_linkage', 'QA-Base__Families_complete_linkage'}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f'missing required columns: {sorted(missing)}')
    df = df.copy()
    df['unique_gain'] = df['DC-OSQA__Unique_strict'] - df['QA-Base__Unique_strict']
    df['family_gain'] = df['DC-OSQA__Families_complete_linkage'] - df['QA-Base__Families_complete_linkage']
    graph_means = df.groupby('Instance', as_index=False).agg(unique_gain=('unique_gain', 'mean'), family_gain=('family_gain', 'mean'), embeddings=('EmbeddingRep', 'nunique')).sort_values('Instance')
    expected_instances = set(range(20))
    observed_instances = set(graph_means['Instance'].astype(int))
    if observed_instances != expected_instances:
        raise SystemExit(f'expected graph instances 0--19; got {sorted(observed_instances)}')
    if not (graph_means['unique_gain'] > 0).all():
        bad = graph_means.loc[graph_means['unique_gain'] <= 0, ['Instance', 'unique_gain']]
        raise SystemExit('non-positive graph mean unique gain:\n' + bad.to_string(index=False))
    if not (graph_means['family_gain'] > 0).all():
        bad = graph_means.loc[graph_means['family_gain'] <= 0, ['Instance', 'family_gain']]
        raise SystemExit('non-positive graph mean family gain:\n' + bad.to_string(index=False))
    rep_values = sorted(df['EmbeddingRep'].dropna().astype(int).unique())
    offsets = np.linspace(-0.24, 0.24, len(rep_values))
    offset_map = dict(zip(rep_values, offsets))
    ninst = 20
    (fig, axes) = plt.subplots(2, 1, figsize=(7.6, 4.6), sharex=True, constrained_layout=False)
    panel_specs = [(axes[0], 'unique_gain', 'Unique certified candidates', 'Gain, DC-OSQA minus QA-Base'), (axes[1], 'family_gain', 'Complete-linkage families', 'Gain, DC-OSQA minus QA-Base')]
    for (panel_label, (ax, gain_col, title, ylabel)) in zip(['(a)', '(b)'], panel_specs):
        for i in range(ninst):
            if i % 2 == 0:
                ax.axvspan(i - 0.5, i + 0.5, color='#F2F4F7', zorder=0)
        for bnd in range(ninst - 1):
            ax.axvline(bnd + 0.5, color='#D9DEE5', linewidth=0.6, zorder=1)
        ax.axhline(0, color=ZERO, linestyle='--', linewidth=1.0, alpha=0.8, zorder=1)
        for rep in rep_values:
            sub = df[df['EmbeddingRep'].astype(int).eq(rep)]
            ax.scatter(sub['Instance'] + offset_map[rep], sub[gain_col], s=16, marker=REP_MARKERS[rep], facecolor=REP_COLORS[rep], edgecolor='#333333', linewidth=0.35, alpha=0.8, zorder=2)
        ax.scatter(graph_means['Instance'], graph_means[gain_col], s=46, marker='D', facecolor=GRAPH_MEAN_COLOR, edgecolor='white', linewidth=1.0, zorder=4)
        ax.set_title(title, fontsize=9.5, pad=4)
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.6, ninst - 0.4)
        clean_axis(ax)
        ax.text(-0.055, 1.0, panel_label, transform=ax.transAxes, fontsize=11, fontweight='bold', ha='left', va='bottom')
    axes[1].set_xticks(range(ninst))
    axes[1].set_xticklabels([str(i) for i in range(ninst)], fontsize=7.5)
    axes[1].set_xlabel('Graph instance')
    legend_handles = [Line2D([0], [0], marker=REP_MARKERS[rep], linestyle='none', markerfacecolor=REP_COLORS[rep], markeredgecolor='#333333', markeredgewidth=0.4, markersize=5.8, label=f'Embedding rep {rep}') for rep in sorted(REP_COLORS)]
    legend_handles.append(Line2D([0], [0], marker='D', linestyle='none', markerfacecolor=GRAPH_MEAN_COLOR, markeredgecolor='white', markeredgewidth=0.8, markersize=10, label='Graph mean'))
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.005), ncol=6, frameon=False, fontsize=7.5)
    fig.subplots_adjust(top=0.9, bottom=0.085, left=0.065, right=0.99, hspace=0.17)
    fig.savefig(OUT_PDF, bbox_inches='tight')
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    plt.close(fig)
    graph_means.to_csv(DER / 'embedding_seed_sensitivity_graph_means_used_for_plot.csv', index=False)
    audit_lines = ['status: PASS', f'input: {PAIR.relative_to(ROOT)}', f'input_rows: {len(df)}', f'input_sha256: {sha256(PAIR)}', f'graph_instances: {len(graph_means)}', f'embedding_pairs: {len(df)}', f"unique_positive_graph_means: {(graph_means['unique_gain'] > 0).sum()}", f"family_positive_graph_means: {(graph_means['family_gain'] > 0).sum()}", f'output_pdf: {OUT_PDF.relative_to(ROOT)}', f'output_png: {OUT_PNG.relative_to(ROOT)}', f'output_pdf_sha256: {sha256(OUT_PDF)}', f'output_png_sha256: {sha256(OUT_PNG)}']
if __name__ == '__main__':
    main()
