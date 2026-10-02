#!/usr/bin/env python3
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
try:
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage, fcluster
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False
ROOT = Path(__file__).resolve().parents[3]
B = ROOT / 'code/main_qabase_vs_dcosqa'
CUR = B / 'curated'
DER = B / 'derived'
for d in [CUR, DER]:
    d.mkdir(parents=True, exist_ok=True)
MAIN_NS = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150]
LARGE_NS = [70, 80, 90, 100, 120, 150]
QA_RAW = ROOT / 'data/raw/qa_base_main_n20_150/raw_solutions.csv'
DC_RAW = ROOT / 'data/raw/main_ba_m2_dcosqa_offset01/raw_candidate_bitstrings.csv'

def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda : handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def read_needed(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    hdr = pd.read_csv(path, nrows=0).columns.tolist()
    use = []
    for c in ['N', 'Nodes', 'instance', 'Instance', 'Solver', 'solver', 'Solver_Canonical', 'solver_canonical', 'solver_raw', 'Bitstring', 'Energy', 'Size', 'K_exact', 'IsValid', 'Occurrences', 'Chain_Break_Frac', 'chain_break_mean', 'Edges', 'Graph_Seed', 'OffsetScale', 'source_dataset', 'source_file']:
        if c in hdr and c not in use:
            use.append(c)
    df = pd.read_csv(path, usecols=use, dtype={'Bitstring': 'string'}, low_memory=False)
    if 'N' not in df.columns and 'Nodes' in df.columns:
        df['N'] = df['Nodes']
    if 'instance' not in df.columns and 'Instance' in df.columns:
        df['instance'] = df['Instance']
    solver_col = next((c for c in ['solver_canonical', 'Solver_Canonical', 'Solver', 'solver', 'solver_raw'] if c in df.columns), None)
    if solver_col is None:
        df['solver_raw'] = ''
    else:
        df['solver_raw'] = df[solver_col].astype(str)
    if 'source_dataset' not in df.columns:
        df['source_dataset'] = path.parent.name
    if 'source_file' not in df.columns:
        df['source_file'] = path.name
    if 'Occurrences' not in df.columns:
        df['Occurrences'] = 1
    if 'IsValid' not in df.columns:
        df['IsValid'] = True
    if 'K_exact' not in df.columns:
        df['K_exact'] = np.nan
    if 'Size' not in df.columns:
        df['Size'] = np.nan
    df['N'] = pd.to_numeric(df['N'], errors='coerce').astype('Int64')
    df['instance'] = pd.to_numeric(df['instance'], errors='coerce').astype('Int64')
    df['Size'] = pd.to_numeric(df['Size'], errors='coerce')
    df['K_exact'] = pd.to_numeric(df['K_exact'], errors='coerce')
    df['Occurrences'] = pd.to_numeric(df['Occurrences'], errors='coerce').fillna(0)
    if df['IsValid'].dtype == bool:
        df['IsValid'] = df['IsValid'].astype(bool)
    else:
        df['IsValid'] = df['IsValid'].astype(str).str.lower().isin(['true', '1', 'yes', 'y'])
    if 'Bitstring' not in df.columns:
        raise SystemExit(f'{path} lacks Bitstring')
    df['Bitstring'] = df['Bitstring'].astype(str)
    if 'Chain_Break_Frac' in df.columns:
        df['chain_break'] = pd.to_numeric(df['Chain_Break_Frac'], errors='coerce')
    elif 'chain_break_mean' in df.columns:
        df['chain_break'] = pd.to_numeric(df['chain_break_mean'], errors='coerce')
    else:
        df['chain_break'] = np.nan
    return df

def canonical_solver_for_qa(df: pd.DataFrame) -> pd.Series:
    low = df['solver_raw'].astype(str).str.lower()
    return low.str.contains('qa-base|qabase|qa_base', regex=True, na=False)

def canonical_solver_for_dc01(df: pd.DataFrame) -> pd.Series:
    low = df['solver_raw'].astype(str).str.lower()
    keep = low.str.contains('dc-offset\\(s=0\\.1\\)|dc-offset\\(s=0\\.10\\)|dc-osqa', regex=True, na=False)
    if 'OffsetScale' in df.columns:
        off = pd.to_numeric(df['OffsetScale'], errors='coerce')
        keep = keep | low.str.contains('dc', na=False) & off.sub(0.1).abs().lt(1e-09)
    return keep

def infer_k_and_certify(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    kmap = df[df['IsValid'].eq(True) & df['Size'].notna()].groupby(['N', 'instance'], dropna=False)['Size'].min().reset_index().rename(columns={'Size': 'K_exact_inferred'})
    df = df.merge(kmap, on=['N', 'instance'], how='left')
    df['K_exact_final'] = df['K_exact'].fillna(df['K_exact_inferred'])
    return df[df['IsValid'].eq(True) & df['Size'].eq(df['K_exact_final'])].copy()

def hamming(a: str, b: str) -> int:
    return sum((x != y for (x, y) in zip(a, b)))

def complete_linkage_families(bitstrings: list[str], tau: int) -> int:
    bs = sorted(set(map(str, bitstrings)))
    n = len(bs)
    if n <= 1:
        return n
    if HAVE_SCIPY:
        arr = np.array([[1 if ch == '1' else 0 for ch in x] for x in bs], dtype=np.int8)
        dist = pdist(arr, metric='hamming') * arr.shape[1]
        Z = linkage(dist, method='complete')
        labels = fcluster(Z, t=float(tau), criterion='distance')
        return int(len(set(labels)))
    clusters = []
    for x in bs:
        placed = False
        for cl in clusters:
            if all((hamming(x, y) <= tau for y in cl)):
                cl.append(x)
                placed = True
                break
        if not placed:
            clusters.append([x])
    return len(clusters)

def entropy_from_counts(counts) -> float:
    arr = np.asarray(counts, dtype=float)
    arr = arr[arr > 0]
    if arr.size == 0:
        return 0.0
    p = arr / arr.sum()
    return float(-(p * np.log2(p)).sum())

def mean_pairwise_hamming(bitstrings: list[str]) -> float:
    bs = sorted(set(map(str, bitstrings)))
    if len(bs) < 2:
        return 0.0
    vals = []
    for i in range(len(bs)):
        for j in range(i + 1, len(bs)):
            vals.append(hamming(bs[i], bs[j]))
    return float(np.mean(vals)) if vals else 0.0
qa = read_needed(QA_RAW)
qa = qa[qa['N'].isin(MAIN_NS) & canonical_solver_for_qa(qa)].copy()
qa['solver_canonical'] = 'QA-Base'
qa['source_role'] = 'QA_CANONICAL_N20_150'
dc = read_needed(DC_RAW)
dc = dc[dc['N'].isin(MAIN_NS) & canonical_solver_for_dc01(dc)].copy()
dc['solver_canonical'] = 'DC-OSQA'
dc['source_role'] = 'DC_OFFSET01_N20_150'
raw = pd.concat([qa, dc], ignore_index=True)
cert = infer_k_and_certify(raw)
cert = cert.groupby(['N', 'instance', 'solver_canonical', 'Bitstring'], dropna=False).agg(Occurrences=('Occurrences', 'sum'), Size=('Size', 'first'), K_exact=('K_exact_final', 'first'), Energy=('Energy', 'min') if 'Energy' in cert.columns else ('Size', 'first'), chain_break=('chain_break', 'mean'), source_dataset=('source_dataset', lambda s: '|'.join(sorted(set(map(str, s))))), source_file=('source_file', lambda s: '|'.join(sorted(set(map(str, s))))), solver_raw=('solver_raw', lambda s: '|'.join(sorted(set(map(str, s))))), source_role=('source_role', lambda s: '|'.join(sorted(set(map(str, s)))))).reset_index()
cert.to_csv(DER / 'fig1_fig2_certified_bitstrings_recomputed_from_raw.csv', index=False)
rows = []
for ((N, inst, solver), g) in cert.groupby(['N', 'instance', 'solver_canonical'], dropna=False):
    bit_occ = g.groupby('Bitstring', dropna=False)['Occurrences'].sum().reset_index().sort_values('Bitstring')
    bs = bit_occ['Bitstring'].astype(str).tolist()
    occ = bit_occ['Occurrences'].astype(float).tolist()
    tau = max(1, int(round(0.1 * int(N))))
    rows.append({'analysis_scope': 'reference_qabase_ba_m2', 'graph_family_id': 'main_ba_m2', 'graph_model': 'BA', 'graph_m': 2, 'graph_m_semantics': 'BA attachment parameter m=2', 'N': int(N), 'instance': int(inst), 'solver_canonical': solver, 'source_dataset': '|'.join(sorted(set(map(str, g['source_dataset'])))), 'source_file': '|'.join(sorted(set(map(str, g['source_file'])))), 'solver_raw': '|'.join(sorted(set(map(str, g['solver_raw'])))), 'source_role': '|'.join(sorted(set(map(str, g['source_role'])))), 'K_exact': float(pd.to_numeric(g['K_exact'], errors='coerce').dropna().iloc[0]) if pd.to_numeric(g['K_exact'], errors='coerce').notna().any() else np.nan, 'Unique_strict': int(len(bs)), 'Strict_occurrences': int(np.sum(occ)), 'Families_complete_linkage': int(complete_linkage_families(bs, tau=tau)), 'Entropy_strict': entropy_from_counts(occ), 'MeanPairwiseHamming_strict': mean_pairwise_hamming(bs), 'zero_yield': False, 'chain_break_mean': float(pd.to_numeric(g['chain_break'], errors='coerce').mean()) if pd.to_numeric(g['chain_break'], errors='coerce').notna().any() else np.nan, 'chain_break_max': float(pd.to_numeric(g['chain_break'], errors='coerce').max()) if pd.to_numeric(g['chain_break'], errors='coerce').notna().any() else np.nan, 'Num_Reads': np.nan, 'Reads_Per_Call': np.nan, 'Budget_Chunks': np.nan, 'Embedding_Source': np.nan, 'Embedding_Status': np.nan, 'family_threshold': tau, 'bitstring_len': int(len(bs[0])) if bs else int(N), 'compute_status': 'OK', 'unique_match': True})
summary = pd.DataFrame(rows)
expected = []
for N in MAIN_NS:
    ninst = 10 if N <= 60 else 20
    for solver in ['QA-Base', 'DC-OSQA']:
        for inst in range(ninst):
            expected.append((N, inst, solver))
present = set(map(tuple, summary[['N', 'instance', 'solver_canonical']].values.tolist())) if len(summary) else set()
zero_rows = []
for (N, inst, solver) in expected:
    if (N, inst, solver) in present:
        continue
    tau = max(1, int(round(0.1 * int(N))))
    zero_rows.append({'analysis_scope': 'reference_qabase_ba_m2', 'graph_family_id': 'main_ba_m2', 'graph_model': 'BA', 'graph_m': 2, 'graph_m_semantics': 'BA attachment parameter m=2', 'N': int(N), 'instance': int(inst), 'solver_canonical': solver, 'source_dataset': 'zero_yield_raw_group_retained', 'source_file': 'raw candidate group had no certified candidates', 'solver_raw': solver, 'source_role': 'ZERO_YIELD_RETAINED', 'K_exact': np.nan, 'Unique_strict': 0, 'Strict_occurrences': 0, 'Families_complete_linkage': 0, 'Entropy_strict': 0.0, 'MeanPairwiseHamming_strict': 0.0, 'zero_yield': True, 'chain_break_mean': np.nan, 'chain_break_max': np.nan, 'Num_Reads': np.nan, 'Reads_Per_Call': np.nan, 'Budget_Chunks': np.nan, 'Embedding_Source': np.nan, 'Embedding_Status': np.nan, 'family_threshold': tau, 'bitstring_len': int(N), 'compute_status': 'ZERO_YIELD_RETAINED', 'unique_match': True})
if zero_rows:
    summary = pd.concat([summary, pd.DataFrame(zero_rows)], ignore_index=True)
summary = summary.sort_values(['N', 'solver_canonical', 'instance']).reset_index(drop=True)
summary.to_csv(CUR / 'fig1_main_only_selected_rows.csv', index=False)
summary.to_csv(CUR / 'fig1_qabase_clean_only_selected_rows.csv', index=False)
summary.to_csv(CUR / 'fig1_selected_rows_with_budget_extension.csv', index=False)
summary.to_csv(DER / 'fig1_fig2_main_qabase_dcosqa_from_raw.csv', index=False)

def sem(x):
    x = pd.to_numeric(x, errors='coerce').dropna()
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
fig1_byN = summary.groupby(['N', 'solver_canonical'], dropna=False).agg(Unique_strict_mean=('Unique_strict', 'mean'), Unique_strict_sem=('Unique_strict', sem), Families_complete_linkage_mean=('Families_complete_linkage', 'mean'), Families_complete_linkage_sem=('Families_complete_linkage', sem), n_instances=('instance', 'nunique')).reset_index().sort_values(['N', 'solver_canonical'])
fig1_byN.to_csv(CUR / 'fig1_main_by_N.csv', index=False)
large = summary[summary['N'].isin(LARGE_NS)].copy()
wide = large.pivot_table(index=['N', 'instance'], columns='solver_canonical', values=['Unique_strict', 'Families_complete_linkage'], aggfunc='first')
wide.columns = [f'{a}__{b}' for (a, b) in wide.columns]
wide = wide.reset_index()
gain_rows = []
for metric in ['Unique_strict', 'Families_complete_linkage']:
    dc_col = f'{metric}__DC-OSQA'
    qa_col = f'{metric}__QA-Base'
    for (_, r) in wide.iterrows():
        if pd.isna(r.get(dc_col)) or pd.isna(r.get(qa_col)):
            continue
        gain_rows.append({'comparison': 'DC-OSQA vs QA-Base', 'metric': metric, 'N': int(r['N']), 'instance': int(r['instance']), 'gain': float(r[dc_col] - r[qa_col]), 'DC-OSQA': float(r[dc_col]), 'QA-Base': float(r[qa_col])})
gains = pd.DataFrame(gain_rows)
gains.to_csv(CUR / 'fig2_paired_instance_gains.csv', index=False)
gain_sum = gains.groupby(['comparison', 'metric', 'N'], dropna=False).agg(mean_gain=('gain', 'mean'), sem_gain=('gain', sem), n_pairs=('instance', 'nunique')).reset_index().sort_values(['metric', 'N'])
gain_sum.to_csv(CUR / 'fig2_paired_gain_summary_by_N.csv', index=False)
key = summary.groupby(['N', 'solver_canonical'], dropna=False).agg(rows=('instance', 'size'), n_instances=('instance', 'nunique'), unique_mean=('Unique_strict', 'mean'), family_mean=('Families_complete_linkage', 'mean'), family_nonnull=('Families_complete_linkage', lambda s: s.notna().sum())).reset_index().sort_values(['N', 'solver_canonical'])
manifest = pd.DataFrame([{'role': 'QA-Base canonical N=20--150', 'path': QA_RAW.relative_to(ROOT).as_posix(), 'bytes': QA_RAW.stat().st_size, 'sha256': sha256(QA_RAW)}, {'role': 'DC-OSQA recorded label DC-Offset(s=0.1)', 'path': DC_RAW.relative_to(ROOT).as_posix(), 'bytes': DC_RAW.stat().st_size, 'sha256': sha256(DC_RAW)}])
zero_audit = summary[summary['zero_yield'].eq(True)][['N', 'instance', 'solver_canonical', 'compute_status']]
status = 'PASS' if len(summary) == 340 else 'CHECK'
if status != 'PASS':
    raise SystemExit(1)
B = Path(__file__).resolve().parents[1]
CUR = B / 'curated'
p = CUR / 'fig1_main_only_selected_rows.csv'
df = pd.read_csv(p, low_memory=False)
df['N'] = pd.to_numeric(df['N'], errors='coerce').astype('Int64')
df['instance'] = pd.to_numeric(df['instance'], errors='coerce').astype('Int64')
df['solver_canonical'] = df['solver_canonical'].replace({'DC-Offset(s=0.1)': 'DC-OSQA', 'DC-Offset': 'DC-OSQA', 'dc_offset': 'DC-OSQA'})
main_ns = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150]
key = df.groupby(['N', 'solver_canonical'], dropna=False).agg(rows=('instance', 'size'), n_instances=('instance', 'nunique'), unique_mean=('Unique_strict', 'mean'), family_nonnull=('Families_complete_linkage', lambda s: s.notna().sum()), family_mean=('Families_complete_linkage', 'mean')).reset_index().sort_values(['N', 'solver_canonical'])
expected = []
for N in main_ns:
    for solver in ['QA-Base', 'DC-OSQA']:
        expected.append((N, solver))
present = set(map(tuple, key[['N', 'solver_canonical']].dropna().values.tolist()))
missing = [(N, s) for (N, s) in expected if (N, s) not in present]
bad_family = key[key['solver_canonical'].isin(['QA-Base', 'DC-OSQA']) & key['family_nonnull'].eq(0)]
txt = pd.Series([''] * len(df), index=df.index, dtype='object')
for c in ['source_dataset', 'source_datasets', 'source_file', 'budget_label', 'Budget', 'Read_Budget', 'Total_Read_Budget', 'Reads_Per_Call', 'Num_Reads']:
    if c in df.columns:
        txt = txt.str.cat(df[c].astype(str), sep=' ')
budget = df[txt.str.lower().str.contains('50k|50000|40k|40000|10k|10000|budget_extension|extension', regex=True, na=False)].copy()
if missing or len(bad_family) or len(budget):
    raise SystemExit('CHECK main Fig1/Fig2 input validation failed; see audit/PAPER_MAIN_FIG1_FIG2_*.csv')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'code/main_qabase_vs_dcosqa'
CUR = BASE / 'curated'
MEAN_INPUT = CUR / 'fig1_main_by_N.csv'
GAIN_INPUT = CUR / 'fig2_paired_instance_gains.csv'
GAIN_SUMMARY = CUR / 'fig2_paired_gain_summary_by_N.csv'
OUT_PDF = CUR / 'Fig2_qabase_vs_dcosqa_n20_150.pdf'
OUT_PNG = CUR / 'Fig2_qabase_vs_dcosqa_n20_150.png'
SOLVERS = ['QA-Base', 'DC-OSQA']
N_MAIN = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150]
N_GAIN = [70, 80, 90, 100, 120, 150]
QA_COLOR = '#666666'
DC_COLOR = '#2F6F8F'
POINT_COLOR = '#7EAAC2'

def clean_axis(ax: plt.Axes, grid_axis: str='y') -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis=grid_axis, alpha=0.22)
    ax.set_axisbelow(True)
    ax.tick_params(direction='out')

def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(0.01, 1.05, label, transform=ax.transAxes, fontsize=12.5, fontweight='bold', ha='left', va='bottom', clip_on=False, zorder=20)

def pick(df: pd.DataFrame, names: list[str], required: bool=True):
    for name in names:
        if name in df.columns:
            return name
    if required:
        raise KeyError(f'Missing one of {names}; columns={df.columns.tolist()}')
    return None

def plot_mean_panel(ax: plt.Axes, df: pd.DataFrame, n_col: str, solver_col: str, mean_col: str, sem_col: str | None, ylabel: str) -> None:
    present_n = [n for n in N_MAIN if n in set(pd.to_numeric(df[n_col], errors='coerce').dropna().astype(int))]
    styles = {'QA-Base': dict(color=QA_COLOR, marker='o', linestyle='--'), 'DC-OSQA': dict(color=DC_COLOR, marker='s', linestyle='-')}
    for solver in SOLVERS:
        sub = df[df[solver_col].eq(solver)].copy()
        sub[n_col] = pd.to_numeric(sub[n_col], errors='coerce')
        sub = sub[sub[n_col].isin(present_n)].sort_values(n_col)
        x = sub[n_col].to_numpy(float)
        y = pd.to_numeric(sub[mean_col], errors='coerce').to_numpy(float)
        style = styles[solver]
        ax.plot(x, y, linewidth=1.9, markersize=5.0, label=solver, zorder=3, **style)
        if sem_col and sem_col in sub.columns:
            sem = pd.to_numeric(sub[sem_col], errors='coerce').fillna(0).to_numpy(float)
            ax.fill_between(x, y - sem, y + sem, color=style['color'], alpha=0.13, linewidth=0, zorder=1)
    ax.set_xlabel('Graph size, N')
    ax.set_ylabel(ylabel)
    ax.set_xticks(present_n)
    ax.set_xticklabels([str(x) for x in present_n], rotation=0)
    clean_axis(ax)

def plot_gain_panel(ax: plt.Axes, gains: pd.DataFrame, summary: pd.DataFrame, metric: str, ylabel: str) -> None:
    sub = gains[gains['comparison'].eq('DC-OSQA vs QA-Base') & gains['metric'].eq(metric) & gains['N'].isin(N_GAIN)].copy()
    actual_n = [n for n in N_GAIN if n in set(sub['N'].astype(int))]
    positions = {n: i for (i, n) in enumerate(actual_n)}
    for n in actual_n:
        d = sub[sub['N'].eq(n)].sort_values('instance')
        jitter = np.linspace(-0.17, 0.17, len(d))
        ax.scatter(positions[n] + jitter, pd.to_numeric(d['gain'], errors='coerce'), s=18, color=POINT_COLOR, alpha=0.68, edgecolor='white', linewidth=0.25, zorder=3)
    sm = summary[summary['comparison'].eq('DC-OSQA vs QA-Base') & summary['metric'].eq(metric) & summary['N'].isin(actual_n)].sort_values('N')
    mean_x = [positions[int(n)] for n in sm['N']]
    mean_y = pd.to_numeric(sm['mean_gain'], errors='coerce').to_numpy(float)
    mean_sem = pd.to_numeric(sm['sem_gain'], errors='coerce').fillna(0).to_numpy(float)
    ax.errorbar(mean_x, mean_y, yerr=mean_sem, fmt='D', markersize=6.2,
                color=DC_COLOR, markeredgecolor='white', markeredgewidth=0.7,
                ecolor=DC_COLOR, elinewidth=1.15, capsize=3.0, capthick=1.15,
                zorder=5, label=r'Mean gain $\pm$ s.e.m.')
    ax.axhline(0, color='#555555', linestyle='--', linewidth=0.9, zorder=1)
    ax.set_xticks(range(len(actual_n)))
    ax.set_xticklabels([str(n) for n in actual_n])
    ax.set_xlabel('Graph size, N')
    ax.set_ylabel(ylabel)
    clean_axis(ax)

def main() -> None:
    CUR.mkdir(parents=True, exist_ok=True)
    mean_df = pd.read_csv(MEAN_INPUT)
    gains = pd.read_csv(GAIN_INPUT)
    gain_summary = pd.read_csv(GAIN_SUMMARY)
    n_col = pick(mean_df, ['N'])
    solver_col = pick(mean_df, ['solver_canonical', 'Solver', 'solver'])
    unique_mean = pick(mean_df, ['Unique_strict_mean', 'mean_unique', 'Unique_mean', 'Unique_strict'])
    unique_sem = pick(mean_df, ['Unique_strict_sem', 'Unique_se', 'Unique_sem', 'sem_unique'], required=False)
    family_mean = pick(mean_df, ['Families_complete_linkage_mean', 'mean_families', 'Families_mean', 'Families_complete_linkage'])
    family_sem = pick(mean_df, ['Families_complete_linkage_sem', 'Families_se', 'Families_sem', 'sem_families'], required=False)
    mean_df = mean_df[mean_df[solver_col].isin(SOLVERS) & mean_df[n_col].isin(N_MAIN)].copy()
    (fig, axes) = plt.subplots(2, 2, figsize=(10.8, 7.6), constrained_layout=True)
    plot_mean_panel(axes[0, 0], mean_df, n_col, solver_col, unique_mean, unique_sem, 'Unique certified candidates')
    plot_mean_panel(axes[0, 1], mean_df, n_col, solver_col, family_mean, family_sem, 'Complete-linkage families')
    plot_gain_panel(axes[1, 0], gains, gain_summary, 'Unique_strict', 'DC-OSQA − QA-Base\nunique candidates')
    plot_gain_panel(axes[1, 1], gains, gain_summary, 'Families_complete_linkage', 'DC-OSQA − QA-Base\nfamilies')
    for (ax, label) in zip(axes.flat, ['(a)', '(b)', '(c)', '(d)']):
        panel_label(ax, label)
    axes[0, 1].legend(frameon=False, fontsize=8.7, loc='upper left')
    axes[1, 1].legend(frameon=False, fontsize=8.7, loc='upper left')
    fig.savefig(OUT_PDF, bbox_inches='tight')
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    plt.close(fig)
    ok = OUT_PDF.is_file() and OUT_PNG.is_file() and (OUT_PDF.stat().st_size > 0) and (OUT_PNG.stat().st_size > 0)
    if not ok:
        raise SystemExit(1)
if __name__ == '__main__':
    main()
