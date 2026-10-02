#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
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
out_name = 'sa_dc_recovery_n150_overlap_pairs_from_raw.csv'
recomputed.to_csv(DERIVED / out_name, index=False)
ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parents[1]
DER = BASE / 'derived'
CUR = BASE / 'curated'
DER.mkdir(exist_ok=True)
SOURCE = DER / 'sa_dc_recovery_n150_overlap_pairs_from_raw.csv'
OUTPUT = DER / 'sa_dc_recovery_plot_input_from_raw.csv'
if not SOURCE.exists():
    raise SystemExit(f'missing raw-recomputed overlap table: {SOURCE}')
df = pd.read_csv(SOURCE, low_memory=False)
required = ['instance', 'A', 'B', 'A_unique', 'B_unique', 'intersection', 'union', 'A_only', 'B_only']
missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f'missing required overlap columns: {missing}')
rows = []
for (_, r) in df.iterrows():
    a = str(r['A'])
    b = str(r['B'])
    a_lower = a.lower()
    b_lower = b.lower()
    if 'dc-osqa' in a_lower and 'sa-recovery' in b_lower:
        dc_unique = float(r['A_unique'])
        sa_unique = float(r['B_unique'])
        dc_only = float(r['A_only'])
        sa_only = float(r['B_only'])
        sa_label = b
    elif 'sa-recovery' in a_lower and 'dc-osqa' in b_lower:
        dc_unique = float(r['B_unique'])
        sa_unique = float(r['A_unique'])
        dc_only = float(r['B_only'])
        sa_only = float(r['A_only'])
        sa_label = a
    else:
        continue
    label_lower = sa_label.lower()
    if '10k' in label_lower:
        comparison = 'DC 1k vs SA 10k'
    elif '50k' in label_lower:
        comparison = 'DC 1k vs SA 50k'
    else:
        continue
    shared = float(r['intersection'])
    union = float(r['union'])
    rows.append({'instance': int(r['instance']), 'comparison': comparison, 'dc_unique': int(round(dc_unique)), 'sa_unique': int(round(sa_unique)), 'shared': int(round(shared)), 'union': int(round(union)), 'recovery_fraction_of_dc': shared / dc_unique if dc_unique > 0 else 0.0, 'dc_only_fraction_of_union': dc_only / union if union > 0 else 0.0, 'shared_fraction_of_union': shared / union if union > 0 else 0.0, 'sa_only_fraction_of_union': sa_only / union if union > 0 else 0.0})
out = pd.DataFrame(rows)
expected_columns = ['instance', 'comparison', 'dc_unique', 'sa_unique', 'shared', 'union', 'recovery_fraction_of_dc', 'dc_only_fraction_of_union', 'shared_fraction_of_union', 'sa_only_fraction_of_union']
if out.empty:
    pairs = df[['A', 'B']].drop_duplicates().sort_values(['A', 'B'])
    raise SystemExit('no DC-OSQA versus SA-Recovery 10k/50k rows found\n' + pairs.to_string(index=False))
out = out[expected_columns].sort_values(['instance', 'comparison']).reset_index(drop=True)
counts = out.groupby('comparison')['instance'].nunique()
expected_comparisons = {'DC 1k vs SA 10k', 'DC 1k vs SA 50k'}
if set(counts.index) != expected_comparisons:
    raise SystemExit('unexpected comparison set:\n' + counts.to_string())
if not counts.eq(20).all():
    raise SystemExit('each SA/DC comparison must contain 20 instances:\n' + counts.to_string())
fraction_sum = out['dc_only_fraction_of_union'] + out['shared_fraction_of_union'] + out['sa_only_fraction_of_union']
if not np.allclose(fraction_sum, 1.0, atol=1e-10):
    bad = out.loc[~np.isclose(fraction_sum, 1.0, atol=1e-10)].copy()
    raise SystemExit('union composition fractions do not sum to one:\n' + bad.to_string(index=False))
out.to_csv(OUTPUT, index=False)
ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'code/supp_sa_dc_complementarity_n150'
FIG4 = ROOT / 'code/main_qpu_read_budget_extension_n150'
RAW = ROOT / 'data/raw'
DC_STREAM = FIG4 / 'derived/verified_full_candidate_stream_from_raw.csv.gz'
KEXACT_FILE = Path(__file__).resolve().parents[3] / 'data' / 'metadata' / 'ba_m2_n150_kexact_reference.csv'
OUTPUT = BASE / 'derived/sa_dc_same_budget_overlap_from_raw.csv'
SUMMARY = BASE / 'derived/sa_dc_same_budget_overlap_summary.csv'
N = 150
BUDGETS = [1000, 10000, 50000]
REPETITIONS = 100
RNG_SEED = 20260521
SA_SOURCES = {1000: [('sa_all', 'SA-Max')], 10000: [('sa_recovery_n150_10k_pilot', 'SA-Recovery(10000r,10000s)'), ('sa_recovery_n150_10k_remaining', 'SA-Recovery(10000r,10000s)')], 50000: [('sa_recovery_n150_50k', 'SA-Recovery(50000r,10000s)'), ('sa_recovery_n150_50k_remaining', 'SA-Recovery(50000r,10000s)')]}

def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(['true', '1', 'yes', 'y', 't'])

def load_kexact() -> dict[int, int]:
    df = pd.read_csv(KEXACT_FILE)
    df = df[pd.to_numeric(df['N'], errors='coerce').eq(N)].copy()
    df['instance'] = pd.to_numeric(df['instance'], errors='raise').astype(int)
    df['K_exact'] = pd.to_numeric(df['K_exact'], errors='raise').astype(int)
    result = dict(zip(df['instance'], df['K_exact']))
    if set(result) != set(range(20)):
        raise RuntimeError('K_exact does not cover instances 0–19')
    return result

def load_dc_reads(kexact: dict[int, int]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    required = ['N', 'instance', 'solver', 'Bitstring', 'K_exact', 'strict_exact_verified', 'Occurrences', 'source_order', 'source_row_order']
    stores = {instance: {'bits': [], 'strict': []} for instance in range(20)}
    for chunk in pd.read_csv(DC_STREAM, usecols=required, chunksize=100000, low_memory=False):
        chunk = chunk[pd.to_numeric(chunk['N'], errors='coerce').eq(N) & chunk['solver'].astype(str).eq('DC-OSQA')].copy()
        if chunk.empty:
            continue
        chunk['instance'] = pd.to_numeric(chunk['instance'], errors='raise').astype(int)
        chunk['K_exact'] = pd.to_numeric(chunk['K_exact'], errors='raise').astype(int)
        chunk['Occurrences'] = pd.to_numeric(chunk['Occurrences'], errors='raise').astype(int)
        chunk['source_order'] = pd.to_numeric(chunk['source_order'], errors='raise').astype(int)
        chunk['source_row_order'] = pd.to_numeric(chunk['source_row_order'], errors='raise').astype(int)
        chunk['strict_exact_verified'] = truthy(chunk['strict_exact_verified'])
        expected = chunk['instance'].map(kexact)
        if not chunk['K_exact'].eq(expected).all():
            raise RuntimeError('DC K_exact mismatch')
        chunk = chunk.sort_values(['source_order', 'source_row_order'], kind='stable')
        for (instance, group) in chunk.groupby('instance', sort=False):
            occurrences = group['Occurrences'].to_numpy(np.int64)
            stores[int(instance)]['bits'].append(np.repeat(group['Bitstring'].astype(str).to_numpy(), occurrences))
            stores[int(instance)]['strict'].append(np.repeat(group['strict_exact_verified'].to_numpy(bool), occurrences))
    result = {}
    for instance in range(20):
        bits = np.concatenate(stores[instance]['bits'])
        strict = np.concatenate(stores[instance]['strict'])
        if len(bits) != 50000:
            raise RuntimeError(f'DC instance {instance}: expected 50,000 reads, found {len(bits)}')
        result[instance] = (bits, strict)
    return result

def load_sa_sets(kexact: dict[int, int]) -> dict[tuple[int, int], set[str]]:
    result: dict[tuple[int, int], set[str]] = {}
    for (budget, source_specs) in SA_SOURCES.items():
        per_instance_rows = {instance: [] for instance in range(20)}
        for (source_dataset, expected_solver) in source_specs:
            path = RAW / source_dataset / 'raw_solutions.csv'
            if not path.is_file():
                raise FileNotFoundError(path)
            df = pd.read_csv(path, low_memory=False)
            df = df[pd.to_numeric(df['Nodes'], errors='coerce').eq(N) & df['Solver'].astype(str).eq(expected_solver)].copy()
            if df.empty:
                continue
            df['Instance'] = pd.to_numeric(df['Instance'], errors='raise').astype(int)
            df['Size'] = pd.to_numeric(df['Size'], errors='raise').astype(int)
            df['Occurrences'] = pd.to_numeric(df['Occurrences'], errors='raise').astype(int)
            df['IsValid'] = truthy(df['IsValid'])
            for (instance, group) in df.groupby('Instance', sort=False):
                per_instance_rows[int(instance)].append(group)
        for instance in range(20):
            groups = per_instance_rows[instance]
            if len(groups) != 1:
                raise RuntimeError(f'SA budget {budget}, instance {instance}: expected one source partition, found {len(groups)}')
            group = groups[0]
            read_count = int(group['Occurrences'].sum())
            if read_count != budget:
                raise RuntimeError(f'SA budget {budget}, instance {instance}: expected {budget} reads, found {read_count}')
            strict = group['IsValid'] & group['Size'].eq(kexact[instance])
            result[instance, budget] = set(group.loc[strict, 'Bitstring'].dropna().astype(str))
    return result

def sampled_dc_set(bits: np.ndarray, strict: np.ndarray, instance: int, budget: int, repeat: int) -> set[str]:
    if budget == 50000:
        indices = np.arange(50000, dtype=np.int64)
    else:
        seed = int(RNG_SEED + instance * 100000 + budget * 10 + repeat)
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(bits), size=budget, replace=False)
    strict_indices = indices[strict[indices]]
    return set(bits[strict_indices].astype(str))

def run_stage_01() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    kexact = load_kexact()
    dc_reads = load_dc_reads(kexact)
    sa_sets = load_sa_sets(kexact)
    rows = []
    for budget in BUDGETS:
        repeats = [0] if budget == 50000 else range(REPETITIONS)
        for instance in range(20):
            (bits, strict) = dc_reads[instance]
            sa_set = sa_sets[instance, budget]
            for repeat in repeats:
                dc_set = sampled_dc_set(bits, strict, instance, budget, int(repeat))
                shared = dc_set & sa_set
                union = dc_set | sa_set
                dc_only = dc_set - sa_set
                sa_only = sa_set - dc_set
                union_n = len(union)
                rows.append({'N': N, 'instance': instance, 'budget': budget, 'repeat': int(repeat), 'dc_unique': len(dc_set), 'sa_unique': len(sa_set), 'shared': len(shared), 'union': union_n, 'jaccard': len(shared) / union_n if union_n else 0.0, 'dc_only_fraction_of_union': len(dc_only) / union_n if union_n else 0.0, 'shared_fraction_of_union': len(shared) / union_n if union_n else 0.0, 'sa_only_fraction_of_union': len(sa_only) / union_n if union_n else 0.0})
    detail = pd.DataFrame(rows)
    fraction_sum = detail['dc_only_fraction_of_union'] + detail['shared_fraction_of_union'] + detail['sa_only_fraction_of_union']
    if not np.allclose(fraction_sum, 1.0, atol=1e-12):
        raise RuntimeError('Union fractions do not sum to one')
    expected_rows = 20 * 100 * 2 + 20
    if len(detail) != expected_rows:
        raise RuntimeError(f'Expected {expected_rows} detail rows; found {len(detail)}')
    detail.to_csv(OUTPUT, index=False)
    instance_summary = detail.groupby(['budget', 'instance'], as_index=False).agg(dc_unique=('dc_unique', 'mean'), sa_unique=('sa_unique', 'mean'), shared=('shared', 'mean'), union=('union', 'mean'), jaccard=('jaccard', 'mean'), dc_only_fraction_of_union=('dc_only_fraction_of_union', 'mean'), shared_fraction_of_union=('shared_fraction_of_union', 'mean'), sa_only_fraction_of_union=('sa_only_fraction_of_union', 'mean'), dc_repeats=('repeat', 'size'))
    summary = instance_summary.groupby('budget', as_index=False).agg(instances=('instance', 'nunique'), dc_unique_mean=('dc_unique', 'mean'), sa_unique_mean=('sa_unique', 'mean'), shared_mean=('shared', 'mean'), union_mean=('union', 'mean'), jaccard_mean=('jaccard', 'mean'), jaccard_sem=('jaccard', lambda x: x.std(ddof=1) / np.sqrt(len(x))), dc_only_fraction_mean=('dc_only_fraction_of_union', 'mean'), shared_fraction_mean=('shared_fraction_of_union', 'mean'), shared_fraction_sem=('shared_fraction_of_union', lambda x: x.std(ddof=1) / np.sqrt(len(x))), sa_only_fraction_mean=('sa_only_fraction_of_union', 'mean'))
    summary.to_csv(SUMMARY, index=False)
if __name__ == '__main__':
    run_stage_01()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'code/supp_sa_dc_complementarity_n150'
CUR = BASE / 'curated'
DER = BASE / 'derived'
RECOVERY_INPUT = DER / 'sa_dc_recovery_plot_input_from_raw.csv'
SAME_BUDGET_DETAIL = DER / 'sa_dc_same_budget_overlap_from_raw.csv'
SAME_BUDGET_SUMMARY = DER / 'sa_dc_same_budget_overlap_summary.csv'
OUT_PDF = CUR / 'sa_dc_complementarity_n150.pdf'
OUT_PNG = CUR / 'sa_dc_complementarity_n150.png'
PANEL_A_SUMMARY = CUR / 'sa_dc_paired_recovery_summary.csv'
PANEL_B_SUMMARY = CUR / 'sa_dc_same_budget_summary.csv'
SA_DARK = '#6A3D9A'
SA_MID = '#B072C0'
SA_LIGHT = '#C9ADD2'
DC_BLUE = '#0072B2'
SHARED_GRAY = '#B8B8B8'
TEXT = '#222222'
GRID = '#B0B0B0'

def clean_axis(ax: plt.Axes, grid_axis: str='y') -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, alpha=0.2, linewidth=0.75)
    ax.set_axisbelow(True)

def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(0.01, 1.08, label, transform=ax.transAxes, fontsize=14, fontweight='bold', ha='left', va='bottom', clip_on=False, zorder=20)

def load_panel_a() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(RECOVERY_INPUT)
    required = {'instance', 'comparison', 'recovery_fraction_of_dc'}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f'Panel (a) input missing columns: {sorted(missing)}')
    comparisons = ['DC 1k vs SA 10k', 'DC 1k vs SA 50k']
    df = df[df['comparison'].isin(comparisons)].copy()
    counts = df.groupby('comparison')['instance'].nunique().reindex(comparisons)
    if len(df) != 40 or not counts.eq(20).all():
        raise RuntimeError('Panel (a) requires 20 instances for each comparison:\n' + counts.to_string())
    summary = df.groupby('comparison', sort=False).agg(mean_recovery_fraction=('recovery_fraction_of_dc', 'mean'), median_recovery_fraction=('recovery_fraction_of_dc', 'median'), sem_recovery_fraction=('recovery_fraction_of_dc', lambda x: x.std(ddof=1) / np.sqrt(len(x))), n_instances=('instance', 'nunique')).reindex(comparisons).reset_index()
    summary['SA_budget'] = ['10k', '50k']
    return (df, summary)

def load_panel_b() -> tuple[pd.DataFrame, pd.DataFrame]:
    detail = pd.read_csv(SAME_BUDGET_DETAIL)
    summary = pd.read_csv(SAME_BUDGET_SUMMARY)
    required_detail = {'instance', 'budget', 'repeat', 'dc_only_fraction_of_union', 'shared_fraction_of_union', 'sa_only_fraction_of_union'}
    required_summary = {'budget', 'instances', 'dc_only_fraction_mean', 'shared_fraction_mean', 'shared_fraction_sem', 'sa_only_fraction_mean'}
    missing_detail = required_detail - set(detail.columns)
    missing_summary = required_summary - set(summary.columns)
    if missing_detail:
        raise KeyError(f'Panel (b) detail missing columns: {sorted(missing_detail)}')
    if missing_summary:
        raise KeyError(f'Panel (b) summary missing columns: {sorted(missing_summary)}')
    budgets = [1000, 10000, 50000]
    summary['budget'] = pd.to_numeric(summary['budget'], errors='raise').astype(int)
    summary = summary[summary['budget'].isin(budgets)].set_index('budget').reindex(budgets).reset_index()
    if len(detail) != 4020:
        raise RuntimeError(f'Expected 4,020 same-budget rows; found {len(detail)}')
    if len(summary) != 3:
        raise RuntimeError(f'Expected three same-budget summary rows; found {len(summary)}')
    if not summary['instances'].eq(20).all():
        raise RuntimeError('Same-budget summary does not contain 20 instances per budget')
    fractions = summary['sa_only_fraction_mean'] + summary['shared_fraction_mean'] + summary['dc_only_fraction_mean']
    if not np.allclose(fractions, 1.0, atol=1e-10):
        raise RuntimeError('Mean union-composition fractions do not sum to one')
    return (detail, summary)

def draw_panel_a(ax: plt.Axes, df: pd.DataFrame, summary: pd.DataFrame) -> None:
    comparisons = ['DC 1k vs SA 10k', 'DC 1k vs SA 50k']
    pivot = df.pivot(index='instance', columns='comparison', values='recovery_fraction_of_dc').reindex(columns=comparisons).sort_index()
    y10 = pivot[comparisons[0]].to_numpy(float)
    y50 = pivot[comparisons[1]].to_numpy(float)
    mean10 = float(np.mean(y10))
    mean50 = float(np.mean(y50))
    rng = np.random.default_rng(20260710)
    x10 = rng.uniform(-0.055, 0.055, len(y10))
    x50 = 1 + rng.uniform(-0.055, 0.055, len(y50))
    ax.scatter(x10, y10, s=30, marker='o', facecolor=SA_MID, edgecolor=SA_DARK, linewidth=0.8, alpha=0.9, zorder=3)
    ax.scatter(x50, y50, s=30, marker='s', facecolor=SA_DARK, edgecolor='white', linewidth=0.55, alpha=0.92, zorder=3)
    ax.scatter(0, mean10, s=140, marker='o', facecolor=SA_MID, edgecolor='#3A1B45', linewidth=2.0, zorder=5)
    ax.scatter(1, mean50, s=140, marker='s', facecolor=SA_DARK, edgecolor='white', linewidth=0.9, zorder=5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['SA 10k', 'SA 50k'])
    ax.set_xlim(-0.27, 1.27)
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel('Fraction of DC-OSQA 1k\ncertified support recovered')
    ax.text(0.5, 1.015, 'Large markers: across-instance means', transform=ax.transAxes, ha='center', va='bottom', fontsize=8.3, color='#555555')
    clean_axis(ax)
    panel_label(ax, '(a)')

def draw_panel_b(ax: plt.Axes, summary: pd.DataFrame) -> None:
    x = np.arange(3)
    width = 0.62
    sa = summary['sa_only_fraction_mean'].to_numpy(float) * 100
    shared = summary['shared_fraction_mean'].to_numpy(float) * 100
    shared_sem = summary['shared_fraction_sem'].to_numpy(float) * 100
    dc = summary['dc_only_fraction_mean'].to_numpy(float) * 100
    ax.bar(x, sa, width=width, color=SA_DARK, edgecolor='#333333', linewidth=0.7, hatch='///', label='SA-only', zorder=3)
    ax.bar(x, shared, width=width, bottom=sa, color=SHARED_GRAY, edgecolor='#333333', linewidth=0.7, label='Shared', zorder=4)
    ax.bar(x, dc, width=width, bottom=sa + shared, color=DC_BLUE, edgecolor='#333333', linewidth=0.7, hatch='xx', label='DC-only', zorder=3)
    for i in range(3):
        if sa[i] >= 8:
            ax.text(x[i], sa[i] / 2, f'{sa[i]:.1f}%', ha='center', va='center', color='white', fontsize=8.8, fontweight='bold')
        if dc[i] >= 8:
            ax.text(x[i], sa[i] + shared[i] + dc[i] / 2, f'{dc[i]:.1f}%', ha='center', va='center', color='white', fontsize=8.8, fontweight='bold')
        shared_center = sa[i] + shared[i] / 2
        ax.annotate(f'Shared {shared[i]:.1f}%\n± {shared_sem[i]:.1f}% SEM', xy=(x[i], shared_center), xytext=(x[i], 104.0), ha='center', va='bottom', fontsize=8.0, color=TEXT, arrowprops={'arrowstyle': '-', 'color': '#555555', 'linewidth': 0.75, 'shrinkA': 1, 'shrinkB': 1}, clip_on=False)
    ax.set_xticks(x)
    ax.set_xticklabels(['1k vs 1k', '10k vs 10k', '50k vs 50k'])
    ax.set_ylim(0, 112)
    ax.set_ylabel('Mean union composition (%)')
    clean_axis(ax)
    panel_label(ax, '(b)')
    legend_handles = [Patch(facecolor=SA_DARK, edgecolor='#333333', hatch='///', label='SA-only'), Patch(facecolor=SHARED_GRAY, edgecolor='#333333', label='Shared'), Patch(facecolor=DC_BLUE, edgecolor='#333333', hatch='xx', label='DC-only')]
    ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=8.7, handlelength=1.35, columnspacing=1.4)

def run_stage_02() -> None:
    CUR.mkdir(parents=True, exist_ok=True)
    (panel_a, panel_a_summary) = load_panel_a()
    (panel_b_detail, panel_b_summary) = load_panel_b()
    (fig, axes) = plt.subplots(1, 2, figsize=(11.1, 4.35), gridspec_kw={'width_ratios': [0.95, 1.2]})
    draw_panel_a(axes[0], panel_a, panel_a_summary)
    draw_panel_b(axes[1], panel_b_summary)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.23, wspace=0.34)
    panel_a_summary[['SA_budget', 'mean_recovery_fraction', 'median_recovery_fraction', 'sem_recovery_fraction', 'n_instances']].to_csv(PANEL_A_SUMMARY, index=False)
    panel_b_summary.to_csv(PANEL_B_SUMMARY, index=False)
    fig.savefig(OUT_PDF, bbox_inches='tight')
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    plt.close(fig)
    outputs_ok = all((path.is_file() and path.stat().st_size > 0 for path in [OUT_PDF, OUT_PNG, PANEL_A_SUMMARY, PANEL_B_SUMMARY]))
    expected_shared = np.array([0.007933, 0.053132, 0.142001])
    observed_shared = panel_b_summary['shared_fraction_mean'].to_numpy(float)
    numerical_ok = np.allclose(observed_shared, expected_shared, atol=5e-06)
    status = 'PASS' if outputs_ok and numerical_ok else 'FAIL'
    if status != 'PASS':
        raise SystemExit(1)
if __name__ == '__main__':
    run_stage_02()
