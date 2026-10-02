#!/usr/bin/env python3
from pathlib import Path
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.special import gammaln
PACKAGE = Path(__file__).resolve().parents[3]
STAGE = PACKAGE / 'code/main_qpu_read_budget_extension_n150'
UNIQUE = STAGE / 'derived/strict_exact_unique_budget_curve_summary.csv'
UNIQUE_BY_INSTANCE = STAGE / 'derived/strict_exact_unique_budget_curve_by_instance.csv'
FAMILY = STAGE / 'derived/strict_exact_family_curve_summary.csv'
FULL_STREAM = STAGE / 'derived/verified_full_candidate_stream_from_raw.csv.gz'
OUT_INPUT = STAGE / 'derived/fig5_qpu_read_budget_extension_plot_input.csv'
OUT_PDF = STAGE / 'figures/Fig5_qpu_read_budget_extension_n150.pdf'
OUT_PNG = STAGE / 'figures/Fig5_qpu_read_budget_extension_n150.png'
BUDGETS = [1000, 2000, 5000, 10000, 20000, 50000]
SOLVERS = ['QA-Base', 'DC-OSQA']

def expected_unique_without_replacement(counts, total_reads, draws):
    counts = np.asarray(counts, dtype=np.int64)
    counts = counts[counts > 0]
    if draws == total_reads:
        return float(len(counts))
    eligible = total_reads - counts >= draws
    absent = np.zeros(len(counts), dtype=float)
    absent[eligible] = np.exp(
        gammaln(total_reads - counts[eligible] + 1)
        - gammaln(draws + 1)
        - gammaln(total_reads - counts[eligible] - draws + 1)
        - (gammaln(total_reads + 1)
           - gammaln(draws + 1)
           - gammaln(total_reads - draws + 1))
    )
    return float(np.sum(1.0 - absent))

def build_exact_unique_curve():
    if not FULL_STREAM.exists():
        raise FileNotFoundError(FULL_STREAM)
    required = ['instance', 'solver', 'Bitstring', 'strict_exact_verified', 'Occurrences']
    stream = pd.read_csv(FULL_STREAM, usecols=required, dtype={'Bitstring': 'string'}, low_memory=False)
    strict_flag = stream['strict_exact_verified']
    if strict_flag.dtype != bool:
        strict_flag = strict_flag.astype(str).str.strip().str.lower().isin(['true', '1', 'yes'])
    stream['strict_exact_verified'] = strict_flag
    stream['Occurrences'] = pd.to_numeric(stream['Occurrences'], errors='raise').astype(np.int64)
    totals = stream.groupby(['instance', 'solver'])['Occurrences'].sum()
    if len(totals) != 40 or not totals.eq(50000).all():
        raise RuntimeError('Expected 50,000 raw reads for each of 40 instance--solver groups')
    strict = stream[stream['strict_exact_verified']].groupby(
        ['instance', 'solver', 'Bitstring'], as_index=False)['Occurrences'].sum()
    rows = []
    for (instance, solver), group in strict.groupby(['instance', 'solver']):
        counts = group['Occurrences'].to_numpy(np.int64)
        for budget in BUDGETS:
            rows.append({
                'N': 150,
                'instance': int(instance),
                'solver': str(solver),
                'budget': int(budget),
                'expected_unique_strict_exact_candidates': expected_unique_without_replacement(counts, 50000, budget),
            })
    by_instance = pd.DataFrame(rows).sort_values(['solver', 'budget', 'instance'])
    expected_keys = {(instance, solver, budget) for instance in range(20) for solver in SOLVERS for budget in BUDGETS}
    found_keys = set(zip(by_instance['instance'], by_instance['solver'], by_instance['budget']))
    if found_keys != expected_keys:
        raise RuntimeError('Exact unique-support curve has incomplete keys')
    summary = by_instance.groupby(['N', 'budget', 'solver'], as_index=False).agg(
        instances=('instance', 'nunique'),
        mean_unique_strict_exact_candidates=('expected_unique_strict_exact_candidates', 'mean'),
        std_unique_strict_exact_candidates=('expected_unique_strict_exact_candidates', 'std'),
    )
    summary['sem_unique_strict_exact_candidates'] = (
        summary['std_unique_strict_exact_candidates'] / np.sqrt(summary['instances'])
    )
    summary['estimator'] = 'exact finite-population expectation from 50,000 raw reads'
    by_instance.to_csv(UNIQUE_BY_INSTANCE, index=False)
    summary.to_csv(UNIQUE, index=False)

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda : handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def load_plot_input():
    # Retain the exact finite-population curve as an auxiliary input for the
    # supplementary novelty diagnostics. Main Figure 5 itself uses the same
    # canonical cumulative read prefixes for unique support and families.
    build_exact_unique_curve()
    if not FAMILY.exists():
        raise FileNotFoundError(FAMILY)
    family = pd.read_csv(FAMILY)
    family_required = {
        'N',
        'budget',
        'solver',
        'mean_unique_strict_exact_candidates',
        'sem_unique_strict_exact_candidates',
        'mean_family',
        'sem_family',
        'n',
    }
    missing_family = family_required - set(family.columns)
    if missing_family:
        raise RuntimeError('Family summary missing columns: ' + ', '.join(sorted(missing_family)))
    data = family[[
        'N',
        'budget',
        'solver',
        'mean_unique_strict_exact_candidates',
        'sem_unique_strict_exact_candidates',
        'mean_family',
        'sem_family',
        'n',
    ]].copy()
    data = data[data['solver'].isin(SOLVERS) & data['budget'].isin(BUDGETS)].copy()
    data['budget'] = data['budget'].astype(int)
    data['N'] = data['N'].astype(int)
    data = data.sort_values(['solver', 'budget']).reset_index(drop=True)
    expected_keys = {(solver, budget) for solver in SOLVERS for budget in BUDGETS}
    found_keys = set(zip(data['solver'], data['budget']))
    numeric_columns = [
        'mean_unique_strict_exact_candidates',
        'sem_unique_strict_exact_candidates',
        'mean_family',
        'sem_family',
    ]
    finite = np.isfinite(data[numeric_columns].to_numpy(dtype=float)).all()
    if len(data) != 12:
        raise RuntimeError(f'Expected 12 plot-input rows, found {len(data)}')
    if found_keys != expected_keys:
        raise RuntimeError('Solver-budget plot keys are incomplete')
    if not data['N'].eq(150).all():
        raise RuntimeError('Plot input contains N other than 150')
    if not data['n'].eq(20).all():
        raise RuntimeError('Family summary must contain 20 graph instances per solver and budget')
    if not finite:
        raise RuntimeError('Plot input contains non-finite values')
    return data

def plot(data):
    styles = {
        'QA-Base': {'color': '#666666', 'marker': 'o', 'linestyle': '--'},
        'DC-OSQA': {'color': '#2F6F8F', 'marker': 's', 'linestyle': '-'},
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    panels = [
        ('mean_unique_strict_exact_candidates', 'sem_unique_strict_exact_candidates', 'Unique certified candidates'),
        ('mean_family', 'sem_family', r'Hamming families ($\tau=0.1N$)'),
    ]
    for ax, (mean_col, sem_col, ylabel) in zip(axes, panels):
        for solver in SOLVERS:
            subset = data[data['solver'].eq(solver)].sort_values('budget')
            x = subset['budget'].to_numpy(dtype=float)
            y = subset[mean_col].to_numpy(dtype=float)
            e = subset[sem_col].to_numpy(dtype=float)
            style = styles[solver]
            ax.plot(x, y, linewidth=2.0, markersize=5.0, label=solver, **style)
            ax.fill_between(x, y - e, y + e, color=style['color'], alpha=0.16, linewidth=0)
        ax.set_xscale('log')
        ax.set_xticks(BUDGETS)
        ax.set_xticklabels([f'{budget // 1000}k' for budget in BUDGETS])
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_xlabel('QPU read budget')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc='upper left', frameon=True)
    axes[0].text(-0.16, 1.04, '(a)', transform=axes[0].transAxes, fontsize=11, fontweight='bold', va='bottom')
    axes[1].text(-0.16, 1.04, '(b)', transform=axes[1].transAxes, fontsize=11, fontweight='bold', va='bottom')
    fig.tight_layout(w_pad=2.2)
    fig.savefig(OUT_PDF, bbox_inches='tight')
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    bundle = Path(__file__).resolve().parents[1]
    (bundle / 'figures').mkdir(parents=True, exist_ok=True)
    data = load_plot_input()
    data.to_csv(OUT_INPUT, index=False)
    plot(data)
    checks = {'classification': 'MANUSCRIPT_READ_BUDGET_SENSITIVITY', 'plot_input_rows': int(len(data)), 'expected_plot_input_rows': 12, 'solver_count': int(data['solver'].nunique()), 'budget_count': int(data['budget'].nunique()), 'support_definition': 'IsValid AND Size == recorded graph-level K_exact', 'unique_budget_definition': 'deterministic occurrence prefix in canonical raw source and row order', 'family_budget_definition': 'deterministic occurrence prefix in canonical raw source and row order', 'auxiliary_unique_curve_definition': 'exact finite-population expected unique support under sampling without replacement from each 50,000-read multiset; generated for supplementary novelty diagnostics', 'family_definition': 'complete-linkage Hamming families at tau=0.1N', 'uncertainty_definition': 'standard error across 20 graph instances', 'outputs': {'auxiliary_unique_instance_table': str(UNIQUE_BY_INSTANCE), 'auxiliary_unique_summary': str(UNIQUE), 'prefix_summary': str(FAMILY), 'plot_input': str(OUT_INPUT), 'pdf': str(OUT_PDF), 'png': str(OUT_PNG)}, 'sha256': {'auxiliary_unique_instance_table': sha256(UNIQUE_BY_INSTANCE), 'auxiliary_unique_summary': sha256(UNIQUE), 'prefix_summary': sha256(FAMILY), 'plot_input': sha256(OUT_INPUT), 'pdf': sha256(OUT_PDF), 'png': sha256(OUT_PNG)}, 'pass': bool(len(data) == 12 and OUT_PDF.exists() and (OUT_PDF.stat().st_size > 0) and OUT_PNG.exists() and (OUT_PNG.stat().st_size > 0))}
    return 0 if checks['pass'] else 2
if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        raise
