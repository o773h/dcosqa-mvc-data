#!/usr/bin/env python3
"""Robustness analyses separating certified yield from conditional diversity.

Inputs are the package-generated certified candidate tables.  All bitstrings are
read explicitly as strings so leading zeros are preserved.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.special import gammaln
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'code/supp_yield_diversity_robustness'
DERIVED = BASE / 'derived'
CURATED = BASE / 'curated'
MAIN_CERT = ROOT / 'code/main_qabase_vs_dcosqa/derived/fig1_fig2_certified_bitstrings_recomputed_from_raw.csv'
FULL_STREAM = ROOT / 'code/main_qpu_read_budget_extension_n150/derived/verified_full_candidate_stream_from_raw.csv.gz'
GAIN_INPUT = ROOT / 'code/main_qabase_vs_dcosqa/curated/fig2_paired_instance_gains.csv'

QA_COLOR = '#5F5F5F'
DC_COLOR = '#2F6F8F'
PAIR_COLOR = '#B0B0B0'
BOOT_SEED = 20260713
BOOT_REPS = 5000
TAU_FRACTIONS = [0.00, 0.05, 0.10, 0.15, 0.20]
CERTIFIED_DRAW_GRID = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]


def sem(values) -> float:
    x = pd.to_numeric(pd.Series(values), errors='coerce').dropna().to_numpy(float)
    return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float('nan')


def clean_axis(ax: plt.Axes, grid_axis: str = 'y') -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out')
    ax.grid(axis=grid_axis, alpha=0.22)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.05, label, transform=ax.transAxes, fontsize=12.5,
            fontweight='bold', ha='left', va='bottom', clip_on=False)


def expected_unique_without_replacement(counts: np.ndarray, draws: int) -> float:
    counts = np.asarray(counts, dtype=np.int64)
    counts = counts[counts > 0]
    total = int(counts.sum())
    if draws < 0 or draws > total:
        return float('nan')
    if draws == 0 or counts.size == 0:
        return 0.0
    if draws == total:
        return float(counts.size)
    # P(type i absent) = choose(total-count_i, draws) / choose(total, draws).
    absent = np.zeros(counts.size, dtype=float)
    eligible = (total - counts) >= draws
    absent[eligible] = np.exp(
        gammaln(total - counts[eligible] + 1)
        - gammaln(draws + 1)
        - gammaln(total - counts[eligible] - draws + 1)
        - (gammaln(total + 1) - gammaln(draws + 1) - gammaln(total - draws + 1))
    )
    return float(np.sum(1.0 - absent))


def load_full_stream() -> pd.DataFrame:
    required = ['N', 'instance', 'solver', 'Bitstring', 'strict_exact_verified', 'Occurrences']
    data = pd.read_csv(FULL_STREAM, usecols=required, dtype={'Bitstring': 'string'}, low_memory=False)
    if data['Bitstring'].isna().any() or not data['Bitstring'].str.fullmatch('[01]{150}').all():
        raise RuntimeError('Malformed N=150 bitstrings in verified stream')
    strict = data['strict_exact_verified']
    if strict.dtype != bool:
        strict = strict.astype(str).str.strip().str.lower().isin(['true', '1', 'yes'])
    data['strict_exact_verified'] = strict
    data['Occurrences'] = pd.to_numeric(data['Occurrences'], errors='raise').astype(np.int64)
    return data


def make_yield_and_rarefaction(full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    totals = full.groupby(['N', 'instance', 'solver'], as_index=False)['Occurrences'].sum().rename(columns={'Occurrences': 'total_reads'})
    strict = full[full['strict_exact_verified']].copy()
    strict_counts = strict.groupby(['N', 'instance', 'solver', 'Bitstring'], as_index=False)['Occurrences'].sum()
    hits = strict_counts.groupby(['N', 'instance', 'solver'], as_index=False).agg(
        certified_reads=('Occurrences', 'sum'), observed_unique=('Bitstring', 'nunique'))
    yield_table = totals.merge(hits, on=['N', 'instance', 'solver'], how='left')
    yield_table[['certified_reads', 'observed_unique']] = yield_table[['certified_reads', 'observed_unique']].fillna(0)
    yield_table['certified_yield'] = yield_table['certified_reads'] / yield_table['total_reads']
    if not yield_table['total_reads'].eq(50000).all():
        raise RuntimeError('Expected exactly 50,000 reads per N=150 instance and solver')

    rows = []
    for (n, instance, solver), group in strict_counts.groupby(['N', 'instance', 'solver']):
        counts = group['Occurrences'].to_numpy(np.int64)
        total_certified = int(counts.sum())
        for draws in CERTIFIED_DRAW_GRID:
            if draws <= total_certified:
                rows.append({
                    'N': int(n), 'instance': int(instance), 'solver': str(solver),
                    'certified_draws': int(draws), 'certified_reads_available': total_certified,
                    'expected_unique': expected_unique_without_replacement(counts, draws),
                })
    rare = pd.DataFrame(rows)
    # Retain a paired instance set at each depth so solver means are directly comparable.
    paired = rare.groupby(['certified_draws', 'instance'])['solver'].nunique().reset_index(name='solver_count')
    paired = paired[paired['solver_count'].eq(2)][['certified_draws', 'instance']]
    rare = rare.merge(paired, on=['certified_draws', 'instance'], how='inner', validate='many_to_one')
    return yield_table.sort_values(['instance', 'solver']), rare.sort_values(['certified_draws', 'solver', 'instance'])


def family_counts(bitstrings: list[str], n: int) -> dict[float, int]:
    unique = sorted(set(bitstrings))
    if len(unique) <= 1:
        return {fraction: len(unique) for fraction in TAU_FRACTIONS}
    matrix = np.fromiter((int(ch) for bitstring in unique for ch in bitstring), dtype=np.int8).reshape(len(unique), n)
    condensed = pdist(matrix, metric='hamming') * n
    tree = linkage(condensed, method='complete')
    result = {0.0: len(unique)}
    for fraction in TAU_FRACTIONS[1:]:
        tau = max(1, int(round(fraction * n)))
        result[fraction] = int(np.unique(fcluster(tree, t=tau, criterion='distance')).size)
    return result


def make_threshold_sensitivity(main_cert: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = main_cert[main_cert['N'].isin([70, 80, 90, 100, 120, 150])].copy()
    rows = []
    for (n, instance, solver), group in data.groupby(['N', 'instance', 'solver_canonical']):
        counts = family_counts(group['Bitstring'].astype(str).tolist(), int(n))
        for fraction, count in counts.items():
            rows.append({'N': int(n), 'instance': int(instance), 'solver': solver,
                         'tau_fraction': fraction, 'tau_vertices': max(0, int(round(fraction * int(n)))),
                         'families': count})
    instance_table = pd.DataFrame(rows)
    expected = pd.DataFrame([
        {'N': n, 'instance': instance, 'solver': solver, 'tau_fraction': fraction,
         'tau_vertices': max(0, int(round(fraction * n)))}
        for n in [70, 80, 90, 100, 120, 150]
        for instance in range(20)
        for solver in ['QA-Base', 'DC-OSQA']
        for fraction in TAU_FRACTIONS
    ])
    instance_table = expected.merge(
        instance_table[['N', 'instance', 'solver', 'tau_fraction', 'families']],
        on=['N', 'instance', 'solver', 'tau_fraction'], how='left', validate='one_to_one')
    instance_table['families'] = instance_table['families'].fillna(0).astype(int)
    wide = instance_table.pivot(index=['N', 'instance', 'tau_fraction'], columns='solver', values='families').reset_index()
    wide['paired_gain_dc_minus_qa'] = wide['DC-OSQA'] - wide['QA-Base']
    rng = np.random.default_rng(BOOT_SEED)
    summary_rows = []
    for fraction, group in wide.groupby('tau_fraction'):
        values = group['paired_gain_dc_minus_qa'].to_numpy(float)
        boot = np.empty(BOOT_REPS, dtype=float)
        strata = [g['paired_gain_dc_minus_qa'].to_numpy(float) for _, g in group.groupby('N')]
        for b in range(BOOT_REPS):
            sampled = np.concatenate([rng.choice(x, size=len(x), replace=True) for x in strata])
            boot[b] = sampled.mean()
        summary_rows.append({'tau_fraction': fraction, 'n_pairs': len(values), 'mean_gain': values.mean(),
                             'sem_gain': sem(values), 'bootstrap_ci_low': np.quantile(boot, 0.025),
                             'bootstrap_ci_high': np.quantile(boot, 0.975)})
    return instance_table.sort_values(['tau_fraction', 'N', 'instance', 'solver']), pd.DataFrame(summary_rows).sort_values('tau_fraction')


def make_observed_support(main_cert: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (n, instance), group in main_cert.groupby(['N', 'instance']):
        qa = set(group.loc[group['solver_canonical'].eq('QA-Base'), 'Bitstring'].astype(str))
        dc = set(group.loc[group['solver_canonical'].eq('DC-OSQA'), 'Bitstring'].astype(str))
        union = qa | dc
        overlap = qa & dc
        best = max(len(qa), len(dc))
        rows.append({'N': int(n), 'instance': int(instance), 'qa_unique': len(qa), 'dc_unique': len(dc),
                     'observed_union_lower_bound': len(union), 'observed_overlap': len(overlap),
                     'union_gain_over_best': len(union) - best,
                     'union_to_best_ratio': len(union) / best if best else float('nan'),
                     'multiple_observed_optima': len(union) >= 2})
    instance = pd.DataFrame(rows).sort_values(['N', 'instance'])
    summary = instance.groupby('N', as_index=False).agg(
        n_instances=('instance', 'nunique'), mean_observed_union=('observed_union_lower_bound', 'mean'),
        sem_observed_union=('observed_union_lower_bound', sem), mean_union_to_best_ratio=('union_to_best_ratio', 'mean'),
        sem_union_to_best_ratio=('union_to_best_ratio', sem),
        fraction_multiple_observed_optima=('multiple_observed_optima', 'mean'))
    return instance, summary


def make_effect_sizes(gains: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOT_SEED + 1)
    rows = []
    for metric, group in gains.groupby('metric'):
        group = group.copy()
        values = group['gain'].to_numpy(float)
        sd = np.std(values, ddof=1)
        dz = float(np.mean(values) / sd) if sd > 0 else float('nan')
        boot_mean = np.empty(BOOT_REPS, dtype=float)
        boot_dz = np.empty(BOOT_REPS, dtype=float)
        strata = [g['gain'].to_numpy(float) for _, g in group.groupby('N')]
        for b in range(BOOT_REPS):
            sampled = np.concatenate([rng.choice(x, size=len(x), replace=True) for x in strata])
            boot_mean[b] = sampled.mean()
            sampled_sd = np.std(sampled, ddof=1)
            boot_dz[b] = sampled.mean() / sampled_sd if sampled_sd > 0 else np.nan
        rows.append({'metric': metric, 'n_pairs': len(values), 'mean_paired_difference': np.mean(values),
                     'mean_difference_ci_low': np.quantile(boot_mean, 0.025),
                     'mean_difference_ci_high': np.quantile(boot_mean, 0.975),
                     'cohen_dz': dz, 'cohen_dz_ci_low': np.nanquantile(boot_dz, 0.025),
                     'cohen_dz_ci_high': np.nanquantile(boot_dz, 0.975),
                     'fraction_positive': np.mean(values > 0)})
    return pd.DataFrame(rows)


def gini_coefficient(values: np.ndarray) -> float:
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    if n == 0 or x.sum() <= 0:
        return float('nan')
    return float(2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum()) - (n + 1) / n)


def make_direct_50k_metrics(full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    strict = full[full['strict_exact_verified']].groupby(
        ['instance', 'solver', 'Bitstring'], as_index=False)['Occurrences'].sum()
    rows = []
    for (instance, solver), group in strict.groupby(['instance', 'solver']):
        counts = group['Occurrences'].to_numpy(float)
        probabilities = counts / counts.sum()
        unique = len(counts)
        rows.append({
            'instance': int(instance), 'solver': solver, 'Unique': unique,
            'Certified_occurrences': int(counts.sum()),
            'NormalizedEntropy': float(-(probabilities * np.log2(probabilities)).sum() / np.log2(unique)) if unique > 1 else 0.0,
            'ESSFraction': float((1.0 / np.sum(probabilities ** 2)) / unique),
            'Top10Share': float(np.sort(probabilities)[-10:].sum()),
            'Gini': gini_coefficient(counts),
            'DuplicateRate': float(1.0 - unique / counts.sum()),
        })
    instance_table = pd.DataFrame(rows).sort_values(['instance', 'solver'])
    summary_rows = []
    lower_is_better = {'Top10Share', 'Gini', 'DuplicateRate'}
    for metric in ['Unique', 'Certified_occurrences', 'NormalizedEntropy', 'ESSFraction', 'Top10Share', 'Gini', 'DuplicateRate']:
        wide = instance_table.pivot(index='instance', columns='solver', values=metric)
        difference = wide['DC-OSQA'] - wide['QA-Base']
        dc_wins = int((difference < 0).sum()) if metric in lower_is_better else int((difference > 0).sum())
        qa_wins = int((difference > 0).sum()) if metric in lower_is_better else int((difference < 0).sum())
        summary_rows.append({
            'metric': metric, 'n_pairs': len(wide), 'dc_mean': wide['DC-OSQA'].mean(),
            'qa_mean': wide['QA-Base'].mean(), 'dc_wins': dc_wins, 'qa_wins': qa_wins,
            'ties': int((difference == 0).sum()), 'mean_dc_minus_qa': difference.mean(),
            'wilcoxon_two_sided_p': float(wilcoxon(difference).pvalue),
        })
    return instance_table, pd.DataFrame(summary_rows)


def plot_figure(yield_table: pd.DataFrame, rare: pd.DataFrame, threshold_summary: pd.DataFrame,
                support_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6), constrained_layout=True)

    pivot = yield_table.pivot(index='instance', columns='solver', values='certified_yield').dropna()
    for _, row in pivot.iterrows():
        axes[0, 0].plot([0, 1], [row['QA-Base'], row['DC-OSQA']], color=PAIR_COLOR, lw=0.65, alpha=0.6)
    axes[0, 0].scatter(np.zeros(len(pivot)), pivot['QA-Base'], s=20, color=QA_COLOR, marker='o', zorder=3)
    axes[0, 0].scatter(np.ones(len(pivot)), pivot['DC-OSQA'], s=24, color=DC_COLOR, marker='s', zorder=4)
    axes[0, 0].scatter([0, 1], [pivot['QA-Base'].mean(), pivot['DC-OSQA'].mean()], s=80, marker='D',
                       color=['#353535', '#174E6B'], edgecolor='white', linewidth=0.8, zorder=5)
    axes[0, 0].set_xticks([0, 1], ['QA-Base', 'DC-OSQA'])
    axes[0, 0].set_ylabel('Certified yield (certified reads / 50,000)')
    axes[0, 0].set_yscale('log')
    clean_axis(axes[0, 0])

    styles = {'QA-Base': (QA_COLOR, 'o', '--'), 'DC-OSQA': (DC_COLOR, 's', '-')}
    rare_summary = rare.groupby(['certified_draws', 'solver'], as_index=False).agg(
        mean_expected_unique=('expected_unique', 'mean'), sem_expected_unique=('expected_unique', sem),
        eligible_instances=('instance', 'nunique'))
    for solver, (color, marker, line) in styles.items():
        sub = rare_summary[rare_summary['solver'].eq(solver)]
        x = sub['certified_draws'].to_numpy(float)
        y = sub['mean_expected_unique'].to_numpy(float)
        e = sub['sem_expected_unique'].fillna(0).to_numpy(float)
        axes[0, 1].plot(x, y, color=color, marker=marker, linestyle=line, lw=1.8, ms=4.5, label=solver)
        axes[0, 1].fill_between(x, y - e, y + e, color=color, alpha=0.13, linewidth=0)
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_xlabel('Certified draws (yield conditioned out)')
    axes[0, 1].set_ylabel('Expected unique certified candidates')
    axes[0, 1].legend(frameon=False, fontsize=8.7)
    clean_axis(axes[0, 1])

    x = threshold_summary['tau_fraction'].to_numpy(float)
    y = threshold_summary['mean_gain'].to_numpy(float)
    lo = threshold_summary['bootstrap_ci_low'].to_numpy(float)
    hi = threshold_summary['bootstrap_ci_high'].to_numpy(float)
    axes[1, 0].plot(x, y, color=DC_COLOR, marker='D', lw=1.8)
    axes[1, 0].fill_between(x, lo, hi, color=DC_COLOR, alpha=0.16, linewidth=0)
    axes[1, 0].axhline(0, color='#555555', linestyle='--', lw=0.9)
    axes[1, 0].set_xticks(TAU_FRACTIONS, [f'{v:.2f}' for v in TAU_FRACTIONS])
    axes[1, 0].set_xlabel('Hamming threshold, tau / N')
    axes[1, 0].set_ylabel('Mean paired family gain\n(DC-OSQA − QA-Base)')
    clean_axis(axes[1, 0])

    x = support_summary['N'].to_numpy(int)
    y = support_summary['mean_union_to_best_ratio'].to_numpy(float)
    e = support_summary['sem_union_to_best_ratio'].fillna(0).to_numpy(float)
    axes[1, 1].plot(x, y, color=DC_COLOR, marker='s', lw=1.8)
    axes[1, 1].fill_between(x, y - e, y + e, color=DC_COLOR, alpha=0.15, linewidth=0)
    axes[1, 1].axhline(1, color='#555555', linestyle='--', lw=0.9)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xlabel('Graph size, N')
    axes[1, 1].set_ylabel('Observed union / best single-protocol support')
    clean_axis(axes[1, 1])

    for ax, label in zip(axes.flat, ['(a)', '(b)', '(c)', '(d)']):
        panel_label(ax, label)
    out_pdf = CURATED / 'yield_diversity_robustness.pdf'
    out_png = CURATED / 'yield_diversity_robustness.png'
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    CURATED.mkdir(parents=True, exist_ok=True)
    for path in [MAIN_CERT, FULL_STREAM, GAIN_INPUT]:
        if not path.is_file():
            raise FileNotFoundError(path)
    full = load_full_stream()
    main_cert = pd.read_csv(MAIN_CERT, dtype={'Bitstring': 'string'}, low_memory=False)
    expected_lengths = main_cert['Bitstring'].str.len().eq(main_cert['N'])
    if not expected_lengths.all():
        raise RuntimeError('Malformed bitstring length in main certified table')
    gains = pd.read_csv(GAIN_INPUT)

    yield_table, rare = make_yield_and_rarefaction(full)
    threshold_instance, threshold_summary = make_threshold_sensitivity(main_cert)
    support_instance, support_summary = make_observed_support(main_cert)
    effect_sizes = make_effect_sizes(gains)
    direct_instance, direct_summary = make_direct_50k_metrics(full)

    yield_table.to_csv(DERIVED / 'certified_yield_n150_by_instance.csv', index=False)
    rare.to_csv(DERIVED / 'conditional_diversity_rarefaction_n150.csv', index=False)
    threshold_instance.to_csv(DERIVED / 'hamming_threshold_sensitivity_by_instance.csv', index=False)
    threshold_summary.to_csv(DERIVED / 'hamming_threshold_sensitivity_summary.csv', index=False)
    support_instance.to_csv(DERIVED / 'observed_support_lower_bound_by_instance.csv', index=False)
    support_summary.to_csv(DERIVED / 'observed_support_lower_bound_summary.csv', index=False)
    effect_sizes.to_csv(DERIVED / 'paired_effect_sizes_bootstrap.csv', index=False)
    direct_instance.to_csv(DERIVED / 'n150_50k_direct_metrics_by_instance.csv', index=False)
    direct_summary.to_csv(DERIVED / 'n150_50k_direct_metrics_summary.csv', index=False)
    plot_figure(yield_table, rare, threshold_summary, support_summary)


if __name__ == '__main__':
    main()
