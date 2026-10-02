#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
BUNDLE = Path(__file__).resolve().parents[1]
ROOT = BUNDLE.parents[1]
RAW_ROOTS = [ROOT / 'data/raw', ROOT / 'combined_raw']
METADATA_SOURCE = ROOT / 'data/metadata/ba_m2_n70_150_concentration_analysis_input.csv'
DERIVED = BUNDLE / 'derived'
AUDIT = BUNDLE / 'audit'
OUT_DATA = DERIVED / 'dc_vs_bc_source_concentration_ba_m2_N70_150_analysis_ready_strict_exact_from_raw.csv'
OUT_SOURCES = AUDIT / f'{METADATA_SOURCE.stem}_raw_sources_used.csv'
KEY = ['N', 'instance', 'solver_canonical', 'source_dataset', 'solver_raw']
STRICT_COLUMNS = ['Unique_strict', 'Strict_occurrences', 'zero_yield']

def find_raw(source_dataset: str) -> Path:
    candidates = [root / source_dataset / 'raw_solutions.csv' for root in RAW_ROOTS]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'raw source not found for {source_dataset}; checked: ' + ', '.join((str(path) for path in candidates)))

def normalize_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(['true', '1', 'yes'])

def load_raw(source_dataset: str, cache: dict[str, pd.DataFrame], source_rows: list[dict]) -> pd.DataFrame:
    if source_dataset in cache:
        return cache[source_dataset]
    path = find_raw(source_dataset)
    data = pd.read_csv(path, low_memory=False)
    rename = {}
    if 'Nodes' not in data.columns and 'N' in data.columns:
        rename['N'] = 'Nodes'
    if 'Instance' not in data.columns and 'instance' in data.columns:
        rename['instance'] = 'Instance'
    if rename:
        data = data.rename(columns=rename)
    required = {'Nodes', 'Instance', 'Size', 'IsValid', 'Bitstring'}
    missing = required - set(data.columns)
    if missing:
        raise RuntimeError(f'{path}: missing raw columns {sorted(missing)}')
    for column in ['Nodes', 'Instance', 'Size']:
        data[column] = pd.to_numeric(data[column], errors='coerce')
    if 'Occurrences' not in data.columns:
        data['Occurrences'] = 1
    data['Occurrences'] = pd.to_numeric(data['Occurrences'], errors='coerce').fillna(0)
    data['IsValid'] = normalize_bool(data['IsValid'])
    source_rows.append({'source_dataset': source_dataset, 'raw_path': path.relative_to(ROOT).as_posix(), 'rows': int(len(data))})
    cache[source_dataset] = data
    return data

def run_stage_01() -> int:
    DERIVED.mkdir(exist_ok=True)
    if not METADATA_SOURCE.exists():
        raise FileNotFoundError(METADATA_SOURCE)
    metadata = pd.read_csv(METADATA_SOURCE, low_memory=False)
    required = set(KEY + ['K_exact'])
    missing = required - set(metadata.columns)
    if missing:
        raise RuntimeError('metadata source missing columns: ' + ', '.join(sorted(missing)))
    missing_strict = set(STRICT_COLUMNS) - set(metadata.columns)
    if missing_strict:
        raise RuntimeError('metadata source missing strict-count columns: ' + ', '.join(sorted(missing_strict)))
    if metadata.duplicated(KEY).any():
        raise RuntimeError(f'metadata source has duplicate keys: {KEY}')
    raw_cache = {}
    source_rows = []
    count_rows = []
    key_table = metadata[KEY + ['K_exact']].drop_duplicates()
    for row in key_table.itertuples(index=False):
        values = row._asdict()
        source_dataset = str(values['source_dataset'])
        solver_raw = str(values['solver_raw'])
        n = int(values['N'])
        instance = int(values['instance'])
        k_exact = float(values['K_exact'])
        raw = load_raw(source_dataset, raw_cache, source_rows)
        if 'Solver' in raw.columns:
            solver_column = 'Solver'
        elif 'solver_raw' in raw.columns:
            solver_column = 'solver_raw'
        else:
            raise RuntimeError(f'{source_dataset} lacks Solver/solver_raw')
        selected = raw[raw['Nodes'].eq(n) & raw['Instance'].eq(instance) & raw[solver_column].astype(str).eq(solver_raw)]
        certified = selected[selected['IsValid'] & selected['Size'].eq(k_exact)]
        unique_strict = int(certified['Bitstring'].nunique())
        strict_occurrences = float(certified['Occurrences'].sum())
        count_rows.append({'N': n, 'instance': instance, 'solver_canonical': values['solver_canonical'], 'source_dataset': source_dataset, 'solver_raw': solver_raw, 'Unique_strict': unique_strict, 'Strict_occurrences': strict_occurrences, 'zero_yield': unique_strict == 0})
    raw_counts = pd.DataFrame(count_rows)
    if raw_counts.duplicated(KEY).any():
        raise RuntimeError(f'recomputed counts have duplicate keys: {KEY}')
    output = metadata.drop(columns=STRICT_COLUMNS).merge(raw_counts[KEY + STRICT_COLUMNS], on=KEY, how='left', validate='one_to_one')
    output = output[metadata.columns]
    numeric = output[['Unique_strict', 'Strict_occurrences']].apply(pd.to_numeric, errors='coerce')
    finite = bool(np.isfinite(numeric.to_numpy(dtype=float)).all())
    nonnegative = bool(numeric.ge(0).all().all())
    expected_zero = numeric['Unique_strict'].eq(0)
    actual_zero = normalize_bool(output['zero_yield'])
    zero_consistent = bool(expected_zero.eq(actual_zero).all())
    checks = {'classification': 'METADATA_GUIDED_RAW_TO_FIGURE', 'metadata_source': METADATA_SOURCE.relative_to(ROOT).as_posix(), 'strict_exact_rule': 'IsValid == True AND Size == K_exact', 'metadata_rows': int(len(metadata)), 'output_rows': int(len(output)), 'raw_count_rows': int(len(raw_counts)), 'key_columns': KEY, 'strict_columns': STRICT_COLUMNS, 'row_count_preserved': bool(len(output) == len(metadata)), 'output_key_unique': bool(not output.duplicated(KEY).any()), 'finite_strict_counts': finite, 'nonnegative_strict_counts': nonnegative, 'zero_yield_consistent': zero_consistent, 'raw_source_count': int(len(raw_cache)), 'outputs': {'data': OUT_DATA.relative_to(ROOT).as_posix(), 'sources': OUT_SOURCES.relative_to(ROOT).as_posix()}}
    checks['pass'] = bool(checks['row_count_preserved'] and checks['output_key_unique'] and checks['finite_strict_counts'] and checks['nonnegative_strict_counts'] and checks['zero_yield_consistent'])
    output.to_csv(OUT_DATA, index=False)
    return 0 if checks['pass'] else 2
builder_status = run_stage_01()
if builder_status != 0:
    raise SystemExit(builder_status)
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE = Path(__file__).resolve().parents[1]
INPUT = BASE / 'derived' / 'dc_vs_bc_source_concentration_ba_m2_N70_150_analysis_ready_strict_exact_from_raw.csv'
OUT_DIR = BASE / 'curated'
OUT_PDF = OUT_DIR / 'dc_vs_bc_paired_gain.pdf'
OUT_PNG = OUT_DIR / 'dc_vs_bc_paired_gain.png'
OUT_INSTANCE = BASE / 'derived' / 'dc_vs_bc_paired_instance_gains.csv'
OUT_SUMMARY = BASE / 'derived' / 'dc_vs_bc_paired_gain_summary.csv'
N_ORDER = [70, 80, 90, 100, 120, 150]
STYLE = {'DC-OSQA': {'color': '#2F6F8F', 'marker': 's', 'linestyle': '-'}, 'BC-OSQA': {'color': '#6B846E', 'marker': '^', 'linestyle': '--'}}

def sem(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors='coerce').dropna()
    if len(values) <= 1:
        return float('nan')
    return float(values.std(ddof=1) / math.sqrt(len(values)))

def clean_axis(ax: plt.Axes) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out')

def run_stage_02() -> None:
    df = pd.read_csv(INPUT)
    required = {'N', 'instance', 'solver_canonical', 'Unique_strict', 'Families_complete_linkage'}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f'missing columns: {missing}')
    df = df[df['solver_canonical'].isin(['QA-Base', 'DC-OSQA', 'BC-OSQA'])].copy()
    metric_specs = [('Unique_strict', 'Unique certified candidates'), ('Families_complete_linkage', 'Complete-linkage families')]
    instance_rows = []
    summary_rows = []
    for (metric, _) in metric_specs:
        pivot = df.pivot_table(index=['N', 'instance'], columns='solver_canonical', values=metric, aggfunc='first')
        for solver in ['DC-OSQA', 'BC-OSQA']:
            valid = pivot.dropna(subset=['QA-Base', solver]).copy()
            valid['gain'] = valid[solver] - valid['QA-Base']
            for ((n, instance), row) in valid.iterrows():
                instance_rows.append({'metric': metric, 'solver': solver, 'N': int(n), 'instance': int(instance), 'gain': float(row['gain'])})
            for (n, group) in valid.groupby(level='N'):
                values = group['gain']
                summary_rows.append({'metric': metric, 'solver': solver, 'N': int(n), 'pairs': len(values), 'mean_gain': float(values.mean()), 'sem_gain': sem(values)})
    instance_df = pd.DataFrame(instance_rows)
    summary_df = pd.DataFrame(summary_rows)
    instance_df.to_csv(OUT_INSTANCE, index=False)
    summary_df.to_csv(OUT_SUMMARY, index=False)
    (fig, axes) = plt.subplots(1, 2, figsize=(9.2, 3.75), constrained_layout=True)
    x = np.arange(len(N_ORDER), dtype=float)
    for (panel_idx, (ax, (metric, ylabel))) in enumerate(zip(axes, metric_specs)):
        for solver in ['DC-OSQA', 'BC-OSQA']:
            s = summary_df[summary_df['metric'].eq(metric) & summary_df['solver'].eq(solver)].set_index('N').reindex(N_ORDER)
            y = s['mean_gain'].to_numpy(float)
            e = s['sem_gain'].to_numpy(float)
            style = STYLE[solver]
            ax.fill_between(x, y - e, y + e, color=style['color'], alpha=0.13, linewidth=0, zorder=1)
            ax.plot(x, y, color=style['color'], marker=style['marker'], linestyle=style['linestyle'], linewidth=2.0, markersize=5.7, label=solver, zorder=3)
        ax.axhline(0, color='#555555', linestyle='--', linewidth=0.9, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in N_ORDER])
        ax.set_xlim(-0.35, len(N_ORDER) - 0.65)
        ax.set_xlabel('N')
        ax.set_ylabel(f'Paired gain over QA-Base\n({ylabel})')
        ax.text(-0.14, 1.04, f'({chr(97 + panel_idx)})', transform=ax.transAxes, fontsize=12, fontweight='bold')
        clean_axis(ax)
    axes[1].legend(frameon=False, fontsize=9, loc='upper left')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches='tight')
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    plt.close(fig)
if __name__ == '__main__':
    run_stage_02()
