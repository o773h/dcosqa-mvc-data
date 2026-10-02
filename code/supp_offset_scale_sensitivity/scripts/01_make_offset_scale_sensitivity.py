#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
BUNDLE = Path(__file__).resolve().parents[1]
ROOT = BUNDLE.parents[1]
RAW_ROOTS = [ROOT / 'data/raw', ROOT / 'combined_raw']
METADATA_REL = 'analysis_ready/offset_scale_sensitivity_input.csv'
METADATA_SOURCE = BUNDLE / METADATA_REL
DERIVED = BUNDLE / 'derived'
AUDIT = BUNDLE / 'audit'
OUT_DATA = DERIVED / 'offset_scale_sensitivity_input_strict_exact_from_raw.csv'
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
INPUT = BASE / 'derived' / 'offset_scale_sensitivity_input_strict_exact_from_raw.csv'
OUT_DIR = BASE / 'curated'
OUT_PDF = OUT_DIR / 'offset_scale_sensitivity.pdf'
OUT_PNG = OUT_DIR / 'offset_scale_sensitivity.png'
OUT_SUMMARY = BASE / 'derived' / 'offset_scale_sensitivity_summary.csv'
ALPHAS = [0.05, 0.1, 0.2]
MANUSCRIPT_N = [70, 90, 120, 150]
QA_COLOR = '#5F5F5F'
DC_COLOR = '#2F6F8F'

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
    required = {'N', 'instance', 'solver_family', 'alpha', 'Unique_strict', 'Families_complete_linkage'}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f'missing columns: {missing}')
    df['N'] = pd.to_numeric(df['N'], errors='coerce')
    df['alpha'] = pd.to_numeric(df['alpha'], errors='coerce')
    df = df[df['N'].isin(MANUSCRIPT_N)].copy()
    metric_specs = [('Unique_strict', 'Unique certified candidates'), ('Families_complete_linkage', 'Complete-linkage families')]
    summary_rows = []
    for (metric, _) in metric_specs:
        qa = df[df['solver_family'].eq('QA-Base')][metric]
        qa_mean = float(pd.to_numeric(qa, errors='coerce').mean())
        qa_sem = sem(qa)
        for alpha in ALPHAS:
            values = df[df['solver_family'].eq('DC-OSQA') & np.isclose(df['alpha'], alpha)][metric]
            summary_rows.append({'metric': metric, 'series': 'DC-OSQA', 'alpha': alpha, 'observations': len(values), 'mean': float(pd.to_numeric(values, errors='coerce').mean()), 'sem': sem(values)})
            summary_rows.append({'metric': metric, 'series': 'QA-Base', 'alpha': alpha, 'observations': len(qa), 'mean': qa_mean, 'sem': qa_sem})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_SUMMARY, index=False)
    (fig, axes) = plt.subplots(1, 2, figsize=(8.8, 3.65), constrained_layout=True)
    x = np.arange(len(ALPHAS), dtype=float)
    for (panel_idx, (ax, (metric, ylabel))) in enumerate(zip(axes, metric_specs)):
        for (series, color, marker, linestyle) in [('QA-Base', QA_COLOR, 'o', '--'), ('DC-OSQA', DC_COLOR, 's', '-')]:
            s = summary[summary['metric'].eq(metric) & summary['series'].eq(series)].set_index('alpha').reindex(ALPHAS)
            y = s['mean'].to_numpy(float)
            e = s['sem'].to_numpy(float)
            ax.fill_between(x, y - e, y + e, color=color, alpha=0.12, linewidth=0, zorder=1)
            ax.plot(x, y, color=color, marker=marker, linestyle=linestyle, linewidth=2.0, markersize=5.8, label=series, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(['0.05', '0.10', '0.20'])
        ax.set_xlim(-0.25, len(ALPHAS) - 0.75)
        ax.set_xlabel('Anneal-offset scale, $\\alpha$')
        ax.set_ylabel(ylabel)
        ax.text(-0.14, 1.04, f'({chr(97 + panel_idx)})', transform=ax.transAxes, fontsize=12, fontweight='bold')
        clean_axis(ax)
    axes[0].legend(frameon=False, fontsize=9, loc='best')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches='tight')
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    plt.close(fig)
if __name__ == '__main__':
    run_stage_02()
