#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[3]
B = ROOT / 'code/supp_budget_diagnostic_qabase_dc'
DER = B / 'derived'
DER.mkdir(parents=True, exist_ok=True)

QA_RAW = ROOT / 'data/raw/qa_base_budget10k_n150/raw_solutions.csv'
DC_RAW = ROOT / 'data/raw/qa_budget10k_n150/raw_solutions.csv'
THRESHOLD = 0.01
EXPECTED_INSTANCES = 20

def load_chain_break(path: Path, solver_pattern: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, usecols=['Instance', 'Solver', 'Occurrences', 'Chain_Break_Frac'])
    df = df[df['Solver'].astype(str).str.contains(solver_pattern, regex=False, na=False)].copy()
    df['Chain_Break_Frac'] = pd.to_numeric(df['Chain_Break_Frac'], errors='raise')
    df['Occurrences'] = pd.to_numeric(df['Occurrences'], errors='raise').astype(int)
    if df['Chain_Break_Frac'].isna().any():
        raise RuntimeError(f'missing Chain_Break_Frac values in {path}')
    if (df['Occurrences'] < 1).any():
        raise RuntimeError(f'nonpositive Occurrences values in {path}')
    return df

def per_instance_stats(df: pd.DataFrame, label: str) -> pd.DataFrame:
    def weighted(group: pd.DataFrame) -> pd.Series:
        values = group['Chain_Break_Frac'].to_numpy(dtype=float)
        weights = group['Occurrences'].to_numpy(dtype=float)
        total = weights.sum()
        return pd.Series({
            'mean_fraction': float(np.average(values, weights=weights)),
            'max_fraction': float(values.max()),
            'fraction_over_threshold': float(weights[values > THRESHOLD].sum() / total),
            'reads': int(total),
        })
    out = df.groupby('Instance').apply(weighted, include_groups=False)
    if len(out) != EXPECTED_INSTANCES:
        raise RuntimeError(f'{label}: expected {EXPECTED_INSTANCES} instances; got {len(out)}')
    out.insert(0, 'solver_canonical', label)
    return out.reset_index()

qa = per_instance_stats(load_chain_break(QA_RAW, 'QA-Base'), 'QA-Base')
dc = per_instance_stats(load_chain_break(DC_RAW, 'DC-Offset(s=0.1)'), 'DC-OSQA')
by_instance = pd.concat([qa, dc], ignore_index=True)
by_instance.to_csv(DER / 'chain_break_by_instance_10k_n150.csv', index=False)

summary_rows = []
for metric in ['mean_fraction', 'max_fraction', 'fraction_over_threshold']:
    qa_vals = qa.set_index('Instance')[metric]
    dc_vals = dc.set_index('Instance')[metric]
    diff = (dc_vals - qa_vals).dropna()
    nonzero = diff[diff != 0.0]
    p_value = float(wilcoxon(nonzero).pvalue) if len(nonzero) else float('nan')
    summary_rows.append({
        'metric': metric,
        'pairs': int(len(diff)),
        'dc_mean': float(dc_vals.mean()),
        'qa_mean': float(qa_vals.mean()),
        'mean_diff': float(diff.mean()),
        'median_diff': float(diff.median()),
        'wilcoxon_two_sided_p': p_value,
    })
summary = pd.DataFrame(summary_rows)
summary.to_csv(DER / 'chain_break_summary_10k_n150.csv', index=False)
print(summary.to_string(index=False))
