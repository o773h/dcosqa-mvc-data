#!/usr/bin/env python3
from pathlib import Path
import gzip
import hashlib
import io
import json
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
STAGE = ROOT / 'code/main_qpu_read_budget_extension_n150'
SOURCE_SELECTION = STAGE / 'config/canonical_raw_source_selection.csv'
KEXACT_REFERENCE = Path(__file__).resolve().parents[3] / 'data' / 'metadata' / 'ba_m2_n150_kexact_reference.csv'
OUT_STREAM = STAGE / 'derived/verified_full_candidate_stream_from_raw.csv.gz'
CHUNK_SIZE = 100000
OUTPUT_COLUMNS = ['N', 'instance', 'solver', 'Bitstring', 'Size', 'K_exact', 'IsValid', 'strict_exact_verified', 'Occurrences', 'Energy', 'Chain_Break_Frac', 'Budget_Chunk_ID', 'source_dataset', 'source_key', 'source_role', 'source_file', 'source_order', 'source_row_order']

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda : handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def truthy(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({'true', '1', 'yes', 'y', 't'})

def canonical_solver(value: object) -> str:
    text = str(value).strip()
    low = text.lower()
    if 'dc-osqa' in low or 'dc_osqa' in low or 'dc-offset' in low or ('offset' in low) or ('inhomo' in low):
        return 'DC-OSQA'
    if 'qa-base' in low or 'qa_base' in low or low == 'standard' or ('standard qa' in low):
        return 'QA-Base'
    return text

def load_sources() -> pd.DataFrame:
    data = pd.read_csv(SOURCE_SELECTION)
    required = {'source_key', 'solver', 'role', 'selected_path', 'sha256'}
    missing = required - set(data.columns)
    if missing:
        raise RuntimeError('Source selection missing columns: ' + ', '.join(sorted(missing)))
    if len(data) != 8:
        raise RuntimeError(f'Expected 8 canonical sources, found {len(data)}')
    data = data.reset_index(drop=True)
    data['source_order'] = np.arange(len(data), dtype=int)
    for row in data.itertuples(index=False):
        path = Path(row.selected_path)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != str(row.sha256):
            raise RuntimeError(f'Source hash mismatch: {path}\nexpected={row.sha256}\nactual={actual}')
    return data

def load_kexact(path, graph_size):
    if not path.is_file():
        raise FileNotFoundError(path)
    data = pd.read_csv(path)
    required = {'N', 'instance', 'K_exact'}
    missing = required - set(data.columns)
    if missing:
        raise RuntimeError('K_exact reference missing columns: ' + ', '.join(sorted(missing)))
    data = data.copy()
    data['N'] = pd.to_numeric(data['N'], errors='raise').astype(int)
    data['instance'] = pd.to_numeric(data['instance'], errors='raise').astype(int)
    data['K_exact'] = pd.to_numeric(data['K_exact'], errors='raise').astype(int)
    data = data[data['N'].eq(int(graph_size))].copy()
    expected_instances = set(range(20))
    observed_instances = set(data['instance'])
    if observed_instances != expected_instances:
        raise RuntimeError('K_exact reference does not cover instances 0-19')
    if data.duplicated('instance').any():
        raise RuntimeError('Duplicate instance in K_exact reference')
    return dict(zip(data['instance'], data['K_exact']))

def run_stage_01() -> int:
    bundle_dir = Path(__file__).resolve().parents[1]
    (bundle_dir / 'derived').mkdir(parents=True, exist_ok=True)
    (bundle_dir / 'figures').mkdir(parents=True, exist_ok=True)
    sources = load_sources()
    kexact = load_kexact(KEXACT_REFERENCE, 150)
    if OUT_STREAM.exists():
        OUT_STREAM.unlink()
    wrote_header = False
    source_records = []
    instance_records = {}
    total_rows = 0
    total_occurrences = 0
    total_strict_rows = 0
    total_strict_occurrences = 0
    raw_output = OUT_STREAM.open('wb')
    gzip_output = gzip.GzipFile(filename='', mode='wb', fileobj=raw_output, mtime=0)
    text_output = io.TextIOWrapper(gzip_output, encoding='utf-8', newline='')
    for source in sources.itertuples(index=False):
        source_path = Path(source.selected_path)
        if not source_path.is_absolute():
            source_path = ROOT / source_path
        header = pd.read_csv(source_path, nrows=0)
        required_columns = {'Nodes', 'Instance', 'Solver', 'Bitstring', 'Energy', 'Size', 'IsValid', 'Occurrences', 'Chain_Break_Frac', 'Budget_Chunk_ID', 'Solver_Canonical', 'source_dataset'}
        missing = required_columns - set(header.columns)
        if missing:
            raise RuntimeError(f'{source_path}: missing columns ' + ', '.join(sorted(missing)))
        source_rows = 0
        source_occurrences = 0
        source_strict_rows = 0
        source_strict_occurrences = 0
        source_row_offset = 0
        for chunk in pd.read_csv(source_path, usecols=sorted(required_columns), dtype={'Bitstring': 'string'}, chunksize=CHUNK_SIZE):
            n_rows = len(chunk)
            chunk['source_row_order'] = np.arange(source_row_offset, source_row_offset + n_rows, dtype=np.int64)
            source_row_offset += n_rows
            chunk['Nodes'] = pd.to_numeric(chunk['Nodes'], errors='raise').astype(int)
            chunk['Instance'] = pd.to_numeric(chunk['Instance'], errors='raise').astype(int)
            chunk['Size'] = pd.to_numeric(chunk['Size'], errors='raise').astype(int)
            chunk['Occurrences'] = pd.to_numeric(chunk['Occurrences'], errors='raise').astype(int)
            if not chunk['Nodes'].eq(150).all():
                raise RuntimeError(f'{source_path}: contains Nodes != 150')
            if chunk['Occurrences'].le(0).any():
                raise RuntimeError(f'{source_path}: non-positive Occurrences')
            solver = chunk['Solver_Canonical'].map(canonical_solver)
            expected_solver = canonical_solver(source.solver)
            if not solver.eq(expected_solver).all():
                raise RuntimeError(f'{source_path}: solver mismatch')
            bitstrings = chunk['Bitstring'].astype(str).str.strip()
            valid_bits = bitstrings.str.fullmatch('[01]{150}')
            if not valid_bits.all():
                raise RuntimeError(f'{source_path}: malformed bitstrings')
            chunk_kexact = chunk['Instance'].map(kexact)
            if chunk_kexact.isna().any():
                raise RuntimeError(f'{source_path}: missing K_exact join')
            is_valid = truthy(chunk['IsValid'])
            strict = is_valid & chunk['Size'].eq(chunk_kexact.astype(int))
            source_file = source_path.relative_to(ROOT).as_posix()
            out = pd.DataFrame({'N': 150, 'instance': chunk['Instance'], 'solver': solver, 'Bitstring': bitstrings, 'Size': chunk['Size'], 'K_exact': chunk_kexact.astype(int), 'IsValid': is_valid, 'strict_exact_verified': strict, 'Occurrences': chunk['Occurrences'], 'Energy': chunk['Energy'], 'Chain_Break_Frac': chunk['Chain_Break_Frac'], 'Budget_Chunk_ID': chunk['Budget_Chunk_ID'], 'source_dataset': chunk['source_dataset'], 'source_key': source.source_key, 'source_role': source.role, 'source_file': source_file, 'source_order': int(source.source_order), 'source_row_order': chunk['source_row_order']})[OUTPUT_COLUMNS]
            out.to_csv(text_output, header=not wrote_header, index=False)
            wrote_header = True
            chunk_occurrences = int(out['Occurrences'].sum())
            chunk_strict_rows = int(out['strict_exact_verified'].sum())
            chunk_strict_occurrences = int(out.loc[out['strict_exact_verified'], 'Occurrences'].sum())
            source_rows += n_rows
            source_occurrences += chunk_occurrences
            source_strict_rows += chunk_strict_rows
            source_strict_occurrences += chunk_strict_occurrences
            total_rows += n_rows
            total_occurrences += chunk_occurrences
            total_strict_rows += chunk_strict_rows
            total_strict_occurrences += chunk_strict_occurrences
            grouped = out.groupby(['instance', 'solver'], sort=False)
            for ((instance, solver_name), group) in grouped:
                key = (int(instance), str(solver_name))
                record = instance_records.setdefault(key, {'N': 150, 'instance': int(instance), 'solver': str(solver_name), 'K_exact': int(kexact[int(instance)]), 'rows': 0, 'occurrences': 0, 'strict_exact_rows': 0, 'strict_exact_occurrences': 0})
                record['rows'] += len(group)
                record['occurrences'] += int(group['Occurrences'].sum())
                mask = group['strict_exact_verified'].astype(bool)
                record['strict_exact_rows'] += int(mask.sum())
                record['strict_exact_occurrences'] += int(group.loc[mask, 'Occurrences'].sum())
        source_records.append({'source_order': int(source.source_order), 'source_key': source.source_key, 'solver': expected_solver, 'role': source.role, 'source_file': source_path.relative_to(ROOT).as_posix(), 'rows': source_rows, 'occurrences': source_occurrences, 'strict_exact_rows': source_strict_rows, 'strict_exact_occurrences': source_strict_occurrences, 'sha256': sha256(source_path)})
    text_output.close()
    raw_output.close()
    source_summary = pd.DataFrame(source_records)
    instance_summary = pd.DataFrame(sorted(instance_records.values(), key=lambda row: (row['solver'], row['instance'])))
    expected_keys = {(instance, solver) for instance in range(20) for solver in ['QA-Base', 'DC-OSQA']}
    found_keys = {(int(row['instance']), str(row['solver'])) for row in instance_records.values()}
    per_solver_occurrences = instance_summary.groupby('solver')['occurrences'].sum().to_dict()
    complete_stream = found_keys == expected_keys and per_solver_occurrences == {'QA-Base': 1000000, 'DC-OSQA': 1000000} and instance_summary['occurrences'].eq(50000).all()
    manifest = {'status': 'PASS_VERIFIED_FULL_CANDIDATE_STREAM' if complete_stream else 'FAIL', 'canonical_source_count': int(len(sources)), 'rows': int(total_rows), 'occurrences': int(total_occurrences), 'strict_exact_rows': int(total_strict_rows), 'strict_exact_occurrences': int(total_strict_occurrences), 'instance_solver_rows': int(len(instance_summary)), 'per_solver_occurrences': {key: int(value) for (key, value) in per_solver_occurrences.items()}, 'all_instance_solver_occurrences_50000': bool(instance_summary['occurrences'].eq(50000).all()), 'coverage_complete': bool(found_keys == expected_keys), 'output': str(OUT_STREAM), 'output_bytes': int(OUT_STREAM.stat().st_size), 'output_sha256': sha256(OUT_STREAM), 'pass': bool(complete_stream)}
    return 0 if manifest['pass'] else 2
if __name__ == '__main__':
    try:
        raise SystemExit(run_stage_01())
    except Exception as exc:
        raise
