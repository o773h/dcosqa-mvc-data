#!/usr/bin/env python3
"""Rebuild the Figure 5 Hamming-family curve from the canonical 50k raw stream.

For each graph instance and protocol, the canonical candidate rows are expanded
conceptually in their stored source/row order.  At each read budget, the script
takes the deterministic occurrence prefix, retains strictly certified MVC
candidates, removes duplicate bitstrings, and applies complete-linkage
clustering at normalized Hamming threshold 0.1 (15 bits for N=150).

The instance table is checkpointed after every clustering case.  Re-running the
script resumes from that checkpoint.  Use ``--force`` to discard the checkpoint
and recompute all 240 instance--protocol--budget cases from the verified stream.
Large condensed distance vectors are stored temporarily as memory-mapped files
to avoid requiring the full pairwise matrix in RAM.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist


PACKAGE = Path(__file__).resolve().parents[3]
STAGE = PACKAGE / "code/main_qpu_read_budget_extension_n150"
STREAM = STAGE / "derived/verified_full_candidate_stream_from_raw.csv.gz"
OUT_DETAIL = STAGE / "derived/strict_exact_family_curve_by_instance.csv"
OUT_SUMMARY = STAGE / "derived/strict_exact_family_curve_summary.csv"
OUT_AUDIT = STAGE / "derived/strict_exact_family_curve_audit.json"

N = 150
THRESHOLD_FRACTION = 0.1
TAU_BITS = int(THRESHOLD_FRACTION * N)
BUDGETS = [1000, 2000, 5000, 10000, 20000, 50000]
SOLVERS = ["QA-Base", "DC-OSQA"]
EXPECTED_KEYS = {
    (instance, solver, budget)
    for instance in range(20)
    for solver in SOLVERS
    for budget in BUDGETS
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "y", "t"}
    )


def load_ordered_rows() -> dict[tuple[int, str], list[tuple[str, int, bool]]]:
    if not STREAM.is_file():
        raise FileNotFoundError(
            f"Missing verified stream: {STREAM}\n"
            "Run scripts/01_build_verified_candidate_stream.py first."
        )
    required = {
        "N",
        "instance",
        "solver",
        "Bitstring",
        "K_exact",
        "strict_exact_verified",
        "Occurrences",
        "source_order",
        "source_row_order",
    }
    header = pd.read_csv(STREAM, nrows=0)
    missing = required - set(header.columns)
    if missing:
        raise RuntimeError("Verified stream missing columns: " + ", ".join(sorted(missing)))

    data = pd.read_csv(
        STREAM,
        usecols=sorted(required),
        dtype={"Bitstring": "string"},
        low_memory=False,
    )
    for column in [
        "N",
        "instance",
        "K_exact",
        "Occurrences",
        "source_order",
        "source_row_order",
    ]:
        data[column] = pd.to_numeric(data[column], errors="raise").astype(int)
    if not data["N"].eq(N).all():
        raise RuntimeError("Verified stream contains N other than 150")
    if data["Occurrences"].le(0).any():
        raise RuntimeError("Verified stream contains non-positive Occurrences")
    if not data["solver"].isin(SOLVERS).all():
        bad = sorted(set(data.loc[~data["solver"].isin(SOLVERS), "solver"].astype(str)))
        raise RuntimeError(f"Unexpected solver labels: {bad}")
    data["strict_exact_verified"] = truthy(data["strict_exact_verified"])
    data = data.sort_values(
        ["source_order", "source_row_order"], kind="stable"
    ).reset_index(drop=True)

    stores: dict[tuple[int, str], list[tuple[str, int, bool]]] = {
        (instance, solver): []
        for instance in range(20)
        for solver in SOLVERS
    }
    for record in data.itertuples(index=False):
        bitstring = str(record.Bitstring)
        if len(bitstring) != N or set(bitstring) - {"0", "1"}:
            raise RuntimeError(
                f"Malformed bitstring for instance={record.instance}, solver={record.solver}"
            )
        key = (int(record.instance), str(record.solver))
        stores[key].append(
            (bitstring, int(record.Occurrences), bool(record.strict_exact_verified))
        )

    for key, rows in stores.items():
        available = sum(occurrences for _, occurrences, _ in rows)
        if available != 50000:
            raise RuntimeError(f"{key}: expected 50,000 reads, found {available}")
    return stores


def unique_strict_prefix(
    rows: list[tuple[str, int, bool]], budget: int
) -> tuple[list[str], int]:
    consumed = 0
    strict_occurrences = 0
    unique_bits: dict[str, None] = {}
    for bitstring, occurrences, strict_exact in rows:
        if consumed >= budget:
            break
        take = min(int(occurrences), int(budget - consumed))
        if strict_exact and take > 0:
            strict_occurrences += take
            unique_bits.setdefault(bitstring, None)
        consumed += take
    if consumed != budget:
        raise RuntimeError(f"Prefix budget {budget} incomplete: {consumed}")
    return list(unique_bits), int(strict_occurrences)


def bitstrings_to_matrix(bitstrings: list[str]) -> np.ndarray:
    if not bitstrings:
        return np.empty((0, N), dtype=np.uint8)
    joined = "".join(bitstrings).encode("ascii")
    matrix = np.frombuffer(joined, dtype=np.uint8).reshape(len(bitstrings), N)
    if not np.isin(matrix, [ord("0"), ord("1")]).all():
        raise RuntimeError("Non-binary character found in bitstrings")
    return (matrix == ord("1")).astype(np.uint8, copy=False)


def complete_linkage_family_count(
    bitstrings: list[str], memmap_threshold_gib: float, temp_dir: Path
) -> tuple[int, float, str]:
    n_unique = len(bitstrings)
    if n_unique == 0:
        return 0, 0.0, "trivial"
    if n_unique == 1:
        return 1, 0.0, "trivial"

    matrix = bitstrings_to_matrix(bitstrings)
    condensed_count = n_unique * (n_unique - 1) // 2
    condensed_gib = condensed_count * 8 / 1024**3
    distance_path: Path | None = None
    distances: np.ndarray | np.memmap
    storage = "memory"
    try:
        if condensed_gib >= memmap_threshold_gib:
            temp_dir.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(
                prefix=f"family_n{n_unique}_", suffix=".float64", dir=temp_dir
            )
            os.close(descriptor)
            distance_path = Path(name)
            distances = np.memmap(
                distance_path, mode="w+", dtype=np.float64, shape=(condensed_count,)
            )
            pdist(matrix, metric="hamming", out=distances)
            distances.flush()
            storage = "memmap"
        else:
            distances = pdist(matrix, metric="hamming")
        tree = linkage(distances, method="complete")
        labels = fcluster(
            tree, t=THRESHOLD_FRACTION, criterion="distance"
        )
        return int(np.unique(labels).size), float(condensed_gib), storage
    finally:
        if "distances" in locals():
            del distances
        if "tree" in locals():
            del tree
        del matrix
        gc.collect()
        if distance_path is not None and distance_path.exists():
            distance_path.unlink()


def load_checkpoint(force: bool) -> pd.DataFrame:
    if force:
        for path in (OUT_DETAIL, OUT_SUMMARY, OUT_AUDIT):
            if path.exists():
                path.unlink()
    if not OUT_DETAIL.exists():
        return pd.DataFrame()
    data = pd.read_csv(OUT_DETAIL)
    required = {
        "N",
        "instance",
        "solver",
        "budget",
        "families_complete_linkage",
        "unique_strict_exact_candidates",
        "strict_exact_occurrences",
    }
    if not required.issubset(data.columns):
        raise RuntimeError("Existing family checkpoint has an incompatible schema")
    if data.duplicated(["instance", "solver", "budget"]).any():
        raise RuntimeError("Existing family checkpoint contains duplicate keys")
    found = set(
        zip(data["instance"].astype(int), data["solver"], data["budget"].astype(int))
    )
    if not found.issubset(EXPECTED_KEYS):
        raise RuntimeError("Existing family checkpoint contains unexpected keys")
    return data


def write_checkpoint(records: list[dict[str, object]]) -> None:
    data = pd.DataFrame(records).sort_values(["solver", "instance", "budget"])
    temporary = OUT_DETAIL.with_suffix(".csv.tmp")
    data.to_csv(temporary, index=False)
    temporary.replace(OUT_DETAIL)


def build_summary(detail: pd.DataFrame) -> pd.DataFrame:
    summary = detail.groupby(["solver", "budget"], as_index=False).agg(
        mean_family=("families_complete_linkage", "mean"),
        sem_family=(
            "families_complete_linkage",
            lambda values: float(values.std(ddof=1) / math.sqrt(len(values)))
            if len(values) > 1
            else 0.0,
        ),
        mean_unique_strict_exact_candidates=("unique_strict_exact_candidates", "mean"),
        std_unique_strict_exact_candidates=("unique_strict_exact_candidates", "std"),
        sem_unique_strict_exact_candidates=(
            "unique_strict_exact_candidates",
            lambda values: float(values.std(ddof=1) / math.sqrt(len(values)))
            if len(values) > 1
            else 0.0,
        ),
        mean_strict_exact_occurrences=("strict_exact_occurrences", "mean"),
        n=("families_complete_linkage", "size"),
    )
    summary.insert(0, "N", N)
    return summary.sort_values(["solver", "budget"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="discard the checkpoint and recompute every family count",
    )
    parser.add_argument(
        "--memmap-threshold-gib",
        type=float,
        default=1.0,
        help="use a temporary memory-mapped distance vector at or above this size",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "qst_family_distances",
        help="directory for temporary memory-mapped distance vectors",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.memmap_threshold_gib <= 0:
        raise ValueError("--memmap-threshold-gib must be positive")
    OUT_DETAIL.parent.mkdir(parents=True, exist_ok=True)
    stores = load_ordered_rows()
    checkpoint = load_checkpoint(args.force)
    records = checkpoint.to_dict("records") if not checkpoint.empty else []
    completed = {
        (int(row["instance"]), str(row["solver"]), int(row["budget"]))
        for row in records
    }
    if completed != EXPECTED_KEYS:
        for path in (OUT_SUMMARY, OUT_AUDIT):
            if path.exists():
                path.unlink()

    pending: list[tuple[int, str, int, list[str], int]] = []
    for solver in SOLVERS:
        for instance in range(20):
            for budget in BUDGETS:
                key = (instance, solver, budget)
                if key in completed:
                    continue
                bitstrings, strict_occurrences = unique_strict_prefix(
                    stores[instance, solver], budget
                )
                pending.append(
                    (instance, solver, budget, bitstrings, strict_occurrences)
                )

    # Small cases first, so a long run leaves the most useful checkpoint possible.
    pending.sort(key=lambda item: (len(item[3]), item[1], item[0], item[2]))
    total_pending = len(pending)
    for case_number, (instance, solver, budget, bitstrings, strict_occurrences) in enumerate(
        pending, start=1
    ):
        started = time.time()
        family_count, condensed_gib, storage = complete_linkage_family_count(
            bitstrings, args.memmap_threshold_gib, args.temp_dir
        )
        elapsed = time.time() - started
        records.append(
            {
                "N": N,
                "instance": int(instance),
                "solver": solver,
                "budget": int(budget),
                "sampled_reads": int(budget),
                "strict_exact_occurrences": int(strict_occurrences),
                "unique_strict_exact_candidates": int(len(bitstrings)),
                "families_complete_linkage": int(family_count),
                "tau_bits": int(TAU_BITS),
                "distance_threshold_fraction": float(THRESHOLD_FRACTION),
                "condensed_float64_gib": float(condensed_gib),
                "distance_storage": storage,
                "elapsed_seconds": float(elapsed),
            }
        )
        write_checkpoint(records)
        print(
            f"[{case_number}/{total_pending}] instance={instance} solver={solver} "
            f"budget={budget} unique={len(bitstrings)} families={family_count} "
            f"distance_gib={condensed_gib:.3f} storage={storage} elapsed={elapsed:.1f}s",
            flush=True,
        )

    detail = pd.read_csv(OUT_DETAIL)
    found_keys = set(
        zip(detail["instance"].astype(int), detail["solver"], detail["budget"].astype(int))
    )
    if found_keys != EXPECTED_KEYS:
        missing = sorted(EXPECTED_KEYS - found_keys)
        raise RuntimeError(f"Family curve is incomplete; missing keys: {missing[:10]}")
    summary = build_summary(detail)
    if len(summary) != 12 or not summary["n"].eq(20).all():
        raise RuntimeError("Family summary does not contain 20 instances for all 12 rows")
    summary.to_csv(OUT_SUMMARY, index=False)
    audit = {
        "status": "PASS_RAW_TO_FAMILY_CURVE",
        "verified_stream": str(STREAM.relative_to(PACKAGE)),
        "verified_stream_sha256": sha256(STREAM),
        "detail": str(OUT_DETAIL.relative_to(PACKAGE)),
        "detail_sha256": sha256(OUT_DETAIL),
        "summary": str(OUT_SUMMARY.relative_to(PACKAGE)),
        "summary_sha256": sha256(OUT_SUMMARY),
        "instance_solver_budget_rows": int(len(detail)),
        "summary_rows": int(len(summary)),
        "budget_definition": "deterministic occurrence prefix in canonical raw source and row order",
        "certification_definition": "IsValid AND Size == graph-level K_exact",
        "family_definition": "complete-linkage clustering of unique certified bitstrings at normalized Hamming threshold 0.1 (15 bits for N=150)",
        "pass": True,
    }
    OUT_AUDIT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
