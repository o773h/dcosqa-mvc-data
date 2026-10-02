#!/usr/bin/env python3
"""Compute and plot the random 3-regular boundary control."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parents[1]
VERIFIED_STREAM = BASE / "derived" / "verified_candidate_stream.csv.gz"
DERIVED = BASE / "derived"
CURATED = BASE / "curated"

N = 150
TAU_FRACTION = 0.1
EXPECTED_SOLVERS = {"QA-Base", "DC-OSQA"}


def truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "y", "t"}
    )


def canonical_solver(value: object) -> str:
    text = str(value).strip()
    low = text.lower()
    if low in {"qa-base", "qabase", "qa_base", "qa base"}:
        return "QA-Base"
    if low in {"dc-osqa", "dcosqa", "dc_osqa", "dc osqa"}:
        return "DC-OSQA"
    if "qa-base" in low or "qabase" in low:
        return "QA-Base"
    if "dc" in low:
        return "DC-OSQA"
    return text


def complete_linkage_family_count(bitstrings: pd.Series, n: int) -> int:
    values = sorted(set(bitstrings.astype(str)))
    if len(values) <= 1:
        return len(values)
    matrix = np.array(
        [[1 if character == "1" else 0 for character in value] for value in values],
        dtype=np.uint8,
    )
    distances = pdist(matrix, metric="hamming") * int(n)
    hierarchy = linkage(distances, method="complete")
    labels = fcluster(
        hierarchy, t=TAU_FRACTION * int(n), criterion="distance"
    )
    return int(np.unique(labels).size)


def safe_wilcoxon(dc: np.ndarray, qa: np.ndarray) -> float:
    difference = np.asarray(dc, dtype=float) - np.asarray(qa, dtype=float)
    if difference.size == 0 or np.allclose(difference, 0):
        return float("nan")
    return float(
        wilcoxon(difference, alternative="two-sided", zero_method="wilcox").pvalue
    )


def make_summary(pair: pd.DataFrame) -> pd.DataFrame:
    metric_map = [
        ("Unique certified candidates", "Unique_strict"),
        ("Certified occurrences", "Strict_occurrences"),
        ("Complete-linkage Hamming families", "Families_complete_linkage"),
    ]
    rows = []
    for label, metric in metric_map:
        qa = pair[f"QA-Base__{metric}"].to_numpy(float)
        dc = pair[f"DC-OSQA__{metric}"].to_numpy(float)
        gain = dc - qa
        rows.append(
            {
                "metric": label,
                "pairs": len(pair),
                "qa_mean": qa.mean(),
                "dc_mean": dc.mean(),
                "mean_gain": gain.mean(),
                "median_gain": np.median(gain),
                "wins": int((gain > 0).sum()),
                "losses": int((gain < 0).sum()),
                "ties": int((gain == 0).sum()),
                "wilcoxon_two_sided_p": safe_wilcoxon(dc, qa),
            }
        )
    return pd.DataFrame(rows)


def validate_claims(summary: pd.DataFrame) -> None:
    expected = {
        "Unique certified candidates": (299.35, 180.75, -118.60, -68.5, 1, 19, 0),
        "Certified occurrences": (438.80, 258.95, -179.85, -152.5, 1, 19, 0),
        "Complete-linkage Hamming families": (27.70, 19.35, -8.35, -3.0, 1, 18, 1),
    }
    for label, values in expected.items():
        row = summary.loc[summary["metric"].eq(label)]
        if len(row) != 1:
            raise RuntimeError(f"missing regular-3 summary row: {label}")
        record = row.iloc[0]
        observed = (
            round(float(record["qa_mean"]), 2),
            round(float(record["dc_mean"]), 2),
            round(float(record["mean_gain"]), 2),
            round(float(record["median_gain"]), 1),
            int(record["wins"]),
            int(record["losses"]),
            int(record["ties"]),
        )
        if observed != values:
            raise RuntimeError(
                f"regular-3 manuscript claim mismatch for {label}: "
                f"observed={observed}, expected={values}"
            )


def clean_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)


def plot_figure(pair: pd.DataFrame) -> None:
    metrics = [
        (
            "DC-OSQA__Unique_strict",
            "QA-Base__Unique_strict",
            "Unique certified\ncandidates",
        ),
        (
            "DC-OSQA__Strict_occurrences",
            "QA-Base__Strict_occurrences",
            "Certified\noccurrences",
        ),
        (
            "DC-OSQA__Families_complete_linkage",
            "QA-Base__Families_complete_linkage",
            "Complete-linkage\nfamilies",
        ),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(10.7, 4.25), constrained_layout=True)
    for panel, (axis, (dc_column, qa_column, title)) in enumerate(zip(axes, metrics)):
        gain = (
            pd.to_numeric(pair[dc_column], errors="raise")
            - pd.to_numeric(pair[qa_column], errors="raise")
        ).to_numpy(float)
        ordered_gain = np.sort(gain)
        x = np.arange(len(ordered_gain))
        point_colors = np.where(
            ordered_gain > 0,
            "#2F6F8F",
            np.where(ordered_gain < 0, "#777777", "#BBBBBB"),
        )
        axis.vlines(
            x, 0, ordered_gain, color=point_colors, alpha=0.42, linewidth=1.0, zorder=1
        )
        axis.scatter(
            x,
            ordered_gain,
            s=32,
            color=point_colors,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        axis.axhline(0, color="#333333", linestyle="--", linewidth=0.95)
        mean_gain = float(gain.mean())
        axis.axhline(mean_gain, color="#B65A1B", linewidth=1.45)
        wins = int((gain > 0).sum())
        losses = int((gain < 0).sum())
        ties = int((gain == 0).sum())
        axis.text(
            0.52,
            1.015,
            f"DC higher {wins}/{len(gain)} · QA higher {losses}/{len(gain)} · "
            f"ties {ties}/{len(gain)}",
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=7.8,
            color="#444444",
            clip_on=False,
        )
        axis.set_xlabel("Paired graph instances\n(sorted by gain)")
        axis.set_ylabel(f"{title}\nDC-OSQA − QA-Base")
        axis.set_xticks([])
        clean_axis(axis)
        axis.text(
            -0.10,
            1.015,
            f"({chr(97 + panel)})",
            transform=axis.transAxes,
            fontsize=12.5,
            fontweight="bold",
            ha="left",
            va="bottom",
            clip_on=False,
            zorder=20,
        )
    CURATED.mkdir(parents=True, exist_ok=True)
    figure.savefig(CURATED / "schedule_regular3_controls.pdf", bbox_inches="tight")
    figure.savefig(
        CURATED / "schedule_regular3_controls.png", dpi=300, bbox_inches="tight"
    )
    plt.close(figure)


def main() -> None:
    if not VERIFIED_STREAM.is_file():
        raise FileNotFoundError(
            f"missing verified stream; run 00_build_verified_candidate_stream.py: "
            f"{VERIFIED_STREAM}"
        )
    data = pd.read_csv(
        VERIFIED_STREAM, dtype={"Bitstring": "string"}, low_memory=False
    )
    required = {
        "Nodes",
        "Instance",
        "Solver",
        "Solver_Canonical",
        "Bitstring",
        "Occurrences",
        "EmbeddingRep",
        "EmbeddingSeed",
        "K_exact",
        "strict_exact_verified",
        "Graph_Seed",
        "graph_hash_sha256",
    }
    missing = required - set(data.columns)
    if missing:
        raise RuntimeError(f"verified regular-3 stream missing: {sorted(missing)}")
    data["Solver_Canonical"] = data["Solver_Canonical"].map(canonical_solver)
    if set(data["Solver_Canonical"]) != EXPECTED_SOLVERS:
        raise RuntimeError("unexpected solver in verified regular-3 stream")
    data["K_exact"] = pd.to_numeric(data["K_exact"], errors="raise").astype(int)
    data["Occurrences"] = pd.to_numeric(data["Occurrences"], errors="raise").astype(int)

    DERIVED.mkdir(parents=True, exist_ok=True)
    k_exact = data[["Instance", "K_exact"]].drop_duplicates().sort_values("Instance")
    if len(k_exact) != 20 or k_exact.groupby("Instance")["K_exact"].nunique().gt(1).any():
        raise RuntimeError("regular-3 K_exact registry is incomplete or inconsistent")
    k_exact.to_csv(DERIVED / "regular3_n150_kexact_by_instance.csv", index=False)

    strict = data.loc[truthy(data["strict_exact_verified"])].copy()
    key = ["Nodes", "Instance", "Solver_Canonical", "EmbeddingRep", "EmbeddingSeed"]
    counts = strict.groupby(key, dropna=False).agg(
        K_exact=("K_exact", "first"),
        Exact_rows=("Bitstring", "size"),
        Unique_strict=("Bitstring", pd.Series.nunique),
        Strict_occurrences=("Occurrences", "sum"),
    ).reset_index()

    family_rows = []
    for key_values, group in strict.groupby(key, dropna=False):
        record = dict(zip(key, key_values))
        record["Families_complete_linkage"] = complete_linkage_family_count(
            group["Bitstring"], int(group["Nodes"].iloc[0])
        )
        record["Exact_unique_bitstrings_for_families"] = int(
            group["Bitstring"].nunique()
        )
        family_rows.append(record)
    families = pd.DataFrame(family_rows)

    keyspace = data[key].drop_duplicates()
    by_solver = keyspace.merge(counts, on=key, how="left").merge(
        families, on=key, how="left"
    )
    by_solver = by_solver.merge(k_exact, on="Instance", how="left", validate="many_to_one")
    if "K_exact_x" in by_solver.columns:
        by_solver["K_exact"] = by_solver["K_exact_x"].fillna(by_solver["K_exact_y"])
        by_solver = by_solver.drop(columns=["K_exact_x", "K_exact_y"])
    for column in [
        "Exact_rows",
        "Unique_strict",
        "Strict_occurrences",
        "Families_complete_linkage",
        "Exact_unique_bitstrings_for_families",
    ]:
        by_solver[column] = by_solver[column].fillna(0).astype(int)
    by_solver["K_exact"] = by_solver["K_exact"].astype(int)
    by_solver.to_csv(DERIVED / "regular3_n150_exact_by_solver_instance.csv", index=False)

    pair = by_solver.pivot_table(
        index=["Instance", "EmbeddingRep", "EmbeddingSeed"],
        columns="Solver_Canonical",
        values=["Unique_strict", "Strict_occurrences", "Families_complete_linkage"],
        aggfunc="first",
    )
    pair.columns = [f"{solver}__{metric}" for metric, solver in pair.columns]
    pair = pair.reset_index()
    required_pair = [
        f"{solver}__{metric}"
        for solver in ["QA-Base", "DC-OSQA"]
        for metric in [
            "Unique_strict",
            "Strict_occurrences",
            "Families_complete_linkage",
        ]
    ]
    pair = pair.dropna(subset=required_pair)
    if len(pair) != 20:
        raise RuntimeError(f"expected 20 paired regular-3 instances, found {len(pair)}")
    pair.to_csv(DERIVED / "regular3_n150_exact_pair_table.csv", index=False)

    summary = make_summary(pair)
    validate_claims(summary)
    summary.to_csv(DERIVED / "regular3_n150_exact_summary_stats.csv", index=False)
    plot_figure(pair)
    print("PASS regular-3 S9 analysis and manuscript-claim checks")


if __name__ == "__main__":
    main()
