#!/usr/bin/env python3
"""Recompute the matched ER N=150 QA-Base/DC-OSQA control from raw rows."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "code/supp_er_n150_qabase_vs_dcosqa"
RAW = BASE / "raw"
REGISTRY = BASE / "graph_registry.csv"
DERIVED = BASE / "derived"
CURATED = BASE / "curated"
OUT_PDF = CURATED / "er_n150_qabase_vs_dcosqa.pdf"
OUT_PNG = CURATED / "er_n150_qabase_vs_dcosqa.png"
N = 150
P = 4.0 / (N - 1)
TAU = 15


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def family_count(bitstrings: list[str]) -> int:
    unique = sorted(set(bitstrings))
    if len(unique) <= 1:
        return len(unique)
    matrix = np.array([[int(char) for char in bitstring] for bitstring in unique], dtype=np.int8)
    tree = linkage(pdist(matrix, metric="hamming") * N, method="complete")
    return int(np.unique(fcluster(tree, t=TAU, criterion="distance")).size)


def validate_registry(registry: pd.DataFrame) -> dict[int, nx.Graph]:
    required = {"Instance", "N", "K_exact", "Edges", "Graph_Seed"}
    missing = required - set(registry.columns)
    if missing:
        raise RuntimeError(f"Graph registry missing columns: {sorted(missing)}")
    if set(registry["Instance"].astype(int)) != set(range(10)):
        raise RuntimeError("ER registry must cover instances 0--9")
    graphs: dict[int, nx.Graph] = {}
    for row in registry.itertuples(index=False):
        instance = int(row.Instance)
        graph = nx.erdos_renyi_graph(N, p=P, seed=int(row.Graph_Seed))
        if graph.number_of_edges() != int(row.Edges):
            raise RuntimeError(f"ER edge-count mismatch for instance {instance}")
        graphs[instance] = graph
    return graphs


def load_raw(path: Path, solver: str) -> pd.DataFrame:
    required = {"Nodes", "Instance", "Bitstring", "Size", "IsValid", "Occurrences"}
    frame = pd.read_csv(path, dtype={"Bitstring": "string"}, low_memory=False)
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path.name} missing columns: {sorted(missing)}")
    if not frame["Nodes"].eq(N).all():
        raise RuntimeError(f"{path.name} contains Nodes != {N}")
    bits = frame["Bitstring"].str.strip()
    if bits.isna().any() or not bits.str.fullmatch(f"[01]{{{N}}}").all():
        raise RuntimeError(f"Malformed bitstrings in {path.name}")
    frame["Bitstring"] = bits
    frame["Occurrences"] = pd.to_numeric(frame["Occurrences"], errors="raise").astype(int)
    frame["Size"] = pd.to_numeric(frame["Size"], errors="raise").astype(int)
    frame["solver"] = solver
    totals = frame.groupby("Instance")["Occurrences"].sum()
    if set(totals.index.astype(int)) != set(range(10)) or not totals.eq(1000).all():
        raise RuntimeError(f"{path.name} must contain 1,000 reads for each instance")
    return frame


def recompute() -> tuple[pd.DataFrame, pd.DataFrame]:
    registry = pd.read_csv(REGISTRY)
    graphs = validate_registry(registry)
    kexact = registry.set_index("Instance")["K_exact"].astype(int).to_dict()
    frames = [
        load_raw(RAW / "qa_base_raw.csv.gz", "QA-Base"),
        load_raw(RAW / "dc_osqa_raw.csv.gz", "DC-OSQA"),
    ]
    data = pd.concat(frames, ignore_index=True)
    rows = []
    validity_mismatches = 0
    for (instance, solver), group in data.groupby(["Instance", "solver"], sort=True):
        instance = int(instance)
        graph = graphs[instance]
        edges = list(graph.edges())
        strict_bitstrings: list[str] = []
        strict_occurrences = 0
        for row in group.itertuples(index=False):
            bitstring = str(row.Bitstring)
            selected = np.fromiter((char == "1" for char in bitstring), dtype=bool, count=N)
            feasible = all(selected[u] or selected[v] for u, v in edges)
            if feasible != bool(as_bool(pd.Series([row.IsValid])).iloc[0]):
                validity_mismatches += 1
            if feasible and int(selected.sum()) == int(kexact[instance]):
                strict_bitstrings.append(bitstring)
                strict_occurrences += int(row.Occurrences)
        rows.append({
            "N": N,
            "instance": instance,
            "solver": solver,
            "K_exact": int(kexact[instance]),
            "total_reads": int(group["Occurrences"].sum()),
            "unique_certified_candidates": int(len(set(strict_bitstrings))),
            "certified_occurrences": int(strict_occurrences),
            "hamming_families": family_count(strict_bitstrings),
        })
    if validity_mismatches:
        raise RuntimeError(f"Stored and graph-recomputed validity disagree in {validity_mismatches} rows")
    by_solver = pd.DataFrame(rows).sort_values(["instance", "solver"])
    paired = by_solver.pivot(
        index="instance",
        columns="solver",
        values=["unique_certified_candidates", "certified_occurrences", "hamming_families"],
    )
    paired.columns = [f"{metric}__{solver}" for metric, solver in paired.columns]
    paired = paired.reset_index()
    for metric in ["unique_certified_candidates", "certified_occurrences", "hamming_families"]:
        paired[f"gain__{metric}"] = paired[f"{metric}__DC-OSQA"] - paired[f"{metric}__QA-Base"]
    return by_solver, paired


def statistics(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ["unique_certified_candidates", "certified_occurrences", "hamming_families"]:
        qa = paired[f"{metric}__QA-Base"].to_numpy(float)
        dc = paired[f"{metric}__DC-OSQA"].to_numpy(float)
        gain = dc - qa
        p_value = float(wilcoxon(dc, qa, alternative="two-sided", zero_method="wilcox").pvalue) if not np.allclose(gain, 0) else np.nan
        rows.append({
            "metric": metric,
            "n_pairs": len(gain),
            "qa_mean": float(qa.mean()),
            "dc_mean": float(dc.mean()),
            "mean_gain_dc_minus_qa": float(gain.mean()),
            "median_gain_dc_minus_qa": float(np.median(gain)),
            "dc_wins": int((gain > 0).sum()),
            "qa_wins": int((gain < 0).sum()),
            "ties": int((gain == 0).sum()),
            "wilcoxon_two_sided_p": p_value,
        })
    return pd.DataFrame(rows)


def plot(paired: pd.DataFrame) -> None:
    panels = [
        ("gain__unique_certified_candidates", "Unique certified candidates"),
        ("gain__hamming_families", r"Hamming families ($\tau=0.1N$)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.75))
    for label, ax, (column, ylabel) in zip(["(a)", "(b)"], axes, panels):
        values = paired[column].to_numpy(float)
        x = np.arange(len(values))
        ax.axhline(0, color="#555555", linestyle="--", linewidth=1.0)
        ax.vlines(x, 0, values, color="#B7B7B7", linewidth=1.0)
        ax.scatter(x, values, c=np.where(values > 0, "#2F6F8F", np.where(values < 0, "#7A7A7A", "#C49A5A")), s=31, zorder=3)
        ax.axhline(values.mean(), color="#D55E00", linewidth=1.4, label=f"Mean {values.mean():.1f}")
        ax.set_xticks(x)
        ax.set_xticklabels(paired["instance"].astype(int))
        ax.set_xlabel("ER graph instance")
        ax.set_ylabel("DC-OSQA minus QA-Base\n" + ylabel)
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="lower left")
        ax.text(-0.14, 1.04, label, transform=ax.transAxes, fontsize=12, fontweight="bold")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    CURATED.mkdir(parents=True, exist_ok=True)
    by_solver, paired = recompute()
    stats = statistics(paired)
    by_solver.to_csv(DERIVED / "er_n150_metrics_by_instance_solver.csv", index=False)
    paired.to_csv(DERIVED / "er_n150_paired_gains.csv", index=False)
    stats.to_csv(DERIVED / "er_n150_paired_statistics.csv", index=False)
    plot(paired)
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
