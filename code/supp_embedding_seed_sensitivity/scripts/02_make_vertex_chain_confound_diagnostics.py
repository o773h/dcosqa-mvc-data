#!/usr/bin/env python3
"""Diagnose degree/chain confounding on the canonical BA m=2 comparison."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "code/supp_embedding_seed_sensitivity"
MAPPING = ROOT / "data/metadata/canonical_embeddings/ba_m2_vertex_chain_mappings.csv.gz"
QA_RAW = ROOT / "data/raw/qa_base_main_n20_150/raw_solutions.csv"
DC_RAW = ROOT / "data/raw/main_ba_m2_dcosqa_offset01/raw_candidate_bitstrings.csv"
DERIVED = BASE / "derived"
CURATED = BASE / "curated"
OUT_PDF = CURATED / "vertex_chain_confound_diagnostics.pdf"
OUT_PNG = CURATED / "vertex_chain_confound_diagnostics.png"
SIZES = (70, 80, 90, 100, 120, 150)


def truthy(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def load_mapping() -> pd.DataFrame:
    frame = pd.read_csv(MAPPING)
    required = {"Graph_Type", "Graph_M", "Nodes", "Instance", "LogicalNode", "PhysicalChain", "ChainLength", "Degree"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Canonical mapping missing columns: {sorted(missing)}")
    frame = frame[
        frame["Graph_Type"].eq("BA")
        & frame["Graph_M"].eq(2)
        & frame["Nodes"].isin(SIZES)
    ].copy()
    chains = frame["PhysicalChain"].map(json.loads)
    if not chains.map(lambda chain: isinstance(chain, list) and len(chain) > 0).all():
        raise RuntimeError("Malformed physical chain in canonical mapping")
    if not chains.map(len).eq(frame["ChainLength"].astype(int)).all():
        raise RuntimeError("Chain-length mismatch in canonical mapping")
    expected_groups = {(nodes, instance) for nodes in SIZES for instance in range(20)}
    observed_groups = set(zip(frame["Nodes"].astype(int), frame["Instance"].astype(int)))
    if observed_groups != expected_groups or len(frame) != 12200:
        raise RuntimeError("Canonical larger-size mapping coverage is incomplete")
    return frame.rename(columns={"LogicalNode": "vertex", "Degree": "MappingDegree"})[
        ["Nodes", "Instance", "vertex", "ChainLength", "MappingDegree"]
    ]


def degree_chain_correlations(mapping: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (nodes, instance), group in mapping.groupby(["Nodes", "Instance"]):
        rows.append({
            "Nodes": int(nodes),
            "Instance": int(instance),
            "rho_degree_chain_length": float(spearmanr(group["MappingDegree"], group["ChainLength"]).statistic),
            "mean_chain_length": float(group["ChainLength"].mean()),
            "maximum_chain_length": int(group["ChainLength"].max()),
        })
    result = pd.DataFrame(rows).sort_values(["Nodes", "Instance"])
    if len(result) != 120:
        raise RuntimeError(f"Expected 120 canonical graph mappings, found {len(result)}")
    return result


def certified_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    dc = pd.read_csv(DC_RAW, dtype={"Bitstring": "string"}, low_memory=False)
    dc = dc[dc["Solver"].eq("DC-Offset(s=0.1)") & dc["Nodes"].isin(SIZES)].copy()
    dc["Size"] = pd.to_numeric(dc["Size"], errors="raise")
    dc["K_exact"] = pd.to_numeric(dc["K_exact"], errors="raise")
    dc = dc[truthy(dc["IsValid"]) & dc["Size"].eq(dc["K_exact"])]

    reference = dc.groupby(["Nodes", "Instance"], as_index=False)["K_exact"].first()
    if len(reference) != 120:
        raise RuntimeError("Canonical K_exact coverage is incomplete")
    qa = pd.read_csv(QA_RAW, dtype={"Bitstring": "string"}, low_memory=False)
    qa = qa[qa["Nodes"].isin(SIZES)].merge(
        reference, on=["Nodes", "Instance"], how="left", validate="many_to_one"
    )
    qa["Size"] = pd.to_numeric(qa["Size"], errors="raise")
    qa = qa[truthy(qa["IsValid"]) & qa["Size"].eq(qa["K_exact"])]
    return qa, dc


def inclusion_probabilities() -> pd.DataFrame:
    qa, dc = certified_rows()
    rows = []
    for solver, frame in (("QA-Base", qa), ("DC-OSQA", dc)):
        for (nodes, instance), group in frame.groupby(["Nodes", "Instance"]):
            nodes = int(nodes)
            total = int(group["Occurrences"].sum())
            if total <= 0:
                continue
            included = np.zeros(nodes, dtype=float)
            for bitstring, occurrences in zip(group["Bitstring"], group["Occurrences"]):
                bits = str(bitstring).strip()
                if len(bits) != nodes or set(bits) - {"0", "1"}:
                    raise RuntimeError(f"Malformed bitstring for N={nodes}, instance={instance}")
                included += np.fromiter((char == "1" for char in bits), dtype=float, count=nodes) * int(occurrences)
            for vertex, probability in enumerate(included / total):
                rows.append({
                    "Nodes": nodes,
                    "Instance": int(instance),
                    "solver": solver,
                    "vertex": vertex,
                    "inclusion_probability": float(probability),
                    "certified_occurrences": total,
                })
    return pd.DataFrame(rows)


def partial_rank_correlation(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    ranked_x = rankdata(x)
    ranked_y = rankdata(y)
    ranked_control = rankdata(control)
    design = np.column_stack([np.ones(len(x)), ranked_control])
    residual_x = ranked_x - design @ np.linalg.lstsq(design, ranked_x, rcond=None)[0]
    residual_y = ranked_y - design @ np.linalg.lstsq(design, ranked_y, rcond=None)[0]
    return float(np.corrcoef(residual_x, residual_y)[0, 1])


def inclusion_shift_correlations(mapping: pd.DataFrame) -> pd.DataFrame:
    inclusion = inclusion_probabilities()
    pivot = inclusion.pivot(
        index=["Nodes", "Instance", "vertex"], columns="solver", values="inclusion_probability"
    ).dropna(subset=["QA-Base", "DC-OSQA"]).reset_index()
    pivot["inclusion_shift"] = pivot["DC-OSQA"] - pivot["QA-Base"]
    joined = pivot.merge(mapping, on=["Nodes", "Instance", "vertex"], how="inner", validate="one_to_one")
    rows = []
    for (nodes, instance), group in joined.groupby(["Nodes", "Instance"]):
        degree = group["MappingDegree"].to_numpy(float)
        chain = group["ChainLength"].to_numpy(float)
        shift = group["inclusion_shift"].to_numpy(float)
        rows.append({
            "Nodes": int(nodes),
            "Instance": int(instance),
            "vertices": len(group),
            "rho_degree_inclusion_shift": float(spearmanr(degree, shift).statistic),
            "rho_chain_inclusion_shift": float(spearmanr(chain, shift).statistic),
            "partial_rho_degree_given_chain": partial_rank_correlation(degree, shift, chain),
            "partial_rho_chain_given_degree": partial_rank_correlation(chain, shift, degree),
        })
    result = pd.DataFrame(rows).sort_values(["Nodes", "Instance"])
    if len(result) != 118 or not result["vertices"].eq(result["Nodes"]).all():
        raise RuntimeError(f"Expected 118 complete canonical protocol pairs, found {len(result)}")
    return result


def summarize(degree_chain: pd.DataFrame, shift: pd.DataFrame) -> pd.DataFrame:
    records = []
    for frame, column in (
        (degree_chain, "rho_degree_chain_length"),
        (shift, "rho_degree_inclusion_shift"),
        (shift, "rho_chain_inclusion_shift"),
        (shift, "partial_rho_degree_given_chain"),
        (shift, "partial_rho_chain_given_degree"),
    ):
        values = frame[column].to_numpy(float)
        records.append({
            "metric": column,
            "n_graph_instances": len(values),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "positive_instances": int((values > 0).sum()),
            "negative_instances": int((values < 0).sum()),
            "zero_instances": int((values == 0).sum()),
        })
    return pd.DataFrame(records)


def plot(degree_chain: pd.DataFrame, shift: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))
    values = np.sort(degree_chain["rho_degree_chain_length"].to_numpy(float))
    axes[0].scatter(np.arange(1, len(values) + 1), values, s=18, color="#2F6F8F", alpha=0.75)
    axes[0].axhline(values.mean(), color="#D55E00", linewidth=1.4, label=f"Mean {values.mean():.2f}")
    axes[0].set_xlabel("Canonical graph instance (sorted)")
    axes[0].set_ylabel("Degree-chain-length\nSpearman correlation, ρ")
    axes[0].legend(frameon=False, loc="lower right")

    columns = [
        "rho_degree_inclusion_shift",
        "rho_chain_inclusion_shift",
        "partial_rho_degree_given_chain",
        "partial_rho_chain_given_degree",
    ]
    labels = [
        "Degree\n(unadjusted)",
        "Chain length\n(unadjusted)",
        "Degree\n(adjusted for\nchain length)",
        "Chain length\n(adjusted for\ndegree)",
    ]
    data = [shift[column].to_numpy(float) for column in columns]
    boxes = axes[1].boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True, widths=0.58)
    colors = ["#2F6F8F", "#8B8B8B", "#2F6F8F", "#8B8B8B"]
    for box, color in zip(boxes["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.28)
        box.set_edgecolor(color)
    rng = np.random.default_rng(20260715)
    for position, (values_i, color) in enumerate(zip(data, colors), start=1):
        axes[1].scatter(position + rng.uniform(-0.12, 0.12, len(values_i)), values_i, s=10, color=color, alpha=0.34)
    axes[1].axhline(0, color="#555555", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Rank correlation\nwith inclusion shift")
    axes[1].tick_params(axis="x", labelsize=8.2)

    for label, ax in zip(["(a)", "(b)"], axes):
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.14, 1.04, label, transform=ax.transAxes, fontsize=12, fontweight="bold")
    fig.tight_layout(w_pad=2.2)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    CURATED.mkdir(parents=True, exist_ok=True)
    mapping = load_mapping()
    degree_chain = degree_chain_correlations(mapping)
    shift = inclusion_shift_correlations(mapping)
    summary = summarize(degree_chain, shift)
    degree_chain.to_csv(DERIVED / "degree_chain_correlation_by_instance.csv", index=False)
    shift.to_csv(DERIVED / "vertex_chain_inclusion_shift_correlations.csv", index=False)
    summary.to_csv(DERIVED / "vertex_chain_confound_summary.csv", index=False)
    plot(degree_chain, shift)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
