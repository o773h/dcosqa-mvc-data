#!/usr/bin/env python3
"""Build the verified candidate stream for the random 3-regular control.

The archived experiment runner generated instance ``i`` with
``networkx.random_regular_graph(3, 150, seed=350000 + i)``.  The released
edge reference was reconstructed from that rule and validated against every
recorded raw candidate.  This script repeats the graph, feasibility, energy,
and exact-cardinality checks before creating the analysis input.
"""
from __future__ import annotations

import gzip
import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "raw" / "raw_solutions.csv"
EDGE_REFERENCE = BASE / "reference" / "regular3_graph_edges.csv"
GRAPH_REGISTRY = BASE / "reference" / "regular3_graph_registry.csv"
OUTPUT = BASE / "derived" / "verified_candidate_stream.csv.gz"

N = 150
REGULAR_DEGREE = 3
EXPECTED_INSTANCES = set(range(20))
EXPECTED_SOLVERS = {"QA-Base", "DC-OSQA"}
EXPECTED_RAW_ROWS = 34_848
EXPECTED_TOTAL_OCCURRENCES = 40_000


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


def canonical_graph_hash(edges: list[tuple[int, int]]) -> str:
    normalized = sorted((min(u, v), max(u, v)) for u, v in edges)
    payload = f"N={N}\n" + "".join(f"{u},{v}\n" for u, v in normalized)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def load_graphs() -> tuple[dict[int, list[tuple[int, int]]], pd.DataFrame]:
    registry = pd.read_csv(GRAPH_REGISTRY)
    required_registry = {
        "Instance",
        "N",
        "Graph_Seed",
        "Regular_Degree",
        "Edge_Count",
        "graph_hash_sha256",
        "provenance_status",
    }
    missing = required_registry - set(registry.columns)
    if missing:
        raise RuntimeError(f"graph registry missing columns: {sorted(missing)}")

    for column in ["Instance", "N", "Graph_Seed", "Regular_Degree", "Edge_Count"]:
        registry[column] = pd.to_numeric(registry[column], errors="raise").astype(int)
    if len(registry) != 20 or set(registry["Instance"]) != EXPECTED_INSTANCES:
        raise RuntimeError("graph registry must contain exactly instances 0--19")
    if registry["Instance"].duplicated().any():
        raise RuntimeError("duplicate instance in graph registry")
    if not registry["N"].eq(N).all():
        raise RuntimeError("regular-3 registry contains N != 150")
    if not registry["Regular_Degree"].eq(REGULAR_DEGREE).all():
        raise RuntimeError("regular-3 registry contains degree != 3")
    if not registry["Edge_Count"].eq(N * REGULAR_DEGREE // 2).all():
        raise RuntimeError("regular-3 registry contains an unexpected edge count")
    expected_seeds = registry["Instance"].map(lambda i: 350_000 + int(i))
    if not registry["Graph_Seed"].eq(expected_seeds).all():
        raise RuntimeError("graph seeds do not follow archived-runner assignment")

    edge_table = pd.read_csv(EDGE_REFERENCE)
    required_edges = {"Instance", "u", "v"}
    missing = required_edges - set(edge_table.columns)
    if missing:
        raise RuntimeError(f"edge reference missing columns: {sorted(missing)}")
    for column in ["Instance", "u", "v"]:
        edge_table[column] = pd.to_numeric(edge_table[column], errors="raise").astype(int)
    if edge_table.duplicated(["Instance", "u", "v"]).any():
        raise RuntimeError("duplicate edge in regular-3 edge reference")

    graphs: dict[int, list[tuple[int, int]]] = {}
    for record in registry.sort_values("Instance").itertuples(index=False):
        instance = int(record.Instance)
        group = edge_table.loc[edge_table["Instance"].eq(instance), ["u", "v"]]
        edges = sorted(
            (min(int(row.u), int(row.v)), max(int(row.u), int(row.v)))
            for row in group.itertuples(index=False)
        )
        if len(edges) != int(record.Edge_Count):
            raise RuntimeError(f"instance {instance}: edge-count mismatch")
        if any(u == v or u < 0 or v >= N for u, v in edges):
            raise RuntimeError(f"instance {instance}: malformed edge reference")
        degree = np.zeros(N, dtype=int)
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        if not np.all(degree == REGULAR_DEGREE):
            raise RuntimeError(f"instance {instance}: graph is not 3-regular")
        graph_hash = canonical_graph_hash(edges)
        if graph_hash != str(record.graph_hash_sha256):
            raise RuntimeError(f"instance {instance}: graph hash mismatch")
        graphs[instance] = edges

    if set(edge_table["Instance"]) != EXPECTED_INSTANCES:
        raise RuntimeError("edge reference does not cover exactly instances 0--19")
    return graphs, registry.sort_values("Instance").reset_index(drop=True)


def solve_exact_mvc(edges: list[tuple[int, int]]) -> int:
    matrix = lil_matrix((len(edges), N), dtype=float)
    for row, (u, v) in enumerate(edges):
        matrix[row, u] = 1.0
        matrix[row, v] = 1.0
    result = milp(
        c=np.ones(N, dtype=float),
        integrality=np.ones(N, dtype=int),
        bounds=Bounds(np.zeros(N), np.ones(N)),
        constraints=LinearConstraint(
            matrix.tocsr(), np.ones(len(edges)), np.full(len(edges), np.inf)
        ),
    )
    if not result.success or result.fun is None:
        raise RuntimeError(f"HiGHS failed to certify MVC optimum: {result.message}")
    optimum = int(round(float(result.fun)))
    if not np.isclose(result.fun, optimum):
        raise RuntimeError("non-integral MVC objective returned by HiGHS")
    return optimum


def deterministic_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_output = path.open("wb")
    gzip_output = gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0)
    text_output = io.TextIOWrapper(gzip_output, encoding="utf-8", newline="")
    frame.to_csv(text_output, index=False)
    text_output.close()
    raw_output.close()


def main() -> None:
    for path in [RAW, EDGE_REFERENCE, GRAPH_REGISTRY]:
        if not path.is_file():
            raise FileNotFoundError(path)

    graphs, registry = load_graphs()
    raw = pd.read_csv(RAW, dtype={"Bitstring": "string"}, low_memory=False)
    required_raw = {
        "Nodes",
        "Instance",
        "Solver",
        "Bitstring",
        "Energy",
        "Size",
        "IsValid",
        "Occurrences",
        "Chain_Break_Frac",
        "EmbeddingRep",
        "EmbeddingSeed",
        "Solver_Canonical",
        "Embedding_Source",
        "Embedding_Status",
    }
    missing = required_raw - set(raw.columns)
    if missing:
        raise RuntimeError(f"raw regular-3 table missing columns: {sorted(missing)}")
    if len(raw) != EXPECTED_RAW_ROWS:
        raise RuntimeError(f"expected {EXPECTED_RAW_ROWS} raw rows, found {len(raw)}")

    for column in ["Nodes", "Instance", "Size", "Occurrences", "EmbeddingRep", "EmbeddingSeed"]:
        raw[column] = pd.to_numeric(raw[column], errors="raise").astype(int)
    raw["Energy"] = pd.to_numeric(raw["Energy"], errors="raise")
    if not raw["Nodes"].eq(N).all():
        raise RuntimeError("raw regular-3 table contains Nodes != 150")
    if set(raw["Instance"]) != EXPECTED_INSTANCES:
        raise RuntimeError("raw regular-3 table does not cover instances 0--19")
    if raw["Occurrences"].le(0).any():
        raise RuntimeError("raw regular-3 table contains non-positive Occurrences")
    if int(raw["Occurrences"].sum()) != EXPECTED_TOTAL_OCCURRENCES:
        raise RuntimeError("regular-3 occurrence total is not 40,000")

    bitstrings = raw["Bitstring"].astype(str).str.strip()
    if not bitstrings.str.fullmatch(f"[01]{{{N}}}").all():
        raise RuntimeError("malformed regular-3 bitstring")
    raw["Bitstring"] = bitstrings
    observed_size = bitstrings.map(lambda value: value.count("1")).astype(int)
    if not observed_size.eq(raw["Size"]).all():
        raise RuntimeError("recorded Size differs from bitstring cardinality")

    raw["Solver_Canonical"] = raw["Solver_Canonical"].map(canonical_solver)
    if set(raw["Solver_Canonical"]) != EXPECTED_SOLVERS:
        raise RuntimeError("unexpected regular-3 solver label")
    per_condition = raw.groupby(["Instance", "Solver_Canonical"])["Occurrences"].sum()
    if len(per_condition) != 40 or not per_condition.eq(1000).all():
        raise RuntimeError("expected 1,000 occurrences for every instance--solver pair")

    k_exact = {instance: solve_exact_mvc(edges) for instance, edges in graphs.items()}
    registry_by_instance = registry.set_index("Instance")
    recorded_validity = truthy(raw["IsValid"])
    recomputed_validity = np.zeros(len(raw), dtype=bool)
    recomputed_energy = np.zeros(len(raw), dtype=float)

    for instance, indices in raw.groupby("Instance").groups.items():
        edges = graphs[int(instance)]
        for index in indices:
            values = np.fromiter(
                (character == "1" for character in raw.at[index, "Bitstring"]),
                dtype=bool,
                count=N,
            )
            recomputed_validity[index] = all(values[u] or values[v] for u, v in edges)
            recomputed_energy[index] = int(values.sum()) - 2 * sum(
                int(values[u]) + int(values[v]) - int(values[u] and values[v])
                for u, v in edges
            )

    if not np.array_equal(recomputed_validity, recorded_validity.to_numpy(bool)):
        mismatch = raw.loc[
            recomputed_validity != recorded_validity.to_numpy(bool),
            ["Instance", "Solver_Canonical", "Bitstring", "IsValid"],
        ].head()
        raise RuntimeError("recomputed validity mismatch\n" + mismatch.to_string(index=False))
    if not np.allclose(recomputed_energy, raw["Energy"].to_numpy(float), atol=1e-9):
        raise RuntimeError("recomputed QUBO energy differs from recorded Energy")

    raw["IsValid"] = recomputed_validity
    raw["Graph_Seed"] = raw["Instance"].map(
        registry_by_instance["Graph_Seed"].astype(int)
    )
    raw["Edges"] = raw["Instance"].map(registry_by_instance["Edge_Count"].astype(int))
    raw["graph_hash_sha256"] = raw["Instance"].map(
        registry_by_instance["graph_hash_sha256"].astype(str)
    )
    raw["K_exact"] = raw["Instance"].map(k_exact).astype(int)
    raw["strict_exact_verified"] = raw["IsValid"] & raw["Size"].eq(raw["K_exact"])

    ordered = list(pd.read_csv(RAW, nrows=0).columns)
    ordered += [
        "Graph_Seed",
        "Edges",
        "graph_hash_sha256",
        "K_exact",
        "strict_exact_verified",
    ]
    output = raw[ordered].sort_values(
        ["Instance", "Solver_Canonical", "Bitstring"], kind="mergesort"
    )
    deterministic_gzip_csv(output, OUTPUT)

    strict = output.loc[output["strict_exact_verified"]]
    exact_summary = strict.groupby("Solver_Canonical").agg(
        strict_rows=("Bitstring", "size"),
        strict_occurrences=("Occurrences", "sum"),
    )
    print("PASS regular-3 verified stream")
    print(f"rows={len(output)} occurrences={int(output['Occurrences'].sum())}")
    print("K_exact=" + ",".join(str(k_exact[i]) for i in range(20)))
    print(exact_summary.to_string())
    print(f"output={OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
