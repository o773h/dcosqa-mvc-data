#!/usr/bin/env python3
"""Validate the released canonical BA m=2 logical-to-physical mappings."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import networkx as nx
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MAPPING = ROOT / "data/metadata/canonical_embeddings/ba_m2_vertex_chain_mappings.csv.gz"
INVENTORY = ROOT / "data/metadata/canonical_embeddings/ba_m2_embedding_inventory.csv"
RECORDED = ROOT / "data/raw/main_ba_m2_dcosqa_offset01/raw_candidate_bitstrings.csv"
REGENERATED = ROOT / "code/supp_embedding_seed_sensitivity/mappings/raw"
REGENERATED_PATTERN = re.compile(r"rep(?P<rep>\d+)_seed(?P<seed>\d+)_N150_inst(?P<instance>\d+)\.csv$")


def mapping_hash(nodes: int, instance: int, mapping: dict[int, list[int]]) -> str:
    lines = [f"N={nodes}", f"instance={instance}"]
    for vertex in sorted(mapping):
        chain = ",".join(str(qubit) for qubit in sorted(mapping[vertex]))
        lines.append(f"{vertex}:{chain}")
    return hashlib.sha256(("\n".join(lines) + "\n").encode("ascii")).hexdigest()


def main() -> None:
    frame = pd.read_csv(MAPPING)
    inventory = pd.read_csv(INVENTORY)
    forbidden = {"DC_Centrality", "BC_Centrality"} & set(frame.columns)
    if forbidden:
        raise RuntimeError(f"Redundant centrality columns present: {sorted(forbidden)}")

    expected = {(n, i) for n in (20, 30, 40, 50, 60) for i in range(10)}
    expected |= {(n, i) for n in (70, 80, 90, 100, 120, 150) for i in range(20)}
    observed = set(zip(frame["Nodes"].astype(int), frame["Instance"].astype(int)))
    if observed != expected or len(inventory) != len(expected) or len(frame) != 14200:
        raise RuntimeError("Canonical embedding coverage is incomplete")

    recorded = pd.read_csv(
        RECORDED,
        usecols=["Nodes", "Instance", "ChainLen_Mean", "ChainLen_Max"],
        low_memory=False,
    ).groupby(["Nodes", "Instance"])[["ChainLen_Mean", "ChainLen_Max"]].first()
    inventory = inventory.set_index(["Nodes", "Instance"])

    for (nodes, instance), group in frame.groupby(["Nodes", "Instance"], sort=True):
        nodes, instance = int(nodes), int(instance)
        if len(group) != nodes or group["LogicalNode"].nunique() != nodes:
            raise RuntimeError(f"Logical coverage mismatch for N={nodes}, instance={instance}")
        seed = 42 + 1000 * nodes + instance
        if not group["Graph_Seed"].eq(seed).all() or not group["Embedding_Seed"].eq(seed).all():
            raise RuntimeError(f"Seed mismatch for N={nodes}, instance={instance}")

        degrees = dict(nx.barabasi_albert_graph(nodes, 2, seed=seed).degree())
        mapping: dict[int, list[int]] = {}
        physical_qubits: list[int] = []
        for row in group.itertuples(index=False):
            vertex = int(row.LogicalNode)
            chain = [int(qubit) for qubit in json.loads(row.PhysicalChain)]
            if len(chain) != int(row.ChainLength) or int(row.Degree) != degrees[vertex]:
                raise RuntimeError(f"Mapping content mismatch for N={nodes}, instance={instance}, vertex={vertex}")
            mapping[vertex] = chain
            physical_qubits.extend(chain)
        if len(physical_qubits) != len(set(physical_qubits)):
            raise RuntimeError(f"Physical qubit reused for N={nodes}, instance={instance}")

        lengths = pd.Series([len(chain) for chain in mapping.values()], dtype=float)
        inv = inventory.loc[(nodes, instance)]
        if mapping_hash(nodes, instance, mapping) != inv["Mapping_SHA256"]:
            raise RuntimeError(f"Mapping hash mismatch for N={nodes}, instance={instance}")
        if not (
            int(inv["Logical_Variables"]) == nodes
            and int(inv["Physical_Qubits"]) == len(physical_qubits)
            and abs(float(inv["Mean_Chain_Length"]) - lengths.mean()) < 1e-12
            and int(inv["Min_Chain_Length"]) == int(lengths.min())
            and int(inv["Max_Chain_Length"]) == int(lengths.max())
        ):
            raise RuntimeError(f"Inventory mismatch for N={nodes}, instance={instance}")
        archived = recorded.loc[(nodes, instance)]
        if abs(float(archived["ChainLen_Mean"]) - lengths.mean()) >= 1e-12 or int(archived["ChainLen_Max"]) != int(lengths.max()):
            raise RuntimeError(f"Recorded chain diagnostic mismatch for N={nodes}, instance={instance}")

    regenerated_files = sorted(REGENERATED.glob("*.csv"))
    if len(regenerated_files) != 100:
        raise RuntimeError(f"Expected 100 regenerated mappings, found {len(regenerated_files)}")
    for path in regenerated_files:
        match = REGENERATED_PATTERN.search(path.name)
        if not match:
            raise RuntimeError(f"Unrecognized regenerated mapping filename: {path.name}")
        instance, rep, seed = (int(match.group(name)) for name in ("instance", "rep", "seed"))
        if seed != 9150000 + 100 * instance + rep:
            raise RuntimeError(f"Embedding-seed mismatch in {path.name}")
        regenerated = pd.read_csv(path)
        if {"DC_Centrality", "BC_Centrality"} & set(regenerated.columns):
            raise RuntimeError(f"Redundant centrality column in {path.name}")
        if len(regenerated) != 150 or regenerated["LogicalNode"].nunique() != 150:
            raise RuntimeError(f"Logical coverage mismatch in {path.name}")
        degrees = dict(nx.barabasi_albert_graph(150, 2, seed=150042 + instance).degree())
        physical_qubits: list[int] = []
        for row in regenerated.itertuples(index=False):
            chain = [int(qubit) for qubit in json.loads(row.PhysicalChain)]
            if len(chain) != int(row.ChainLength) or int(row.Degree) != degrees[int(row.LogicalNode)]:
                raise RuntimeError(f"Mapping content mismatch in {path.name}")
            physical_qubits.extend(chain)
        if len(physical_qubits) != len(set(physical_qubits)):
            raise RuntimeError(f"Physical qubit reused in {path.name}")

    print(
        f"Validated {len(expected)} canonical embeddings ({len(frame)} vertex-chain rows) "
        f"and {len(regenerated_files)} regenerated embeddings."
    )


if __name__ == "__main__":
    main()
