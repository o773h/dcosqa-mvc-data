#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dcosqa_idealized_simulation.py
==============================

Idealized closed-system (noiseless, 0 K, unitary) quantum-annealing simulation
supporting the manuscript

    "Degree-conditioned anneal offsets reshape certified-optimum sampling
     for minimum vertex cover" (DC-OSQA).

WHAT THIS IS
------------
A *proof-of-principle* numerical experiment on small SURROGATE Barabasi-Albert
(m=2) MVC instances. It asks a single question:

    In a purely coherent, idealized model, can a degree-conditioned *local anneal
    schedule* alone redistribute the final Born probability inside the exact
    minimum-vertex-cover (MVC) ground-manifold, and does DEGREE ALIGNMENT
    contribute beyond the offset multiset, a mean-matched uniform schedule perturbation,
    or an unstructured random assignment?

PRIMARY CLAIM
-------------
Within the explicitly defined closed-system model, degree-aligned local schedules
are sufficient to redistribute exact MVC ground-manifold occupation, broaden its
conditional effective support, and increase expected distinct-optimum discovery
over a finite range of independent-read budgets relative to the zero-offset
baseline. Degree alignment is assessed separately against mean-matched uniform and
shuffled assignment controls.

The claim is model-specific and concerns sufficiency, not the microscopic dynamics
of the open-system QPU. Alpha is a dimensionless control strength of this model.

WHAT THIS IS NOT
----------------
This is NOT a device model. It deliberately omits:
  thermal excitation / relaxation, decoherence, bath coupling, analog control
  error, calibration drift, readout error, minor-embedding, physical chains /
  chain breaks, and per-physical-qubit inhomogeneity.
It is not designed to reproduce the hardware effect magnitude and makes NO claim
of dynamical equivalence with the N = 70-150 QPU experiments. Report it as
"selected small surrogate BA instances" and "in the spirit of prior
inhomogeneous-driving studies".

METHOD (per instance)
---------------------
1. Build BA(N, m=2) with the manuscript seed convention  seed = 42 + 1000*N + i.
2. Enumerate all 2^N bitstrings; compute the exact minimum-cover manifold M.
3. Convert the MVC QUBO (lambda_card=1, lambda_edge=2) to an Ising Hamiltonian,
   applying the schedule
   in ISING space (this ordering matters under heterogeneous per-qubit schedules):
       h_i = A/2 - B*d_i/4 ,   J_ij = B/4 .
4. Start from |+>^{⊗N}.
5. Per-qubit local schedule via a monotone, endpoint-pinned offset (alpha <= 0.2):
       delta_i = -alpha * d_i / d_max ,   s_i(s) = clip(s + 4*delta_i*s*(1-s), 0, 1).
6. Propagate the full state vector using a 2nd-order (Strang) split-operator
   approximation. The driver terms sigma^x_i mutually commute, so the driver
   exponential within each split step is exact; the driver/problem splitting and
   midpoint treatment of the time dependence are 2nd order in dt (convergence is
   checked).
7. Analyse P(z) = |<z|psi(T)>|^2 restricted to M: yield, normalized entropy,
   Simpson effective support, and unconditional finite-budget discovery.
8. Test the result across a fixed alpha grid and record time-resolved
   ground-manifold yield and conditional effective support during the evolution.

OUTPUTS
-------
  <outdir>/results.json          machine-readable results (incl. bootstrap CIs, dEUB curves)
  <outdir>/paired_stats.csv      per-instance paired DC-Base / DC-Shuffle / dEUB table
  <outdir>/alpha_sensitivity_summary.csv
  <outdir>/time_resolved_summary.csv
  <outdir>/fig_size_robustness.png
  <outdir>/fig_eub.png           unconditional discovery gain  Delta E[U_R]  vs draw budget
  <outdir>/fig_controls.png
  <outdir>/fig_schedule.png
  <outdir>/fig_alpha_sensitivity.png
  <outdir>/fig_time_resolved.png
  <outdir>/fig_mechanism_sensitivity.{png,pdf}  alpha + dynamics manuscript figure
  <outdir>/fig_supplement_simulation.{png,pdf}  compact manuscript figure

CONFIRMATORY CONFIGURATION
  python dcosqa_idealized_simulation.py all --alpha 0.10
The `size` stage reports paired 95% bootstrap CIs for the DC-Base and DC-Shuffle
effective-support contrasts. For UNCONDITIONAL finite-budget discovery it reports
(i) the mean gain over a fixed approximately log-spaced budget grid and
(ii) a two-sided max-absolute sign-flip permutation test that accounts for the
budget-grid search. Peak gain and its budget are descriptive only.

USAGE
-----
  python dcosqa_idealized_simulation.py all            # full pipeline (N=12,14,16,18,20)
  python dcosqa_idealized_simulation.py size --sizes 18 20 --workers 20 \
      --outdir idealized_output                         # resumable heavy extension
  python dcosqa_idealized_simulation.py verify         # sanity + convergence checks
  python dcosqa_idealized_simulation.py size           # size-robustness only
  python dcosqa_idealized_simulation.py controls       # 5-scheme control panel
  python dcosqa_idealized_simulation.py schedule       # schedule-shape robustness
  python dcosqa_idealized_simulation.py alpha          # alpha sensitivity at smallest N
  python dcosqa_idealized_simulation.py dynamics       # time-resolved redistribution
  python dcosqa_idealized_simulation.py extended --sizes 14 16 18 20 --workers 20
                                                      # every missing large-N diagnostic
  python dcosqa_idealized_simulation.py plot           # regenerate figures from results.json

Config is at the top of the file (CONFIG). Everything is deterministic given
CONFIG.rng_seed.

Dependencies: numpy, networkx, matplotlib.
Runtime scales ~ O(N * 2^N * n_steps) per annealing condition. N=18--20 use
instance-level multiprocessing and resumable records under <outdir>/instance_cache/;
N>=22 is impractical for this full 20-shuffle exact-statevector panel.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

import numpy as np
import networkx as nx

# Imported lazily by plot_all(). Spawned numerical workers therefore do not each
# import matplotlib or build a font cache.
plt = None

# ======================================================================================
# Configuration
# ======================================================================================
@dataclass
class Config:
    # --- problem / physics ---
    m_ba: int = 2                     # Barabasi-Albert attachment parameter (paper: m=2)
    A_qubo: float = 1.0               # QUBO cardinality weight  (paper A = 1)
    B_qubo: float = 2.0               # QUBO penalty weight      (paper B = 2)
    T: float = 40.0                   # dimensionless total anneal time (NOT the 20 us QPU time)
    n_steps: int = 600                # Strang steps (convergence-checked; see `verify`)
    alpha_main: float = 0.10          # fixed operating point of the idealized model
    alpha_grid: tuple = (0.05, 0.10, 0.15, 0.20)

    # --- instance selection (manuscript seed convention: seed = 42 + 1000*N + i) ---
    sizes: tuple = (12, 14, 16, 18, 20)  # exact-statevector robustness range
    seed_pool: int = 60               # scan i = 0 .. seed_pool-1 from the fixed pool
    min_degeneracy: int = 4           # include instances with |M| >= this
    max_instances: int | None = None  # None => include ALL eligible in the pool (recommended)

    # --- local parallel execution ---
    workers: int = 1                  # increase on a sufficiently provisioned system
    parallel_min_N: int = 18          # preserve the validated serial RNG sequence for N<=16
    resume: bool = True               # reuse completed per-instance size-stage records

    # --- controls / statistics ---
    shuffle_reps: int = 20            # random-assignment repetitions (matches hardware control)
    schedules: tuple = ("linear", "convex_power", "trig")
    rng_seed: int = 0                 # master RNG seed (full determinism)
    n_boot: int = 5000                # bootstrap resamples for 95% CIs
    n_perm: int = 5000                # sign-flip permutations for the max-statistic p-value
    trajectory_grid: tuple = tuple(np.linspace(0.05, 1.0, 20))
    # unconditional discovery metric E[U_R] (independent-draw approximation), log budget grid
    budget_grid: tuple = tuple(int(x) for x in np.unique(np.round(np.logspace(0, 4, 33)).astype(int)))

    # --- io ---
    outdir: str = "sim_out"


CONFIG = Config()


# ======================================================================================
# Core physics
# ======================================================================================
def paper_seed(N: int, i: int) -> int:
    """Manuscript deterministic seed convention."""
    return 42 + 1000 * N + i


def build_instance(N: int, i: int, cfg: Config):
    """Return (G, degrees) for the i-th instance at size N using the paper seed."""
    G = nx.barabasi_albert_graph(N, cfg.m_ba, seed=paper_seed(N, i))
    deg = np.array([d for _, d in sorted(G.degree())], dtype=float)
    return G, deg


def enumerate_manifold(N: int, G: nx.Graph):
    """
    Enumerate all 2^N assignments and return (bits, K_exact, M_mask).
    bits[z, i] in {0,1}; x_i = 1 means vertex i is IN the cover.
    M_mask selects feasible covers of minimum size (the exact MVC ground-manifold).
    """
    idx = np.arange(2 ** N, dtype=np.int64)
    bits = ((idx[:, None] >> np.arange(N)[None, :]) & 1).astype(np.int8)
    size = bits.sum(1)
    feasible = np.ones(2 ** N, dtype=bool)
    for u, v in G.edges():
        feasible &= (bits[:, u] | bits[:, v]).astype(bool)   # every edge covered
    K = int(size[feasible].min())
    M = feasible & (size == K)
    return bits, K, M


def ising_params(deg: np.ndarray, cfg: Config):
    """
    MVC QUBO -> Ising.  With x_i = (1+z_i)/2 the collected fields are
        h_i = A/2 - B*d_i/4 ,   J_ij = B/4  (uniform over edges).
    """
    A, B = cfg.A_qubo, cfg.B_qubo
    h = A / 2.0 - B * deg / 4.0
    J = B / 4.0
    return h, J


def schedule(name: str, s: np.ndarray):
    """
    Annealing envelope (A: transverse driver, B: problem), monotone in s,
    with common endpoints A(0)=B(1)=1 and A(1)=B(0)=0.
    `s` may be a per-qubit array s_i.
    """
    if name == "linear":
        return 1.0 - s, s
    if name == "convex_power":                    # idealized steeper problem ramp
        return (1.0 - s) ** 1.6, s ** 1.8
    if name == "trig":
        return np.cos(np.pi * s / 2.0), np.sin(np.pi * s / 2.0)
    raise ValueError(f"unknown schedule '{name}'")


def local_s(s: float, delta: np.ndarray) -> np.ndarray:
    """
    Endpoint-pinned, monotone (for |alpha| <= 0.25) per-qubit anneal fraction:
        s_i(s) = clip(s + 4*delta_i*s*(1-s), 0, 1).
    delta_i < 0 delays qubit i (keeps it in the driver-dominated regime longer).
    """
    return np.clip(s + 4.0 * delta * s * (1.0 - s), 0.0, 1.0)


def build_problem_features(bits, edges):
    """Return fixed int8 z_i and z_i z_j rows for one graph instance."""
    z = (2 * bits.T - 1).astype(np.int8, copy=False)
    edge_rows = np.stack([z[u] * z[v] for u, v in edges], axis=0)
    return np.concatenate([z, edge_rows], axis=0)


def ising_diagonal(bits, edges, h, J, Bvec, problem_features=None):
    """Diagonal problem energy without materializing a dense float spin matrix.

    At N=20, converting the (2^N, N) int8 bit table to float64 at every split
    step would allocate about 160 MiB. Column-wise accumulation keeps the peak
    temporary storage O(2^N) while producing the same float64 diagonal.
    """
    if problem_features is not None:
        coefficients = np.concatenate([
            Bvec * h,
            np.fromiter((np.sqrt(Bvec[u] * Bvec[v]) * J for u, v in edges),
                        dtype=np.float64, count=len(edges)),
        ])
        # einsum avoids the spurious Accelerate matmul warnings observed for
        # mixed int8/float64 operands and was >3x faster at N=20 in validation.
        return np.einsum("t,tx->x", coefficients, problem_features,
                         dtype=np.float64, optimize=True)

    diag = np.zeros(bits.shape[0], dtype=np.float64)
    for i, coefficient in enumerate(Bvec * h):
        diag += coefficient * (2.0 * bits[:, i] - 1.0)
    for u, v in edges:
        zu = 2.0 * bits[:, u] - 1.0
        zv = 2.0 * bits[:, v] - 1.0
        diag += np.sqrt(Bvec[u] * Bvec[v]) * J * (zu * zv)
    return diag


def _apply_driver_rotation(psi: np.ndarray, qubit: int, c: float, isn: complex):
    """Apply one sigma-x rotation using strided block views.

    This replaces O(N 2^N) cached int64 index arrays with two O(2^N) temporary
    amplitude blocks, which is substantially smaller for parallel N=18--20 runs.
    """
    block = 1 << qubit
    view = psi.reshape(-1, 2, block)
    pu = view[:, 0, :].copy()
    pv = view[:, 1, :].copy()
    view[:, 0, :] = c * pu + isn * pv
    view[:, 1, :] = isn * pu + c * pv


def evolve(N, bits, edges, h, J, delta, sched_name, cfg: Config,
           trajectory_s=None, problem_features=None):
    """
    Full state-vector propagation from |+>^N under the time-dependent Hamiltonian,
    using a 2nd-order (Strang) split-operator step:
        exp(-i dt/2 H_prob) exp(-i dt H_driver) exp(-i dt/2 H_prob).
    H_driver = -sum_i A_i sigma^x_i (commuting -> exact rotation), evaluated at s_i.
    Returns P(z) = |amplitude|^2 for every computational basis state. If
    `trajectory_s` is supplied, also returns probabilities at the closest completed
    split steps to those fixed anneal fractions.
    """
    dim = 2 ** N
    dt = cfg.T / cfg.n_steps
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    snapshot_steps = {}
    snapshots = {}
    if trajectory_s is not None:
        for sval in trajectory_s:
            step = int(np.clip(np.rint(float(sval) * cfg.n_steps), 1, cfg.n_steps))
            snapshot_steps.setdefault(step, []).append(float(sval))
    for k in range(cfg.n_steps):
        s = (k + 0.5) / cfg.n_steps               # midpoint rule (2nd order in time)
        si = local_s(s, delta)
        Ai, Bi = schedule(sched_name, si)
        D = ising_diagonal(bits, edges, h, J, Bi, problem_features)
        half = np.exp(-1j * 0.5 * dt * D)
        psi *= half                               # half problem step
        for i in range(N):                        # exact commuting driver step
            th = dt * Ai[i]
            c = np.cos(th)
            isn = 1j * np.sin(th)
            _apply_driver_rotation(psi, i, c, isn)
        psi *= half                               # half problem step
        completed = k + 1
        if completed in snapshot_steps:
            prob = np.abs(psi) ** 2
            for sval in snapshot_steps[completed]:
                snapshots[sval] = prob.copy()
    final = np.abs(psi) ** 2
    if trajectory_s is None:
        return final
    return final, snapshots


# ======================================================================================
# Metrics
# ======================================================================================
def manifold_metrics(P, M, bits):
    """
    Restrict Born probabilities to the minimum-cover manifold and summarise:
      yield      : total probability mass in M (single-read certified probability)
      entropy    : normalized Shannon entropy of the conditional distribution over M
      seff       : Simpson effective number of optima = 1 / sum p_j^2
    """
    wM = P * M
    yld = wM.sum()
    p = wM / yld
    pe = p[M]
    entropy = -(pe * np.log(pe + 1e-18)).sum() / np.log(M.sum())
    seff = 1.0 / (pe ** 2).sum()
    return dict(yld=float(yld), entropy=float(entropy), seff=float(seff),
                w_manifold=P[M])   # w_manifold: UNCONDITIONAL per-read probabilities


def eub_curve(w_manifold, budget_grid):
    """
    Expected number of DISTINCT optima observed in B independent reads:
        E[U_R] = sum_{x in M} [1 - (1 - w(x))^R],
    where w(x) is the UNCONDITIONAL single-read probability (includes yield).
    This is the discovery metric aligned with the manuscript's unique-count outcome
    (independent-draw approximation, as in the main-text Discussion).
    """
    B = np.asarray(budget_grid, dtype=float)
    w = np.asarray(w_manifold, dtype=float)[:, None]
    return np.sum(1.0 - (1.0 - w) ** B[None, :], axis=0)


def bootstrap_ci(values, n_boot, rng, ci=95):
    """Percentile bootstrap CI for the mean of paired per-instance values."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    boots = np.array([values[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [(100 - ci) / 2.0, 100 - (100 - ci) / 2.0])
    return float(values.mean()), float(lo), float(hi)


def bootstrap_curve_ci(curves, n_boot, rng, ci=95):
    """Pointwise percentile-bootstrap band for a mean paired curve (descriptive)."""
    curves = np.asarray(curves, dtype=float)
    if curves.ndim != 2 or curves.shape[0] == 0:
        raise ValueError("curves must have shape (n_instances, n_grid_points)")
    n = curves.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = curves[idx].mean(axis=1)
    q = [(100 - ci) / 2.0, 100 - (100 - ci) / 2.0]
    lo, hi = np.percentile(boot_means, q, axis=0)
    return curves.mean(axis=0), lo, hi


# ======================================================================================
# Offset-assignment schemes (the controls)
# ======================================================================================
def assign_offsets(deg, scheme, alpha, rng):
    """
    Build the per-vertex offset vector delta for a given control scheme.
      dc          : degree-aligned         delta_i = -alpha * d_i / d_max      (the rule)
      shuffle     : same MULTISET, permuted across vertices (alignment removed)
      anti        : same multiset, reversed by degree (most-negative -> lowest degree)
      homogeneous : mean-matched uniform offset (heterogeneity removed)
      random      : independent Uniform[-alpha, 0] offsets
    """
    c = deg / deg.max()
    base = -alpha * c
    if scheme == "dc":
        return base
    if scheme == "shuffle":
        return rng.permutation(base)
    if scheme == "anti":
        vals = np.sort(base)                      # ascending (most negative first)
        out = np.empty_like(base)
        out[np.argsort(deg)] = vals               # lowest degree gets most-negative
        return out
    if scheme == "homogeneous":
        return -alpha * c.mean() * np.ones_like(c)
    if scheme == "random":
        return -alpha * rng.random(len(c))
    raise ValueError(f"unknown scheme '{scheme}'")


# ======================================================================================
# Instance pool (fixed seed pool; include ALL eligible; report searched vs included)
# ======================================================================================
def instance_pool(N: int, cfg: Config):
    """
    Deterministically scan i = 0..seed_pool-1 using seed = 42 + 1000*N + i and
    include every instance whose exact MVC degeneracy |M| >= min_degeneracy.
    Returns (instances, n_searched). Each instance is a dict with everything cached.
    """
    instances = []
    for i in range(cfg.seed_pool):
        G, deg = build_instance(N, i, cfg)
        bits, K, M = enumerate_manifold(N, G)
        if int(M.sum()) >= cfg.min_degeneracy:
            h, J = ising_params(deg, cfg)
            instances.append(dict(i=i, seed=paper_seed(N, i), N=N, deg=deg,
                                  bits=bits, edges=list(G.edges()), h=h, J=J,
                                  K=K, M=M, degen=int(M.sum())))
            if cfg.max_instances is not None and len(instances) >= cfg.max_instances:
                break
    return instances, cfg.seed_pool


def _stable_instance_rng_seed(cfg: Config, N: int, i: int, sched_name: str) -> int:
    """Deterministic RNG seed independent of worker count and completion order."""
    token = f"size-v3|{cfg.rng_seed}|{N}|{i}|{sched_name}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little")


def _size_cache_signature(cfg: Config, N: int, sched_name: str) -> str:
    """Short signature for every setting that changes a size-stage record."""
    payload = dict(
        schema="exact-size-v3",
        N=N,
        m_ba=cfg.m_ba,
        A_qubo=cfg.A_qubo,
        B_qubo=cfg.B_qubo,
        T=cfg.T,
        n_steps=cfg.n_steps,
        alpha_main=cfg.alpha_main,
        min_degeneracy=cfg.min_degeneracy,
        shuffle_reps=cfg.shuffle_reps,
        budget_grid=list(cfg.budget_grid),
        rng_seed=cfg.rng_seed,
        schedule=sched_name,
    )
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _size_cache_path(cfg: Config, N: int, i: int, sched_name: str) -> Path:
    signature = _size_cache_signature(cfg, N, sched_name)
    return (Path(cfg.outdir) / "instance_cache" /
            f"size_N{N}_{sched_name}_{signature}" / f"instance_{i:03d}.json")


def _compute_size_instance(task):
    """Worker entry point for one exact size-stage graph instance."""
    N, i, cfg_dict, sched_name = task
    cfg = Config(**cfg_dict)
    G, deg = build_instance(N, i, cfg)
    bits, K, M = enumerate_manifold(N, G)
    degen = int(M.sum())
    record = dict(
        schema="exact-size-v3",
        N=N,
        instance=i,
        seed=paper_seed(N, i),
        eligible=bool(degen >= cfg.min_degeneracy),
        degen=degen,
        K=K,
    )
    if not record["eligible"]:
        return record

    h, J = ising_params(deg, cfg)
    edges = list(G.edges())
    problem_features = build_problem_features(bits, edges)
    rng = np.random.default_rng(_stable_instance_rng_seed(cfg, N, i, sched_name))

    P0 = evolve(N, bits, edges, h, J, np.zeros(N), sched_name, cfg,
                problem_features=problem_features)
    base = manifold_metrics(P0, M, bits)
    Pdc = evolve(N, bits, edges, h, J,
                 assign_offsets(deg, "dc", cfg.alpha_main, rng), sched_name, cfg,
                 problem_features=problem_features)
    dc = manifold_metrics(Pdc, M, bits)

    shuffle_gains = []
    for _ in range(cfg.shuffle_reps):
        Psh = evolve(N, bits, edges, h, J,
                     assign_offsets(deg, "shuffle", cfg.alpha_main, rng), sched_name, cfg,
                     problem_features=problem_features)
        shuffle_gains.append(manifold_metrics(Psh, M, bits)["seff"] - base["seff"])

    eb = eub_curve(base["w_manifold"], cfg.budget_grid)
    ed = eub_curve(dc["w_manifold"], cfg.budget_grid)
    deub = ed - eb
    dc_gain = dc["seff"] - base["seff"]
    shuffle_gain = float(np.mean(shuffle_gains))
    relative_gain = dc_gain / base["seff"]

    record["row"] = dict(
        seed=record["seed"],
        instance=i,
        degen=degen,
        seff_base=base["seff"],
        seff_dc=dc["seff"],
        dc_gain=dc_gain,
        dc_relative_gain=relative_gain,
        dc_log_ratio=float(np.log(dc["seff"] / base["seff"])),
        shuffle_gain=shuffle_gain,
        dc_minus_shuffle=dc_gain - shuffle_gain,
        dc_minus_shuffle_relative=(dc_gain - shuffle_gain) / base["seff"],
        ent_base=base["entropy"],
        ent_dc=dc["entropy"],
        yld_base=base["yld"],
        yld_dc=dc["yld"],
        deub_curve=deub.tolist(),
        eub_budget_mean=float(deub.mean()),
        eub_budget_fraction_mean=float(deub.mean() / degen),
        eub_peak_gain=float(deub.max()),
        eub_peak_B=int(cfg.budget_grid[int(deub.argmax())]),
    )
    return record


def _write_json_atomic(path: Path, payload) -> None:
    """Write one completed record atomically so interrupted runs can resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, sort_keys=True)
    os.replace(temporary, path)


def _parallel_paired_size(N: int, cfg: Config, sched_name="linear"):
    """Compute/cache one record per graph and return eligible rows in seed order."""
    cfg_dict = asdict(cfg)
    records = {}
    missing = []
    for i in range(cfg.seed_pool):
        path = _size_cache_path(cfg, N, i, sched_name)
        if cfg.resume and path.is_file():
            try:
                with path.open() as handle:
                    record = json.load(handle)
                if record.get("schema") == "exact-size-v3":
                    records[i] = record
                    continue
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        missing.append(i)

    workers = max(1, min(int(cfg.workers), len(missing) or 1))
    dim = 1 << N
    estimated_gib = (N + 176) * dim / (1024 ** 3)
    print(f"    exact workers={workers}; estimated numerical peak about "
          f"{estimated_gib:.2f} GiB/worker ({estimated_gib * workers:.1f} GiB total)")
    if records:
        print(f"    resumed {len(records)}/{cfg.seed_pool} cached instance records")

    # Prevent each process from starting an additional multithreaded BLAS pool.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(variable, "1")

    tasks = [(N, i, cfg_dict, sched_name) for i in missing]
    if workers == 1:
        completed_records = map(_compute_size_instance, tasks)
        for record in completed_records:
            i = int(record["instance"])
            records[i] = record
            _write_json_atomic(_size_cache_path(cfg, N, i, sched_name), record)
            print(f"    completed N={N} instance {i:02d} "
                  f"(|M|={record['degen']}, eligible={record['eligible']})", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_compute_size_instance, task): task[1] for task in tasks}
            for future in as_completed(futures):
                i = futures[future]
                record = future.result()
                records[i] = record
                _write_json_atomic(_size_cache_path(cfg, N, i, sched_name), record)
                print(f"    completed N={N} instance {i:02d} "
                      f"(|M|={record['degen']}, eligible={record['eligible']})", flush=True)

    eligible = [records[i]["row"] for i in sorted(records) if records[i]["eligible"]]
    if cfg.max_instances is not None:
        eligible = eligible[:cfg.max_instances]
    return eligible, cfg.seed_pool


def _stable_extended_rng_seed(cfg: Config, N: int, i: int, label: str) -> int:
    """Deterministic RNG seed for one extended-control component."""
    token = f"exact-extended-v1|{cfg.rng_seed}|{N}|{i}|{label}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little")


def _extended_cache_signature(cfg: Config, N: int) -> str:
    """Short signature for settings that change an extended-control record."""
    payload = dict(
        schema="exact-extended-v1",
        N=N,
        m_ba=cfg.m_ba,
        A_qubo=cfg.A_qubo,
        B_qubo=cfg.B_qubo,
        T=cfg.T,
        n_steps=cfg.n_steps,
        alpha_main=cfg.alpha_main,
        alpha_grid=list(cfg.alpha_grid),
        min_degeneracy=cfg.min_degeneracy,
        shuffle_reps=cfg.shuffle_reps,
        schedules=list(cfg.schedules),
        trajectory_grid=list(cfg.trajectory_grid),
        budget_grid=list(cfg.budget_grid),
        rng_seed=cfg.rng_seed,
    )
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _extended_cache_path(cfg: Config, N: int, i: int) -> Path:
    signature = _extended_cache_signature(cfg, N)
    return (Path(cfg.outdir) / "instance_cache" /
            f"extended_N{N}_{signature}" / f"instance_{i:03d}.json")


def _compute_extended_instance(task):
    """Worker for controls, schedule, alpha, and dynamics on one eligible graph.

    The completed size-stage row supplies the already validated linear-schedule
    baseline/DC/shuffle endpoint contrasts. Baseline and DC are propagated once
    more with trajectory snapshots; their final metrics are checked against that
    row. All other propagations are new.
    """
    N, i, cfg_dict, size_row = task
    cfg = Config(**cfg_dict)
    G, deg = build_instance(N, i, cfg)
    bits, K, M = enumerate_manifold(N, G)
    degen = int(M.sum())
    if degen < cfg.min_degeneracy:
        raise RuntimeError(f"Extended task received ineligible N={N}, instance={i}")
    if degen != int(size_row["degen"]):
        raise RuntimeError(
            f"Degeneracy mismatch for N={N}, instance={i}: {degen} != {size_row['degen']}")

    h, J = ising_params(deg, cfg)
    edges = list(G.edges())
    features = build_problem_features(bits, edges)
    sgrid = np.asarray(cfg.trajectory_grid, dtype=float)
    zero = np.zeros(N)
    dc_offsets = -cfg.alpha_main * deg / deg.max()

    # One baseline/DC pair supplies both final metrics and time-resolved snapshots.
    P0, base_snap = evolve(N, bits, edges, h, J, zero, "linear", cfg,
                           trajectory_s=sgrid, problem_features=features)
    Pdc, dc_snap = evolve(N, bits, edges, h, J, dc_offsets, "linear", cfg,
                          trajectory_s=sgrid, problem_features=features)
    base = manifold_metrics(P0, M, bits)
    dc = manifold_metrics(Pdc, M, bits)
    if not np.isclose(base["seff"], size_row["seff_base"], rtol=0, atol=1e-9):
        raise RuntimeError(f"Baseline endpoint mismatch for N={N}, instance={i}")
    if not np.isclose(dc["seff"], size_row["seff_dc"], rtol=0, atol=1e-9):
        raise RuntimeError(f"DC endpoint mismatch for N={N}, instance={i}")

    controls = {
        "dc": float(dc["seff"] - base["seff"]),
        "shuffle": float(size_row["shuffle_gain"]),
    }
    for scheme in ("homogeneous", "anti"):
        P = evolve(N, bits, edges, h, J,
                   assign_offsets(deg, scheme, cfg.alpha_main,
                                  np.random.default_rng(0)),
                   "linear", cfg, problem_features=features)
        controls[scheme] = float(manifold_metrics(P, M, bits)["seff"] - base["seff"])
    rng_random = np.random.default_rng(
        _stable_extended_rng_seed(cfg, N, i, "controls-random"))
    random_gains = []
    for _ in range(cfg.shuffle_reps):
        P = evolve(N, bits, edges, h, J,
                   assign_offsets(deg, "random", cfg.alpha_main, rng_random),
                   "linear", cfg, problem_features=features)
        random_gains.append(manifold_metrics(P, M, bits)["seff"] - base["seff"])
    controls["random"] = float(np.mean(random_gains))

    schedule_rows = {
        "linear": {
            "dc_gain": float(dc["seff"] - base["seff"]),
            "shuffle_gain": float(size_row["shuffle_gain"]),
        }
    }
    for sched_name in cfg.schedules:
        if sched_name == "linear":
            continue
        Pb = evolve(N, bits, edges, h, J, zero, sched_name, cfg,
                    problem_features=features)
        mb = manifold_metrics(Pb, M, bits)
        Pd = evolve(N, bits, edges, h, J, dc_offsets, sched_name, cfg,
                    problem_features=features)
        md = manifold_metrics(Pd, M, bits)
        rng_shuffle = np.random.default_rng(
            _stable_extended_rng_seed(cfg, N, i, f"schedule-{sched_name}-shuffle"))
        shuffle_gains = []
        for _ in range(cfg.shuffle_reps):
            Ps = evolve(
                N, bits, edges, h, J,
                assign_offsets(deg, "shuffle", cfg.alpha_main, rng_shuffle),
                sched_name, cfg, problem_features=features)
            shuffle_gains.append(
                manifold_metrics(Ps, M, bits)["seff"] - mb["seff"])
        schedule_rows[sched_name] = {
            "dc_gain": float(md["seff"] - mb["seff"]),
            "shuffle_gain": float(np.mean(shuffle_gains)),
        }

    eb = eub_curve(base["w_manifold"], cfg.budget_grid)
    alpha_rows = {}
    for alpha in cfg.alpha_grid:
        key = str(float(alpha))
        if np.isclose(alpha, cfg.alpha_main, rtol=0, atol=1e-15):
            ma = dc
        else:
            Pa = evolve(N, bits, edges, h, J,
                        -float(alpha) * deg / deg.max(), "linear", cfg,
                        problem_features=features)
            ma = manifold_metrics(Pa, M, bits)
        deub = eub_curve(ma["w_manifold"], cfg.budget_grid) - eb
        alpha_rows[key] = {
            "dc_gain": float(ma["seff"] - base["seff"]),
            "yield_gain": float(ma["yld"] - base["yld"]),
            "eub_budget_mean": float(deub.mean()),
        }

    dynamics = {}
    for label, snapshots in (("base", base_snap), ("dc", dc_snap)):
        metrics = [manifold_metrics(snapshots[float(s)], M, bits) for s in sgrid]
        dynamics[f"{label}_yield"] = [float(m["yld"]) for m in metrics]
        dynamics[f"{label}_seff"] = [float(m["seff"]) for m in metrics]
    dynamics["dc_base_yield"] = (
        np.asarray(dynamics["dc_yield"]) - np.asarray(dynamics["base_yield"])).tolist()
    dynamics["dc_base_seff"] = (
        np.asarray(dynamics["dc_seff"]) - np.asarray(dynamics["base_seff"])).tolist()

    return {
        "schema": "exact-extended-v1",
        "N": N,
        "instance": i,
        "seed": paper_seed(N, i),
        "K": K,
        "degen": degen,
        "controls": controls,
        "schedule": schedule_rows,
        "alpha": alpha_rows,
        "dynamics": dynamics,
    }


def _parallel_extended_records(N: int, cfg: Config):
    """Load size-stage eligibility, then compute/cache all missing diagnostics."""
    size_records = {}
    missing_size = []
    for i in range(cfg.seed_pool):
        path = _size_cache_path(cfg, N, i, "linear")
        try:
            with path.open() as handle:
                record = json.load(handle)
            if record.get("schema") != "exact-size-v3":
                raise ValueError("unexpected schema")
            size_records[i] = record
        except (OSError, ValueError, json.JSONDecodeError):
            missing_size.append(i)
    if missing_size:
        preview = ", ".join(str(i) for i in missing_size[:8])
        raise RuntimeError(
            f"N={N}: missing {len(missing_size)} size-stage cache records "
            f"({preview}). Run the size stage first.")

    eligible_ids = [
        i for i in sorted(size_records) if size_records[i]["eligible"]
    ]
    if cfg.max_instances is not None:
        eligible_ids = eligible_ids[:cfg.max_instances]
    records = {}
    missing = []
    for i in eligible_ids:
        path = _extended_cache_path(cfg, N, i)
        if cfg.resume and path.is_file():
            try:
                with path.open() as handle:
                    record = json.load(handle)
                if record.get("schema") == "exact-extended-v1":
                    records[i] = record
                    continue
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        missing.append(i)

    workers = max(1, min(int(cfg.workers), len(missing) or 1))
    dim = 1 << N
    estimated_gib = (N + 176) * dim / (1024 ** 3)
    print(f"    extended workers={workers}; estimated numerical peak about "
          f"{estimated_gib:.2f} GiB/worker ({estimated_gib * workers:.1f} GiB total)")
    if records:
        print(f"    resumed {len(records)}/{len(eligible_ids)} extended records")

    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(variable, "1")

    cfg_dict = asdict(cfg)
    tasks = [(N, i, cfg_dict, size_records[i]["row"]) for i in missing]
    if workers == 1:
        iterator = map(_compute_extended_instance, tasks)
        for record in iterator:
            i = int(record["instance"])
            records[i] = record
            _write_json_atomic(_extended_cache_path(cfg, N, i), record)
            print(f"    completed extended N={N} instance {i:02d}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_compute_extended_instance, task): task[1]
                for task in tasks
            }
            for future in as_completed(futures):
                i = futures[future]
                record = future.result()
                records[i] = record
                _write_json_atomic(_extended_cache_path(cfg, N, i), record)
                print(f"    completed extended N={N} instance {i:02d}", flush=True)
    return [records[i] for i in eligible_ids], cfg.seed_pool


# ======================================================================================
# Stages
# ======================================================================================
def stage_verify(cfg: Config):
    """(a) Ising ground-manifold must equal the classical MVC manifold.
       (b) split-operator convergence in n_steps."""
    print("== verify ==")
    ok_all = True
    for i in range(3):
        G, deg = build_instance(12, i, cfg)
        bits, K, M = enumerate_manifold(12, G)
        h, J = ising_params(deg, cfg)
        edges = list(G.edges())
        features = build_problem_features(bits, edges)
        D1 = ising_diagonal(bits, edges, h, J, np.ones(12), features)  # s=1, no offset
        ground = np.where(D1 <= D1.min() + 1e-9)[0]
        ok = set(ground.tolist()) == set(np.where(M)[0].tolist())
        ok_all &= ok
        print(f"  seed {paper_seed(12,i)}: |M|={int(M.sum())} K={K}  Ising-ground==manifold: {ok}")
    # convergence: dc broadening at alpha_main for one instance, over several n_steps
    G, deg = build_instance(12, 1, cfg)
    bits, K, M = enumerate_manifold(12, G)
    h, J = ising_params(deg, cfg)
    edges = list(G.edges())
    features = build_problem_features(bits, edges)
    c = deg / deg.max()
    print("  convergence (dSeff at alpha=%.2f):" % cfg.alpha_main)
    for ns in (300, 600, 1200):
        cc = Config(**{**asdict(cfg), "n_steps": ns})
        P0 = evolve(12, bits, edges, h, J, np.zeros(12), "linear", cc,
                    problem_features=features)
        P1 = evolve(12, bits, edges, h, J, -cfg.alpha_main * c, "linear", cc,
                    problem_features=features)
        s0 = manifold_metrics(P0, M, bits)["seff"]
        s1 = manifold_metrics(P1, M, bits)["seff"]
        print(f"    n_steps={ns:4d}: Seff {s0:.4f} -> {s1:.4f}  (dSeff {s1-s0:+.4f})")
    return {"ising_ground_ok": bool(ok_all)}


def _paired(instances, cfg: Config, sched_name="linear"):
    """Per-instance paired DC-Base, DC-Shuffle, and discovery contrasts."""
    rng = np.random.default_rng(cfg.rng_seed)
    rows = []
    for inst in instances:
        N = inst["N"]; deg = inst["deg"]; bits = inst["bits"]; M = inst["M"]
        h, J, edges = inst["h"], inst["J"], inst["edges"]
        problem_features = build_problem_features(bits, edges)
        P0 = evolve(N, bits, edges, h, J, np.zeros(N), sched_name, cfg,
                    problem_features=problem_features)
        base = manifold_metrics(P0, M, bits)
        Pdc = evolve(N, bits, edges, h, J,
                     assign_offsets(deg, "dc", cfg.alpha_main, rng), sched_name, cfg,
                     problem_features=problem_features)
        dc = manifold_metrics(Pdc, M, bits)
        sh_gain = []
        for _ in range(cfg.shuffle_reps):
            Psh = evolve(N, bits, edges, h, J,
                         assign_offsets(deg, "shuffle", cfg.alpha_main, rng), sched_name, cfg,
                         problem_features=problem_features)
            sh_gain.append(manifold_metrics(Psh, M, bits)["seff"] - base["seff"])
        eb = eub_curve(base["w_manifold"], cfg.budget_grid)   # unconditional E[U_R], Base
        ed = eub_curve(dc["w_manifold"], cfg.budget_grid)     # unconditional E[U_R], DC
        deub = ed - eb
        dc_gain = dc["seff"] - base["seff"]
        shuffle_gain = float(np.mean(sh_gain))
        rows.append(dict(seed=inst["seed"], instance=inst["i"], degen=inst["degen"],
                         seff_base=base["seff"], seff_dc=dc["seff"],
                         dc_gain=dc_gain,
                         dc_relative_gain=dc_gain / base["seff"],
                         dc_log_ratio=float(np.log(dc["seff"] / base["seff"])),
                         shuffle_gain=shuffle_gain,
                         dc_minus_shuffle=dc_gain - shuffle_gain,
                         dc_minus_shuffle_relative=(dc_gain - shuffle_gain) / base["seff"],
                         ent_base=base["entropy"], ent_dc=dc["entropy"],
                         yld_base=base["yld"], yld_dc=dc["yld"],
                         eub_base=eb, eub_dc=ed,                       # arrays (stripped before JSON)
                         deub_curve=deub,
                         eub_budget_mean=float(deub.mean()),
                         eub_budget_fraction_mean=float(deub.mean() / inst["degen"]),
                         eub_peak_gain=float(deub.max()),
                         eub_peak_B=int(cfg.budget_grid[int(deub.argmax())])))
    return rows


def stage_size(cfg: Config):
    """Size-robustness: paired DC-Base and DC-Shuffle at each N in cfg.sizes (linear schedule)."""
    print("== size robustness ==")
    out = {}
    for N in cfg.sizes:
        if N >= cfg.parallel_min_N:
            rows, searched = _parallel_paired_size(N, cfg, "linear")
        else:
            instances, searched = instance_pool(N, cfg)
            rows = _paired(instances, cfg, "linear")
        if not rows:
            raise RuntimeError(f"No eligible size-stage instances for N={N}")
        rng = np.random.default_rng(cfg.rng_seed + 999 + N)
        rng_p = np.random.default_rng(cfg.rng_seed + 4242 + N)
        dcg = np.array([r["dc_gain"] for r in rows])
        relg = np.array([r["dc_relative_gain"] for r in rows])
        logg = np.array([r["dc_log_ratio"] for r in rows])
        shg = np.array([r["shuffle_gain"] for r in rows])
        dsrel = np.array([r["dc_minus_shuffle_relative"] for r in rows])
        deub = np.array([r["deub_curve"] for r in rows])   # n_instances x n_budgets
        mean_deub = deub.mean(0)
        pk = int(mean_deub.argmax()); peakB = int(cfg.budget_grid[pk])
        # paired bootstrap CIs for the conditional (Simpson) gains
        dcm, dclo, dchi = bootstrap_ci(dcg, cfg.n_boot, rng)
        relm, rello, relhi = bootstrap_ci(relg, cfg.n_boot, rng)
        logm, loglo, loghi = bootstrap_ci(logg, cfg.n_boot, rng)
        dsm, dslo, dshi = bootstrap_ci(dcg - shg, cfg.n_boot, rng)
        dsrelm, dsrello, dsrelhi = bootstrap_ci(dsrel, cfg.n_boot, rng)
        # ---- Peak-selection-robust inference for finite-budget discovery dE[U_R] ----
        # (1) Mean over the fixed approximately log-spaced budget grid.
        # This is a discrete grid mean, not a continuous numerical integral.
        budget_mean = deub.mean(axis=1)
        budget_m, budget_lo, budget_hi = bootstrap_ci(budget_mean, cfg.n_boot, rng)
        budget_fraction = np.array([r["eub_budget_fraction_mean"] for r in rows])
        budget_frac_m, budget_frac_lo, budget_frac_hi = bootstrap_ci(
            budget_fraction, cfg.n_boot, rng)
        # (2) Two-sided max-absolute paired sign-flip permutation test. One sign is
        # applied to each graph's entire curve, preserving within-curve dependence.
        obs_max = float(np.abs(mean_deub).max())
        perm_max = np.empty(cfg.n_perm)
        for b in range(cfg.n_perm):
            signs = rng_p.choice([-1.0, 1.0], size=deub.shape[0])[:, None]
            perm_max[b] = np.abs((deub * signs).mean(0)).max()
        maxstat_p = float((np.sum(perm_max >= obs_max) + 1) / (cfg.n_perm + 1))
        # (3) pointwise bootstrap band for the descriptive curve
        boot = np.array([deub[rng.integers(0, deub.shape[0], deub.shape[0])].mean(0)
                         for _ in range(cfg.n_boot)])
        band_lo = np.percentile(boot, 2.5, axis=0).tolist()
        band_hi = np.percentile(boot, 97.5, axis=0).tolist()
        # peak CI kept ONLY as descriptive (selection-biased; NOT used as confirmatory evidence)
        _, eub_lo, eub_hi = bootstrap_ci(deub[:, pk], cfg.n_boot, rng)
        for r in rows:                                        # strip big arrays before storing
            r.pop("eub_base", None); r.pop("eub_dc", None)
        summ = dict(
            N=N, alpha=cfg.alpha_main, searched=searched, included=len(rows),
            dc_gain_mean=dcm, dc_gain_ci=[dclo, dchi], dc_gain_se=float(dcg.std() / np.sqrt(len(dcg))),
            dc_relative_gain_mean=relm, dc_relative_gain_ci=[rello, relhi],
            dc_log_ratio_mean=logm, dc_log_ratio_ci=[loglo, loghi],
            shuffle_gain_mean=float(shg.mean()),
            dc_minus_shuffle_mean=dsm, dc_minus_shuffle_ci=[dslo, dshi],
            dc_minus_shuffle_relative_mean=dsrelm,
            dc_minus_shuffle_relative_ci=[dsrello, dsrelhi],
            dc_gt_shuffle=int(np.sum(dcg > shg)), dc_gt_zero=int(np.sum(dcg > 0)),
            # discovery gain: confirmatory statistics robust to selecting a peak budget
            eub_budget_mean=budget_m, eub_budget_mean_ci=[budget_lo, budget_hi],
            eub_budget_fraction_mean=budget_frac_m,
            eub_budget_fraction_ci=[budget_frac_lo, budget_frac_hi],
            eub_maxstat_p=maxstat_p, eub_maxstat_significant=bool(maxstat_p < 0.05),
            # discovery gain: DESCRIPTIVE only (peak is data-selected -> not confirmatory)
            eub_peak_gain_descriptive=float(mean_deub[pk]), eub_peak_B_descriptive=peakB,
            eub_peak_ci_descriptive=[eub_lo, eub_hi],
            mean_deub_curve=mean_deub.tolist(), deub_band_lo=band_lo, deub_band_hi=band_hi,
            budget_grid=list(cfg.budget_grid),
            rows=rows,
        )
        out[str(N)] = summ
        print(f"  N={N} (alpha={cfg.alpha_main}): included {len(rows)}/{searched} | "
              f"dc_gain {dcm:+.3f} [{dclo:+.3f},{dchi:+.3f}] | "
              f"relative {relm:+.3f} [{rello:+.3f},{relhi:+.3f}] | "
              f"dc-shuffle {dsm:+.3f} [{dslo:+.3f},{dshi:+.3f}] ({int(np.sum(dcg > shg))}/{len(rows)}) | "
              f"dEUB grid-mean {budget_m:+.4f} [{budget_lo:+.4f},{budget_hi:+.4f}], "
              f"two-sided max-|stat| p={maxstat_p:.4f} | "
              f"peak(descr) {mean_deub[pk]:+.3f}@B~{peakB}")
    # The three size-specific max-statistic tests form a fixed family.
    mtests = max(1, len(out))
    for summ in out.values():
        summ["eub_maxstat_p_bonferroni"] = min(1.0, summ["eub_maxstat_p"] * mtests)
    return out


def stage_controls(cfg: Config, N=12):
    """5-scheme control panel at alpha_main (linear schedule) on the N-instance pool."""
    print("== controls panel ==")
    instances, searched = instance_pool(N, cfg)
    rng = np.random.default_rng(cfg.rng_seed)
    schemes = ["dc", "homogeneous", "shuffle", "anti", "random"]
    raw = {s: [] for s in schemes}
    for inst in instances:
        Nn = inst["N"]; deg = inst["deg"]; bits = inst["bits"]; M = inst["M"]
        h, J, edges = inst["h"], inst["J"], inst["edges"]
        features = build_problem_features(bits, edges)
        s0 = manifold_metrics(evolve(Nn, bits, edges, h, J, np.zeros(Nn), "linear", cfg,
                                     problem_features=features), M, bits)["seff"]
        for sc in schemes:
            reps = cfg.shuffle_reps if sc in ("shuffle", "random") else 1
            g = []
            for _ in range(reps):
                P = evolve(Nn, bits, edges, h, J,
                           assign_offsets(deg, sc, cfg.alpha_main, rng), "linear", cfg,
                           problem_features=features)
                g.append(manifold_metrics(P, M, bits)["seff"] - s0)
            raw[sc].append(float(np.mean(g)))
    summary = {sc: dict(mean=float(np.mean(raw[sc])),
                        se=float(np.std(raw[sc]) / np.sqrt(len(raw[sc])))) for sc in schemes}
    summary["_meta"] = dict(N=N, searched=searched, included=len(instances))
    for sc in schemes:
        print(f"  {sc:12s}: dSeff {summary[sc]['mean']:+.3f} ± {summary[sc]['se']:.3f}")
    summary["_raw"] = raw
    return summary


def stage_schedule(cfg: Config, N=12):
    """Schedule-shape robustness: paired DC-Base and DC-Shuffle for each schedule."""
    print("== schedule robustness ==")
    instances, searched = instance_pool(N, cfg)
    out = {}
    for name in cfg.schedules:
        rows = _paired(instances, cfg, name)
        dcg = np.array([r["dc_gain"] for r in rows])
        shg = np.array([r["shuffle_gain"] for r in rows])
        out[name] = dict(dc_gain_mean=float(dcg.mean()), shuffle_gain_mean=float(shg.mean()),
                         dc_gain_se=float(dcg.std() / np.sqrt(len(dcg))),
                         dc_gt_shuffle=int(np.sum(dcg > shg)), included=len(rows))
        print(f"  {name:10s}: dc {out[name]['dc_gain_mean']:+.3f}, "
              f"shuffle {out[name]['shuffle_gain_mean']:+.3f}, "
              f"dc>shuffle {out[name]['dc_gt_shuffle']}/{len(rows)}")
    out["_meta"] = dict(N=N, searched=searched, included=len(instances))
    return out


def stage_alpha(cfg: Config, N=12):
    """Fixed alpha sensitivity on the smallest surrogate size.

    This stage isolates the operating window of the degree-aligned rule relative
    to the zero-offset baseline. Degree-alignment specificity is tested separately
    at alpha_main by `stage_size` and `stage_controls`; repeating all shuffled
    assignments at every alpha would not address the fixed sensitivity
    question. This is not an optimization of alpha.
    """
    print("== alpha sensitivity ==")
    instances, searched = instance_pool(N, cfg)
    alpha_values = tuple(float(a) for a in cfg.alpha_grid)
    raw = {str(a): [] for a in alpha_values}

    for inst in instances:
        Nn = inst["N"]; deg = inst["deg"]; bits = inst["bits"]; M = inst["M"]
        h, J, edges = inst["h"], inst["J"], inst["edges"]
        features = build_problem_features(bits, edges)
        P0 = evolve(Nn, bits, edges, h, J, np.zeros(Nn), "linear", cfg,
                    problem_features=features)
        base = manifold_metrics(P0, M, bits)
        eb = eub_curve(base["w_manifold"], cfg.budget_grid)
        c = deg / deg.max()

        for alpha in alpha_values:
            delta = -alpha * c
            Pdc = evolve(Nn, bits, edges, h, J, delta, "linear", cfg,
                         problem_features=features)
            dc = manifold_metrics(Pdc, M, bits)
            deub = eub_curve(dc["w_manifold"], cfg.budget_grid) - eb
            raw[str(alpha)].append(dict(
                seed=inst["seed"], degen=inst["degen"],
                dc_gain=float(dc["seff"] - base["seff"]),
                yield_gain=float(dc["yld"] - base["yld"]),
                eub_budget_mean=float(deub.mean()),
            ))

    out = {}
    for j, alpha in enumerate(alpha_values):
        rows = raw[str(alpha)]
        rng = np.random.default_rng(cfg.rng_seed + 7100 + j)
        summary = {"alpha": alpha, "rows": rows}
        for key in ("dc_gain", "yield_gain", "eub_budget_mean"):
            values = np.array([r[key] for r in rows])
            mean, lo, hi = bootstrap_ci(values, cfg.n_boot, rng)
            summary[f"{key}_mean"] = mean
            summary[f"{key}_ci"] = [lo, hi]
        out[str(alpha)] = summary
        print(f"  alpha={alpha:.3f}: dc_gain {summary['dc_gain_mean']:+.3f} "
              f"[{summary['dc_gain_ci'][0]:+.3f},{summary['dc_gain_ci'][1]:+.3f}] | "
              f"yield {summary['yield_gain_mean']:+.4f} | "
              f"dEUB-grid {summary['eub_budget_mean_mean']:+.4f}")
    out["_meta"] = dict(N=N, searched=searched, included=len(instances),
                        alpha_values=list(alpha_values),
                        interpretation="fixed sensitivity analysis; not alpha optimization")
    return out


def stage_dynamics(cfg: Config, N=12):
    """Time-resolved target-manifold occupation and conditional support.

    These curves diagnose when the final DC--Base contrast forms inside the
    idealized unitary model. Degree-alignment specificity is handled by the
    endpoint shuffled-assignment controls. The curves do not estimate device
    freeze-out. Pointwise bootstrap bands are descriptive.
    """
    print("== time-resolved redistribution ==")
    instances, searched = instance_pool(N, cfg)
    sgrid = np.asarray(cfg.trajectory_grid, dtype=float)
    records = []

    for inst in instances:
        Nn = inst["N"]; deg = inst["deg"]; bits = inst["bits"]; M = inst["M"]
        h, J, edges = inst["h"], inst["J"], inst["edges"]
        features = build_problem_features(bits, edges)
        delta = -cfg.alpha_main * deg / deg.max()
        _, base_snap = evolve(Nn, bits, edges, h, J, np.zeros(Nn), "linear", cfg,
                              trajectory_s=sgrid, problem_features=features)
        _, dc_snap = evolve(Nn, bits, edges, h, J, delta, "linear", cfg,
                            trajectory_s=sgrid, problem_features=features)

        bm = [manifold_metrics(base_snap[float(s)], M, bits) for s in sgrid]
        dm = [manifold_metrics(dc_snap[float(s)], M, bits) for s in sgrid]
        base_yield = np.array([m["yld"] for m in bm])
        dc_yield = np.array([m["yld"] for m in dm])
        base_seff = np.array([m["seff"] for m in bm])
        dc_seff = np.array([m["seff"] for m in dm])
        records.append(dict(
            seed=inst["seed"], degen=inst["degen"],
            base_yield=base_yield, dc_yield=dc_yield,
            base_seff=base_seff, dc_seff=dc_seff,
            dc_base_yield=dc_yield-base_yield,
            dc_base_seff=dc_seff-base_seff,
        ))

    rng = np.random.default_rng(cfg.rng_seed + 8200)
    out = {"N": N, "alpha": cfg.alpha_main, "searched": searched,
           "included": len(records), "s_grid": sgrid.tolist(), "curves": {}}
    for key in ("base_yield", "dc_yield", "base_seff", "dc_seff",
                "dc_base_yield", "dc_base_seff"):
        arr = np.array([r[key] for r in records])
        mean, lo, hi = bootstrap_curve_ci(arr, cfg.n_boot, rng)
        out["curves"][key] = {"mean": mean.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
    out["rows"] = [{"seed": r["seed"], "degen": r["degen"],
                    **{k: np.asarray(v).tolist() for k, v in r.items()
                       if k not in ("seed", "degen")}} for r in records]
    final = -1
    print(f"  N={N}: included {len(records)}/{searched} | final dc-base dSeff "
          f"{out['curves']['dc_base_seff']['mean'][final]:+.3f} | final yield shift "
          f"{out['curves']['dc_base_yield']['mean'][final]:+.4f}")
    return out


def stage_extended(cfg: Config):
    """Run the full sensitivity/control suite for every size in ``cfg.sizes``.

    This stage is intended for the expensive N=18--20 extension. It requires the
    completed linear size-stage cache, reuses its baseline/DC/shuffle endpoint
    results, and adds all remaining assignment, schedule, alpha, and trajectory
    diagnostics with per-instance multiprocessing and resumable records.
    """
    print("== full extended controls ==")
    out = {}
    control_schemes = ("dc", "homogeneous", "shuffle", "anti", "random")
    curve_names = ("base_yield", "dc_yield", "base_seff", "dc_seff",
                   "dc_base_yield", "dc_base_seff")
    for N in cfg.sizes:
        print(f"  -- N={N} --")
        records, searched = _parallel_extended_records(N, cfg)
        if not records:
            raise RuntimeError(f"No eligible extended-control instances for N={N}")
        rng = np.random.default_rng(cfg.rng_seed + 12000 + N)

        controls = {"_meta": {
            "N": N, "searched": searched, "included": len(records),
            "error_bars": "paired instance-bootstrap 95% confidence intervals",
        }}
        dc_values = np.asarray([r["controls"]["dc"] for r in records], dtype=float)
        for scheme in control_schemes:
            values = np.asarray([r["controls"][scheme] for r in records], dtype=float)
            mean, lo, hi = bootstrap_ci(values, cfg.n_boot, rng)
            entry = {
                "mean": mean,
                "ci": [lo, hi],
                "se": float(values.std() / np.sqrt(len(values))),
            }
            if scheme != "dc":
                contrast = dc_values - values
                cm, clo, chi = bootstrap_ci(contrast, cfg.n_boot, rng)
                entry.update({
                    "dc_minus_control_mean": cm,
                    "dc_minus_control_ci": [clo, chi],
                    "dc_gt_control": int(np.sum(dc_values > values)),
                })
            controls[scheme] = entry

        schedule_out = {"_meta": {
            "N": N, "searched": searched, "included": len(records),
            "error_bars": "paired instance-bootstrap 95% confidence intervals",
        }}
        for sched_name in cfg.schedules:
            dcg = np.asarray(
                [r["schedule"][sched_name]["dc_gain"] for r in records], dtype=float)
            shg = np.asarray(
                [r["schedule"][sched_name]["shuffle_gain"] for r in records], dtype=float)
            dcm, dclo, dchi = bootstrap_ci(dcg, cfg.n_boot, rng)
            shm, shlo, shhi = bootstrap_ci(shg, cfg.n_boot, rng)
            dm, dlo, dhi = bootstrap_ci(dcg - shg, cfg.n_boot, rng)
            schedule_out[sched_name] = {
                "dc_gain_mean": dcm, "dc_gain_ci": [dclo, dchi],
                "shuffle_gain_mean": shm, "shuffle_gain_ci": [shlo, shhi],
                "dc_minus_shuffle_mean": dm,
                "dc_minus_shuffle_ci": [dlo, dhi],
                "dc_gt_shuffle": int(np.sum(dcg > shg)),
            }

        alpha_out = {"_meta": {
            "N": N, "searched": searched, "included": len(records),
            "alpha_values": [float(a) for a in cfg.alpha_grid],
            "interpretation": "fixed sensitivity analysis; not alpha optimization",
        }}
        for alpha in cfg.alpha_grid:
            key = str(float(alpha))
            entry = {"alpha": float(alpha)}
            for metric in ("dc_gain", "yield_gain", "eub_budget_mean"):
                values = np.asarray(
                    [r["alpha"][key][metric] for r in records], dtype=float)
                mean, lo, hi = bootstrap_ci(values, cfg.n_boot, rng)
                entry[f"{metric}_mean"] = mean
                entry[f"{metric}_ci"] = [lo, hi]
            alpha_out[key] = entry

        dynamics = {
            "N": N, "alpha": cfg.alpha_main, "searched": searched,
            "included": len(records), "s_grid": list(cfg.trajectory_grid),
            "curves": {},
        }
        for curve_name in curve_names:
            curves = np.asarray(
                [r["dynamics"][curve_name] for r in records], dtype=float)
            mean, lo, hi = bootstrap_curve_ci(curves, cfg.n_boot, rng)
            dynamics["curves"][curve_name] = {
                "mean": mean.tolist(), "lo": lo.tolist(), "hi": hi.tolist()
            }

        out[str(N)] = {
            "N": N,
            "searched": searched,
            "included": len(records),
            "controls": controls,
            "schedule": schedule_out,
            "alpha": alpha_out,
            "dynamics": dynamics,
            "rows": records,
        }
        print(
            f"  N={N}: included {len(records)}/{searched} | "
            f"DC-homogeneous {controls['homogeneous']['dc_minus_control_mean']:+.3f} "
            f"[{controls['homogeneous']['dc_minus_control_ci'][0]:+.3f},"
            f"{controls['homogeneous']['dc_minus_control_ci'][1]:+.3f}] | "
            f"DC-anti {controls['anti']['dc_minus_control_mean']:+.3f} | "
            f"DC-random {controls['random']['dc_minus_control_mean']:+.3f}")
    return out


def synchronize_extended_primary_statistics(extended, size_results, cfg: Config):
    """Use one canonical estimate whenever the same contrast appears repeatedly.

    The expensive endpoint samples are identical across the size, controls,
    schedule, and alpha summaries. Re-bootstrapping those same vectors with
    different RNG streams would produce harmless but visually inconsistent
    confidence limits. The size-stage summaries are therefore canonical for the
    fixed alpha_main linear-schedule outcomes.
    """
    main_alpha_key = str(float(cfg.alpha_main))
    for N, result in extended.items():
        if N not in size_results:
            raise KeyError(f"Missing canonical size summary for N={N}")
        primary = size_results[N]
        controls = result["controls"]
        schedule_linear = result["schedule"]["linear"]
        alpha_main = result["alpha"][main_alpha_key]

        # Degree-aligned minus zero-offset baseline.
        controls["dc"]["mean"] = primary["dc_gain_mean"]
        controls["dc"]["ci"] = list(primary["dc_gain_ci"])
        controls["dc"]["se"] = primary["dc_gain_se"]
        schedule_linear["dc_gain_mean"] = primary["dc_gain_mean"]
        schedule_linear["dc_gain_ci"] = list(primary["dc_gain_ci"])
        alpha_main["dc_gain_mean"] = primary["dc_gain_mean"]
        alpha_main["dc_gain_ci"] = list(primary["dc_gain_ci"])

        # DC minus shuffled multiset.
        controls["shuffle"]["dc_minus_control_mean"] = \
            primary["dc_minus_shuffle_mean"]
        controls["shuffle"]["dc_minus_control_ci"] = \
            list(primary["dc_minus_shuffle_ci"])
        controls["shuffle"]["dc_gt_control"] = primary["dc_gt_shuffle"]
        schedule_linear["dc_minus_shuffle_mean"] = \
            primary["dc_minus_shuffle_mean"]
        schedule_linear["dc_minus_shuffle_ci"] = \
            list(primary["dc_minus_shuffle_ci"])
        schedule_linear["dc_gt_shuffle"] = primary["dc_gt_shuffle"]

        # The shuffle marginal is not a primary size-stage CI, so the controls
        # summary is canonical for its repeated linear-schedule appearance.
        schedule_linear["shuffle_gain_mean"] = controls["shuffle"]["mean"]
        schedule_linear["shuffle_gain_ci"] = list(controls["shuffle"]["ci"])

        # Fixed-budget-grid discovery at alpha_main.
        alpha_main["eub_budget_mean_mean"] = primary["eub_budget_mean"]
        alpha_main["eub_budget_mean_ci"] = \
            list(primary["eub_budget_mean_ci"])
    return extended


# ======================================================================================
# Plotting
# ======================================================================================
def plot_all(results, cfg: Config):
    global plt
    if plt is None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as _plt
            plt = _plt
            from matplotlib.patches import Patch  # noqa: F401
            from matplotlib.lines import Line2D  # noqa: F401
            globals()["Patch"] = Patch
            globals()["Line2D"] = Line2D
        except Exception:  # pragma: no cover
            print("matplotlib unavailable; skipping plots")
            return
    BLUE, VERMILLION, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#9aa5b1"

    # Colorblind-safe (Okabe--Ito) palette keyed by graph size, so every
    # size-resolved series is consistent across panels and avoids the default
    # matplotlib red/green pair for adjacent categories.
    _SIZE_CB_CYCLE = ["#0072B2", "#56B4E9", "#009E73", "#E69F00", "#D55E00",
                      "#CC79A7", "#F0E442", "#000000"]
    _size_keys = sorted(results.get("size", {}), key=int)
    SIZE_COLOR = {int(k): _SIZE_CB_CYCLE[i % len(_SIZE_CB_CYCLE)]
                  for i, k in enumerate(_size_keys)}

    def ci_yerr(means, cis):
        means = np.asarray(means, dtype=float)
        lo = means - np.asarray([ci[0] for ci in cis], dtype=float)
        hi = np.asarray([ci[1] for ci in cis], dtype=float) - means
        return np.vstack([lo, hi])

    def panel_label(ax, label):
        ax.text(0.0, 1.025, label, transform=ax.transAxes,
                fontsize=12, fontweight="bold", ha="left", va="bottom",
                clip_on=False)

    # (1) size robustness
    if "size" in results:
        sz = results["size"]; Ns = sorted(sz.keys(), key=int)
        x = np.arange(len(Ns)); w = 0.35
        dc = [sz[n]["dc_gain_mean"] for n in Ns]
        dc_ci = [sz[n]["dc_gain_ci"] for n in Ns]
        aligned = [sz[n]["dc_minus_shuffle_mean"] for n in Ns]
        aligned_ci = [sz[n]["dc_minus_shuffle_ci"] for n in Ns]
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.bar(x - w/2, dc, w, yerr=ci_yerr(dc, dc_ci), capsize=3,
               label="DC minus no offset", color=BLUE, hatch="//")
        ax.bar(x + w/2, aligned, w, yerr=ci_yerr(aligned, aligned_ci), capsize=3,
               label="degree-aligned minus shuffled multiset", color=VERMILLION, hatch="\\\\")
        for k, n in enumerate(Ns):
            ax.annotate(f"{sz[n]['dc_gt_shuffle']}/{sz[n]['included']}",
                        (x[k]+w/2, aligned_ci[k][1]), textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=8)
        ax.axhline(0, color="k", lw=.8)
        ax.set_xticks(x); ax.set_xticklabels([f"N={n}\n(n={sz[n]['included']})" for n in Ns])
        ax.set_ylabel("conditional effective-support contrast (95% CI)")
        ax.set_title("Degree-aligned schedule contrasts across surrogate sizes")
        ax.legend(fontsize=9); ax.grid(alpha=.3, axis="y")
        fig.tight_layout(); fig.savefig(os.path.join(cfg.outdir, "fig_size_robustness.png"), dpi=170)
        plt.close(fig)

        # Scale-normalized companion: used to distinguish a genuine size trend
        # from changes in baseline effective support or optimum-manifold size.
        if all("dc_relative_gain_mean" in sz[n] for n in Ns):
            fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.0))
            nvals = np.asarray([int(n) for n in Ns])
            specifications = [
                ("dc_relative_gain_mean", "dc_relative_gain_ci",
                 r"$(S_{\rm DC}-S_{\rm Base})/S_{\rm Base}$", BLUE, "o",
                 "(a) Relative DC--baseline support gain"),
                ("dc_minus_shuffle_relative_mean", "dc_minus_shuffle_relative_ci",
                 r"$(S_{\rm DC}-S_{\rm shuf})/S_{\rm Base}$", VERMILLION, "s",
                 "(b) Relative alignment-specific gain"),
                ("eub_budget_fraction_mean", "eub_budget_fraction_ci",
                 r"mean $\Delta\mathbb{E}[U_R]/|\mathcal{M}|$", GREEN, "D",
                 "(c) Manifold-normalized discovery gain"),
            ]
            for ax, (mean_key, ci_key, ylabel, color, marker, title) in zip(axes, specifications):
                means = [sz[n][mean_key] for n in Ns]
                cis = [sz[n][ci_key] for n in Ns]
                ax.errorbar(nvals, means, yerr=ci_yerr(means, cis), color=color,
                            marker=marker, capsize=3)
                ax.axhline(0, color="k", lw=.8)
                ax.set_xticks(nvals)
                ax.set_xlabel(r"graph size $N$")
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.grid(alpha=.25)
            fig.suptitle("Scale-normalized exact-statevector diagnostics", fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, .93))
            fig.savefig(os.path.join(cfg.outdir, "fig_size_scaling_normalized.png"), dpi=190)
            fig.savefig(os.path.join(cfg.outdir, "fig_size_scaling_normalized.pdf"))
            plt.close(fig)

    # (1b) unconditional finite-budget discovery gain Delta E[U_R] vs draw budget
    if "size" in results:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        for n in sorted(results["size"].keys(), key=int):
            s = results["size"][n]
            p_adj = s.get("eub_maxstat_p_bonferroni", s.get("eub_maxstat_p", float("nan")))
            line, = ax.plot(s["budget_grid"], s["mean_deub_curve"], marker="o", ms=3,
                            color=SIZE_COLOR.get(int(n)),
                            label=rf"$N={n}$ ($n={s['included']}$, adjusted $p={p_adj:.3g}$)")
            if "deub_band_lo" in s:            # pointwise 95% bootstrap band (descriptive)
                ax.fill_between(s["budget_grid"], s["deub_band_lo"], s["deub_band_hi"],
                                color=line.get_color(), alpha=0.12)
        ax.axhline(0, color="k", lw=.8); ax.set_xscale("log")
        ax.set_xlabel(r"independent-draw budget $R$")
        ax.set_ylabel(r"unconditional $\Delta\mathbb{E}[U_R]$ (degree-aligned minus zero offset)")
        ax.set_title(rf"Finite-budget discovery gain (nominal $\alpha={cfg.alpha_main}$)")
        ax.legend(fontsize=9); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(os.path.join(cfg.outdir, "fig_eub.png"), dpi=170)
        plt.close(fig)

    # (2) controls panel
    if "controls" in results:
        p = results["controls"]; order = ["dc", "homogeneous", "shuffle", "anti", "random"]
        labels = {"dc": "degree-\naligned", "homogeneous": "mean-matched\nuniform",
                  "shuffle": "shuffled\nmultiset", "anti": "degree-\nreversed", "random": "random"}
        means = [p[s]["mean"] for s in order]; ses = [p[s]["se"] for s in order]
        cols = [BLUE if s == "dc" else GREY for s in order]
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.bar(range(len(order)), means, yerr=ses, color=cols, capsize=3)
        ax.axhline(0, color="k", lw=.8)
        ax.set_xticks(range(len(order))); ax.set_xticklabels([labels[s] for s in order], fontsize=9)
        ax.set_ylabel("effective-support change vs no offset")
        ax.set_title(f"Dependence on offset assignment\n(nominal alpha={cfg.alpha_main}, N={p['_meta']['N']}, n={p['_meta']['included']})")
        ax.grid(alpha=.3, axis="y")
        fig.tight_layout(); fig.savefig(os.path.join(cfg.outdir, "fig_controls.png"), dpi=170)
        plt.close(fig)

    # (3) schedule robustness
    if "schedule" in results:
        sc = results["schedule"]; names = [n for n in sc if not n.startswith("_")]
        x = np.arange(len(names)); w = 0.35
        dc = [sc[n]["dc_gain_mean"] for n in names]; sh = [sc[n]["shuffle_gain_mean"] for n in names]
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.bar(x - w/2, dc, w, label="degree-aligned (DC)", color=BLUE)
        ax.bar(x + w/2, sh, w, label="shuffled multiset", color=GREY)
        for k, n in enumerate(names):
            ax.annotate(f"{sc[n]['dc_gt_shuffle']}/{sc[n]['included']}",
                        (x[k]-w/2, dc[k]), textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=8)
        ax.axhline(0, color="k", lw=.8)
        display = {"linear": "linear", "convex_power": "convex\npower", "trig": "trigonometric"}
        ax.set_xticks(x); ax.set_xticklabels([display.get(n, n) for n in names])
        ax.set_ylabel("effective-support gain vs no offset")
        ax.set_title(f"Schedule-shape robustness (N={sc['_meta']['N']}, n={sc['_meta']['included']})")
        ax.legend(fontsize=9); ax.grid(alpha=.3, axis="y")
        fig.tight_layout(); fig.savefig(os.path.join(cfg.outdir, "fig_schedule.png"), dpi=170)
        plt.close(fig)

    # (3b) fixed alpha sensitivity
    if "alpha" in results:
        aa = results["alpha"]
        keys = sorted((k for k in aa if not k.startswith("_")), key=float)
        avals = np.array([aa[k]["alpha"] for k in keys])
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

        ax = axes[0]
        vals = [aa[k]["dc_gain_mean"] for k in keys]
        cis = [aa[k]["dc_gain_ci"] for k in keys]
        ax.errorbar(avals, vals, yerr=ci_yerr(vals, cis), marker="o",
                    color=BLUE, capsize=3)
        ax.axhline(0, color="k", lw=.8)
        ax.set_xlabel(r"offset scale $\alpha$")
        ax.set_ylabel(r"conditional $\Delta S_{\mathrm{eff}}$ (95% CI)")
        ax.set_title("(a) Effective support")
        ax.grid(alpha=.25)

        ax = axes[1]
        vals = [aa[k]["yield_gain_mean"] for k in keys]
        cis = [aa[k]["yield_gain_ci"] for k in keys]
        ax.errorbar(avals, vals, yerr=ci_yerr(vals, cis), marker="^", color=GREY,
                    capsize=3)
        ax.axhline(0, color="k", lw=.8)
        ax.set_xlabel(r"offset scale $\alpha$")
        ax.set_ylabel("target-manifold occupation shift (95% CI)")
        ax.set_title("(b) Occupation trade-off")
        ax.grid(alpha=.25)

        ax = axes[2]
        vals = [aa[k]["eub_budget_mean_mean"] for k in keys]
        cis = [aa[k]["eub_budget_mean_ci"] for k in keys]
        ax.errorbar(avals, vals, yerr=ci_yerr(vals, cis), marker="D", color=GREEN,
                    capsize=3)
        ax.axhline(0, color="k", lw=.8)
        ax.set_xlabel(r"offset scale $\alpha$")
        ax.set_ylabel(r"mean $\Delta\mathbb{E}[U_R]$ over budget grid (95% CI)")
        ax.set_title("(c) Expected discovery")
        ax.grid(alpha=.25)
        fig.suptitle(rf"Fixed offset-scale sensitivity ($N={aa['_meta']['N']}$)")
        fig.tight_layout(rect=(0, 0, 1, .94))
        fig.savefig(os.path.join(cfg.outdir, "fig_alpha_sensitivity.png"), dpi=190)
        fig.savefig(os.path.join(cfg.outdir, "fig_alpha_sensitivity.pdf"))
        plt.close(fig)

    # (3c) time-resolved target-manifold redistribution
    if "dynamics" in results:
        dy = results["dynamics"]; sgrid = np.asarray(dy["s_grid"])
        curves = dy["curves"]
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
        ax = axes[0]
        for key, label, color, ls in (
                ("base_yield", "baseline", GREY, "--"),
                ("dc_yield", "degree-aligned", BLUE, "-")):
            c = curves[key]
            ax.plot(sgrid, c["mean"], label=label, color=color, ls=ls)
            ax.fill_between(sgrid, c["lo"], c["hi"], color=color, alpha=.11)
        ax.set_xlabel(r"anneal fraction $s$")
        ax.set_ylabel(r"target-manifold probability $Y_{\mathcal{M}}(s)$")
        ax.set_title("(a) Occupation of the final optimum manifold")
        ax.legend(fontsize=8); ax.grid(alpha=.25)

        ax = axes[1]
        for key, label, color, ls in (
                ("dc_base_seff", "degree-aligned minus zero offset", BLUE, "-"),):
            c = curves[key]
            ax.plot(sgrid, c["mean"], label=label, color=color, ls=ls)
            ax.fill_between(sgrid, c["lo"], c["hi"], color=color, alpha=.13)
        ax.axhline(0, color="k", lw=.8)
        ax.set_xlabel(r"anneal fraction $s$")
        ax.set_ylabel(r"conditional $\Delta S_{\mathrm{eff}}(s)$")
        ax.set_title("(b) Formation of the effective-support contrast")
        ax.legend(fontsize=8); ax.grid(alpha=.25)
        fig.suptitle(rf"Time-resolved redistribution in the idealized model "
                     rf"($N={dy['N']}$, $\alpha={dy['alpha']}$; pointwise 95% bands)")
        fig.tight_layout(rect=(0, 0, 1, .93))
        fig.savefig(os.path.join(cfg.outdir, "fig_time_resolved.png"), dpi=190)
        fig.savefig(os.path.join(cfg.outdir, "fig_time_resolved.pdf"))
        plt.close(fig)

    # (3d) combined manuscript figure for operating-window and dynamical diagnostics.
    # The primary N=12 pool and the independent N=18--20 extension use the same
    # fixed alpha grid and trajectory fractions.
    if "alpha" in results and "dynamics" in results:
        panels = [(
            str(results["alpha"]["_meta"]["N"]),
            results["alpha"],
            results["dynamics"],
        )]
        if "extended" in results:
            panels.extend(
                (n, results["extended"][n]["alpha"],
                 results["extended"][n]["dynamics"])
                for n in sorted(results["extended"], key=int)
            )
        # Cycled so that any number of size panels is covered; colours are
        # overridden per size by SIZE_COLOR below.
        _base_pal = [BLUE, VERMILLION, GREEN, "#CC79A7", "#333333"]
        _base_mrk = ["o", "s", "D", "^", "v"]
        palette = [_base_pal[i % len(_base_pal)] for i in range(len(panels))]
        markers = [_base_mrk[i % len(_base_mrk)] for i in range(len(panels))]
        # Four panels: the operating window (alpha sweep) on top and the
        # time-resolved dynamical trade-off below. The budget-grid-mean
        # discovery panel is omitted; curve-wise discovery inference is
        # reported with the full curves and the max-statistic test.
        fig = plt.figure(figsize=(11.6, 7.7))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05],
                              hspace=.38, wspace=.30)
        axes = [fig.add_subplot(gs[0, j]) for j in range(2)]
        ax_yield = fig.add_subplot(gs[1, 0])
        ax_support = fig.add_subplot(gs[1, 1])

        metric_specs = [
            ("dc_gain_mean", "dc_gain_ci",
             r"conditional $\Delta S_{\mathrm{eff}}$ (95% CI)"
             "\n" r"(broader support $\uparrow$)",
             "(a)"),
            ("yield_gain_mean", "yield_gain_ci",
            "target-manifold occupation shift (95% CI)\n"
             r"(more occupation $\uparrow$)",
             "(b)"),
        ]
        # Small horizontal dodge so the three size series do not overlap on the
        # discrete alpha grid; error bars stay readable where CIs are wide.
        _dodge = np.linspace(-0.004, 0.004, len(panels))
        for ax, (mean_key, ci_key, ylabel, label) in zip(axes, metric_specs):
            for _i, ((n, alpha_result, _), color, marker) in enumerate(
                    zip(panels, palette, markers)):
                color = SIZE_COLOR.get(int(n), color)
                keys = sorted(
                    (k for k in alpha_result if not k.startswith("_")),
                    key=float)
                avals = np.array([alpha_result[k]["alpha"] for k in keys])
                vals = [alpha_result[k][mean_key] for k in keys]
                cis = [alpha_result[k][ci_key] for k in keys]
                ax.errorbar(avals + _dodge[_i], vals, yerr=ci_yerr(vals, cis),
                            marker=marker, ms=4, color=color, capsize=2.5,
                            lw=1.1, elinewidth=1.0,
                            label=rf"$N={n}$")
            ax.axhline(0, color="k", lw=.8)
            ax.grid(alpha=.25)
            ax.set_xlabel(r"offset scale $\alpha$")
            # Only the tested alpha values are labelled on the axis.
            _alpha_ticks = sorted({float(a) for a in cfg.alpha_grid})
            ax.set_xticks(_alpha_ticks)
            ax.set_xticklabels([f"{a:g}" for a in _alpha_ticks])
            ax.set_ylabel(ylabel)
            panel_label(ax, label)
        axes[0].legend(fontsize=8)

        for (n, _, dynamics), color in zip(panels, palette):
            color = SIZE_COLOR.get(int(n), color)
            sgrid = np.asarray(dynamics["s_grid"])
            curve = dynamics["curves"]["dc_base_yield"]
            ax_yield.plot(sgrid, curve["mean"], color=color,
                          label=rf"$N={n}$")
            ax_yield.fill_between(sgrid, curve["lo"], curve["hi"],
                                  color=color, alpha=.11)
        ax_yield.axhline(0, color="k", lw=.8)
        ax_yield.set_xlabel(r"anneal fraction $s$")
        ax_yield.set_ylabel(
            r"degree-aligned minus zero offset $Y_{\mathcal{M}}(s)$")
        panel_label(ax_yield, "(c)")
        ax_yield.legend(fontsize=8)
        ax_yield.grid(alpha=.25)

        for (n, _, dynamics), color in zip(panels, palette):
            color = SIZE_COLOR.get(int(n), color)
            sgrid = np.asarray(dynamics["s_grid"])
            curve = dynamics["curves"]["dc_base_seff"]
            ax_support.plot(sgrid, curve["mean"], color=color,
                            label=rf"$N={n}$")
            ax_support.fill_between(sgrid, curve["lo"], curve["hi"],
                                    color=color, alpha=.12)
        ax_support.axhline(0, color="k", lw=.8)
        ax_support.grid(alpha=.25)
        ax_support.set_xlabel(r"anneal fraction $s$")
        ax_support.set_ylabel(r"conditional $\Delta S_{\mathrm{eff}}(s)$")
        panel_label(ax_support, "(d)")
        fig.savefig(os.path.join(cfg.outdir, "fig_mechanism_sensitivity.png"),
                    dpi=210, bbox_inches="tight")
        fig.savefig(os.path.join(cfg.outdir, "fig_mechanism_sensitivity.pdf"),
                    bbox_inches="tight")
        plt.close(fig)

    # (4) compact confirmatory figure for the Supplement
    if "size" in results:
        sz = results["size"]; Ns = sorted(sz.keys(), key=int)
        x = np.arange(len(Ns)); w = 0.34
        dc = [sz[n]["dc_gain_mean"] for n in Ns]
        dc_ci = [sz[n]["dc_gain_ci"] for n in Ns]
        aligned = [sz[n]["dc_minus_shuffle_mean"] for n in Ns]
        aligned_ci = [sz[n]["dc_minus_shuffle_ci"] for n in Ns]
        budget = [sz[n]["eub_budget_mean"] for n in Ns]
        budget_ci = [sz[n]["eub_budget_mean_ci"] for n in Ns]
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
        ax = axes[0]
        ax.bar(x-w/2, dc, w, yerr=ci_yerr(dc, dc_ci), capsize=3,
               color=BLUE, hatch="//", label="DC minus no offset")
        ax.bar(x+w/2, aligned, w, yerr=ci_yerr(aligned, aligned_ci), capsize=3,
               color=VERMILLION, hatch="\\\\", label="degree-aligned minus shuffled multiset")
        ax.axhline(0, color="k", lw=.8)
        ax.set_xticks(x); ax.set_xticklabels(
            [f"N={n}\n(n={sz[n]['included']}; wins={sz[n]['dc_gt_shuffle']})" for n in Ns])
        ax.set_ylabel("conditional effective-support contrast (95% CI)")
        ax.set_title("(a) Degree-aligned versus zero-offset and shuffled controls")
        ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=.25, axis="y")

        ax = axes[1]
        bars = ax.bar(x, budget, 0.52, yerr=ci_yerr(budget, budget_ci), capsize=3,
                      color=GREEN, hatch="..")
        for k, n in enumerate(Ns):
            p_adj = sz[n].get("eub_maxstat_p_bonferroni", sz[n]["eub_maxstat_p"])
            ax.text(bars[k].get_x()+bars[k].get_width()/2, 0.97,
                    f"max-|stat|\nBonf. p={p_adj:.3g}",
                    transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)
        ax.axhline(0, color="k", lw=.8)
        ax.set_xticks(x); ax.set_xticklabels([f"N={n}" for n in Ns])
        ax.set_ylabel(r"mean $\Delta\mathbb{E}[U_R]$ over fixed budget grid (95% CI)")
        ax.set_title("(b) Unconditional finite-budget discovery gain")
        ax.grid(alpha=.25, axis="y")
        fig.suptitle(rf"Idealized closed-system simulation at nominal $\alpha={cfg.alpha_main}$",
                     fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(os.path.join(cfg.outdir, "fig_confirmatory.png"), dpi=200)
        fig.savefig(os.path.join(cfg.outdir, "fig_confirmatory.pdf"))
        plt.close(fig)

    # (5) compact three-panel manuscript figure: conditional support, discovery curve,
    # and assignment controls. Schedule-shape robustness remains a textual/sensitivity
    # result so that the figure stays focused on the two fixed outcomes.
    if "size" in results and "controls" in results:
        sz = results["size"]; Ns = sorted(sz.keys(), key=int)
        x = np.arange(len(Ns)); w = 0.34
        dc = [sz[n]["dc_gain_mean"] for n in Ns]
        dc_ci = [sz[n]["dc_gain_ci"] for n in Ns]
        aligned = [sz[n]["dc_minus_shuffle_mean"] for n in Ns]
        aligned_ci = [sz[n]["dc_minus_shuffle_ci"] for n in Ns]
        fig = plt.figure(figsize=(11.6, 7.1))
        gs = fig.add_gridspec(2, 2, hspace=.40, wspace=.28)
        axes = [fig.add_subplot(gs[0, 0]),
                fig.add_subplot(gs[0, 1]),
                fig.add_subplot(gs[1, :])]

        ax = axes[0]
        ax.bar(x-w/2, dc, w, yerr=ci_yerr(dc, dc_ci), capsize=3,
               color=BLUE, hatch="//", label="degree-aligned minus zero offset")
        ax.bar(x+w/2, aligned, w, yerr=ci_yerr(aligned, aligned_ci), capsize=3,
               color=VERMILLION, hatch="\\\\", label="degree-aligned minus shuffled")
        ax.axhline(0, color="k", lw=.8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"N={n}\n{sz[n]['dc_gt_shuffle']}/{sz[n]['included']}" for n in Ns],
            fontsize=8)
        ax.set_ylabel(r"conditional $\Delta S_{\mathrm{eff}}$ (95% CI)"
                      "\n" r"(broader support $\uparrow$)")
        panel_label(ax, "(a)")
        ax.legend(fontsize=7.5, loc="upper left")
        ax.grid(alpha=.25, axis="y")

        ax = axes[1]
        for n in Ns:
            s = sz[n]
            line, = ax.plot(s["budget_grid"], s["mean_deub_curve"], marker="o", ms=2.8,
                            color=SIZE_COLOR.get(int(n)),
                            label=rf"$N={n}$")
            ax.fill_between(s["budget_grid"], s["deub_band_lo"], s["deub_band_hi"],
                            color=line.get_color(), alpha=.13)
        ax.axhline(0, color="k", lw=.8)
        ax.set_xscale("log")
        ax.set_xlabel(r"independent-draw budget $R$")
        ax.set_ylabel(r"unconditional $\Delta\mathbb{E}[U_R]$"
                      "\n" r"(more distinct optima $\uparrow$)")
        panel_label(ax, "(b)")
        ax.legend(fontsize=8)
        ax.grid(alpha=.25)

        ax = axes[2]
        order = ["dc", "homogeneous", "shuffle", "anti", "random"]
        labels = {"dc": "degree-\naligned", "homogeneous": "mean-matched\nuniform",
                  "shuffle": "shuffled\nmultiset", "anti": "degree-\nreversed",
                  "random": "random"}
        base_controls = results["controls"]
        control_panels = [(
            str(base_controls["_meta"]["N"]),
            base_controls,
            base_controls["_meta"]["included"],
        )]
        if "extended" in results:
            control_panels.extend(
                (n, results["extended"][n]["controls"],
                 results["extended"][n]["included"])
                for n in sorted(results["extended"], key=int)
            )
        # Cycled locally so the panel covers any number of size conditions.
        _cp_pal = [BLUE, VERMILLION, GREEN, "#CC79A7", "#333333"]
        _cp_mrk = ["o", "s", "D", "^", "v"]
        ctrl_palette = [_cp_pal[i % len(_cp_pal)] for i in range(len(control_panels))]
        ctrl_markers = [_cp_mrk[i % len(_cp_mrk)] for i in range(len(control_panels))]
        control_x = np.arange(len(order))
        offsets = np.linspace(-0.17, 0.17, len(control_panels))
        for offset, (n, controls, included), color, marker in zip(
                offsets, control_panels, ctrl_palette, ctrl_markers):
            means = [controls[s]["mean"] for s in order]
            ses = [controls[s]["se"] for s in order]
            ax.errorbar(control_x + offset, means, yerr=ses,
                        color=SIZE_COLOR.get(int(n), color),
                        marker=marker, ls="none", capsize=3,
                        label=rf"$N={n}$ ($n={included}$)")
        ax.axhline(0, color="k", lw=.8)
        ax.set_xticks(control_x)
        ax.set_xticklabels([labels[s] for s in order], fontsize=7.5)
        ax.set_ylabel(r"$\Delta S_{\mathrm{eff}}$ versus zero offset (s.e.m.)"
                      "\n" r"(broader support $\uparrow$)")
        panel_label(ax, "(c)")
        ax.legend(fontsize=7.5)
        ax.grid(alpha=.25, axis="y")

        fig.savefig(os.path.join(cfg.outdir, "fig_supplement_simulation.png"),
                    dpi=220, bbox_inches="tight")
        fig.savefig(os.path.join(cfg.outdir, "fig_supplement_simulation.pdf"),
                    bbox_inches="tight")
        plt.close(fig)

    # (6) full N=18--20 extension: every assignment control, schedule shape,
    # alpha value, and time-resolved endpoint contrast on the same fixed pools.
    if "extended" in results:
        ext = results["extended"]
        Ns = sorted(ext.keys(), key=int)
        _bp = [BLUE, VERMILLION, GREEN, "#CC79A7", "#333333"]
        _bm = ["o", "s", "D", "^", "v"]
        colors = [_bp[i % len(_bp)] for i in range(len(Ns))]
        markers = [_bm[i % len(_bm)] for i in range(len(Ns))]
        fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.4))

        ax = axes[0, 0]
        schemes = ["dc", "homogeneous", "shuffle", "anti", "random"]
        labels = ["degree-\naligned", "mean-matched\nuniform", "shuffled\nmultiset",
                  "degree-\nreversed", "random"]
        x = np.arange(len(schemes))
        offsets = np.linspace(-0.15, 0.15, len(Ns))
        for offset, n, color, marker in zip(offsets, Ns, colors, markers):
            controls = ext[n]["controls"]
            means = [controls[s]["mean"] for s in schemes]
            cis = [controls[s]["ci"] for s in schemes]
            ax.errorbar(x + offset, means, yerr=ci_yerr(means, cis),
                        color=color, marker=marker, ls="none", capsize=3,
                        label=rf"$N={n}$ ($n={ext[n]['included']}$)")
        ax.axhline(0, color="k", lw=.8)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(r"$\Delta S_{\mathrm{eff}}$ versus zero offset (95% CI)")
        ax.set_title("(a) Offset-assignment controls")
        ax.legend(fontsize=8); ax.grid(alpha=.25, axis="y")

        ax = axes[0, 1]
        schedules = list(cfg.schedules)
        x = np.arange(len(schedules))
        display = {"linear": "linear", "convex_power": "convex\npower",
                   "trig": "trigonometric"}
        offsets = np.linspace(-0.06, 0.06, len(Ns))
        for offset, n, color, marker in zip(offsets, Ns, colors, markers):
            schedule_result = ext[n]["schedule"]
            means = [schedule_result[s]["dc_minus_shuffle_mean"] for s in schedules]
            cis = [schedule_result[s]["dc_minus_shuffle_ci"] for s in schedules]
            ax.errorbar(x + offset, means, yerr=ci_yerr(means, cis), color=color,
                        marker=marker, ls="none", capsize=3, label=rf"$N={n}$")
        ax.axhline(0, color="k", lw=.8)
        ax.set_xticks(x); ax.set_xticklabels([display.get(s, s) for s in schedules])
        ax.set_ylabel(r"$S_{\mathrm{DC}}-S_{\mathrm{shuffle}}$ (95% CI)")
        ax.set_title("(b) Schedule-shape robustness")
        ax.legend(fontsize=8); ax.grid(alpha=.25)

        ax = axes[1, 0]
        alpha_values = np.asarray(cfg.alpha_grid, dtype=float)
        for n, color, marker in zip(Ns, colors, markers):
            alpha_result = ext[n]["alpha"]
            means = [alpha_result[str(float(a))]["dc_gain_mean"] for a in alpha_values]
            cis = [alpha_result[str(float(a))]["dc_gain_ci"] for a in alpha_values]
            ax.errorbar(alpha_values, means, yerr=ci_yerr(means, cis),
                        color=color, marker=marker, capsize=3, label=rf"$N={n}$")
        ax.axhline(0, color="k", lw=.8)
        ax.set_xlabel(r"offset scale $\alpha$")
        ax.set_ylabel(r"conditional $\Delta S_{\mathrm{eff}}$ (95% CI)")
        ax.set_title("(c) Fixed offset-strength sensitivity")
        ax.legend(fontsize=8); ax.grid(alpha=.25)

        ax = axes[1, 1]
        for n, color in zip(Ns, colors):
            dynamics = ext[n]["dynamics"]
            sgrid = np.asarray(dynamics["s_grid"], dtype=float)
            curve = dynamics["curves"]["dc_base_seff"]
            ax.plot(sgrid, curve["mean"], color=color, label=rf"$N={n}$")
            ax.fill_between(sgrid, curve["lo"], curve["hi"], color=color, alpha=.12)
        ax.axhline(0, color="k", lw=.8)
        ax.set_xlabel(r"anneal fraction $s$")
        ax.set_ylabel(r"conditional $\Delta S_{\mathrm{eff}}(s)$")
        ax.set_title("(d) Time-resolved support redistribution")
        ax.legend(fontsize=8); ax.grid(alpha=.25)

        fig.suptitle("Full large-size sensitivity suite in the idealized model",
                     fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, .96))
        fig.savefig(os.path.join(cfg.outdir, "fig_extended_full_controls.png"),
                    dpi=210)
        fig.savefig(os.path.join(cfg.outdir, "fig_extended_full_controls.pdf"))
        plt.close(fig)


# ======================================================================================
# IO helpers
# ======================================================================================
def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items() if k != "_raw"}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def write_outputs(results, cfg: Config):
    os.makedirs(cfg.outdir, exist_ok=True)
    with open(os.path.join(cfg.outdir, "results.json"), "w") as f:
        json.dump({"config": asdict(cfg), "results": _json_safe(results)}, f, indent=2)
    # paired stats CSV (size stage)
    if "size" in results:
        import csv
        with open(os.path.join(cfg.outdir, "paired_stats.csv"), "w", newline="") as f:
            wtr = csv.writer(f)
            wtr.writerow(["N", "seed", "degeneracy", "seff_base", "seff_dc",
                          "dc_gain", "dc_relative_gain", "dc_log_ratio",
                          "shuffle_gain", "dc_minus_shuffle", "dc_minus_shuffle_relative",
                          "ent_base", "ent_dc",
                          "yld_base", "yld_dc", "eub_budget_mean",
                          "eub_budget_fraction_mean",
                          "eub_peak_gain_descriptive", "eub_peak_B_descriptive"])
            for N, summ in results["size"].items():
                for r in summ["rows"]:
                    wtr.writerow([N, r["seed"], r["degen"], f"{r['seff_base']:.4f}",
                                  f"{r['seff_dc']:.4f}", f"{r['dc_gain']:.4f}",
                                  f"{r['dc_relative_gain']:.6f}", f"{r['dc_log_ratio']:.6f}",
                                  f"{r['shuffle_gain']:.4f}", f"{r['dc_minus_shuffle']:.4f}",
                                  f"{r['dc_minus_shuffle_relative']:.6f}",
                                  f"{r['ent_base']:.4f}", f"{r['ent_dc']:.4f}", f"{r['yld_base']:.4f}",
                                  f"{r['yld_dc']:.4f}", f"{r['eub_budget_mean']:.4f}",
                                  f"{r['eub_budget_fraction_mean']:.6f}",
                                  f"{r['eub_peak_gain']:.4f}", r["eub_peak_B"]])
    if "alpha" in results:
        import csv
        aa = results["alpha"]
        keys = sorted((k for k in aa if not k.startswith("_")), key=float)
        fields = ["dc_gain", "yield_gain", "eub_budget_mean"]
        with open(os.path.join(cfg.outdir, "alpha_sensitivity_summary.csv"), "w", newline="") as f:
            wtr = csv.writer(f)
            wtr.writerow(["alpha"] + [f"{m}_{x}" for m in fields for x in ("mean", "ci_lo", "ci_hi")])
            for k in keys:
                row = [aa[k]["alpha"]]
                for m in fields:
                    row.extend([aa[k][f"{m}_mean"], *aa[k][f"{m}_ci"]])
                wtr.writerow(row)
    if "dynamics" in results:
        import csv
        dy = results["dynamics"]
        curve_names = ["base_yield", "dc_yield", "base_seff", "dc_seff",
                       "dc_base_yield", "dc_base_seff"]
        with open(os.path.join(cfg.outdir, "time_resolved_summary.csv"), "w", newline="") as f:
            wtr = csv.writer(f)
            wtr.writerow(["s"] + [f"{name}_{x}" for name in curve_names
                                    for x in ("mean", "ci_lo", "ci_hi")])
            for j, sval in enumerate(dy["s_grid"]):
                row = [sval]
                for name in curve_names:
                    c = dy["curves"][name]
                    row.extend([c["mean"][j], c["lo"][j], c["hi"][j]])
                wtr.writerow(row)
    if "extended" in results:
        import csv
        ext = results["extended"]
        with open(os.path.join(cfg.outdir, "extended_controls_summary.csv"),
                  "w", newline="") as f:
            fields = [
                "N", "included", "scheme", "mean", "ci_lo", "ci_hi", "se",
                "dc_minus_control_mean", "dc_minus_control_ci_lo",
                "dc_minus_control_ci_hi", "dc_gt_control",
            ]
            wtr = csv.DictWriter(f, fieldnames=fields)
            wtr.writeheader()
            for N, result in ext.items():
                for scheme in ("dc", "homogeneous", "shuffle", "anti", "random"):
                    entry = result["controls"][scheme]
                    wtr.writerow({
                        "N": N, "included": result["included"], "scheme": scheme,
                        "mean": entry["mean"], "ci_lo": entry["ci"][0],
                        "ci_hi": entry["ci"][1], "se": entry["se"],
                        "dc_minus_control_mean": entry.get("dc_minus_control_mean", ""),
                        "dc_minus_control_ci_lo":
                            entry.get("dc_minus_control_ci", ["", ""])[0],
                        "dc_minus_control_ci_hi":
                            entry.get("dc_minus_control_ci", ["", ""])[1],
                        "dc_gt_control": entry.get("dc_gt_control", ""),
                    })
        with open(os.path.join(cfg.outdir, "extended_schedule_summary.csv"),
                  "w", newline="") as f:
            fields = [
                "N", "included", "schedule", "dc_gain_mean", "dc_gain_ci_lo",
                "dc_gain_ci_hi", "shuffle_gain_mean", "shuffle_gain_ci_lo",
                "shuffle_gain_ci_hi", "dc_minus_shuffle_mean",
                "dc_minus_shuffle_ci_lo", "dc_minus_shuffle_ci_hi",
                "dc_gt_shuffle",
            ]
            wtr = csv.DictWriter(f, fieldnames=fields)
            wtr.writeheader()
            for N, result in ext.items():
                for sched_name in cfg.schedules:
                    entry = result["schedule"][sched_name]
                    wtr.writerow({
                        "N": N, "included": result["included"],
                        "schedule": sched_name,
                        "dc_gain_mean": entry["dc_gain_mean"],
                        "dc_gain_ci_lo": entry["dc_gain_ci"][0],
                        "dc_gain_ci_hi": entry["dc_gain_ci"][1],
                        "shuffle_gain_mean": entry["shuffle_gain_mean"],
                        "shuffle_gain_ci_lo": entry["shuffle_gain_ci"][0],
                        "shuffle_gain_ci_hi": entry["shuffle_gain_ci"][1],
                        "dc_minus_shuffle_mean": entry["dc_minus_shuffle_mean"],
                        "dc_minus_shuffle_ci_lo":
                            entry["dc_minus_shuffle_ci"][0],
                        "dc_minus_shuffle_ci_hi":
                            entry["dc_minus_shuffle_ci"][1],
                        "dc_gt_shuffle": entry["dc_gt_shuffle"],
                    })
        with open(os.path.join(cfg.outdir, "extended_alpha_summary.csv"),
                  "w", newline="") as f:
            fields = [
                "N", "included", "alpha", "dc_gain_mean", "dc_gain_ci_lo",
                "dc_gain_ci_hi", "yield_gain_mean", "yield_gain_ci_lo",
                "yield_gain_ci_hi", "eub_budget_mean", "eub_budget_ci_lo",
                "eub_budget_ci_hi",
            ]
            wtr = csv.DictWriter(f, fieldnames=fields)
            wtr.writeheader()
            for N, result in ext.items():
                for alpha in cfg.alpha_grid:
                    entry = result["alpha"][str(float(alpha))]
                    wtr.writerow({
                        "N": N, "included": result["included"], "alpha": alpha,
                        "dc_gain_mean": entry["dc_gain_mean"],
                        "dc_gain_ci_lo": entry["dc_gain_ci"][0],
                        "dc_gain_ci_hi": entry["dc_gain_ci"][1],
                        "yield_gain_mean": entry["yield_gain_mean"],
                        "yield_gain_ci_lo": entry["yield_gain_ci"][0],
                        "yield_gain_ci_hi": entry["yield_gain_ci"][1],
                        "eub_budget_mean": entry["eub_budget_mean_mean"],
                        "eub_budget_ci_lo": entry["eub_budget_mean_ci"][0],
                        "eub_budget_ci_hi": entry["eub_budget_mean_ci"][1],
                    })
        with open(os.path.join(cfg.outdir, "extended_dynamics_summary.csv"),
                  "w", newline="") as f:
            curve_names = [
                "base_yield", "dc_yield", "base_seff", "dc_seff",
                "dc_base_yield", "dc_base_seff",
            ]
            fields = ["N", "included", "s"] + [
                f"{name}_{suffix}" for name in curve_names
                for suffix in ("mean", "ci_lo", "ci_hi")
            ]
            wtr = csv.DictWriter(f, fieldnames=fields)
            wtr.writeheader()
            for N, result in ext.items():
                dynamics = result["dynamics"]
                for j, sval in enumerate(dynamics["s_grid"]):
                    row = {"N": N, "included": result["included"], "s": sval}
                    for name in curve_names:
                        curve = dynamics["curves"][name]
                        row[f"{name}_mean"] = curve["mean"][j]
                        row[f"{name}_ci_lo"] = curve["lo"][j]
                        row[f"{name}_ci_hi"] = curve["hi"][j]
                    wtr.writerow(row)
    print(f"\nWrote machine-readable results to '{cfg.outdir}/'")


# ======================================================================================
# Entry point
# ======================================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", default="all",
                    choices=["all", "verify", "size", "controls", "schedule",
                             "alpha", "dynamics", "extended", "plot"])
    ap.add_argument("--outdir", default=CONFIG.outdir)
    ap.add_argument("--sizes", type=int, nargs="+", default=list(CONFIG.sizes))
    ap.add_argument("--seed-pool", type=int, default=CONFIG.seed_pool)
    ap.add_argument("--min-degeneracy", type=int, default=CONFIG.min_degeneracy)
    ap.add_argument("--max-instances", type=int, default=CONFIG.max_instances)
    ap.add_argument("--n-steps", type=int, default=CONFIG.n_steps)
    ap.add_argument("--alpha", type=float, default=CONFIG.alpha_main)
    ap.add_argument("--shuffle-reps", type=int, default=CONFIG.shuffle_reps)
    ap.add_argument("--workers", type=int, default=CONFIG.workers,
                    help="instance-level worker processes for N >= --parallel-min-N")
    ap.add_argument("--parallel-min-N", type=int, default=CONFIG.parallel_min_N)
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore completed per-instance size-stage cache records")
    args = ap.parse_args(argv)

    cfg = Config(**{**asdict(CONFIG),
                    "outdir": args.outdir, "sizes": tuple(args.sizes),
                    "seed_pool": args.seed_pool, "min_degeneracy": args.min_degeneracy,
                    "max_instances": args.max_instances, "n_steps": args.n_steps,
                    "alpha_main": args.alpha, "shuffle_reps": args.shuffle_reps,
                    "workers": args.workers, "parallel_min_N": args.parallel_min_N,
                    "resume": not args.no_resume})
    os.makedirs(cfg.outdir, exist_ok=True)

    if args.stage == "plot":
        result_path = os.path.join(cfg.outdir, "results.json")
        with open(result_path) as f:
            stored = json.load(f)
        valid_fields = Config.__dataclass_fields__
        stored_values = {k: v for k, v in stored["config"].items() if k in valid_fields}
        stored_cfg = Config(**{**stored_values, "outdir": args.outdir})
        plot_all(stored["results"], stored_cfg)
        print(f"Regenerated figures from '{result_path}'")
        return

    results = {}
    if args.stage in ("all", "verify"):
        results["verify"] = stage_verify(cfg)
    if args.stage in ("all", "size"):
        results["size"] = stage_size(cfg)
    if args.stage in ("all", "controls"):
        results["controls"] = stage_controls(cfg, N=min(cfg.sizes))
    if args.stage in ("all", "schedule"):
        results["schedule"] = stage_schedule(cfg, N=min(cfg.sizes))
    if args.stage in ("all", "alpha"):
        results["alpha"] = stage_alpha(cfg, N=min(cfg.sizes))
    if args.stage in ("all", "dynamics"):
        results["dynamics"] = stage_dynamics(cfg, N=min(cfg.sizes))
    if args.stage == "extended":
        results["extended"] = stage_extended(cfg)
        # Preserve the completed size-stage results in the same output package.
        result_path = os.path.join(cfg.outdir, "results.json")
        if os.path.isfile(result_path):
            with open(result_path) as f:
                stored = json.load(f)
            previous = stored.get("results", {})
            if "size" in previous:
                results["extended"] = synchronize_extended_primary_statistics(
                    results["extended"], previous["size"], cfg)
                results = {"size": previous["size"], **results}
            else:
                raise RuntimeError(
                    "Extended post-processing requires size results in results.json")
    if results:
        write_outputs(results, cfg)
        plot_all(results, cfg)
        print(f"Wrote available figures to '{cfg.outdir}/'")


if __name__ == "__main__":
    main()
