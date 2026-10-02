# ER N=150 QA-Base versus DC-OSQA control

This analysis contains ten connected Erdos-Renyi graph instances at `N=150`.
The graph-generation probability is `p=4/(N-1)`, which targets an
unconditional mean degree of four. Disconnected draws were rejected using
deterministic seed increments. The accepted graphs contain 297--346 edges
(mean 317.9), whereas each NetworkX Barabasi-Albert `m=2`, `N=150` graph
contains 296 edges. This is therefore a nominal-mean-degree topology-transfer
control rather than an exact realized-density match.

The analysis script reconstructs every graph from its released seed, validates
the recorded edge count, recomputes vertex-cover feasibility from every raw
bitstring, applies the released graph-level `K_exact`, and recomputes unique
certified candidates and complete-linkage Hamming families at `tau=0.1N=15`.

Run from the package root:

```bash
python3 code/supp_er_n150_qabase_vs_dcosqa/scripts/01_make_er_n150_qabase_vs_dcosqa.py
```

This ten-instance experiment is a topology-transfer control.
