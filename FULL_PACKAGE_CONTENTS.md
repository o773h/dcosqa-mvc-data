# Data and analysis code for *Degree-conditioned anneal offsets reshape certified-optimum sampling for minimum vertex cover*

This package contains the recorded data, metadata, analysis code, and
manuscript figures used in the accompanying study.

Main Figure 1 is a workflow schematic and is not generated from this package.

## Layout

```text
data/
    raw/         Recorded QPU and simulated-annealing candidate tables,
                 one directory per experiment.
    metadata/    Graph, protocol, exact-solution, budget, source-mapping, and
                 canonical logical-to-physical embedding metadata.

code/
    <result>/    Analysis scripts and their inputs, one directory per manuscript result.
    reproduce_all_figures.py   Top-level analysis and figure runner.

figures/
    main/        Main-text figures.
    supplement/  Supplementary figures.

README.md · DATA_DICTIONARY.md · DATA_PROVENANCE.md
CITATION.cff · LICENSE · requirements.txt
MANIFEST.csv · SHA256SUMS.txt
```

Raw candidate tables store one row per distinct returned bitstring within a
recorded condition; `Occurrences` gives the multiplicity of that bitstring in
the read stream. Some tables are per-read solution dumps and some carry
additional certification and graph fields; column definitions are in
`DATA_DICTIONARY.md`.

`DATA_PROVENANCE.md` maps each reported result to its input files and filters.
This crosswalk is useful because several source exports overlap in name and
content.

## Reproduction

Use Python 3.10 or later:

```bash
python3 -m pip install -r requirements.txt
python3 code/reproduce_all_figures.py
```

The runner executes the analysis scripts and copies generated PDF and PNG
figures to `figures/main/` and `figures/supplement/`.

The canonical BA `m=2` QA-Base and DC-OSQA conditions used one shared fixed
embedding per graph instance. The 170 mappings are released as a compact
vertex-level table and inventory under `data/metadata/canonical_embeddings/`.
`code/validate_canonical_embeddings.py` verifies graph degrees, physical-chain
integrity, mapping hashes, complete coverage, and agreement with the recorded
chain-length diagnostics.

All degree-resolved BA `m=2` analyses use the same fixed actual-degree groups:
low (`d <= 2`), middle (`3 <= d <= 4`), and high (`d >= 5`). The definition is
tied to the graph model rather than to `N`: degree two is the minimum and modal
class, and the mean degree approaches four. The ranges are not recomputed from
per-graph quantiles. Across `N=70`--`150`, they contain 47.4--50.0%,
28.5--31.1%, and 21.1--22.4% of vertices, respectively. The same definition is
used in the `N=150` matched-shuffle diagnostic. `requirements-lock.txt` records
the reference validation environment.

Main Figure 5 recomputes cumulative-prefix support and complete-linkage Hamming
families for 240 instance--protocol--budget cases. This is the longest step and
creates resumable files under
`code/main_qpu_read_budget_extension_n150/derived/`.

The BA m=3 and m=4 offset-scale analysis (Figure S7) uses the released verified
candidate stream at
`code/supp_bam34_alpha_controls/derived/verified_candidate_stream.csv.gz`.

For the random 3-regular control (Figure S9), the runner first rebuilds the
verified candidate stream from the released raw table and graph references
(`code/supp_schedule_regular3_controls/reference/`): it reconstructs each graph
from its archived seed, recomputes `K_exact` with HiGHS, and sets
`strict_exact_verified` before generating the figure.

For the idealized closed-system calculation (Figures S12 and S13), the top-level
runner replots the released verified summaries. To recompute the state-vector
evolution, bootstrap intervals, permutation tests, and figures from the fixed
graph pools, run the size stage followed by the extended controls:

```bash
python3 code/supp_idealized_closed_system/scripts/01_run_idealized_closed_system.py all \
  --sizes 12 14 16 18 20 --workers 20 \
  --outdir code/supp_idealized_closed_system/derived
python3 code/supp_idealized_closed_system/scripts/01_run_idealized_closed_system.py extended \
  --sizes 14 16 18 20 --workers 20 \
  --outdir code/supp_idealized_closed_system/derived
python3 code/supp_idealized_closed_system/scripts/00_replot_verified_results.py
```

The second command adds the assignment, schedule-shape, offset-strength, and
time-resolved diagnostics at \(N=14,16,18,\) and 20 to the corresponding
\(N=12\) diagnostics produced by the first command. The calculation uses only generated
BA graphs and stores no QPU data or intermediate state vectors. Per-instance
cache files make the \(N=18\)--20 stages resumable and may be deleted after the
verified summaries have been written.

## Analysis-to-figure mapping

Each directory below lives under `code/`.

| Analysis directory | Manuscript output |
|---|---|
| `main_qabase_vs_dcosqa` | Main Figures 2 and 3 |
| `supp_shuffle20_control` | Main Figure 4 |
| `main_qpu_read_budget_extension_n150` | Main Figure 5 |
| `supp_sa_dc_complementarity_n150` | Supplementary Figure S1 |
| `supp_dc_vs_bc_paired_gain` | Supplementary Figure S2 |
| `supp_offset_scale_sensitivity` | Supplementary Figure S3 |
| `supp_embedding_seed_sensitivity` | Supplementary Figures S4 and S5 |
| `supp_shuffle20_degree_inclusion_alignment` | Supplementary Figure S6 |
| `supp_bam34_alpha_controls` | Supplementary Figure S7 |
| `supp_degree_bin_vertex_inclusion_shift` | Supplementary Figure S8 |
| `supp_schedule_regular3_controls` | Supplementary Figure S9 |
| `supp_er_n150_qabase_vs_dcosqa` | Supplementary Figure S10 |
| `supp_yield_diversity_robustness` | Supplementary Figure S11 |
| `supp_idealized_closed_system` | Supplementary Figures S12 and S13 |
| `supp_n200_qabase_extension` | Supplementary N=200 statistics |
| `supp_sa_recovery_support_decomposition` | Supplementary SA recovery and overlap statistics |
| `supp_budget_diagnostic_qabase_dc` | Supplementary budget and chain-break statistics |

## Integrity and licensing

`MANIFEST.csv` records each released file, size, SHA-256 digest, and role.
`SHA256SUMS.txt` can be checked with:

```bash
shasum -a 256 -c SHA256SUMS.txt
```

Data are released under CC BY 4.0 and Python code under the MIT license. See
`LICENSE` and `CITATION.cff`.
