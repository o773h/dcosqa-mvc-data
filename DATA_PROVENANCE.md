# Data provenance and file-to-figure map

This document identifies the input files and filters used for each reported
result. Several source exports overlap, so the tables below distinguish the
canonical input for each analysis.

The top-level runner, `python3 code/reproduce_all_figures.py`, applies these
selections automatically. The details below support manual verification.

## Source-selection notes

1. **The canonical DC-OSQA file for the primary comparison mixes two solvers.**
   `data/raw/main_ba_m2_dcosqa_offset01/raw_candidate_bitstrings.csv` contains
   **both** `DC-Offset(s=0.1)` and `BC-Offset(s=0.1)` rows. The primary
   DC-OSQA vs QA-Base comparison uses **only** rows with
   `Solver == 'DC-Offset(s=0.1)'`. Without this filter the DC counts are
   inflated (e.g. mean unique candidates 691.7 instead of 455.8).

2. **There is more than one DC-OSQA export, and they are not equally complete.**
   The primary comparison (Fig. 2/3) uses
   `data/raw/main_ba_m2_dcosqa_offset01/…` which has the **full 20 instances
   per size** for N = 70–150 (120 protocol–instance pairs). A second export,
   `data/raw/qa_inhomo_all/…` (used only for the DC-vs-BC comparison,
   Fig. S2), is a **partial export**: it contains only 10 instances at N = 80
   and N = 100, i.e. 100 pairs. The two are the **same DC-OSQA runs** — on every
   shared instance the certified-candidate counts are identical — so the
   difference is coverage, not a different experiment. Using the partial export
   for the primary comparison gives 471.9 over 100 pairs instead of the
   published 455.8 over 120 pairs.

3. **File names do not always match their role.**
   `data/metadata/ba_m2_n70_150_concentration_analysis_input.csv` is the input
   for the **DC-vs-BC** figure (Fig. S2), not the primary concentration figure
   (Fig. 3). The primary concentration numbers are computed from the same
   certified bitstrings as Fig. 2 (`main_ba_m2_dcosqa_offset01`, filtered to
   `DC-Offset(s=0.1)`).

## Canonical vs. auxiliary datasets

| Dataset | Role | Coverage | Use for |
|---|---|---|---|
| `data/raw/main_ba_m2_dcosqa_offset01/raw_candidate_bitstrings.csv` | **Canonical DC-OSQA** (and BC) certified candidates | 20 inst/size, N=70–150 | Fig. 2, Fig. 3, Fig. S8, Fig. S11 |
| `data/raw/qa_base_main_n20_150/raw_solutions.csv` | **Canonical QA-Base** reads | 20 inst/size, N=20–150 | Fig. 2, Fig. 3, Fig. S8 |
| `data/raw/qa_inhomo_all/raw_solutions.csv` | Auxiliary DC/BC export (partial) | 10 inst at N=80,100 | Fig. S2 (DC-vs-BC) only |
| `data/metadata/ba_m2_n70_150_concentration_analysis_input.csv` | Prebuilt DC-vs-BC input | 100 pairs | Fig. S2 (DC-vs-BC) only |
| `data/raw/qa_base_ba_main/raw_solutions.csv` | Auxiliary QA-Base export | — | Fig. S2 QA rows only |

## Canonical physical embeddings

`data/metadata/canonical_embeddings/ba_m2_vertex_chain_mappings.csv.gz`
contains the 170 logical-to-physical mappings used by the canonical BA `m=2`
comparison: 10 instances per size for `N=20`--`60` and 20 instances per size
for `N=70`--`150`. Within each graph instance, QA-Base and DC-OSQA used the
same fixed mapping and differed only in the anneal-offset vector. The companion
`ba_m2_embedding_inventory.csv` records graph and embedding seeds, chain-length
summaries, and a normalized mapping hash.

`code/validate_canonical_embeddings.py` checks all 170 mappings against the
deterministically reconstructed BA graphs and against the aggregate chain-length
diagnostics retained in the canonical DC-OSQA candidate table.

The primary comparison uses the 170 canonical mappings described above. The
three auxiliary datasets in the preceding table are not part of that comparison.

## Figure → input crosswalk

For any figure, the authoritative input path is the `*_RAW` / `SRC` / `RAW`
variable at the top of its script under `code/<dir>/scripts/`.
The main figures and supplementary controls are listed here.

| Figure | Script | Canonical input(s) | Filter |
|---|---|---|---|
| Fig. 2 (support) | `main_qabase_vs_dcosqa/01_…` | `main_ba_m2_dcosqa_offset01` (DC) + `qa_base_main_n20_150` (QA) | DC: `Solver=='DC-Offset(s=0.1)'` |
| Fig. 3 (concentration) | `main_qabase_vs_dcosqa/02_…` | same certified bitstrings as Fig. 2 | over certified occurrences |
| Fig. 4 (shuffle) | `supp_shuffle20_control/01_…` | `supp_shuffle20_control/raw/` | — |
| Fig. 5 (50k budget) | `main_qpu_read_budget_extension_n150/…` | raw via `config/canonical_raw_source_selection.csv` → `derived/verified_full_candidate_stream_from_raw.csv.gz` | strict_exact_verified |
| Fig. S1 (SA overlap) | `supp_sa_dc_complementarity_n150/…` | DC stream + `data/metadata/sa_recovery_n150_*`; matched-budget summary retained at `code/supp_sa_dc_complementarity_n150/curated/sa_dc_same_budget_summary.csv` | panel (a): canonical DC 1k recovery; panel (b): matched output budgets |
| Fig. S2 (DC vs BC) | `supp_dc_vs_bc_paired_gain/01_…` | `data/metadata/ba_m2_n70_150_concentration_analysis_input.csv` (partial, 100 pairs) | DC/BC = `*-Offset(s=0.1)` |
| Fig. S5 (chain length) | `supp_embedding_seed_sensitivity/02_…` | canonical embedding table + the Fig. 2 QA-Base and DC-OSQA raw inputs | `N=70`--`150`; conditional on certified occurrences in both protocols |
| Fig. S7 (BA m=3,4) | `supp_bam34_alpha_controls/01_…` | `supp_bam34_alpha_controls/derived/verified_candidate_stream.csv.gz` + `reference/…kexact.csv` | — |
| Fig. S8 (degree bins) | `supp_degree_bin_vertex_inclusion_shift/01_…` | `main_qabase_vs_dcosqa/derived/fig1_fig2_certified_bitstrings_recomputed_from_raw.csv` (regenerated by Fig. 2 script) | — |
| Fig. S9 (3-regular) | `supp_schedule_regular3_controls/01_…` | `supp_schedule_regular3_controls/raw/` + graph metadata | — |
| Fig. S11 (yield–diversity) | `supp_yield_diversity_robustness/01_…` | Fig. 2 certified CSV + qpu `verified_full_candidate_stream…gz` | — |
| Figs. S12--S13 (idealized QA) | `supp_idealized_closed_system/01_…` | fixed generated BA seed pools for \(N=12,14,16,18,20\); verified summaries in `derived/results.json` | retain every graph with at least four exact minimum covers; full assignment, schedule, offset-strength, and time-resolved controls at all five sizes |

All certified counts use the same rule everywhere: a read is a **certified
candidate** only if it is a feasible vertex cover **and** its cardinality equals
the independently ILP-certified minimum `K_exact`.

## Simulated-annealing acquisition metadata

The released SA candidate tables and analysis scripts reproduce the reported
recovery and set-overlap summaries. Run-level metadata needed to regenerate the
original SA trajectories---the software version, random seeds, initialization,
and update schedule---were not retained; the SA component is therefore released
as an output-level recovery analysis rather than a trajectory-level rerun.

## Manual check of the headline statistic

Mean unique certified candidates for DC-OSQA over N = 70–150 (published 455.783):

```python
import pandas as pd
dc = pd.read_csv('data/raw/main_ba_m2_dcosqa_offset01/raw_candidate_bitstrings.csv',
                 low_memory=False)
dc = dc[dc['Solver'] == 'DC-Offset(s=0.1)']                 # <- required filter
dc = dc[(dc['Nodes'] >= 70) & (dc['Nodes'] <= 150)]         # already certified
u = dc.groupby(['Nodes', 'Instance'])['Bitstring'].nunique()
# fill the full 6 sizes x 20 instances grid with zeros for zero-yield pairs
idx = [(n, i) for n in (70, 80, 90, 100, 120, 150) for i in range(20)]
print(u.reindex(idx, fill_value=0).mean())                  # -> 455.783
```

The matching QA-Base value (171.700) is obtained from
`data/raw/qa_base_main_n20_150/raw_solutions.csv`, keeping reads with
`IsValid` true and `Size == K_exact` (join `K_exact` from the DC file above),
counting distinct `Bitstring` per (Nodes, Instance), and averaging over the same
120-pair grid.
