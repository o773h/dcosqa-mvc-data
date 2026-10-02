#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ANALYSIS_SCRIPTS = [
    "code/validate_canonical_embeddings.py",
    "code/main_qabase_vs_dcosqa/scripts/01_make_main_fig2_qabase_vs_dcosqa.py",
    "code/main_qabase_vs_dcosqa/scripts/02_make_main_fig3_sampling_concentration.py",
    "code/main_qpu_read_budget_extension_n150/scripts/01_build_verified_candidate_stream.py",
    "code/main_qpu_read_budget_extension_n150/scripts/02_build_strict_exact_family_curve_from_raw.py",
    "code/main_qpu_read_budget_extension_n150/scripts/02_make_main_fig5_qpu_read_budget_extension.py",
    "code/supp_sa_dc_complementarity_n150/scripts/01_make_sa_dc_complementarity_n150.py",
    "code/supp_sa_recovery_support_decomposition/scripts/01_make_sa_recovery_support_decomposition.py",
    "code/supp_degree_bin_vertex_inclusion_shift/scripts/01_make_degree_bin_vertex_inclusion_shift.py",
    "code/supp_embedding_seed_sensitivity/scripts/01_make_embedding_seed_sensitivity.py",
    "code/supp_embedding_seed_sensitivity/scripts/02_make_vertex_chain_confound_diagnostics.py",
    "code/supp_schedule_regular3_controls/scripts/00_build_verified_candidate_stream.py",
    "code/supp_schedule_regular3_controls/scripts/01_make_schedule_regular3_controls.py",
    "code/supp_shuffle20_control/scripts/01_make_shuffle20_control.py",
    "code/supp_shuffle20_degree_inclusion_alignment/scripts/01_make_shuffle20_degree_inclusion_alignment.py",
    "code/supp_dc_vs_bc_paired_gain/scripts/01_make_dc_vs_bc_paired_gain.py",
    "code/supp_offset_scale_sensitivity/scripts/01_make_offset_scale_sensitivity.py",
    "code/supp_bam34_alpha_controls/scripts/01_make_bam34_alpha_controls.py",
    "code/supp_n200_qabase_extension/scripts/01_make_n200_qabase_extension.py",
    "code/supp_budget_diagnostic_qabase_dc/scripts/01_make_budget_diagnostic.py",
    "code/supp_budget_diagnostic_qabase_dc/scripts/02_make_chain_break_diagnostics.py",
    "code/supp_yield_diversity_robustness/scripts/01_make_yield_diversity_robustness.py",
    "code/supp_er_n150_qabase_vs_dcosqa/scripts/01_make_er_n150_qabase_vs_dcosqa.py",
    "code/supp_idealized_closed_system/scripts/00_replot_verified_results.py",
]

PUBLICATION_PAIRS = [
    (
        "code/main_qabase_vs_dcosqa/curated/Fig2_qabase_vs_dcosqa_n20_150",
        "figures/main/Fig2_qabase_vs_dcosqa_n20_150",
    ),
    (
        "code/main_qabase_vs_dcosqa/curated/Fig3_sampling_concentration_paired",
        "figures/main/Fig3_sampling_concentration_paired",
    ),
    (
        "code/supp_shuffle20_control/curated/shuffle20_control",
        "figures/main/Fig4_topology_assignment_control_n150",
    ),
    (
        "code/main_qpu_read_budget_extension_n150/figures/Fig5_qpu_read_budget_extension_n150",
        "figures/main/Fig5_qpu_read_budget_extension_n150",
    ),
    (
        "code/supp_sa_dc_complementarity_n150/curated/sa_dc_complementarity_n150",
        "figures/supplement/sa_dc_complementarity_n150",
    ),
    (
        "code/supp_dc_vs_bc_paired_gain/curated/dc_vs_bc_paired_gain",
        "figures/supplement/dc_vs_bc_paired_gain",
    ),
    (
        "code/supp_offset_scale_sensitivity/curated/offset_scale_sensitivity",
        "figures/supplement/offset_scale_sensitivity",
    ),
    (
        "code/supp_embedding_seed_sensitivity/curated/embedding_seed_sensitivity",
        "figures/supplement/embedding_seed_sensitivity",
    ),
    (
        "code/supp_embedding_seed_sensitivity/curated/vertex_chain_confound_diagnostics",
        "figures/supplement/vertex_chain_confound_diagnostics",
    ),
    (
        "code/supp_shuffle20_degree_inclusion_alignment/curated/shuffle20_degree_inclusion_alignment",
        "figures/supplement/shuffle20_degree_inclusion_alignment",
    ),
    (
        "code/supp_bam34_alpha_controls/curated/bam34_alpha_controls",
        "figures/supplement/bam34_alpha_controls",
    ),
    (
        "code/supp_degree_bin_vertex_inclusion_shift/curated/degree_bin_vertex_inclusion_shift",
        "figures/supplement/degree_bin_vertex_inclusion_shift",
    ),
    (
        "code/supp_schedule_regular3_controls/curated/schedule_regular3_controls",
        "figures/supplement/schedule_regular3_controls",
    ),
    (
        "code/supp_er_n150_qabase_vs_dcosqa/curated/er_n150_qabase_vs_dcosqa",
        "figures/supplement/er_n150_qabase_vs_dcosqa",
    ),
    (
        "code/supp_yield_diversity_robustness/curated/yield_diversity_robustness",
        "figures/supplement/yield_diversity_robustness",
    ),
    (
        "code/supp_idealized_closed_system/curated/idealized_closed_system_endpoint",
        "figures/supplement/idealized_closed_system_endpoint",
    ),
    (
        "code/supp_idealized_closed_system/curated/idealized_closed_system_sensitivity",
        "figures/supplement/idealized_closed_system_sensitivity",
    ),
]


def run_scripts() -> list[tuple[str, int]]:
    results = []
    for relative in ANALYSIS_SCRIPTS:
        completed = subprocess.run(
            [sys.executable, relative], cwd=ROOT, check=False
        )
        results.append((relative, completed.returncode))
    return results


def publish_figures() -> list[str]:
    failures = []
    for source_text, target_text in PUBLICATION_PAIRS:
        source_stem = ROOT / source_text
        target_stem = ROOT / target_text
        for suffix in (".pdf", ".png"):
            source = source_stem.with_suffix(suffix)
            target = target_stem.with_suffix(suffix)
            if not source.is_file() or source.stat().st_size == 0:
                failures.append(f"missing source: {source}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if not target.is_file() or target.stat().st_size == 0:
                failures.append(f"publication mismatch: {source} -> {target}")
    return failures


def remove_transient_curated_figures() -> None:
    """Remove duplicate staging figures after successful publication copying."""
    for source_text, _ in PUBLICATION_PAIRS:
        source_stem = ROOT / source_text
        if source_stem.parent.name != "curated":
            continue
        for suffix in (".pdf", ".png"):
            source = source_stem.with_suffix(suffix)
            if source.is_file():
                source.unlink()


def main() -> int:
    results = run_scripts()
    script_failures = [
        (path, code) for path, code in results if code != 0
    ]
    publication_failures = []
    if not script_failures:
        publication_failures = publish_figures()
        if not publication_failures:
            remove_transient_curated_figures()

    print("== analysis script summary ==")
    print("PASS", len(results) - len(script_failures))
    print("FAIL", len(script_failures))
    if script_failures:
        print("== failed analysis scripts ==")
        for path, code in script_failures:
            print(f"FAIL {code}: {path}")

    print("== publication summary ==")
    publication_count = len(PUBLICATION_PAIRS) * 2
    if script_failures:
        print("SKIPPED because one or more analysis scripts failed")
    else:
        print("PASS", publication_count - len(publication_failures))
        print("FAIL", len(publication_failures))
    for failure in publication_failures:
        print("FAIL", failure)
    return int(bool(script_failures or publication_failures))


if __name__ == "__main__":
    raise SystemExit(main())
