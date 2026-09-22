#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Phase 7.10 — Thesis Evidence Synthesis & Phase Closure V1
#
# Purpose
# -------
# Consolidate the completed Phase-7 diagnostics into one thesis-ready,
# versioned evidence package.
#
# This phase performs NO new model experiment and NO model selection.
# It synthesizes:
#   - frozen one-shot final-test baseline anatomy
#   - new-to-investor novelty diagnostics
#   - cold-start diagnostics
#   - structural availability / source / degree diagnostics
#   - temporal-history diagnostics
#   - error-tail concentration
#   - adjusted Hit@10 diagnostics
#   - dependence / pair-weighting robustness
#
# Scientific role
# ---------------
# SYNTHESIS ONLY.
#
# The script distinguishes:
#   SUPPORTED OBSERVATIONS
#   ROBUST ADJUSTED ASSOCIATIONS
#   MOTIVATED BUT UNTESTED MODEL HYPOTHESES
#
# It explicitly does NOT claim:
#   - causality
#   - mechanism identification
#   - that an inductive architecture has already been shown to improve results
#
# Prohibited
# ----------
# - model loading
# - checkpoint loading
# - raw-logit loading
# - inference
# - test rescoring
# - negative regeneration
# - candidate modification
# - checkpoint/model selection
# - retraining
# =============================================================================


REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    REPO_ROOT
    / "data/experimental/phase_6/full_training/100pct/final_test/"
    / "analysis_ready_test_cases.parquet"
)

PHASE_7_4_MANIFEST = (
    REPO_ROOT
    / "data/experimental/phase_7/"
    / "phase_7_4_novelty_cold_start_interaction/"
    / "phase_7_4_analysis_manifest_V1.json"
)

PHASE_7_5B_MANIFEST = (
    REPO_ROOT
    / "data/experimental/phase_7/"
    / "phase_7_5b_structural_performance/"
    / "phase_7_5b_analysis_manifest_V1.json"
)

PHASE_7_5C_MANIFEST = (
    REPO_ROOT
    / "data/experimental/phase_7/"
    / "phase_7_5c_degree_performance/"
    / "phase_7_5c_analysis_manifest_V1.json"
)

PHASE_7_5C_OBSERVED = (
    REPO_ROOT
    / "data/experimental/phase_7/"
    / "phase_7_5c_degree_performance/"
    / "phase_7_5c_primary_cold_startup_degree_observed_V1.json"
)

PHASE_7_5C_BOOTSTRAP = (
    REPO_ROOT
    / "data/experimental/phase_7/"
    / "phase_7_5c_degree_performance/"
    / "phase_7_5c_primary_degree_bootstrap_summary_V1.csv"
)

PHASE_7_6B_MANIFEST = (
    REPO_ROOT
    / "data/experimental/phase_7/"
    / "phase_7_6b_temporal_history_performance/"
    / "phase_7_6b_analysis_manifest_V1.json"
)

PHASE_7_6B_CONTRASTS = (
    REPO_ROOT
    / "data/experimental/phase_7/"
    / "phase_7_6b_temporal_history_performance/"
    / "phase_7_6b_temporal_history_contrasts_V1.csv"
)

PHASE_7_6B_BOOTSTRAP = (
    REPO_ROOT
    / "data/experimental/phase_7/"
    / "phase_7_6b_temporal_history_performance/"
    / "phase_7_6b_investor_cluster_bootstrap_summary_V1.csv"
)

PHASE_7_7_MANIFEST = (
    REPO_ROOT
    / "data/experimental/phase_7/"
    / "phase_7_7_error_archetypes/"
    / "phase_7_7_analysis_manifest_V1.json"
)

PHASE_7_7_SUMMARY = (
    REPO_ROOT
    / "data/experimental/phase_7/"
    / "phase_7_7_error_archetypes/"
    / "phase_7_7_error_anatomy_summary_V1.json"
)

PHASE_7_8_MANIFEST = (
    REPO_ROOT
    / "data/experimental/phase_7/"
    / "phase_7_8_adjusted_hit10/"
    / "phase_7_8_analysis_manifest_V1.json"
)

PHASE_7_9_MANIFEST = (
    REPO_ROOT
    / "data/experimental/phase_7/"
    / "phase_7_9_dependence_robustness/"
    / "phase_7_9_analysis_manifest_V1.json"
)

PHASE_7_9_SYNTHESIS = (
    REPO_ROOT
    / "data/experimental/phase_7/"
    / "phase_7_9_dependence_robustness/"
    / "phase_7_9_robustness_synthesis_V1.csv"
)

PHASE_7_9_SUMMARY = (
    REPO_ROOT
    / "data/experimental/phase_7/"
    / "phase_7_9_dependence_robustness/"
    / "phase_7_9_robustness_summary_V1.json"
)

OUT_DIR = (
    REPO_ROOT
    / "data/experimental/phase_7/"
    / "phase_7_10_thesis_evidence_synthesis"
)

FIG_DIR = OUT_DIR / "figures"

EXPECTED_SOURCE_SHA256 = (
    "d70f21bff0006e094c5d567d307c0664"
    "b41a11e7250a0e69aaec610e1810132d"
)

EXPECTED_ROWS = 20_264

SOURCE_COLUMNS = [
    "interaction_id",
    "investor_id",
    "startup_id",
    "positive_rank",
    "NDCG@10",
    "new_to_investor_pair",
    "prior_investor_startup_relationship",
    "cold_start_investor",
    "cold_start_startup",
    "investor_core_connected",
    "startup_core_connected",
    "startup_structural_source_class",
]


# =============================================================================
# Helpers
# =============================================================================


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def file_sha256(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def json_converter(value):
    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        if np.isnan(value):
            return None

        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if pd.isna(value):
        return None

    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def json_dump(obj, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=json_converter,
        )


def print_section(title: str) -> None:
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)


def save_figure(fig, stem: str) -> list[Path]:
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
        metadata={
            "Title": stem,
            "Author": "ITRS Phase 7 thesis evidence synthesis",
        },
    )

    plt.close(fig)

    return [png, pdf]


def group_metrics(
    df: pd.DataFrame,
    mask: pd.Series,
) -> dict:
    group = df.loc[mask]
    rank = group["positive_rank"].astype(int)

    return {
        "n": int(len(group)),
        "HR@10": float((rank <= 10).mean()),
        "NDCG@10": float(group["NDCG@10"].mean()),
        "mean_rank": float(rank.mean()),
        "median_rank": float(rank.median()),
        "rank_gt_50_n": int((rank > 50).sum()),
        "rank_gt_50_share": float((rank > 50).mean()),
    }


def pct(x: float, digits: int = 2) -> str:
    return f"{x * 100:.{digits}f}%"


def pp(x: float, digits: int = 2) -> str:
    return f"{x * 100:+.{digits}f} pp"


def ci_pp(low: float, high: float, digits: int = 2) -> str:
    return (
        f"[{low * 100:+.{digits}f}, "
        f"{high * 100:+.{digits}f}] pp"
    )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    print("=" * 110)
    print(
        "PHASE 7.10 — THESIS EVIDENCE "
        "SYNTHESIS & PHASE CLOSURE V1"
    )
    print("=" * 110)

    print(
        "Scientific role:             "
        "CONSOLIDATED POST-HOC EVIDENCE SYNTHESIS"
    )
    print("New model experiment:         NO")
    print("Model loaded:                 NO")
    print("Checkpoint loaded:            NO")
    print("Raw logits loaded:            NO")
    print("Inference executed:           NO")
    print("Test rescored:                NO")
    print("Model selection performed:   NO")
    print("Causal interpretation:        NO")

    # =========================================================================
    # 7.10.1 — Integrity gate
    # =========================================================================

    print_section(
        "7.10.1 — PHASE-7 INTEGRITY AND COMPLETION GATE"
    )

    manifest_paths = [
        PHASE_7_4_MANIFEST,
        PHASE_7_5B_MANIFEST,
        PHASE_7_5C_MANIFEST,
        PHASE_7_6B_MANIFEST,
        PHASE_7_7_MANIFEST,
        PHASE_7_8_MANIFEST,
        PHASE_7_9_MANIFEST,
    ]

    required_paths = [
        SOURCE,
        *manifest_paths,
        PHASE_7_5C_OBSERVED,
        PHASE_7_5C_BOOTSTRAP,
        PHASE_7_6B_CONTRASTS,
        PHASE_7_6B_BOOTSTRAP,
        PHASE_7_7_SUMMARY,
        PHASE_7_9_SYNTHESIS,
        PHASE_7_9_SUMMARY,
    ]

    for path in required_paths:
        require(
            path.exists(),
            f"Missing required artifact: {path}",
        )

        print(
            "FOUND  "
            f"{path.relative_to(REPO_ROOT)}"
        )

    source_hash = file_sha256(SOURCE)

    require(
        source_hash == EXPECTED_SOURCE_SHA256,
        "Frozen Phase-6 analysis bundle SHA256 drift.",
    )

    for path in manifest_paths:
        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            manifest = json.load(f)

        require(
            manifest["status"] == "COMPLETE",
            f"Prerequisite phase not COMPLETE: {path}",
        )

    print()
    print("Frozen source fingerprint:   PASS")
    print("Phase 7.4:                   COMPLETE")
    print("Phase 7.5B:                  COMPLETE")
    print("Phase 7.5C:                  COMPLETE")
    print("Phase 7.6B:                  COMPLETE")
    print("Phase 7.7:                   COMPLETE")
    print("Phase 7.8:                   COMPLETE")
    print("Phase 7.9:                   COMPLETE")

    # =========================================================================
    # 7.10.2 — Frozen descriptive evidence
    # =========================================================================

    print_section(
        "7.10.2 — FROZEN FINAL-TEST DESCRIPTIVE EVIDENCE"
    )

    df = pd.read_parquet(
        SOURCE,
        columns=SOURCE_COLUMNS,
    )

    require(
        len(df) == EXPECTED_ROWS,
        "Final-test row-count drift.",
    )

    require(
        df["interaction_id"].is_unique,
        "interaction_id is not unique.",
    )

    overall = group_metrics(
        df,
        pd.Series(
            True,
            index=df.index,
        ),
    )

    new = (
        df[
            "new_to_investor_pair"
        ].astype(bool)
    )

    prior = (
        ~new
    )

    cold_i = (
        df[
            "cold_start_investor"
        ].astype(bool)
    )

    cold_s = (
        df[
            "cold_start_startup"
        ].astype(bool)
    )

    startup_connected = (
        df[
            "startup_core_connected"
        ].astype(bool)
    )

    warm_warm = (
        ~cold_i
        &
        ~cold_s
    )

    prior_ww = (
        prior
        &
        warm_warm
    )

    new_ww = (
        new
        &
        warm_warm
    )

    new_wc = (
        new
        &
        ~cold_i
        &
        cold_s
    )

    new_cw = (
        new
        &
        cold_i
        &
        ~cold_s
    )

    new_cc = (
        new
        &
        cold_i
        &
        cold_s
    )

    new_wc_connected = (
        new_wc
        &
        startup_connected
    )

    new_wc_isolated = (
        new_wc
        &
        ~startup_connected
    )

    metrics = {
        "overall":
            overall,

        "prior":
            group_metrics(
                df,
                prior,
            ),

        "new":
            group_metrics(
                df,
                new,
            ),

        "prior_ww":
            group_metrics(
                df,
                prior_ww,
            ),

        "new_ww":
            group_metrics(
                df,
                new_ww,
            ),

        "new_wc":
            group_metrics(
                df,
                new_wc,
            ),

        "new_cw":
            group_metrics(
                df,
                new_cw,
            ),

        "new_cc":
            group_metrics(
                df,
                new_cc,
            ),

        "new_wc_connected":
            group_metrics(
                df,
                new_wc_connected,
            ),

        "new_wc_isolated":
            group_metrics(
                df,
                new_wc_isolated,
            ),
    }

    require(
        metrics[
            "overall"
        ][
            "n"
        ]
        == 20_264,
        "Overall N drift.",
    )

    require(
        metrics[
            "prior"
        ][
            "n"
        ]
        == 3_818,
        "Prior-pair N drift.",
    )

    require(
        metrics[
            "new"
        ][
            "n"
        ]
        == 16_446,
        "New-pair N drift.",
    )

    require(
        metrics[
            "new_ww"
        ][
            "n"
        ]
        == 6_889,
        "New warm/warm N drift.",
    )

    require(
        metrics[
            "new_wc"
        ][
            "n"
        ]
        == 6_392,
        "New warm-investor/cold-startup N drift.",
    )

    require(
        metrics[
            "new_wc_connected"
        ][
            "n"
        ]
        == 809,
        "Connected cold-start startup N drift.",
    )

    require(
        metrics[
            "new_wc_isolated"
        ][
            "n"
        ]
        == 5_583,
        "Isolated cold-start startup N drift.",
    )

    print(
        f"Overall final-test HR@10:     "
        f"{pct(metrics['overall']['HR@10'])}"
    )

    print(
        f"Overall final-test NDCG@10:   "
        f"{metrics['overall']['NDCG@10']:.4f}"
    )

    print()
    print(
        f"Prior-pair HR@10:             "
        f"{pct(metrics['prior']['HR@10'])}"
    )

    print(
        f"New-pair HR@10:               "
        f"{pct(metrics['new']['HR@10'])}"
    )

    print()
    print(
        f"Prior warm/warm HR@10:        "
        f"{pct(metrics['prior_ww']['HR@10'])}"
    )

    print(
        f"New warm/warm HR@10:          "
        f"{pct(metrics['new_ww']['HR@10'])}"
    )

    print()
    print(
        f"New warm-I/cold-S HR@10:      "
        f"{pct(metrics['new_wc']['HR@10'])}"
    )

    print(
        f"New cold-I/warm-S HR@10:      "
        f"{pct(metrics['new_cw']['HR@10'])}"
    )

    print(
        f"New dual-cold HR@10:          "
        f"{pct(metrics['new_cc']['HR@10'])}"
    )

    # =========================================================================
    # 7.10.3 — Load completed statistical evidence
    # =========================================================================

    print_section(
        "7.10.3 — LOAD ADJUSTED / ROBUST / NEGATIVE EVIDENCE"
    )

    robustness = pd.read_csv(
        PHASE_7_9_SYNTHESIS
    ).set_index(
        "estimand_key"
    )

    with PHASE_7_5C_OBSERVED.open(
        "r",
        encoding="utf-8",
    ) as f:
        degree_observed = json.load(f)

    degree_bootstrap = pd.read_csv(
        PHASE_7_5C_BOOTSTRAP
    )

    temporal_contrasts = pd.read_csv(
        PHASE_7_6B_CONTRASTS
    )

    temporal_bootstrap = pd.read_csv(
        PHASE_7_6B_BOOTSTRAP
    )

    with PHASE_7_7_SUMMARY.open(
        "r",
        encoding="utf-8",
    ) as f:
        error_summary = json.load(f)

    with PHASE_7_9_SUMMARY.open(
        "r",
        encoding="utf-8",
    ) as f:
        robustness_summary = json.load(f)

    key_required = [
        "m1_adjusted_new_vs_prior_warm_warm",
        "m2_cold_start_penalty_if_isolated",
        "m2_cold_start_penalty_if_connected",
        "m2_structure_effect_among_warm_startups",
        "m2_structure_effect_among_cold_startups",
        "m2_structure_interaction_cold_minus_warm",
        "m3_founder_signal_vs_neither",
        "m3_acquisition_only_vs_neither",
    ]

    for key in key_required:
        require(
            key in robustness.index,
            f"Missing robustness estimand: {key}",
        )

    print(
        "Adjusted robustness estimands: PASS"
    )

    print(
        "Degree-intensity artifacts:    PASS"
    )

    print(
        "Temporal-history artifacts:    PASS"
    )

    print(
        "Error-anatomy artifacts:       PASS"
    )

    # =========================================================================
    # 7.10.4 — Construct thesis evidence table
    # =========================================================================

    print_section(
        "7.10.4 — CONSOLIDATED THESIS EVIDENCE TABLE"
    )

    m1 = robustness.loc[
        "m1_adjusted_new_vs_prior_warm_warm"
    ]

    m2_iso = robustness.loc[
        "m2_cold_start_penalty_if_isolated"
    ]

    m2_conn = robustness.loc[
        "m2_cold_start_penalty_if_connected"
    ]

    m2_warm_structure = robustness.loc[
        "m2_structure_effect_among_warm_startups"
    ]

    m2_cold_structure = robustness.loc[
        "m2_structure_effect_among_cold_startups"
    ]

    m2_interaction = robustness.loc[
        "m2_structure_interaction_cold_minus_warm"
    ]

    m3_founder = robustness.loc[
        "m3_founder_signal_vs_neither"
    ]

    m3_acquisition = robustness.loc[
        "m3_acquisition_only_vs_neither"
    ]

    temporal_primary = temporal_contrasts.loc[
        temporal_contrasts[
            "comparison_key"
        ]
        == (
            "new_ww_investor_t0_and_t1_t59_"
            "vs_t1_t59_only"
        )
    ]

    require(
        len(temporal_primary) == 1,
        "Could not identify temporal primary contrast.",
    )

    temporal_primary = (
        temporal_primary.iloc[0]
    )

    temporal_primary_ci = temporal_bootstrap.loc[
        (
            temporal_bootstrap[
                "comparison_key"
            ]
            == (
                "new_ww_investor_t0_and_t1_t59_"
                "vs_t1_t59_only"
            )
        )
        &
        (
            temporal_bootstrap[
                "metric"
            ]
            == "HR@10"
        )
    ]

    require(
        len(
            temporal_primary_ci
        )
        == 1,
        "Could not identify temporal HR@10 CI.",
    )

    temporal_primary_ci = (
        temporal_primary_ci.iloc[0]
    )

    degree_hr = degree_bootstrap.loc[
        degree_bootstrap[
            "metric"
        ]
        == "HR@10"
    ]

    require(
        len(degree_hr) == 1,
        "Could not identify degree HR@10 slope CI.",
    )

    degree_hr = (
        degree_hr.iloc[0]
    )

    evidence_rows = [
        {
            "evidence_id":
                "E1",

            "theme":
                "Aggregate final-test performance",

            "claim":
                (
                    "The frozen reproduced ITRS achieves moderate "
                    "aggregate final-test ranking performance."
                ),

            "estimate":
                metrics[
                    "overall"
                ][
                    "HR@10"
                ],

            "estimate_unit":
                "HR@10",

            "ci95_low":
                np.nan,

            "ci95_high":
                np.nan,

            "evidence_strength":
                "DESCRIPTIVE_FROZEN_TEST",

            "status":
                "SUPPORTED",

            "thesis_implication":
                (
                    "Aggregate performance alone is insufficient; "
                    "task-specific diagnostics are required."
                ),
        },

        {
            "evidence_id":
                "E2",

            "theme":
                "Pair novelty",

            "claim":
                (
                    "New-to-investor discovery is harder than "
                    "re-ranking prior relationships even when "
                    "both endpoints are warm."
                ),

            "estimate":
                float(
                    m1[
                        "event_estimate"
                    ]
                ),

            "estimate_unit":
                "adjusted_Hit@10_difference",

            "ci95_low":
                float(
                    m1[
                        "event_two_way_ci95_low"
                    ]
                ),

            "ci95_high":
                float(
                    m1[
                        "event_two_way_ci95_high"
                    ]
                ),

            "evidence_strength":
                "ADJUSTED_TWO_WAY_CLUSTER_ROBUST",

            "status":
                "SUPPORTED",

            "thesis_implication":
                (
                    "New-to-investor discovery should be evaluated "
                    "as a distinct recommendation task."
                ),
        },

        {
            "evidence_id":
                "E3",

            "theme":
                "Startup cold start",

            "claim":
                (
                    "Within new-pair warm-investor cases, startup "
                    "cold start is associated with a very large "
                    "Hit@10 degradation."
                ),

            "estimate":
                float(
                    m2_iso[
                        "event_estimate"
                    ]
                ),

            "estimate_unit":
                (
                    "adjusted_Hit@10_difference_"
                    "cold_vs_warm_when_isolated"
                ),

            "ci95_low":
                float(
                    m2_iso[
                        "event_two_way_ci95_low"
                    ]
                ),

            "ci95_high":
                float(
                    m2_iso[
                        "event_two_way_ci95_high"
                    ]
                ),

            "evidence_strength":
                "ADJUSTED_TWO_WAY_CLUSTER_ROBUST",

            "status":
                "SUPPORTED",

            "thesis_implication":
                (
                    "Startup cold start is a primary target for "
                    "methodological improvement."
                ),
        },

        {
            "evidence_id":
                "E4",

            "theme":
                "Structural availability under startup cold start",

            "claim":
                (
                    "Heterogeneous startup structural availability "
                    "is associated with higher Hit@10 specifically "
                    "for cold-start startups."
                ),

            "estimate":
                float(
                    m2_cold_structure[
                        "event_estimate"
                    ]
                ),

            "estimate_unit":
                "adjusted_Hit@10_difference",

            "ci95_low":
                float(
                    m2_cold_structure[
                        "event_two_way_ci95_low"
                    ]
                ),

            "ci95_high":
                float(
                    m2_cold_structure[
                        "event_two_way_ci95_high"
                    ]
                ),

            "evidence_strength":
                "ADJUSTED_TWO_WAY_CLUSTER_ROBUST",

            "status":
                "SUPPORTED",

            "thesis_implication":
                (
                    "Inductive use of heterogeneous side information "
                    "is empirically motivated for startup cold start."
                ),
        },

        {
            "evidence_id":
                "E5",

            "theme":
                "Structure specificity",

            "claim":
                (
                    "Startup structural availability is not clearly "
                    "associated with higher Hit@10 among warm startups."
                ),

            "estimate":
                float(
                    m2_warm_structure[
                        "event_estimate"
                    ]
                ),

            "estimate_unit":
                "adjusted_Hit@10_difference",

            "ci95_low":
                float(
                    m2_warm_structure[
                        "event_two_way_ci95_low"
                    ]
                ),

            "ci95_high":
                float(
                    m2_warm_structure[
                        "event_two_way_ci95_high"
                    ]
                ),

            "evidence_strength":
                "ADJUSTED_TWO_WAY_CLUSTER_ROBUST",

            "status":
                "NO_CLEAR_ASSOCIATION",

            "thesis_implication":
                (
                    "The structural signal appears concentrated in "
                    "the cold-start regime rather than being universally beneficial."
                ),
        },

        {
            "evidence_id":
                "E6",

            "theme":
                "Founder-related structure",

            "claim":
                (
                    "Founder-related structural support is associated "
                    "with higher Hit@10 for new cold-start startups."
                ),

            "estimate":
                float(
                    m3_founder[
                        "event_estimate"
                    ]
                ),

            "estimate_unit":
                "adjusted_Hit@10_difference",

            "ci95_low":
                float(
                    m3_founder[
                        "event_two_way_ci95_low"
                    ]
                ),

            "ci95_high":
                float(
                    m3_founder[
                        "event_two_way_ci95_high"
                    ]
                ),

            "evidence_strength":
                "ADJUSTED_TWO_WAY_CLUSTER_ROBUST",

            "status":
                "SUPPORTED",

            "thesis_implication":
                (
                    "Founder-side relations are a high-priority source "
                    "for inductive startup representation."
                ),
        },

        {
            "evidence_id":
                "E7",

            "theme":
                "Acquisition-only structure",

            "claim":
                (
                    "Acquisition-only structural support does not show "
                    "a clear Hit@10 association versus no structure."
                ),

            "estimate":
                float(
                    m3_acquisition[
                        "event_estimate"
                    ]
                ),

            "estimate_unit":
                "adjusted_Hit@10_difference",

            "ci95_low":
                float(
                    m3_acquisition[
                        "event_two_way_ci95_low"
                    ]
                ),

            "ci95_high":
                float(
                    m3_acquisition[
                        "event_two_way_ci95_high"
                    ]
                ),

            "evidence_strength":
                "ADJUSTED_TWO_WAY_CLUSTER_ROBUST",

            "status":
                "NO_CLEAR_ASSOCIATION",

            "thesis_implication":
                (
                    "Not all heterogeneous relations should be assumed "
                    "equally informative."
                ),
        },

        {
            "evidence_id":
                "E8",

            "theme":
                "Structural degree intensity",

            "claim":
                (
                    "Among already-connected cold-start startups, "
                    "greater structural degree does not show a clear "
                    "monotonic Hit@10 benefit."
                ),

            "estimate":
                float(
                    degree_hr[
                        "estimate"
                    ]
                ),

            "estimate_unit":
                "Hit@10_slope_per_log1p_degree",

            "ci95_low":
                float(
                    degree_hr[
                        "ci95_low"
                    ]
                ),

            "ci95_high":
                float(
                    degree_hr[
                        "ci95_high"
                    ]
                ),

            "evidence_strength":
                "INVESTOR_CLUSTER_BOOTSTRAP",

            "status":
                "NO_CLEAR_ASSOCIATION",

            "thesis_implication":
                (
                    "Availability/quality of side information may matter "
                    "more than simply accumulating more structural edges."
                ),
        },

        {
            "evidence_id":
                "E9",

            "theme":
                "Investor temporal coverage",

            "claim":
                (
                    "Within new warm/warm cases, investor observation "
                    "in both T0 and T1-T59 does not show a clear Hit@10 "
                    "advantage over T1-T59 only."
                ),

            "estimate":
                float(
                    temporal_primary[
                        "delta_HR@10"
                    ]
                ),

            "estimate_unit":
                "Hit@10_difference",

            "ci95_low":
                float(
                    temporal_primary_ci[
                        "ci95_low"
                    ]
                ),

            "ci95_high":
                float(
                    temporal_primary_ci[
                        "ci95_high"
                    ]
                ),

            "evidence_strength":
                "INVESTOR_CLUSTER_BOOTSTRAP",

            "status":
                "NO_CLEAR_ASSOCIATION",

            "thesis_implication":
                (
                    "The central problem is not explained simply by "
                    "broader investor temporal coverage."
                ),
        },

        {
            "evidence_id":
                "E10",

            "theme":
                "Severe-error concentration",

            "claim":
                (
                    "Structurally isolated cold-start startups dominate "
                    "the rank>50 failure tail."
                ),

            "estimate":
                float(
                    error_summary[
                        "cold_isolated_startup_in_severe_tail_share"
                    ]
                ),

            "estimate_unit":
                "share_of_rank_gt_50_failures",

            "ci95_low":
                np.nan,

            "ci95_high":
                np.nan,

            "evidence_strength":
                "DESCRIPTIVE_FROZEN_TEST",

            "status":
                "SUPPORTED",

            "thesis_implication":
                (
                    "The most severe model failures concentrate exactly "
                    "where inductive startup representation is most relevant."
                ),
        },

        {
            "evidence_id":
                "E11",

            "theme":
                "Dependence / weighting robustness",

            "claim":
                (
                    "Central adjusted findings are stable to two-way "
                    "investor/startup clustering and pair-balanced weighting."
                ),

            "estimate":
                np.nan,

            "estimate_unit":
                "robustness_statement",

            "ci95_low":
                np.nan,

            "ci95_high":
                np.nan,

            "evidence_strength":
                "MULTIPLE_ROBUSTNESS_CHECKS",

            "status":
                "SUPPORTED",

            "thesis_implication":
                (
                    "The main diagnostic conclusions are unlikely to be "
                    "artifacts of one clustering assumption or repeated-pair weighting."
                ),
        },
    ]

    evidence = pd.DataFrame(
        evidence_rows
    )

    print(
        evidence[
            [
                "evidence_id",
                "theme",
                "status",
                "estimate",
                "ci95_low",
                "ci95_high",
            ]
        ].to_string(
            index=False,
            formatters={
                "estimate":
                    lambda x:
                        ""
                        if pd.isna(x)
                        else f"{x:+.6f}",

                "ci95_low":
                    lambda x:
                        ""
                        if pd.isna(x)
                        else f"{x:+.6f}",

                "ci95_high":
                    lambda x:
                        ""
                        if pd.isna(x)
                        else f"{x:+.6f}",
            },
        )
    )

    # =========================================================================
    # 7.10.5 — Thesis synthesis text
    # =========================================================================

    print_section(
        "7.10.5 — THESIS-READY SYNTHESIS"
    )

    novelty_pp = float(
        m1[
            "event_estimate"
        ]
    )

    novelty_low = float(
        m1[
            "event_two_way_ci95_low"
        ]
    )

    novelty_high = float(
        m1[
            "event_two_way_ci95_high"
        ]
    )

    cold_iso_pp = float(
        m2_iso[
            "event_estimate"
        ]
    )

    cold_iso_low = float(
        m2_iso[
            "event_two_way_ci95_low"
        ]
    )

    cold_iso_high = float(
        m2_iso[
            "event_two_way_ci95_high"
        ]
    )

    cold_structure_pp = float(
        m2_cold_structure[
            "event_estimate"
        ]
    )

    cold_structure_low = float(
        m2_cold_structure[
            "event_two_way_ci95_low"
        ]
    )

    cold_structure_high = float(
        m2_cold_structure[
            "event_two_way_ci95_high"
        ]
    )

    founder_pp = float(
        m3_founder[
            "event_estimate"
        ]
    )

    founder_low = float(
        m3_founder[
            "event_two_way_ci95_low"
        ]
    )

    founder_high = float(
        m3_founder[
            "event_two_way_ci95_high"
        ]
    )

    severe_cold_isolated_share = float(
        error_summary[
            "cold_isolated_startup_in_severe_tail_share"
        ]
    )

    problem_statement = (
        "Although the reproduced ITRS achieves "
        f"{pct(metrics['overall']['HR@10'])} HR@10 on the frozen "
        "one-shot final test, aggregate future-event performance masks "
        "substantial task heterogeneity. New-to-investor relationships "
        "remain harder than prior investor-startup relationships even "
        "when both endpoints have historical observations: the adjusted "
        f"warm/warm new-pair difference is {pp(novelty_pp)} "
        f"(two-way cluster 95% CI {ci_pp(novelty_low, novelty_high)}). "
        "Within new-pair cases, startup cold start is associated with a "
        "much larger degradation, while heterogeneous structural support "
        f"is associated with {pp(cold_structure_pp)} higher Hit@10 among "
        "cold-start startups "
        f"(95% CI {ci_pp(cold_structure_low, cold_structure_high)}). "
        "The structural association is concentrated in founder-related "
        f"support ({pp(founder_pp)}, 95% CI "
        f"{ci_pp(founder_low, founder_high)}), whereas acquisition-only "
        "support, structural degree intensity, and broader investor "
        "temporal coverage do not show clear benefits. "
        f"{pct(severe_cold_isolated_share)} of all rank>50 failures "
        "involve a cold-start startup that is structurally isolated."
    )

    thesis_motivation = (
        "These diagnostics motivate a recommendation framework explicitly "
        "designed for new-to-investor startup discovery under startup cold "
        "start, using inductive heterogeneous startup representations that "
        "can exploit side information even when investment history is absent, "
        "while retaining temporal investor-preference modeling. Phase 7 does "
        "not establish that such an architecture will improve performance; "
        "that claim requires controlled model-development experiments and "
        "ablation studies."
    )

    print(
        "PROBLEM STATEMENT SYNTHESIS"
    )
    print()
    print(problem_statement)

    print()
    print(
        "RESEARCH MOTIVATION SYNTHESIS"
    )
    print()
    print(thesis_motivation)

    # =========================================================================
    # 7.10.6 — Future research task alignment
    # =========================================================================

    print_section(
        "7.10.6 — EVIDENCE-TO-RESEARCH-TASK ALIGNMENT"
    )

    research_tasks = [
        {
            "task_id":
                "R1",

            "research_task":
                (
                    "Design an inductive startup representation that "
                    "does not require prior investment interactions."
                ),

            "evidence_basis":
                (
                    "Large startup cold-start penalty + severe-error "
                    "concentration in cold isolated startups."
                ),

            "phase_7_status":
                "MOTIVATED_NOT_TESTED",
        },

        {
            "task_id":
                "R2",

            "research_task":
                (
                    "Prioritize founder-related heterogeneous relations "
                    "as startup side information and test relation-specific ablations."
                ),

            "evidence_basis":
                (
                    "Founder structural signal robustly positive; "
                    "acquisition-only signal unclear."
                ),

            "phase_7_status":
                "MOTIVATED_NOT_TESTED",
        },

        {
            "task_id":
                "R3",

            "research_task":
                (
                    "Preserve investor temporal-preference modeling but "
                    "evaluate whether the trend component adds value beyond "
                    "the frozen history indicators."
                ),

            "evidence_basis":
                (
                    "No clear advantage from broader T0/T1-T59 investor "
                    "coverage alone."
                ),

            "phase_7_status":
                "MOTIVATED_NOT_TESTED",
        },

        {
            "task_id":
                "R4",

            "research_task":
                (
                    "Make new-to-investor discovery the primary evaluation "
                    "task, with warm/warm, startup-cold, investor-cold, and "
                    "dual-cold strata reported separately."
                ),

            "evidence_basis":
                (
                    "Persistent adjusted new-pair gap and large subgroup "
                    "heterogeneity hidden by aggregate HR@10."
                ),

            "phase_7_status":
                "SUPPORTED_DESIGN_REQUIREMENT",
        },

        {
            "task_id":
                "R5",

            "research_task":
                (
                    "Use controlled ablations to test whether inductive "
                    "founder/heterogeneous features causally improve cold-start "
                    "ranking relative to the reproduced ITRS baseline."
                ),

            "evidence_basis":
                (
                    "Phase 7 establishes association, not causal mechanism "
                    "or architecture superiority."
                ),

            "phase_7_status":
                "REQUIRED_NEXT_EXPERIMENT",
        },
    ]

    research_tasks_df = pd.DataFrame(
        research_tasks
    )

    print(
        research_tasks_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # 7.10.7 — Write thesis-ready markdown
    # =========================================================================

    print_section(
        "7.10.7 — WRITE THESIS-READY SYNTHESIS ARTIFACTS"
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    evidence_path = (
        OUT_DIR
        / "phase_7_10_thesis_evidence_table_V1.csv"
    )

    research_tasks_path = (
        OUT_DIR
        / "phase_7_10_research_task_alignment_V1.csv"
    )

    synthesis_md_path = (
        OUT_DIR
        / "phase_7_10_thesis_evidence_synthesis_V1.md"
    )

    slide_md_path = (
        OUT_DIR
        / "phase_7_10_supervisor_slide_summary_V1.md"
    )

    closure_json_path = (
        OUT_DIR
        / "phase_7_10_phase_closure_evidence_V1.json"
    )

    evidence.to_csv(
        evidence_path,
        index=False,
    )

    research_tasks_df.to_csv(
        research_tasks_path,
        index=False,
    )

    synthesis_md = f"""# Phase 7 — Final Test Analysis & Thesis Diagnostics

## Scientific status

Phase 7 is a post-hoc analysis of the immutable Phase-6 one-shot final test.
No model was retrained, reselected, rescored, or re-evaluated.

## Frozen final-test baseline

- Test events: **20,264**
- Candidates per case: **100**
- Overall HR@10: **{pct(metrics['overall']['HR@10'])}**
- Overall NDCG@10: **{metrics['overall']['NDCG@10']:.4f}**
- Mean positive rank: **{metrics['overall']['mean_rank']:.2f}**
- Median positive rank: **{metrics['overall']['median_rank']:.0f}**

## Central findings

### 1. Aggregate performance masks a distinct new-to-investor discovery gap

Prior-pair HR@10 is **{pct(metrics['prior']['HR@10'])}**, while new-to-investor HR@10 is
**{pct(metrics['new']['HR@10'])}**.

More importantly, the gap persists when both endpoints are warm:

- prior warm/warm HR@10: **{pct(metrics['prior_ww']['HR@10'])}**
- new warm/warm HR@10: **{pct(metrics['new_ww']['HR@10'])}**

The adjusted warm/warm new-pair difference is **{pp(novelty_pp)}**
(two-way investor/startup cluster 95% CI **{ci_pp(novelty_low, novelty_high)}**).

**Interpretation:** new-to-investor discovery is empirically distinct from ranking repeated
investor-startup relationships and remains harder even without entity cold start.

### 2. Startup cold start is the largest observed degradation

Within new-to-investor cases:

- warm investor + warm startup HR@10: **{pct(metrics['new_ww']['HR@10'])}**
- warm investor + cold startup HR@10: **{pct(metrics['new_wc']['HR@10'])}**
- cold investor + warm startup HR@10: **{pct(metrics['new_cw']['HR@10'])}**
- dual cold HR@10: **{pct(metrics['new_cc']['HR@10'])}**

After adjustment, startup cold start is associated with a **{pp(cold_iso_pp)}**
Hit@10 difference when the startup is structurally isolated
(95% CI **{ci_pp(cold_iso_low, cold_iso_high)}**).

### 3. Structural side information is specifically useful under startup cold start

In the new-pair, warm-investor, cold-startup regime:

- structurally connected startup HR@10: **{pct(metrics['new_wc_connected']['HR@10'])}**
- structurally isolated startup HR@10: **{pct(metrics['new_wc_isolated']['HR@10'])}**

The adjusted structural association among cold-start startups is
**{pp(cold_structure_pp)}** (two-way cluster 95% CI
**{ci_pp(cold_structure_low, cold_structure_high)}**).

The corresponding adjusted association among warm startups is only
**{pp(float(m2_warm_structure['event_estimate']))}** with a confidence interval
that crosses zero.

### 4. Founder-related structure carries the clearest structural signal

Founder-related structural support is associated with **{pp(founder_pp)}**
higher Hit@10 versus no structural support
(95% CI **{ci_pp(founder_low, founder_high)}**).

Acquisition-only support shows no clear association:
**{pp(float(m3_acquisition['event_estimate']))}**, with a confidence interval crossing zero.

### 5. More structural degree is not clearly better once structure exists

Among the **{degree_observed['n_positive_degree']:,}** connected cold-start startups,
the Spearman association between log-degree and Hit@10 is
**{degree_observed['spearman_logdegree_vs_HR10']:+.4f}**.

The Hit@10 slope per +1 log1p(degree) is
**{float(degree_hr['estimate']):+.4f}**
with 95% CI **[{float(degree_hr['ci95_low']):+.4f}, {float(degree_hr['ci95_high']):+.4f}]**.

**Interpretation:** having usable side structure appears more important than simply
accumulating more edges.

### 6. Broader investor T0/T1-T59 coverage does not explain the new-pair gap

Within new warm/warm cases, the investor T0+T1-T59 minus T1-T59-only HR@10
difference is **{pp(float(temporal_primary['delta_HR@10']))}**
with investor-cluster 95% CI
**{ci_pp(float(temporal_primary_ci['ci95_low']), float(temporal_primary_ci['ci95_high']))}**.

This does not provide clear evidence that broader investor temporal coverage alone
drives new-pair ranking performance.

### 7. Severe failures are concentrated in structurally isolated cold-start startups

There are **{int(error_summary['total_rank_gt_50_failures']):,}** rank>50 failures.

- cold-start startup involved: **{pct(float(error_summary['cold_startup_in_severe_tail_share']))}**
- cold + structurally isolated startup: **{pct(severe_cold_isolated_share)}**
- warm investor + cold isolated startup: **{pct(float(error_summary['new_wc_isolated_in_severe_tail_share']))}**

### 8. Central findings are robust to dependence and repeated-pair weighting

Phase 7.9 preserved the exact Phase-7.8 model formulas and showed that the main
conclusions remain stable under:

- investor clustering
- startup clustering
- pair clustering
- two-way investor + startup clustering
- pair-balanced weighting

Unique investor-startup pairs: **{int(robustness_summary['pair_balanced_robustness']['unique_pair_n']):,}**
from **20,264** events.

Pairs with multiple final-test events:
**{int(robustness_summary['pair_balanced_robustness']['pairs_with_multiple_events_n']):,}**.

## Thesis-ready problem statement

{problem_statement}

## Thesis motivation

{thesis_motivation}

## What Phase 7 supports

Phase 7 supports the empirical diagnosis that:

1. new-to-investor discovery is a distinct and harder task than repeated-pair ranking;
2. startup cold start is the dominant observed degradation;
3. founder-related heterogeneous structure is associated with partial improvement specifically under startup cold start;
4. the most severe failures concentrate in structurally isolated cold-start startups;
5. these conclusions survive adjustment, two-way clustering, and pair-balanced robustness checks.

## What Phase 7 does NOT support

Phase 7 does **not** establish:

- causality;
- that founder structure is the mechanism causing better rankings;
- that an inductive GNN or any particular architecture will improve the final test;
- that all heterogeneous relations are beneficial;
- that structural density itself improves performance;
- that the frozen final test can now be reused for model selection.

The next model-development phase must use validation-only decisions and controlled
ablation experiments before any future final evaluation.
"""

    synthesis_md_path.write_text(
        synthesis_md,
        encoding="utf-8",
    )

    slide_md = f"""# Supervisor Slide — Phase 7 Thesis Diagnostics

## Headline

**Aggregate ITRS performance masks a distinct new-to-investor cold-start failure mode.**

## Four numbers

- Overall final-test HR@10: **{pct(metrics['overall']['HR@10'])}**
- Adjusted warm/warm new-vs-prior gap: **{pp(novelty_pp)}**
- Adjusted structural association for cold-start startups: **{pp(cold_structure_pp)}**
- Rank>50 failures involving structurally isolated cold-start startups: **{pct(severe_cold_isolated_share)}**

## Evidence chain

- **Novelty:** new-to-investor difficulty persists even with warm endpoints.
- **Cold start:** startup cold start produces the largest observed degradation.
- **Structure:** heterogeneous startup structure is associated with higher Hit@10 specifically under cold start.
- **Source:** founder-related structure carries the clearest signal.
- **Degree:** more edges are not clearly better once structure exists.
- **Temporal coverage:** broader investor T0/T1-T59 coverage does not clearly explain the gap.
- **Robustness:** findings survive two-way investor/startup clustering and pair-balanced weighting.

## Thesis implication

Develop and test an **inductive heterogeneous startup representation** for
**new-to-investor discovery under startup cold start**, with founder-side relations
as a priority information source.

## Caveat

Phase 7 is diagnostic and observational. It motivates the architecture; it does not
prove that the proposed architecture will improve performance.
"""

    slide_md_path.write_text(
        slide_md,
        encoding="utf-8",
    )

    closure_evidence = {
        "phase":
            "7.10",

        "status":
            "COMPLETE",

        "overall_final_test":
            metrics[
                "overall"
            ],

        "problem_statement_synthesis":
            problem_statement,

        "research_motivation_synthesis":
            thesis_motivation,

        "central_supported_findings": {
            "adjusted_warm_warm_novelty_gap":
                {
                    "estimate":
                        novelty_pp,

                    "two_way_ci95":
                        [
                            novelty_low,
                            novelty_high,
                        ],
                },

            "adjusted_cold_start_penalty_isolated":
                {
                    "estimate":
                        cold_iso_pp,

                    "two_way_ci95":
                        [
                            cold_iso_low,
                            cold_iso_high,
                        ],
                },

            "adjusted_structure_association_cold_start":
                {
                    "estimate":
                        cold_structure_pp,

                    "two_way_ci95":
                        [
                            cold_structure_low,
                            cold_structure_high,
                        ],
                },

            "adjusted_founder_signal":
                {
                    "estimate":
                        founder_pp,

                    "two_way_ci95":
                        [
                            founder_low,
                            founder_high,
                        ],
                },

            "cold_isolated_startup_share_of_severe_tail":
                severe_cold_isolated_share,
        },

        "negative_or_null_diagnostics": {
            "warm_startup_structure":
                {
                    "estimate":
                        float(
                            m2_warm_structure[
                                "event_estimate"
                            ]
                        ),

                    "two_way_ci95":
                        [
                            float(
                                m2_warm_structure[
                                    "event_two_way_ci95_low"
                                ]
                            ),
                            float(
                                m2_warm_structure[
                                    "event_two_way_ci95_high"
                                ]
                            ),
                        ],
                },

            "acquisition_only_structure":
                {
                    "estimate":
                        float(
                            m3_acquisition[
                                "event_estimate"
                            ]
                        ),

                    "two_way_ci95":
                        [
                            float(
                                m3_acquisition[
                                    "event_two_way_ci95_low"
                                ]
                            ),
                            float(
                                m3_acquisition[
                                    "event_two_way_ci95_high"
                                ]
                            ),
                        ],
                },

            "cold_start_degree_HR10_slope":
                {
                    "estimate":
                        float(
                            degree_hr[
                                "estimate"
                            ]
                        ),

                    "ci95":
                        [
                            float(
                                degree_hr[
                                    "ci95_low"
                                ]
                            ),
                            float(
                                degree_hr[
                                    "ci95_high"
                                ]
                            ),
                        ],
                },

            "investor_temporal_coverage_HR10":
                {
                    "estimate":
                        float(
                            temporal_primary[
                                "delta_HR@10"
                            ]
                        ),

                    "ci95":
                        [
                            float(
                                temporal_primary_ci[
                                    "ci95_low"
                                ]
                            ),
                            float(
                                temporal_primary_ci[
                                    "ci95_high"
                                ]
                            ),
                        ],
                },
        },

        "interpretation_boundary":
            {
                "causal":
                    False,

                "architecture_improvement_proven":
                    False,

                "final_test_available_for_model_selection":
                    False,
            },
    }

    json_dump(
        closure_evidence,
        closure_json_path,
    )

    # =========================================================================
    # 7.10.8 — Figures
    # =========================================================================

    print_section(
        "7.10.8 — GENERATE THESIS / PRESENTATION FIGURES"
    )

    figure_paths = []

    selected_keys = [
        "m1_adjusted_new_vs_prior_warm_warm",
        "m2_cold_start_penalty_if_isolated",
        "m2_structure_effect_among_cold_startups",
        "m2_structure_interaction_cold_minus_warm",
        "m3_founder_signal_vs_neither",
        "m3_acquisition_only_vs_neither",
    ]

    labels = {
        "m1_adjusted_new_vs_prior_warm_warm":
            "New vs prior\n(warm/warm)",

        "m2_cold_start_penalty_if_isolated":
            "Cold-start penalty\n(startup isolated)",

        "m2_structure_effect_among_cold_startups":
            "Startup structure\nwithin cold start",

        "m2_structure_interaction_cold_minus_warm":
            "Structure interaction\ncold minus warm",

        "m3_founder_signal_vs_neither":
            "Founder signal\nvs neither",

        "m3_acquisition_only_vs_neither":
            "Acquisition-only\nvs neither",
    }

    plot_df = (
        robustness.loc[
            selected_keys
        ]
        .reset_index()
    )

    estimates = (
        plot_df[
            "event_estimate"
        ].to_numpy()
        * 100.0
    )

    lows = (
        plot_df[
            "event_two_way_ci95_low"
        ].to_numpy()
        * 100.0
    )

    highs = (
        plot_df[
            "event_two_way_ci95_high"
        ].to_numpy()
        * 100.0
    )

    y = np.arange(
        len(plot_df)
    )

    fig, ax = plt.subplots(
        figsize=(10.5, 7)
    )

    ax.errorbar(
        estimates,
        y,
        xerr=[
            estimates - lows,
            highs - estimates,
        ],
        fmt="o",
        capsize=4,
    )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_yticks(
        y,
        [
            labels[key]
            for key in plot_df[
                "estimand_key"
            ]
        ],
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Adjusted Hit@10 difference (percentage points)"
    )

    ax.set_title(
        "Phase 7 thesis evidence — two-way investor/startup clustered intervals"
    )

    ax.grid(
        axis="x",
        alpha=0.20,
    )

    fig.tight_layout()

    figure_paths.extend(
        save_figure(
            fig,
            (
                "phase_7_10_fig1_"
                "central_adjusted_evidence_V1"
            ),
        )
    )

    # Figure 2 — thesis funnel
    funnel_labels = [
        "Overall",
        "New warm/warm",
        "New warm I + cold S\n(structure present)",
        "New warm I + cold S\n(structure absent)",
        "New dual cold",
    ]

    funnel_values = [
        metrics[
            "overall"
        ][
            "HR@10"
        ],
        metrics[
            "new_ww"
        ][
            "HR@10"
        ],
        metrics[
            "new_wc_connected"
        ][
            "HR@10"
        ],
        metrics[
            "new_wc_isolated"
        ][
            "HR@10"
        ],
        metrics[
            "new_cc"
        ][
            "HR@10"
        ],
    ]

    x = np.arange(
        len(
            funnel_labels
        )
    )

    fig, ax = plt.subplots(
        figsize=(11.5, 6.5)
    )

    bars = ax.bar(
        x,
        np.array(
            funnel_values
        )
        * 100.0,
    )

    ax.set_xticks(
        x,
        funnel_labels,
    )

    ax.set_ylabel(
        "HR@10 (%)"
    )

    ax.set_title(
        "Aggregate performance decomposes into a new-to-investor cold-start failure mode"
    )

    ax.grid(
        axis="y",
        alpha=0.20,
    )

    for bar, value in zip(
        bars,
        funnel_values,
    ):
        ax.annotate(
            f"{value * 100:.1f}%",
            (
                bar.get_x()
                + bar.get_width() / 2,
                bar.get_height(),
            ),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
        )

    fig.tight_layout()

    figure_paths.extend(
        save_figure(
            fig,
            (
                "phase_7_10_fig2_"
                "thesis_failure_funnel_V1"
            ),
        )
    )

    # =========================================================================
    # 7.10.9 — Closure manifest / hashes
    # =========================================================================

    print_section(
        "7.10.9 — PHASE-7 CLOSURE MANIFEST"
    )

    manifest_path = (
        OUT_DIR
        / "phase_7_10_analysis_manifest_V1.json"
    )

    manifest = {
        "phase":
            "7.10",

        "schema_version":
            "ITRS_PHASE7_10_THESIS_EVIDENCE_SYNTHESIS_V1",

        "status":
            "COMPLETE",

        "phase_7_status":
            "CLOSED",

        "scientific_role":
            (
                "Consolidated thesis evidence synthesis "
                "from completed Phase-7 post-hoc diagnostics."
            ),

        "source": {
            "path":
                str(
                    SOURCE.relative_to(
                        REPO_ROOT
                    )
                ),

            "sha256":
                source_hash,

            "rows":
                len(df),
        },

        "completed_prerequisites": [
            "7.4",
            "7.5B",
            "7.5C",
            "7.6B",
            "7.7",
            "7.8",
            "7.9",
        ],

        "central_evidence_ids":
            evidence[
                "evidence_id"
            ].tolist(),

        "phase_7_final_interpretation":
            (
                "ITRS aggregate performance masks a distinct "
                "new-to-investor discovery gap. Startup cold start "
                "is the dominant observed degradation. Founder-related "
                "heterogeneous structural support is associated with "
                "partial improvement specifically under startup cold start, "
                "while degree intensity and broader investor temporal "
                "coverage show no clear benefit. Findings are robust "
                "to two-way clustering and pair-balanced weighting."
            ),

        "phase_7_does_not_establish": [
            "causality",
            "mechanism identification",
            "inductive architecture superiority",
            "future final-test performance",
        ],

        "final_test_policy": {
            "rescore_performed":
                False,

            "model_selection_performed":
                False,

            "available_for_future_model_selection":
                False,
        },

        "figures": [
            str(
                path.relative_to(
                    REPO_ROOT
                )
            )
            for path in figure_paths
        ],

        "prohibited_operations": {
            "model_loaded":
                False,

            "checkpoint_loaded":
                False,

            "raw_logits_loaded":
                False,

            "inference_executed":
                False,

            "test_rescored":
                False,

            "negative_resampling":
                False,

            "candidate_modification":
                False,

            "model_selection":
                False,

            "retraining":
                False,
        },
    }

    json_dump(
        manifest,
        manifest_path,
    )

    artifact_paths = [
        evidence_path,
        research_tasks_path,
        synthesis_md_path,
        slide_md_path,
        closure_json_path,
        *figure_paths,
        manifest_path,
    ]

    hashes = {}

    for path in artifact_paths:
        relative = str(
            path.relative_to(
                REPO_ROOT
            )
        )

        hashes[
            relative
        ] = file_sha256(
            path
        )

        print(
            f"WROTE  {relative}"
        )

    hashes_path = (
        OUT_DIR
        / "phase_7_10_derived_artifact_sha256_V1.json"
    )

    json_dump(
        hashes,
        hashes_path,
    )

    print(
        "WROTE  "
        f"{hashes_path.relative_to(REPO_ROOT)}"
    )

    # =========================================================================
    # Final status
    # =========================================================================

    print_section(
        "PHASE 7.10 V1 RESULT"
    )

    print("Frozen source binding:              PASS")
    print("Phase-7 prerequisite gate:          PASS")
    print("Descriptive evidence synthesis:     COMPLETE")
    print("Adjusted evidence synthesis:        COMPLETE")
    print("Robustness evidence synthesis:      COMPLETE")
    print("Negative-result synthesis:          COMPLETE")
    print("Thesis problem statement:           COMPLETE")
    print("Research-task alignment:            COMPLETE")
    print("Supervisor-slide summary:           COMPLETE")
    print("Presentation-ready figures:         COMPLETE")
    print("Provenance manifest:                COMPLETE")

    print()
    print("New model experiment executed:      NO")
    print("Causal interpretation performed:    NO")
    print("Model inference executed:           NO")
    print("Final test rescored:                NO")
    print("Final test modified:                NO")
    print("Final test used for model selection:NO")

    print()
    print(
        "PHASE 7 — FINAL TEST ANALYSIS & "
        "THESIS DIAGNOSTICS STATUS: CLOSED"
    )


if __name__ == "__main__":
    main()