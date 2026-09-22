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
# Phase 7.8 — Adjusted Hit@10 Statistical Diagnostics V1
#
# Purpose
# -------
# Test whether the central Phase-7 associations survive mutual adjustment
# using the frozen final-test events.
#
# PRIMARY ENDPOINT
# ----------------
#   Hit@10 = 1[positive_rank <= 10]
#
# PRIMARY ESTIMATOR
# -----------------
# Linear probability model (OLS) with investor-cluster robust (CR1) covariance.
#
# Why a linear probability model?
# -------------------------------
# - coefficients are directly interpretable as adjusted probability
#   differences / percentage-point differences;
# - no odds-ratio translation is required;
# - the goal is post-hoc diagnostic adjustment, not causal modeling.
#
# LPM caveat
# ----------
# This phase reports the range of fitted probabilities and fraction outside
# [0, 1]. Phase 7.9 will handle additional robustness / dependence checks.
#
# PRE-SPECIFIED MODELS
# --------------------
#
# M1 — ENTITY-WARM PAIR NOVELTY
# Population:
#   warm investor + warm startup (prior + new)
#
# Outcome:
#   Hit@10
#
# Predictors:
#   new_to_investor_pair                  [PRIMARY]
#   investor_core_connected
#   startup_core_connected
#   investor_seen_in_t0
#   startup_seen_in_t0
#
# Main estimand:
#   adjusted new-pair minus prior-pair Hit@10 difference.
#
#
# M2 — NEW-PAIR STARTUP COLD START + STRUCTURAL AVAILABILITY
# Population:
#   new-to-investor + warm investor
#
# Outcome:
#   Hit@10
#
# Predictors:
#   cold_start_startup                    [PRIMARY]
#   startup_core_connected
#   cold_start_startup × startup_core_connected
#   investor_core_connected
#   investor_seen_in_t0
#   startup_seen_in_t0
#
# Main estimands:
#   - cold-start startup penalty when startup structurally isolated
#   - cold-start startup penalty when startup structurally connected
#   - structural-connectivity association among warm startups
#   - structural-connectivity association among cold-start startups
#   - difference in structural association between cold and warm startups
#
#
# M3 — STRUCTURAL SOURCE WITHIN NEW COLD-START STARTUPS
# Population:
#   new-to-investor + warm investor + cold-start startup
#
# Outcome:
#   Hit@10
#
# Structural coding:
#   baseline = neither founder nor acquisition support
#
#   founder_signal =
#       founder_only OR founder_and_acquisition
#
#   acquisition_only =
#       acquisition_only
#
# The 3 founder+acquisition cases are merged with founder support rather than
# estimated as an unstable stand-alone coefficient.
#
# Additional adjustment:
#   investor_core_connected
#   investor_seen_in_t0
#
# Main estimands:
#   - founder structural signal vs neither
#   - acquisition-only signal vs neither
#
#
# INTERPRETATION BOUNDARY
# -----------------------
# These are adjusted observational associations. The covariates are not
# asserted to constitute a causal adjustment set, and no causal interpretation
# is permitted.
#
# PROHIBITED
# ----------
# - model loading
# - checkpoint loading
# - raw-logit loading
# - inference
# - test rescoring
# - negative regeneration
# - candidate modification
# - checkpoint/model selection
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

PHASE_7_6B_MANIFEST = (
    REPO_ROOT
    / "data/experimental/phase_7/"
    / "phase_7_6b_temporal_history_performance/"
    / "phase_7_6b_analysis_manifest_V1.json"
)

PHASE_7_7_MANIFEST = (
    REPO_ROOT
    / "data/experimental/phase_7/"
    / "phase_7_7_error_archetypes/"
    / "phase_7_7_analysis_manifest_V1.json"
)

OUT_DIR = (
    REPO_ROOT
    / "data/experimental/phase_7/"
    / "phase_7_8_adjusted_hit10"
)

FIG_DIR = OUT_DIR / "figures"

EXPECTED_SOURCE_SHA256 = (
    "d70f21bff0006e094c5d567d307c0664"
    "b41a11e7250a0e69aaec610e1810132d"
)

EXPECTED_ROWS = 20_264

EXPECTED_M1_N = 10_707
EXPECTED_M2_N = 13_281
EXPECTED_M3_N = 6_392

Z_975 = 1.959963984540054

SOURCE_COLUMNS = [
    "interaction_id",
    "investor_id",
    "startup_id",
    "positive_rank",
    "new_to_investor_pair",
    "cold_start_investor",
    "cold_start_startup",
    "investor_core_connected",
    "startup_core_connected",
    "investor_seen_in_t0",
    "startup_seen_in_t0",
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
            "Author": "ITRS Phase 7 adjusted diagnostics",
        },
    )

    plt.close(fig)

    return [png, pdf]


def fit_cluster_robust_lpm(
    y: np.ndarray,
    X: np.ndarray,
    clusters: np.ndarray,
    column_names: list[str],
) -> dict:
    """
    OLS linear probability model with CR1 investor-cluster robust covariance.

    CR1 correction:
        G/(G-1) * (N-1)/(N-K)
    """

    y = np.asarray(
        y,
        dtype=float,
    ).reshape(-1)

    X = np.asarray(
        X,
        dtype=float,
    )

    clusters = np.asarray(
        clusters
    )

    n, k = X.shape

    require(
        len(y) == n,
        "y and X length mismatch.",
    )

    require(
        len(clusters) == n,
        "cluster and X length mismatch.",
    )

    require(
        len(column_names) == k,
        "column-name count does not match X.",
    )

    rank = int(
        np.linalg.matrix_rank(X)
    )

    require(
        rank == k,
        (
            f"Design matrix is rank deficient: "
            f"rank={rank}, k={k}."
        ),
    )

    xtx = X.T @ X
    xtx_inv = np.linalg.inv(
        xtx
    )

    beta = (
        xtx_inv
        @ X.T
        @ y
    )

    fitted = (
        X
        @ beta
    )

    residual = (
        y
        - fitted
    )

    unique_clusters = pd.unique(
        clusters
    )

    g = len(
        unique_clusters
    )

    require(
        g > 1,
        "Need at least two clusters.",
    )

    meat = np.zeros(
        (
            k,
            k,
        ),
        dtype=float,
    )

    for cluster_id in (
        unique_clusters
    ):
        mask = (
            clusters
            == cluster_id
        )

        Xg = X[
            mask
        ]

        ug = residual[
            mask
        ]

        score = (
            Xg.T
            @ ug
        ).reshape(
            -1,
            1,
        )

        meat += (
            score
            @ score.T
        )

    correction = (
        (g / (g - 1))
        * ((n - 1) / (n - k))
    )

    covariance = (
        correction
        * xtx_inv
        @ meat
        @ xtx_inv
    )

    se = np.sqrt(
        np.maximum(
            np.diag(
                covariance
            ),
            0.0,
        )
    )

    ci_low = (
        beta
        - Z_975
        * se
    )

    ci_high = (
        beta
        + Z_975
        * se
    )

    coefficient_table = pd.DataFrame(
        {
            "term":
                column_names,

            "estimate":
                beta,

            "cluster_robust_se":
                se,

            "ci95_low":
                ci_low,

            "ci95_high":
                ci_high,
        }
    )

    outside = (
        (fitted < 0.0)
        |
        (fitted > 1.0)
    )

    diagnostics = {
        "n":
            int(n),

        "k":
            int(k),

        "rank":
            rank,

        "cluster_count":
            int(g),

        "fitted_min":
            float(
                fitted.min()
            ),

        "fitted_max":
            float(
                fitted.max()
            ),

        "fitted_outside_0_1_n":
            int(
                outside.sum()
            ),

        "fitted_outside_0_1_share":
            float(
                outside.mean()
            ),

        "outcome_mean":
            float(
                y.mean()
            ),

        "residual_mean":
            float(
                residual.mean()
            ),
    }

    return {
        "beta":
            beta,

        "covariance":
            covariance,

        "coefficient_table":
            coefficient_table,

        "diagnostics":
            diagnostics,

        "column_names":
            column_names,
    }


def linear_combination(
    fit: dict,
    weights: dict[str, float],
    label: str,
    estimand_key: str,
    model_key: str,
) -> dict:
    names = fit[
        "column_names"
    ]

    c = np.array(
        [
            weights.get(
                name,
                0.0,
            )
            for name in names
        ],
        dtype=float,
    )

    beta = fit[
        "beta"
    ]

    covariance = fit[
        "covariance"
    ]

    estimate = float(
        c
        @ beta
    )

    variance = float(
        c
        @ covariance
        @ c
    )

    variance = max(
        variance,
        0.0,
    )

    se = float(
        np.sqrt(
            variance
        )
    )

    low = (
        estimate
        - Z_975
        * se
    )

    high = (
        estimate
        + Z_975
        * se
    )

    return {
        "model_key":
            model_key,

        "estimand_key":
            estimand_key,

        "estimand_label":
            label,

        "estimate":
            estimate,

        "cluster_robust_se":
            se,

        "ci95_low":
            low,

        "ci95_high":
            high,
    }


def coefficient_lookup(
    fit: dict,
    term: str,
) -> dict:
    table = fit[
        "coefficient_table"
    ]

    row = table.loc[
        table[
            "term"
        ]
        == term
    ]

    require(
        len(row) == 1,
        f"Expected exactly one coefficient for {term}.",
    )

    return row.iloc[0].to_dict()


def print_estimands(
    table: pd.DataFrame,
) -> None:
    print(
        table[
            [
                "estimand_label",
                "estimate",
                "cluster_robust_se",
                "ci95_low",
                "ci95_high",
            ]
        ].to_string(
            index=False,
            formatters={
                "estimate":
                    lambda x: f"{x:+.6f}",
                "cluster_robust_se":
                    lambda x: f"{x:.6f}",
                "ci95_low":
                    lambda x: f"{x:+.6f}",
                "ci95_high":
                    lambda x: f"{x:+.6f}",
            },
        )
    )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    print("=" * 110)
    print(
        "PHASE 7.8 — ADJUSTED HIT@10 "
        "STATISTICAL DIAGNOSTICS V1"
    )
    print("=" * 110)

    print(
        "Scientific role:             "
        "PRE-SPECIFIED ADJUSTED POST-HOC DIAGNOSTIC"
    )
    print("Primary endpoint:             Hit@10")
    print(
        "Primary estimator:            "
        "Linear probability model + investor-cluster CR1 covariance"
    )
    print("Causal interpretation:        NO")
    print("Model loaded:                 NO")
    print("Checkpoint loaded:            NO")
    print("Raw logits loaded:            NO")
    print("Inference executed:           NO")
    print("Test rescored:                NO")
    print("Model selection performed:   NO")

    # =========================================================================
    # 7.8.1 — Integrity gate
    # =========================================================================

    print_section(
        "7.8.1 — ANALYSIS INTEGRITY GATE"
    )

    prerequisite_paths = [
        PHASE_7_4_MANIFEST,
        PHASE_7_5B_MANIFEST,
        PHASE_7_6B_MANIFEST,
        PHASE_7_7_MANIFEST,
    ]

    for path in (
        SOURCE,
        *prerequisite_paths,
    ):
        require(
            path.exists(),
            f"Missing required artifact: {path}",
        )

        print(
            "FOUND  "
            f"{path.relative_to(REPO_ROOT)}"
        )

    source_hash = file_sha256(
        SOURCE
    )

    require(
        source_hash
        == EXPECTED_SOURCE_SHA256,
        "Frozen source SHA256 drift.",
    )

    for path in (
        prerequisite_paths
    ):
        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            manifest = json.load(f)

        require(
            manifest[
                "status"
            ]
            == "COMPLETE",
            (
                f"Prerequisite manifest is not "
                f"COMPLETE: {path}"
            ),
        )

    print()
    print("Frozen source fingerprint:   PASS")
    print("Phase 7.4 prerequisite:       PASS")
    print("Phase 7.5B prerequisite:      PASS")
    print("Phase 7.6B prerequisite:      PASS")
    print("Phase 7.7 prerequisite:       PASS")

    # =========================================================================
    # 7.8.2 — Load frozen fields
    # =========================================================================

    print_section(
        "7.8.2 — LOAD FROZEN HIT@10 AND PRE-SPECIFIED COVARIATES"
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
        df[
            "interaction_id"
        ].is_unique,
        "interaction_id is not unique.",
    )

    require(
        df[
            "positive_rank"
        ].between(
            1,
            100,
        ).all(),
        "positive_rank outside 1..100.",
    )

    require(
        df.notna().all().all(),
        "Required Phase-7.8 fields contain missing values.",
    )

    df[
        "hit10"
    ] = (
        df[
            "positive_rank"
        ]
        .le(10)
        .astype(float)
    )

    print(
        f"Test cases:                   "
        f"{len(df):,}"
    )

    print(
        f"Overall Hit@10 mean:          "
        f"{df['hit10'].mean():.6f}"
    )

    # =========================================================================
    # 7.8.3 — M1: adjusted entity-warm pair novelty
    # =========================================================================

    print_section(
        "7.8.3 — M1: ADJUSTED ENTITY-WARM PAIR NOVELTY"
    )

    warm_warm = (
        ~df[
            "cold_start_investor"
        ].astype(bool)
        &
        ~df[
            "cold_start_startup"
        ].astype(bool)
    )

    m1 = df.loc[
        warm_warm
    ].copy()

    require(
        len(m1)
        == EXPECTED_M1_N,
        "M1 population count drift.",
    )

    X1 = np.column_stack(
        [
            np.ones(
                len(m1),
                dtype=float,
            ),

            m1[
                "new_to_investor_pair"
            ].astype(float),

            m1[
                "investor_core_connected"
            ].astype(float),

            m1[
                "startup_core_connected"
            ].astype(float),

            m1[
                "investor_seen_in_t0"
            ].astype(float),

            m1[
                "startup_seen_in_t0"
            ].astype(float),
        ]
    )

    m1_columns = [
        "intercept",
        "new_to_investor_pair",
        "investor_core_connected",
        "startup_core_connected",
        "investor_seen_in_t0",
        "startup_seen_in_t0",
    ]

    fit1 = fit_cluster_robust_lpm(
        y=m1[
            "hit10"
        ].to_numpy(),
        X=X1,
        clusters=m1[
            "investor_id"
        ].to_numpy(),
        column_names=m1_columns,
    )

    m1_estimand = linear_combination(
        fit=fit1,
        weights={
            "new_to_investor_pair":
                1.0,
        },
        label=(
            "Adjusted new-pair - prior-pair Hit@10 "
            "within warm/warm endpoints"
        ),
        estimand_key=(
            "m1_adjusted_new_vs_prior_warm_warm"
        ),
        model_key="M1",
    )

    print(
        f"Population N:                 "
        f"{fit1['diagnostics']['n']:,}"
    )

    print(
        f"Investor clusters:            "
        f"{fit1['diagnostics']['cluster_count']:,}"
    )

    print()
    print_estimands(
        pd.DataFrame(
            [
                m1_estimand
            ]
        )
    )

    print()
    print(
        "LPM fitted-probability diagnostic:"
    )

    print(
        f"  range:                      "
        f"[{fit1['diagnostics']['fitted_min']:.4f}, "
        f"{fit1['diagnostics']['fitted_max']:.4f}]"
    )

    print(
        f"  outside [0,1]:              "
        f"{fit1['diagnostics']['fitted_outside_0_1_n']:,} "
        f"({fit1['diagnostics']['fitted_outside_0_1_share']:.2%})"
    )

    # =========================================================================
    # 7.8.4 — M2: startup cold-start + structure within new warm investors
    # =========================================================================

    print_section(
        "7.8.4 — M2: NEW-PAIR STARTUP COLD START × STRUCTURAL AVAILABILITY"
    )

    new_warm_investor = (
        df[
            "new_to_investor_pair"
        ].astype(bool)
        &
        ~df[
            "cold_start_investor"
        ].astype(bool)
    )

    m2 = df.loc[
        new_warm_investor
    ].copy()

    require(
        len(m2)
        == EXPECTED_M2_N,
        "M2 population count drift.",
    )

    m2[
        "cold_startup_x_structure"
    ] = (
        m2[
            "cold_start_startup"
        ].astype(float)
        *
        m2[
            "startup_core_connected"
        ].astype(float)
    )

    X2 = np.column_stack(
        [
            np.ones(
                len(m2),
                dtype=float,
            ),

            m2[
                "cold_start_startup"
            ].astype(float),

            m2[
                "startup_core_connected"
            ].astype(float),

            m2[
                "cold_startup_x_structure"
            ].astype(float),

            m2[
                "investor_core_connected"
            ].astype(float),

            m2[
                "investor_seen_in_t0"
            ].astype(float),

            m2[
                "startup_seen_in_t0"
            ].astype(float),
        ]
    )

    m2_columns = [
        "intercept",
        "cold_start_startup",
        "startup_core_connected",
        "cold_startup_x_structure",
        "investor_core_connected",
        "investor_seen_in_t0",
        "startup_seen_in_t0",
    ]

    fit2 = fit_cluster_robust_lpm(
        y=m2[
            "hit10"
        ].to_numpy(),
        X=X2,
        clusters=m2[
            "investor_id"
        ].to_numpy(),
        column_names=m2_columns,
    )

    m2_estimands = [
        linear_combination(
            fit=fit2,
            weights={
                "cold_start_startup":
                    1.0,
            },
            label=(
                "Adjusted cold-start startup penalty "
                "when startup structurally isolated"
            ),
            estimand_key=(
                "m2_cold_start_penalty_if_isolated"
            ),
            model_key="M2",
        ),

        linear_combination(
            fit=fit2,
            weights={
                "cold_start_startup":
                    1.0,

                "cold_startup_x_structure":
                    1.0,
            },
            label=(
                "Adjusted cold-start startup penalty "
                "when startup structurally connected"
            ),
            estimand_key=(
                "m2_cold_start_penalty_if_connected"
            ),
            model_key="M2",
        ),

        linear_combination(
            fit=fit2,
            weights={
                "startup_core_connected":
                    1.0,
            },
            label=(
                "Adjusted startup-structure association "
                "among warm startups"
            ),
            estimand_key=(
                "m2_structure_effect_among_warm_startups"
            ),
            model_key="M2",
        ),

        linear_combination(
            fit=fit2,
            weights={
                "startup_core_connected":
                    1.0,

                "cold_startup_x_structure":
                    1.0,
            },
            label=(
                "Adjusted startup-structure association "
                "among cold-start startups"
            ),
            estimand_key=(
                "m2_structure_effect_among_cold_startups"
            ),
            model_key="M2",
        ),

        linear_combination(
            fit=fit2,
            weights={
                "cold_startup_x_structure":
                    1.0,
            },
            label=(
                "Difference in startup-structure association: "
                "cold-start minus warm startup"
            ),
            estimand_key=(
                "m2_structure_interaction_cold_minus_warm"
            ),
            model_key="M2",
        ),
    ]

    m2_estimands_df = pd.DataFrame(
        m2_estimands
    )

    print(
        f"Population N:                 "
        f"{fit2['diagnostics']['n']:,}"
    )

    print(
        f"Investor clusters:            "
        f"{fit2['diagnostics']['cluster_count']:,}"
    )

    print()
    print_estimands(
        m2_estimands_df
    )

    print()
    print(
        "LPM fitted-probability diagnostic:"
    )

    print(
        f"  range:                      "
        f"[{fit2['diagnostics']['fitted_min']:.4f}, "
        f"{fit2['diagnostics']['fitted_max']:.4f}]"
    )

    print(
        f"  outside [0,1]:              "
        f"{fit2['diagnostics']['fitted_outside_0_1_n']:,} "
        f"({fit2['diagnostics']['fitted_outside_0_1_share']:.2%})"
    )

    # =========================================================================
    # 7.8.5 — M3: structural source within new cold-start startups
    # =========================================================================

    print_section(
        "7.8.5 — M3: COLD-START STARTUP STRUCTURAL SOURCE"
    )

    new_wc = (
        df[
            "new_to_investor_pair"
        ].astype(bool)
        &
        ~df[
            "cold_start_investor"
        ].astype(bool)
        &
        df[
            "cold_start_startup"
        ].astype(bool)
    )

    m3 = df.loc[
        new_wc
    ].copy()

    require(
        len(m3)
        == EXPECTED_M3_N,
        "M3 population count drift.",
    )

    valid_source_classes = {
        "founder_only",
        "acquisition_only",
        "founder_and_acquisition",
        "neither",
    }

    require(
        set(
            m3[
                "startup_structural_source_class"
            ].unique()
        )
        <= valid_source_classes,
        "Unexpected startup source class in M3.",
    )

    m3[
        "founder_signal"
    ] = (
        m3[
            "startup_structural_source_class"
        ].isin(
            [
                "founder_only",
                "founder_and_acquisition",
            ]
        )
        .astype(float)
    )

    m3[
        "acquisition_only_signal"
    ] = (
        m3[
            "startup_structural_source_class"
        ]
        .eq(
            "acquisition_only"
        )
        .astype(float)
    )

    source_counts = (
        m3[
            "startup_structural_source_class"
        ]
        .value_counts()
    )

    print(
        "Source-class counts:"
    )

    for key in (
        "founder_only",
        "acquisition_only",
        "founder_and_acquisition",
        "neither",
    ):
        print(
            f"  {key:<24} "
            f"{int(source_counts.get(key, 0)):>6,}"
        )

    X3 = np.column_stack(
        [
            np.ones(
                len(m3),
                dtype=float,
            ),

            m3[
                "founder_signal"
            ].astype(float),

            m3[
                "acquisition_only_signal"
            ].astype(float),

            m3[
                "investor_core_connected"
            ].astype(float),

            m3[
                "investor_seen_in_t0"
            ].astype(float),
        ]
    )

    m3_columns = [
        "intercept",
        "founder_signal",
        "acquisition_only_signal",
        "investor_core_connected",
        "investor_seen_in_t0",
    ]

    fit3 = fit_cluster_robust_lpm(
        y=m3[
            "hit10"
        ].to_numpy(),
        X=X3,
        clusters=m3[
            "investor_id"
        ].to_numpy(),
        column_names=m3_columns,
    )

    m3_estimands = [
        linear_combination(
            fit=fit3,
            weights={
                "founder_signal":
                    1.0,
            },
            label=(
                "Adjusted founder structural signal "
                "vs neither"
            ),
            estimand_key=(
                "m3_founder_signal_vs_neither"
            ),
            model_key="M3",
        ),

        linear_combination(
            fit=fit3,
            weights={
                "acquisition_only_signal":
                    1.0,
            },
            label=(
                "Adjusted acquisition-only structural signal "
                "vs neither"
            ),
            estimand_key=(
                "m3_acquisition_only_vs_neither"
            ),
            model_key="M3",
        ),
    ]

    m3_estimands_df = pd.DataFrame(
        m3_estimands
    )

    print()
    print(
        f"Population N:                 "
        f"{fit3['diagnostics']['n']:,}"
    )

    print(
        f"Investor clusters:            "
        f"{fit3['diagnostics']['cluster_count']:,}"
    )

    print()
    print_estimands(
        m3_estimands_df
    )

    print()
    print(
        "LPM fitted-probability diagnostic:"
    )

    print(
        f"  range:                      "
        f"[{fit3['diagnostics']['fitted_min']:.4f}, "
        f"{fit3['diagnostics']['fitted_max']:.4f}]"
    )

    print(
        f"  outside [0,1]:              "
        f"{fit3['diagnostics']['fitted_outside_0_1_n']:,} "
        f"({fit3['diagnostics']['fitted_outside_0_1_share']:.2%})"
    )

    # =========================================================================
    # 7.8.6 — Consolidate adjusted thesis estimands
    # =========================================================================

    print_section(
        "7.8.6 — CONSOLIDATED ADJUSTED THESIS ESTIMANDS"
    )

    all_estimands = pd.concat(
        [
            pd.DataFrame(
                [
                    m1_estimand
                ]
            ),
            m2_estimands_df,
            m3_estimands_df,
        ],
        ignore_index=True,
    )

    print_estimands(
        all_estimands
    )

    # =========================================================================
    # 7.8.7 — Interpretation scaffold
    # =========================================================================

    print_section(
        "7.8.7 — INTERPRETATION SCAFFOLD"
    )

    print(
        "Questions this phase can support:"
    )

    print(
        "  1) Does the warm/warm new-pair gap persist "
        "after adjustment for structural coverage and "
        "literal T0 exposure indicators?"
    )

    print(
        "  2) Within new pairs and warm investors, does "
        "startup cold start remain strongly associated "
        "with lower Hit@10 after structural/history adjustment?"
    )

    print(
        "  3) Is startup structural availability still "
        "associated with higher Hit@10 specifically among "
        "cold-start startups after adjustment?"
    )

    print(
        "  4) Is the structural association concentrated "
        "in founder-related support rather than acquisition-only support?"
    )

    print()
    print(
        "What this phase CANNOT support:"
    )

    print(
        "  - causal effects;"
    )

    print(
        "  - identification of the mechanism by which "
        "structure helps;"
    )

    print(
        "  - proof that a proposed inductive model will improve results;"
    )

    print(
        "  - full dependence robustness beyond investor clustering."
    )

    # =========================================================================
    # 7.8.8 — Write artifacts
    # =========================================================================

    print_section(
        "7.8.8 — WRITE ANALYSIS ARTIFACTS"
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    m1_coef_path = (
        OUT_DIR
        / "phase_7_8_M1_warm_warm_novelty_coefficients_V1.csv"
    )

    m2_coef_path = (
        OUT_DIR
        / "phase_7_8_M2_cold_start_structure_coefficients_V1.csv"
    )

    m3_coef_path = (
        OUT_DIR
        / "phase_7_8_M3_structural_source_coefficients_V1.csv"
    )

    estimands_path = (
        OUT_DIR
        / "phase_7_8_adjusted_estimands_V1.csv"
    )

    diagnostics_path = (
        OUT_DIR
        / "phase_7_8_model_diagnostics_V1.json"
    )

    evidence_path = (
        OUT_DIR
        / "phase_7_8_thesis_evidence_V1.json"
    )

    fit1[
        "coefficient_table"
    ].to_csv(
        m1_coef_path,
        index=False,
    )

    fit2[
        "coefficient_table"
    ].to_csv(
        m2_coef_path,
        index=False,
    )

    fit3[
        "coefficient_table"
    ].to_csv(
        m3_coef_path,
        index=False,
    )

    all_estimands.to_csv(
        estimands_path,
        index=False,
    )

    diagnostics = {
        "M1":
            fit1[
                "diagnostics"
            ],

        "M2":
            fit2[
                "diagnostics"
            ],

        "M3":
            fit3[
                "diagnostics"
            ],
    }

    json_dump(
        diagnostics,
        diagnostics_path,
    )

    evidence = {
        "phase":
            "7.8",

        "primary_endpoint":
            "Hit@10",

        "estimator":
            (
                "Linear probability model with "
                "investor-cluster robust CR1 covariance."
            ),

        "M1":
            {
                "population":
                    (
                        "warm investor + warm startup"
                    ),

                "primary_estimand":
                    m1_estimand,
            },

        "M2":
            {
                "population":
                    (
                        "new-to-investor + warm investor"
                    ),

                "estimands":
                    m2_estimands,
            },

        "M3":
            {
                "population":
                    (
                        "new-to-investor + warm investor "
                        "+ cold-start startup"
                    ),

                "estimands":
                    m3_estimands,

                "founder_and_acquisition_handling":
                    (
                        "Merged with founder support "
                        "because N=3."
                    ),
            },

        "causal_interpretation":
            False,

        "dependence_limitation":
            (
                "Investor clustering only. "
                "Startup and multiway robustness "
                "deferred to Phase 7.9."
            ),
    }

    json_dump(
        evidence,
        evidence_path,
    )

    # =========================================================================
    # 7.8.9 — Figures
    # =========================================================================

    print_section(
        "7.8.9 — GENERATE PRESENTATION-READY FIGURES"
    )

    figure_paths = []

    # Figure 1 — selected adjusted effects
    selected_keys = [
        "m1_adjusted_new_vs_prior_warm_warm",
        "m2_cold_start_penalty_if_isolated",
        "m2_cold_start_penalty_if_connected",
        "m2_structure_effect_among_cold_startups",
        "m3_founder_signal_vs_neither",
        "m3_acquisition_only_vs_neither",
    ]

    selected = (
        all_estimands.loc[
            all_estimands[
                "estimand_key"
            ].isin(
                selected_keys
            )
        ]
        .set_index(
            "estimand_key"
        )
        .loc[
            selected_keys
        ]
        .reset_index()
    )

    display_labels = {
        "m1_adjusted_new_vs_prior_warm_warm":
            "New vs prior\n(warm/warm)",

        "m2_cold_start_penalty_if_isolated":
            "Cold-start penalty\n(startup isolated)",

        "m2_cold_start_penalty_if_connected":
            "Cold-start penalty\n(startup connected)",

        "m2_structure_effect_among_cold_startups":
            "Startup structure\nwithin cold start",

        "m3_founder_signal_vs_neither":
            "Founder signal\nvs neither",

        "m3_acquisition_only_vs_neither":
            "Acquisition-only\nvs neither",
    }

    estimates = (
        selected[
            "estimate"
        ].to_numpy()
        * 100.0
    )

    lows = (
        selected[
            "ci95_low"
        ].to_numpy()
        * 100.0
    )

    highs = (
        selected[
            "ci95_high"
        ].to_numpy()
        * 100.0
    )

    y = np.arange(
        len(selected)
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
            display_labels[
                key
            ]
            for key in selected[
                "estimand_key"
            ]
        ],
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Adjusted Hit@10 difference (percentage points)"
    )

    ax.set_title(
        "Adjusted Phase-7 thesis diagnostics\n"
        "Investor-cluster robust 95% intervals"
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
                "phase_7_8_fig1_"
                "adjusted_hit10_forest_V1"
            ),
        )
    )

    # Figure 2 — M2 conceptual 2x2 adjusted contrasts
    m2_plot_keys = [
        "m2_structure_effect_among_warm_startups",
        "m2_structure_effect_among_cold_startups",
        "m2_structure_interaction_cold_minus_warm",
    ]

    m2_plot = (
        m2_estimands_df.loc[
            m2_estimands_df[
                "estimand_key"
            ].isin(
                m2_plot_keys
            )
        ]
        .set_index(
            "estimand_key"
        )
        .loc[
            m2_plot_keys
        ]
        .reset_index()
    )

    m2_labels = {
        "m2_structure_effect_among_warm_startups":
            "Structure association\nwarm startups",

        "m2_structure_effect_among_cold_startups":
            "Structure association\ncold-start startups",

        "m2_structure_interaction_cold_minus_warm":
            "Difference:\ncold minus warm",
    }

    estimates = (
        m2_plot[
            "estimate"
        ].to_numpy()
        * 100.0
    )

    lows = (
        m2_plot[
            "ci95_low"
        ].to_numpy()
        * 100.0
    )

    highs = (
        m2_plot[
            "ci95_high"
        ].to_numpy()
        * 100.0
    )

    x = np.arange(
        len(
            m2_plot
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.errorbar(
        x,
        estimates,
        yerr=[
            estimates - lows,
            highs - estimates,
        ],
        fmt="o",
        capsize=4,
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_xticks(
        x,
        [
            m2_labels[
                key
            ]
            for key in m2_plot[
                "estimand_key"
            ]
        ],
    )

    ax.set_ylabel(
        "Adjusted Hit@10 difference (percentage points)"
    )

    ax.set_title(
        "Does startup structural availability matter more under cold start?"
    )

    ax.grid(
        axis="y",
        alpha=0.20,
    )

    fig.tight_layout()

    figure_paths.extend(
        save_figure(
            fig,
            (
                "phase_7_8_fig2_"
                "cold_start_structure_interaction_V1"
            ),
        )
    )

    # =========================================================================
    # 7.8.10 — Manifest / hashes
    # =========================================================================

    manifest_path = (
        OUT_DIR
        / "phase_7_8_analysis_manifest_V1.json"
    )

    manifest = {
        "phase":
            "7.8",

        "schema_version":
            "ITRS_PHASE7_8_ADJUSTED_HIT10_V1",

        "status":
            "COMPLETE",

        "scientific_role":
            (
                "Adjusted post-hoc Hit@10 diagnostic "
                "using pre-specified linear probability models."
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

        "primary_endpoint":
            "Hit@10",

        "estimator":
            {
                "model":
                    "linear_probability_model",

                "covariance":
                    "investor_cluster_CR1",

                "ci":
                    "normal_95_percent",
            },

        "model_specs": {
            "M1":
                m1_columns,

            "M2":
                m2_columns,

            "M3":
                m3_columns,
        },

        "causal_interpretation":
            False,

        "model_selection":
            False,

        "dependence_limitation":
            (
                "Investor clustering only; additional "
                "startup / multiway robustness deferred "
                "to Phase 7.9."
            ),

        "figures": [
            str(
                p.relative_to(
                    REPO_ROOT
                )
            )
            for p in figure_paths
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

            "checkpoint_or_model_selection":
                False,
        },
    }

    json_dump(
        manifest,
        manifest_path,
    )

    artifact_paths = [
        m1_coef_path,
        m2_coef_path,
        m3_coef_path,
        estimands_path,
        diagnostics_path,
        evidence_path,
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
        / "phase_7_8_derived_artifact_sha256_V1.json"
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
        "PHASE 7.8 V1 RESULT"
    )

    print("Frozen source binding:              PASS")
    print("Primary endpoint Hit@10:            PASS")
    print("M1 adjusted novelty diagnostic:     COMPLETE")
    print("M2 cold-start/structure diagnostic: COMPLETE")
    print("M3 structural-source diagnostic:    COMPLETE")
    print("Investor-cluster CR1 uncertainty:   COMPLETE")
    print("LPM probability diagnostics:        COMPLETE")
    print("Presentation-ready figures:         COMPLETE")
    print("Provenance manifest:                COMPLETE")

    print()
    print("Causal interpretation performed:    NO")
    print("Model inference executed:           NO")
    print("Final test rescored:                NO")
    print("Final test modified:                NO")

    print()
    print(
        "PHASE 7.8 ADJUSTED HIT@10 "
        "STATISTICAL DIAGNOSTICS STATUS: COMPLETE"
    )


if __name__ == "__main__":
    main()