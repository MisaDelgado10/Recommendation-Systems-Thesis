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
# Phase 7.9 — Dependence & Weighting Robustness Diagnostics V1
#
# Purpose
# -------
# Stress-test the central Phase-7.8 adjusted Hit@10 estimands against:
#
#   A) dependence assumptions:
#        - investor-cluster CR1        [Phase-7.8 baseline]
#        - startup-cluster CR1
#        - investor-startup pair CR1
#        - two-way investor + startup CR1
#
#   B) event weighting:
#        - original event-level analysis
#        - pair-balanced analysis, where repeated investor-startup events
#          are collapsed to one pair row with mean Hit@10
#
# No model specification is searched or selected in this phase.
# The Phase-7.8 model formulas and estimands are reused exactly.
#
# Why pair-balanced robustness?
# -----------------------------
# The frozen final test is event-level, which remains the PRIMARY evaluation
# unit. However, investors and startups can occur in multiple test events, and
# some investor-startup pairs can also repeat. Pair-balanced analysis checks
# whether the main adjusted findings are driven by repeated pair exposure.
#
# Two-way clustered covariance
# ----------------------------
# Cameron-Gelbach-Miller style inclusion-exclusion:
#
#     V_two_way = V_investor + V_startup - V_pair
#
# where each component uses the same CR1 finite-sample correction.
#
# IMPORTANT
# ---------
# This phase changes ONLY statistical summarization / weighting for robustness.
# It does NOT:
#   - retrain
#   - load the model/checkpoint/logits
#   - rerun inference
#   - rescore the final test
#   - change candidates/negatives
#   - perform model selection
#
# Interpretation remains observational, not causal.
# =============================================================================


REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    REPO_ROOT
    / "data/experimental/phase_6/full_training/100pct/final_test/"
    / "analysis_ready_test_cases.parquet"
)

PHASE_7_8_MANIFEST = (
    REPO_ROOT
    / "data/experimental/phase_7/"
    / "phase_7_8_adjusted_hit10/"
    / "phase_7_8_analysis_manifest_V1.json"
)

PHASE_7_8_ESTIMANDS = (
    REPO_ROOT
    / "data/experimental/phase_7/"
    / "phase_7_8_adjusted_hit10/"
    / "phase_7_8_adjusted_estimands_V1.csv"
)

OUT_DIR = (
    REPO_ROOT
    / "data/experimental/phase_7/"
    / "phase_7_9_dependence_robustness"
)

FIG_DIR = OUT_DIR / "figures"

EXPECTED_SOURCE_SHA256 = (
    "d70f21bff0006e094c5d567d307c0664"
    "b41a11e7250a0e69aaec610e1810132d"
)

EXPECTED_ROWS = 20_264

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

KEY_ESTIMAND_ORDER = [
    "m1_adjusted_new_vs_prior_warm_warm",
    "m2_cold_start_penalty_if_isolated",
    "m2_cold_start_penalty_if_connected",
    "m2_structure_effect_among_warm_startups",
    "m2_structure_effect_among_cold_startups",
    "m2_structure_interaction_cold_minus_warm",
    "m3_founder_signal_vs_neither",
    "m3_acquisition_only_vs_neither",
]

ESTIMAND_LABELS = {
    "m1_adjusted_new_vs_prior_warm_warm":
        "New vs prior | warm/warm",

    "m2_cold_start_penalty_if_isolated":
        "Cold-start penalty | startup isolated",

    "m2_cold_start_penalty_if_connected":
        "Cold-start penalty | startup connected",

    "m2_structure_effect_among_warm_startups":
        "Startup structure | warm startups",

    "m2_structure_effect_among_cold_startups":
        "Startup structure | cold-start startups",

    "m2_structure_interaction_cold_minus_warm":
        "Structure interaction | cold minus warm",

    "m3_founder_signal_vs_neither":
        "Founder signal vs neither",

    "m3_acquisition_only_vs_neither":
        "Acquisition-only vs neither",
}


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
            "Author": "ITRS Phase 7 robustness diagnostics",
        },
    )

    plt.close(fig)

    return [png, pdf]


def fit_ols(
    y: np.ndarray,
    X: np.ndarray,
    column_names: list[str],
) -> dict:
    y = np.asarray(
        y,
        dtype=float,
    ).reshape(-1)

    X = np.asarray(
        X,
        dtype=float,
    )

    n, k = X.shape

    require(
        len(y) == n,
        "y/X length mismatch.",
    )

    require(
        len(column_names) == k,
        "column-name count mismatch.",
    )

    rank = int(
        np.linalg.matrix_rank(X)
    )

    require(
        rank == k,
        f"Design rank deficient: rank={rank}, k={k}.",
    )

    xtx_inv = np.linalg.inv(
        X.T @ X
    )

    beta = (
        xtx_inv
        @ X.T
        @ y
    )

    fitted = X @ beta
    residual = y - fitted

    return {
        "y": y,
        "X": X,
        "n": n,
        "k": k,
        "rank": rank,
        "xtx_inv": xtx_inv,
        "beta": beta,
        "fitted": fitted,
        "residual": residual,
        "column_names": column_names,
    }


def one_way_cluster_covariance(
    fit: dict,
    clusters: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """
    CR1 one-way cluster-robust covariance.
    """

    X = fit["X"]
    residual = fit["residual"]
    xtx_inv = fit["xtx_inv"]

    clusters = np.asarray(
        clusters
    )

    require(
        len(clusters) == fit["n"],
        "Cluster vector length mismatch.",
    )

    unique_clusters = pd.unique(
        clusters
    )

    g = len(
        unique_clusters
    )

    require(
        g > 1,
        "Need >1 cluster.",
    )

    meat = np.zeros(
        (
            fit["k"],
            fit["k"],
        ),
        dtype=float,
    )

    for cluster_id in unique_clusters:
        mask = (
            clusters
            == cluster_id
        )

        score = (
            X[
                mask
            ].T
            @ residual[
                mask
            ]
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
        * (
            (fit["n"] - 1)
            / (fit["n"] - fit["k"])
        )
    )

    covariance = (
        correction
        * xtx_inv
        @ meat
        @ xtx_inv
    )

    diagnostics = {
        "cluster_count":
            int(g),

        "cr1_correction":
            float(correction),
    }

    return (
        covariance,
        diagnostics,
    )


def two_way_cluster_covariance(
    fit: dict,
    investor_clusters: np.ndarray,
    startup_clusters: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """
    Two-way CR1 covariance:
        V_investor + V_startup - V_pair
    """

    investor_clusters = np.asarray(
        investor_clusters
    )

    startup_clusters = np.asarray(
        startup_clusters
    )

    require(
        len(investor_clusters)
        == fit["n"],
        "Investor cluster length mismatch.",
    )

    require(
        len(startup_clusters)
        == fit["n"],
        "Startup cluster length mismatch.",
    )

    pair_clusters = np.array(
        [
            f"{inv}|||{startup}"
            for inv, startup in zip(
                investor_clusters,
                startup_clusters,
            )
        ],
        dtype=object,
    )

    v_inv, d_inv = (
        one_way_cluster_covariance(
            fit,
            investor_clusters,
        )
    )

    v_startup, d_startup = (
        one_way_cluster_covariance(
            fit,
            startup_clusters,
        )
    )

    v_pair, d_pair = (
        one_way_cluster_covariance(
            fit,
            pair_clusters,
        )
    )

    covariance = (
        v_inv
        + v_startup
        - v_pair
    )

    diagnostics = {
        "investor_cluster_count":
            d_inv[
                "cluster_count"
            ],

        "startup_cluster_count":
            d_startup[
                "cluster_count"
            ],

        "pair_cluster_count":
            d_pair[
                "cluster_count"
            ],

        "investor_cr1_correction":
            d_inv[
                "cr1_correction"
            ],

        "startup_cr1_correction":
            d_startup[
                "cr1_correction"
            ],

        "pair_cr1_correction":
            d_pair[
                "cr1_correction"
            ],
    }

    return (
        covariance,
        diagnostics,
    )


def covariance_variance_diagnostics(
    covariance: np.ndarray,
) -> dict:
    eigenvalues = np.linalg.eigvalsh(
        (
            covariance
            + covariance.T
        )
        / 2.0
    )

    return {
        "min_eigenvalue":
            float(
                eigenvalues.min()
            ),

        "negative_eigenvalue_n":
            int(
                (
                    eigenvalues
                    < -1e-12
                ).sum()
            ),
    }


def linear_combination(
    fit: dict,
    covariance: np.ndarray,
    weights: dict[str, float],
    estimand_key: str,
    estimand_label: str,
    model_key: str,
    analysis_unit: str,
    covariance_type: str,
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

    estimate = float(
        c
        @ fit["beta"]
    )

    raw_variance = float(
        c
        @ covariance
        @ c
    )

    require(
        raw_variance >= -1e-10,
        (
            f"Negative estimand variance "
            f"{raw_variance} for {estimand_key} "
            f"under {covariance_type}."
        ),
    )

    variance = max(
        raw_variance,
        0.0,
    )

    se = float(
        np.sqrt(
            variance
        )
    )

    return {
        "model_key":
            model_key,

        "analysis_unit":
            analysis_unit,

        "covariance_type":
            covariance_type,

        "estimand_key":
            estimand_key,

        "estimand_label":
            estimand_label,

        "estimate":
            estimate,

        "se":
            se,

        "ci95_low":
            estimate
            - Z_975
            * se,

        "ci95_high":
            estimate
            + Z_975
            * se,

        "raw_variance":
            raw_variance,
    }


def model_specs_for_dataframe(
    df: pd.DataFrame,
) -> dict[str, dict]:
    """
    Reconstruct the exact Phase-7.8 model populations/designs.
    """

    specs = {}

    # -------------------------------------------------------------------------
    # M1
    # -------------------------------------------------------------------------

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

    fit1 = fit_ols(
        y=m1[
            "hit10"
        ].to_numpy(),
        X=X1,
        column_names=m1_columns,
    )

    specs["M1"] = {
        "data":
            m1,

        "fit":
            fit1,

        "estimands":
            [
                {
                    "estimand_key":
                        "m1_adjusted_new_vs_prior_warm_warm",

                    "estimand_label":
                        (
                            "Adjusted new-pair - prior-pair "
                            "Hit@10 within warm/warm endpoints"
                        ),

                    "weights":
                        {
                            "new_to_investor_pair":
                                1.0,
                        },
                },
            ],
    }

    # -------------------------------------------------------------------------
    # M2
    # -------------------------------------------------------------------------

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

    fit2 = fit_ols(
        y=m2[
            "hit10"
        ].to_numpy(),
        X=X2,
        column_names=m2_columns,
    )

    specs["M2"] = {
        "data":
            m2,

        "fit":
            fit2,

        "estimands":
            [
                {
                    "estimand_key":
                        "m2_cold_start_penalty_if_isolated",

                    "estimand_label":
                        (
                            "Adjusted cold-start startup penalty "
                            "when startup structurally isolated"
                        ),

                    "weights":
                        {
                            "cold_start_startup":
                                1.0,
                        },
                },

                {
                    "estimand_key":
                        "m2_cold_start_penalty_if_connected",

                    "estimand_label":
                        (
                            "Adjusted cold-start startup penalty "
                            "when startup structurally connected"
                        ),

                    "weights":
                        {
                            "cold_start_startup":
                                1.0,

                            "cold_startup_x_structure":
                                1.0,
                        },
                },

                {
                    "estimand_key":
                        "m2_structure_effect_among_warm_startups",

                    "estimand_label":
                        (
                            "Adjusted startup-structure association "
                            "among warm startups"
                        ),

                    "weights":
                        {
                            "startup_core_connected":
                                1.0,
                        },
                },

                {
                    "estimand_key":
                        "m2_structure_effect_among_cold_startups",

                    "estimand_label":
                        (
                            "Adjusted startup-structure association "
                            "among cold-start startups"
                        ),

                    "weights":
                        {
                            "startup_core_connected":
                                1.0,

                            "cold_startup_x_structure":
                                1.0,
                        },
                },

                {
                    "estimand_key":
                        "m2_structure_interaction_cold_minus_warm",

                    "estimand_label":
                        (
                            "Difference in startup-structure association: "
                            "cold-start minus warm startup"
                        ),

                    "weights":
                        {
                            "cold_startup_x_structure":
                                1.0,
                        },
                },
            ],
    }

    # -------------------------------------------------------------------------
    # M3
    # -------------------------------------------------------------------------

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

    fit3 = fit_ols(
        y=m3[
            "hit10"
        ].to_numpy(),
        X=X3,
        column_names=m3_columns,
    )

    specs["M3"] = {
        "data":
            m3,

        "fit":
            fit3,

        "estimands":
            [
                {
                    "estimand_key":
                        "m3_founder_signal_vs_neither",

                    "estimand_label":
                        (
                            "Adjusted founder structural "
                            "signal vs neither"
                        ),

                    "weights":
                        {
                            "founder_signal":
                                1.0,
                        },
                },

                {
                    "estimand_key":
                        "m3_acquisition_only_vs_neither",

                    "estimand_label":
                        (
                            "Adjusted acquisition-only structural "
                            "signal vs neither"
                        ),

                    "weights":
                        {
                            "acquisition_only_signal":
                                1.0,
                        },
                },
            ],
    }

    return specs


def collapse_to_pair_level(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Collapse repeated investor-startup events to one pair row with mean Hit@10.

    All model predictors must be invariant within investor-startup pair.
    """

    invariant_columns = [
        "new_to_investor_pair",
        "cold_start_investor",
        "cold_start_startup",
        "investor_core_connected",
        "startup_core_connected",
        "investor_seen_in_t0",
        "startup_seen_in_t0",
        "startup_structural_source_class",
    ]

    grouped = df.groupby(
        [
            "investor_id",
            "startup_id",
        ],
        sort=False,
        dropna=False,
    )

    audit_rows = []

    for col in invariant_columns:
        max_nunique = int(
            grouped[col]
            .nunique(
                dropna=False
            )
            .max()
        )

        audit_rows.append(
            {
                "field":
                    col,

                "max_within_pair_nunique":
                    max_nunique,

                "pair_invariant":
                    bool(
                        max_nunique
                        == 1
                    ),
            }
        )

        require(
            max_nunique == 1,
            (
                f"Pair collapse invalid: "
                f"{col} varies within pair."
            ),
        )

    pair_df = (
        grouped.agg(
            event_n=(
                "interaction_id",
                "size",
            ),

            hit10=(
                "hit10",
                "mean",
            ),

            new_to_investor_pair=(
                "new_to_investor_pair",
                "first",
            ),

            cold_start_investor=(
                "cold_start_investor",
                "first",
            ),

            cold_start_startup=(
                "cold_start_startup",
                "first",
            ),

            investor_core_connected=(
                "investor_core_connected",
                "first",
            ),

            startup_core_connected=(
                "startup_core_connected",
                "first",
            ),

            investor_seen_in_t0=(
                "investor_seen_in_t0",
                "first",
            ),

            startup_seen_in_t0=(
                "startup_seen_in_t0",
                "first",
            ),

            startup_structural_source_class=(
                "startup_structural_source_class",
                "first",
            ),
        )
        .reset_index()
    )

    pair_df[
        "interaction_id"
    ] = (
        pair_df[
            "investor_id"
        ].astype(str)
        + "|||"
        + pair_df[
            "startup_id"
        ].astype(str)
    )

    audit = pd.DataFrame(
        audit_rows
    )

    return (
        pair_df,
        audit,
    )


def build_covariance_set(
    spec: dict,
) -> tuple[dict[str, np.ndarray], dict]:
    data = spec[
        "data"
    ]

    fit = spec[
        "fit"
    ]

    investor = data[
        "investor_id"
    ].to_numpy()

    startup = data[
        "startup_id"
    ].to_numpy()

    pair = np.array(
        [
            f"{inv}|||{st}"
            for inv, st in zip(
                investor,
                startup,
            )
        ],
        dtype=object,
    )

    v_investor, d_investor = (
        one_way_cluster_covariance(
            fit,
            investor,
        )
    )

    v_startup, d_startup = (
        one_way_cluster_covariance(
            fit,
            startup,
        )
    )

    v_pair, d_pair = (
        one_way_cluster_covariance(
            fit,
            pair,
        )
    )

    v_two_way, d_two_way = (
        two_way_cluster_covariance(
            fit,
            investor,
            startup,
        )
    )

    covariances = {
        "investor_cluster_CR1":
            v_investor,

        "startup_cluster_CR1":
            v_startup,

        "pair_cluster_CR1":
            v_pair,

        "two_way_investor_startup_CR1":
            v_two_way,
    }

    diagnostics = {
        "investor_cluster_CR1":
            {
                **d_investor,
                **covariance_variance_diagnostics(
                    v_investor
                ),
            },

        "startup_cluster_CR1":
            {
                **d_startup,
                **covariance_variance_diagnostics(
                    v_startup
                ),
            },

        "pair_cluster_CR1":
            {
                **d_pair,
                **covariance_variance_diagnostics(
                    v_pair
                ),
            },

        "two_way_investor_startup_CR1":
            {
                **d_two_way,
                **covariance_variance_diagnostics(
                    v_two_way
                ),
            },
    }

    return (
        covariances,
        diagnostics,
    )


def estimands_for_covariances(
    model_key: str,
    spec: dict,
    covariances: dict[str, np.ndarray],
    analysis_unit: str,
) -> pd.DataFrame:
    rows = []

    for covariance_type, covariance in (
        covariances.items()
    ):
        for estimand in spec[
            "estimands"
        ]:
            rows.append(
                linear_combination(
                    fit=spec[
                        "fit"
                    ],
                    covariance=covariance,
                    weights=estimand[
                        "weights"
                    ],
                    estimand_key=estimand[
                        "estimand_key"
                    ],
                    estimand_label=estimand[
                        "estimand_label"
                    ],
                    model_key=model_key,
                    analysis_unit=analysis_unit,
                    covariance_type=covariance_type,
                )
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    print("=" * 110)
    print(
        "PHASE 7.9 — DEPENDENCE & WEIGHTING "
        "ROBUSTNESS DIAGNOSTICS V1"
    )
    print("=" * 110)

    print(
        "Scientific role:             "
        "PRE-SPECIFIED ROBUSTNESS / DEPENDENCE DIAGNOSTIC"
    )
    print("Primary endpoint:             Hit@10")
    print("Phase-7.8 formulas changed:   NO")
    print("Event-level analysis primary: YES")
    print("Pair-balanced analysis:       ROBUSTNESS ONLY")
    print("Causal interpretation:        NO")
    print("Model loaded:                 NO")
    print("Checkpoint loaded:            NO")
    print("Raw logits loaded:            NO")
    print("Inference executed:           NO")
    print("Test rescored:                NO")
    print("Model selection performed:   NO")

    # =========================================================================
    # 7.9.1 — Integrity gate
    # =========================================================================

    print_section(
        "7.9.1 — ANALYSIS INTEGRITY GATE"
    )

    for path in (
        SOURCE,
        PHASE_7_8_MANIFEST,
        PHASE_7_8_ESTIMANDS,
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

    with PHASE_7_8_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as f:
        phase_7_8_manifest = (
            json.load(f)
        )

    require(
        phase_7_8_manifest[
            "status"
        ]
        == "COMPLETE",
        "Phase 7.8 is not COMPLETE.",
    )

    phase_7_8_estimands = (
        pd.read_csv(
            PHASE_7_8_ESTIMANDS
        )
    )

    print()
    print("Frozen source fingerprint:   PASS")
    print("Phase 7.8 prerequisite:       PASS")

    # =========================================================================
    # 7.9.2 — Load frozen events / reconstruct exact Phase-7.8 models
    # =========================================================================

    print_section(
        "7.9.2 — RECONSTRUCT EXACT PHASE-7.8 EVENT-LEVEL MODELS"
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

    df[
        "hit10"
    ] = (
        df[
            "positive_rank"
        ]
        .le(10)
        .astype(float)
    )

    event_specs = (
        model_specs_for_dataframe(
            df
        )
    )

    require(
        event_specs[
            "M1"
        ][
            "fit"
        ][
            "n"
        ]
        == 10_707,
        "M1 event N drift.",
    )

    require(
        event_specs[
            "M2"
        ][
            "fit"
        ][
            "n"
        ]
        == 13_281,
        "M2 event N drift.",
    )

    require(
        event_specs[
            "M3"
        ][
            "fit"
        ][
            "n"
        ]
        == 6_392,
        "M3 event N drift.",
    )

    print("M1 event N:                  10,707")
    print("M2 event N:                  13,281")
    print("M3 event N:                   6,392")

    # =========================================================================
    # 7.9.3 — Event-level covariance sensitivity
    # =========================================================================

    print_section(
        "7.9.3 — EVENT-LEVEL CLUSTER-COVARIANCE SENSITIVITY"
    )

    event_estimand_frames = []
    event_cov_diagnostics = {}

    for model_key in (
        "M1",
        "M2",
        "M3",
    ):
        covariances, diagnostics = (
            build_covariance_set(
                event_specs[
                    model_key
                ]
            )
        )

        event_cov_diagnostics[
            model_key
        ] = diagnostics

        frame = (
            estimands_for_covariances(
                model_key=model_key,
                spec=event_specs[
                    model_key
                ],
                covariances=covariances,
                analysis_unit="event",
            )
        )

        event_estimand_frames.append(
            frame
        )

    event_estimands = pd.concat(
        event_estimand_frames,
        ignore_index=True,
    )

    # Validate investor-cluster implementation against Phase 7.8.
    baseline = (
        event_estimands.loc[
            event_estimands[
                "covariance_type"
            ]
            == "investor_cluster_CR1"
        ]
        .set_index(
            "estimand_key"
        )
    )

    prior = (
        phase_7_8_estimands
        .set_index(
            "estimand_key"
        )
    )

    for key in KEY_ESTIMAND_ORDER:
        require(
            key in baseline.index,
            f"Missing event baseline estimand {key}.",
        )

        require(
            key in prior.index,
            f"Missing Phase-7.8 estimand {key}.",
        )

        require(
            np.isclose(
                float(
                    baseline.loc[
                        key,
                        "estimate",
                    ]
                ),
                float(
                    prior.loc[
                        key,
                        "estimate",
                    ]
                ),
                atol=1e-12,
                rtol=0.0,
            ),
            (
                f"Phase-7.8 estimate mismatch "
                f"for {key}."
            ),
        )

        require(
            np.isclose(
                float(
                    baseline.loc[
                        key,
                        "ci95_low",
                    ]
                ),
                float(
                    prior.loc[
                        key,
                        "ci95_low",
                    ]
                ),
                atol=1e-10,
                rtol=0.0,
            ),
            (
                f"Phase-7.8 lower-CI mismatch "
                f"for {key}."
            ),
        )

        require(
            np.isclose(
                float(
                    baseline.loc[
                        key,
                        "ci95_high",
                    ]
                ),
                float(
                    prior.loc[
                        key,
                        "ci95_high",
                    ]
                ),
                atol=1e-10,
                rtol=0.0,
            ),
            (
                f"Phase-7.8 upper-CI mismatch "
                f"for {key}."
            ),
        )

    print(
        "Phase-7.8 investor-cluster "
        "estimand binding: PASS"
    )

    for key in KEY_ESTIMAND_ORDER:
        subset = (
            event_estimands.loc[
                event_estimands[
                    "estimand_key"
                ]
                == key,
                [
                    "covariance_type",
                    "estimate",
                    "se",
                    "ci95_low",
                    "ci95_high",
                ],
            ]
        )

        print()
        print(
            ESTIMAND_LABELS[key]
        )

        print(
            subset.to_string(
                index=False,
                formatters={
                    "estimate":
                        lambda x: f"{x:+.6f}",
                    "se":
                        lambda x: f"{x:.6f}",
                    "ci95_low":
                        lambda x: f"{x:+.6f}",
                    "ci95_high":
                        lambda x: f"{x:+.6f}",
                },
            )
        )

    # =========================================================================
    # 7.9.4 — Pair-collapse audit
    # =========================================================================

    print_section(
        "7.9.4 — PAIR-BALANCED COLLAPSE AUDIT"
    )

    pair_df, pair_audit = (
        collapse_to_pair_level(
            df
        )
    )

    pair_n = len(
        pair_df
    )

    repeated_pair_n = int(
        (
            pair_df[
                "event_n"
            ]
            > 1
        ).sum()
    )

    repeated_event_n = int(
        pair_df.loc[
            pair_df[
                "event_n"
            ]
            > 1,
            "event_n",
        ].sum()
    )

    max_events_per_pair = int(
        pair_df[
            "event_n"
        ].max()
    )

    print(
        f"Event rows:                   "
        f"{len(df):,}"
    )

    print(
        f"Unique investor-startup pairs:"
        f" {pair_n:,}"
    )

    print(
        f"Pairs with >1 test event:     "
        f"{repeated_pair_n:,}"
    )

    print(
        f"Events belonging to repeated pairs: "
        f"{repeated_event_n:,}"
    )

    print(
        f"Maximum events in one pair:   "
        f"{max_events_per_pair:,}"
    )

    print()
    print(
        pair_audit.to_string(
            index=False
        )
    )

    require(
        pair_audit[
            "pair_invariant"
        ].all(),
        "Not all model predictors are pair invariant.",
    )

    # =========================================================================
    # 7.9.5 — Pair-balanced models / covariance sensitivity
    # =========================================================================

    print_section(
        "7.9.5 — PAIR-BALANCED MODEL ROBUSTNESS"
    )

    pair_specs = (
        model_specs_for_dataframe(
            pair_df
        )
    )

    pair_estimand_frames = []
    pair_cov_diagnostics = {}

    for model_key in (
        "M1",
        "M2",
        "M3",
    ):
        covariances, diagnostics = (
            build_covariance_set(
                pair_specs[
                    model_key
                ]
            )
        )

        pair_cov_diagnostics[
            model_key
        ] = diagnostics

        frame = (
            estimands_for_covariances(
                model_key=model_key,
                spec=pair_specs[
                    model_key
                ],
                covariances=covariances,
                analysis_unit="pair_balanced",
            )
        )

        pair_estimand_frames.append(
            frame
        )

    pair_estimands = pd.concat(
        pair_estimand_frames,
        ignore_index=True,
    )

    pair_two_way = (
        pair_estimands.loc[
            pair_estimands[
                "covariance_type"
            ]
            == "two_way_investor_startup_CR1"
        ]
        .set_index(
            "estimand_key"
        )
    )

    event_two_way = (
        event_estimands.loc[
            event_estimands[
                "covariance_type"
            ]
            == "two_way_investor_startup_CR1"
        ]
        .set_index(
            "estimand_key"
        )
    )

    pair_comparison_rows = []

    for key in KEY_ESTIMAND_ORDER:
        event_row = (
            event_two_way.loc[
                key
            ]
        )

        pair_row = (
            pair_two_way.loc[
                key
            ]
        )

        pair_comparison_rows.append(
            {
                "estimand_key":
                    key,

                "estimand_label":
                    ESTIMAND_LABELS[
                        key
                    ],

                "event_estimate":
                    float(
                        event_row[
                            "estimate"
                        ]
                    ),

                "event_ci95_low":
                    float(
                        event_row[
                            "ci95_low"
                        ]
                    ),

                "event_ci95_high":
                    float(
                        event_row[
                            "ci95_high"
                        ]
                    ),

                "pair_balanced_estimate":
                    float(
                        pair_row[
                            "estimate"
                        ]
                    ),

                "pair_balanced_ci95_low":
                    float(
                        pair_row[
                            "ci95_low"
                        ]
                    ),

                "pair_balanced_ci95_high":
                    float(
                        pair_row[
                            "ci95_high"
                        ]
                    ),

                "pair_minus_event_estimate":
                    float(
                        pair_row[
                            "estimate"
                        ]
                        - event_row[
                            "estimate"
                        ]
                    ),

                "same_sign":
                    bool(
                        np.sign(
                            pair_row[
                                "estimate"
                            ]
                        )
                        == np.sign(
                            event_row[
                                "estimate"
                            ]
                        )
                    ),

                "event_ci_excludes_zero":
                    bool(
                        (
                            event_row[
                                "ci95_low"
                            ]
                            > 0
                        )
                        or
                        (
                            event_row[
                                "ci95_high"
                            ]
                            < 0
                        )
                    ),

                "pair_ci_excludes_zero":
                    bool(
                        (
                            pair_row[
                                "ci95_low"
                            ]
                            > 0
                        )
                        or
                        (
                            pair_row[
                                "ci95_high"
                            ]
                            < 0
                        )
                    ),
            }
        )

    pair_comparison = pd.DataFrame(
        pair_comparison_rows
    )

    print(
        pair_comparison[
            [
                "estimand_label",
                "event_estimate",
                "pair_balanced_estimate",
                "pair_minus_event_estimate",
                "same_sign",
                "event_ci_excludes_zero",
                "pair_ci_excludes_zero",
            ]
        ].to_string(
            index=False,
            formatters={
                "event_estimate":
                    lambda x: f"{x:+.6f}",
                "pair_balanced_estimate":
                    lambda x: f"{x:+.6f}",
                "pair_minus_event_estimate":
                    lambda x: f"{x:+.6f}",
            },
        )
    )

    # =========================================================================
    # 7.9.6 — Key robustness synthesis
    # =========================================================================

    print_section(
        "7.9.6 — KEY ROBUSTNESS SYNTHESIS"
    )

    synthesis_rows = []

    for key in KEY_ESTIMAND_ORDER:
        event_cov_subset = (
            event_estimands.loc[
                event_estimands[
                    "estimand_key"
                ]
                == key
            ]
        )

        all_event_cov_same_sign = bool(
            (
                np.sign(
                    event_cov_subset[
                        "estimate"
                    ]
                )
                == np.sign(
                    event_cov_subset[
                        "estimate"
                    ].iloc[0]
                )
            ).all()
        )

        two_way_event = (
            event_two_way.loc[
                key
            ]
        )

        two_way_pair = (
            pair_two_way.loc[
                key
            ]
        )

        synthesis_rows.append(
            {
                "estimand_key":
                    key,

                "estimand_label":
                    ESTIMAND_LABELS[
                        key
                    ],

                "event_estimate":
                    float(
                        two_way_event[
                            "estimate"
                        ]
                    ),

                "event_two_way_ci95_low":
                    float(
                        two_way_event[
                            "ci95_low"
                        ]
                    ),

                "event_two_way_ci95_high":
                    float(
                        two_way_event[
                            "ci95_high"
                        ]
                    ),

                "pair_balanced_estimate":
                    float(
                        two_way_pair[
                            "estimate"
                        ]
                    ),

                "pair_two_way_ci95_low":
                    float(
                        two_way_pair[
                            "ci95_low"
                        ]
                    ),

                "pair_two_way_ci95_high":
                    float(
                        two_way_pair[
                            "ci95_high"
                        ]
                    ),

                "all_event_covariance_estimates_same_sign":
                    all_event_cov_same_sign,

                "event_two_way_ci_excludes_zero":
                    bool(
                        (
                            two_way_event[
                                "ci95_low"
                            ]
                            > 0
                        )
                        or
                        (
                            two_way_event[
                                "ci95_high"
                            ]
                            < 0
                        )
                    ),

                "pair_two_way_ci_excludes_zero":
                    bool(
                        (
                            two_way_pair[
                                "ci95_low"
                            ]
                            > 0
                        )
                        or
                        (
                            two_way_pair[
                                "ci95_high"
                            ]
                            < 0
                        )
                    ),
            }
        )

    synthesis = pd.DataFrame(
        synthesis_rows
    )

    print(
        synthesis.to_string(
            index=False,
            formatters={
                "event_estimate":
                    lambda x: f"{x:+.6f}",
                "event_two_way_ci95_low":
                    lambda x: f"{x:+.6f}",
                "event_two_way_ci95_high":
                    lambda x: f"{x:+.6f}",
                "pair_balanced_estimate":
                    lambda x: f"{x:+.6f}",
                "pair_two_way_ci95_low":
                    lambda x: f"{x:+.6f}",
                "pair_two_way_ci95_high":
                    lambda x: f"{x:+.6f}",
            },
        )
    )

    # =========================================================================
    # 7.9.7 — Covariance matrix diagnostics
    # =========================================================================

    print_section(
        "7.9.7 — COVARIANCE MATRIX DIAGNOSTICS"
    )

    covariance_diagnostic_rows = []

    for analysis_unit, container in (
        (
            "event",
            event_cov_diagnostics,
        ),
        (
            "pair_balanced",
            pair_cov_diagnostics,
        ),
    ):
        for model_key, cov_types in (
            container.items()
        ):
            for covariance_type, diag in (
                cov_types.items()
            ):
                row = {
                    "analysis_unit":
                        analysis_unit,

                    "model_key":
                        model_key,

                    "covariance_type":
                        covariance_type,

                    **diag,
                }

                covariance_diagnostic_rows.append(
                    row
                )

    covariance_diagnostics = pd.DataFrame(
        covariance_diagnostic_rows
    )

    print(
        covariance_diagnostics[
            [
                col
                for col in [
                    "analysis_unit",
                    "model_key",
                    "covariance_type",
                    "cluster_count",
                    "investor_cluster_count",
                    "startup_cluster_count",
                    "pair_cluster_count",
                    "min_eigenvalue",
                    "negative_eigenvalue_n",
                ]
                if col
                in covariance_diagnostics.columns
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # 7.9.8 — Interpretation boundary
    # =========================================================================

    print_section(
        "7.9.8 — INTERPRETATION BOUNDARY"
    )

    print(
        "Primary scientific unit remains the frozen test EVENT."
    )

    print(
        "Pair-balanced estimates are sensitivity analyses only."
    )

    print()
    print(
        "Two-way clustering addresses dependence shared within "
        "investors and within startups using an inclusion-exclusion "
        "cluster covariance."
    )

    print()
    print(
        "Robustness of sign / interval across these analyses strengthens "
        "confidence that a finding is not merely an artifact of one "
        "clustering assumption or repeated-pair weighting."
    )

    print()
    print(
        "It still does NOT establish causality or remove all possible "
        "dependence/model-specification concerns."
    )

    # =========================================================================
    # 7.9.9 — Write artifacts
    # =========================================================================

    print_section(
        "7.9.9 — WRITE ROBUSTNESS ARTIFACTS"
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    event_estimands_path = (
        OUT_DIR
        / "phase_7_9_event_covariance_sensitivity_V1.csv"
    )

    pair_estimands_path = (
        OUT_DIR
        / "phase_7_9_pair_balanced_covariance_sensitivity_V1.csv"
    )

    pair_audit_path = (
        OUT_DIR
        / "phase_7_9_pair_collapse_invariance_audit_V1.csv"
    )

    pair_comparison_path = (
        OUT_DIR
        / "phase_7_9_event_vs_pair_balanced_estimands_V1.csv"
    )

    covariance_diag_path = (
        OUT_DIR
        / "phase_7_9_covariance_diagnostics_V1.csv"
    )

    synthesis_path = (
        OUT_DIR
        / "phase_7_9_robustness_synthesis_V1.csv"
    )

    summary_path = (
        OUT_DIR
        / "phase_7_9_robustness_summary_V1.json"
    )

    event_estimands.to_csv(
        event_estimands_path,
        index=False,
    )

    pair_estimands.to_csv(
        pair_estimands_path,
        index=False,
    )

    pair_audit.to_csv(
        pair_audit_path,
        index=False,
    )

    pair_comparison.to_csv(
        pair_comparison_path,
        index=False,
    )

    covariance_diagnostics.to_csv(
        covariance_diag_path,
        index=False,
    )

    synthesis.to_csv(
        synthesis_path,
        index=False,
    )

    summary = {
        "phase":
            "7.9",

        "scientific_role":
            (
                "Dependence and event-weighting robustness "
                "for the exact Phase-7.8 adjusted Hit@10 models."
            ),

        "event_level_primary":
            True,

        "covariance_sensitivity": [
            "investor_cluster_CR1",
            "startup_cluster_CR1",
            "pair_cluster_CR1",
            "two_way_investor_startup_CR1",
        ],

        "two_way_formula":
            (
                "V_investor + V_startup - V_investor_startup_pair"
            ),

        "pair_balanced_robustness": {
            "unique_pair_n":
                pair_n,

            "pairs_with_multiple_events_n":
                repeated_pair_n,

            "events_in_repeated_pairs_n":
                repeated_event_n,

            "max_events_per_pair":
                max_events_per_pair,

            "outcome":
                (
                    "mean Hit@10 across events within pair"
                ),
        },

        "causal_interpretation":
            False,

        "model_specification_search":
            False,

        "final_test_rescored":
            False,
    }

    json_dump(
        summary,
        summary_path,
    )

    # =========================================================================
    # 7.9.10 — Figures
    # =========================================================================

    print_section(
        "7.9.10 — GENERATE PRESENTATION-READY FIGURES"
    )

    figure_paths = []

    # Figure 1 — event-level covariance sensitivity
    selected_keys = [
        "m1_adjusted_new_vs_prior_warm_warm",
        "m2_cold_start_penalty_if_isolated",
        "m2_structure_effect_among_cold_startups",
        "m3_founder_signal_vs_neither",
    ]

    plot_df = (
        event_estimands.loc[
            event_estimands[
                "estimand_key"
            ].isin(
                selected_keys
            )
            &
            event_estimands[
                "covariance_type"
            ].isin(
                [
                    "investor_cluster_CR1",
                    "startup_cluster_CR1",
                    "two_way_investor_startup_CR1",
                ]
            )
        ]
        .copy()
    )

    cov_order = [
        "investor_cluster_CR1",
        "startup_cluster_CR1",
        "two_way_investor_startup_CR1",
    ]

    cov_display = {
        "investor_cluster_CR1":
            "Investor cluster",

        "startup_cluster_CR1":
            "Startup cluster",

        "two_way_investor_startup_CR1":
            "Two-way cluster",
    }

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    y_base = np.arange(
        len(
            selected_keys
        )
    ) * 1.2

    offsets = {
        "investor_cluster_CR1":
            -0.18,

        "startup_cluster_CR1":
            0.0,

        "two_way_investor_startup_CR1":
            0.18,
    }

    for covariance_type in cov_order:
        sub = (
            plot_df.loc[
                plot_df[
                    "covariance_type"
                ]
                == covariance_type
            ]
            .set_index(
                "estimand_key"
            )
            .loc[
                selected_keys
            ]
        )

        estimates = (
            sub[
                "estimate"
            ].to_numpy()
            * 100.0
        )

        lows = (
            sub[
                "ci95_low"
            ].to_numpy()
            * 100.0
        )

        highs = (
            sub[
                "ci95_high"
            ].to_numpy()
            * 100.0
        )

        y = (
            y_base
            + offsets[
                covariance_type
            ]
        )

        ax.errorbar(
            estimates,
            y,
            xerr=[
                estimates - lows,
                highs - estimates,
            ],
            fmt="o",
            capsize=3,
            label=cov_display[
                covariance_type
            ],
        )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_yticks(
        y_base,
        [
            ESTIMAND_LABELS[
                key
            ]
            for key in selected_keys
        ],
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Adjusted Hit@10 difference (percentage points)"
    )

    ax.set_title(
        "Dependence-robustness of central Phase-7 estimands"
    )

    ax.legend()
    ax.grid(
        axis="x",
        alpha=0.20,
    )

    fig.tight_layout()

    figure_paths.extend(
        save_figure(
            fig,
            (
                "phase_7_9_fig1_"
                "cluster_covariance_sensitivity_V1"
            ),
        )
    )

    # Figure 2 — event vs pair-balanced two-way estimates
    selected_keys = [
        "m1_adjusted_new_vs_prior_warm_warm",
        "m2_cold_start_penalty_if_isolated",
        "m2_structure_effect_among_cold_startups",
        "m2_structure_interaction_cold_minus_warm",
        "m3_founder_signal_vs_neither",
    ]

    compare = (
        pair_comparison.loc[
            pair_comparison[
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

    y = np.arange(
        len(compare)
    )

    event_est = (
        compare[
            "event_estimate"
        ].to_numpy()
        * 100.0
    )

    pair_est = (
        compare[
            "pair_balanced_estimate"
        ].to_numpy()
        * 100.0
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    ax.scatter(
        event_est,
        y - 0.10,
        label="Event-level",
    )

    ax.scatter(
        pair_est,
        y + 0.10,
        label="Pair-balanced",
    )

    for i in range(
        len(compare)
    ):
        ax.plot(
            [
                event_est[i],
                pair_est[i],
            ],
            [
                y[i] - 0.10,
                y[i] + 0.10,
            ],
            linewidth=1.0,
        )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_yticks(
        y,
        [
            ESTIMAND_LABELS[
                key
            ]
            for key in compare[
                "estimand_key"
            ]
        ],
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Adjusted Hit@10 difference (percentage points)"
    )

    ax.set_title(
        "Event-weighted vs pair-balanced robustness\n"
        "Point estimates; two-way clustered inference stored in artifacts"
    )

    ax.legend()
    ax.grid(
        axis="x",
        alpha=0.20,
    )

    fig.tight_layout()

    figure_paths.extend(
        save_figure(
            fig,
            (
                "phase_7_9_fig2_"
                "event_vs_pair_balanced_V1"
            ),
        )
    )

    # =========================================================================
    # 7.9.11 — Manifest / hashes
    # =========================================================================

    manifest_path = (
        OUT_DIR
        / "phase_7_9_analysis_manifest_V1.json"
    )

    manifest = {
        "phase":
            "7.9",

        "schema_version":
            "ITRS_PHASE7_9_DEPENDENCE_ROBUSTNESS_V1",

        "status":
            "COMPLETE",

        "scientific_role":
            (
                "Dependence and event-weighting robustness "
                "for exact Phase-7.8 adjusted Hit@10 models."
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

        "phase_7_8_binding":
            "PASS",

        "event_level_primary":
            True,

        "pair_balanced_secondary":
            True,

        "covariance_estimators": [
            "investor_cluster_CR1",
            "startup_cluster_CR1",
            "pair_cluster_CR1",
            "two_way_investor_startup_CR1",
        ],

        "two_way_covariance_formula":
            (
                "V_investor + V_startup - V_pair"
            ),

        "pair_collapse_predictor_invariance":
            bool(
                pair_audit[
                    "pair_invariant"
                ].all()
            ),

        "causal_interpretation":
            False,

        "model_specification_search":
            False,

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

            "model_selection":
                False,
        },
    }

    json_dump(
        manifest,
        manifest_path,
    )

    artifact_paths = [
        event_estimands_path,
        pair_estimands_path,
        pair_audit_path,
        pair_comparison_path,
        covariance_diag_path,
        synthesis_path,
        summary_path,
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
        / "phase_7_9_derived_artifact_sha256_V1.json"
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
        "PHASE 7.9 V1 RESULT"
    )

    print("Frozen source binding:              PASS")
    print("Exact Phase-7.8 model binding:      PASS")
    print("Investor-cluster covariance:        COMPLETE")
    print("Startup-cluster covariance:         COMPLETE")
    print("Pair-cluster covariance:            COMPLETE")
    print("Two-way cluster covariance:         COMPLETE")
    print("Pair-collapse invariance audit:     COMPLETE")
    print("Pair-balanced robustness:           COMPLETE")
    print("Robustness synthesis:               COMPLETE")
    print("Presentation-ready figures:         COMPLETE")
    print("Provenance manifest:                COMPLETE")

    print()
    print("Event-level final test remains primary: YES")
    print("Causal interpretation performed:       NO")
    print("Model inference executed:              NO")
    print("Final test rescored:                    NO")
    print("Final test modified:                    NO")

    print()
    print(
        "PHASE 7.9 DEPENDENCE & WEIGHTING "
        "ROBUSTNESS STATUS: COMPLETE"
    )


if __name__ == "__main__":
    main()