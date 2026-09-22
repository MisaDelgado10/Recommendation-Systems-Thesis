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
# Phase 7.5C — Degree–Performance Diagnostic V1
#
# Purpose
# -------
# Examine whether structural degree is associated with ranking performance
# beyond a simple connected / isolated indicator.
#
# Scientific scope
# ----------------
# This is a descriptive / association diagnostic on the immutable Phase-6
# final test. It does NOT establish causality.
#
# Pre-specified degree treatment
# ----------------------------
# - degree == 0 is treated as structural isolation
# - positive degree is analyzed continuously using log1p(total_degree)
# - NO outcome-tuned degree bins
# - NO threshold search
#
# Primary thesis-focused analysis:
#   new-to-investor, warm investor, cold-start startup
#
# Secondary:
#   new-to-investor, warm/warm
#   new-to-investor overall
#
# Degree fields:
#   startup_core_total_degree
#   investor_core_total_degree
#
# Prohibited
# ----------
# - model loading
# - checkpoint loading
# - inference
# - test rescoring
# - candidate modification
# - model selection
# =============================================================================


REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    REPO_ROOT
    / "data/experimental/phase_6/full_training/100pct/final_test/"
    / "analysis_ready_test_cases.parquet"
)

PHASE_7_5A_MANIFEST = (
    REPO_ROOT
    / "data/experimental/phase_7/"
    / "phase_7_5a_structural_coverage_composition/"
    / "phase_7_5a_analysis_manifest_V2.json"
)

PHASE_7_5B_MANIFEST = (
    REPO_ROOT
    / "data/experimental/phase_7/"
    / "phase_7_5b_structural_performance/"
    / "phase_7_5b_analysis_manifest_V1.json"
)

OUT_DIR = (
    REPO_ROOT
    / "data/experimental/phase_7/"
    / "phase_7_5c_degree_performance"
)

FIG_DIR = OUT_DIR / "figures"

EXPECTED_SOURCE_SHA256 = (
    "d70f21bff0006e094c5d567d307c0664"
    "b41a11e7250a0e69aaec610e1810132d"
)

EXPECTED_ROWS = 20_264

BOOTSTRAP_REPS = 2_000
BOOTSTRAP_SEED = 7_105_003

SOURCE_COLUMNS = [
    "interaction_id",
    "investor_id",
    "startup_id",
    "positive_rank",
    "NDCG@10",
    "new_to_investor_pair",
    "cold_start_investor",
    "cold_start_startup",
    "investor_core_connected",
    "startup_core_connected",
    "investor_core_total_degree",
    "startup_core_total_degree",
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
            "Author": "ITRS Phase 7 diagnostic analysis",
        },
    )

    plt.close(fig)

    return [png, pdf]


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    require(
        len(x) == len(y),
        "Correlation arrays differ in length.",
    )

    require(
        len(x) >= 2,
        "Need at least two observations for correlation.",
    )

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return float(
        np.corrcoef(x, y)[0, 1]
    )


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = pd.Series(x).rank(
        method="average"
    ).to_numpy(dtype=float)

    y_rank = pd.Series(y).rank(
        method="average"
    ).to_numpy(dtype=float)

    return pearson_corr(
        x_rank,
        y_rank,
    )


def simple_slope(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x_centered = x - x.mean()

    denom = float(
        np.sum(
            x_centered ** 2
        )
    )

    require(
        denom > 0,
        "Cannot estimate slope with zero x variance.",
    )

    return float(
        np.sum(
            x_centered
            * (y - y.mean())
        )
        / denom
    )


def summarize_degree_association(
    df: pd.DataFrame,
    degree_col: str,
    entity_label: str,
    analysis_label: str,
) -> dict:
    working = df[
        [
            degree_col,
            "positive_rank",
            "NDCG@10",
        ]
    ].copy()

    working["hit10"] = (
        working["positive_rank"]
        .le(10)
        .astype(float)
    )

    working["log1p_degree"] = np.log1p(
        working[
            degree_col
        ].astype(float)
    )

    positive = working.loc[
        working[
            degree_col
        ] > 0
    ].copy()

    result = {
        "analysis_label":
            analysis_label,

        "entity":
            entity_label,

        "degree_column":
            degree_col,

        "n_all":
            int(len(working)),

        "n_degree_zero":
            int(
                (
                    working[
                        degree_col
                    ] == 0
                ).sum()
            ),

        "n_degree_positive":
            int(
                (
                    working[
                        degree_col
                    ] > 0
                ).sum()
            ),

        "positive_degree_min":
            (
                int(
                    positive[
                        degree_col
                    ].min()
                )
                if len(positive) > 0
                else None
            ),

        "positive_degree_median":
            (
                float(
                    positive[
                        degree_col
                    ].median()
                )
                if len(positive) > 0
                else None
            ),

        "positive_degree_max":
            (
                int(
                    positive[
                        degree_col
                    ].max()
                )
                if len(positive) > 0
                else None
            ),
    }

    if len(positive) >= 2 and (
        positive[degree_col].nunique() >= 2
    ):
        x = positive[
            "log1p_degree"
        ].to_numpy(
            dtype=float
        )

        result[
            "positive_degree_spearman_logdegree_vs_hit10"
        ] = spearman_corr(
            x,
            positive[
                "hit10"
            ].to_numpy(
                dtype=float
            ),
        )

        result[
            "positive_degree_spearman_logdegree_vs_ndcg10"
        ] = spearman_corr(
            x,
            positive[
                "NDCG@10"
            ].to_numpy(
                dtype=float
            ),
        )

        result[
            "positive_degree_spearman_logdegree_vs_rank"
        ] = spearman_corr(
            x,
            positive[
                "positive_rank"
            ].to_numpy(
                dtype=float
            ),
        )

        result[
            "positive_degree_linear_slope_hit10_per_log1p_degree"
        ] = simple_slope(
            x,
            positive[
                "hit10"
            ].to_numpy(
                dtype=float
            ),
        )

        result[
            "positive_degree_linear_slope_ndcg10_per_log1p_degree"
        ] = simple_slope(
            x,
            positive[
                "NDCG@10"
            ].to_numpy(
                dtype=float
            ),
        )

        result[
            "positive_degree_linear_slope_rank_per_log1p_degree"
        ] = simple_slope(
            x,
            positive[
                "positive_rank"
            ].to_numpy(
                dtype=float
            ),
        )
    else:
        result[
            "positive_degree_spearman_logdegree_vs_hit10"
        ] = None

        result[
            "positive_degree_spearman_logdegree_vs_ndcg10"
        ] = None

        result[
            "positive_degree_spearman_logdegree_vs_rank"
        ] = None

        result[
            "positive_degree_linear_slope_hit10_per_log1p_degree"
        ] = None

        result[
            "positive_degree_linear_slope_ndcg10_per_log1p_degree"
        ] = None

        result[
            "positive_degree_linear_slope_rank_per_log1p_degree"
        ] = None

    return result


def investor_cluster_bootstrap_slope(
    df: pd.DataFrame,
    degree_col: str,
    reps: int,
    seed: int,
) -> pd.DataFrame:
    """
    Investor-cluster bootstrap for positive-degree cases only.

    Estimates simple unadjusted slopes of:
      Hit@10 ~ log1p(degree)
      NDCG@10 ~ log1p(degree)
      rank ~ log1p(degree)

    This is an association diagnostic, not a causal model.
    """

    working = df.loc[
        df[
            degree_col
        ] > 0,
        [
            "investor_id",
            degree_col,
            "positive_rank",
            "NDCG@10",
        ],
    ].copy()

    working["hit10"] = (
        working["positive_rank"]
        .le(10)
        .astype(float)
    )

    working["log1p_degree"] = np.log1p(
        working[
            degree_col
        ].astype(float)
    )

    investors = (
        working[
            "investor_id"
        ]
        .drop_duplicates()
        .to_numpy()
    )

    groups = {
        investor_id:
            group[
                [
                    "log1p_degree",
                    "hit10",
                    "NDCG@10",
                    "positive_rank",
                ]
            ].to_numpy(
                dtype=float
            )
        for investor_id, group
        in working.groupby(
            "investor_id",
            sort=False,
        )
    }

    rng = np.random.default_rng(
        seed
    )

    rows = []

    for bootstrap_index in range(
        reps
    ):
        sampled_ids = rng.choice(
            investors,
            size=len(investors),
            replace=True,
        )

        sampled_arrays = [
            groups[i]
            for i in sampled_ids
        ]

        sample = np.vstack(
            sampled_arrays
        )

        x = sample[:, 0]
        hit10 = sample[:, 1]
        ndcg10 = sample[:, 2]
        rank = sample[:, 3]

        if np.unique(x).size < 2:
            continue

        rows.append(
            {
                "bootstrap_index":
                    bootstrap_index,

                "slope_HR10":
                    simple_slope(
                        x,
                        hit10,
                    ),

                "slope_NDCG10":
                    simple_slope(
                        x,
                        ndcg10,
                    ),

                "slope_rank":
                    simple_slope(
                        x,
                        rank,
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def percentile_ci(
    values: pd.Series,
) -> tuple[float, float]:
    clean = (
        values
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    require(
        len(clean) > 0,
        "Cannot compute CI from empty bootstrap distribution.",
    )

    low, high = np.quantile(
        clean,
        [
            0.025,
            0.975,
        ],
    )

    return (
        float(low),
        float(high),
    )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    print("=" * 110)
    print(
        "PHASE 7.5C — DEGREE–PERFORMANCE DIAGNOSTIC V1"
    )
    print("=" * 110)

    print(
        "Scientific role:             "
        "PRE-SPECIFIED CONTINUOUS DEGREE ASSOCIATION DIAGNOSTIC"
    )
    print("Outcome-tuned degree bins:    NO")
    print("Threshold search:             NO")
    print("Model loaded:                 NO")
    print("Checkpoint loaded:            NO")
    print("Inference executed:           NO")
    print("Test rescored:                NO")
    print("Model selection performed:   NO")

    # =========================================================================
    # 7.5C.1 — Integrity gate
    # =========================================================================

    print_section(
        "7.5C.1 — ANALYSIS INTEGRITY GATE"
    )

    for path in (
        SOURCE,
        PHASE_7_5A_MANIFEST,
        PHASE_7_5B_MANIFEST,
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
        source_hash == EXPECTED_SOURCE_SHA256,
        "Frozen source SHA256 drift.",
    )

    with PHASE_7_5A_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as f:
        phase_7_5a = json.load(f)

    with PHASE_7_5B_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as f:
        phase_7_5b = json.load(f)

    require(
        phase_7_5a["status"]
        == "COMPLETE",
        "Phase 7.5A is not COMPLETE.",
    )

    require(
        phase_7_5b["status"]
        == "COMPLETE",
        "Phase 7.5B is not COMPLETE.",
    )

    print()
    print("Frozen source fingerprint:   PASS")
    print("Phase 7.5A prerequisite:      PASS")
    print("Phase 7.5B prerequisite:      PASS")

    # =========================================================================
    # 7.5C.2 — Load frozen fields / semantic binding
    # =========================================================================

    print_section(
        "7.5C.2 — LOAD DEGREE AND PERFORMANCE FIELDS"
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

    require(
        (
            df[
                "startup_core_connected"
            ].astype(bool)
            ==
            df[
                "startup_core_total_degree"
            ].gt(0)
        ).all(),
        "Startup connected/degree binding failed.",
    )

    require(
        (
            df[
                "investor_core_connected"
            ].astype(bool)
            ==
            df[
                "investor_core_total_degree"
            ].gt(0)
        ).all(),
        "Investor connected/degree binding failed.",
    )

    print(
        f"Test cases:                   "
        f"{len(df):,}"
    )
    print(
        "Startup connected/degree:    PASS"
    )
    print(
        "Investor connected/degree:   PASS"
    )

    # =========================================================================
    # 7.5C.3 — Define analysis subsets
    # =========================================================================

    print_section(
        "7.5C.3 — PRE-SPECIFIED ANALYSIS SUBSETS"
    )

    new_mask = (
        df[
            "new_to_investor_pair"
        ].astype(bool)
    )

    new_ww_mask = (
        new_mask
        &
        ~df[
            "cold_start_investor"
        ].astype(bool)
        &
        ~df[
            "cold_start_startup"
        ].astype(bool)
    )

    new_wc_mask = (
        new_mask
        &
        ~df[
            "cold_start_investor"
        ].astype(bool)
        &
        df[
            "cold_start_startup"
        ].astype(bool)
    )

    subsets = {
        "all_test":
            df,

        "new_to_investor":
            df.loc[
                new_mask
            ],

        "new_warm_warm":
            df.loc[
                new_ww_mask
            ],

        "new_warm_investor_cold_startup":
            df.loc[
                new_wc_mask
            ],
    }

    for name, subset in (
        subsets.items()
    ):
        print(
            f"{name:<38} "
            f"{len(subset):>7,}"
        )

    require(
        len(
            subsets[
                "new_warm_investor_cold_startup"
            ]
        )
        == 6_392,
        "Primary new_wc subgroup count drift.",
    )

    # =========================================================================
    # 7.5C.4 — Startup degree association
    # =========================================================================

    print_section(
        "7.5C.4 — STARTUP DEGREE ASSOCIATION"
    )

    startup_rows = []

    for name in (
        "all_test",
        "new_to_investor",
        "new_warm_warm",
        "new_warm_investor_cold_startup",
    ):
        result = (
            summarize_degree_association(
                df=subsets[name],
                degree_col=(
                    "startup_core_total_degree"
                ),
                entity_label="startup",
                analysis_label=name,
            )
        )

        startup_rows.append(
            result
        )

    startup_association = pd.DataFrame(
        startup_rows
    )

    print(
        startup_association[
            [
                "analysis_label",
                "n_all",
                "n_degree_zero",
                "n_degree_positive",
                "positive_degree_median",
                (
                    "positive_degree_spearman_"
                    "logdegree_vs_hit10"
                ),
                (
                    "positive_degree_spearman_"
                    "logdegree_vs_ndcg10"
                ),
                (
                    "positive_degree_spearman_"
                    "logdegree_vs_rank"
                ),
            ]
        ].to_string(
            index=False,
            formatters={
                (
                    "positive_degree_spearman_"
                    "logdegree_vs_hit10"
                ):
                    lambda x: f"{x:+.4f}",

                (
                    "positive_degree_spearman_"
                    "logdegree_vs_ndcg10"
                ):
                    lambda x: f"{x:+.4f}",

                (
                    "positive_degree_spearman_"
                    "logdegree_vs_rank"
                ):
                    lambda x: f"{x:+.4f}",
            },
        )
    )

    # =========================================================================
    # 7.5C.5 — Investor degree association
    # =========================================================================

    print_section(
        "7.5C.5 — INVESTOR DEGREE ASSOCIATION"
    )

    investor_rows = []

    for name in (
        "all_test",
        "new_to_investor",
        "new_warm_warm",
    ):
        result = (
            summarize_degree_association(
                df=subsets[name],
                degree_col=(
                    "investor_core_total_degree"
                ),
                entity_label="investor",
                analysis_label=name,
            )
        )

        investor_rows.append(
            result
        )

    investor_association = pd.DataFrame(
        investor_rows
    )

    print(
        investor_association[
            [
                "analysis_label",
                "n_all",
                "n_degree_zero",
                "n_degree_positive",
                "positive_degree_median",
                (
                    "positive_degree_spearman_"
                    "logdegree_vs_hit10"
                ),
                (
                    "positive_degree_spearman_"
                    "logdegree_vs_ndcg10"
                ),
                (
                    "positive_degree_spearman_"
                    "logdegree_vs_rank"
                ),
            ]
        ].to_string(
            index=False,
            formatters={
                (
                    "positive_degree_spearman_"
                    "logdegree_vs_hit10"
                ):
                    lambda x: f"{x:+.4f}",

                (
                    "positive_degree_spearman_"
                    "logdegree_vs_ndcg10"
                ):
                    lambda x: f"{x:+.4f}",

                (
                    "positive_degree_spearman_"
                    "logdegree_vs_rank"
                ):
                    lambda x: f"{x:+.4f}",
            },
        )
    )

    # =========================================================================
    # 7.5C.6 — Primary positive-degree cold-start startup slope
    # =========================================================================

    print_section(
        "7.5C.6 — PRIMARY CONTINUOUS DEGREE TEST: COLD-START STARTUPS"
    )

    primary = (
        subsets[
            "new_warm_investor_cold_startup"
        ]
        .loc[
            lambda x:
                x[
                    "startup_core_total_degree"
                ] > 0
        ]
        .copy()
    )

    require(
        len(primary) == 809,
        (
            "Primary positive-degree cold-start "
            "startup count drift."
        ),
    )

    require(
        primary[
            "startup_core_total_degree"
        ].nunique()
        >= 2,
        (
            "Insufficient degree variation for "
            "continuous analysis."
        ),
    )

    primary[
        "log1p_startup_degree"
    ] = np.log1p(
        primary[
            "startup_core_total_degree"
        ].astype(float)
    )

    primary[
        "hit10"
    ] = (
        primary[
            "positive_rank"
        ]
        .le(10)
        .astype(float)
    )

    x = primary[
        "log1p_startup_degree"
    ].to_numpy(
        dtype=float
    )

    observed = {
        "n_positive_degree":
            int(len(primary)),

        "degree_min":
            int(
                primary[
                    "startup_core_total_degree"
                ].min()
            ),

        "degree_median":
            float(
                primary[
                    "startup_core_total_degree"
                ].median()
            ),

        "degree_max":
            int(
                primary[
                    "startup_core_total_degree"
                ].max()
            ),

        "spearman_logdegree_vs_HR10":
            spearman_corr(
                x,
                primary[
                    "hit10"
                ].to_numpy(
                    dtype=float
                ),
            ),

        "spearman_logdegree_vs_NDCG10":
            spearman_corr(
                x,
                primary[
                    "NDCG@10"
                ].to_numpy(
                    dtype=float
                ),
            ),

        "spearman_logdegree_vs_rank":
            spearman_corr(
                x,
                primary[
                    "positive_rank"
                ].to_numpy(
                    dtype=float
                ),
            ),

        "linear_slope_HR10_per_log1p_degree":
            simple_slope(
                x,
                primary[
                    "hit10"
                ].to_numpy(
                    dtype=float
                ),
            ),

        "linear_slope_NDCG10_per_log1p_degree":
            simple_slope(
                x,
                primary[
                    "NDCG@10"
                ].to_numpy(
                    dtype=float
                ),
            ),

        "linear_slope_rank_per_log1p_degree":
            simple_slope(
                x,
                primary[
                    "positive_rank"
                ].to_numpy(
                    dtype=float
                ),
            ),
    }

    print(
        f"Positive-degree N:            "
        f"{observed['n_positive_degree']:,}"
    )

    print(
        f"Degree range:                 "
        f"{observed['degree_min']}.."
        f"{observed['degree_max']}"
    )

    print(
        f"Median positive degree:       "
        f"{observed['degree_median']:.1f}"
    )

    print()
    print(
        f"Spearman log-degree vs HR@10: "
        f"{observed['spearman_logdegree_vs_HR10']:+.4f}"
    )

    print(
        f"Spearman log-degree vs NDCG:  "
        f"{observed['spearman_logdegree_vs_NDCG10']:+.4f}"
    )

    print(
        f"Spearman log-degree vs rank:  "
        f"{observed['spearman_logdegree_vs_rank']:+.4f}"
    )

    print()
    print(
        f"Slope HR@10 / log1p degree:   "
        f"{observed['linear_slope_HR10_per_log1p_degree']:+.6f}"
    )

    print(
        f"Slope NDCG / log1p degree:    "
        f"{observed['linear_slope_NDCG10_per_log1p_degree']:+.6f}"
    )

    print(
        f"Slope rank / log1p degree:    "
        f"{observed['linear_slope_rank_per_log1p_degree']:+.6f}"
    )

    # =========================================================================
    # 7.5C.7 — Investor-cluster bootstrap for primary degree slopes
    # =========================================================================

    print_section(
        "7.5C.7 — INVESTOR-CLUSTER BOOTSTRAP FOR PRIMARY DEGREE SLOPES"
    )

    bootstrap = (
        investor_cluster_bootstrap_slope(
            df=(
                subsets[
                    "new_warm_investor_cold_startup"
                ]
            ),
            degree_col=(
                "startup_core_total_degree"
            ),
            reps=BOOTSTRAP_REPS,
            seed=BOOTSTRAP_SEED,
        )
    )

    require(
        len(bootstrap) > 0,
        "No valid bootstrap replicates.",
    )

    ci_rows = []

    observed_map = {
        "HR@10":
            observed[
                "linear_slope_HR10_per_log1p_degree"
            ],

        "NDCG@10":
            observed[
                "linear_slope_NDCG10_per_log1p_degree"
            ],

        "positive_rank":
            observed[
                "linear_slope_rank_per_log1p_degree"
            ],
    }

    bootstrap_map = {
        "HR@10":
            "slope_HR10",

        "NDCG@10":
            "slope_NDCG10",

        "positive_rank":
            "slope_rank",
    }

    for metric, column in (
        bootstrap_map.items()
    ):
        low, high = percentile_ci(
            bootstrap[
                column
            ]
        )

        ci_rows.append(
            {
                "metric":
                    metric,

                "estimate":
                    observed_map[
                        metric
                    ],

                "ci95_low":
                    low,

                "ci95_high":
                    high,

                "bootstrap_unit":
                    "investor_id",

                "repetitions":
                    len(bootstrap),
            }
        )

    bootstrap_summary = pd.DataFrame(
        ci_rows
    )

    print(
        bootstrap_summary.to_string(
            index=False,
            formatters={
                "estimate":
                    lambda x: f"{x:+.6f}",
                "ci95_low":
                    lambda x: f"{x:+.6f}",
                "ci95_high":
                    lambda x: f"{x:+.6f}",
            },
        )
    )

    # =========================================================================
    # 7.5C.8 — Descriptive unique-degree table (NO degree bins)
    # =========================================================================

    print_section(
        "7.5C.8 — PRIMARY UNIQUE-DEGREE DESCRIPTIVE TABLE"
    )

    unique_degree = (
        primary.groupby(
            "startup_core_total_degree",
            sort=True,
        )
        .agg(
            n=(
                "positive_rank",
                "size",
            ),

            Hits10=(
                "hit10",
                "sum",
            ),

            HR10=(
                "hit10",
                "mean",
            ),

            NDCG10=(
                "NDCG@10",
                "mean",
            ),

            mean_rank=(
                "positive_rank",
                "mean",
            ),

            median_rank=(
                "positive_rank",
                "median",
            ),
        )
        .reset_index()
    )

    print(
        unique_degree.head(
            20
        ).to_string(
            index=False,
            formatters={
                "HR10":
                    lambda x: f"{x:.4f}",
                "NDCG10":
                    lambda x: f"{x:.4f}",
                "mean_rank":
                    lambda x: f"{x:.2f}",
                "median_rank":
                    lambda x: f"{x:.1f}",
            },
        )
    )

    print()
    print(
        "No degree bins were constructed."
    )

    # =========================================================================
    # 7.5C.9 — Interpretation boundary
    # =========================================================================

    print_section(
        "7.5C.9 — INTERPRETATION BOUNDARY"
    )

    print(
        "This phase separates two structural questions:"
    )

    print(
        "  1) availability: degree 0 vs degree >0 "
        "(already examined in Phase 7.5B);"
    )

    print(
        "  2) intensity: among structurally connected "
        "entities, whether larger audited degree is "
        "associated with better ranking."
    )

    print()
    print(
        "A positive connectivity effect in 7.5B combined "
        "with a weak degree-intensity trend in 7.5C would "
        "suggest that having SOME usable side structure may "
        "matter more than simply accumulating more edges."
    )

    print()
    print(
        "No causal interpretation is permitted."
    )

    # =========================================================================
    # 7.5C.10 — Write artifacts
    # =========================================================================

    print_section(
        "7.5C.10 — WRITE ANALYSIS ARTIFACTS"
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    startup_assoc_path = (
        OUT_DIR
        / "phase_7_5c_startup_degree_associations_V1.csv"
    )

    investor_assoc_path = (
        OUT_DIR
        / "phase_7_5c_investor_degree_associations_V1.csv"
    )

    primary_observed_path = (
        OUT_DIR
        / "phase_7_5c_primary_cold_startup_degree_observed_V1.json"
    )

    bootstrap_path = (
        OUT_DIR
        / "phase_7_5c_primary_degree_bootstrap_replicates_V1.csv"
    )

    bootstrap_summary_path = (
        OUT_DIR
        / "phase_7_5c_primary_degree_bootstrap_summary_V1.csv"
    )

    unique_degree_path = (
        OUT_DIR
        / "phase_7_5c_primary_unique_degree_metrics_V1.csv"
    )

    startup_association.to_csv(
        startup_assoc_path,
        index=False,
    )

    investor_association.to_csv(
        investor_assoc_path,
        index=False,
    )

    json_dump(
        observed,
        primary_observed_path,
    )

    bootstrap.to_csv(
        bootstrap_path,
        index=False,
    )

    bootstrap_summary.to_csv(
        bootstrap_summary_path,
        index=False,
    )

    unique_degree.to_csv(
        unique_degree_path,
        index=False,
    )

    # =========================================================================
    # 7.5C.11 — Figures
    # =========================================================================

    print_section(
        "7.5C.11 — GENERATE PRESENTATION-READY FIGURES"
    )

    figure_paths = []

    # Figure 1 — positive startup degree vs HR@10, exact unique degrees
    plot_degree = (
        unique_degree.loc[
            unique_degree[
                "n"
            ] >= 10
        ]
        .copy()
    )

    fig, ax = plt.subplots(
        figsize=(10.5, 6.5)
    )

    ax.scatter(
        np.log1p(
            plot_degree[
                "startup_core_total_degree"
            ].to_numpy(
                dtype=float
            )
        ),
        plot_degree[
            "HR10"
        ].to_numpy()
        * 100.0,
        s=np.maximum(
            20,
            plot_degree[
                "n"
            ].to_numpy(
                dtype=float
            )
            * 1.5,
        ),
        alpha=0.7,
    )

    ax.set_xlabel(
        "log1p(startup core total degree)"
    )

    ax.set_ylabel(
        "HR@10 (%)"
    )

    ax.set_title(
        "Cold-start startup degree and Top-10 retrieval\n"
        "Exact degree values with at least 10 cases; no outcome-tuned bins"
    )

    ax.grid(
        alpha=0.20,
    )

    fig.tight_layout()

    figure_paths.extend(
        save_figure(
            fig,
            (
                "phase_7_5c_fig1_"
                "cold_startup_degree_HR10_V1"
            ),
        )
    )

    # Figure 2 — positive startup degree vs mean rank
    fig, ax = plt.subplots(
        figsize=(10.5, 6.5)
    )

    ax.scatter(
        np.log1p(
            plot_degree[
                "startup_core_total_degree"
            ].to_numpy(
                dtype=float
            )
        ),
        plot_degree[
            "mean_rank"
        ].to_numpy(),
        s=np.maximum(
            20,
            plot_degree[
                "n"
            ].to_numpy(
                dtype=float
            )
            * 1.5,
        ),
        alpha=0.7,
    )

    ax.set_xlabel(
        "log1p(startup core total degree)"
    )

    ax.set_ylabel(
        "Mean positive rank"
    )

    ax.set_title(
        "Cold-start startup degree and rank severity\n"
        "Exact degree values with at least 10 cases; no outcome-tuned bins"
    )

    ax.grid(
        alpha=0.20,
    )

    fig.tight_layout()

    figure_paths.extend(
        save_figure(
            fig,
            (
                "phase_7_5c_fig2_"
                "cold_startup_degree_mean_rank_V1"
            ),
        )
    )

    # Figure 3 — bootstrap slope forest
    estimates = (
        bootstrap_summary[
            "estimate"
        ].to_numpy()
    )

    lows = (
        bootstrap_summary[
            "ci95_low"
        ].to_numpy()
    )

    highs = (
        bootstrap_summary[
            "ci95_high"
        ].to_numpy()
    )

    labels = [
        "HR@10 slope",
        "NDCG@10 slope",
        "Positive-rank slope",
    ]

    y = np.arange(
        len(labels)
    )

    fig, ax = plt.subplots(
        figsize=(9.5, 6)
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
        labels,
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Unadjusted slope per +1 log1p(degree)"
    )

    ax.set_title(
        "Cold-start startup degree–performance associations\n"
        "Investor-cluster 95% bootstrap intervals"
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
                "phase_7_5c_fig3_"
                "degree_slope_forest_V1"
            ),
        )
    )

    # =========================================================================
    # 7.5C.12 — Manifest / hashes
    # =========================================================================

    manifest_path = (
        OUT_DIR
        / "phase_7_5c_analysis_manifest_V1.json"
    )

    manifest = {
        "phase":
            "7.5C",

        "schema_version":
            "ITRS_PHASE7_5C_DEGREE_PERFORMANCE_V1",

        "status":
            "COMPLETE",

        "scientific_role":
            (
                "Pre-specified continuous degree association "
                "diagnostic on the frozen Phase-6 final test."
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

        "degree_policy": {
            "zero_degree":
                "structural isolation",

            "positive_degree_transform":
                "log1p(total_degree)",

            "outcome_tuned_bins":
                False,

            "threshold_search":
                False,
        },

        "primary_analysis":
            (
                "startup_core_total_degree among "
                "new-to-investor warm-investor/"
                "cold-startup positive-degree cases"
            ),

        "uncertainty": {
            "method":
                (
                    "Percentile investor-cluster bootstrap "
                    "of unadjusted continuous slopes."
                ),

            "repetitions":
                BOOTSTRAP_REPS,

            "seed":
                BOOTSTRAP_SEED,
        },

        "interpretation_boundary":
            (
                "Associations between degree intensity and "
                "ranking do not establish causality."
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
            "model_loaded": False,
            "checkpoint_loaded": False,
            "inference_executed": False,
            "test_rescored": False,
            "candidate_modification": False,
            "model_selection": False,
        },
    }

    json_dump(
        manifest,
        manifest_path,
    )

    artifact_paths = [
        startup_assoc_path,
        investor_assoc_path,
        primary_observed_path,
        bootstrap_path,
        bootstrap_summary_path,
        unique_degree_path,
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
        / "phase_7_5c_derived_artifact_sha256_V1.json"
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
        "PHASE 7.5C V1 RESULT"
    )

    print("Frozen source binding:             PASS")
    print("Degree/connectivity semantics:     PASS")
    print("No outcome-tuned degree bins:      PASS")
    print("Startup degree diagnostics:        COMPLETE")
    print("Investor degree diagnostics:       COMPLETE")
    print("Primary cold-start startup slope:  COMPLETE")
    print("Investor-cluster uncertainty:      COMPLETE")
    print("Presentation-ready figures:        COMPLETE")
    print("Provenance manifest:               COMPLETE")

    print()
    print("Model inference executed:          NO")
    print("Final test rescored:               NO")
    print("Final test modified:               NO")

    print()
    print(
        "PHASE 7.5C DEGREE–PERFORMANCE "
        "DIAGNOSTIC STATUS: COMPLETE"
    )


if __name__ == "__main__":
    main()