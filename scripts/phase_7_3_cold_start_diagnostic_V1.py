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
# Phase 7.3 — Cold-Start Diagnostic V1
#
# Scientific question:
#   How does frozen ITRS ranking performance vary across investor/startup
#   cold-start configurations?
#
# Scientific boundary:
#   - immutable Phase-6 one-shot final test
#   - post-hoc analysis only
#   - no model loading
#   - no checkpoint loading
#   - no raw-logit loading
#   - no inference
#   - no test rescoring
#   - no negative resampling
#   - no candidate modification
#   - no model selection
#
# Critical interpretation warning:
#   Cold-start configuration and pair novelty are NOT independent.
#
#   All cases involving a cold investor and/or cold startup are new-to-investor.
#   Warm/warm cases contain BOTH:
#       - prior relationships
#       - new-to-investor relationships
#
#   Therefore Phase 7.3 is descriptive/unadjusted.
#   Phase 7.4 will perform the key novelty × cold-start stratification.
# =============================================================================


REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    REPO_ROOT
    / "data/experimental/phase_6/full_training/100pct/final_test/"
    / "analysis_ready_test_cases.parquet"
)

PHASE_7_0_CONTRACT = (
    REPO_ROOT
    / "data/experimental/phase_7/phase_7_0_analysis_contract/"
    / "phase_7_0_analysis_contract_V2.json"
)

PHASE_7_2_MANIFEST = (
    REPO_ROOT
    / "data/experimental/phase_7/phase_7_2_pair_novelty/"
    / "phase_7_2_analysis_manifest_V1.json"
)

OUT_DIR = (
    REPO_ROOT
    / "data/experimental/phase_7/phase_7_3_cold_start"
)

FIG_DIR = OUT_DIR / "figures"

EXPECTED_SOURCE_SHA256 = (
    "d70f21bff0006e094c5d567d307c0664"
    "b41a11e7250a0e69aaec610e1810132d"
)

EXPECTED_ROWS = 20_264

EXPECTED_COUNTS = {
    "Warm investor + Warm startup": 10_707,
    "Warm investor + Cold startup": 6_392,
    "Cold investor + Warm startup": 1_453,
    "Cold investor + Cold startup": 1_712,
}

REGIME_ORDER = [
    "Warm investor + Warm startup",
    "Warm investor + Cold startup",
    "Cold investor + Warm startup",
    "Cold investor + Cold startup",
]

RANK_LABELS = [
    "Rank 1",
    "Ranks 2–5",
    "Ranks 6–10",
    "Ranks 11–20",
    "Ranks 21–50",
    "Ranks 51–100",
]

BOOTSTRAP_REPS = 2_000
BOOTSTRAP_SEED = 7_103_001


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
        f"Object of type {type(value).__name__} "
        "is not JSON serializable"
    )


def json_dump(obj, path: Path) -> None:

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

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


def save_figure(
    fig,
    stem: str,
) -> list[Path]:

    png_path = FIG_DIR / f"{stem}.png"
    pdf_path = FIG_DIR / f"{stem}.pdf"

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={
            "Title": stem,
            "Author": "ITRS Phase 7 diagnostic analysis",
        },
    )

    plt.close(fig)

    return [
        png_path,
        pdf_path,
    ]


def calculate_metrics(
    group: pd.DataFrame,
) -> dict:

    rank = (
        group["positive_rank"]
        .astype(int)
    )

    return {

        "n": int(len(group)),

        "Hits@1": int(
            (rank <= 1).sum()
        ),

        "Hits@5": int(
            (rank <= 5).sum()
        ),

        "Hits@10": int(
            (rank <= 10).sum()
        ),

        "HR@1": float(
            (rank <= 1).mean()
        ),

        "HR@5": float(
            (rank <= 5).mean()
        ),

        "HR@10": float(
            (rank <= 10).mean()
        ),

        "NDCG@10": float(
            group["NDCG@10"].mean()
        ),

        "mean_rank": float(
            rank.mean()
        ),

        "median_rank": float(
            rank.median()
        ),

        "std_rank": float(
            rank.std(ddof=1)
        ),

        "P25_rank": float(
            rank.quantile(0.25)
        ),

        "P75_rank": float(
            rank.quantile(0.75)
        ),

        "P90_rank": float(
            rank.quantile(0.90)
        ),

        "P95_rank": float(
            rank.quantile(0.95)
        ),

        "rank_gt_10_n": int(
            (rank > 10).sum()
        ),

        "rank_gt_10_share": float(
            (rank > 10).mean()
        ),

        "rank_gt_20_n": int(
            (rank > 20).sum()
        ),

        "rank_gt_20_share": float(
            (rank > 20).mean()
        ),

        "rank_gt_50_n": int(
            (rank > 50).sum()
        ),

        "rank_gt_50_share": float(
            (rank > 50).mean()
        ),

        "rank_gt_75_n": int(
            (rank > 75).sum()
        ),

        "rank_gt_75_share": float(
            (rank > 75).mean()
        ),
    }


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


def investor_cluster_bootstrap(
    df: pd.DataFrame,
    reps: int,
    seed: int,
) -> pd.DataFrame:
    """
    One-way cluster bootstrap by investor_id.

    All test events from a sampled investor are carried together.

    This addresses within-investor dependence only.
    Startup dependence will be revisited during Phase 7 robustness analysis.
    """

    working = df[
        [
            "investor_id",
            "cold_start_regime",
            "positive_rank",
            "NDCG@10",
        ]
    ].copy()

    working["hit10"] = (
        working["positive_rank"]
        .le(10)
        .astype(int)
    )

    investors = (
        working[
            "investor_id"
        ]
        .drop_duplicates()
        .to_numpy()
    )

    n_clusters = len(investors)

    arrays = {}

    for regime in REGIME_ORDER:

        grouped = (
            working.loc[
                working[
                    "cold_start_regime"
                ]
                == regime
            ]
            .groupby(
                "investor_id",
                sort=False,
            )
            .agg(
                n=(
                    "positive_rank",
                    "size",
                ),
                hits10=(
                    "hit10",
                    "sum",
                ),
                ndcg_sum=(
                    "NDCG@10",
                    "sum",
                ),
                rank_sum=(
                    "positive_rank",
                    "sum",
                ),
            )
            .reindex(
                investors,
                fill_value=0,
            )
        )

        arrays[regime] = {
            key: (
                grouped[key]
                .to_numpy(
                    dtype=float
                )
            )
            for key in (
                "n",
                "hits10",
                "ndcg_sum",
                "rank_sum",
            )
        }

    rng = np.random.default_rng(
        seed
    )

    rows = []

    reference = (
        "Warm investor + Warm startup"
    )

    for bootstrap_index in range(
        reps
    ):

        sampled = rng.integers(
            0,
            n_clusters,
            size=n_clusters,
        )

        result = {
            "bootstrap_index":
                bootstrap_index,
        }

        valid = True

        for regime in REGIME_ORDER:

            regime_arrays = (
                arrays[regime]
            )

            n = float(
                regime_arrays[
                    "n"
                ][sampled].sum()
            )

            if n <= 0:

                valid = False
                break

            result[
                f"{regime}__HR@10"
            ] = float(
                regime_arrays[
                    "hits10"
                ][sampled].sum()
                / n
            )

            result[
                f"{regime}__NDCG@10"
            ] = float(
                regime_arrays[
                    "ndcg_sum"
                ][sampled].sum()
                / n
            )

            result[
                f"{regime}__mean_rank"
            ] = float(
                regime_arrays[
                    "rank_sum"
                ][sampled].sum()
                / n
            )

        if not valid:
            continue

        for regime in REGIME_ORDER[1:]:

            result[
                f"{regime}__delta_HR@10_vs_warm_warm"
            ] = (
                result[
                    f"{regime}__HR@10"
                ]
                - result[
                    f"{reference}__HR@10"
                ]
            )

            result[
                f"{regime}__delta_NDCG@10_vs_warm_warm"
            ] = (
                result[
                    f"{regime}__NDCG@10"
                ]
                - result[
                    f"{reference}__NDCG@10"
                ]
            )

            result[
                f"{regime}__delta_mean_rank_vs_warm_warm"
            ] = (
                result[
                    f"{regime}__mean_rank"
                ]
                - result[
                    f"{reference}__mean_rank"
                ]
            )

        rows.append(
            result
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
        "PHASE 7.3 — COLD-START DIAGNOSTIC V1"
    )
    print("=" * 110)

    print(
        "Scientific role:             "
        "POST-HOC UNADJUSTED COLD-START DIAGNOSTIC"
    )

    print("Model loaded:                NO")
    print("Checkpoint loaded:           NO")
    print("Raw logits loaded:           NO")
    print("Inference executed:          NO")
    print("Test rescored:               NO")
    print("Model selection performed:   NO")

    # =========================================================================
    # 7.3.1 — Integrity gate
    # =========================================================================

    print_section(
        "7.3.1 — ANALYSIS INTEGRITY GATE"
    )

    for path in (
        SOURCE,
        PHASE_7_0_CONTRACT,
        PHASE_7_2_MANIFEST,
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
        "Frozen final-test bundle SHA256 drift.",
    )

    with PHASE_7_0_CONTRACT.open(
        "r",
        encoding="utf-8",
    ) as f:

        phase_7_0 = json.load(f)

    with PHASE_7_2_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as f:

        phase_7_2 = json.load(f)

    require(
        phase_7_0["status"]
        == "PASS",
        "Phase 7.0 contract is not PASS.",
    )

    require(
        phase_7_2["status"]
        == "COMPLETE",
        "Phase 7.2 is not COMPLETE.",
    )

    print()
    print("Frozen source fingerprint:   PASS")
    print("Phase 7.0 contract:           PASS")
    print("Phase 7.2 prerequisite:       PASS")

    # =========================================================================
    # 7.3.2 — Load frozen cases / construct regime
    # =========================================================================

    print_section(
        "7.3.2 — CONSTRUCT FROZEN COLD-START REGIMES"
    )

    df = pd.read_parquet(
        SOURCE
    )

    require(
        len(df) == EXPECTED_ROWS,
        "Final-test row-count drift.",
    )

    require(
        df["interaction_id"].is_unique,
        "interaction_id is not unique.",
    )

    df[
        "cold_start_regime"
    ] = np.select(
        [
            (
                ~df[
                    "cold_start_investor"
                ]
                &
                ~df[
                    "cold_start_startup"
                ]
            ),
            (
                ~df[
                    "cold_start_investor"
                ]
                &
                df[
                    "cold_start_startup"
                ]
            ),
            (
                df[
                    "cold_start_investor"
                ]
                &
                ~df[
                    "cold_start_startup"
                ]
            ),
            (
                df[
                    "cold_start_investor"
                ]
                &
                df[
                    "cold_start_startup"
                ]
            ),
        ],
        REGIME_ORDER,
        default="INVALID",
    )

    require(
        ~df[
            "cold_start_regime"
        ].eq("INVALID").any(),
        "Invalid cold-start regime encountered.",
    )

    # Verify against frozen interaction_cold_start_status.
    expected_frozen_status = {
        "Warm investor + Warm startup":
            "warm_investor__warm_startup",
        "Warm investor + Cold startup":
            "warm_investor__cold_startup",
        "Cold investor + Warm startup":
            "cold_investor__warm_startup",
        "Cold investor + Cold startup":
            "cold_investor__cold_startup",
    }

    for regime, frozen_status in (
        expected_frozen_status.items()
    ):

        rows = df.loc[
            df["cold_start_regime"]
            == regime
        ]

        require(
            rows[
                "interaction_cold_start_status"
            ]
            .eq(
                frozen_status
            )
            .all(),
            (
                f"Frozen interaction cold-start "
                f"status mismatch for {regime}."
            ),
        )

    counts = (
        df[
            "cold_start_regime"
        ]
        .value_counts()
    )

    for regime in REGIME_ORDER:

        observed = int(
            counts[regime]
        )

        expected = (
            EXPECTED_COUNTS[
                regime
            ]
        )

        require(
            observed == expected,
            (
                f"Count drift for {regime}: "
                f"expected {expected}, "
                f"observed {observed}."
            ),
        )

        print(
            f"{regime:<34} "
            f"{observed:>7,} "
            f"({observed / len(df):6.2%})"
        )

    print()
    print(
        "Frozen cold-start semantics:  PASS"
    )

    # =========================================================================
    # 7.3.3 — Pair-novelty composition
    # =========================================================================

    print_section(
        "7.3.3 — PAIR-NOVELTY COMPOSITION OF COLD-START REGIMES"
    )

    df[
        "pair_regime"
    ] = np.where(
        df[
            "new_to_investor_pair"
        ],
        "New-to-investor",
        "Prior relationship",
    )

    novelty_composition = (
        df.groupby(
            [
                "cold_start_regime",
                "pair_regime",
            ],
            observed=False,
        )
        .size()
        .rename("n")
        .reset_index()
    )

    regime_sizes = (
        df[
            "cold_start_regime"
        ]
        .value_counts()
    )

    novelty_composition[
        "within_cold_regime_share"
    ] = (
        novelty_composition[
            "n"
        ]
        / novelty_composition[
            "cold_start_regime"
        ].map(
            regime_sizes
        )
    )

    print(
        novelty_composition
        .sort_values(
            [
                "cold_start_regime",
                "pair_regime",
            ]
        )
        .to_string(
            index=False,
            formatters={
                "within_cold_regime_share":
                    lambda x: f"{x:.2%}",
            },
        )
    )

    for regime in REGIME_ORDER[1:]:

        subset = df.loc[
            df[
                "cold_start_regime"
            ]
            == regime
        ]

        require(
            subset[
                "new_to_investor_pair"
            ].all(),
            (
                f"Cold regime {regime} "
                "contains a prior relationship."
            ),
        )

    warm_warm = df.loc[
        df[
            "cold_start_regime"
        ]
        == REGIME_ORDER[0]
    ]

    warm_warm_new_n = int(
        warm_warm[
            "new_to_investor_pair"
        ].sum()
    )

    warm_warm_prior_n = int(
        (
            ~warm_warm[
                "new_to_investor_pair"
            ]
        ).sum()
    )

    print()
    print(
        f"Warm/warm new-to-investor:    "
        f"{warm_warm_new_n:,}"
    )

    print(
        f"Warm/warm prior relationship: "
        f"{warm_warm_prior_n:,}"
    )

    # =========================================================================
    # 7.3.4 — Cold-start performance
    # =========================================================================

    print_section(
        "7.3.4 — PERFORMANCE BY COLD-START REGIME"
    )

    metric_rows = []

    for regime in REGIME_ORDER:

        subgroup = df.loc[
            df[
                "cold_start_regime"
            ]
            == regime
        ]

        metrics = calculate_metrics(
            subgroup
        )

        metrics[
            "cold_start_regime"
        ] = regime

        metrics[
            "test_share"
        ] = (
            len(subgroup)
            / len(df)
        )

        metrics[
            "new_to_investor_share"
        ] = float(
            subgroup[
                "new_to_investor_pair"
            ].mean()
        )

        metric_rows.append(
            metrics
        )

    metrics_df = pd.DataFrame(
        metric_rows
    )

    metrics_df = metrics_df[
        [
            "cold_start_regime",
            "n",
            "test_share",
            "new_to_investor_share",
            "Hits@1",
            "HR@1",
            "Hits@5",
            "HR@5",
            "Hits@10",
            "HR@10",
            "NDCG@10",
            "mean_rank",
            "median_rank",
            "std_rank",
            "P25_rank",
            "P75_rank",
            "P90_rank",
            "P95_rank",
            "rank_gt_10_n",
            "rank_gt_10_share",
            "rank_gt_20_n",
            "rank_gt_20_share",
            "rank_gt_50_n",
            "rank_gt_50_share",
            "rank_gt_75_n",
            "rank_gt_75_share",
        ]
    ]

    print(
        metrics_df[
            [
                "cold_start_regime",
                "n",
                "new_to_investor_share",
                "HR@1",
                "HR@5",
                "HR@10",
                "NDCG@10",
                "mean_rank",
                "median_rank",
                "P75_rank",
                "rank_gt_50_share",
            ]
        ].to_string(
            index=False,
            formatters={
                "new_to_investor_share":
                    lambda x: f"{x:.2%}",
                "HR@1":
                    lambda x: f"{x:.4f}",
                "HR@5":
                    lambda x: f"{x:.4f}",
                "HR@10":
                    lambda x: f"{x:.4f}",
                "NDCG@10":
                    lambda x: f"{x:.4f}",
                "mean_rank":
                    lambda x: f"{x:.2f}",
                "median_rank":
                    lambda x: f"{x:.1f}",
                "P75_rank":
                    lambda x: f"{x:.1f}",
                "rank_gt_50_share":
                    lambda x: f"{x:.2%}",
            },
        )
    )

    # =========================================================================
    # 7.3.5 — Raw contrasts against warm/warm
    # =========================================================================

    print_section(
        "7.3.5 — RAW CONTRASTS VS WARM INVESTOR + WARM STARTUP"
    )

    indexed = (
        metrics_df
        .set_index(
            "cold_start_regime"
        )
    )

    reference_name = (
        "Warm investor + Warm startup"
    )

    reference = indexed.loc[
        reference_name
    ]

    contrast_rows = []

    for regime in REGIME_ORDER[1:]:

        row = indexed.loc[
            regime
        ]

        contrast_rows.append(
            {
                "comparison":
                    f"{regime} - {reference_name}",

                "cold_start_regime":
                    regime,

                "delta_HR@1":
                    float(
                        row["HR@1"]
                        - reference["HR@1"]
                    ),

                "delta_HR@5":
                    float(
                        row["HR@5"]
                        - reference["HR@5"]
                    ),

                "delta_HR@10":
                    float(
                        row["HR@10"]
                        - reference["HR@10"]
                    ),

                "delta_NDCG@10":
                    float(
                        row["NDCG@10"]
                        - reference["NDCG@10"]
                    ),

                "delta_mean_rank":
                    float(
                        row["mean_rank"]
                        - reference["mean_rank"]
                    ),

                "delta_median_rank":
                    float(
                        row["median_rank"]
                        - reference["median_rank"]
                    ),

                "delta_rank_gt_50_share":
                    float(
                        row[
                            "rank_gt_50_share"
                        ]
                        - reference[
                            "rank_gt_50_share"
                        ]
                    ),
            }
        )

    contrasts_df = pd.DataFrame(
        contrast_rows
    )

    print(
        contrasts_df.to_string(
            index=False,
            formatters={
                "delta_HR@1":
                    lambda x: f"{x:+.4f}",
                "delta_HR@5":
                    lambda x: f"{x:+.4f}",
                "delta_HR@10":
                    lambda x: f"{x:+.4f}",
                "delta_NDCG@10":
                    lambda x: f"{x:+.4f}",
                "delta_mean_rank":
                    lambda x: f"{x:+.2f}",
                "delta_median_rank":
                    lambda x: f"{x:+.2f}",
                "delta_rank_gt_50_share":
                    lambda x: f"{x:+.2%}",
            },
        )
    )

    # =========================================================================
    # 7.3.6 — Rank bands
    # =========================================================================

    print_section(
        "7.3.6 — RANK-BAND DECOMPOSITION"
    )

    df["rank_band"] = pd.cut(
        df["positive_rank"],
        bins=[
            0,
            1,
            5,
            10,
            20,
            50,
            100,
        ],
        labels=RANK_LABELS,
        include_lowest=True,
        ordered=True,
    )

    rank_bands = (
        df.groupby(
            [
                "cold_start_regime",
                "rank_band",
            ],
            observed=False,
        )
        .size()
        .rename("n")
        .reset_index()
    )

    rank_bands[
        "within_regime_share"
    ] = (
        rank_bands["n"]
        / rank_bands[
            "cold_start_regime"
        ].map(
            regime_sizes
        )
    )

    print(
        rank_bands.to_string(
            index=False,
            formatters={
                "within_regime_share":
                    lambda x: f"{x:.2%}",
            },
        )
    )

    # =========================================================================
    # 7.3.7 — HR@k curves
    # =========================================================================

    print_section(
        "7.3.7 — CUMULATIVE HR@K BY COLD-START REGIME"
    )

    cdf_rows = []

    for regime in REGIME_ORDER:

        ranks = (
            df.loc[
                df[
                    "cold_start_regime"
                ]
                == regime,
                "positive_rank",
            ]
            .astype(int)
        )

        for k in range(
            1,
            101,
        ):

            cdf_rows.append(
                {
                    "cold_start_regime":
                        regime,
                    "k":
                        k,
                    "HR@k":
                        float(
                            (ranks <= k)
                            .mean()
                        ),
                }
            )

    cdf_df = pd.DataFrame(
        cdf_rows
    )

    for cutoff in (
        1,
        5,
        10,
        20,
        50,
        75,
    ):

        print()
        print(
            f"Cutoff k={cutoff}"
        )

        subset = cdf_df.loc[
            cdf_df["k"]
            == cutoff
        ]

        for regime in REGIME_ORDER:

            value = float(
                subset.loc[
                    subset[
                        "cold_start_regime"
                    ]
                    == regime,
                    "HR@k",
                ].iloc[0]
            )

            print(
                f"  {regime:<34} "
                f"{value:.2%}"
            )

    # =========================================================================
    # 7.3.8 — Investor-cluster bootstrap
    # =========================================================================

    print_section(
        "7.3.8 — INVESTOR-CLUSTER BOOTSTRAP UNCERTAINTY"
    )

    print(
        f"Bootstrap replicates:         "
        f"{BOOTSTRAP_REPS:,}"
    )

    print(
        f"Bootstrap seed:               "
        f"{BOOTSTRAP_SEED}"
    )

    print(
        f"Investor clusters:            "
        f"{df['investor_id'].nunique():,}"
    )

    bootstrap_df = (
        investor_cluster_bootstrap(
            df=df,
            reps=BOOTSTRAP_REPS,
            seed=BOOTSTRAP_SEED,
        )
    )

    require(
        len(bootstrap_df)
        == BOOTSTRAP_REPS,
        (
            "Unexpected number of valid "
            "bootstrap replicates."
        ),
    )

    bootstrap_summary_rows = []

    for regime in REGIME_ORDER:

        actual = indexed.loc[
            regime
        ]

        for metric in (
            "HR@10",
            "NDCG@10",
            "mean_rank",
        ):

            column = (
                f"{regime}__{metric}"
            )

            low, high = percentile_ci(
                bootstrap_df[
                    column
                ]
            )

            bootstrap_summary_rows.append(
                {
                    "cold_start_regime":
                        regime,

                    "comparison":
                        "absolute",

                    "metric":
                        metric,

                    "estimate":
                        float(
                            actual[
                                metric
                            ]
                        ),

                    "ci95_low":
                        low,

                    "ci95_high":
                        high,

                    "bootstrap_unit":
                        "investor_id",

                    "repetitions":
                        BOOTSTRAP_REPS,
                }
            )

    contrast_lookup = (
        contrasts_df
        .set_index(
            "cold_start_regime"
        )
    )

    for regime in REGIME_ORDER[1:]:

        mapping = {
            "HR@10":
                (
                    f"{regime}"
                    "__delta_HR@10_vs_warm_warm"
                ),

            "NDCG@10":
                (
                    f"{regime}"
                    "__delta_NDCG@10_vs_warm_warm"
                ),

            "mean_rank":
                (
                    f"{regime}"
                    "__delta_mean_rank_vs_warm_warm"
                ),
        }

        actual_mapping = {
            "HR@10":
                "delta_HR@10",
            "NDCG@10":
                "delta_NDCG@10",
            "mean_rank":
                "delta_mean_rank",
        }

        for metric, column in (
            mapping.items()
        ):

            low, high = percentile_ci(
                bootstrap_df[
                    column
                ]
            )

            bootstrap_summary_rows.append(
                {
                    "cold_start_regime":
                        regime,

                    "comparison":
                        (
                            "difference_vs_"
                            "warm_investor_warm_startup"
                        ),

                    "metric":
                        metric,

                    "estimate":
                        float(
                            contrast_lookup.loc[
                                regime,
                                actual_mapping[
                                    metric
                                ],
                            ]
                        ),

                    "ci95_low":
                        low,

                    "ci95_high":
                        high,

                    "bootstrap_unit":
                        "investor_id",

                    "repetitions":
                        BOOTSTRAP_REPS,
                }
            )

    bootstrap_summary = pd.DataFrame(
        bootstrap_summary_rows
    )

    contrast_bootstrap = (
        bootstrap_summary.loc[
            bootstrap_summary[
                "comparison"
            ]
            != "absolute"
        ]
    )

    print(
        contrast_bootstrap.to_string(
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
    # 7.3.9 — Interpretation audit
    # =========================================================================

    print_section(
        "7.3.9 — INTERPRETATION AUDIT"
    )

    print(
        "Observed design structure:"
    )

    print(
        "  Warm/warm contains BOTH new-to-investor "
        "and prior relationships."
    )

    print(
        "  Every regime containing a cold entity "
        "contains only new-to-investor cases."
    )

    print()

    print(
        "Therefore:"
    )

    print(
        "  Raw cold-start contrasts are "
        "DESCRIPTIVE ASSOCIATIONS."
    )

    print(
        "  They do not isolate the effect of "
        "cold start from pair novelty."
    )

    print()

    print(
        "Phase 7.4 will resolve this by:"
    )

    print(
        "  A) comparing new vs prior within "
        "warm/warm cases;"
    )

    print(
        "  B) comparing cold-start regimes "
        "within new-to-investor cases only."
    )

    # =========================================================================
    # 7.3.10 — Write artifacts
    # =========================================================================

    print_section(
        "7.3.10 — WRITE ANALYSIS ARTIFACTS"
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        OUT_DIR
        / "phase_7_3_cold_start_metrics_V1.csv"
    )

    novelty_composition_path = (
        OUT_DIR
        / "phase_7_3_pair_novelty_composition_V1.csv"
    )

    contrasts_path = (
        OUT_DIR
        / "phase_7_3_raw_contrasts_V1.csv"
    )

    rank_bands_path = (
        OUT_DIR
        / "phase_7_3_rank_bands_V1.csv"
    )

    cdf_path = (
        OUT_DIR
        / "phase_7_3_rank_cdf_V1.csv"
    )

    bootstrap_reps_path = (
        OUT_DIR
        / "phase_7_3_investor_cluster_bootstrap_replicates_V1.csv"
    )

    bootstrap_summary_path = (
        OUT_DIR
        / "phase_7_3_investor_cluster_bootstrap_summary_V1.csv"
    )

    insights_path = (
        OUT_DIR
        / "phase_7_3_descriptive_insights_V1.json"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    novelty_composition.to_csv(
        novelty_composition_path,
        index=False,
    )

    contrasts_df.to_csv(
        contrasts_path,
        index=False,
    )

    rank_bands.to_csv(
        rank_bands_path,
        index=False,
    )

    cdf_df.to_csv(
        cdf_path,
        index=False,
    )

    bootstrap_df.to_csv(
        bootstrap_reps_path,
        index=False,
    )

    bootstrap_summary.to_csv(
        bootstrap_summary_path,
        index=False,
    )

    insights = {

        "phase":
            "7.3",

        "scientific_scope":
            (
                "Unadjusted descriptive comparison "
                "across frozen investor/startup "
                "cold-start regimes."
            ),

        "group_sizes":
            EXPECTED_COUNTS,

        "warm_warm_pair_composition": {
            "new_to_investor_n":
                warm_warm_new_n,
            "prior_relationship_n":
                warm_warm_prior_n,
            "new_to_investor_share":
                (
                    warm_warm_new_n
                    / EXPECTED_COUNTS[
                        reference_name
                    ]
                ),
        },

        "design_constraint": (
            "All regimes involving any cold-start "
            "entity are new-to-investor. Warm/warm "
            "contains both new and prior pairs."
        ),

        "uncertainty": {
            "method":
                (
                    "Percentile bootstrap with "
                    "investor_id as resampling unit."
                ),
            "repetitions":
                BOOTSTRAP_REPS,
            "seed":
                BOOTSTRAP_SEED,
            "limitation":
                (
                    "One-way investor clustering only; "
                    "startup dependence and multiway "
                    "robustness are deferred."
                ),
        },

        "interpretation_boundary": (
            "No causal or isolated cold-start effect "
            "can be inferred from Phase 7.3."
        ),
    }

    json_dump(
        insights,
        insights_path,
    )

    # =========================================================================
    # 7.3.11 — Figure 1: top-k performance
    # =========================================================================

    print_section(
        "7.3.11 — GENERATE PRESENTATION-READY FIGURES"
    )

    figure_paths = []

    topk_metrics = [
        "HR@1",
        "HR@5",
        "HR@10",
    ]

    x = np.arange(
        len(REGIME_ORDER)
    )

    width = 0.24

    fig, ax = plt.subplots(
        figsize=(13, 7)
    )

    for metric_i, metric in enumerate(
        topk_metrics
    ):

        values = (
            indexed
            .loc[
                REGIME_ORDER,
                metric,
            ]
            .to_numpy()
            * 100.0
        )

        bars = ax.bar(
            x
            + (
                metric_i - 1
            ) * width,
            values,
            width=width,
            label=metric,
        )

        for bar, value in zip(
            bars,
            values,
        ):

            ax.annotate(
                f"{value:.1f}%",
                (
                    bar.get_x()
                    + bar.get_width() / 2,
                    bar.get_height(),
                ),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    ax.set_xticks(
        x,
        REGIME_ORDER,
        rotation=15,
        ha="right",
    )

    ax.set_ylabel(
        "Hit rate (%)"
    )

    ax.set_title(
        "ITRS retrieval performance by cold-start configuration"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.20,
    )

    fig.tight_layout()

    figure_paths.extend(
        save_figure(
            fig,
            (
                "phase_7_3_fig1_"
                "topk_by_cold_start_regime_V1"
            ),
        )
    )

    # =========================================================================
    # Figure 2: HR@k curves
    # =========================================================================

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    for regime in REGIME_ORDER:

        subset = cdf_df.loc[
            cdf_df[
                "cold_start_regime"
            ]
            == regime
        ]

        ax.plot(
            subset["k"],
            subset["HR@k"]
            * 100.0,
            linewidth=2.0,
            label=regime,
        )

    ax.axvline(
        10,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_xlim(
        1,
        100,
    )

    ax.set_ylim(
        0,
        101,
    )

    ax.set_xlabel(
        "Recommendation cutoff k"
    )

    ax.set_ylabel(
        "Cumulative hit rate (%)"
    )

    ax.set_title(
        "Positive-startup retrieval curves by cold-start regime"
    )

    ax.legend(
        fontsize=9,
    )

    ax.grid(
        alpha=0.20,
    )

    fig.tight_layout()

    figure_paths.extend(
        save_figure(
            fig,
            (
                "phase_7_3_fig2_"
                "rank_cdf_by_cold_start_regime_V1"
            ),
        )
    )

    # =========================================================================
    # Figure 3: rank-band composition
    # =========================================================================

    rank_pivot = (
        rank_bands.pivot(
            index="cold_start_regime",
            columns="rank_band",
            values="within_regime_share",
        )
        .reindex(
            REGIME_ORDER
        )
        .reindex(
            columns=RANK_LABELS
        )
        .fillna(0.0)
        * 100.0
    )

    fig, ax = plt.subplots(
        figsize=(13, 7)
    )

    bottom = np.zeros(
        len(rank_pivot)
    )

    x = np.arange(
        len(rank_pivot)
    )

    for band in RANK_LABELS:

        values = (
            rank_pivot[
                band
            ]
            .to_numpy()
        )

        ax.bar(
            x,
            values,
            bottom=bottom,
            label=band,
        )

        bottom += values

    ax.set_xticks(
        x,
        rank_pivot.index,
        rotation=15,
        ha="right",
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.set_ylabel(
        "Share of subgroup (%)"
    )

    ax.set_title(
        "Rank-error composition by cold-start regime"
    )

    ax.legend(
        bbox_to_anchor=(
            1.02,
            1,
        ),
        loc="upper left",
    )

    ax.grid(
        axis="y",
        alpha=0.15,
    )

    fig.tight_layout()

    figure_paths.extend(
        save_figure(
            fig,
            (
                "phase_7_3_fig3_"
                "rank_band_composition_V1"
            ),
        )
    )

    # =========================================================================
    # Figure 4: novelty composition — explicit confounding visualization
    # =========================================================================

    novelty_pivot = (
        novelty_composition.pivot(
            index="cold_start_regime",
            columns="pair_regime",
            values="within_cold_regime_share",
        )
        .reindex(
            REGIME_ORDER
        )
        .reindex(
            columns=[
                "Prior relationship",
                "New-to-investor",
            ]
        )
        .fillna(0.0)
        * 100.0
    )

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    bottom = np.zeros(
        len(novelty_pivot)
    )

    x = np.arange(
        len(novelty_pivot)
    )

    for pair_regime in (
        "Prior relationship",
        "New-to-investor",
    ):

        values = (
            novelty_pivot[
                pair_regime
            ]
            .to_numpy()
        )

        bars = ax.bar(
            x,
            values,
            bottom=bottom,
            label=pair_regime,
        )

        bottom += values

    ax.set_xticks(
        x,
        novelty_pivot.index,
        rotation=15,
        ha="right",
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.set_ylabel(
        "Share of cold-start regime (%)"
    )

    ax.set_title(
        "Pair novelty is structurally confounded with cold-start status"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.15,
    )

    fig.tight_layout()

    figure_paths.extend(
        save_figure(
            fig,
            (
                "phase_7_3_fig4_"
                "pair_novelty_composition_by_cold_start_V1"
            ),
        )
    )

    # =========================================================================
    # 7.3.12 — Provenance manifest
    # =========================================================================

    manifest_path = (
        OUT_DIR
        / "phase_7_3_analysis_manifest_V1.json"
    )

    manifest = {

        "phase":
            "7.3",

        "schema_version":
            "ITRS_PHASE7_3_COLD_START_DIAGNOSTIC_V1",

        "status":
            "COMPLETE",

        "scientific_role":
            (
                "Post-hoc unadjusted cold-start "
                "diagnostic on the immutable "
                "Phase-6 final test."
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

        "group_order":
            REGIME_ORDER,

        "prerequisites": {
            "phase_7_0":
                str(
                    PHASE_7_0_CONTRACT.relative_to(
                        REPO_ROOT
                    )
                ),
            "phase_7_2":
                str(
                    PHASE_7_2_MANIFEST.relative_to(
                        REPO_ROOT
                    )
                ),
        },

        "uncertainty_method": {
            "method":
                (
                    "One-way investor-cluster "
                    "percentile bootstrap"
                ),
            "repetitions":
                BOOTSTRAP_REPS,
            "seed":
                BOOTSTRAP_SEED,
        },

        "interpretation_boundary":
            (
                "Cold-start and pair novelty are "
                "structurally associated in this "
                "test set. Phase 7.3 does not "
                "identify an isolated cold-start "
                "effect."
            ),

        "tables": [
            str(
                metrics_path.relative_to(
                    REPO_ROOT
                )
            ),
            str(
                novelty_composition_path.relative_to(
                    REPO_ROOT
                )
            ),
            str(
                contrasts_path.relative_to(
                    REPO_ROOT
                )
            ),
            str(
                rank_bands_path.relative_to(
                    REPO_ROOT
                )
            ),
            str(
                cdf_path.relative_to(
                    REPO_ROOT
                )
            ),
            str(
                bootstrap_summary_path.relative_to(
                    REPO_ROOT
                )
            ),
        ],

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
            "model_selection":
                False,
        },
    }

    json_dump(
        manifest,
        manifest_path,
    )

    # =========================================================================
    # Artifact hashes
    # =========================================================================

    artifact_paths = [
        metrics_path,
        novelty_composition_path,
        contrasts_path,
        rank_bands_path,
        cdf_path,
        bootstrap_reps_path,
        bootstrap_summary_path,
        insights_path,
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
        / "phase_7_3_derived_artifact_sha256_V1.json"
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
        "PHASE 7.3 V1 RESULT"
    )

    print(
        "Frozen source binding:              PASS"
    )

    print(
        "Cold-start semantic binding:        PASS"
    )

    print(
        "Frozen subgroup counts:             PASS"
    )

    print(
        "Pair-novelty composition audit:     COMPLETE"
    )

    print(
        "Cold-start performance metrics:     COMPLETE"
    )

    print(
        "Raw warm/warm contrasts:            COMPLETE"
    )

    print(
        "Rank-distribution diagnostics:      COMPLETE"
    )

    print(
        "Investor-cluster uncertainty:       COMPLETE"
    )

    print(
        "Presentation-ready figures:         COMPLETE"
    )

    print(
        "Provenance manifest:                COMPLETE"
    )

    print()
    print(
        "Model inference executed:           NO"
    )

    print(
        "Final test rescored:                NO"
    )

    print(
        "Final test modified:                NO"
    )

    print()
    print(
        "IMPORTANT: COLD-START AND PAIR NOVELTY "
        "ARE NOT YET ISOLATED."
    )

    print()
    print(
        "PHASE 7.3 COLD-START "
        "DIAGNOSTIC STATUS: COMPLETE"
    )


if __name__ == "__main__":
    main()