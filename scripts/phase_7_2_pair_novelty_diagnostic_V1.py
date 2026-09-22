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
# Phase 7.2 — Pair-Novelty Diagnostic V1
#
# Question:
#   Does the reproduced ITRS rank future investment targets differently for
#   genuinely new investor-startup pairs versus previously observed pairs?
#
# Scientific boundary:
#   - immutable Phase-6 final test
#   - post-hoc analysis only
#   - no inference
#   - no rescoring
#   - no candidate modification
#   - no model/checkpoint selection
#
# Important:
#   This is an UNADJUSTED pair-novelty comparison.
#
#   Cold-start observations are known to occur only in the new-to-investor
#   population. Therefore the raw new-vs-prior contrast MUST NOT be interpreted
#   as the isolated effect of pair novelty.
#
# Uncertainty:
#   A one-way investor-cluster bootstrap is included because repeated events
#   from the same investor violate simple event-independence assumptions.
#
#   This does NOT yet handle startup clustering or all multiway dependence.
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

PHASE_7_1_MANIFEST = (
    REPO_ROOT
    / "data/experimental/phase_7/phase_7_1_baseline_rank_anatomy/"
    / "phase_7_1_analysis_manifest_V1.json"
)

OUT_DIR = (
    REPO_ROOT
    / "data/experimental/phase_7/phase_7_2_pair_novelty"
)

FIG_DIR = OUT_DIR / "figures"

EXPECTED_SOURCE_SHA256 = (
    "d70f21bff0006e094c5d567d307c0664"
    "b41a11e7250a0e69aaec610e1810132d"
)

EXPECTED_ROWS = 20_264

EXPECTED_NEW_N = 16_446
EXPECTED_PRIOR_N = 3_818

BOOTSTRAP_REPS = 2_000
BOOTSTRAP_SEED = 7_102_001

PAIR_ORDER = [
    "Prior relationship",
    "New-to-investor",
]

FLOAT_TOL = 5e-13


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

    png_path = (
        FIG_DIR
        / f"{stem}.png"
    )

    pdf_path = (
        FIG_DIR
        / f"{stem}.pdf"
    )

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
            "Author": (
                "ITRS Phase 7 diagnostic analysis"
            ),
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

    n = len(group)

    return {

        "n": int(n),

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


def investor_cluster_bootstrap(
    df: pd.DataFrame,
    reps: int,
    seed: int,
) -> pd.DataFrame:
    """
    One-way investor-cluster bootstrap.

    Sampling unit:
        investor_id

    All test events belonging to a sampled investor are carried together.

    This addresses within-investor dependence but does not yet account for
    clustering by startup. That will be handled in later robustness analysis.
    """

    working = df[
        [
            "investor_id",
            "pair_regime",
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
        working["investor_id"]
        .drop_duplicates()
        .to_numpy()
    )

    n_clusters = len(investors)

    arrays = {}

    for regime in PAIR_ORDER:

        sub = (
            working.loc[
                working["pair_regime"]
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
                sub[key]
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

    for bootstrap_index in range(reps):

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

        for regime in PAIR_ORDER:

            arrays_regime = (
                arrays[regime]
            )

            n = float(
                arrays_regime[
                    "n"
                ][sampled].sum()
            )

            if n <= 0:

                valid = False
                break

            result[
                f"{regime}__HR@10"
            ] = float(
                arrays_regime[
                    "hits10"
                ][sampled].sum()
                / n
            )

            result[
                f"{regime}__NDCG@10"
            ] = float(
                arrays_regime[
                    "ndcg_sum"
                ][sampled].sum()
                / n
            )

            result[
                f"{regime}__mean_rank"
            ] = float(
                arrays_regime[
                    "rank_sum"
                ][sampled].sum()
                / n
            )

        if not valid:
            continue

        result["delta_HR@10"] = (
            result[
                "New-to-investor__HR@10"
            ]
            - result[
                "Prior relationship__HR@10"
            ]
        )

        result["delta_NDCG@10"] = (
            result[
                "New-to-investor__NDCG@10"
            ]
            - result[
                "Prior relationship__NDCG@10"
            ]
        )

        result["delta_mean_rank"] = (
            result[
                "New-to-investor__mean_rank"
            ]
            - result[
                "Prior relationship__mean_rank"
            ]
        )

        rows.append(result)

    return pd.DataFrame(rows)


def percentile_ci(
    values: pd.Series,
) -> tuple[float, float]:

    clean = (
        values.dropna()
        .to_numpy(
            dtype=float
        )
    )

    require(
        len(clean) > 0,
        "Cannot calculate CI from empty bootstrap distribution.",
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
        "PHASE 7.2 — PAIR-NOVELTY DIAGNOSTIC V1"
    )
    print("=" * 110)

    print(
        "Scientific role:             "
        "POST-HOC UNADJUSTED SUBGROUP DIAGNOSTIC"
    )

    print("Model loaded:                NO")
    print("Checkpoint loaded:           NO")
    print("Raw logits loaded:           NO")
    print("Inference executed:          NO")
    print("Test rescored:               NO")
    print("Model selection performed:   NO")

    # =========================================================================
    # 7.2.1 — Integrity gate
    # =========================================================================

    print_section(
        "7.2.1 — ANALYSIS INTEGRITY GATE"
    )

    for path in (
        SOURCE,
        PHASE_7_0_CONTRACT,
        PHASE_7_1_MANIFEST,
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
        "Frozen test-analysis bundle SHA256 drift.",
    )

    with PHASE_7_0_CONTRACT.open(
        "r",
        encoding="utf-8",
    ) as f:

        phase_7_0 = json.load(f)

    with PHASE_7_1_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as f:

        phase_7_1 = json.load(f)

    require(
        phase_7_0["status"]
        == "PASS",
        "Phase 7.0 contract is not PASS.",
    )

    require(
        phase_7_1["status"]
        == "COMPLETE",
        "Phase 7.1 is not COMPLETE.",
    )

    print()
    print("Frozen source fingerprint:   PASS")
    print("Phase 7.0 contract:           PASS")
    print("Phase 7.1 prerequisite:       PASS")

    # =========================================================================
    # 7.2.2 — Load immutable final-test cases
    # =========================================================================

    print_section(
        "7.2.2 — LOAD FROZEN TEST CASES"
    )

    df = pd.read_parquet(
        SOURCE
    )

    require(
        len(df)
        == EXPECTED_ROWS,
        "Final-test row-count drift.",
    )

    require(
        df["interaction_id"].is_unique,
        "interaction_id is not unique.",
    )

    require(
        (
            df["new_to_investor_pair"]
            ==
            ~df[
                "prior_investor_startup_relationship"
            ]
        ).all(),
        (
            "Pair novelty and prior-relationship "
            "fields are not complements."
        ),
    )

    df["pair_regime"] = np.where(
        df[
            "new_to_investor_pair"
        ].astype(bool),
        "New-to-investor",
        "Prior relationship",
    )

    counts = (
        df["pair_regime"]
        .value_counts()
    )

    require(
        int(
            counts[
                "New-to-investor"
            ]
        )
        == EXPECTED_NEW_N,
        "New-to-investor count drift.",
    )

    require(
        int(
            counts[
                "Prior relationship"
            ]
        )
        == EXPECTED_PRIOR_N,
        "Prior-relationship count drift.",
    )

    print(
        f"New-to-investor:              "
        f"{EXPECTED_NEW_N:,} "
        f"({EXPECTED_NEW_N / len(df):.2%})"
    )

    print(
        f"Prior relationship:           "
        f"{EXPECTED_PRIOR_N:,} "
        f"({EXPECTED_PRIOR_N / len(df):.2%})"
    )

    # =========================================================================
    # 7.2.3 — Cold-start composition audit
    # =========================================================================

    print_section(
        "7.2.3 — COLD-START COMPOSITION BY PAIR REGIME"
    )

    df["cold_state"] = np.select(
        [
            (
                df["cold_start_investor"]
                & df["cold_start_startup"]
            ),
            (
                df["cold_start_investor"]
                & ~df["cold_start_startup"]
            ),
            (
                ~df["cold_start_investor"]
                & df["cold_start_startup"]
            ),
        ],
        [
            "Cold investor + cold startup",
            "Cold investor + warm startup",
            "Warm investor + cold startup",
        ],
        default=(
            "Warm investor + warm startup"
        ),
    )

    cold_order = [
        "Warm investor + warm startup",
        "Warm investor + cold startup",
        "Cold investor + warm startup",
        "Cold investor + cold startup",
    ]

    composition = (
        df.groupby(
            [
                "pair_regime",
                "cold_state",
            ],
            observed=False,
        )
        .size()
        .rename("n")
        .reset_index()
    )

    composition["pair_n"] = (
        composition[
            "pair_regime"
        ].map(
            df[
                "pair_regime"
            ].value_counts()
        )
    )

    composition["within_pair_share"] = (
        composition["n"]
        / composition["pair_n"]
    )

    print(
        composition[
            [
                "pair_regime",
                "cold_state",
                "n",
                "within_pair_share",
            ]
        ]
        .sort_values(
            [
                "pair_regime",
                "cold_state",
            ]
        )
        .to_string(
            index=False,
            formatters={
                "within_pair_share":
                    lambda x: f"{x:.2%}",
            },
        )
    )

    prior_has_cold = bool(
        (
            (
                df["pair_regime"]
                == "Prior relationship"
            )
            &
            (
                df["cold_start_investor"]
                | df["cold_start_startup"]
            )
        ).any()
    )

    print()
    print(
        "Any cold-start case among "
        "prior relationships:       "
        f"{prior_has_cold}"
    )

    if not prior_has_cold:

        print(
            "Observed structural zero:       "
            "prior relationship × any cold start"
        )

    # =========================================================================
    # 7.2.4 — Pair-regime metrics
    # =========================================================================

    print_section(
        "7.2.4 — PAIR-NOVELTY PERFORMANCE"
    )

    metric_rows = []

    for regime in PAIR_ORDER:

        subgroup = df.loc[
            df["pair_regime"]
            == regime
        ]

        metrics = calculate_metrics(
            subgroup
        )

        metrics[
            "pair_regime"
        ] = regime

        metrics[
            "test_share"
        ] = (
            len(subgroup)
            / len(df)
        )

        metric_rows.append(
            metrics
        )

    metrics_df = pd.DataFrame(
        metric_rows
    )

    column_order = [
        "pair_regime",
        "n",
        "test_share",
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

    metrics_df = metrics_df[
        column_order
    ]

    print(
        metrics_df[
            [
                "pair_regime",
                "n",
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
    # 7.2.5 — Raw effect-size contrasts
    # =========================================================================

    print_section(
        "7.2.5 — RAW NEW-TO-INVESTOR CONTRASTS"
    )

    metric_indexed = (
        metrics_df
        .set_index(
            "pair_regime"
        )
    )

    prior = metric_indexed.loc[
        "Prior relationship"
    ]

    new = metric_indexed.loc[
        "New-to-investor"
    ]

    raw_contrasts = {

        "orientation":
            "New-to-investor minus Prior relationship",

        "delta_HR@1":
            float(
                new["HR@1"]
                - prior["HR@1"]
            ),

        "delta_HR@5":
            float(
                new["HR@5"]
                - prior["HR@5"]
            ),

        "delta_HR@10":
            float(
                new["HR@10"]
                - prior["HR@10"]
            ),

        "delta_NDCG@10":
            float(
                new["NDCG@10"]
                - prior["NDCG@10"]
            ),

        "delta_mean_rank":
            float(
                new["mean_rank"]
                - prior["mean_rank"]
            ),

        "delta_median_rank":
            float(
                new["median_rank"]
                - prior["median_rank"]
            ),

        "delta_rank_gt_50_share":
            float(
                new["rank_gt_50_share"]
                - prior[
                    "rank_gt_50_share"
                ]
            ),
    }

    print(
        "Contrast orientation: "
        "New-to-investor - Prior relationship"
    )

    print()

    print(
        f"Δ HR@1:                       "
        f"{raw_contrasts['delta_HR@1']:+.6f} "
        f"({raw_contrasts['delta_HR@1']:+.2%})"
    )

    print(
        f"Δ HR@5:                       "
        f"{raw_contrasts['delta_HR@5']:+.6f} "
        f"({raw_contrasts['delta_HR@5']:+.2%})"
    )

    print(
        f"Δ HR@10:                      "
        f"{raw_contrasts['delta_HR@10']:+.6f} "
        f"({raw_contrasts['delta_HR@10']:+.2%})"
    )

    print(
        f"Δ NDCG@10:                    "
        f"{raw_contrasts['delta_NDCG@10']:+.6f}"
    )

    print(
        f"Δ mean rank:                  "
        f"{raw_contrasts['delta_mean_rank']:+.2f}"
    )

    print(
        f"Δ median rank:                "
        f"{raw_contrasts['delta_median_rank']:+.2f}"
    )

    print(
        f"Δ P(rank > 50):               "
        f"{raw_contrasts['delta_rank_gt_50_share']:+.2%}"
    )

    # =========================================================================
    # 7.2.6 — Rank-band decomposition
    # =========================================================================

    print_section(
        "7.2.6 — RANK-BAND DECOMPOSITION BY PAIR REGIME"
    )

    rank_labels = [
        "Rank 1",
        "Ranks 2–5",
        "Ranks 6–10",
        "Ranks 11–20",
        "Ranks 21–50",
        "Ranks 51–100",
    ]

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
        labels=rank_labels,
        include_lowest=True,
        ordered=True,
    )

    rank_bands = (
        df.groupby(
            [
                "pair_regime",
                "rank_band",
            ],
            observed=False,
        )
        .size()
        .rename("n")
        .reset_index()
    )

    pair_sizes = (
        df["pair_regime"]
        .value_counts()
    )

    rank_bands[
        "within_pair_share"
    ] = (
        rank_bands["n"]
        / rank_bands[
            "pair_regime"
        ].map(pair_sizes)
    )

    print(
        rank_bands.to_string(
            index=False,
            formatters={
                "within_pair_share":
                    lambda x: f"{x:.2%}",
            },
        )
    )

    # =========================================================================
    # 7.2.7 — Empirical cumulative rank curves
    # =========================================================================

    print_section(
        "7.2.7 — EMPIRICAL CUMULATIVE RANK CURVES"
    )

    cdf_rows = []

    for regime in PAIR_ORDER:

        ranks = (
            df.loc[
                df["pair_regime"]
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
                    "pair_regime":
                        regime,
                    "k": k,
                    "HR@k": float(
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

        for _, row in (
            subset.iterrows()
        ):

            print(
                f"  {row['pair_regime']:<20} "
                f"{row['HR@k']:.2%}"
            )

    # =========================================================================
    # 7.2.8 — Investor-cluster bootstrap
    # =========================================================================

    print_section(
        "7.2.8 — INVESTOR-CLUSTER BOOTSTRAP UNCERTAINTY"
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

    actual_lookup = {

        "Prior relationship__HR@10":
            float(
                prior["HR@10"]
            ),

        "New-to-investor__HR@10":
            float(
                new["HR@10"]
            ),

        "Prior relationship__NDCG@10":
            float(
                prior["NDCG@10"]
            ),

        "New-to-investor__NDCG@10":
            float(
                new["NDCG@10"]
            ),

        "Prior relationship__mean_rank":
            float(
                prior["mean_rank"]
            ),

        "New-to-investor__mean_rank":
            float(
                new["mean_rank"]
            ),

        "delta_HR@10":
            raw_contrasts[
                "delta_HR@10"
            ],

        "delta_NDCG@10":
            raw_contrasts[
                "delta_NDCG@10"
            ],

        "delta_mean_rank":
            raw_contrasts[
                "delta_mean_rank"
            ],
    }

    for metric_name in (
        "Prior relationship__HR@10",
        "New-to-investor__HR@10",
        "Prior relationship__NDCG@10",
        "New-to-investor__NDCG@10",
        "Prior relationship__mean_rank",
        "New-to-investor__mean_rank",
        "delta_HR@10",
        "delta_NDCG@10",
        "delta_mean_rank",
    ):

        low, high = percentile_ci(
            bootstrap_df[
                metric_name
            ]
        )

        bootstrap_summary_rows.append(
            {
                "metric":
                    metric_name,
                "estimate":
                    actual_lookup[
                        metric_name
                    ],
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

    bootstrap_summary = (
        pd.DataFrame(
            bootstrap_summary_rows
        )
    )

    print(
        bootstrap_summary[
            bootstrap_summary[
                "metric"
            ].isin(
                [
                    "delta_HR@10",
                    "delta_NDCG@10",
                    "delta_mean_rank",
                ]
            )
        ].to_string(
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
    # 7.2.9 — Interpretation safeguards
    # =========================================================================

    print_section(
        "7.2.9 — INTERPRETATION SAFEGUARDS"
    )

    new_any_cold = int(
        (
            (
                df["pair_regime"]
                == "New-to-investor"
            )
            &
            (
                df["cold_start_investor"]
                | df["cold_start_startup"]
            )
        ).sum()
    )

    new_warm_warm = int(
        (
            (
                df["pair_regime"]
                == "New-to-investor"
            )
            &
            ~df["cold_start_investor"]
            &
            ~df["cold_start_startup"]
        ).sum()
    )

    new_any_cold_share = (
        new_any_cold
        / EXPECTED_NEW_N
    )

    new_warm_warm_share = (
        new_warm_warm
        / EXPECTED_NEW_N
    )

    print(
        f"New-pair cases with any cold entity: "
        f"{new_any_cold:,} "
        f"({new_any_cold_share:.2%})"
    )

    print(
        f"New-pair warm/warm cases:           "
        f"{new_warm_warm:,} "
        f"({new_warm_warm_share:.2%})"
    )

    print()

    print(
        "RAW CONTRAST INTERPRETATION:"
    )

    print(
        "  New-to-investor performance differences "
        "are descriptive associations."
    )

    print(
        "  They combine pair novelty with different "
        "cold-start composition."
    )

    print(
        "  They must NOT be interpreted as the "
        "causal or isolated effect of novelty."
    )

    print()

    print(
        "NEXT IDENTIFICATION STEP:"
    )

    print(
        "  Compare new vs prior relationships "
        "within warm-investor/warm-startup cases."
    )

    # =========================================================================
    # 7.2.10 — Machine-readable insight artifact
    # =========================================================================

    descriptive_insights = {

        "phase":
            "7.2",

        "scientific_scope":
            (
                "Unadjusted association between "
                "pair novelty and ranking performance."
            ),

        "group_sizes": {
            "new_to_investor":
                EXPECTED_NEW_N,
            "prior_relationship":
                EXPECTED_PRIOR_N,
        },

        "raw_contrasts":
            raw_contrasts,

        "confounding_structure": {
            "prior_relationship_has_any_cold_start":
                prior_has_cold,
            "new_pair_any_cold_n":
                new_any_cold,
            "new_pair_any_cold_share":
                new_any_cold_share,
            "new_pair_warm_warm_n":
                new_warm_warm,
            "new_pair_warm_warm_share":
                new_warm_warm_share,
            "interpretation":
                (
                    "Cold-start status is unequally "
                    "distributed across pair regimes, "
                    "so the raw pair-novelty contrast "
                    "does not isolate novelty."
                ),
        },

        "uncertainty": {
            "method":
                (
                    "Percentile bootstrap with "
                    "investor_id as the resampling unit."
                ),
            "repetitions":
                BOOTSTRAP_REPS,
            "seed":
                BOOTSTRAP_SEED,
            "limitation":
                (
                    "This handles within-investor "
                    "dependence only. Startup clustering "
                    "and other dependence structures are "
                    "reserved for robustness analysis."
                ),
        },

        "interpretation_boundary": (
            "Descriptive and unadjusted statistical "
            "evidence only; no causal claim."
        ),
    }

    # =========================================================================
    # 7.2.11 — Write tables
    # =========================================================================

    print_section(
        "7.2.10 — WRITE ANALYSIS ARTIFACTS"
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
        / "phase_7_2_pair_regime_metrics_V1.csv"
    )

    composition_path = (
        OUT_DIR
        / "phase_7_2_pair_cold_start_composition_V1.csv"
    )

    rank_bands_path = (
        OUT_DIR
        / "phase_7_2_pair_rank_bands_V1.csv"
    )

    cdf_path = (
        OUT_DIR
        / "phase_7_2_pair_rank_cdf_V1.csv"
    )

    contrasts_path = (
        OUT_DIR
        / "phase_7_2_raw_contrasts_V1.json"
    )

    bootstrap_replicates_path = (
        OUT_DIR
        / "phase_7_2_investor_cluster_bootstrap_replicates_V1.csv"
    )

    bootstrap_summary_path = (
        OUT_DIR
        / "phase_7_2_investor_cluster_bootstrap_summary_V1.csv"
    )

    insights_path = (
        OUT_DIR
        / "phase_7_2_descriptive_insights_V1.json"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    composition.to_csv(
        composition_path,
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

    json_dump(
        raw_contrasts,
        contrasts_path,
    )

    bootstrap_df.to_csv(
        bootstrap_replicates_path,
        index=False,
    )

    bootstrap_summary.to_csv(
        bootstrap_summary_path,
        index=False,
    )

    json_dump(
        descriptive_insights,
        insights_path,
    )

    # =========================================================================
    # 7.2.12 — Figure 1: Top-k hit rates
    # =========================================================================

    print_section(
        "7.2.11 — GENERATE PRESENTATION-READY FIGURES"
    )

    figure_paths = []

    topk_metrics = [
        "HR@1",
        "HR@5",
        "HR@10",
    ]

    x = np.arange(
        len(topk_metrics)
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(10, 6.5)
    )

    for i, regime in enumerate(
        PAIR_ORDER
    ):

        row = metric_indexed.loc[
            regime
        ]

        values = np.array(
            [
                row[m]
                for m in topk_metrics
            ]
        ) * 100.0

        bars = ax.bar(
            x
            + (
                i - 0.5
            ) * width,
            values,
            width=width,
            label=regime,
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
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(
        x,
        topk_metrics,
    )

    ax.set_ylabel(
        "Hit rate (%)"
    )

    ax.set_title(
        "ITRS retrieval performance by investor–startup pair novelty"
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
                "phase_7_2_fig1_"
                "topk_by_pair_novelty_V1"
            ),
        )
    )

    # =========================================================================
    # Figure 2: CDF / HR@k curve
    # =========================================================================

    fig, ax = plt.subplots(
        figsize=(11, 6.5)
    )

    for regime in PAIR_ORDER:

        subset = cdf_df.loc[
            cdf_df["pair_regime"]
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
        "Positive-startup rank distribution by pair novelty"
    )

    ax.legend()

    ax.grid(
        alpha=0.20,
    )

    fig.tight_layout()

    figure_paths.extend(
        save_figure(
            fig,
            (
                "phase_7_2_fig2_"
                "rank_cdf_by_pair_novelty_V1"
            ),
        )
    )

    # =========================================================================
    # Figure 3: 100% stacked rank-band composition
    # =========================================================================

    rank_band_pivot = (
        rank_bands.pivot(
            index="pair_regime",
            columns="rank_band",
            values="within_pair_share",
        )
        .reindex(
            PAIR_ORDER
        )
        .reindex(
            columns=rank_labels
        )
        .fillna(0.0)
        * 100.0
    )

    fig, ax = plt.subplots(
        figsize=(11, 6.5)
    )

    bottom = np.zeros(
        len(rank_band_pivot)
    )

    x = np.arange(
        len(rank_band_pivot)
    )

    for band in rank_labels:

        values = (
            rank_band_pivot[
                band
            ].to_numpy()
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
        rank_band_pivot.index,
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.set_ylabel(
        "Share of subgroup (%)"
    )

    ax.set_title(
        "Rank-error composition by pair novelty"
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
                "phase_7_2_fig3_"
                "rank_band_composition_by_pair_V1"
            ),
        )
    )

    # =========================================================================
    # Figure 4: cold-start composition of pair regimes
    # =========================================================================

    composition_complete = (
        composition.pivot(
            index="pair_regime",
            columns="cold_state",
            values="within_pair_share",
        )
        .reindex(
            PAIR_ORDER
        )
        .reindex(
            columns=cold_order
        )
        .fillna(0.0)
        * 100.0
    )

    fig, ax = plt.subplots(
        figsize=(11, 6.5)
    )

    bottom = np.zeros(
        len(composition_complete)
    )

    x = np.arange(
        len(composition_complete)
    )

    for state in cold_order:

        values = (
            composition_complete[
                state
            ].to_numpy()
        )

        ax.bar(
            x,
            values,
            bottom=bottom,
            label=state,
        )

        bottom += values

    ax.set_xticks(
        x,
        composition_complete.index,
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.set_ylabel(
        "Share of pair-regime cases (%)"
    )

    ax.set_title(
        "Cold-start composition differs strongly across pair regimes"
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
                "phase_7_2_fig4_"
                "cold_start_composition_by_pair_V1"
            ),
        )
    )

    # =========================================================================
    # 7.2.12 — Manifest
    # =========================================================================

    manifest_path = (
        OUT_DIR
        / "phase_7_2_analysis_manifest_V1.json"
    )

    manifest = {

        "phase":
            "7.2",

        "schema_version":
            "ITRS_PHASE7_2_PAIR_NOVELTY_DIAGNOSTIC_V1",

        "status":
            "COMPLETE",

        "scientific_role":
            (
                "Post-hoc unadjusted comparison "
                "of new-to-investor versus prior "
                "investor-startup relationships."
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

        "prerequisites": {
            "phase_7_0":
                str(
                    PHASE_7_0_CONTRACT.relative_to(
                        REPO_ROOT
                    )
                ),
            "phase_7_1":
                str(
                    PHASE_7_1_MANIFEST.relative_to(
                        REPO_ROOT
                    )
                ),
        },

        "uncertainty_method": {
            "primary_here":
                (
                    "One-way investor-cluster "
                    "percentile bootstrap"
                ),
            "bootstrap_repetitions":
                BOOTSTRAP_REPS,
            "bootstrap_seed":
                BOOTSTRAP_SEED,
            "limitation":
                (
                    "Does not yet address startup "
                    "clustering or adjusted confounding."
                ),
        },

        "interpretation_boundary":
            (
                "Raw pair-regime contrasts combine "
                "novelty and unequal cold-start "
                "composition. No causal or isolated "
                "novelty-effect claim is permitted."
            ),

        "tables": [
            str(
                metrics_path.relative_to(
                    REPO_ROOT
                )
            ),
            str(
                composition_path.relative_to(
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
    }

    json_dump(
        manifest,
        manifest_path,
    )

    # =========================================================================
    # Hash all derived artifacts
    # =========================================================================

    artifact_paths = [
        metrics_path,
        composition_path,
        rank_bands_path,
        cdf_path,
        contrasts_path,
        bootstrap_replicates_path,
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
        / "phase_7_2_derived_artifact_sha256_V1.json"
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
        "PHASE 7.2 V1 RESULT"
    )

    print(
        "Frozen source binding:             PASS"
    )

    print(
        "Pair-regime semantic binding:      PASS"
    )

    print(
        "Cold-start composition audit:      COMPLETE"
    )

    print(
        "Pair-regime performance metrics:   COMPLETE"
    )

    print(
        "Raw effect-size contrasts:         COMPLETE"
    )

    print(
        "Rank-distribution comparison:      COMPLETE"
    )

    print(
        "Investor-cluster bootstrap:        COMPLETE"
    )

    print(
        "Presentation-ready figures:        COMPLETE"
    )

    print(
        "Provenance manifest:               COMPLETE"
    )

    print()
    print(
        "Model inference executed:          NO"
    )

    print(
        "Final test rescored:               NO"
    )

    print(
        "Final test modified:               NO"
    )

    print()
    print(
        "IMPORTANT: RAW NOVELTY CONTRAST "
        "IS CONFOUNDED BY COLD-START COMPOSITION."
    )

    print()
    print(
        "PHASE 7.2 PAIR-NOVELTY "
        "DIAGNOSTIC STATUS: COMPLETE"
    )


if __name__ == "__main__":
    main()