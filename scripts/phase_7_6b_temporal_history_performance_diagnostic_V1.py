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
# Phase 7.6B — Temporal History Performance Diagnostic V1
#
# Core question
# -------------
# Within NEW-TO-INVESTOR, WARM-INVESTOR, WARM-STARTUP cases, is broader
# pre-T60 investor temporal coverage associated with different ranking
# performance?
#
# Primary comparison (pre-registered in Phase 7.6A):
#
#   investor T0 + T1-T59
#       minus
#   investor T1-T59 only
#
# Controlled comparison:
#
#   repeat the same investor-history comparison while holding the startup
#   history class fixed at T1-T59 only.
#
# Why investor-side only for formal inference?
# --------------------------------------------
# Phase 7.6A showed:
#
#   new warm/warm investors:
#       T1-T59 only      = 5,170
#       T0 + T1-T59      = 1,710
#       T0 only          = 9
#
#   new warm/warm startups:
#       T1-T59 only      = 6,775
#       T0 + T1-T59      = 90
#       T0 only          = 24
#
# With the pre-specified minimum group N=100, the startup temporal-history
# contrast is descriptive only.
#
# IMPORTANT SEMANTICS
# -------------------
# We use literal temporal labels only.
# We do NOT rename T0/T1-T59 classes as "recent", "long", "deep", etc.
#
# Interpretation boundary
# -----------------------
# Observational association only. This phase does not establish that temporal
# coverage causes better performance or identify a mechanism.
#
# Uncertainty
# -----------
# Percentile investor-cluster bootstrap.
# Startup / multiway dependence is deferred to Phase 7.9.
#
# Prohibited
# ----------
# - model loading
# - checkpoint loading
# - raw-logit loading
# - inference
# - test rescoring
# - negative resampling
# - candidate modification
# - model selection
# =============================================================================


REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    REPO_ROOT
    / "data/experimental/phase_6/full_training/100pct/final_test/"
    / "analysis_ready_test_cases.parquet"
)

PHASE_7_6A_MANIFEST = (
    REPO_ROOT
    / "data/experimental/phase_7/"
    / "phase_7_6a_temporal_history_composition/"
    / "phase_7_6a_analysis_manifest_V1.json"
)

PHASE_7_6A_PREREG = (
    REPO_ROOT
    / "data/experimental/phase_7/"
    / "phase_7_6a_temporal_history_composition/"
    / "phase_7_6a_preregistered_7_6b_comparisons_V1.json"
)

OUT_DIR = (
    REPO_ROOT
    / "data/experimental/phase_7/"
    / "phase_7_6b_temporal_history_performance"
)

FIG_DIR = OUT_DIR / "figures"

EXPECTED_SOURCE_SHA256 = (
    "d70f21bff0006e094c5d567d307c0664"
    "b41a11e7250a0e69aaec610e1810132d"
)

EXPECTED_ROWS = 20_264

EXPECTED_NEW_WW = 6_889

EXPECTED_NEW_WW_INVESTOR_COUNTS = {
    "t0_only": 9,
    "t1_t59_only": 5_170,
    "t0_and_t1_t59": 1_710,
}

EXPECTED_NEW_WW_STARTUP_COUNTS = {
    "t0_only": 24,
    "t1_t59_only": 6_775,
    "t0_and_t1_t59": 90,
}

EXPECTED_CONTROLLED_COUNTS = {
    "investor_t1_t59_only__startup_t1_t59_only": 5_101,
    "investor_t0_and_t1_t59__startup_t1_t59_only": 1_665,
}

HISTORY_ORDER = [
    "cold",
    "t0_only",
    "t1_t59_only",
    "t0_and_t1_t59",
]

MIN_FORMAL_GROUP_N = 100

BOOTSTRAP_REPS = 2_000
BOOTSTRAP_SEED = 7_106_002

SOURCE_COLUMNS = [
    "interaction_id",
    "investor_id",
    "startup_id",
    "positive_rank",
    "NDCG@10",
    "new_to_investor_pair",
    "cold_start_investor",
    "cold_start_startup",
    "investor_seen_in_t0",
    "investor_seen_in_t1_t59",
    "startup_seen_in_t0",
    "startup_seen_in_t1_t59",
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


def build_history_class(
    seen_t0: pd.Series,
    seen_t1_t59: pd.Series,
) -> pd.Series:
    t0 = seen_t0.astype(bool)
    t1_t59 = seen_t1_t59.astype(bool)

    return pd.Series(
        np.select(
            [
                (~t0) & (~t1_t59),
                t0 & (~t1_t59),
                (~t0) & t1_t59,
                t0 & t1_t59,
            ],
            HISTORY_ORDER,
            default="INVALID",
        ),
        index=seen_t0.index,
    )


def calculate_metrics(
    group: pd.DataFrame,
) -> dict:
    rank = (
        group["positive_rank"]
        .astype(int)
    )

    return {
        "n": int(len(group)),
        "Hits@1": int((rank <= 1).sum()),
        "Hits@5": int((rank <= 5).sum()),
        "Hits@10": int((rank <= 10).sum()),
        "HR@1": float((rank <= 1).mean()),
        "HR@5": float((rank <= 5).mean()),
        "HR@10": float((rank <= 10).mean()),
        "NDCG@10": float(group["NDCG@10"].mean()),
        "mean_rank": float(rank.mean()),
        "median_rank": float(rank.median()),
        "P25_rank": float(rank.quantile(0.25)),
        "P75_rank": float(rank.quantile(0.75)),
        "P90_rank": float(rank.quantile(0.90)),
        "P95_rank": float(rank.quantile(0.95)),
        "rank_gt_10_n": int((rank > 10).sum()),
        "rank_gt_10_share": float((rank > 10).mean()),
        "rank_gt_50_n": int((rank > 50).sum()),
        "rank_gt_50_share": float((rank > 50).mean()),
    }


def metric_table(
    df: pd.DataFrame,
    class_col: str,
    order: list[str],
) -> pd.DataFrame:
    rows = []

    for history_class in order:
        subgroup = df.loc[
            df[class_col] == history_class
        ]

        if len(subgroup) == 0:
            continue

        row = calculate_metrics(
            subgroup
        )

        row["history_class"] = (
            history_class
        )

        row[
            "formal_inference_eligible"
        ] = bool(
            len(subgroup)
            >= MIN_FORMAL_GROUP_N
        )

        rows.append(row)

    return pd.DataFrame(rows)


def build_contrast(
    df: pd.DataFrame,
    mask_a: pd.Series,
    mask_b: pd.Series,
    comparison_key: str,
    comparison_label: str,
    group_a_label: str,
    group_b_label: str,
    priority: str,
) -> dict:
    a = df.loc[mask_a]
    b = df.loc[mask_b]

    require(
        len(a) > 0 and len(b) > 0,
        f"Empty comparison group in {comparison_key}.",
    )

    ma = calculate_metrics(a)
    mb = calculate_metrics(b)

    formal = bool(
        min(
            ma["n"],
            mb["n"],
        )
        >= MIN_FORMAL_GROUP_N
    )

    return {
        "comparison_key":
            comparison_key,

        "comparison_label":
            comparison_label,

        "priority":
            priority,

        "group_a_label":
            group_a_label,

        "group_b_label":
            group_b_label,

        "group_a_n":
            ma["n"],

        "group_b_n":
            mb["n"],

        "formal_bootstrap_eligible":
            formal,

        "minimum_group_n_threshold":
            MIN_FORMAL_GROUP_N,

        "group_a_HR@1":
            ma["HR@1"],

        "group_b_HR@1":
            mb["HR@1"],

        "delta_HR@1":
            ma["HR@1"]
            - mb["HR@1"],

        "group_a_HR@5":
            ma["HR@5"],

        "group_b_HR@5":
            mb["HR@5"],

        "delta_HR@5":
            ma["HR@5"]
            - mb["HR@5"],

        "group_a_HR@10":
            ma["HR@10"],

        "group_b_HR@10":
            mb["HR@10"],

        "delta_HR@10":
            ma["HR@10"]
            - mb["HR@10"],

        "group_a_NDCG@10":
            ma["NDCG@10"],

        "group_b_NDCG@10":
            mb["NDCG@10"],

        "delta_NDCG@10":
            ma["NDCG@10"]
            - mb["NDCG@10"],

        "group_a_mean_rank":
            ma["mean_rank"],

        "group_b_mean_rank":
            mb["mean_rank"],

        "delta_mean_rank":
            ma["mean_rank"]
            - mb["mean_rank"],

        "group_a_median_rank":
            ma["median_rank"],

        "group_b_median_rank":
            mb["median_rank"],

        "delta_median_rank":
            ma["median_rank"]
            - mb["median_rank"],

        "group_a_rank_gt_50_share":
            ma["rank_gt_50_share"],

        "group_b_rank_gt_50_share":
            mb["rank_gt_50_share"],

        "delta_rank_gt_50_share":
            ma["rank_gt_50_share"]
            - mb["rank_gt_50_share"],
    }


def cluster_bootstrap_comparison(
    df: pd.DataFrame,
    mask_a: pd.Series,
    mask_b: pd.Series,
    reps: int,
    seed: int,
) -> pd.DataFrame:
    working = df.loc[
        mask_a | mask_b,
        [
            "investor_id",
            "positive_rank",
            "NDCG@10",
        ],
    ].copy()

    working[
        "analysis_group"
    ] = np.where(
        mask_a.loc[
            working.index
        ],
        "A",
        "B",
    )

    working["hit10"] = (
        working[
            "positive_rank"
        ]
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

    arrays = {}

    for key in (
        "A",
        "B",
    ):
        grouped = (
            working.loc[
                working[
                    "analysis_group"
                ]
                == key
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

        arrays[key] = {
            col:
                grouped[col]
                .to_numpy(
                    dtype=float
                )
            for col in (
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

    for bootstrap_index in range(
        reps
    ):
        sampled = rng.integers(
            0,
            len(investors),
            size=len(investors),
        )

        result = {
            "bootstrap_index":
                bootstrap_index,
        }

        valid = True

        for key in (
            "A",
            "B",
        ):
            arr = arrays[key]

            n = float(
                arr["n"][
                    sampled
                ].sum()
            )

            if n <= 0:
                valid = False
                break

            result[
                f"{key}__HR@10"
            ] = float(
                arr[
                    "hits10"
                ][
                    sampled
                ].sum()
                / n
            )

            result[
                f"{key}__NDCG@10"
            ] = float(
                arr[
                    "ndcg_sum"
                ][
                    sampled
                ].sum()
                / n
            )

            result[
                f"{key}__mean_rank"
            ] = float(
                arr[
                    "rank_sum"
                ][
                    sampled
                ].sum()
                / n
            )

        if not valid:
            continue

        for metric in (
            "HR@10",
            "NDCG@10",
            "mean_rank",
        ):
            result[
                f"delta_{metric}"
            ] = (
                result[
                    f"A__{metric}"
                ]
                - result[
                    f"B__{metric}"
                ]
            )

        rows.append(
            result
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
        (
            "Cannot compute CI from "
            "empty bootstrap distribution."
        ),
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
        "PHASE 7.6B — TEMPORAL HISTORY "
        "PERFORMANCE DIAGNOSTIC V1"
    )
    print("=" * 110)

    print(
        "Scientific role:             "
        "PRE-REGISTERED POST-HOC TEMPORAL-HISTORY DIAGNOSTIC"
    )
    print("Model loaded:                NO")
    print("Checkpoint loaded:           NO")
    print("Raw logits loaded:           NO")
    print("Inference executed:          NO")
    print("Test rescored:               NO")
    print("Model selection performed:   NO")

    # =========================================================================
    # 7.6B.1 — Integrity / preregistration gate
    # =========================================================================

    print_section(
        "7.6B.1 — INTEGRITY AND PRE-REGISTRATION GATE"
    )

    for path in (
        SOURCE,
        PHASE_7_6A_MANIFEST,
        PHASE_7_6A_PREREG,
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

    with PHASE_7_6A_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as f:
        phase_7_6a = json.load(f)

    with PHASE_7_6A_PREREG.open(
        "r",
        encoding="utf-8",
    ) as f:
        prereg = json.load(f)

    require(
        phase_7_6a[
            "status"
        ]
        == "COMPLETE",
        "Phase 7.6A is not COMPLETE.",
    )

    require(
        phase_7_6a[
            "outcome_blindness"
        ][
            "performance_metrics_examined"
        ]
        is False,
        (
            "Phase 7.6A does not confirm "
            "outcome-blind preregistration."
        ),
    )

    required_prereg = {
        "new_ww_investor_temporal_coverage",
        "new_ww_startup_temporal_coverage",
        (
            "new_ww_investor_temporal_coverage_"
            "control_startup_class"
        ),
        (
            "new_ww_startup_temporal_coverage_"
            "control_investor_class"
        ),
        "new_ww_joint_temporal_support",
        "pair_history_descriptive_only",
        "minimum_group_n_for_formal_bootstrap",
    }

    require(
        required_prereg
        <= set(
            prereg.keys()
        ),
        (
            "Phase 7.6A preregistration "
            "is missing required entries."
        ),
    )

    require(
        int(
            prereg[
                "minimum_group_n_for_formal_bootstrap"
            ]
        )
        == MIN_FORMAL_GROUP_N,
        (
            "Minimum-N rule differs from "
            "Phase 7.6A preregistration."
        ),
    )

    print()
    print("Frozen source fingerprint:   PASS")
    print("Phase 7.6A prerequisite:      PASS")
    print("Outcome-blind preregistration:PASS")
    print(
        f"Minimum formal group N:       "
        f"{MIN_FORMAL_GROUP_N}"
    )

    # =========================================================================
    # 7.6B.2 — Load frozen outcomes / construct classes
    # =========================================================================

    print_section(
        "7.6B.2 — LOAD FROZEN OUTCOMES AND TEMPORAL-HISTORY CLASSES"
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
        "investor_history_class"
    ] = build_history_class(
        df[
            "investor_seen_in_t0"
        ],
        df[
            "investor_seen_in_t1_t59"
        ],
    )

    df[
        "startup_history_class"
    ] = build_history_class(
        df[
            "startup_seen_in_t0"
        ],
        df[
            "startup_seen_in_t1_t59"
        ],
    )

    new_ww_mask = (
        df[
            "new_to_investor_pair"
        ].astype(bool)
        &
        ~df[
            "cold_start_investor"
        ].astype(bool)
        &
        ~df[
            "cold_start_startup"
        ].astype(bool)
    )

    new_ww = df.loc[
        new_ww_mask
    ].copy()

    require(
        len(new_ww)
        == EXPECTED_NEW_WW,
        "new warm/warm count drift.",
    )

    print(
        f"New warm/warm cases:          "
        f"{len(new_ww):,}"
    )
    print(
        "History-class construction:  PASS"
    )

    # =========================================================================
    # 7.6B.3 — Descriptive class performance
    # =========================================================================

    print_section(
        "7.6B.3 — NEW WARM/WARM PERFORMANCE BY HISTORY CLASS"
    )

    investor_metrics = metric_table(
        df=new_ww,
        class_col=(
            "investor_history_class"
        ),
        order=HISTORY_ORDER,
    )

    startup_metrics = metric_table(
        df=new_ww,
        class_col=(
            "startup_history_class"
        ),
        order=HISTORY_ORDER,
    )

    for key, expected in (
        EXPECTED_NEW_WW_INVESTOR_COUNTS.items()
    ):
        observed = int(
            investor_metrics.loc[
                investor_metrics[
                    "history_class"
                ]
                == key,
                "n",
            ].iloc[0]
        )

        require(
            observed == expected,
            (
                f"Investor new_ww history count "
                f"drift for {key}."
            ),
        )

    for key, expected in (
        EXPECTED_NEW_WW_STARTUP_COUNTS.items()
    ):
        observed = int(
            startup_metrics.loc[
                startup_metrics[
                    "history_class"
                ]
                == key,
                "n",
            ].iloc[0]
        )

        require(
            observed == expected,
            (
                f"Startup new_ww history count "
                f"drift for {key}."
            ),
        )

    print(
        "INVESTOR HISTORY"
    )

    print(
        investor_metrics[
            [
                "history_class",
                "n",
                "formal_inference_eligible",
                "HR@1",
                "HR@5",
                "HR@10",
                "NDCG@10",
                "mean_rank",
                "median_rank",
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
                "rank_gt_50_share":
                    lambda x: f"{x:.2%}",
            },
        )
    )

    print()
    print(
        "STARTUP HISTORY"
    )

    print(
        startup_metrics[
            [
                "history_class",
                "n",
                "formal_inference_eligible",
                "HR@1",
                "HR@5",
                "HR@10",
                "NDCG@10",
                "mean_rank",
                "median_rank",
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
                "rank_gt_50_share":
                    lambda x: f"{x:.2%}",
            },
        )
    )

    # =========================================================================
    # 7.6B.4 — Primary investor temporal-coverage contrast
    # =========================================================================

    print_section(
        "7.6B.4 — PRIMARY INVESTOR TEMPORAL-COVERAGE CONTRAST"
    )

    inv_both_mask = (
        new_ww_mask
        &
        df[
            "investor_history_class"
        ].eq(
            "t0_and_t1_t59"
        )
    )

    inv_t1_only_mask = (
        new_ww_mask
        &
        df[
            "investor_history_class"
        ].eq(
            "t1_t59_only"
        )
    )

    primary = build_contrast(
        df=df,
        mask_a=inv_both_mask,
        mask_b=inv_t1_only_mask,
        comparison_key=(
            "new_ww_investor_t0_and_t1_t59_"
            "vs_t1_t59_only"
        ),
        comparison_label=(
            "New warm/warm: investor T0+T1-T59 "
            "- investor T1-T59 only"
        ),
        group_a_label=(
            "Investor T0 + T1-T59"
        ),
        group_b_label=(
            "Investor T1-T59 only"
        ),
        priority="PRIMARY",
    )

    require(
        primary[
            "group_a_n"
        ]
        == 1_710,
        "Primary group A count drift.",
    )

    require(
        primary[
            "group_b_n"
        ]
        == 5_170,
        "Primary group B count drift.",
    )

    require(
        primary[
            "formal_bootstrap_eligible"
        ],
        "Primary contrast unexpectedly ineligible.",
    )

    print(
        f"Investor T0+T1-T59 N:        "
        f"{primary['group_a_n']:,}"
    )

    print(
        f"Investor T1-T59 only N:      "
        f"{primary['group_b_n']:,}"
    )

    print()
    print(
        f"T0+T1-T59 HR@10:             "
        f"{primary['group_a_HR@10']:.2%}"
    )

    print(
        f"T1-T59-only HR@10:           "
        f"{primary['group_b_HR@10']:.2%}"
    )

    print(
        f"ΔHR@10:                       "
        f"{primary['delta_HR@10']:+.2%}"
    )

    print()
    print(
        f"ΔNDCG@10:                     "
        f"{primary['delta_NDCG@10']:+.4f}"
    )

    print(
        f"Δmean rank:                   "
        f"{primary['delta_mean_rank']:+.2f}"
    )

    print(
        f"Δrank>50 share:               "
        f"{primary['delta_rank_gt_50_share']:+.2%}"
    )

    # =========================================================================
    # 7.6B.5 — Controlled comparison: startup class fixed
    # =========================================================================

    print_section(
        "7.6B.5 — CONTROLLED INVESTOR CONTRAST WITH STARTUP HISTORY FIXED"
    )

    startup_t1_only = (
        df[
            "startup_history_class"
        ].eq(
            "t1_t59_only"
        )
    )

    controlled_a = (
        inv_both_mask
        &
        startup_t1_only
    )

    controlled_b = (
        inv_t1_only_mask
        &
        startup_t1_only
    )

    controlled = build_contrast(
        df=df,
        mask_a=controlled_a,
        mask_b=controlled_b,
        comparison_key=(
            "new_ww_startup_t1_t59_only__"
            "investor_t0_and_t1_t59_vs_t1_t59_only"
        ),
        comparison_label=(
            "New warm/warm with startup T1-T59 only: "
            "investor T0+T1-T59 - investor T1-T59 only"
        ),
        group_a_label=(
            "Investor T0 + T1-T59"
        ),
        group_b_label=(
            "Investor T1-T59 only"
        ),
        priority="PRIMARY_CONTROLLED",
    )

    require(
        controlled[
            "group_a_n"
        ]
        == EXPECTED_CONTROLLED_COUNTS[
            (
                "investor_t0_and_t1_t59__"
                "startup_t1_t59_only"
            )
        ],
        "Controlled group A count drift.",
    )

    require(
        controlled[
            "group_b_n"
        ]
        == EXPECTED_CONTROLLED_COUNTS[
            (
                "investor_t1_t59_only__"
                "startup_t1_t59_only"
            )
        ],
        "Controlled group B count drift.",
    )

    require(
        controlled[
            "formal_bootstrap_eligible"
        ],
        (
            "Controlled contrast "
            "unexpectedly ineligible."
        ),
    )

    print(
        f"Investor T0+T1-T59 N:        "
        f"{controlled['group_a_n']:,}"
    )

    print(
        f"Investor T1-T59 only N:      "
        f"{controlled['group_b_n']:,}"
    )

    print(
        "Startup history held fixed:   "
        "T1-T59 only"
    )

    print()
    print(
        f"ΔHR@10:                       "
        f"{controlled['delta_HR@10']:+.2%}"
    )

    print(
        f"ΔNDCG@10:                     "
        f"{controlled['delta_NDCG@10']:+.4f}"
    )

    print(
        f"Δmean rank:                   "
        f"{controlled['delta_mean_rank']:+.2f}"
    )

    # =========================================================================
    # 7.6B.6 — Startup temporal-history contrast: descriptive only
    # =========================================================================

    print_section(
        "7.6B.6 — STARTUP TEMPORAL-HISTORY CONTRAST: DESCRIPTIVE ONLY"
    )

    startup_both_mask = (
        new_ww_mask
        &
        df[
            "startup_history_class"
        ].eq(
            "t0_and_t1_t59"
        )
    )

    startup_t1_mask = (
        new_ww_mask
        &
        df[
            "startup_history_class"
        ].eq(
            "t1_t59_only"
        )
    )

    startup_contrast = build_contrast(
        df=df,
        mask_a=startup_both_mask,
        mask_b=startup_t1_mask,
        comparison_key=(
            "new_ww_startup_t0_and_t1_t59_"
            "vs_t1_t59_only"
        ),
        comparison_label=(
            "New warm/warm: startup T0+T1-T59 "
            "- startup T1-T59 only"
        ),
        group_a_label=(
            "Startup T0 + T1-T59"
        ),
        group_b_label=(
            "Startup T1-T59 only"
        ),
        priority="DESCRIPTIVE_SMALL_N",
    )

    require(
        startup_contrast[
            "group_a_n"
        ]
        == 90,
        "Startup both-window count drift.",
    )

    require(
        not startup_contrast[
            "formal_bootstrap_eligible"
        ],
        (
            "Startup temporal contrast should be "
            "ineligible under minimum-N rule."
        ),
    )

    print(
        f"Startup T0+T1-T59 N:         "
        f"{startup_contrast['group_a_n']:,}"
    )

    print(
        f"Startup T1-T59 only N:       "
        f"{startup_contrast['group_b_n']:,}"
    )

    print(
        f"Formal bootstrap eligible:    "
        f"{startup_contrast['formal_bootstrap_eligible']}"
    )

    print()
    print(
        f"Descriptive ΔHR@10:           "
        f"{startup_contrast['delta_HR@10']:+.2%}"
    )

    print(
        f"Descriptive ΔNDCG@10:         "
        f"{startup_contrast['delta_NDCG@10']:+.4f}"
    )

    print(
        f"Descriptive Δmean rank:       "
        f"{startup_contrast['delta_mean_rank']:+.2f}"
    )

    # =========================================================================
    # 7.6B.7 — Joint temporal-support performance matrix
    # =========================================================================

    print_section(
        "7.6B.7 — NEW WARM/WARM JOINT TEMPORAL-SUPPORT PERFORMANCE"
    )

    joint_rows = []

    for investor_class in (
        "t0_only",
        "t1_t59_only",
        "t0_and_t1_t59",
    ):
        for startup_class in (
            "t0_only",
            "t1_t59_only",
            "t0_and_t1_t59",
        ):
            subgroup = new_ww.loc[
                (
                    new_ww[
                        "investor_history_class"
                    ]
                    == investor_class
                )
                &
                (
                    new_ww[
                        "startup_history_class"
                    ]
                    == startup_class
                )
            ]

            if len(subgroup) == 0:
                continue

            metrics = calculate_metrics(
                subgroup
            )

            joint_rows.append(
                {
                    "investor_history_class":
                        investor_class,

                    "startup_history_class":
                        startup_class,

                    "n":
                        metrics["n"],

                    "formal_cell_n_ge_100":
                        bool(
                            metrics["n"]
                            >= MIN_FORMAL_GROUP_N
                        ),

                    "HR@10":
                        metrics["HR@10"],

                    "NDCG@10":
                        metrics["NDCG@10"],

                    "mean_rank":
                        metrics["mean_rank"],

                    "median_rank":
                        metrics["median_rank"],

                    "rank_gt_50_share":
                        metrics[
                            "rank_gt_50_share"
                        ],
                }
            )

    joint_metrics = pd.DataFrame(
        joint_rows
    )

    print(
        joint_metrics.to_string(
            index=False,
            formatters={
                "HR@10":
                    lambda x: f"{x:.4f}",
                "NDCG@10":
                    lambda x: f"{x:.4f}",
                "mean_rank":
                    lambda x: f"{x:.2f}",
                "median_rank":
                    lambda x: f"{x:.1f}",
                "rank_gt_50_share":
                    lambda x: f"{x:.2%}",
            },
        )
    )

    # =========================================================================
    # 7.6B.8 — Investor-cluster bootstrap
    # =========================================================================

    print_section(
        "7.6B.8 — INVESTOR-CLUSTER BOOTSTRAP"
    )

    contrasts = pd.DataFrame(
        [
            primary,
            controlled,
            startup_contrast,
        ]
    )

    mask_lookup = {
        primary[
            "comparison_key"
        ]:
            (
                inv_both_mask,
                inv_t1_only_mask,
            ),

        controlled[
            "comparison_key"
        ]:
            (
                controlled_a,
                controlled_b,
            ),
    }

    bootstrap_frames = []
    bootstrap_summary_rows = []

    formal = contrasts.loc[
        contrasts[
            "formal_bootstrap_eligible"
        ]
    ].reset_index(
        drop=True
    )

    print(
        f"Bootstrap replicates:         "
        f"{BOOTSTRAP_REPS:,}"
    )

    print(
        f"Base seed:                    "
        f"{BOOTSTRAP_SEED}"
    )

    for i, row in (
        formal.iterrows()
    ):
        key = row[
            "comparison_key"
        ]

        require(
            key in mask_lookup,
            (
                f"No mask registered for "
                f"formal contrast {key}."
            ),
        )

        seed = (
            BOOTSTRAP_SEED
            + i
            + 1
        )

        mask_a, mask_b = (
            mask_lookup[
                key
            ]
        )

        reps = (
            cluster_bootstrap_comparison(
                df=df,
                mask_a=mask_a,
                mask_b=mask_b,
                reps=BOOTSTRAP_REPS,
                seed=seed,
            )
        )

        require(
            len(reps)
            == BOOTSTRAP_REPS,
            (
                f"Unexpected valid bootstrap "
                f"replicate count for {key}."
            ),
        )

        reps.insert(
            0,
            "comparison_key",
            key,
        )

        bootstrap_frames.append(
            reps
        )

        for metric, (
            bootstrap_col,
            estimate_col,
        ) in {
            "HR@10":
                (
                    "delta_HR@10",
                    "delta_HR@10",
                ),

            "NDCG@10":
                (
                    "delta_NDCG@10",
                    "delta_NDCG@10",
                ),

            "mean_rank":
                (
                    "delta_mean_rank",
                    "delta_mean_rank",
                ),
        }.items():
            low, high = percentile_ci(
                reps[
                    bootstrap_col
                ]
            )

            bootstrap_summary_rows.append(
                {
                    "comparison_key":
                        key,

                    "comparison_label":
                        row[
                            "comparison_label"
                        ],

                    "metric":
                        metric,

                    "estimate":
                        float(
                            row[
                                estimate_col
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

                    "seed":
                        seed,
                }
            )

    bootstrap_replicates = pd.concat(
        bootstrap_frames,
        ignore_index=True,
    )

    bootstrap_summary = pd.DataFrame(
        bootstrap_summary_rows
    )

    print(
        bootstrap_summary.loc[
            bootstrap_summary[
                "metric"
            ]
            == "HR@10",
            [
                "comparison_key",
                "estimate",
                "ci95_low",
                "ci95_high",
            ],
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
    # 7.6B.9 — Cumulative HR@k curves for primary contrast
    # =========================================================================

    print_section(
        "7.6B.9 — PRIMARY TEMPORAL-HISTORY HR@K CURVES"
    )

    cdf_rows = []

    primary_groups = {
        "Investor T0 + T1-T59":
            inv_both_mask,

        "Investor T1-T59 only":
            inv_t1_only_mask,
    }

    for label, mask in (
        primary_groups.items()
    ):
        ranks = (
            df.loc[
                mask,
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
                    "group":
                        label,

                    "k":
                        k,

                    "HR@k":
                        float(
                            (
                                ranks
                                <= k
                            ).mean()
                        ),
                }
            )

    cdf = pd.DataFrame(
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

        subset = cdf.loc[
            cdf[
                "k"
            ]
            == cutoff
        ]

        for _, row in (
            subset.iterrows()
        ):
            print(
                f"  {row['group']:<26} "
                f"{row['HR@k']:.2%}"
            )

    # =========================================================================
    # 7.6B.10 — Interpretation scaffold
    # =========================================================================

    print_section(
        "7.6B.10 — INTERPRETATION SCAFFOLD"
    )

    primary_ci = (
        bootstrap_summary.loc[
            (
                bootstrap_summary[
                    "comparison_key"
                ]
                == primary[
                    "comparison_key"
                ]
            )
            &
            (
                bootstrap_summary[
                    "metric"
                ]
                == "HR@10"
            )
        ]
        .iloc[0]
    )

    controlled_ci = (
        bootstrap_summary.loc[
            (
                bootstrap_summary[
                    "comparison_key"
                ]
                == controlled[
                    "comparison_key"
                ]
            )
            &
            (
                bootstrap_summary[
                    "metric"
                ]
                == "HR@10"
            )
        ]
        .iloc[0]
    )

    print(
        "Primary investor-history comparison:"
    )

    print(
        f"  ΔHR@10 = "
        f"{primary['delta_HR@10']:+.2%}"
    )

    print(
        f"  investor-cluster 95% CI = "
        f"[{primary_ci['ci95_low']:+.2%}, "
        f"{primary_ci['ci95_high']:+.2%}]"
    )

    print()
    print(
        "Controlled comparison "
        "(startup T1-T59 only):"
    )

    print(
        f"  ΔHR@10 = "
        f"{controlled['delta_HR@10']:+.2%}"
    )

    print(
        f"  investor-cluster 95% CI = "
        f"[{controlled_ci['ci95_low']:+.2%}, "
        f"{controlled_ci['ci95_high']:+.2%}]"
    )

    print()
    print(
        "Startup temporal-history comparison remains "
        "descriptive because the T0+T1-T59 startup "
        f"group has N={startup_contrast['group_a_n']}, "
        f"below the frozen threshold of "
        f"{MIN_FORMAL_GROUP_N}."
    )

    print()
    print(
        "Interpretation boundary:"
    )

    print(
        "  A difference between investor history classes "
        "is an observational association with temporal "
        "coverage, not a causal effect."
    )

    print(
        "  Literal T0 / T1-T59 semantics are retained; "
        "no unsupported recency/depth interpretation "
        "is introduced."
    )

    # =========================================================================
    # 7.6B.11 — Write artifacts
    # =========================================================================

    print_section(
        "7.6B.11 — WRITE ANALYSIS ARTIFACTS"
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    investor_metrics_path = (
        OUT_DIR
        / "phase_7_6b_new_ww_investor_history_metrics_V1.csv"
    )

    startup_metrics_path = (
        OUT_DIR
        / "phase_7_6b_new_ww_startup_history_metrics_V1.csv"
    )

    contrast_path = (
        OUT_DIR
        / "phase_7_6b_temporal_history_contrasts_V1.csv"
    )

    joint_path = (
        OUT_DIR
        / "phase_7_6b_new_ww_joint_history_performance_V1.csv"
    )

    bootstrap_reps_path = (
        OUT_DIR
        / "phase_7_6b_investor_cluster_bootstrap_replicates_V1.csv"
    )

    bootstrap_summary_path = (
        OUT_DIR
        / "phase_7_6b_investor_cluster_bootstrap_summary_V1.csv"
    )

    cdf_path = (
        OUT_DIR
        / "phase_7_6b_primary_temporal_history_rank_cdf_V1.csv"
    )

    evidence_path = (
        OUT_DIR
        / "phase_7_6b_thesis_evidence_V1.json"
    )

    investor_metrics.to_csv(
        investor_metrics_path,
        index=False,
    )

    startup_metrics.to_csv(
        startup_metrics_path,
        index=False,
    )

    contrasts.to_csv(
        contrast_path,
        index=False,
    )

    joint_metrics.to_csv(
        joint_path,
        index=False,
    )

    bootstrap_replicates.to_csv(
        bootstrap_reps_path,
        index=False,
    )

    bootstrap_summary.to_csv(
        bootstrap_summary_path,
        index=False,
    )

    cdf.to_csv(
        cdf_path,
        index=False,
    )

    evidence = {
        "phase":
            "7.6B",

        "scientific_scope":
            (
                "Pre-registered post-hoc temporal-history "
                "performance diagnostic within new-to-investor "
                "warm/warm cases."
            ),

        "primary_comparison":
            primary,

        "controlled_comparison":
            controlled,

        "startup_temporal_history":
            {
                "status":
                    "DESCRIPTIVE_ONLY",

                "comparison":
                    startup_contrast,

                "reason":
                    (
                        "T0+T1-T59 startup group below "
                        "pre-specified minimum N."
                    ),
            },

        "minimum_group_n_for_formal_bootstrap":
            MIN_FORMAL_GROUP_N,

        "uncertainty": {
            "method":
                (
                    "Percentile investor-cluster bootstrap"
                ),

            "repetitions":
                BOOTSTRAP_REPS,

            "base_seed":
                BOOTSTRAP_SEED,

            "limitation":
                (
                    "One-way investor clustering only; "
                    "startup and multiway dependence "
                    "deferred to Phase 7.9."
                ),
        },

        "semantic_boundary":
            (
                "T0 and T1-T59 labels retain their "
                "literal frozen meanings; no additional "
                "recency/depth semantics are asserted."
            ),

        "interpretation_boundary":
            (
                "Observed differences are associations, "
                "not causal effects."
            ),
    }

    json_dump(
        evidence,
        evidence_path,
    )

    # =========================================================================
    # 7.6B.12 — Figures
    # =========================================================================

    print_section(
        "7.6B.12 — GENERATE PRESENTATION-READY FIGURES"
    )

    figure_paths = []

    # Figure 1 — primary top-k comparison
    primary_a = calculate_metrics(
        df.loc[
            inv_both_mask
        ]
    )

    primary_b = calculate_metrics(
        df.loc[
            inv_t1_only_mask
        ]
    )

    metric_names = [
        "HR@1",
        "HR@5",
        "HR@10",
    ]

    values_a = np.array(
        [
            primary_a[m]
            for m in metric_names
        ]
    ) * 100.0

    values_b = np.array(
        [
            primary_b[m]
            for m in metric_names
        ]
    ) * 100.0

    x = np.arange(
        len(metric_names)
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(10.5, 6.5)
    )

    bars_a = ax.bar(
        x - width / 2,
        values_a,
        width=width,
        label=(
            "Investor T0 + T1-T59 "
            f"(N={primary_a['n']:,})"
        ),
    )

    bars_b = ax.bar(
        x + width / 2,
        values_b,
        width=width,
        label=(
            "Investor T1-T59 only "
            f"(N={primary_b['n']:,})"
        ),
    )

    for bars, values in (
        (bars_a, values_a),
        (bars_b, values_b),
    ):
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
                xytext=(
                    0,
                    4,
                ),
                textcoords="offset points",
                ha="center",
            )

    ax.set_xticks(
        x,
        metric_names,
    )

    ax.set_ylabel(
        "Hit rate (%)"
    )

    ax.set_title(
        "New-to-investor warm/warm ranking by investor temporal coverage"
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
                "phase_7_6b_fig1_"
                "investor_temporal_coverage_topk_V1"
            ),
        )
    )

    # Figure 2 — primary HR@k curves
    fig, ax = plt.subplots(
        figsize=(11, 6.5)
    )

    for label in (
        "Investor T0 + T1-T59",
        "Investor T1-T59 only",
    ):
        subset = cdf.loc[
            cdf[
                "group"
            ]
            == label
        ]

        ax.plot(
            subset[
                "k"
            ],
            subset[
                "HR@k"
            ] * 100.0,
            linewidth=2.0,
            label=label,
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
        "New warm/warm retrieval by investor temporal-history class"
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
                "phase_7_6b_fig2_"
                "investor_temporal_coverage_rank_cdf_V1"
            ),
        )
    )

    # Figure 3 — HR@10 contrast forest
    forest = (
        bootstrap_summary.loc[
            bootstrap_summary[
                "metric"
            ]
            == "HR@10"
        ]
        .copy()
    )

    label_map = {
        primary[
            "comparison_key"
        ]:
            "Primary\n(all new warm/warm)",

        controlled[
            "comparison_key"
        ]:
            "Controlled\n(startup T1-T59 only)",
    }

    forest[
        "display_label"
    ] = (
        forest[
            "comparison_key"
        ].map(
            label_map
        )
    )

    estimates = (
        forest[
            "estimate"
        ].to_numpy()
        * 100.0
    )

    lows = (
        forest[
            "ci95_low"
        ].to_numpy()
        * 100.0
    )

    highs = (
        forest[
            "ci95_high"
        ].to_numpy()
        * 100.0
    )

    y = np.arange(
        len(forest)
    )

    fig, ax = plt.subplots(
        figsize=(9.5, 5.5)
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
        forest[
            "display_label"
        ],
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        (
            "HR@10 difference: "
            "T0+T1-T59 minus T1-T59-only "
            "(percentage points)"
        )
    )

    ax.set_title(
        "Investor temporal-coverage contrasts\n"
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
                "phase_7_6b_fig3_"
                "investor_temporal_HR10_forest_V1"
            ),
        )
    )

    # Figure 4 — descriptive startup history classes
    startup_plot = (
        startup_metrics
        .set_index(
            "history_class"
        )
        .reindex(
            [
                "t0_only",
                "t1_t59_only",
                "t0_and_t1_t59",
            ]
        )
        .dropna(
            subset=["n"]
        )
    )

    x = np.arange(
        len(
            startup_plot
        )
    )

    values = (
        startup_plot[
            "HR@10"
        ].to_numpy()
        * 100.0
    )

    labels = [
        (
            f"{history_class}\n"
            f"N={int(startup_plot.loc[history_class, 'n']):,}"
        )
        for history_class in (
            startup_plot.index
        )
    ]

    fig, ax = plt.subplots(
        figsize=(9.5, 6)
    )

    bars = ax.bar(
        x,
        values,
    )

    ax.set_xticks(
        x,
        labels,
    )

    ax.set_ylabel(
        "HR@10 (%)"
    )

    ax.set_title(
        "Startup temporal-history classes in new warm/warm cases\n"
        "Descriptive only for small T0-containing groups"
    )

    ax.grid(
        axis="y",
        alpha=0.20,
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
            xytext=(
                0,
                5,
            ),
            textcoords="offset points",
            ha="center",
        )

    fig.tight_layout()

    figure_paths.extend(
        save_figure(
            fig,
            (
                "phase_7_6b_fig4_"
                "startup_history_descriptive_HR10_V1"
            ),
        )
    )

    # =========================================================================
    # 7.6B.13 — Manifest / hashes
    # =========================================================================

    manifest_path = (
        OUT_DIR
        / "phase_7_6b_analysis_manifest_V1.json"
    )

    manifest = {
        "phase":
            "7.6B",

        "schema_version":
            "ITRS_PHASE7_6B_TEMPORAL_HISTORY_PERFORMANCE_V1",

        "status":
            "COMPLETE",

        "scientific_role":
            (
                "Pre-registered post-hoc temporal-history "
                "performance diagnostic."
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

        "preregistration": {
            "manifest":
                str(
                    PHASE_7_6A_MANIFEST.relative_to(
                        REPO_ROOT
                    )
                ),

            "comparison_contract":
                str(
                    PHASE_7_6A_PREREG.relative_to(
                        REPO_ROOT
                    )
                ),

            "outcomes_inspected_before_preregistration":
                False,
        },

        "primary_comparison":
            primary[
                "comparison_key"
            ],

        "controlled_comparison":
            controlled[
                "comparison_key"
            ],

        "startup_temporal_history_inference":
            "DESCRIPTIVE_ONLY_SMALL_N",

        "minimum_group_n_for_formal_bootstrap":
            MIN_FORMAL_GROUP_N,

        "uncertainty": {
            "method":
                (
                    "Percentile investor-cluster bootstrap"
                ),

            "repetitions":
                BOOTSTRAP_REPS,

            "base_seed":
                BOOTSTRAP_SEED,
        },

        "semantic_boundary":
            (
                "Literal T0/T1-T59 classes only; "
                "no unsupported recency or history-depth "
                "labels."
            ),

        "interpretation_boundary":
            (
                "Temporal-history performance differences "
                "are observational associations."
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

            "model_selection":
                False,
        },
    }

    json_dump(
        manifest,
        manifest_path,
    )

    artifact_paths = [
        investor_metrics_path,
        startup_metrics_path,
        contrast_path,
        joint_path,
        bootstrap_reps_path,
        bootstrap_summary_path,
        cdf_path,
        evidence_path,
        *figure_paths,
        manifest_path,
    ]

    hashes = {}

    for path in (
        artifact_paths
    ):
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
        / "phase_7_6b_derived_artifact_sha256_V1.json"
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
        "PHASE 7.6B V1 RESULT"
    )

    print("Frozen source binding:                PASS")
    print("Outcome-blind preregistration gate:   PASS")
    print("Literal history-class binding:        PASS")
    print("Investor history metrics:             COMPLETE")
    print("Startup history metrics:              COMPLETE")
    print("Primary investor temporal contrast:   COMPLETE")
    print("Controlled investor contrast:         COMPLETE")
    print("Startup small-N descriptive analysis: COMPLETE")
    print("Joint temporal-support matrix:        COMPLETE")
    print("Investor-cluster uncertainty:         COMPLETE")
    print("Presentation-ready figures:           COMPLETE")
    print("Provenance manifest:                  COMPLETE")

    print()
    print("Model inference executed:             NO")
    print("Final test rescored:                  NO")
    print("Final test modified:                  NO")

    print()
    print(
        "PHASE 7.6B TEMPORAL HISTORY "
        "PERFORMANCE STATUS: COMPLETE"
    )


if __name__ == "__main__":
    main()