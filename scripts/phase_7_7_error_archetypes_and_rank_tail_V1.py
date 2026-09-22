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
# Phase 7.7 — Error Archetypes & Rank-Tail Anatomy V1
#
# Purpose
# -------
# Identify where the frozen model's ranking failures are concentrated using
# diagnostic dimensions that were established BEFORE this phase:
#
#   - pair novelty / entity cold-start status          (Phase 7.4)
#   - startup structural availability under cold start (Phase 7.5B)
#   - fixed rank-severity bands                         (Phase 7.1)
#
# This phase is DESCRIPTIVE. It does not tune the model, select a checkpoint,
# search for thresholds, or introduce outcome-optimized subgroup definitions.
#
# Frozen rank-severity bands
# --------------------------
#   Top-10 hit       : rank 1-10
#   Near miss        : rank 11-20
#   Deep miss        : rank 21-50
#   Severe miss      : rank 51-100
#
# Pre-specified archetype partition
# ---------------------------------
#   1. prior_ww
#   2. new_ww
#   3. new_wc_startup_connected
#   4. new_wc_startup_isolated
#   5. new_cw
#   6. new_cc_startup_connected
#   7. new_cc_startup_isolated
#
# These seven categories form a mutually exclusive, exhaustive partition of
# all 20,264 frozen final-test events.
#
# Key outputs
# -----------
# - performance and tail metrics by archetype
# - contribution of each archetype to Top-10 failures and severe rank>50 errors
# - severe-error enrichment relative to archetype exposure
# - rank-band composition / HR@k curves
# - tiny collision diagnostic (descriptive only)
# - deterministic representative severe cases for later qualitative inspection
#
# Prohibited
# ----------
# - model loading
# - checkpoint loading
# - raw-logit loading
# - inference
# - test rescoring
# - candidate modification
# - negative resampling
# - model selection
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

OUT_DIR = (
    REPO_ROOT
    / "data/experimental/phase_7/"
    / "phase_7_7_error_archetypes"
)

FIG_DIR = OUT_DIR / "figures"

EXPECTED_SOURCE_SHA256 = (
    "d70f21bff0006e094c5d567d307c0664"
    "b41a11e7250a0e69aaec610e1810132d"
)

EXPECTED_ROWS = 20_264
EXPECTED_TOTAL_SEVERE = 3_933
EXPECTED_TOTAL_TOP10_FAILURES = 9_192

ARCHETYPE_ORDER = [
    "prior_ww",
    "new_ww",
    "new_wc_startup_connected",
    "new_wc_startup_isolated",
    "new_cw",
    "new_cc_startup_connected",
    "new_cc_startup_isolated",
]

ARCHETYPE_LABELS = {
    "prior_ww":
        "Prior pair | warm investor + warm startup",

    "new_ww":
        "New pair | warm investor + warm startup",

    "new_wc_startup_connected":
        "New pair | warm investor + cold startup | startup connected",

    "new_wc_startup_isolated":
        "New pair | warm investor + cold startup | startup isolated",

    "new_cw":
        "New pair | cold investor + warm startup",

    "new_cc_startup_connected":
        "New pair | dual cold | startup connected",

    "new_cc_startup_isolated":
        "New pair | dual cold | startup isolated",
}

EXPECTED_ARCHETYPE_COUNTS = {
    "prior_ww": 3_818,
    "new_ww": 6_889,
    "new_wc_startup_connected": 809,
    "new_wc_startup_isolated": 5_583,
    "new_cw": 1_453,
    "new_cc_startup_connected": 148,
    "new_cc_startup_isolated": 1_564,
}

RANK_BAND_ORDER = [
    "Top-10 hit",
    "Near miss (11-20)",
    "Deep miss (21-50)",
    "Severe miss (51-100)",
]

SOURCE_COLUMNS = [
    "interaction_id",
    "investor_id",
    "investor_name",
    "startup_id",
    "startup_name",
    "announced_on",
    "investment_type",
    "positive_rank",
    "NDCG@10",
    "new_to_investor_pair",
    "prior_investor_startup_relationship",
    "cold_start_investor",
    "cold_start_startup",
    "investor_core_connected",
    "startup_core_connected",
    "investor_core_total_degree",
    "startup_core_total_degree",
    "investor_structural_source_class",
    "startup_structural_source_class",
    "has_realized_other_T60_positive_collision",
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


def build_archetype(
    df: pd.DataFrame,
) -> pd.Series:
    new = (
        df["new_to_investor_pair"]
        .astype(bool)
    )

    cold_i = (
        df["cold_start_investor"]
        .astype(bool)
    )

    cold_s = (
        df["cold_start_startup"]
        .astype(bool)
    )

    startup_connected = (
        df["startup_core_connected"]
        .astype(bool)
    )

    return pd.Series(
        np.select(
            [
                (~new) & (~cold_i) & (~cold_s),
                new & (~cold_i) & (~cold_s),
                new & (~cold_i) & cold_s & startup_connected,
                new & (~cold_i) & cold_s & (~startup_connected),
                new & cold_i & (~cold_s),
                new & cold_i & cold_s & startup_connected,
                new & cold_i & cold_s & (~startup_connected),
            ],
            ARCHETYPE_ORDER,
            default="INVALID",
        ),
        index=df.index,
    )


def build_rank_band(
    rank: pd.Series,
) -> pd.Series:
    return pd.Series(
        np.select(
            [
                rank.le(10),
                rank.between(11, 20),
                rank.between(21, 50),
                rank.between(51, 100),
            ],
            RANK_BAND_ORDER,
            default="INVALID",
        ),
        index=rank.index,
    )


def calculate_metrics(
    group: pd.DataFrame,
) -> dict:
    rank = (
        group["positive_rank"]
        .astype(int)
    )

    return {
        "n":
            int(len(group)),

        "Hits@1":
            int((rank <= 1).sum()),

        "Hits@5":
            int((rank <= 5).sum()),

        "Hits@10":
            int((rank <= 10).sum()),

        "HR@1":
            float((rank <= 1).mean()),

        "HR@5":
            float((rank <= 5).mean()),

        "HR@10":
            float((rank <= 10).mean()),

        "NDCG@10":
            float(group["NDCG@10"].mean()),

        "mean_rank":
            float(rank.mean()),

        "median_rank":
            float(rank.median()),

        "P75_rank":
            float(rank.quantile(0.75)),

        "P90_rank":
            float(rank.quantile(0.90)),

        "P95_rank":
            float(rank.quantile(0.95)),

        "top10_failure_n":
            int((rank > 10).sum()),

        "top10_failure_rate":
            float((rank > 10).mean()),

        "rank_gt_50_n":
            int((rank > 50).sum()),

        "rank_gt_50_rate":
            float((rank > 50).mean()),

        "rank_gt_75_n":
            int((rank > 75).sum()),

        "rank_gt_75_rate":
            float((rank > 75).mean()),
    }


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    print("=" * 110)
    print(
        "PHASE 7.7 — ERROR ARCHETYPES & "
        "RANK-TAIL ANATOMY V1"
    )
    print("=" * 110)

    print(
        "Scientific role:             "
        "DESCRIPTIVE FROZEN-TEST FAILURE ANATOMY"
    )
    print("Outcome-tuned subgroups:      NO")
    print("Rank thresholds searched:     NO")
    print("Model loaded:                NO")
    print("Checkpoint loaded:           NO")
    print("Raw logits loaded:           NO")
    print("Inference executed:          NO")
    print("Test rescored:               NO")
    print("Model selection performed:   NO")

    # =========================================================================
    # 7.7.1 — Integrity gate
    # =========================================================================

    print_section(
        "7.7.1 — ANALYSIS INTEGRITY GATE"
    )

    for path in (
        SOURCE,
        PHASE_7_4_MANIFEST,
        PHASE_7_5B_MANIFEST,
        PHASE_7_6B_MANIFEST,
    ):
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
        "Frozen source SHA256 drift.",
    )

    for path, expected_status in (
        (PHASE_7_4_MANIFEST, "COMPLETE"),
        (PHASE_7_5B_MANIFEST, "COMPLETE"),
        (PHASE_7_6B_MANIFEST, "COMPLETE"),
    ):
        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            manifest = json.load(f)

        require(
            manifest["status"] == expected_status,
            (
                f"Prerequisite manifest not "
                f"{expected_status}: {path}"
            ),
        )

    print()
    print("Frozen source fingerprint:   PASS")
    print("Phase 7.4 prerequisite:       PASS")
    print("Phase 7.5B prerequisite:      PASS")
    print("Phase 7.6B prerequisite:      PASS")

    # =========================================================================
    # 7.7.2 — Load / bind archetypes
    # =========================================================================

    print_section(
        "7.7.2 — CONSTRUCT PRE-SPECIFIED ERROR ARCHETYPE PARTITION"
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
        df["positive_rank"]
        .between(1, 100)
        .all(),
        "positive_rank outside 1..100.",
    )

    df["error_archetype"] = (
        build_archetype(df)
    )

    require(
        ~df["error_archetype"]
        .eq("INVALID")
        .any(),
        "Invalid error-archetype combination.",
    )

    archetype_counts = (
        df["error_archetype"]
        .value_counts()
    )

    for key in ARCHETYPE_ORDER:
        observed = int(
            archetype_counts[key]
        )

        expected = (
            EXPECTED_ARCHETYPE_COUNTS[key]
        )

        require(
            observed == expected,
            (
                f"Archetype count drift for {key}: "
                f"{observed} != {expected}"
            ),
        )

        print(
            f"{ARCHETYPE_LABELS[key]:<73} "
            f"{observed:>6,} "
            f"({observed / len(df):6.2%})"
        )

    require(
        sum(
            EXPECTED_ARCHETYPE_COUNTS.values()
        )
        == EXPECTED_ROWS,
        "Archetypes do not partition all final-test events.",
    )

    print()
    print("Mutually exclusive partition: PASS")
    print("Exhaustive partition:         PASS")

    # =========================================================================
    # 7.7.3 — Archetype performance / tail metrics
    # =========================================================================

    print_section(
        "7.7.3 — PERFORMANCE AND TAIL METRICS BY ARCHETYPE"
    )

    metric_rows = []

    for key in ARCHETYPE_ORDER:
        subgroup = df.loc[
            df["error_archetype"]
            == key
        ]

        metrics = calculate_metrics(
            subgroup
        )

        metrics[
            "archetype"
        ] = key

        metrics[
            "archetype_label"
        ] = ARCHETYPE_LABELS[key]

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

    total_top10_failures = int(
        (
            df["positive_rank"]
            > 10
        ).sum()
    )

    total_severe = int(
        (
            df["positive_rank"]
            > 50
        ).sum()
    )

    require(
        total_top10_failures
        == EXPECTED_TOTAL_TOP10_FAILURES,
        "Top-10 failure count drift.",
    )

    require(
        total_severe
        == EXPECTED_TOTAL_SEVERE,
        "Severe rank>50 count drift.",
    )

    metrics_df[
        "share_of_all_top10_failures"
    ] = (
        metrics_df[
            "top10_failure_n"
        ]
        / total_top10_failures
    )

    metrics_df[
        "share_of_all_severe_failures"
    ] = (
        metrics_df[
            "rank_gt_50_n"
        ]
        / total_severe
    )

    metrics_df[
        "severe_failure_enrichment"
    ] = (
        metrics_df[
            "share_of_all_severe_failures"
        ]
        / metrics_df[
            "test_share"
        ]
    )

    print(
        metrics_df[
            [
                "archetype_label",
                "n",
                "HR@10",
                "NDCG@10",
                "mean_rank",
                "median_rank",
                "top10_failure_rate",
                "rank_gt_50_rate",
                "share_of_all_severe_failures",
                "severe_failure_enrichment",
            ]
        ].to_string(
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
                "top10_failure_rate":
                    lambda x: f"{x:.2%}",
                "rank_gt_50_rate":
                    lambda x: f"{x:.2%}",
                "share_of_all_severe_failures":
                    lambda x: f"{x:.2%}",
                "severe_failure_enrichment":
                    lambda x: f"{x:.2f}x",
            },
        )
    )

    # =========================================================================
    # 7.7.4 — Rank-band decomposition
    # =========================================================================

    print_section(
        "7.7.4 — FIXED RANK-SEVERITY BAND DECOMPOSITION"
    )

    df["rank_severity_band"] = (
        build_rank_band(
            df[
                "positive_rank"
            ]
        )
    )

    require(
        ~df[
            "rank_severity_band"
        ]
        .eq("INVALID")
        .any(),
        "Invalid rank-severity band.",
    )

    band_counts = (
        df.groupby(
            [
                "error_archetype",
                "rank_severity_band",
            ],
            observed=False,
        )
        .size()
        .rename("n")
        .reset_index()
    )

    archetype_size_map = (
        df[
            "error_archetype"
        ]
        .value_counts()
    )

    band_counts[
        "within_archetype_share"
    ] = (
        band_counts["n"]
        / band_counts[
            "error_archetype"
        ].map(
            archetype_size_map
        )
    )

    band_order_map = {
        key: i
        for i, key in enumerate(
            RANK_BAND_ORDER
        )
    }

    archetype_order_map = {
        key: i
        for i, key in enumerate(
            ARCHETYPE_ORDER
        )
    }

    band_counts[
        "_archetype_order"
    ] = (
        band_counts[
            "error_archetype"
        ].map(
            archetype_order_map
        )
    )

    band_counts[
        "_band_order"
    ] = (
        band_counts[
            "rank_severity_band"
        ].map(
            band_order_map
        )
    )

    band_counts = (
        band_counts
        .sort_values(
            [
                "_archetype_order",
                "_band_order",
            ]
        )
        .drop(
            columns=[
                "_archetype_order",
                "_band_order",
            ]
        )
        .reset_index(drop=True)
    )

    band_counts[
        "archetype_label"
    ] = (
        band_counts[
            "error_archetype"
        ].map(
            ARCHETYPE_LABELS
        )
    )

    print(
        band_counts[
            [
                "archetype_label",
                "rank_severity_band",
                "n",
                "within_archetype_share",
            ]
        ].to_string(
            index=False,
            formatters={
                "within_archetype_share":
                    lambda x: f"{x:.2%}",
            },
        )
    )

    # =========================================================================
    # 7.7.5 — Severe-failure concentration
    # =========================================================================

    print_section(
        "7.7.5 — SEVERE-FAILURE CONCENTRATION"
    )

    severe = df.loc[
        df["positive_rank"]
        > 50
    ].copy()

    require(
        len(severe)
        == EXPECTED_TOTAL_SEVERE,
        "Severe-case count drift.",
    )

    severe_counts = (
        severe[
            "error_archetype"
        ]
        .value_counts()
        .reindex(
            ARCHETYPE_ORDER,
            fill_value=0,
        )
    )

    print(
        f"All rank>50 failures:         "
        f"{len(severe):,}"
    )

    print()

    for key in ARCHETYPE_ORDER:
        n = int(
            severe_counts[key]
        )

        print(
            f"{ARCHETYPE_LABELS[key]:<73} "
            f"{n:>6,} "
            f"({n / len(severe):6.2%})"
        )

    severe_with_cold_startup = int(
        (
            severe[
                "cold_start_startup"
            ].astype(bool)
        ).sum()
    )

    severe_with_isolated_cold_startup = int(
        (
            severe[
                "cold_start_startup"
            ].astype(bool)
            &
            ~severe[
                "startup_core_connected"
            ].astype(bool)
        ).sum()
    )

    severe_new_wc_isolated = int(
        severe[
            "error_archetype"
        ].eq(
            "new_wc_startup_isolated"
        ).sum()
    )

    print()
    print(
        "SEVERE-TAIL SUMMARY"
    )

    print(
        f"  Cold-start startup involved:       "
        f"{severe_with_cold_startup:,} "
        f"({severe_with_cold_startup / len(severe):.2%})"
    )

    print(
        f"  Cold + structurally isolated startup: "
        f"{severe_with_isolated_cold_startup:,} "
        f"({severe_with_isolated_cold_startup / len(severe):.2%})"
    )

    print(
        f"  Warm investor + cold isolated startup: "
        f"{severe_new_wc_isolated:,} "
        f"({severe_new_wc_isolated / len(severe):.2%})"
    )

    # =========================================================================
    # 7.7.6 — Top-10 failure concentration
    # =========================================================================

    print_section(
        "7.7.6 — TOP-10 FAILURE CONCENTRATION"
    )

    misses = df.loc[
        df["positive_rank"]
        > 10
    ].copy()

    require(
        len(misses)
        == EXPECTED_TOTAL_TOP10_FAILURES,
        "Top-10 miss count drift.",
    )

    miss_counts = (
        misses[
            "error_archetype"
        ]
        .value_counts()
        .reindex(
            ARCHETYPE_ORDER,
            fill_value=0,
        )
    )

    print(
        f"All Top-10 failures:          "
        f"{len(misses):,}"
    )

    print()

    for key in ARCHETYPE_ORDER:
        n = int(
            miss_counts[key]
        )

        print(
            f"{ARCHETYPE_LABELS[key]:<73} "
            f"{n:>6,} "
            f"({n / len(misses):6.2%})"
        )

    # =========================================================================
    # 7.7.7 — HR@k curves by archetype
    # =========================================================================

    print_section(
        "7.7.7 — CUMULATIVE HR@K BY ERROR ARCHETYPE"
    )

    cdf_rows = []

    for key in ARCHETYPE_ORDER:
        ranks = (
            df.loc[
                df[
                    "error_archetype"
                ]
                == key,
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
                    "archetype":
                        key,

                    "archetype_label":
                        ARCHETYPE_LABELS[
                            key
                        ],

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

        for key in ARCHETYPE_ORDER:
            value = float(
                subset.loc[
                    subset[
                        "archetype"
                    ]
                    == key,
                    "HR@k",
                ].iloc[0]
            )

            print(
                f"  {ARCHETYPE_LABELS[key]:<71} "
                f"{value:.2%}"
            )

    # =========================================================================
    # 7.7.8 — Collision diagnostic
    # =========================================================================

    print_section(
        "7.7.8 — REALIZED-POSITIVE COLLISION DIAGNOSTIC"
    )

    collision = (
        df[
            "has_realized_other_T60_positive_collision"
        ].astype(bool)
    )

    collision_rows = []

    for label, mask in (
        (
            "collision",
            collision,
        ),
        (
            "no_collision",
            ~collision,
        ),
    ):
        subgroup = df.loc[
            mask
        ]

        metrics = calculate_metrics(
            subgroup
        )

        collision_rows.append(
            {
                "collision_status":
                    label,

                **metrics,
            }
        )

    collision_df = pd.DataFrame(
        collision_rows
    )

    print(
        collision_df[
            [
                "collision_status",
                "n",
                "HR@10",
                "NDCG@10",
                "mean_rank",
                "median_rank",
                "rank_gt_50_rate",
            ]
        ].to_string(
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
                "rank_gt_50_rate":
                    lambda x: f"{x:.2%}",
            },
        )
    )

    print()
    print(
        "Collision analysis is descriptive only; "
        "the collision subgroup is tiny."
    )

    # =========================================================================
    # 7.7.9 — Deterministic severe-case export
    # =========================================================================

    print_section(
        "7.7.9 — DETERMINISTIC REPRESENTATIVE SEVERE CASES"
    )

    representative_rows = []

    for key in ARCHETYPE_ORDER:
        subgroup = (
            df.loc[
                (
                    df[
                        "error_archetype"
                    ]
                    == key
                )
                &
                (
                    df[
                        "positive_rank"
                    ]
                    > 50
                )
            ]
            .sort_values(
                [
                    "positive_rank",
                    "interaction_id",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .head(10)
            .copy()
        )

        if len(subgroup) == 0:
            continue

        subgroup[
            "selection_rule"
        ] = (
            "highest_positive_rank_then_interaction_id"
        )

        representative_rows.append(
            subgroup[
                [
                    "error_archetype",
                    "interaction_id",
                    "investor_id",
                    "investor_name",
                    "startup_id",
                    "startup_name",
                    "announced_on",
                    "investment_type",
                    "positive_rank",
                    "NDCG@10",
                    "investor_core_connected",
                    "startup_core_connected",
                    "investor_core_total_degree",
                    "startup_core_total_degree",
                    "investor_structural_source_class",
                    "startup_structural_source_class",
                    "selection_rule",
                ]
            ]
        )

    if representative_rows:
        representatives = pd.concat(
            representative_rows,
            ignore_index=True,
        )
    else:
        representatives = pd.DataFrame()

    print(
        f"Representative severe cases exported: "
        f"{len(representatives):,}"
    )

    print(
        "Selection is deterministic and descriptive; "
        "examples are not used for statistical inference."
    )

    # =========================================================================
    # 7.7.10 — Interpretation scaffold
    # =========================================================================

    print_section(
        "7.7.10 — ERROR-ANATOMY INTERPRETATION SCAFFOLD"
    )

    ranked_contributors = (
        metrics_df.sort_values(
            "share_of_all_severe_failures",
            ascending=False,
        )
    )

    print(
        "Largest contributors to the rank>50 tail:"
    )

    for _, row in (
        ranked_contributors.head(
            4
        ).iterrows()
    ):
        print(
            f"  {row['archetype_label']}: "
            f"{int(row['rank_gt_50_n']):,} cases, "
            f"{row['share_of_all_severe_failures']:.2%} "
            f"of severe errors, "
            f"{row['severe_failure_enrichment']:.2f}x "
            f"exposure-relative enrichment"
        )

    print()
    print(
        "Interpretation boundary:"
    )

    print(
        "  Error archetypes describe where frozen-test "
        "failures concentrate."
    )

    print(
        "  They do not establish causal mechanisms."
    )

    print(
        "  The severe-error enrichment ratio compares "
        "error share with test-set exposure; it is not "
        "an inferential risk ratio."
    )

    # =========================================================================
    # 7.7.11 — Write artifacts
    # =========================================================================

    print_section(
        "7.7.11 — WRITE ANALYSIS ARTIFACTS"
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
        / "phase_7_7_error_archetype_metrics_V1.csv"
    )

    bands_path = (
        OUT_DIR
        / "phase_7_7_rank_severity_bands_V1.csv"
    )

    cdf_path = (
        OUT_DIR
        / "phase_7_7_archetype_rank_cdf_V1.csv"
    )

    collision_path = (
        OUT_DIR
        / "phase_7_7_collision_diagnostic_V1.csv"
    )

    representative_path = (
        OUT_DIR
        / "phase_7_7_representative_severe_cases_V1.csv"
    )

    summary_path = (
        OUT_DIR
        / "phase_7_7_error_anatomy_summary_V1.json"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    band_counts.to_csv(
        bands_path,
        index=False,
    )

    cdf.to_csv(
        cdf_path,
        index=False,
    )

    collision_df.to_csv(
        collision_path,
        index=False,
    )

    representatives.to_csv(
        representative_path,
        index=False,
    )

    summary = {
        "phase":
            "7.7",

        "total_test_cases":
            len(df),

        "total_top10_failures":
            total_top10_failures,

        "total_rank_gt_50_failures":
            total_severe,

        "cold_startup_in_severe_tail_n":
            severe_with_cold_startup,

        "cold_startup_in_severe_tail_share":
            (
                severe_with_cold_startup
                / total_severe
            ),

        "cold_isolated_startup_in_severe_tail_n":
            severe_with_isolated_cold_startup,

        "cold_isolated_startup_in_severe_tail_share":
            (
                severe_with_isolated_cold_startup
                / total_severe
            ),

        "new_wc_isolated_in_severe_tail_n":
            severe_new_wc_isolated,

        "new_wc_isolated_in_severe_tail_share":
            (
                severe_new_wc_isolated
                / total_severe
            ),

        "rank_band_policy":
            RANK_BAND_ORDER,

        "archetype_policy":
            ARCHETYPE_LABELS,

        "interpretation_boundary":
            (
                "Descriptive concentration / enrichment only; "
                "no causal or inferential interpretation."
            ),
    }

    json_dump(
        summary,
        summary_path,
    )

    # =========================================================================
    # 7.7.12 — Figures
    # =========================================================================

    print_section(
        "7.7.12 — GENERATE PRESENTATION-READY FIGURES"
    )

    figure_paths = []

    short_labels = [
        "Prior\nwarm/warm",
        "New\nwarm/warm",
        "New warm I /\ncold S connected",
        "New warm I /\ncold S isolated",
        "New cold I /\nwarm S",
        "New dual cold /\nS connected",
        "New dual cold /\nS isolated",
    ]

    ordered_metrics = (
        metrics_df
        .set_index(
            "archetype"
        )
        .loc[
            ARCHETYPE_ORDER
        ]
    )

    # Figure 1 — severe-error contribution
    values = (
        ordered_metrics[
            "share_of_all_severe_failures"
        ]
        .to_numpy()
        * 100.0
    )

    x = np.arange(
        len(
            ARCHETYPE_ORDER
        )
    )

    fig, ax = plt.subplots(
        figsize=(13, 7)
    )

    bars = ax.bar(
        x,
        values,
    )

    ax.set_xticks(
        x,
        short_labels,
    )

    ax.set_ylabel(
        "Share of all rank>50 failures (%)"
    )

    ax.set_title(
        "Where the frozen ITRS model's severe ranking failures concentrate"
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
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    fig.tight_layout()

    figure_paths.extend(
        save_figure(
            fig,
            (
                "phase_7_7_fig1_"
                "severe_error_contribution_V1"
            ),
        )
    )

    # Figure 2 — severe-error enrichment
    enrichment = (
        ordered_metrics[
            "severe_failure_enrichment"
        ]
        .to_numpy()
    )

    fig, ax = plt.subplots(
        figsize=(13, 7)
    )

    bars = ax.bar(
        x,
        enrichment,
    )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_xticks(
        x,
        short_labels,
    )

    ax.set_ylabel(
        "Severe-error enrichment vs test-set exposure (x)"
    )

    ax.set_title(
        "Which archetypes are over-represented in the rank>50 tail?"
    )

    ax.grid(
        axis="y",
        alpha=0.20,
    )

    for bar, value in zip(
        bars,
        enrichment,
    ):
        ax.annotate(
            f"{value:.2f}x",
            (
                bar.get_x()
                + bar.get_width() / 2,
                bar.get_height(),
            ),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    fig.tight_layout()

    figure_paths.extend(
        save_figure(
            fig,
            (
                "phase_7_7_fig2_"
                "severe_error_enrichment_V1"
            ),
        )
    )

    # Figure 3 — stacked rank-severity bands
    pivot = (
        band_counts.pivot(
            index="error_archetype",
            columns="rank_severity_band",
            values="within_archetype_share",
        )
        .reindex(
            ARCHETYPE_ORDER
        )
        .reindex(
            columns=RANK_BAND_ORDER
        )
        .fillna(0.0)
        * 100.0
    )

    fig, ax = plt.subplots(
        figsize=(13.5, 7)
    )

    bottom = np.zeros(
        len(
            ARCHETYPE_ORDER
        )
    )

    for band in RANK_BAND_ORDER:
        values = (
            pivot[
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
        short_labels,
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.set_ylabel(
        "Share of archetype (%)"
    )

    ax.set_title(
        "Rank-severity composition by pre-specified error archetype"
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
                "phase_7_7_fig3_"
                "rank_severity_composition_V1"
            ),
        )
    )

    # Figure 4 — CDF for selected thesis-relevant archetypes
    selected = [
        "new_ww",
        "new_wc_startup_connected",
        "new_wc_startup_isolated",
        "new_cc_startup_isolated",
    ]

    fig, ax = plt.subplots(
        figsize=(11, 6.5)
    )

    for key in selected:
        subset = cdf.loc[
            cdf[
                "archetype"
            ]
            == key
        ]

        ax.plot(
            subset[
                "k"
            ],
            subset[
                "HR@k"
            ] * 100.0,
            linewidth=2.0,
            label=ARCHETYPE_LABELS[
                key
            ],
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
        "Rank CDF of thesis-relevant error archetypes"
    )

    ax.legend(
        fontsize=8,
    )

    ax.grid(
        alpha=0.20,
    )

    fig.tight_layout()

    figure_paths.extend(
        save_figure(
            fig,
            (
                "phase_7_7_fig4_"
                "selected_archetype_rank_cdf_V1"
            ),
        )
    )

    # =========================================================================
    # 7.7.13 — Manifest / hashes
    # =========================================================================

    manifest_path = (
        OUT_DIR
        / "phase_7_7_analysis_manifest_V1.json"
    )

    manifest = {
        "phase":
            "7.7",

        "schema_version":
            "ITRS_PHASE7_7_ERROR_ARCHETYPES_V1",

        "status":
            "COMPLETE",

        "scientific_role":
            (
                "Descriptive frozen-test failure anatomy "
                "using pre-established diagnostic dimensions."
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

        "archetype_partition":
            ARCHETYPE_LABELS,

        "rank_severity_policy":
            {
                "top10_hit":
                    "positive_rank <= 10",

                "near_miss":
                    "11 <= positive_rank <= 20",

                "deep_miss":
                    "21 <= positive_rank <= 50",

                "severe_miss":
                    "51 <= positive_rank <= 100",
            },

        "outcome_tuned_subgroups":
            False,

        "threshold_search":
            False,

        "representative_case_selection":
            (
                "Within each archetype, rank>50 cases sorted "
                "by positive_rank descending then interaction_id "
                "ascending; first 10 exported."
            ),

        "interpretation_boundary":
            (
                "Descriptive error concentration only; "
                "no causal mechanism inferred."
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
        metrics_path,
        bands_path,
        cdf_path,
        collision_path,
        representative_path,
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
        / "phase_7_7_derived_artifact_sha256_V1.json"
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
        "PHASE 7.7 V1 RESULT"
    )

    print("Frozen source binding:              PASS")
    print("Archetype partition binding:        PASS")
    print("Fixed rank-severity bands:          PASS")
    print("Archetype performance metrics:      COMPLETE")
    print("Severe-tail concentration:          COMPLETE")
    print("Top-10 failure concentration:       COMPLETE")
    print("Rank-distribution diagnostics:      COMPLETE")
    print("Collision diagnostic:               COMPLETE")
    print("Representative severe-case export: COMPLETE")
    print("Presentation-ready figures:         COMPLETE")
    print("Provenance manifest:                COMPLETE")

    print()
    print("Model inference executed:           NO")
    print("Final test rescored:                NO")
    print("Final test modified:                NO")

    print()
    print(
        "PHASE 7.7 ERROR ARCHETYPES & "
        "RANK-TAIL ANATOMY STATUS: COMPLETE"
    )


if __name__ == "__main__":
    main()