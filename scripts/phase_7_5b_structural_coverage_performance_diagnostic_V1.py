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
# Phase 7.5B — Structural Coverage Performance Diagnostic V1
#
# Core question
# -------------
# Is frozen ITRS ranking performance associated with availability of audited
# heterogeneous graph structure, especially when investment history is absent?
#
# Scientific design
# -----------------
# Comparisons were frozen outcome-blind in Phase 7.5A V2 before this script
# inspects positive_rank / HR / NDCG outcomes.
#
# PRIMARY thesis-focused structural comparison:
#
#   Within NEW-TO-INVESTOR cases with:
#       warm investor + cold-start startup
#
#   compare:
#       startup structurally connected
#       vs
#       startup structurally isolated
#
# This is especially relevant because the startup has no prior investment
# history by construction, while the structural fields may still supply
# founder/acquisition-side information.
#
# Secondary checks hold investor structural connectivity fixed:
#
#   investor isolated:
#       startup connected only vs neither connected
#
#   investor connected:
#       both connected vs investor connected only
#
# Cold-investor and dual-cold diagnostics are also reported, but formal
# bootstrap contrasts require both compared groups to have at least
# MIN_FORMAL_GROUP_N cases.
#
# IMPORTANT INTERPRETATION BOUNDARY
# ---------------------------------
# These are observational associations inside the frozen test.
# Better performance for structurally connected entities would NOT prove that
# structure causally "rescues" performance, nor that a particular new model
# architecture will improve results.
#
# Uncertainty
# -----------
# Percentile bootstrap with investor_id as the cluster-resampling unit.
# Startup clustering / multiway dependence is deferred to Phase 7.9.
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

PHASE_7_5A_MANIFEST = (
    REPO_ROOT
    / "data/experimental/phase_7/"
    / "phase_7_5a_structural_coverage_composition/"
    / "phase_7_5a_analysis_manifest_V2.json"
)

PHASE_7_5A_PREREG = (
    REPO_ROOT
    / "data/experimental/phase_7/"
    / "phase_7_5a_structural_coverage_composition/"
    / "phase_7_5a_preregistered_7_5b_comparisons_V2.json"
)

OUT_DIR = (
    REPO_ROOT
    / "data/experimental/phase_7/"
    / "phase_7_5b_structural_performance"
)

FIG_DIR = OUT_DIR / "figures"

EXPECTED_SOURCE_SHA256 = (
    "d70f21bff0006e094c5d567d307c0664"
    "b41a11e7250a0e69aaec610e1810132d"
)

EXPECTED_ROWS = 20_264

EXPECTED_DIAGNOSTIC_COUNTS = {
    "prior_ww": 3_818,
    "new_ww": 6_889,
    "new_wc": 6_392,
    "new_cw": 1_453,
    "new_cc": 1_712,
}

EXPECTED_NEW_WC_STARTUP_CONNECTED = 809
EXPECTED_NEW_WC_STARTUP_ISOLATED = 5_583

EXPECTED_NEW_WC_INVESTOR_ISOLATED_STARTUP_CONNECTED = 676
EXPECTED_NEW_WC_NEITHER = 4_935

EXPECTED_NEW_WC_BOTH_CONNECTED = 133
EXPECTED_NEW_WC_INVESTOR_ONLY = 648

EXPECTED_NEW_CW_INVESTOR_CONNECTED = 37
EXPECTED_NEW_CW_INVESTOR_ISOLATED = 1_416

EXPECTED_NEW_CC_STARTUP_CONNECTED = 148
EXPECTED_NEW_CC_STARTUP_ISOLATED = 1_564

EXPECTED_NEW_CC_INVESTOR_CONNECTED = 56
EXPECTED_NEW_CC_INVESTOR_ISOLATED = 1_656

DIAGNOSTIC_LABELS = {
    "prior_ww": "Prior pair | Warm investor + Warm startup",
    "new_ww": "New pair | Warm investor + Warm startup",
    "new_wc": "New pair | Warm investor + Cold startup",
    "new_cw": "New pair | Cold investor + Warm startup",
    "new_cc": "New pair | Cold investor + Cold startup",
}

DIAGNOSTIC_ORDER = [
    "prior_ww",
    "new_ww",
    "new_wc",
    "new_cw",
    "new_cc",
]

STRUCTURAL_ORDER = [
    "Both connected",
    "Investor connected only",
    "Startup connected only",
    "Neither connected",
]

SOURCE_CLASS_ORDER = [
    "founder_and_acquisition",
    "founder_only",
    "acquisition_only",
    "neither",
]

BOOTSTRAP_REPS = 2_000
BOOTSTRAP_SEED = 7_105_002

# Pre-specified before inspecting structural performance outcomes.
MIN_FORMAL_GROUP_N = 100

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
    "investor_structural_source_class",
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
        f"Object of type {type(value).__name__} "
        "is not JSON serializable"
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


def build_diagnostic_group(
    df: pd.DataFrame,
) -> pd.Series:
    new = df["new_to_investor_pair"].astype(bool)
    cold_i = df["cold_start_investor"].astype(bool)
    cold_s = df["cold_start_startup"].astype(bool)

    return pd.Series(
        np.select(
            [
                (~new) & (~cold_i) & (~cold_s),
                new & (~cold_i) & (~cold_s),
                new & (~cold_i) & cold_s,
                new & cold_i & (~cold_s),
                new & cold_i & cold_s,
            ],
            DIAGNOSTIC_ORDER,
            default="INVALID",
        ),
        index=df.index,
    )


def build_joint_structural_status(
    df: pd.DataFrame,
) -> pd.Series:
    investor_connected = (
        df["investor_core_connected"]
        .astype(bool)
    )

    startup_connected = (
        df["startup_core_connected"]
        .astype(bool)
    )

    return pd.Series(
        np.select(
            [
                investor_connected & startup_connected,
                investor_connected & ~startup_connected,
                ~investor_connected & startup_connected,
                ~investor_connected & ~startup_connected,
            ],
            STRUCTURAL_ORDER,
            default="INVALID",
        ),
        index=df.index,
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


def build_metric_table(
    df: pd.DataFrame,
    group_col: str,
    group_order: list[str],
    label_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    rows = []

    for group_key in group_order:
        subgroup = df.loc[
            df[group_col] == group_key
        ]

        if len(subgroup) == 0:
            continue

        row = calculate_metrics(subgroup)
        row["group_key"] = group_key

        if label_map is None:
            row["group_label"] = group_key
        else:
            row["group_label"] = label_map.get(
                group_key,
                group_key,
            )

        rows.append(row)

    return pd.DataFrame(rows)


def contrast_from_groups(
    df: pd.DataFrame,
    mask_a: pd.Series,
    mask_b: pd.Series,
    comparison_key: str,
    comparison_label: str,
    group_a_label: str,
    group_b_label: str,
    priority: str,
    formal_candidate: bool = True,
) -> dict:
    a = df.loc[mask_a]
    b = df.loc[mask_b]

    require(
        len(a) > 0 and len(b) > 0,
        f"Empty comparison group in {comparison_key}.",
    )

    ma = calculate_metrics(a)
    mb = calculate_metrics(b)

    min_n = min(
        ma["n"],
        mb["n"],
    )

    formal_eligible = bool(
        formal_candidate
        and min_n >= MIN_FORMAL_GROUP_N
    )

    return {
        "comparison_key": comparison_key,
        "comparison_label": comparison_label,
        "priority": priority,
        "group_a_label": group_a_label,
        "group_b_label": group_b_label,
        "group_a_n": ma["n"],
        "group_b_n": mb["n"],
        "formal_bootstrap_eligible": formal_eligible,
        "minimum_group_n_threshold": MIN_FORMAL_GROUP_N,
        "group_a_HR@10": ma["HR@10"],
        "group_b_HR@10": mb["HR@10"],
        "delta_HR@10": (
            ma["HR@10"]
            - mb["HR@10"]
        ),
        "group_a_NDCG@10": ma["NDCG@10"],
        "group_b_NDCG@10": mb["NDCG@10"],
        "delta_NDCG@10": (
            ma["NDCG@10"]
            - mb["NDCG@10"]
        ),
        "group_a_mean_rank": ma["mean_rank"],
        "group_b_mean_rank": mb["mean_rank"],
        "delta_mean_rank": (
            ma["mean_rank"]
            - mb["mean_rank"]
        ),
        "group_a_median_rank": ma["median_rank"],
        "group_b_median_rank": mb["median_rank"],
        "delta_median_rank": (
            ma["median_rank"]
            - mb["median_rank"]
        ),
        "group_a_rank_gt_50_share": (
            ma["rank_gt_50_share"]
        ),
        "group_b_rank_gt_50_share": (
            mb["rank_gt_50_share"]
        ),
        "delta_rank_gt_50_share": (
            ma["rank_gt_50_share"]
            - mb["rank_gt_50_share"]
        ),
    }


def percentile_ci(
    values: pd.Series,
) -> tuple[float, float]:
    clean = (
        values
        .dropna()
        .to_numpy(dtype=float)
    )

    require(
        len(clean) > 0,
        "Cannot compute CI from empty bootstrap values.",
    )

    low, high = np.quantile(
        clean,
        [0.025, 0.975],
    )

    return (
        float(low),
        float(high),
    )


def cluster_bootstrap_comparison(
    df: pd.DataFrame,
    mask_a: pd.Series,
    mask_b: pd.Series,
    reps: int,
    seed: int,
) -> pd.DataFrame:
    """
    Investor-cluster bootstrap for one pre-specified two-group contrast.

    All events from a sampled investor are carried together.
    """

    working = df.loc[
        mask_a | mask_b,
        [
            "investor_id",
            "positive_rank",
            "NDCG@10",
        ],
    ].copy()

    working["analysis_group"] = np.where(
        mask_a.loc[working.index],
        "A",
        "B",
    )

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

    arrays = {}

    for group_key in (
        "A",
        "B",
    ):
        grouped = (
            working.loc[
                working["analysis_group"]
                == group_key
            ]
            .groupby(
                "investor_id",
                sort=False,
            )
            .agg(
                n=("positive_rank", "size"),
                hits10=("hit10", "sum"),
                ndcg_sum=("NDCG@10", "sum"),
                rank_sum=("positive_rank", "sum"),
            )
            .reindex(
                investors,
                fill_value=0,
            )
        )

        arrays[group_key] = {
            col: (
                grouped[col]
                .to_numpy(dtype=float)
            )
            for col in (
                "n",
                "hits10",
                "ndcg_sum",
                "rank_sum",
            )
        }

    rng = np.random.default_rng(seed)

    rows = []

    for bootstrap_index in range(reps):
        sampled = rng.integers(
            0,
            len(investors),
            size=len(investors),
        )

        result = {
            "bootstrap_index": bootstrap_index,
        }

        valid = True

        for group_key in (
            "A",
            "B",
        ):
            arr = arrays[group_key]

            n = float(
                arr["n"][sampled].sum()
            )

            if n <= 0:
                valid = False
                break

            result[
                f"{group_key}__HR@10"
            ] = float(
                arr["hits10"][sampled].sum()
                / n
            )

            result[
                f"{group_key}__NDCG@10"
            ] = float(
                arr["ndcg_sum"][sampled].sum()
                / n
            )

            result[
                f"{group_key}__mean_rank"
            ] = float(
                arr["rank_sum"][sampled].sum()
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

        rows.append(result)

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    print("=" * 110)
    print(
        "PHASE 7.5B — STRUCTURAL COVERAGE "
        "PERFORMANCE DIAGNOSTIC V1"
    )
    print("=" * 110)

    print(
        "Scientific role:             "
        "PRE-REGISTERED POST-HOC STRUCTURAL ASSOCIATION DIAGNOSTIC"
    )
    print("Model loaded:                NO")
    print("Checkpoint loaded:           NO")
    print("Raw logits loaded:           NO")
    print("Inference executed:          NO")
    print("Test rescored:               NO")
    print("Model selection performed:   NO")

    # =========================================================================
    # 7.5B.1 — Integrity / preregistration gate
    # =========================================================================

    print_section(
        "7.5B.1 — INTEGRITY AND PRE-REGISTRATION GATE"
    )

    for path in (
        SOURCE,
        PHASE_7_4_MANIFEST,
        PHASE_7_5A_MANIFEST,
        PHASE_7_5A_PREREG,
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
        "Frozen Phase-6 bundle SHA256 drift.",
    )

    with PHASE_7_4_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as f:
        phase_7_4 = json.load(f)

    with PHASE_7_5A_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as f:
        phase_7_5a = json.load(f)

    with PHASE_7_5A_PREREG.open(
        "r",
        encoding="utf-8",
    ) as f:
        prereg = json.load(f)

    require(
        phase_7_4["status"] == "COMPLETE",
        "Phase 7.4 is not COMPLETE.",
    )

    require(
        phase_7_5a["status"] == "COMPLETE",
        "Phase 7.5A V2 is not COMPLETE.",
    )

    require(
        phase_7_5a[
            "outcome_blindness"
        ][
            "performance_metrics_examined"
        ]
        is False,
        (
            "Phase 7.5A does not confirm "
            "outcome-blind pre-registration."
        ),
    )

    required_preregistered = {
        "overall_joint_structure",
        "new_pair_joint_structure",
        "new_warm_warm_joint_structure",
        "cold_startup_structural_rescue",
        "cold_investor_structural_rescue",
        "dual_cold_structural_support",
        "source_class_diagnostic",
    }

    require(
        required_preregistered
        <= set(prereg.keys()),
        (
            "Phase 7.5A preregistration is "
            "missing planned comparisons."
        ),
    )

    print()
    print("Frozen source fingerprint:   PASS")
    print("Phase 7.4 prerequisite:       PASS")
    print("Phase 7.5A prerequisite:      PASS")
    print("Outcome-blind preregistration:PASS")

    # =========================================================================
    # 7.5B.2 — Load frozen outcome + structural fields
    # =========================================================================

    print_section(
        "7.5B.2 — LOAD FROZEN OUTCOMES AND PRE-REGISTERED STRUCTURAL FIELDS"
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

    df["diagnostic_group"] = (
        build_diagnostic_group(df)
    )

    df[
        "joint_structural_status"
    ] = build_joint_structural_status(
        df
    )

    require(
        ~df["diagnostic_group"]
        .eq("INVALID")
        .any(),
        "Invalid novelty/cold-start group.",
    )

    require(
        ~df[
            "joint_structural_status"
        ]
        .eq("INVALID")
        .any(),
        "Invalid joint structural status.",
    )

    counts = (
        df["diagnostic_group"]
        .value_counts()
    )

    for key, expected in (
        EXPECTED_DIAGNOSTIC_COUNTS.items()
    ):
        require(
            int(counts[key]) == expected,
            f"Diagnostic count drift for {key}.",
        )

    print(
        f"Test cases:                   "
        f"{len(df):,}"
    )
    print(
        "Diagnostic group binding:    PASS"
    )
    print(
        "Joint structural binding:    PASS"
    )

    # =========================================================================
    # 7.5B.3 — Overall / new-pair / warm-new joint structure
    # =========================================================================

    print_section(
        "7.5B.3 — PRE-REGISTERED JOINT STRUCTURAL PERFORMANCE"
    )

    overall_joint = build_metric_table(
        df=df,
        group_col="joint_structural_status",
        group_order=STRUCTURAL_ORDER,
    )

    new_df = df.loc[
        df["new_to_investor_pair"]
        .astype(bool)
    ].copy()

    new_joint = build_metric_table(
        df=new_df,
        group_col="joint_structural_status",
        group_order=STRUCTURAL_ORDER,
    )

    new_ww = df.loc[
        df["diagnostic_group"]
        == "new_ww"
    ].copy()

    new_ww_joint = build_metric_table(
        df=new_ww,
        group_col="joint_structural_status",
        group_order=STRUCTURAL_ORDER,
    )

    print(
        "ALL TEST CASES"
    )
    print(
        overall_joint[
            [
                "group_label",
                "n",
                "HR@10",
                "NDCG@10",
                "mean_rank",
                "median_rank",
                "rank_gt_50_share",
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
                "rank_gt_50_share":
                    lambda x: f"{x:.2%}",
            },
        )
    )

    print()
    print(
        "NEW-TO-INVESTOR ONLY"
    )
    print(
        new_joint[
            [
                "group_label",
                "n",
                "HR@10",
                "NDCG@10",
                "mean_rank",
                "median_rank",
                "rank_gt_50_share",
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
                "rank_gt_50_share":
                    lambda x: f"{x:.2%}",
            },
        )
    )

    print()
    print(
        "NEW-TO-INVESTOR WARM/WARM ONLY"
    )
    print(
        new_ww_joint[
            [
                "group_label",
                "n",
                "HR@10",
                "NDCG@10",
                "mean_rank",
                "median_rank",
                "rank_gt_50_share",
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
                "rank_gt_50_share":
                    lambda x: f"{x:.2%}",
            },
        )
    )

    # =========================================================================
    # 7.5B.4 — Primary cold-start startup structural-support comparison
    # =========================================================================

    print_section(
        "7.5B.4 — PRIMARY: STRUCTURAL SUPPORT FOR COLD-START STARTUPS"
    )

    new_wc_mask = (
        df["diagnostic_group"]
        == "new_wc"
    )

    new_wc = df.loc[
        new_wc_mask
    ]

    startup_connected_mask = (
        new_wc_mask
        &
        df[
            "startup_core_connected"
        ].astype(bool)
    )

    startup_isolated_mask = (
        new_wc_mask
        &
        ~df[
            "startup_core_connected"
        ].astype(bool)
    )

    require(
        int(startup_connected_mask.sum())
        == EXPECTED_NEW_WC_STARTUP_CONNECTED,
        "new_wc startup-connected count drift.",
    )

    require(
        int(startup_isolated_mask.sum())
        == EXPECTED_NEW_WC_STARTUP_ISOLATED,
        "new_wc startup-isolated count drift.",
    )

    primary_startup_contrast = (
        contrast_from_groups(
            df=df,
            mask_a=startup_connected_mask,
            mask_b=startup_isolated_mask,
            comparison_key=(
                "new_wc_startup_connected_vs_isolated"
            ),
            comparison_label=(
                "New warm-investor/cold-startup: "
                "startup connected - startup isolated"
            ),
            group_a_label=(
                "Structurally connected cold-start startup"
            ),
            group_b_label=(
                "Structurally isolated cold-start startup"
            ),
            priority="PRIMARY",
        )
    )

    print(
        f"Connected startup N:          "
        f"{primary_startup_contrast['group_a_n']:,}"
    )
    print(
        f"Isolated startup N:           "
        f"{primary_startup_contrast['group_b_n']:,}"
    )
    print()
    print(
        f"Connected HR@10:              "
        f"{primary_startup_contrast['group_a_HR@10']:.2%}"
    )
    print(
        f"Isolated HR@10:               "
        f"{primary_startup_contrast['group_b_HR@10']:.2%}"
    )
    print(
        f"ΔHR@10:                       "
        f"{primary_startup_contrast['delta_HR@10']:+.2%}"
    )
    print()
    print(
        f"Connected NDCG@10:            "
        f"{primary_startup_contrast['group_a_NDCG@10']:.4f}"
    )
    print(
        f"Isolated NDCG@10:             "
        f"{primary_startup_contrast['group_b_NDCG@10']:.4f}"
    )
    print(
        f"ΔNDCG@10:                     "
        f"{primary_startup_contrast['delta_NDCG@10']:+.4f}"
    )
    print()
    print(
        f"Connected mean rank:          "
        f"{primary_startup_contrast['group_a_mean_rank']:.2f}"
    )
    print(
        f"Isolated mean rank:           "
        f"{primary_startup_contrast['group_b_mean_rank']:.2f}"
    )
    print(
        f"Δmean rank:                   "
        f"{primary_startup_contrast['delta_mean_rank']:+.2f}"
    )

    # =========================================================================
    # 7.5B.5 — Hold investor connectivity fixed
    # =========================================================================

    print_section(
        "7.5B.5 — STARTUP STRUCTURE WITH INVESTOR CONNECTIVITY HELD FIXED"
    )

    investor_isolated_mask = (
        ~df[
            "investor_core_connected"
        ].astype(bool)
    )

    investor_connected_mask = (
        df[
            "investor_core_connected"
        ].astype(bool)
    )

    wc_startup_only_mask = (
        new_wc_mask
        &
        investor_isolated_mask
        &
        df[
            "startup_core_connected"
        ].astype(bool)
    )

    wc_neither_mask = (
        new_wc_mask
        &
        investor_isolated_mask
        &
        ~df[
            "startup_core_connected"
        ].astype(bool)
    )

    require(
        int(wc_startup_only_mask.sum())
        == EXPECTED_NEW_WC_INVESTOR_ISOLATED_STARTUP_CONNECTED,
        (
            "new_wc startup-only count drift."
        ),
    )

    require(
        int(wc_neither_mask.sum())
        == EXPECTED_NEW_WC_NEITHER,
        "new_wc neither count drift.",
    )

    investor_isolated_startup_contrast = (
        contrast_from_groups(
            df=df,
            mask_a=wc_startup_only_mask,
            mask_b=wc_neither_mask,
            comparison_key=(
                "new_wc_investor_isolated_"
                "startup_connected_vs_isolated"
            ),
            comparison_label=(
                "New warm-investor/cold-startup, "
                "investor structurally isolated: "
                "startup connected - startup isolated"
            ),
            group_a_label=(
                "Startup connected only"
            ),
            group_b_label=(
                "Neither connected"
            ),
            priority="PRIMARY_CONTROLLED",
        )
    )

    wc_both_mask = (
        new_wc_mask
        &
        investor_connected_mask
        &
        df[
            "startup_core_connected"
        ].astype(bool)
    )

    wc_investor_only_mask = (
        new_wc_mask
        &
        investor_connected_mask
        &
        ~df[
            "startup_core_connected"
        ].astype(bool)
    )

    require(
        int(wc_both_mask.sum())
        == EXPECTED_NEW_WC_BOTH_CONNECTED,
        "new_wc both-connected count drift.",
    )

    require(
        int(wc_investor_only_mask.sum())
        == EXPECTED_NEW_WC_INVESTOR_ONLY,
        "new_wc investor-only count drift.",
    )

    investor_connected_startup_contrast = (
        contrast_from_groups(
            df=df,
            mask_a=wc_both_mask,
            mask_b=wc_investor_only_mask,
            comparison_key=(
                "new_wc_investor_connected_"
                "startup_connected_vs_isolated"
            ),
            comparison_label=(
                "New warm-investor/cold-startup, "
                "investor structurally connected: "
                "startup connected - startup isolated"
            ),
            group_a_label=(
                "Both connected"
            ),
            group_b_label=(
                "Investor connected only"
            ),
            priority="PRIMARY_CONTROLLED",
        )
    )

    controlled_rows = [
        investor_isolated_startup_contrast,
        investor_connected_startup_contrast,
    ]

    controlled_df = pd.DataFrame(
        controlled_rows
    )

    print(
        controlled_df[
            [
                "comparison_label",
                "group_a_n",
                "group_b_n",
                "group_a_HR@10",
                "group_b_HR@10",
                "delta_HR@10",
                "delta_NDCG@10",
                "delta_mean_rank",
            ]
        ].to_string(
            index=False,
            formatters={
                "group_a_HR@10":
                    lambda x: f"{x:.2%}",
                "group_b_HR@10":
                    lambda x: f"{x:.2%}",
                "delta_HR@10":
                    lambda x: f"{x:+.2%}",
                "delta_NDCG@10":
                    lambda x: f"{x:+.4f}",
                "delta_mean_rank":
                    lambda x: f"{x:+.2f}",
            },
        )
    )

    # =========================================================================
    # 7.5B.6 — Cold-investor structural diagnostic
    # =========================================================================

    print_section(
        "7.5B.6 — COLD-INVESTOR STRUCTURAL DIAGNOSTIC"
    )

    new_cw_mask = (
        df["diagnostic_group"]
        == "new_cw"
    )

    cw_investor_connected_mask = (
        new_cw_mask
        &
        df[
            "investor_core_connected"
        ].astype(bool)
    )

    cw_investor_isolated_mask = (
        new_cw_mask
        &
        ~df[
            "investor_core_connected"
        ].astype(bool)
    )

    require(
        int(
            cw_investor_connected_mask.sum()
        )
        == EXPECTED_NEW_CW_INVESTOR_CONNECTED,
        "new_cw investor-connected count drift.",
    )

    require(
        int(
            cw_investor_isolated_mask.sum()
        )
        == EXPECTED_NEW_CW_INVESTOR_ISOLATED,
        "new_cw investor-isolated count drift.",
    )

    cold_investor_contrast = (
        contrast_from_groups(
            df=df,
            mask_a=cw_investor_connected_mask,
            mask_b=cw_investor_isolated_mask,
            comparison_key=(
                "new_cw_investor_connected_vs_isolated"
            ),
            comparison_label=(
                "New cold-investor/warm-startup: "
                "investor connected - investor isolated"
            ),
            group_a_label=(
                "Structurally connected cold-start investor"
            ),
            group_b_label=(
                "Structurally isolated cold-start investor"
            ),
            priority="EXPLORATORY_SMALL_N",
        )
    )

    print(
        f"Connected cold investors:     "
        f"{cold_investor_contrast['group_a_n']:,}"
    )
    print(
        f"Isolated cold investors:      "
        f"{cold_investor_contrast['group_b_n']:,}"
    )
    print(
        f"Formal bootstrap eligible:    "
        f"{cold_investor_contrast['formal_bootstrap_eligible']}"
    )
    print(
        f"ΔHR@10:                       "
        f"{cold_investor_contrast['delta_HR@10']:+.2%}"
    )
    print(
        f"ΔNDCG@10:                     "
        f"{cold_investor_contrast['delta_NDCG@10']:+.4f}"
    )
    print(
        f"Δmean rank:                   "
        f"{cold_investor_contrast['delta_mean_rank']:+.2f}"
    )

    # =========================================================================
    # 7.5B.7 — Dual-cold structural support
    # =========================================================================

    print_section(
        "7.5B.7 — DUAL-COLD STRUCTURAL SUPPORT"
    )

    new_cc_mask = (
        df["diagnostic_group"]
        == "new_cc"
    )

    cc_startup_connected_mask = (
        new_cc_mask
        &
        df[
            "startup_core_connected"
        ].astype(bool)
    )

    cc_startup_isolated_mask = (
        new_cc_mask
        &
        ~df[
            "startup_core_connected"
        ].astype(bool)
    )

    require(
        int(
            cc_startup_connected_mask.sum()
        )
        == EXPECTED_NEW_CC_STARTUP_CONNECTED,
        "new_cc startup-connected count drift.",
    )

    require(
        int(
            cc_startup_isolated_mask.sum()
        )
        == EXPECTED_NEW_CC_STARTUP_ISOLATED,
        "new_cc startup-isolated count drift.",
    )

    dual_cold_startup_contrast = (
        contrast_from_groups(
            df=df,
            mask_a=cc_startup_connected_mask,
            mask_b=cc_startup_isolated_mask,
            comparison_key=(
                "new_cc_startup_connected_vs_isolated"
            ),
            comparison_label=(
                "New dual-cold: "
                "startup connected - startup isolated"
            ),
            group_a_label=(
                "Cold-start startup connected"
            ),
            group_b_label=(
                "Cold-start startup isolated"
            ),
            priority="SECONDARY",
        )
    )

    cc_investor_connected_mask = (
        new_cc_mask
        &
        df[
            "investor_core_connected"
        ].astype(bool)
    )

    cc_investor_isolated_mask = (
        new_cc_mask
        &
        ~df[
            "investor_core_connected"
        ].astype(bool)
    )

    require(
        int(
            cc_investor_connected_mask.sum()
        )
        == EXPECTED_NEW_CC_INVESTOR_CONNECTED,
        "new_cc investor-connected count drift.",
    )

    require(
        int(
            cc_investor_isolated_mask.sum()
        )
        == EXPECTED_NEW_CC_INVESTOR_ISOLATED,
        "new_cc investor-isolated count drift.",
    )

    dual_cold_investor_contrast = (
        contrast_from_groups(
            df=df,
            mask_a=cc_investor_connected_mask,
            mask_b=cc_investor_isolated_mask,
            comparison_key=(
                "new_cc_investor_connected_vs_isolated"
            ),
            comparison_label=(
                "New dual-cold: "
                "investor connected - investor isolated"
            ),
            group_a_label=(
                "Cold-start investor connected"
            ),
            group_b_label=(
                "Cold-start investor isolated"
            ),
            priority="EXPLORATORY_SMALL_N",
        )
    )

    dual_joint = build_metric_table(
        df=df.loc[new_cc_mask],
        group_col="joint_structural_status",
        group_order=STRUCTURAL_ORDER,
    )

    dual_contrasts_df = pd.DataFrame(
        [
            dual_cold_startup_contrast,
            dual_cold_investor_contrast,
        ]
    )

    print(
        dual_contrasts_df[
            [
                "comparison_label",
                "group_a_n",
                "group_b_n",
                "formal_bootstrap_eligible",
                "delta_HR@10",
                "delta_NDCG@10",
                "delta_mean_rank",
            ]
        ].to_string(
            index=False,
            formatters={
                "delta_HR@10":
                    lambda x: f"{x:+.2%}",
                "delta_NDCG@10":
                    lambda x: f"{x:+.4f}",
                "delta_mean_rank":
                    lambda x: f"{x:+.2f}",
            },
        )
    )

    print()
    print(
        "DUAL-COLD JOINT STRUCTURAL STATES"
    )

    print(
        dual_joint[
            [
                "group_label",
                "n",
                "HR@10",
                "NDCG@10",
                "mean_rank",
                "median_rank",
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
            },
        )
    )

    # =========================================================================
    # 7.5B.8 — Cold-start startup source-class diagnostic
    # =========================================================================

    print_section(
        "7.5B.8 — COLD-START STARTUP STRUCTURAL SOURCE CLASS"
    )

    new_wc_source = build_metric_table(
        df=df.loc[new_wc_mask],
        group_col=(
            "startup_structural_source_class"
        ),
        group_order=SOURCE_CLASS_ORDER,
    )

    print(
        new_wc_source[
            [
                "group_label",
                "n",
                "HR@10",
                "NDCG@10",
                "mean_rank",
                "median_rank",
                "rank_gt_50_share",
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
                "rank_gt_50_share":
                    lambda x: f"{x:.2%}",
            },
        )
    )

    source_contrasts = []

    for source_class in (
        "founder_only",
        "acquisition_only",
        "founder_and_acquisition",
    ):
        source_mask = (
            new_wc_mask
            &
            df[
                "startup_structural_source_class"
            ].eq(
                source_class
            )
        )

        neither_mask = (
            new_wc_mask
            &
            df[
                "startup_structural_source_class"
            ].eq(
                "neither"
            )
        )

        if int(source_mask.sum()) == 0:
            continue

        source_contrasts.append(
            contrast_from_groups(
                df=df,
                mask_a=source_mask,
                mask_b=neither_mask,
                comparison_key=(
                    f"new_wc_source_{source_class}_vs_neither"
                ),
                comparison_label=(
                    "New warm-investor/cold-startup: "
                    f"{source_class} - neither"
                ),
                group_a_label=source_class,
                group_b_label="neither",
                priority="SOURCE_CLASS",
            )
        )

    source_contrasts_df = pd.DataFrame(
        source_contrasts
    )

    print()
    print(
        "SOURCE-CLASS CONTRASTS VS NEITHER"
    )

    print(
        source_contrasts_df[
            [
                "comparison_label",
                "group_a_n",
                "group_b_n",
                "formal_bootstrap_eligible",
                "delta_HR@10",
                "delta_NDCG@10",
                "delta_mean_rank",
            ]
        ].to_string(
            index=False,
            formatters={
                "delta_HR@10":
                    lambda x: f"{x:+.2%}",
                "delta_NDCG@10":
                    lambda x: f"{x:+.4f}",
                "delta_mean_rank":
                    lambda x: f"{x:+.2f}",
            },
        )
    )

    # =========================================================================
    # 7.5B.9 — Freeze all contrasts / apply minimum-N rule
    # =========================================================================

    print_section(
        "7.5B.9 — FORMAL CONTRAST ELIGIBILITY"
    )

    all_contrast_rows = [
        primary_startup_contrast,
        investor_isolated_startup_contrast,
        investor_connected_startup_contrast,
        cold_investor_contrast,
        dual_cold_startup_contrast,
        dual_cold_investor_contrast,
        *source_contrasts,
    ]

    contrasts_df = pd.DataFrame(
        all_contrast_rows
    )

    print(
        contrasts_df[
            [
                "comparison_key",
                "priority",
                "group_a_n",
                "group_b_n",
                "formal_bootstrap_eligible",
                "delta_HR@10",
                "delta_NDCG@10",
                "delta_mean_rank",
            ]
        ].to_string(
            index=False,
            formatters={
                "delta_HR@10":
                    lambda x: f"{x:+.2%}",
                "delta_NDCG@10":
                    lambda x: f"{x:+.4f}",
                "delta_mean_rank":
                    lambda x: f"{x:+.2f}",
            },
        )
    )

    # =========================================================================
    # 7.5B.10 — Investor-cluster bootstrap
    # =========================================================================

    print_section(
        "7.5B.10 — INVESTOR-CLUSTER BOOTSTRAP"
    )

    print(
        f"Bootstrap replicates:         "
        f"{BOOTSTRAP_REPS:,}"
    )
    print(
        f"Base bootstrap seed:          "
        f"{BOOTSTRAP_SEED}"
    )
    print(
        f"Minimum formal group N:       "
        f"{MIN_FORMAL_GROUP_N:,}"
    )

    mask_lookup = {
        (
            "new_wc_startup_connected_vs_isolated"
        ): (
            startup_connected_mask,
            startup_isolated_mask,
        ),

        (
            "new_wc_investor_isolated_"
            "startup_connected_vs_isolated"
        ): (
            wc_startup_only_mask,
            wc_neither_mask,
        ),

        (
            "new_wc_investor_connected_"
            "startup_connected_vs_isolated"
        ): (
            wc_both_mask,
            wc_investor_only_mask,
        ),

        (
            "new_cw_investor_connected_vs_isolated"
        ): (
            cw_investor_connected_mask,
            cw_investor_isolated_mask,
        ),

        (
            "new_cc_startup_connected_vs_isolated"
        ): (
            cc_startup_connected_mask,
            cc_startup_isolated_mask,
        ),

        (
            "new_cc_investor_connected_vs_isolated"
        ): (
            cc_investor_connected_mask,
            cc_investor_isolated_mask,
        ),
    }

    for row in source_contrasts:
        key = row["comparison_key"]

        source_class = (
            key
            .replace(
                "new_wc_source_",
                "",
            )
            .replace(
                "_vs_neither",
                "",
            )
        )

        mask_lookup[key] = (
            (
                new_wc_mask
                &
                df[
                    "startup_structural_source_class"
                ].eq(
                    source_class
                )
            ),
            (
                new_wc_mask
                &
                df[
                    "startup_structural_source_class"
                ].eq(
                    "neither"
                )
            ),
        )

    bootstrap_summary_rows = []
    bootstrap_rep_frames = []

    formal_rows = contrasts_df.loc[
        contrasts_df[
            "formal_bootstrap_eligible"
        ]
    ].reset_index(drop=True)

    for i, row in formal_rows.iterrows():
        key = row["comparison_key"]

        mask_a, mask_b = (
            mask_lookup[key]
        )

        comparison_seed = (
            BOOTSTRAP_SEED
            + i
            + 1
        )

        reps_df = (
            cluster_bootstrap_comparison(
                df=df,
                mask_a=mask_a,
                mask_b=mask_b,
                reps=BOOTSTRAP_REPS,
                seed=comparison_seed,
            )
        )

        require(
            len(reps_df)
            == BOOTSTRAP_REPS,
            (
                f"Unexpected bootstrap replicate "
                f"count for {key}."
            ),
        )

        reps_df.insert(
            0,
            "comparison_key",
            key,
        )

        bootstrap_rep_frames.append(
            reps_df
        )

        metric_map = {
            "HR@10": (
                "delta_HR@10",
                "delta_HR@10",
            ),
            "NDCG@10": (
                "delta_NDCG@10",
                "delta_NDCG@10",
            ),
            "mean_rank": (
                "delta_mean_rank",
                "delta_mean_rank",
            ),
        }

        for metric, (
            bootstrap_col,
            estimate_col,
        ) in metric_map.items():
            low, high = percentile_ci(
                reps_df[
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
                        comparison_seed,
                }
            )

    if bootstrap_rep_frames:
        bootstrap_replicates = pd.concat(
            bootstrap_rep_frames,
            ignore_index=True,
        )
    else:
        bootstrap_replicates = pd.DataFrame()

    bootstrap_summary = pd.DataFrame(
        bootstrap_summary_rows
    )

    if not bootstrap_summary.empty:
        hr_summary = (
            bootstrap_summary.loc[
                bootstrap_summary[
                    "metric"
                ]
                == "HR@10"
            ]
        )

        print(
            hr_summary[
                [
                    "comparison_key",
                    "estimate",
                    "ci95_low",
                    "ci95_high",
                ]
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
    # 7.5B.11 — Primary rank CDF
    # =========================================================================

    print_section(
        "7.5B.11 — PRIMARY COLD-START STARTUP HR@K CURVE"
    )

    cdf_rows = []

    primary_groups = {
        "Connected cold-start startup":
            startup_connected_mask,

        "Isolated cold-start startup":
            startup_isolated_mask,
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
                            (ranks <= k)
                            .mean()
                        ),
                }
            )

    primary_cdf = pd.DataFrame(
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

        subset = (
            primary_cdf.loc[
                primary_cdf["k"]
                == cutoff
            ]
        )

        for _, row in (
            subset.iterrows()
        ):
            print(
                f"  {row['group']:<30} "
                f"{row['HR@k']:.2%}"
            )

    # =========================================================================
    # 7.5B.12 — Evidence interpretation scaffold
    # =========================================================================

    print_section(
        "7.5B.12 — INTERPRETATION SCAFFOLD"
    )

    primary_bootstrap_hr = None

    if not bootstrap_summary.empty:
        match = bootstrap_summary.loc[
            (
                bootstrap_summary[
                    "comparison_key"
                ]
                == (
                    "new_wc_startup_"
                    "connected_vs_isolated"
                )
            )
            &
            (
                bootstrap_summary[
                    "metric"
                ]
                == "HR@10"
            )
        ]

        if len(match) == 1:
            primary_bootstrap_hr = (
                match.iloc[0]
            )

    print(
        "Primary question:"
    )
    print(
        "  Among new-to-investor cases with a warm investor "
        "and a cold-start startup, is startup structural "
        "connectivity associated with better ranking?"
    )

    print()
    print(
        f"Observed ΔHR@10:              "
        f"{primary_startup_contrast['delta_HR@10']:+.2%}"
    )

    if primary_bootstrap_hr is not None:
        print(
            f"Investor-cluster 95% CI:      "
            f"[{primary_bootstrap_hr['ci95_low']:+.2%}, "
            f"{primary_bootstrap_hr['ci95_high']:+.2%}]"
        )

    print()
    print(
        "Controlled structural checks:"
    )
    print(
        "  1) Investor isolated: startup connected only vs neither."
    )
    print(
        "  2) Investor connected: both connected vs investor-only."
    )

    print()
    print(
        "Interpretation boundary:"
    )
    print(
        "  Positive structural contrasts indicate ASSOCIATION with "
        "better frozen-test ranking, not causal rescue."
    )
    print(
        "  Later robustness analysis will revisit startup clustering "
        "and multiway event dependence."
    )

    # =========================================================================
    # 7.5B.13 — Write artifacts
    # =========================================================================

    print_section(
        "7.5B.13 — WRITE ANALYSIS ARTIFACTS"
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall_path = (
        OUT_DIR
        / "phase_7_5b_overall_joint_structure_metrics_V1.csv"
    )

    new_joint_path = (
        OUT_DIR
        / "phase_7_5b_new_pair_joint_structure_metrics_V1.csv"
    )

    new_ww_path = (
        OUT_DIR
        / "phase_7_5b_new_warm_warm_joint_structure_metrics_V1.csv"
    )

    contrasts_path = (
        OUT_DIR
        / "phase_7_5b_structural_contrasts_V1.csv"
    )

    source_path = (
        OUT_DIR
        / "phase_7_5b_new_wc_startup_source_class_metrics_V1.csv"
    )

    dual_joint_path = (
        OUT_DIR
        / "phase_7_5b_dual_cold_joint_structure_metrics_V1.csv"
    )

    bootstrap_reps_path = (
        OUT_DIR
        / "phase_7_5b_investor_cluster_bootstrap_replicates_V1.csv"
    )

    bootstrap_summary_path = (
        OUT_DIR
        / "phase_7_5b_investor_cluster_bootstrap_summary_V1.csv"
    )

    cdf_path = (
        OUT_DIR
        / "phase_7_5b_primary_cold_startup_rank_cdf_V1.csv"
    )

    evidence_path = (
        OUT_DIR
        / "phase_7_5b_thesis_evidence_V1.json"
    )

    overall_joint.to_csv(
        overall_path,
        index=False,
    )

    new_joint.to_csv(
        new_joint_path,
        index=False,
    )

    new_ww_joint.to_csv(
        new_ww_path,
        index=False,
    )

    contrasts_df.to_csv(
        contrasts_path,
        index=False,
    )

    new_wc_source.to_csv(
        source_path,
        index=False,
    )

    dual_joint.to_csv(
        dual_joint_path,
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

    primary_cdf.to_csv(
        cdf_path,
        index=False,
    )

    evidence = {
        "phase":
            "7.5B",

        "scientific_scope":
            (
                "Pre-registered post-hoc association between "
                "audited heterogeneous structural coverage and "
                "frozen ITRS ranking performance."
            ),

        "primary_comparison":
            primary_startup_contrast,

        "controlled_startup_checks": {
            (
                "investor_isolated_"
                "startup_connected_vs_isolated"
            ):
                investor_isolated_startup_contrast,

            (
                "investor_connected_"
                "startup_connected_vs_isolated"
            ):
                investor_connected_startup_contrast,
        },

        "cold_investor_diagnostic":
            cold_investor_contrast,

        "dual_cold": {
            "startup_structure":
                dual_cold_startup_contrast,

            "investor_structure":
                dual_cold_investor_contrast,
        },

        "minimum_group_n_for_formal_bootstrap":
            MIN_FORMAL_GROUP_N,

        "uncertainty": {
            "method":
                (
                    "Percentile investor-cluster bootstrap "
                    "for pre-specified contrasts satisfying "
                    "the minimum subgroup-size rule."
                ),

            "repetitions":
                BOOTSTRAP_REPS,

            "base_seed":
                BOOTSTRAP_SEED,

            "limitation":
                (
                    "One-way investor clustering only. "
                    "Startup and multiway dependence are "
                    "deferred to Phase 7.9."
                ),
        },

        "interpretation_boundary": {
            "supported_if_positive":
                (
                    "Structural availability is associated with "
                    "better ranking in the examined frozen-test "
                    "regime."
                ),

            "not_supported":
                (
                    "Causal rescue, mechanism identification, or "
                    "proof that a proposed inductive architecture "
                    "will improve performance."
                ),
        },
    }

    json_dump(
        evidence,
        evidence_path,
    )

    # =========================================================================
    # 7.5B.14 — Figures
    # =========================================================================

    print_section(
        "7.5B.14 — GENERATE PRESENTATION-READY FIGURES"
    )

    figure_paths = []

    # -------------------------------------------------------------------------
    # Figure 1 — Primary cold-start startup structural comparison
    # -------------------------------------------------------------------------

    connected_metrics = calculate_metrics(
        df.loc[
            startup_connected_mask
        ]
    )

    isolated_metrics = calculate_metrics(
        df.loc[
            startup_isolated_mask
        ]
    )

    metric_names = [
        "HR@1",
        "HR@5",
        "HR@10",
    ]

    connected_values = np.array(
        [
            connected_metrics[m]
            for m in metric_names
        ]
    ) * 100.0

    isolated_values = np.array(
        [
            isolated_metrics[m]
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
        connected_values,
        width=width,
        label=(
            "Structurally connected "
            f"(N={connected_metrics['n']:,})"
        ),
    )

    bars_b = ax.bar(
        x + width / 2,
        isolated_values,
        width=width,
        label=(
            "Structurally isolated "
            f"(N={isolated_metrics['n']:,})"
        ),
    )

    for bars, values in (
        (bars_a, connected_values),
        (bars_b, isolated_values),
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
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )

    ax.set_xticks(
        x,
        metric_names,
    )

    ax.set_ylabel(
        "Hit rate (%)"
    )

    ax.set_title(
        "Cold-start startup ranking by heterogeneous structural availability\n"
        "New-to-investor, warm-investor cases only"
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
                "phase_7_5b_fig1_"
                "cold_startup_structural_HR_V1"
            ),
        )
    )

    # -------------------------------------------------------------------------
    # Figure 2 — Primary HR@k curves
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 6.5)
    )

    for label in (
        "Connected cold-start startup",
        "Isolated cold-start startup",
    ):
        subset = primary_cdf.loc[
            primary_cdf["group"]
            == label
        ]

        ax.plot(
            subset["k"],
            subset["HR@k"]
            * 100.0,
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
        "Positive-startup retrieval with vs without structural support\n"
        "New-to-investor cold-start startups"
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
                "phase_7_5b_fig2_"
                "cold_startup_structural_rank_cdf_V1"
            ),
        )
    )

    # -------------------------------------------------------------------------
    # Figure 3 — HR@10 forest of formal structural contrasts
    # -------------------------------------------------------------------------

    if not bootstrap_summary.empty:
        forest = bootstrap_summary.loc[
            bootstrap_summary[
                "metric"
            ]
            == "HR@10"
        ].copy()

        if len(forest) > 0:
            forest["label"] = (
                forest[
                    "comparison_key"
                ].replace(
                    {
                        (
                            "new_wc_startup_"
                            "connected_vs_isolated"
                        ):
                            "Cold startup: connected vs isolated",

                        (
                            "new_wc_investor_isolated_"
                            "startup_connected_vs_isolated"
                        ):
                            "Cold startup structure\n(investor isolated)",

                        (
                            "new_wc_investor_connected_"
                            "startup_connected_vs_isolated"
                        ):
                            "Cold startup structure\n(investor connected)",

                        (
                            "new_cc_startup_"
                            "connected_vs_isolated"
                        ):
                            "Dual-cold: startup\nconnected vs isolated",

                        (
                            "new_wc_source_"
                            "founder_only_vs_neither"
                        ):
                            "Founder-only vs neither",

                        (
                            "new_wc_source_"
                            "acquisition_only_vs_neither"
                        ):
                            "Acquisition-only vs neither",
                    }
                )
            )

            estimates = (
                forest["estimate"]
                .to_numpy()
                * 100.0
            )

            lows = (
                forest["ci95_low"]
                .to_numpy()
                * 100.0
            )

            highs = (
                forest["ci95_high"]
                .to_numpy()
                * 100.0
            )

            y = np.arange(
                len(forest)
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
                forest["label"],
            )

            ax.invert_yaxis()

            ax.set_xlabel(
                "Difference in HR@10 (percentage points)"
            )

            ax.set_title(
                "Pre-registered structural contrasts\n"
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
                        "phase_7_5b_fig3_"
                        "structural_HR10_contrast_forest_V1"
                    ),
                )
            )

    # -------------------------------------------------------------------------
    # Figure 4 — Structural source class in new_wc
    # -------------------------------------------------------------------------

    source_plot = (
        new_wc_source
        .set_index(
            "group_key"
        )
        .reindex(
            SOURCE_CLASS_ORDER
        )
        .dropna(
            subset=["n"]
        )
    )

    x = np.arange(
        len(source_plot)
    )

    values = (
        source_plot[
            "HR@10"
        ].to_numpy()
        * 100.0
    )

    fig, ax = plt.subplots(
        figsize=(10.5, 6.5)
    )

    bars = ax.bar(
        x,
        values,
    )

    tick_labels = [
        (
            f"{key}\n"
            f"N={int(source_plot.loc[key, 'n']):,}"
        )
        for key in source_plot.index
    ]

    ax.set_xticks(
        x,
        tick_labels,
        rotation=10,
        ha="right",
    )

    ax.set_ylabel(
        "HR@10 (%)"
    )

    ax.set_title(
        "Cold-start startup performance by structural source class\n"
        "New-to-investor, warm-investor cases"
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
        )

    fig.tight_layout()

    figure_paths.extend(
        save_figure(
            fig,
            (
                "phase_7_5b_fig4_"
                "cold_startup_source_class_HR10_V1"
            ),
        )
    )

    # =========================================================================
    # 7.5B.15 — Provenance manifest / hashes
    # =========================================================================

    manifest_path = (
        OUT_DIR
        / "phase_7_5b_analysis_manifest_V1.json"
    )

    manifest = {
        "phase":
            "7.5B",

        "schema_version":
            "ITRS_PHASE7_5B_STRUCTURAL_PERFORMANCE_V1",

        "status":
            "COMPLETE",

        "scientific_role":
            (
                "Pre-registered post-hoc structural "
                "coverage-performance association diagnostic."
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
                    PHASE_7_5A_MANIFEST.relative_to(
                        REPO_ROOT
                    )
                ),

            "comparison_contract":
                str(
                    PHASE_7_5A_PREREG.relative_to(
                        REPO_ROOT
                    )
                ),

            "outcomes_inspected_before_preregistration":
                False,
        },

        "primary_comparison":
            (
                "new_wc_startup_connected_vs_isolated"
            ),

        "minimum_group_n_for_formal_bootstrap":
            MIN_FORMAL_GROUP_N,

        "uncertainty": {
            "method":
                (
                    "Percentile bootstrap with "
                    "investor_id as cluster unit."
                ),

            "repetitions":
                BOOTSTRAP_REPS,

            "base_seed":
                BOOTSTRAP_SEED,
        },

        "interpretation_boundary":
            (
                "Structural-performance associations are "
                "observational. They do not establish causal "
                "rescue or validate a future architecture."
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
            "raw_logits_loaded": False,
            "inference_executed": False,
            "test_rescored": False,
            "negative_resampling": False,
            "candidate_modification": False,
            "model_selection": False,
        },
    }

    json_dump(
        manifest,
        manifest_path,
    )

    artifact_paths = [
        overall_path,
        new_joint_path,
        new_ww_path,
        contrasts_path,
        source_path,
        dual_joint_path,
        bootstrap_reps_path,
        bootstrap_summary_path,
        cdf_path,
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
        / "phase_7_5b_derived_artifact_sha256_V1.json"
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
        "PHASE 7.5B V1 RESULT"
    )

    print("Frozen source binding:                 PASS")
    print("Outcome-blind preregistration gate:    PASS")
    print("Overall structural diagnostic:         COMPLETE")
    print("New-pair structural diagnostic:        COMPLETE")
    print("New warm/warm structural diagnostic:   COMPLETE")
    print("Cold-start startup primary comparison: COMPLETE")
    print("Investor-connectivity controlled checks:COMPLETE")
    print("Cold-investor small-N diagnostic:      COMPLETE")
    print("Dual-cold structural diagnostic:       COMPLETE")
    print("Structural source-class diagnostic:    COMPLETE")
    print("Investor-cluster uncertainty:          COMPLETE")
    print("Presentation-ready figures:            COMPLETE")
    print("Provenance manifest:                   COMPLETE")

    print()
    print("Model inference executed:              NO")
    print("Final test rescored:                   NO")
    print("Final test modified:                   NO")

    print()
    print(
        "PHASE 7.5B STRUCTURAL COVERAGE "
        "PERFORMANCE STATUS: COMPLETE"
    )


if __name__ == "__main__":
    main()