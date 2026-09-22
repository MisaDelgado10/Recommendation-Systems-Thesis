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
# Phase 7.6A — Temporal History Support Composition Audit V1
#
# Purpose
# -------
# Audit the actual pre-T60 temporal-history regimes available in the frozen
# final test BEFORE examining their relationship with recommendation outcomes.
#
# This phase is deliberately outcome-blind.
#
# We use only audited history indicators already present in the immutable
# Phase-6 analysis bundle:
#
#   investor_seen_before_t60
#   investor_seen_in_t1_t59
#   investor_seen_in_t0
#   investor_t0_only_history
#
#   startup_seen_before_t60
#   startup_seen_in_t1_t59
#   startup_seen_in_t0
#   startup_t0_only_history
#
# Literal history classes
# -----------------------
#   cold
#       not seen in T0 and not seen in T1-T59
#
#   t0_only
#       seen in T0, not seen in T1-T59
#
#   t1_t59_only
#       seen in T1-T59, not seen in T0
#
#   t0_and_t1_t59
#       seen in both T0 and T1-T59
#
# IMPORTANT:
# We intentionally use literal temporal labels rather than terms such as
# "long history", "recent history", or "history depth", because those terms
# would add interpretation beyond the frozen indicator definitions.
#
# Scientific boundary
# -------------------
# - immutable Phase-6 final-test cases
# - no positive_rank loaded
# - no HR/NDCG loaded
# - no performance analysis
# - no model/checkpoint/logit loading
# - no inference
# - no rescoring
# - no model selection
#
# Phase 7.6B comparisons are frozen here before outcome inspection.
# =============================================================================


REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    REPO_ROOT
    / "data/experimental/phase_6/full_training/100pct/final_test/"
    / "analysis_ready_test_cases.parquet"
)

PHASE_7_0_CONTRACT = (
    REPO_ROOT
    / "data/experimental/phase_7/"
    / "phase_7_0_analysis_contract/"
    / "phase_7_0_analysis_contract_V2.json"
)

PHASE_7_5C_MANIFEST = (
    REPO_ROOT
    / "data/experimental/phase_7/"
    / "phase_7_5c_degree_performance/"
    / "phase_7_5c_analysis_manifest_V1.json"
)

OUT_DIR = (
    REPO_ROOT
    / "data/experimental/phase_7/"
    / "phase_7_6a_temporal_history_composition"
)

FIG_DIR = OUT_DIR / "figures"

EXPECTED_SOURCE_SHA256 = (
    "d70f21bff0006e094c5d567d307c0664"
    "b41a11e7250a0e69aaec610e1810132d"
)

EXPECTED_ROWS = 20_264

EXPECTED_INVESTOR_CLASSES = {
    "cold": 3_165,
    "t0_only": 12,
    "t1_t59_only": 12_493,
    "t0_and_t1_t59": 4_594,
}

EXPECTED_STARTUP_CLASSES = {
    "cold": 8_104,
    "t0_only": 30,
    "t1_t59_only": 11_994,
    "t0_and_t1_t59": 136,
}

EXPECTED_DIAGNOSTIC_COUNTS = {
    "prior_ww": 3_818,
    "new_ww": 6_889,
    "new_wc": 6_392,
    "new_cw": 1_453,
    "new_cc": 1_712,
}

HISTORY_ORDER = [
    "cold",
    "t0_only",
    "t1_t59_only",
    "t0_and_t1_t59",
]

DIAGNOSTIC_ORDER = [
    "prior_ww",
    "new_ww",
    "new_wc",
    "new_cw",
    "new_cc",
]

DIAGNOSTIC_LABELS = {
    "prior_ww":
        "Prior pair | Warm investor + Warm startup",

    "new_ww":
        "New pair | Warm investor + Warm startup",

    "new_wc":
        "New pair | Warm investor + Cold startup",

    "new_cw":
        "New pair | Cold investor + Warm startup",

    "new_cc":
        "New pair | Cold investor + Cold startup",
}

MIN_FORMAL_GROUP_N = 100

SOURCE_COLUMNS = [
    "interaction_id",
    "new_to_investor_pair",
    "cold_start_investor",
    "cold_start_startup",
    "investor_seen_before_t60",
    "investor_seen_in_t1_t59",
    "investor_seen_in_t0",
    "investor_t0_only_history",
    "startup_seen_before_t60",
    "startup_seen_in_t1_t59",
    "startup_seen_in_t0",
    "startup_t0_only_history",
    "pair_seen_before_t60",
    "pair_seen_in_t1_t59",
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


def build_diagnostic_group(
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


def class_count_table(
    series: pd.Series,
    expected: dict[str, int],
) -> pd.DataFrame:
    table = (
        series.value_counts()
        .reindex(
            HISTORY_ORDER,
            fill_value=0,
        )
        .rename_axis(
            "history_class"
        )
        .reset_index(
            name="n"
        )
    )

    table[
        "share"
    ] = (
        table["n"]
        / table["n"].sum()
    )

    for _, row in (
        table.iterrows()
    ):
        key = str(
            row[
                "history_class"
            ]
        )

        require(
            int(
                row["n"]
            )
            == expected[key],
            (
                f"History-class count drift "
                f"for {key}: "
                f"{int(row['n'])} != "
                f"{expected[key]}"
            ),
        )

    return table


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    print("=" * 110)
    print(
        "PHASE 7.6A — TEMPORAL HISTORY "
        "SUPPORT COMPOSITION AUDIT V1"
    )
    print("=" * 110)

    print(
        "Scientific role:             "
        "OUTCOME-BLIND TEMPORAL HISTORY AUDIT"
    )
    print("Performance columns loaded:  NO")
    print("Performance analyzed:        NO")
    print("Model loaded:                NO")
    print("Checkpoint loaded:           NO")
    print("Inference executed:          NO")
    print("Test rescored:               NO")
    print("Model selection performed:   NO")

    # =========================================================================
    # 7.6A.1 — Integrity gate
    # =========================================================================

    print_section(
        "7.6A.1 — ANALYSIS INTEGRITY GATE"
    )

    for path in (
        SOURCE,
        PHASE_7_0_CONTRACT,
        PHASE_7_5C_MANIFEST,
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

    with PHASE_7_0_CONTRACT.open(
        "r",
        encoding="utf-8",
    ) as f:
        phase_7_0 = json.load(f)

    with PHASE_7_5C_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as f:
        phase_7_5c = json.load(f)

    require(
        phase_7_0["status"] == "PASS",
        "Phase 7.0 contract is not PASS.",
    )

    require(
        phase_7_5c["status"] == "COMPLETE",
        "Phase 7.5C is not COMPLETE.",
    )

    print()
    print("Frozen source fingerprint:   PASS")
    print("Phase 7.0 prerequisite:       PASS")
    print("Phase 7.5C prerequisite:      PASS")

    # =========================================================================
    # 7.6A.2 — Load outcome-blind history fields
    # =========================================================================

    print_section(
        "7.6A.2 — LOAD OUTCOME-BLIND HISTORY VARIABLES"
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

    forbidden_outcomes = {
        "positive_rank",
        "HR@10",
        "NDCG@10",
    }

    require(
        forbidden_outcomes.isdisjoint(
            df.columns
        ),
        (
            "Outcome-blind history audit "
            "loaded performance columns."
        ),
    )

    require(
        df.notna().all().all(),
        "History audit fields contain missing values.",
    )

    print(
        f"Test cases:                   "
        f"{len(df):,}"
    )
    print(
        f"Columns loaded:               "
        f"{len(df.columns)}"
    )
    print(
        "Performance columns excluded: PASS"
    )

    # =========================================================================
    # 7.6A.3 — Bind history semantics
    # =========================================================================

    print_section(
        "7.6A.3 — TEMPORAL HISTORY SEMANTIC BINDING"
    )

    investor_before_reconstructed = (
        df[
            "investor_seen_in_t0"
        ].astype(bool)
        |
        df[
            "investor_seen_in_t1_t59"
        ].astype(bool)
    )

    startup_before_reconstructed = (
        df[
            "startup_seen_in_t0"
        ].astype(bool)
        |
        df[
            "startup_seen_in_t1_t59"
        ].astype(bool)
    )

    require(
        (
            investor_before_reconstructed
            ==
            df[
                "investor_seen_before_t60"
            ].astype(bool)
        ).all(),
        (
            "Investor seen-before-T60 does not equal "
            "T0 OR T1-T59."
        ),
    )

    require(
        (
            startup_before_reconstructed
            ==
            df[
                "startup_seen_before_t60"
            ].astype(bool)
        ).all(),
        (
            "Startup seen-before-T60 does not equal "
            "T0 OR T1-T59."
        ),
    )

    investor_t0_only_reconstructed = (
        df[
            "investor_seen_in_t0"
        ].astype(bool)
        &
        ~df[
            "investor_seen_in_t1_t59"
        ].astype(bool)
    )

    startup_t0_only_reconstructed = (
        df[
            "startup_seen_in_t0"
        ].astype(bool)
        &
        ~df[
            "startup_seen_in_t1_t59"
        ].astype(bool)
    )

    require(
        (
            investor_t0_only_reconstructed
            ==
            df[
                "investor_t0_only_history"
            ].astype(bool)
        ).all(),
        "Investor T0-only semantic binding failed.",
    )

    require(
        (
            startup_t0_only_reconstructed
            ==
            df[
                "startup_t0_only_history"
            ].astype(bool)
        ).all(),
        "Startup T0-only semantic binding failed.",
    )

    require(
        (
            df[
                "cold_start_investor"
            ].astype(bool)
            ==
            ~df[
                "investor_seen_before_t60"
            ].astype(bool)
        ).all(),
        "Investor cold-start semantic binding failed.",
    )

    require(
        (
            df[
                "cold_start_startup"
            ].astype(bool)
            ==
            ~df[
                "startup_seen_before_t60"
            ].astype(bool)
        ).all(),
        "Startup cold-start semantic binding failed.",
    )

    print(
        "Investor pre-T60 = T0 OR T1-T59: PASS"
    )
    print(
        "Startup pre-T60 = T0 OR T1-T59:  PASS"
    )
    print(
        "Investor T0-only binding:         PASS"
    )
    print(
        "Startup T0-only binding:          PASS"
    )
    print(
        "Cold-start/history binding:       PASS"
    )

    # =========================================================================
    # 7.6A.4 — Construct literal history classes
    # =========================================================================

    print_section(
        "7.6A.4 — CONSTRUCT LITERAL HISTORY CLASSES"
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

    require(
        ~df[
            "investor_history_class"
        ].eq("INVALID").any(),
        "Invalid investor history class.",
    )

    require(
        ~df[
            "startup_history_class"
        ].eq("INVALID").any(),
        "Invalid startup history class.",
    )

    investor_counts = class_count_table(
        df[
            "investor_history_class"
        ],
        EXPECTED_INVESTOR_CLASSES,
    )

    startup_counts = class_count_table(
        df[
            "startup_history_class"
        ],
        EXPECTED_STARTUP_CLASSES,
    )

    print(
        "INVESTOR HISTORY CLASSES"
    )

    print(
        investor_counts.to_string(
            index=False,
            formatters={
                "share":
                    lambda x: f"{x:.2%}",
            },
        )
    )

    print()
    print(
        "STARTUP HISTORY CLASSES"
    )

    print(
        startup_counts.to_string(
            index=False,
            formatters={
                "share":
                    lambda x: f"{x:.2%}",
            },
        )
    )

    # =========================================================================
    # 7.6A.5 — History classes by thesis diagnostic group
    # =========================================================================

    print_section(
        "7.6A.5 — HISTORY SUPPORT BY NOVELTY × COLD-START GROUP"
    )

    df[
        "diagnostic_group"
    ] = build_diagnostic_group(
        df
    )

    require(
        ~df[
            "diagnostic_group"
        ].eq("INVALID").any(),
        "Invalid diagnostic group.",
    )

    diagnostic_counts = (
        df[
            "diagnostic_group"
        ]
        .value_counts()
    )

    for key, expected in (
        EXPECTED_DIAGNOSTIC_COUNTS.items()
    ):
        require(
            int(
                diagnostic_counts[
                    key
                ]
            )
            == expected,
            f"Diagnostic-group count drift for {key}.",
        )

    investor_by_group = (
        df.groupby(
            [
                "diagnostic_group",
                "investor_history_class",
            ],
            observed=False,
        )
        .size()
        .rename("n")
        .reset_index()
    )

    startup_by_group = (
        df.groupby(
            [
                "diagnostic_group",
                "startup_history_class",
            ],
            observed=False,
        )
        .size()
        .rename("n")
        .reset_index()
    )

    diagnostic_sizes = (
        df[
            "diagnostic_group"
        ]
        .value_counts()
    )

    investor_by_group[
        "within_group_share"
    ] = (
        investor_by_group["n"]
        / investor_by_group[
            "diagnostic_group"
        ].map(
            diagnostic_sizes
        )
    )

    startup_by_group[
        "within_group_share"
    ] = (
        startup_by_group["n"]
        / startup_by_group[
            "diagnostic_group"
        ].map(
            diagnostic_sizes
        )
    )

    investor_by_group[
        "diagnostic_label"
    ] = (
        investor_by_group[
            "diagnostic_group"
        ].map(
            DIAGNOSTIC_LABELS
        )
    )

    startup_by_group[
        "diagnostic_label"
    ] = (
        startup_by_group[
            "diagnostic_group"
        ].map(
            DIAGNOSTIC_LABELS
        )
    )

    diag_order = {
        key: i
        for i, key in enumerate(
            DIAGNOSTIC_ORDER
        )
    }

    history_order = {
        key: i
        for i, key in enumerate(
            HISTORY_ORDER
        )
    }

    for table, class_col in (
        (
            investor_by_group,
            "investor_history_class",
        ),
        (
            startup_by_group,
            "startup_history_class",
        ),
    ):
        table[
            "_diag_order"
        ] = (
            table[
                "diagnostic_group"
            ].map(
                diag_order
            )
        )

        table[
            "_history_order"
        ] = (
            table[
                class_col
            ].map(
                history_order
            )
        )

        table.sort_values(
            [
                "_diag_order",
                "_history_order",
            ],
            inplace=True,
        )

        table.drop(
            columns=[
                "_diag_order",
                "_history_order",
            ],
            inplace=True,
        )

        table.reset_index(
            drop=True,
            inplace=True,
        )

    print(
        "INVESTOR HISTORY BY DIAGNOSTIC GROUP"
    )

    print(
        investor_by_group[
            [
                "diagnostic_label",
                "investor_history_class",
                "n",
                "within_group_share",
            ]
        ].to_string(
            index=False,
            formatters={
                "within_group_share":
                    lambda x: f"{x:.2%}",
            },
        )
    )

    print()
    print(
        "STARTUP HISTORY BY DIAGNOSTIC GROUP"
    )

    print(
        startup_by_group[
            [
                "diagnostic_label",
                "startup_history_class",
                "n",
                "within_group_share",
            ]
        ].to_string(
            index=False,
            formatters={
                "within_group_share":
                    lambda x: f"{x:.2%}",
            },
        )
    )

    # =========================================================================
    # 7.6A.6 — New warm/warm joint history matrix
    # =========================================================================

    print_section(
        "7.6A.6 — NEW WARM/WARM JOINT TEMPORAL-HISTORY SUPPORT"
    )

    new_ww = df.loc[
        df[
            "diagnostic_group"
        ]
        == "new_ww"
    ].copy()

    require(
        len(new_ww) == 6_889,
        "new_ww count drift.",
    )

    require(
        ~new_ww[
            "investor_history_class"
        ].eq("cold").any(),
        (
            "new_ww contains a cold investor "
            "history class."
        ),
    )

    require(
        ~new_ww[
            "startup_history_class"
        ].eq("cold").any(),
        (
            "new_ww contains a cold startup "
            "history class."
        ),
    )

    joint_history = (
        new_ww.groupby(
            [
                "investor_history_class",
                "startup_history_class",
            ],
            observed=False,
        )
        .size()
        .rename("n")
        .reset_index()
    )

    joint_history[
        "share_of_new_ww"
    ] = (
        joint_history["n"]
        / len(new_ww)
    )

    joint_history[
        "_investor_order"
    ] = (
        joint_history[
            "investor_history_class"
        ].map(
            history_order
        )
    )

    joint_history[
        "_startup_order"
    ] = (
        joint_history[
            "startup_history_class"
        ].map(
            history_order
        )
    )

    joint_history = (
        joint_history
        .sort_values(
            [
                "_investor_order",
                "_startup_order",
            ]
        )
        .drop(
            columns=[
                "_investor_order",
                "_startup_order",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    print(
        joint_history.to_string(
            index=False,
            formatters={
                "share_of_new_ww":
                    lambda x: f"{x:.2%}",
            },
        )
    )

    # =========================================================================
    # 7.6A.7 — Pair-history support audit
    # =========================================================================

    print_section(
        "7.6A.7 — PAIR-HISTORY SUPPORT AUDIT"
    )

    pair_before_n = int(
        df[
            "pair_seen_before_t60"
        ].sum()
    )

    pair_t1_t59_n = int(
        df[
            "pair_seen_in_t1_t59"
        ].sum()
    )

    pair_before_not_t1_t59_n = int(
        (
            df[
                "pair_seen_before_t60"
            ].astype(bool)
            &
            ~df[
                "pair_seen_in_t1_t59"
            ].astype(bool)
        ).sum()
    )

    require(
        (
            df[
                "pair_seen_in_t1_t59"
            ].astype(bool)
            <=
            df[
                "pair_seen_before_t60"
            ].astype(bool)
        ).all(),
        (
            "pair_seen_in_t1_t59 is not a subset "
            "of pair_seen_before_t60."
        ),
    )

    print(
        f"Pair seen before T60:         "
        f"{pair_before_n:,}"
    )

    print(
        f"Pair seen in T1-T59:          "
        f"{pair_t1_t59_n:,}"
    )

    print(
        f"Pair seen before T60 but "
        f"not T1-T59:                  "
        f"{pair_before_not_t1_t59_n:,}"
    )

    print()
    print(
        "No pair-level T0 indicator is available in "
        "the frozen analysis bundle, so the residual "
        "case(s) are not assigned a more specific "
        "pair-history class."
    )

    # =========================================================================
    # 7.6A.8 — Pre-register Phase 7.6B comparisons
    # =========================================================================

    print_section(
        "7.6A.8 — PRE-REGISTER PHASE 7.6B COMPARISONS"
    )

    planned_comparisons = {
        "new_ww_investor_temporal_coverage":
            (
                "Within new-to-investor warm/warm cases, compare "
                "investors seen in both T0 and T1-T59 versus "
                "investors seen in T1-T59 only. T0-only investors "
                "remain descriptive because of tiny N."
            ),

        "new_ww_startup_temporal_coverage":
            (
                "Within new-to-investor warm/warm cases, compare "
                "startups seen in both T0 and T1-T59 versus "
                "startups seen in T1-T59 only when both groups "
                "meet the pre-specified minimum N. T0-only startups "
                "remain descriptive."
            ),

        "new_ww_investor_temporal_coverage_control_startup_class":
            (
                "Repeat the investor comparison within a fixed "
                "startup history class where both compared groups "
                "meet the minimum N."
            ),

        "new_ww_startup_temporal_coverage_control_investor_class":
            (
                "Repeat the startup comparison within a fixed "
                "investor history class where both compared groups "
                "meet the minimum N."
            ),

        "new_ww_joint_temporal_support":
            (
                "Describe the joint investor/startup temporal-history "
                "matrix and compare sufficiently large cells using "
                "the pre-specified minimum-N rule; do not infer from "
                "tiny T0-only cells."
            ),

        "pair_history_descriptive_only":
            (
                "Pair-level T1-T59 history is reported descriptively. "
                "No T0 pair indicator exists in the frozen bundle, "
                "so no fabricated pair-history decomposition is allowed."
            ),

        "minimum_group_n_for_formal_bootstrap":
            MIN_FORMAL_GROUP_N,
    }

    for key, description in (
        planned_comparisons.items()
    ):
        print(
            f"{key}"
        )
        print(
            f"  {description}"
        )

    # =========================================================================
    # 7.6A.9 — Write artifacts
    # =========================================================================

    print_section(
        "7.6A.9 — WRITE AUDIT ARTIFACTS"
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    investor_counts_path = (
        OUT_DIR
        / "phase_7_6a_investor_history_classes_V1.csv"
    )

    startup_counts_path = (
        OUT_DIR
        / "phase_7_6a_startup_history_classes_V1.csv"
    )

    investor_group_path = (
        OUT_DIR
        / "phase_7_6a_investor_history_by_diagnostic_group_V1.csv"
    )

    startup_group_path = (
        OUT_DIR
        / "phase_7_6a_startup_history_by_diagnostic_group_V1.csv"
    )

    joint_path = (
        OUT_DIR
        / "phase_7_6a_new_ww_joint_history_matrix_V1.csv"
    )

    pair_audit_path = (
        OUT_DIR
        / "phase_7_6a_pair_history_support_V1.json"
    )

    prereg_path = (
        OUT_DIR
        / "phase_7_6a_preregistered_7_6b_comparisons_V1.json"
    )

    investor_counts.to_csv(
        investor_counts_path,
        index=False,
    )

    startup_counts.to_csv(
        startup_counts_path,
        index=False,
    )

    investor_by_group.to_csv(
        investor_group_path,
        index=False,
    )

    startup_by_group.to_csv(
        startup_group_path,
        index=False,
    )

    joint_history.to_csv(
        joint_path,
        index=False,
    )

    pair_audit = {
        "pair_seen_before_t60_n":
            pair_before_n,

        "pair_seen_in_t1_t59_n":
            pair_t1_t59_n,

        "pair_seen_before_t60_not_t1_t59_n":
            pair_before_not_t1_t59_n,

        "limitation":
            (
                "No pair_seen_in_t0 field exists in "
                "the frozen analysis bundle."
            ),
    }

    json_dump(
        pair_audit,
        pair_audit_path,
    )

    json_dump(
        planned_comparisons,
        prereg_path,
    )

    # =========================================================================
    # 7.6A.10 — Figures
    # =========================================================================

    print_section(
        "7.6A.10 — GENERATE PRESENTATION-READY FIGURES"
    )

    figure_paths = []

    # Figure 1 — overall history-class composition
    fig, ax = plt.subplots(
        figsize=(10.5, 6.5)
    )

    x = np.arange(
        len(HISTORY_ORDER)
    )

    width = 0.36

    investor_values = (
        investor_counts[
            "share"
        ].to_numpy()
        * 100.0
    )

    startup_values = (
        startup_counts[
            "share"
        ].to_numpy()
        * 100.0
    )

    ax.bar(
        x - width / 2,
        investor_values,
        width=width,
        label="Investor",
    )

    ax.bar(
        x + width / 2,
        startup_values,
        width=width,
        label="Startup",
    )

    ax.set_xticks(
        x,
        HISTORY_ORDER,
        rotation=15,
        ha="right",
    )

    ax.set_ylabel(
        "Share of final-test cases (%)"
    )

    ax.set_title(
        "Literal pre-T60 temporal-history classes"
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
                "phase_7_6a_fig1_"
                "overall_history_classes_V1"
            ),
        )
    )

    # Figure 2 — investor history by diagnostic group
    investor_pivot = (
        investor_by_group.pivot(
            index="diagnostic_group",
            columns="investor_history_class",
            values="within_group_share",
        )
        .reindex(
            DIAGNOSTIC_ORDER
        )
        .reindex(
            columns=HISTORY_ORDER
        )
        .fillna(0.0)
        * 100.0
    )

    fig, ax = plt.subplots(
        figsize=(13, 7)
    )

    x = np.arange(
        len(investor_pivot)
    )

    bottom = np.zeros(
        len(investor_pivot)
    )

    for history_class in HISTORY_ORDER:
        values = (
            investor_pivot[
                history_class
            ].to_numpy()
        )

        ax.bar(
            x,
            values,
            bottom=bottom,
            label=history_class,
        )

        bottom += values

    ax.set_xticks(
        x,
        [
            "Prior\nwarm/warm",
            "New\nwarm/warm",
            "New\nwarm investor\ncold startup",
            "New\ncold investor\nwarm startup",
            "New\ndual cold",
        ],
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.set_ylabel(
        "Share of diagnostic group (%)"
    )

    ax.set_title(
        "Investor temporal-history composition across thesis regimes"
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
                "phase_7_6a_fig2_"
                "investor_history_by_diagnostic_group_V1"
            ),
        )
    )

    # Figure 3 — startup history by diagnostic group
    startup_pivot = (
        startup_by_group.pivot(
            index="diagnostic_group",
            columns="startup_history_class",
            values="within_group_share",
        )
        .reindex(
            DIAGNOSTIC_ORDER
        )
        .reindex(
            columns=HISTORY_ORDER
        )
        .fillna(0.0)
        * 100.0
    )

    fig, ax = plt.subplots(
        figsize=(13, 7)
    )

    x = np.arange(
        len(startup_pivot)
    )

    bottom = np.zeros(
        len(startup_pivot)
    )

    for history_class in HISTORY_ORDER:
        values = (
            startup_pivot[
                history_class
            ].to_numpy()
        )

        ax.bar(
            x,
            values,
            bottom=bottom,
            label=history_class,
        )

        bottom += values

    ax.set_xticks(
        x,
        [
            "Prior\nwarm/warm",
            "New\nwarm/warm",
            "New\nwarm investor\ncold startup",
            "New\ncold investor\nwarm startup",
            "New\ndual cold",
        ],
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.set_ylabel(
        "Share of diagnostic group (%)"
    )

    ax.set_title(
        "Startup temporal-history composition across thesis regimes"
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
                "phase_7_6a_fig3_"
                "startup_history_by_diagnostic_group_V1"
            ),
        )
    )

    # Figure 4 — joint history matrix for new warm/warm
    joint_pivot = (
        joint_history.pivot(
            index="investor_history_class",
            columns="startup_history_class",
            values="n",
        )
        .reindex(
            HISTORY_ORDER
        )
        .reindex(
            columns=HISTORY_ORDER
        )
        .fillna(0)
    )

    fig, ax = plt.subplots(
        figsize=(8.5, 7)
    )

    image = ax.imshow(
        joint_pivot.to_numpy(
            dtype=float
        ),
        aspect="auto",
    )

    ax.set_xticks(
        np.arange(
            len(HISTORY_ORDER)
        ),
        HISTORY_ORDER,
        rotation=20,
        ha="right",
    )

    ax.set_yticks(
        np.arange(
            len(HISTORY_ORDER)
        ),
        HISTORY_ORDER,
    )

    ax.set_xlabel(
        "Startup history class"
    )

    ax.set_ylabel(
        "Investor history class"
    )

    ax.set_title(
        "New warm/warm joint temporal-history support"
    )

    for i in range(
        joint_pivot.shape[0]
    ):
        for j in range(
            joint_pivot.shape[1]
        ):
            value = int(
                joint_pivot.iloc[
                    i,
                    j,
                ]
            )

            ax.text(
                j,
                i,
                f"{value:,}",
                ha="center",
                va="center",
            )

    fig.colorbar(
        image,
        ax=ax,
        label="Cases",
    )

    fig.tight_layout()

    figure_paths.extend(
        save_figure(
            fig,
            (
                "phase_7_6a_fig4_"
                "new_ww_joint_history_matrix_V1"
            ),
        )
    )

    # =========================================================================
    # 7.6A.11 — Manifest / hashes
    # =========================================================================

    manifest_path = (
        OUT_DIR
        / "phase_7_6a_analysis_manifest_V1.json"
    )

    manifest = {
        "phase":
            "7.6A",

        "schema_version":
            "ITRS_PHASE7_6A_TEMPORAL_HISTORY_COMPOSITION_V1",

        "status":
            "COMPLETE",

        "scientific_role":
            (
                "Outcome-blind audit of literal pre-T60 "
                "temporal-history support before "
                "performance comparisons."
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

            "columns_loaded":
                SOURCE_COLUMNS,
        },

        "outcome_blindness": {
            "positive_rank_loaded":
                False,

            "HR@10_loaded":
                False,

            "NDCG@10_loaded":
                False,

            "performance_metrics_examined":
                False,
        },

        "history_class_definition": {
            "cold":
                "not seen in T0 or T1-T59",

            "t0_only":
                "seen in T0 and not T1-T59",

            "t1_t59_only":
                "seen in T1-T59 and not T0",

            "t0_and_t1_t59":
                "seen in both T0 and T1-T59",
        },

        "minimum_group_n_for_formal_bootstrap":
            MIN_FORMAL_GROUP_N,

        "phase_7_6b_comparisons_frozen_before_outcomes":
            planned_comparisons,

        "pair_history_limitation":
            (
                "No pair-level T0 indicator exists "
                "in the frozen analysis bundle."
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

            "inference_executed":
                False,

            "test_rescored":
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
        investor_counts_path,
        startup_counts_path,
        investor_group_path,
        startup_group_path,
        joint_path,
        pair_audit_path,
        prereg_path,
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
        / "phase_7_6a_derived_artifact_sha256_V1.json"
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
        "PHASE 7.6A V1 RESULT"
    )

    print("Frozen source binding:             PASS")
    print("Outcome-blind column gate:         PASS")
    print("History semantic binding:          PASS")
    print("Investor history classes:          COMPLETE")
    print("Startup history classes:           COMPLETE")
    print("Diagnostic-group composition:      COMPLETE")
    print("New warm/warm joint matrix:        COMPLETE")
    print("Pair-history support audit:        COMPLETE")
    print("7.6B comparisons pre-registered:   COMPLETE")
    print("Presentation-ready figures:        COMPLETE")
    print("Provenance manifest:               COMPLETE")

    print()
    print("Performance outcomes inspected:    NO")
    print("Model inference executed:          NO")
    print("Final test rescored:               NO")
    print("Final test modified:               NO")

    print()
    print(
        "PHASE 7.6A TEMPORAL HISTORY "
        "SUPPORT COMPOSITION STATUS: COMPLETE"
    )


if __name__ == "__main__":
    main()