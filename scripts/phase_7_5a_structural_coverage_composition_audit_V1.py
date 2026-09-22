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
# Phase 7.5A — Structural Coverage Composition Audit V1
#
# Purpose
# -------
# Audit the availability and composition of heterogeneous graph structure
# BEFORE examining structural-performance associations.
#
# This is intentionally outcome-blind with respect to recommendation metrics.
#
# No HR/NDCG/rank performance comparison is performed here.
#
# Scientific boundary
# -------------------
# - frozen Phase-6 one-shot test only
# - no model loading
# - no inference
# - no rescoring
# - no candidate modification
# - no model selection
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

PHASE_7_4_MANIFEST = (
    REPO_ROOT
    / "data/experimental/phase_7/"
    / "phase_7_4_novelty_cold_start_interaction/"
    / "phase_7_4_analysis_manifest_V1.json"
)

OUT_DIR = (
    REPO_ROOT
    / "data/experimental/phase_7/"
    / "phase_7_5a_structural_coverage_composition"
)

FIG_DIR = OUT_DIR / "figures"

EXPECTED_SOURCE_SHA256 = (
    "d70f21bff0006e094c5d567d307c0664"
    "b41a11e7250a0e69aaec610e1810132d"
)

EXPECTED_ROWS = 20_264

EXPECTED_INVESTOR_CONNECTED = 2_225
EXPECTED_STARTUP_CONNECTED = 4_011

EXPECTED_JOINT_COUNTS = {
    "Both connected": 604,
    "Investor connected only": 1_621,
    "Startup connected only": 3_407,
    "Neither connected": 14_632,
}

DIAGNOSTIC_GROUPS = {
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
                investor_connected
                & startup_connected,

                investor_connected
                & ~startup_connected,

                ~investor_connected
                & startup_connected,

                ~investor_connected
                & ~startup_connected,
            ],
            STRUCTURAL_ORDER,
            default="INVALID",
        ),
        index=df.index,
    )


# =============================================================================
# Main
# =============================================================================


def main() -> None:

    print("=" * 110)
    print(
        "PHASE 7.5A — STRUCTURAL COVERAGE "
        "COMPOSITION AUDIT V1"
    )
    print("=" * 110)

    print(
        "Scientific role:             "
        "OUTCOME-BLIND STRUCTURAL COVERAGE AUDIT"
    )

    print("Performance analyzed:        NO")
    print("Model loaded:                NO")
    print("Checkpoint loaded:           NO")
    print("Inference executed:          NO")
    print("Test rescored:               NO")
    print("Model selection performed:   NO")

    # =========================================================================
    # 7.5A.1 — Integrity gate
    # =========================================================================

    print_section(
        "7.5A.1 — ANALYSIS INTEGRITY GATE"
    )

    for path in (
        SOURCE,
        PHASE_7_0_CONTRACT,
        PHASE_7_4_MANIFEST,
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
        "Frozen Phase-6 bundle SHA256 drift.",
    )

    with PHASE_7_0_CONTRACT.open(
        "r",
        encoding="utf-8",
    ) as f:
        phase_7_0 = json.load(f)

    with PHASE_7_4_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as f:
        phase_7_4 = json.load(f)

    require(
        phase_7_0["status"] == "PASS",
        "Phase 7.0 contract is not PASS.",
    )

    require(
        phase_7_4["status"] == "COMPLETE",
        "Phase 7.4 is not COMPLETE.",
    )

    print()
    print("Frozen source fingerprint:   PASS")
    print("Phase 7.0 prerequisite:       PASS")
    print("Phase 7.4 prerequisite:       PASS")

    # =========================================================================
    # 7.5A.2 — Load frozen cases
    # =========================================================================

    print_section(
        "7.5A.2 — LOAD STRUCTURAL VARIABLES"
    )

    df = pd.read_parquet(SOURCE)

    require(
        len(df) == EXPECTED_ROWS,
        "Final-test row-count drift.",
    )

    required = [
        "investor_core_connected",
        "startup_core_connected",
        "investor_core_total_degree",
        "startup_core_total_degree",
        "investor_structural_source_class",
        "startup_structural_source_class",
        "new_to_investor_pair",
        "cold_start_investor",
        "cold_start_startup",
    ]

    for col in required:

        require(
            col in df.columns,
            f"Missing structural field: {col}",
        )

        require(
            df[col].notna().all(),
            f"{col} contains missing values.",
        )

    print(
        f"Test cases:                   "
        f"{len(df):,}"
    )

    print("Required structural fields:   PASS")

    # =========================================================================
    # 7.5A.3 — Structural semantic binding
    # =========================================================================

    print_section(
        "7.5A.3 — STRUCTURAL SEMANTIC BINDING"
    )

    investor_connected_from_degree = (
        df["investor_core_total_degree"]
        .gt(0)
    )

    startup_connected_from_degree = (
        df["startup_core_total_degree"]
        .gt(0)
    )

    require(
        (
            investor_connected_from_degree
            ==
            df["investor_core_connected"]
            .astype(bool)
        ).all(),
        (
            "investor_core_connected != "
            "(investor_core_total_degree > 0)"
        ),
    )

    require(
        (
            startup_connected_from_degree
            ==
            df["startup_core_connected"]
            .astype(bool)
        ).all(),
        (
            "startup_core_connected != "
            "(startup_core_total_degree > 0)"
        ),
    )

    valid_source_classes = set(
        SOURCE_CLASS_ORDER
    )

    require(
        set(
            df[
                "investor_structural_source_class"
            ].unique()
        )
        <= valid_source_classes,
        "Unexpected investor structural source class.",
    )

    require(
        set(
            df[
                "startup_structural_source_class"
            ].unique()
        )
        <= valid_source_classes,
        "Unexpected startup structural source class.",
    )

    require(
        (
            df[
                "investor_structural_source_class"
            ].eq("neither")
            ==
            ~df[
                "investor_core_connected"
            ].astype(bool)
        ).all(),
        (
            "Investor source class 'neither' "
            "does not match core isolation."
        ),
    )

    require(
        (
            df[
                "startup_structural_source_class"
            ].eq("neither")
            ==
            ~df[
                "startup_core_connected"
            ].astype(bool)
        ).all(),
        (
            "Startup source class 'neither' "
            "does not match core isolation."
        ),
    )

    print(
        "Investor connected == degree>0:     PASS"
    )

    print(
        "Startup connected == degree>0:      PASS"
    )

    print(
        "Investor source-class binding:      PASS"
    )

    print(
        "Startup source-class binding:       PASS"
    )

    # =========================================================================
    # 7.5A.4 — Joint structural coverage
    # =========================================================================

    print_section(
        "7.5A.4 — JOINT STRUCTURAL COVERAGE"
    )

    df[
        "joint_structural_status"
    ] = build_joint_structural_status(
        df
    )

    require(
        ~df[
            "joint_structural_status"
        ]
        .eq("INVALID")
        .any(),
        "Invalid structural combination.",
    )

    structural_counts = (
        df[
            "joint_structural_status"
        ]
        .value_counts()
        .reindex(
            STRUCTURAL_ORDER
        )
    )

    for status in STRUCTURAL_ORDER:

        observed = int(
            structural_counts[
                status
            ]
        )

        expected = (
            EXPECTED_JOINT_COUNTS[
                status
            ]
        )

        require(
            observed == expected,
            (
                f"Structural-count drift "
                f"for {status}: "
                f"{observed} != {expected}"
            ),
        )

        print(
            f"{status:<28} "
            f"{observed:>7,} "
            f"({observed / len(df):6.2%})"
        )

    investor_connected_n = int(
        df[
            "investor_core_connected"
        ].sum()
    )

    startup_connected_n = int(
        df[
            "startup_core_connected"
        ].sum()
    )

    require(
        investor_connected_n
        == EXPECTED_INVESTOR_CONNECTED,
        "Investor structural count drift.",
    )

    require(
        startup_connected_n
        == EXPECTED_STARTUP_CONNECTED,
        "Startup structural count drift.",
    )

    # =========================================================================
    # 7.5A.5 — Structural coverage across thesis diagnostic groups
    # =========================================================================

    print_section(
        "7.5A.5 — STRUCTURAL COVERAGE BY NOVELTY × COLD-START GROUP"
    )

    df[
        "diagnostic_group"
    ] = build_diagnostic_group(
        df
    )

    require(
        ~df[
            "diagnostic_group"
        ]
        .eq("INVALID")
        .any(),
        "Invalid Phase-7.4 diagnostic group.",
    )

    structural_by_group = (
        df.groupby(
            [
                "diagnostic_group",
                "joint_structural_status",
            ],
            observed=False,
        )
        .size()
        .rename("n")
        .reset_index()
    )

    diagnostic_sizes = (
        df["diagnostic_group"]
        .value_counts()
    )

    structural_by_group[
        "within_group_share"
    ] = (
        structural_by_group["n"]
        / structural_by_group[
            "diagnostic_group"
        ].map(
            diagnostic_sizes
        )
    )

    structural_by_group[
        "diagnostic_label"
    ] = (
        structural_by_group[
            "diagnostic_group"
        ].map(
            DIAGNOSTIC_GROUPS
        )
    )

    print(
        structural_by_group[
            [
                "diagnostic_label",
                "joint_structural_status",
                "n",
                "within_group_share",
            ]
        ]
        .sort_values(
            [
                "diagnostic_group",
                "joint_structural_status",
            ]
        )
        .to_string(
            index=False,
            formatters={
                "within_group_share":
                    lambda x: f"{x:.2%}",
            },
        )
    )

    # =========================================================================
    # 7.5A.6 — Structural support specifically under cold start
    # =========================================================================

    print_section(
        "7.5A.6 — STRUCTURAL SUPPORT UNDER ENTITY COLD START"
    )

    cold_startup = df.loc[
        df[
            "cold_start_startup"
        ].astype(bool)
    ]

    cold_investor = df.loc[
        df[
            "cold_start_investor"
        ].astype(bool)
    ]

    cold_startup_connected_n = int(
        cold_startup[
            "startup_core_connected"
        ].sum()
    )

    cold_startup_isolated_n = int(
        (
            ~cold_startup[
                "startup_core_connected"
            ]
        ).sum()
    )

    cold_investor_connected_n = int(
        cold_investor[
            "investor_core_connected"
        ].sum()
    )

    cold_investor_isolated_n = int(
        (
            ~cold_investor[
                "investor_core_connected"
            ]
        ).sum()
    )

    print(
        "COLD-START STARTUPS"
    )

    print(
        f"  Structurally connected:     "
        f"{cold_startup_connected_n:,} "
        f"({cold_startup_connected_n / len(cold_startup):.2%})"
    )

    print(
        f"  Structurally isolated:      "
        f"{cold_startup_isolated_n:,} "
        f"({cold_startup_isolated_n / len(cold_startup):.2%})"
    )

    print()

    print(
        "COLD-START INVESTORS"
    )

    print(
        f"  Structurally connected:     "
        f"{cold_investor_connected_n:,} "
        f"({cold_investor_connected_n / len(cold_investor):.2%})"
    )

    print(
        f"  Structurally isolated:      "
        f"{cold_investor_isolated_n:,} "
        f"({cold_investor_isolated_n / len(cold_investor):.2%})"
    )

    # Dual cold.
    dual_cold = df.loc[
        df[
            "diagnostic_group"
        ]
        == "new_cc"
    ]

    dual_cold_structural = (
        dual_cold[
            "joint_structural_status"
        ]
        .value_counts()
        .reindex(
            STRUCTURAL_ORDER,
            fill_value=0,
        )
        .rename_axis(
            "joint_structural_status"
        )
        .reset_index(
            name="n"
        )
    )

    dual_cold_structural[
        "share"
    ] = (
        dual_cold_structural["n"]
        / len(dual_cold)
    )

    print()
    print(
        "DUAL-COLD NEW-PAIR CASES"
    )

    print(
        dual_cold_structural
        .to_string(
            index=False,
            formatters={
                "share":
                    lambda x: f"{x:.2%}",
            },
        )
    )

    # =========================================================================
    # 7.5A.7 — Structural source classes
    # =========================================================================

    print_section(
        "7.5A.7 — STRUCTURAL SOURCE CLASSES"
    )

    investor_source = (
        df[
            "investor_structural_source_class"
        ]
        .value_counts()
        .reindex(
            SOURCE_CLASS_ORDER,
            fill_value=0,
        )
        .rename_axis(
            "structural_source_class"
        )
        .reset_index(
            name="n"
        )
    )

    investor_source[
        "share"
    ] = (
        investor_source["n"]
        / len(df)
    )

    startup_source = (
        df[
            "startup_structural_source_class"
        ]
        .value_counts()
        .reindex(
            SOURCE_CLASS_ORDER,
            fill_value=0,
        )
        .rename_axis(
            "structural_source_class"
        )
        .reset_index(
            name="n"
        )
    )

    startup_source[
        "share"
    ] = (
        startup_source["n"]
        / len(df)
    )

    print(
        "INVESTOR STRUCTURAL SOURCE"
    )

    print(
        investor_source.to_string(
            index=False,
            formatters={
                "share":
                    lambda x: f"{x:.2%}",
            },
        )
    )

    print()
    print(
        "STARTUP STRUCTURAL SOURCE"
    )

    print(
        startup_source.to_string(
            index=False,
            formatters={
                "share":
                    lambda x: f"{x:.2%}",
            },
        )
    )

    # Cold-start source-class composition.
    cold_startup_source = (
        cold_startup[
            "startup_structural_source_class"
        ]
        .value_counts()
        .reindex(
            SOURCE_CLASS_ORDER,
            fill_value=0,
        )
        .rename_axis(
            "structural_source_class"
        )
        .reset_index(
            name="n"
        )
    )

    cold_startup_source[
        "share"
    ] = (
        cold_startup_source["n"]
        / len(cold_startup)
    )

    cold_investor_source = (
        cold_investor[
            "investor_structural_source_class"
        ]
        .value_counts()
        .reindex(
            SOURCE_CLASS_ORDER,
            fill_value=0,
        )
        .rename_axis(
            "structural_source_class"
        )
        .reset_index(
            name="n"
        )
    )

    cold_investor_source[
        "share"
    ] = (
        cold_investor_source["n"]
        / len(cold_investor)
    )

    print()
    print(
        "COLD-START STARTUP SOURCE CLASS"
    )

    print(
        cold_startup_source.to_string(
            index=False,
            formatters={
                "share":
                    lambda x: f"{x:.2%}",
            },
        )
    )

    print()
    print(
        "COLD-START INVESTOR SOURCE CLASS"
    )

    print(
        cold_investor_source.to_string(
            index=False,
            formatters={
                "share":
                    lambda x: f"{x:.2%}",
            },
        )
    )

    # =========================================================================
    # 7.5A.8 — Pre-register performance comparisons for 7.5B
    # =========================================================================

    print_section(
        "7.5A.8 — PRE-REGISTER PHASE 7.5B COMPARISONS"
    )

    planned_comparisons = {
        "overall_joint_structure": (
            "Compare both connected / investor-only / "
            "startup-only / neither across all test cases."
        ),

        "new_pair_joint_structure": (
            "Repeat joint structural comparison within "
            "new-to-investor cases only."
        ),

        "new_warm_warm_joint_structure": (
            "Repeat structural comparison within "
            "new-to-investor warm-investor/warm-startup "
            "cases to reduce cold-start confounding."
        ),

        "cold_startup_structural_rescue": (
            "Within new warm-investor/cold-startup cases, "
            "compare structurally connected versus isolated startups."
        ),

        "cold_investor_structural_rescue": (
            "Within new cold-investor/warm-startup cases, "
            "compare structurally connected versus isolated investors."
        ),

        "dual_cold_structural_support": (
            "Within new dual-cold cases, compare joint "
            "structural-coverage regimes where subgroup sizes permit."
        ),

        "source_class_diagnostic": (
            "Compare founder-only / acquisition-only / "
            "founder-and-acquisition / neither descriptively, "
            "subject to subgroup size."
        ),
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
    # 7.5A.9 — Write artifacts
    # =========================================================================

    print_section(
        "7.5A.9 — WRITE AUDIT ARTIFACTS"
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joint_path = (
        OUT_DIR
        / "phase_7_5a_joint_structural_counts_V1.csv"
    )

    group_path = (
        OUT_DIR
        / "phase_7_5a_structure_by_diagnostic_group_V1.csv"
    )

    dual_path = (
        OUT_DIR
        / "phase_7_5a_dual_cold_structural_coverage_V1.csv"
    )

    investor_source_path = (
        OUT_DIR
        / "phase_7_5a_investor_source_classes_V1.csv"
    )

    startup_source_path = (
        OUT_DIR
        / "phase_7_5a_startup_source_classes_V1.csv"
    )

    cold_startup_source_path = (
        OUT_DIR
        / "phase_7_5a_cold_startup_source_classes_V1.csv"
    )

    cold_investor_source_path = (
        OUT_DIR
        / "phase_7_5a_cold_investor_source_classes_V1.csv"
    )

    prereg_path = (
        OUT_DIR
        / "phase_7_5a_preregistered_7_5b_comparisons_V1.json"
    )

    joint_table = (
        structural_counts
        .rename_axis(
            "joint_structural_status"
        )
        .reset_index(
            name="n"
        )
    )

    joint_table["share"] = (
        joint_table["n"]
        / len(df)
    )

    joint_table.to_csv(
        joint_path,
        index=False,
    )

    structural_by_group.to_csv(
        group_path,
        index=False,
    )

    dual_cold_structural.to_csv(
        dual_path,
        index=False,
    )

    investor_source.to_csv(
        investor_source_path,
        index=False,
    )

    startup_source.to_csv(
        startup_source_path,
        index=False,
    )

    cold_startup_source.to_csv(
        cold_startup_source_path,
        index=False,
    )

    cold_investor_source.to_csv(
        cold_investor_source_path,
        index=False,
    )

    json_dump(
        planned_comparisons,
        prereg_path,
    )

    # =========================================================================
    # 7.5A.10 — Figures
    # =========================================================================

    print_section(
        "7.5A.10 — GENERATE PRESENTATION-READY FIGURES"
    )

    figure_paths = []

    # -------------------------------------------------------------------------
    # Figure 1 — overall structural coverage
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10.5, 6.5)
    )

    x = np.arange(
        len(joint_table)
    )

    values = (
        joint_table["share"]
        .to_numpy()
        * 100.0
    )

    bars = ax.bar(
        x,
        values,
    )

    ax.set_xticks(
        x,
        joint_table[
            "joint_structural_status"
        ],
        rotation=15,
        ha="right",
    )

    ax.set_ylabel(
        "Share of final-test cases (%)"
    )

    ax.set_title(
        "Core heterogeneous graph coverage in the frozen final test"
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
                "phase_7_5a_fig1_"
                "overall_structural_coverage_V1"
            ),
        )
    )

    # -------------------------------------------------------------------------
    # Figure 2 — structural coverage by thesis diagnostic group
    # -------------------------------------------------------------------------

    structural_pivot = (
        structural_by_group.pivot(
            index="diagnostic_group",
            columns="joint_structural_status",
            values="within_group_share",
        )
        .reindex(
            DIAGNOSTIC_ORDER
        )
        .reindex(
            columns=STRUCTURAL_ORDER
        )
        .fillna(0.0)
        * 100.0
    )

    fig, ax = plt.subplots(
        figsize=(13, 7)
    )

    x = np.arange(
        len(structural_pivot)
    )

    bottom = np.zeros(
        len(structural_pivot)
    )

    for status in STRUCTURAL_ORDER:

        values = (
            structural_pivot[
                status
            ]
            .to_numpy()
        )

        ax.bar(
            x,
            values,
            bottom=bottom,
            label=status,
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
        "Structural graph coverage across novelty and cold-start regimes"
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
                "phase_7_5a_fig2_"
                "structure_by_diagnostic_group_V1"
            ),
        )
    )

    # -------------------------------------------------------------------------
    # Figure 3 — structural source classes for cold-start startups
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10.5, 6.5)
    )

    x = np.arange(
        len(cold_startup_source)
    )

    values = (
        cold_startup_source[
            "share"
        ]
        .to_numpy()
        * 100.0
    )

    bars = ax.bar(
        x,
        values,
    )

    ax.set_xticks(
        x,
        cold_startup_source[
            "structural_source_class"
        ],
        rotation=15,
        ha="right",
    )

    ax.set_ylabel(
        "Share of cold-start startup cases (%)"
    )

    ax.set_title(
        "Where structural information for cold-start startups comes from"
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
                "phase_7_5a_fig3_"
                "cold_startup_structural_sources_V1"
            ),
        )
    )

    # =========================================================================
    # 7.5A.11 — Manifest
    # =========================================================================

    manifest_path = (
        OUT_DIR
        / "phase_7_5a_analysis_manifest_V1.json"
    )

    manifest = {
        "phase":
            "7.5A",

        "schema_version":
            "ITRS_PHASE7_5A_STRUCTURAL_COVERAGE_COMPOSITION_V1",

        "status":
            "COMPLETE",

        "scientific_role":
            (
                "Outcome-blind audit of heterogeneous "
                "graph structural coverage before "
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
        },

        "structural_semantics": {
            "investor_connected_equals_degree_gt_zero":
                True,
            "startup_connected_equals_degree_gt_zero":
                True,
            "source_class_neither_equals_isolated":
                True,
        },

        "cold_start_structural_support": {
            "cold_startup_n":
                int(
                    len(cold_startup)
                ),

            "cold_startup_connected_n":
                cold_startup_connected_n,

            "cold_startup_isolated_n":
                cold_startup_isolated_n,

            "cold_investor_n":
                int(
                    len(cold_investor)
                ),

            "cold_investor_connected_n":
                cold_investor_connected_n,

            "cold_investor_isolated_n":
                cold_investor_isolated_n,
        },

        "performance_metrics_examined":
            False,

        "phase_7_5b_comparisons_frozen_before_outcomes":
            planned_comparisons,

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

    artifact_paths = [
        joint_path,
        group_path,
        dual_path,
        investor_source_path,
        startup_source_path,
        cold_startup_source_path,
        cold_investor_source_path,
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

        hashes[relative] = (
            file_sha256(path)
        )

        print(
            f"WROTE  {relative}"
        )

    hashes_path = (
        OUT_DIR
        / "phase_7_5a_derived_artifact_sha256_V1.json"
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
        "PHASE 7.5A V1 RESULT"
    )

    print("Frozen source binding:             PASS")
    print("Connected/degree semantics:        PASS")
    print("Structural source-class binding:   PASS")
    print("Joint structural coverage audit:   COMPLETE")
    print("Cold-start structural audit:       COMPLETE")
    print("Source-class composition audit:    COMPLETE")
    print("7.5B comparisons pre-registered:   COMPLETE")
    print("Presentation-ready figures:        COMPLETE")

    print()
    print("Performance outcomes inspected:    NO")
    print("Model inference executed:          NO")
    print("Final test rescored:               NO")
    print("Final test modified:               NO")

    print()
    print(
        "PHASE 7.5A STRUCTURAL COVERAGE "
        "COMPOSITION STATUS: COMPLETE"
    )


if __name__ == "__main__":
    main()