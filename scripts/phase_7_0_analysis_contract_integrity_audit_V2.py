#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Phase 7.0 — Analysis Contract & Integrity Audit V2
#
# Scientific role:
#   Post-hoc integrity/schema audit of the immutable Phase-6 one-shot
#   final-test analysis bundle.
#
# V2 correction:
#   V1 incorrectly assumed:
#
#       test_position == case_index
#
#   Phase 6.9c defines:
#
#       validation cases = first 2,251 evaluation rows
#       test cases       = following 20,264 evaluation rows
#
#   Therefore:
#
#       case_index / matrix_row_index = 2251..22514
#       test_position                 = 0..20263
#
#       test_position == case_index - 2251
#
# Explicitly prohibited:
#   - model loading
#   - checkpoint loading
#   - inference
#   - test rescoring
#   - negative resampling
#   - candidate regeneration
#   - model selection
#   - optimizer/backward
# =============================================================================


REPO_ROOT = Path(__file__).resolve().parents[1]

BUNDLE = (
    REPO_ROOT
    / "data/experimental/phase_6/full_training/100pct/final_test/"
    / "analysis_ready_test_cases.parquet"
)

BUNDLE_MANIFEST = (
    REPO_ROOT
    / "data/experimental/phase_6/full_training/100pct/final_test/"
    / "analysis_ready_test_cases_manifest.json"
)

FINAL_TEST_SUMMARY = (
    REPO_ROOT
    / "data/experimental/phase_6/full_training/100pct/final_test/"
    / "final_test_summary.json"
)

OUT_DIR = (
    REPO_ROOT
    / "data/experimental/phase_7/phase_7_0_analysis_contract"
)


# =============================================================================
# Frozen Phase-6 constants
# =============================================================================

EXPECTED_BUNDLE_SHA256 = (
    "d70f21bff0006e094c5d567d307c0664"
    "b41a11e7250a0e69aaec610e1810132d"
)

VALIDATION_CASES = 2_251
TEST_CASES = 20_264
TOTAL_EVALUATION_CASES = 22_515

EXPECTED_COLUMNS = 74
EXPECTED_HITS_AT_10 = 11_072

EXPECTED_HR10 = 0.5463876825898144
EXPECTED_NDCG10 = 0.39870973012365235
EXPECTED_MEAN_RANK = 23.034346624555862
EXPECTED_MEDIAN_RANK = 7.0

EXPECTED_OBSERVED_SUBGROUP_COUNTS = {
    "new_to_investor": 16_446,
    "cold_start_investor": 3_165,
    "cold_start_startup": 8_104,
    "investor_core_connected": 2_225,
    "startup_core_connected": 4_011,
}

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


def grouped_count_table(
    df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:

    return (
        df.groupby(
            group_cols,
            dropna=False,
            observed=False,
        )
        .size()
        .reset_index(name="n")
        .sort_values(group_cols)
        .reset_index(drop=True)
    )


def build_schema_inventory(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for col in df.columns:

        s = df[col]

        null_count = int(s.isna().sum())

        sample_values = (
            s.dropna()
            .astype(str)
            .drop_duplicates()
            .head(5)
            .tolist()
        )

        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "rows": len(s),
                "null_count": null_count,
                "null_pct": null_count / len(s),
                "unique_non_null": int(
                    s.nunique(dropna=True)
                ),
                "sample_values": " | ".join(
                    sample_values
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================


def main() -> None:

    print("=" * 110)
    print(
        "PHASE 7.0 — ANALYSIS CONTRACT "
        "& INTEGRITY AUDIT V2"
    )
    print("=" * 110)

    print("Scientific role:             POST-HOC ANALYSIS AUDIT ONLY")
    print("Model loaded:                NO")
    print("Checkpoint loaded:           NO")
    print("Inference executed:          NO")
    print("Test rescored:               NO")
    print("Negatives regenerated:       NO")
    print("Model selection performed:   NO")
    print("Optimizer/backward:          NO")

    print()
    print(
        "V2 correction:              "
        "test_position is TEST-LOCAL;"
    )
    print(
        "                            "
        "case_index/matrix_row_index are GLOBAL evaluation rows"
    )

    # =========================================================================
    # 7.0.1 — Source existence and fingerprints
    # =========================================================================

    print_section(
        "7.0.1 — SOURCE EXISTENCE AND FINGERPRINTS"
    )

    for path in (
        BUNDLE,
        BUNDLE_MANIFEST,
        FINAL_TEST_SUMMARY,
    ):
        require(
            path.exists(),
            f"Missing required frozen input: {path}",
        )

        print(
            "FOUND  "
            f"{path.relative_to(REPO_ROOT)}"
        )

    bundle_hash = file_sha256(BUNDLE)

    print()
    print("Bundle SHA256:")
    print(f"  observed: {bundle_hash}")
    print(f"  expected: {EXPECTED_BUNDLE_SHA256}")

    require(
        bundle_hash == EXPECTED_BUNDLE_SHA256,
        "Analysis-ready test bundle SHA256 drift.",
    )

    with BUNDLE_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as f:
        manifest = json.load(f)

    with FINAL_TEST_SUMMARY.open(
        "r",
        encoding="utf-8",
    ) as f:
        final_summary = json.load(f)

    require(
        manifest["output"]["sha256"]
        == EXPECTED_BUNDLE_SHA256,
        (
            "Bundle manifest does not bind to "
            "expected analysis bundle SHA256."
        ),
    )

    require(
        manifest["rows"] == TEST_CASES,
        "Manifest test-case count drift.",
    )

    require(
        manifest["integrity"]["test_rescored"]
        is False,
        "Manifest unexpectedly reports test rescoring.",
    )

    print("Bundle fingerprint:          PASS")
    print("Manifest fingerprint bind:   PASS")
    print("Post-hoc role binding:        PASS")

    # =========================================================================
    # 7.0.2 — Load frozen bundle
    # =========================================================================

    print_section(
        "7.0.2 — LOAD FROZEN ANALYSIS BUNDLE"
    )

    df = pd.read_parquet(BUNDLE)

    print(
        f"Rows:                         "
        f"{len(df):,}"
    )

    print(
        f"Columns:                      "
        f"{len(df.columns):,}"
    )

    require(
        len(df) == TEST_CASES,
        (
            f"Expected {TEST_CASES:,} rows, "
            f"observed {len(df):,}."
        ),
    )

    require(
        len(df.columns) == EXPECTED_COLUMNS,
        (
            f"Expected {EXPECTED_COLUMNS} columns, "
            f"observed {len(df.columns)}."
        ),
    )

    require(
        list(df.columns) == manifest["columns"],
        (
            "Parquet column order/schema differs "
            "from frozen manifest."
        ),
    )

    print("Row count:                    PASS")
    print("Column count:                 PASS")
    print("Manifest schema binding:      PASS")

    # =========================================================================
    # 7.0.3 — Identifier and evaluation-coordinate integrity
    # =========================================================================

    print_section(
        "7.0.3 — IDENTIFIER AND EVALUATION-COORDINATE INTEGRITY"
    )

    require(
        df["interaction_id"].is_unique,
        "interaction_id is not unique.",
    )

    require(
        df["test_position"].is_unique,
        "test_position is not unique.",
    )

    require(
        df["case_index"].is_unique,
        "case_index is not unique.",
    )

    require(
        df["matrix_row_index"].is_unique,
        "matrix_row_index is not unique.",
    )

    # Test-local coordinate.
    expected_test_positions = list(
        range(TEST_CASES)
    )

    require(
        df["test_position"].tolist()
        == expected_test_positions,
        (
            "test_position is not exactly "
            "0..20263 in frozen test order."
        ),
    )

    # Global evaluation-matrix coordinate.
    expected_global_rows = list(
        range(
            VALIDATION_CASES,
            TOTAL_EVALUATION_CASES,
        )
    )

    require(
        df["case_index"].tolist()
        == expected_global_rows,
        (
            "case_index does not correspond to "
            "the frozen contiguous test suffix "
            "2251..22514."
        ),
    )

    require(
        df["matrix_row_index"].tolist()
        == expected_global_rows,
        (
            "matrix_row_index does not correspond "
            "to frozen test rows 2251..22514."
        ),
    )

    require(
        (
            df["case_index"]
            == df["matrix_row_index"]
        ).all(),
        (
            "case_index and matrix_row_index "
            "differ."
        ),
    )

    # Correct mapping between local and global coordinates.
    require(
        (
            df["test_position"]
            == (
                df["case_index"]
                - VALIDATION_CASES
            )
        ).all(),
        (
            "test_position != "
            "case_index - VALIDATION_CASES."
        ),
    )

    require(
        (
            df["test_position"]
            == (
                df["matrix_row_index"]
                - VALIDATION_CASES
            )
        ).all(),
        (
            "test_position != "
            "matrix_row_index - VALIDATION_CASES."
        ),
    )

    require(
        df["experiment_split"]
        .eq("test")
        .all(),
        (
            "experiment_split contains "
            "non-test observations."
        ),
    )

    require(
        df["evaluation_split"]
        .eq("test")
        .all(),
        (
            "evaluation_split contains "
            "non-test observations."
        ),
    )

    print("interaction_id unique:        PASS")
    print("test_position unique:         PASS")
    print("case_index unique:            PASS")
    print("matrix_row_index unique:      PASS")
    print("Test-local order 0..20263:    PASS")
    print("Global test rows 2251..22514: PASS")
    print("case_index == matrix row:     PASS")
    print("Local/global offset mapping:  PASS")
    print("All observations are test:    PASS")

    print()
    print(
        "Index interpretation:"
    )

    print(
        f"  validation rows:            "
        f"0..{VALIDATION_CASES - 1}"
    )

    print(
        f"  test matrix rows:           "
        f"{VALIDATION_CASES}.."
        f"{TOTAL_EVALUATION_CASES - 1}"
    )

    print(
        f"  final-test local positions: "
        f"0..{TEST_CASES - 1}"
    )

    # =========================================================================
    # 7.0.4 — Reconstruct frozen metrics from rank only
    # =========================================================================

    print_section(
        "7.0.4 — RECONSTRUCT RANK-DERIVED TEST METRICS"
    )

    require(
        df["positive_rank"].notna().all(),
        "positive_rank contains missing values.",
    )

    require(
        df["positive_rank"]
        .between(1, 100)
        .all(),
        (
            "positive_rank contains values "
            "outside 1..100."
        ),
    )

    rank = df["positive_rank"].astype(int)

    reconstructed_hr10 = (
        rank <= 10
    ).astype(int)

    reconstructed_ndcg10 = pd.Series(
        np.where(
            rank <= 10,
            1.0
            / np.log2(
                rank.to_numpy(
                    dtype=float
                )
                + 1.0
            ),
            0.0,
        ),
        index=df.index,
    )

    hr_row_match = (
        reconstructed_hr10.to_numpy()
        == df["HR@10"]
        .astype(int)
        .to_numpy()
    )

    ndcg_row_abs_error = np.abs(
        reconstructed_ndcg10.to_numpy(
            dtype=float
        )
        - df["NDCG@10"].to_numpy(
            dtype=float
        )
    )

    require(
        hr_row_match.all(),
        (
            "Stored HR@10 disagrees with "
            "positive_rank."
        ),
    )

    require(
        float(
            ndcg_row_abs_error.max()
        )
        <= FLOAT_TOL,
        (
            "Stored NDCG@10 disagrees with "
            "positive_rank."
        ),
    )

    reconstructed_hits = int(
        reconstructed_hr10.sum()
    )

    reconstructed_hr10_mean = float(
        reconstructed_hr10.mean()
    )

    reconstructed_ndcg10_mean = float(
        reconstructed_ndcg10.mean()
    )

    mean_rank = float(rank.mean())
    median_rank = float(rank.median())

    print(
        f"Hits@10:                      "
        f"{reconstructed_hits:,}"
    )

    print(
        f"HR@10:                        "
        f"{reconstructed_hr10_mean:.12f}"
    )

    print(
        f"NDCG@10:                      "
        f"{reconstructed_ndcg10_mean:.12f}"
    )

    print(
        f"Mean positive rank:           "
        f"{mean_rank:.6f}"
    )

    print(
        f"Median positive rank:         "
        f"{median_rank:.0f}"
    )

    require(
        reconstructed_hits
        == EXPECTED_HITS_AT_10,
        "Frozen Hits@10 drift.",
    )

    require(
        abs(
            reconstructed_hr10_mean
            - EXPECTED_HR10
        )
        <= FLOAT_TOL,
        "Frozen HR@10 drift.",
    )

    require(
        abs(
            reconstructed_ndcg10_mean
            - EXPECTED_NDCG10
        )
        <= FLOAT_TOL,
        "Frozen NDCG@10 drift.",
    )

    require(
        abs(
            mean_rank
            - EXPECTED_MEAN_RANK
        )
        <= FLOAT_TOL,
        "Frozen mean rank drift.",
    )

    require(
        median_rank
        == EXPECTED_MEDIAN_RANK,
        "Frozen median rank drift.",
    )

    require(
        final_summary["test"][
            "cases"
        ]
        == TEST_CASES,
        (
            "final_test_summary test-case "
            "count drift."
        ),
    )

    require(
        final_summary["test"][
            "hit_count_at_10"
        ]
        == reconstructed_hits,
        (
            "final_test_summary Hits@10 "
            "differs from bundle."
        ),
    )

    require(
        abs(
            final_summary["test"][
                "HR@10"
            ]
            - reconstructed_hr10_mean
        )
        <= FLOAT_TOL,
        (
            "final_test_summary HR@10 "
            "differs from bundle."
        ),
    )

    require(
        abs(
            final_summary["test"][
                "NDCG@10"
            ]
            - reconstructed_ndcg10_mean
        )
        <= FLOAT_TOL,
        (
            "final_test_summary NDCG@10 "
            "differs from bundle."
        ),
    )

    require(
        final_summary["integrity"][
            "test_passes_executed"
        ]
        == 1,
        (
            "Final-test execution count "
            "is not exactly one."
        ),
    )

    require(
        final_summary["integrity"][
            "model_state_unchanged"
        ]
        is True,
        "Frozen final model state changed.",
    )

    require(
        final_summary["integrity"][
            "checkpoint_reselection_performed"
        ]
        is False,
        (
            "Checkpoint reselection unexpectedly "
            "reported."
        ),
    )

    print()
    print("Row-level HR reconstruction:  PASS")
    print("Row-level NDCG reconstruction:PASS")
    print("Frozen aggregate metrics:     PASS")
    print("One-shot integrity:           PASS")

    # =========================================================================
    # 7.0.5 — Thesis-variable availability and semantic contract
    # =========================================================================

    print_section(
        "7.0.5 — THESIS SUBGROUP LOGICAL CONTRACT"
    )

    required_cols = [
        "new_to_investor_pair",
        "prior_investor_startup_relationship",
        "cold_start_investor",
        "cold_start_startup",
        "interaction_cold_start_status",
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
        "pair_repeats_within_t60",
        "t60_pair_event_count",
        "investor_core_connected",
        "startup_core_connected",
        "investor_core_total_degree",
        "startup_core_total_degree",
        "investor_structural_source_class",
        "startup_structural_source_class",
    ]

    missing_required = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    require(
        not missing_required,
        (
            "Missing required thesis-analysis "
            f"fields: {missing_required}"
        ),
    )

    semantic_boolean_cols = [
        "new_to_investor_pair",
        "prior_investor_startup_relationship",
        "cold_start_investor",
        "cold_start_startup",
        "investor_seen_before_t60",
        "startup_seen_before_t60",
        "pair_seen_before_t60",
    ]

    for col in semantic_boolean_cols:
        require(
            df[col].notna().all(),
            (
                f"Semantic field {col} "
                "contains missing values."
            ),
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
        (
            "cold_start_investor semantics "
            "disagree with "
            "investor_seen_before_t60."
        ),
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
        (
            "cold_start_startup semantics "
            "disagree with "
            "startup_seen_before_t60."
        ),
    )

    require(
        (
            df[
                "prior_investor_startup_relationship"
            ].astype(bool)
            ==
            df[
                "pair_seen_before_t60"
            ].astype(bool)
        ).all(),
        (
            "prior relationship alias "
            "disagrees with "
            "pair_seen_before_t60."
        ),
    )

    require(
        (
            df[
                "new_to_investor_pair"
            ].astype(bool)
            ==
            ~df[
                "pair_seen_before_t60"
            ].astype(bool)
        ).all(),
        (
            "new_to_investor_pair disagrees "
            "with inverse pair_seen_before_t60."
        ),
    )

    require(
        (
            df[
                "new_to_investor_pair"
            ].astype(bool)
            ==
            ~df[
                "prior_investor_startup_relationship"
            ].astype(bool)
        ).all(),
        (
            "New/repeat pair fields are "
            "not exact complements."
        ),
    )

    print("Cold-investor alias:          PASS")
    print("Cold-startup alias:           PASS")
    print("Prior-pair alias:             PASS")
    print("New-pair complement:          PASS")

    # =========================================================================
    # Diagnostic labels
    # =========================================================================

    audit = df.copy()

    audit["pair_regime"] = np.where(
        audit[
            "new_to_investor_pair"
        ].astype(bool),
        "new_to_investor",
        "prior_relationship",
    )

    audit["investor_temperature"] = (
        np.where(
            audit[
                "cold_start_investor"
            ].astype(bool),
            "cold_investor",
            "warm_investor",
        )
    )

    audit["startup_temperature"] = (
        np.where(
            audit[
                "cold_start_startup"
            ].astype(bool),
            "cold_startup",
            "warm_startup",
        )
    )

    audit["structural_joint_status"] = (
        pd.Series(
            np.where(
                audit[
                    "investor_core_connected"
                ].astype(bool),
                "investor_connected",
                "investor_isolated",
            ),
            index=audit.index,
        )
        + "__"
        + pd.Series(
            np.where(
                audit[
                    "startup_core_connected"
                ].astype(bool),
                "startup_connected",
                "startup_isolated",
            ),
            index=audit.index,
        )
    )

    # =========================================================================
    # 7.0.6 — Logical subgroup cross-tabs
    # =========================================================================

    print_section(
        "7.0.6 — OBSERVED THESIS SUBGROUP COMBINATIONS"
    )

    pair_cold_cross = grouped_count_table(
        audit,
        [
            "pair_regime",
            "investor_temperature",
            "startup_temperature",
        ],
    )

    interaction_status_cross = grouped_count_table(
        audit,
        [
            "pair_regime",
            "interaction_cold_start_status",
        ],
    )

    structural_cross = grouped_count_table(
        audit,
        [
            "structural_joint_status",
        ],
    )

    print(
        "PAIR NOVELTY × INVESTOR COLD START "
        "× STARTUP COLD START"
    )
    print("-" * 110)

    print(
        pair_cold_cross.to_string(
            index=False
        )
    )

    print()
    print(
        "PAIR NOVELTY × FROZEN "
        "INTERACTION COLD-START STATUS"
    )
    print("-" * 110)

    print(
        interaction_status_cross.to_string(
            index=False
        )
    )

    print()
    print(
        "JOINT CORE STRUCTURAL COVERAGE"
    )
    print("-" * 110)

    print(
        structural_cross.to_string(
            index=False
        )
    )

    # =========================================================================
    # 7.0.7 — Basic subgroup counts
    # =========================================================================

    print_section(
        "7.0.7 — BASIC SUBGROUP COUNTS"
    )

    subgroup_counts = {
        "test_cases": int(len(df)),
        "new_to_investor": int(
            df[
                "new_to_investor_pair"
            ].astype(bool).sum()
        ),
        "prior_relationship": int(
            df[
                "prior_investor_startup_relationship"
            ].astype(bool).sum()
        ),
        "cold_start_investor": int(
            df[
                "cold_start_investor"
            ].astype(bool).sum()
        ),
        "cold_start_startup": int(
            df[
                "cold_start_startup"
            ].astype(bool).sum()
        ),
        "investor_core_connected": int(
            df[
                "investor_core_connected"
            ].astype(bool).sum()
        ),
        "startup_core_connected": int(
            df[
                "startup_core_connected"
            ].astype(bool).sum()
        ),
        "collision_flag_true": int(
            df[
                "has_realized_other_T60_positive_collision"
            ].astype(bool).sum()
        ),
    }

    for key, value in subgroup_counts.items():

        pct = value / len(df) * 100.0

        print(
            f"{key:<32} "
            f"{value:>8,}   "
            f"({pct:6.2f}%)"
        )

    # Cross-check counts already frozen/observed before Phase 7.
    for key, expected in (
        EXPECTED_OBSERVED_SUBGROUP_COUNTS.items()
    ):

        observed = subgroup_counts[key]

        require(
            observed == expected,
            (
                f"Previously observed subgroup "
                f"count drift for {key}: "
                f"expected {expected}, "
                f"observed {observed}."
            ),
        )

    print()
    print(
        "Previously observed Phase-6 "
        "subgroup counts: PASS"
    )

    # =========================================================================
    # 7.0.8 — Event-level dependence structure
    # =========================================================================

    print_section(
        "7.0.8 — EVENT-LEVEL DEPENDENCE STRUCTURE"
    )

    investor_counts = (
        df["investor_id"]
        .value_counts(dropna=False)
    )

    startup_counts = (
        df["startup_id"]
        .value_counts(dropna=False)
    )

    dependence = {
        "unique_investors": int(
            df["investor_id"]
            .nunique(dropna=False)
        ),
        "unique_startups": int(
            df["startup_id"]
            .nunique(dropna=False)
        ),
        "investors_with_multiple_test_events": int(
            (investor_counts > 1).sum()
        ),
        "startups_with_multiple_test_events": int(
            (startup_counts > 1).sum()
        ),
        "test_events_from_repeated_investors": int(
            investor_counts[
                investor_counts > 1
            ].sum()
        ),
        "test_events_from_repeated_startups": int(
            startup_counts[
                startup_counts > 1
            ].sum()
        ),
        "max_test_events_per_investor": int(
            investor_counts.max()
        ),
        "max_test_events_per_startup": int(
            startup_counts.max()
        ),
        "median_test_events_per_investor": float(
            investor_counts.median()
        ),
        "median_test_events_per_startup": float(
            startup_counts.median()
        ),
    }

    for key, value in dependence.items():

        if isinstance(value, float):

            print(
                f"{key:<42} "
                f"{value:>10.2f}"
            )

        else:

            print(
                f"{key:<42} "
                f"{value:>10,}"
            )

    repeated_investor_event_share = (
        dependence[
            "test_events_from_repeated_investors"
        ]
        / TEST_CASES
    )

    repeated_startup_event_share = (
        dependence[
            "test_events_from_repeated_startups"
        ]
        / TEST_CASES
    )

    print()
    print(
        "Share of events from investors "
        "appearing >1×: "
        f"{repeated_investor_event_share:.2%}"
    )

    print(
        "Share of events from startups "
        "appearing >1×:  "
        f"{repeated_startup_event_share:.2%}"
    )

    # =========================================================================
    # 7.0.9 — Schema / missingness inventory
    # =========================================================================

    print_section(
        "7.0.9 — SCHEMA AND MISSINGNESS INVENTORY"
    )

    schema = build_schema_inventory(df)

    print(
        schema[
            [
                "column",
                "dtype",
                "null_count",
                "null_pct",
                "unique_non_null",
            ]
        ].to_string(index=False)
    )

    total_nulls = int(
        df.isna()
        .sum()
        .sum()
    )

    columns_with_missing = int(
        (df.isna().sum() > 0).sum()
    )

    print()
    print(
        f"Total missing cells:          "
        f"{total_nulls:,}"
    )

    print(
        f"Columns with missing values:  "
        f"{columns_with_missing:,}"
    )

    # =========================================================================
    # 7.0.10 — History and structural-variable support
    # =========================================================================

    print_section(
        "7.0.10 — HISTORY / STRUCTURE VARIABLE SUPPORT"
    )

    inspect_cols = [
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
        "pair_repeats_within_t60",
        "t60_pair_event_count",
        "investor_core_connected",
        "startup_core_connected",
        "investor_core_total_degree",
        "startup_core_total_degree",
        "investor_structural_source_class",
        "startup_structural_source_class",
    ]

    variable_support = []

    for col in inspect_cols:

        s = df[col]

        record = {
            "column": col,
            "dtype": str(s.dtype),
            "null_count": int(
                s.isna().sum()
            ),
            "unique_non_null": int(
                s.nunique(dropna=True)
            ),
            "min": None,
            "median": None,
            "max": None,
        }

        if (
            pd.api.types.is_numeric_dtype(s)
            and s.notna().any()
        ):

            record["min"] = float(s.min())
            record["median"] = float(
                s.median()
            )
            record["max"] = float(s.max())

        variable_support.append(record)

        print()
        print(col)
        print("-" * len(col))

        counts = (
            s.value_counts(
                dropna=False
            )
            .head(25)
        )

        print(
            counts.to_string()
        )

    variable_support_df = pd.DataFrame(
        variable_support
    )

    # =========================================================================
    # 7.0.11 — Derived Phase-7 audit artifacts
    # =========================================================================

    print_section(
        "7.0.11 — WRITE PHASE-7 AUDIT ARTIFACTS"
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    schema_path = (
        OUT_DIR
        / "phase_7_0_schema_inventory_V2.csv"
    )

    pair_cold_path = (
        OUT_DIR
        / "phase_7_0_pair_cold_cross_tab_V2.csv"
    )

    interaction_status_path = (
        OUT_DIR
        / "phase_7_0_pair_interaction_status_cross_tab_V2.csv"
    )

    structural_path = (
        OUT_DIR
        / "phase_7_0_structural_joint_status_counts_V2.csv"
    )

    variable_support_path = (
        OUT_DIR
        / "phase_7_0_variable_support_V2.csv"
    )

    metrics_path = (
        OUT_DIR
        / "phase_7_0_reconstructed_frozen_metrics_V2.json"
    )

    contract_path = (
        OUT_DIR
        / "phase_7_0_analysis_contract_V2.json"
    )

    schema.to_csv(
        schema_path,
        index=False,
    )

    pair_cold_cross.to_csv(
        pair_cold_path,
        index=False,
    )

    interaction_status_cross.to_csv(
        interaction_status_path,
        index=False,
    )

    structural_cross.to_csv(
        structural_path,
        index=False,
    )

    variable_support_df.to_csv(
        variable_support_path,
        index=False,
    )

    reconstructed_metrics = {
        "n_test_cases": TEST_CASES,
        "hits_at_10": reconstructed_hits,
        "HR@10": reconstructed_hr10_mean,
        "NDCG@10": reconstructed_ndcg10_mean,
        "mean_positive_rank": mean_rank,
        "median_positive_rank": median_rank,
        "positive_rank_min": int(
            rank.min()
        ),
        "positive_rank_max": int(
            rank.max()
        ),
        "rank_derived_only": True,
        "model_inference_performed": False,
        "test_rescored": False,
    }

    json_dump(
        reconstructed_metrics,
        metrics_path,
    )

    contract = {

        "phase": "7.0",

        "schema_version":
            "ITRS_PHASE7_0_ANALYSIS_CONTRACT_V2",

        "status": "PASS",

        "scientific_role": (
            "Post-hoc diagnostic analysis of "
            "the immutable Phase-6 one-shot "
            "final test."
        ),

        "audit_trail": {

            "V1": {
                "status": "FAILED_ANALYSIS_ASSUMPTION",
                "failed_assumption": (
                    "test_position == case_index"
                ),
                "scientific_impact": (
                    "NONE — failure occurred during "
                    "read-only integrity checks before "
                    "any subgroup analysis."
                ),
            },

            "V2": {
                "status": "PASS",
                "correction": (
                    "test_position is test-local; "
                    "case_index/matrix_row_index use "
                    "the full validation+test "
                    "evaluation coordinate."
                ),
            },
        },

        "frozen_evaluation_coordinates": {
            "validation_cases": VALIDATION_CASES,
            "test_cases": TEST_CASES,
            "total_cases": TOTAL_EVALUATION_CASES,
            "validation_global_range": [
                0,
                VALIDATION_CASES - 1,
            ],
            "test_global_range": [
                VALIDATION_CASES,
                TOTAL_EVALUATION_CASES - 1,
            ],
            "test_local_range": [
                0,
                TEST_CASES - 1,
            ],
            "mapping": (
                "test_position = "
                "case_index - validation_cases"
            ),
        },

        "frozen_source": {
            "analysis_bundle": str(
                BUNDLE.relative_to(
                    REPO_ROOT
                )
            ),
            "analysis_bundle_sha256":
                bundle_hash,
            "manifest": str(
                BUNDLE_MANIFEST.relative_to(
                    REPO_ROOT
                )
            ),
            "final_test_summary": str(
                FINAL_TEST_SUMMARY.relative_to(
                    REPO_ROOT
                )
            ),
            "rows": len(df),
            "columns": len(
                df.columns
            ),
        },

        "phase_7_rules": {
            "model_retraining_allowed":
                False,
            "checkpoint_reselection_allowed":
                False,
            "test_rescoring_allowed":
                False,
            "negative_resampling_allowed":
                False,
            "candidate_regeneration_allowed":
                False,
            "test_based_model_selection_allowed":
                False,
            "event_level_evaluation_is_primary":
                True,
            "subgroup_n_must_be_reported":
                True,
            (
                "descriptive_vs_inferential_"
                "vs_interpretive_claims_separated"
            ):
                True,
        },

        "metric_reconstruction":
            reconstructed_metrics,

        "subgroup_counts":
            subgroup_counts,

        "dependence_structure":
            dependence,

        "logical_contract": {
            (
                "cold_start_investor_equals_"
                "not_seen_before_t60"
            ):
                True,
            (
                "cold_start_startup_equals_"
                "not_seen_before_t60"
            ):
                True,
            (
                "prior_relationship_equals_"
                "pair_seen_before_t60"
            ):
                True,
            (
                "new_to_investor_equals_"
                "not_pair_seen_before_t60"
            ):
                True,
            (
                "new_and_prior_relationship_"
                "are_complements"
            ):
                True,
        },

        "available_primary_thesis_fields":
            required_cols,

        "derived_artifacts": {
            "schema_inventory": str(
                schema_path.relative_to(
                    REPO_ROOT
                )
            ),
            "pair_cold_cross_tab": str(
                pair_cold_path.relative_to(
                    REPO_ROOT
                )
            ),
            (
                "pair_interaction_status_"
                "cross_tab"
            ): str(
                interaction_status_path.relative_to(
                    REPO_ROOT
                )
            ),
            (
                "structural_joint_status_counts"
            ): str(
                structural_path.relative_to(
                    REPO_ROOT
                )
            ),
            "variable_support": str(
                variable_support_path.relative_to(
                    REPO_ROOT
                )
            ),
            "reconstructed_metrics": str(
                metrics_path.relative_to(
                    REPO_ROOT
                )
            ),
        },
    }

    json_dump(
        contract,
        contract_path,
    )

    artifact_paths = [
        schema_path,
        pair_cold_path,
        interaction_status_path,
        structural_path,
        variable_support_path,
        metrics_path,
        contract_path,
    ]

    artifact_hashes = {}

    for artifact in artifact_paths:

        relative = str(
            artifact.relative_to(
                REPO_ROOT
            )
        )

        artifact_hashes[
            relative
        ] = file_sha256(
            artifact
        )

        print(
            f"WROTE  {relative}"
        )

    hashes_path = (
        OUT_DIR
        / "phase_7_0_derived_artifact_sha256_V2.json"
    )

    json_dump(
        artifact_hashes,
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
        "PHASE 7.0 V2 RESULT"
    )

    print("Frozen bundle fingerprint:       PASS")
    print("Frozen row/schema binding:        PASS")
    print("Evaluation-coordinate integrity: PASS")
    print("Rank-derived metric recreation:   PASS")
    print("New/repeat semantic binding:      PASS")
    print("Cold-start semantic binding:      PASS")
    print("Known subgroup count binding:     PASS")
    print("History-variable inventory:       COMPLETE")
    print("Structural-variable inventory:    COMPLETE")
    print("Dependence audit:                 COMPLETE")

    print()
    print("Model inference executed:         NO")
    print("Final test rescored:              NO")
    print("Final test modified:              NO")

    print()
    print(
        "PHASE 7.0 ANALYSIS CONTRACT STATUS: PASS"
    )


if __name__ == "__main__":
    main()