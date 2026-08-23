from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PHASE 2.3.4 — FINAL TEMPORAL-SPLIT INTEGRITY AUDIT
# =============================================================================
#
# PURPOSE
# -------
#
# Verify the complete Phase-2 temporal reconstruction:
#
# canonical
#   |
#   +-- temporal experiment <= 2026-03-31
#   |       |
#   |       +-- T0-T59 historical training pool
#   |       |
#   |       +-- T60
#   |               +-- validation
#   |               +-- test
#   |
#   +-- post-endpoint > 2026-03-31
#
# This script also creates:
#
#   1. one model-ready temporal split table;
#   2. one T60 investor-startup holdout manifest;
#   3. integrity-check and split-diagnostic audit tables.
#
# No negative sampling or graph construction occurs here.
# =============================================================================


# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------

CANONICAL_PATH = Path(
    "data/processed/interactions.parquet"
)

TEMPORAL_PATH = Path(
    "data/experimental/phase_2/temporal/"
    "interactions_itrs_temporal.parquet"
)

POST_ENDPOINT_PATH = Path(
    "data/experimental/phase_2/temporal/"
    "post_endpoint_interactions.parquet"
)

SEGMENT_METADATA_PATH = Path(
    "data/experimental/phase_2/temporal/"
    "itrs_segment_metadata.csv"
)

T60_SPLIT_PATH = Path(
    "data/experimental/phase_2/model_ready/"
    "t60_validation_test_split.parquet"
)

T60_ASSIGNMENT_PATH = Path(
    "data/experimental/phase_2/model_ready/"
    "t60_split_assignments.csv"
)


# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

MODEL_READY_DIR = Path(
    "data/experimental/phase_2/model_ready"
)

AUDIT_DIR = Path(
    "data/experimental/phase_2/audits"
)

FULL_SPLIT_OUTPUT = (
    MODEL_READY_DIR
    / "interactions_itrs_temporal_split.parquet"
)

HOLDOUT_PAIR_OUTPUT = (
    MODEL_READY_DIR
    / "t60_holdout_pair_manifest.parquet"
)

INTEGRITY_OUTPUT = (
    AUDIT_DIR
    / "phase_2_final_temporal_integrity_checks.csv"
)

SPLIT_SUMMARY_OUTPUT = (
    AUDIT_DIR
    / "phase_2_final_temporal_split_summary.csv"
)

DISTRIBUTION_OUTPUT = (
    AUDIT_DIR
    / "phase_2_validation_test_distribution_diagnostics.csv"
)

OVERLAP_OUTPUT = (
    AUDIT_DIR
    / "phase_2_validation_test_overlap_diagnostics.csv"
)


# -----------------------------------------------------------------------------
# Expected Phase-2 values
# -----------------------------------------------------------------------------

EXPECTED_CANONICAL_ROWS = 1_208_051

EXPECTED_TEMPORAL_ROWS = 1_195_937
EXPECTED_POST_ENDPOINT_ROWS = 12_114

EXPECTED_T0_ROWS = 100_173
EXPECTED_T1_T59_ROWS = 1_073_249

EXPECTED_HISTORY_ROWS = 1_173_422

EXPECTED_T60_ROWS = 22_515
EXPECTED_VALIDATION_ROWS = 2_251
EXPECTED_TEST_ROWS = 20_264

EXPECTED_PAIR_OVERLAP = 33
EXPECTED_FUNDING_ROUND_OVERLAP = 1_315

EXPECTED_SEGMENT_ROWS = 61

DETAILED_START_DATE = pd.Timestamp(
    "2011-04-01"
)

T0_END_DATE = pd.Timestamp(
    "2011-03-31"
)

EXPERIMENT_END_DATE = pd.Timestamp(
    "2026-03-31"
)

N_DETAILED_SEGMENTS = 60
MONTHS_PER_SEGMENT = 3


def separator(char="=", width=120):
    print(char * width)


def pct(num, den):

    if den == 0:
        return np.nan

    return num / den * 100


def add_check(
    checks,
    name,
    passed,
    observed,
    expected,
    note="",
):

    checks.append(
        {
            "check": name,
            "status": (
                "PASS"
                if passed
                else "FAIL"
            ),
            "observed": observed,
            "expected": expected,
            "note": note,
        }
    )


def main():

    separator()
    print(
        "PHASE 2.3.4 — "
        "FINAL TEMPORAL-SPLIT INTEGRITY AUDIT"
    )
    separator()

    checks = []

    # =========================================================================
    # 1. Load all Phase-2 artifacts
    # =========================================================================

    print("\nLoading Phase-2 artifacts...")

    canonical = pd.read_parquet(
        CANONICAL_PATH
    )

    temporal = pd.read_parquet(
        TEMPORAL_PATH
    )

    post_endpoint = pd.read_parquet(
        POST_ENDPOINT_PATH
    )

    metadata = pd.read_csv(
        SEGMENT_METADATA_PATH
    )

    t60_split = pd.read_parquet(
        T60_SPLIT_PATH
    )

    t60_assignment = pd.read_csv(
        T60_ASSIGNMENT_PATH
    )

    for frame in [
        canonical,
        temporal,
        post_endpoint,
        t60_split,
    ]:

        frame["announced_on"] = pd.to_datetime(
            frame["announced_on"],
            errors="raise",
        )

    print("Loaded.")

    # =========================================================================
    # 2. Global row-count reconstruction
    # =========================================================================

    separator("-")
    print("GLOBAL DATASET RECONSTRUCTION")
    separator("-")

    add_check(
        checks,
        "canonical_row_count",
        len(canonical)
        == EXPECTED_CANONICAL_ROWS,
        len(canonical),
        EXPECTED_CANONICAL_ROWS,
    )

    add_check(
        checks,
        "temporal_row_count",
        len(temporal)
        == EXPECTED_TEMPORAL_ROWS,
        len(temporal),
        EXPECTED_TEMPORAL_ROWS,
    )

    add_check(
        checks,
        "post_endpoint_row_count",
        len(post_endpoint)
        == EXPECTED_POST_ENDPOINT_ROWS,
        len(post_endpoint),
        EXPECTED_POST_ENDPOINT_ROWS,
    )

    add_check(
        checks,
        "temporal_plus_post_equals_canonical",
        (
            len(temporal)
            + len(post_endpoint)
            == len(canonical)
        ),
        (
            len(temporal)
            + len(post_endpoint)
        ),
        len(canonical),
    )

    # IDs must remain unique inside all generated layers.

    for name, frame in [
        ("canonical", canonical),
        ("temporal", temporal),
        ("post_endpoint", post_endpoint),
        ("t60_split", t60_split),
    ]:

        duplicate_count = int(
            frame["interaction_id"]
            .duplicated()
            .sum()
        )

        add_check(
            checks,
            f"{name}_interaction_id_uniqueness",
            duplicate_count == 0,
            duplicate_count,
            0,
        )

    # -------------------------------------------------------------------------
    # Verify temporal and post-endpoint ID sets do not overlap.
    # -------------------------------------------------------------------------

    post_ids = set(
        post_endpoint[
            "interaction_id"
        ]
    )

    temporal_post_overlap = int(
        temporal[
            "interaction_id"
        ]
        .isin(post_ids)
        .sum()
    )

    add_check(
        checks,
        "temporal_post_endpoint_id_overlap",
        temporal_post_overlap == 0,
        temporal_post_overlap,
        0,
    )

    # =========================================================================
    # 3. Date-boundary integrity
    # =========================================================================

    separator("-")
    print("DATE-BOUNDARY INTEGRITY")
    separator("-")

    temporal_after_endpoint = int(
        (
            temporal["announced_on"]
            > EXPERIMENT_END_DATE
        )
        .sum()
    )

    add_check(
        checks,
        "temporal_rows_after_experiment_endpoint",
        temporal_after_endpoint == 0,
        temporal_after_endpoint,
        0,
    )

    post_on_or_before_endpoint = int(
        (
            post_endpoint["announced_on"]
            <= EXPERIMENT_END_DATE
        )
        .sum()
    )

    add_check(
        checks,
        "post_endpoint_rows_on_or_before_endpoint",
        post_on_or_before_endpoint == 0,
        post_on_or_before_endpoint,
        0,
    )

    print(
        f"Temporal date range:      "
        f"{temporal['announced_on'].min().date()} -> "
        f"{temporal['announced_on'].max().date()}"
    )

    print(
        f"Post-endpoint range:      "
        f"{post_endpoint['announced_on'].min().date()} -> "
        f"{post_endpoint['announced_on'].max().date()}"
    )

    # =========================================================================
    # 4. Segment-metadata integrity
    # =========================================================================

    separator("-")
    print("SEGMENT-METADATA INTEGRITY")
    separator("-")

    metadata["segment_start_date"] = (
        pd.to_datetime(
            metadata["segment_start_date"],
            errors="coerce",
        )
    )

    metadata["segment_end_date"] = (
        pd.to_datetime(
            metadata["segment_end_date"],
            errors="coerce",
        )
    )

    add_check(
        checks,
        "segment_metadata_row_count",
        len(metadata)
        == EXPECTED_SEGMENT_ROWS,
        len(metadata),
        EXPECTED_SEGMENT_ROWS,
    )

    expected_segment_numbers = set(
        range(0, 61)
    )

    actual_segment_numbers = set(
        metadata[
            "segment_number"
        ]
        .astype(int)
    )

    add_check(
        checks,
        "segment_numbers_exactly_0_to_60",
        actual_segment_numbers
        == expected_segment_numbers,
        len(actual_segment_numbers),
        61,
    )

    # -------------------------------------------------------------------------
    # T0 boundary
    # -------------------------------------------------------------------------

    t0 = temporal[
        temporal["segment_number"] == 0
    ]

    t0_bad_dates = int(
        (
            t0["announced_on"]
            > T0_END_DATE
        )
        .sum()
    )

    add_check(
        checks,
        "t0_date_boundary",
        t0_bad_dates == 0,
        t0_bad_dates,
        0,
    )

    add_check(
        checks,
        "t0_row_count",
        len(t0)
        == EXPECTED_T0_ROWS,
        len(t0),
        EXPECTED_T0_ROWS,
    )

    # -------------------------------------------------------------------------
    # Exact T1-T60 date boundaries
    # -------------------------------------------------------------------------

    first_month = (
        DETAILED_START_DATE
        .to_period("M")
    )

    segment_boundary_failures = 0
    segment_event_date_failures = 0

    for segment in range(
        1,
        N_DETAILED_SEGMENTS + 1,
    ):

        expected_start_month = (
            first_month
            + (
                segment - 1
            )
            * MONTHS_PER_SEGMENT
        )

        expected_end_month = (
            expected_start_month
            + (
                MONTHS_PER_SEGMENT - 1
            )
        )

        expected_start = (
            expected_start_month
            .to_timestamp(
                how="start"
            )
            .normalize()
        )

        expected_end = (
            expected_end_month
            .to_timestamp(
                how="end"
            )
            .normalize()
        )

        meta_row = metadata[
            metadata["segment_number"]
            == segment
        ].iloc[0]

        metadata_match = (
            meta_row[
                "segment_start_date"
            ]
            == expected_start
            and
            meta_row[
                "segment_end_date"
            ]
            == expected_end
        )

        if not metadata_match:
            segment_boundary_failures += 1

        segment_df = temporal[
            temporal["segment_number"]
            == segment
        ]

        bad_events = (
            (
                segment_df["announced_on"]
                < expected_start
            )
            |
            (
                segment_df["announced_on"]
                > expected_end
            )
        ).sum()

        segment_event_date_failures += int(
            bad_events
        )

    add_check(
        checks,
        "t1_t60_metadata_boundary_failures",
        segment_boundary_failures == 0,
        segment_boundary_failures,
        0,
    )

    add_check(
        checks,
        "events_outside_assigned_segment_bounds",
        segment_event_date_failures == 0,
        segment_event_date_failures,
        0,
    )

    # =========================================================================
    # 5. Temporal-role reconstruction
    # =========================================================================

    separator("-")
    print("TEMPORAL ROLE RECONSTRUCTION")
    separator("-")

    t1_t59 = temporal[
        temporal["segment_number"]
        .between(1, 59)
    ]

    history = temporal[
        temporal["segment_number"]
        < 60
    ]

    t60 = temporal[
        temporal["segment_number"]
        == 60
    ]

    add_check(
        checks,
        "t1_t59_row_count",
        len(t1_t59)
        == EXPECTED_T1_T59_ROWS,
        len(t1_t59),
        EXPECTED_T1_T59_ROWS,
    )

    add_check(
        checks,
        "history_t0_t59_row_count",
        len(history)
        == EXPECTED_HISTORY_ROWS,
        len(history),
        EXPECTED_HISTORY_ROWS,
    )

    add_check(
        checks,
        "t60_row_count",
        len(t60)
        == EXPECTED_T60_ROWS,
        len(t60),
        EXPECTED_T60_ROWS,
    )

    # =========================================================================
    # 6. T60 validation/test reconstruction
    # =========================================================================

    separator("-")
    print("T60 SPLIT INTEGRITY")
    separator("-")

    validation = t60_split[
        t60_split[
            "evaluation_split"
        ]
        == "validation"
    ]

    test = t60_split[
        t60_split[
            "evaluation_split"
        ]
        == "test"
    ]

    add_check(
        checks,
        "validation_row_count",
        len(validation)
        == EXPECTED_VALIDATION_ROWS,
        len(validation),
        EXPECTED_VALIDATION_ROWS,
    )

    add_check(
        checks,
        "test_row_count",
        len(test)
        == EXPECTED_TEST_ROWS,
        len(test),
        EXPECTED_TEST_ROWS,
    )

    split_t60_ids = set(
        t60_split[
            "interaction_id"
        ]
    )

    source_t60_ids = set(
        t60[
            "interaction_id"
        ]
    )

    add_check(
        checks,
        "t60_split_ids_match_temporal_t60",
        split_t60_ids
        == source_t60_ids,
        len(split_t60_ids),
        len(source_t60_ids),
    )

    validation_ids = set(
        validation[
            "interaction_id"
        ]
    )

    test_ids = set(
        test[
            "interaction_id"
        ]
    )

    interaction_overlap = len(
        validation_ids
        & test_ids
    )

    add_check(
        checks,
        "validation_test_interaction_overlap",
        interaction_overlap == 0,
        interaction_overlap,
        0,
    )

    # -------------------------------------------------------------------------
    # Verify assignment CSV against the parquet split.
    # -------------------------------------------------------------------------

    parquet_assignment = (
        t60_split[
            [
                "interaction_id",
                "evaluation_split",
            ]
        ]
        .sort_values(
            "interaction_id"
        )
        .reset_index(drop=True)
    )

    csv_assignment = (
        t60_assignment[
            [
                "interaction_id",
                "evaluation_split",
            ]
        ]
        .sort_values(
            "interaction_id"
        )
        .reset_index(drop=True)
    )

    assignments_equal = (
        parquet_assignment
        .equals(
            csv_assignment
        )
    )

    add_check(
        checks,
        "assignment_csv_matches_split_parquet",
        assignments_equal,
        assignments_equal,
        True,
    )

    # =========================================================================
    # 7. Check historical vs T60 funding-round separation
    # =========================================================================

    separator("-")
    print("TEMPORAL LEAKAGE BOUNDARY")
    separator("-")

    historical_round_ids = set(
        history[
            "funding_round_id"
        ]
        .unique()
    )

    t60_round_ids = set(
        t60[
            "funding_round_id"
        ]
        .unique()
    )

    history_t60_round_overlap = len(
        historical_round_ids
        & t60_round_ids
    )

    add_check(
        checks,
        "funding_round_overlap_history_vs_t60",
        history_t60_round_overlap == 0,
        history_t60_round_overlap,
        0,
        (
            "A funding round must not span the historical "
            "training period and T60."
        ),
    )

    # =========================================================================
    # 8. Validation/test overlap diagnostics
    # =========================================================================

    validation_pairs = (
        validation[
            [
                "investor_id",
                "startup_id",
            ]
        ]
        .drop_duplicates()
    )

    test_pairs = (
        test[
            [
                "investor_id",
                "startup_id",
            ]
        ]
        .drop_duplicates()
    )

    pair_overlap = (
        validation_pairs.merge(
            test_pairs,
            on=[
                "investor_id",
                "startup_id",
            ],
            how="inner",
        )
    )

    validation_round_ids = set(
        validation[
            "funding_round_id"
        ]
        .unique()
    )

    test_round_ids = set(
        test[
            "funding_round_id"
        ]
        .unique()
    )

    funding_round_overlap = (
        validation_round_ids
        & test_round_ids
    )

    add_check(
        checks,
        "validation_test_pair_overlap_reproduced",
        len(pair_overlap)
        == EXPECTED_PAIR_OVERLAP,
        len(pair_overlap),
        EXPECTED_PAIR_OVERLAP,
        (
            "Diagnostic only; pair overlap is preserved "
            "to reproduce the event-level paper split."
        ),
    )

    add_check(
        checks,
        "validation_test_funding_round_overlap_reproduced",
        len(funding_round_overlap)
        == EXPECTED_FUNDING_ROUND_OVERLAP,
        len(funding_round_overlap),
        EXPECTED_FUNDING_ROUND_OVERLAP,
        (
            "Diagnostic only; both subsets are inside held-out T60."
        ),
    )

    # =========================================================================
    # 9. Distribution diagnostics
    # =========================================================================

    def distribution_row(
        name,
        subset,
    ):

        n = len(subset)

        return {
            "split": name,

            "interactions": n,

            "new_pair_event_share_pct": pct(
                subset[
                    "new_to_investor_pair"
                ].sum(),
                n,
            ),

            "cold_investor_event_share_pct": pct(
                (
                    ~subset[
                        "investor_seen_before_t60"
                    ]
                ).sum(),
                n,
            ),

            "cold_startup_event_share_pct": pct(
                (
                    ~subset[
                        "startup_seen_before_t60"
                    ]
                ).sum(),
                n,
            ),

            "warm_investor_warm_startup_share_pct": pct(
                (
                    subset[
                        "interaction_cold_start_status"
                    ]
                    ==
                    "warm_investor__warm_startup"
                ).sum(),
                n,
            ),
        }

    distribution = pd.DataFrame(
        [
            distribution_row(
                "full_t60",
                t60_split,
            ),
            distribution_row(
                "validation",
                validation,
            ),
            distribution_row(
                "test",
                test,
            ),
        ]
    )

    full_row = (
        distribution[
            distribution["split"]
            == "full_t60"
        ]
        .iloc[0]
    )

    for metric in [
        "new_pair_event_share_pct",
        "cold_investor_event_share_pct",
        "cold_startup_event_share_pct",
        "warm_investor_warm_startup_share_pct",
    ]:

        distribution[
            f"{metric}_delta_vs_full_t60_pp"
        ] = (
            distribution[metric]
            - full_row[metric]
        )

    # =========================================================================
    # 10. Build complete experiment split
    # =========================================================================

    separator("-")
    print("MATERIALIZING FINAL TEMPORAL SPLIT")
    separator("-")

    full_split = (
        temporal.copy()
    )

    full_split[
        "experiment_split"
    ] = "train"

    assignment_map = (
        t60_assignment
        .set_index(
            "interaction_id"
        )[
            "evaluation_split"
        ]
    )

    t60_mask = (
        full_split[
            "segment_number"
        ]
        == 60
    )

    full_split.loc[
        t60_mask,
        "experiment_split",
    ] = (
        full_split.loc[
            t60_mask,
            "interaction_id",
        ]
        .map(
            assignment_map
        )
    )

    missing_final_assignments = int(
        full_split.loc[
            t60_mask,
            "experiment_split",
        ]
        .isna()
        .sum()
    )

    add_check(
        checks,
        "t60_missing_final_split_assignments",
        missing_final_assignments == 0,
        missing_final_assignments,
        0,
    )

    final_train = (
        full_split[
            full_split[
                "experiment_split"
            ]
            == "train"
        ]
    )

    final_validation = (
        full_split[
            full_split[
                "experiment_split"
            ]
            == "validation"
        ]
    )

    final_test = (
        full_split[
            full_split[
                "experiment_split"
            ]
            == "test"
        ]
    )

    add_check(
        checks,
        "final_train_row_count",
        len(final_train)
        == EXPECTED_HISTORY_ROWS,
        len(final_train),
        EXPECTED_HISTORY_ROWS,
    )

    add_check(
        checks,
        "final_validation_row_count",
        len(final_validation)
        == EXPECTED_VALIDATION_ROWS,
        len(final_validation),
        EXPECTED_VALIDATION_ROWS,
    )

    add_check(
        checks,
        "final_test_row_count",
        len(final_test)
        == EXPECTED_TEST_ROWS,
        len(final_test),
        EXPECTED_TEST_ROWS,
    )

    # =========================================================================
    # 11. Create future graph-leakage holdout manifest
    # =========================================================================
    #
    # Any future investment-edge graph used for training must not include
    # T60 holdout investment relationships.
    #
    # This manifest preserves unique investor-startup holdout pairs and says
    # whether each appears in validation, test, or both.
    # =========================================================================

    holdout_pair_manifest = (
        t60_split.groupby(
            [
                "investor_id",
                "startup_id",
            ],
            observed=True,
        )
        .agg(
            t60_event_count=(
                "interaction_id",
                "size",
            ),

            validation_event_count=(
                "evaluation_split",
                lambda x:
                    int(
                        (
                            x
                            == "validation"
                        )
                        .sum()
                    ),
            ),

            test_event_count=(
                "evaluation_split",
                lambda x:
                    int(
                        (
                            x
                            == "test"
                        )
                        .sum()
                    ),
            ),

            pair_seen_before_t60=(
                "pair_seen_before_t60",
                "max",
            ),

            new_to_investor_pair=(
                "new_to_investor_pair",
                "max",
            ),
        )
        .reset_index()
    )

    holdout_pair_manifest[
        "appears_in_validation"
    ] = (
        holdout_pair_manifest[
            "validation_event_count"
        ]
        > 0
    )

    holdout_pair_manifest[
        "appears_in_test"
    ] = (
        holdout_pair_manifest[
            "test_event_count"
        ]
        > 0
    )

    holdout_pair_manifest[
        "appears_in_both_validation_and_test"
    ] = (
        holdout_pair_manifest[
            "appears_in_validation"
        ]
        &
        holdout_pair_manifest[
            "appears_in_test"
        ]
    )

    add_check(
        checks,
        "holdout_pair_manifest_count",
        len(
            holdout_pair_manifest
        )
        == 22_327,
        len(
            holdout_pair_manifest
        ),
        22_327,
    )

    # =========================================================================
    # 12. Build final split summary
    # =========================================================================

    split_summary = (
        full_split.groupby(
            "experiment_split",
            observed=True,
        )
        .agg(
            interactions=(
                "interaction_id",
                "size",
            ),
            unique_investors=(
                "investor_id",
                "nunique",
            ),
            unique_startups=(
                "startup_id",
                "nunique",
            ),
            unique_pairs=(
                "interaction_id",
                lambda idx:
                    full_split.loc[
                        idx.index,
                        [
                            "investor_id",
                            "startup_id",
                        ],
                    ]
                    .drop_duplicates()
                    .shape[0],
            ),
            funding_rounds=(
                "funding_round_id",
                "nunique",
            ),
            min_segment=(
                "segment_number",
                "min",
            ),
            max_segment=(
                "segment_number",
                "max",
            ),
        )
        .reset_index()
    )

    # =========================================================================
    # 13. Overlap diagnostic table
    # =========================================================================

    overlap_diagnostics = pd.DataFrame(
        [
            {
                "metric": (
                    "validation_test_interaction_overlap"
                ),
                "count": (
                    interaction_overlap
                ),
                "interpretation": (
                    "Must be zero."
                ),
            },
            {
                "metric": (
                    "validation_test_unique_pair_overlap"
                ),
                "count": (
                    len(pair_overlap)
                ),
                "interpretation": (
                    "Expected under event-level random split; "
                    "preserved for paper fidelity."
                ),
            },
            {
                "metric": (
                    "validation_test_funding_round_overlap"
                ),
                "count": (
                    len(
                        funding_round_overlap
                    )
                ),
                "interpretation": (
                    "Expected because one funding round can contain "
                    "multiple investor interactions; both are held out."
                ),
            },
            {
                "metric": (
                    "history_t60_funding_round_overlap"
                ),
                "count": (
                    history_t60_round_overlap
                ),
                "interpretation": (
                    "Must be zero to preserve the temporal holdout."
                ),
            },
        ]
    )

    # =========================================================================
    # 14. Final integrity status
    # =========================================================================

    checks_df = pd.DataFrame(
        checks
    )

    failed_checks = (
        checks_df[
            checks_df[
                "status"
            ]
            == "FAIL"
        ]
    )

    separator("-")
    print("FINAL INTEGRITY CHECKS")
    separator("-")

    print(
        checks_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # 15. Print distribution diagnostics
    # =========================================================================

    separator("-")
    print(
        "VALIDATION / TEST DISTRIBUTION DIAGNOSTICS"
    )
    separator("-")

    print(
        distribution.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    # =========================================================================
    # 16. Print overlap diagnostics
    # =========================================================================

    separator("-")
    print(
        "HOLDOUT OVERLAP DIAGNOSTICS"
    )
    separator("-")

    print(
        overlap_diagnostics.to_string(
            index=False
        )
    )

    print(
        f"\nValidation pair-overlap share: "
        f"{pct(len(pair_overlap), len(validation_pairs)):.3f}%"
    )

    print(
        f"Test pair-overlap share:       "
        f"{pct(len(pair_overlap), len(test_pairs)):.3f}%"
    )

    print(
        f"\nValidation funding rounds also represented in test: "
        f"{pct(len(funding_round_overlap), len(validation_round_ids)):.3f}%"
    )

    print(
        f"Test funding rounds also represented in validation: "
        f"{pct(len(funding_round_overlap), len(test_round_ids)):.3f}%"
    )

    # =========================================================================
    # 17. Save final Phase-2 artifacts
    # =========================================================================

    MODEL_READY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    full_split.to_parquet(
        FULL_SPLIT_OUTPUT,
        index=False,
    )

    holdout_pair_manifest.to_parquet(
        HOLDOUT_PAIR_OUTPUT,
        index=False,
    )

    checks_df.to_csv(
        INTEGRITY_OUTPUT,
        index=False,
    )

    split_summary.to_csv(
        SPLIT_SUMMARY_OUTPUT,
        index=False,
    )

    distribution.to_csv(
        DISTRIBUTION_OUTPUT,
        index=False,
    )

    overlap_diagnostics.to_csv(
        OVERLAP_OUTPUT,
        index=False,
    )

    # =========================================================================
    # 18. Stop if anything failed
    # =========================================================================

    if len(failed_checks) > 0:

        separator("!")
        print(
            "PHASE 2.3.4 FAILED — "
            "DO NOT CLOSE PHASE 2"
        )
        separator("!")

        print(
            failed_checks.to_string(
                index=False
            )
        )

        raise RuntimeError(
            "One or more final Phase-2 integrity checks failed."
        )

    # =========================================================================
    # 19. Success summary
    # =========================================================================

    separator()
    print(
        "PHASE 2.3.4 FINAL INTEGRITY AUDIT PASSED"
    )
    separator()

    print(
        f"""
Final temporal experiment:

Canonical interactions:
    {len(canonical):,}

ITRS temporal experiment:
    {len(full_split):,}

Historical training pool (T0-T59):
    {len(final_train):,}

Validation (10% of T60, floor rule):
    {len(final_validation):,}

Test:
    {len(final_test):,}

Post-endpoint canonical interactions preserved:
    {len(post_endpoint):,}

Temporal architecture:
    T0      <= 2011-03-31
    T1-T60   2011-04-01 -> 2026-03-31
    T60      2026-01-01 -> 2026-03-31

Detailed segments:
    60

Months per detailed segment:
    3

Outputs written to:

{FULL_SPLIT_OUTPUT}
{HOLDOUT_PAIR_OUTPUT}
{INTEGRITY_OUTPUT}
{SPLIT_SUMMARY_OUTPUT}
{DISTRIBUTION_OUTPUT}
{OVERLAP_OUTPUT}

No negative sampling has been performed.
No heterogeneous graph has been constructed.
No investment type has been filtered.
No cold-start entity has been removed.
The Phase-1 canonical dataset remains immutable.

Phase 2 is ready for closure.
"""
    )


if __name__ == "__main__":
    main()