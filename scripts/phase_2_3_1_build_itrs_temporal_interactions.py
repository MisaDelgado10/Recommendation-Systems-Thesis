from pathlib import Path

import pandas as pd


# =============================================================================
# PHASE 2.3.1 — BUILD SELECTED ITRS TEMPORAL INTERACTION LAYER
# =============================================================================
#
# SELECTED PAPER-GROUNDED TEMPORAL DESIGN
#
# T0:
#     all canonical investment events <= 2011-03-31
#
# T1-T60:
#     60 consecutive calendar-quarter-aligned 3-month segments
#
# Detailed horizon:
#     2011-04-01 -> 2026-03-31
#
# T60:
#     2026-01-01 -> 2026-03-31
#
# Post-endpoint data:
#     > 2026-03-31
#
# IMPORTANT:
#     The Phase-1 canonical dataset is NEVER modified.
# =============================================================================


INPUT_PATH = Path(
    "data/processed/interactions.parquet"
)

OUTPUT_DIR = Path(
    "data/experimental/phase_2/temporal"
)

TEMPORAL_OUTPUT = (
    OUTPUT_DIR
    / "interactions_itrs_temporal.parquet"
)

SEGMENT_METADATA_OUTPUT = (
    OUTPUT_DIR
    / "itrs_segment_metadata.csv"
)

POST_ENDPOINT_OUTPUT = (
    OUTPUT_DIR
    / "post_endpoint_interactions.parquet"
)


EXPECTED_CANONICAL_ROWS = 1_208_051

EXPECTED_ELIGIBLE_ROWS = 1_195_937
EXPECTED_T0_ROWS = 100_173
EXPECTED_DETAILED_ROWS = 1_095_764
EXPECTED_T60_ROWS = 22_515

DATE_COL = "announced_on"


DETAILED_START_DATE = pd.Timestamp(
    "2011-04-01"
)

DETAILED_END_DATE = pd.Timestamp(
    "2026-03-31"
)

T0_END_DATE = pd.Timestamp(
    "2011-03-31"
)

N_SEGMENTS = 60
MONTHS_PER_SEGMENT = 3


def separator(char="=", width=110):
    print(char * width)


def main():

    separator()
    print(
        "PHASE 2.3.1 — "
        "BUILD SELECTED ITRS TEMPORAL INTERACTION LAYER"
    )
    separator()

    # -------------------------------------------------------------------------
    # 1. Load the immutable canonical interaction table
    # -------------------------------------------------------------------------

    df = pd.read_parquet(
        INPUT_PATH
    )

    if len(df) != EXPECTED_CANONICAL_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_CANONICAL_ROWS:,} "
            f"canonical interactions; "
            f"found {len(df):,}."
        )

    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL],
        errors="raise",
    )

    print(
        f"\nCanonical interactions: "
        f"{len(df):,}"
    )

    print(
        f"Canonical range:        "
        f"{df[DATE_COL].min().date()} -> "
        f"{df[DATE_COL].max().date()}"
    )

    # -------------------------------------------------------------------------
    # 2. Separate experimental temporal view from post-endpoint tail
    # -------------------------------------------------------------------------

    temporal = (
        df[
            df[DATE_COL]
            <= DETAILED_END_DATE
        ]
        .copy()
    )

    post_endpoint = (
        df[
            df[DATE_COL]
            > DETAILED_END_DATE
        ]
        .copy()
    )

    print(
        f"\nTemporal experiment interactions: "
        f"{len(temporal):,}"
    )

    print(
        f"Post-endpoint interactions:        "
        f"{len(post_endpoint):,}"
    )

    # -------------------------------------------------------------------------
    # 3. Initialize temporal metadata
    # -------------------------------------------------------------------------

    temporal["segment_number"] = 0

    temporal["segment_label"] = "T0"

    temporal["temporal_role"] = (
        "compressed_prehistory"
    )

    # -------------------------------------------------------------------------
    # 4. Assign T1-T60
    # -------------------------------------------------------------------------

    detailed_mask = (
        temporal[DATE_COL]
        >= DETAILED_START_DATE
    )

    detailed_month = (
        temporal.loc[
            detailed_mask,
            DATE_COL,
        ]
        .dt.to_period("M")
    )

    start_month = (
        DETAILED_START_DATE
        .to_period("M")
    )

    month_offset = (
        detailed_month.map(
            lambda period:
                period.ordinal
                - start_month.ordinal
        )
    )

    segment_number = (
        month_offset
        // MONTHS_PER_SEGMENT
        + 1
    ).astype("int16")

    temporal.loc[
        detailed_mask,
        "segment_number",
    ] = segment_number

    temporal.loc[
        detailed_mask,
        "segment_label",
    ] = (
        "T"
        + temporal.loc[
            detailed_mask,
            "segment_number",
        ]
        .astype(str)
    )

    # -------------------------------------------------------------------------
    # 5. Assign experimental role
    # -------------------------------------------------------------------------

    train_history_mask = (
        temporal["segment_number"]
        .between(1, 59)
    )

    evaluation_pool_mask = (
        temporal["segment_number"]
        == 60
    )

    temporal.loc[
        train_history_mask,
        "temporal_role",
    ] = "detailed_train_history"

    temporal.loc[
        evaluation_pool_mask,
        "temporal_role",
    ] = "t60_evaluation_pool"

    # -------------------------------------------------------------------------
    # 6. Explicit segment dates
    # -------------------------------------------------------------------------

    temporal[
        "segment_start_date"
    ] = pd.NaT

    temporal[
        "segment_end_date"
    ] = pd.NaT

    # T0 has an open historical start.
    temporal.loc[
        temporal["segment_number"] == 0,
        "segment_end_date",
    ] = T0_END_DATE

    for segment in range(
        1,
        N_SEGMENTS + 1,
    ):

        segment_start_month = (
            start_month
            + (
                (segment - 1)
                * MONTHS_PER_SEGMENT
            )
        )

        segment_end_month = (
            segment_start_month
            + (
                MONTHS_PER_SEGMENT
                - 1
            )
        )

        segment_start = (
            segment_start_month
            .to_timestamp(
                how="start"
            )
            .normalize()
        )

        segment_end = (
            segment_end_month
            .to_timestamp(
                how="end"
            )
            .normalize()
        )

        mask = (
            temporal[
                "segment_number"
            ]
            == segment
        )

        temporal.loc[
            mask,
            "segment_start_date",
        ] = segment_start

        temporal.loc[
            mask,
            "segment_end_date",
        ] = segment_end

    # -------------------------------------------------------------------------
    # 7. Deterministic within-segment ordering metadata
    #
    # This is NOT a claim that the true event sequence is known.
    #
    # It exists only for reproducible storage/debugging.
    # ITRS treats events within the same temporal segment as unordered.
    # -------------------------------------------------------------------------

    temporal = temporal.sort_values(
        [
            "segment_number",
            DATE_COL,
            "investor_id",
            "funding_round_id",
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    # -------------------------------------------------------------------------
    # 8. Build segment metadata
    # -------------------------------------------------------------------------

    metadata_rows = []

    # T0
    t0 = temporal[
        temporal["segment_number"]
        == 0
    ]

    metadata_rows.append(
        {
            "segment_number": 0,
            "segment_label": "T0",
            "segment_start_date": None,
            "segment_end_date": (
                T0_END_DATE.date()
            ),
            "temporal_role": (
                "compressed_prehistory"
            ),
            "interactions": len(t0),
            "active_investors": (
                t0["investor_id"]
                .nunique()
            ),
            "active_startups": (
                t0["startup_id"]
                .nunique()
            ),
            "active_pairs": (
                t0[
                    [
                        "investor_id",
                        "startup_id",
                    ]
                ]
                .drop_duplicates()
                .shape[0]
            ),
            "funding_rounds": (
                t0["funding_round_id"]
                .nunique()
            ),
        }
    )

    # T1-T60
    for segment in range(
        1,
        N_SEGMENTS + 1,
    ):

        subset = temporal[
            temporal[
                "segment_number"
            ]
            == segment
        ]

        if len(subset) == 0:
            raise ValueError(
                f"T{segment} has zero interactions."
            )

        metadata_rows.append(
            {
                "segment_number": (
                    segment
                ),
                "segment_label": (
                    f"T{segment}"
                ),
                "segment_start_date": (
                    subset[
                        "segment_start_date"
                    ]
                    .iloc[0]
                    .date()
                ),
                "segment_end_date": (
                    subset[
                        "segment_end_date"
                    ]
                    .iloc[0]
                    .date()
                ),
                "temporal_role": (
                    "t60_evaluation_pool"
                    if segment == 60
                    else "detailed_train_history"
                ),
                "interactions": (
                    len(subset)
                ),
                "active_investors": (
                    subset[
                        "investor_id"
                    ]
                    .nunique()
                ),
                "active_startups": (
                    subset[
                        "startup_id"
                    ]
                    .nunique()
                ),
                "active_pairs": (
                    subset[
                        [
                            "investor_id",
                            "startup_id",
                        ]
                    ]
                    .drop_duplicates()
                    .shape[0]
                ),
                "funding_rounds": (
                    subset[
                        "funding_round_id"
                    ]
                    .nunique()
                ),
            }
        )

    metadata = pd.DataFrame(
        metadata_rows
    )

    # -------------------------------------------------------------------------
    # 9. Integrity checks
    # -------------------------------------------------------------------------

    separator("-")
    print("TEMPORAL CONSTRUCTION CHECKS")
    separator("-")

    t0_count = int(
        (
            temporal[
                "segment_number"
            ]
            == 0
        )
        .sum()
    )

    detailed_count = int(
        (
            temporal[
                "segment_number"
            ]
            .between(
                1,
                60,
            )
        )
        .sum()
    )

    t60_count = int(
        (
            temporal[
                "segment_number"
            ]
            == 60
        )
        .sum()
    )

    unique_detailed_segments = (
        temporal.loc[
            temporal[
                "segment_number"
            ]
            .between(
                1,
                60,
            ),
            "segment_number",
        ]
        .nunique()
    )

    print(
        f"Eligible temporal rows:       "
        f"{len(temporal):,}"
    )

    print(
        f"T0 rows:                      "
        f"{t0_count:,}"
    )

    print(
        f"T1-T60 rows:                  "
        f"{detailed_count:,}"
    )

    print(
        f"T60 rows:                     "
        f"{t60_count:,}"
    )

    print(
        f"Detailed segment count:       "
        f"{unique_detailed_segments}"
    )

    print(
        f"Post-endpoint rows:           "
        f"{len(post_endpoint):,}"
    )

    if len(temporal) != EXPECTED_ELIGIBLE_ROWS:
        raise ValueError(
            "Eligible temporal row count "
            "does not match Phase 2.2.2."
        )

    if t0_count != EXPECTED_T0_ROWS:
        raise ValueError(
            "T0 row count does not match "
            "Phase 2.2.2."
        )

    if detailed_count != EXPECTED_DETAILED_ROWS:
        raise ValueError(
            "Detailed-window count does not "
            "match Phase 2.2.2."
        )

    if t60_count != EXPECTED_T60_ROWS:
        raise ValueError(
            "T60 row count does not match "
            "Phase 2.2.2."
        )

    if unique_detailed_segments != 60:
        raise ValueError(
            "Expected exactly 60 detailed segments."
        )

    if (
        len(temporal)
        + len(post_endpoint)
        != EXPECTED_CANONICAL_ROWS
    ):
        raise ValueError(
            "Temporal + post-endpoint rows "
            "do not reconstruct canonical data."
        )

    # -------------------------------------------------------------------------
    # 10. Verify exact important boundaries
    # -------------------------------------------------------------------------

    t1 = metadata[
        metadata[
            "segment_number"
        ] == 1
    ].iloc[0]

    t60 = metadata[
        metadata[
            "segment_number"
        ] == 60
    ].iloc[0]

    print(
        f"\nT1:  "
        f"{t1['segment_start_date']} -> "
        f"{t1['segment_end_date']}"
    )

    print(
        f"T60: "
        f"{t60['segment_start_date']} -> "
        f"{t60['segment_end_date']}"
    )

    if str(
        t1[
            "segment_start_date"
        ]
    ) != "2011-04-01":
        raise ValueError(
            "Unexpected T1 start."
        )

    if str(
        t60[
            "segment_end_date"
        ]
    ) != "2026-03-31":
        raise ValueError(
            "Unexpected T60 end."
        )

    # -------------------------------------------------------------------------
    # 11. Save experimental temporal layer
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporal.to_parquet(
        TEMPORAL_OUTPUT,
        index=False,
    )

    metadata.to_csv(
        SEGMENT_METADATA_OUTPUT,
        index=False,
    )

    post_endpoint.to_parquet(
        POST_ENDPOINT_OUTPUT,
        index=False,
    )

    # -------------------------------------------------------------------------
    # 12. Print metadata summary
    # -------------------------------------------------------------------------

    separator("-")
    print("SELECTED TEMPORAL SEGMENT SUMMARY")
    separator("-")

    print(
        metadata.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # 13. Final status
    # -------------------------------------------------------------------------

    separator()
    print(
        "PHASE 2.3.1 TEMPORAL LAYER COMPLETE"
    )
    separator()

    print(f"""
Outputs written to:

{TEMPORAL_OUTPUT}
{SEGMENT_METADATA_OUTPUT}
{POST_ENDPOINT_OUTPUT}

Selected ITRS temporal structure:

T0:
    <= 2011-03-31

T1-T60:
    2011-04-01 -> 2026-03-31

T60:
    2026-01-01 -> 2026-03-31

The temporal layer currently labels T60 only as an EVALUATION POOL.

No validation/test random split has been performed yet.
No cold-start entity has been removed.
No negative instance has been sampled.
No minimum investor-history criterion has been imposed.
No investment types have been filtered.

The canonical Phase-1 dataset remains unchanged.
""")


if __name__ == "__main__":
    main()