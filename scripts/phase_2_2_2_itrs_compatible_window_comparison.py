from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PHASE 2.2.2 — ITRS-COMPATIBLE 15-YEAR / 60-SEGMENT WINDOW COMPARISON
# =============================================================================
#
# PAPER-GROUNDED DESIGN
# ---------------------
#
# ITRS:
#
#   - uses t = 60 detailed temporal segments;
#   - each detailed segment spans 3 months;
#   - therefore detailed history spans 180 months = 15 years;
#   - investment events older than the 15-year detailed horizon are assigned
#     to T0;
#   - intra-segment event order is ignored;
#   - the final detailed segment is the evaluation period.
#
# This audit compares endpoint choices while KEEPING this temporal
# architecture fixed.
#
# No endpoint is selected here.
# =============================================================================


INPUT_PATH = Path(
    "data/processed/interactions.parquet"
)

BUFFER_AUDIT_PATH = Path(
    "data/experimental/phase_2/audits/"
    "snapshot_maturity_buffer_sensitivity.csv"
)

OUTPUT_DIR = Path(
    "data/experimental/phase_2/audits"
)

WINDOW_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "itrs_compatible_window_summary.csv"
)

SEGMENT_OUTPUT = (
    OUTPUT_DIR
    / "itrs_compatible_segment_diagnostics.csv"
)

INVESTOR_DEPTH_OUTPUT = (
    OUTPUT_DIR
    / "itrs_compatible_investor_segment_depth_summary.csv"
)


EXPECTED_ROWS = 1_208_051

DATE_COL = "announced_on"
INVESTOR_COL = "investor_id"
STARTUP_COL = "startup_id"
ROUND_COL = "funding_round_id"


N_DETAILED_SEGMENTS = 60
MONTHS_PER_SEGMENT = 3

N_DETAILED_MONTHS = (
    N_DETAILED_SEGMENTS
    * MONTHS_PER_SEGMENT
)


def separator(char="=", width=125):
    print(char * width)


def pct(numerator, denominator):

    if denominator == 0:
        return np.nan

    return numerator / denominator * 100


def build_endpoint_scenarios(buffer_audit):

    """
    Collapse buffer scenarios that produce the same eligible month.

    Example:
        7, 14 and 30 days all currently map to 2026-04-30.

    This prevents running identical temporal windows three times while
    preserving which hypothetical buffers correspond to each endpoint.
    """

    temp = buffer_audit.copy()

    temp["latest_eligible_month_end"] = pd.to_datetime(
        temp["latest_eligible_month_end"]
    )

    temp = temp.dropna(
        subset=["latest_eligible_month_end"]
    )

    scenarios = (
        temp.groupby(
            "latest_eligible_month_end",
            as_index=False,
        )
        .agg(
            minimum_buffer_days=(
                "hypothetical_buffer_days",
                "min",
            ),
            maximum_buffer_days=(
                "hypothetical_buffer_days",
                "max",
            ),
            buffer_scenarios=(
                "hypothetical_buffer_days",
                lambda x: "/".join(
                    str(int(v))
                    for v in sorted(set(x))
                ),
            ),
        )
        .sort_values(
            "latest_eligible_month_end",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return scenarios


def assign_itrs_segments(
    df,
    detailed_start_month,
    detailed_end_month,
):

    """
    Assign:

        T0      = everything before detailed_start_month
        T1-T60  = exactly 60 consecutive 3-month blocks

    Anything after detailed_end_month must already have been excluded.
    """

    result = df.copy()

    event_month = (
        result[DATE_COL]
        .dt.to_period("M")
    )

    start_ordinal = (
        detailed_start_month.ordinal
    )

    month_ordinal = (
        event_month.map(
            lambda p: p.ordinal
        )
    )

    result["month_offset_from_detailed_start"] = (
        month_ordinal - start_ordinal
    )

    result["segment_number"] = 0

    detailed_mask = (
        result[
            "month_offset_from_detailed_start"
        ] >= 0
    )

    result.loc[
        detailed_mask,
        "segment_number",
    ] = (
        result.loc[
            detailed_mask,
            "month_offset_from_detailed_start",
        ]
        // MONTHS_PER_SEGMENT
        + 1
    )

    result["segment_number"] = (
        result["segment_number"]
        .astype("int16")
    )

    invalid = result[
        result["segment_number"]
        > N_DETAILED_SEGMENTS
    ]

    if len(invalid) > 0:

        raise ValueError(
            "Found events assigned beyond T60. "
            "Check detailed-window construction."
        )

    return result


def main():

    separator()
    print(
        "PHASE 2.2.2 — "
        "ITRS-COMPATIBLE 15-YEAR / 60-SEGMENT WINDOW COMPARISON"
    )
    separator()

    # -------------------------------------------------------------------------
    # 1. Load canonical interaction data
    # -------------------------------------------------------------------------

    df = pd.read_parquet(
        INPUT_PATH,
        columns=[
            "interaction_id",
            DATE_COL,
            INVESTOR_COL,
            STARTUP_COL,
            ROUND_COL,
        ],
    )

    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} rows; "
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
        f"Canonical date range:   "
        f"{df[DATE_COL].min().date()} -> "
        f"{df[DATE_COL].max().date()}"
    )

    # -------------------------------------------------------------------------
    # 2. Load data-derived endpoint scenarios from Phase 2.1.6
    # -------------------------------------------------------------------------

    buffer_audit = pd.read_csv(
        BUFFER_AUDIT_PATH
    )

    scenarios = build_endpoint_scenarios(
        buffer_audit
    )

    separator("-")
    print("UNIQUE ENDPOINT SCENARIOS")
    separator("-")

    print(
        scenarios.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # 3. Evaluate each endpoint under the exact 15-year design
    # -------------------------------------------------------------------------

    summary_rows = []
    segment_rows = []
    depth_rows = []

    for _, scenario in scenarios.iterrows():

        end_date = pd.Timestamp(
            scenario[
                "latest_eligible_month_end"
            ]
        )

        end_month = (
            end_date.to_period("M")
        )

        # -------------------------------------------------------------
        # Exactly 180 inclusive months.
        #
        # Example:
        #
        # end = 2026-03
        #
        # start = 2011-04
        #
        # Apr 2011 ... Mar 2026 = exactly 180 months.
        # -------------------------------------------------------------

        detailed_start_month = (
            end_month
            - (N_DETAILED_MONTHS - 1)
        )

        detailed_start_date = (
            detailed_start_month
            .to_timestamp(how="start")
            .normalize()
        )

        # -------------------------------------------------------------
        # Use all historical events through endpoint.
        #
        # Pre-detailed-window events remain available for T0.
        # -------------------------------------------------------------

        eligible = (
            df[
                df[DATE_COL]
                <= end_date
            ]
            .copy()
        )

        segmented = assign_itrs_segments(
            eligible,
            detailed_start_month,
            end_month,
        )

        # -------------------------------------------------------------
        # Validate exact segment structure
        # -------------------------------------------------------------

        detailed_events = segmented[
            segmented["segment_number"] >= 1
        ]

        t0_events = segmented[
            segmented["segment_number"] == 0
        ]

        observed_detailed_segments = sorted(
            detailed_events[
                "segment_number"
            ]
            .unique()
        )

        expected_segments = list(
            range(
                1,
                N_DETAILED_SEGMENTS + 1,
            )
        )

        if observed_detailed_segments != expected_segments:

            missing_segments = sorted(
                set(expected_segments)
                - set(observed_detailed_segments)
            )

        else:

            missing_segments = []

        # -------------------------------------------------------------
        # Build explicit segment diagnostics T1 ... T60
        # -------------------------------------------------------------

        scenario_segment_rows = []

        for segment_number in expected_segments:

            segment_start_month = (
                detailed_start_month
                + (
                    (segment_number - 1)
                    * MONTHS_PER_SEGMENT
                )
            )

            segment_end_month = (
                segment_start_month
                + (
                    MONTHS_PER_SEGMENT - 1
                )
            )

            segment_start_date = (
                segment_start_month
                .to_timestamp(how="start")
                .normalize()
            )

            segment_end_date = (
                segment_end_month
                .to_timestamp(how="end")
                .normalize()
            )

            segment_df = segmented[
                segmented["segment_number"]
                == segment_number
            ]

            segment_rows_current = {
                "endpoint": end_date.date(),
                "buffer_scenarios": scenario[
                    "buffer_scenarios"
                ],
                "detailed_start_date": (
                    detailed_start_date.date()
                ),
                "segment_number": (
                    segment_number
                ),
                "segment_label": (
                    f"T{segment_number}"
                ),
                "segment_start_date": (
                    segment_start_date.date()
                ),
                "segment_end_date": (
                    segment_end_date.date()
                ),
                "interactions": (
                    len(segment_df)
                ),
                "active_investors": (
                    segment_df[
                        INVESTOR_COL
                    ]
                    .nunique()
                ),
                "active_startups": (
                    segment_df[
                        STARTUP_COL
                    ]
                    .nunique()
                ),
                "active_pairs": (
                    segment_df[
                        [
                            INVESTOR_COL,
                            STARTUP_COL,
                        ]
                    ]
                    .drop_duplicates()
                    .shape[0]
                ),
                "funding_rounds": (
                    segment_df[
                        ROUND_COL
                    ]
                    .nunique()
                ),
            }

            scenario_segment_rows.append(
                segment_rows_current
            )

            segment_rows.append(
                segment_rows_current
            )

        segment_table = pd.DataFrame(
            scenario_segment_rows
        )

        # -------------------------------------------------------------
        # T0 diagnostics
        # -------------------------------------------------------------

        t0_interactions = len(
            t0_events
        )

        t0_investors = (
            t0_events[
                INVESTOR_COL
            ]
            .nunique()
        )

        t0_startups = (
            t0_events[
                STARTUP_COL
            ]
            .nunique()
        )

        # Investors active in the detailed horizon.
        detailed_investor_ids = (
            detailed_events[
                INVESTOR_COL
            ]
            .unique()
        )

        t0_investor_ids = set(
            t0_events[
                INVESTOR_COL
            ]
            .unique()
        )

        detailed_investors_with_t0_history = sum(
            investor_id in t0_investor_ids
            for investor_id
            in detailed_investor_ids
        )

        # -------------------------------------------------------------
        # Detailed segment density
        # -------------------------------------------------------------

        segment_interactions = (
            segment_table[
                "interactions"
            ]
        )

        zero_detailed_segments = int(
            (
                segment_interactions == 0
            ).sum()
        )

        # -------------------------------------------------------------
        # Training/history and final-segment diagnostics
        # -------------------------------------------------------------
        #
        # Paper notation around T0 vs "first t-1 fragments" is slightly
        # ambiguous.
        #
        # Therefore we separately report:
        #
        #   T0 prehistory
        #   T1-T59 detailed training fragments
        #   T60 final evaluation fragment
        #
        # rather than silently merging them into one interpretation.
        # -------------------------------------------------------------

        detailed_train = segmented[
            segmented["segment_number"]
            .between(
                1,
                N_DETAILED_SEGMENTS - 1,
            )
        ]

        final_segment = segmented[
            segmented["segment_number"]
            == N_DETAILED_SEGMENTS
        ]

        final_start_month = (
            detailed_start_month
            + (
                (
                    N_DETAILED_SEGMENTS - 1
                )
                * MONTHS_PER_SEGMENT
            )
        )

        final_start_date = (
            final_start_month
            .to_timestamp(how="start")
            .normalize()
        )

        # Everything strictly before T60.
        all_pre_final_history = segmented[
            segmented[DATE_COL]
            < final_start_date
        ]

        # -------------------------------------------------------------
        # Final-segment investor cold-start diagnostics
        # -------------------------------------------------------------

        final_investor_ids = set(
            final_segment[
                INVESTOR_COL
            ]
            .unique()
        )

        prior_investor_ids = set(
            all_pre_final_history[
                INVESTOR_COL
            ]
            .unique()
        )

        cold_start_final_investors = (
            final_investor_ids
            - prior_investor_ids
        )

        final_events_from_cold_investors = (
            final_segment[
                final_segment[
                    INVESTOR_COL
                ]
                .isin(
                    cold_start_final_investors
                )
            ]
        )

        # -------------------------------------------------------------
        # Final-segment startup cold-start diagnostics
        # -------------------------------------------------------------

        final_startup_ids = set(
            final_segment[
                STARTUP_COL
            ]
            .unique()
        )

        prior_startup_ids = set(
            all_pre_final_history[
                STARTUP_COL
            ]
            .unique()
        )

        cold_start_final_startups = (
            final_startup_ids
            - prior_startup_ids
        )

        final_events_to_cold_startups = (
            final_segment[
                final_segment[
                    STARTUP_COL
                ]
                .isin(
                    cold_start_final_startups
                )
            ]
        )

        # -------------------------------------------------------------
        # Investor temporal-depth distribution
        # -------------------------------------------------------------

        investor_segment_depth = (
            detailed_events[
                [
                    INVESTOR_COL,
                    "segment_number",
                ]
            ]
            .drop_duplicates()
            .groupby(
                INVESTOR_COL,
                observed=True,
            )
            .size()
        )

        depth_row = {
            "endpoint": end_date.date(),
            "buffer_scenarios": scenario[
                "buffer_scenarios"
            ],
            "detailed_start_date": (
                detailed_start_date.date()
            ),
            "investors_in_detailed_window": (
                len(investor_segment_depth)
            ),
            "median_active_detailed_segments": (
                investor_segment_depth.median()
            ),
            "active_segments_q75": (
                investor_segment_depth.quantile(
                    0.75
                )
            ),
            "active_segments_q90": (
                investor_segment_depth.quantile(
                    0.90
                )
            ),
            "active_segments_q95": (
                investor_segment_depth.quantile(
                    0.95
                )
            ),
            "active_segments_q99": (
                investor_segment_depth.quantile(
                    0.99
                )
            ),
            "investors_2plus_segments": (
                (
                    investor_segment_depth >= 2
                ).sum()
            ),
            "investors_2plus_segments_share_pct": pct(
                (
                    investor_segment_depth >= 2
                ).sum(),
                len(investor_segment_depth),
            ),
            "investors_3plus_segments": (
                (
                    investor_segment_depth >= 3
                ).sum()
            ),
            "investors_3plus_segments_share_pct": pct(
                (
                    investor_segment_depth >= 3
                ).sum(),
                len(investor_segment_depth),
            ),
            "investors_5plus_segments": (
                (
                    investor_segment_depth >= 5
                ).sum()
            ),
            "investors_5plus_segments_share_pct": pct(
                (
                    investor_segment_depth >= 5
                ).sum(),
                len(investor_segment_depth),
            ),
            "investors_10plus_segments": (
                (
                    investor_segment_depth >= 10
                ).sum()
            ),
            "investors_10plus_segments_share_pct": pct(
                (
                    investor_segment_depth >= 10
                ).sum(),
                len(investor_segment_depth),
            ),
        }

        depth_rows.append(
            depth_row
        )

        # -------------------------------------------------------------
        # Final-segment relative activity
        # -------------------------------------------------------------

        prior_8_segments = (
            segment_table[
                segment_table[
                    "segment_number"
                ]
                .between(
                    52,
                    59,
                )
            ][
                "interactions"
            ]
        )

        prior_8_median = (
            prior_8_segments.median()
        )

        final_segment_interactions = (
            len(final_segment)
        )

        final_vs_prior8_median_pct = pct(
            final_segment_interactions,
            prior_8_median,
        )

        # -------------------------------------------------------------
        # Calendar-quarter alignment
        # -------------------------------------------------------------

        calendar_quarter_aligned = (
            detailed_start_month.month
            in {1, 4, 7, 10}
            and
            end_month.month
            in {3, 6, 9, 12}
        )

        # -------------------------------------------------------------
        # Validation/test feasibility diagnostic
        #
        # Paper says 10% of T_t is validation, remainder test.
        #
        # This does NOT perform the split.
        # -------------------------------------------------------------

        approximate_validation_events = int(
            np.floor(
                final_segment_interactions
                * 0.10
            )
        )

        approximate_test_events = (
            final_segment_interactions
            - approximate_validation_events
        )

        # -------------------------------------------------------------
        # Scenario summary
        # -------------------------------------------------------------

        summary_rows.append(
            {
                "endpoint": (
                    end_date.date()
                ),

                "buffer_scenarios": (
                    scenario[
                        "buffer_scenarios"
                    ]
                ),

                "minimum_buffer_days": int(
                    scenario[
                        "minimum_buffer_days"
                    ]
                ),

                "maximum_buffer_days": int(
                    scenario[
                        "maximum_buffer_days"
                    ]
                ),

                "detailed_start_date": (
                    detailed_start_date.date()
                ),

                "detailed_start_month": str(
                    detailed_start_month
                ),

                "detailed_end_month": str(
                    end_month
                ),

                "detailed_months": (
                    N_DETAILED_MONTHS
                ),

                "detailed_segments": (
                    N_DETAILED_SEGMENTS
                ),

                "months_per_segment": (
                    MONTHS_PER_SEGMENT
                ),

                "calendar_quarter_aligned": (
                    calendar_quarter_aligned
                ),

                # All history through endpoint
                "eligible_interactions_through_endpoint": (
                    len(eligible)
                ),

                # T0
                "t0_interactions": (
                    t0_interactions
                ),

                "t0_interaction_share_pct": pct(
                    t0_interactions,
                    len(eligible),
                ),

                "t0_investors": (
                    t0_investors
                ),

                "t0_startups": (
                    t0_startups
                ),

                # Detailed horizon
                "detailed_window_interactions": (
                    len(detailed_events)
                ),

                "detailed_window_interaction_share_pct": pct(
                    len(detailed_events),
                    len(eligible),
                ),

                "detailed_window_investors": (
                    detailed_events[
                        INVESTOR_COL
                    ]
                    .nunique()
                ),

                "detailed_window_startups": (
                    detailed_events[
                        STARTUP_COL
                    ]
                    .nunique()
                ),

                "detailed_investors_with_t0_history": (
                    detailed_investors_with_t0_history
                ),

                "detailed_investors_with_t0_history_share_pct": pct(
                    detailed_investors_with_t0_history,
                    len(
                        detailed_investor_ids
                    ),
                ),

                # Detailed segment structure
                "zero_detailed_segments": (
                    zero_detailed_segments
                ),

                "min_segment_interactions": int(
                    segment_interactions.min()
                ),

                "median_segment_interactions": float(
                    segment_interactions.median()
                ),

                "mean_segment_interactions": float(
                    segment_interactions.mean()
                ),

                "max_segment_interactions": int(
                    segment_interactions.max()
                ),

                # Training-period diagnostics
                "t1_to_t59_interactions": (
                    len(detailed_train)
                ),

                "t0_plus_t1_to_t59_history_interactions": (
                    len(all_pre_final_history)
                ),

                # Final T60
                "t60_start_date": (
                    final_start_date.date()
                ),

                "t60_end_date": (
                    end_date.date()
                ),

                "t60_interactions": (
                    final_segment_interactions
                ),

                "t60_active_investors": (
                    final_segment[
                        INVESTOR_COL
                    ]
                    .nunique()
                ),

                "t60_active_startups": (
                    final_segment[
                        STARTUP_COL
                    ]
                    .nunique()
                ),

                "t60_active_pairs": (
                    final_segment[
                        [
                            INVESTOR_COL,
                            STARTUP_COL,
                        ]
                    ]
                    .drop_duplicates()
                    .shape[0]
                ),

                "t60_funding_rounds": (
                    final_segment[
                        ROUND_COL
                    ]
                    .nunique()
                ),

                "t60_vs_prior8_segment_median_pct": (
                    final_vs_prior8_median_pct
                ),

                # Cold-start
                "t60_cold_start_investors": (
                    len(
                        cold_start_final_investors
                    )
                ),

                "t60_cold_start_investor_share_pct": pct(
                    len(
                        cold_start_final_investors
                    ),
                    len(
                        final_investor_ids
                    ),
                ),

                "t60_events_from_cold_start_investors": (
                    len(
                        final_events_from_cold_investors
                    )
                ),

                "t60_cold_start_startups": (
                    len(
                        cold_start_final_startups
                    )
                ),

                "t60_cold_start_startup_share_pct": pct(
                    len(
                        cold_start_final_startups
                    ),
                    len(
                        final_startup_ids
                    ),
                ),

                "t60_events_to_cold_start_startups": (
                    len(
                        final_events_to_cold_startups
                    )
                ),

                # Approximate paper-style split
                "approx_validation_events_floor_10pct": (
                    approximate_validation_events
                ),

                "approx_test_events_remaining": (
                    approximate_test_events
                ),

                # Integrity
                "missing_detailed_segments": (
                    ",".join(
                        str(v)
                        for v in missing_segments
                    )
                ),
            }
        )

    # -------------------------------------------------------------------------
    # 4. Build output tables
    # -------------------------------------------------------------------------

    summary = pd.DataFrame(
        summary_rows
    )

    segments = pd.DataFrame(
        segment_rows
    )

    depth = pd.DataFrame(
        depth_rows
    )

    # -------------------------------------------------------------------------
    # 5. Save outputs
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        WINDOW_SUMMARY_OUTPUT,
        index=False,
    )

    segments.to_csv(
        SEGMENT_OUTPUT,
        index=False,
    )

    depth.to_csv(
        INVESTOR_DEPTH_OUTPUT,
        index=False,
    )

    # -------------------------------------------------------------------------
    # 6. Print main window comparison
    # -------------------------------------------------------------------------

    separator("-")
    print(
        "ITRS-COMPATIBLE WINDOW COMPARISON"
    )
    separator("-")

    display_columns = [
        "endpoint",
        "buffer_scenarios",
        "detailed_start_date",
        "calendar_quarter_aligned",

        "eligible_interactions_through_endpoint",

        "t0_interactions",
        "t0_interaction_share_pct",

        "detailed_window_interactions",
        "detailed_window_interaction_share_pct",

        "detailed_investors_with_t0_history_share_pct",

        "zero_detailed_segments",
        "min_segment_interactions",
        "median_segment_interactions",

        "t60_interactions",
        "t60_vs_prior8_segment_median_pct",

        "t60_cold_start_investor_share_pct",
        "t60_cold_start_startup_share_pct",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    # -------------------------------------------------------------------------
    # 7. Print investor temporal-depth comparison
    # -------------------------------------------------------------------------

    separator("-")
    print(
        "INVESTOR TEMPORAL-SEGMENT DEPTH"
    )
    separator("-")

    print(
        depth.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    # -------------------------------------------------------------------------
    # 8. Print T60 evaluation feasibility
    # -------------------------------------------------------------------------

    separator("-")
    print(
        "FINAL T60 EVALUATION-PERIOD DIAGNOSTICS"
    )
    separator("-")

    t60_columns = [
        "endpoint",
        "buffer_scenarios",
        "t60_start_date",
        "t60_end_date",

        "t60_interactions",
        "t60_active_investors",
        "t60_active_startups",
        "t60_funding_rounds",

        "t60_vs_prior8_segment_median_pct",

        "t60_cold_start_investors",
        "t60_cold_start_investor_share_pct",
        "t60_events_from_cold_start_investors",

        "t60_cold_start_startups",
        "t60_cold_start_startup_share_pct",
        "t60_events_to_cold_start_startups",

        "approx_validation_events_floor_10pct",
        "approx_test_events_remaining",
    ]

    print(
        summary[
            t60_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    # -------------------------------------------------------------------------
    # 9. Print segment counts for each endpoint
    # -------------------------------------------------------------------------

    separator("-")
    print(
        "LAST 8 DETAILED SEGMENTS FOR EACH ENDPOINT"
    )
    separator("-")

    for endpoint in summary["endpoint"]:

        endpoint_str = str(
            endpoint
        )

        subset = segments[
            segments[
                "endpoint"
            ].astype(str)
            == endpoint_str
        ]

        subset = subset.tail(8)

        print(
            f"\nENDPOINT: {endpoint_str}"
        )

        print(
            subset[
                [
                    "segment_label",
                    "segment_start_date",
                    "segment_end_date",
                    "interactions",
                    "active_investors",
                    "active_startups",
                    "funding_rounds",
                ]
            ].to_string(
                index=False
            )
        )

    # -------------------------------------------------------------------------
    # 10. Integrity checks
    # -------------------------------------------------------------------------

    separator("-")
    print(
        "STRUCTURAL INTEGRITY CHECKS"
    )
    separator("-")

    for _, row in summary.iterrows():

        print(
            f"\nEndpoint {row['endpoint']}:"
        )

        print(
            f"  Detailed months:          "
            f"{int(row['detailed_months'])}"
        )

        print(
            f"  Detailed segments:        "
            f"{int(row['detailed_segments'])}"
        )

        print(
            f"  Months per segment:       "
            f"{int(row['months_per_segment'])}"
        )

        print(
            f"  Missing detailed segments:"
            f" {row['missing_detailed_segments']}"
        )

        reconstructed = (
            int(
                row[
                    "t0_interactions"
                ]
            )
            +
            int(
                row[
                    "detailed_window_interactions"
                ]
            )
        )

        expected = int(
            row[
                "eligible_interactions_through_endpoint"
            ]
        )

        print(
            f"  T0 + detailed events:     "
            f"{reconstructed:,}"
        )

        print(
            f"  Eligible interactions:    "
            f"{expected:,}"
        )

        if reconstructed != expected:
            raise ValueError(
                "T0 + detailed horizon does not "
                "reconstruct endpoint-eligible interactions."
            )

        if int(
            row[
                "detailed_segments"
            ]
        ) != 60:
            raise ValueError(
                "Detailed temporal design is not exactly 60 segments."
            )

        if int(
            row[
                "months_per_segment"
            ]
        ) != 3:
            raise ValueError(
                "Detailed segment duration is not exactly 3 months."
            )

    # -------------------------------------------------------------------------
    # 11. End
    # -------------------------------------------------------------------------

    separator()
    print(
        "PHASE 2.2.2 AUDIT COMPLETE"
    )
    separator()

    print(f"""
Outputs written to:

{WINDOW_SUMMARY_OUTPUT}
{SEGMENT_OUTPUT}
{INVESTOR_DEPTH_OUTPUT}

IMPORTANT:

Every candidate uses the SAME paper-grounded architecture:

    T0       = all pre-history before the detailed horizon
    T1-T60   = 60 consecutive segments
    duration = 3 months per detailed segment
    horizon  = exactly 15 years

This audit does NOT:
- select the final endpoint,
- assume that 60 days is the true Crunchbase reporting lag,
- remove cold-start investors or startups,
- perform the validation/test random split,
- sample negative instances,
- choose a minimum investor-history threshold,
- filter investment types,
- modify data/processed/interactions.parquet.

The next step is to choose the most defensible ITRS-compatible temporal
construction from the observed comparison.
""")


if __name__ == "__main__":
    main()