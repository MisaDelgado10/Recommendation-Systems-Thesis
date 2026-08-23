from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PHASE 2.1.7.1 — CANDIDATE START-YEAR COVERAGE AUDIT
# =============================================================================
#
# PURPOSE
# -------
# Evaluate the consequences of every possible calendar-year starting boundary.
#
# This script DOES NOT select a temporal cutoff.
#
# For isolation of the starting-boundary problem, the audit uses May 31, 2026
# as a REFERENCE ENDPOINT because Phase 2.1.4 established it as the latest
# calendar-complete month.
#
# Phase 2.1.6 separately established that recent database maturity remains
# uncertain because actual Crunchbase reporting lag is not observable.
#
# Therefore:
#
#     REFERENCE_END_DATE != final experimental endpoint
#
# =============================================================================


INPUT_PATH = Path(
    "data/processed/interactions.parquet"
)

OUTPUT_DIR = Path(
    "data/experimental/phase_2/audits"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "candidate_start_year_diagnostics.csv"
)

EXPECTED_CANONICAL_ROWS = 1_208_051

DATE_COL = "announced_on"
INVESTOR_COL = "investor_id"
STARTUP_COL = "startup_id"
ROUND_COL = "funding_round_id"


# Latest calendar-complete month.
#
# This is a reference endpoint only.
# It is NOT the final model endpoint.
REFERENCE_END_DATE = pd.Timestamp(
    "2026-05-31"
)


def separator(char="=", width=120):
    print(char * width)


def pct(numerator, denominator):

    if denominator == 0:
        return np.nan

    return (
        numerator
        / denominator
        * 100
    )


def main():

    separator()
    print(
        "PHASE 2.1.7.1 — "
        "CANDIDATE START-YEAR COVERAGE AUDIT"
    )
    separator()

    # -------------------------------------------------------------------------
    # 1. Load canonical interactions
    # -------------------------------------------------------------------------

    columns = [
        "interaction_id",
        DATE_COL,
        INVESTOR_COL,
        STARTUP_COL,
        ROUND_COL,
    ]

    df = pd.read_parquet(
        INPUT_PATH,
        columns=columns,
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
    # 2. Build reference calendar-complete view
    # -------------------------------------------------------------------------

    reference_df = (
        df[
            df[DATE_COL]
            <= REFERENCE_END_DATE
        ]
        .copy()
    )

    excluded_after_reference = (
        len(df)
        - len(reference_df)
    )

    reference_interactions = len(
        reference_df
    )

    reference_investors = (
        reference_df[INVESTOR_COL]
        .nunique()
    )

    reference_startups = (
        reference_df[STARTUP_COL]
        .nunique()
    )

    reference_pairs = (
        reference_df[
            [
                INVESTOR_COL,
                STARTUP_COL,
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    reference_rounds = (
        reference_df[ROUND_COL]
        .nunique()
    )

    print(
        f"\nReference endpoint:     "
        f"{REFERENCE_END_DATE.date()}"
    )

    print(
        f"Reference interactions: "
        f"{reference_interactions:,}"
    )

    print(
        f"Excluded after reference endpoint: "
        f"{excluded_after_reference:,}"
    )

    # -------------------------------------------------------------------------
    # 3. Add month / quarter variables
    # -------------------------------------------------------------------------

    reference_df["month"] = (
        reference_df[DATE_COL]
        .dt.to_period("M")
    )

    reference_df["quarter"] = (
        reference_df[DATE_COL]
        .dt.to_period("Q")
    )

    # -------------------------------------------------------------------------
    # 4. Precompute lifecycle information
    #
    # This makes left-censoring calculations explicit.
    # -------------------------------------------------------------------------

    investor_lifecycle = (
        reference_df.groupby(
            INVESTOR_COL,
            observed=True,
        )[DATE_COL]
        .agg(
            first_event_date="min",
            last_event_date="max",
        )
    )

    startup_lifecycle = (
        reference_df.groupby(
            STARTUP_COL,
            observed=True,
        )[DATE_COL]
        .agg(
            first_event_date="min",
            last_event_date="max",
        )
    )

    # -------------------------------------------------------------------------
    # 5. Full monthly / quarterly activity vectors
    # -------------------------------------------------------------------------

    min_month = (
        reference_df[DATE_COL]
        .min()
        .to_period("M")
    )

    max_month = (
        REFERENCE_END_DATE
        .to_period("M")
    )

    full_month_index = pd.period_range(
        min_month,
        max_month,
        freq="M",
    )

    monthly_counts = (
        reference_df.groupby(
            "month",
            observed=True,
        )
        .size()
        .reindex(
            full_month_index,
            fill_value=0,
        )
    )

    min_quarter = (
        reference_df[DATE_COL]
        .min()
        .to_period("Q")
    )

    max_quarter = (
        REFERENCE_END_DATE
        .to_period("Q")
    )

    full_quarter_index = pd.period_range(
        min_quarter,
        max_quarter,
        freq="Q",
    )

    quarterly_counts = (
        reference_df.groupby(
            "quarter",
            observed=True,
        )
        .size()
        .reindex(
            full_quarter_index,
            fill_value=0,
        )
    )

    # -------------------------------------------------------------------------
    # 6. Objective continuity landmarks
    #
    # These are descriptive properties, NOT cutoff recommendations.
    # -------------------------------------------------------------------------

    zero_months = monthly_counts[
        monthly_counts == 0
    ]

    zero_quarters = quarterly_counts[
        quarterly_counts == 0
    ]

    if len(zero_months) > 0:

        last_zero_month = (
            zero_months.index.max()
        )

        continuous_monthly_start = (
            last_zero_month + 1
        )

    else:

        last_zero_month = None
        continuous_monthly_start = (
            monthly_counts.index.min()
        )

    if len(zero_quarters) > 0:

        last_zero_quarter = (
            zero_quarters.index.max()
        )

        continuous_quarterly_start = (
            last_zero_quarter + 1
        )

    else:

        last_zero_quarter = None
        continuous_quarterly_start = (
            quarterly_counts.index.min()
        )

    # -------------------------------------------------------------------------
    # 7. Evaluate every calendar-year starting boundary
    # -------------------------------------------------------------------------

    first_year = (
        reference_df[DATE_COL]
        .dt.year
        .min()
    )

    final_year = (
        REFERENCE_END_DATE.year
    )

    candidate_rows = []

    for start_year in range(
        first_year,
        final_year + 1,
    ):

        start_date = pd.Timestamp(
            year=start_year,
            month=1,
            day=1,
        )

        window = reference_df[
            reference_df[DATE_COL]
            >= start_date
        ]

        if len(window) == 0:
            continue

        # ---------------------------------------------------------------------
        # Dataset coverage
        # ---------------------------------------------------------------------

        interactions = len(window)

        active_investors = (
            window[INVESTOR_COL]
            .nunique()
        )

        active_startups = (
            window[STARTUP_COL]
            .nunique()
        )

        active_pairs = (
            window[
                [
                    INVESTOR_COL,
                    STARTUP_COL,
                ]
            ]
            .drop_duplicates()
            .shape[0]
        )

        active_rounds = (
            window[ROUND_COL]
            .nunique()
        )

        # ---------------------------------------------------------------------
        # Calendar continuity
        # ---------------------------------------------------------------------

        candidate_month_index = (
            pd.period_range(
                start=start_date.to_period("M"),
                end=max_month,
                freq="M",
            )
        )

        candidate_month_counts = (
            monthly_counts
            .reindex(
                candidate_month_index,
                fill_value=0,
            )
        )

        candidate_quarter_index = (
            pd.period_range(
                start=start_date.to_period("Q"),
                end=max_quarter,
                freq="Q",
            )
        )

        candidate_quarter_counts = (
            quarterly_counts
            .reindex(
                candidate_quarter_index,
                fill_value=0,
            )
        )

        # ---------------------------------------------------------------------
        # Investor event sequence lengths INSIDE candidate window
        # ---------------------------------------------------------------------

        investor_event_counts = (
            window.groupby(
                INVESTOR_COL,
                observed=True,
            )
            .size()
        )

        sequence_median = (
            investor_event_counts
            .median()
        )

        sequence_q75 = (
            investor_event_counts
            .quantile(0.75)
        )

        sequence_q90 = (
            investor_event_counts
            .quantile(0.90)
        )

        sequence_q95 = (
            investor_event_counts
            .quantile(0.95)
        )

        sequence_q99 = (
            investor_event_counts
            .quantile(0.99)
        )

        sequence_max = (
            investor_event_counts
            .max()
        )

        # ---------------------------------------------------------------------
        # Sequence-length diagnostics
        #
        # These thresholds are descriptive.
        # They are NOT eligibility criteria.
        # ---------------------------------------------------------------------

        exactly_1 = (
            investor_event_counts == 1
        )

        at_least_2 = (
            investor_event_counts >= 2
        )

        at_least_3 = (
            investor_event_counts >= 3
        )

        at_least_5 = (
            investor_event_counts >= 5
        )

        at_least_10 = (
            investor_event_counts >= 10
        )

        # ---------------------------------------------------------------------
        # Interaction mass contributed by richer histories
        # ---------------------------------------------------------------------

        interactions_from_2plus = (
            investor_event_counts[
                at_least_2
            ]
            .sum()
        )

        interactions_from_3plus = (
            investor_event_counts[
                at_least_3
            ]
            .sum()
        )

        interactions_from_5plus = (
            investor_event_counts[
                at_least_5
            ]
            .sum()
        )

        interactions_from_10plus = (
            investor_event_counts[
                at_least_10
            ]
            .sum()
        )

        # ---------------------------------------------------------------------
        # Active-month depth
        #
        # This guards against interpreting several events on one date/month
        # as a long temporal history.
        # ---------------------------------------------------------------------

        investor_active_month_counts = (
            window[
                [
                    INVESTOR_COL,
                    "month",
                ]
            ]
            .drop_duplicates()
            .groupby(
                INVESTOR_COL,
                observed=True,
            )
            .size()
        )

        active_month_median = (
            investor_active_month_counts
            .median()
        )

        active_month_q75 = (
            investor_active_month_counts
            .quantile(0.75)
        )

        active_month_q90 = (
            investor_active_month_counts
            .quantile(0.90)
        )

        investors_2plus_active_months = (
            (
                investor_active_month_counts
                >= 2
            )
            .sum()
        )

        investors_3plus_active_months = (
            (
                investor_active_month_counts
                >= 3
            )
            .sum()
        )

        investors_5plus_active_months = (
            (
                investor_active_month_counts
                >= 5
            )
            .sum()
        )

        # ---------------------------------------------------------------------
        # Left censoring
        #
        # Investors are left-censored if:
        #
        # 1. they are active inside this candidate window, AND
        # 2. their first known canonical event occurred before the window.
        # ---------------------------------------------------------------------

        active_lifecycle = (
            investor_lifecycle[
                investor_lifecycle[
                    "last_event_date"
                ]
                >= start_date
            ]
        )

        investors_with_pre_window_history = (
            (
                active_lifecycle[
                    "first_event_date"
                ]
                < start_date
            )
            .sum()
        )

        # Same concept for startups, useful for candidate-space interpretation.
        active_startup_lifecycle = (
            startup_lifecycle[
                startup_lifecycle[
                    "last_event_date"
                ]
                >= start_date
            ]
        )

        startups_with_pre_window_history = (
            (
                active_startup_lifecycle[
                    "first_event_date"
                ]
                < start_date
            )
            .sum()
        )

        # ---------------------------------------------------------------------
        # Assemble candidate row
        # ---------------------------------------------------------------------

        candidate_rows.append(
            {
                "start_year": start_year,
                "start_date": start_date.date(),
                "reference_end_date": (
                    REFERENCE_END_DATE.date()
                ),

                "window_months": (
                    len(candidate_month_index)
                ),

                "window_quarters": (
                    len(candidate_quarter_index)
                ),

                "interactions": interactions,

                "interaction_retention_pct": pct(
                    interactions,
                    reference_interactions,
                ),

                "active_investors": (
                    active_investors
                ),

                "investor_retention_pct": pct(
                    active_investors,
                    reference_investors,
                ),

                "active_startups": (
                    active_startups
                ),

                "startup_retention_pct": pct(
                    active_startups,
                    reference_startups,
                ),

                "active_pairs": (
                    active_pairs
                ),

                "pair_retention_pct": pct(
                    active_pairs,
                    reference_pairs,
                ),

                "funding_rounds": (
                    active_rounds
                ),

                "funding_round_retention_pct": pct(
                    active_rounds,
                    reference_rounds,
                ),

                # Calendar continuity
                "zero_interaction_months": int(
                    (
                        candidate_month_counts
                        == 0
                    ).sum()
                ),

                "min_monthly_interactions": int(
                    candidate_month_counts.min()
                ),

                "median_monthly_interactions": float(
                    candidate_month_counts.median()
                ),

                "zero_interaction_quarters": int(
                    (
                        candidate_quarter_counts
                        == 0
                    ).sum()
                ),

                "min_quarterly_interactions": int(
                    candidate_quarter_counts.min()
                ),

                "median_quarterly_interactions": float(
                    candidate_quarter_counts.median()
                ),

                # Sequence distribution
                "median_sequence_length": float(
                    sequence_median
                ),

                "sequence_q75": float(
                    sequence_q75
                ),

                "sequence_q90": float(
                    sequence_q90
                ),

                "sequence_q95": float(
                    sequence_q95
                ),

                "sequence_q99": float(
                    sequence_q99
                ),

                "max_sequence_length": int(
                    sequence_max
                ),

                "single_event_investors": int(
                    exactly_1.sum()
                ),

                "single_event_investor_share_pct": pct(
                    exactly_1.sum(),
                    active_investors,
                ),

                "investors_2plus_events": int(
                    at_least_2.sum()
                ),

                "investors_2plus_events_share_pct": pct(
                    at_least_2.sum(),
                    active_investors,
                ),

                "investors_3plus_events": int(
                    at_least_3.sum()
                ),

                "investors_3plus_events_share_pct": pct(
                    at_least_3.sum(),
                    active_investors,
                ),

                "investors_5plus_events": int(
                    at_least_5.sum()
                ),

                "investors_5plus_events_share_pct": pct(
                    at_least_5.sum(),
                    active_investors,
                ),

                "investors_10plus_events": int(
                    at_least_10.sum()
                ),

                "investors_10plus_events_share_pct": pct(
                    at_least_10.sum(),
                    active_investors,
                ),

                # Interaction mass
                "interaction_share_from_2plus_event_investors_pct": pct(
                    interactions_from_2plus,
                    interactions,
                ),

                "interaction_share_from_3plus_event_investors_pct": pct(
                    interactions_from_3plus,
                    interactions,
                ),

                "interaction_share_from_5plus_event_investors_pct": pct(
                    interactions_from_5plus,
                    interactions,
                ),

                "interaction_share_from_10plus_event_investors_pct": pct(
                    interactions_from_10plus,
                    interactions,
                ),

                # Active temporal depth
                "median_active_months": float(
                    active_month_median
                ),

                "active_months_q75": float(
                    active_month_q75
                ),

                "active_months_q90": float(
                    active_month_q90
                ),

                "investors_2plus_active_months": int(
                    investors_2plus_active_months
                ),

                "investors_2plus_active_months_share_pct": pct(
                    investors_2plus_active_months,
                    active_investors,
                ),

                "investors_3plus_active_months": int(
                    investors_3plus_active_months
                ),

                "investors_3plus_active_months_share_pct": pct(
                    investors_3plus_active_months,
                    active_investors,
                ),

                "investors_5plus_active_months": int(
                    investors_5plus_active_months
                ),

                "investors_5plus_active_months_share_pct": pct(
                    investors_5plus_active_months,
                    active_investors,
                ),

                # Left censoring
                "investors_with_pre_window_history": int(
                    investors_with_pre_window_history
                ),

                "investor_left_censoring_share_pct": pct(
                    investors_with_pre_window_history,
                    active_investors,
                ),

                "startups_with_pre_window_history": int(
                    startups_with_pre_window_history
                ),

                "startup_left_censoring_share_pct": pct(
                    startups_with_pre_window_history,
                    active_startups,
                ),
            }
        )

    candidate_table = pd.DataFrame(
        candidate_rows
    )

    # -------------------------------------------------------------------------
    # 8. Save full diagnostics
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_table.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # -------------------------------------------------------------------------
    # 9. Print objective continuity landmarks
    # -------------------------------------------------------------------------

    separator("-")
    print("OBJECTIVE CALENDAR-CONTINUITY LANDMARKS")
    separator("-")

    print(
        f"Last zero-interaction month:        "
        f"{last_zero_month}"
    )

    print(
        f"Continuous nonzero monthly coverage "
        f"after:                             "
        f"{continuous_monthly_start}"
    )

    print(
        f"Last zero-interaction quarter:      "
        f"{last_zero_quarter}"
    )

    print(
        f"Continuous nonzero quarterly "
        f"coverage after:                    "
        f"{continuous_quarterly_start}"
    )

    print(
        """
These are descriptive continuity landmarks only.

They do NOT identify the final temporal cutoff.
A month having at least one interaction is not sufficient evidence
that the period is suitable for ITRS.
"""
    )

    # -------------------------------------------------------------------------
    # 10. Print compact candidate-year table
    # -------------------------------------------------------------------------

    separator("-")
    print("CANDIDATE START YEARS — COMPACT DIAGNOSTICS")
    separator("-")

    compact_columns = [
        "start_year",
        "window_months",

        "interactions",
        "interaction_retention_pct",

        "active_investors",
        "investor_retention_pct",

        "active_startups",
        "startup_retention_pct",

        "zero_interaction_months",

        "min_monthly_interactions",
        "median_monthly_interactions",

        "median_sequence_length",
        "sequence_q90",

        "single_event_investor_share_pct",
        "investors_2plus_events_share_pct",
        "investors_5plus_events_share_pct",

        "investors_2plus_active_months_share_pct",

        "investor_left_censoring_share_pct",
    ]

    print(
        candidate_table[
            compact_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    # -------------------------------------------------------------------------
    # 11. Print richer-history diagnostic
    # -------------------------------------------------------------------------

    separator("-")
    print(
        "INTERACTION MASS FROM INVESTORS "
        "WITH RICHER WITHIN-WINDOW HISTORIES"
    )
    separator("-")

    history_columns = [
        "start_year",
        "interaction_retention_pct",

        "investors_2plus_events",
        "interaction_share_from_2plus_event_investors_pct",

        "investors_3plus_events",
        "interaction_share_from_3plus_event_investors_pct",

        "investors_5plus_events",
        "interaction_share_from_5plus_event_investors_pct",

        "investors_10plus_events",
        "interaction_share_from_10plus_event_investors_pct",
    ]

    print(
        candidate_table[
            history_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    # -------------------------------------------------------------------------
    # 12. Global integrity checks
    # -------------------------------------------------------------------------

    separator("-")
    print("REFERENCE VIEW CONSISTENCY")
    separator("-")

    print(
        f"Canonical interactions:              "
        f"{len(df):,}"
    )

    print(
        f"Reference interactions through "
        f"2026-05-31:                         "
        f"{reference_interactions:,}"
    )

    print(
        f"Interactions after reference end:    "
        f"{excluded_after_reference:,}"
    )

    print(
        f"Reference investors:                 "
        f"{reference_investors:,}"
    )

    print(
        f"Reference startups:                  "
        f"{reference_startups:,}"
    )

    print(
        f"Reference investor-startup pairs:    "
        f"{reference_pairs:,}"
    )

    print(
        f"Reference funding rounds:            "
        f"{reference_rounds:,}"
    )

    # -------------------------------------------------------------------------
    # 13. End
    # -------------------------------------------------------------------------

    separator()
    print("PHASE 2.1.7.1 AUDIT COMPLETE")
    separator()

    print(f"""
Output written to:

{OUTPUT_PATH}

IMPORTANT:

This audit compares EVERY calendar-year starting boundary.

It does NOT:
- select a starting cutoff,
- select a final endpoint,
- select a maturity buffer,
- select monthly or quarterly segments,
- choose t=60,
- remove investors based on sequence length,
- filter investment types,
- modify the canonical Phase-1 dataset.

The next step is to interpret the candidate-start tradeoffs and combine them
with the endpoint-maturity sensitivity from Phase 2.1.6.
""")


if __name__ == "__main__":
    main()