from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PHASE 2.1.7.2 — JOINT START / ENDPOINT SENSITIVITY AUDIT
# =============================================================================
#
# PURPOSE
# -------
# Combine:
#
#   Phase 2.1.7.1 start-boundary diagnostics
#   Phase 2.1.6.2 endpoint maturity scenarios
#
# without selecting a final temporal window.
#
# Candidate start years used here are diagnostic LANDMARKS derived
# automatically from observed data properties.
# =============================================================================


INPUT_PATH = Path(
    "data/processed/interactions.parquet"
)

START_AUDIT_PATH = Path(
    "data/experimental/phase_2/audits/"
    "candidate_start_year_diagnostics.csv"
)

BUFFER_AUDIT_PATH = Path(
    "data/experimental/phase_2/audits/"
    "snapshot_maturity_buffer_sensitivity.csv"
)

OUTPUT_DIR = Path(
    "data/experimental/phase_2/audits"
)

LANDMARK_OUTPUT = (
    OUTPUT_DIR
    / "candidate_start_landmarks.csv"
)

JOINT_OUTPUT = (
    OUTPUT_DIR
    / "joint_start_endpoint_sensitivity.csv"
)


EXPECTED_ROWS = 1_208_051

DATE_COL = "announced_on"
INVESTOR_COL = "investor_id"
STARTUP_COL = "startup_id"
ROUND_COL = "funding_round_id"


# -------------------------------------------------------------------------
# Diagnostic thresholds
#
# These are NOT model eligibility criteria.
#
# They are used only to identify informative locations on the observed
# start-year tradeoff curve.
# -------------------------------------------------------------------------

INTERACTION_RETENTION_LEVELS = [
    99,
    95,
    90,
    85,
    80,
    75,
    50,
]

LEFT_CENSORING_LEVELS = [
    1,
    5,
    10,
    20,
    25,
]


def separator(char="=", width=120):
    print(char * width)


def pct(num, den):

    if den == 0:
        return np.nan

    return num / den * 100


def first_year_at_or_below(
    df,
    column,
    threshold,
):

    matches = df[
        df[column] <= threshold
    ]

    if len(matches) == 0:
        return None

    return int(
        matches.iloc[0]["start_year"]
    )


def first_year_at_or_above(
    df,
    column,
    threshold,
):

    matches = df[
        df[column] >= threshold
    ]

    if len(matches) == 0:
        return None

    return int(
        matches.iloc[0]["start_year"]
    )


def main():

    separator()
    print(
        "PHASE 2.1.7.2 — "
        "JOINT START / ENDPOINT SENSITIVITY AUDIT"
    )
    separator()

    # ---------------------------------------------------------------------
    # 1. Load prior audit outputs
    # ---------------------------------------------------------------------

    start_audit = pd.read_csv(
        START_AUDIT_PATH
    ).sort_values(
        "start_year"
    )

    buffer_audit = pd.read_csv(
        BUFFER_AUDIT_PATH
    ).sort_values(
        "hypothetical_buffer_days"
    )

    # ---------------------------------------------------------------------
    # 2. Derive diagnostic start-year landmarks
    # ---------------------------------------------------------------------

    landmark_records = []

    # Earliest Jan-1 start with no zero-interaction months.
    zero_free = start_audit[
        start_audit[
            "zero_interaction_months"
        ] == 0
    ]

    if len(zero_free) > 0:

        year = int(
            zero_free.iloc[0][
                "start_year"
            ]
        )

        landmark_records.append(
            {
                "landmark_type": (
                    "first_zero_free_calendar_year"
                ),
                "threshold": 0,
                "start_year": year,
            }
        )

    # Interaction-retention landmarks.
    for level in INTERACTION_RETENTION_LEVELS:

        year = first_year_at_or_below(
            start_audit,
            "interaction_retention_pct",
            level,
        )

        if year is not None:

            landmark_records.append(
                {
                    "landmark_type": (
                        "interaction_retention_at_or_below_pct"
                    ),
                    "threshold": level,
                    "start_year": year,
                }
            )

    # Left-censoring landmarks.
    for level in LEFT_CENSORING_LEVELS:

        year = first_year_at_or_above(
            start_audit,
            "investor_left_censoring_share_pct",
            level,
        )

        if year is not None:

            landmark_records.append(
                {
                    "landmark_type": (
                        "investor_left_censoring_at_or_above_pct"
                    ),
                    "threshold": level,
                    "start_year": year,
                }
            )

    landmark_table = pd.DataFrame(
        landmark_records
    )

    # Several criteria can identify the same year.
    landmark_years = sorted(
        landmark_table[
            "start_year"
        ]
        .dropna()
        .astype(int)
        .unique()
    )

    # ---------------------------------------------------------------------
    # 3. Load canonical data
    # ---------------------------------------------------------------------

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

    # Full canonical first-event dates are required for left-censoring
    # identification.
    investor_first_date = (
        df.groupby(
            INVESTOR_COL,
            observed=True,
        )[DATE_COL]
        .min()
    )

    # ---------------------------------------------------------------------
    # 4. Evaluate landmark start years across all endpoint scenarios
    # ---------------------------------------------------------------------

    rows = []

    for _, buffer_row in buffer_audit.iterrows():

        buffer_days = int(
            buffer_row[
                "hypothetical_buffer_days"
            ]
        )

        end_date = pd.to_datetime(
            buffer_row[
                "latest_eligible_month_end"
            ]
        )

        if pd.isna(end_date):
            continue

        for start_year in landmark_years:

            start_date = pd.Timestamp(
                year=start_year,
                month=1,
                day=1,
            )

            # Skip logically invalid windows.
            if start_date > end_date:
                continue

            window = df[
                (df[DATE_COL] >= start_date)
                &
                (df[DATE_COL] <= end_date)
            ]

            interactions = len(window)

            active_investors = (
                window[INVESTOR_COL]
                .nunique()
            )

            active_startups = (
                window[STARTUP_COL]
                .nunique()
            )

            active_rounds = (
                window[ROUND_COL]
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

            # -------------------------------------------------------------
            # Sequence characteristics inside this exact start/end window.
            # -------------------------------------------------------------

            investor_counts = (
                window.groupby(
                    INVESTOR_COL,
                    observed=True,
                )
                .size()
            )

            single_event_share = pct(
                (
                    investor_counts == 1
                ).sum(),
                active_investors,
            )

            investors_2plus = (
                investor_counts >= 2
            )

            investors_5plus = (
                investor_counts >= 5
            )

            interactions_from_2plus = (
                investor_counts[
                    investors_2plus
                ]
                .sum()
            )

            interactions_from_5plus = (
                investor_counts[
                    investors_5plus
                ]
                .sum()
            )

            # -------------------------------------------------------------
            # Active-month depth
            # -------------------------------------------------------------

            temp = window[
                [
                    INVESTOR_COL,
                    DATE_COL,
                ]
            ].copy()

            temp["month"] = (
                temp[DATE_COL]
                .dt.to_period("M")
            )

            active_month_counts = (
                temp[
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

            # -------------------------------------------------------------
            # Left censoring
            # -------------------------------------------------------------

            active_investor_ids = (
                investor_counts.index
            )

            active_first_dates = (
                investor_first_date
                .reindex(
                    active_investor_ids
                )
            )

            left_censored_count = int(
                (
                    active_first_dates
                    < start_date
                )
                .sum()
            )

            # -------------------------------------------------------------
            # Window duration
            # -------------------------------------------------------------

            window_months = len(
                pd.period_range(
                    start_date.to_period("M"),
                    end_date.to_period("M"),
                    freq="M",
                )
            )

            rows.append(
                {
                    "hypothetical_buffer_days": (
                        buffer_days
                    ),

                    "end_date": (
                        end_date.date()
                    ),

                    "start_year": (
                        start_year
                    ),

                    "window_months": (
                        window_months
                    ),

                    "interactions": (
                        interactions
                    ),

                    "canonical_interaction_retention_pct": pct(
                        interactions,
                        len(df),
                    ),

                    "active_investors": (
                        active_investors
                    ),

                    "active_startups": (
                        active_startups
                    ),

                    "active_pairs": (
                        active_pairs
                    ),

                    "funding_rounds": (
                        active_rounds
                    ),

                    "median_sequence_length": (
                        float(
                            investor_counts.median()
                        )
                    ),

                    "sequence_q90": (
                        float(
                            investor_counts.quantile(
                                0.90
                            )
                        )
                    ),

                    "single_event_investor_share_pct": (
                        single_event_share
                    ),

                    "investors_2plus_events_share_pct": pct(
                        investors_2plus.sum(),
                        active_investors,
                    ),

                    "investors_5plus_events_share_pct": pct(
                        investors_5plus.sum(),
                        active_investors,
                    ),

                    "interaction_share_from_2plus_event_investors_pct": pct(
                        interactions_from_2plus,
                        interactions,
                    ),

                    "interaction_share_from_5plus_event_investors_pct": pct(
                        interactions_from_5plus,
                        interactions,
                    ),

                    "median_active_months": (
                        float(
                            active_month_counts.median()
                        )
                    ),

                    "investors_2plus_active_months_share_pct": pct(
                        (
                            active_month_counts
                            >= 2
                        ).sum(),
                        active_investors,
                    ),

                    "investor_left_censoring_share_pct": pct(
                        left_censored_count,
                        active_investors,
                    ),
                }
            )

    joint = pd.DataFrame(
        rows
    )

    # ---------------------------------------------------------------------
    # 5. Save outputs
    # ---------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    landmark_table.to_csv(
        LANDMARK_OUTPUT,
        index=False,
    )

    joint.to_csv(
        JOINT_OUTPUT,
        index=False,
    )

    # ---------------------------------------------------------------------
    # 6. Print landmark definitions
    # ---------------------------------------------------------------------

    separator("-")
    print("DATA-DERIVED START-YEAR LANDMARKS")
    separator("-")

    print(
        landmark_table.to_string(
            index=False
        )
    )

    print(
        "\nUnique landmark start years:"
    )

    print(
        ", ".join(
            str(year)
            for year in landmark_years
        )
    )

    # ---------------------------------------------------------------------
    # 7. Main joint sensitivity table
    # ---------------------------------------------------------------------

    separator("-")
    print("JOINT START / ENDPOINT SENSITIVITY")
    separator("-")

    display_columns = [
        "hypothetical_buffer_days",
        "end_date",
        "start_year",
        "window_months",
        "interactions",
        "canonical_interaction_retention_pct",
        "active_investors",
        "median_sequence_length",
        "sequence_q90",
        "single_event_investor_share_pct",
        "investors_5plus_events_share_pct",
        "interaction_share_from_5plus_event_investors_pct",
        "investor_left_censoring_share_pct",
    ]

    print(
        joint[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    # ---------------------------------------------------------------------
    # 8. Endpoint effect holding start year fixed
    # ---------------------------------------------------------------------

    separator("-")
    print(
        "ENDPOINT SENSITIVITY BY START-YEAR LANDMARK"
    )
    separator("-")

    for start_year in landmark_years:

        subset = (
            joint[
                joint["start_year"]
                == start_year
            ]
            .sort_values(
                "hypothetical_buffer_days"
            )
        )

        if len(subset) == 0:
            continue

        baseline = subset.iloc[0]

        temp = subset[
            [
                "hypothetical_buffer_days",
                "end_date",
                "interactions",
                "active_investors",
                "median_sequence_length",
                "single_event_investor_share_pct",
            ]
        ].copy()

        temp[
            "interaction_change_vs_0_buffer_pct"
        ] = (
            (
                temp["interactions"]
                / baseline["interactions"]
            )
            - 1
        ) * 100

        print(
            f"\nSTART YEAR {start_year}"
        )

        print(
            temp.to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}",
            )
        )

    # ---------------------------------------------------------------------
    # 9. Interpretation reminder
    # ---------------------------------------------------------------------

    separator("-")
    print("INTERPRETATION CONSTRAINT")
    separator("-")

    print(
        """
The endpoint buffers remain HYPOTHETICAL maturity scenarios.

They do not estimate actual Crunchbase reporting delay.

The landmark start years are diagnostic points derived from the observed
tradeoff curve. They are NOT selected temporal cutoffs.

This audit asks whether conclusions about the starting boundary remain
stable when recent right-censoring risk is handled more conservatively.
"""
    )

    # ---------------------------------------------------------------------
    # 10. End
    # ---------------------------------------------------------------------

    separator()
    print(
        "PHASE 2.1.7.2 AUDIT COMPLETE"
    )
    separator()

    print(f"""
Outputs written to:

{LANDMARK_OUTPUT}
{JOINT_OUTPUT}

No start date has been selected.
No final endpoint has been selected.
No maturity buffer has been selected.
No minimum investor history has been selected.
No t=60 segment construction has been performed.
No investment-type filtering has been performed.
The Phase-1 canonical interaction layer remains unchanged.

Next:
Phase 2.1.8 — Temporal Distribution Audit Conclusions.
""")


if __name__ == "__main__":
    main()