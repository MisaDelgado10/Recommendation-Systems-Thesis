from pathlib import Path

import pandas as pd


# =============================================================================
# PHASE 2.1.6.2 — SNAPSHOT-AGE / CENSORING-EXPOSURE AUDIT
# =============================================================================
#
# IMPORTANT:
#
# This script does NOT estimate actual Crunchbase reporting lag.
#
# The available raw funding dataset contains announced_on but no record-
# creation or update timestamp. Therefore reporting latency cannot be
# directly measured.
#
# Instead, this script measures how much time different historical periods
# had to "mature" before the approximate extraction completion date.
# =============================================================================


INPUT_PATH = Path(
    "data/processed/interactions.parquet"
)

OUTPUT_DIR = Path(
    "data/experimental/phase_2/audits"
)

MONTHLY_OUTPUT = (
    OUTPUT_DIR / "snapshot_age_monthly.csv"
)

BUFFER_OUTPUT = (
    OUTPUT_DIR / "snapshot_maturity_buffer_sensitivity.csv"
)


EXPECTED_ROWS = 1_208_051

DATE_COL = "announced_on"


# -------------------------------------------------------------------------
# Data-collection provenance
# -------------------------------------------------------------------------
#
# User's collection process was completed approximately June 2, 2026.
#
# This is a reference date for sensitivity analysis, NOT a proven
# Crunchbase database snapshot timestamp.
#
# The raw filename includes 20260531, whose exact semantics are currently
# unknown and should not be treated as authoritative snapshot metadata.
# -------------------------------------------------------------------------

COLLECTION_REFERENCE_DATE = pd.Timestamp("2026-06-02")


# -------------------------------------------------------------------------
# Hypothetical maturity buffers
# -------------------------------------------------------------------------
#
# These values are NOT assumptions about actual Crunchbase reporting lag.
#
# They are sensitivity scenarios only.
# -------------------------------------------------------------------------

MATURITY_BUFFERS_DAYS = [
    0,
    7,
    14,
    30,
    60,
    90,
    120,
    180,
]


def separator(char="=", width=105):
    print(char * width)


def main():

    separator()
    print(
        "PHASE 2.1.6.2 — "
        "SNAPSHOT-AGE / CENSORING-EXPOSURE AUDIT"
    )
    separator()

    # ---------------------------------------------------------------------
    # 1. Load canonical event dates
    # ---------------------------------------------------------------------

    df = pd.read_parquet(
        INPUT_PATH,
        columns=[
            "interaction_id",
            DATE_COL,
            "investor_id",
            "startup_id",
        ],
    )

    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} interactions; "
            f"found {len(df):,}."
        )

    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL],
        errors="raise",
    )

    print(
        f"\nCanonical interactions:       "
        f"{len(df):,}"
    )

    print(
        f"Canonical event range:        "
        f"{df[DATE_COL].min().date()} -> "
        f"{df[DATE_COL].max().date()}"
    )

    print(
        f"Collection reference date:    "
        f"{COLLECTION_REFERENCE_DATE.date()}"
    )

    # ---------------------------------------------------------------------
    # 2. Build month variable
    # ---------------------------------------------------------------------

    df["month"] = (
        df[DATE_COL].dt.to_period("M")
    )

    monthly = (
        df.groupby(
            "month",
            observed=True,
        )
        .agg(
            interactions=(
                "interaction_id",
                "size",
            ),
            active_investors=(
                "investor_id",
                "nunique",
            ),
            active_startups=(
                "startup_id",
                "nunique",
            ),
            first_observed_date=(
                DATE_COL,
                "min",
            ),
            last_observed_date=(
                DATE_COL,
                "max",
            ),
        )
    )

    # ---------------------------------------------------------------------
    # 3. Month-end maturity exposure
    # ---------------------------------------------------------------------
    #
    # Question:
    #
    # How many calendar days elapsed between the END of a month and our
    # approximate collection completion date?
    #
    # Positive:
    #   month ended before collection.
    #
    # Zero:
    #   month-end equals collection date.
    #
    # Negative:
    #   collection occurred before that calendar month ended.
    # ---------------------------------------------------------------------

    monthly["calendar_month_end"] = (
        monthly.index.to_timestamp(
            how="end"
        )
        .normalize()
    )

    monthly["days_from_month_end_to_collection"] = (
        COLLECTION_REFERENCE_DATE
        - monthly["calendar_month_end"]
    ).dt.days

    # ---------------------------------------------------------------------
    # 4. Last observed event age
    # ---------------------------------------------------------------------

    monthly["days_from_last_event_to_collection"] = (
        COLLECTION_REFERENCE_DATE
        - monthly["last_observed_date"]
    ).dt.days

    # ---------------------------------------------------------------------
    # 5. Complete-calendar-month indicator
    # ---------------------------------------------------------------------

    monthly["calendar_month_finished_before_collection"] = (
        monthly["calendar_month_end"]
        < COLLECTION_REFERENCE_DATE
    )

    monthly["observed_through_calendar_month_end"] = (
        monthly["last_observed_date"]
        == monthly["calendar_month_end"]
    )

    # ---------------------------------------------------------------------
    # 6. Maturity-buffer sensitivity
    #
    # For a hypothetical buffer B:
    #
    #     eligible_date <= collection_date - B
    #
    # We then identify the latest COMPLETE calendar month whose month-end
    # satisfies this rule.
    #
    # Again: no buffer is selected here.
    # ---------------------------------------------------------------------

    buffer_rows = []

    for buffer_days in MATURITY_BUFFERS_DAYS:

        maturity_boundary = (
            COLLECTION_REFERENCE_DATE
            - pd.Timedelta(days=buffer_days)
        )

        eligible_months = monthly[
            (
                monthly[
                    "calendar_month_finished_before_collection"
                ]
            )
            &
            (
                monthly["calendar_month_end"]
                <= maturity_boundary
            )
        ]

        if len(eligible_months) > 0:

            latest_month = eligible_months.index.max()

            latest_month_end = (
                eligible_months.loc[
                    latest_month,
                    "calendar_month_end",
                ]
            )

            interactions_through_boundary = (
                df.loc[
                    df[DATE_COL] <= latest_month_end
                ].shape[0]
            )

            investors_through_boundary = (
                df.loc[
                    df[DATE_COL] <= latest_month_end,
                    "investor_id",
                ]
                .nunique()
            )

            startups_through_boundary = (
                df.loc[
                    df[DATE_COL] <= latest_month_end,
                    "startup_id",
                ]
                .nunique()
            )

        else:

            latest_month = None
            latest_month_end = pd.NaT
            interactions_through_boundary = 0
            investors_through_boundary = 0
            startups_through_boundary = 0

        excluded_interactions = (
            len(df)
            - interactions_through_boundary
        )

        buffer_rows.append(
            {
                "hypothetical_buffer_days": buffer_days,
                "maturity_boundary_date": maturity_boundary.date(),
                "latest_eligible_complete_month": (
                    str(latest_month)
                    if latest_month is not None
                    else None
                ),
                "latest_eligible_month_end": latest_month_end,
                "interactions_through_boundary": (
                    interactions_through_boundary
                ),
                "interactions_excluded_from_recent_tail": (
                    excluded_interactions
                ),
                "interaction_retention_pct": (
                    interactions_through_boundary
                    / len(df)
                    * 100
                ),
                "unique_investors_through_boundary": (
                    investors_through_boundary
                ),
                "unique_startups_through_boundary": (
                    startups_through_boundary
                ),
            }
        )

    buffer_summary = pd.DataFrame(
        buffer_rows
    )

    # ---------------------------------------------------------------------
    # 7. Save
    # ---------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    monthly_output = monthly.reset_index()

    monthly_output["month"] = (
        monthly_output["month"].astype(str)
    )

    monthly_output.to_csv(
        MONTHLY_OUTPUT,
        index=False,
    )

    buffer_summary.to_csv(
        BUFFER_OUTPUT,
        index=False,
    )

    # ---------------------------------------------------------------------
    # 8. Recent month view
    # ---------------------------------------------------------------------

    separator("-")
    print("RECENT MONTH-END MATURITY EXPOSURE")
    separator("-")

    recent = monthly.tail(18)[
        [
            "interactions",
            "first_observed_date",
            "last_observed_date",
            "calendar_month_end",
            "days_from_month_end_to_collection",
            "days_from_last_event_to_collection",
            "calendar_month_finished_before_collection",
            "observed_through_calendar_month_end",
        ]
    ]

    print(
        recent.to_string()
    )

    # ---------------------------------------------------------------------
    # 9. Buffer sensitivity
    # ---------------------------------------------------------------------

    separator("-")
    print("HYPOTHETICAL MATURITY-BUFFER SENSITIVITY")
    separator("-")

    print(
        buffer_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ---------------------------------------------------------------------
    # 10. Important interpretation
    # ---------------------------------------------------------------------

    separator("-")
    print("INTERPRETATION CONSTRAINT")
    separator("-")

    print(
        """
The buffer table is NOT an estimate of Crunchbase reporting lag.

For example, a 30-day row does NOT mean that Crunchbase requires
30 days to mature.

It answers only:

    "If a future experimental design required at least 30 calendar
     days between month-end and the collection reference date,
     what would the latest eligible complete month be, and how much
     canonical data would remain?"

Actual reporting lag remains unidentifiable from this single snapshot.
"""
    )

    # ---------------------------------------------------------------------
    # 11. Integrity
    # ---------------------------------------------------------------------

    separator("-")
    print("GLOBAL CONSISTENCY CHECK")
    separator("-")

    print(
        f"Canonical interactions: "
        f"{len(df):,}"
    )

    print(
        f"Monthly interaction sum: "
        f"{monthly['interactions'].sum():,}"
    )

    if monthly["interactions"].sum() != EXPECTED_ROWS:
        raise ValueError(
            "Monthly totals do not reproduce the canonical dataset."
        )

    separator()
    print("PHASE 2.1.6.2 AUDIT COMPLETE")
    separator()

    print(f"""
Outputs written to:

{MONTHLY_OUTPUT}
{BUFFER_OUTPUT}

No reporting-lag duration has been selected.
No maturity buffer has been selected.
No recent month has been removed from the canonical layer.
No temporal starting cutoff has been selected.
No 60-segment design has been selected.

Next:
Phase 2.1.7 — Temporal Coverage and Candidate-Window Diagnostics.
""")


if __name__ == "__main__":
    main()