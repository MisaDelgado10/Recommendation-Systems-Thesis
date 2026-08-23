from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PHASE 2.1.4 — RECENT-PERIOD COMPLETENESS AUDIT
# =============================================================================

INPUT_PATH = Path("data/processed/interactions.parquet")

OUTPUT_DIR = Path("data/experimental/phase_2/audits")

MONTHLY_OUTPUT = (
    OUTPUT_DIR / "recent_period_completeness_monthly.csv"
)

DAILY_OUTPUT = (
    OUTPUT_DIR / "recent_period_completeness_daily.csv"
)

EXPECTED_ROWS = 1_208_051

DATE_COL = "announced_on"
INVESTOR_COL = "investor_id"
STARTUP_COL = "startup_id"
ROUND_COL = "funding_round_id"

# Diagnostic horizon only.
# This is NOT a modeling window or temporal cutoff.
RECENT_MONTHS_TO_DISPLAY = 36

# Number of calendar days to print near the dataset endpoint.
ENDPOINT_DAYS_TO_DISPLAY = 75


def separator(char="=", width=110):
    print(char * width)


def safe_ratio(numerator, denominator):
    if denominator == 0 or pd.isna(denominator):
        return np.nan

    return numerator / denominator


def main():

    separator()
    print("PHASE 2.1.4 — RECENT-PERIOD COMPLETENESS AUDIT")
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

    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} rows, "
            f"found {len(df):,}."
        )

    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL],
        errors="raise",
    )

    min_date = df[DATE_COL].min()
    max_date = df[DATE_COL].max()

    print(f"\nCanonical interactions: {len(df):,}")
    print(f"Minimum date:           {min_date.date()}")
    print(f"Maximum date:           {max_date.date()}")

    # -------------------------------------------------------------------------
    # 2. Calendar variables
    # -------------------------------------------------------------------------

    df["month"] = df[DATE_COL].dt.to_period("M")
    df["day"] = df[DATE_COL].dt.normalize()

    min_month = df["month"].min()
    max_month = df["month"].max()

    full_month_index = pd.period_range(
        min_month,
        max_month,
        freq="M",
        name="month",
    )

    # -------------------------------------------------------------------------
    # 3. Basic monthly activity
    # -------------------------------------------------------------------------

    interactions = (
        df.groupby("month", observed=True)
        .size()
        .rename("interactions")
    )

    active_investors = (
        df.groupby("month", observed=True)[INVESTOR_COL]
        .nunique()
        .rename("active_investors")
    )

    active_startups = (
        df.groupby("month", observed=True)[STARTUP_COL]
        .nunique()
        .rename("active_startups")
    )

    funding_rounds = (
        df.groupby("month", observed=True)[ROUND_COL]
        .nunique()
        .rename("funding_rounds")
    )

    active_pairs = (
        df[
            [
                "month",
                INVESTOR_COL,
                STARTUP_COL,
            ]
        ]
        .drop_duplicates()
        .groupby("month", observed=True)
        .size()
        .rename("active_investor_startup_pairs")
    )

    monthly = pd.concat(
        [
            interactions,
            active_investors,
            active_startups,
            active_pairs,
            funding_rounds,
        ],
        axis=1,
    ).reindex(
        full_month_index,
        fill_value=0,
    )

    # -------------------------------------------------------------------------
    # 4. Intra-month temporal coverage
    # -------------------------------------------------------------------------

    date_stats = (
        df.groupby("month", observed=True)[DATE_COL]
        .agg(
            first_event_date="min",
            last_event_date="max",
            active_event_dates="nunique",
        )
    )

    monthly = monthly.join(
        date_stats,
        how="left",
    )

    monthly["calendar_days"] = [
        period.days_in_month
        for period in monthly.index
    ]

    monthly["first_event_day"] = (
        monthly["first_event_date"].dt.day
    )

    monthly["last_event_day"] = (
        monthly["last_event_date"].dt.day
    )

    monthly["days_from_last_event_to_month_end"] = (
        monthly["calendar_days"]
        - monthly["last_event_day"]
    )

    monthly["active_event_day_share_pct"] = (
        monthly["active_event_dates"]
        / monthly["calendar_days"]
        * 100
    )

    # -------------------------------------------------------------------------
    # 5. First-half / second-half distribution
    # -------------------------------------------------------------------------

    df["day_of_month"] = df[DATE_COL].dt.day

    first_half = (
        df[df["day_of_month"] <= 15]
        .groupby("month", observed=True)
        .size()
        .rename("first_half_interactions")
    )

    second_half = (
        df[df["day_of_month"] > 15]
        .groupby("month", observed=True)
        .size()
        .rename("second_half_interactions")
    )

    monthly = monthly.join(
        [
            first_half,
            second_half,
        ],
        how="left",
    )

    monthly[
        [
            "first_half_interactions",
            "second_half_interactions",
        ]
    ] = (
        monthly[
            [
                "first_half_interactions",
                "second_half_interactions",
            ]
        ]
        .fillna(0)
        .astype("int64")
    )

    monthly["second_half_share_pct"] = np.where(
        monthly["interactions"] > 0,
        monthly["second_half_interactions"]
        / monthly["interactions"]
        * 100,
        np.nan,
    )

    # -------------------------------------------------------------------------
    # 6. Last-seven-calendar-days distribution
    # -------------------------------------------------------------------------

    df["days_in_month"] = (
        df[DATE_COL].dt.days_in_month
    )

    df["is_last_7_calendar_days"] = (
        df["day_of_month"]
        > (df["days_in_month"] - 7)
    )

    last_7 = (
        df[df["is_last_7_calendar_days"]]
        .groupby("month", observed=True)
        .size()
        .rename("last_7_days_interactions")
    )

    monthly = monthly.join(
        last_7,
        how="left",
    )

    monthly["last_7_days_interactions"] = (
        monthly["last_7_days_interactions"]
        .fillna(0)
        .astype("int64")
    )

    monthly["last_7_days_share_pct"] = np.where(
        monthly["interactions"] > 0,
        monthly["last_7_days_interactions"]
        / monthly["interactions"]
        * 100,
        np.nan,
    )

    # -------------------------------------------------------------------------
    # 7. Previous-year benchmark
    # -------------------------------------------------------------------------

    monthly["prev_year_interactions"] = (
        monthly["interactions"].shift(12)
    )

    monthly["ratio_to_prev_year_pct"] = (
        monthly["interactions"]
        / monthly["prev_year_interactions"]
        * 100
    )

    monthly["prev_year_active_investors"] = (
        monthly["active_investors"].shift(12)
    )

    monthly["investor_ratio_to_prev_year_pct"] = (
        monthly["active_investors"]
        / monthly["prev_year_active_investors"]
        * 100
    )

    monthly["prev_year_active_startups"] = (
        monthly["active_startups"].shift(12)
    )

    monthly["startup_ratio_to_prev_year_pct"] = (
        monthly["active_startups"]
        / monthly["prev_year_active_startups"]
        * 100
    )

    # -------------------------------------------------------------------------
    # 8. Three-year same-calendar-month benchmark
    # -------------------------------------------------------------------------

    prior_3y_medians = []

    for period in monthly.index:

        previous_values = []

        for years_back in [1, 2, 3]:

            previous_period = period - (12 * years_back)

            if previous_period in monthly.index:
                previous_values.append(
                    monthly.loc[
                        previous_period,
                        "interactions",
                    ]
                )

        if previous_values:
            prior_3y_medians.append(
                float(np.median(previous_values))
            )
        else:
            prior_3y_medians.append(np.nan)

    monthly["prior_3y_same_month_median"] = (
        prior_3y_medians
    )

    monthly["ratio_to_prior_3y_month_median_pct"] = (
        monthly["interactions"]
        / monthly["prior_3y_same_month_median"]
        * 100
    )

    # -------------------------------------------------------------------------
    # 9. Prior rolling 12-month benchmark
    #
    # shift(1) is crucial: the current month must not contribute to
    # its own baseline.
    # -------------------------------------------------------------------------

    monthly["prior_12m_median_interactions"] = (
        monthly["interactions"]
        .shift(1)
        .rolling(
            window=12,
            min_periods=6,
        )
        .median()
    )

    monthly["ratio_to_prior_12m_median_pct"] = (
        monthly["interactions"]
        / monthly["prior_12m_median_interactions"]
        * 100
    )

    # -------------------------------------------------------------------------
    # 10. Explicit global-endpoint diagnostics
    # -------------------------------------------------------------------------

    monthly["contains_global_max_date"] = False

    monthly.loc[
        max_month,
        "contains_global_max_date",
    ] = True

    monthly["calendar_coverage_through_dataset_endpoint_pct"] = 100.0

    monthly.loc[
        max_month,
        "calendar_coverage_through_dataset_endpoint_pct",
    ] = (
        max_date.day
        / max_month.days_in_month
        * 100
    )

    # This is an objective calendar fact, not a statistical judgment.
    monthly["terminal_calendar_month_is_partial"] = False

    if max_date.day < max_month.days_in_month:
        monthly.loc[
            max_month,
            "terminal_calendar_month_is_partial",
        ] = True

    # -------------------------------------------------------------------------
    # 11. Build daily endpoint audit
    # -------------------------------------------------------------------------

    daily = (
        df.groupby("day", observed=True)
        .agg(
            interactions=("interaction_id", "size"),
            active_investors=(INVESTOR_COL, "nunique"),
            active_startups=(STARTUP_COL, "nunique"),
            funding_rounds=(ROUND_COL, "nunique"),
        )
    )

    full_daily_index = pd.date_range(
        start=min_date.normalize(),
        end=max_date.normalize(),
        freq="D",
        name="day",
    )

    daily = daily.reindex(
        full_daily_index,
        fill_value=0,
    )

    daily["day_of_week"] = (
        daily.index.day_name()
    )

    daily["month"] = (
        daily.index.to_period("M").astype(str)
    )

    # -------------------------------------------------------------------------
    # 12. Save audit outputs
    # -------------------------------------------------------------------------

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

    daily_output = daily.reset_index()

    daily_output.to_csv(
        DAILY_OUTPUT,
        index=False,
    )

    # -------------------------------------------------------------------------
    # 13. Print recent monthly completeness diagnostics
    # -------------------------------------------------------------------------

    separator("-")
    print(
        f"MOST RECENT {RECENT_MONTHS_TO_DISPLAY} MONTHS — "
        "COMPLETENESS DIAGNOSTICS"
    )
    separator("-")

    recent = monthly.tail(
        RECENT_MONTHS_TO_DISPLAY
    )[
        [
            "interactions",
            "prev_year_interactions",
            "ratio_to_prev_year_pct",
            "prior_3y_same_month_median",
            "ratio_to_prior_3y_month_median_pct",
            "active_event_dates",
            "first_event_day",
            "last_event_day",
            "days_from_last_event_to_month_end",
            "second_half_share_pct",
            "last_7_days_share_pct",
            "calendar_coverage_through_dataset_endpoint_pct",
            "terminal_calendar_month_is_partial",
        ]
    ].copy()

    print(
        recent.to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )

    # -------------------------------------------------------------------------
    # 14. Print 2026 separately
    # -------------------------------------------------------------------------

    separator("-")
    print("2026 MONTH-BY-MONTH DIAGNOSTICS")
    separator("-")

    months_2026 = monthly[
        monthly.index.year == 2026
    ][
        [
            "interactions",
            "active_investors",
            "active_startups",
            "funding_rounds",
            "prev_year_interactions",
            "ratio_to_prev_year_pct",
            "prior_3y_same_month_median",
            "ratio_to_prior_3y_month_median_pct",
            "active_event_dates",
            "last_event_day",
            "days_from_last_event_to_month_end",
            "second_half_share_pct",
            "last_7_days_share_pct",
        ]
    ]

    print(
        months_2026.to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )

    # -------------------------------------------------------------------------
    # 15. Print endpoint daily activity
    # -------------------------------------------------------------------------

    separator("-")
    print(
        f"LAST {ENDPOINT_DAYS_TO_DISPLAY} CALENDAR DAYS "
        "BEFORE DATASET ENDPOINT"
    )
    separator("-")

    endpoint_daily = daily.tail(
        ENDPOINT_DAYS_TO_DISPLAY
    )

    print(
        endpoint_daily.to_string()
    )

    # -------------------------------------------------------------------------
    # 16. Endpoint summary
    # -------------------------------------------------------------------------

    separator("-")
    print("DATASET ENDPOINT SUMMARY")
    separator("-")

    endpoint_month = monthly.loc[max_month]

    print(
        f"Global maximum date:                    "
        f"{max_date.date()}"
    )

    print(
        f"Terminal calendar month:                "
        f"{max_month}"
    )

    print(
        f"Days in terminal month:                 "
        f"{max_month.days_in_month}"
    )

    print(
        f"Dataset reaches calendar day:           "
        f"{max_date.day}"
    )

    print(
        f"Calendar coverage of terminal month:    "
        f"{endpoint_month['calendar_coverage_through_dataset_endpoint_pct']:.2f}%"
    )

    print(
        f"Interactions in terminal month:         "
        f"{int(endpoint_month['interactions']):,}"
    )

    print(
        f"Previous-year same-month interactions:  "
        f"{int(endpoint_month['prev_year_interactions']):,}"
    )

    print(
        f"Ratio to previous year:                 "
        f"{endpoint_month['ratio_to_prev_year_pct']:.4f}%"
    )

    print(
        f"Active event dates in terminal month:   "
        f"{int(endpoint_month['active_event_dates']):,}"
    )

    print(
        f"Terminal calendar month partial:        "
        f"{endpoint_month['terminal_calendar_month_is_partial']}"
    )

    # -------------------------------------------------------------------------
    # 17. Integrity checks
    # -------------------------------------------------------------------------

    separator("-")
    print("GLOBAL CONSISTENCY CHECKS")
    separator("-")

    print(
        f"Monthly interaction sum:       "
        f"{monthly['interactions'].sum():,}"
    )

    print(
        f"Daily interaction sum:         "
        f"{daily['interactions'].sum():,}"
    )

    print(
        f"Canonical interaction count:   "
        f"{EXPECTED_ROWS:,}"
    )

    if monthly["interactions"].sum() != EXPECTED_ROWS:
        raise ValueError(
            "Monthly aggregation does not reproduce canonical row count."
        )

    if daily["interactions"].sum() != EXPECTED_ROWS:
        raise ValueError(
            "Daily aggregation does not reproduce canonical row count."
        )

    # -------------------------------------------------------------------------
    # 18. End
    # -------------------------------------------------------------------------

    separator()
    print("PHASE 2.1.4 AUDIT COMPLETE")
    separator()

    print(f"""
Outputs written to:

{MONTHLY_OUTPUT}
{DAILY_OUTPUT}

IMPORTANT:

This audit measures recent-period completeness only.

It does NOT:
- select the final temporal cutoff,
- remove June 2026,
- remove May 2026,
- choose the 60 ITRS temporal segments,
- impose a minimum investor history,
- filter investment types,
- modify the Phase-1 canonical interaction dataset.

The next decision must be based on the observed endpoint diagnostics.
""")


if __name__ == "__main__":
    main()