from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PHASE 2.1.3 — MONTHLY AND QUARTERLY TEMPORAL DISTRIBUTION
# =============================================================================

INPUT_PATH = Path("data/processed/interactions.parquet")

OUTPUT_DIR = Path("data/experimental/phase_2/audits")

MONTHLY_OUTPUT = OUTPUT_DIR / "monthly_temporal_distribution.csv"
QUARTERLY_OUTPUT = OUTPUT_DIR / "quarterly_temporal_distribution.csv"

EXPECTED_ROWS = 1_208_051

DATE_COL = "announced_on"
INVESTOR_COL = "investor_id"
STARTUP_COL = "startup_id"
ROUND_COL = "funding_round_id"


def separator(char="=", width=110):
    print(char * width)


def build_period_table(
    df,
    period_col,
    full_period_index,
):

    # -------------------------------------------------------------------------
    # Core activity
    # -------------------------------------------------------------------------

    interactions = (
        df.groupby(period_col, observed=True)
        .size()
        .rename("interactions")
    )

    active_investors = (
        df.groupby(period_col, observed=True)[INVESTOR_COL]
        .nunique()
        .rename("active_investors")
    )

    active_startups = (
        df.groupby(period_col, observed=True)[STARTUP_COL]
        .nunique()
        .rename("active_startups")
    )

    funding_rounds = (
        df.groupby(period_col, observed=True)[ROUND_COL]
        .nunique()
        .rename("funding_rounds")
    )

    active_pairs = (
        df[
            [
                period_col,
                INVESTOR_COL,
                STARTUP_COL,
            ]
        ]
        .drop_duplicates()
        .groupby(period_col, observed=True)
        .size()
        .rename("active_investor_startup_pairs")
    )

    result = pd.concat(
        [
            interactions,
            active_investors,
            active_startups,
            active_pairs,
            funding_rounds,
        ],
        axis=1,
    )

    # -------------------------------------------------------------------------
    # Reindex to the complete calendar
    #
    # This is important. If a month or quarter contains zero interactions,
    # we want an explicit zero row rather than silently losing the period.
    # -------------------------------------------------------------------------

    result = result.reindex(
        full_period_index,
        fill_value=0,
    )

    # -------------------------------------------------------------------------
    # Derived metrics
    # -------------------------------------------------------------------------

    result["interaction_share_pct"] = (
        result["interactions"]
        / len(df)
        * 100
    )

    result["events_per_active_investor"] = np.where(
        result["active_investors"] > 0,
        result["interactions"] / result["active_investors"],
        np.nan,
    )

    result["events_per_active_startup"] = np.where(
        result["active_startups"] > 0,
        result["interactions"] / result["active_startups"],
        np.nan,
    )

    result["events_per_active_pair"] = np.where(
        result["active_investor_startup_pairs"] > 0,
        result["interactions"]
        / result["active_investor_startup_pairs"],
        np.nan,
    )

    result["investors_per_funding_round"] = np.where(
        result["funding_rounds"] > 0,
        result["interactions"] / result["funding_rounds"],
        np.nan,
    )

    # Change from immediately preceding calendar period.
    result["interactions_period_change_pct"] = (
        result["interactions"]
        .pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan)
        * 100
    )

    result["active_investors_period_change_pct"] = (
        result["active_investors"]
        .pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan)
        * 100
    )

    result["active_startups_period_change_pct"] = (
        result["active_startups"]
        .pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan)
        * 100
    )

    result["active_pairs_period_change_pct"] = (
        result["active_investor_startup_pairs"]
        .pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan)
        * 100
    )

    return result


def add_new_entity_counts(
    df,
    result,
    period_col,
):

    # -------------------------------------------------------------------------
    # First observed periods
    #
    # Again, "new" means newly observed in our canonical investment history,
    # NOT real-world entity creation.
    # -------------------------------------------------------------------------

    investor_first = (
        df.groupby(INVESTOR_COL, observed=True)[period_col]
        .min()
    )

    startup_first = (
        df.groupby(STARTUP_COL, observed=True)[period_col]
        .min()
    )

    pair_first = (
        df.groupby(
            [INVESTOR_COL, STARTUP_COL],
            observed=True,
        )[period_col]
        .min()
    )

    new_investors = (
        investor_first
        .value_counts()
        .rename("new_investors")
    )

    new_startups = (
        startup_first
        .value_counts()
        .rename("new_startups")
    )

    new_pairs = (
        pair_first
        .value_counts()
        .rename("new_investor_startup_pairs")
    )

    result = result.join(
        [
            new_investors,
            new_startups,
            new_pairs,
        ],
        how="left",
    )

    new_cols = [
        "new_investors",
        "new_startups",
        "new_investor_startup_pairs",
    ]

    result[new_cols] = (
        result[new_cols]
        .fillna(0)
        .astype("int64")
    )

    return result


def main():

    separator()
    print("PHASE 2.1.3 — MONTHLY AND QUARTERLY TEMPORAL DISTRIBUTION")
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

    print(f"\nInteractions loaded: {len(df):,}")
    print(
        f"Date range: "
        f"{df[DATE_COL].min().date()} -> "
        f"{df[DATE_COL].max().date()}"
    )

    # -------------------------------------------------------------------------
    # 2. Construct calendar period variables
    # -------------------------------------------------------------------------

    df["month"] = df[DATE_COL].dt.to_period("M")
    df["quarter"] = df[DATE_COL].dt.to_period("Q")

    min_month = df["month"].min()
    max_month = df["month"].max()

    min_quarter = df["quarter"].min()
    max_quarter = df["quarter"].max()

    full_month_index = pd.period_range(
        start=min_month,
        end=max_month,
        freq="M",
        name="month",
    )

    full_quarter_index = pd.period_range(
        start=min_quarter,
        end=max_quarter,
        freq="Q",
        name="quarter",
    )

    print(
        f"Calendar months covered: "
        f"{len(full_month_index):,}"
    )

    print(
        f"Calendar quarters covered: "
        f"{len(full_quarter_index):,}"
    )

    # -------------------------------------------------------------------------
    # 3. Build monthly table
    # -------------------------------------------------------------------------

    monthly = build_period_table(
        df,
        "month",
        full_month_index,
    )

    monthly = add_new_entity_counts(
        df,
        monthly,
        "month",
    )

    # Month-over-month comparison against same month one year earlier.
    # This is useful later because it reduces ordinary seasonality effects.
    monthly["interactions_yoy_pct"] = (
        monthly["interactions"]
        .pct_change(
            periods=12,
            fill_method=None,
        )
        .replace([np.inf, -np.inf], np.nan)
        * 100
    )

    monthly["active_investors_yoy_pct"] = (
        monthly["active_investors"]
        .pct_change(
            periods=12,
            fill_method=None,
        )
        .replace([np.inf, -np.inf], np.nan)
        * 100
    )

    monthly["active_startups_yoy_pct"] = (
        monthly["active_startups"]
        .pct_change(
            periods=12,
            fill_method=None,
        )
        .replace([np.inf, -np.inf], np.nan)
        * 100
    )

    # -------------------------------------------------------------------------
    # 4. Build quarterly table
    # -------------------------------------------------------------------------

    quarterly = build_period_table(
        df,
        "quarter",
        full_quarter_index,
    )

    quarterly = add_new_entity_counts(
        df,
        quarterly,
        "quarter",
    )

    quarterly["interactions_yoy_pct"] = (
        quarterly["interactions"]
        .pct_change(
            periods=4,
            fill_method=None,
        )
        .replace([np.inf, -np.inf], np.nan)
        * 100
    )

    quarterly["active_investors_yoy_pct"] = (
        quarterly["active_investors"]
        .pct_change(
            periods=4,
            fill_method=None,
        )
        .replace([np.inf, -np.inf], np.nan)
        * 100
    )

    quarterly["active_startups_yoy_pct"] = (
        quarterly["active_startups"]
        .pct_change(
            periods=4,
            fill_method=None,
        )
        .replace([np.inf, -np.inf], np.nan)
        * 100
    )

    # -------------------------------------------------------------------------
    # 5. Convert count columns explicitly
    # -------------------------------------------------------------------------

    count_cols = [
        "interactions",
        "active_investors",
        "active_startups",
        "active_investor_startup_pairs",
        "funding_rounds",
        "new_investors",
        "new_startups",
        "new_investor_startup_pairs",
    ]

    monthly[count_cols] = (
        monthly[count_cols]
        .astype("int64")
    )

    quarterly[count_cols] = (
        quarterly[count_cols]
        .astype("int64")
    )

    # -------------------------------------------------------------------------
    # 6. Save results
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    monthly_output = monthly.reset_index()
    quarterly_output = quarterly.reset_index()

    # Convert Period objects to strings for clean CSV output.
    monthly_output["month"] = (
        monthly_output["month"].astype(str)
    )

    quarterly_output["quarter"] = (
        quarterly_output["quarter"].astype(str)
    )

    monthly_output.to_csv(
        MONTHLY_OUTPUT,
        index=False,
    )

    quarterly_output.to_csv(
        QUARTERLY_OUTPUT,
        index=False,
    )

    # -------------------------------------------------------------------------
    # 7. Missing / zero-activity calendar periods
    # -------------------------------------------------------------------------

    zero_months = monthly[
        monthly["interactions"] == 0
    ]

    zero_quarters = quarterly[
        quarterly["interactions"] == 0
    ]

    separator("-")
    print("CALENDAR COVERAGE")
    separator("-")

    print(
        f"Total calendar months in range:      "
        f"{len(monthly):,}"
    )

    print(
        f"Months with >=1 interaction:         "
        f"{(monthly['interactions'] > 0).sum():,}"
    )

    print(
        f"Zero-interaction months:             "
        f"{len(zero_months):,}"
    )

    print(
        f"\nTotal calendar quarters in range:    "
        f"{len(quarterly):,}"
    )

    print(
        f"Quarters with >=1 interaction:       "
        f"{(quarterly['interactions'] > 0).sum():,}"
    )

    print(
        f"Zero-interaction quarters:           "
        f"{len(zero_quarters):,}"
    )

    if len(zero_months) > 0:
        print("\nFirst 30 zero-interaction months:")
        print(
            ", ".join(
                zero_months.index.astype(str)[:30]
            )
        )

    if len(zero_quarters) > 0:
        print("\nZero-interaction quarters:")
        print(
            ", ".join(
                zero_quarters.index.astype(str)
            )
        )

    # -------------------------------------------------------------------------
    # 8. Recent monthly activity
    # -------------------------------------------------------------------------

    separator("-")
    print("MOST RECENT 36 MONTHS")
    separator("-")

    monthly_display = monthly.tail(36)[
        [
            "interactions",
            "active_investors",
            "active_startups",
            "active_investor_startup_pairs",
            "funding_rounds",
            "new_investors",
            "new_startups",
            "interactions_period_change_pct",
            "interactions_yoy_pct",
        ]
    ].copy()

    print(
        monthly_display.to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )

    # -------------------------------------------------------------------------
    # 9. Recent quarterly activity
    # -------------------------------------------------------------------------

    separator("-")
    print("MOST RECENT 20 QUARTERS")
    separator("-")

    quarterly_display = quarterly.tail(20)[
        [
            "interactions",
            "active_investors",
            "active_startups",
            "active_investor_startup_pairs",
            "funding_rounds",
            "new_investors",
            "new_startups",
            "interactions_period_change_pct",
            "interactions_yoy_pct",
        ]
    ].copy()

    print(
        quarterly_display.to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )

    # -------------------------------------------------------------------------
    # 10. Highest-activity periods
    # -------------------------------------------------------------------------

    separator("-")
    print("TOP 15 MONTHS BY INTERACTION COUNT")
    separator("-")

    print(
        monthly
        .nlargest(15, "interactions")[
            [
                "interactions",
                "active_investors",
                "active_startups",
                "funding_rounds",
            ]
        ]
        .to_string()
    )

    separator("-")
    print("TOP 10 QUARTERS BY INTERACTION COUNT")
    separator("-")

    print(
        quarterly
        .nlargest(10, "interactions")[
            [
                "interactions",
                "active_investors",
                "active_startups",
                "funding_rounds",
            ]
        ]
        .to_string()
    )

    # -------------------------------------------------------------------------
    # 11. Integrity checks
    # -------------------------------------------------------------------------

    separator("-")
    print("GLOBAL CONSISTENCY CHECKS")
    separator("-")

    print(
        f"Monthly interaction sum:       "
        f"{monthly['interactions'].sum():,}"
    )

    print(
        f"Quarterly interaction sum:     "
        f"{quarterly['interactions'].sum():,}"
    )

    print(
        f"Canonical interaction count:   "
        f"{EXPECTED_ROWS:,}"
    )

    if monthly["interactions"].sum() != EXPECTED_ROWS:
        raise ValueError(
            "Monthly interaction total does not match canonical data."
        )

    if quarterly["interactions"].sum() != EXPECTED_ROWS:
        raise ValueError(
            "Quarterly interaction total does not match canonical data."
        )

    # -------------------------------------------------------------------------
    # 12. Output summary
    # -------------------------------------------------------------------------

    separator()
    print("PHASE 2.1.3 AUDIT COMPLETE")
    separator()

    print(f"""
Outputs written to:

{MONTHLY_OUTPUT}
{QUARTERLY_OUTPUT}

No temporal cutoff has been selected.
No historical interaction has been removed.
No 60-segment construction has been attempted.
No minimum investor-history requirement has been imposed.
No investment types have been filtered.

The next step is Phase 2.1.4 — Recent-Period Completeness Audit.
""")


if __name__ == "__main__":
    main()