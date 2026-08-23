from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PHASE 2.1.2 — ANNUAL TEMPORAL DISTRIBUTION
# =============================================================================

INPUT_PATH = Path("data/processed/interactions.parquet")

OUTPUT_DIR = Path("data/experimental/phase_2/audits")
OUTPUT_PATH = OUTPUT_DIR / "annual_temporal_distribution.csv"

EXPECTED_ROWS = 1_208_051

DATE_COL = "announced_on"
INVESTOR_COL = "investor_id"
STARTUP_COL = "startup_id"
ROUND_COL = "funding_round_id"


def separator(char="=", width=100):
    print(char * width)


def main():

    separator()
    print("PHASE 2.1.2 — ANNUAL TEMPORAL DISTRIBUTION")
    separator()

    # -------------------------------------------------------------------------
    # 1. Load only the canonical columns needed for this audit
    # -------------------------------------------------------------------------

    required_columns = [
        "interaction_id",
        DATE_COL,
        INVESTOR_COL,
        STARTUP_COL,
        ROUND_COL,
    ]

    print("\nLoading canonical interaction data...")

    df = pd.read_parquet(
        INPUT_PATH,
        columns=required_columns,
    )

    print(f"Rows loaded: {len(df):,}")

    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} canonical interactions, "
            f"but loaded {len(df):,}."
        )

    # -------------------------------------------------------------------------
    # 2. Parse canonical event date
    # -------------------------------------------------------------------------

    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL],
        errors="raise",
    )

    if df[DATE_COL].isna().any():
        raise ValueError("Unexpected missing values in announced_on.")

    df["year"] = df[DATE_COL].dt.year.astype("int16")

    print(
        f"Temporal range: "
        f"{df[DATE_COL].min().date()} -> "
        f"{df[DATE_COL].max().date()}"
    )

    print(
        f"Year range: "
        f"{df['year'].min()} -> "
        f"{df['year'].max()}"
    )

    # -------------------------------------------------------------------------
    # 3. Core annual activity metrics
    # -------------------------------------------------------------------------

    interactions = (
        df.groupby("year", observed=True)
        .size()
        .rename("interactions")
    )

    active_investors = (
        df.groupby("year", observed=True)[INVESTOR_COL]
        .nunique()
        .rename("active_investors")
    )

    active_startups = (
        df.groupby("year", observed=True)[STARTUP_COL]
        .nunique()
        .rename("active_startups")
    )

    funding_rounds = (
        df.groupby("year", observed=True)[ROUND_COL]
        .nunique()
        .rename("funding_rounds")
    )

    # A pair is counted once per year even if the same investor invested
    # in the same startup multiple times during that calendar year.
    annual_pairs = (
        df[
            [
                "year",
                INVESTOR_COL,
                STARTUP_COL,
            ]
        ]
        .drop_duplicates()
        .groupby("year", observed=True)
        .size()
        .rename("active_investor_startup_pairs")
    )

    annual = pd.concat(
        [
            interactions,
            active_investors,
            active_startups,
            annual_pairs,
            funding_rounds,
        ],
        axis=1,
    ).sort_index()

    # -------------------------------------------------------------------------
    # 4. First-observed-year metrics
    # -------------------------------------------------------------------------
    #
    # These describe expansion of the observed Crunchbase investment graph.
    # They DO NOT imply that the real-world entity was created in this year.
    # They mean only: first canonical interaction observed in our dataset.
    # -------------------------------------------------------------------------

    investor_first_year = (
        df.groupby(INVESTOR_COL, observed=True)["year"]
        .min()
    )

    startup_first_year = (
        df.groupby(STARTUP_COL, observed=True)["year"]
        .min()
    )

    pair_first_year = (
        df.groupby(
            [INVESTOR_COL, STARTUP_COL],
            observed=True,
        )["year"]
        .min()
    )

    new_investors = (
        investor_first_year
        .value_counts()
        .sort_index()
        .rename("new_investors")
    )

    new_startups = (
        startup_first_year
        .value_counts()
        .sort_index()
        .rename("new_startups")
    )

    new_pairs = (
        pair_first_year
        .value_counts()
        .sort_index()
        .rename("new_investor_startup_pairs")
    )

    annual = annual.join(
        [
            new_investors,
            new_startups,
            new_pairs,
        ],
        how="left",
    )

    annual[
        [
            "new_investors",
            "new_startups",
            "new_investor_startup_pairs",
        ]
    ] = (
        annual[
            [
                "new_investors",
                "new_startups",
                "new_investor_startup_pairs",
            ]
        ]
        .fillna(0)
        .astype("int64")
    )

    # -------------------------------------------------------------------------
    # 5. Derived activity measures
    # -------------------------------------------------------------------------

    total_interactions = len(df)

    annual["interaction_share_pct"] = (
        annual["interactions"]
        / total_interactions
        * 100
    )

    annual["cumulative_interactions"] = (
        annual["interactions"].cumsum()
    )

    annual["cumulative_interaction_share_pct"] = (
        annual["cumulative_interactions"]
        / total_interactions
        * 100
    )

    annual["events_per_active_investor"] = (
        annual["interactions"]
        / annual["active_investors"]
    )

    annual["events_per_active_startup"] = (
        annual["interactions"]
        / annual["active_startups"]
    )

    annual["events_per_active_pair"] = (
        annual["interactions"]
        / annual["active_investor_startup_pairs"]
    )

    annual["investors_per_funding_round"] = (
        annual["interactions"]
        / annual["funding_rounds"]
    )

    # -------------------------------------------------------------------------
    # 6. Year-over-year change
    # -------------------------------------------------------------------------

    annual["interactions_yoy_pct"] = (
        annual["interactions"]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
        * 100
    )

    annual["active_investors_yoy_pct"] = (
        annual["active_investors"]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
        * 100
    )

    annual["active_startups_yoy_pct"] = (
        annual["active_startups"]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
        * 100
    )

    annual["active_pairs_yoy_pct"] = (
        annual["active_investor_startup_pairs"]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
        * 100
    )

    # -------------------------------------------------------------------------
    # 7. Calendar-date diagnostics
    # -------------------------------------------------------------------------
    #
    # Several very early observations from Phase 2.1.1 occurred on January 1.
    # We should measure this rather than assume whether those dates are exact
    # or year-level placeholders.
    # -------------------------------------------------------------------------

    january_first = (
        (df[DATE_COL].dt.month == 1)
        & (df[DATE_COL].dt.day == 1)
    )

    jan1_by_year = (
        df.loc[january_first]
        .groupby("year", observed=True)
        .size()
        .rename("jan_1_interactions")
    )

    annual = annual.join(
        jan1_by_year,
        how="left",
    )

    annual["jan_1_interactions"] = (
        annual["jan_1_interactions"]
        .fillna(0)
        .astype("int64")
    )

    annual["jan_1_share_pct"] = (
        annual["jan_1_interactions"]
        / annual["interactions"]
        * 100
    )

    # -------------------------------------------------------------------------
    # 8. Ensure integer count columns
    # -------------------------------------------------------------------------

    count_columns = [
        "interactions",
        "active_investors",
        "active_startups",
        "active_investor_startup_pairs",
        "funding_rounds",
        "new_investors",
        "new_startups",
        "new_investor_startup_pairs",
        "cumulative_interactions",
        "jan_1_interactions",
    ]

    annual[count_columns] = annual[count_columns].astype("int64")

    # -------------------------------------------------------------------------
    # 9. Save audit table
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    annual_output = annual.reset_index()

    annual_output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # -------------------------------------------------------------------------
    # 10. Print main annual table
    # -------------------------------------------------------------------------

    separator("-")
    print("ANNUAL TEMPORAL DISTRIBUTION")
    separator("-")

    display_columns = [
        "interactions",
        "active_investors",
        "active_startups",
        "active_investor_startup_pairs",
        "funding_rounds",
        "new_investors",
        "new_startups",
        "new_investor_startup_pairs",
        "interaction_share_pct",
        "cumulative_interaction_share_pct",
        "jan_1_share_pct",
    ]

    display = annual[display_columns].copy()

    for col in [
        "interaction_share_pct",
        "cumulative_interaction_share_pct",
        "jan_1_share_pct",
    ]:
        display[col] = display[col].map(
            lambda x: f"{x:.4f}%"
        )

    print(display.to_string())

    # -------------------------------------------------------------------------
    # 11. Recent-year view
    # -------------------------------------------------------------------------

    separator("-")
    print("MOST RECENT 15 CALENDAR YEARS")
    separator("-")

    recent = annual.tail(15)[
        [
            "interactions",
            "active_investors",
            "active_startups",
            "active_investor_startup_pairs",
            "funding_rounds",
            "interaction_share_pct",
            "interactions_yoy_pct",
        ]
    ].copy()

    for col in [
        "interaction_share_pct",
        "interactions_yoy_pct",
    ]:
        recent[col] = recent[col].map(
            lambda x: (
                "NA"
                if pd.isna(x)
                else f"{x:.2f}%"
            )
        )

    print(recent.to_string())

    # -------------------------------------------------------------------------
    # 12. Global consistency checks
    # -------------------------------------------------------------------------

    separator("-")
    print("GLOBAL CONSISTENCY CHECKS")
    separator("-")

    print(
        f"Annual interaction sum:             "
        f"{annual['interactions'].sum():,}"
    )

    print(
        f"Expected canonical interactions:    "
        f"{EXPECTED_ROWS:,}"
    )

    print(
        f"Unique investors from first-years:  "
        f"{annual['new_investors'].sum():,}"
    )

    print(
        f"Expected unique investors:          "
        f"{df[INVESTOR_COL].nunique():,}"
    )

    print(
        f"Unique startups from first-years:   "
        f"{annual['new_startups'].sum():,}"
    )

    print(
        f"Expected unique startups:           "
        f"{df[STARTUP_COL].nunique():,}"
    )

    print(
        f"Unique pairs from first-years:      "
        f"{annual['new_investor_startup_pairs'].sum():,}"
    )

    print(
        f"Expected unique pairs:              "
        f"{df[[INVESTOR_COL, STARTUP_COL]].drop_duplicates().shape[0]:,}"
    )

    # -------------------------------------------------------------------------
    # 13. Audit status
    # -------------------------------------------------------------------------

    separator()
    print("PHASE 2.1.2 AUDIT COMPLETE")
    separator()

    print(f"""
Output written to:

{OUTPUT_PATH}

No temporal cutoff has been selected.
No historical observations have been removed.
No 60-segment construction has been attempted.
No minimum investor-history requirement has been imposed.
No investment types have been filtered.

The next audit should examine the temporal distribution at MONTHLY and
QUARTERLY resolution before interpreting recent-period completeness.
""")


if __name__ == "__main__":
    main()