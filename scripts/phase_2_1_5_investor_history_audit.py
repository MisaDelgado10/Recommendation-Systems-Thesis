from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PHASE 2.1.5 — INVESTOR HISTORY / SEQUENCE-LENGTH DISTRIBUTION
# =============================================================================

INPUT_PATH = Path(
    "data/processed/interactions.parquet"
)

OUTPUT_DIR = Path(
    "data/experimental/phase_2/audits"
)

INVESTOR_SUMMARY_OUTPUT = (
    OUTPUT_DIR / "investor_history_summary.csv"
)

SEQUENCE_EXACT_OUTPUT = (
    OUTPUT_DIR / "investor_sequence_length_exact.csv"
)

SEQUENCE_BUCKET_OUTPUT = (
    OUTPUT_DIR / "investor_sequence_length_buckets.csv"
)

QUANTILES_OUTPUT = (
    OUTPUT_DIR / "investor_history_quantiles.csv"
)

GAP_QUANTILES_OUTPUT = (
    OUTPUT_DIR / "investor_interevent_gap_quantiles.csv"
)

SAME_DAY_OUTPUT = (
    OUTPUT_DIR / "investor_same_day_ties.csv"
)


EXPECTED_ROWS = 1_208_051
EXPECTED_INVESTORS = 165_975

DATE_COL = "announced_on"
INVESTOR_COL = "investor_id"
STARTUP_COL = "startup_id"
ROUND_COL = "funding_round_id"

# From Phase 2.1.4:
# This is NOT the final modeling-window cutoff.
# It is only the last fully observed calendar month.
LAST_COMPLETE_MONTH_END = pd.Timestamp("2026-05-31")


def separator(char="=", width=110):
    print(char * width)


def build_investor_summary(df):

    # -------------------------------------------------------------------------
    # 1. Basic per-investor history
    # -------------------------------------------------------------------------

    summary = (
        df.groupby(
            INVESTOR_COL,
            observed=True,
        )
        .agg(
            interaction_events=(
                "interaction_id",
                "size",
            ),
            unique_startups=(
                STARTUP_COL,
                "nunique",
            ),
            unique_funding_rounds=(
                ROUND_COL,
                "nunique",
            ),
            active_dates=(
                DATE_COL,
                "nunique",
            ),
            first_event_date=(
                DATE_COL,
                "min",
            ),
            last_event_date=(
                DATE_COL,
                "max",
            ),
        )
    )

    # -------------------------------------------------------------------------
    # 2. Active months / quarters / years
    # -------------------------------------------------------------------------

    active_months = (
        df.groupby(
            INVESTOR_COL,
            observed=True,
        )["month"]
        .nunique()
        .rename("active_months")
    )

    active_quarters = (
        df.groupby(
            INVESTOR_COL,
            observed=True,
        )["quarter"]
        .nunique()
        .rename("active_quarters")
    )

    active_years = (
        df.groupby(
            INVESTOR_COL,
            observed=True,
        )["year"]
        .nunique()
        .rename("active_years")
    )

    summary = summary.join(
        [
            active_months,
            active_quarters,
            active_years,
        ]
    )

    # -------------------------------------------------------------------------
    # 3. History span
    # -------------------------------------------------------------------------

    summary["history_span_days"] = (
        summary["last_event_date"]
        - summary["first_event_date"]
    ).dt.days

    summary["history_span_years"] = (
        summary["history_span_days"] / 365.25
    )

    # -------------------------------------------------------------------------
    # 4. Repeat-startup behavior
    #
    # interaction_events - unique_startups gives the number of event slots
    # beyond the first observed event with each startup.
    # -------------------------------------------------------------------------

    summary["repeat_startup_event_excess"] = (
        summary["interaction_events"]
        - summary["unique_startups"]
    )

    summary["repeat_startup_event_share_pct"] = (
        summary["repeat_startup_event_excess"]
        / summary["interaction_events"]
        * 100
    )

    summary["has_repeat_startup"] = (
        summary["repeat_startup_event_excess"] > 0
    )

    # -------------------------------------------------------------------------
    # 5. Sequence-density measures
    # -------------------------------------------------------------------------

    summary["events_per_active_month"] = (
        summary["interaction_events"]
        / summary["active_months"]
    )

    summary["events_per_active_quarter"] = (
        summary["interaction_events"]
        / summary["active_quarters"]
    )

    summary["events_per_active_year"] = (
        summary["interaction_events"]
        / summary["active_years"]
    )

    summary["events_per_active_date"] = (
        summary["interaction_events"]
        / summary["active_dates"]
    )

    return summary


def main():

    separator()
    print(
        "PHASE 2.1.5 — INVESTOR HISTORY / "
        "SEQUENCE-LENGTH DISTRIBUTION"
    )
    separator()

    # -------------------------------------------------------------------------
    # 1. Load canonical data
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
            f"Expected {EXPECTED_ROWS:,} rows; "
            f"found {len(df):,}."
        )

    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL],
        errors="raise",
    )

    print(f"\nCanonical interactions: {len(df):,}")
    print(
        f"Canonical investors:    "
        f"{df[INVESTOR_COL].nunique():,}"
    )

    print(
        f"Date range:             "
        f"{df[DATE_COL].min().date()} -> "
        f"{df[DATE_COL].max().date()}"
    )

    # -------------------------------------------------------------------------
    # 2. Add period variables
    # -------------------------------------------------------------------------

    df["month"] = (
        df[DATE_COL].dt.to_period("M")
    )

    df["quarter"] = (
        df[DATE_COL].dt.to_period("Q")
    )

    df["year"] = (
        df[DATE_COL].dt.year
    )

    # -------------------------------------------------------------------------
    # 3. Build investor summary
    # -------------------------------------------------------------------------

    investor_summary = build_investor_summary(df)

    if len(investor_summary) != EXPECTED_INVESTORS:
        raise ValueError(
            f"Expected {EXPECTED_INVESTORS:,} investors; "
            f"found {len(investor_summary):,}."
        )

    # -------------------------------------------------------------------------
    # 4. Validate investor-round uniqueness
    #
    # By Phase-1 definition, one interaction is one investor participating
    # in one funding round. Therefore each investor's interaction count
    # should equal that investor's number of unique funding rounds.
    # -------------------------------------------------------------------------

    round_mismatch = (
        investor_summary["interaction_events"]
        != investor_summary["unique_funding_rounds"]
    )

    num_round_mismatch = int(round_mismatch.sum())

    # -------------------------------------------------------------------------
    # 5. Exact sequence-length distribution
    # -------------------------------------------------------------------------

    exact_sequence = (
        investor_summary["interaction_events"]
        .value_counts()
        .sort_index()
        .rename_axis("sequence_length")
        .reset_index(name="investor_count")
    )

    exact_sequence["investor_share_pct"] = (
        exact_sequence["investor_count"]
        / EXPECTED_INVESTORS
        * 100
    )

    exact_sequence["interactions_contributed"] = (
        exact_sequence["sequence_length"]
        * exact_sequence["investor_count"]
    )

    exact_sequence["interaction_share_pct"] = (
        exact_sequence["interactions_contributed"]
        / EXPECTED_ROWS
        * 100
    )

    # -------------------------------------------------------------------------
    # 6. Sequence-length buckets
    # -------------------------------------------------------------------------

    bucket_edges = [
        0,
        1,
        2,
        5,
        10,
        20,
        50,
        100,
        250,
        500,
        np.inf,
    ]

    bucket_labels = [
        "1",
        "2",
        "3-5",
        "6-10",
        "11-20",
        "21-50",
        "51-100",
        "101-250",
        "251-500",
        "501+",
    ]

    investor_summary["sequence_length_bucket"] = pd.cut(
        investor_summary["interaction_events"],
        bins=bucket_edges,
        labels=bucket_labels,
        include_lowest=True,
        right=True,
    )

    bucket_summary = (
        investor_summary.groupby(
            "sequence_length_bucket",
            observed=False,
        )
        .agg(
            investor_count=(
                "interaction_events",
                "size",
            ),
            interactions_contributed=(
                "interaction_events",
                "sum",
            ),
            median_unique_startups=(
                "unique_startups",
                "median",
            ),
            median_active_months=(
                "active_months",
                "median",
            ),
            median_active_years=(
                "active_years",
                "median",
            ),
            median_history_span_years=(
                "history_span_years",
                "median",
            ),
        )
        .reset_index()
    )

    bucket_summary["investor_share_pct"] = (
        bucket_summary["investor_count"]
        / EXPECTED_INVESTORS
        * 100
    )

    bucket_summary["interaction_share_pct"] = (
        bucket_summary["interactions_contributed"]
        / EXPECTED_ROWS
        * 100
    )

    # -------------------------------------------------------------------------
    # 7. Global investor-history quantiles
    # -------------------------------------------------------------------------

    quantile_columns = [
        "interaction_events",
        "unique_startups",
        "active_dates",
        "active_months",
        "active_quarters",
        "active_years",
        "history_span_days",
        "history_span_years",
        "events_per_active_month",
        "events_per_active_year",
        "repeat_startup_event_share_pct",
    ]

    quantile_levels = [
        0.00,
        0.01,
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
        0.995,
        0.999,
        1.00,
    ]

    quantiles = (
        investor_summary[quantile_columns]
        .quantile(quantile_levels)
        .T
    )

    quantiles.columns = [
        f"q_{int(q * 1000):04d}"
        for q in quantile_levels
    ]

    quantiles = quantiles.reset_index(
        names="metric"
    )

    # -------------------------------------------------------------------------
    # 8. Inter-event gap reconstruction
    #
    # Same-day events will legitimately produce gap_days = 0.
    # -------------------------------------------------------------------------

    ordered = df.sort_values(
        [
            INVESTOR_COL,
            DATE_COL,
            ROUND_COL,
        ],
        kind="mergesort",
    ).copy()

    ordered["previous_event_date"] = (
        ordered.groupby(
            INVESTOR_COL,
            observed=True,
        )[DATE_COL]
        .shift(1)
    )

    ordered["gap_days"] = (
        ordered[DATE_COL]
        - ordered["previous_event_date"]
    ).dt.days

    gap_values = (
        ordered["gap_days"]
        .dropna()
    )

    positive_gap_values = (
        gap_values[gap_values > 0]
    )

    gap_levels = [
        0.00,
        0.01,
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
        1.00,
    ]

    gap_rows = []

    for name, values in [
        ("all_consecutive_gaps", gap_values),
        ("positive_consecutive_gaps", positive_gap_values),
    ]:

        row = {
            "gap_set": name,
            "count": len(values),
            "mean_days": values.mean(),
        }

        q = values.quantile(gap_levels)

        for level, value in q.items():
            row[
                f"q_{int(level * 1000):04d}"
            ] = value

        gap_rows.append(row)

    gap_quantiles = pd.DataFrame(gap_rows)

    # -------------------------------------------------------------------------
    # 9. Same-day investor-event ties
    #
    # A tie means one investor has >1 canonical investment event carrying
    # the same announced_on value.
    # -------------------------------------------------------------------------

    same_day_counts = (
        df.groupby(
            [
                INVESTOR_COL,
                DATE_COL,
            ],
            observed=True,
        )
        .size()
        .rename("events_same_day")
        .reset_index()
    )

    same_day_ties = (
        same_day_counts[
            same_day_counts["events_same_day"] > 1
        ]
        .copy()
    )

    tied_investors = (
        same_day_ties[INVESTOR_COL]
        .nunique()
    )

    tied_groups = len(same_day_ties)

    tied_interactions = (
        same_day_ties["events_same_day"].sum()
    )

    max_same_day_events = (
        same_day_ties["events_same_day"].max()
        if len(same_day_ties) > 0
        else 0
    )

    # -------------------------------------------------------------------------
    # 10. Complete-calendar-month sensitivity check
    #
    # We do NOT replace the canonical analysis with this view.
    # We only quantify the effect of excluding objectively partial June 2026.
    # -------------------------------------------------------------------------

    complete_period_df = df[
        df[DATE_COL] <= LAST_COMPLETE_MONTH_END
    ]

    excluded_terminal_interactions = (
        len(df) - len(complete_period_df)
    )

    affected_terminal_investors = (
        df.loc[
            df[DATE_COL] > LAST_COMPLETE_MONTH_END,
            INVESTOR_COL,
        ]
        .nunique()
    )

    # -------------------------------------------------------------------------
    # 11. Save outputs
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    investor_summary.reset_index().to_csv(
        INVESTOR_SUMMARY_OUTPUT,
        index=False,
    )

    exact_sequence.to_csv(
        SEQUENCE_EXACT_OUTPUT,
        index=False,
    )

    bucket_summary.to_csv(
        SEQUENCE_BUCKET_OUTPUT,
        index=False,
    )

    quantiles.to_csv(
        QUANTILES_OUTPUT,
        index=False,
    )

    gap_quantiles.to_csv(
        GAP_QUANTILES_OUTPUT,
        index=False,
    )

    same_day_ties.to_csv(
        SAME_DAY_OUTPUT,
        index=False,
    )

    # -------------------------------------------------------------------------
    # 12. Print high-level sequence statistics
    # -------------------------------------------------------------------------

    separator("-")
    print("GLOBAL INVESTOR HISTORY SUMMARY")
    separator("-")

    print(
        f"Investors:                         "
        f"{len(investor_summary):,}"
    )

    print(
        f"Interactions:                      "
        f"{investor_summary['interaction_events'].sum():,}"
    )

    print(
        f"Minimum sequence length:           "
        f"{investor_summary['interaction_events'].min():,}"
    )

    print(
        f"Maximum sequence length:           "
        f"{investor_summary['interaction_events'].max():,}"
    )

    print(
        f"Mean sequence length:              "
        f"{investor_summary['interaction_events'].mean():.3f}"
    )

    print(
        f"Median sequence length:            "
        f"{investor_summary['interaction_events'].median():.1f}"
    )

    print(
        f"Investors with repeat startups:    "
        f"{investor_summary['has_repeat_startup'].sum():,}"
    )

    print(
        f"Repeat-startup investor share:     "
        f"{investor_summary['has_repeat_startup'].mean() * 100:.2f}%"
    )

    print(
        f"Investor-round count mismatches:   "
        f"{num_round_mismatch:,}"
    )

    # -------------------------------------------------------------------------
    # 13. Sequence-length bucket table
    # -------------------------------------------------------------------------

    separator("-")
    print("SEQUENCE-LENGTH BUCKETS")
    separator("-")

    print(
        bucket_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    # -------------------------------------------------------------------------
    # 14. Selected investor-history quantiles
    # -------------------------------------------------------------------------

    separator("-")
    print("INVESTOR-HISTORY QUANTILES")
    separator("-")

    print(
        quantiles.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    # -------------------------------------------------------------------------
    # 15. Inter-event gaps
    # -------------------------------------------------------------------------

    separator("-")
    print("CONSECUTIVE INVESTMENT GAP DISTRIBUTION")
    separator("-")

    print(
        gap_quantiles.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print(
        f"\nZero-day consecutive gaps: "
        f"{(gap_values == 0).sum():,}"
    )

    print(
        f"Zero-day gap share:        "
        f"{(gap_values == 0).mean() * 100:.2f}%"
    )

    # -------------------------------------------------------------------------
    # 16. Same-day tie diagnostics
    # -------------------------------------------------------------------------

    separator("-")
    print("SAME-DAY INVESTOR SEQUENCE TIES")
    separator("-")

    print(
        f"Investors with >=1 same-day tie:   "
        f"{tied_investors:,}"
    )

    print(
        f"Tied investor-date groups:         "
        f"{tied_groups:,}"
    )

    print(
        f"Interactions inside tied groups:   "
        f"{tied_interactions:,}"
    )

    print(
        f"Maximum events by one investor "
        f"on one date:                       "
        f"{max_same_day_events:,}"
    )

    # -------------------------------------------------------------------------
    # 17. Complete-period sensitivity
    # -------------------------------------------------------------------------

    separator("-")
    print("TERMINAL PARTIAL-MONTH SENSITIVITY")
    separator("-")

    print(
        f"Canonical interactions:             "
        f"{len(df):,}"
    )

    print(
        f"Interactions through 2026-05-31:    "
        f"{len(complete_period_df):,}"
    )

    print(
        f"Interactions in partial June 2026:  "
        f"{excluded_terminal_interactions:,}"
    )

    print(
        f"Investors affected by partial June: "
        f"{affected_terminal_investors:,}"
    )

    # -------------------------------------------------------------------------
    # 18. Integrity checks
    # -------------------------------------------------------------------------

    separator("-")
    print("GLOBAL CONSISTENCY CHECKS")
    separator("-")

    print(
        f"Investor-summary event sum:         "
        f"{investor_summary['interaction_events'].sum():,}"
    )

    print(
        f"Canonical interaction count:        "
        f"{EXPECTED_ROWS:,}"
    )

    print(
        f"Investor-summary rows:              "
        f"{len(investor_summary):,}"
    )

    print(
        f"Expected investor count:            "
        f"{EXPECTED_INVESTORS:,}"
    )

    if (
        investor_summary["interaction_events"].sum()
        != EXPECTED_ROWS
    ):
        raise ValueError(
            "Investor sequence counts do not sum to canonical interactions."
        )

    if len(investor_summary) != EXPECTED_INVESTORS:
        raise ValueError(
            "Investor summary does not reproduce canonical investor count."
        )

    if num_round_mismatch != 0:
        raise ValueError(
            "Investor interaction count differs from unique "
            "funding-round count for at least one investor."
        )

    # -------------------------------------------------------------------------
    # 19. End
    # -------------------------------------------------------------------------

    separator()
    print("PHASE 2.1.5 AUDIT COMPLETE")
    separator()

    print(f"""
Outputs written to:

{INVESTOR_SUMMARY_OUTPUT}
{SEQUENCE_EXACT_OUTPUT}
{SEQUENCE_BUCKET_OUTPUT}
{QUANTILES_OUTPUT}
{GAP_QUANTILES_OUTPUT}
{SAME_DAY_OUTPUT}

No minimum investor sequence length has been selected.
No investor has been removed.
No temporal starting cutoff has been selected.
No 60-segment design has been chosen.
No investment-type filtering has been performed.
The canonical Phase-1 dataset remains unchanged.

Next:
Phase 2.1.6 — Temporal Coverage and Candidate-Window Diagnostics.
""")


if __name__ == "__main__":
    main()