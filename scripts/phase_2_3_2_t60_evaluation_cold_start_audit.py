from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PHASE 2.3.2 — T60 EVALUATION & INTERACTION-COLD-START AUDIT
# =============================================================================
#
# PURPOSE
# -------
# Audit the final ITRS evaluation fragment T60 before performing the paper's
# random 10% validation / 90% test split.
#
# IMPORTANT TERMINOLOGY
# ---------------------
#
# "interaction-cold" means the entity has no INVESTMENT INTERACTION before T60.
#
# It does NOT mean that the entity is absent from Crunchbase or from a future
# heterogeneous graph.
#
# No filtering or train/validation/test split is performed in this audit.
# =============================================================================


INPUT_PATH = Path(
    "data/experimental/phase_2/temporal/"
    "interactions_itrs_temporal.parquet"
)

OUTPUT_DIR = Path(
    "data/experimental/phase_2/audits"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "t60_evaluation_cold_start_summary.csv"
)

INTERACTION_FLAGS_OUTPUT = (
    OUTPUT_DIR
    / "t60_evaluation_interaction_flags.parquet"
)

INVESTOR_HISTORY_OUTPUT = (
    OUTPUT_DIR
    / "t60_investor_prior_history.csv"
)

INVESTOR_BUCKET_OUTPUT = (
    OUTPUT_DIR
    / "t60_investor_prior_history_buckets.csv"
)

STARTUP_HISTORY_OUTPUT = (
    OUTPUT_DIR
    / "t60_startup_prior_history.csv"
)

PAIR_OUTPUT = (
    OUTPUT_DIR
    / "t60_pair_novelty_summary.csv"
)


EXPECTED_TEMPORAL_ROWS = 1_195_937

EXPECTED_T60_ROWS = 22_515
EXPECTED_T60_INVESTORS = 11_884
EXPECTED_T60_STARTUPS = 8_992
EXPECTED_T60_PAIRS = 22_327

T60_SEGMENT = 60


def separator(char="=", width=120):
    print(char * width)


def pct(numerator, denominator):

    if denominator == 0:
        return np.nan

    return numerator / denominator * 100


def bucket_event_history(value):

    if value == 0:
        return "0"
    if value == 1:
        return "1"
    if value == 2:
        return "2"
    if value <= 5:
        return "3-5"
    if value <= 10:
        return "6-10"
    if value <= 20:
        return "11-20"
    if value <= 50:
        return "21-50"

    return "51+"


def bucket_segment_history(value):

    if value == 0:
        return "0"
    if value == 1:
        return "1"
    if value == 2:
        return "2"
    if value <= 5:
        return "3-5"
    if value <= 10:
        return "6-10"
    if value <= 20:
        return "11-20"
    if value <= 40:
        return "21-40"

    return "41-60"


def main():

    separator()
    print(
        "PHASE 2.3.2 — "
        "T60 EVALUATION & INTERACTION-COLD-START AUDIT"
    )
    separator()

    # -------------------------------------------------------------------------
    # 1. Load selected temporal interaction layer
    # -------------------------------------------------------------------------

    df = pd.read_parquet(
        INPUT_PATH
    )

    if len(df) != EXPECTED_TEMPORAL_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_TEMPORAL_ROWS:,} temporal interactions; "
            f"found {len(df):,}."
        )

    print(
        f"\nTemporal interactions: "
        f"{len(df):,}"
    )

    print(
        f"Segments present:      "
        f"{df['segment_number'].min()} -> "
        f"{df['segment_number'].max()}"
    )

    # -------------------------------------------------------------------------
    # 2. Separate history and evaluation fragment
    # -------------------------------------------------------------------------

    history = (
        df[
            df["segment_number"] < T60_SEGMENT
        ]
        .copy()
    )

    t0 = (
        history[
            history["segment_number"] == 0
        ]
        .copy()
    )

    detailed_history = (
        history[
            history["segment_number"]
            .between(1, 59)
        ]
        .copy()
    )

    t60 = (
        df[
            df["segment_number"] == T60_SEGMENT
        ]
        .copy()
    )

    if len(t60) != EXPECTED_T60_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_T60_ROWS:,} T60 events; "
            f"found {len(t60):,}."
        )

    # -------------------------------------------------------------------------
    # 3. Basic T60 checks
    # -------------------------------------------------------------------------

    t60_investor_count = (
        t60["investor_id"]
        .nunique()
    )

    t60_startup_count = (
        t60["startup_id"]
        .nunique()
    )

    t60_pair_count = (
        t60[
            [
                "investor_id",
                "startup_id",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    if t60_investor_count != EXPECTED_T60_INVESTORS:
        raise ValueError(
            "Unexpected T60 investor count."
        )

    if t60_startup_count != EXPECTED_T60_STARTUPS:
        raise ValueError(
            "Unexpected T60 startup count."
        )

    if t60_pair_count != EXPECTED_T60_PAIRS:
        raise ValueError(
            "Unexpected T60 pair count."
        )

    # -------------------------------------------------------------------------
    # 4. Historical entity sets
    # -------------------------------------------------------------------------

    prior_investors = set(
        history["investor_id"]
        .unique()
    )

    detailed_prior_investors = set(
        detailed_history["investor_id"]
        .unique()
    )

    t0_investors = set(
        t0["investor_id"]
        .unique()
    )

    prior_startups = set(
        history["startup_id"]
        .unique()
    )

    detailed_prior_startups = set(
        detailed_history["startup_id"]
        .unique()
    )

    t0_startups = set(
        t0["startup_id"]
        .unique()
    )

    # -------------------------------------------------------------------------
    # 5. Add investor/startup prior-history flags to every T60 interaction
    # -------------------------------------------------------------------------

    t60["investor_seen_before_t60"] = (
        t60["investor_id"]
        .isin(prior_investors)
    )

    t60["investor_seen_in_t1_t59"] = (
        t60["investor_id"]
        .isin(detailed_prior_investors)
    )

    t60["investor_seen_in_t0"] = (
        t60["investor_id"]
        .isin(t0_investors)
    )

    t60["investor_t0_only_history"] = (
        t60["investor_seen_in_t0"]
        &
        ~t60["investor_seen_in_t1_t59"]
    )

    t60["startup_seen_before_t60"] = (
        t60["startup_id"]
        .isin(prior_startups)
    )

    t60["startup_seen_in_t1_t59"] = (
        t60["startup_id"]
        .isin(detailed_prior_startups)
    )

    t60["startup_seen_in_t0"] = (
        t60["startup_id"]
        .isin(t0_startups)
    )

    t60["startup_t0_only_history"] = (
        t60["startup_seen_in_t0"]
        &
        ~t60["startup_seen_in_t1_t59"]
    )

    # -------------------------------------------------------------------------
    # 6. Investor-startup pair history
    # -------------------------------------------------------------------------

    prior_pairs = (
        history[
            [
                "investor_id",
                "startup_id",
            ]
        ]
        .drop_duplicates()
        .assign(
            pair_seen_before_t60=True
        )
    )

    detailed_prior_pairs = (
        detailed_history[
            [
                "investor_id",
                "startup_id",
            ]
        ]
        .drop_duplicates()
        .assign(
            pair_seen_in_t1_t59=True
        )
    )

    t60 = t60.merge(
        prior_pairs,
        on=[
            "investor_id",
            "startup_id",
        ],
        how="left",
        validate="many_to_one",
    )

    t60 = t60.merge(
        detailed_prior_pairs,
        on=[
            "investor_id",
            "startup_id",
        ],
        how="left",
        validate="many_to_one",
    )

    t60["pair_seen_before_t60"] = (
        t60["pair_seen_before_t60"]
        .fillna(False)
        .astype(bool)
    )

    t60["pair_seen_in_t1_t59"] = (
        t60["pair_seen_in_t1_t59"]
        .fillna(False)
        .astype(bool)
    )

    t60["new_to_investor_pair"] = (
        ~t60["pair_seen_before_t60"]
    )

    # -------------------------------------------------------------------------
    # 7. Four-way interaction-cold-start status
    # -------------------------------------------------------------------------

    t60["interaction_cold_start_status"] = np.select(
        [
            (
                t60["investor_seen_before_t60"]
                &
                t60["startup_seen_before_t60"]
            ),
            (
                t60["investor_seen_before_t60"]
                &
                ~t60["startup_seen_before_t60"]
            ),
            (
                ~t60["investor_seen_before_t60"]
                &
                t60["startup_seen_before_t60"]
            ),
            (
                ~t60["investor_seen_before_t60"]
                &
                ~t60["startup_seen_before_t60"]
            ),
        ],
        [
            "warm_investor__warm_startup",
            "warm_investor__cold_startup",
            "cold_investor__warm_startup",
            "cold_investor__cold_startup",
        ],
        default="unexpected",
    )

    # -------------------------------------------------------------------------
    # 8. Repeated pairs WITHIN T60
    #
    # This matters because an event-level random validation/test split
    # could place the same investor-startup pair in both subsets.
    # -------------------------------------------------------------------------

    t60_pair_event_counts = (
        t60.groupby(
            [
                "investor_id",
                "startup_id",
            ],
            observed=True,
        )
        .size()
        .rename("t60_pair_event_count")
        .reset_index()
    )

    t60 = t60.merge(
        t60_pair_event_counts,
        on=[
            "investor_id",
            "startup_id",
        ],
        how="left",
        validate="many_to_one",
    )

    t60["pair_repeats_within_t60"] = (
        t60["t60_pair_event_count"] > 1
    )

    repeated_t60_pairs = (
        t60_pair_event_counts[
            t60_pair_event_counts[
                "t60_pair_event_count"
            ] > 1
        ]
    )

    repeated_t60_pair_count = (
        len(repeated_t60_pairs)
    )

    interactions_in_repeated_t60_pairs = (
        repeated_t60_pairs[
            "t60_pair_event_count"
        ]
        .sum()
    )

    max_events_one_pair_t60 = (
        t60_pair_event_counts[
            "t60_pair_event_count"
        ]
        .max()
    )

    # -------------------------------------------------------------------------
    # 9. Build prior-history statistics for T60 investors
    # -------------------------------------------------------------------------

    t60_investor_ids = (
        t60[
            ["investor_id"]
        ]
        .drop_duplicates()
    )

    investor_prior = (
        history.groupby(
            "investor_id",
            observed=True,
        )
        .agg(
            prior_interactions=(
                "interaction_id",
                "size",
            ),
            prior_unique_startups=(
                "startup_id",
                "nunique",
            ),
            prior_active_segments=(
                "segment_number",
                "nunique",
            ),
            first_prior_segment=(
                "segment_number",
                "min",
            ),
            last_prior_segment=(
                "segment_number",
                "max",
            ),
        )
        .reset_index()
    )

    investor_detailed_segments = (
        detailed_history[
            [
                "investor_id",
                "segment_number",
            ]
        ]
        .drop_duplicates()
        .groupby(
            "investor_id",
            observed=True,
        )
        .size()
        .rename(
            "prior_active_detailed_segments"
        )
        .reset_index()
    )

    investor_prior = investor_prior.merge(
        investor_detailed_segments,
        on="investor_id",
        how="left",
        validate="one_to_one",
    )

    investor_prior[
        "prior_active_detailed_segments"
    ] = (
        investor_prior[
            "prior_active_detailed_segments"
        ]
        .fillna(0)
        .astype("int64")
    )

    investor_t0_flag = (
        t0[
            ["investor_id"]
        ]
        .drop_duplicates()
        .assign(
            has_t0_history=True
        )
    )

    investor_prior = investor_prior.merge(
        investor_t0_flag,
        on="investor_id",
        how="left",
        validate="one_to_one",
    )

    investor_prior["has_t0_history"] = (
        investor_prior[
            "has_t0_history"
        ]
        .fillna(False)
        .astype(bool)
    )

    t60_investor_history = (
        t60_investor_ids.merge(
            investor_prior,
            on="investor_id",
            how="left",
            validate="one_to_one",
        )
    )

    fill_zero_cols = [
        "prior_interactions",
        "prior_unique_startups",
        "prior_active_segments",
        "prior_active_detailed_segments",
    ]

    t60_investor_history[
        fill_zero_cols
    ] = (
        t60_investor_history[
            fill_zero_cols
        ]
        .fillna(0)
        .astype("int64")
    )

    t60_investor_history[
        "has_t0_history"
    ] = (
        t60_investor_history[
            "has_t0_history"
        ]
        .fillna(False)
        .astype(bool)
    )

    t60_investor_history[
        "is_interaction_cold_start"
    ] = (
        t60_investor_history[
            "prior_interactions"
        ] == 0
    )

    t60_investor_history[
        "prior_event_history_bucket"
    ] = (
        t60_investor_history[
            "prior_interactions"
        ]
        .map(bucket_event_history)
    )

    t60_investor_history[
        "prior_detailed_segment_bucket"
    ] = (
        t60_investor_history[
            "prior_active_detailed_segments"
        ]
        .map(bucket_segment_history)
    )

    # -------------------------------------------------------------------------
    # 10. Investor-history bucket summary
    # -------------------------------------------------------------------------

    investor_bucket_order = [
        "0",
        "1",
        "2",
        "3-5",
        "6-10",
        "11-20",
        "21-50",
        "51+",
    ]

    investor_bucket_summary = (
        t60_investor_history.groupby(
            "prior_event_history_bucket",
            observed=True,
        )
        .agg(
            investor_count=(
                "investor_id",
                "size",
            ),
            median_prior_unique_startups=(
                "prior_unique_startups",
                "median",
            ),
            median_prior_active_detailed_segments=(
                "prior_active_detailed_segments",
                "median",
            ),
        )
        .reindex(
            investor_bucket_order
        )
        .fillna(0)
        .reset_index()
    )

    investor_bucket_summary[
        "investor_share_pct"
    ] = (
        investor_bucket_summary[
            "investor_count"
        ]
        / t60_investor_count
        * 100
    )

    # -------------------------------------------------------------------------
    # 11. Build prior-history statistics for T60 startups
    # -------------------------------------------------------------------------

    t60_startup_ids = (
        t60[
            ["startup_id"]
        ]
        .drop_duplicates()
    )

    startup_prior = (
        history.groupby(
            "startup_id",
            observed=True,
        )
        .agg(
            prior_investment_interactions=(
                "interaction_id",
                "size",
            ),
            prior_unique_investors=(
                "investor_id",
                "nunique",
            ),
            prior_active_segments=(
                "segment_number",
                "nunique",
            ),
        )
        .reset_index()
    )

    t60_startup_history = (
        t60_startup_ids.merge(
            startup_prior,
            on="startup_id",
            how="left",
            validate="one_to_one",
        )
    )

    startup_fill_cols = [
        "prior_investment_interactions",
        "prior_unique_investors",
        "prior_active_segments",
    ]

    t60_startup_history[
        startup_fill_cols
    ] = (
        t60_startup_history[
            startup_fill_cols
        ]
        .fillna(0)
        .astype("int64")
    )

    t60_startup_history[
        "is_interaction_cold_start"
    ] = (
        t60_startup_history[
            "prior_investment_interactions"
        ] == 0
    )

    # -------------------------------------------------------------------------
    # 12. Pair-level summary
    # -------------------------------------------------------------------------

    t60_unique_pairs = (
        t60[
            [
                "investor_id",
                "startup_id",
                "pair_seen_before_t60",
                "pair_seen_in_t1_t59",
                "new_to_investor_pair",
                "t60_pair_event_count",
            ]
        ]
        .drop_duplicates(
            subset=[
                "investor_id",
                "startup_id",
            ]
        )
    )

    pair_summary = pd.DataFrame(
        [
            {
                "metric": (
                    "unique_t60_pairs"
                ),
                "count": (
                    len(t60_unique_pairs)
                ),
                "share_pct": 100.0,
            },
            {
                "metric": (
                    "new_to_investor_pairs"
                ),
                "count": int(
                    t60_unique_pairs[
                        "new_to_investor_pair"
                    ]
                    .sum()
                ),
                "share_pct": pct(
                    t60_unique_pairs[
                        "new_to_investor_pair"
                    ]
                    .sum(),
                    len(t60_unique_pairs),
                ),
            },
            {
                "metric": (
                    "previously_seen_pairs"
                ),
                "count": int(
                    t60_unique_pairs[
                        "pair_seen_before_t60"
                    ]
                    .sum()
                ),
                "share_pct": pct(
                    t60_unique_pairs[
                        "pair_seen_before_t60"
                    ]
                    .sum(),
                    len(t60_unique_pairs),
                ),
            },
            {
                "metric": (
                    "pairs_repeated_within_t60"
                ),
                "count": int(
                    repeated_t60_pair_count
                ),
                "share_pct": pct(
                    repeated_t60_pair_count,
                    len(t60_unique_pairs),
                ),
            },
        ]
    )

    # -------------------------------------------------------------------------
    # 13. Four-way event-status summary
    # -------------------------------------------------------------------------

    cold_start_status = (
        t60[
            "interaction_cold_start_status"
        ]
        .value_counts()
        .rename_axis(
            "interaction_cold_start_status"
        )
        .reset_index(
            name="interaction_count"
        )
    )

    cold_start_status[
        "interaction_share_pct"
    ] = (
        cold_start_status[
            "interaction_count"
        ]
        / len(t60)
        * 100
    )

    # -------------------------------------------------------------------------
    # 14. Main summary
    # -------------------------------------------------------------------------

    unique_cold_investors = (
        t60_investor_history[
            "is_interaction_cold_start"
        ]
        .sum()
    )

    unique_cold_startups = (
        t60_startup_history[
            "is_interaction_cold_start"
        ]
        .sum()
    )

    t0_only_t60_investors = (
        t60[
            [
                "investor_id",
                "investor_t0_only_history",
            ]
        ]
        .drop_duplicates(
            subset=["investor_id"]
        )[
            "investor_t0_only_history"
        ]
        .sum()
    )

    events_new_to_investor = int(
        t60[
            "new_to_investor_pair"
        ]
        .sum()
    )

    events_repeat_prior_pair = int(
        t60[
            "pair_seen_before_t60"
        ]
        .sum()
    )

    summary = pd.DataFrame(
        [
            {
                "metric": "t60_interactions",
                "value": len(t60),
            },
            {
                "metric": "t60_unique_investors",
                "value": t60_investor_count,
            },
            {
                "metric": "t60_unique_startups",
                "value": t60_startup_count,
            },
            {
                "metric": "t60_unique_pairs",
                "value": t60_pair_count,
            },
            {
                "metric": (
                    "interaction_cold_start_investors"
                ),
                "value": int(
                    unique_cold_investors
                ),
            },
            {
                "metric": (
                    "interaction_cold_start_investor_share_pct"
                ),
                "value": pct(
                    unique_cold_investors,
                    t60_investor_count,
                ),
            },
            {
                "metric": (
                    "interaction_cold_start_startups"
                ),
                "value": int(
                    unique_cold_startups
                ),
            },
            {
                "metric": (
                    "interaction_cold_start_startup_share_pct"
                ),
                "value": pct(
                    unique_cold_startups,
                    t60_startup_count,
                ),
            },
            {
                "metric": (
                    "t60_investors_with_t0_only_history"
                ),
                "value": int(
                    t0_only_t60_investors
                ),
            },
            {
                "metric": (
                    "events_new_to_investor_pair"
                ),
                "value": (
                    events_new_to_investor
                ),
            },
            {
                "metric": (
                    "events_new_to_investor_pair_share_pct"
                ),
                "value": pct(
                    events_new_to_investor,
                    len(t60),
                ),
            },
            {
                "metric": (
                    "events_repeating_prior_pair"
                ),
                "value": (
                    events_repeat_prior_pair
                ),
            },
            {
                "metric": (
                    "events_repeating_prior_pair_share_pct"
                ),
                "value": pct(
                    events_repeat_prior_pair,
                    len(t60),
                ),
            },
            {
                "metric": (
                    "pairs_repeating_within_t60"
                ),
                "value": int(
                    repeated_t60_pair_count
                ),
            },
            {
                "metric": (
                    "interactions_in_repeated_t60_pairs"
                ),
                "value": int(
                    interactions_in_repeated_t60_pairs
                ),
            },
            {
                "metric": (
                    "max_events_for_one_pair_in_t60"
                ),
                "value": int(
                    max_events_one_pair_t60
                ),
            },
        ]
    )

    # -------------------------------------------------------------------------
    # 15. Save audit outputs
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    t60.to_parquet(
        INTERACTION_FLAGS_OUTPUT,
        index=False,
    )

    t60_investor_history.to_csv(
        INVESTOR_HISTORY_OUTPUT,
        index=False,
    )

    investor_bucket_summary.to_csv(
        INVESTOR_BUCKET_OUTPUT,
        index=False,
    )

    t60_startup_history.to_csv(
        STARTUP_HISTORY_OUTPUT,
        index=False,
    )

    pair_summary.to_csv(
        PAIR_OUTPUT,
        index=False,
    )

    # -------------------------------------------------------------------------
    # 16. Print T60 overview
    # -------------------------------------------------------------------------

    separator("-")
    print(
        "T60 EVALUATION-POOL OVERVIEW"
    )
    separator("-")

    print(
        f"T60 interactions:             "
        f"{len(t60):,}"
    )

    print(
        f"T60 investors:                "
        f"{t60_investor_count:,}"
    )

    print(
        f"T60 startups:                 "
        f"{t60_startup_count:,}"
    )

    print(
        f"T60 unique pairs:             "
        f"{t60_pair_count:,}"
    )

    # -------------------------------------------------------------------------
    # 17. Print entity cold-start diagnostics
    # -------------------------------------------------------------------------

    separator("-")
    print(
        "INTERACTION-COLD-START DIAGNOSTICS"
    )
    separator("-")

    print(
        f"Interaction-cold investors:   "
        f"{int(unique_cold_investors):,} "
        f"({pct(unique_cold_investors, t60_investor_count):.2f}%)"
    )

    print(
        f"Interaction-cold startups:    "
        f"{int(unique_cold_startups):,} "
        f"({pct(unique_cold_startups, t60_startup_count):.2f}%)"
    )

    print(
        f"T0-only history investors:    "
        f"{int(t0_only_t60_investors):,}"
    )

    print(
        "\nT60 event status combinations:"
    )

    print(
        cold_start_status.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    # -------------------------------------------------------------------------
    # 18. Print pair novelty
    # -------------------------------------------------------------------------

    separator("-")
    print(
        "INVESTOR-STARTUP PAIR NOVELTY"
    )
    separator("-")

    print(
        pair_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print(
        f"\nT60 events on new-to-investor pairs: "
        f"{events_new_to_investor:,} "
        f"({pct(events_new_to_investor, len(t60)):.2f}%)"
    )

    print(
        f"T60 events repeating prior pairs:    "
        f"{events_repeat_prior_pair:,} "
        f"({pct(events_repeat_prior_pair, len(t60)):.2f}%)"
    )

    # -------------------------------------------------------------------------
    # 19. Print within-T60 repeated-pair diagnostics
    # -------------------------------------------------------------------------

    separator("-")
    print(
        "WITHIN-T60 REPEATED PAIRS"
    )
    separator("-")

    print(
        f"Unique T60 pairs:                  "
        f"{t60_pair_count:,}"
    )

    print(
        f"Pairs with >1 T60 event:           "
        f"{repeated_t60_pair_count:,}"
    )

    print(
        f"Interactions inside repeated pairs:"
        f" {interactions_in_repeated_t60_pairs:,}"
    )

    print(
        f"Max events for one T60 pair:       "
        f"{max_events_one_pair_t60:,}"
    )

    print(
        """
If repeated T60 pairs exist, a purely event-level random validation/test
split could place the same investor-startup pair into both subsets.

This audit does NOT change the paper's protocol. It only measures the
potential consequence before the split is implemented.
"""
    )

    # -------------------------------------------------------------------------
    # 20. Print T60 investor prior-history distribution
    # -------------------------------------------------------------------------

    separator("-")
    print(
        "T60 INVESTOR PRIOR-HISTORY BUCKETS"
    )
    separator("-")

    print(
        investor_bucket_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    separator("-")
    print(
        "T60 INVESTOR HISTORY QUANTILES"
    )
    separator("-")

    investor_quantile_cols = [
        "prior_interactions",
        "prior_unique_startups",
        "prior_active_segments",
        "prior_active_detailed_segments",
    ]

    print(
        t60_investor_history[
            investor_quantile_cols
        ]
        .quantile(
            [
                0.00,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
                1.00,
            ]
        )
        .to_string()
    )

    # -------------------------------------------------------------------------
    # 21. Print startup prior-history distribution
    # -------------------------------------------------------------------------

    separator("-")
    print(
        "T60 STARTUP PRIOR-HISTORY QUANTILES"
    )
    separator("-")

    startup_quantile_cols = [
        "prior_investment_interactions",
        "prior_unique_investors",
        "prior_active_segments",
    ]

    print(
        t60_startup_history[
            startup_quantile_cols
        ]
        .quantile(
            [
                0.00,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
                1.00,
            ]
        )
        .to_string()
    )

    # -------------------------------------------------------------------------
    # 22. Global consistency checks
    # -------------------------------------------------------------------------

    separator("-")
    print(
        "GLOBAL CONSISTENCY CHECKS"
    )
    separator("-")

    print(
        f"Temporal interactions:        "
        f"{len(df):,}"
    )

    print(
        f"History T0-T59:               "
        f"{len(history):,}"
    )

    print(
        f"T60 evaluation pool:          "
        f"{len(t60):,}"
    )

    print(
        f"History + T60:                "
        f"{len(history) + len(t60):,}"
    )

    if (
        len(history)
        + len(t60)
        != EXPECTED_TEMPORAL_ROWS
    ):
        raise ValueError(
            "History + T60 does not reconstruct "
            "the temporal interaction layer."
        )

    if (
        len(t60_unique_pairs)
        != EXPECTED_T60_PAIRS
    ):
        raise ValueError(
            "T60 unique-pair count changed "
            "during audit."
        )

    separator()
    print(
        "PHASE 2.3.2 AUDIT COMPLETE"
    )
    separator()

    print(f"""
Outputs written to:

{SUMMARY_OUTPUT}
{INTERACTION_FLAGS_OUTPUT}
{INVESTOR_HISTORY_OUTPUT}
{INVESTOR_BUCKET_OUTPUT}
{STARTUP_HISTORY_OUTPUT}
{PAIR_OUTPUT}

No T60 interaction has been removed.
No cold-start investor or startup has been removed.
No repeated investor-startup pair has been removed.
No validation/test split has been performed.
No negative candidate has been sampled.
No minimum history threshold has been chosen.
The selected temporal layer remains unchanged.

Next:
Interpret T60 evaluation feasibility before reproducing the paper's
10% validation / 90% test split.
""")


if __name__ == "__main__":
    main()