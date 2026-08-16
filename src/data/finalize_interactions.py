from __future__ import annotations

from pathlib import Path

import polars as pl


# =========================================================
# Configuration
# =========================================================

CANDIDATE_PATH = Path(
    "data/interim/candidate_interactions.parquet"
)

PROCESSED_DIR = Path(
    "data/processed"
)

OUTPUT_PATH = (
    PROCESSED_DIR
    / "interactions.parquet"
)

SUMMARY_OUTPUT = (
    PROCESSED_DIR
    / "interactions_summary.csv"
)

SAME_DAY_SUMMARY_OUTPUT = (
    PROCESSED_DIR
    / "same_day_interaction_summary.csv"
)


# =========================================================
# Main
# =========================================================

def main() -> None:

    print("=" * 80)
    print("PHASE 1.18 — FINALIZE CANONICAL INTERACTION DATASET")
    print("=" * 80)

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # 1. Verify candidate dataset
    # -----------------------------------------------------

    print("\n[1/9] Checking candidate interaction dataset...")

    if not CANDIDATE_PATH.exists():

        raise FileNotFoundError(
            f"Required file not found:\n"
            f"{CANDIDATE_PATH.resolve()}"
        )

    print("Candidate interaction dataset found.")

    # -----------------------------------------------------
    # 2. Load candidate interactions
    # -----------------------------------------------------

    print("\n[2/9] Loading candidate interactions...")

    interactions = pl.read_parquet(
        CANDIDATE_PATH
    )

    print(
        f"Rows loaded: "
        f"{interactions.height:,}"
    )

    # -----------------------------------------------------
    # 3. Verify required schema
    # -----------------------------------------------------

    print("\n[3/9] Verifying required columns...")

    required_columns = {
        "funding_round_id",
        "investor_id",
        "investor_name",
        "startup_id",
        "startup_name",
        "announced_on",
        "investment_type",
        "startup_resolution_method",
        "parse_quality_status",
        "canonical_name_looks_malformed",
    }

    available_columns = set(
        interactions.columns
    )

    missing_columns = (
        required_columns
        - available_columns
    )

    if missing_columns:

        raise RuntimeError(
            "Required columns missing:\n"
            f"{sorted(missing_columns)}"
        )

    print("Required columns found.")

    # -----------------------------------------------------
    # 4. Add canonical interaction identifier
    #
    # One row represents:
    #
    # investor × Crunchbase funding round
    #
    # Since one funding round belongs to one startup,
    # this should uniquely identify our raw interaction.
    # -----------------------------------------------------

    print(
        "\n[4/9] Creating canonical interaction IDs..."
    )

    interactions = (
        interactions
        .with_columns(

            pl.concat_str(
                [
                    pl.col(
                        "funding_round_id"
                    ),

                    pl.col(
                        "investor_id"
                    ),
                ],
                separator="::",
            )
            .alias(
                "interaction_id"
            )
        )
    )

    duplicate_interaction_ids = (
        interactions
        .group_by(
            "interaction_id"
        )
        .len()
        .filter(
            pl.col("len") > 1
        )
        .height
    )

    print(
        f"Duplicate canonical interaction IDs: "
        f"{duplicate_interaction_ids:,}"
    )

    if duplicate_interaction_ids != 0:

        raise RuntimeError(
            "Canonical interaction_id is not unique."
        )

    # -----------------------------------------------------
    # 5. Validate required fields
    # -----------------------------------------------------

    print(
        "\n[5/9] Running final integrity checks..."
    )

    required_non_null = [
        "interaction_id",
        "funding_round_id",
        "investor_id",
        "startup_id",
        "announced_on",
    ]

    for column in required_non_null:

        null_count = (
            interactions
            .select(
                pl.col(
                    column
                ).null_count()
            )
            .item()
        )

        print(
            f"Missing {column:<20} "
            f"{null_count:,}"
        )

        if null_count != 0:

            raise RuntimeError(
                f"{column} contains null values."
            )

    # -----------------------------------------------------
    # 6. Characterize same-day observations
    #
    # Important:
    # we AUDIT these, but DO NOT remove them.
    # -----------------------------------------------------

    print(
        "\n[6/9] Characterizing same-day repeated observations..."
    )

    same_day = (
        interactions
        .group_by(
            [
                "investor_id",
                "startup_id",
                "announced_on",
            ]
        )
        .agg(

            pl.len().alias(
                "row_count"
            ),

            pl.col(
                "funding_round_id"
            )
            .n_unique()
            .alias(
                "unique_funding_rounds"
            ),

            pl.col(
                "investment_type"
            )
            .n_unique()
            .alias(
                "unique_investment_types"
            ),

            pl.col(
                "investment_type"
            )
            .unique()
            .alias(
                "investment_types"
            ),
        )
        .filter(
            pl.col(
                "row_count"
            )
            > 1
        )
    )

    same_day_groups = (
        same_day.height
    )

    same_day_rows = (
        same_day
        .select(
            pl.col(
                "row_count"
            )
            .sum()
        )
        .item()
        if same_day_groups
        else 0
    )

    same_day_extra_rows = (
        same_day_rows
        - same_day_groups
    )

    same_day_multiple_types = (
        same_day
        .filter(
            pl.col(
                "unique_investment_types"
            )
            > 1
        )
        .height
    )

    same_day_single_type = (
        same_day
        .filter(
            pl.col(
                "unique_investment_types"
            )
            == 1
        )
        .height
    )

    # Verify that these really are different source rounds.
    same_day_nonunique_round_ids = (
        same_day
        .filter(
            pl.col(
                "row_count"
            )
            !=
            pl.col(
                "unique_funding_rounds"
            )
        )
        .height
    )

    print(
        f"Same-day repeated groups:          "
        f"{same_day_groups:,}"
    )

    print(
        f"Rows belonging to those groups:   "
        f"{same_day_rows:,}"
    )

    print(
        f"Extra rows beyond one per group:  "
        f"{same_day_extra_rows:,}"
    )

    print(
        f"Groups with multiple round types: "
        f"{same_day_multiple_types:,}"
    )

    print(
        f"Groups with one round type only:  "
        f"{same_day_single_type:,}"
    )

    print(
        f"Groups containing repeated "
        f"funding-round IDs:                "
        f"{same_day_nonunique_round_ids:,}"
    )

    if same_day_nonunique_round_ids != 0:

        raise RuntimeError(
            "A same-day group contains repeated "
            "funding_round IDs. Investigate before "
            "finalizing."
        )

    # Human-readable diagnostic.
    if same_day_groups:

        (
            same_day
            .with_columns(

                pl.col(
                    "investment_types"
                )
                .list.join("|")
            )
            .write_csv(
                SAME_DAY_SUMMARY_OUTPUT
            )
        )

    # -----------------------------------------------------
    # 7. Calculate final dataset statistics
    # -----------------------------------------------------

    print(
        "\n[7/9] Calculating final dataset statistics..."
    )

    total_interactions = (
        interactions.height
    )

    unique_investors = (
        interactions
        .select(
            pl.col(
                "investor_id"
            ).n_unique()
        )
        .item()
    )

    unique_startups = (
        interactions
        .select(
            pl.col(
                "startup_id"
            ).n_unique()
        )
        .item()
    )

    unique_rounds = (
        interactions
        .select(
            pl.col(
                "funding_round_id"
            ).n_unique()
        )
        .item()
    )

    unique_pairs = (
        interactions
        .select(
            [
                "investor_id",
                "startup_id",
            ]
        )
        .unique()
        .height
    )

    earliest_date = (
        interactions
        .select(
            pl.col(
                "announced_on"
            ).min()
        )
        .item()
    )

    latest_date = (
        interactions
        .select(
            pl.col(
                "announced_on"
            ).max()
        )
        .item()
    )

    # Number of investors with repeated investments
    # into the same startup across the dataset.
    pair_frequency = (
        interactions
        .group_by(
            [
                "investor_id",
                "startup_id",
            ]
        )
        .agg(
            pl.len().alias(
                "event_count"
            )
        )
    )

    repeated_pairs = (
        pair_frequency
        .filter(
            pl.col(
                "event_count"
            )
            > 1
        )
        .height
    )

    max_pair_events = (
        pair_frequency
        .select(
            pl.col(
                "event_count"
            ).max()
        )
        .item()
    )

    # -----------------------------------------------------
    # 8. Save final canonical interaction table
    # -----------------------------------------------------

    print(
        "\n[8/9] Saving final canonical interactions..."
    )

    # Put identifiers first for readability.
    final_interactions = (
        interactions
        .select(
            "interaction_id",
            "funding_round_id",

            "investor_id",
            "investor_name",

            "startup_id",
            "startup_name",

            "announced_on",
            "investment_type",

            "startup_resolution_method",
            "parse_quality_status",
            "canonical_name_looks_malformed",
        )
        .sort(
            [
                "announced_on",
                "funding_round_id",
                "investor_id",
            ]
        )
    )

    final_interactions.write_parquet(
        OUTPUT_PATH,
        compression="zstd",
    )

    # -----------------------------------------------------
    # 9. Save summary
    # -----------------------------------------------------

    print(
        "\n[9/9] Saving dataset summary..."
    )

    summary = pl.DataFrame(
        {
            "metric": [
                "total_interactions",
                "unique_investors",
                "unique_startups",
                "unique_funding_rounds",
                "unique_investor_startup_pairs",
                "repeated_investor_startup_pairs",
                "max_events_for_one_pair",
                "same_day_repeated_groups",
                "same_day_rows",
                "same_day_extra_rows",
                "same_day_multiple_types",
                "same_day_single_type",
            ],

            "value": [
                total_interactions,
                unique_investors,
                unique_startups,
                unique_rounds,
                unique_pairs,
                repeated_pairs,
                max_pair_events,
                same_day_groups,
                same_day_rows,
                same_day_extra_rows,
                same_day_multiple_types,
                same_day_single_type,
            ],
        }
    )

    summary.write_csv(
        SUMMARY_OUTPUT
    )

    # =====================================================
    # Final output
    # =====================================================

    print(
        "\n" + "-" * 80
    )

    print(
        "PHASE 1.18 SUMMARY"
    )

    print(
        "-" * 80
    )

    print(
        f"Canonical interactions:             "
        f"{total_interactions:,}"
    )

    print(
        f"Unique investors:                   "
        f"{unique_investors:,}"
    )

    print(
        f"Unique startups:                    "
        f"{unique_startups:,}"
    )

    print(
        f"Unique funding rounds:              "
        f"{unique_rounds:,}"
    )

    print(
        f"Unique investor-startup pairs:      "
        f"{unique_pairs:,}"
    )

    print(
        f"Pairs with >1 investment event:     "
        f"{repeated_pairs:,}"
    )

    print(
        f"Maximum events for one pair:        "
        f"{max_pair_events:,}"
    )

    print(
        f"Same-day repeated groups:           "
        f"{same_day_groups:,}"
    )

    print(
        f"Same-day groups / multiple types:   "
        f"{same_day_multiple_types:,}"
    )

    print(
        f"Same-day groups / one type:         "
        f"{same_day_single_type:,}"
    )

    print(
        f"Date range:                         "
        f"{earliest_date} → {latest_date}"
    )

    print(
        f"\nSaved final interactions:\n"
        f"{OUTPUT_PATH}"
    )

    print(
        f"\nSaved summary:\n"
        f"{SUMMARY_OUTPUT}"
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "PHASE 1 COMPLETE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()