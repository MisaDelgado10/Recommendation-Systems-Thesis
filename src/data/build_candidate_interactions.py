from __future__ import annotations

from pathlib import Path

import polars as pl


# =========================================================
# Configuration
# =========================================================

INVESTOR_MENTIONS_PATH = Path(
    "data/interim/investor_mentions.parquet"
)

STARTUP_RESOLUTION_PATH = Path(
    "data/interim/startup_resolution_by_round.parquet"
)

INTERIM_DIR = Path(
    "data/interim"
)

CANDIDATE_OUTPUT = (
    INTERIM_DIR
    / "candidate_interactions.parquet"
)

EXCLUDED_OUTPUT = (
    INTERIM_DIR
    / "interaction_exclusion_summary.csv"
)

EXACT_DUPLICATE_OUTPUT = (
    INTERIM_DIR
    / "interaction_exact_duplicate_audit.csv"
)

SAME_DAY_DUPLICATE_OUTPUT = (
    INTERIM_DIR
    / "interaction_same_day_duplicate_audit.csv"
)


# =========================================================
# Main
# =========================================================

def main() -> None:

    print("=" * 80)
    print("PHASE 1.17 — BUILD AND AUDIT CANONICAL INVESTMENT EVENTS")
    print("=" * 80)

    INTERIM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # 1. Verify inputs
    # -----------------------------------------------------

    print("\n[1/10] Checking input files...")

    for path in [
        INVESTOR_MENTIONS_PATH,
        STARTUP_RESOLUTION_PATH,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n"
                f"{path.resolve()}"
            )

    print("Input files found.")

    # -----------------------------------------------------
    # 2. Load investor mentions
    # -----------------------------------------------------

    print(
        "\n[2/10] Loading investor mentions..."
    )

    mentions = pl.read_parquet(
        INVESTOR_MENTIONS_PATH
    )

    print(
        f"Investor mention rows: "
        f"{mentions.height:,}"
    )

    # -----------------------------------------------------
    # 3. Load startup resolution table
    # -----------------------------------------------------

    print(
        "\n[3/10] Loading startup resolution..."
    )

    startups = pl.read_parquet(
        STARTUP_RESOLUTION_PATH
    )

    print(
        f"Startup-resolution rows: "
        f"{startups.height:,}"
    )

    # -----------------------------------------------------
    # 4. Join investor and startup identity
    # -----------------------------------------------------

    print(
        "\n[4/10] Joining investor and startup identities..."
    )

    joined = (
        mentions
        .join(
            startups.select(
                "funding_round_id",
                "startup_id",
                "startup_resolution_method",
            ),
            on="funding_round_id",
            how="left",
            validate="m:1",
        )
    )

    print(
        f"Rows after join: "
        f"{joined.height:,}"
    )

    if joined.height != mentions.height:

        raise RuntimeError(
            "Joining startup resolution changed the "
            "number of investor-mention rows."
        )

    # -----------------------------------------------------
    # 5. Audit exclusion reasons BEFORE filtering
    # -----------------------------------------------------

    print(
        "\n[5/10] Auditing interaction eligibility..."
    )

    eligibility = (
        joined
        .with_columns(

            (
                pl.col(
                    "id_resolution_status"
                )
                == "resolved_unique"
            )
            .alias(
                "investor_is_resolved"
            ),

            pl.col(
                "startup_id"
            )
            .is_not_null()
            .alias(
                "startup_is_resolved"
            ),

            pl.col(
                "announced_on"
            )
            .str.to_date(
                "%Y-%m-%d",
                strict=False,
            )
            .alias(
                "announced_date"
            ),
        )

        .with_columns(

            pl.col(
                "announced_date"
            )
            .is_not_null()
            .alias(
                "date_is_valid"
            )
        )
    )

    # -----------------------------------------------------
    # Build mutually exclusive exclusion categories.
    # -----------------------------------------------------

    eligibility = (
        eligibility
        .with_columns(

            pl.when(
                (
                    pl.col(
                        "investor_is_resolved"
                    )
                    &
                    pl.col(
                        "startup_is_resolved"
                    )
                    &
                    pl.col(
                        "date_is_valid"
                    )
                )
            )
            .then(
                pl.lit(
                    "eligible"
                )
            )

            .when(
                (
                    ~pl.col(
                        "investor_is_resolved"
                    )
                )
                &
                (
                    ~pl.col(
                        "startup_is_resolved"
                    )
                )
            )
            .then(
                pl.lit(
                    "unresolved_investor_and_startup"
                )
            )

            .when(
                ~pl.col(
                    "investor_is_resolved"
                )
            )
            .then(
                pl.lit(
                    "unresolved_investor"
                )
            )

            .when(
                ~pl.col(
                    "startup_is_resolved"
                )
            )
            .then(
                pl.lit(
                    "unresolved_startup"
                )
            )

            .when(
                ~pl.col(
                    "date_is_valid"
                )
            )
            .then(
                pl.lit(
                    "invalid_date"
                )
            )

            .otherwise(
                pl.lit(
                    "unexpected"
                )
            )

            .alias(
                "eligibility_status"
            )
        )
    )

    eligibility_summary = (
        eligibility
        .group_by(
            "eligibility_status"
        )
        .len()
        .sort(
            "len",
            descending=True,
        )
    )

    print(
        "\nINTERACTION ELIGIBILITY:"
    )

    print(
        eligibility_summary
    )

    eligibility_summary.write_csv(
        EXCLUDED_OUTPUT
    )

    # -----------------------------------------------------
    # 6. Build candidate canonical interaction table
    # -----------------------------------------------------

    print(
        "\n[6/10] Building high-confidence "
        "candidate interactions..."
    )

    candidate = (
        eligibility
        .filter(
            pl.col(
                "eligibility_status"
            )
            == "eligible"
        )
        .select(
            "funding_round_id",

            pl.col(
                "investor_id"
            ),

            pl.col(
                "investor_name"
            ),

            pl.col(
                "startup_id"
            ),

            pl.col(
                "startup_name_raw"
            )
            .alias(
                "startup_name"
            ),

            pl.col(
                "announced_date"
            )
            .alias(
                "announced_on"
            ),

            "investment_type",

            "startup_resolution_method",

            "parse_quality_status",

            "canonical_name_looks_malformed",
        )
    )

    print(
        f"Candidate interactions: "
        f"{candidate.height:,}"
    )

    # -----------------------------------------------------
    # 7. Basic integrity checks
    # -----------------------------------------------------

    print(
        "\n[7/10] Running interaction integrity checks..."
    )

    required_non_null = [
        "funding_round_id",
        "investor_id",
        "startup_id",
        "announced_on",
    ]

    for column in required_non_null:

        null_count = (
            candidate
            .select(
                pl.col(
                    column
                )
                .null_count()
            )
            .item()
        )

        print(
            f"Missing {column:<20} "
            f"{null_count:,}"
        )

        if null_count != 0:

            raise RuntimeError(
                f"Required column {column} "
                f"contains null values."
            )

    unique_investors = (
        candidate
        .select(
            pl.col(
                "investor_id"
            )
            .n_unique()
        )
        .item()
    )

    unique_startups = (
        candidate
        .select(
            pl.col(
                "startup_id"
            )
            .n_unique()
        )
        .item()
    )

    unique_rounds = (
        candidate
        .select(
            pl.col(
                "funding_round_id"
            )
            .n_unique()
        )
        .item()
    )

    earliest_date = (
        candidate
        .select(
            pl.col(
                "announced_on"
            )
            .min()
        )
        .item()
    )

    latest_date = (
        candidate
        .select(
            pl.col(
                "announced_on"
            )
            .max()
        )
        .item()
    )

    print(
        f"\nUnique investors: "
        f"{unique_investors:,}"
    )

    print(
        f"Unique startups:  "
        f"{unique_startups:,}"
    )

    print(
        f"Unique funding rounds: "
        f"{unique_rounds:,}"
    )

    print(
        f"Date range: "
        f"{earliest_date} → {latest_date}"
    )

    # -----------------------------------------------------
    # 8. Audit exact duplicates
    #
    # Same investor + startup + funding round.
    #
    # If this occurs, something may have been duplicated
    # during extraction or processing.
    # -----------------------------------------------------

    print(
        "\n[8/10] Auditing exact interaction duplicates..."
    )

    exact_duplicate_groups = (
        candidate
        .group_by(
            [
                "investor_id",
                "startup_id",
                "funding_round_id",
            ]
        )
        .agg(
            pl.len().alias(
                "row_count"
            )
        )
        .filter(
            pl.col(
                "row_count"
            )
            > 1
        )
        .sort(
            "row_count",
            descending=True,
        )
    )

    print(
        f"Exact duplicate groups "
        f"(investor + startup + funding round): "
        f"{exact_duplicate_groups.height:,}"
    )

    if exact_duplicate_groups.height:

        exact_duplicate_groups.write_csv(
            EXACT_DUPLICATE_OUTPUT
        )

        print(
            "Saved exact duplicate audit to:"
        )

        print(
            EXACT_DUPLICATE_OUTPUT
        )

    # -----------------------------------------------------
    # 9. Audit same-day repeated investment events
    #
    # Same investor + startup + date, but potentially
    # different funding-round IDs.
    #
    # We DO NOT remove these yet.
    # -----------------------------------------------------

    print(
        "\n[9/10] Auditing same-day repeated events..."
    )

    same_day_groups = (
        candidate
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
                "funding_round_id"
            )
            .unique()
            .alias(
                "funding_round_ids"
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
        .sort(
            "row_count",
            descending=True,
        )
    )

    print(
        f"Same investor + startup + date groups "
        f"with multiple rows: "
        f"{same_day_groups.height:,}"
    )

    if same_day_groups.height:

        print(
            "\nTop 20 same-day repeated event groups:"
        )

        with pl.Config(
            tbl_rows=20,
            tbl_cols=-1,
            fmt_str_lengths=100,
        ):

            print(
                same_day_groups
                .head(20)
            )

        # CSV cannot store list columns directly,
        # so flatten only for the diagnostic export.
        (
            same_day_groups
            .with_columns(

                pl.col(
                    "funding_round_ids"
                )
                .list.join("|"),

                pl.col(
                    "investment_types"
                )
                .list.join("|"),
            )
            .write_csv(
                SAME_DAY_DUPLICATE_OUTPUT
            )
        )

        print(
            "\nSaved same-day duplicate audit to:"
        )

        print(
            SAME_DAY_DUPLICATE_OUTPUT
        )

    # -----------------------------------------------------
    # 10. Save candidate interaction table
    # -----------------------------------------------------

    print(
        "\n[10/10] Saving candidate interaction dataset..."
    )

    candidate.write_parquet(
        CANDIDATE_OUTPUT,
        compression="zstd",
    )

    print(
        f"\nSaved:\n"
        f"{CANDIDATE_OUTPUT}"
    )

    # =====================================================
    # Final summary
    # =====================================================

    print(
        "\n" + "-" * 80
    )

    print(
        "PHASE 1.17 SUMMARY"
    )

    print(
        "-" * 80
    )

    print(
        f"Candidate interactions:               "
        f"{candidate.height:,}"
    )

    print(
        f"Unique investors:                     "
        f"{unique_investors:,}"
    )

    print(
        f"Unique startups:                      "
        f"{unique_startups:,}"
    )

    print(
        f"Unique funding rounds:                "
        f"{unique_rounds:,}"
    )

    print(
        f"Earliest event:                       "
        f"{earliest_date}"
    )

    print(
        f"Latest event:                         "
        f"{latest_date}"
    )

    print(
        f"Exact duplicate groups:               "
        f"{exact_duplicate_groups.height:,}"
    )

    print(
        f"Same-day repeated event groups:       "
        f"{same_day_groups.height:,}"
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "PHASE 1.17 COMPLETE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()