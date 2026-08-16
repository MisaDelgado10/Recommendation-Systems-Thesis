from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import polars as pl


# =========================================================
# Configuration
# =========================================================

RAW_DIR = Path("data/raw")

INVESTOR_MENTIONS_PATH = Path(
    "data/interim/investor_mentions.parquet"
)

INTERIM_DIR = Path("data/interim")

DUPLICATE_COMPANY_NAMES_OUTPUT = (
    INTERIM_DIR
    / "company_name_id_ambiguities.csv"
)

UNMATCHED_STARTUPS_OUTPUT = (
    INTERIM_DIR
    / "startup_name_unmatched.csv"
)


# =========================================================
# Conservative normalization
# =========================================================

def normalize_name(
    expr: pl.Expr,
) -> pl.Expr:
    """
    Conservative normalization only.

    We currently only remove leading/trailing whitespace.

    We deliberately DO NOT:
    - lowercase
    - remove punctuation
    - remove accents
    - remove corporate suffixes
    - fuzzy-match
    """

    return expr.str.strip_chars()


# =========================================================
# Main
# =========================================================

def main() -> None:

    print("=" * 80)
    print("PHASE 1.13 — STARTUP / COMPANY ID RESOLUTION AUDIT")
    print("=" * 80)

    INTERIM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # 1. Verify investor mention table
    # -----------------------------------------------------

    print(
        "\n[1/8] Checking investor mention table..."
    )

    if not INVESTOR_MENTIONS_PATH.exists():

        raise FileNotFoundError(
            f"File not found:\n"
            f"{INVESTOR_MENTIONS_PATH.resolve()}"
        )

    print("Investor mention table found.")

    # -----------------------------------------------------
    # 2. Discover company CSV shards
    # -----------------------------------------------------

    print(
        "\n[2/8] Discovering company CSV files..."
    )

    company_files = sorted(
        RAW_DIR.glob("companies*.csv")
    )

    if not company_files:

        raise FileNotFoundError(
            "\nNo company CSV files were found.\n"
            "Expected files matching:\n"
            "data/raw/companies*.csv"
        )

    print(
        f"Company files found: "
        f"{len(company_files):,}"
    )

    for path in company_files:

        print(
            f"  {path.name}"
        )

    # -----------------------------------------------------
    # 3. Load startup names from investor mention table
    #
    # We only need the distinct startup names for the audit,
    # not all 1.3M investor mentions.
    # -----------------------------------------------------

    print(
        "\n[3/8] Loading unique startup names "
        "from investor mention table..."
    )

    startup_names = (
        pl.scan_parquet(
            INVESTOR_MENTIONS_PATH
        )
        .select(
            "startup_name_raw"
        )
        .filter(
            pl.col(
                "startup_name_raw"
            ).is_not_null()
        )
        .with_columns(
            normalize_name(
                pl.col(
                    "startup_name_raw"
                )
            ).alias(
                "startup_name_normalized"
            )
        )
        .select(
            "startup_name_normalized"
        )
        .unique()
        .collect()
    )

    print(
        f"Unique startup names appearing "
        f"in funding interactions: "
        f"{startup_names.height:,}"
    )

    # -----------------------------------------------------
    # 4. Read company IDs + names from every shard
    # -----------------------------------------------------

    print(
        "\n[4/8] Reading company master shards..."
    )

    company_frames = []

    for index, path in enumerate(
        company_files,
        start=1,
    ):

        print(
            f"  Reading "
            f"{index}/{len(company_files)}: "
            f"{path.name}"
        )

        frame = (
            pl.scan_csv(
                path,
                schema_overrides={
                    "id": pl.String,
                    "name": pl.String,
                },
                null_values=[
                    "",
                    "null",
                    "NULL",
                    "None",
                ],
                empty_string_is_null=True,
            )
            .select(
                pl.col(
                    "id"
                ).alias(
                    "company_id"
                ),

                pl.col(
                    "name"
                ).alias(
                    "company_name"
                ),
            )
            .filter(
                pl.col(
                    "company_name"
                ).is_not_null()
            )
            .with_columns(
                normalize_name(
                    pl.col(
                        "company_name"
                    )
                ).alias(
                    "company_name_normalized"
                )
            )
            .collect()
        )

        print(
            f"    Rows loaded: "
            f"{frame.height:,}"
        )

        company_frames.append(
            frame
        )

    # -----------------------------------------------------
    # 5. Combine company shards
    # -----------------------------------------------------

    print(
        "\n[5/8] Combining company master..."
    )

    companies = pl.concat(
        company_frames,
        how="vertical",
    )

    print(
        f"Total company rows loaded: "
        f"{companies.height:,}"
    )

    company_summary = (
        companies
        .select(
            pl.col(
                "company_id"
            )
            .n_unique()
            .alias(
                "unique_company_ids"
            ),

            pl.col(
                "company_name_normalized"
            )
            .n_unique()
            .alias(
                "unique_company_names"
            ),

            pl.col(
                "company_id"
            )
            .null_count()
            .alias(
                "missing_company_ids"
            ),
        )
    )

    print(
        "\nCompany master summary:"
    )

    print(
        company_summary
    )

    # -----------------------------------------------------
    # 6. Determine company-name multiplicity
    # -----------------------------------------------------

    print(
        "\n[6/8] Auditing duplicate company names..."
    )

    company_name_resolution = (
        companies
        .group_by(
            "company_name_normalized"
        )
        .agg(
            pl.col(
                "company_id"
            )
            .n_unique()
            .alias(
                "company_id_count"
            ),

            pl.col(
                "company_id"
            )
            .unique()
            .alias(
                "candidate_company_ids"
            ),
        )
    )

    duplicate_company_names = (
        company_name_resolution
        .filter(
            pl.col(
                "company_id_count"
            )
            > 1
        )
        .sort(
            "company_id_count",
            descending=True,
        )
    )

    print(
        f"Company names mapping to "
        f"multiple IDs: "
        f"{duplicate_company_names.height:,}"
    )

    if duplicate_company_names.height:

        print(
            "\nTop duplicated company names:"
        )

        print(
            duplicate_company_names
            .head(30)
        )

        (
            duplicate_company_names
            .with_columns(
                pl.col(
                    "candidate_company_ids"
                )
                .list.join("|")
            )
            .write_csv(
                DUPLICATE_COMPANY_NAMES_OUTPUT
            )
        )

        print(
            "\nSaved company-name ambiguities to:"
        )

        print(
            DUPLICATE_COMPANY_NAMES_OUTPUT
        )

    # -----------------------------------------------------
    # 7. Match funding startup names to company master
    # -----------------------------------------------------

    print(
        "\n[7/8] Matching funding startup names "
        "to company master..."
    )

    startup_resolution = (
        startup_names
        .join(
            company_name_resolution,
            left_on=
                "startup_name_normalized",
            right_on=
                "company_name_normalized",
            how="left",
        )
        .with_columns(

            pl.when(
                pl.col(
                    "company_id_count"
                ).is_null()
            )
            .then(
                pl.lit(
                    "unmatched_master"
                )
            )

            .when(
                pl.col(
                    "company_id_count"
                )
                == 1
            )
            .then(
                pl.lit(
                    "resolved_unique"
                )
            )

            .otherwise(
                pl.lit(
                    "ambiguous_multiple_ids"
                )
            )
            .alias(
                "startup_resolution_status"
            )
        )
    )

    resolution_summary = (
        startup_resolution
        .group_by(
            "startup_resolution_status"
        )
        .len()
        .sort(
            "len",
            descending=True,
        )
    )

    print(
        "\nUNIQUE STARTUP-NAME RESOLUTION:"
    )

    print(
        resolution_summary
    )

    # -----------------------------------------------------
    # 7.1 Save unmatched names
    # -----------------------------------------------------

    unmatched_startups = (
        startup_resolution
        .filter(
            pl.col(
                "startup_resolution_status"
            )
            == "unmatched_master"
        )
        .select(
            "startup_name_normalized"
        )
        .sort(
            "startup_name_normalized"
        )
    )

    print(
        f"\nUnmatched unique startup names: "
        f"{unmatched_startups.height:,}"
    )

    if unmatched_startups.height:

        unmatched_startups.write_csv(
            UNMATCHED_STARTUPS_OUTPUT
        )

        print(
            "Saved unmatched startup names to:"
        )

        print(
            UNMATCHED_STARTUPS_OUTPUT
        )

        print(
            "\nFirst 50 unmatched startup names:"
        )

        print(
            unmatched_startups
            .head(50)
        )

    # -----------------------------------------------------
    # 8. Measure impact at INVESTOR-MENTION level
    #
    # Important:
    # resolving unique startup names is useful, but what we
    # really care about is how many of the 1.3M potential
    # interactions are affected.
    # -----------------------------------------------------

    print(
        "\n[8/8] Measuring interaction-level impact..."
    )

    mention_level = (
        pl.scan_parquet(
            INVESTOR_MENTIONS_PATH
        )
        .with_columns(
            normalize_name(
                pl.col(
                    "startup_name_raw"
                )
            ).alias(
                "startup_name_normalized"
            )
        )
        .join(
            startup_resolution.lazy(),
            on=
                "startup_name_normalized",
            how="left",
        )
        .collect()
    )

    interaction_summary = (
        mention_level
        .group_by(
            "startup_resolution_status"
        )
        .len()
        .sort(
            "len",
            descending=True,
        )
    )

    print(
        "\nINVESTOR-MENTION LEVEL "
        "STARTUP RESOLUTION:"
    )

    print(
        interaction_summary
    )

    # -----------------------------------------------------
    # Also measure cases where BOTH investor and startup
    # can already be uniquely resolved.
    # -----------------------------------------------------

    fully_resolved_mentions = (
        mention_level
        .filter(
            (
                pl.col(
                    "id_resolution_status"
                )
                == "resolved_unique"
            )
            &
            (
                pl.col(
                    "startup_resolution_status"
                )
                == "resolved_unique"
            )
        )
        .height
    )

    print(
        "\nMentions with both investor and "
        "startup uniquely resolvable:"
    )

    print(
        f"{fully_resolved_mentions:,}"
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "PHASE 1.13 AUDIT COMPLETE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()