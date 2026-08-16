from __future__ import annotations

from pathlib import Path

import polars as pl


# =========================================================
# Configuration
# =========================================================

RAW_DIR = Path(
    "data/raw"
)

INTERIM_DIR = Path(
    "data/interim"
)

IDENTITY_MASTER_OUTPUT = (
    INTERIM_DIR
    / "company_identity_master_raw.parquet"
)

DUPLICATE_ID_SUMMARY_OUTPUT = (
    INTERIM_DIR
    / "company_duplicate_id_summary.csv"
)

DUPLICATE_ID_CONFLICTS_OUTPUT = (
    INTERIM_DIR
    / "company_duplicate_id_conflicts.csv"
)

WEBSITE_AMBIGUITIES_OUTPUT = (
    INTERIM_DIR
    / "company_website_ambiguities.csv"
)


# =========================================================
# Conservative normalization
# =========================================================

def normalize_name(
    expr: pl.Expr,
) -> pl.Expr:
    """
    Conservative company-name normalization.

    For now we only:
    - strip leading whitespace
    - strip trailing whitespace

    We DO NOT:
    - lowercase
    - remove punctuation
    - remove legal suffixes
    - fuzzy-match names
    """

    return expr.str.strip_chars()


def normalize_website_exact(
    expr: pl.Expr,
) -> pl.Expr:
    """
    Phase 1.14 website treatment.

    We ONLY remove leading/trailing whitespace.

    We deliberately DO NOT yet:
    - remove http:// or https://
    - remove www.
    - lowercase domains
    - remove trailing slashes
    - extract domains
    - remove query parameters

    Therefore this column should be interpreted as an
    "exact website string after whitespace trimming",
    not as a canonical website/domain.
    """

    return expr.str.strip_chars()


# =========================================================
# Main
# =========================================================

def main() -> None:

    print("=" * 80)
    print("PHASE 1.14 — COMPANY IDENTITY MASTER AUDIT")
    print("=" * 80)

    INTERIM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # 1. Discover company files
    # -----------------------------------------------------

    print(
        "\n[1/9] Discovering company CSV shards..."
    )

    company_files = sorted(
        RAW_DIR.glob("companies*.csv")
    )

    if not company_files:

        raise FileNotFoundError(
            "\nNo company CSV files were found.\n"
            "Expected:\n"
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
    # 2. Verify schema of every company file
    #
    # We explicitly check instead of assuming every shard
    # contains the same columns.
    # -----------------------------------------------------

    print(
        "\n[2/9] Verifying company-file schemas..."
    )

    required_columns = {
        "id",
        "name",
        "website",
    }

    for path in company_files:

        schema = (
            pl.scan_csv(
                path,
                infer_schema_length=100,
            )
            .collect_schema()
        )

        available = set(
            schema.names()
        )

        missing = (
            required_columns
            - available
        )

        if missing:

            raise RuntimeError(
                f"\nRequired columns missing from "
                f"{path.name}:\n"
                f"{sorted(missing)}"
            )

        print(
            f"  {path.name}: OK"
        )

    # -----------------------------------------------------
    # 3. Load identity columns from every shard
    # -----------------------------------------------------

    print(
        "\n[3/9] Loading company identity fields..."
    )

    company_frames: list[
        pl.DataFrame
    ] = []

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
                    "website": pl.String,
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

                normalize_name(
                    pl.col(
                        "name"
                    )
                ).alias(
                    "company_name_normalized"
                ),

                pl.col(
                    "website"
                ).alias(
                    "website"
                ),

                normalize_website_exact(
                    pl.col(
                        "website"
                    )
                ).alias(
                    "website_exact"
                ),

                pl.lit(
                    path.name
                ).alias(
                    "source_file"
                ),
            )
            .collect()
        )

        print(
            f"    Rows: "
            f"{frame.height:,}"
        )

        company_frames.append(
            frame
        )

    # -----------------------------------------------------
    # 4. Combine shards
    # -----------------------------------------------------

    print(
        "\n[4/9] Combining company identity data..."
    )

    companies = pl.concat(
        company_frames,
        how="vertical",
    )

    print(
        f"Total identity rows: "
        f"{companies.height:,}"
    )

    # -----------------------------------------------------
    # 5. Basic identity-master summary
    # -----------------------------------------------------

    print(
        "\n[5/9] Auditing identity-master coverage..."
    )

    summary = (
        companies
        .select(
            pl.len().alias(
                "total_rows"
            ),

            pl.col(
                "company_id"
            )
            .n_unique()
            .alias(
                "unique_company_ids"
            ),

            pl.col(
                "company_id"
            )
            .null_count()
            .alias(
                "missing_company_ids"
            ),

            pl.col(
                "company_name"
            )
            .null_count()
            .alias(
                "missing_company_names"
            ),

            pl.col(
                "website_exact"
            )
            .is_not_null()
            .sum()
            .alias(
                "rows_with_website"
            ),

            pl.col(
                "website_exact"
            )
            .drop_nulls()
            .n_unique()
            .alias(
                "unique_exact_websites"
            ),
        )
    )

    print(
        "\nCompany identity summary:"
    )

    print(
        summary
    )

    # -----------------------------------------------------
    # 6. Audit duplicate company IDs
    # -----------------------------------------------------

    print(
        "\n[6/9] Auditing repeated company IDs..."
    )

    id_stats = (
        companies
        .group_by(
            "company_id"
        )
        .agg(
            pl.len().alias(
                "row_count"
            ),

            pl.col(
                "company_name_normalized"
            )
            .drop_nulls()
            .n_unique()
            .alias(
                "unique_name_count"
            ),

            pl.col(
                "website_exact"
            )
            .drop_nulls()
            .n_unique()
            .alias(
                "unique_website_count"
            ),

            pl.col(
                "source_file"
            )
            .n_unique()
            .alias(
                "source_file_count"
            ),
        )
    )

    duplicated_ids = (
        id_stats
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

    duplicate_id_groups = (
        duplicated_ids.height
    )

    duplicate_rows_total = (
        duplicated_ids
        .select(
            pl.col(
                "row_count"
            )
            .sum()
        )
        .item()
        if duplicated_ids.height
        else 0
    )

    extra_duplicate_rows = (
        duplicate_rows_total
        - duplicate_id_groups
    )

    conflicting_names = (
        duplicated_ids
        .filter(
            pl.col(
                "unique_name_count"
            )
            > 1
        )
        .height
    )

    conflicting_websites = (
        duplicated_ids
        .filter(
            pl.col(
                "unique_website_count"
            )
            > 1
        )
        .height
    )

    print(
        f"\nCompany IDs appearing more than once: "
        f"{duplicate_id_groups:,}"
    )

    print(
        f"Extra repeated rows beyond one per ID: "
        f"{extra_duplicate_rows:,}"
    )

    print(
        f"Repeated IDs with conflicting names: "
        f"{conflicting_names:,}"
    )

    print(
        f"Repeated IDs with conflicting websites: "
        f"{conflicting_websites:,}"
    )

    # Save summary of every duplicated ID.
    if duplicated_ids.height:

        duplicated_ids.write_csv(
            DUPLICATE_ID_SUMMARY_OUTPUT
        )

        print(
            "\nSaved repeated-ID summary to:"
        )

        print(
            DUPLICATE_ID_SUMMARY_OUTPUT
        )

    # -----------------------------------------------------
    # 6.1 Save actual rows for conflicting IDs
    # -----------------------------------------------------

    conflicting_id_values = (
        duplicated_ids
        .filter(
            (
                pl.col(
                    "unique_name_count"
                )
                > 1
            )
            |
            (
                pl.col(
                    "unique_website_count"
                )
                > 1
            )
        )
        .get_column(
            "company_id"
        )
        .to_list()
    )

    if conflicting_id_values:

        conflict_rows = (
            companies
            .filter(
                pl.col(
                    "company_id"
                )
                .is_in(
                    conflicting_id_values
                )
            )
            .sort(
                [
                    "company_id",
                    "source_file",
                ]
            )
        )

        conflict_rows.write_csv(
            DUPLICATE_ID_CONFLICTS_OUTPUT
        )

        print(
            "\nSaved conflicting duplicated-ID "
            "records to:"
        )

        print(
            DUPLICATE_ID_CONFLICTS_OUTPUT
        )

    # -----------------------------------------------------
    # 7. Audit exact website uniqueness
    # -----------------------------------------------------

    print(
        "\n[7/9] Auditing exact website strings..."
    )

    website_stats = (
        companies
        .filter(
            pl.col(
                "website_exact"
            )
            .is_not_null()
        )
        .group_by(
            "website_exact"
        )
        .agg(
            pl.col(
                "company_id"
            )
            .n_unique()
            .alias(
                "company_id_count"
            ),

            pl.len().alias(
                "row_count"
            ),
        )
    )

    unique_website_groups = (
        website_stats
        .filter(
            pl.col(
                "company_id_count"
            )
            == 1
        )
        .height
    )

    ambiguous_website_groups = (
        website_stats
        .filter(
            pl.col(
                "company_id_count"
            )
            > 1
        )
        .height
    )

    print(
        f"Exact websites mapping to one company ID: "
        f"{unique_website_groups:,}"
    )

    print(
        f"Exact websites mapping to multiple IDs: "
        f"{ambiguous_website_groups:,}"
    )

    website_ambiguities = (
        website_stats
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

    if website_ambiguities.height:

        print(
            "\nTop exact websites shared by "
            "multiple company IDs:"
        )

        print(
            website_ambiguities
            .head(30)
        )

        website_ambiguities.write_csv(
            WEBSITE_AMBIGUITIES_OUTPUT
        )

        print(
            "\nSaved website ambiguities to:"
        )

        print(
            WEBSITE_AMBIGUITIES_OUTPUT
        )

    # -----------------------------------------------------
    # 8. Save compact RAW identity master
    #
    # We do NOT deduplicate records here.
    # -----------------------------------------------------

    print(
        "\n[8/9] Saving raw company identity master..."
    )

    companies.write_parquet(
        IDENTITY_MASTER_OUTPUT,
        compression="zstd",
    )

    print(
        f"Saved:\n"
        f"{IDENTITY_MASTER_OUTPUT}"
    )

    # -----------------------------------------------------
    # 9. Final summary
    # -----------------------------------------------------

    print(
        "\n[9/9] Audit complete."
    )

    print(
        "\n" + "-" * 80
    )

    print(
        "PHASE 1.14 SUMMARY"
    )

    print(
        "-" * 80
    )

    print(
        f"Identity rows:                         "
        f"{companies.height:,}"
    )

    print(
        f"Unique company IDs:                    "
        f"{companies.select(pl.col('company_id').n_unique()).item():,}"
    )

    print(
        f"IDs appearing more than once:          "
        f"{duplicate_id_groups:,}"
    )

    print(
        f"Extra repeated rows:                   "
        f"{extra_duplicate_rows:,}"
    )

    print(
        f"Repeated IDs with conflicting names:   "
        f"{conflicting_names:,}"
    )

    print(
        f"Repeated IDs with conflicting websites:"
        f" {conflicting_websites:,}"
    )

    print(
        f"Unique exact website groups:           "
        f"{unique_website_groups:,}"
    )

    print(
        f"Ambiguous exact website groups:        "
        f"{ambiguous_website_groups:,}"
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "PHASE 1.14 COMPLETE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()