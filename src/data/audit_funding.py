from pathlib import Path

import polars as pl


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

FUNDING_PATH = Path(
    "data/raw/CRUNCHBASE_funding_20260531_797177.csv"
)

REQUIRED_COLUMNS = [
    "id",
    "announced_on",
    "investors",
    "org_name",
    "investment_type",
]


def main() -> None:
    # -----------------------------------------------------
    # 1. Check that the source file exists
    # -----------------------------------------------------

    if not FUNDING_PATH.exists():
        raise FileNotFoundError(
            f"Funding file not found:\n{FUNDING_PATH.resolve()}"
        )

    print("=" * 70)
    print("PHASE 1 — CRUNCHBASE FUNDING DATA AUDIT")
    print("=" * 70)

    print(f"\nFile: {FUNDING_PATH}")
    print(
        f"File size: "
        f"{FUNDING_PATH.stat().st_size / (1024 ** 2):,.2f} MB"
    )

    # -----------------------------------------------------
    # 2. Lazily read the CSV
    # -----------------------------------------------------

    funding = pl.scan_csv(
        FUNDING_PATH,
        schema_overrides={
            "id": pl.String,
            "announced_on": pl.String,
            "investors": pl.String,
            "org_name": pl.String,
            "investment_type": pl.String,
        },
        null_values=["", "null", "NULL", "None"],
    )

    # -----------------------------------------------------
    # 3. Verify expected columns
    # -----------------------------------------------------

    schema = funding.collect_schema()
    available_columns = schema.names()

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in available_columns
    ]

    print("\nRequired columns:")

    for column in REQUIRED_COLUMNS:
        status = (
            "FOUND"
            if column in available_columns
            else "MISSING"
        )

        print(f"  {column:<20} {status}")

    if missing_columns:
        raise ValueError(
            "The following required columns are missing: "
            f"{missing_columns}"
        )

    # -----------------------------------------------------
    # 4. Parse dates conservatively
    # -----------------------------------------------------

    parsed_date = (
        pl.col("announced_on")
        .str.to_date(
            format="%Y-%m-%d",
            strict=False,
        )
    )

    # -----------------------------------------------------
    # 5. Compute dataset-level statistics
    # -----------------------------------------------------

    summary = (
        funding
        .select(
            pl.len().alias("total_funding_round_rows"),

            pl.col("id")
            .null_count()
            .alias("missing_funding_round_id"),

            pl.col("announced_on")
            .null_count()
            .alias("missing_announced_on"),

            pl.col("investors")
            .null_count()
            .alias("missing_investors"),

            pl.col("org_name")
            .null_count()
            .alias("missing_org_name"),

            pl.col("investment_type")
            .null_count()
            .alias("missing_investment_type"),

            pl.col("org_name")
            .n_unique()
            .alias("unique_startup_names"),

            parsed_date
            .min()
            .alias("earliest_parsed_date"),

            parsed_date
            .max()
            .alias("latest_parsed_date"),

            (
                pl.col("announced_on").is_not_null()
                & parsed_date.is_null()
            )
            .sum()
            .alias("unparseable_dates"),
        )
        .collect()
    )

    print("\nDataset summary:")
    with pl.Config(tbl_cols=-1):
        print(summary)

    # -----------------------------------------------------
    # 6. Examine possible investor separators
    # -----------------------------------------------------

    separator_stats = (
        funding
        .filter(pl.col("investors").is_not_null())
        .select(
            pl.len()
            .alias("rows_with_investor_data"),

            pl.col("investors")
            .str.contains(",")
            .sum()
            .alias("rows_containing_comma"),

            pl.col("investors")
            .str.contains(";")
            .sum()
            .alias("rows_containing_semicolon"),

            pl.col("investors")
            .str.contains(r"\|")
            .sum()
            .alias("rows_containing_pipe"),
        )
        .collect()
    )

    print("\nInvestor-field separator audit:")
    print(separator_stats)

    # -----------------------------------------------------
    # 7. Print actual raw examples
    # -----------------------------------------------------

    samples = (
        funding
        .filter(pl.col("investors").is_not_null())
        .select(
            "id",
            "announced_on",
            "org_name",
            "investors",
        )
        .head(20)
        .collect()
    )

    print("\nFirst 20 rows with investor information:")
    print(samples)

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()