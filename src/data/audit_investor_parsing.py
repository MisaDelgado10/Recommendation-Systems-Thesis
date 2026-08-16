from pathlib import Path

import polars as pl


# =========================================================
# Configuration
# =========================================================

FUNDING_PATH = Path(
    "data/raw/CRUNCHBASE_funding_20260531_797177.csv"
)

INVESTOR_PATH = Path(
    "data/raw/CRUNCHBASE_investor_20260531_333726.csv"
)


def normalize_name(expr: pl.Expr) -> pl.Expr:
    """
    Conservative normalization only.

    We deliberately do NOT:
    - lowercase names
    - remove punctuation
    - change accents
    - perform fuzzy matching

    At this stage we only remove surrounding whitespace.
    """

    return expr.str.strip_chars()


def main() -> None:

    print("=" * 80)
    print("PHASE 1 — INVESTOR PARSING AUDIT")
    print("=" * 80)

    # -----------------------------------------------------
    # 1. Verify source files
    # -----------------------------------------------------

    for path in [FUNDING_PATH, INVESTOR_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path.resolve()}"
            )

    # -----------------------------------------------------
    # 2. Read only the columns we need
    # -----------------------------------------------------

    funding = (
        pl.scan_csv(
            FUNDING_PATH,
            schema_overrides={
                "id": pl.String,
                "investors": pl.String,
            },
            null_values=["", "null", "NULL", "None"],
        )
        .select(
            "id",
            "investors",
        )
    )

    investors = (
        pl.scan_csv(
            INVESTOR_PATH,
            schema_overrides={
                "id": pl.String,
                "name": pl.String,
            },
            null_values=["", "null", "NULL", "None"],
        )
        .select(
            pl.col("id").alias("investor_id"),
            pl.col("name").alias("investor_name"),
        )
        .with_columns(
            normalize_name(
                pl.col("investor_name")
            ).alias("investor_name_normalized")
        )
    )

    # -----------------------------------------------------
    # 3. Basic investor-master audit
    # -----------------------------------------------------

    investor_summary = (
        investors
        .select(
            pl.len().alias("investor_rows"),

            pl.col("investor_id")
            .null_count()
            .alias("missing_id"),

            pl.col("investor_name")
            .null_count()
            .alias("missing_name"),

            pl.col("investor_id")
            .n_unique()
            .alias("unique_ids"),

            pl.col("investor_name_normalized")
            .n_unique()
            .alias("unique_normalized_names"),
        )
        .collect()
    )

    print("\nInvestor master summary:")
    print(investor_summary)

    # -----------------------------------------------------
    # 4. Do canonical investor names contain delimiters?
    # -----------------------------------------------------

    delimiter_summary = (
        investors
        .filter(
            pl.col("investor_name_normalized").is_not_null()
        )
        .select(
            pl.col("investor_name_normalized")
            .str.contains(",")
            .sum()
            .alias("canonical_names_with_comma"),

            pl.col("investor_name_normalized")
            .str.contains(";")
            .sum()
            .alias("canonical_names_with_semicolon"),

            pl.col("investor_name_normalized")
            .str.contains(r"\|")
            .sum()
            .alias("canonical_names_with_pipe"),
        )
        .collect()
    )

    print("\nDelimiter characters inside canonical investor names:")
    print(delimiter_summary)

    # -----------------------------------------------------
    # 5. Show canonical names containing delimiters
    # -----------------------------------------------------

    suspicious_canonical_names = (
        investors
        .filter(
            pl.col("investor_name_normalized")
            .str.contains(r"[,;|]")
        )
        .select(
            "investor_id",
            "investor_name",
        )
        .head(50)
        .collect()
    )

    print("\nExamples of canonical investor names containing delimiters:")
    print(suspicious_canonical_names)

    # -----------------------------------------------------
    # 6. Find duplicate canonical names
    # -----------------------------------------------------

    duplicate_names = (
        investors
        .filter(
            pl.col("investor_name_normalized").is_not_null()
        )
        .group_by("investor_name_normalized")
        .agg(
            pl.len().alias("entity_count")
        )
        .filter(
            pl.col("entity_count") > 1
        )
        .sort(
            "entity_count",
            descending=True,
        )
    )

    duplicate_summary = (
        duplicate_names
        .select(
            pl.len().alias("duplicate_name_groups"),
            pl.col("entity_count")
            .sum()
            .alias("investor_entities_in_duplicate_groups"),
        )
        .collect()
    )

    print("\nDuplicate canonical investor-name summary:")
    print(duplicate_summary)

    print("\nTop duplicate investor names:")
    print(
        duplicate_names
        .head(30)
        .collect()
    )

    # -----------------------------------------------------
    # 7. Inspect special funding strings
    # -----------------------------------------------------

    special_funding_strings = (
        funding
        .filter(
            pl.col("investors").is_not_null()
            & (
                pl.col("investors").str.contains(";")
                | pl.col("investors").str.contains(r"\|")
            )
        )
        .select(
            "id",
            "investors",
        )
        .collect()
    )

    print("\nFunding rows containing semicolon or pipe:")
    print(special_funding_strings)

    # -----------------------------------------------------
    # 8. Candidate parsing: comma split
    # -----------------------------------------------------

    parsed_tokens = (
        funding
        .filter(
            pl.col("investors").is_not_null()
        )
        .with_columns(
            pl.col("investors")
            .str.split(",")
            .alias("candidate_investor_names")
        )
        .explode("candidate_investor_names")
        .with_columns(
            normalize_name(
                pl.col("candidate_investor_names")
            ).alias("candidate_name_normalized")
        )
        .filter(
            pl.col("candidate_name_normalized")
            .is_not_null()
            & (
                pl.col("candidate_name_normalized")
                .str.len_chars()
                > 0
            )
        )
        .select(
            pl.col("id").alias("funding_round_id"),
            "investors",
            "candidate_name_normalized",
        )
    )

    # -----------------------------------------------------
    # 9. Exact match candidate tokens against investor master
    # -----------------------------------------------------

    canonical_names = (
        investors
        .select(
            "investor_name_normalized"
        )
        .unique()
    )

    matched = (
        parsed_tokens
        .join(
            canonical_names,
            left_on="candidate_name_normalized",
            right_on="investor_name_normalized",
            how="left",
        )
        .with_columns(
            pl.col("candidate_name_normalized")
            .is_not_null()
            .alias("has_candidate")
        )
    )

    # Because the right-side join key itself is consumed by
    # the join, use membership against the canonical name set
    # to calculate exact-match status directly.
    canonical_name_list = (
        canonical_names
        .collect()
        .get_column("investor_name_normalized")
        .to_list()
    )

    parsed_tokens_checked = (
        parsed_tokens
        .with_columns(
            pl.col("candidate_name_normalized")
            .is_in(canonical_name_list)
            .alias("exact_match")
        )
    )

    match_summary = (
        parsed_tokens_checked
        .select(
            pl.len().alias("candidate_tokens"),

            pl.col("exact_match")
            .sum()
            .alias("exact_matches"),

            (~pl.col("exact_match"))
            .sum()
            .alias("unmatched_tokens"),

            pl.col("candidate_name_normalized")
            .n_unique()
            .alias("unique_candidate_names"),
        )
        .collect()
    )

    print("\nComma-split exact-match summary:")
    print(match_summary)

    # -----------------------------------------------------
    # 10. Most frequent unmatched tokens
    # -----------------------------------------------------

    unmatched = (
        parsed_tokens_checked
        .filter(
            ~pl.col("exact_match")
        )
        .group_by(
            "candidate_name_normalized"
        )
        .agg(
            pl.len().alias("occurrences")
        )
        .sort(
            "occurrences",
            descending=True,
        )
    )

    print("\nMost frequent unmatched candidate names:")
    print(
        unmatched
        .head(50)
        .collect()
    )

    # -----------------------------------------------------
    # 11. Show raw funding examples associated with failures
    # -----------------------------------------------------

    unmatched_examples = (
        parsed_tokens_checked
        .filter(
            ~pl.col("exact_match")
        )
        .select(
            "funding_round_id",
            "investors",
            "candidate_name_normalized",
        )
        .head(50)
        .collect()
    )

    print("\nExamples of unmatched parsing results:")
    print(unmatched_examples)

    print("\n" + "=" * 80)
    print("INVESTOR PARSING AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()