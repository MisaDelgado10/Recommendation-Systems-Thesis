from __future__ import annotations

from pathlib import Path

import polars as pl


# =========================================================
# Configuration
# =========================================================

FUNDING_PATH = Path(
    "data/raw/CRUNCHBASE_funding_20260531_797177.csv"
)

INVESTOR_MENTIONS_PATH = Path(
    "data/interim/investor_mentions.parquet"
)

COMPANY_IDENTITY_PATH = Path(
    "data/interim/company_identity_master_raw.parquet"
)

COMPANY_CONFLICTS_PATH = Path(
    "data/interim/company_duplicate_id_conflicts.csv"
)

INTERIM_DIR = Path(
    "data/interim"
)

ROUND_AUDIT_OUTPUT = (
    INTERIM_DIR
    / "startup_website_resolution_audit.parquet"
)

NAME_WEBSITE_CONFLICT_OUTPUT = (
    INTERIM_DIR
    / "startup_name_website_conflicts.csv"
)

AMBIGUOUS_NAME_RECOVERY_OUTPUT = (
    INTERIM_DIR
    / "startup_ambiguous_name_website_candidates.csv"
)

UNMATCHED_NAME_RECOVERY_OUTPUT = (
    INTERIM_DIR
    / "startup_unmatched_name_website_candidates.csv"
)


# =========================================================
# Conservative normalization
# =========================================================

def normalize_name(
    expr: pl.Expr,
) -> pl.Expr:
    """
    Same conservative company-name normalization used
    previously: strip surrounding whitespace only.
    """

    return expr.str.strip_chars()


def normalize_website_exact(
    expr: pl.Expr,
) -> pl.Expr:
    """
    Exact website evidence.

    We only strip surrounding whitespace.

    We DO NOT:
    - lowercase
    - remove http/https
    - remove www
    - remove trailing slash
    - extract domains
    - alter query strings

    This keeps Phase 1.15 strictly based on literal
    Crunchbase website values.
    """

    return expr.str.strip_chars()


# =========================================================
# Main
# =========================================================

def main() -> None:

    print("=" * 80)
    print("PHASE 1.15 — EXACT WEBSITE STARTUP RESOLUTION AUDIT")
    print("=" * 80)

    INTERIM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # 1. Verify required files
    # -----------------------------------------------------

    print("\n[1/11] Checking required files...")

    required_files = [
        FUNDING_PATH,
        INVESTOR_MENTIONS_PATH,
        COMPANY_IDENTITY_PATH,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n"
                f"{path.resolve()}"
            )

    print("Required files found.")

    # -----------------------------------------------------
    # 2. Inspect the one known duplicate-company-ID conflict
    # -----------------------------------------------------

    print(
        "\n[2/11] Inspecting duplicate company-ID conflicts..."
    )

    if COMPANY_CONFLICTS_PATH.exists():

        conflict_df = pl.read_csv(
            COMPANY_CONFLICTS_PATH
        )

        print(
            f"Conflict rows found: "
            f"{conflict_df.height:,}"
        )

        print("\nComplete conflict data:")

        with pl.Config(
            tbl_rows=-1,
            tbl_cols=-1,
            fmt_str_lengths=120,
        ):
            print(conflict_df)

    else:

        print(
            "Conflict file not found. "
            "Skipping this diagnostic."
        )

    # -----------------------------------------------------
    # 3. Build one startup record per funding round
    #
    # investor_mentions contains one row per investor.
    # We need one row per funding round for company identity.
    # -----------------------------------------------------

    print(
        "\n[3/11] Building funding-round startup universe..."
    )

    rounds = (
        pl.scan_parquet(
            INVESTOR_MENTIONS_PATH
        )
        .select(
            "funding_round_id",
            "startup_name_raw",
        )
        .unique()
        .collect()
    )

    print(
        f"Unique funding-round/startup records: "
        f"{rounds.height:,}"
    )

    # -----------------------------------------------------
    # 4. Read organization website from funding data
    # -----------------------------------------------------

    print(
        "\n[4/11] Reading funding-round organization websites..."
    )

    funding_identity = (
        pl.scan_csv(
            FUNDING_PATH,
            schema_overrides={
                "id": pl.String,
                "org_name": pl.String,
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
                "funding_round_id"
            ),

            pl.col(
                "org_name"
            ).alias(
                "funding_org_name"
            ),

            pl.col(
                "website"
            ).alias(
                "funding_website"
            ),
        )
        .collect()
    )

    round_identity = (
        rounds
        .join(
            funding_identity,
            on="funding_round_id",
            how="left",
        )
        .with_columns(
            normalize_name(
                pl.col(
                    "startup_name_raw"
                )
            ).alias(
                "startup_name_normalized"
            ),

            normalize_website_exact(
                pl.col(
                    "funding_website"
                )
            ).alias(
                "website_exact"
            ),
        )
    )

    # -----------------------------------------------------
    # 4.1 Validate that startup_name_raw came from org_name
    # -----------------------------------------------------

    source_name_mismatches = (
        round_identity
        .filter(
            (
                pl.col(
                    "startup_name_raw"
                ).is_not_null()
            )
            &
            (
                pl.col(
                    "funding_org_name"
                ).is_not_null()
            )
            &
            (
                pl.col(
                    "startup_name_raw"
                )
                !=
                pl.col(
                    "funding_org_name"
                )
            )
        )
        .height
    )

    missing_funding_joins = (
        round_identity
        .filter(
            pl.col(
                "funding_org_name"
            ).is_null()
            &
            pl.col(
                "funding_website"
            ).is_null()
        )
        .height
    )

    print(
        f"Startup-name / funding-org mismatches: "
        f"{source_name_mismatches:,}"
    )

    print(
        f"Funding rounds not joined back to source: "
        f"{missing_funding_joins:,}"
    )

    # -----------------------------------------------------
    # 5. Load compact company identity master
    # -----------------------------------------------------

    print(
        "\n[5/11] Loading company identity master..."
    )

    companies = (
        pl.scan_parquet(
            COMPANY_IDENTITY_PATH
        )
    )

    # -----------------------------------------------------
    # 6. Build exact company-name resolution dictionary
    #
    # n_unique(company_id) is important because Phase 1.14
    # found repeated rows for some IDs.
    # -----------------------------------------------------

    print(
        "\n[6/11] Building exact name and website mappings..."
    )

    name_resolution = (
        companies
        .filter(
            pl.col(
                "company_name_normalized"
            ).is_not_null()
        )
        .group_by(
            "company_name_normalized"
        )
        .agg(
            pl.col(
                "company_id"
            )
            .n_unique()
            .alias(
                "name_company_id_count"
            ),

            pl.col(
                "company_id"
            )
            .unique()
            .alias(
                "name_candidate_company_ids"
            ),
        )
        .collect()
    )

    website_resolution = (
        companies
        .filter(
            pl.col(
                "website_exact"
            ).is_not_null()
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
                "website_company_id_count"
            ),

            pl.col(
                "company_id"
            )
            .unique()
            .alias(
                "website_candidate_company_ids"
            ),
        )
        .collect()
    )

    # -----------------------------------------------------
    # 7. Attach name + website evidence to funding rounds
    # -----------------------------------------------------

    print(
        "\n[7/11] Matching funding startups against "
        "name and website evidence..."
    )

    audit = (
        round_identity

        .join(
            name_resolution,
            left_on=
                "startup_name_normalized",
            right_on=
                "company_name_normalized",
            how="left",
        )

        .join(
            website_resolution,
            on=
                "website_exact",
            how="left",
        )

        .with_columns(

            # ---------------------------------------------
            # Exact-name resolution status
            # ---------------------------------------------

            pl.when(
                pl.col(
                    "startup_name_normalized"
                ).is_null()
            )
            .then(
                pl.lit(
                    "missing_name"
                )
            )

            .when(
                pl.col(
                    "name_company_id_count"
                ).is_null()
            )
            .then(
                pl.lit(
                    "unmatched_master"
                )
            )

            .when(
                pl.col(
                    "name_company_id_count"
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
                "name_resolution_status"
            ),

            # ---------------------------------------------
            # Exact-website resolution status
            # ---------------------------------------------

            pl.when(
                pl.col(
                    "website_exact"
                ).is_null()
            )
            .then(
                pl.lit(
                    "missing_website"
                )
            )

            .when(
                pl.col(
                    "website_company_id_count"
                ).is_null()
            )
            .then(
                pl.lit(
                    "unmatched_website"
                )
            )

            .when(
                pl.col(
                    "website_company_id_count"
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
                "website_resolution_status"
            ),
        )

        # ---------------------------------------------
        # Extract ID when mapping is unique
        # ---------------------------------------------

        .with_columns(

            pl.when(
                pl.col(
                    "name_company_id_count"
                )
                == 1
            )
            .then(
                pl.col(
                    "name_candidate_company_ids"
                )
                .list.first()
            )
            .otherwise(
                None
            )
            .alias(
                "name_resolved_company_id"
            ),

            pl.when(
                pl.col(
                    "website_company_id_count"
                )
                == 1
            )
            .then(
                pl.col(
                    "website_candidate_company_ids"
                )
                .list.first()
            )
            .otherwise(
                None
            )
            .alias(
                "website_resolved_company_id"
            ),
        )
    )

    # -----------------------------------------------------
    # 8. Validate website using already-resolved names
    # -----------------------------------------------------

    print(
        "\n[8/11] Validating website evidence against "
        "unique name matches..."
    )

    validation_population = (
        audit
        .filter(
            (
                pl.col(
                    "name_resolution_status"
                )
                == "resolved_unique"
            )
            &
            (
                pl.col(
                    "website_resolution_status"
                )
                == "resolved_unique"
            )
        )
        .with_columns(
            (
                pl.col(
                    "name_resolved_company_id"
                )
                ==
                pl.col(
                    "website_resolved_company_id"
                )
            )
            .alias(
                "name_website_agree"
            )
        )
    )

    validation_summary = (
        validation_population
        .group_by(
            "name_website_agree"
        )
        .len()
        .sort(
            "name_website_agree",
            descending=True,
        )
    )

    print(
        "\nNAME ↔ WEBSITE VALIDATION:"
    )

    print(
        validation_summary
    )

    agreement_count = (
        validation_population
        .filter(
            pl.col(
                "name_website_agree"
            )
        )
        .height
    )

    conflict_count = (
        validation_population
        .filter(
            ~pl.col(
                "name_website_agree"
            )
        )
        .height
    )

    print(
        f"\nUnique-name + unique-website agreements: "
        f"{agreement_count:,}"
    )

    print(
        f"Unique-name + unique-website conflicts:  "
        f"{conflict_count:,}"
    )

    # -----------------------------------------------------
    # Save conflicts for manual inspection
    # -----------------------------------------------------

    validation_conflicts = (
        validation_population
        .filter(
            ~pl.col(
                "name_website_agree"
            )
        )
        .select(
            "funding_round_id",
            "startup_name_raw",
            "funding_website",
            "name_resolved_company_id",
            "website_resolved_company_id",
        )
    )

    if validation_conflicts.height:

        validation_conflicts.write_csv(
            NAME_WEBSITE_CONFLICT_OUTPUT
        )

        print(
            "\nSaved name/website conflicts to:"
        )

        print(
            NAME_WEBSITE_CONFLICT_OUTPUT
        )

        print(
            "\nFirst 30 name/website conflicts:"
        )

        with pl.Config(
            tbl_rows=30,
            tbl_cols=-1,
            fmt_str_lengths=100,
        ):
            print(
                validation_conflicts
                .head(30)
            )

    # -----------------------------------------------------
    # 9. Test ambiguous-name recovery
    # -----------------------------------------------------

    print(
        "\n[9/11] Testing website evidence on "
        "ambiguous startup names..."
    )

    ambiguous_name_with_unique_website = (
        audit
        .filter(
            (
                pl.col(
                    "name_resolution_status"
                )
                == "ambiguous_multiple_ids"
            )
            &
            (
                pl.col(
                    "website_resolution_status"
                )
                == "resolved_unique"
            )
        )

        .with_columns(
            pl.col(
                "name_candidate_company_ids"
            )
            .list.contains(
                pl.col(
                    "website_resolved_company_id"
                )
            )
            .alias(
                "website_id_is_name_candidate"
            )
        )
    )

    ambiguous_recovery_summary = (
        ambiguous_name_with_unique_website
        .group_by(
            "website_id_is_name_candidate"
        )
        .len()
        .sort(
            "website_id_is_name_candidate",
            descending=True,
        )
    )

    print(
        "\nAMBIGUOUS NAME + UNIQUE WEBSITE:"
    )

    print(
        ambiguous_recovery_summary
    )

    consistent_ambiguous_recoveries = (
        ambiguous_name_with_unique_website
        .filter(
            pl.col(
                "website_id_is_name_candidate"
            )
        )
    )

    inconsistent_ambiguous_recoveries = (
        ambiguous_name_with_unique_website
        .filter(
            ~pl.col(
                "website_id_is_name_candidate"
            )
        )
    )

    print(
        "\nPotentially resolvable ambiguous-name rounds "
        "where website selects an existing name candidate:"
    )

    print(
        f"{consistent_ambiguous_recoveries.height:,}"
    )

    print(
        "\nWebsite/name candidate conflicts:"
    )

    print(
        f"{inconsistent_ambiguous_recoveries.height:,}"
    )

    if ambiguous_name_with_unique_website.height:

        (
            ambiguous_name_with_unique_website
            .select(
                "funding_round_id",
                "startup_name_raw",
                "funding_website",

                pl.col(
                    "name_candidate_company_ids"
                )
                .list.join("|")
                .alias(
                    "name_candidate_company_ids_pipe_separated"
                ),

                "website_resolved_company_id",
                "website_id_is_name_candidate",
            )
            .write_csv(
                AMBIGUOUS_NAME_RECOVERY_OUTPUT
            )
        )

        print(
            "\nSaved ambiguous-name website audit to:"
        )

        print(
            AMBIGUOUS_NAME_RECOVERY_OUTPUT
        )

    # -----------------------------------------------------
    # 10. Test unmatched-name recovery
    # -----------------------------------------------------

    print(
        "\n[10/11] Testing website evidence on "
        "unmatched startup names..."
    )

    unmatched_name_with_unique_website = (
        audit
        .filter(
            (
                pl.col(
                    "name_resolution_status"
                )
                == "unmatched_master"
            )
            &
            (
                pl.col(
                    "website_resolution_status"
                )
                == "resolved_unique"
            )
        )
    )

    print(
        "\nUnmatched-name funding rounds with "
        "a unique exact-website company candidate:"
    )

    print(
        f"{unmatched_name_with_unique_website.height:,}"
    )

    if unmatched_name_with_unique_website.height:

        (
            unmatched_name_with_unique_website
            .select(
                "funding_round_id",
                "startup_name_raw",
                "funding_website",
                "website_resolved_company_id",
            )
            .write_csv(
                UNMATCHED_NAME_RECOVERY_OUTPUT
            )
        )

        print(
            "\nSaved unmatched-name website candidates to:"
        )

        print(
            UNMATCHED_NAME_RECOVERY_OUTPUT
        )

        print(
            "\nFirst 30 unmatched-name website candidates:"
        )

        with pl.Config(
            tbl_rows=30,
            tbl_cols=-1,
            fmt_str_lengths=100,
        ):
            print(
                unmatched_name_with_unique_website
                .select(
                    "funding_round_id",
                    "startup_name_raw",
                    "funding_website",
                    "website_resolved_company_id",
                )
                .head(30)
            )

    # -----------------------------------------------------
    # 11. Measure impact at investor-mention level
    # -----------------------------------------------------

    print(
        "\n[11/11] Measuring investor-mention-level impact..."
    )

    round_resolution = (
        audit
        .select(
            "funding_round_id",
            "name_resolution_status",
            "website_resolution_status",
            "name_resolved_company_id",
            "website_resolved_company_id",
            "name_candidate_company_ids",
        )

        .with_columns(

            pl.when(
                pl.col(
                    "name_resolution_status"
                )
                == "resolved_unique"
            )
            .then(
                pl.lit(
                    "resolved_by_name"
                )
            )

            .when(
                (
                    pl.col(
                        "name_resolution_status"
                    )
                    == "ambiguous_multiple_ids"
                )
                &
                (
                    pl.col(
                        "website_resolution_status"
                    )
                    == "resolved_unique"
                )
                &
                (
                    pl.col(
                        "name_candidate_company_ids"
                    )
                    .list.contains(
                        pl.col(
                            "website_resolved_company_id"
                        )
                    )
                )
            )
            .then(
                pl.lit(
                    "website_consistent_ambiguity_candidate"
                )
            )

            .when(
                (
                    pl.col(
                        "name_resolution_status"
                    )
                    == "unmatched_master"
                )
                &
                (
                    pl.col(
                        "website_resolution_status"
                    )
                    == "resolved_unique"
                )
            )
            .then(
                pl.lit(
                    "website_candidate_for_unmatched_name"
                )
            )

            .otherwise(
                pl.lit(
                    "still_unresolved"
                )
            )
            .alias(
                "audit_resolution_category"
            )
        )
    )

    mentions = (
        pl.scan_parquet(
            INVESTOR_MENTIONS_PATH
        )
        .select(
            "funding_round_id",
            "id_resolution_status",
        )
        .collect()
    )

    mention_audit = (
        mentions
        .join(
            round_resolution,
            on="funding_round_id",
            how="left",
        )
    )

    mention_summary = (
        mention_audit
        .group_by(
            "audit_resolution_category"
        )
        .len()
        .sort(
            "len",
            descending=True,
        )
    )

    print(
        "\nINVESTOR-MENTION LEVEL "
        "STARTUP RESOLUTION POTENTIAL:"
    )

    print(
        mention_summary
    )

    # -----------------------------------------------------
    # Save complete round audit
    # -----------------------------------------------------

    audit.write_parquet(
        ROUND_AUDIT_OUTPUT,
        compression="zstd",
    )

    print(
        "\nSaved complete round-level audit to:"
    )

    print(
        ROUND_AUDIT_OUTPUT
    )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print(
        "\n" + "-" * 80
    )

    print(
        "PHASE 1.15 SUMMARY"
    )

    print(
        "-" * 80
    )

    print(
        f"Unique-name + unique-website agreements: "
        f"{agreement_count:,}"
    )

    print(
        f"Unique-name + unique-website conflicts:  "
        f"{conflict_count:,}"
    )

    print(
        "Ambiguous-name rounds potentially "
        "resolved consistently by website: "
        f"{consistent_ambiguous_recoveries.height:,}"
    )

    print(
        "Ambiguous-name website conflicts:       "
        f"{inconsistent_ambiguous_recoveries.height:,}"
    )

    print(
        "Unmatched-name rounds with unique "
        "website candidate: "
        f"{unmatched_name_with_unique_website.height:,}"
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "PHASE 1.15 COMPLETE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()