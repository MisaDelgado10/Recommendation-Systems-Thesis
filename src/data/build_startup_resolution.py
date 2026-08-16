from __future__ import annotations

from pathlib import Path

import polars as pl


# =========================================================
# Configuration
# =========================================================

AUDIT_PATH = Path(
    "data/interim/startup_website_resolution_audit.parquet"
)

INVESTOR_MENTIONS_PATH = Path(
    "data/interim/investor_mentions.parquet"
)

INTERIM_DIR = Path(
    "data/interim"
)

OUTPUT_PATH = (
    INTERIM_DIR
    / "startup_resolution_by_round.parquet"
)

UNRESOLVED_OUTPUT = (
    INTERIM_DIR
    / "startup_resolution_unresolved.csv"
)


# =========================================================
# Main
# =========================================================

def main() -> None:

    print("=" * 80)
    print("PHASE 1.16 — BUILD FINAL STARTUP RESOLUTION TABLE")
    print("=" * 80)

    INTERIM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # 1. Verify inputs
    # -----------------------------------------------------

    print("\n[1/8] Checking input files...")

    for path in [
        AUDIT_PATH,
        INVESTOR_MENTIONS_PATH,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n"
                f"{path.resolve()}"
            )

    print("Input files found.")

    # -----------------------------------------------------
    # 2. Load Phase 1.15 audit
    # -----------------------------------------------------

    print(
        "\n[2/8] Loading startup website-resolution audit..."
    )

    audit = pl.read_parquet(
        AUDIT_PATH
    )

    print(
        f"Funding-round records: "
        f"{audit.height:,}"
    )

    # -----------------------------------------------------
    # 3. Create deterministic resolution status
    # -----------------------------------------------------

    print(
        "\n[3/8] Applying evidence-based "
        "startup resolution rules..."
    )

    resolution = (
        audit

        .with_columns(

            # =============================================
            # Rule 1:
            # exact company name maps uniquely.
            # =============================================

            pl.when(
                pl.col(
                    "name_resolution_status"
                )
                == "resolved_unique"
            )
            .then(
                pl.col(
                    "name_resolved_company_id"
                )
            )

            # =============================================
            # Rule 2:
            #
            # Name is ambiguous,
            # website is unique,
            # website-selected ID belongs to the candidate
            # set produced by the exact company name.
            # =============================================

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
                pl.col(
                    "website_resolved_company_id"
                )
            )

            # =============================================
            # Everything else remains unresolved.
            # =============================================

            .otherwise(
                None
            )
            .alias(
                "startup_id"
            )
        )

        .with_columns(

            # =============================================
            # Resolution-method label
            # =============================================

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
                    "resolved_by_name_and_website"
                )
            )

            # =============================================
            # Four cases observed in Phase 1.15:
            # unique website points outside name candidate
            # set.
            # =============================================

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
                    ~pl.col(
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
                    "unresolved_name_website_conflict"
                )
            )

            # =============================================
            # Website-only candidates:
            #
            # preserved as evidence, but NOT accepted.
            # =============================================

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
                    "unresolved_website_only_candidate"
                )
            )

            .when(
                pl.col(
                    "name_resolution_status"
                )
                == "missing_name"
            )
            .then(
                pl.lit(
                    "unresolved_missing_name"
                )
            )

            .when(
                pl.col(
                    "name_resolution_status"
                )
                == "ambiguous_multiple_ids"
            )
            .then(
                pl.lit(
                    "unresolved_ambiguous_name"
                )
            )

            .when(
                pl.col(
                    "name_resolution_status"
                )
                == "unmatched_master"
            )
            .then(
                pl.lit(
                    "unresolved_unmatched_name"
                )
            )

            .otherwise(
                pl.lit(
                    "unresolved_other"
                )
            )
            .alias(
                "startup_resolution_method"
            )
        )
    )

    # -----------------------------------------------------
    # 4. Keep only fields needed downstream
    # -----------------------------------------------------

    print(
        "\n[4/8] Building compact startup-resolution table..."
    )

    startup_resolution = (
        resolution
        .select(
            "funding_round_id",

            "startup_name_raw",
            "startup_name_normalized",

            "funding_website",
            "website_exact",

            "startup_id",

            "startup_resolution_method",

            "name_resolution_status",
            "website_resolution_status",

            "name_resolved_company_id",
            "website_resolved_company_id",

            "name_candidate_company_ids",
            "name_company_id_count",

            "website_candidate_company_ids",
            "website_company_id_count",
        )
    )

    # -----------------------------------------------------
    # 5. Integrity checks
    # -----------------------------------------------------

    print(
        "\n[5/8] Running integrity checks..."
    )

    # There must be one row per funding round.
    duplicate_rounds = (
        startup_resolution
        .group_by(
            "funding_round_id"
        )
        .len()
        .filter(
            pl.col("len") > 1
        )
        .height
    )

    missing_round_ids = (
        startup_resolution
        .select(
            pl.col(
                "funding_round_id"
            )
            .null_count()
        )
        .item()
    )

    # Resolved statuses must always have a startup ID.
    invalid_resolved = (
        startup_resolution
        .filter(
            pl.col(
                "startup_resolution_method"
            )
            .is_in(
                [
                    "resolved_by_name",
                    "resolved_by_name_and_website",
                ]
            )
        )
        .filter(
            pl.col(
                "startup_id"
            )
            .is_null()
        )
        .height
    )

    # Unresolved statuses must not be assigned startup IDs.
    invalid_unresolved = (
        startup_resolution
        .filter(
            pl.col(
                "startup_resolution_method"
            )
            .str.starts_with(
                "unresolved"
            )
        )
        .filter(
            pl.col(
                "startup_id"
            )
            .is_not_null()
        )
        .height
    )

    print(
        f"Duplicate funding rounds:       "
        f"{duplicate_rounds:,}"
    )

    print(
        f"Missing funding-round IDs:      "
        f"{missing_round_ids:,}"
    )

    print(
        f"Resolved rows missing ID:       "
        f"{invalid_resolved:,}"
    )

    print(
        f"Unresolved rows with forced ID: "
        f"{invalid_unresolved:,}"
    )

    if duplicate_rounds != 0:

        raise RuntimeError(
            "Duplicate funding_round_id rows "
            "exist in startup-resolution table."
        )

    if missing_round_ids != 0:

        raise RuntimeError(
            "Missing funding_round_id detected."
        )

    if invalid_resolved != 0:

        raise RuntimeError(
            "Resolved startup rows without "
            "startup IDs were detected."
        )

    if invalid_unresolved != 0:

        raise RuntimeError(
            "Unresolved startup rows were "
            "incorrectly assigned IDs."
        )

    print(
        "Integrity checks passed."
    )

    # -----------------------------------------------------
    # 6. Resolution summary
    # -----------------------------------------------------

    print(
        "\n[6/8] Calculating startup-resolution summary..."
    )

    resolution_summary = (
        startup_resolution
        .group_by(
            "startup_resolution_method"
        )
        .len()
        .sort(
            "len",
            descending=True,
        )
    )

    print(
        "\nFUNDING-ROUND LEVEL RESOLUTION:"
    )

    print(
        resolution_summary
    )

    resolved_rounds = (
        startup_resolution
        .filter(
            pl.col(
                "startup_id"
            )
            .is_not_null()
        )
        .height
    )

    unresolved_rounds = (
        startup_resolution.height
        - resolved_rounds
    )

    unique_resolved_startups = (
        startup_resolution
        .filter(
            pl.col(
                "startup_id"
            )
            .is_not_null()
        )
        .select(
            pl.col(
                "startup_id"
            )
            .n_unique()
        )
        .item()
    )

    print(
        f"\nResolved funding rounds:   "
        f"{resolved_rounds:,}"
    )

    print(
        f"Unresolved funding rounds: "
        f"{unresolved_rounds:,}"
    )

    print(
        f"Unique resolved startups:  "
        f"{unique_resolved_startups:,}"
    )

    # -----------------------------------------------------
    # 7. Measure resolution at investor-mention level
    # -----------------------------------------------------

    print(
        "\n[7/8] Measuring investor-mention impact..."
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

    mention_resolution = (
        mentions
        .join(
            startup_resolution
            .select(
                "funding_round_id",
                "startup_id",
                "startup_resolution_method",
            ),
            on="funding_round_id",
            how="left",
        )
    )

    mention_summary = (
        mention_resolution
        .group_by(
            "startup_resolution_method"
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
        mention_summary
    )

    resolved_mentions = (
        mention_resolution
        .filter(
            pl.col(
                "startup_id"
            )
            .is_not_null()
        )
        .height
    )

    # -----------------------------------------------------
    # This is VERY important:
    #
    # how many rows already have BOTH:
    #
    # a unique investor ID
    # and
    # a resolved startup ID?
    # -----------------------------------------------------

    fully_resolved_mentions = (
        mention_resolution
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
                    "startup_id"
                )
                .is_not_null()
            )
        )
        .height
    )

    print(
        f"\nStartup-resolved mentions: "
        f"{resolved_mentions:,}"
    )

    print(
        "\nMentions with BOTH investor and "
        "startup uniquely resolved:"
    )

    print(
        f"{fully_resolved_mentions:,}"
    )

    # -----------------------------------------------------
    # 8. Save outputs
    # -----------------------------------------------------

    print(
        "\n[8/8] Saving startup-resolution table..."
    )

    startup_resolution.write_parquet(
        OUTPUT_PATH,
        compression="zstd",
    )

    # Save unresolved diagnostic as CSV.
    #
    # Convert list fields into flat strings first.
    unresolved = (
        startup_resolution
        .filter(
            pl.col(
                "startup_id"
            )
            .is_null()
        )
        .select(
            "funding_round_id",
            "startup_name_raw",
            "funding_website",
            "startup_resolution_method",

            pl.col(
                "name_candidate_company_ids"
            )
            .list.join("|")
            .alias(
                "name_candidate_company_ids"
            ),

            "website_resolved_company_id",
        )
    )

    unresolved.write_csv(
        UNRESOLVED_OUTPUT
    )

    print(
        f"\nSaved startup resolution table:\n"
        f"{OUTPUT_PATH}"
    )

    print(
        f"\nSaved unresolved diagnostic:\n"
        f"{UNRESOLVED_OUTPUT}"
    )

    # =====================================================
    # Final summary
    # =====================================================

    print(
        "\n" + "-" * 80
    )

    print(
        "PHASE 1.16 SUMMARY"
    )

    print(
        "-" * 80
    )

    print(
        f"Funding rounds:                         "
        f"{startup_resolution.height:,}"
    )

    print(
        f"Resolved funding rounds:                "
        f"{resolved_rounds:,}"
    )

    print(
        f"Unresolved funding rounds:              "
        f"{unresolved_rounds:,}"
    )

    print(
        f"Unique resolved startups:               "
        f"{unique_resolved_startups:,}"
    )

    print(
        f"Startup-resolved investor mentions:     "
        f"{resolved_mentions:,}"
    )

    print(
        f"Investor + startup fully resolved:      "
        f"{fully_resolved_mentions:,}"
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "PHASE 1.16 COMPLETE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()