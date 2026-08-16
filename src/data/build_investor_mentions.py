from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import polars as pl

from investor_parsing import (
    find_single_unknown_recovery,
    normalize_name,
    parse_investor_string,
)


# =========================================================
# Configuration
# =========================================================

FUNDING_PATH = Path(
    "data/raw/CRUNCHBASE_funding_20260531_797177.csv"
)

INVESTOR_PATH = Path(
    "data/raw/CRUNCHBASE_investor_20260531_333726.csv"
)

INTERIM_DIR = Path(
    "data/interim"
)

OUTPUT_PATH = (
    INTERIM_DIR
    / "investor_mentions.parquet"
)

UNRESOLVED_ROUNDS_PATH = (
    INTERIM_DIR
    / "investor_mentions_unresolved_rounds.csv"
)


# =========================================================
# Explicit output schema
# =========================================================
#
# We define the schema explicitly instead of asking Polars
# to infer it from the first rows.
#
# This is important because some columns contain mostly null
# values but occasionally contain strings, e.g.
# alternate_repair_solution.
# =========================================================

MENTION_SCHEMA = {
    "funding_round_id": pl.String,
    "announced_on": pl.String,
    "startup_name_raw": pl.String,
    "investment_type": pl.String,
    "investors_raw": pl.String,
    "mention_order": pl.Int64,
    "investor_name": pl.String,
    "investor_id": pl.String,
    "candidate_investor_ids": pl.List(pl.String),
    "candidate_id_count": pl.Int64,
    "id_resolution_status": pl.String,
    "parse_quality_status": pl.String,
    "source_contains_empty_chunk": pl.Boolean,
    "canonical_name_looks_malformed": pl.Boolean,
    "alternate_repair_solution": pl.String,
}


# =========================================================
# Quality helpers
# =========================================================

def canonical_name_looks_malformed(
    name: str,
) -> bool:
    """
    Flag malformed patterns that were actually observed
    during our Crunchbase investor-master audit.

    We observed canonical investor names that:
    - end with a comma
    - contain consecutive commas

    We preserve those names exactly as they exist in the
    Crunchbase investor master, but mark them for auditing.
    """

    return (
        name.startswith(",")
        or name.endswith(",")
        or ",," in name
    )


def resolve_name_to_ids(
    investor_name: str,
    name_to_ids: dict[str, list[str]],
) -> tuple[
    str,
    str | None,
    list[str],
]:
    """
    Resolve a parsed investor name against the Crunchbase
    investor master.

    Returns
    -------
    resolution_status
        One of:
        - resolved_unique
        - ambiguous_multiple_ids
        - unmatched_master

    resolved_id
        The Crunchbase investor ID when exactly one
        matching entity exists. Otherwise None.

    candidate_ids
        All Crunchbase IDs associated with the name.
    """

    candidate_ids = name_to_ids.get(
        investor_name,
        [],
    )

    # No matching investor entity in investor master.
    if len(candidate_ids) == 0:

        return (
            "unmatched_master",
            None,
            [],
        )

    # Exactly one investor ID.
    if len(candidate_ids) == 1:

        return (
            "resolved_unique",
            candidate_ids[0],
            candidate_ids,
        )

    # Same canonical name maps to multiple Crunchbase IDs.
    return (
        "ambiguous_multiple_ids",
        None,
        candidate_ids,
    )


# =========================================================
# Mention construction
# =========================================================

def build_mention_row(
    *,
    funding_round_id: str,
    announced_on: str,
    startup_name: str | None,
    investment_type: str,
    investors_raw: str,
    investor_name: str,
    mention_order: int,
    parse_quality_status: str,
    source_contains_empty_chunk: bool,
    alternate_repair_solution: str | None,
    name_to_ids: dict[str, list[str]],
) -> dict:
    """
    Construct one normalized investor-mention record.

    One row represents one investor mentioned in one
    Crunchbase funding round.
    """

    (
        resolution_status,
        investor_id,
        candidate_ids,
    ) = resolve_name_to_ids(
        investor_name,
        name_to_ids,
    )

    return {
        "funding_round_id":
            funding_round_id,

        "announced_on":
            announced_on,

        "startup_name_raw":
            startup_name,

        "investment_type":
            investment_type,

        "investors_raw":
            investors_raw,

        "mention_order":
            mention_order,

        "investor_name":
            investor_name,

        "investor_id":
            investor_id,

        "candidate_investor_ids":
            candidate_ids,

        "candidate_id_count":
            len(candidate_ids),

        "id_resolution_status":
            resolution_status,

        "parse_quality_status":
            parse_quality_status,

        "source_contains_empty_chunk":
            source_contains_empty_chunk,

        "canonical_name_looks_malformed":
            canonical_name_looks_malformed(
                investor_name
            ),

        "alternate_repair_solution":
            alternate_repair_solution,
    }


# =========================================================
# Main
# =========================================================

def main() -> None:

    print("=" * 80)
    print("PHASE 1.12 — BUILD INVESTOR MENTION TABLE")
    print("=" * 80)

    INTERIM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # 1. Verify source files
    # -----------------------------------------------------

    print("\n[1/10] Checking source files...")

    for path in [
        FUNDING_PATH,
        INVESTOR_PATH,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n"
                f"{path.resolve()}"
            )

    print("Source files found.")

    # -----------------------------------------------------
    # 2. Load investor master
    # -----------------------------------------------------

    print("\n[2/10] Loading Crunchbase investor master...")

    investor_df = (
        pl.scan_csv(
            INVESTOR_PATH,
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
            "id",
            "name",
        )
        .filter(
            pl.col("name").is_not_null()
        )
        .collect()
    )

    print(
        f"Investor master rows: "
        f"{investor_df.height:,}"
    )

    # -----------------------------------------------------
    # 3. Build canonical name -> Crunchbase IDs dictionary
    # -----------------------------------------------------

    print(
        "\n[3/10] Building canonical investor-name dictionary..."
    )

    name_to_ids: dict[
        str,
        list[str],
    ] = defaultdict(list)

    for (
        investor_id,
        investor_name,
    ) in investor_df.iter_rows():

        normalized_name = normalize_name(
            investor_name
        )

        name_to_ids[
            normalized_name
        ].append(
            investor_id
        )

    canonical_names = set(
        name_to_ids.keys()
    )

    max_name_chunks = max(
        name.count(",") + 1
        for name in canonical_names
    )

    print(
        f"Canonical names: "
        f"{len(canonical_names):,}"
    )

    print(
        "Maximum comma-separated chunks "
        f"inside one canonical name: "
        f"{max_name_chunks}"
    )

    # -----------------------------------------------------
    # 4. Load funding records
    # -----------------------------------------------------

    print(
        "\n[4/10] Loading funding rounds with investor data..."
    )

    funding_df = (
        pl.scan_csv(
            FUNDING_PATH,
            schema_overrides={
                "id": pl.String,
                "announced_on": pl.String,
                "org_name": pl.String,
                "investment_type": pl.String,
                "investors": pl.String,
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
            "id",
            "announced_on",
            "org_name",
            "investment_type",
            "investors",
        )
        .filter(
            pl.col("investors").is_not_null()
        )
        .collect()
    )

    print(
        f"Funding rows with investors: "
        f"{funding_df.height:,}"
    )

    # -----------------------------------------------------
    # 5. Parse investor strings and build mentions
    # -----------------------------------------------------

    print(
        "\n[5/10] Parsing investor mentions..."
    )

    mention_rows: list[dict] = []

    unresolved_rounds: list[dict] = []

    # Cache repeated investor strings.
    #
    # Many funding rounds may contain identical investor
    # strings, so this avoids repeating dictionary
    # segmentation work unnecessarily.
    parsing_cache: dict[str, dict] = {}

    recovery_cache: dict[str, dict | None] = {}

    total_funding_rows = funding_df.height

    for row_number, (
        funding_round_id,
        announced_on,
        startup_name,
        investment_type,
        investors_raw,
    ) in enumerate(
        funding_df.iter_rows(),
        start=1,
    ):

        # Optional progress update every 100,000 rows.
        if row_number % 100_000 == 0:

            print(
                f"  Processed "
                f"{row_number:,} / "
                f"{total_funding_rows:,} "
                f"funding rows..."
            )

        # -------------------------------------------------
        # Parse once per unique raw investor string
        # -------------------------------------------------

        if investors_raw not in parsing_cache:

            parsing_cache[
                investors_raw
            ] = parse_investor_string(
                raw_value=investors_raw,
                canonical_names=canonical_names,
                max_name_chunks=max_name_chunks,
            )

        parsed = parsing_cache[
            investors_raw
        ]

        strict = parsed[
            "strict_solutions"
        ]

        repair = parsed[
            "repair_solutions"
        ]

        contains_empty = parsed[
            "contains_empty_chunk"
        ]

        # =================================================
        # Case A
        # Exactly one valid full segmentation.
        # =================================================

        if len(strict) == 1:

            names = strict[0]

            alternate_repair = None

            # ---------------------------------------------
            # Empty chunk with two possible interpretations.
            #
            # Example observed in our data:
            #
            # Amit Agarwal,,Anthology Fund...
            #
            # Strict interpretation might correspond to:
            # "Amit Agarwal,"
            #
            # while removing the empty chunk corresponds to:
            # "Amit Agarwal"
            #
            # We retain the strict canonical interpretation
            # but preserve the alternate version.
            # ---------------------------------------------

            if (
                contains_empty
                and len(repair) == 1
            ):

                parse_quality = (
                    "empty_chunk_dual_interpretation"
                )

                alternate_repair = (
                    " || ".join(
                        repair[0]
                    )
                )

            # ---------------------------------------------
            # Empty chunk exists, but only strict canonical
            # interpretation is supported by investor master.
            # ---------------------------------------------

            elif contains_empty:

                parse_quality = (
                    "empty_chunk_strict_only"
                )

            # ---------------------------------------------
            # Normal funding record.
            # ---------------------------------------------

            else:

                parse_quality = (
                    "clean_full_parse"
                )

            for order, name in enumerate(
                names
            ):

                mention_rows.append(
                    build_mention_row(
                        funding_round_id=
                            funding_round_id,

                        announced_on=
                            announced_on,

                        startup_name=
                            startup_name,

                        investment_type=
                            investment_type,

                        investors_raw=
                            investors_raw,

                        investor_name=
                            name,

                        mention_order=
                            order,

                        parse_quality_status=
                            parse_quality,

                        source_contains_empty_chunk=
                            contains_empty,

                        alternate_repair_solution=
                            alternate_repair,

                        name_to_ids=
                            name_to_ids,
                    )
                )

            continue

        # =================================================
        # Case B
        # More than one full segmentation exists.
        #
        # Our previous audit found zero such rows.
        #
        # If one appears, do not arbitrarily choose.
        # =================================================

        if len(strict) > 1:

            unresolved_rounds.append(
                {
                    "funding_round_id":
                        funding_round_id,

                    "investors":
                        investors_raw,

                    "reason":
                        "ambiguous_full_segmentation",
                }
            )

            continue

        # =================================================
        # Case C
        # No complete canonical segmentation.
        #
        # Our previous audit found 17 such funding rounds,
        # all caused by HALA Ventures being absent from the
        # investor master.
        #
        # We conservatively recover exactly one unknown
        # source span and preserve it as unmatched.
        # =================================================

        if investors_raw not in recovery_cache:

            recovery_cache[
                investors_raw
            ] = find_single_unknown_recovery(
                raw_value=investors_raw,
                canonical_names=canonical_names,
                max_name_chunks=max_name_chunks,
            )

        recovery = recovery_cache[
            investors_raw
        ]

        if recovery is None:

            unresolved_rounds.append(
                {
                    "funding_round_id":
                        funding_round_id,

                    "investors":
                        investors_raw,

                    "reason":
                        "no_conservative_recovery",
                }
            )

            continue

        # -------------------------------------------------
        # Reconstruct original mention order:
        #
        # known names before unknown
        # unknown source investor
        # known names after unknown
        # -------------------------------------------------

        ordered_mentions: list[
            tuple[str, str]
        ] = []

        for name in recovery["before"]:

            ordered_mentions.append(
                (
                    name,
                    "partial_parse_known",
                )
            )

        ordered_mentions.append(
            (
                recovery["unknown"],
                "partial_parse_unmatched",
            )
        )

        for name in recovery["after"]:

            ordered_mentions.append(
                (
                    name,
                    "partial_parse_known",
                )
            )

        for order, (
            name,
            quality_status,
        ) in enumerate(
            ordered_mentions
        ):

            mention_rows.append(
                build_mention_row(
                    funding_round_id=
                        funding_round_id,

                    announced_on=
                        announced_on,

                    startup_name=
                        startup_name,

                    investment_type=
                        investment_type,

                    investors_raw=
                        investors_raw,

                    investor_name=
                        name,

                    mention_order=
                        order,

                    parse_quality_status=
                        quality_status,

                    source_contains_empty_chunk=
                        False,

                    alternate_repair_solution=
                        None,

                    name_to_ids=
                        name_to_ids,
                )
            )

    print(
        f"\nRaw investor mentions built: "
        f"{len(mention_rows):,}"
    )

    # -----------------------------------------------------
    # 6. Stop if any funding rounds remain unresolved
    # -----------------------------------------------------

    print(
        "\n[6/10] Checking for unresolved funding rounds..."
    )

    if unresolved_rounds:

        unresolved_df = pl.DataFrame(
            unresolved_rounds,
            schema={
                "funding_round_id": pl.String,
                "investors": pl.String,
                "reason": pl.String,
            },
        )

        unresolved_df.write_csv(
            UNRESOLVED_ROUNDS_PATH
        )

        print(
            f"\nUnresolved rounds: "
            f"{unresolved_df.height:,}"
        )

        print(
            f"Saved diagnostic file to:\n"
            f"{UNRESOLVED_ROUNDS_PATH}"
        )

        raise RuntimeError(
            "\nSome funding rounds could not be "
            "parsed conservatively.\n"
            "The mention table was NOT saved."
        )

    print(
        "No unresolved funding rounds remain."
    )

    # -----------------------------------------------------
    # 7. Construct DataFrame using EXPLICIT schema
    # -----------------------------------------------------

    print(
        "\n[7/10] Constructing Polars mention table..."
    )

    mentions_df = pl.from_dicts(
        mention_rows,
        schema=MENTION_SCHEMA,
        strict=True,
    )

    print(
        f"Constructed DataFrame with "
        f"{mentions_df.height:,} rows."
    )

    # -----------------------------------------------------
    # 7.1 Print schema
    # -----------------------------------------------------

    print("\nConstructed mention-table schema:")

    for (
        column_name,
        dtype,
    ) in mentions_df.schema.items():

        print(
            f"  {column_name:<35} "
            f"{dtype}"
        )

    # -----------------------------------------------------
    # 8. Verify all source funding rounds are represented
    # -----------------------------------------------------

    print(
        "\n[8/10] Validating funding-round coverage..."
    )

    source_rounds = (
        funding_df
        .select(
            pl.col("id").n_unique()
        )
        .item()
    )

    represented_rounds = (
        mentions_df
        .select(
            pl.col(
                "funding_round_id"
            ).n_unique()
        )
        .item()
    )

    print(
        f"Source funding rounds:      "
        f"{source_rounds:,}"
    )

    print(
        f"Represented funding rounds: "
        f"{represented_rounds:,}"
    )

    if represented_rounds != source_rounds:

        raise RuntimeError(
            "Not every funding round with investor data "
            "is represented in the mention table.\n"
            f"Source rounds:      "
            f"{source_rounds:,}\n"
            f"Represented rounds: "
            f"{represented_rounds:,}"
        )

    print(
        "Funding-round coverage validation passed."
    )

    # -----------------------------------------------------
    # 9. Additional integrity checks
    # -----------------------------------------------------

    print(
        "\n[9/10] Running integrity checks..."
    )

    # Funding round IDs should never be null.
    missing_round_ids = (
        mentions_df
        .select(
            pl.col(
                "funding_round_id"
            ).null_count()
        )
        .item()
    )

    # Investor names should never be null.
    missing_investor_names = (
        mentions_df
        .select(
            pl.col(
                "investor_name"
            ).null_count()
        )
        .item()
    )

    # Announced dates were previously audited and should
    # remain present.
    missing_dates = (
        mentions_df
        .select(
            pl.col(
                "announced_on"
            ).null_count()
        )
        .item()
    )

    # Every uniquely resolved investor must have exactly
    # one candidate ID and a non-null investor_id.
    invalid_unique_resolution = (
        mentions_df
        .filter(
            pl.col(
                "id_resolution_status"
            )
            == "resolved_unique"
        )
        .filter(
            pl.col("investor_id").is_null()
            | (
                pl.col(
                    "candidate_id_count"
                )
                != 1
            )
        )
        .height
    )

    # Ambiguous investors should not receive a forced ID.
    invalid_ambiguous_resolution = (
        mentions_df
        .filter(
            pl.col(
                "id_resolution_status"
            )
            == "ambiguous_multiple_ids"
        )
        .filter(
            pl.col(
                "investor_id"
            ).is_not_null()
        )
        .height
    )

    print(
        f"Missing funding-round IDs:   "
        f"{missing_round_ids:,}"
    )

    print(
        f"Missing investor names:      "
        f"{missing_investor_names:,}"
    )

    print(
        f"Missing announced dates:     "
        f"{missing_dates:,}"
    )

    print(
        "Invalid unique resolutions:  "
        f"{invalid_unique_resolution:,}"
    )

    print(
        "Forced ambiguous IDs:        "
        f"{invalid_ambiguous_resolution:,}"
    )

    if missing_round_ids != 0:

        raise RuntimeError(
            "Missing funding-round IDs were found."
        )

    if missing_investor_names != 0:

        raise RuntimeError(
            "Missing investor names were found."
        )

    if missing_dates != 0:

        raise RuntimeError(
            "Missing announced dates were found."
        )

    if invalid_unique_resolution != 0:

        raise RuntimeError(
            "Some resolved_unique rows have invalid "
            "ID mappings."
        )

    if invalid_ambiguous_resolution != 0:

        raise RuntimeError(
            "Some ambiguous investor names were "
            "incorrectly assigned a forced ID."
        )

    print(
        "Integrity checks passed."
    )

    # -----------------------------------------------------
    # 10. Save Parquet
    # -----------------------------------------------------

    print(
        "\n[10/10] Saving investor mention table..."
    )

    mentions_df.write_parquet(
        OUTPUT_PATH,
        compression="zstd",
    )

    # =====================================================
    # Final summary
    # =====================================================

    print("\n" + "-" * 80)
    print("OUTPUT SUMMARY")
    print("-" * 80)

    print(
        f"Investor mention rows:       "
        f"{mentions_df.height:,}"
    )

    print(
        f"Funding rounds represented:  "
        f"{represented_rounds:,}"
    )

    # -----------------------------------------------------
    # ID-resolution distribution
    # -----------------------------------------------------

    print("\nID resolution:")

    resolution_summary = (
        mentions_df
        .group_by(
            "id_resolution_status"
        )
        .len()
        .sort(
            "len",
            descending=True,
        )
    )

    print(
        resolution_summary
    )

    # -----------------------------------------------------
    # Parsing-quality distribution
    # -----------------------------------------------------

    print("\nParsing quality:")

    parsing_summary = (
        mentions_df
        .group_by(
            "parse_quality_status"
        )
        .len()
        .sort(
            "len",
            descending=True,
        )
    )

    print(
        parsing_summary
    )

    # -----------------------------------------------------
    # Malformed canonical-name mentions
    # -----------------------------------------------------

    malformed_mentions = (
        mentions_df
        .filter(
            pl.col(
                "canonical_name_looks_malformed"
            )
        )
        .height
    )

    print(
        "\nMalformed canonical-name mentions:"
    )

    print(
        f"{malformed_mentions:,}"
    )

    # -----------------------------------------------------
    # Unique resolved investors
    # -----------------------------------------------------

    unique_resolved_investors = (
        mentions_df
        .filter(
            pl.col(
                "id_resolution_status"
            )
            == "resolved_unique"
        )
        .select(
            pl.col(
                "investor_id"
            ).n_unique()
        )
        .item()
    )

    print(
        "\nUnique resolved investors:"
    )

    print(
        f"{unique_resolved_investors:,}"
    )

    # -----------------------------------------------------
    # Unmatched source names
    # -----------------------------------------------------

    unmatched_names = (
        mentions_df
        .filter(
            pl.col(
                "id_resolution_status"
            )
            == "unmatched_master"
        )
        .group_by(
            "investor_name"
        )
        .len()
        .sort(
            "len",
            descending=True,
        )
    )

    print(
        "\nUnmatched investor names:"
    )

    print(
        unmatched_names
    )

    # -----------------------------------------------------
    # File output
    # -----------------------------------------------------

    print(
        f"\nSaved to:\n"
        f"{OUTPUT_PATH}"
    )

    print("\n" + "=" * 80)
    print("PHASE 1.12 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()