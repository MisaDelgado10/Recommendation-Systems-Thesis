from __future__ import annotations

from collections import Counter
from pathlib import Path

import polars as pl

from audit_canonical_parsing import (
    find_segmentations,
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

INTERIM_DIR = Path("data/interim")

EMPTY_CHUNK_OUTPUT = (
    INTERIM_DIR / "investor_empty_chunk_audit.csv"
)

UNPARSED_DIAGNOSTIC_OUTPUT = (
    INTERIM_DIR / "investor_unparsed_diagnostic.csv"
)


# =========================================================
# Helper
# =========================================================

def find_blocking_spans(
    raw_value: str,
    canonical_names: set[str],
    max_name_chunks: int,
) -> list[dict]:
    """
    For an unparseable investor string, remove one contiguous
    span of comma chunks and test whether the remainder becomes
    uniquely parseable.

    This is DIAGNOSTIC ONLY.

    It allows us to identify a possible missing canonical
    investor name without automatically changing the data.
    """

    normalized = normalize_name(raw_value)

    chunks = [
        chunk.strip()
        for chunk in normalized.split(",")
    ]

    candidates: list[dict] = []

    # Try the smallest possible removed span first.
    for width in range(
        1,
        min(max_name_chunks, len(chunks)) + 1,
    ):

        for start in range(
            0,
            len(chunks) - width + 1,
        ):

            removed_chunks = chunks[
                start : start + width
            ]

            remaining_chunks = tuple(
                chunks[:start]
                + chunks[start + width :]
            )

            if not remaining_chunks:
                continue

            solutions = find_segmentations(
                chunks=remaining_chunks,
                canonical_names=canonical_names,
                max_name_chunks=max_name_chunks,
            )

            if len(solutions) == 1:

                candidates.append(
                    {
                        "blocking_text":
                            ",".join(removed_chunks),

                        "removed_chunk_count":
                            width,

                        "remaining_solution":
                            " || ".join(
                                solutions[0]
                            ),
                    }
                )

        # If removing one chunk solves it, we don't need
        # to test two-, three-, etc. chunk removals.
        if candidates:
            break

    return candidates


# =========================================================
# Main
# =========================================================

def main() -> None:

    print("=" * 80)
    print("PHASE 1.11 — INVESTOR PARSER EDGE-CASE AUDIT")
    print("=" * 80)

    INTERIM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # 1. Load investor names
    # -----------------------------------------------------

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

    canonical_names = {
        normalize_name(name)
        for name in investor_df
        .get_column("name")
        .to_list()
    }

    max_name_chunks = max(
        name.count(",") + 1
        for name in canonical_names
    )

    print(
        f"\nCanonical investor names: "
        f"{len(canonical_names):,}"
    )

    # -----------------------------------------------------
    # 2. Audit malformed-looking canonical names
    # -----------------------------------------------------

    leading_comma_names = sorted(
        name
        for name in canonical_names
        if name.startswith(",")
    )

    trailing_comma_names = sorted(
        name
        for name in canonical_names
        if name.endswith(",")
    )

    double_comma_names = sorted(
        name
        for name in canonical_names
        if ",," in name
    )

    print("\nMalformed-looking canonical names:")

    print(
        f"Starts with comma: "
        f"{len(leading_comma_names):,}"
    )

    print(
        f"Ends with comma:   "
        f"{len(trailing_comma_names):,}"
    )

    print(
        f"Contains ',,':     "
        f"{len(double_comma_names):,}"
    )

    if leading_comma_names:
        print("\nExamples starting with comma:")
        for value in leading_comma_names[:20]:
            print(repr(value))

    if trailing_comma_names:
        print("\nExamples ending with comma:")
        for value in trailing_comma_names[:20]:
            print(repr(value))

    if double_comma_names:
        print("\nExamples containing double comma:")
        for value in double_comma_names[:20]:
            print(repr(value))

    # -----------------------------------------------------
    # 3. Confirm HALA Ventures locally
    # -----------------------------------------------------

    print("\nExact canonical-name checks:")

    names_to_check = [
        "HALA Ventures",
        "Techstars",
        "500 Global",
        "Plug and Play",
        "National Science Foundation",
    ]

    for name in names_to_check:

        print(
            f"{name:<30} "
            f"{'FOUND' if name in canonical_names else 'NOT FOUND'}"
        )

    # -----------------------------------------------------
    # 4. Load funding rows
    # -----------------------------------------------------

    funding_df = (
        pl.scan_csv(
            FUNDING_PATH,
            schema_overrides={
                "id": pl.String,
                "investors": pl.String,
            },
            null_values=[
                "",
                "null",
                "NULL",
                "None",
            ],
        )
        .select(
            "id",
            "investors",
        )
        .filter(
            pl.col("investors").is_not_null()
        )
        .collect()
    )

    # -----------------------------------------------------
    # 5. Investigate every row containing an empty chunk
    # -----------------------------------------------------

    empty_chunk_rows = []

    unparsed_rows = []

    for funding_round_id, raw_investors in (
        funding_df.iter_rows()
    ):

        result = parse_investor_string(
            raw_value=raw_investors,
            canonical_names=canonical_names,
            max_name_chunks=max_name_chunks,
        )

        strict_solutions = result[
            "strict_solutions"
        ]

        repair_solutions = result[
            "repair_solutions"
        ]

        # -------------------------------------------------
        # Empty chunks
        # -------------------------------------------------

        if result["contains_empty_chunk"]:

            empty_chunk_rows.append(
                {
                    "funding_round_id":
                        funding_round_id,

                    "investors":
                        raw_investors,

                    "strict_solution_count":
                        len(strict_solutions),

                    "strict_solution":
                        (
                            " || ".join(
                                strict_solutions[0]
                            )
                            if strict_solutions
                            else None
                        ),

                    "repair_solution_count":
                        len(repair_solutions),

                    "repair_solution":
                        (
                            " || ".join(
                                repair_solutions[0]
                            )
                            if repair_solutions
                            else None
                        ),
                }
            )

        # -------------------------------------------------
        # Fully unparsed rows
        # -------------------------------------------------

        if len(strict_solutions) == 0:

            blockers = find_blocking_spans(
                raw_value=raw_investors,
                canonical_names=canonical_names,
                max_name_chunks=max_name_chunks,
            )

            if blockers:

                for blocker in blockers:

                    unparsed_rows.append(
                        {
                            "funding_round_id":
                                funding_round_id,

                            "investors":
                                raw_investors,

                            "possible_blocking_text":
                                blocker[
                                    "blocking_text"
                                ],

                            "removed_chunk_count":
                                blocker[
                                    "removed_chunk_count"
                                ],

                            "remaining_solution":
                                blocker[
                                    "remaining_solution"
                                ],
                        }
                    )

            else:

                unparsed_rows.append(
                    {
                        "funding_round_id":
                            funding_round_id,

                        "investors":
                            raw_investors,

                        "possible_blocking_text":
                            None,

                        "removed_chunk_count":
                            None,

                        "remaining_solution":
                            None,
                    }
                )

    # -----------------------------------------------------
    # 6. Save empty-chunk audit
    # -----------------------------------------------------

    empty_df = pl.DataFrame(
        empty_chunk_rows
    )

    empty_df.write_csv(
        EMPTY_CHUNK_OUTPUT
    )

    print("\n" + "-" * 80)
    print("EMPTY-CHUNK AUDIT")
    print("-" * 80)

    print(
        f"Rows containing empty chunks: "
        f"{empty_df.height:,}"
    )

    if empty_df.height:

        summary = (
            empty_df
            .group_by(
                "strict_solution_count",
                "repair_solution_count",
            )
            .len()
            .sort(
                [
                    "strict_solution_count",
                    "repair_solution_count",
                ]
            )
        )

        print("\nEmpty-chunk outcome summary:")
        print(summary)

        print("\nFirst 20 empty-chunk rows:")
        print(
            empty_df.head(20)
        )

    # -----------------------------------------------------
    # 7. Save unparsed diagnostic
    # -----------------------------------------------------

    unparsed_df = pl.DataFrame(
        unparsed_rows
    )

    unparsed_df.write_csv(
        UNPARSED_DIAGNOSTIC_OUTPUT
    )

    print("\n" + "-" * 80)
    print("UNPARSED-ROW DIAGNOSTIC")
    print("-" * 80)

    print(
        f"Diagnostic rows generated: "
        f"{unparsed_df.height:,}"
    )

    if unparsed_df.height:

        blocker_counts = (
            unparsed_df
            .filter(
                pl.col(
                    "possible_blocking_text"
                ).is_not_null()
            )
            .group_by(
                "possible_blocking_text"
            )
            .len()
            .sort(
                "len",
                descending=True,
            )
        )

        print(
            "\nPossible blocking-text frequencies:"
        )

        print(blocker_counts)

        print(
            "\nFull unparsed diagnostic:"
        )

        print(unparsed_df)

    # -----------------------------------------------------
    # 8. Finish
    # -----------------------------------------------------

    print("\nSaved:")
    print(EMPTY_CHUNK_OUTPUT)
    print(UNPARSED_DIAGNOSTIC_OUTPUT)

    print("\n" + "=" * 80)
    print("EDGE-CASE AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()