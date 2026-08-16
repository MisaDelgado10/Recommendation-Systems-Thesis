from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import re

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

INTERIM_DIR = Path("data/interim")

UNPARSED_OUTPUT = (
    INTERIM_DIR / "investor_parsing_unparsed_examples.csv"
)

AMBIGUOUS_OUTPUT = (
    INTERIM_DIR / "investor_parsing_ambiguous_examples.csv"
)

AMBIGUOUS_ID_OUTPUT = (
    INTERIM_DIR / "investor_name_id_ambiguities.csv"
)


# =========================================================
# Conservative string normalization
# =========================================================

COMMA_SPACING = re.compile(r"\s*,\s*")


def normalize_name(value: str) -> str:
    """
    Conservative normalization.

    We only:
    - remove leading/trailing whitespace
    - normalize whitespace around commas

    We DO NOT:
    - lowercase
    - remove punctuation
    - remove accents
    - fuzzy-match names
    """

    value = value.strip()

    return COMMA_SPACING.sub(",", value)


# =========================================================
# Dictionary segmentation
# =========================================================

def find_segmentations(
    chunks: tuple[str, ...],
    canonical_names: set[str],
    max_name_chunks: int,
    max_solutions: int = 2,
) -> list[tuple[str, ...]]:
    """
    Find up to `max_solutions` ways of segmenting comma-separated
    chunks into exact canonical Crunchbase investor names.

    Returning:
        []               -> no valid segmentation
        [solution]       -> unique segmentation
        [solution1, ...] -> ambiguous segmentation
    """

    n = len(chunks)

    @lru_cache(maxsize=None)
    def solve(position: int) -> tuple[tuple[str, ...], ...]:

        if position == n:
            return ((),)

        solutions: list[tuple[str, ...]] = []

        max_end = min(
            n,
            position + max_name_chunks,
        )

        for end in range(position + 1, max_end + 1):

            candidate = ",".join(
                chunks[position:end]
            )

            if candidate not in canonical_names:
                continue

            tails = solve(end)

            for tail in tails:
                solutions.append(
                    (candidate,) + tail
                )

                if len(solutions) >= max_solutions:
                    return tuple(solutions)

        return tuple(solutions)

    return list(solve(0))


def parse_investor_string(
    raw_value: str,
    canonical_names: set[str],
    max_name_chunks: int,
) -> dict:

    normalized = normalize_name(raw_value)

    chunks = tuple(
        part.strip()
        for part in normalized.split(",")
    )

    contains_empty_chunk = any(
        chunk == ""
        for chunk in chunks
    )

    # -----------------------------------------------------
    # First: strict parsing.
    # We do NOT silently remove empty chunks.
    # -----------------------------------------------------

    strict_solutions = find_segmentations(
        chunks=chunks,
        canonical_names=canonical_names,
        max_name_chunks=max_name_chunks,
    )

    # -----------------------------------------------------
    # Diagnostic only:
    # could an empty-chunk row become parseable if the
    # empty fragments were removed?
    #
    # We DO NOT accept this as final parsing yet.
    # -----------------------------------------------------

    repair_solutions: list[tuple[str, ...]] = []

    if contains_empty_chunk:

        non_empty_chunks = tuple(
            chunk
            for chunk in chunks
            if chunk != ""
        )

        repair_solutions = find_segmentations(
            chunks=non_empty_chunks,
            canonical_names=canonical_names,
            max_name_chunks=max_name_chunks,
        )

    return {
        "normalized": normalized,
        "contains_empty_chunk": contains_empty_chunk,
        "strict_solutions": strict_solutions,
        "repair_solutions": repair_solutions,
    }


# =========================================================
# Main audit
# =========================================================

def main() -> None:

    print("=" * 80)
    print("PHASE 1.10 — CANONICAL INVESTOR PARSING AUDIT")
    print("=" * 80)

    INTERIM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # 1. Verify files
    # -----------------------------------------------------

    for path in [FUNDING_PATH, INVESTOR_PATH]:

        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n"
                f"{path.resolve()}"
            )

    # -----------------------------------------------------
    # 2. Load investor master
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

    # -----------------------------------------------------
    # 3. Construct:
    #
    # canonical name -> Crunchbase IDs
    # -----------------------------------------------------

    name_to_ids: dict[str, list[str]] = defaultdict(list)

    for investor_id, investor_name in investor_df.iter_rows():

        canonical_name = normalize_name(
            investor_name
        )

        name_to_ids[canonical_name].append(
            investor_id
        )

    canonical_names = set(
        name_to_ids.keys()
    )

    print(
        "\nCanonical investor names:",
        f"{len(canonical_names):,}",
    )

    # -----------------------------------------------------
    # 4. Determine how many comma chunks the longest
    # canonical investor name contains.
    #
    # This lets the parser avoid testing impossible spans.
    # -----------------------------------------------------

    max_name_chunks = max(
        name.count(",") + 1
        for name in canonical_names
    )

    print(
        "Maximum comma-separated chunks "
        "inside one canonical investor name:",
        max_name_chunks,
    )

    # -----------------------------------------------------
    # 5. Identify names mapping to >1 investor ID
    # -----------------------------------------------------

    ambiguous_name_to_ids = {
        name: ids
        for name, ids in name_to_ids.items()
        if len(ids) > 1
    }

    print(
        "Canonical names mapping to multiple IDs:",
        f"{len(ambiguous_name_to_ids):,}",
    )

    # -----------------------------------------------------
    # 6. Load only funding ID + investors
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

    print(
        "\nFunding rows with investor data:",
        f"{funding_df.height:,}",
    )

    # -----------------------------------------------------
    # 7. Counters
    # -----------------------------------------------------

    unique_parse_rows = 0
    ambiguous_parse_rows = 0
    unparsed_rows = 0

    rows_with_empty_chunks = 0
    repairable_empty_chunk_rows = 0

    total_parsed_investor_mentions = 0

    unique_id_mentions = 0
    ambiguous_id_mentions = 0

    ambiguous_id_name_frequency = Counter()

    unparsed_examples = []
    ambiguous_examples = []

    # Cache identical investor strings so we don't solve
    # the same segmentation problem repeatedly.
    parsing_cache: dict[str, dict] = {}

    # -----------------------------------------------------
    # 8. Parse every funding investor string
    # -----------------------------------------------------

    for funding_round_id, raw_investors in funding_df.iter_rows():

        cache_key = raw_investors

        if cache_key not in parsing_cache:

            parsing_cache[cache_key] = (
                parse_investor_string(
                    raw_value=raw_investors,
                    canonical_names=canonical_names,
                    max_name_chunks=max_name_chunks,
                )
            )

        result = parsing_cache[cache_key]

        strict_solutions = result[
            "strict_solutions"
        ]

        repair_solutions = result[
            "repair_solutions"
        ]

        # -------------------------------------------------
        # Empty chunk audit
        # -------------------------------------------------

        if result["contains_empty_chunk"]:

            rows_with_empty_chunks += 1

            if len(repair_solutions) == 1:
                repairable_empty_chunk_rows += 1

        # -------------------------------------------------
        # No segmentation found
        # -------------------------------------------------

        if len(strict_solutions) == 0:

            unparsed_rows += 1

            if len(unparsed_examples) < 100:

                unparsed_examples.append(
                    {
                        "funding_round_id":
                            funding_round_id,

                        "investors":
                            raw_investors,

                        "normalized":
                            result["normalized"],

                        "contains_empty_chunk":
                            result[
                                "contains_empty_chunk"
                            ],

                        "repairable_after_empty_removal":
                            len(repair_solutions) == 1,
                    }
                )

            continue

        # -------------------------------------------------
        # More than one possible segmentation
        # -------------------------------------------------

        if len(strict_solutions) > 1:

            ambiguous_parse_rows += 1

            if len(ambiguous_examples) < 100:

                ambiguous_examples.append(
                    {
                        "funding_round_id":
                            funding_round_id,

                        "investors":
                            raw_investors,

                        "solution_1":
                            " || ".join(
                                strict_solutions[0]
                            ),

                        "solution_2":
                            " || ".join(
                                strict_solutions[1]
                            ),
                    }
                )

            continue

        # -------------------------------------------------
        # Exactly one valid segmentation
        # -------------------------------------------------

        unique_parse_rows += 1

        parsed_names = strict_solutions[0]

        total_parsed_investor_mentions += (
            len(parsed_names)
        )

        # -------------------------------------------------
        # Check whether names map uniquely to IDs
        # -------------------------------------------------

        for name in parsed_names:

            ids = name_to_ids[name]

            if len(ids) == 1:

                unique_id_mentions += 1

            else:

                ambiguous_id_mentions += 1

                ambiguous_id_name_frequency[
                    name
                ] += 1

    # -----------------------------------------------------
    # 9. Print row-level parsing summary
    # -----------------------------------------------------

    print("\n" + "-" * 80)
    print("ROW-LEVEL PARSING SUMMARY")
    print("-" * 80)

    print(
        f"Unique valid segmentations:     "
        f"{unique_parse_rows:,}"
    )

    print(
        f"Ambiguous segmentations:        "
        f"{ambiguous_parse_rows:,}"
    )

    print(
        f"No valid segmentation:          "
        f"{unparsed_rows:,}"
    )

    print(
        f"Rows containing empty chunks:   "
        f"{rows_with_empty_chunks:,}"
    )

    print(
        f"Repairable after empty removal: "
        f"{repairable_empty_chunk_rows:,}"
    )

    # -----------------------------------------------------
    # 10. Print entity-ID resolution summary
    # -----------------------------------------------------

    print("\n" + "-" * 80)
    print("INVESTOR-MENTION ID RESOLUTION")
    print("-" * 80)

    print(
        f"Parsed investor mentions:       "
        f"{total_parsed_investor_mentions:,}"
    )

    print(
        f"Mentions with unique ID:         "
        f"{unique_id_mentions:,}"
    )

    print(
        f"Mentions with ambiguous IDs:     "
        f"{ambiguous_id_mentions:,}"
    )

    # -----------------------------------------------------
    # 11. Show most frequent ambiguous-ID names
    # -----------------------------------------------------

    print("\nMost frequent parsed names "
          "that map to multiple investor IDs:")

    for name, count in (
        ambiguous_id_name_frequency
        .most_common(30)
    ):

        print(
            f"{count:>8,}  |  "
            f"{len(name_to_ids[name]):>3} IDs  |  "
            f"{name}"
        )

    # -----------------------------------------------------
    # 12. Save unresolved parsing examples
    # -----------------------------------------------------

    if unparsed_examples:

        pl.DataFrame(
            unparsed_examples
        ).write_csv(
            UNPARSED_OUTPUT
        )

        print(
            "\nSaved unparsed examples to:"
        )

        print(
            UNPARSED_OUTPUT
        )

    # -----------------------------------------------------
    # 13. Save ambiguous segmentation examples
    # -----------------------------------------------------

    if ambiguous_examples:

        pl.DataFrame(
            ambiguous_examples
        ).write_csv(
            AMBIGUOUS_OUTPUT
        )

        print(
            "\nSaved ambiguous segmentation "
            "examples to:"
        )

        print(
            AMBIGUOUS_OUTPUT
        )

    # -----------------------------------------------------
    # 14. Save duplicated canonical names + IDs
    # -----------------------------------------------------

    ambiguity_rows = []

    for name, ids in ambiguous_name_to_ids.items():

        ambiguity_rows.append(
            {
                "investor_name": name,
                "number_of_ids": len(ids),
                "investor_ids": "|".join(ids),
            }
        )

    (
        pl.DataFrame(
            ambiguity_rows
        )
        .sort(
            "number_of_ids",
            descending=True,
        )
        .write_csv(
            AMBIGUOUS_ID_OUTPUT
        )
    )

    print(
        "\nSaved canonical name → multiple-ID "
        "mapping to:"
    )

    print(
        AMBIGUOUS_ID_OUTPUT
    )

    # -----------------------------------------------------
    # 15. Finish
    # -----------------------------------------------------

    print("\n" + "=" * 80)
    print("CANONICAL PARSING AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()