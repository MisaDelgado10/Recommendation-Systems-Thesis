from __future__ import annotations

from functools import lru_cache
import re


# =========================================================
# Conservative normalization
# =========================================================

COMMA_SPACING = re.compile(r"\s*,\s*")


def normalize_name(value: str) -> str:
    """
    Conservative normalization.

    Only:
    - strip surrounding whitespace
    - normalize whitespace around commas

    We deliberately do NOT:
    - lowercase
    - remove punctuation
    - remove accents
    - remove corporate suffixes
    - perform fuzzy matching
    """

    value = value.strip()

    return COMMA_SPACING.sub(",", value)


# =========================================================
# Exact dictionary segmentation
# =========================================================

def find_segmentations(
    chunks: tuple[str, ...],
    canonical_names: set[str],
    max_name_chunks: int,
    max_solutions: int = 2,
) -> list[tuple[str, ...]]:
    """
    Segment comma-separated chunks into exact canonical
    investor names.

    Returns at most `max_solutions`.

    []          -> no valid segmentation
    [solution]  -> unique segmentation
    [a, b]      -> ambiguous segmentation
    """

    n = len(chunks)

    @lru_cache(maxsize=None)
    def solve(
        position: int,
    ) -> tuple[tuple[str, ...], ...]:

        if position == n:
            return ((),)

        solutions: list[tuple[str, ...]] = []

        max_end = min(
            n,
            position + max_name_chunks,
        )

        for end in range(
            position + 1,
            max_end + 1,
        ):

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


# =========================================================
# Full investor-field parser
# =========================================================

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

    strict_solutions = find_segmentations(
        chunks=chunks,
        canonical_names=canonical_names,
        max_name_chunks=max_name_chunks,
    )

    repair_solutions = []

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
        "chunks": chunks,
        "contains_empty_chunk": contains_empty_chunk,
        "strict_solutions": strict_solutions,
        "repair_solutions": repair_solutions,
    }


# =========================================================
# Partial recovery for exactly one unknown source token
# =========================================================

def find_single_unknown_recovery(
    raw_value: str,
    canonical_names: set[str],
    max_name_chunks: int,
) -> dict | None:
    """
    Diagnostic/conservative recovery.

    Find exactly one contiguous source span that is NOT in
    the canonical investor dictionary while both the left and
    right sides can be uniquely segmented into canonical names.

    We preserve the unknown span as an unmatched source token.
    We do NOT assign it an investor ID.
    """

    normalized = normalize_name(raw_value)

    chunks = tuple(
        part.strip()
        for part in normalized.split(",")
    )

    candidates = []

    for width in range(
        1,
        min(max_name_chunks, len(chunks)) + 1,
    ):

        for start in range(
            len(chunks) - width + 1
        ):

            end = start + width

            unknown = ",".join(
                chunks[start:end]
            )

            if not unknown:
                continue

            if unknown in canonical_names:
                continue

            before_chunks = chunks[:start]
            after_chunks = chunks[end:]

            if before_chunks:

                before = find_segmentations(
                    chunks=before_chunks,
                    canonical_names=canonical_names,
                    max_name_chunks=max_name_chunks,
                )

            else:
                before = [()]

            if after_chunks:

                after = find_segmentations(
                    chunks=after_chunks,
                    canonical_names=canonical_names,
                    max_name_chunks=max_name_chunks,
                )

            else:
                after = [()]

            if (
                len(before) == 1
                and len(after) == 1
            ):

                candidates.append(
                    {
                        "before": before[0],
                        "unknown": unknown,
                        "after": after[0],
                    }
                )

        # Prefer the smallest unknown span.
        if candidates:
            break

    # Important:
    # if several different recoveries are possible,
    # we refuse to choose one.
    if len(candidates) != 1:
        return None

    return candidates[0]