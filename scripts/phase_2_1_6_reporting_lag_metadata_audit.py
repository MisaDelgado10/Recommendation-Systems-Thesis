from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


# =============================================================================
# PHASE 2.1.6 — REPORTING-LAG / RIGHT-CENSORING AUDIT
# SUBPHASE 2.1.6.1 — METADATA AVAILABILITY
# =============================================================================

DATA_ROOT = Path("data")


FUNDING_KEYWORDS = [
    "funding",
    "round",
]


TEMPORAL_METADATA_KEYWORDS = [
    "announced",
    "created",
    "updated",
    "modified",
    "published",
    "date",
    "time",
    "timestamp",
]


ID_KEYWORDS = [
    "uuid",
    "id",
    "funding_round",
]


def separator(char="=", width=100):
    print(char * width)


def looks_like_funding_file(path):
    name = path.name.lower()

    return (
        any(keyword in name for keyword in FUNDING_KEYWORDS)
        and path.suffix.lower() in {".csv", ".parquet"}
    )


def get_columns(path):

    suffix = path.suffix.lower()

    if suffix == ".parquet":
        parquet = pq.ParquetFile(path)

        return [
            field.name
            for field in parquet.schema_arrow
        ]

    if suffix == ".csv":
        return list(
            pd.read_csv(
                path,
                nrows=0,
                low_memory=False,
            ).columns
        )

    return []


def classify_columns(columns):

    temporal = []
    identifiers = []

    for col in columns:

        lower = col.lower()

        if any(
            keyword in lower
            for keyword in TEMPORAL_METADATA_KEYWORDS
        ):
            temporal.append(col)

        if any(
            keyword in lower
            for keyword in ID_KEYWORDS
        ):
            identifiers.append(col)

    return temporal, identifiers


def main():

    separator()
    print(
        "PHASE 2.1.6.1 — "
        "REPORTING-LAG METADATA AVAILABILITY AUDIT"
    )
    separator()

    print(f"\nScanning under: {DATA_ROOT.resolve()}")

    candidate_files = sorted(
        path
        for path in DATA_ROOT.rglob("*")
        if path.is_file()
        and looks_like_funding_file(path)
    )

    print(
        f"\nCandidate funding-related files found: "
        f"{len(candidate_files):,}"
    )

    if not candidate_files:

        print(
            "\nNo funding-related CSV or Parquet files "
            "were detected."
        )

        return

    for i, path in enumerate(candidate_files, start=1):

        separator("-")
        print(f"CANDIDATE FILE {i}")
        separator("-")

        print(f"Path: {path}")

        try:
            columns = get_columns(path)

        except Exception as exc:

            print(
                f"Could not inspect schema: "
                f"{type(exc).__name__}: {exc}"
            )

            continue

        temporal, identifiers = classify_columns(
            columns
        )

        print(
            f"\nTotal columns: {len(columns):,}"
        )

        print("\nFULL SCHEMA:")
        for col in columns:
            print(f"- {col}")

        print(
            "\nPOTENTIAL TEMPORAL / "
            "REPORTING-METADATA COLUMNS:"
        )

        if temporal:
            for col in temporal:
                print(f"- {col}")
        else:
            print("- NONE DETECTED")

        print("\nPOTENTIAL IDENTIFIER COLUMNS:")

        if identifiers:
            for col in identifiers:
                print(f"- {col}")
        else:
            print("- NONE DETECTED")

    separator()
    print("PHASE 2.1.6.1 AUDIT COMPLETE")
    separator()

    print(
        """
No reporting lag has been estimated.

No assumption has been made that created/updated
timestamps exist.

No recent month has been removed.

The purpose of this audit is only to determine whether
the available Crunchbase files contain metadata capable
of measuring reporting latency.
"""
    )


if __name__ == "__main__":
    main()