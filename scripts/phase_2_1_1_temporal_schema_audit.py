from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


# =============================================================================
# PHASE 2.1.1 — TEMPORAL SCHEMA AND INPUT INTEGRITY AUDIT
# =============================================================================

INPUT_PATH = Path("data/processed/interactions.parquet")

EXPECTED_ROWS = 1_208_051


def print_separator(char="=", width=80):
    print(char * width)


def find_candidate_columns(columns, keywords):
    """
    Return columns whose names contain at least one supplied keyword.

    This is used only for schema discovery.
    It does NOT automatically select any column for modeling.
    """
    matches = []

    for col in columns:
        lower_col = col.lower()

        if any(keyword in lower_col for keyword in keywords):
            matches.append(col)

    return matches


def summarize_candidate_column(df, col):
    print(f"\nColumn: {col}")
    print(f"  dtype:        {df[col].dtype}")
    print(f"  non-null:     {df[col].notna().sum():,}")
    print(f"  null:         {df[col].isna().sum():,}")
    print(f"  unique:       {df[col].nunique(dropna=True):,}")

    sample = df[col].dropna().head(5).tolist()
    print(f"  sample:       {sample}")


def summarize_date_candidate(df, col):
    """
    Attempt a diagnostic datetime conversion.

    This does not modify the dataframe and does not establish that the column
    should be used as the canonical event date.
    """
    parsed = pd.to_datetime(df[col], errors="coerce")

    original_non_null = df[col].notna().sum()
    parsed_non_null = parsed.notna().sum()

    print(f"\nTemporal candidate: {col}")
    print(f"  original dtype:       {df[col].dtype}")
    print(f"  original non-null:    {original_non_null:,}")
    print(f"  parseable as date:    {parsed_non_null:,}")

    if original_non_null > 0:
        parse_rate = parsed_non_null / original_non_null
        print(f"  parse success rate:   {parse_rate:.6%}")

    if parsed_non_null > 0:
        print(f"  minimum date:         {parsed.min()}")
        print(f"  maximum date:         {parsed.max()}")

        years = parsed.dt.year

        print(f"  minimum year:         {years.min()}")
        print(f"  maximum year:         {years.max()}")

        print("  first parsed values:")
        print(parsed.dropna().head(5).to_string(index=False))


def main():

    print_separator()
    print("PHASE 2.1.1 — TEMPORAL SCHEMA AND INPUT INTEGRITY AUDIT")
    print_separator()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Canonical interaction file not found:\n{INPUT_PATH}"
        )

    # -------------------------------------------------------------------------
    # 1. Inspect parquet schema before analysis
    # -------------------------------------------------------------------------

    parquet_file = pq.ParquetFile(INPUT_PATH)

    print("\nPARQUET METADATA")
    print("-" * 80)

    print(f"Path:                 {INPUT_PATH}")
    print(f"Parquet row groups:   {parquet_file.num_row_groups:,}")

    metadata_rows = parquet_file.metadata.num_rows

    print(f"Parquet rows:         {metadata_rows:,}")
    print(f"Expected Phase-1 rows:{EXPECTED_ROWS:,}")

    if metadata_rows == EXPECTED_ROWS:
        print("Row-count check:      PASS")
    else:
        print("Row-count check:      WARNING — differs from Phase-1 closure statistics")

    print("\nPARQUET SCHEMA")
    print("-" * 80)

    for i, field in enumerate(parquet_file.schema_arrow):
        print(f"{i:>3}. {field.name:<40} {field.type}")

    # -------------------------------------------------------------------------
    # 2. Load canonical interactions
    # -------------------------------------------------------------------------

    print("\nLoading canonical interactions...")

    df = pd.read_parquet(INPUT_PATH)

    print("Loaded.")

    print("\nDATAFRAME INTEGRITY")
    print("-" * 80)

    print(f"Rows:                 {len(df):,}")
    print(f"Columns:              {len(df.columns):,}")

    if len(df) == EXPECTED_ROWS:
        print("Phase-1 row count:    PASS")
    else:
        print("Phase-1 row count:    WARNING")

    # -------------------------------------------------------------------------
    # 3. Verify known canonical interaction identifier
    # -------------------------------------------------------------------------

    if "interaction_id" in df.columns:

        duplicate_interaction_ids = df["interaction_id"].duplicated().sum()
        unique_interactions = df["interaction_id"].nunique(dropna=True)
        missing_interaction_ids = df["interaction_id"].isna().sum()

        print(f"\ninteraction_id:")
        print(f"  unique:             {unique_interactions:,}")
        print(f"  missing:            {missing_interaction_ids:,}")
        print(f"  duplicate IDs:      {duplicate_interaction_ids:,}")

        if (
            unique_interactions == EXPECTED_ROWS
            and missing_interaction_ids == 0
            and duplicate_interaction_ids == 0
        ):
            print("  integrity check:    PASS")
        else:
            print("  integrity check:    WARNING")
    else:
        print("\nWARNING: interaction_id not found.")

    # -------------------------------------------------------------------------
    # 4. Full column audit
    # -------------------------------------------------------------------------

    print("\nCOLUMN-LEVEL SUMMARY")
    print("-" * 80)

    summary = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[col].dtype) for col in df.columns],
            "non_null": [df[col].notna().sum() for col in df.columns],
            "null": [df[col].isna().sum() for col in df.columns],
            "unique": [df[col].nunique(dropna=True) for col in df.columns],
        }
    )

    summary["null_pct"] = summary["null"] / len(df) * 100

    print(
        summary.to_string(
            index=False,
            formatters={"null_pct": lambda x: f"{x:.4f}%"}
        )
    )

    # -------------------------------------------------------------------------
    # 5. Discover temporal candidates
    # -------------------------------------------------------------------------

    temporal_keywords = [
        "date",
        "time",
        "year",
        "month",
        "announced",
        "closed",
        "funded",
        "created",
        "updated",
    ]

    temporal_candidates = find_candidate_columns(
        df.columns,
        temporal_keywords,
    )

    print("\nTEMPORAL COLUMN CANDIDATES")
    print("-" * 80)

    if temporal_candidates:
        for col in temporal_candidates:
            print(f"- {col}")
    else:
        print("No temporal candidates detected from column names.")

    # -------------------------------------------------------------------------
    # 6. Discover entity-ID candidates
    # -------------------------------------------------------------------------

    investor_candidates = find_candidate_columns(
        df.columns,
        ["investor"],
    )

    startup_candidates = find_candidate_columns(
        df.columns,
        [
            "startup",
            "company",
            "organization",
            "organisation",
            "org_",
        ],
    )

    funding_round_candidates = find_candidate_columns(
        df.columns,
        [
            "funding_round",
            "round_id",
        ],
    )

    print("\nINVESTOR COLUMN CANDIDATES")
    print("-" * 80)
    for col in investor_candidates:
        print(f"- {col}")

    print("\nSTARTUP / ORGANIZATION COLUMN CANDIDATES")
    print("-" * 80)
    for col in startup_candidates:
        print(f"- {col}")

    print("\nFUNDING-ROUND COLUMN CANDIDATES")
    print("-" * 80)
    for col in funding_round_candidates:
        print(f"- {col}")

    # -------------------------------------------------------------------------
    # 7. Inspect candidate entity columns
    # -------------------------------------------------------------------------

    candidate_entity_columns = list(
        dict.fromkeys(
            investor_candidates
            + startup_candidates
            + funding_round_candidates
        )
    )

    print("\nENTITY-CANDIDATE DIAGNOSTICS")
    print("-" * 80)

    for col in candidate_entity_columns:
        summarize_candidate_column(df, col)

    # -------------------------------------------------------------------------
    # 8. Diagnostic parsing of candidate temporal columns
    # -------------------------------------------------------------------------

    print("\nTEMPORAL-CANDIDATE DIAGNOSTICS")
    print("-" * 80)

    for col in temporal_candidates:
        summarize_date_candidate(df, col)

    # -------------------------------------------------------------------------
    # 9. Final audit summary
    # -------------------------------------------------------------------------

    print("\n")
    print_separator()
    print("PHASE 2.1.1 AUDIT COMPLETE")
    print_separator()

    print(
        """
No temporal field has been selected yet.
No dates have been removed.
No cutoff has been selected.
No temporal segments have been constructed.
No minimum sequence length has been imposed.
No investment types have been filtered.

The next step is to identify, from this audit, the exact columns representing:

1. canonical investment-event date
2. canonical investor ID
3. canonical startup/company ID
4. canonical funding-round ID

Only after those fields are verified should Phase 2.1.2 construct the
year/month/quarter temporal distributions.
"""
    )


if __name__ == "__main__":
    main()