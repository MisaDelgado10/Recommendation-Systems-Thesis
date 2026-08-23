from pathlib import Path
import json
import pandas as pd


# =============================================================================
# PHASE 4.1.3a — DESCRIPTION-SOURCE SCHEMA DISCOVERY
# =============================================================================

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/experimental/phase_4/audits")
OUT_DIR.mkdir(parents=True, exist_ok=True)

INVESTOR_PATH = RAW_DIR / "CRUNCHBASE_investor_20260531_333726.csv"

# Phase 1/3 established that the Company registry is distributed across
# data/raw/companies*.csv shards.
COMPANY_PATHS = sorted(RAW_DIR.glob("companies*.csv"))


def banner(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def inspect_csv_schema(path: Path, source_name: str) -> pd.DataFrame:
    """
    Read only a small sample.

    This subphase is schema discovery only. We deliberately do NOT calculate
    canonical-universe coverage yet.
    """
    sample = pd.read_csv(
        path,
        nrows=5,
        low_memory=False,
    )

    rows = []

    for col in sample.columns:
        values = (
            sample[col]
            .dropna()
            .astype(str)
            .head(3)
            .tolist()
        )

        rows.append(
            {
                "source": source_name,
                "column": col,
                "sample_dtype": str(sample[col].dtype),
                "sample_values": " || ".join(values),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# 1. FILE PRESENCE
# =============================================================================

banner("PHASE 4.1.3a — DESCRIPTION-SOURCE SCHEMA DISCOVERY")

print("\nAUTHORITATIVE SOURCE FILES")
print("-" * 100)

print(f"Investor file exists: {INVESTOR_PATH.exists()}")
print(f"Investor path:        {INVESTOR_PATH}")

print(f"\nCompany shards found: {len(COMPANY_PATHS)}")

for path in COMPANY_PATHS:
    print(f"  - {path}")


if not INVESTOR_PATH.exists():
    raise FileNotFoundError(
        f"Frozen Phase-3 Investor registry not found: {INVESTOR_PATH}"
    )

if not COMPANY_PATHS:
    raise FileNotFoundError(
        "No data/raw/companies*.csv files were found."
    )


# =============================================================================
# 2. INVESTOR SCHEMA
# =============================================================================

banner("INVESTOR RAW SCHEMA")

investor_schema = inspect_csv_schema(
    INVESTOR_PATH,
    source_name="investor",
)

print(f"Columns: {len(investor_schema)}")
print()

for _, row in investor_schema.iterrows():
    print(f"{row['column']:<45} {row['sample_dtype']}")


# =============================================================================
# 3. COMPANY SCHEMA CONSISTENCY
# =============================================================================

banner("COMPANY SHARD SCHEMA CONSISTENCY")

company_column_sets = {}

for path in COMPANY_PATHS:
    header = pd.read_csv(
        path,
        nrows=0,
    )

    company_column_sets[path.name] = list(header.columns)

reference_name = COMPANY_PATHS[0].name
reference_columns = company_column_sets[reference_name]

schema_mismatch_count = 0

for filename, columns in company_column_sets.items():
    if columns != reference_columns:
        schema_mismatch_count += 1

        print(f"\nSCHEMA MISMATCH: {filename}")

        missing = sorted(set(reference_columns) - set(columns))
        extra = sorted(set(columns) - set(reference_columns))

        print(f"  Missing relative to reference: {missing}")
        print(f"  Extra relative to reference:   {extra}")

print()
print(f"Reference shard:          {reference_name}")
print(f"Reference column count:   {len(reference_columns)}")
print(f"Schema-mismatching shards:{schema_mismatch_count}")


# =============================================================================
# 4. COMPANY REPRESENTATIVE SCHEMA
# =============================================================================

banner("COMPANY RAW SCHEMA")

company_schema = inspect_csv_schema(
    COMPANY_PATHS[0],
    source_name="company",
)

print(f"Columns: {len(company_schema)}")
print()

for _, row in company_schema.iterrows():
    print(f"{row['column']:<45} {row['sample_dtype']}")


# =============================================================================
# 5. POTENTIAL DESCRIPTION-RELATED COLUMN NAMES
# =============================================================================

banner("NAME-BASED CANDIDATE FIELD DISCOVERY")

# IMPORTANT:
# These keywords do NOT select features.
# They only help us locate columns worth auditing in the next subphase.

candidate_keywords = [
    "description",
    "short_description",
    "overview",
    "summary",
    "category",
    "categories",
    "industry",
    "industries",
    "sector",
    "type",
    "stage",
    "location",
    "city",
    "region",
    "country",
    "market",
    "tag",
    "label",
    "focus",
    "investment",
]


def find_candidate_columns(columns):
    matches = []

    for col in columns:
        lower = col.lower()

        matched_keywords = [
            keyword
            for keyword in candidate_keywords
            if keyword in lower
        ]

        if matched_keywords:
            matches.append(
                {
                    "column": col,
                    "matched_keywords": ",".join(matched_keywords),
                }
            )

    return pd.DataFrame(matches)


investor_candidates = find_candidate_columns(
    investor_schema["column"].tolist()
)
investor_candidates.insert(0, "source", "investor")

company_candidates = find_candidate_columns(
    company_schema["column"].tolist()
)
company_candidates.insert(0, "source", "company")

candidate_fields = pd.concat(
    [investor_candidates, company_candidates],
    ignore_index=True,
)

print("\nPossible Investor fields:")
print("-" * 100)

if investor_candidates.empty:
    print("None discovered from column names.")
else:
    print(
        investor_candidates[
            ["column", "matched_keywords"]
        ].to_string(index=False)
    )


print("\nPossible Company fields:")
print("-" * 100)

if company_candidates.empty:
    print("None discovered from column names.")
else:
    print(
        company_candidates[
            ["column", "matched_keywords"]
        ].to_string(index=False)
    )


# =============================================================================
# 6. SAMPLE VALUES FOR POSSIBLE CANDIDATES
# =============================================================================

banner("SAMPLE VALUES FOR POSSIBLE DESCRIPTION FIELDS")


def print_candidate_samples(
    path: Path,
    source_name: str,
    candidate_df: pd.DataFrame,
):
    if candidate_df.empty:
        print(f"\n{source_name}: no candidate columns.")
        return

    cols = candidate_df["column"].tolist()

    df = pd.read_csv(
        path,
        usecols=cols,
        nrows=10,
        low_memory=False,
    )

    print(f"\n{source_name.upper()}")
    print("-" * 100)

    for col in cols:
        print(f"\nCOLUMN: {col}")
        values = df[col].dropna().astype(str).head(5)

        if values.empty:
            print("  <no non-null value in sample>")
        else:
            for value in values:
                # Keep terminal output readable.
                if len(value) > 300:
                    value = value[:300] + "..."

                print(f"  {value}")


print_candidate_samples(
    INVESTOR_PATH,
    "investor",
    investor_candidates,
)

print_candidate_samples(
    COMPANY_PATHS[0],
    "company",
    company_candidates,
)


# =============================================================================
# 7. SAVE AUDIT OUTPUTS
# =============================================================================

schema_inventory = pd.concat(
    [
        investor_schema,
        company_schema,
    ],
    ignore_index=True,
)

schema_inventory_path = (
    OUT_DIR / "description_source_schema_inventory.csv"
)

candidate_fields_path = (
    OUT_DIR / "description_candidate_field_inventory.csv"
)

schema_inventory.to_csv(
    schema_inventory_path,
    index=False,
)

candidate_fields.to_csv(
    candidate_fields_path,
    index=False,
)


metadata = {
    "phase": "4.1.3a",
    "purpose": "description-source schema discovery",
    "investor_source": str(INVESTOR_PATH),
    "company_glob": "data/raw/companies*.csv",
    "company_shards_found": len(COMPANY_PATHS),
    "investor_column_count": int(len(investor_schema)),
    "company_column_count": int(len(company_schema)),
    "company_schema_mismatch_count": int(schema_mismatch_count),
    "candidate_field_selection_status": "NOT_SELECTED",
}

metadata_path = (
    OUT_DIR / "description_source_schema_metadata.json"
)

with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False,
    )


# =============================================================================
# 8. FINAL SUMMARY
# =============================================================================

banner("PHASE 4.1.3a SUMMARY")

print(f"Investor columns discovered:   {len(investor_schema):,}")
print(f"Company columns discovered:    {len(company_schema):,}")
print(f"Company shards inspected:      {len(COMPANY_PATHS):,}")
print(f"Company schema mismatches:     {schema_mismatch_count:,}")

print(
    f"Name-based candidate fields:   "
    f"{len(candidate_fields):,}"
)

print("\nOutputs:")
print(f"  {schema_inventory_path}")
print(f"  {candidate_fields_path}")
print(f"  {metadata_path}")

print("\nIMPORTANT:")
print(
    "No description field has been selected in this subphase."
)
print(
    "The next step will measure canonical-role coverage and "
    "semantic suitability."
)

print("\nPHASE 4.1.3a STATUS: COMPLETE")