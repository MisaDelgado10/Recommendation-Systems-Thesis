from pathlib import Path
import json

import numpy as np
import pandas as pd


# =============================================================================
# PHASE 4.1.3b — CANONICAL DESCRIPTION-FIELD COVERAGE AUDIT
# =============================================================================

RAW_DIR = Path("data/raw")
PHASE3_GRAPH_DIR = Path("data/experimental/phase_3/graph")
OUT_DIR = Path("data/experimental/phase_4/audits")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NODE_PATH = PHASE3_GRAPH_DIR / "nodes.parquet"

INVESTOR_PATH = RAW_DIR / "CRUNCHBASE_investor_20260531_333726.csv"
COMPANY_PATHS = sorted(RAW_DIR.glob("companies*.csv"))

EXPECTED_INVESTORS = 165_975
EXPECTED_STARTUPS = 311_589
EXPECTED_ROLE_NODES = 477_564

CHUNK_SIZE = 200_000


# =============================================================================
# Candidate fields discovered in Phase 4.1.3a
# =============================================================================

INVESTOR_FIELDS = [
    "description",
    "short_description",
    "company_type",
    "locations",
    "city",
    "region",
    "country",
    "location_groups",
    "categories",
    "category_groups",
    "investor_stage",
    "investor_type",
    "num_investments_funding_rounds",
    "num_lead_investments",
    "num_diversity_spotlight_investments",
    "last_equity_funding_type",
    "last_funding_type",
]

COMPANY_FIELDS = [
    "description",
    "short_description",
    "company_type",
    "last_equity_funding_type",
    "funding_stage",
    "last_funding_type",
    "categories",
    "category_groups",
    "locations",
    "city",
    "region",
    "country",
    "location_groups",
    "hub_tags",
]


# =============================================================================
# Semantic / temporal-risk labels
#
# IMPORTANT:
# These labels do NOT select the final ITRS features.
# They document what each field represents and how cautiously it must be
# treated in the next decision subphase.
# =============================================================================

FIELD_CLASSIFICATION = {
    # -------------------------------------------------------------------------
    # Investor
    # -------------------------------------------------------------------------
    ("investor", "description"): {
        "semantic_class": "text_candidate",
        "temporal_risk": "unversioned_current_snapshot",
        "initial_disposition": "AUDIT",
    },
    ("investor", "short_description"): {
        "semantic_class": "text_candidate",
        "temporal_risk": "unversioned_current_snapshot",
        "initial_disposition": "AUDIT",
    },
    ("investor", "company_type"): {
        "semantic_class": "entity_attribute_candidate",
        "temporal_risk": "unversioned_current_snapshot",
        "initial_disposition": "AUDIT",
    },
    ("investor", "locations"): {
        "semantic_class": "location_attribute",
        "temporal_risk": "unversioned_current_snapshot",
        "initial_disposition": "AUDIT",
    },
    ("investor", "city"): {
        "semantic_class": "location_attribute",
        "temporal_risk": "unversioned_current_snapshot",
        "initial_disposition": "AUDIT",
    },
    ("investor", "region"): {
        "semantic_class": "location_attribute",
        "temporal_risk": "unversioned_current_snapshot",
        "initial_disposition": "AUDIT",
    },
    ("investor", "country"): {
        "semantic_class": "location_attribute",
        "temporal_risk": "unversioned_current_snapshot",
        "initial_disposition": "AUDIT",
    },
    ("investor", "location_groups"): {
        "semantic_class": "location_attribute",
        "temporal_risk": "unversioned_current_snapshot",
        "initial_disposition": "AUDIT",
    },
    ("investor", "categories"): {
        "semantic_class": "label_candidate",
        "temporal_risk": "unversioned_current_snapshot",
        "initial_disposition": "AUDIT",
    },
    ("investor", "category_groups"): {
        "semantic_class": "label_candidate",
        "temporal_risk": "unversioned_current_snapshot",
        "initial_disposition": "AUDIT",
    },
    ("investor", "investor_stage"): {
        "semantic_class": "investment_profile_candidate",
        "temporal_risk": "potentially_history_derived",
        "initial_disposition": "AUDIT_WITH_CAUTION",
    },
    ("investor", "investor_type"): {
        "semantic_class": "label_candidate",
        "temporal_risk": "unversioned_current_snapshot",
        "initial_disposition": "AUDIT",
    },

    # These are explicitly derived from investment behavior.
    ("investor", "num_investments_funding_rounds"): {
        "semantic_class": "interaction_derived_numeric",
        "temporal_risk": "high",
        "initial_disposition": "DO_NOT_USE_AS_DESCRIPTION",
    },
    ("investor", "num_lead_investments"): {
        "semantic_class": "interaction_derived_numeric",
        "temporal_risk": "high",
        "initial_disposition": "DO_NOT_USE_AS_DESCRIPTION",
    },
    ("investor", "num_diversity_spotlight_investments"): {
        "semantic_class": "interaction_derived_numeric",
        "temporal_risk": "high",
        "initial_disposition": "DO_NOT_USE_AS_DESCRIPTION",
    },
    ("investor", "last_equity_funding_type"): {
        "semantic_class": "dynamic_funding_attribute",
        "temporal_risk": "high",
        "initial_disposition": "DO_NOT_USE_AS_DESCRIPTION",
    },
    ("investor", "last_funding_type"): {
        "semantic_class": "dynamic_funding_attribute",
        "temporal_risk": "high",
        "initial_disposition": "DO_NOT_USE_AS_DESCRIPTION",
    },

    # -------------------------------------------------------------------------
    # Startup
    # -------------------------------------------------------------------------
    ("startup", "description"): {
        "semantic_class": "text_candidate",
        "temporal_risk": "unversioned_current_snapshot",
        "initial_disposition": "AUDIT",
    },
    ("startup", "short_description"): {
        "semantic_class": "text_candidate",
        "temporal_risk": "unversioned_current_snapshot",
        "initial_disposition": "AUDIT",
    },
    ("startup", "company_type"): {
        "semantic_class": "entity_attribute_candidate",
        "temporal_risk": "unversioned_current_snapshot",
        "initial_disposition": "AUDIT",
    },
    ("startup", "categories"): {
        "semantic_class": "label_candidate",
        "temporal_risk": "unversioned_current_snapshot",
        "initial_disposition": "AUDIT",
    },
    ("startup", "category_groups"): {
        "semantic_class": "label_candidate",
        "temporal_risk": "unversioned_current_snapshot",
        "initial_disposition": "AUDIT",
    },
    ("startup", "locations"): {
        "semantic_class": "location_attribute",
        "temporal_risk": "unversioned_current_snapshot",
        "initial_disposition": "AUDIT",
    },
    ("startup", "city"): {
        "semantic_class": "location_attribute",
        "temporal_risk": "unversioned_current_snapshot",
        "initial_disposition": "AUDIT",
    },
    ("startup", "region"): {
        "semantic_class": "location_attribute",
        "temporal_risk": "unversioned_current_snapshot",
        "initial_disposition": "AUDIT",
    },
    ("startup", "country"): {
        "semantic_class": "location_attribute",
        "temporal_risk": "unversioned_current_snapshot",
        "initial_disposition": "AUDIT",
    },
    ("startup", "location_groups"): {
        "semantic_class": "location_attribute",
        "temporal_risk": "unversioned_current_snapshot",
        "initial_disposition": "AUDIT",
    },

    # Explicitly dynamic / outcome-sensitive fields.
    ("startup", "last_equity_funding_type"): {
        "semantic_class": "dynamic_funding_attribute",
        "temporal_risk": "high",
        "initial_disposition": "DO_NOT_USE_AS_DESCRIPTION",
    },
    ("startup", "funding_stage"): {
        "semantic_class": "dynamic_funding_attribute",
        "temporal_risk": "high",
        "initial_disposition": "DO_NOT_USE_AS_DESCRIPTION",
    },
    ("startup", "last_funding_type"): {
        "semantic_class": "dynamic_funding_attribute",
        "temporal_risk": "high",
        "initial_disposition": "DO_NOT_USE_AS_DESCRIPTION",
    },
    ("startup", "hub_tags"): {
        "semantic_class": "status_or_outcome_tag",
        "temporal_risk": "high",
        "initial_disposition": "DO_NOT_USE_AS_DESCRIPTION",
    },
}


# =============================================================================
# Helpers
# =============================================================================

def banner(title: str) -> None:
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)


def present_mask(series: pd.Series) -> pd.Series:
    """
    Treat null and blank strings as unavailable.

    Do not tokenize or otherwise interpret field contents here.
    """
    not_null = series.notna()

    text = series.astype("string").str.strip()

    return (
        not_null
        & text.notna()
        & text.ne("")
        & ~text.str.lower().isin(["nan", "none", "null"])
    )


def coverage_record(
    df: pd.DataFrame,
    id_col: str,
    field: str,
    source: str,
    canonical_count: int,
) -> dict:

    mask = present_mask(df[field])

    available_ids = df.loc[mask, id_col].nunique()
    unique_raw_values = (
        df.loc[mask, field]
        .astype("string")
        .str.strip()
        .nunique()
    )

    record = {
        "source": source,
        "field": field,
        "canonical_entities": int(canonical_count),
        "entities_with_value": int(available_ids),
        "entities_without_value": int(canonical_count - available_ids),
        "coverage_pct": float(
            100.0 * available_ids / canonical_count
        ),
        "unique_nonempty_raw_values": int(unique_raw_values),
    }

    classification = FIELD_CLASSIFICATION.get(
        (source, field),
        {
            "semantic_class": "UNCLASSIFIED",
            "temporal_risk": "UNCLASSIFIED",
            "initial_disposition": "UNCLASSIFIED",
        },
    )

    record.update(classification)

    return record


def text_stats(
    df: pd.DataFrame,
    id_col: str,
    field: str,
    source: str,
) -> dict:

    mask = present_mask(df[field])

    values = (
        df.loc[mask, [id_col, field]]
        .drop_duplicates()
        .copy()
    )

    if values.empty:
        return {
            "source": source,
            "field": field,
            "nonempty_rows": 0,
            "median_chars": np.nan,
            "p25_chars": np.nan,
            "p75_chars": np.nan,
            "p95_chars": np.nan,
            "median_words": np.nan,
            "p95_words": np.nan,
        }

    text = values[field].astype("string").str.strip()

    char_lengths = text.str.len()
    word_lengths = text.str.split().str.len()

    return {
        "source": source,
        "field": field,
        "nonempty_rows": int(len(values)),
        "median_chars": float(char_lengths.median()),
        "p25_chars": float(char_lengths.quantile(0.25)),
        "p75_chars": float(char_lengths.quantile(0.75)),
        "p95_chars": float(char_lengths.quantile(0.95)),
        "median_words": float(word_lengths.median()),
        "p95_words": float(word_lengths.quantile(0.95)),
    }


def text_overlap_summary(
    df: pd.DataFrame,
    id_col: str,
    source: str,
    canonical_count: int,
) -> dict:

    long_ids = set(
        df.loc[
            present_mask(df["description"]),
            id_col,
        ].dropna().astype(str)
    )

    short_ids = set(
        df.loc[
            present_mask(df["short_description"]),
            id_col,
        ].dropna().astype(str)
    )

    both = long_ids & short_ids
    long_only = long_ids - short_ids
    short_only = short_ids - long_ids
    neither = canonical_count - len(long_ids | short_ids)

    return {
        "source": source,
        "canonical_entities": canonical_count,
        "both_description_and_short": len(both),
        "description_only": len(long_only),
        "short_description_only": len(short_only),
        "neither_text_field": int(neither),
        "any_text_available": len(long_ids | short_ids),
        "any_text_coverage_pct": (
            100.0 * len(long_ids | short_ids) / canonical_count
        ),
    }


# =============================================================================
# 1. Frozen Phase-3 role population
# =============================================================================

banner("PHASE 4.1.3b — CANONICAL DESCRIPTION-FIELD COVERAGE AUDIT")

if not NODE_PATH.exists():
    raise FileNotFoundError(
        f"Frozen Phase-3 node table not found: {NODE_PATH}"
    )

nodes = pd.read_parquet(
    NODE_PATH,
    columns=[
        "node_type",
        "raw_entity_id",
    ],
)

print("\nFROZEN ROLE POPULATION")
print("-" * 110)

print(f"Role-node rows: {len(nodes):,}")

if len(nodes) != EXPECTED_ROLE_NODES:
    raise AssertionError(
        f"Expected {EXPECTED_ROLE_NODES:,} Phase-3 role nodes, "
        f"found {len(nodes):,}."
    )

investor_ids = set(
    nodes.loc[
        nodes["node_type"].eq("investor"),
        "raw_entity_id",
    ]
    .astype(str)
)

startup_ids = set(
    nodes.loc[
        nodes["node_type"].eq("startup"),
        "raw_entity_id",
    ]
    .astype(str)
)

print(f"Investor role IDs: {len(investor_ids):,}")
print(f"Startup role IDs:  {len(startup_ids):,}")

if len(investor_ids) != EXPECTED_INVESTORS:
    raise AssertionError(
        f"Expected {EXPECTED_INVESTORS:,} investors, "
        f"found {len(investor_ids):,}."
    )

if len(startup_ids) != EXPECTED_STARTUPS:
    raise AssertionError(
        f"Expected {EXPECTED_STARTUPS:,} startups, "
        f"found {len(startup_ids):,}."
    )


# =============================================================================
# 2. Investor canonical subset
# =============================================================================

banner("INVESTOR CANONICAL DESCRIPTION SUBSET")

investor_usecols = ["id"] + INVESTOR_FIELDS

investor_raw = pd.read_csv(
    INVESTOR_PATH,
    usecols=investor_usecols,
    dtype={"id": "string"},
    low_memory=False,
)

investor_raw["id"] = investor_raw["id"].astype(str)

investor_df = investor_raw.loc[
    investor_raw["id"].isin(investor_ids)
].copy()

investor_unique_ids = investor_df["id"].nunique()

print(f"Raw Investor rows:          {len(investor_raw):,}")
print(f"Canonical Investor rows:    {len(investor_df):,}")
print(f"Canonical unique IDs:       {investor_unique_ids:,}")
print(
    f"Canonical duplicate IDs:    "
    f"{investor_df['id'].duplicated().sum():,}"
)

missing_investor_ids = investor_ids - set(investor_df["id"])

print(
    f"Canonical IDs absent raw:   "
    f"{len(missing_investor_ids):,}"
)

if investor_unique_ids != EXPECTED_INVESTORS:
    raise AssertionError(
        "Investor registry no longer gives full frozen canonical coverage."
    )


# =============================================================================
# 3. Startup canonical subset — streamed over 24 shards
# =============================================================================

banner("STARTUP CANONICAL DESCRIPTION SUBSET")

company_usecols = ["id"] + COMPANY_FIELDS

startup_parts = []

total_raw_company_rows = 0

for shard_number, path in enumerate(COMPANY_PATHS, start=1):

    print(
        f"[{shard_number:02d}/{len(COMPANY_PATHS):02d}] "
        f"Scanning {path.name}"
    )

    shard_matches = []

    for chunk in pd.read_csv(
        path,
        usecols=company_usecols,
        dtype={"id": "string"},
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        total_raw_company_rows += len(chunk)

        chunk["id"] = chunk["id"].astype(str)

        match = chunk.loc[
            chunk["id"].isin(startup_ids)
        ].copy()

        if not match.empty:
            match["_source_shard"] = path.name
            shard_matches.append(match)

    if shard_matches:
        startup_parts.append(
            pd.concat(
                shard_matches,
                ignore_index=True,
            )
        )


if not startup_parts:
    raise RuntimeError(
        "No canonical Startup IDs were recovered from Companies shards."
    )

startup_df = pd.concat(
    startup_parts,
    ignore_index=True,
)

startup_unique_ids = startup_df["id"].nunique()

startup_row_duplicates = (
    len(startup_df) - startup_unique_ids
)

print()
print(f"Raw Company rows scanned:   {total_raw_company_rows:,}")
print(f"Canonical matched rows:     {len(startup_df):,}")
print(f"Canonical unique IDs:       {startup_unique_ids:,}")
print(f"Extra rows from overlaps:   {startup_row_duplicates:,}")

missing_startup_ids = startup_ids - set(startup_df["id"])

print(
    f"Canonical IDs absent raw:   "
    f"{len(missing_startup_ids):,}"
)

if startup_unique_ids != EXPECTED_STARTUPS:
    raise AssertionError(
        "Company shards no longer give full frozen canonical Startup coverage."
    )


# =============================================================================
# 4. Coverage audit
# =============================================================================

banner("ENTITY-LEVEL FIELD COVERAGE")

coverage_records = []

for field in INVESTOR_FIELDS:
    coverage_records.append(
        coverage_record(
            investor_df,
            id_col="id",
            field=field,
            source="investor",
            canonical_count=EXPECTED_INVESTORS,
        )
    )

for field in COMPANY_FIELDS:
    coverage_records.append(
        coverage_record(
            startup_df,
            id_col="id",
            field=field,
            source="startup",
            canonical_count=EXPECTED_STARTUPS,
        )
    )

coverage_df = pd.DataFrame(coverage_records)

for source in ["investor", "startup"]:

    print()
    print(source.upper())
    print("-" * 110)

    display = (
        coverage_df.loc[
            coverage_df["source"].eq(source),
            [
                "field",
                "entities_with_value",
                "coverage_pct",
                "unique_nonempty_raw_values",
                "semantic_class",
                "temporal_risk",
                "initial_disposition",
            ],
        ]
        .sort_values(
            ["initial_disposition", "coverage_pct"],
            ascending=[True, False],
        )
    )

    print(
        display.to_string(
            index=False,
            formatters={
                "coverage_pct": lambda x: f"{x:8.3f}",
            },
        )
    )


# =============================================================================
# 5. Text coverage and length
# =============================================================================

banner("TEXT-FIELD AVAILABILITY")

text_stats_df = pd.DataFrame(
    [
        text_stats(
            investor_df,
            "id",
            "description",
            "investor",
        ),
        text_stats(
            investor_df,
            "id",
            "short_description",
            "investor",
        ),
        text_stats(
            startup_df,
            "id",
            "description",
            "startup",
        ),
        text_stats(
            startup_df,
            "id",
            "short_description",
            "startup",
        ),
    ]
)

print(text_stats_df.to_string(index=False))


text_overlap_df = pd.DataFrame(
    [
        text_overlap_summary(
            investor_df,
            "id",
            "investor",
            EXPECTED_INVESTORS,
        ),
        text_overlap_summary(
            startup_df,
            "id",
            "startup",
            EXPECTED_STARTUPS,
        ),
    ]
)

print("\nTEXT OVERLAP")
print("-" * 110)

print(
    text_overlap_df.to_string(
        index=False,
        formatters={
            "any_text_coverage_pct": lambda x: f"{x:8.3f}",
        },
    )
)


# =============================================================================
# 6. Company duplicate-ID / field-consistency audit
# =============================================================================

banner("COMPANY SHARD DUPLICATE-ID FIELD CONSISTENCY")

startup_id_counts = startup_df["id"].value_counts()

duplicate_startup_ids = set(
    startup_id_counts[
        startup_id_counts.gt(1)
    ].index
)

duplicate_rows = startup_df.loc[
    startup_df["id"].isin(duplicate_startup_ids)
].copy()

print(
    f"Canonical Startup IDs represented >1 time: "
    f"{len(duplicate_startup_ids):,}"
)

print(
    f"Rows belonging to duplicated IDs:         "
    f"{len(duplicate_rows):,}"
)

conflict_records = []

for field in COMPANY_FIELDS:

    field_df = duplicate_rows.loc[
        present_mask(duplicate_rows[field]),
        ["id", field],
    ].copy()

    if field_df.empty:
        conflicting_ids = 0
        comparable_ids = 0

    else:
        field_df[field] = (
            field_df[field]
            .astype("string")
            .str.strip()
        )

        nunique = (
            field_df
            .groupby("id")[field]
            .nunique(dropna=True)
        )

        comparable_ids = int(len(nunique))
        conflicting_ids = int(nunique.gt(1).sum())

    conflict_records.append(
        {
            "source": "startup",
            "field": field,
            "duplicated_ids_with_any_value": comparable_ids,
            "ids_with_conflicting_nonempty_values": conflicting_ids,
            "conflict_pct_among_comparable_duplicates": (
                100.0 * conflicting_ids / comparable_ids
                if comparable_ids
                else 0.0
            ),
        }
    )

conflict_df = pd.DataFrame(conflict_records)

print(
    conflict_df.to_string(
        index=False,
        formatters={
            "conflict_pct_among_comparable_duplicates":
                lambda x: f"{x:8.4f}",
        },
    )
)


# =============================================================================
# 7. Raw delimiter-character audit
#
# IMPORTANT:
# This does NOT parse labels.
#
# It only tells us which fields visibly contain commas, semicolons, or pipes
# so that Phase 4.1.3c knows where delimiter semantics must be investigated.
# =============================================================================

banner("DELIMITER-PRESENCE AUDIT — NO TOKENIZATION")

DELIMITER_FIELDS = {
    "investor": [
        "categories",
        "category_groups",
        "investor_stage",
        "investor_type",
        "locations",
        "location_groups",
    ],
    "startup": [
        "categories",
        "category_groups",
        "locations",
        "location_groups",
        "hub_tags",
    ],
}

delimiter_records = []

source_frames = {
    "investor": investor_df,
    "startup": startup_df,
}

for source, fields in DELIMITER_FIELDS.items():

    frame = source_frames[source]

    for field in fields:

        mask = present_mask(frame[field])

        values = (
            frame.loc[mask, field]
            .astype("string")
        )

        delimiter_records.append(
            {
                "source": source,
                "field": field,
                "nonempty_rows": int(len(values)),
                "rows_with_comma": int(
                    values.str.contains(",", regex=False).sum()
                ),
                "rows_with_semicolon": int(
                    values.str.contains(";", regex=False).sum()
                ),
                "rows_with_pipe": int(
                    values.str.contains("|", regex=False).sum()
                ),
            }
        )

delimiter_df = pd.DataFrame(delimiter_records)

print(delimiter_df.to_string(index=False))


# =============================================================================
# 8. Semantic / risk matrix
# =============================================================================

banner("SEMANTIC AND TEMPORAL-RISK MATRIX")

semantic_records = []

for (source, field), values in FIELD_CLASSIFICATION.items():

    semantic_records.append(
        {
            "source": source,
            "field": field,
            **values,
        }
    )

semantic_df = pd.DataFrame(semantic_records)

print(
    semantic_df.sort_values(
        ["source", "initial_disposition", "field"]
    ).to_string(index=False)
)


# =============================================================================
# 9. Save audit outputs
# =============================================================================

coverage_path = (
    OUT_DIR / "canonical_description_field_coverage.csv"
)

text_stats_path = (
    OUT_DIR / "canonical_description_text_stats.csv"
)

text_overlap_path = (
    OUT_DIR / "canonical_description_text_overlap.csv"
)

conflict_path = (
    OUT_DIR / "canonical_company_field_conflict_audit.csv"
)

delimiter_path = (
    OUT_DIR / "description_candidate_delimiter_presence.csv"
)

semantic_path = (
    OUT_DIR / "description_field_semantic_risk_matrix.csv"
)

metadata_path = (
    OUT_DIR / "canonical_description_coverage_metadata.json"
)


coverage_df.to_csv(
    coverage_path,
    index=False,
)

text_stats_df.to_csv(
    text_stats_path,
    index=False,
)

text_overlap_df.to_csv(
    text_overlap_path,
    index=False,
)

conflict_df.to_csv(
    conflict_path,
    index=False,
)

delimiter_df.to_csv(
    delimiter_path,
    index=False,
)

semantic_df.to_csv(
    semantic_path,
    index=False,
)


metadata = {
    "phase": "4.1.3b",
    "purpose": (
        "canonical role-specific description-field coverage "
        "and consistency audit"
    ),
    "frozen_phase_3_node_table": str(NODE_PATH),
    "canonical_investor_count": EXPECTED_INVESTORS,
    "canonical_startup_count": EXPECTED_STARTUPS,
    "canonical_role_node_count": EXPECTED_ROLE_NODES,
    "company_shards_scanned": len(COMPANY_PATHS),
    "raw_company_rows_scanned": int(total_raw_company_rows),
    "matched_company_rows": int(len(startup_df)),
    "matched_unique_startup_ids": int(startup_unique_ids),
    "duplicated_startup_ids_across_source_rows": int(
        len(duplicate_startup_ids)
    ),
    "investor_text_fields_selected": False,
    "startup_text_fields_selected": False,
    "label_fields_selected": False,
    "label_tokenization_performed": False,
    "phase_2_reopened": False,
    "phase_3_reopened": False,
}

with open(
    metadata_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False,
    )


# =============================================================================
# 10. Final integrity summary
# =============================================================================

banner("PHASE 4.1.3b SUMMARY")

print(
    f"Frozen Investor IDs recovered: "
    f"{investor_unique_ids:,}/{EXPECTED_INVESTORS:,}"
)

print(
    f"Frozen Startup IDs recovered:  "
    f"{startup_unique_ids:,}/{EXPECTED_STARTUPS:,}"
)

print(
    f"Startup IDs duplicated across Company source rows: "
    f"{len(duplicate_startup_ids):,}"
)

print("\nOutputs:")
print(f"  {coverage_path}")
print(f"  {text_stats_path}")
print(f"  {text_overlap_path}")
print(f"  {conflict_path}")
print(f"  {delimiter_path}")
print(f"  {semantic_path}")
print(f"  {metadata_path}")

print("\nIMPORTANT:")
print(
    "No Text_o, Text_b, Labels_o, or Labels_b mapping "
    "has been frozen."
)

print(
    "No multi-value field has been tokenized."
)

print(
    "Phase-2 temporal decisions and Phase-3 graph decisions "
    "remain unchanged."
)

print("\nPHASE 4.1.3b STATUS: COMPLETE")