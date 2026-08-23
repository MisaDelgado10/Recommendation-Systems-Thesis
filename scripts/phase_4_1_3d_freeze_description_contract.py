from pathlib import Path
from collections import Counter
import json

import numpy as np
import pandas as pd


# =============================================================================
# PHASE 4.1.3d — FREEZE CRUNCHBASE DESCRIPTION CONTRACT
# =============================================================================

RAW_DIR = Path("data/raw")

NODE_PATH = Path(
    "data/experimental/phase_3/graph/nodes.parquet"
)

OUT_DIR = Path(
    "data/experimental/phase_4/description_contract"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

INVESTOR_PATH = (
    RAW_DIR / "CRUNCHBASE_investor_20260531_333726.csv"
)

COMPANY_PATHS = sorted(
    RAW_DIR.glob("companies*.csv")
)

CHUNK_SIZE = 200_000


# =============================================================================
# Frozen Phase-3 population
# =============================================================================

EXPECTED_INVESTORS = 165_975
EXPECTED_STARTUPS = 311_589
EXPECTED_ROLE_NODES = 477_564

# Phase-4.1.3b text audit
EXPECTED_INVESTORS_WITH_TEXT = 165_970
EXPECTED_STARTUPS_WITH_TEXT = 311_586

# Phase-4.1.3c label audit
EXPECTED_INVESTORS_WITH_LABELS = 61_957
EXPECTED_STARTUPS_WITH_LABELS = 303_688

# Phase-4.1.3c.1 corrected shared vocabulary
EXPECTED_SHARED_LABEL_VOCAB = 802

PAPER_TOP_K = 2000


# =============================================================================
# Audited embedded-comma category
# =============================================================================

HVAC_CATEGORY = (
    "Heating, Ventilation, and Air Conditioning (HVAC)"
)

HVAC_SENTINEL = (
    "__ITRS_CRUNCHBASE_PROTECTED_HVAC_CATEGORY__"
)


# =============================================================================
# Helpers
# =============================================================================

def banner(title: str) -> None:
    print()
    print("=" * 115)
    print(title)
    print("=" * 115)


def normalize_scalar(value):
    """
    Convert a raw scalar to stripped text or None.
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    if value.lower() in {
        "nan",
        "none",
        "null",
    }:
        return None

    return value


def resolve_text(
    description,
    short_description,
):
    """
    Frozen Phase-4 description-source policy.

    Priority:
      1. description
      2. short_description
      3. missing
    """

    description = normalize_scalar(
        description
    )

    short_description = normalize_scalar(
        short_description
    )

    if description is not None:
        return (
            description,
            "description",
        )

    if short_description is not None:
        return (
            short_description,
            "short_description_fallback",
        )

    return (
        None,
        "missing",
    )


def parse_categories(value):
    """
    Final audited Crunchbase categories parser.

    Phase 4.1.3c.1 established that every literal comma+space
    sequence in the frozen canonical categories universe is part
    of exactly one legitimate Crunchbase category:

        Heating, Ventilation, and Air Conditioning (HVAC)

    The parser therefore protects that exact category before
    splitting the normal comma-delimited category list.

    Any remaining ', ' is treated as an unexpected schema/content
    change and raises an error rather than silently misparsing.
    """

    value = normalize_scalar(value)

    if value is None:
        return []

    protected = value.replace(
        HVAC_CATEGORY,
        HVAC_SENTINEL,
    )

    # Leakage / parser-drift style safety principle:
    # unexpected source syntax must fail loudly.
    if ", " in protected:
        raise ValueError(
            "Unexpected embedded-comma category pattern "
            f"after HVAC protection: {value}"
        )

    tokens = []

    for token in protected.split(","):

        token = token.strip()

        if not token:
            continue

        token = token.replace(
            HVAC_SENTINEL,
            HVAC_CATEGORY,
        )

        tokens.append(token)

    # No duplicate labels within one entity should survive.
    if len(tokens) != len(set(tokens)):
        raise ValueError(
            "Duplicate parsed category token detected: "
            f"{value}"
        )

    return tokens


# =============================================================================
# 1. Load frozen Phase-3 nodes
# =============================================================================

banner(
    "PHASE 4.1.3d — "
    "FREEZE CRUNCHBASE DESCRIPTION CONTRACT"
)

nodes = pd.read_parquet(
    NODE_PATH,
    columns=[
        "node_id",
        "node_type",
        "raw_entity_id",
    ],
)

if len(nodes) != EXPECTED_ROLE_NODES:
    raise AssertionError(
        f"Expected {EXPECTED_ROLE_NODES:,} role nodes; "
        f"found {len(nodes):,}."
    )

if nodes["node_id"].duplicated().any():
    raise AssertionError(
        "Duplicate frozen Phase-3 node_id detected."
    )

investor_nodes = nodes.loc[
    nodes["node_type"].eq("investor")
].copy()

startup_nodes = nodes.loc[
    nodes["node_type"].eq("startup")
].copy()

if len(investor_nodes) != EXPECTED_INVESTORS:
    raise AssertionError(
        "Frozen Investor role population mismatch."
    )

if len(startup_nodes) != EXPECTED_STARTUPS:
    raise AssertionError(
        "Frozen Startup role population mismatch."
    )

investor_ids = set(
    investor_nodes["raw_entity_id"]
    .astype(str)
)

startup_ids = set(
    startup_nodes["raw_entity_id"]
    .astype(str)
)

print(
    f"Frozen Investors: {len(investor_ids):,}"
)

print(
    f"Frozen Startups:  {len(startup_ids):,}"
)


# =============================================================================
# 2. Load canonical Investor description fields
# =============================================================================

banner("LOADING INVESTOR DESCRIPTION INPUTS")

investor_raw = pd.read_csv(
    INVESTOR_PATH,
    usecols=[
        "id",
        "description",
        "short_description",
        "categories",
    ],
    dtype={
        "id": "string",
    },
    low_memory=False,
)

investor_raw["id"] = (
    investor_raw["id"].astype(str)
)

investor_raw = investor_raw.loc[
    investor_raw["id"].isin(investor_ids)
].copy()

if (
    investor_raw["id"].nunique()
    != EXPECTED_INVESTORS
):
    raise AssertionError(
        "Investor registry does not recover "
        "the frozen Investor universe."
    )

if investor_raw["id"].duplicated().any():
    raise AssertionError(
        "Unexpected duplicate Investor IDs."
    )

print(
    f"Canonical Investors recovered: "
    f"{len(investor_raw):,}"
)


# =============================================================================
# 3. Load canonical Startup description fields
# =============================================================================

banner("LOADING STARTUP DESCRIPTION INPUTS")

startup_parts = []

for i, path in enumerate(
    COMPANY_PATHS,
    start=1,
):

    print(
        f"[{i:02d}/{len(COMPANY_PATHS):02d}] "
        f"{path.name}"
    )

    for chunk in pd.read_csv(
        path,
        usecols=[
            "id",
            "description",
            "short_description",
            "categories",
        ],
        dtype={
            "id": "string",
        },
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):

        chunk["id"] = (
            chunk["id"].astype(str)
        )

        matched = chunk.loc[
            chunk["id"].isin(startup_ids)
        ].copy()

        if not matched.empty:
            startup_parts.append(
                matched
            )


startup_raw = pd.concat(
    startup_parts,
    ignore_index=True,
)

print()
print(
    f"Matched Startup source rows: "
    f"{len(startup_raw):,}"
)

print(
    f"Unique Startup IDs:          "
    f"{startup_raw['id'].nunique():,}"
)


# =============================================================================
# 4. Consume Phase-4.1.3b duplicate consistency result
#
# We still verify selected fields here so contract materialization fails if
# source data changed after the audit.
# =============================================================================

banner("VERIFYING STARTUP DUPLICATE CONSISTENCY")

duplicate_ids = set(
    startup_raw.loc[
        startup_raw["id"].duplicated(
            keep=False
        ),
        "id",
    ]
)

print(
    f"Duplicated canonical Startup IDs: "
    f"{len(duplicate_ids):,}"
)

for field in [
    "description",
    "short_description",
    "categories",
]:

    subset = startup_raw.loc[
        startup_raw["id"].isin(
            duplicate_ids
        ),
        [
            "id",
            field,
        ],
    ].copy()

    subset[field] = (
        subset[field]
        .astype("string")
        .str.strip()
    )

    conflicts = (
        subset
        .dropna(subset=[field])
        .groupby("id")[field]
        .nunique(dropna=True)
        .gt(1)
        .sum()
    )

    print(
        f"{field:<25} conflicts: "
        f"{int(conflicts):,}"
    )

    if conflicts:
        raise AssertionError(
            f"Unexpected Startup duplicate conflict "
            f"in {field}."
        )


startup_raw = (
    startup_raw
    .drop_duplicates(
        subset=["id"],
        keep="first",
    )
    .copy()
)

if (
    startup_raw["id"].nunique()
    != EXPECTED_STARTUPS
):
    raise AssertionError(
        "Startup registry does not recover "
        "the frozen Startup universe."
    )


# =============================================================================
# 5. Resolve frozen text and category inputs
# =============================================================================

banner("RESOLVING DESCRIPTION INPUT CONTRACT")


def build_role_inputs(
    frame,
    role,
):

    records = []

    for row in frame.itertuples(
        index=False
    ):

        text, text_source = resolve_text(
            row.description,
            row.short_description,
        )

        labels = parse_categories(
            row.categories
        )

        records.append(
            {
                "raw_entity_id": str(row.id),
                "node_type": role,
                "text": text,
                "text_source": text_source,
                "has_text": text is not None,
                "labels": labels,
                "label_count": len(labels),
                "has_labels": len(labels) > 0,
            }
        )

    return pd.DataFrame(records)


investor_inputs = build_role_inputs(
    investor_raw,
    "investor",
)

startup_inputs = build_role_inputs(
    startup_raw,
    "startup",
)


# =============================================================================
# 6. Integrity checks on text coverage
# =============================================================================

banner("TEXT CONTRACT INTEGRITY")

investor_text_count = int(
    investor_inputs["has_text"].sum()
)

startup_text_count = int(
    startup_inputs["has_text"].sum()
)

print(
    f"Investors with text: "
    f"{investor_text_count:,}"
)

print(
    f"Startups with text:  "
    f"{startup_text_count:,}"
)

print(
    f"Investors missing text: "
    f"{EXPECTED_INVESTORS - investor_text_count:,}"
)

print(
    f"Startups missing text:  "
    f"{EXPECTED_STARTUPS - startup_text_count:,}"
)

if (
    investor_text_count
    != EXPECTED_INVESTORS_WITH_TEXT
):
    raise AssertionError(
        "Investor text coverage differs "
        "from Phase 4.1.3b."
    )

if (
    startup_text_count
    != EXPECTED_STARTUPS_WITH_TEXT
):
    raise AssertionError(
        "Startup text coverage differs "
        "from Phase 4.1.3b."
    )


# =============================================================================
# 7. Integrity checks on label coverage
# =============================================================================

banner("LABEL CONTRACT INTEGRITY")

investor_label_count = int(
    investor_inputs["has_labels"].sum()
)

startup_label_count = int(
    startup_inputs["has_labels"].sum()
)

print(
    f"Investors with labels: "
    f"{investor_label_count:,}"
)

print(
    f"Startups with labels:  "
    f"{startup_label_count:,}"
)

if (
    investor_label_count
    != EXPECTED_INVESTORS_WITH_LABELS
):
    raise AssertionError(
        "Investor label coverage differs "
        "from Phase 4.1.3c."
    )

if (
    startup_label_count
    != EXPECTED_STARTUPS_WITH_LABELS
):
    raise AssertionError(
        "Startup label coverage differs "
        "from Phase 4.1.3c."
    )


# =============================================================================
# 8. Build shared category vocabulary
# =============================================================================

banner("BUILDING SHARED CATEGORY VOCABULARY")

label_frequency = Counter()


for labels in investor_inputs["labels"]:
    label_frequency.update(
        set(labels)
    )


for labels in startup_inputs["labels"]:
    label_frequency.update(
        set(labels)
    )


ranked_labels = sorted(
    label_frequency.items(),
    key=lambda item: (
        -item[1],
        item[0],
    ),
)

vocabulary_size = len(
    ranked_labels
)

print(
    f"Corrected shared vocabulary: "
    f"{vocabulary_size:,}"
)

if (
    vocabulary_size
    != EXPECTED_SHARED_LABEL_VOCAB
):
    raise AssertionError(
        f"Expected corrected shared vocabulary "
        f"of {EXPECTED_SHARED_LABEL_VOCAB:,}, "
        f"found {vocabulary_size:,}."
    )


# =============================================================================
# 9. Paper top-2000 rule
# =============================================================================

selected_label_count = min(
    PAPER_TOP_K,
    vocabulary_size,
)

if selected_label_count != vocabulary_size:
    raise AssertionError(
        "Unexpected label filtering: "
        "Crunchbase vocabulary should be "
        "smaller than paper top-k."
    )


vocabulary_records = []

for rank, (
    label,
    entity_frequency,
) in enumerate(
    ranked_labels,
    start=1,
):

    vocabulary_records.append(
        {
            "label_index": rank - 1,
            "rank_by_entity_frequency": rank,
            "label": label,
            "entity_frequency": entity_frequency,
            "selected_by_paper_top_k": (
                rank <= PAPER_TOP_K
            ),
        }
    )


vocabulary_df = pd.DataFrame(
    vocabulary_records
)


# =============================================================================
# 10. Specific HVAC integrity assertions
# =============================================================================

banner("HVAC PARSER INTEGRITY")

if HVAC_CATEGORY not in set(
    vocabulary_df["label"]
):
    raise AssertionError(
        "Protected HVAC category missing "
        "from final vocabulary."
    )


for artifact in [
    "Heating",
    "Ventilation",
    "and Air Conditioning (HVAC)",
]:

    if artifact in set(
        vocabulary_df["label"]
    ):
        raise AssertionError(
            f"Provisional artifact survived: "
            f"{artifact}"
        )


print(
    "HVAC category preserved as one label: PASS"
)

print(
    "All three provisional fragments removed: PASS"
)


# =============================================================================
# 11. Attach frozen graph node_id
# =============================================================================

banner("ATTACHING PHASE-3 GRAPH IDENTITIES")

role_inputs = pd.concat(
    [
        investor_inputs,
        startup_inputs,
    ],
    ignore_index=True,
)

manifest = nodes.merge(
    role_inputs,
    on=[
        "node_type",
        "raw_entity_id",
    ],
    how="left",
    validate="one_to_one",
)

if len(manifest) != EXPECTED_ROLE_NODES:
    raise AssertionError(
        "Description manifest row count mismatch."
    )

if manifest["text_source"].isna().any():
    raise AssertionError(
        "At least one frozen graph node failed "
        "description-input attachment."
    )


# =============================================================================
# 12. Serialize labels deterministically
#
# Use JSON text rather than relying on implementation-specific parquet
# handling of object/list columns.
# =============================================================================

manifest["labels_json"] = (
    manifest["labels"]
    .apply(
        lambda labels:
            json.dumps(
                labels,
                ensure_ascii=False,
            )
    )
)

manifest = manifest.drop(
    columns=["labels"]
)


# =============================================================================
# 13. Missing-text entities
# =============================================================================

missing_text_df = manifest.loc[
    ~manifest["has_text"],
    [
        "node_id",
        "node_type",
        "raw_entity_id",
        "text_source",
        "has_labels",
        "label_count",
        "labels_json",
    ],
].copy()

print()
print(
    f"Total role nodes missing text: "
    f"{len(missing_text_df):,}"
)

if len(missing_text_df) != 8:
    raise AssertionError(
        "Expected exactly 8 role nodes "
        "without description text."
    )


# =============================================================================
# 14. Audit summary
# =============================================================================

audit_df = pd.DataFrame(
    [
        {
            "metric":
                "frozen_role_nodes",
            "value":
                EXPECTED_ROLE_NODES,
        },
        {
            "metric":
                "investors",
            "value":
                EXPECTED_INVESTORS,
        },
        {
            "metric":
                "startups",
            "value":
                EXPECTED_STARTUPS,
        },
        {
            "metric":
                "investors_with_text",
            "value":
                investor_text_count,
        },
        {
            "metric":
                "startups_with_text",
            "value":
                startup_text_count,
        },
        {
            "metric":
                "investors_with_categories",
            "value":
                investor_label_count,
        },
        {
            "metric":
                "startups_with_categories",
            "value":
                startup_label_count,
        },
        {
            "metric":
                "shared_category_vocabulary",
            "value":
                vocabulary_size,
        },
        {
            "metric":
                "paper_top_k",
            "value":
                PAPER_TOP_K,
        },
        {
            "metric":
                "labels_retained",
            "value":
                selected_label_count,
        },
        {
            "metric":
                "missing_text_role_nodes",
            "value":
                len(missing_text_df),
        },
    ]
)


# =============================================================================
# 15. Frozen model contract
# =============================================================================

contract = {
    "phase": "4.1.3d",
    "status": "FROZEN",
    "module": "ITRS description input contract",

    "population": {
        "investors": EXPECTED_INVESTORS,
        "startups": EXPECTED_STARTUPS,
        "role_nodes": EXPECTED_ROLE_NODES,
    },

    "text": {
        "investor_field_priority": [
            "description",
            "short_description",
        ],
        "startup_field_priority": [
            "description",
            "short_description",
        ],
        "resolution_rule": (
            "description if nonempty; otherwise "
            "short_description; otherwise missing"
        ),
        "missing_text_policy": (
            "No fabricated text. Missing entities are retained. "
            "Zero text representation will be assigned during "
            "description feature construction."
        ),
        "investors_with_text":
            investor_text_count,
        "startups_with_text":
            startup_text_count,
        "temporal_provenance":
            "current_snapshot_unversioned",
    },

    "labels": {
        "primary_field": "categories",
        "representation": "shared multi-hot vocabulary",
        "parser": (
            "protect exact audited HVAC category; "
            "split remaining comma-delimited values; "
            "strip whitespace"
        ),
        "embedded_comma_category": HVAC_CATEGORY,
        "shared_vocabulary_size":
            vocabulary_size,
        "paper_top_k":
            PAPER_TOP_K,
        "labels_retained":
            selected_label_count,
        "investors_with_labels":
            investor_label_count,
        "startups_with_labels":
            startup_label_count,
        "missing_label_policy":
            "all-zero multi-hot vector",
        "temporal_provenance":
            "current_snapshot_unversioned",
    },

    "excluded_primary_description_fields": [
        "category_groups",
        "investor_type",
        "investor_stage",
        "locations",
        "city",
        "region",
        "country",
        "location_groups",
        "funding_stage",
        "last_funding_type",
        "last_equity_funding_type",
        "hub_tags",
        "num_investments_funding_rounds",
        "num_lead_investments",
        "num_diversity_spotlight_investments",
    ],

    "secondary_ablation": {
        "field": "category_groups",
        "status": "not used in primary ITRS reproduction",
    },

    "paper_vs_reproduction": {
        "paper_text_source":
            "one description document per entity",
        "crunchbase_text_adaptation":
            "long description with short-description fallback",
        "paper_label_semantics":
            "industries and fields",
        "crunchbase_label_mapping":
            "categories",
        "paper_max_labels":
            2000,
        "crunchbase_observed_labels":
            vocabulary_size,
        "top_k_effect":
            "none; observed vocabulary is below 2000",
    },

    "not_yet_frozen": [
        "English text tokenizer",
        "Doc2Vec algorithm variant",
        "Doc2Vec epochs",
        "Doc2Vec other paper-unspecified parameters",
        "description MLP hidden dimensions",
        "40-dimensional text/label branch split",
    ],

    "phase_2_reopened": False,
    "phase_3_reopened": False,
}


# =============================================================================
# 16. Save frozen outputs
# =============================================================================

manifest_path = (
    OUT_DIR
    / "description_input_manifest.parquet"
)

vocabulary_path = (
    OUT_DIR
    / "description_label_vocabulary.csv"
)

audit_path = (
    OUT_DIR
    / "description_contract_audit.csv"
)

missing_text_path = (
    OUT_DIR
    / "description_missing_text_entities.csv"
)

contract_path = (
    OUT_DIR
    / "description_contract.json"
)


manifest.to_parquet(
    manifest_path,
    index=False,
)

vocabulary_df.to_csv(
    vocabulary_path,
    index=False,
)

audit_df.to_csv(
    audit_path,
    index=False,
)

missing_text_df.to_csv(
    missing_text_path,
    index=False,
)

with open(
    contract_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        contract,
        f,
        indent=2,
        ensure_ascii=False,
    )


# =============================================================================
# 17. Final summary
# =============================================================================

banner("PHASE 4.1.3d FINAL SUMMARY")

print(
    f"Description input rows:       "
    f"{len(manifest):,}"
)

print(
    f"Investors with text:          "
    f"{investor_text_count:,}"
)

print(
    f"Startups with text:           "
    f"{startup_text_count:,}"
)

print(
    f"Investors with labels:        "
    f"{investor_label_count:,}"
)

print(
    f"Startups with labels:         "
    f"{startup_label_count:,}"
)

print(
    f"Final shared label vocabulary:"
    f" {vocabulary_size:,}"
)

print(
    f"Paper top-k:                  "
    f"{PAPER_TOP_K:,}"
)

print(
    f"Labels retained:              "
    f"{selected_label_count:,}"
)

print(
    f"Missing-text role nodes:      "
    f"{len(missing_text_df):,}"
)

print()
print("Outputs:")

for path in [
    manifest_path,
    vocabulary_path,
    audit_path,
    missing_text_path,
    contract_path,
]:
    print(f"  {path}")


print()
print(
    "Text field mapping:              FROZEN"
)

print(
    "Label field mapping:             FROZEN"
)

print(
    "Category parser:                 FROZEN"
)

print(
    "Shared label vocabulary:         FROZEN"
)

print(
    "English tokenizer:               NOT YET FROZEN"
)

print(
    "Doc2Vec training configuration:  NOT YET FROZEN"
)

print(
    "Description neural architecture: NOT YET FROZEN"
)

print()
print(
    "PHASE 4.1.3d STATUS: COMPLETE — "
    "DESCRIPTION INPUT CONTRACT FROZEN"
)