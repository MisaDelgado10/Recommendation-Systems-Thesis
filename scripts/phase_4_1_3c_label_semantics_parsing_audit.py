from pathlib import Path
import json

import numpy as np
import pandas as pd


# =============================================================================
# PHASE 4.1.3c — LABEL SEMANTICS, PARSING, AND TOP-2000 AUDIT
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

LABEL_FIELDS = [
    "categories",
    "category_groups",
]


# =============================================================================
# Helpers
# =============================================================================

def banner(title: str) -> None:
    print()
    print("=" * 115)
    print(title)
    print("=" * 115)


def present_mask(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()

    return (
        series.notna()
        & text.notna()
        & text.ne("")
        & ~text.str.lower().isin(["nan", "none", "null"])
    )


def provisional_tokenize(
    df: pd.DataFrame,
    id_col: str,
    field: str,
    source: str,
):
    """
    PROVISIONAL comma tokenization for audit only.

    This does not freeze comma as the final parser.
    """

    work = df.loc[
        present_mask(df[field]),
        [id_col, field],
    ].copy()

    work[id_col] = work[id_col].astype(str)

    work["raw_value"] = (
        work[field]
        .astype("string")
        .str.strip()
    )

    # Number of fragments the provisional comma parser would generate.
    work["raw_fragment_count"] = (
        work["raw_value"]
        .str.split(",")
        .str.len()
    )

    exploded = (
        work[
            [id_col, "raw_value", "raw_fragment_count"]
        ]
        .assign(
            raw_token=lambda x:
                x["raw_value"].str.split(",")
        )
        .explode("raw_token")
        .reset_index(drop=True)
    )

    exploded["raw_token"] = (
        exploded["raw_token"]
        .astype("string")
    )

    exploded["token"] = (
        exploded["raw_token"]
        .str.strip()
    )

    exploded["empty_token"] = (
        exploded["token"].isna()
        | exploded["token"].eq("")
    )

    exploded["whitespace_changed"] = (
        exploded["raw_token"].fillna("")
        != exploded["token"].fillna("")
    )

    nonempty = exploded.loc[
        ~exploded["empty_token"]
    ].copy()

    nonempty["duplicate_entity_token"] = (
        nonempty.duplicated(
            subset=[id_col, "token"],
            keep="first",
        )
    )

    unique_entity_tokens = (
        nonempty
        .drop_duplicates(
            subset=[id_col, "token"],
            keep="first",
        )
        .copy()
    )

    unique_entity_tokens["source"] = source
    unique_entity_tokens["field"] = field

    unique_entity_tokens["entity_key"] = (
        source
        + "::"
        + unique_entity_tokens[id_col].astype(str)
    )

    return (
        work,
        exploded,
        nonempty,
        unique_entity_tokens,
    )


def build_vocabulary(
    entity_tokens: pd.DataFrame,
    source: str,
    field: str,
) -> pd.DataFrame:

    vocab = (
        entity_tokens
        .groupby("token", as_index=False)
        .agg(
            entity_count=("entity_key", "nunique"),
        )
    )

    # Because we use unique entity-token pairs, entity_count is the relevant
    # frequency for one-hot labels.
    vocab = vocab.sort_values(
        ["entity_count", "token"],
        ascending=[False, True],
    ).reset_index(drop=True)

    vocab["rank_by_entity_frequency"] = (
        np.arange(len(vocab)) + 1
    )

    vocab["source"] = source
    vocab["field"] = field

    return vocab[
        [
            "source",
            "field",
            "rank_by_entity_frequency",
            "token",
            "entity_count",
        ]
    ]


def token_count_summary(
    unique_entity_tokens: pd.DataFrame,
    source: str,
    field: str,
    canonical_entities: int,
    raw_work: pd.DataFrame,
    exploded: pd.DataFrame,
    nonempty: pd.DataFrame,
) -> dict:

    per_entity = (
        unique_entity_tokens
        .groupby("entity_key")
        .size()
    )

    labeled_entities = int(per_entity.size)

    return {
        "source": source,
        "field": field,
        "canonical_entities": canonical_entities,
        "labeled_entities": labeled_entities,
        "label_coverage_pct":
            100.0 * labeled_entities / canonical_entities,
        "raw_nonempty_rows": int(len(raw_work)),
        "raw_token_fragments": int(len(exploded)),
        "nonempty_token_occurrences": int(len(nonempty)),
        "unique_entity_token_pairs":
            int(len(unique_entity_tokens)),
        "unique_tokens":
            int(unique_entity_tokens["token"].nunique()),
        "empty_token_fragments":
            int(exploded["empty_token"].sum()),
        "duplicate_entity_token_occurrences":
            int(nonempty["duplicate_entity_token"].sum()),
        "median_labels_per_labeled_entity":
            float(per_entity.median()) if labeled_entities else np.nan,
        "mean_labels_per_labeled_entity":
            float(per_entity.mean()) if labeled_entities else np.nan,
        "p95_labels_per_labeled_entity":
            float(per_entity.quantile(0.95))
            if labeled_entities else np.nan,
        "max_labels_per_labeled_entity":
            int(per_entity.max()) if labeled_entities else 0,
    }


def top_k_retention(
    entity_tokens: pd.DataFrame,
    vocab: pd.DataFrame,
    source: str,
    field: str,
    canonical_entities: int,
    k: int = 2000,
) -> dict:

    top_tokens = set(
        vocab.loc[
            vocab["rank_by_entity_frequency"].le(k),
            "token",
        ]
    )

    retained = entity_tokens.loc[
        entity_tokens["token"].isin(top_tokens)
    ]

    all_labeled_entities = (
        entity_tokens["entity_key"].nunique()
    )

    retained_entities = (
        retained["entity_key"].nunique()
    )

    total_pairs = len(entity_tokens)
    retained_pairs = len(retained)

    return {
        "source": source,
        "field": field,
        "k": k,
        "unique_tokens_total": int(len(vocab)),
        "tokens_selected": int(len(top_tokens)),
        "entity_token_pairs_total": int(total_pairs),
        "entity_token_pairs_retained":
            int(retained_pairs),
        "pair_retention_pct":
            (
                100.0 * retained_pairs / total_pairs
                if total_pairs
                else np.nan
            ),
        "labeled_entities_total":
            int(all_labeled_entities),
        "entities_with_top_k_label":
            int(retained_entities),
        "retention_among_labeled_entities_pct":
            (
                100.0
                * retained_entities
                / all_labeled_entities
                if all_labeled_entities
                else np.nan
            ),
        "coverage_of_canonical_entities_pct":
            (
                100.0
                * retained_entities
                / canonical_entities
            ),
    }


def suspicious_vocabulary(vocab: pd.DataFrame) -> pd.DataFrame:
    """
    Heuristics only.

    These rows deserve human inspection. A token appearing here is NOT
    automatically invalid.
    """

    out = vocab.copy()

    token = out["token"].astype("string")
    lower = token.str.lower()

    out["length_le_2"] = token.str.len().le(2)
    out["numeric_only"] = token.str.fullmatch(r"\d+").fillna(False)

    out["bare_conjunction"] = lower.isin(
        ["and", "or", "&"]
    )

    out["starts_with_conjunction"] = (
        lower.str.startswith("and ")
        | lower.str.startswith("or ")
    )

    out["starts_or_ends_punctuation"] = (
        token.str.match(r"^[,;|:]").fillna(False)
        | token.str.contains(r"[,;|:]$", regex=True).fillna(False)
    )

    flags = [
        "length_le_2",
        "numeric_only",
        "bare_conjunction",
        "starts_with_conjunction",
        "starts_or_ends_punctuation",
    ]

    return out.loc[
        out[flags].any(axis=1)
    ].copy()


def casefold_collision_audit(vocab: pd.DataFrame) -> pd.DataFrame:

    temp = vocab[
        ["source", "field", "token", "entity_count"]
    ].copy()

    temp["casefold_token"] = (
        temp["token"]
        .astype("string")
        .str.casefold()
    )

    groups = (
        temp
        .groupby(
            ["source", "field", "casefold_token"],
            as_index=False,
        )
        .agg(
            variants=("token", lambda x: " || ".join(sorted(set(x)))),
            variant_count=("token", "nunique"),
            combined_entity_frequency=("entity_count", "sum"),
        )
    )

    return groups.loc[
        groups["variant_count"].gt(1)
    ].copy()


# =============================================================================
# 1. Load frozen Phase-3 role universe
# =============================================================================

banner("PHASE 4.1.3c — LABEL SEMANTICS, PARSING, AND TOP-2000 AUDIT")

nodes = pd.read_parquet(
    NODE_PATH,
    columns=[
        "node_type",
        "raw_entity_id",
    ],
)

if len(nodes) != EXPECTED_ROLE_NODES:
    raise AssertionError(
        f"Expected {EXPECTED_ROLE_NODES:,} frozen role nodes; "
        f"found {len(nodes):,}."
    )

investor_ids = set(
    nodes.loc[
        nodes["node_type"].eq("investor"),
        "raw_entity_id",
    ].astype(str)
)

startup_ids = set(
    nodes.loc[
        nodes["node_type"].eq("startup"),
        "raw_entity_id",
    ].astype(str)
)

if len(investor_ids) != EXPECTED_INVESTORS:
    raise AssertionError("Frozen Investor population mismatch.")

if len(startup_ids) != EXPECTED_STARTUPS:
    raise AssertionError("Frozen Startup population mismatch.")

print(f"Frozen Investors: {len(investor_ids):,}")
print(f"Frozen Startups:  {len(startup_ids):,}")


# =============================================================================
# 2. Load canonical Investor label candidates
# =============================================================================

banner("LOADING INVESTOR LABEL CANDIDATES")

investor_df = pd.read_csv(
    INVESTOR_PATH,
    usecols=["id"] + LABEL_FIELDS,
    dtype={"id": "string"},
    low_memory=False,
)

investor_df["id"] = investor_df["id"].astype(str)

investor_df = investor_df.loc[
    investor_df["id"].isin(investor_ids)
].copy()

if investor_df["id"].nunique() != EXPECTED_INVESTORS:
    raise AssertionError(
        "Investor label audit does not recover all frozen Investors."
    )

if investor_df["id"].duplicated().any():
    raise AssertionError(
        "Unexpected duplicate Investor IDs."
    )

print(f"Canonical Investor rows: {len(investor_df):,}")


# =============================================================================
# 3. Load canonical Startup label candidates
# =============================================================================

banner("LOADING STARTUP LABEL CANDIDATES")

startup_parts = []

for shard_number, path in enumerate(COMPANY_PATHS, start=1):

    print(
        f"[{shard_number:02d}/{len(COMPANY_PATHS):02d}] "
        f"{path.name}"
    )

    for chunk in pd.read_csv(
        path,
        usecols=["id"] + LABEL_FIELDS,
        dtype={"id": "string"},
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):

        chunk["id"] = chunk["id"].astype(str)

        matched = chunk.loc[
            chunk["id"].isin(startup_ids)
        ].copy()

        if not matched.empty:
            startup_parts.append(matched)


startup_raw = pd.concat(
    startup_parts,
    ignore_index=True,
)

print(
    f"\nMatched Startup source rows before deduplication: "
    f"{len(startup_raw):,}"
)

print(
    f"Unique Startup IDs before deduplication:          "
    f"{startup_raw['id'].nunique():,}"
)

# Phase 4.1.3b established that duplicate canonical Company rows have
# no conflicting values in categories/category_groups.
#
# Therefore we consume that audited result rather than reopening the decision.
startup_df = startup_raw.drop_duplicates(
    subset=["id"],
    keep="first",
).copy()

if startup_df["id"].nunique() != EXPECTED_STARTUPS:
    raise AssertionError(
        "Startup label audit does not recover all frozen Startups."
    )

print(
    f"Canonical Startup rows after audited deduplication: "
    f"{len(startup_df):,}"
)


# =============================================================================
# 4. Provisional tokenization audit
# =============================================================================

banner("PROVISIONAL COMMA-TOKENIZATION AUDIT")

source_frames = {
    "investor": (
        investor_df,
        EXPECTED_INVESTORS,
    ),
    "startup": (
        startup_df,
        EXPECTED_STARTUPS,
    ),
}

all_entity_tokens = {}
all_vocabularies = {}

summary_records = []
vocab_frames = []
suspicious_frames = []
casefold_frames = []
extreme_frames = []


for source, (frame, canonical_count) in source_frames.items():

    for field in LABEL_FIELDS:

        print()
        print(f"{source.upper()} — {field}")
        print("-" * 115)

        (
            raw_work,
            exploded,
            nonempty,
            unique_entity_tokens,
        ) = provisional_tokenize(
            frame,
            id_col="id",
            field=field,
            source=source,
        )

        key = (source, field)

        all_entity_tokens[key] = unique_entity_tokens

        vocab = build_vocabulary(
            unique_entity_tokens,
            source=source,
            field=field,
        )

        all_vocabularies[key] = vocab
        vocab_frames.append(vocab)

        summary = token_count_summary(
            unique_entity_tokens,
            source=source,
            field=field,
            canonical_entities=canonical_count,
            raw_work=raw_work,
            exploded=exploded,
            nonempty=nonempty,
        )

        summary_records.append(summary)

        print(
            f"Labeled entities:             "
            f"{summary['labeled_entities']:,}"
        )

        print(
            f"Coverage:                     "
            f"{summary['label_coverage_pct']:.3f}%"
        )

        print(
            f"Unique parsed tokens:         "
            f"{summary['unique_tokens']:,}"
        )

        print(
            f"Median labels/entity:         "
            f"{summary['median_labels_per_labeled_entity']:.2f}"
        )

        print(
            f"P95 labels/entity:            "
            f"{summary['p95_labels_per_labeled_entity']:.2f}"
        )

        print(
            f"Maximum labels/entity:        "
            f"{summary['max_labels_per_labeled_entity']:,}"
        )

        print(
            f"Empty fragments:              "
            f"{summary['empty_token_fragments']:,}"
        )

        print(
            f"Duplicate entity-token rows:  "
            f"{summary['duplicate_entity_token_occurrences']:,}"
        )

        # ---------------------------------------------------------------------
        # Suspicious vocabulary
        # ---------------------------------------------------------------------

        suspicious = suspicious_vocabulary(vocab)

        if not suspicious.empty:
            suspicious_frames.append(suspicious)

        print(
            f"Suspicious vocabulary tokens: "
            f"{len(suspicious):,}"
        )

        # ---------------------------------------------------------------------
        # Case-only collisions
        # ---------------------------------------------------------------------

        casefold = casefold_collision_audit(vocab)

        if not casefold.empty:
            casefold_frames.append(casefold)

        print(
            f"Casefold collision groups:    "
            f"{len(casefold):,}"
        )

        # ---------------------------------------------------------------------
        # Extreme raw rows for manual inspection
        # ---------------------------------------------------------------------

        extreme = (
            raw_work
            .sort_values(
                "raw_fragment_count",
                ascending=False,
            )
            .head(50)
            .copy()
        )

        extreme["source"] = source
        extreme["field"] = field

        extreme_frames.append(
            extreme[
                [
                    "source",
                    "field",
                    "id",
                    "raw_fragment_count",
                    "raw_value",
                ]
            ]
        )

        # ---------------------------------------------------------------------
        # Top-20 preview
        # ---------------------------------------------------------------------

        print("\nTop 20 provisional tokens:")

        print(
            vocab.head(20).to_string(
                index=False
            )
        )


summary_df = pd.DataFrame(summary_records)

vocabulary_df = pd.concat(
    vocab_frames,
    ignore_index=True,
)


# =============================================================================
# 5. Candidate-field coverage relationship
# =============================================================================

banner("CATEGORIES VS CATEGORY_GROUPS COVERAGE RELATIONSHIP")

coverage_relation_records = []

for source, (frame, canonical_count) in source_frames.items():

    categories_ids = set(
        frame.loc[
            present_mask(frame["categories"]),
            "id",
        ].astype(str)
    )

    groups_ids = set(
        frame.loc[
            present_mask(frame["category_groups"]),
            "id",
        ].astype(str)
    )

    record = {
        "source": source,
        "canonical_entities": canonical_count,
        "both": len(categories_ids & groups_ids),
        "categories_only": len(categories_ids - groups_ids),
        "category_groups_only": len(groups_ids - categories_ids),
        "neither": (
            canonical_count
            - len(categories_ids | groups_ids)
        ),
    }

    coverage_relation_records.append(record)

coverage_relation_df = pd.DataFrame(
    coverage_relation_records
)

print(
    coverage_relation_df.to_string(index=False)
)


# =============================================================================
# 6. Cross-role vocabulary overlap
# =============================================================================

banner("INVESTOR / STARTUP VOCABULARY OVERLAP")

overlap_records = []

for field in LABEL_FIELDS:

    investor_vocab = set(
        all_vocabularies[
            ("investor", field)
        ]["token"]
    )

    startup_vocab = set(
        all_vocabularies[
            ("startup", field)
        ]["token"]
    )

    intersection = investor_vocab & startup_vocab
    union = investor_vocab | startup_vocab

    overlap_records.append(
        {
            "field": field,
            "investor_unique_tokens":
                len(investor_vocab),
            "startup_unique_tokens":
                len(startup_vocab),
            "shared_tokens":
                len(intersection),
            "union_tokens":
                len(union),
            "jaccard_pct":
                (
                    100.0
                    * len(intersection)
                    / len(union)
                    if union
                    else np.nan
                ),
        }
    )

overlap_df = pd.DataFrame(overlap_records)

print(
    overlap_df.to_string(
        index=False,
        formatters={
            "jaccard_pct": lambda x: f"{x:.3f}",
        },
    )
)


# =============================================================================
# 7. Paper-style top-2000 retention audit — role-specific
# =============================================================================

banner("TOP-2000 RETENTION — ROLE-SPECIFIC VOCABULARIES")

top2000_records = []

for source, canonical_count in [
    ("investor", EXPECTED_INVESTORS),
    ("startup", EXPECTED_STARTUPS),
]:

    for field in LABEL_FIELDS:

        tokens = all_entity_tokens[
            (source, field)
        ]

        vocab = all_vocabularies[
            (source, field)
        ]

        top2000_records.append(
            top_k_retention(
                entity_tokens=tokens,
                vocab=vocab,
                source=source,
                field=field,
                canonical_entities=canonical_count,
                k=2000,
            )
        )


# =============================================================================
# 8. Paper-style top-2000 retention audit — shared vocabulary
# =============================================================================

banner("TOP-2000 RETENTION — SHARED INVESTOR/STARTUP VOCABULARY")

shared_vocab_frames = []

for field in LABEL_FIELDS:

    combined = pd.concat(
        [
            all_entity_tokens[
                ("investor", field)
            ],
            all_entity_tokens[
                ("startup", field)
            ],
        ],
        ignore_index=True,
    )

    shared_vocab = (
        combined
        .groupby("token", as_index=False)
        .agg(
            entity_count=("entity_key", "nunique"),
        )
        .sort_values(
            ["entity_count", "token"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    shared_vocab[
        "rank_by_entity_frequency"
    ] = np.arange(len(shared_vocab)) + 1

    shared_vocab["source"] = "combined"
    shared_vocab["field"] = field

    shared_vocab_frames.append(
        shared_vocab[
            [
                "source",
                "field",
                "rank_by_entity_frequency",
                "token",
                "entity_count",
            ]
        ]
    )

    top2000_records.append(
        top_k_retention(
            entity_tokens=combined,
            vocab=shared_vocab,
            source="combined",
            field=field,
            canonical_entities=EXPECTED_ROLE_NODES,
            k=2000,
        )
    )


shared_vocabulary_df = pd.concat(
    shared_vocab_frames,
    ignore_index=True,
)

top2000_df = pd.DataFrame(
    top2000_records
)

print(
    top2000_df.to_string(
        index=False,
        formatters={
            "pair_retention_pct":
                lambda x: f"{x:.4f}",
            "retention_among_labeled_entities_pct":
                lambda x: f"{x:.4f}",
            "coverage_of_canonical_entities_pct":
                lambda x: f"{x:.4f}",
        },
    )
)


# =============================================================================
# 9. Frequency-tail audit
# =============================================================================

banner("LABEL FREQUENCY-TAIL AUDIT")

frequency_tail_records = []

combined_vocab_for_tail = pd.concat(
    [
        vocabulary_df,
        shared_vocabulary_df,
    ],
    ignore_index=True,
)

for (source, field), group in combined_vocab_for_tail.groupby(
    ["source", "field"]
):

    counts = group["entity_count"]

    frequency_tail_records.append(
        {
            "source": source,
            "field": field,
            "unique_tokens": int(len(group)),
            "support_eq_1": int(counts.eq(1).sum()),
            "support_lt_5": int(counts.lt(5).sum()),
            "support_lt_10": int(counts.lt(10).sum()),
            "support_ge_10": int(counts.ge(10).sum()),
            "support_ge_100": int(counts.ge(100).sum()),
            "support_ge_1000": int(counts.ge(1000).sum()),
        }
    )

frequency_tail_df = pd.DataFrame(
    frequency_tail_records
)

print(
    frequency_tail_df.to_string(index=False)
)


# =============================================================================
# 10. Paper-grounded semantic narrowing
# =============================================================================

banner("PAPER-GROUNDED LABEL-CANDIDATE NARROWING")

semantic_decision_df = pd.DataFrame(
    [
        {
            "field": "categories",
            "paper_semantic_alignment":
                "DIRECT",
            "reason":
                "Crunchbase categories represent industries / "
                "business fields, matching ITRS label semantics.",
            "phase_4_status":
                "PRIMARY_LABEL_CANDIDATE",
        },
        {
            "field": "category_groups",
            "paper_semantic_alignment":
                "DIRECT_BUT_COARSER",
            "reason":
                "Industry/field taxonomy at a broader aggregation "
                "level; useful as an alternative or ablation.",
            "phase_4_status":
                "SECONDARY_LABEL_CANDIDATE",
        },
        {
            "field": "investor_type",
            "paper_semantic_alignment":
                "INDIRECT",
            "reason":
                "Describes investor organizational type rather than "
                "the industries/fields to which the entity belongs.",
            "phase_4_status":
                "EXCLUDE_FROM_PRIMARY_ITRS_LABELS",
        },
        {
            "field": "investor_stage",
            "paper_semantic_alignment":
                "INDIRECT_AND_HISTORY_SENSITIVE",
            "reason":
                "Represents investment-stage profile and may be "
                "derived from observed investment behavior.",
            "phase_4_status":
                "EXCLUDE_FROM_PRIMARY_ITRS_LABELS",
        },
        {
            "field": "location_fields",
            "paper_semantic_alignment":
                "INDIRECT",
            "reason":
                "Geography is not the industry/field label semantics "
                "described by ITRS.",
            "phase_4_status":
                "EXCLUDE_FROM_PRIMARY_ITRS_LABELS",
        },
        {
            "field": "funding_or_outcome_fields",
            "paper_semantic_alignment":
                "NO",
            "reason":
                "Dynamic and potentially future-sensitive; not "
                "description labels.",
            "phase_4_status":
                "EXCLUDE",
        },
    ]
)

print(
    semantic_decision_df.to_string(index=False)
)


# =============================================================================
# 11. Consolidate diagnostics
# =============================================================================

if suspicious_frames:
    suspicious_df = pd.concat(
        suspicious_frames,
        ignore_index=True,
    )
else:
    suspicious_df = pd.DataFrame(
        columns=[
            "source",
            "field",
            "rank_by_entity_frequency",
            "token",
            "entity_count",
        ]
    )

if casefold_frames:
    casefold_df = pd.concat(
        casefold_frames,
        ignore_index=True,
    )
else:
    casefold_df = pd.DataFrame(
        columns=[
            "source",
            "field",
            "casefold_token",
            "variants",
            "variant_count",
            "combined_entity_frequency",
        ]
    )

extreme_df = pd.concat(
    extreme_frames,
    ignore_index=True,
)


# =============================================================================
# 12. Save outputs
# =============================================================================

summary_path = (
    OUT_DIR / "description_label_provisional_parse_summary.csv"
)

vocab_path = (
    OUT_DIR / "description_label_provisional_vocabulary.csv"
)

shared_vocab_path = (
    OUT_DIR / "description_label_shared_vocabulary.csv"
)

overlap_path = (
    OUT_DIR / "description_label_cross_role_overlap.csv"
)

top2000_path = (
    OUT_DIR / "description_label_top2000_retention.csv"
)

tail_path = (
    OUT_DIR / "description_label_frequency_tail.csv"
)

suspicious_path = (
    OUT_DIR / "description_label_suspicious_tokens.csv"
)

casefold_path = (
    OUT_DIR / "description_label_casefold_collisions.csv"
)

extreme_path = (
    OUT_DIR / "description_label_extreme_raw_rows.csv"
)

coverage_relation_path = (
    OUT_DIR / "description_label_candidate_coverage_relationship.csv"
)

semantic_path = (
    OUT_DIR / "description_label_semantic_decision_matrix.csv"
)

metadata_path = (
    OUT_DIR / "description_label_parsing_metadata.json"
)


summary_df.to_csv(summary_path, index=False)
vocabulary_df.to_csv(vocab_path, index=False)
shared_vocabulary_df.to_csv(shared_vocab_path, index=False)
overlap_df.to_csv(overlap_path, index=False)
top2000_df.to_csv(top2000_path, index=False)
frequency_tail_df.to_csv(tail_path, index=False)
suspicious_df.to_csv(suspicious_path, index=False)
casefold_df.to_csv(casefold_path, index=False)
extreme_df.to_csv(extreme_path, index=False)
coverage_relation_df.to_csv(coverage_relation_path, index=False)
semantic_decision_df.to_csv(semantic_path, index=False)


metadata = {
    "phase": "4.1.3c",
    "purpose": (
        "Audit Crunchbase industry/field label semantics, "
        "provisional comma parsing, vocabulary size, cross-role "
        "overlap, and paper-style top-2000 retention."
    ),
    "paper_label_semantics": (
        "Industries and fields to which the entity belongs."
    ),
    "paper_label_encoding": "one_hot",
    "paper_label_frequency_rule": (
        "Low-frequency labels removed; top 2000 retained."
    ),
    "top_k": 2000,
    "provisional_parser": "comma_split_then_strip",
    "parser_frozen": False,
    "label_mapping_frozen": False,
    "primary_candidate": "categories",
    "secondary_candidate": "category_groups",
    "company_duplicate_resolution": (
        "consume Phase-4.1.3b audit; keep first identical source row"
    ),
    "phase_2_reopened": False,
    "phase_3_reopened": False,
}

with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False,
    )


# =============================================================================
# 13. Final summary
# =============================================================================

banner("PHASE 4.1.3c SUMMARY")

print("\nPrimary paper-semantic label candidate:")
print("  categories")

print("\nSecondary paper-semantic label candidate:")
print("  category_groups")

print("\nImportant:")
print(
    "Comma tokenization remains PROVISIONAL until this audit "
    "is reviewed."
)

print(
    "No final Labels_o / Labels_b vocabulary has been frozen."
)

print(
    "No one-hot matrices have been materialized."
)

print(
    "No Phase-2 temporal or Phase-3 graph decision was changed."
)

print("\nOutputs:")
for path in [
    summary_path,
    vocab_path,
    shared_vocab_path,
    overlap_path,
    top2000_path,
    tail_path,
    suspicious_path,
    casefold_path,
    extreme_path,
    coverage_relation_path,
    semantic_path,
    metadata_path,
]:
    print(f"  {path}")

print("\nPHASE 4.1.3c STATUS: COMPLETE")