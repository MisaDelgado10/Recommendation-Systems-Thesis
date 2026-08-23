from pathlib import Path
from collections import Counter
import os
import re
import sys

import numpy as np
import pandas as pd
import scipy
import gensim

from gensim.models.doc2vec import Doc2Vec
from gensim.models import doc2vec


# =============================================================================
# PHASE 4.2.1b — DOC2VEC VOCABULARY AND CONFIGURATION PRE-FREEZE AUDIT
# =============================================================================

MANIFEST_PATH = Path(
    "data/experimental/phase_4/"
    "description_contract/"
    "description_input_manifest.parquet"
)

OUT_DIR = Path(
    "data/experimental/phase_4/audits"
)
OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


EXPECTED_ROLE_NODES = 477_564
EXPECTED_INVESTORS = 165_975
EXPECTED_STARTUPS = 311_589

EXPECTED_SOURCE_MISSING = 8
EXPECTED_TOKEN_EMPTY = 41


# =============================================================================
# Frozen tokenizer from Phase 4.2.1a
# =============================================================================

TOKEN_RE = re.compile(
    r"\b[^\W_]+(?:[-'’][^\W_]+)*\b",
    flags=re.UNICODE,
)


def tokenize(text):

    if pd.isna(text):
        return []

    text = str(text).strip()

    if not text:
        return []

    return [
        token.casefold()
        for token in TOKEN_RE.findall(text)
    ]


def banner(title):

    print()
    print("=" * 115)
    print(title)
    print("=" * 115)


# =============================================================================
# 1. Environment
# =============================================================================

banner(
    "PHASE 4.2.1b — "
    "DOC2VEC VOCABULARY AND CONFIGURATION PRE-FREEZE AUDIT"
)

print("\nENVIRONMENT")
print("-" * 115)

print(
    f"Python:               "
    f"{sys.version.splitlines()[0]}"
)

print(
    f"NumPy:                "
    f"{np.__version__}"
)

print(
    f"SciPy:                "
    f"{scipy.__version__}"
)

print(
    f"Gensim:               "
    f"{gensim.__version__}"
)

print(
    f"Doc2Vec FAST_VERSION: "
    f"{doc2vec.FAST_VERSION}"
)

print(
    f"PYTHONHASHSEED:        "
    f"{os.environ.get('PYTHONHASHSEED')}"
)


if gensim.__version__ != "4.4.0":
    raise AssertionError(
        "Expected frozen candidate Gensim version 4.4.0."
    )

if doc2vec.FAST_VERSION < 0:
    raise AssertionError(
        "Optimized Doc2Vec implementation is unavailable."
    )


# =============================================================================
# 2. Gensim constructor / resolved-parameter audit
# =============================================================================

banner("GENSIM DOC2VEC RESOLVED PARAMETER AUDIT")


# Blank model: no training occurs.
default_candidate = Doc2Vec(
    vector_size=32,
    window=3,
    dm=1,
    min_count=5,
    workers=1,
    seed=42,
    epochs=5,
)


attrs = [
    "vector_size",
    "window",
    "dm",
    "dm_mean",
    "dm_concat",
    "dbow_words",
    "min_count",
    "sample",
    "workers",
    "epochs",
    "alpha",
    "min_alpha",
    "hs",
    "negative",
    "ns_exponent",
    "seed",
    "shrink_windows",
]


resolved_records = []

for attr in attrs:

    value = getattr(
        default_candidate,
        attr,
        "<MISSING>",
    )

    print(
        f"{attr:<20} {value}"
    )

    resolved_records.append(
        {
            "parameter": attr,
            "resolved_value": str(value),
        }
    )


resolved_df = pd.DataFrame(
    resolved_records
)


# =============================================================================
# 3. Load frozen description manifest
# =============================================================================

banner("LOADING FROZEN DESCRIPTION MANIFEST")


manifest = pd.read_parquet(
    MANIFEST_PATH,
    columns=[
        "node_id",
        "node_type",
        "has_text",
        "text",
    ],
)


if len(manifest) != EXPECTED_ROLE_NODES:
    raise AssertionError(
        "Frozen description manifest row count changed."
    )


role_counts = (
    manifest["node_type"]
    .value_counts()
    .to_dict()
)

assert (
    role_counts.get("investor", 0)
    == EXPECTED_INVESTORS
)

assert (
    role_counts.get("startup", 0)
    == EXPECTED_STARTUPS
)


source_missing = int(
    (~manifest["has_text"]).sum()
)

if source_missing != EXPECTED_SOURCE_MISSING:
    raise AssertionError(
        "Source-level missing-text count changed."
    )


print(
    f"Frozen role nodes:       "
    f"{len(manifest):,}"
)

print(
    f"Source-missing text:     "
    f"{source_missing:,}"
)


# =============================================================================
# 4. Tokenize once
# =============================================================================

banner("TOKENIZING WITH FROZEN TOKENIZER")


manifest["tokens"] = (
    manifest["text"]
    .apply(tokenize)
)

manifest[
    "token_count_before_min_count"
] = (
    manifest["tokens"]
    .apply(len)
)


token_empty_mask = (
    manifest["has_text"]
    & manifest[
        "token_count_before_min_count"
    ].eq(0)
)


token_empty = int(
    token_empty_mask.sum()
)


print(
    f"Source-present but token-empty: "
    f"{token_empty:,}"
)


if token_empty != EXPECTED_TOKEN_EMPTY:
    raise AssertionError(
        "Frozen tokenizer no longer reproduces "
        "the 41 audited token-empty documents."
    )


print("\nToken-empty by role:")

print(
    manifest.loc[
        token_empty_mask,
        "node_type",
    ]
    .value_counts()
    .to_string()
)


# =============================================================================
# 5. Role-specific min_count audit
#
# IMPORTANT:
# ITRS pretrains Investor and Startup Doc2Vec corpora separately.
# Therefore vocabulary frequencies are calculated separately.
# =============================================================================

banner("ROLE-SPECIFIC MIN_COUNT RETENTION AUDIT")


MIN_COUNT_CANDIDATES = [
    1,
    2,
    3,
    5,
    10,
]


audit_records = []


for role in [
    "investor",
    "startup",
]:

    role_df = manifest.loc[
        manifest["node_type"].eq(role)
    ].copy()


    counter = Counter()

    for tokens in role_df["tokens"]:
        counter.update(tokens)


    total_occurrences = sum(
        counter.values()
    )


    print()
    print(
        f"{role.upper()} raw token vocabulary: "
        f"{len(counter):,}"
    )

    print(
        f"{role.upper()} token occurrences:    "
        f"{total_occurrences:,}"
    )


    for min_count in MIN_COUNT_CANDIDATES:

        retained_vocab = {
            token
            for token, count
            in counter.items()
            if count >= min_count
        }


        retained_lengths = (
            role_df["tokens"]
            .apply(
                lambda tokens:
                    sum(
                        token in retained_vocab
                        for token in tokens
                    )
            )
        )


        source_present = (
            role_df["has_text"]
        )

        tokenizable_before = (
            role_df[
                "token_count_before_min_count"
            ].gt(0)
        )


        empty_after_pruning = (
            source_present
            & retained_lengths.eq(0)
        )


        newly_empty_due_to_min_count = (
            source_present
            & tokenizable_before
            & retained_lengths.eq(0)
        )


        retained_occurrences = sum(
            count
            for token, count
            in counter.items()
            if token in retained_vocab
        )


        record = {
            "node_type":
                role,

            "min_count":
                min_count,

            "raw_vocabulary":
                len(counter),

            "retained_vocabulary":
                len(retained_vocab),

            "vocabulary_retention_pct":
                (
                    100.0
                    * len(retained_vocab)
                    / len(counter)
                    if counter
                    else 0.0
                ),

            "raw_token_occurrences":
                total_occurrences,

            "retained_token_occurrences":
                retained_occurrences,

            "token_occurrence_retention_pct":
                (
                    100.0
                    * retained_occurrences
                    / total_occurrences
                    if total_occurrences
                    else 0.0
                ),

            "documents_empty_after_pruning":
                int(
                    empty_after_pruning.sum()
                ),

            "documents_newly_empty_due_to_min_count":
                int(
                    newly_empty_due_to_min_count.sum()
                ),

            "documents_with_at_least_one_retained_token":
                int(
                    retained_lengths.gt(0).sum()
                ),

            "median_retained_tokens":
                float(
                    retained_lengths.median()
                ),

            "p95_retained_tokens":
                float(
                    retained_lengths.quantile(
                        0.95
                    )
                ),
        }


        audit_records.append(
            record
        )


        print()
        print(
            f"min_count={min_count}"
        )

        print(
            f"  retained vocabulary: "
            f"{len(retained_vocab):,}"
        )

        print(
            f"  vocabulary retention: "
            f"{record['vocabulary_retention_pct']:.3f}%"
        )

        print(
            f"  token occurrence retention: "
            f"{record['token_occurrence_retention_pct']:.3f}%"
        )

        print(
            f"  docs empty after pruning: "
            f"{record['documents_empty_after_pruning']:,}"
        )

        print(
            f"  newly empty due to min_count: "
            f"{record['documents_newly_empty_due_to_min_count']:,}"
        )


audit_df = pd.DataFrame(
    audit_records
)


# =============================================================================
# 6. Inspect entities newly emptied by min_count=5
# =============================================================================

banner(
    "MIN_COUNT=5 NEWLY EMPTY DOCUMENT INSPECTION"
)


newly_empty_frames = []


for role in [
    "investor",
    "startup",
]:

    role_df = manifest.loc[
        manifest["node_type"].eq(role)
    ].copy()


    counter = Counter()

    for tokens in role_df["tokens"]:
        counter.update(tokens)


    retained_vocab = {
        token
        for token, count
        in counter.items()
        if count >= 5
    }


    role_df[
        "retained_token_count_min5"
    ] = (
        role_df["tokens"]
        .apply(
            lambda tokens:
                sum(
                    token in retained_vocab
                    for token in tokens
                )
        )
    )


    newly_empty = role_df.loc[
        role_df[
            "token_count_before_min_count"
        ].gt(0)
        & role_df[
            "retained_token_count_min5"
        ].eq(0),
        [
            "node_id",
            "node_type",
            "text",
            "tokens",
        ],
    ].copy()


    newly_empty[
        "tokens_json"
    ] = (
        newly_empty["tokens"]
        .apply(
            lambda x:
                str(x)
        )
    )


    newly_empty = (
        newly_empty
        .drop(
            columns=["tokens"]
        )
    )


    newly_empty_frames.append(
        newly_empty
    )


if newly_empty_frames:

    newly_empty_df = pd.concat(
        newly_empty_frames,
        ignore_index=True,
    )

else:

    newly_empty_df = pd.DataFrame()


print(
    f"Documents newly empty under min_count=5: "
    f"{len(newly_empty_df):,}"
)


if not newly_empty_df.empty:

    print()
    print(
        newly_empty_df.head(
            50
        ).to_string(
            index=False
        )
    )


# =============================================================================
# 7. Save outputs
# =============================================================================

resolved_path = (
    OUT_DIR
    / "doc2vec_resolved_parameter_audit.csv"
)

min_count_path = (
    OUT_DIR
    / "doc2vec_min_count_retention_audit.csv"
)

newly_empty_path = (
    OUT_DIR
    / "doc2vec_min_count5_newly_empty_documents.csv"
)


resolved_df.to_csv(
    resolved_path,
    index=False,
)

audit_df.to_csv(
    min_count_path,
    index=False,
)

newly_empty_df.to_csv(
    newly_empty_path,
    index=False,
)


# =============================================================================
# 8. Final summary
# =============================================================================

banner("PHASE 4.2.1b SUMMARY")


print(
    "Frozen tokenizer consumed: PASS"
)

print(
    "Source description contract reopened: NO"
)

print(
    "Doc2Vec training performed:          NO"
)

print(
    "Doc2Vec configuration frozen:        NO"
)

print()
print(
    "This subphase measures the effect of "
    "the candidate min_count before freezing it."
)

print()
print("Outputs:")

for path in [
    resolved_path,
    min_count_path,
    newly_empty_path,
]:

    print(
        f"  {path}"
    )


print(
    "\nPHASE 4.2.1b STATUS: COMPLETE — "
    "PRE-FREEZE AUDIT ONLY"
)