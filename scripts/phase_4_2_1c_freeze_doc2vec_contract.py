from pathlib import Path
from collections import Counter
import hashlib
import json
import os
import platform
import re
import sys

import gensim
import numpy as np
import pandas as pd
import scipy

from gensim.models.doc2vec import Doc2Vec
from gensim.models import doc2vec


# =============================================================================
# PHASE 4.2.1c — FREEZE COMPLETE DOC2VEC CONTRACT
# =============================================================================

MANIFEST_PATH = Path(
    "data/experimental/phase_4/"
    "description_contract/"
    "description_input_manifest.parquet"
)

OUT_DIR = Path(
    "data/experimental/phase_4/doc2vec_contract"
)
OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# Frozen populations
# =============================================================================

EXPECTED_ROLE_NODES = 477_564

EXPECTED_INVESTORS = 165_975
EXPECTED_STARTUPS = 311_589

EXPECTED_SOURCE_MISSING_INVESTOR = 5
EXPECTED_SOURCE_MISSING_STARTUP = 3

EXPECTED_TOKENIZER_EMPTY_INVESTOR = 12
EXPECTED_TOKENIZER_EMPTY_STARTUP = 29

EXPECTED_MINCOUNT_EMPTY_INVESTOR = 2_427
EXPECTED_MINCOUNT_EMPTY_STARTUP = 194

EXPECTED_ELIGIBLE_INVESTOR = 163_531
EXPECTED_ELIGIBLE_STARTUP = 311_363

EXPECTED_ZERO_TEXT_FEATURE = 2_670
EXPECTED_ELIGIBLE_TOTAL = 474_894

EXPECTED_INVESTOR_VOCAB = 36_505
EXPECTED_STARTUP_VOCAB = 57_605


# =============================================================================
# Frozen tokenizer
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


# =============================================================================
# Frozen Doc2Vec configuration
# =============================================================================

DOC2VEC_CONFIG = {
    # -------------------------------------------------------------------------
    # PAPER_SPECIFIED
    # -------------------------------------------------------------------------
    "vector_size": 32,
    "window": 3,

    # -------------------------------------------------------------------------
    # PAPER_UNSPECIFIED_REPRODUCTION_CHOICE
    # -------------------------------------------------------------------------
    "dm": 1,
    "dm_mean": 1,
    "dm_concat": 0,
    "dm_tag_count": 1,
    "dbow_words": 0,

    "min_count": 5,
    "sample": 0.001,

    "hs": 0,
    "negative": 5,
    "ns_exponent": 0.75,

    "alpha": 0.025,
    "min_alpha": 0.0001,

    "epochs": 5,

    "workers": 1,
    "seed": 42,

    "shrink_windows": True,

    "sorted_vocab": 1,
    "batch_words": 10_000,

    "max_vocab_size": None,
    "max_final_vocab": None,

    "compute_loss": False,
}


# =============================================================================
# Helpers
# =============================================================================

def banner(title):

    print()
    print("=" * 118)
    print(title)
    print("=" * 118)


def sha256_vocabulary(counter, min_count):

    digest = hashlib.sha256()

    for token, count in sorted(
        counter.items(),
        key=lambda x: x[0],
    ):

        if count < min_count:
            continue

        digest.update(
            token.encode("utf-8")
        )

        digest.update(b"\t")

        digest.update(
            str(count).encode("ascii")
        )

        digest.update(b"\n")

    return digest.hexdigest()


def build_vocabulary_df(
    counter,
    role,
    min_count,
):

    rows = [
        {
            "node_type": role,
            "token": token,
            "frequency": count,
        }
        for token, count
        in counter.items()
        if count >= min_count
    ]

    df = pd.DataFrame(rows)

    df = (
        df
        .sort_values(
            [
                "frequency",
                "token",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    df[
        "frequency_rank"
    ] = (
        np.arange(len(df))
        + 1
    )

    return df[
        [
            "node_type",
            "frequency_rank",
            "token",
            "frequency",
        ]
    ]


# =============================================================================
# 1. Environment integrity
# =============================================================================

banner(
    "PHASE 4.2.1c — "
    "FREEZE COMPLETE DOC2VEC CONTRACT"
)

print("\nENVIRONMENT")
print("-" * 118)

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
        "Expected Gensim 4.4.0."
    )

if doc2vec.FAST_VERSION < 0:
    raise AssertionError(
        "Optimized Doc2Vec routines unavailable."
    )

if (
    os.environ.get("PYTHONHASHSEED")
    != "42"
):
    raise AssertionError(
        "Run with PYTHONHASHSEED=42."
    )


# =============================================================================
# 2. Verify explicitly resolved Gensim configuration
# =============================================================================

banner("DOC2VEC PARAMETER RESOLUTION")


candidate = Doc2Vec(
    **DOC2VEC_CONFIG
)


resolved = {
    "vector_size":
        candidate.vector_size,

    "window":
        candidate.window,

    "dm":
        int(bool(candidate.dm)),

    # Gensim stores the operational mean/sum switch as cbow_mean.
    "dm_mean_operational_cbow_mean":
        candidate.cbow_mean,

    "dm_concat":
        candidate.dm_concat,

    "dm_tag_count":
        candidate.dm_tag_count,

    "dbow_words":
        candidate.dbow_words,

    "min_count":
        candidate.min_count,

    "sample":
        candidate.sample,

    "hs":
        candidate.hs,

    "negative":
        candidate.negative,

    "ns_exponent":
        candidate.ns_exponent,

    "alpha":
        candidate.alpha,

    "min_alpha":
        candidate.min_alpha,

    "epochs":
        candidate.epochs,

    "workers":
        candidate.workers,

    "seed":
        candidate.seed,

    "shrink_windows":
        candidate.shrink_windows,

    "sorted_vocab":
        candidate.sorted_vocab,

    "batch_words":
        candidate.batch_words,
}


for key, value in resolved.items():

    print(
        f"{key:<32} {value}"
    )


# Explicit assertions prevent hidden-default drift.

assert resolved["vector_size"] == 32
assert resolved["window"] == 3

assert resolved["dm"] == 1

# dm_mean=1 should resolve operationally to mean aggregation.
assert (
    resolved[
        "dm_mean_operational_cbow_mean"
    ]
    == 1
)

assert resolved["dm_concat"] == 0
assert resolved["dm_tag_count"] == 1
assert resolved["dbow_words"] == 0

assert resolved["min_count"] == 5
assert resolved["sample"] == 0.001

assert resolved["hs"] == 0
assert resolved["negative"] == 5
assert resolved["ns_exponent"] == 0.75

assert resolved["alpha"] == 0.025
assert resolved["min_alpha"] == 0.0001

assert resolved["epochs"] == 5
assert resolved["workers"] == 1
assert resolved["seed"] == 42

assert resolved["shrink_windows"] is True
assert resolved["sorted_vocab"] == 1
assert resolved["batch_words"] == 10_000


# =============================================================================
# 3. Load frozen description contract
# =============================================================================

banner("LOADING FROZEN DESCRIPTION INPUTS")


manifest = pd.read_parquet(
    MANIFEST_PATH,
    columns=[
        "node_id",
        "node_type",
        "raw_entity_id",
        "has_text",
        "text",
    ],
)


if len(manifest) != EXPECTED_ROLE_NODES:

    raise AssertionError(
        "Description-input population changed."
    )


if manifest["node_id"].duplicated().any():

    raise AssertionError(
        "Duplicate node_id in frozen description manifest."
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


print(
    f"Frozen role nodes: "
    f"{len(manifest):,}"
)


# =============================================================================
# 4. Apply frozen tokenizer
# =============================================================================

banner("APPLYING FROZEN TOKENIZER")


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


# =============================================================================
# 5. Build role-specific vocabularies
#
# The paper pretrains the two Doc2Vec corpora separately.
# Therefore min_count is evaluated independently per role.
# =============================================================================

banner("BUILDING FROZEN ROLE-SPECIFIC VOCABULARIES")


role_counters = {}

vocabulary_frames = []

vocabulary_hashes = {}


for role in [
    "investor",
    "startup",
]:

    role_df = manifest.loc[
        manifest["node_type"].eq(role)
    ]

    counter = Counter()

    for tokens in role_df["tokens"]:
        counter.update(tokens)

    role_counters[role] = counter


    vocab_df = build_vocabulary_df(
        counter=counter,
        role=role,
        min_count=DOC2VEC_CONFIG[
            "min_count"
        ],
    )


    vocabulary_frames.append(
        vocab_df
    )


    vocab_hash = sha256_vocabulary(
        counter,
        DOC2VEC_CONFIG[
            "min_count"
        ],
    )

    vocabulary_hashes[role] = (
        vocab_hash
    )


    print()
    print(
        f"{role.upper()} retained vocabulary: "
        f"{len(vocab_df):,}"
    )

    print(
        f"{role.upper()} vocabulary SHA256:   "
        f"{vocab_hash}"
    )


investor_vocab_df = (
    vocabulary_frames[0]
)

startup_vocab_df = (
    vocabulary_frames[1]
)


if (
    len(investor_vocab_df)
    != EXPECTED_INVESTOR_VOCAB
):

    raise AssertionError(
        "Investor retained vocabulary changed."
    )


if (
    len(startup_vocab_df)
    != EXPECTED_STARTUP_VOCAB
):

    raise AssertionError(
        "Startup retained vocabulary changed."
    )


# =============================================================================
# 6. Calculate final Doc2Vec eligibility
# =============================================================================

banner("FINAL DOCUMENT ELIGIBILITY")


retained_vocab = {
    role: {
        token
        for token, count
        in counter.items()
        if count
        >= DOC2VEC_CONFIG[
            "min_count"
        ]
    }
    for role, counter
    in role_counters.items()
}


def retained_count(row):

    vocab = retained_vocab[
        row.node_type
    ]

    return sum(
        token in vocab
        for token in row.tokens
    )


manifest[
    "retained_token_count"
] = [
    retained_count(row)
    for row in manifest[
        [
            "node_type",
            "tokens",
        ]
    ].itertuples(
        index=False
    )
]


def text_status(row):

    if not row.has_text:

        return "SOURCE_MISSING"

    if (
        row.token_count_before_min_count
        == 0
    ):

        return "TOKENIZER_EMPTY"

    if (
        row.retained_token_count
        == 0
    ):

        return "MIN_COUNT_EMPTY"

    return "ELIGIBLE"


manifest[
    "doc2vec_text_status"
] = [
    text_status(row)
    for row in manifest[
        [
            "has_text",
            "token_count_before_min_count",
            "retained_token_count",
        ]
    ].itertuples(
        index=False
    )
]


manifest[
    "doc2vec_zero_vector"
] = (
    manifest[
        "doc2vec_text_status"
    ]
    .ne("ELIGIBLE")
)


# =============================================================================
# 7. Verify exact expected counts
# =============================================================================

summary = (
    manifest
    .groupby(
        [
            "node_type",
            "doc2vec_text_status",
        ],
        dropna=False,
    )
    .size()
    .reset_index(
        name="entities"
    )
)


print(
    summary.to_string(
        index=False
    )
)


def count_status(
    role,
    status,
):

    mask = (
        manifest[
            "node_type"
        ].eq(role)
        & manifest[
            "doc2vec_text_status"
        ].eq(status)
    )

    return int(
        mask.sum()
    )


# Investor
assert (
    count_status(
        "investor",
        "SOURCE_MISSING",
    )
    == EXPECTED_SOURCE_MISSING_INVESTOR
)

assert (
    count_status(
        "investor",
        "TOKENIZER_EMPTY",
    )
    == EXPECTED_TOKENIZER_EMPTY_INVESTOR
)

assert (
    count_status(
        "investor",
        "MIN_COUNT_EMPTY",
    )
    == EXPECTED_MINCOUNT_EMPTY_INVESTOR
)

assert (
    count_status(
        "investor",
        "ELIGIBLE",
    )
    == EXPECTED_ELIGIBLE_INVESTOR
)


# Startup
assert (
    count_status(
        "startup",
        "SOURCE_MISSING",
    )
    == EXPECTED_SOURCE_MISSING_STARTUP
)

assert (
    count_status(
        "startup",
        "TOKENIZER_EMPTY",
    )
    == EXPECTED_TOKENIZER_EMPTY_STARTUP
)

assert (
    count_status(
        "startup",
        "MIN_COUNT_EMPTY",
    )
    == EXPECTED_MINCOUNT_EMPTY_STARTUP
)

assert (
    count_status(
        "startup",
        "ELIGIBLE",
    )
    == EXPECTED_ELIGIBLE_STARTUP
)


zero_count = int(
    manifest[
        "doc2vec_zero_vector"
    ].sum()
)

eligible_count = int(
    manifest[
        "doc2vec_text_status"
    ]
    .eq("ELIGIBLE")
    .sum()
)


assert (
    zero_count
    == EXPECTED_ZERO_TEXT_FEATURE
)

assert (
    eligible_count
    == EXPECTED_ELIGIBLE_TOTAL
)


print()
print(
    f"Doc2Vec-eligible nodes: "
    f"{eligible_count:,}"
)

print(
    f"Zero-text-vector nodes: "
    f"{zero_count:,}"
)


# =============================================================================
# 8. Save frozen role vocabularies
# =============================================================================

investor_vocab_path = (
    OUT_DIR
    / "investor_doc2vec_vocabulary.csv"
)

startup_vocab_path = (
    OUT_DIR
    / "startup_doc2vec_vocabulary.csv"
)


investor_vocab_df.to_csv(
    investor_vocab_path,
    index=False,
)

startup_vocab_df.to_csv(
    startup_vocab_path,
    index=False,
)


# =============================================================================
# 9. Save eligibility manifest
#
# Do not serialize the large token-list column here. Training will reproduce
# tokenization from the frozen tokenizer definition.
# =============================================================================

eligibility_path = (
    OUT_DIR
    / "doc2vec_text_eligibility.parquet"
)


eligibility_df = manifest[
    [
        "node_id",
        "node_type",
        "raw_entity_id",
        "has_text",
        "token_count_before_min_count",
        "retained_token_count",
        "doc2vec_text_status",
        "doc2vec_zero_vector",
    ]
].copy()


eligibility_df.to_parquet(
    eligibility_path,
    index=False,
)


# =============================================================================
# 10. Environment snapshot
# =============================================================================

environment = {
    "python_version":
        sys.version,

    "platform":
        platform.platform(),

    "numpy_version":
        np.__version__,

    "scipy_version":
        scipy.__version__,

    "gensim_version":
        gensim.__version__,

    "doc2vec_fast_version":
        int(
            doc2vec.FAST_VERSION
        ),

    "pythonhashseed":
        os.environ.get(
            "PYTHONHASHSEED"
        ),
}


environment_path = (
    OUT_DIR
    / "doc2vec_environment.json"
)


with open(
    environment_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        environment,
        f,
        indent=2,
        ensure_ascii=False,
    )


# =============================================================================
# 11. Frozen Doc2Vec contract
# =============================================================================

contract = {
    "phase": "4.2.1c",

    "status": "FROZEN",

    "component":
        "ITRS description text pretraining",

    "implementation": {
        "library":
            "gensim",

        "version":
            "4.4.0",

        "optimized_doc2vec":
            True,
    },

    "paper_specified": {
        "separate_investor_and_startup_models":
            True,

        "vector_size":
            32,

        "window":
            3,
    },

    "text_preprocessing": {
        "tokenizer":
            "unicode_compound",

        "regex":
            TOKEN_RE.pattern,

        "case_normalization":
            "Unicode casefold",

        "preserve_internal_hyphens":
            True,

        "preserve_internal_apostrophes":
            True,

        "remove_stopwords":
            False,

        "stemming":
            False,

        "lemmatization":
            False,

        "remove_numeric_tokens":
            False,

        "remove_urls":
            False,

        "remove_emails":
            False,
    },

    "paper_unspecified_reproduction_choices":
        DOC2VEC_CONFIG,

    "algorithm_interpretation": {
        "name":
            "PV-DM",

        "dm":
            1,

        "dm_mean":
            1,

        "dm_concat":
            0,

        "reason":
            (
                "ITRS reports a window size but does not "
                "identify the Doc2Vec variant. PV-DM is "
                "used as an explicit reproduction choice."
            ),
    },

    "vocabulary": {
        "frequency_scope":
            "role-specific",

        "min_count":
            5,

        "investor_retained_tokens":
            int(
                len(
                    investor_vocab_df
                )
            ),

        "startup_retained_tokens":
            int(
                len(
                    startup_vocab_df
                )
            ),

        "investor_vocabulary_sha256":
            vocabulary_hashes[
                "investor"
            ],

        "startup_vocabulary_sha256":
            vocabulary_hashes[
                "startup"
            ],
    },

    "document_eligibility": {
        "eligible":
            eligible_count,

        "zero_vector":
            zero_count,

        "zero_vector_policy":
            (
                "Any node with source-missing text, "
                "tokenizer-empty text, or zero retained "
                "tokens after role-specific min_count "
                "receives an exact 32-dimensional zero "
                "Doc2Vec text vector in model-ready "
                "features."
            ),
    },

    "reproducibility": {
        "seed":
            42,

        "workers":
            1,

        "pythonhashseed":
            "42",
    },

    "temporal_provenance":
        "current_snapshot_unversioned",

    "transductive_side_information": {
        "enabled":
            True,

        "description":
            (
                "Doc2Vec vocabulary/pretraining consumes "
                "the frozen entity-description universe, "
                "including entities that may appear in "
                "validation/test. It consumes no holdout "
                "interaction labels or T60 investment "
                "edges."
            ),
    },

    "not_yet_frozen": [
        "description text MLP architecture",
        "description label MLP architecture",
        "text-vs-label output dimension split",
    ],

    "phase_2_reopened":
        False,

    "phase_3_reopened":
        False,

    "phase_4_1_3_reopened":
        False,
}


contract_path = (
    OUT_DIR
    / "doc2vec_contract.json"
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
# 12. Save audit summary
# =============================================================================

audit_path = (
    OUT_DIR
    / "doc2vec_contract_audit.csv"
)


audit_records = [
    {
        "metric":
            "frozen_role_nodes",
        "value":
            EXPECTED_ROLE_NODES,
    },
    {
        "metric":
            "investor_vocabulary",
        "value":
            len(
                investor_vocab_df
            ),
    },
    {
        "metric":
            "startup_vocabulary",
        "value":
            len(
                startup_vocab_df
            ),
    },
    {
        "metric":
            "doc2vec_eligible_nodes",
        "value":
            eligible_count,
    },
    {
        "metric":
            "zero_text_vector_nodes",
        "value":
            zero_count,
    },
    {
        "metric":
            "vector_size",
        "value":
            32,
    },
    {
        "metric":
            "window",
        "value":
            3,
    },
    {
        "metric":
            "min_count",
        "value":
            5,
    },
    {
        "metric":
            "epochs",
        "value":
            5,
    },
]


pd.DataFrame(
    audit_records
).to_csv(
    audit_path,
    index=False,
)


# =============================================================================
# 13. Final summary
# =============================================================================

banner(
    "PHASE 4.2.1c FINAL SUMMARY"
)


print(
    f"Investor vocabulary:       "
    f"{len(investor_vocab_df):,}"
)

print(
    f"Startup vocabulary:        "
    f"{len(startup_vocab_df):,}"
)

print(
    f"Doc2Vec-eligible nodes:    "
    f"{eligible_count:,}"
)

print(
    f"Zero text vectors:         "
    f"{zero_count:,}"
)

print()
print(
    "Doc2Vec algorithm:         "
    "PV-DM"
)

print(
    "Vector size:               "
    "32"
)

print(
    "Window:                    "
    "3"
)

print(
    "min_count:                 "
    "5"
)

print(
    "epochs:                    "
    "5"
)

print(
    "workers:                   "
    "1"
)

print(
    "seed:                      "
    "42"
)

print(
    "PYTHONHASHSEED:             "
    "42"
)

print()
print("Outputs:")

for path in [
    investor_vocab_path,
    startup_vocab_path,
    eligibility_path,
    environment_path,
    contract_path,
    audit_path,
]:
    print(f"  {path}")


print()
print(
    "Tokenizer:                 FROZEN"
)

print(
    "Doc2Vec configuration:     FROZEN"
)

print(
    "Role-specific vocabulary:  FROZEN"
)

print(
    "Zero-vector policy:        FROZEN"
)

print(
    "Doc2Vec models trained:    NO"
)

print()
print(
    "PHASE 4.2.1c STATUS: COMPLETE — "
    "DOC2VEC CONTRACT FROZEN"
)