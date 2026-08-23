from pathlib import Path
import hashlib
import json
import os
import re
import sys
import time

import gensim
import numpy as np
import pandas as pd
import scipy

from gensim.models.doc2vec import (
    Doc2Vec,
    TaggedDocument,
)
from gensim.models import doc2vec


# =============================================================================
# PHASE 4.2.2 — TRAIN AND AUDIT ROLE-SPECIFIC DOC2VEC MODELS
# =============================================================================

DESCRIPTION_MANIFEST_PATH = Path(
    "data/experimental/phase_4/"
    "description_contract/"
    "description_input_manifest.parquet"
)

ELIGIBILITY_PATH = Path(
    "data/experimental/phase_4/"
    "doc2vec_contract/"
    "doc2vec_text_eligibility.parquet"
)

CONTRACT_PATH = Path(
    "data/experimental/phase_4/"
    "doc2vec_contract/"
    "doc2vec_contract.json"
)

INVESTOR_VOCAB_PATH = Path(
    "data/experimental/phase_4/"
    "doc2vec_contract/"
    "investor_doc2vec_vocabulary.csv"
)

STARTUP_VOCAB_PATH = Path(
    "data/experimental/phase_4/"
    "doc2vec_contract/"
    "startup_doc2vec_vocabulary.csv"
)


OUT_DIR = Path(
    "data/experimental/phase_4/doc2vec"
)

MODEL_DIR = OUT_DIR / "models"
VECTOR_DIR = OUT_DIR / "vectors"
AUDIT_DIR = OUT_DIR / "audits"

for directory in [
    OUT_DIR,
    MODEL_DIR,
    VECTOR_DIR,
    AUDIT_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# Frozen expectations
# =============================================================================

EXPECTED_ROLE_NODES = 477_564

EXPECTED_INVESTORS = 165_975
EXPECTED_STARTUPS = 311_589

EXPECTED_INVESTOR_ELIGIBLE = 163_531
EXPECTED_STARTUP_ELIGIBLE = 311_363

EXPECTED_INVESTOR_ZERO = 2_444
EXPECTED_STARTUP_ZERO = 226

EXPECTED_ELIGIBLE_TOTAL = 474_894
EXPECTED_ZERO_TOTAL = 2_670

EXPECTED_INVESTOR_VOCAB = 36_505
EXPECTED_STARTUP_VOCAB = 57_605

VECTOR_SIZE = 32

DETERMINISM_SAMPLE_DOCS = 2_000


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
# Helpers
# =============================================================================

def banner(title):

    print()
    print("=" * 120)
    print(title)
    print("=" * 120)


def sha256_file(path):

    digest = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def hash_numpy_array(array):

    array = np.ascontiguousarray(
        array
    )

    digest = hashlib.sha256()

    digest.update(
        str(array.shape).encode(
            "ascii"
        )
    )

    digest.update(b"\n")

    digest.update(
        str(array.dtype).encode(
            "ascii"
        )
    )

    digest.update(b"\n")

    digest.update(
        array.tobytes(
            order="C"
        )
    )

    return digest.hexdigest()


def model_file_family(model_path):
    """
    Gensim may save large arrays into sidecar files.

    Collect every file beginning with the model basename so
    the complete saved model artifact is hashed.
    """

    prefix = model_path.name

    return sorted(
        [
            path
            for path in model_path.parent.iterdir()
            if path.is_file()
            and path.name.startswith(prefix)
        ],
        key=lambda p: p.name,
    )


# =============================================================================
# Re-iterable corpus
# =============================================================================

class RoleCorpus:

    def __init__(
        self,
        frame,
    ):

        self.node_ids = (
            frame["node_id"]
            .astype(str)
            .tolist()
        )

        self.texts = (
            frame["text"]
            .tolist()
        )


    def __len__(self):

        return len(
            self.node_ids
        )


    def __iter__(self):

        for node_id, text in zip(
            self.node_ids,
            self.texts,
        ):

            tokens = tokenize(
                text
            )

            yield TaggedDocument(
                words=tokens,
                tags=[node_id],
            )


# =============================================================================
# Load frozen contract
# =============================================================================

banner(
    "PHASE 4.2.2 — "
    "TRAIN AND AUDIT ROLE-SPECIFIC DOC2VEC MODELS"
)


if (
    os.environ.get(
        "PYTHONHASHSEED"
    )
    != "42"
):

    raise AssertionError(
        "Run with PYTHONHASHSEED=42."
    )


with open(
    CONTRACT_PATH,
    "r",
    encoding="utf-8",
) as f:

    frozen_contract = json.load(f)


if (
    frozen_contract.get(
        "status"
    )
    != "FROZEN"
):

    raise AssertionError(
        "Doc2Vec contract is not frozen."
    )


config = (
    frozen_contract[
        "paper_unspecified_reproduction_choices"
    ]
    .copy()
)


# Paper-specified values are stored separately in the contract.
config[
    "vector_size"
] = (
    frozen_contract[
        "paper_specified"
    ][
        "vector_size"
    ]
)

config[
    "window"
] = (
    frozen_contract[
        "paper_specified"
    ][
        "window"
    ]
)


if config["vector_size"] != 32:
    raise AssertionError(
        "Frozen vector_size changed."
    )

if config["window"] != 3:
    raise AssertionError(
        "Frozen window changed."
    )


print("\nFROZEN DOC2VEC CONFIGURATION")
print("-" * 120)

for key in sorted(config):

    print(
        f"{key:<25} "
        f"{config[key]}"
    )


# =============================================================================
# Environment integrity
# =============================================================================

banner("ENVIRONMENT INTEGRITY")


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
        "Gensim version differs from frozen contract."
    )

if doc2vec.FAST_VERSION < 0:
    raise AssertionError(
        "Optimized Doc2Vec routines unavailable."
    )


# =============================================================================
# Load frozen manifests
# =============================================================================

banner("LOADING FROZEN INPUT MANIFESTS")


description = pd.read_parquet(
    DESCRIPTION_MANIFEST_PATH,
    columns=[
        "node_id",
        "node_type",
        "raw_entity_id",
        "text",
    ],
)


eligibility = pd.read_parquet(
    ELIGIBILITY_PATH,
    columns=[
        "node_id",
        "node_type",
        "doc2vec_text_status",
        "doc2vec_zero_vector",
    ],
)


if len(description) != EXPECTED_ROLE_NODES:
    raise AssertionError(
        "Description manifest population changed."
    )


if len(eligibility) != EXPECTED_ROLE_NODES:
    raise AssertionError(
        "Eligibility population changed."
    )


if (
    description[
        "node_id"
    ].duplicated().any()
):

    raise AssertionError(
        "Duplicate node IDs in description manifest."
    )


if (
    eligibility[
        "node_id"
    ].duplicated().any()
):

    raise AssertionError(
        "Duplicate node IDs in eligibility manifest."
    )


manifest = description.merge(
    eligibility[
        [
            "node_id",
            "node_type",
            "doc2vec_text_status",
            "doc2vec_zero_vector",
        ]
    ],
    on=[
        "node_id",
        "node_type",
    ],
    how="left",
    validate="one_to_one",
)


if (
    manifest[
        "doc2vec_text_status"
    ].isna().any()
):

    raise AssertionError(
        "Eligibility attachment failed."
    )


# Explicit feature row order.
manifest[
    "doc2vec_feature_row"
] = np.arange(
    len(manifest),
    dtype=np.int64,
)


print(
    f"Frozen role nodes: "
    f"{len(manifest):,}"
)


# =============================================================================
# Verify eligibility counts
# =============================================================================

banner("ELIGIBILITY INTEGRITY")


eligibility_summary = (
    manifest
    .groupby(
        [
            "node_type",
            "doc2vec_text_status",
        ]
    )
    .size()
    .reset_index(
        name="entities"
    )
)


print(
    eligibility_summary.to_string(
        index=False
    )
)


def role_status_count(
    role,
    status,
):

    return int(
        (
            manifest[
                "node_type"
            ].eq(role)
            & manifest[
                "doc2vec_text_status"
            ].eq(status)
        ).sum()
    )


investor_eligible_count = (
    role_status_count(
        "investor",
        "ELIGIBLE",
    )
)

startup_eligible_count = (
    role_status_count(
        "startup",
        "ELIGIBLE",
    )
)


investor_zero_count = int(
    (
        manifest[
            "node_type"
        ].eq("investor")
        & manifest[
            "doc2vec_zero_vector"
        ]
    ).sum()
)

startup_zero_count = int(
    (
        manifest[
            "node_type"
        ].eq("startup")
        & manifest[
            "doc2vec_zero_vector"
        ]
    ).sum()
)


assert (
    investor_eligible_count
    == EXPECTED_INVESTOR_ELIGIBLE
)

assert (
    startup_eligible_count
    == EXPECTED_STARTUP_ELIGIBLE
)

assert (
    investor_zero_count
    == EXPECTED_INVESTOR_ZERO
)

assert (
    startup_zero_count
    == EXPECTED_STARTUP_ZERO
)


# =============================================================================
# Load frozen vocabulary manifests
# =============================================================================

banner("LOADING FROZEN ROLE VOCABULARIES")


frozen_vocab = {
    "investor":
        pd.read_csv(
            INVESTOR_VOCAB_PATH,
            keep_default_na=False,
            na_filter=False,
            dtype={
                "node_type": "string",
                "frequency_rank": "int64",
                "token": "string",
                "frequency": "int64",
            },
        ),

    "startup":
        pd.read_csv(
            STARTUP_VOCAB_PATH,
            keep_default_na=False,
            na_filter=False,
            dtype={
                "node_type": "string",
                "frequency_rank": "int64",
                "token": "string",
                "frequency": "int64",
            },
        ),
}


# =============================================================================
# Vocabulary lexical-integrity guard
#
# Tokens are arbitrary strings. Values such as "nan" and "null" are legitimate
# vocabulary items and must never be interpreted as missing values by Pandas.
# =============================================================================

for role, vocab_df in frozen_vocab.items():

    if vocab_df["token"].isna().any():
        raise AssertionError(
            f"{role}: vocabulary token column "
            "contains parsed NA values."
        )

    if vocab_df["token"].eq("").any():
        raise AssertionError(
            f"{role}: empty vocabulary token detected."
        )


print()
print("LEXICAL CSV READBACK INTEGRITY")

for role, vocab_df in frozen_vocab.items():

    print(
        f"{role:<10} "
        f"rows={len(vocab_df):,} "
        f"parsed_NA={int(vocab_df['token'].isna().sum()):,}"
    )


print()
print("Known NA-like lexical tokens:")

for role, vocab_df in frozen_vocab.items():

    known = vocab_df.loc[
        vocab_df["token"].isin(
            [
                "nan",
                "null",
                "NA",
                "N/A",
                "None",
            ]
        ),
        [
            "frequency_rank",
            "token",
            "frequency",
        ],
    ]

    print()
    print(role.upper())

    if known.empty:
        print("  none")
    else:
        print(
            known.to_string(
                index=False
            )
        )


assert (
    len(
        frozen_vocab[
            "investor"
        ]
    )
    == EXPECTED_INVESTOR_VOCAB
)

assert (
    len(
        frozen_vocab[
            "startup"
        ]
    )
    == EXPECTED_STARTUP_VOCAB
)


# =============================================================================
# Exact small-sample deterministic training audit
# =============================================================================

banner(
    "DETERMINISM CHECK — "
    "FIXED 2,000-DOCUMENT SAMPLE"
)


determinism_records = []


def train_small_model(
    sample_df,
):

    corpus = RoleCorpus(
        sample_df
    )

    model = Doc2Vec(
        **config
    )

    model.build_vocab(
        corpus
    )

    model.train(
        corpus,
        total_examples=
            model.corpus_count,
        epochs=
            model.epochs,
    )

    return model


for role in [
    "investor",
    "startup",
]:

    role_sample = (
        manifest.loc[
            manifest[
                "node_type"
            ].eq(role)
            & manifest[
                "doc2vec_text_status"
            ].eq("ELIGIBLE")
        ]
        .head(
            DETERMINISM_SAMPLE_DOCS
        )
        .copy()
    )


    if (
        len(role_sample)
        != DETERMINISM_SAMPLE_DOCS
    ):

        raise AssertionError(
            "Insufficient documents for determinism sample."
        )


    print()
    print(
        f"{role.upper()} "
        f"determinism sample..."
    )


    model_a = train_small_model(
        role_sample
    )

    model_b = train_small_model(
        role_sample
    )


    same_dv_keys = (
        model_a.dv.index_to_key
        == model_b.dv.index_to_key
    )

    exact_document_vectors = (
        np.array_equal(
            model_a.dv.vectors,
            model_b.dv.vectors,
        )
    )

    exact_word_vectors = (
        model_a.wv.index_to_key
        == model_b.wv.index_to_key
        and np.array_equal(
            model_a.wv.vectors,
            model_b.wv.vectors,
        )
    )


    determinism_records.append(
        {
            "node_type":
                role,

            "sample_documents":
                len(role_sample),

            "same_document_tag_order":
                same_dv_keys,

            "exact_document_vectors":
                exact_document_vectors,

            "exact_word_vectors":
                exact_word_vectors,
        }
    )


    print(
        f"  document tag order: "
        f"{same_dv_keys}"
    )

    print(
        f"  document vectors exact: "
        f"{exact_document_vectors}"
    )

    print(
        f"  word vectors exact: "
        f"{exact_word_vectors}"
    )


    if not (
        same_dv_keys
        and exact_document_vectors
        and exact_word_vectors
    ):

        raise AssertionError(
            f"{role} Doc2Vec determinism "
            f"check failed."
        )


    del model_a
    del model_b


determinism_df = pd.DataFrame(
    determinism_records
)


# =============================================================================
# Training helper
# =============================================================================

def train_role_model(
    role,
):

    banner(
        f"TRAINING {role.upper()} DOC2VEC"
    )


    role_all = manifest.loc[
        manifest[
            "node_type"
        ].eq(role)
    ].copy()


    role_eligible = role_all.loc[
        role_all[
            "doc2vec_text_status"
        ].eq("ELIGIBLE")
    ].copy()


    expected_docs = {
        "investor":
            EXPECTED_INVESTOR_ELIGIBLE,

        "startup":
            EXPECTED_STARTUP_ELIGIBLE,
    }[
        role
    ]


    if (
        len(role_eligible)
        != expected_docs
    ):

        raise AssertionError(
            f"{role} eligible-document count changed."
        )


    corpus = RoleCorpus(
        role_eligible
    )


    model = Doc2Vec(
        **config
    )


    # -------------------------------------------------------------------------
    # Build vocabulary
    # -------------------------------------------------------------------------

    print(
        f"Building {role} vocabulary..."
    )

    build_start = time.time()


    model.build_vocab(
        corpus
    )


    build_seconds = (
        time.time()
        - build_start
    )


    print(
        f"Vocabulary built in "
        f"{build_seconds:.2f} s"
    )

    print(
        f"Gensim vocabulary: "
        f"{len(model.wv):,}"
    )

    print(
        f"Document tags:    "
        f"{len(model.dv):,}"
    )


    # -------------------------------------------------------------------------
    # Frozen vocabulary exactness audit
    # -------------------------------------------------------------------------

    frozen = (
        frozen_vocab[
            role
        ]
        .copy()
    )


    frozen_tokens = set(
        frozen["token"]
        .astype(str)
    )

    gensim_tokens = set(
        model.wv.index_to_key
    )


    missing_from_gensim = (
        frozen_tokens
        - gensim_tokens
    )

    extra_in_gensim = (
        gensim_tokens
        - frozen_tokens
    )


    print(
        f"Frozen vocabulary tokens missing "
        f"from Gensim: {len(missing_from_gensim):,}"
    )

    print(
        f"Unexpected Gensim vocabulary tokens: "
        f"{len(extra_in_gensim):,}"
    )


    if missing_from_gensim:

        print()
        print("Missing frozen tokens:")

        for token in sorted(
            missing_from_gensim
        ):
            print(repr(token))


    if extra_in_gensim:

        print()
        print("Unexpected Gensim tokens:")

        for token in sorted(
            extra_in_gensim
        ):
            print(repr(token))


    # -------------------------------------------------------------------------
    # Frequency exactness
    # -------------------------------------------------------------------------

    frozen_frequency = dict(
        zip(
            frozen["token"].astype(str),
            frozen["frequency"].astype(int),
        )
    )


    frequency_mismatches = []


    for token in model.wv.index_to_key:

        actual = int(
            model.wv.get_vecattr(
                token,
                "count",
            )
        )

        expected = (
            frozen_frequency[
                token
            ]
        )


        if actual != expected:

            frequency_mismatches.append(
                (
                    token,
                    expected,
                    actual,
                )
            )


    print(
        f"Vocabulary frequency mismatches: "
        f"{len(frequency_mismatches):,}"
    )


    if frequency_mismatches:

        print(
            frequency_mismatches[
                :20
            ]
        )

        raise AssertionError(
            f"{role}: vocabulary frequencies "
            "differ from frozen contract."
        )


    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------

    print()
    print(
        f"Training {role} model "
        f"for {model.epochs} epochs..."
    )


    train_start = time.time()


    model.train(
        corpus,
        total_examples=
            model.corpus_count,
        epochs=
            model.epochs,
    )


    train_seconds = (
        time.time()
        - train_start
    )


    print(
        f"Training completed in "
        f"{train_seconds:.2f} s"
    )


    # -------------------------------------------------------------------------
    # Model vector integrity
    # -------------------------------------------------------------------------

    if (
        model.dv.vectors.shape
        != (
            expected_docs,
            VECTOR_SIZE,
        )
    ):

        raise AssertionError(
            f"{role}: unexpected document "
            f"vector matrix shape "
            f"{model.dv.vectors.shape}."
        )


    if not np.isfinite(
        model.dv.vectors
    ).all():

        raise AssertionError(
            f"{role}: non-finite document vectors."
        )


    if not np.isfinite(
        model.wv.vectors
    ).all():

        raise AssertionError(
            f"{role}: non-finite word vectors."
        )


    # Exact tag identity audit.
    expected_tags = set(
        role_eligible[
            "node_id"
        ].astype(str)
    )

    actual_tags = set(
        model.dv.index_to_key
    )


    if (
        expected_tags
        != actual_tags
    ):

        raise AssertionError(
            f"{role}: document-tag set mismatch."
        )


    # -------------------------------------------------------------------------
    # Extract model-ready vectors for ALL role nodes
    # -------------------------------------------------------------------------

    role_vectors = np.zeros(
        (
            len(role_all),
            VECTOR_SIZE,
        ),
        dtype=np.float32,
    )


    node_to_local_row = {
        node_id: idx
        for idx, node_id
        in enumerate(
            role_all[
                "node_id"
            ].astype(str)
        )
    }


    for node_id in role_eligible[
        "node_id"
    ].astype(str):

        local_row = (
            node_to_local_row[
                node_id
            ]
        )

        role_vectors[
            local_row
        ] = model.dv[
            node_id
        ]


    # -------------------------------------------------------------------------
    # Exact zero policy
    # -------------------------------------------------------------------------

    expected_zero_mask = (
        role_all[
            "doc2vec_zero_vector"
        ].to_numpy(
            dtype=bool
        )
    )


    actual_zero_mask = np.all(
        role_vectors == 0.0,
        axis=1,
    )


    zero_policy_mismatches = int(
        np.sum(
            expected_zero_mask
            != actual_zero_mask
        )
    )


    print(
        f"Zero-policy mismatches: "
        f"{zero_policy_mismatches:,}"
    )


    if zero_policy_mismatches:

        raise AssertionError(
            f"{role}: zero-vector policy mismatch."
        )


    learned_mask = (
        ~expected_zero_mask
    )


    learned_norms = np.linalg.norm(
        role_vectors[
            learned_mask
        ],
        axis=1,
    )


    if np.any(
        learned_norms == 0
    ):

        raise AssertionError(
            f"{role}: eligible document "
            "received a zero vector."
        )


    if not np.isfinite(
        role_vectors
    ).all():

        raise AssertionError(
            f"{role}: model-ready vector matrix "
            "contains NaN/Inf."
        )


    # -------------------------------------------------------------------------
    # Save model and vectors
    # -------------------------------------------------------------------------

    model_path = (
        MODEL_DIR
        / f"{role}_doc2vec.model"
    )

    vector_path = (
        VECTOR_DIR
        / f"{role}_doc2vec_vectors.npy"
    )


    model.save(
        str(
            model_path
        )
    )


    np.save(
        vector_path,
        role_vectors,
    )


    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    summary = {
        "node_type":
            role,

        "role_nodes":
            int(
                len(role_all)
            ),

        "eligible_documents":
            int(
                len(role_eligible)
            ),

        "zero_vector_nodes":
            int(
                expected_zero_mask.sum()
            ),

        "gensim_vocabulary":
            int(
                len(model.wv)
            ),

        "document_vectors":
            int(
                len(model.dv)
            ),

        "vector_size":
            VECTOR_SIZE,

        "vocab_build_seconds":
            build_seconds,

        "training_seconds":
            train_seconds,

        "minimum_learned_norm":
            float(
                learned_norms.min()
            ),

        "median_learned_norm":
            float(
                np.median(
                    learned_norms
                )
            ),

        "maximum_learned_norm":
            float(
                learned_norms.max()
            ),

        "vector_array_sha256":
            hash_numpy_array(
                role_vectors
            ),
    }


    return (
        model,
        role_all,
        role_vectors,
        model_path,
        vector_path,
        summary,
    )


# =============================================================================
# Train independent Investor model
# =============================================================================

(
    investor_model,
    investor_manifest,
    investor_vectors,
    investor_model_path,
    investor_vector_path,
    investor_summary,
) = train_role_model(
    "investor"
)


# =============================================================================
# Train independent Startup model
# =============================================================================

(
    startup_model,
    startup_manifest,
    startup_vectors,
    startup_model_path,
    startup_vector_path,
    startup_summary,
) = train_role_model(
    "startup"
)


# =============================================================================
# Verify models are genuinely separate
# =============================================================================

banner("CROSS-ROLE MODEL-SEPARATION AUDIT")


if (
    investor_model
    is startup_model
):

    raise AssertionError(
        "Investor and Startup models "
        "must be independent objects."
    )


print(
    "Independent model objects: PASS"
)

print(
    f"Investor vocabulary: "
    f"{len(investor_model.wv):,}"
)

print(
    f"Startup vocabulary:  "
    f"{len(startup_model.wv):,}"
)


# A shared token should generally have different learned vectors
# because the two models are independently initialized/trained.
shared_tokens = (
    set(
        investor_model.wv.index_to_key
    )
    & set(
        startup_model.wv.index_to_key
    )
)


if not shared_tokens:

    raise AssertionError(
        "Unexpected: role vocabularies share no tokens."
    )


audit_token = sorted(
    shared_tokens
)[0]


same_shared_word_vector = (
    np.array_equal(
        investor_model.wv[
            audit_token
        ],
        startup_model.wv[
            audit_token
        ],
    )
)


print(
    f"Audit shared token: "
    f"{repr(audit_token)}"
)

print(
    f"Exact same vector across roles: "
    f"{same_shared_word_vector}"
)


if same_shared_word_vector:

    raise AssertionError(
        "Independent role models unexpectedly "
        "produced identical audit word vector."
    )


# =============================================================================
# Build combined vector matrix in explicit description-manifest order
# =============================================================================

banner("BUILDING COMBINED DOC2VEC FEATURE MATRIX")


combined_vectors = np.zeros(
    (
        EXPECTED_ROLE_NODES,
        VECTOR_SIZE,
    ),
    dtype=np.float32,
)


investor_rows = (
    manifest[
        "node_type"
    ].eq(
        "investor"
    )
    .to_numpy()
)


startup_rows = (
    manifest[
        "node_type"
    ].eq(
        "startup"
    )
    .to_numpy()
)


if (
    investor_rows.sum()
    != EXPECTED_INVESTORS
):

    raise AssertionError(
        "Investor combined-row count mismatch."
    )


if (
    startup_rows.sum()
    != EXPECTED_STARTUPS
):

    raise AssertionError(
        "Startup combined-row count mismatch."
    )


combined_vectors[
    investor_rows
] = investor_vectors

combined_vectors[
    startup_rows
] = startup_vectors


if (
    combined_vectors.shape
    != (
        EXPECTED_ROLE_NODES,
        VECTOR_SIZE,
    )
):

    raise AssertionError(
        "Combined vector shape mismatch."
    )


if not np.isfinite(
    combined_vectors
).all():

    raise AssertionError(
        "Combined vectors contain NaN/Inf."
    )


expected_combined_zero = (
    manifest[
        "doc2vec_zero_vector"
    ].to_numpy(
        dtype=bool
    )
)


actual_combined_zero = np.all(
    combined_vectors == 0.0,
    axis=1,
)


if not np.array_equal(
    expected_combined_zero,
    actual_combined_zero,
):

    raise AssertionError(
        "Combined zero-vector policy mismatch."
    )


print(
    f"Combined vector shape: "
    f"{combined_vectors.shape}"
)

print(
    f"Combined zero rows:    "
    f"{actual_combined_zero.sum():,}"
)


combined_vector_path = (
    VECTOR_DIR
    / "doc2vec_vectors_all.npy"
)


np.save(
    combined_vector_path,
    combined_vectors,
)


# =============================================================================
# Save vector-row manifest
# =============================================================================

vector_manifest_path = (
    VECTOR_DIR
    / "doc2vec_vector_manifest.parquet"
)


vector_manifest = manifest[
    [
        "doc2vec_feature_row",
        "node_id",
        "node_type",
        "raw_entity_id",
        "doc2vec_text_status",
        "doc2vec_zero_vector",
    ]
].copy()


vector_manifest.to_parquet(
    vector_manifest_path,
    index=False,
)


# =============================================================================
# Model artifact hashes
# =============================================================================

banner("ARTIFACT HASH AUDIT")


hash_records = []


for role, model_path in [
    (
        "investor",
        investor_model_path,
    ),
    (
        "startup",
        startup_model_path,
    ),
]:

    for path in model_file_family(
        model_path
    ):

        digest = sha256_file(
            path
        )

        hash_records.append(
            {
                "artifact_group":
                    f"{role}_model",

                "path":
                    str(path),

                "sha256":
                    digest,

                "bytes":
                    path.stat().st_size,
            }
        )

        print(
            f"{path.name}: "
            f"{digest}"
        )


for artifact_group, path in [
    (
        "investor_vectors",
        investor_vector_path,
    ),
    (
        "startup_vectors",
        startup_vector_path,
    ),
    (
        "combined_vectors",
        combined_vector_path,
    ),
    (
        "vector_manifest",
        vector_manifest_path,
    ),
]:

    digest = sha256_file(
        path
    )

    hash_records.append(
        {
            "artifact_group":
                artifact_group,

            "path":
                str(path),

            "sha256":
                digest,

            "bytes":
                path.stat().st_size,
        }
    )


hash_df = pd.DataFrame(
    hash_records
)


hash_path = (
    AUDIT_DIR
    / "doc2vec_artifact_hashes.csv"
)


hash_df.to_csv(
    hash_path,
    index=False,
)


# =============================================================================
# Training audit
# =============================================================================

banner("TRAINING AUDIT")


training_summary_df = pd.DataFrame(
    [
        investor_summary,
        startup_summary,
    ]
)


print(
    training_summary_df.to_string(
        index=False
    )
)


training_summary_path = (
    AUDIT_DIR
    / "doc2vec_training_audit.csv"
)


training_summary_df.to_csv(
    training_summary_path,
    index=False,
)


determinism_path = (
    AUDIT_DIR
    / "doc2vec_determinism_audit.csv"
)


determinism_df.to_csv(
    determinism_path,
    index=False,
)


# =============================================================================
# Metadata
# =============================================================================

metadata = {
    "phase":
        "4.2.2",

    "status":
        "COMPLETE",

    "models": {
        "investor": {
            "model_path":
                str(
                    investor_model_path
                ),

            "vector_path":
                str(
                    investor_vector_path
                ),

            "role_nodes":
                EXPECTED_INVESTORS,

            "eligible":
                EXPECTED_INVESTOR_ELIGIBLE,

            "zero_vectors":
                EXPECTED_INVESTOR_ZERO,

            "vocabulary":
                EXPECTED_INVESTOR_VOCAB,
        },

        "startup": {
            "model_path":
                str(
                    startup_model_path
                ),

            "vector_path":
                str(
                    startup_vector_path
                ),

            "role_nodes":
                EXPECTED_STARTUPS,

            "eligible":
                EXPECTED_STARTUP_ELIGIBLE,

            "zero_vectors":
                EXPECTED_STARTUP_ZERO,

            "vocabulary":
                EXPECTED_STARTUP_VOCAB,
        },
    },

    "combined_vectors": {
        "path":
            str(
                combined_vector_path
            ),

        "manifest":
            str(
                vector_manifest_path
            ),

        "shape": [
            EXPECTED_ROLE_NODES,
            VECTOR_SIZE,
        ],

        "zero_rows":
            EXPECTED_ZERO_TOTAL,
    },

    "training": {
        "separate_role_models":
            True,

        "vector_size":
            config[
                "vector_size"
            ],

        "window":
            config[
                "window"
            ],

        "epochs":
            config[
                "epochs"
            ],

        "workers":
            config[
                "workers"
            ],

        "seed":
            config[
                "seed"
            ],

        "pythonhashseed":
            os.environ.get(
                "PYTHONHASHSEED"
            ),
    },

    "determinism": {
        "sample_documents_per_role":
            DETERMINISM_SAMPLE_DOCS,

        "sample_repeat_training_exact":
            True,
    },

    "temporal_provenance":
        "current_snapshot_unversioned",

    "holdout_interaction_labels_consumed":
        False,

    "phase_2_reopened":
        False,

    "phase_3_reopened":
        False,

    "doc2vec_contract_reopened":
        False,
}


metadata_path = (
    OUT_DIR
    / "doc2vec_training_metadata.json"
)


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
# Final integrity
# =============================================================================

banner("PHASE 4.2.2 FINAL SUMMARY")


print(
    f"Investor model documents: "
    f"{len(investor_model.dv):,}"
)

print(
    f"Investor vocabulary:      "
    f"{len(investor_model.wv):,}"
)

print(
    f"Startup model documents:  "
    f"{len(startup_model.dv):,}"
)

print(
    f"Startup vocabulary:       "
    f"{len(startup_model.wv):,}"
)

print()
print(
    f"Combined vector shape:    "
    f"{combined_vectors.shape}"
)

print(
    f"Exact zero rows:          "
    f"{actual_combined_zero.sum():,}"
)

print(
    f"NaN/Inf:                  "
    f"{not np.isfinite(combined_vectors).all()}"
)

print()
print(
    "Separate models:             PASS"
)

print(
    "Frozen vocabularies:         PASS"
)

print(
    "Frozen frequencies:          PASS"
)

print(
    "32-D vector integrity:       PASS"
)

print(
    "Zero-vector policy:          PASS"
)

print(
    "Sample repeat determinism:   PASS"
)

print()
print("Outputs:")

for path in [
    investor_model_path,
    startup_model_path,
    investor_vector_path,
    startup_vector_path,
    combined_vector_path,
    vector_manifest_path,
    training_summary_path,
    determinism_path,
    hash_path,
    metadata_path,
]:

    print(
        f"  {path}"
    )


print()
print(
    "PHASE 4.2.2 STATUS: COMPLETE — "
    "ROLE-SPECIFIC DOC2VEC MODELS TRAINED"
)