from pathlib import Path
from collections import Counter
import json
import platform
import re
import sys

import numpy as np
import pandas as pd


# =============================================================================
# PHASE 4.2.1a — ENGLISH TEXT PREPROCESSING AUDIT
# =============================================================================

CONTRACT_DIR = Path(
    "data/experimental/phase_4/description_contract"
)

MANIFEST_PATH = (
    CONTRACT_DIR / "description_input_manifest.parquet"
)

CONTRACT_PATH = (
    CONTRACT_DIR / "description_contract.json"
)

OUT_DIR = Path(
    "data/experimental/phase_4/audits"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Frozen Phase-4.1.3 expectations
# =============================================================================

EXPECTED_ROLE_NODES = 477_564
EXPECTED_INVESTORS = 165_975
EXPECTED_STARTUPS = 311_589

EXPECTED_INVESTORS_WITH_TEXT = 165_970
EXPECTED_STARTUPS_WITH_TEXT = 311_586

EXPECTED_MISSING_TEXT = 8


# =============================================================================
# Tokenizer candidates
#
# IMPORTANT:
# These are AUDIT candidates only.
# No tokenizer is frozen by this script.
# =============================================================================

# Candidate A:
# Unicode word tokens, punctuation separates terms.
#
# Examples:
#   "AI-powered" -> ["ai", "powered"]
#   "Mexico's"   -> ["mexico", "s"]
#
UNICODE_BASIC_RE = re.compile(
    r"\b[^\W_]+\b",
    flags=re.UNICODE,
)


# Candidate B:
# Unicode word tokens while preserving internal hyphens/apostrophes.
#
# Examples:
#   "AI-powered" -> ["ai-powered"]
#   "company's"  -> ["company's"]
#
UNICODE_COMPOUND_RE = re.compile(
    r"\b[^\W_]+(?:[-'’][^\W_]+)*\b",
    flags=re.UNICODE,
)


# Candidate C:
# ASCII-only alphanumeric compound tokenizer.
#
# This is deliberately included as a comparison because it may lose accented
# or non-English text. It is NOT assumed to be desirable.
#
ASCII_COMPOUND_RE = re.compile(
    r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*"
)


URL_RE = re.compile(
    r"https?://\S+|www\.\S+",
    flags=re.IGNORECASE,
)

EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

CJK_RE = re.compile(
    "["
    "\u3400-\u4DBF"
    "\u4E00-\u9FFF"
    "\u3040-\u30FF"
    "\uAC00-\uD7AF"
    "]"
)

CYRILLIC_RE = re.compile(
    r"[\u0400-\u04FF]"
)

ARABIC_RE = re.compile(
    r"[\u0600-\u06FF]"
)


# =============================================================================
# Helpers
# =============================================================================

def banner(title):
    print()
    print("=" * 115)
    print(title)
    print("=" * 115)


def normalize_text(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    return value


def tokenize_unicode_basic(text):
    return [
        token.casefold()
        for token in UNICODE_BASIC_RE.findall(text)
    ]


def tokenize_unicode_compound(text):
    return [
        token.casefold()
        for token in UNICODE_COMPOUND_RE.findall(text)
    ]


def tokenize_ascii_compound(text):
    return [
        token.lower()
        for token in ASCII_COMPOUND_RE.findall(text)
    ]


TOKENIZERS = {
    "unicode_basic": tokenize_unicode_basic,
    "unicode_compound": tokenize_unicode_compound,
    "ascii_compound": tokenize_ascii_compound,
}


def percentile(values, q):
    if not values:
        return np.nan

    return float(
        np.quantile(
            np.asarray(values),
            q,
        )
    )


# =============================================================================
# 1. Load and verify frozen contract
# =============================================================================

banner(
    "PHASE 4.2.1a — "
    "ENGLISH TEXT PREPROCESSING AUDIT"
)

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(
        f"Frozen description manifest not found: "
        f"{MANIFEST_PATH}"
    )

if not CONTRACT_PATH.exists():
    raise FileNotFoundError(
        f"Frozen description contract not found: "
        f"{CONTRACT_PATH}"
    )


with open(
    CONTRACT_PATH,
    "r",
    encoding="utf-8",
) as f:
    contract = json.load(f)


if contract.get("status") != "FROZEN":
    raise AssertionError(
        "Description contract is not marked FROZEN."
    )


manifest = pd.read_parquet(
    MANIFEST_PATH,
    columns=[
        "node_id",
        "node_type",
        "raw_entity_id",
        "text",
        "text_source",
        "has_text",
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


if role_counts.get("investor", 0) != EXPECTED_INVESTORS:
    raise AssertionError(
        "Frozen Investor population changed."
    )

if role_counts.get("startup", 0) != EXPECTED_STARTUPS:
    raise AssertionError(
        "Frozen Startup population changed."
    )


text_counts = (
    manifest.loc[
        manifest["has_text"]
    ]
    .groupby("node_type")
    .size()
    .to_dict()
)


if (
    text_counts.get("investor", 0)
    != EXPECTED_INVESTORS_WITH_TEXT
):
    raise AssertionError(
        "Frozen Investor text coverage changed."
    )

if (
    text_counts.get("startup", 0)
    != EXPECTED_STARTUPS_WITH_TEXT
):
    raise AssertionError(
        "Frozen Startup text coverage changed."
    )


missing_count = int(
    (~manifest["has_text"]).sum()
)

if missing_count != EXPECTED_MISSING_TEXT:
    raise AssertionError(
        "Frozen missing-text count changed."
    )


print(
    f"Frozen role nodes:      {len(manifest):,}"
)
print(
    f"Investors:              "
    f"{role_counts['investor']:,}"
)
print(
    f"Startups:               "
    f"{role_counts['startup']:,}"
)
print(
    f"Documents with text:    "
    f"{manifest['has_text'].sum():,}"
)
print(
    f"Documents without text: "
    f"{missing_count:,}"
)


# =============================================================================
# 2. Environment inventory
# =============================================================================

banner("ENVIRONMENT INVENTORY")

environment = {
    "python_version": sys.version,
    "platform": platform.platform(),
    "pandas_version": pd.__version__,
    "numpy_version": np.__version__,
}


try:
    import gensim

    environment[
        "gensim_available"
    ] = True

    environment[
        "gensim_version"
    ] = gensim.__version__

except Exception as exc:

    environment[
        "gensim_available"
    ] = False

    environment[
        "gensim_import_error"
    ] = repr(exc)


for key, value in environment.items():
    print(
        f"{key:<25} {value}"
    )


# =============================================================================
# 3. Raw text-character audit
# =============================================================================

banner("RAW TEXT CHARACTER / CONTENT AUDIT")


raw_records = []
non_ascii_examples = []


for role in [
    "investor",
    "startup",
]:

    frame = manifest.loc[
        manifest["node_type"].eq(role)
        & manifest["has_text"]
    ].copy()

    docs = [
        normalize_text(value)
        for value in frame["text"]
    ]

    docs = [
        text
        for text in docs
        if text is not None
    ]


    char_lengths = []
    whitespace_word_counts = []

    docs_non_ascii = 0
    docs_with_url = 0
    docs_with_email = 0
    docs_with_digit = 0
    docs_with_hyphen = 0
    docs_with_apostrophe = 0
    docs_with_cjk = 0
    docs_with_cyrillic = 0
    docs_with_arabic = 0

    total_chars = 0
    non_ascii_chars = 0


    for node_id, text in zip(
        frame["node_id"],
        docs,
    ):

        char_lengths.append(
            len(text)
        )

        whitespace_word_counts.append(
            len(text.split())
        )

        total_chars += len(text)

        current_non_ascii = sum(
            ord(ch) > 127
            for ch in text
        )

        non_ascii_chars += (
            current_non_ascii
        )

        has_non_ascii = (
            current_non_ascii > 0
        )

        if has_non_ascii:
            docs_non_ascii += 1

            if len(non_ascii_examples) < 100:
                non_ascii_examples.append(
                    {
                        "node_id": node_id,
                        "node_type": role,
                        "text": text[:1000],
                    }
                )

        docs_with_url += bool(
            URL_RE.search(text)
        )

        docs_with_email += bool(
            EMAIL_RE.search(text)
        )

        docs_with_digit += any(
            ch.isdigit()
            for ch in text
        )

        docs_with_hyphen += (
            "-" in text
        )

        docs_with_apostrophe += (
            "'" in text
            or "’" in text
        )

        docs_with_cjk += bool(
            CJK_RE.search(text)
        )

        docs_with_cyrillic += bool(
            CYRILLIC_RE.search(text)
        )

        docs_with_arabic += bool(
            ARABIC_RE.search(text)
        )


    raw_records.append(
        {
            "node_type": role,
            "documents": len(docs),

            "median_chars":
                percentile(
                    char_lengths,
                    0.50,
                ),

            "p95_chars":
                percentile(
                    char_lengths,
                    0.95,
                ),

            "median_whitespace_words":
                percentile(
                    whitespace_word_counts,
                    0.50,
                ),

            "p95_whitespace_words":
                percentile(
                    whitespace_word_counts,
                    0.95,
                ),

            "documents_with_non_ascii":
                docs_non_ascii,

            "non_ascii_document_pct":
                100.0
                * docs_non_ascii
                / len(docs),

            "non_ascii_character_pct":
                (
                    100.0
                    * non_ascii_chars
                    / total_chars
                    if total_chars
                    else 0.0
                ),

            "documents_with_url":
                docs_with_url,

            "documents_with_email":
                docs_with_email,

            "documents_with_digit":
                docs_with_digit,

            "documents_with_hyphen":
                docs_with_hyphen,

            "documents_with_apostrophe":
                docs_with_apostrophe,

            "documents_with_cjk":
                docs_with_cjk,

            "documents_with_cyrillic":
                docs_with_cyrillic,

            "documents_with_arabic":
                docs_with_arabic,
        }
    )


raw_audit_df = pd.DataFrame(
    raw_records
)

print(
    raw_audit_df.to_string(
        index=False,
        formatters={
            "non_ascii_document_pct":
                lambda x: f"{x:.4f}",
            "non_ascii_character_pct":
                lambda x: f"{x:.6f}",
        },
    )
)


# =============================================================================
# 4. Tokenizer comparison
# =============================================================================

banner("TOKENIZER COMPARISON")


tokenizer_records = []
frequency_frames = []


for role in [
    "investor",
    "startup",
]:

    frame = manifest.loc[
        manifest["node_type"].eq(role)
        & manifest["has_text"]
    ]


    for tokenizer_name, tokenizer in TOKENIZERS.items():

        print(
            f"\nProcessing "
            f"{role} / {tokenizer_name} ..."
        )

        token_counter = Counter()

        doc_lengths = []

        zero_token_docs = 0
        numeric_only_occurrences = 0
        single_character_occurrences = 0

        compound_hyphen_occurrences = 0
        compound_apostrophe_occurrences = 0


        for text in frame["text"]:

            text = normalize_text(text)

            if text is None:
                continue

            tokens = tokenizer(text)

            doc_lengths.append(
                len(tokens)
            )

            if not tokens:
                zero_token_docs += 1
                continue

            token_counter.update(
                tokens
            )


            for token in tokens:

                if token.isdigit():
                    numeric_only_occurrences += 1

                if len(token) == 1:
                    single_character_occurrences += 1

                if "-" in token:
                    compound_hyphen_occurrences += 1

                if (
                    "'" in token
                    or "’" in token
                ):
                    compound_apostrophe_occurrences += 1


        total_tokens = sum(
            token_counter.values()
        )

        unique_tokens = len(
            token_counter
        )


        tokenizer_records.append(
            {
                "node_type": role,
                "tokenizer": tokenizer_name,
                "documents": len(frame),
                "zero_token_documents":
                    zero_token_docs,
                "total_token_occurrences":
                    total_tokens,
                "unique_tokens":
                    unique_tokens,
                "median_tokens_per_document":
                    percentile(
                        doc_lengths,
                        0.50,
                    ),
                "p95_tokens_per_document":
                    percentile(
                        doc_lengths,
                        0.95,
                    ),
                "max_tokens_per_document":
                    max(doc_lengths)
                    if doc_lengths
                    else 0,
                "numeric_only_occurrences":
                    numeric_only_occurrences,
                "single_character_occurrences":
                    single_character_occurrences,
                "hyphen_compound_occurrences":
                    compound_hyphen_occurrences,
                "apostrophe_compound_occurrences":
                    compound_apostrophe_occurrences,
            }
        )


        # Save only the 100 most frequent tokens for each role/tokenizer.
        for rank, (
            token,
            count,
        ) in enumerate(
            token_counter.most_common(100),
            start=1,
        ):

            frequency_frames.append(
                {
                    "node_type": role,
                    "tokenizer": tokenizer_name,
                    "rank": rank,
                    "token": token,
                    "count": count,
                }
            )


tokenizer_df = pd.DataFrame(
    tokenizer_records
)

top_tokens_df = pd.DataFrame(
    frequency_frames
)


print()
print(
    tokenizer_df.to_string(
        index=False
    )
)


# =============================================================================
# 5. Frequency-tail audit for the most inclusive Unicode tokenizer
#
# This is still an audit, not a min_count decision.
# =============================================================================

banner(
    "TOKEN FREQUENCY TAIL — UNICODE COMPOUND"
)


tail_records = []


for role in [
    "investor",
    "startup",
]:

    frame = manifest.loc[
        manifest["node_type"].eq(role)
        & manifest["has_text"]
    ]


    counter = Counter()


    for text in frame["text"]:

        text = normalize_text(text)

        if text is None:
            continue

        counter.update(
            tokenize_unicode_compound(
                text
            )
        )


    counts = np.asarray(
        list(counter.values()),
        dtype=np.int64,
    )


    tail_records.append(
        {
            "node_type": role,
            "unique_tokens":
                len(counter),

            "frequency_eq_1":
                int(
                    np.sum(
                        counts == 1
                    )
                ),

            "frequency_lt_3":
                int(
                    np.sum(
                        counts < 3
                    )
                ),

            "frequency_lt_5":
                int(
                    np.sum(
                        counts < 5
                    )
                ),

            "frequency_lt_10":
                int(
                    np.sum(
                        counts < 10
                    )
                ),

            "frequency_ge_10":
                int(
                    np.sum(
                        counts >= 10
                    )
                ),

            "frequency_ge_100":
                int(
                    np.sum(
                        counts >= 100
                    )
                ),

            "frequency_ge_1000":
                int(
                    np.sum(
                        counts >= 1000
                    )
                ),
        }
    )


tail_df = pd.DataFrame(
    tail_records
)

print(
    tail_df.to_string(
        index=False
    )
)


# =============================================================================
# 6. Text-source composition
# =============================================================================

banner("FROZEN TEXT SOURCE COMPOSITION")


text_source_df = (
    manifest
    .groupby(
        [
            "node_type",
            "text_source",
        ],
        dropna=False,
    )
    .size()
    .reset_index(
        name="entities"
    )
)


print(
    text_source_df.to_string(
        index=False
    )
)


# =============================================================================
# 7. Save outputs
# =============================================================================

raw_audit_path = (
    OUT_DIR
    / "description_text_raw_content_audit.csv"
)

tokenizer_path = (
    OUT_DIR
    / "description_text_tokenizer_comparison.csv"
)

top_tokens_path = (
    OUT_DIR
    / "description_text_top_tokens.csv"
)

tail_path = (
    OUT_DIR
    / "description_text_token_frequency_tail.csv"
)

text_source_path = (
    OUT_DIR
    / "description_text_source_composition.csv"
)

non_ascii_path = (
    OUT_DIR
    / "description_text_non_ascii_examples.csv"
)

environment_path = (
    OUT_DIR
    / "description_text_environment.json"
)

metadata_path = (
    OUT_DIR
    / "description_text_preprocessing_audit_metadata.json"
)


raw_audit_df.to_csv(
    raw_audit_path,
    index=False,
)

tokenizer_df.to_csv(
    tokenizer_path,
    index=False,
)

top_tokens_df.to_csv(
    top_tokens_path,
    index=False,
)

tail_df.to_csv(
    tail_path,
    index=False,
)

text_source_df.to_csv(
    text_source_path,
    index=False,
)

pd.DataFrame(
    non_ascii_examples
).to_csv(
    non_ascii_path,
    index=False,
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


metadata = {
    "phase": "4.2.1a",

    "status":
        "AUDIT_ONLY",

    "paper_specified": {
        "separate_role_corpora":
            True,
        "doc2vec_vector_size":
            32,
        "doc2vec_window":
            3,
    },

    "audited_tokenizer_candidates": [
        "unicode_basic",
        "unicode_compound",
        "ascii_compound",
    ],

    "not_frozen": [
        "tokenizer",
        "lowercasing_policy",
        "numeric_token_policy",
        "stopword_policy",
        "doc2vec_dm_or_dbow",
        "doc2vec_min_count",
        "doc2vec_epochs",
        "doc2vec_negative_sampling",
        "doc2vec_workers",
        "doc2vec_seed",
    ],

    "phase_4_1_3_description_contract_reopened":
        False,
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
# 8. Final summary
# =============================================================================

banner("PHASE 4.2.1a SUMMARY")

print(
    "Frozen input contract consumed: PASS"
)

print(
    "Description field mapping reopened: NO"
)

print(
    "Label mapping reopened:             NO"
)

print()
print(
    "Paper-specified Doc2Vec vector size: 32"
)

print(
    "Paper-specified Doc2Vec window:      3"
)

print(
    "Paper-specified separate corpora:    YES"
)

print()
print(
    "Tokenizer frozen:                    NO"
)

print(
    "Doc2Vec variant frozen:              NO"
)

print(
    "min_count frozen:                    NO"
)

print(
    "epochs frozen:                       NO"
)

print()
print("Outputs:")

for path in [
    raw_audit_path,
    tokenizer_path,
    top_tokens_path,
    tail_path,
    text_source_path,
    non_ascii_path,
    environment_path,
    metadata_path,
]:
    print(f"  {path}")


print(
    "\nPHASE 4.2.1a STATUS: COMPLETE"
)