from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import math
import re
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


# =============================================================================
# PHASE 8.2.1D — ORIGIN-SPECIFIC TRAINING-ONLY PREPROCESSING V1
# =============================================================================
#
# Scientific role:
#
#   FIRST_LEARNED_PREPROCESSING_MATERIALIZATION
#
# This is the first Phase-8 step that fits learned preprocessing state.
#
# It does NOT:
#
#   - instantiate B1a/B1b/M1/M2;
#   - instantiate an optimizer;
#   - perform neural-network training;
#   - score T60;
#   - analyze April performance;
#   - inspect any evaluation metric;
#   - use positive/negative evaluation membership to fit preprocessing.
#
# Authoritative method:
#
#   Phase 8.1.12 V5
#
# Authoritative fit population:
#
#   Phase 8.2.1C V2
#
# Key leakage rule:
#
#   Preprocessors fit ONLY on origin-specific preprocessing_fit_startup IDs.
#
#   ALL COLD_t raw values are excluded from preprocessor fit.
#
#   Once preprocessing state is frozen, it may transform every origin-eligible
#   Startup. Later model-fitting code MUST still zero all COLD_t Startup feature
#   slots during parameter fitting, per V5.
#
# P12 implementation clarification:
#
#   The authoritative consistency contract explicitly says:
#
#       "Categories/category_groups pooled from the same deduplicated H(s,t)"
#
#   Therefore P12 uses BOTH sources.
#
#   To avoid silently aliasing a fine category and a coarse category group
#   that happen to share the same surface string, tokens are source-namespaced:
#
#       CAT::<category>
#       GROUP::<category_group>
#
#   Their counts are pooled across unique Companies in H(s,t), then L1
#   normalized independently from P11 venture count.
#
#   This is an implementation disambiguation of already-frozen information,
#   NOT feature discovery.
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

PHASE8_ROOT = Path(
    "data/experimental/phase_8"
)

METHOD_ROOT = (
    PHASE8_ROOT
    / "phase_8_1_inductive_startup_representation"
)

FRAMEWORK_ROOT = (
    PHASE8_ROOT
    / "phase_8_2_framework_development"
)

FEATURE_ROOT = (
    FRAMEWORK_ROOT
    / "phase_8_2_1_feature_materialization"
)

PREFLIGHT_V2 = (
    FRAMEWORK_ROOT
    / "phase_8_2_0_preflight"
    / "phase_8_2_0_preflight_contract_V2.json"
)

EXECUTION_V5 = (
    METHOD_ROOT
    / "phase_8_1_12_execution_contract"
    / "phase_8_1_12_execution_contract_V5.json"
)

CONSISTENCY_V2 = (
    METHOD_ROOT
    / "phase_8_1_10a_inductive_consistency_closure"
    / "phase_8_1_10a_consistency_amendment_V2.json"
)

FOUNDER_CONTRACT_V1 = (
    METHOD_ROOT
    / "phase_8_1_12b_origin_referenced_founder_history"
    / "phase_8_1_12b_origin_referenced_founder_history_contract_V1.json"
)

ARCHITECTURE_V1 = (
    METHOD_ROOT
    / "phase_8_1_11_exact_neural_architecture"
    / "phase_8_1_11_architecture_contract_V1.json"
)

STARTUP_SOURCE = (
    FEATURE_ROOT
    / "phase_8_2_1a_canonical_startup_source_V1.parquet"
)

FOUNDER_V1_SOURCE = (
    FEATURE_ROOT
    / "phase_8_2_1b_founder_history_source_V1.parquet"
)

FOUNDER_V2_SOURCE = (
    FEATURE_ROOT
    / "phase_8_2_1b_origin_specific_founder_history_source_V2.parquet"
)

MASKS_V2 = (
    FEATURE_ROOT
    / "phase_8_2_1c_origin_startup_masks_V2.parquet"
)

FIT_POP_V2 = (
    FEATURE_ROOT
    / "phase_8_2_1c_preprocessing_fit_population_V2.parquet"
)

ORIGIN_SUMMARY_V2 = (
    FEATURE_ROOT
    / "phase_8_2_1c_origin_population_summary_V2.csv"
)

OUTPUT_DIR = (
    FEATURE_ROOT
    / "phase_8_2_1d_preprocessing_V1"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "phase_8_2_1d_manifest_V1.json"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "phase_8_2_1d_origin_summary_V1.csv"
)

HASH_PATH = (
    OUTPUT_DIR
    / "phase_8_2_1d_artifact_sha256_V1.json"
)


# =============================================================================
# FROZEN POPULATION / ORIGINS
# =============================================================================

EXPECTED_STARTUPS = 311_589

ORIGINS = [
    "T57",
    "T58",
    "T59",
    "T60",
    "APRIL_CONFIRMATORY",
]

SCOPE_BY_ORIGIN = {
    "T57":
        "DEV_T57",

    "T58":
        "DEV_T58",

    "T59":
        "DEV_T59",

    "T60":
        "T60_FINAL_IMPROVED",

    "APRIL_CONFIRMATORY":
        "APRIL_FINAL_M2",
}

PAIR_FILE_BY_ORIGIN = {
    "T57":
        FEATURE_ROOT
        / "phase_8_2_1b_t57_founder_history_pairs_V2.parquet",

    "T58":
        FEATURE_ROOT
        / "phase_8_2_1b_t58_founder_history_pairs_V2.parquet",

    "T59":
        FEATURE_ROOT
        / "phase_8_2_1b_t59_founder_history_pairs_V2.parquet",

    "T60":
        FEATURE_ROOT
        / "phase_8_2_1b_t60_founder_history_pairs_V2.parquet",

    "APRIL_CONFIRMATORY":
        FEATURE_ROOT
        / "phase_8_2_1b_april_confirmatory_founder_history_pairs_V2.parquet",
}


# =============================================================================
# FROZEN TEXT SEMANTICS
# =============================================================================

TOKEN_RE = re.compile(
    r"\b[^\W_]+(?:[-'’][^\W_]+)*\b",
    flags=re.UNICODE,
)

HVAC_CATEGORY = (
    "Heating, Ventilation, and Air Conditioning (HVAC)"
)

HVAC_SENTINEL = (
    "__PHASE8_HVAC_EMBEDDED_COMMA_SENTINEL__"
)


# =============================================================================
# RESERVED TOKENS
# =============================================================================

UNK = "__UNK__"
MISSING = "__MISSING__"

REGION_UNK = "REGION::__UNK__"
REGION_MISSING = "REGION::__MISSING__"
LOCATION_GROUP_UNK = "LOCATION_GROUP::__UNK__"

P12_CATEGORY_UNK = "CAT::__UNK__"
P12_GROUP_UNK = "GROUP::__UNK__"


# =============================================================================
# HELPERS
# =============================================================================

def sep(char="=", width=120):
    print(char * width)


def require(condition, message):

    if not bool(condition):
        raise ValueError(message)


def read_json(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def write_json(path, obj):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )


def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def normalize_scalar(value):

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


def tokenize_phase4(text):

    if pd.isna(text):
        return []

    text = str(text).strip()

    if not text:
        return []

    return [
        token.casefold()
        for token
        in TOKEN_RE.findall(text)
    ]


def parse_categories_raw(value):

    value = normalize_scalar(
        value
    )

    if value is None:
        return []

    protected = value.replace(
        HVAC_CATEGORY,
        HVAC_SENTINEL,
    )

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

        tokens.append(
            token
        )

    if len(tokens) != len(set(tokens)):

        raise ValueError(
            "Duplicate parsed category token detected: "
            f"{value}"
        )

    return tokens


def parse_categories_json(value):

    value = normalize_scalar(
        value
    )

    if value is None:
        return []

    parsed = json.loads(
        value
    )

    require(
        isinstance(
            parsed,
            list,
        ),
        (
            "categories_json is not a JSON list: "
            f"{value[:200]}"
        ),
    )

    tokens = []

    for item in parsed:

        item = normalize_scalar(
            item
        )

        if item is None:
            continue

        tokens.append(
            item
        )

    require(
        len(tokens)
        ==
        len(set(tokens)),
        (
            "Duplicate token in categories_json: "
            f"{value[:200]}"
        ),
    )

    return tokens


def parse_simple_multilabel(value):

    value = normalize_scalar(
        value
    )

    if value is None:
        return []

    tokens = []

    seen = set()

    for token in value.split(","):

        token = token.strip()

        if not token:
            continue

        if token in seen:
            continue

        seen.add(
            token
        )

        tokens.append(
            token
        )

    return tokens


def build_vocab(
    token_lists,
    reserved=None,
):

    reserved = (
        list(reserved)
        if reserved is not None
        else []
    )

    reserved_set = set(
        reserved
    )

    observed = set()

    for tokens in token_lists:

        for token in tokens:

            if token not in reserved_set:
                observed.add(
                    token
                )

    vocabulary = (
        reserved
        +
        sorted(
            observed
        )
    )

    require(
        len(vocabulary)
        ==
        len(set(vocabulary)),
        "Vocabulary contains duplicate entries.",
    )

    return vocabulary


def vocabulary_index(vocabulary):

    return {
        token: idx
        for idx, token
        in enumerate(vocabulary)
    }


def transform_multihot(
    token_lists,
    vocabulary,
    unknown_token=None,
):

    vocab = vocabulary_index(
        vocabulary
    )

    rows = []
    cols = []

    for row_index, tokens in enumerate(
        token_lists
    ):

        indices = set()

        for token in tokens:

            if token in vocab:

                indices.add(
                    vocab[token]
                )

            elif unknown_token is not None:

                require(
                    unknown_token
                    in vocab,
                    (
                        "Unknown token requested but reserved "
                        "UNK token missing from vocabulary."
                    ),
                )

                indices.add(
                    vocab[
                        unknown_token
                    ]
                )

        for col_index in sorted(
            indices
        ):

            rows.append(
                row_index
            )

            cols.append(
                col_index
            )

    data = np.ones(
        len(rows),
        dtype=np.float32,
    )

    matrix = sp.csr_matrix(
        (
            data,
            (
                np.asarray(
                    rows,
                    dtype=np.int64,
                ),
                np.asarray(
                    cols,
                    dtype=np.int64,
                ),
            ),
        ),
        shape=(
            len(token_lists),
            len(vocabulary),
        ),
        dtype=np.float32,
    )

    return matrix


def build_scalar_vocab(
    training_values,
):

    values = set()

    for value in training_values:

        value = normalize_scalar(
            value
        )

        if value is not None:
            values.add(
                value
            )

    return [
        MISSING,
        UNK,
    ] + sorted(
        values
    )


def transform_scalar_onehot(
    values,
    vocabulary,
):

    vocab = vocabulary_index(
        vocabulary
    )

    require(
        MISSING in vocab,
        "MISSING token absent.",
    )

    require(
        UNK in vocab,
        "UNK token absent.",
    )

    rows = []
    cols = []

    for row_index, value in enumerate(
        values
    ):

        value = normalize_scalar(
            value
        )

        if value is None:

            token = MISSING

        elif value in vocab:

            token = value

        else:

            token = UNK

        rows.append(
            row_index
        )

        cols.append(
            vocab[token]
        )

    data = np.ones(
        len(rows),
        dtype=np.float32,
    )

    return sp.csr_matrix(
        (
            data,
            (
                np.asarray(
                    rows,
                    dtype=np.int64,
                ),
                np.asarray(
                    cols,
                    dtype=np.int64,
                ),
            ),
        ),
        shape=(
            len(values),
            len(vocabulary),
        ),
        dtype=np.float32,
    )


def phase8_seed(
    namespace,
    scope_id,
    epoch="NA",
    component="NA",
    parameter="NA",
    shape="NA",
    dtype="NA",
    replicate="NA",
    attempt=0,
    dimension="NA",
):

    payload = (
        f"{namespace}"
        f"|base_seed=42"
        f"|scope={scope_id}"
        f"|epoch={epoch}"
        f"|component={component}"
        f"|parameter={parameter}"
        f"|shape={shape}"
        f"|dtype={dtype}"
        f"|replicate={replicate}"
        f"|attempt={attempt}"
        f"|dimension={dimension}"
    )

    digest = hashlib.sha256(
        payload.encode(
            "utf-8"
        )
    ).digest()

    seed_u64 = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )

    return {
        "payload":
            payload,

        "seed_u64":
            seed_u64,

        "sklearn_random_state":
            seed_u64
            %
            (2 ** 32),
    }


def finite_dense(arr, name):

    require(
        np.isfinite(
            arr
        ).all(),
        f"Non-finite values in {name}.",
    )


def finite_sparse(matrix, name):

    require(
        np.isfinite(
            matrix.data
        ).all(),
        f"Non-finite sparse values in {name}.",
    )


def save_sparse(path, matrix):

    sp.save_npz(
        path,
        matrix,
        compressed=True,
    )


def make_region_location_tokens(
    region,
    location_groups,
):

    tokens = []

    region = normalize_scalar(
        region
    )

    if region is None:

        tokens.append(
            REGION_MISSING
        )

    else:

        tokens.append(
            f"REGION::{region}"
        )

    groups = parse_simple_multilabel(
        location_groups
    )

    for group in groups:

        tokens.append(
            f"LOCATION_GROUP::{group}"
        )

    return tokens


def fit_region_location_vocab(
    token_lists,
):

    observed = set()

    for tokens in token_lists:

        for token in tokens:

            if token not in {
                REGION_MISSING,
                REGION_UNK,
                LOCATION_GROUP_UNK,
            }:

                observed.add(
                    token
                )

    return [
        REGION_MISSING,
        REGION_UNK,
        LOCATION_GROUP_UNK,
    ] + sorted(
        observed
    )


def transform_region_location(
    token_lists,
    vocabulary,
):

    vocab = vocabulary_index(
        vocabulary
    )

    rows = []
    cols = []

    for row_index, tokens in enumerate(
        token_lists
    ):

        indices = set()

        for token in tokens:

            if token in vocab:

                indices.add(
                    vocab[token]
                )

            elif token.startswith(
                "REGION::"
            ):

                indices.add(
                    vocab[
                        REGION_UNK
                    ]
                )

            elif token.startswith(
                "LOCATION_GROUP::"
            ):

                indices.add(
                    vocab[
                        LOCATION_GROUP_UNK
                    ]
                )

            else:

                raise ValueError(
                    f"Unexpected geography token: {token}"
                )

        for col in sorted(
            indices
        ):

            rows.append(
                row_index
            )

            cols.append(
                col
            )

    data = np.ones(
        len(rows),
        dtype=np.float32,
    )

    return sp.csr_matrix(
        (
            data,
            (
                np.asarray(
                    rows,
                    dtype=np.int64,
                ),
                np.asarray(
                    cols,
                    dtype=np.int64,
                ),
            ),
        ),
        shape=(
            len(token_lists),
            len(vocabulary),
        ),
        dtype=np.float32,
    )


def age_log1p_years(
    founded_on,
    prediction_origin,
):

    founded_on = pd.to_datetime(
        founded_on,
        errors="coerce",
    )

    if pd.isna(
        founded_on
    ):
        return np.nan

    prediction_origin = pd.Timestamp(
        prediction_origin
    )

    delta_days = (
        prediction_origin
        -
        founded_on
    ).days

    years = max(
        0.0,
        float(delta_days)
        /
        365.25,
    )

    return float(
        np.log1p(
            years
        )
    )


def p12_counter_by_startup(
    pair_df,
):

    counters = defaultdict(
        Counter
    )

    for row in pair_df.itertuples(
        index=False
    ):

        startup_id = str(
            row.startup_id
        )

        categories = parse_categories_raw(
            row.prior_categories_raw
        )

        groups = parse_simple_multilabel(
            row.prior_category_groups_raw
        )

        # One unique Company row contributes at most one count
        # per source-specific token.
        category_tokens = {
            f"CAT::{token}"
            for token
            in categories
        }

        group_tokens = {
            f"GROUP::{token}"
            for token
            in groups
        }

        for token in category_tokens:

            counters[
                startup_id
            ][
                token
            ] += 1

        for token in group_tokens:

            counters[
                startup_id
            ][
                token
            ] += 1

    return counters


def fit_p12_vocab(
    counters,
    fit_ids,
):

    observed = set()

    for startup_id in fit_ids:

        for token in counters.get(
            startup_id,
            {},
        ):

            observed.add(
                token
            )

    observed.discard(
        P12_CATEGORY_UNK
    )

    observed.discard(
        P12_GROUP_UNK
    )

    return [
        P12_CATEGORY_UNK,
        P12_GROUP_UNK,
    ] + sorted(
        observed
    )


def transform_p12(
    counters,
    startup_ids,
    vocabulary,
):

    vocab = vocabulary_index(
        vocabulary
    )

    rows = []
    cols = []
    data = []

    for row_index, startup_id in enumerate(
        startup_ids
    ):

        original = counters.get(
            startup_id,
            {}
        )

        mapped = Counter()

        for token, count in original.items():

            if token in vocab:

                target = token

            elif token.startswith(
                "CAT::"
            ):

                target = P12_CATEGORY_UNK

            elif token.startswith(
                "GROUP::"
            ):

                target = P12_GROUP_UNK

            else:

                raise ValueError(
                    f"Unexpected P12 token: {token}"
                )

            mapped[
                target
            ] += float(
                count
            )

        denominator = float(
            sum(
                mapped.values()
            )
        )

        if denominator <= 0:
            continue

        for token, count in mapped.items():

            rows.append(
                row_index
            )

            cols.append(
                vocab[token]
            )

            data.append(
                float(count)
                /
                denominator
            )

    matrix = sp.csr_matrix(
        (
            np.asarray(
                data,
                dtype=np.float32,
            ),
            (
                np.asarray(
                    rows,
                    dtype=np.int64,
                ),
                np.asarray(
                    cols,
                    dtype=np.int64,
                ),
            ),
        ),
        shape=(
            len(startup_ids),
            len(vocabulary),
        ),
        dtype=np.float32,
    )

    return matrix


def expected_summary_row(
    summary_df,
    origin,
):

    rows = summary_df.loc[
        summary_df[
            "origin_label"
        ].eq(
            origin
        )
    ]

    require(
        len(rows)
        ==
        1,
        (
            "Expected exactly one origin summary row "
            f"for {origin}."
        ),
    )

    return rows.iloc[0]


# =============================================================================
# MAIN
# =============================================================================

def main():

    sep()

    print(
        "PHASE 8.2.1D — "
        "ORIGIN-SPECIFIC TRAINING-ONLY PREPROCESSING V1"
    )

    sep()

    print()

    print(
        "Scientific role:             "
        "FIRST_LEARNED_PREPROCESSING_MATERIALIZATION"
    )

    print("Method design reopened:      NO")
    print("Feature discovery reopened:  NO")
    print("Evaluation membership fit:   NO")
    print("Cold raw values fit:         NO")
    print("Model instantiated:          NO")
    print("Optimizer instantiated:      NO")
    print("Neural training performed:   NO")
    print("Performance inspected:       NO")
    print("T60 scored:                  NO")
    print("April analyzed:              NO")

    # =========================================================================
    # 1. Prerequisite gates
    # =========================================================================

    sep("-")

    print(
        "8.2.1D.1 — PREREQUISITE GATES"
    )

    sep("-")

    required = [
        PREFLIGHT_V2,
        EXECUTION_V5,
        CONSISTENCY_V2,
        FOUNDER_CONTRACT_V1,
        ARCHITECTURE_V1,
        STARTUP_SOURCE,
        FOUNDER_V1_SOURCE,
        FOUNDER_V2_SOURCE,
        MASKS_V2,
        FIT_POP_V2,
        ORIGIN_SUMMARY_V2,
    ] + [
        PAIR_FILE_BY_ORIGIN[
            origin
        ]
        for origin
        in ORIGINS
    ]

    for path in required:

        require(
            path.exists(),
            f"Missing prerequisite: {path}",
        )

        print(
            f"FOUND {path}"
        )

    require(
        not OUTPUT_DIR.exists(),
        (
            "V1 output directory already exists. "
            "Preserve it and create V2 if this is a rerun "
            "after a failed/materialized attempt."
        ),
    )

    preflight = read_json(
        PREFLIGHT_V2
    )

    execution = read_json(
        EXECUTION_V5
    )

    consistency = read_json(
        CONSISTENCY_V2
    )

    founder_contract = read_json(
        FOUNDER_CONTRACT_V1
    )

    architecture = read_json(
        ARCHITECTURE_V1
    )

    require(
        preflight.get(
            "status"
        )
        ==
        "FROZEN",
        "8.2.0 V2 is not frozen.",
    )

    require(
        preflight.get(
            "training_allowed_now"
        )
        is False,
        "8.2.0 unexpectedly authorizes training.",
    )

    require(
        execution.get(
            "schema_version"
        )
        ==
        "PHASE_8_1_12_EXECUTION_CONTRACT_V5",
        "Authoritative execution contract is not V5.",
    )

    require(
        execution.get(
            "method_design_closed"
        )
        is True,
        "Method design is not closed.",
    )

    execution_blob = json.dumps(
        execution,
        sort_keys=True,
    )

    for marker in [
        "ALL_COLD_t_STARTUPS",
        "TRAINING-SIDE",
        "ITRS_PHASE8_SVD",
    ]:

        require(
            marker.lower()
            in
            execution_blob.lower(),
            (
                "Required V5 preprocessing marker missing: "
                f"{marker}"
            ),
        )

    consistency_blob = json.dumps(
        consistency,
        sort_keys=True,
    )

    require(
        "Categories/category_groups pooled"
        in
        consistency_blob,
        (
            "Authoritative P12 dual-source wording "
            "not found."
        ),
    )

    require(
        "H(s,t)"
        in
        consistency_blob,
        "Authoritative H(s,t) marker missing.",
    )

    founder_blob = json.dumps(
        founder_contract,
        sort_keys=True,
    )

    require(
        "prediction_origin"
        in
        founder_blob,
        "Founder contract does not reference prediction_origin.",
    )

    print()
    print(
        "Authoritative preprocessing chain: PASS"
    )

    # =========================================================================
    # 2. Load substrates
    # =========================================================================

    sep("-")

    print(
        "8.2.1D.2 — LOAD AUTHORITATIVE MATERIALIZED SUBSTRATES"
    )

    sep("-")

    startup = pd.read_parquet(
        STARTUP_SOURCE
    )

    founder_v1 = pd.read_parquet(
        FOUNDER_V1_SOURCE,
        columns=[
            "startup_id",
            "startup_founder_count_resolved",
        ],
    )

    founder_v2 = pd.read_parquet(
        FOUNDER_V2_SOURCE,
        columns=[
            "startup_id",
            "founder_prior_startup_count",
            "origin_label",
            "prediction_origin",
        ],
    )

    masks = pd.read_parquet(
        MASKS_V2,
        columns=[
            "startup_node_index",
            "startup_node_id",
            "startup_id",
            "origin_label",
            "prediction_origin",
            "origin_eligible",
            "investment_cold",
            "investment_warm",
            "fit_startup_features_visible",
            "preprocessing_fit_startup",
        ],
    )

    fit_population = pd.read_parquet(
        FIT_POP_V2,
        columns=[
            "origin_label",
            "prediction_origin",
            "startup_node_index",
            "startup_node_id",
            "startup_id",
            "preprocessing_fit_reason",
        ],
    )

    origin_summary = pd.read_csv(
        ORIGIN_SUMMARY_V2
    )

    require(
        len(startup)
        ==
        EXPECTED_STARTUPS,
        "Canonical Startup source population changed.",
    )

    require(
        startup[
            "startup_id"
        ].nunique()
        ==
        EXPECTED_STARTUPS,
        "Canonical Startup source IDs are not unique.",
    )

    require(
        len(founder_v1)
        ==
        EXPECTED_STARTUPS,
        "Founder V1 reusable substrate population changed.",
    )

    require(
        founder_v1[
            "startup_id"
        ].nunique()
        ==
        EXPECTED_STARTUPS,
        "Founder V1 Startup IDs are not unique.",
    )

    require(
        len(founder_v2)
        ==
        EXPECTED_STARTUPS
        *
        len(ORIGINS),
        "Founder V2 origin population changed.",
    )

    require(
        len(masks)
        ==
        EXPECTED_STARTUPS
        *
        len(ORIGINS),
        "Origin-mask population changed.",
    )

    require(
        not fit_population.duplicated(
            subset=[
                "origin_label",
                "startup_id",
            ]
        ).any(),
        "Duplicate preprocessing-fit Startup/origin rows.",
    )

    print(
        f"Canonical Startups:                  {len(startup):,}"
    )

    print(
        f"Founder-count substrate rows:        {len(founder_v1):,}"
    )

    print(
        f"Origin-specific founder rows:        {len(founder_v2):,}"
    )

    print(
        f"Origin-mask rows:                    {len(masks):,}"
    )

    print(
        f"Preprocessing-fit rows:              {len(fit_population):,}"
    )

    # =========================================================================
    # 3. Freeze exact implementation interpretation
    # =========================================================================

    sep("-")

    print(
        "8.2.1D.3 — PREPROCESSING IMPLEMENTATION FREEZE"
    )

    sep("-")

    implementation = {
        "text": {
            "tokenizer":
                "PHASE4_UNICODE_COMPOUND",

            "regex":
                TOKEN_RE.pattern,

            "case_normalization":
                "Unicode casefold",

            "tfidf": {
                "ngram_range":
                    [1, 2],

                "min_df":
                    5,

                "max_df":
                    0.95,

                "max_features":
                    50_000,

                "sublinear_tf":
                    True,

                "norm":
                    "l2",

                "stop_words":
                    None,

                "strip_accents":
                    None,

                "lowercase":
                    False,
            },

            "svd": {
                "components":
                    32,

                "algorithm":
                    "randomized",

                "n_iter":
                    5,

                "n_oversamples":
                    10,

                "power_iteration_normalizer":
                    "auto",

                "randomness_namespace":
                    "ITRS_PHASE8_SVD",
            },
        },

        "P02_categories": {
            "source":
                "8.2.1A categories_json",

            "parser":
                "already Phase-4 audited",

            "representation":
                "SPARSE_MULTI_HOT",

            "fit_scope":
                "PREPROCESSING_FIT_STARTUPS_ONLY",

            "missing":
                "ALL_ZERO",

            "unseen":
                UNK,
        },

        "P03_category_groups": {
            "source":
                "category_groups_raw",

            "parser":
                "comma-delimited strip + within-row dedup",

            "representation":
                "SPARSE_MULTI_HOT",

            "fit_scope":
                "PREPROCESSING_FIT_STARTUPS_ONLY",

            "missing":
                "ALL_ZERO",

            "unseen":
                UNK,
        },

        "P04_country": {
            "representation":
                "ONE_HOT",

            "missing":
                MISSING,

            "unseen":
                UNK,
        },

        "P05_region_location_groups": {
            "representation":
                "SOURCE_NAMESPACED_SPARSE_MULTI_HOT",

            "region_prefix":
                "REGION::",

            "location_group_prefix":
                "LOCATION_GROUP::",

            "region_missing":
                REGION_MISSING,

            "region_unseen":
                REGION_UNK,

            "location_group_unseen":
                LOCATION_GROUP_UNK,
        },

        "P06_company_type": {
            "representation":
                "ONE_HOT",

            "missing":
                MISSING,

            "unseen":
                UNK,
        },

        "P07_age_at_origin": {
            "transform":
                "log1p(max(0,(origin-founded_on).days/365.25))",

            "missing":
                "TRAINING_MEDIAN_AFTER_TRANSFORM",

            "standardization":
                "TRAINING_SIDE_STANDARD_SCALER",
        },

        "P08_founder_count": {
            "source":
                "8.2.1B V1 reusable resolved identity substrate",

            "transform":
                "log1p(nonnegative count)",

            "missing":
                0,

            "standardization":
                "TRAINING_SIDE_STANDARD_SCALER",
        },

        "P11_prior_startup_count": {
            "source":
                "8.2.1B V2 H(s,t)",

            "history_clock":
                "prediction_origin",

            "transform":
                "log1p(nonnegative count)",

            "missing":
                0,

            "standardization":
                "TRAINING_SIDE_STANDARD_SCALER",
        },

        "P12_prior_sector_profile": {
            "source":
                (
                    "prior_categories_raw + "
                    "prior_category_groups_raw "
                    "from unique H(s,t) Companies"
                ),

            "category_namespace":
                "CAT::",

            "category_group_namespace":
                "GROUP::",

            "company_deduplication":
                "ALREADY_MATERIALIZED_BY_8.2.1B_V2",

            "fit_scope":
                "PREPROCESSING_FIT_STARTUPS_ONLY",

            "aggregation":
                "COUNT_EACH_SOURCE_NAMESPACED_LABEL_PER_UNIQUE_H(s,t)_COMPANY",

            "normalization":
                "L1_PER_FOCAL_STARTUP",

            "missing":
                "ALL_ZERO",

            "unseen_category":
                P12_CATEGORY_UNK,

            "unseen_category_group":
                P12_GROUP_UNK,

            "feature_discovery":
                False,
        },

        "vocabulary_ordering":
            (
                "reserved tokens first, then deterministic "
                "Unicode lexicographic ordering"
            ),

        "evaluation_membership_loaded":
            False,

        "cold_values_allowed_to_fit":
            False,

        "transform_after_fit":
            "ALL_ORIGIN_ELIGIBLE_STARTUPS",

        "cold_feature_zeroing_during_model_fit":
            (
                "DEFERRED_TO_LATER_RUNTIME; transformed cold "
                "features may exist on disk but MUST be zeroed "
                "during parameter fitting."
            ),
    }

    print(
        "P12 sources:                   categories + category_groups"
    )

    print(
        "P12 namespaces:                CAT:: + GROUP::"
    )

    print(
        "P12 history clock:             prediction_origin"
    )

    print(
        "P12 normalization:             L1"
    )

    print(
        "Cold values allowed to fit:    NO"
    )

    print(
        "Evaluation membership loaded:  NO"
    )

    # =========================================================================
    # 4. Create output root
    # =========================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    # =========================================================================
    # 5. Per-origin fit + transformation
    # =========================================================================

    sep("-")

    print(
        "8.2.1D.4 — FIT ORIGIN-SPECIFIC PREPROCESSORS"
    )

    sep("-")

    all_output_paths = []

    summary_rows = []

    startup_base = startup.merge(
        founder_v1,
        on="startup_id",
        how="left",
        validate="one_to_one",
    )

    require(
        startup_base[
            "startup_founder_count_resolved"
        ].notna().all(),
        "P08 founder counts missing after authoritative merge.",
    )

    for origin in ORIGINS:

        print()
        sep(".")
        print(
            f"ORIGIN {origin}"
        )
        sep(".")

        scope_id = SCOPE_BY_ORIGIN[
            origin
        ]

        expected = expected_summary_row(
            origin_summary,
            origin,
        )

        origin_masks = (
            masks.loc[
                masks[
                    "origin_label"
                ].eq(
                    origin
                )
            ]
            .copy()
        )

        require(
            len(origin_masks)
            ==
            EXPECTED_STARTUPS,
            (
                f"{origin}: mask population "
                "does not cover all frozen Startups."
            ),
        )

        eligible_masks = (
            origin_masks.loc[
                origin_masks[
                    "origin_eligible"
                ]
            ]
            .sort_values(
                "startup_node_index"
            )
            .reset_index(
                drop=True
            )
        )

        fit_rows = (
            fit_population.loc[
                fit_population[
                    "origin_label"
                ].eq(
                    origin
                )
            ]
            .copy()
        )

        fit_ids = set(
            fit_rows[
                "startup_id"
            ].astype(str)
        )

        mask_fit_ids = set(
            origin_masks.loc[
                origin_masks[
                    "preprocessing_fit_startup"
                ],
                "startup_id",
            ].astype(str)
        )

        require(
            fit_ids
            ==
            mask_fit_ids,
            (
                f"{origin}: preprocessing-fit artifact "
                "does not match authoritative mask flag."
            ),
        )

        cold_ids = set(
            origin_masks.loc[
                origin_masks[
                    "investment_cold"
                ],
                "startup_id",
            ].astype(str)
        )

        require(
            len(
                fit_ids
                &
                cold_ids
            )
            ==
            0,
            (
                f"{origin}: COLD_t Startup entered "
                "preprocessor fit population."
            ),
        )

        require(
            int(
                len(
                    eligible_masks
                )
            )
            ==
            int(
                expected[
                    "origin_eligible_startups"
                ]
            ),
            f"{origin}: eligible-count mismatch.",
        )

        require(
            len(
                fit_ids
            )
            ==
            int(
                expected[
                    "supervised_training_unique_startups"
                ]
            ),
            (
                f"{origin}: preprocessing-fit unique "
                "Startup count mismatch."
            ),
        )

        prediction_origin_values = (
            eligible_masks[
                "prediction_origin"
            ]
            .drop_duplicates()
        )

        require(
            len(
                prediction_origin_values
            )
            ==
            1,
            f"{origin}: multiple prediction origins found.",
        )

        prediction_origin = pd.Timestamp(
            prediction_origin_values.iloc[
                0
            ]
        )

        founder_origin = (
            founder_v2.loc[
                founder_v2[
                    "origin_label"
                ].eq(
                    origin
                ),
                [
                    "startup_id",
                    "founder_prior_startup_count",
                ],
            ]
            .copy()
        )

        require(
            len(
                founder_origin
            )
            ==
            EXPECTED_STARTUPS,
            (
                f"{origin}: P11 source does not cover "
                "all frozen Startups."
            ),
        )

        frame = (
            eligible_masks[
                [
                    "startup_node_index",
                    "startup_node_id",
                    "startup_id",
                    "investment_cold",
                    "investment_warm",
                    "fit_startup_features_visible",
                    "preprocessing_fit_startup",
                ]
            ]
            .merge(
                startup_base,
                on=[
                    "startup_node_index",
                    "startup_node_id",
                    "startup_id",
                ],
                how="left",
                validate="one_to_one",
            )
            .merge(
                founder_origin,
                on="startup_id",
                how="left",
                validate="one_to_one",
            )
        )

        require(
            len(frame)
            ==
            len(
                eligible_masks
            ),
            f"{origin}: feature merge changed row count.",
        )

        require(
            frame[
                "founder_prior_startup_count"
            ].notna().all(),
            f"{origin}: missing P11 after merge.",
        )

        row_fit_mask = (
            frame[
                "startup_id"
            ]
            .astype(str)
            .isin(
                fit_ids
            )
            .to_numpy()
        )

        require(
            int(
                row_fit_mask.sum()
            )
            ==
            len(
                fit_ids
            ),
            f"{origin}: fit-mask cardinality mismatch.",
        )

        startup_ids = (
            frame[
                "startup_id"
            ]
            .astype(str)
            .tolist()
        )

        # =====================================================================
        # 5A. P01 text TF-IDF -> SVD32
        # =====================================================================

        texts = (
            frame[
                "text"
            ]
            .fillna("")
            .astype(str)
            .tolist()
        )

        fit_texts = [
            texts[i]
            for i
            in np.flatnonzero(
                row_fit_mask
            )
        ]

        tfidf = TfidfVectorizer(
            tokenizer=tokenize_phase4,
            preprocessor=None,
            token_pattern=None,
            lowercase=False,
            strip_accents=None,
            stop_words=None,
            analyzer="word",
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.95,
            max_features=50_000,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )

        fit_tfidf = tfidf.fit_transform(
            fit_texts
        )

        require(
            fit_tfidf.shape[0]
            ==
            len(
                fit_ids
            ),
            f"{origin}: TF-IDF fit-row mismatch.",
        )

        require(
            fit_tfidf.shape[1]
            >=
            32,
            (
                f"{origin}: TF-IDF vocabulary too small "
                "for SVD32."
            ),
        )

        require(
            fit_tfidf.shape[1]
            <=
            50_000,
            f"{origin}: TF-IDF vocabulary exceeds frozen cap.",
        )

        svd_seed = phase8_seed(
            namespace="ITRS_PHASE8_SVD",
            scope_id=scope_id,
            component="text_svd",
            parameter="TruncatedSVD32",
            shape=(
                f"{fit_tfidf.shape[0]}x"
                f"{fit_tfidf.shape[1]}"
            ),
            dtype="float32",
            dimension="32",
        )

        svd = TruncatedSVD(
            n_components=32,
            algorithm="randomized",
            n_iter=5,
            n_oversamples=10,
            power_iteration_normalizer="auto",
            random_state=svd_seed[
                "sklearn_random_state"
            ],
        )

        svd.fit(
            fit_tfidf
        )

        all_tfidf = tfidf.transform(
            texts
        )

        text_svd32 = (
            svd.transform(
                all_tfidf
            )
            .astype(
                np.float32
            )
        )

        require(
            text_svd32.shape
            ==
            (
                len(frame),
                32,
            ),
            f"{origin}: SVD output shape mismatch.",
        )

        finite_dense(
            text_svd32,
            f"{origin} text_svd32",
        )

        # =====================================================================
        # 5B. P02 categories
        # =====================================================================

        category_tokens = [
            parse_categories_json(
                value
            )
            for value
            in frame[
                "categories_json"
            ].tolist()
        ]

        fit_category_tokens = [
            category_tokens[i]
            for i
            in np.flatnonzero(
                row_fit_mask
            )
        ]

        category_vocab = build_vocab(
            fit_category_tokens,
            reserved=[
                UNK,
            ],
        )

        categories_matrix = transform_multihot(
            category_tokens,
            category_vocab,
            unknown_token=UNK,
        )

        # =====================================================================
        # 5C. P03 category groups
        # =====================================================================

        category_group_tokens = [
            parse_simple_multilabel(
                value
            )
            for value
            in frame[
                "category_groups_raw"
            ].tolist()
        ]

        fit_category_group_tokens = [
            category_group_tokens[i]
            for i
            in np.flatnonzero(
                row_fit_mask
            )
        ]

        category_group_vocab = build_vocab(
            fit_category_group_tokens,
            reserved=[
                UNK,
            ],
        )

        category_groups_matrix = transform_multihot(
            category_group_tokens,
            category_group_vocab,
            unknown_token=UNK,
        )

        # =====================================================================
        # 5D. P04 country
        # =====================================================================

        countries = frame[
            "country_raw"
        ].tolist()

        country_fit_values = [
            countries[i]
            for i
            in np.flatnonzero(
                row_fit_mask
            )
        ]

        country_vocab = build_scalar_vocab(
            country_fit_values
        )

        country_matrix = transform_scalar_onehot(
            countries,
            country_vocab,
        )

        # =====================================================================
        # 5E. P05 region + location groups
        # =====================================================================

        region_location_tokens = [
            make_region_location_tokens(
                region,
                location_groups,
            )
            for region, location_groups
            in zip(
                frame[
                    "region_raw"
                ].tolist(),
                frame[
                    "location_groups_raw"
                ].tolist(),
            )
        ]

        fit_region_location_tokens = [
            region_location_tokens[i]
            for i
            in np.flatnonzero(
                row_fit_mask
            )
        ]

        region_location_vocab = fit_region_location_vocab(
            fit_region_location_tokens
        )

        region_location_matrix = transform_region_location(
            region_location_tokens,
            region_location_vocab,
        )

        # =====================================================================
        # 5F. P06 company type
        # =====================================================================

        company_types = frame[
            "company_type_raw"
        ].tolist()

        fit_company_types = [
            company_types[i]
            for i
            in np.flatnonzero(
                row_fit_mask
            )
        ]

        company_type_vocab = build_scalar_vocab(
            fit_company_types
        )

        company_type_matrix = transform_scalar_onehot(
            company_types,
            company_type_vocab,
        )

        # =====================================================================
        # 5G. P07 / P08 numeric base channels
        # =====================================================================

        age_raw = np.asarray(
            [
                age_log1p_years(
                    value,
                    prediction_origin,
                )
                for value
                in frame[
                    "founded_on"
                ].tolist()
            ],
            dtype=np.float64,
        )

        founder_count_raw = (
            np.log1p(
                np.maximum(
                    0.0,
                    frame[
                        "startup_founder_count_resolved"
                    ]
                    .astype(
                        np.float64
                    )
                    .to_numpy(),
                )
            )
        )

        fit_age = age_raw[
            row_fit_mask
        ]

        fit_age_nonmissing = fit_age[
            np.isfinite(
                fit_age
            )
        ]

        require(
            len(
                fit_age_nonmissing
            )
            >
            0,
            f"{origin}: no finite training ages.",
        )

        age_median = float(
            np.median(
                fit_age_nonmissing
            )
        )

        age_imputed = age_raw.copy()

        age_imputed[
            ~np.isfinite(
                age_imputed
            )
        ] = age_median

        base_numeric_raw = np.column_stack(
            [
                age_imputed,
                founder_count_raw,
            ]
        )

        base_scaler = StandardScaler(
            with_mean=True,
            with_std=True,
        )

        base_scaler.fit(
            base_numeric_raw[
                row_fit_mask
            ]
        )

        numeric_base = (
            base_scaler.transform(
                base_numeric_raw
            )
            .astype(
                np.float32
            )
        )

        finite_dense(
            numeric_base,
            f"{origin} numeric_base",
        )

        # =====================================================================
        # 5H. P11 prior Startup count
        # =====================================================================

        prior_count_raw = np.log1p(
            np.maximum(
                0.0,
                frame[
                    "founder_prior_startup_count"
                ]
                .astype(
                    np.float64
                )
                .to_numpy(),
            )
        ).reshape(
            -1,
            1,
        )

        prior_count_scaler = StandardScaler(
            with_mean=True,
            with_std=True,
        )

        prior_count_scaler.fit(
            prior_count_raw[
                row_fit_mask
            ]
        )

        prior_count = (
            prior_count_scaler.transform(
                prior_count_raw
            )
            .astype(
                np.float32
            )
        )

        finite_dense(
            prior_count,
            f"{origin} prior_count",
        )

        # =====================================================================
        # 5I. P12 founder prior sector profile
        # =====================================================================

        pair_path = PAIR_FILE_BY_ORIGIN[
            origin
        ]

        pair_df = pd.read_parquet(
            pair_path,
            columns=[
                "startup_id",
                "prior_company_id",
                "prior_categories_raw",
                "prior_category_groups_raw",
                "origin_label",
                "prediction_origin",
            ],
        )

        if len(
            pair_df
        ):

            require(
                pair_df[
                    "origin_label"
                ].eq(
                    origin
                ).all(),
                f"{origin}: P12 pair-file origin mismatch.",
            )

            require(
                not pair_df.duplicated(
                    subset=[
                        "startup_id",
                        "prior_company_id",
                    ]
                ).any(),
                (
                    f"{origin}: duplicate unique-Company rows "
                    "in H(s,t) P12 substrate."
                ),
            )

        p12_counters = p12_counter_by_startup(
            pair_df
        )

        p12_vocab = fit_p12_vocab(
            p12_counters,
            fit_ids,
        )

        p12_matrix = transform_p12(
            p12_counters,
            startup_ids,
            p12_vocab,
        )

        # =====================================================================
        # 5J. Finite audits
        # =====================================================================

        for name, matrix in [
            (
                "categories",
                categories_matrix,
            ),
            (
                "category_groups",
                category_groups_matrix,
            ),
            (
                "country",
                country_matrix,
            ),
            (
                "region_location_groups",
                region_location_matrix,
            ),
            (
                "company_type",
                company_type_matrix,
            ),
            (
                "prior_sector",
                p12_matrix,
            ),
        ]:

            require(
                matrix.shape[0]
                ==
                len(frame),
                (
                    f"{origin}: {name} row-count "
                    "mismatch."
                ),
            )

            finite_sparse(
                matrix,
                f"{origin} {name}",
            )

        if p12_matrix.shape[0]:

            row_sums = np.asarray(
                p12_matrix.sum(
                    axis=1
                )
            ).ravel()

            nonzero = (
                row_sums
                >
                0
            )

            if nonzero.any():

                require(
                    np.allclose(
                        row_sums[
                            nonzero
                        ],
                        1.0,
                        atol=1e-6,
                    ),
                    (
                        f"{origin}: P12 nonzero rows "
                        "are not L1 normalized."
                    ),
                )

        # =====================================================================
        # 5K. Persist origin artifacts
        # =====================================================================

        slug = (
            origin.lower()
            .replace(
                "april_confirmatory",
                "april"
            )
        )

        row_path = (
            OUTPUT_DIR
            / f"phase_8_2_1d_{slug}_startup_rows_V1.parquet"
        )

        text_path = (
            OUTPUT_DIR
            / f"phase_8_2_1d_{slug}_text_svd32_V1.npy"
        )

        categories_path = (
            OUTPUT_DIR
            / f"phase_8_2_1d_{slug}_categories_V1.npz"
        )

        category_groups_path = (
            OUTPUT_DIR
            / f"phase_8_2_1d_{slug}_category_groups_V1.npz"
        )

        country_path = (
            OUTPUT_DIR
            / f"phase_8_2_1d_{slug}_country_V1.npz"
        )

        geography_path = (
            OUTPUT_DIR
            / f"phase_8_2_1d_{slug}_region_location_groups_V1.npz"
        )

        company_type_path = (
            OUTPUT_DIR
            / f"phase_8_2_1d_{slug}_company_type_V1.npz"
        )

        numeric_base_path = (
            OUTPUT_DIR
            / f"phase_8_2_1d_{slug}_numeric_base_V1.npy"
        )

        prior_count_path = (
            OUTPUT_DIR
            / f"phase_8_2_1d_{slug}_prior_count_V1.npy"
        )

        prior_sector_path = (
            OUTPUT_DIR
            / f"phase_8_2_1d_{slug}_prior_sector_V1.npz"
        )

        vocab_path = (
            OUTPUT_DIR
            / f"phase_8_2_1d_{slug}_vocabularies_V1.json"
        )

        numeric_contract_path = (
            OUTPUT_DIR
            / f"phase_8_2_1d_{slug}_numeric_preprocessor_V1.json"
        )

        text_vocab_path = (
            OUTPUT_DIR
            / f"phase_8_2_1d_{slug}_text_vocabulary_V1.json"
        )

        text_idf_path = (
            OUTPUT_DIR
            / f"phase_8_2_1d_{slug}_text_idf_V1.npy"
        )

        svd_state_path = (
            OUTPUT_DIR
            / f"phase_8_2_1d_{slug}_svd_state_V1.npz"
        )

        row_frame = frame[
            [
                "startup_node_index",
                "startup_node_id",
                "startup_id",
                "investment_cold",
                "investment_warm",
                "fit_startup_features_visible",
                "preprocessing_fit_startup",
            ]
        ].copy()

        row_frame[
            "origin_label"
        ] = origin

        row_frame[
            "prediction_origin"
        ] = prediction_origin

        row_frame.to_parquet(
            row_path,
            index=False,
        )

        np.save(
            text_path,
            text_svd32,
        )

        save_sparse(
            categories_path,
            categories_matrix,
        )

        save_sparse(
            category_groups_path,
            category_groups_matrix,
        )

        save_sparse(
            country_path,
            country_matrix,
        )

        save_sparse(
            geography_path,
            region_location_matrix,
        )

        save_sparse(
            company_type_path,
            company_type_matrix,
        )

        np.save(
            numeric_base_path,
            numeric_base,
        )

        np.save(
            prior_count_path,
            prior_count,
        )

        save_sparse(
            prior_sector_path,
            p12_matrix,
        )

        write_json(
            vocab_path,
            {
                "categories":
                    category_vocab,

                "category_groups":
                    category_group_vocab,

                "country":
                    country_vocab,

                "region_location_groups":
                    region_location_vocab,

                "company_type":
                    company_type_vocab,

                "prior_sector":
                    p12_vocab,
            },
        )

        text_vocabulary = sorted(
            tfidf.vocabulary_.items(),
            key=lambda x: x[1],
        )

        write_json(
            text_vocab_path,
            {
                "terms_in_column_order": [
                    term
                    for term, _
                    in text_vocabulary
                ],

                "vocabulary_size":
                    len(
                        text_vocabulary
                    ),
            },
        )

        np.save(
            text_idf_path,
            tfidf.idf_.astype(
                np.float32
            ),
        )

        np.savez_compressed(
            svd_state_path,
            components=svd.components_.astype(
                np.float32
            ),
            singular_values=svd.singular_values_.astype(
                np.float32
            ),
            explained_variance=svd.explained_variance_.astype(
                np.float32
            ),
            explained_variance_ratio=svd.explained_variance_ratio_.astype(
                np.float32
            ),
        )

        write_json(
            numeric_contract_path,
            {
                "origin_label":
                    origin,

                "prediction_origin":
                    prediction_origin.isoformat(),

                "scope_id":
                    scope_id,

                "age_training_median_after_log1p":
                    age_median,

                "base_numeric_columns": [
                    "startup_age_at_origin",
                    "startup_founder_count_resolved",
                ],

                "base_scaler_mean":
                    base_scaler.mean_.tolist(),

                "base_scaler_scale":
                    base_scaler.scale_.tolist(),

                "base_scaler_var":
                    base_scaler.var_.tolist(),

                "prior_count_column":
                    "founder_prior_startup_count",

                "prior_count_history_clock":
                    "prediction_origin",

                "prior_count_scaler_mean":
                    prior_count_scaler.mean_.tolist(),

                "prior_count_scaler_scale":
                    prior_count_scaler.scale_.tolist(),

                "prior_count_scaler_var":
                    prior_count_scaler.var_.tolist(),

                "svd_seed":
                    svd_seed,

                "standard_scaler_ddof":
                    0,
            },
        )

        current_outputs = [
            row_path,
            text_path,
            categories_path,
            category_groups_path,
            country_path,
            geography_path,
            company_type_path,
            numeric_base_path,
            prior_count_path,
            prior_sector_path,
            vocab_path,
            numeric_contract_path,
            text_vocab_path,
            text_idf_path,
            svd_state_path,
        ]

        all_output_paths.extend(
            current_outputs
        )

        summary_row = {
            "origin_label":
                origin,

            "prediction_origin":
                prediction_origin.date().isoformat(),

            "scope_id":
                scope_id,

            "eligible_startups":
                len(frame),

            "fit_startups":
                len(fit_ids),

            "cold_startups":
                int(
                    frame[
                        "investment_cold"
                    ].sum()
                ),

            "cold_in_fit":
                int(
                    (
                        frame[
                            "investment_cold"
                        ].to_numpy()
                        &
                        row_fit_mask
                    ).sum()
                ),

            "text_vocab_size":
                len(
                    tfidf.vocabulary_
                ),

            "text_svd_dim":
                32,

            "category_vocab_size":
                len(
                    category_vocab
                ),

            "category_group_vocab_size":
                len(
                    category_group_vocab
                ),

            "country_vocab_size":
                len(
                    country_vocab
                ),

            "region_location_vocab_size":
                len(
                    region_location_vocab
                ),

            "company_type_vocab_size":
                len(
                    company_type_vocab
                ),

            "p12_vocab_size":
                len(
                    p12_vocab
                ),

            "p12_nonzero_startups":
                int(
                    (
                        np.asarray(
                            p12_matrix.sum(
                                axis=1
                            )
                        ).ravel()
                        >
                        0
                    ).sum()
                ),

            "p12_history_pair_rows":
                len(
                    pair_df
                ),

            "age_training_median_after_transform":
                age_median,

            "svd_random_state":
                svd_seed[
                    "sklearn_random_state"
                ],
        }

        summary_rows.append(
            summary_row
        )

        print(
            f"eligible Startups:           {len(frame):,}"
        )

        print(
            f"preprocessing-fit Startups:  {len(fit_ids):,}"
        )

        print(
            f"COLD_t in fit:               {summary_row['cold_in_fit']:,}"
        )

        print(
            f"TF-IDF vocabulary:           {summary_row['text_vocab_size']:,}"
        )

        print(
            f"categories vocabulary:       {summary_row['category_vocab_size']:,}"
        )

        print(
            f"category_groups vocabulary:  {summary_row['category_group_vocab_size']:,}"
        )

        print(
            f"P12 vocabulary:              {summary_row['p12_vocab_size']:,}"
        )

        print(
            f"P12 nonzero Startups:        {summary_row['p12_nonzero_startups']:,}"
        )

    # =========================================================================
    # 6. Consolidated audit
    # =========================================================================

    sep("-")

    print(
        "8.2.1D.5 — CONSOLIDATED LEAKAGE / SHAPE AUDIT"
    )

    sep("-")

    summary_df = pd.DataFrame(
        summary_rows
    )

    require(
        summary_df[
            "origin_label"
        ].tolist()
        ==
        ORIGINS,
        "Origin execution order changed.",
    )

    require(
        (
            summary_df[
                "cold_in_fit"
            ]
            ==
            0
        ).all(),
        "At least one COLD_t Startup entered preprocessing fit.",
    )

    require(
        (
            summary_df[
                "text_svd_dim"
            ]
            ==
            32
        ).all(),
        "Text SVD dimension drift.",
    )

    require(
        (
            summary_df[
                "text_vocab_size"
            ]
            <=
            50_000
        ).all(),
        "TF-IDF vocabulary exceeded frozen max_features.",
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    all_output_paths.append(
        SUMMARY_PATH
    )

    print(
        "ALL COLD_t excluded from fit:              PASS"
    )

    print(
        "Fit population = 8.2.1C V2:               PASS"
    )

    print(
        "All transformed rows origin-eligible:      PASS"
    )

    print(
        "Phase-4 tokenizer reused exactly:          PASS"
    )

    print(
        "TF-IDF frozen hyperparameters respected:   PASS"
    )

    print(
        "SVD output dimension = 32:                 PASS"
    )

    print(
        "P11 uses prediction-origin H(s,t):         PASS"
    )

    print(
        "P12 uses categories + category_groups:     PASS"
    )

    print(
        "P12 L1 normalization:                      PASS"
    )

    print(
        "Evaluation case membership consumed:       NO"
    )

    print(
        "Performance inspected:                     NO"
    )

    # =========================================================================
    # 7. Manifest
    # =========================================================================

    sep("-")

    print(
        "8.2.1D.6 — PERSIST MANIFEST + ARTIFACT HASHES"
    )

    sep("-")

    manifest = {
        "phase":
            "8.2.1D",

        "version":
            "V1",

        "status":
            "MATERIALIZED_AND_AUDITED",

        "scientific_role":
            "FIRST_LEARNED_PREPROCESSING_MATERIALIZATION",

        "authoritative_execution_contract":
            str(
                EXECUTION_V5
            ),

        "authoritative_fit_population":
            str(
                FIT_POP_V2
            ),

        "authoritative_cold_mask":
            str(
                MASKS_V2
            ),

        "feature_inventory_changed":
            False,

        "feature_discovery_reopened":
            False,

        "method_design_reopened":
            False,

        "P11_history_clock":
            "prediction_origin",

        "P11_history_set":
            "H(s,t)",

        "P12_history_set":
            "H(s,t)",

        "P12_sources": [
            "prior_categories_raw",
            "prior_category_groups_raw",
        ],

        "P12_source_namespacing": {
            "categories":
                "CAT::",

            "category_groups":
                "GROUP::",
        },

        "P12_source_namespacing_role":
            (
                "Implementation disambiguation preventing "
                "cross-source string aliasing; no new information "
                "channel introduced."
            ),

        "P12_normalization":
            "L1_PER_STARTUP",

        "implementation":
            implementation,

        "origin_summary":
            str(
                SUMMARY_PATH
            ),

        "origins":
            ORIGINS,

        "learned_preprocessing_fit":
            True,

        "model_instantiated":
            False,

        "optimizer_instantiated":
            False,

        "training_performed":
            False,

        "performance_inspected":
            False,

        "T60_scored":
            False,

        "April_analyzed":
            False,

        "cold_raw_values_used_for_fit":
            False,

        "evaluation_positive_membership_used_for_fit":
            False,

        "evaluation_negative_membership_used_for_fit":
            False,

        "evaluation_candidate_matrix_used_for_fit":
            False,

        "future_evaluation_membership_used_for_fit":
            False,

        "cold_transformed_features_materialized":
            True,

        "cold_transformed_feature_runtime_rule":
            (
                "May be restored only after parameters are frozen. "
                "During parameter fitting ALL COLD_t Startup feature "
                "slots remain zero."
            ),

        "next_phase":
            (
                "Phase 8.2.2 — strict inductive heterogeneous "
                "graph materialization."
            ),
    }

    write_json(
        MANIFEST_PATH,
        manifest,
    )

    all_output_paths.append(
        MANIFEST_PATH
    )

    hash_map = {
        str(path):
            sha256_file(
                path
            )
        for path
        in all_output_paths
    }

    write_json(
        HASH_PATH,
        hash_map,
    )

    # =========================================================================
    # 8. Final status
    # =========================================================================

    sep()

    print(
        "PHASE 8.2.1D V1 — "
        "TRAINING-ONLY PREPROCESSING MATERIALIZED AND AUDITED"
    )

    sep()

    print()

    print(
        "STATUS:                            MATERIALIZED_AND_AUDITED"
    )

    print(
        "AUTHORITATIVE METHOD CONTRACT:     8.1.12 V5"
    )

    print(
        "FIT POPULATION:                    8.2.1C V2"
    )

    print(
        "COLD_t RAW VALUES USED FOR FIT:    NO"
    )

    print(
        "EVALUATION MEMBERSHIP USED:        NO"
    )

    print(
        "P01 TF-IDF FIT:                    YES"
    )

    print(
        "P01 SVD32 FIT:                     YES"
    )

    print(
        "VOCABULARIES FIT:                  YES"
    )

    print(
        "NUMERIC SCALERS FIT:               YES"
    )

    print(
        "P11 CLOCK:                         prediction_origin"
    )

    print(
        "P12 SOURCES:                       categories + category_groups"
    )

    print(
        "P12 PROFILE:                       source-namespaced + L1"
    )

    print(
        "COLD TRANSFORMED FEATURES SAVED:   YES"
    )

    print(
        "COLD FEATURES DURING MODEL FIT:     MUST REMAIN ZERO"
    )

    print(
        "MODEL INSTANTIATED:                NO"
    )

    print(
        "OPTIMIZER INSTANTIATED:            NO"
    )

    print(
        "TRAINING PERFORMED:                NO"
    )

    print(
        "PERFORMANCE INSPECTED:             NO"
    )

    print(
        "T60 SCORED:                        NO"
    )

    print(
        "APRIL ANALYZED:                    NO"
    )

    print()

    print("NEXT:")
    print(
        "Phase 8.2.2 — strict inductive heterogeneous "
        "graph materialization."
    )


if __name__ == "__main__":

    try:

        main()

    except Exception:

        print()
        sep("!")
        print(
            "PHASE 8.2.1D V1 — FAILED / NON-AUTHORITATIVE"
        )
        sep("!")

        print()
        print(
            "Any partial V1 artifacts must be preserved."
        )
        print(
            "Do NOT overwrite them. Repair with V2."
        )

        raise
