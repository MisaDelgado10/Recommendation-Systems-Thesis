from pathlib import Path
import hashlib
import json
import re
import sys

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from scipy import sparse


# =============================================================================
# PHASE 4.6.1b — FREEZE AUTHORITATIVE STATIC INPUTS
#                  AND COMPLETE ITRS MODEL TOPOLOGY CONTRACT
#
# CLEAN FULL REPLACEMENT
#
# This version fixes:
#
#   1. Doc2Vec hash provenance:
#      no incorrect hardcoded hash for doc2vec_vectors_all.npy.
#
#   2. Static-manifest row-coordinate schemas:
#
#          Phase-3 node table:
#              node_index
#
#          Doc2Vec manifest:
#              doc2vec_feature_row
#
#          Label manifest:
#              label_feature_row
#
#   3. Strong row identity verification:
#
#          node_id
#          node_type
#          raw_entity_id
#
#      must match row-by-row between Phase 3, Doc2Vec, and labels.
#
# THIS SCRIPT DOES NOT:
#
#   - train anything,
#   - choose negative sampling,
#   - freeze the exact Kaiming variant,
#   - create final parameter values,
#   - save a model state,
#   - reopen Phase 2,
#   - reopen Phase 3,
#   - alter frozen Phase-4 outputs.
#
# The complete model topology is instantiated on PyTorch's META device only.
# =============================================================================


# =============================================================================
# INPUT PATHS
# =============================================================================

PHASE_4_ROOT = Path(
    "data/experimental/phase_4"
)

PHASE_3_MODEL_READY_ROOT = Path(
    "data/experimental/phase_3/model_ready"
)


# -----------------------------------------------------------------------------
# Phase 4.6.1a
# -----------------------------------------------------------------------------

INVENTORY_METADATA_PATH = (
    PHASE_4_ROOT
    / "full_model_integration_inventory"
    / "phase_4_6_1a_inventory_metadata.json"
)

INVENTORY_PARAMETER_BUDGET_PATH = (
    PHASE_4_ROOT
    / "full_model_integration_inventory"
    / "full_itrs_parameter_budget.csv"
)


# -----------------------------------------------------------------------------
# Doc2Vec
# -----------------------------------------------------------------------------

DOC2VEC_ALL_PATH = (
    PHASE_4_ROOT
    / "doc2vec"
    / "vectors"
    / "doc2vec_vectors_all.npy"
)

DOC2VEC_INVESTOR_PATH = (
    PHASE_4_ROOT
    / "doc2vec"
    / "vectors"
    / "investor_doc2vec_vectors.npy"
)

DOC2VEC_STARTUP_PATH = (
    PHASE_4_ROOT
    / "doc2vec"
    / "vectors"
    / "startup_doc2vec_vectors.npy"
)

DOC2VEC_MANIFEST_PATH = (
    PHASE_4_ROOT
    / "doc2vec"
    / "vectors"
    / "doc2vec_vector_manifest.parquet"
)

DOC2VEC_ARTIFACT_HASH_AUDIT_PATH = (
    PHASE_4_ROOT
    / "doc2vec"
    / "audits"
    / "doc2vec_artifact_hashes.csv"
)


# -----------------------------------------------------------------------------
# Labels
# -----------------------------------------------------------------------------

LABEL_MATRIX_PATH = (
    PHASE_4_ROOT
    / "description_labels"
    / "description_label_multihot.npz"
)

LABEL_MANIFEST_PATH = (
    PHASE_4_ROOT
    / "description_labels"
    / "description_label_vector_manifest.parquet"
)


# -----------------------------------------------------------------------------
# Frozen contracts
# -----------------------------------------------------------------------------

DOC2VEC_CONTRACT_PATH = (
    PHASE_4_ROOT
    / "doc2vec_contract"
    / "doc2vec_contract.json"
)

DESCRIPTION_CONTRACT_PATH = (
    PHASE_4_ROOT
    / "description_contract"
    / "description_contract.json"
)

DESCRIPTION_NEURAL_CONTRACT_PATH = (
    PHASE_4_ROOT
    / "description_neural_contract"
    / "description_neural_contract.json"
)

TREND_HISTORY_CONTRACT_PATH = (
    PHASE_4_ROOT
    / "trend_contract"
    / "trend_history_semantics_contract.json"
)

TREND_NEURAL_CONTRACT_PATH = (
    PHASE_4_ROOT
    / "trend_neural_contract"
    / "trend_neural_contract.json"
)

TREND_RUNTIME_CONTRACT_PATH = (
    PHASE_4_ROOT
    / "trend_runtime"
    / "trend_runtime_contract.json"
)

RGCN_NEURAL_CONTRACT_PATH = (
    PHASE_4_ROOT
    / "rgcn_neural_contract"
    / "rgcn_neural_contract.json"
)

RGCN_INTEGRATION_CONTRACT_PATH = (
    PHASE_4_ROOT
    / "rgcn_integration"
    / "rgcn_integration_contract.json"
)

SCORING_INPUT_CONTRACT_PATH = (
    PHASE_4_ROOT
    / "scoring_contract"
    / "scoring_input_contract.json"
)

SCORING_NEURAL_CONTRACT_PATH = (
    PHASE_4_ROOT
    / "scoring_neural_contract"
    / "scoring_neural_contract.json"
)

SCORING_FORWARD_CONTRACT_PATH = (
    PHASE_4_ROOT
    / "scoring_module"
    / "scoring_forward_contract.json"
)

PHASE_4_5_CLOSURE_PATH = (
    PHASE_4_ROOT
    / "scoring_module"
    / "phase_4_5_closure_manifest.json"
)


# -----------------------------------------------------------------------------
# Trend runtime
# -----------------------------------------------------------------------------

TREND_PERIOD_PTR_PATH = (
    PHASE_4_ROOT
    / "trend_runtime"
    / "trend_period_ptr.npy"
)

TREND_STARTUP_INDICES_PATH = (
    PHASE_4_ROOT
    / "trend_runtime"
    / "trend_startup_node_indices.npy"
)

TREND_PERIOD_COUNTS_PATH = (
    PHASE_4_ROOT
    / "trend_runtime"
    / "trend_period_startup_counts.npy"
)


# -----------------------------------------------------------------------------
# Frozen graph
# -----------------------------------------------------------------------------

NODE_INDEX_PATH = (
    PHASE_3_MODEL_READY_ROOT
    / "node_index.parquet"
)

RELATION_INDEX_PATH = (
    PHASE_3_MODEL_READY_ROOT
    / "relation_index.csv"
)

EDGE_INDEX_PATH = (
    PHASE_3_MODEL_READY_ROOT
    / "edge_index.npy"
)

EDGE_TYPE_PATH = (
    PHASE_3_MODEL_READY_ROOT
    / "edge_type.npy"
)


# =============================================================================
# OUTPUTS
# =============================================================================

OUT_DIR = (
    PHASE_4_ROOT
    / "full_model_contract"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# FROZEN POPULATION
# =============================================================================

NUM_INVESTORS = 165_975
NUM_STARTUPS = 311_589

NUM_NODES = 477_564

NUM_EDGES = 158_818
NUM_RELATIONS = 12


# =============================================================================
# FROZEN FEATURE DIMENSIONS
# =============================================================================

LATENT_DIM = 40

DOC2VEC_DIM = 32
LABEL_DIM = 802

DESCRIPTION_TEXT_DIM = 20
DESCRIPTION_LABEL_DIM = 20
DESCRIPTION_DIM = 40

TREND_ITEM_DIM = 80
TREND_QUERY_DIM = 80
TREND_HIDDEN_DIM = 40
TREND_DIM = 40

STRUCTURAL_DIM = 40

INVESTOR_SCORING_DIM = 160
STARTUP_SCORING_DIM = 120

PAIR_DIM = 280


# =============================================================================
# FROZEN STATIC COUNTS
# =============================================================================

EXPECTED_DOC2VEC_ZERO_ROWS = 2_670

EXPECTED_LABEL_NNZ = 1_230_068
EXPECTED_LABEL_ZERO_ROWS = 111_919


# =============================================================================
# FROZEN TRAINABLE PARAMETER COUNTS
# =============================================================================

INVESTOR_LATENT_PARAMETERS = (
    NUM_INVESTORS
    * LATENT_DIM
)

STARTUP_LATENT_PARAMETERS = (
    NUM_STARTUPS
    * LATENT_DIM
)

DESCRIPTION_PARAMETERS = 16_720
TREND_PARAMETERS = 32_480
RGCN_PARAMETERS = 19_320
SCORING_PARAMETERS = 46_849

EXPECTED_FULL_PARAMETERS = (
    INVESTOR_LATENT_PARAMETERS
    + STARTUP_LATENT_PARAMETERS
    + DESCRIPTION_PARAMETERS
    + TREND_PARAMETERS
    + RGCN_PARAMETERS
    + SCORING_PARAMETERS
)


assert (
    INVESTOR_LATENT_PARAMETERS
    == 6_639_000
)

assert (
    STARTUP_LATENT_PARAMETERS
    == 12_463_560
)

assert (
    EXPECTED_FULL_PARAMETERS
    == 19_217_929
)


# =============================================================================
# HELPERS
# =============================================================================

def banner(title):

    print()
    print("=" * 120)
    print(title)
    print("=" * 120)


def require(
    condition,
    message,
):

    if not condition:

        raise AssertionError(
            message
        )


def load_json(path):

    require(
        path.exists(),
        f"Missing required JSON file: {path}",
    )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def sha256_file(path):

    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:

        for chunk in iter(
            lambda:
                f.read(
                    1024 * 1024
                ),
            b"",
        ):

            digest.update(
                chunk
            )

    return digest.hexdigest()


def is_sha256(value):

    if value is None:

        return False

    value = str(
        value
    ).strip()

    return bool(
        re.fullmatch(
            r"[0-9a-fA-F]{64}",
            value,
        )
    )


def normalize_sha256(value):

    value = str(
        value
    ).strip().lower()

    require(
        is_sha256(
            value
        ),
        (
            "Invalid SHA-256 value encountered: "
            f"{value}"
        ),
    )

    return value


def normalize_search_text(value):

    value = str(
        value
    ).casefold()

    value = re.sub(
        r"[_\-/\\\.]+",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def dataframe_row_search_text(
    dataframe,
    row_index,
):

    values = []

    row = dataframe.iloc[
        row_index
    ]

    for value in row.tolist():

        if pd.isna(
            value
        ):

            continue

        values.append(
            normalize_search_text(
                value
            )
        )

    return " | ".join(
        values
    )


def resolve_sha256_column(
    dataframe,
):

    preferred = [
        "sha256",
        "sha_256",
        "file_sha256",
        "file_sha_256",
        "artifact_sha256",
        "artifact_sha_256",
        "hash_sha256",
    ]

    normalized_columns = {

        str(column)
        .strip()
        .casefold():

            column

        for column
        in dataframe.columns
    }

    for candidate in preferred:

        if candidate in normalized_columns:

            return normalized_columns[
                candidate
            ]

    fallback = []

    for column in dataframe.columns:

        compact = (
            str(column)
            .casefold()
            .replace(
                "_",
                "",
            )
            .replace(
                "-",
                "",
            )
        )

        if "sha256" in compact:

            fallback.append(
                column
            )

    require(
        len(
            fallback
        )
        == 1,
        (
            "Could not uniquely resolve SHA-256 "
            f"column from {dataframe.columns.tolist()}"
        ),
    )

    return fallback[
        0
    ]


def find_hash_record(
    dataframe,
    sha_column,
    artifact_path,
    aliases,
):

    row_text = [

        dataframe_row_search_text(
            dataframe,
            index,
        )

        for index
        in range(
            len(
                dataframe
            )
        )
    ]

    search_terms = [

        str(
            artifact_path
        ),

        artifact_path.name,

        artifact_path.stem,

        *aliases,
    ]

    normalized_terms = [

        normalize_search_text(
            term
        )

        for term
        in search_terms
    ]

    for term in normalized_terms:

        matches = [

            index

            for index, text
            in enumerate(
                row_text
            )

            if term in text
        ]

        if len(
            matches
        ) == 1:

            row_index = matches[
                0
            ]

            raw_hash = dataframe.iloc[
                row_index
            ][
                sha_column
            ]

            require(
                is_sha256(
                    raw_hash
                ),
                (
                    "Hash audit row matched "
                    f"{artifact_path}, but SHA-256 "
                    f"value is invalid: {raw_hash}"
                ),
            )

            return {

                "found":
                    True,

                "row_index":
                    int(
                        row_index
                    ),

                "sha256":
                    normalize_sha256(
                        raw_hash
                    ),

                "row":
                    dataframe.iloc[
                        row_index
                    ].to_dict(),
            }

        if len(
            matches
        ) > 1:

            # Keep trying increasingly more specific terms.
            continue

    return {

        "found":
            False,

        "row_index":
            None,

        "sha256":
            None,

        "row":
            None,
    }


def normalized_string_array(series):

    return (
        series
        .fillna(
            "__MISSING__"
        )
        .astype(str)
        .to_numpy()
    )


def exact_string_series_match(
    left,
    right,
):

    return np.array_equal(

        normalized_string_array(
            left
        ),

        normalized_string_array(
            right
        ),
    )


def count_parameters(module):

    return sum(

        parameter.numel()

        for parameter
        in module.parameters()
    )


# =============================================================================
# FULL-MODEL TOPOLOGY DEFINITIONS
#
# META DEVICE ONLY.
# =============================================================================

class DescriptionEncoder(nn.Module):

    def __init__(
        self,
        device,
    ):

        super().__init__()

        self.text_projection = nn.Linear(
            DOC2VEC_DIM,
            DESCRIPTION_TEXT_DIM,
            bias=True,
            device=device,
        )

        self.label_projection = nn.Linear(
            LABEL_DIM,
            DESCRIPTION_LABEL_DIM,
            bias=True,
            device=device,
        )

        self.activation = nn.ReLU()


    def forward(
        self,
        doc2vec,
        labels,
    ):

        text_feature = self.activation(
            self.text_projection(
                doc2vec
            )
        )

        label_feature = self.activation(
            self.label_projection(
                labels
            )
        )

        return torch.cat(
            [
                text_feature,
                label_feature,
            ],
            dim=-1,
        )


class TrendExtractor(nn.Module):

    def __init__(
        self,
        device,
    ):

        super().__init__()

        self.attention_weight = nn.Parameter(
            torch.empty(
                TREND_QUERY_DIM,
                TREND_ITEM_DIM,
                device=device,
            )
        )

        self.gru = nn.GRU(
            input_size=TREND_ITEM_DIM,
            hidden_size=TREND_HIDDEN_DIM,
            num_layers=2,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
            device=device,
        )

        self.output_projection = nn.Linear(
            TREND_HIDDEN_DIM,
            TREND_DIM,
            bias=False,
            device=device,
        )

        self.output_activation = nn.Sigmoid()


class BasisRGCNLayer(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        device,
    ):

        super().__init__()

        self.bases = nn.Parameter(
            torch.empty(
                5,
                in_dim,
                out_dim,
                device=device,
            )
        )

        self.coefficients = nn.Parameter(
            torch.empty(
                NUM_RELATIONS,
                5,
                device=device,
            )
        )

        self.root_weight = nn.Parameter(
            torch.empty(
                in_dim,
                out_dim,
                device=device,
            )
        )


class PreferencePropagation(nn.Module):

    def __init__(
        self,
        device,
    ):

        super().__init__()

        self.layer_1 = BasisRGCNLayer(
            40,
            40,
            device=device,
        )

        self.layer_2 = BasisRGCNLayer(
            40,
            40,
            device=device,
        )

        self.activation = nn.ReLU()


class ScoringMLP(nn.Module):

    def __init__(
        self,
        device,
    ):

        super().__init__()

        self.hidden_1 = nn.Linear(
            280,
            128,
            bias=True,
            device=device,
        )

        self.hidden_2 = nn.Linear(
            128,
            64,
            bias=True,
            device=device,
        )

        self.hidden_3 = nn.Linear(
            64,
            32,
            bias=True,
            device=device,
        )

        self.hidden_4 = nn.Linear(
            32,
            16,
            bias=True,
            device=device,
        )

        self.output = nn.Linear(
            16,
            1,
            bias=True,
            device=device,
        )

        self.activation = nn.ReLU()


class ITRSFullModelTopology(nn.Module):

    def __init__(
        self,
        device,
    ):

        super().__init__()


        # ---------------------------------------------------------------------
        # EXACTLY TWO shared latent tables.
        # ---------------------------------------------------------------------

        self.investor_embedding = nn.Embedding(
            NUM_INVESTORS,
            LATENT_DIM,
            device=device,
        )

        self.startup_embedding = nn.Embedding(
            NUM_STARTUPS,
            LATENT_DIM,
            device=device,
        )


        # ---------------------------------------------------------------------
        # Four reconstructed modules.
        # ---------------------------------------------------------------------

        self.description_encoder = DescriptionEncoder(
            device=device,
        )

        self.trend_extractor = TrendExtractor(
            device=device,
        )

        self.preference_propagation = (
            PreferencePropagation(
                device=device,
            )
        )

        self.scoring_mlp = ScoringMLP(
            device=device,
        )


# =============================================================================
# START
# =============================================================================

banner(
    "PHASE 4.6.1b — "
    "AUTHORITATIVE STATIC INPUTS + "
    "COMPLETE ITRS MODEL TOPOLOGY"
)


# =============================================================================
# 1. ENVIRONMENT
# =============================================================================

banner(
    "ENVIRONMENT"
)

print(
    f"Python:  "
    f"{sys.version.splitlines()[0]}"
)

print(
    f"PyTorch: "
    f"{torch.__version__}"
)

print(
    "Topology device: meta"
)


# =============================================================================
# 2. VERIFY PHASE 4.6.1a
# =============================================================================

banner(
    "PHASE 4.6.1a INVENTORY CONTRACT"
)


inventory_metadata = load_json(
    INVENTORY_METADATA_PATH
)

require(
    inventory_metadata.get(
        "status"
    )
    == "COMPLETE_AUDIT_ONLY",
    "Phase 4.6.1a inventory is not complete.",
)


doc2vec_candidates = (
    inventory_metadata[
        "expected_static_inputs"
    ][
        "doc2vec"
    ][
        "candidates"
    ]
)

label_candidates = (
    inventory_metadata[
        "expected_static_inputs"
    ][
        "labels"
    ][
        "candidates"
    ]
)


require(
    len(
        doc2vec_candidates
    )
    == 1,
    (
        "Phase 4.6.1a must contain exactly "
        "one Doc2Vec candidate."
    ),
)

require(
    len(
        label_candidates
    )
    == 1,
    (
        "Phase 4.6.1a must contain exactly "
        "one label candidate."
    ),
)


require(
    Path(
        doc2vec_candidates[
            0
        ]
    )
    == DOC2VEC_ALL_PATH,
    (
        "Unique Doc2Vec candidate differs "
        "from authoritative path."
    ),
)

require(
    Path(
        label_candidates[
            0
        ]
    )
    == LABEL_MATRIX_PATH,
    (
        "Unique label candidate differs "
        "from authoritative path."
    ),
)


print(
    "Phase 4.6.1a inventory: PASS"
)

print()
print(
    "Authoritative Doc2Vec:"
)

print(
    f"  {DOC2VEC_ALL_PATH}"
)

print()
print(
    "Authoritative labels:"
)

print(
    f"  {LABEL_MATRIX_PATH}"
)


# =============================================================================
# 3. REQUIRED FILE EXISTENCE
# =============================================================================

banner(
    "REQUIRED FILE EXISTENCE"
)


required_files = [

    INVENTORY_PARAMETER_BUDGET_PATH,

    DOC2VEC_ALL_PATH,
    DOC2VEC_INVESTOR_PATH,
    DOC2VEC_STARTUP_PATH,
    DOC2VEC_MANIFEST_PATH,
    DOC2VEC_ARTIFACT_HASH_AUDIT_PATH,

    LABEL_MATRIX_PATH,
    LABEL_MANIFEST_PATH,

    NODE_INDEX_PATH,
    RELATION_INDEX_PATH,
    EDGE_INDEX_PATH,
    EDGE_TYPE_PATH,

    TREND_PERIOD_PTR_PATH,
    TREND_STARTUP_INDICES_PATH,
    TREND_PERIOD_COUNTS_PATH,
]


for path in required_files:

    exists = path.exists()

    print(
        f"{str(path):<110} "
        f"{'FOUND' if exists else 'MISSING'}"
    )

    require(
        exists,
        f"Missing required artifact: {path}",
    )


# =============================================================================
# 4. UPSTREAM FROZEN CONTRACT STATUS
# =============================================================================

banner(
    "UPSTREAM CONTRACT STATUS"
)


status_checks = [

    (
        "Doc2Vec",
        DOC2VEC_CONTRACT_PATH,
        "FROZEN",
    ),

    (
        "Description input",
        DESCRIPTION_CONTRACT_PATH,
        "FROZEN",
    ),

    (
        "Description neural",
        DESCRIPTION_NEURAL_CONTRACT_PATH,
        "FROZEN",
    ),

    (
        "Trend history",
        TREND_HISTORY_CONTRACT_PATH,
        "FROZEN",
    ),

    (
        "Trend neural",
        TREND_NEURAL_CONTRACT_PATH,
        "FROZEN",
    ),

    (
        "Trend runtime",
        TREND_RUNTIME_CONTRACT_PATH,
        "FROZEN",
    ),

    (
        "R-GCN neural",
        RGCN_NEURAL_CONTRACT_PATH,
        "FROZEN",
    ),

    (
        "R-GCN integration",
        RGCN_INTEGRATION_CONTRACT_PATH,
        "FROZEN",
    ),

    (
        "Scoring input",
        SCORING_INPUT_CONTRACT_PATH,
        "FROZEN_INPUT_CONTRACT",
    ),

    (
        "Scoring neural",
        SCORING_NEURAL_CONTRACT_PATH,
        "FROZEN",
    ),

    (
        "Scoring forward",
        SCORING_FORWARD_CONTRACT_PATH,
        "FROZEN",
    ),

    (
        "Phase 4.5 closure",
        PHASE_4_5_CLOSURE_PATH,
        "COMPLETE",
    ),
]


for (
    name,
    path,
    expected_status,
) in status_checks:

    payload = load_json(
        path
    )

    actual_status = payload.get(
        "status"
    )

    exact = (
        actual_status
        == expected_status
    )

    print(
        f"{name:<24} "
        f"expected={expected_status:<24} "
        f"actual={str(actual_status):<24} "
        f"{'PASS' if exact else 'FAIL'}"
    )

    require(
        exact,
        f"{name} frozen status changed.",
    )


# =============================================================================
# 5. DOC2VEC ARRAY INTEGRITY
# =============================================================================

banner(
    "AUTHORITATIVE DOC2VEC ARRAYS"
)


doc2vec_all = np.load(
    DOC2VEC_ALL_PATH,
    mmap_mode="r",
)

doc2vec_investor = np.load(
    DOC2VEC_INVESTOR_PATH,
    mmap_mode="r",
)

doc2vec_startup = np.load(
    DOC2VEC_STARTUP_PATH,
    mmap_mode="r",
)


print(
    f"Combined: "
    f"{doc2vec_all.shape} "
    f"{doc2vec_all.dtype}"
)

print(
    f"Investor: "
    f"{doc2vec_investor.shape} "
    f"{doc2vec_investor.dtype}"
)

print(
    f"Startup:  "
    f"{doc2vec_startup.shape} "
    f"{doc2vec_startup.dtype}"
)


require(
    doc2vec_all.shape
    == (
        NUM_NODES,
        DOC2VEC_DIM,
    ),
    "Combined Doc2Vec shape changed.",
)

require(
    doc2vec_investor.shape
    == (
        NUM_INVESTORS,
        DOC2VEC_DIM,
    ),
    "Investor Doc2Vec shape changed.",
)

require(
    doc2vec_startup.shape
    == (
        NUM_STARTUPS,
        DOC2VEC_DIM,
    ),
    "Startup Doc2Vec shape changed.",
)

require(
    doc2vec_all.dtype
    == np.float32,
    "Combined Doc2Vec dtype changed.",
)

require(
    doc2vec_investor.dtype
    == np.float32,
    "Investor Doc2Vec dtype changed.",
)

require(
    doc2vec_startup.dtype
    == np.float32,
    "Startup Doc2Vec dtype changed.",
)


# =============================================================================
# 6. DOC2VEC ROLE-SLICE ROUNDTRIP
# =============================================================================

banner(
    "DOC2VEC ROLE-SLICE ROUNDTRIP"
)


investor_slice_exact = np.array_equal(
    doc2vec_all[
        :NUM_INVESTORS
    ],
    doc2vec_investor,
)

startup_slice_exact = np.array_equal(
    doc2vec_all[
        NUM_INVESTORS:
    ],
    doc2vec_startup,
)


print(
    f"Investor slice exact: "
    f"{investor_slice_exact}"
)

print(
    f"Startup slice exact:  "
    f"{startup_slice_exact}"
)


require(
    investor_slice_exact,
    (
        "Combined Doc2Vec Investor slice "
        "differs from Investor artifact."
    ),
)

require(
    startup_slice_exact,
    (
        "Combined Doc2Vec Startup slice "
        "differs from Startup artifact."
    ),
)


# =============================================================================
# 7. DOC2VEC NUMERICAL INTEGRITY
# =============================================================================

banner(
    "DOC2VEC NUMERICAL INTEGRITY"
)


doc2vec_finite = bool(
    np.isfinite(
        doc2vec_all
    ).all()
)

doc2vec_zero_rows = int(
    np.sum(
        np.all(
            doc2vec_all
            == 0.0,
            axis=1,
        )
    )
)


print(
    f"All finite:    "
    f"{doc2vec_finite}"
)

print(
    f"All-zero rows: "
    f"{doc2vec_zero_rows:,}"
)


require(
    doc2vec_finite,
    "Doc2Vec contains non-finite values.",
)

require(
    doc2vec_zero_rows
    == EXPECTED_DOC2VEC_ZERO_ROWS,
    (
        "Frozen Doc2Vec zero-row count changed. "
        f"Expected {EXPECTED_DOC2VEC_ZERO_ROWS:,}, "
        f"found {doc2vec_zero_rows:,}."
    ),
)


# =============================================================================
# 8. CURRENT DOC2VEC HASHES
# =============================================================================

banner(
    "CURRENT DOC2VEC FILE HASHES"
)


current_combined_hash = sha256_file(
    DOC2VEC_ALL_PATH
)

current_investor_hash = sha256_file(
    DOC2VEC_INVESTOR_PATH
)

current_startup_hash = sha256_file(
    DOC2VEC_STARTUP_PATH
)


print(
    "Combined:"
)

print(
    f"  {current_combined_hash}"
)

print()
print(
    "Investor:"
)

print(
    f"  {current_investor_hash}"
)

print()
print(
    "Startup:"
)

print(
    f"  {current_startup_hash}"
)


# =============================================================================
# 9. READ ORIGINAL PHASE-4.2 HASH AUDIT
# =============================================================================

banner(
    "ORIGINAL PHASE-4.2 DOC2VEC HASH AUDIT"
)


hash_audit = pd.read_csv(
    DOC2VEC_ARTIFACT_HASH_AUDIT_PATH
)


print(
    f"Rows: "
    f"{len(hash_audit)}"
)

print(
    f"Columns:"
)

print(
    f"  {hash_audit.columns.tolist()}"
)


require(
    len(
        hash_audit
    )
    > 0,
    "Phase-4.2 hash audit is empty.",
)


sha_column = resolve_sha256_column(
    hash_audit
)


print()
print(
    "Resolved SHA-256 column:"
)

print(
    f"  {sha_column}"
)


# =============================================================================
# 10. RESOLVE PHASE-4.2 HASH RECORDS
# =============================================================================

banner(
    "RESOLVING FROZEN DOC2VEC HASH PROVENANCE"
)


combined_hash_record = find_hash_record(
    dataframe=hash_audit,
    sha_column=sha_column,
    artifact_path=DOC2VEC_ALL_PATH,
    aliases=[
        "doc2vec vectors all",
        "combined doc2vec vectors",
        "combined vectors",
    ],
)

investor_hash_record = find_hash_record(
    dataframe=hash_audit,
    sha_column=sha_column,
    artifact_path=DOC2VEC_INVESTOR_PATH,
    aliases=[
        "investor doc2vec vectors",
        "investor vectors",
    ],
)

startup_hash_record = find_hash_record(
    dataframe=hash_audit,
    sha_column=sha_column,
    artifact_path=DOC2VEC_STARTUP_PATH,
    aliases=[
        "startup doc2vec vectors",
        "startup vectors",
    ],
)


print(
    "Combined hash stored directly:"
)

print(
    f"  {combined_hash_record['found']}"
)


if combined_hash_record[
    "found"
]:

    print(
        f"  {combined_hash_record['sha256']}"
    )


print()
print(
    "Investor hash stored directly:"
)

print(
    f"  {investor_hash_record['found']}"
)


if investor_hash_record[
    "found"
]:

    print(
        f"  {investor_hash_record['sha256']}"
    )


print()
print(
    "Startup hash stored directly:"
)

print(
    f"  {startup_hash_record['found']}"
)


if startup_hash_record[
    "found"
]:

    print(
        f"  {startup_hash_record['sha256']}"
    )


# =============================================================================
# 11. VERIFY SPLIT HASHES WHERE AVAILABLE
# =============================================================================

banner(
    "DOC2VEC SPLIT HASH VERIFICATION"
)


investor_hash_verified = None
startup_hash_verified = None


if investor_hash_record[
    "found"
]:

    investor_hash_verified = (
        current_investor_hash
        ==
        investor_hash_record[
            "sha256"
        ]
    )

    print(
        f"Investor frozen hash match: "
        f"{investor_hash_verified}"
    )

    require(
        investor_hash_verified,
        (
            "Investor Doc2Vec file differs "
            "from Phase-4.2 frozen hash."
        ),
    )


else:

    print(
        "Investor frozen hash match: "
        "NOT DIRECTLY AVAILABLE"
    )


if startup_hash_record[
    "found"
]:

    startup_hash_verified = (
        current_startup_hash
        ==
        startup_hash_record[
            "sha256"
        ]
    )

    print(
        f"Startup frozen hash match:  "
        f"{startup_hash_verified}"
    )

    require(
        startup_hash_verified,
        (
            "Startup Doc2Vec file differs "
            "from Phase-4.2 frozen hash."
        ),
    )


else:

    print(
        "Startup frozen hash match:  "
        "NOT DIRECTLY AVAILABLE"
    )


# =============================================================================
# 12. ESTABLISH COMBINED DOC2VEC PROVENANCE
# =============================================================================

banner(
    "COMBINED DOC2VEC HASH PROVENANCE"
)


if combined_hash_record[
    "found"
]:

    combined_hash_verified = (
        current_combined_hash
        ==
        combined_hash_record[
            "sha256"
        ]
    )

    print(
        "Provenance:"
    )

    print(
        "  DIRECT_PHASE_4_2_ARTIFACT_HASH"
    )

    print()
    print(
        "Current combined SHA256:"
    )

    print(
        f"  {current_combined_hash}"
    )

    print()
    print(
        "Frozen Phase-4.2 SHA256:"
    )

    print(
        f"  {combined_hash_record['sha256']}"
    )

    print()
    print(
        f"Direct hash match: "
        f"{combined_hash_verified}"
    )


    require(
        combined_hash_verified,
        (
            "Combined Doc2Vec file differs "
            "from its direct Phase-4.2 frozen hash."
        ),
    )


    combined_hash_provenance = (
        "DIRECT_PHASE_4_2_ARTIFACT_HASH"
    )

    direct_combined_hash_available = True


else:

    print(
        "Direct combined-file Phase-4.2 hash:"
    )

    print(
        "  NOT AVAILABLE"
    )

    print()
    print(
        "Using:"
    )

    print(
        "  FROZEN_SPLIT_HASHES_PLUS_EXACT_CONCATENATION"
    )


    require(
        investor_hash_record[
            "found"
        ],
        (
            "No direct combined hash exists and "
            "Investor split hash could not be resolved."
        ),
    )

    require(
        startup_hash_record[
            "found"
        ],
        (
            "No direct combined hash exists and "
            "Startup split hash could not be resolved."
        ),
    )

    require(
        investor_hash_verified is True,
        (
            "Investor split hash is required "
            "for derived combined provenance."
        ),
    )

    require(
        startup_hash_verified is True,
        (
            "Startup split hash is required "
            "for derived combined provenance."
        ),
    )

    require(
        investor_slice_exact,
        (
            "Combined Investor slice failed "
            "exact roundtrip."
        ),
    )

    require(
        startup_slice_exact,
        (
            "Combined Startup slice failed "
            "exact roundtrip."
        ),
    )


    combined_hash_verified = True

    combined_hash_provenance = (
        "DERIVED_FROM_FROZEN_PHASE_4_2_SPLITS"
    )

    direct_combined_hash_available = False


    print()
    print(
        "Investor frozen hash:       PASS"
    )

    print(
        "Startup frozen hash:        PASS"
    )

    print(
        "Investor combined slice:    PASS"
    )

    print(
        "Startup combined slice:     PASS"
    )

    print()
    print(
        "Combined integration SHA256:"
    )

    print(
        f"  {current_combined_hash}"
    )


print()
print(
    "Combined Doc2Vec provenance:"
)

print(
    f"  {combined_hash_provenance}"
)

print(
    "Combined Doc2Vec integrity:"
)

print(
    "  PASS"
)


# =============================================================================
# 13. LABEL MATRIX INTEGRITY
# =============================================================================

banner(
    "AUTHORITATIVE LABEL MATRIX"
)


labels = sparse.load_npz(
    LABEL_MATRIX_PATH
).tocsr()


print(
    f"Shape: "
    f"{labels.shape}"
)

print(
    f"dtype: "
    f"{labels.dtype}"
)

print(
    f"nnz:   "
    f"{labels.nnz:,}"
)


require(
    labels.shape
    == (
        NUM_NODES,
        LABEL_DIM,
    ),
    "Label matrix shape changed.",
)

require(
    labels.dtype
    == np.float32,
    "Label matrix dtype changed.",
)

require(
    labels.nnz
    == EXPECTED_LABEL_NNZ,
    "Label matrix nnz changed.",
)


label_row_nnz = labels.getnnz(
    axis=1
)

label_zero_rows = int(
    np.sum(
        label_row_nnz
        == 0
    )
)


print(
    f"All-zero rows: "
    f"{label_zero_rows:,}"
)


require(
    label_zero_rows
    == EXPECTED_LABEL_ZERO_ROWS,
    (
        "Frozen label zero-row count changed. "
        f"Expected {EXPECTED_LABEL_ZERO_ROWS:,}, "
        f"found {label_zero_rows:,}."
    ),
)


unique_label_values = np.unique(
    labels.data
)


print(
    "Stored non-zero values:"
)

print(
    f"  {unique_label_values.tolist()}"
)


require(
    np.array_equal(
        unique_label_values,
        np.array(
            [
                1.0,
            ],
            dtype=np.float32,
        ),
    ),
    "Label matrix is no longer binary multi-hot.",
)


label_hash = sha256_file(
    LABEL_MATRIX_PATH
)


print()
print(
    "Label artifact SHA256:"
)

print(
    f"  {label_hash}"
)


# =============================================================================
# 14. LOAD GLOBAL NODE TABLE + FEATURE MANIFESTS
# =============================================================================

banner(
    "GLOBAL NODE / STATIC MANIFEST ALIGNMENT"
)


nodes = pd.read_parquet(
    NODE_INDEX_PATH
)

doc2vec_manifest = pd.read_parquet(
    DOC2VEC_MANIFEST_PATH
)

label_manifest = pd.read_parquet(
    LABEL_MANIFEST_PATH
)


print(
    "Node-index columns:"
)

print(
    f"  {nodes.columns.tolist()}"
)

print()
print(
    "Doc2Vec-manifest columns:"
)

print(
    f"  {doc2vec_manifest.columns.tolist()}"
)

print()
print(
    "Label-manifest columns:"
)

print(
    f"  {label_manifest.columns.tolist()}"
)


require(
    len(
        nodes
    )
    == NUM_NODES,
    "Phase-3 node table population changed.",
)

require(
    len(
        doc2vec_manifest
    )
    == NUM_NODES,
    "Doc2Vec manifest population changed.",
)

require(
    len(
        label_manifest
    )
    == NUM_NODES,
    "Label manifest population changed.",
)


# =============================================================================
# 15. EXACT ROW-COORDINATE SCHEMA
#
# These are the ACTUAL known schemas from the audit output.
# =============================================================================

required_node_columns = {
    "node_index",
    "node_id",
    "node_type",
    "raw_entity_id",
}

required_doc2vec_columns = {
    "doc2vec_feature_row",
    "node_id",
    "node_type",
    "raw_entity_id",
    "doc2vec_text_status",
    "doc2vec_zero_vector",
}

required_label_columns = {
    "label_feature_row",
    "node_id",
    "node_type",
    "raw_entity_id",
    "has_labels",
    "label_count",
}


require(
    required_node_columns.issubset(
        set(
            nodes.columns
        )
    ),
    (
        "Phase-3 node table is missing "
        "required identity/index columns."
    ),
)

require(
    required_doc2vec_columns.issubset(
        set(
            doc2vec_manifest.columns
        )
    ),
    (
        "Doc2Vec manifest schema differs "
        "from frozen observed schema."
    ),
)

require(
    required_label_columns.issubset(
        set(
            label_manifest.columns
        )
    ),
    (
        "Label manifest schema differs "
        "from frozen observed schema."
    ),
)


print()
print(
    "Resolved coordinate fields:"
)

print(
    "  Phase-3: node_index"
)

print(
    "  Doc2Vec: doc2vec_feature_row"
)

print(
    "  Labels:  label_feature_row"
)


expected_rows = np.arange(
    NUM_NODES,
    dtype=np.int64,
)


phase3_rows = (
    nodes[
        "node_index"
    ]
    .to_numpy(
        dtype=np.int64
    )
)

doc2vec_rows = (
    doc2vec_manifest[
        "doc2vec_feature_row"
    ]
    .to_numpy(
        dtype=np.int64
    )
)

label_rows = (
    label_manifest[
        "label_feature_row"
    ]
    .to_numpy(
        dtype=np.int64
    )
)


phase3_rows_exact = np.array_equal(
    phase3_rows,
    expected_rows,
)

doc2vec_rows_exact = np.array_equal(
    doc2vec_rows,
    expected_rows,
)

label_rows_exact = np.array_equal(
    label_rows,
    expected_rows,
)

cross_row_exact = (
    np.array_equal(
        phase3_rows,
        doc2vec_rows,
    )
    and
    np.array_equal(
        phase3_rows,
        label_rows,
    )
)


print()
print(
    "Phase-3 node_index == 0..477563:"
)

print(
    f"  {phase3_rows_exact}"
)

print(
    "Doc2Vec feature row == 0..477563:"
)

print(
    f"  {doc2vec_rows_exact}"
)

print(
    "Label feature row == 0..477563:"
)

print(
    f"  {label_rows_exact}"
)

print(
    "All three row coordinates exact:"
)

print(
    f"  {cross_row_exact}"
)


require(
    phase3_rows_exact,
    "Phase-3 node_index ordering changed.",
)

require(
    doc2vec_rows_exact,
    "Doc2Vec feature-row ordering changed.",
)

require(
    label_rows_exact,
    "Label feature-row ordering changed.",
)

require(
    cross_row_exact,
    (
        "Phase-3 / Doc2Vec / label "
        "row coordinates are not aligned."
    ),
)


# =============================================================================
# 16. STRONG ROW-BY-ROW IDENTITY ALIGNMENT
# =============================================================================

banner(
    "STATIC MANIFEST ROW-IDENTITY ALIGNMENT"
)


doc2vec_node_id_exact = (
    exact_string_series_match(
        nodes[
            "node_id"
        ],
        doc2vec_manifest[
            "node_id"
        ],
    )
)

doc2vec_node_type_exact = (
    exact_string_series_match(
        nodes[
            "node_type"
        ],
        doc2vec_manifest[
            "node_type"
        ],
    )
)

doc2vec_raw_entity_exact = (
    exact_string_series_match(
        nodes[
            "raw_entity_id"
        ],
        doc2vec_manifest[
            "raw_entity_id"
        ],
    )
)


label_node_id_exact = (
    exact_string_series_match(
        nodes[
            "node_id"
        ],
        label_manifest[
            "node_id"
        ],
    )
)

label_node_type_exact = (
    exact_string_series_match(
        nodes[
            "node_type"
        ],
        label_manifest[
            "node_type"
        ],
    )
)

label_raw_entity_exact = (
    exact_string_series_match(
        nodes[
            "raw_entity_id"
        ],
        label_manifest[
            "raw_entity_id"
        ],
    )
)


print(
    "Doc2Vec versus Phase-3:"
)

print(
    f"  node_id exact:       "
    f"{doc2vec_node_id_exact}"
)

print(
    f"  node_type exact:     "
    f"{doc2vec_node_type_exact}"
)

print(
    f"  raw_entity_id exact: "
    f"{doc2vec_raw_entity_exact}"
)


print()
print(
    "Labels versus Phase-3:"
)

print(
    f"  node_id exact:       "
    f"{label_node_id_exact}"
)

print(
    f"  node_type exact:     "
    f"{label_node_type_exact}"
)

print(
    f"  raw_entity_id exact: "
    f"{label_raw_entity_exact}"
)


require(
    doc2vec_node_id_exact,
    "Doc2Vec node_id row alignment failed.",
)

require(
    doc2vec_node_type_exact,
    "Doc2Vec node_type row alignment failed.",
)

require(
    doc2vec_raw_entity_exact,
    "Doc2Vec raw_entity_id row alignment failed.",
)

require(
    label_node_id_exact,
    "Label node_id row alignment failed.",
)

require(
    label_node_type_exact,
    "Label node_type row alignment failed.",
)

require(
    label_raw_entity_exact,
    "Label raw_entity_id row alignment failed.",
)


full_identity_alignment = all(
    [
        doc2vec_node_id_exact,
        doc2vec_node_type_exact,
        doc2vec_raw_entity_exact,
        label_node_id_exact,
        label_node_type_exact,
        label_raw_entity_exact,
    ]
)


print()
print(
    "Full static identity alignment:"
)

print(
    f"  {full_identity_alignment}"
)


require(
    full_identity_alignment,
    "Static feature identity alignment failed.",
)


# =============================================================================
# 17. MANIFEST SEMANTICS CROSS-CHECKS
# =============================================================================

banner(
    "STATIC MANIFEST SEMANTICS"
)


manifest_doc2vec_zero_count = int(
    doc2vec_manifest[
        "doc2vec_zero_vector"
    ]
    .astype(bool)
    .sum()
)


manifest_label_zero_count = int(
    (
        label_manifest[
            "label_count"
        ]
        .fillna(
            0
        )
        .astype(int)
        == 0
    )
    .sum()
)


print(
    "Doc2Vec zero rows from matrix:"
)

print(
    f"  {doc2vec_zero_rows:,}"
)

print(
    "Doc2Vec zero rows from manifest:"
)

print(
    f"  {manifest_doc2vec_zero_count:,}"
)


print()
print(
    "Label zero rows from matrix:"
)

print(
    f"  {label_zero_rows:,}"
)

print(
    "Label zero rows from manifest:"
)

print(
    f"  {manifest_label_zero_count:,}"
)


require(
    manifest_doc2vec_zero_count
    == doc2vec_zero_rows,
    (
        "Doc2Vec manifest zero-vector count "
        "does not match vector matrix."
    ),
)

require(
    manifest_label_zero_count
    == label_zero_rows,
    (
        "Label manifest zero-label count "
        "does not match sparse matrix."
    ),
)


# =============================================================================
# 18. ROLE-SLICE ALIGNMENT
# =============================================================================

banner(
    "FROZEN ROLE-SLICE ALIGNMENT"
)


investor_roles_exact = (
    nodes.iloc[
        :NUM_INVESTORS
    ][
        "node_type"
    ]
    .astype(str)
    .eq(
        "investor"
    )
    .all()
)

startup_roles_exact = (
    nodes.iloc[
        NUM_INVESTORS:
    ][
        "node_type"
    ]
    .astype(str)
    .eq(
        "startup"
    )
    .all()
)


print(
    "Investor global range:"
)

print(
    f"  0 .. {NUM_INVESTORS - 1}"
)

print(
    f"  role exact: "
    f"{investor_roles_exact}"
)


print()
print(
    "Startup global range:"
)

print(
    f"  {NUM_INVESTORS} .. {NUM_NODES - 1}"
)

print(
    f"  role exact: "
    f"{startup_roles_exact}"
)


require(
    investor_roles_exact,
    "Investor role slice changed.",
)

require(
    startup_roles_exact,
    "Startup role slice changed.",
)


# =============================================================================
# 19. STARTUP GLOBAL / LOCAL INDEX CONTRACT
# =============================================================================

banner(
    "GLOBAL / LOCAL ROLE INDEX CONTRACT"
)


sample_startup_local = np.array(
    [
        0,
        1,
        123,
        NUM_STARTUPS - 1,
    ],
    dtype=np.int64,
)

sample_startup_global = (
    sample_startup_local
    + NUM_INVESTORS
)

startup_local_roundtrip = (
    sample_startup_global
    - NUM_INVESTORS
)

startup_conversion_exact = np.array_equal(
    sample_startup_local,
    startup_local_roundtrip,
)


print(
    "Investor:"
)

print(
    "  local index == global index"
)

print()
print(
    "Startup:"
)

print(
    "  global = local + 165975"
)

print(
    "  local  = global - 165975"
)

print()
print(
    f"Startup conversion roundtrip: "
    f"{startup_conversion_exact}"
)


require(
    startup_conversion_exact,
    "Startup global/local conversion failed.",
)


# =============================================================================
# 20. TREND CSR CONTRACT
# =============================================================================

banner(
    "TREND CSR INDEX CONTRACT"
)


trend_period_ptr = np.load(
    TREND_PERIOD_PTR_PATH,
    mmap_mode="r",
)

trend_startup_indices = np.load(
    TREND_STARTUP_INDICES_PATH,
    mmap_mode="r",
)

trend_period_counts = np.load(
    TREND_PERIOD_COUNTS_PATH,
    mmap_mode="r",
)


print(
    f"period_ptr: "
    f"{trend_period_ptr.shape} "
    f"{trend_period_ptr.dtype}"
)

print(
    f"startup_indices: "
    f"{trend_startup_indices.shape} "
    f"{trend_startup_indices.dtype}"
)

print(
    f"period_counts: "
    f"{trend_period_counts.shape} "
    f"{trend_period_counts.dtype}"
)


require(
    trend_period_ptr.shape
    == (
        9_958_501,
    ),
    "Trend period pointer shape changed.",
)

require(
    trend_startup_indices.shape
    == (
        1_145_364,
    ),
    "Trend Startup membership shape changed.",
)

require(
    trend_period_counts.shape
    == (
        9_958_500,
    ),
    "Trend period-count shape changed.",
)


trend_indices_are_global_startups = bool(
    np.all(
        (
            trend_startup_indices
            >= NUM_INVESTORS
        )
        &
        (
            trend_startup_indices
            < NUM_NODES
        )
    )
)


print()
print(
    "All trend Startup indices are "
    "global Startup node indices:"
)

print(
    f"  {trend_indices_are_global_startups}"
)


require(
    trend_indices_are_global_startups,
    "Trend CSR Startup index space changed.",
)


# =============================================================================
# 21. GRAPH ARRAY CONTRACT
# =============================================================================

banner(
    "FULL GRAPH ARRAY CONTRACT"
)


edge_index = np.load(
    EDGE_INDEX_PATH,
    mmap_mode="r",
)

edge_type = np.load(
    EDGE_TYPE_PATH,
    mmap_mode="r",
)

relation_index = pd.read_csv(
    RELATION_INDEX_PATH
)


print(
    f"edge_index: "
    f"{edge_index.shape}"
)

print(
    f"edge_type: "
    f"{edge_type.shape}"
)

print(
    f"relations: "
    f"{len(relation_index)}"
)


require(
    edge_index.shape
    == (
        2,
        NUM_EDGES,
    ),
    "Frozen edge_index shape changed.",
)

require(
    edge_type.shape
    == (
        NUM_EDGES,
    ),
    "Frozen edge_type shape changed.",
)

require(
    len(
        relation_index
    )
    == NUM_RELATIONS,
    "Frozen typed relation count changed.",
)


# =============================================================================
# 22. INSTANTIATE COMPLETE META TOPOLOGY
# =============================================================================

banner(
    "COMPLETE ITRS MODEL TOPOLOGY"
)


model = ITRSFullModelTopology(
    device="meta",
)


print(
    model
)


# =============================================================================
# 23. EXACTLY TWO SHARED LATENT TABLES
# =============================================================================

banner(
    "SHARED LATENT PARAMETER OWNERSHIP"
)


embedding_modules = [

    (
        name,
        module,
    )

    for name, module
    in model.named_modules()

    if isinstance(
        module,
        nn.Embedding,
    )
]


print(
    f"Embedding modules found: "
    f"{len(embedding_modules)}"
)


for (
    name,
    module,
) in embedding_modules:

    print(
        f"{name:<30} "
        f"{tuple(module.weight.shape)}"
    )


require(
    len(
        embedding_modules
    )
    == 2,
    (
        "Integrated model must contain "
        "exactly two embedding tables."
    ),
)


embedding_names = [

    name

    for name, _
    in embedding_modules
]


require(
    embedding_names
    == [
        "investor_embedding",
        "startup_embedding",
    ],
    "Unexpected embedding ownership.",
)


print()
print(
    "Separate trend embeddings:      NO"
)

print(
    "Separate structural embeddings: NO"
)

print(
    "Separate scoring embeddings:    NO"
)


# =============================================================================
# 24. FULL PARAMETER BUDGET
# =============================================================================

banner(
    "FULL MODEL PARAMETER BUDGET — META TOPOLOGY"
)


actual_component_counts = {

    "Investor latent embeddings":
        model
        .investor_embedding
        .weight
        .numel(),

    "Startup latent embeddings":
        model
        .startup_embedding
        .weight
        .numel(),

    "Description encoder":
        count_parameters(
            model.description_encoder
        ),

    "Trend module":
        count_parameters(
            model.trend_extractor
        ),

    "R-GCN":
        count_parameters(
            model.preference_propagation
        ),

    "Scoring MLP":
        count_parameters(
            model.scoring_mlp
        ),
}


expected_component_counts = {

    "Investor latent embeddings":
        INVESTOR_LATENT_PARAMETERS,

    "Startup latent embeddings":
        STARTUP_LATENT_PARAMETERS,

    "Description encoder":
        DESCRIPTION_PARAMETERS,

    "Trend module":
        TREND_PARAMETERS,

    "R-GCN":
        RGCN_PARAMETERS,

    "Scoring MLP":
        SCORING_PARAMETERS,
}


parameter_records = []


for component in expected_component_counts:

    actual = int(
        actual_component_counts[
            component
        ]
    )

    expected = int(
        expected_component_counts[
            component
        ]
    )

    exact = (
        actual
        == expected
    )

    print(
        f"{component:<34} "
        f"{actual:>12,} "
        f"{'PASS' if exact else 'FAIL'}"
    )

    require(
        exact,
        (
            "Parameter count mismatch for "
            f"{component}."
        ),
    )

    parameter_records.append(
        {
            "component":
                component,

            "expected_parameters":
                expected,

            "actual_parameters":
                actual,

            "exact":
                exact,
        }
    )


full_parameter_count = count_parameters(
    model
)


print(
    "-" * 60
)

print(
    f"{'FULL ITRS':<34} "
    f"{full_parameter_count:>12,}"
)


require(
    full_parameter_count
    == EXPECTED_FULL_PARAMETERS,
    "Full ITRS parameter total changed.",
)


print()
print(
    "Full trainable parameter budget: PASS"
)


# =============================================================================
# 25. CROSS-CHECK PHASE 4.6.1a PARAMETER BUDGET
# =============================================================================

banner(
    "PHASE 4.6.1a PARAMETER-BUDGET CROSS-CHECK"
)


inventory_budget = pd.read_csv(
    INVENTORY_PARAMETER_BUDGET_PATH
)


require(
    "parameters"
    in inventory_budget.columns,
    (
        "4.6.1a parameter-budget CSV "
        "does not contain 'parameters'."
    ),
)


inventory_parameter_total = int(
    inventory_budget[
        "parameters"
    ].sum()
)


parameter_budget_crosscheck = (
    inventory_parameter_total
    == full_parameter_count
)


print(
    f"Phase 4.6.1a total: "
    f"{inventory_parameter_total:,}"
)

print(
    f"Meta topology total: "
    f"{full_parameter_count:,}"
)

print(
    f"Exact: "
    f"{parameter_budget_crosscheck}"
)


require(
    parameter_budget_crosscheck,
    (
        "Phase 4.6.1a and 4.6.1b "
        "parameter totals differ."
    ),
)


# =============================================================================
# 26. PARAMETER NAMESPACE
# =============================================================================

banner(
    "INTEGRATED PARAMETER NAMESPACE"
)


namespace_records = []


for (
    name,
    parameter,
) in model.named_parameters():

    record = {

        "parameter":
            name,

        "shape":
            str(
                tuple(
                    parameter.shape
                )
            ),

        "numel":
            int(
                parameter.numel()
            ),
    }

    namespace_records.append(
        record
    )

    print(
        f"{name:<58} "
        f"{str(tuple(parameter.shape)):<22} "
        f"{parameter.numel():>12,}"
    )


# =============================================================================
# 27. COMPLETE FORWARD DATAFLOW CONTRACT
# =============================================================================

banner(
    "COMPLETE ITRS FORWARD DATAFLOW CONTRACT"
)


forward_steps = [

    (
        1,
        "Static description lookup",
        (
            "Doc2Vec[global_node] and "
            "Labels[global_node]"
        ),
    ),

    (
        2,
        "Description encoding",
        (
            "F_d = "
            "[ReLU(TextLinear(Doc2Vec)) || "
            "ReLU(LabelLinear(Labels))]"
        ),
    ),

    (
        3,
        "Shared latent lookup",
        (
            "Investor -> L_o[investor_global]; "
            "Startup -> L_b[startup_global - 165975]"
        ),
    ),

    (
        4,
        "Structural propagation",
        (
            "latent_all = cat(L_o.weight,L_b.weight); "
            "F_s_all = RGCN(latent_all,graph)"
        ),
    ),

    (
        5,
        "Trend historical items",
        (
            "historical Startup global s -> "
            "[L_b[s-165975] || F_d(s)]"
        ),
    ),

    (
        6,
        "Trend Investor query",
        (
            "[L_o[investor] || F_d(investor)]"
        ),
    ),

    (
        7,
        "Trend extraction",
        (
            "target T_h consumes T0..T(h-1); "
            "candidate Startup does not modify history"
        ),
    ),

    (
        8,
        "Investor scorer representation",
        (
            "[F_t || L_o || F_d,o || F_s,o] -> 160"
        ),
    ),

    (
        9,
        "Startup scorer representation",
        (
            "[L_b || F_d,b || F_s,b] -> 120"
        ),
    ),

    (
        10,
        "Pair representation",
        "[R_o || R_b] -> 280",
    ),

    (
        11,
        "Scoring MLP",
        (
            "280 -> 128 -> 64 -> 32 -> "
            "16 -> raw logit"
        ),
    ),

    (
        12,
        "Probability",
        "sigmoid(logit)",
    ),

    (
        13,
        "Training objective",
        "BCEWithLogitsLoss(logit,label)",
    ),
]


for (
    step,
    operation,
    detail,
) in forward_steps:

    print(
        f"{step:>2}. {operation}"
    )

    print(
        f"    {detail}"
    )


# =============================================================================
# 28. STATIC / TRAINABLE / DYNAMIC BOUNDARY
# =============================================================================

banner(
    "STATIC / TRAINABLE / DYNAMIC BOUNDARY"
)


print(
    "STATIC persisted inputs:"
)

print(
    "  Doc2Vec vectors"
)

print(
    "  category-label matrix"
)

print(
    "  trend CSR identities"
)

print(
    "  graph arrays"
)


print()
print(
    "TRAINABLE parameters:"
)

print(
    "  L_o"
)

print(
    "  L_b"
)

print(
    "  description projections"
)

print(
    "  trend attention / GRU / projection"
)

print(
    "  R-GCN"
)

print(
    "  scoring MLP"
)


print()
print(
    "DYNAMIC representations:"
)

print(
    "  F_d"
)

print(
    "  F_t"
)

print(
    "  F_s"
)

print(
    "  R_o"
)

print(
    "  R_b"
)

print(
    "  pair representation"
)

print(
    "  logit / probability"
)


# =============================================================================
# 29. RECOMPUTATION CONTRACT
# =============================================================================

banner(
    "RECOMPUTATION CONTRACT"
)


print(
    "F_d:"
)

print(
    "  recompute from current "
    "description parameters"
)


print()
print(
    "F_t:"
)

print(
    "  recompute from current L_o/L_b, "
    "current F_d, and trend parameters"
)


print()
print(
    "F_s:"
)

print(
    "  recompute from current L_o/L_b "
    "and R-GCN parameters"
)


print()
print(
    "Permanent learned-feature cache:"
)

print(
    "  NO"
)


# =============================================================================
# 30. TRAINING DECISIONS STILL OPEN
# =============================================================================

banner(
    "TRAINING DECISIONS STILL OPEN"
)


still_open = [

    "exact global Kaiming initialization variant",

    "global neural seed policy",

    "training negative:positive ratio",

    "training negative candidate eligibility",

    "training historical negative exclusion",

    "training epoch count",

    "early stopping",

    "weight decay",

    "evaluation candidate-generation runtime contract",
]


for item in still_open:

    print(
        f"  - {item}"
    )


print()
print(
    "Phase 4.6.1b changes NONE of these."
)


# =============================================================================
# 31. SAVE DOC2VEC HASH-PROVENANCE AUDIT
# =============================================================================

hash_provenance_records = [

    {
        "artifact":
            "doc2vec_vectors_all",

        "path":
            str(
                DOC2VEC_ALL_PATH
            ),

        "current_sha256":
            current_combined_hash,

        "phase_4_2_hash_found":
            bool(
                combined_hash_record[
                    "found"
                ]
            ),

        "phase_4_2_sha256":
            combined_hash_record[
                "sha256"
            ],

        "verified":
            bool(
                combined_hash_verified
            ),

        "verification_basis":
            combined_hash_provenance,
    },

    {
        "artifact":
            "investor_doc2vec_vectors",

        "path":
            str(
                DOC2VEC_INVESTOR_PATH
            ),

        "current_sha256":
            current_investor_hash,

        "phase_4_2_hash_found":
            bool(
                investor_hash_record[
                    "found"
                ]
            ),

        "phase_4_2_sha256":
            investor_hash_record[
                "sha256"
            ],

        "verified":
            investor_hash_verified,

        "verification_basis":
            (
                "DIRECT_PHASE_4_2_ARTIFACT_HASH"
                if investor_hash_record[
                    "found"
                ]
                else "EXACT_ROLE_SLICE"
            ),
    },

    {
        "artifact":
            "startup_doc2vec_vectors",

        "path":
            str(
                DOC2VEC_STARTUP_PATH
            ),

        "current_sha256":
            current_startup_hash,

        "phase_4_2_hash_found":
            bool(
                startup_hash_record[
                    "found"
                ]
            ),

        "phase_4_2_sha256":
            startup_hash_record[
                "sha256"
            ],

        "verified":
            startup_hash_verified,

        "verification_basis":
            (
                "DIRECT_PHASE_4_2_ARTIFACT_HASH"
                if startup_hash_record[
                    "found"
                ]
                else "EXACT_ROLE_SLICE"
            ),
    },
]


hash_provenance_df = pd.DataFrame(
    hash_provenance_records
)


hash_provenance_path = (
    OUT_DIR
    / "doc2vec_hash_provenance_audit.csv"
)


hash_provenance_df.to_csv(
    hash_provenance_path,
    index=False,
)


# =============================================================================
# 32. SAVE STATIC ALIGNMENT AUDIT
# =============================================================================

alignment_records = [

    {
        "check":
            "unique_doc2vec_candidate",

        "status":
            "PASS",

        "detail":
            str(
                DOC2VEC_ALL_PATH
            ),
    },

    {
        "check":
            "unique_label_candidate",

        "status":
            "PASS",

        "detail":
            str(
                LABEL_MATRIX_PATH
            ),
    },

    {
        "check":
            "doc2vec_role_slice_roundtrip",

        "status":
            "PASS",

        "detail":
            (
                "combined Investor/Startup slices "
                "exactly equal split artifacts"
            ),
    },

    {
        "check":
            "doc2vec_hash_provenance",

        "status":
            "PASS",

        "detail":
            combined_hash_provenance,
    },

    {
        "check":
            "phase3_node_index",

        "status":
            "PASS",

        "detail":
            "node_index == 0..477563",
    },

    {
        "check":
            "doc2vec_feature_row",

        "status":
            "PASS",

        "detail":
            "doc2vec_feature_row == 0..477563",
    },

    {
        "check":
            "label_feature_row",

        "status":
            "PASS",

        "detail":
            "label_feature_row == 0..477563",
    },

    {
        "check":
            "cross_row_coordinate_alignment",

        "status":
            "PASS",

        "detail":
            (
                "node_index == doc2vec_feature_row "
                "== label_feature_row"
            ),
    },

    {
        "check":
            "doc2vec_node_id_alignment",

        "status":
            "PASS",

        "detail":
            "row-by-row exact",
    },

    {
        "check":
            "doc2vec_node_type_alignment",

        "status":
            "PASS",

        "detail":
            "row-by-row exact",
    },

    {
        "check":
            "doc2vec_raw_entity_alignment",

        "status":
            "PASS",

        "detail":
            "row-by-row exact",
    },

    {
        "check":
            "label_node_id_alignment",

        "status":
            "PASS",

        "detail":
            "row-by-row exact",
    },

    {
        "check":
            "label_node_type_alignment",

        "status":
            "PASS",

        "detail":
            "row-by-row exact",
    },

    {
        "check":
            "label_raw_entity_alignment",

        "status":
            "PASS",

        "detail":
            "row-by-row exact",
    },

    {
        "check":
            "doc2vec_manifest_zero_count",

        "status":
            "PASS",

        "detail":
            str(
                manifest_doc2vec_zero_count
            ),
    },

    {
        "check":
            "label_manifest_zero_count",

        "status":
            "PASS",

        "detail":
            str(
                manifest_label_zero_count
            ),
    },

    {
        "check":
            "startup_global_local_conversion",

        "status":
            "PASS",

        "detail":
            (
                "startup_local = "
                "startup_global - 165975"
            ),
    },

    {
        "check":
            "trend_startup_global_indices",

        "status":
            "PASS",

        "detail":
            "all in 165975..477563",
    },
]


alignment_df = pd.DataFrame(
    alignment_records
)


alignment_path = (
    OUT_DIR
    / "full_model_static_input_alignment_audit.csv"
)


alignment_df.to_csv(
    alignment_path,
    index=False,
)


# =============================================================================
# 33. SAVE COMPONENT PARAMETER AUDIT
# =============================================================================

parameter_df = pd.DataFrame(
    parameter_records
)


parameter_path = (
    OUT_DIR
    / "full_model_component_parameter_audit.csv"
)


parameter_df.to_csv(
    parameter_path,
    index=False,
)


# =============================================================================
# 34. SAVE PARAMETER NAMESPACE
# =============================================================================

namespace_df = pd.DataFrame(
    namespace_records
)


namespace_path = (
    OUT_DIR
    / "full_model_parameter_namespace.csv"
)


namespace_df.to_csv(
    namespace_path,
    index=False,
)


# =============================================================================
# 35. SAVE FORWARD DATAFLOW
# =============================================================================

forward_records = [

    {
        "step":
            step,

        "operation":
            operation,

        "detail":
            detail,
    }

    for (
        step,
        operation,
        detail,
    ) in forward_steps
]


forward_df = pd.DataFrame(
    forward_records
)


forward_path = (
    OUT_DIR
    / "full_itrs_forward_dataflow_contract.csv"
)


forward_df.to_csv(
    forward_path,
    index=False,
)


# =============================================================================
# 36. SAVE STATIC ARTIFACT HASHES
# =============================================================================

static_hash_records = [

    {
        "artifact":
            "doc2vec_vectors_all",

        "path":
            str(
                DOC2VEC_ALL_PATH
            ),

        "sha256":
            current_combined_hash,

        "verification_basis":
            combined_hash_provenance,

        "verified":
            True,
    },

    {
        "artifact":
            "investor_doc2vec_vectors",

        "path":
            str(
                DOC2VEC_INVESTOR_PATH
            ),

        "sha256":
            current_investor_hash,

        "verification_basis":
            (
                "DIRECT_PHASE_4_2_ARTIFACT_HASH"
                if investor_hash_record[
                    "found"
                ]
                else "EXACT_ROLE_SLICE"
            ),

        "verified":
            True,
    },

    {
        "artifact":
            "startup_doc2vec_vectors",

        "path":
            str(
                DOC2VEC_STARTUP_PATH
            ),

        "sha256":
            current_startup_hash,

        "verification_basis":
            (
                "DIRECT_PHASE_4_2_ARTIFACT_HASH"
                if startup_hash_record[
                    "found"
                ]
                else "EXACT_ROLE_SLICE"
            ),

        "verified":
            True,
    },

    {
        "artifact":
            "description_label_multihot",

        "path":
            str(
                LABEL_MATRIX_PATH
            ),

        "sha256":
            label_hash,

        "verification_basis":
            "PHASE_4_6_1B_INTEGRATION_HASH",

        "verified":
            True,
    },
]


static_hash_df = pd.DataFrame(
    static_hash_records
)


static_hash_path = (
    OUT_DIR
    / "full_model_static_artifact_hashes.csv"
)


static_hash_df.to_csv(
    static_hash_path,
    index=False,
)


# =============================================================================
# 37. FREEZE STATIC INPUT CONTRACT
# =============================================================================

banner(
    "FREEZING AUTHORITATIVE STATIC-INPUT CONTRACT"
)


static_input_contract = {

    "phase":
        "4.6.1b",

    "status":
        "FROZEN",

    "component":
        (
            "Complete ITRS authoritative "
            "static-input contract"
        ),

    "population":
        {

            "investors":
                NUM_INVESTORS,

            "startups":
                NUM_STARTUPS,

            "nodes":
                NUM_NODES,
        },

    "doc2vec":
        {

            "path":
                str(
                    DOC2VEC_ALL_PATH
                ),

            "shape":
                [
                    NUM_NODES,
                    DOC2VEC_DIM,
                ],

            "dtype":
                "float32",

            "zero_rows":
                doc2vec_zero_rows,

            "sha256":
                current_combined_hash,

            "hash_verified":
                True,

            "hash_verification_basis":
                combined_hash_provenance,

            "direct_phase_4_2_combined_hash_available":
                direct_combined_hash_available,

            "direct_phase_4_2_combined_sha256":
                combined_hash_record[
                    "sha256"
                ],

            "investor_split_sha256":
                current_investor_hash,

            "startup_split_sha256":
                current_startup_hash,

            "role_slice_roundtrip_exact":
                True,

            "manifest_row_column":
                "doc2vec_feature_row",

            "trainable":
                False,
        },

    "labels":
        {

            "path":
                str(
                    LABEL_MATRIX_PATH
                ),

            "storage":
                "scipy_csr",

            "shape":
                [
                    NUM_NODES,
                    LABEL_DIM,
                ],

            "dtype":
                "float32",

            "nnz":
                int(
                    labels.nnz
                ),

            "zero_rows":
                label_zero_rows,

            "nonzero_value":
                1.0,

            "sha256":
                label_hash,

            "manifest_row_column":
                "label_feature_row",

            "trainable":
                False,
        },

    "row_alignment":
        {

            "phase_3_row_coordinate":
                "node_index",

            "doc2vec_row_coordinate":
                "doc2vec_feature_row",

            "label_row_coordinate":
                "label_feature_row",

            "range":
                [
                    0,
                    NUM_NODES - 1,
                ],

            "coordinate_alignment_exact":
                True,

            "node_id_alignment_exact":
                True,

            "node_type_alignment_exact":
                True,

            "raw_entity_id_alignment_exact":
                True,

            "semantics":
                (
                    "Every static feature row maps "
                    "to the same frozen Phase-3 "
                    "global role-node index."
                ),
        },

    "indexing":
        {

            "investor_global_range":
                [
                    0,
                    NUM_INVESTORS - 1,
                ],

            "startup_global_range":
                [
                    NUM_INVESTORS,
                    NUM_NODES - 1,
                ],

            "investor_local_equals_global":
                True,

            "startup_global_to_local":
                "startup_global - 165975",

            "startup_local_to_global":
                "startup_local + 165975",

            "description_index_space":
                "global",

            "structural_index_space":
                "global",

            "trend_startup_index_space":
                "global",

            "startup_latent_index_space":
                "startup-local",
        },

    "trend_runtime":
        {

            "period_ptr":
                str(
                    TREND_PERIOD_PTR_PATH
                ),

            "startup_indices":
                str(
                    TREND_STARTUP_INDICES_PATH
                ),

            "period_counts":
                str(
                    TREND_PERIOD_COUNTS_PATH
                ),
        },

    "graph":
        {

            "node_index":
                str(
                    NODE_INDEX_PATH
                ),

            "relation_index":
                str(
                    RELATION_INDEX_PATH
                ),

            "edge_index":
                str(
                    EDGE_INDEX_PATH
                ),

            "edge_type":
                str(
                    EDGE_TYPE_PATH
                ),

            "nodes":
                NUM_NODES,

            "edges":
                NUM_EDGES,

            "relations":
                NUM_RELATIONS,
        },

    "training_performed":
        False,

    "upstream_reopened":
        {

            "phase_2":
                False,

            "phase_3":
                False,

            "phase_4_2":
                False,

            "phase_4_3":
                False,

            "phase_4_4":
                False,

            "phase_4_5":
                False,
        },
}


static_contract_path = (
    OUT_DIR
    / "full_model_static_input_contract.json"
)


with open(
    static_contract_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        static_input_contract,
        f,
        indent=2,
        ensure_ascii=False,
    )


# =============================================================================
# 38. FREEZE COMPLETE MODEL TOPOLOGY CONTRACT
# =============================================================================

banner(
    "FREEZING COMPLETE ITRS MODEL TOPOLOGY CONTRACT"
)


full_model_contract = {

    "phase":
        "4.6.1b",

    "status":
        "FROZEN",

    "component":
        (
            "Complete ITRS neural topology "
            "and forward dataflow"
        ),

    "latent_embeddings":
        {

            "number_of_tables":
                2,

            "investor":
                {

                    "symbol":
                        "L_o",

                    "shape":
                        [
                            NUM_INVESTORS,
                            LATENT_DIM,
                        ],

                    "parameters":
                        INVESTOR_LATENT_PARAMETERS,
                },

            "startup":
                {

                    "symbol":
                        "L_b",

                    "shape":
                        [
                            NUM_STARTUPS,
                            LATENT_DIM,
                        ],

                    "parameters":
                        STARTUP_LATENT_PARAMETERS,
                },

            "shared_across_modules":
                True,

            "separate_trend_embeddings":
                False,

            "separate_structural_embeddings":
                False,

            "separate_scoring_embeddings":
                False,

            "rgcn_input":
                "cat(L_o.weight,L_b.weight,dim=0)",

            "rgcn_input_shape":
                [
                    NUM_NODES,
                    LATENT_DIM,
                ],
        },

    "description":
        {

            "doc2vec_dim":
                DOC2VEC_DIM,

            "label_dim":
                LABEL_DIM,

            "text_projection":
                [
                    DOC2VEC_DIM,
                    DESCRIPTION_TEXT_DIM,
                ],

            "label_projection":
                [
                    LABEL_DIM,
                    DESCRIPTION_LABEL_DIM,
                ],

            "output_dim":
                DESCRIPTION_DIM,

            "parameters":
                DESCRIPTION_PARAMETERS,
        },

    "trend":
        {

            "query":
                "L_o || F_d,o",

            "query_dim":
                TREND_QUERY_DIM,

            "item":
                "L_b || F_d,b",

            "item_dim":
                TREND_ITEM_DIM,

            "gru_hidden_dim":
                TREND_HIDDEN_DIM,

            "gru_layers":
                2,

            "output_dim":
                TREND_DIM,

            "parameters":
                TREND_PARAMETERS,

            "target_semantics":
                (
                    "target T_h uses historical "
                    "periods T0..T(h-1)"
                ),

            "candidate_affects_history":
                False,
        },

    "preference_propagation":
        {

            "input":
                "cat(L_o.weight,L_b.weight,dim=0)",

            "input_shape":
                [
                    NUM_NODES,
                    LATENT_DIM,
                ],

            "layers":
                2,

            "relations":
                NUM_RELATIONS,

            "bases":
                5,

            "output_dim":
                STRUCTURAL_DIM,

            "parameters":
                RGCN_PARAMETERS,
        },

    "scoring":
        {

            "investor_representation":
                {

                    "formula":
                        (
                            "F_t || L_o || "
                            "F_d,o || F_s,o"
                        ),

                    "dimension":
                        INVESTOR_SCORING_DIM,
                },

            "startup_representation":
                {

                    "formula":
                        (
                            "L_b || F_d,b || F_s,b"
                        ),

                    "dimension":
                        STARTUP_SCORING_DIM,
                },

            "pair_order":
                [
                    "F_t",
                    "L_o",
                    "F_d,o",
                    "F_s,o",
                    "L_b",
                    "F_d,b",
                    "F_s,b",
                ],

            "pair_dimension":
                PAIR_DIM,

            "architecture":
                [
                    280,
                    128,
                    64,
                    32,
                    16,
                    1,
                ],

            "parameters":
                SCORING_PARAMETERS,

            "training_output":
                "raw_logit",

            "probability":
                "sigmoid(logit)",

            "loss":
                "BCEWithLogitsLoss",
        },

    "parameter_budget":
        {

            "investor_latent":
                INVESTOR_LATENT_PARAMETERS,

            "startup_latent":
                STARTUP_LATENT_PARAMETERS,

            "description":
                DESCRIPTION_PARAMETERS,

            "trend":
                TREND_PARAMETERS,

            "rgcn":
                RGCN_PARAMETERS,

            "scoring":
                SCORING_PARAMETERS,

            "total":
                EXPECTED_FULL_PARAMETERS,
        },

    "feature_persistence":
        {

            "F_d":
                False,

            "F_t":
                False,

            "F_s":
                False,

            "reason":
                (
                    "All depend on current "
                    "trainable model parameters."
                ),
        },

    "initialization":
        {

            "paper_family":
                "Kaiming",

            "exact_variant":
                "NOT_YET_FROZEN",

            "meta_device_topology_only":
                True,

            "initialized_parameter_state_created":
                False,
        },

    "negative_sampling":
        {

            "status":
                "NOT_YET_FROZEN",

            "phase_2_deferral_preserved":
                True,
        },

    "training_performed":
        False,

    "numerical_end_to_end_forward_verified":
        False,

    "next_phase":
        {

            "phase":
                "4.6.2",

            "name":
                (
                    "Complete End-to-End "
                    "ITRS Forward + BCE Audit"
                ),
        },
}


full_model_contract_path = (
    OUT_DIR
    / "full_itrs_model_topology_contract.json"
)


with open(
    full_model_contract_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        full_model_contract,
        f,
        indent=2,
        ensure_ascii=False,
    )


# =============================================================================
# 39. CONTRACT HASHES
# =============================================================================

contract_artifacts = [

    (
        "static_input_contract",
        static_contract_path,
    ),

    (
        "full_model_topology_contract",
        full_model_contract_path,
    ),

    (
        "doc2vec_hash_provenance",
        hash_provenance_path,
    ),

    (
        "static_alignment_audit",
        alignment_path,
    ),

    (
        "component_parameter_audit",
        parameter_path,
    ),

    (
        "parameter_namespace",
        namespace_path,
    ),

    (
        "forward_dataflow",
        forward_path,
    ),

    (
        "static_artifact_hashes",
        static_hash_path,
    ),
]


contract_hash_records = []


for (
    artifact,
    path,
) in contract_artifacts:

    contract_hash_records.append(
        {

            "artifact":
                artifact,

            "path":
                str(
                    path
                ),

            "sha256":
                sha256_file(
                    path
                ),

            "bytes":
                int(
                    path.stat().st_size
                ),
        }
    )


contract_hash_df = pd.DataFrame(
    contract_hash_records
)


contract_hash_path = (
    OUT_DIR
    / "phase_4_6_1b_contract_hashes.csv"
)


contract_hash_df.to_csv(
    contract_hash_path,
    index=False,
)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

banner(
    "PHASE 4.6.1b FINAL SUMMARY"
)


print(
    "Authoritative Doc2Vec:"
)

print(
    f"  {DOC2VEC_ALL_PATH}"
)

print(
    "  shape                         [477564, 32]"
)

print(
    "  dtype                         float32"
)

print(
    f"  all-zero rows                 "
    f"{doc2vec_zero_rows:,}"
)

print(
    "  Investor slice exact          PASS"
)

print(
    "  Startup slice exact           PASS"
)

print(
    f"  SHA256                        "
    f"{current_combined_hash}"
)

print(
    f"  provenance                    "
    f"{combined_hash_provenance}"
)


print()
print(
    "Authoritative labels:"
)

print(
    f"  {LABEL_MATRIX_PATH}"
)

print(
    "  shape                         [477564, 802]"
)

print(
    "  storage                       sparse CSR"
)

print(
    f"  nnz                           "
    f"{labels.nnz:,}"
)

print(
    f"  all-zero rows                 "
    f"{label_zero_rows:,}"
)


print()
print(
    "Static row coordinates:"
)

print(
    "  Phase-3                       node_index"
)

print(
    "  Doc2Vec                       doc2vec_feature_row"
)

print(
    "  Labels                        label_feature_row"
)

print(
    "  all exact 0..477563           PASS"
)


print()
print(
    "Static identity alignment:"
)

print(
    "  Doc2Vec node_id               PASS"
)

print(
    "  Doc2Vec node_type             PASS"
)

print(
    "  Doc2Vec raw_entity_id         PASS"
)

print(
    "  Labels node_id                PASS"
)

print(
    "  Labels node_type              PASS"
)

print(
    "  Labels raw_entity_id          PASS"
)


print()
print(
    "Shared latent tables:"
)

print(
    "  L_o [165975,40]"
)

print(
    "  L_b [311589,40]"
)

print(
    "  total embedding tables        2"
)

print(
    "  duplicate branch embeddings   NO"
)


print()
print(
    "Full trainable parameter budget:"
)

print(
    f"  Investor latent               "
    f"{INVESTOR_LATENT_PARAMETERS:,}"
)

print(
    f"  Startup latent                "
    f"{STARTUP_LATENT_PARAMETERS:,}"
)

print(
    f"  Description                   "
    f"{DESCRIPTION_PARAMETERS:,}"
)

print(
    f"  Trend                         "
    f"{TREND_PARAMETERS:,}"
)

print(
    f"  R-GCN                         "
    f"{RGCN_PARAMETERS:,}"
)

print(
    f"  Scoring                       "
    f"{SCORING_PARAMETERS:,}"
)

print(
    "                                ------------"
)

print(
    f"  FULL ITRS                     "
    f"{full_parameter_count:,}"
)


print()
print(
    "Topology:"
)

print(
    "  PyTorch META device           YES"
)

print(
    "  actual parameter values       NO"
)

print(
    "  model state persisted         NO"
)


print()
print(
    "Still NOT frozen:"
)

print(
    "  exact Kaiming variant"
)

print(
    "  global neural seed"
)

print(
    "  training negative sampling"
)

print(
    "  training epochs"
)

print(
    "  early stopping"
)

print(
    "  weight decay"
)

print(
    "  evaluation candidate generation"
)


print()
print(
    "Training performed:             NO"
)

print(
    "Numerical end-to-end forward:   NOT YET"
)

print(
    "Phase 2 reopened:               NO"
)

print(
    "Phase 3 reopened:               NO"
)


print()
print(
    "Outputs:"
)


for path in [

    static_contract_path,

    full_model_contract_path,

    hash_provenance_path,

    alignment_path,

    parameter_path,

    namespace_path,

    forward_path,

    static_hash_path,

    contract_hash_path,
]:

    print(
        f"  {path}"
    )


print()
print(
    "PHASE 4.6.1b STATUS: COMPLETE — "
    "AUTHORITATIVE STATIC INPUTS AND "
    "FULL ITRS TOPOLOGY FROZEN"
)


print()
print(
    "NEXT:"
)

print(
    "PHASE 4.6.2 — "
    "COMPLETE END-TO-END ITRS "
    "FORWARD + BCE AUDIT"
)