from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy import sparse


# =============================================================================
# PHASE 4.6.2 — COMPLETE END-TO-END ITRS FORWARD + BCE AUDIT
#
# PURPOSE
# -------
# Execute one REAL frozen Phase-2 T60 VALIDATION positive event through the
# complete reconstructed ITRS architecture:
#
#   Static Doc2Vec / labels
#       ->
#   Description extraction F_d
#       ->
#   Shared latent embeddings L_o / L_b
#       ->
#   Trend extraction F_t
#       ->
#   R-GCN preference propagation F_s
#       ->
#   280-D pair representation
#       ->
#   4-hidden-layer scoring MLP
#       ->
#   raw logit
#       ->
#   sigmoid probability
#       ->
#   BCEWithLogitsLoss
#       ->
#   full backward pass
#
#
# WHY A REAL VALIDATION POSITIVE?
# -------------------------------
# Negative sampling remains explicitly UNFROZEN.
#
# Therefore this integration audit uses:
#
#   - an actual observed T60 validation investment,
#   - target y = 1,
#   - NO fabricated negative candidate,
#   - NO test example.
#
#
# AUDIT-CASE SELECTION
# --------------------
# To ensure that the selected case genuinely exercises all major branches,
# the validation event must satisfy:
#
#   - Investor has prior historical memberships,
#   - at least one historical period has >= 2 startups,
#   - Investor has structural incoming degree > 0,
#   - candidate Startup has structural incoming degree > 0,
#   - candidate Startup has at least one category label,
#   - Investor Doc2Vec is nonzero,
#   - candidate Startup Doc2Vec is nonzero.
#
# These criteria are ONLY an audit-case selection mechanism.
#
# They DO NOT modify:
#
#   - training eligibility,
#   - evaluation eligibility,
#   - cold-start policy,
#   - negative sampling,
#   - Phase-2 splits,
#   - Phase-3 graph structure.
#
#
# INITIALIZATION
# --------------
# The paper reports Kaiming initialization, but its exact variant remains
# unfrozen.
#
# Therefore this script uses deterministic NON-KAIMING audit-only values.
#
# These values:
#
#   - are never saved as model state,
#   - are never used for training,
#   - do not freeze initialization policy.
#
#
# IMPORTANT
# ---------
# THIS SCRIPT DOES NOT TRAIN.
#
# It performs exactly:
#
#   one forward,
#   one BCE calculation,
#   one backward pass.
#
# There is NO optimizer and NO optimizer.step().
# =============================================================================


# =============================================================================
# ROOTS
# =============================================================================

PHASE_2_ROOT = Path(
    "data/experimental/phase_2"
)

PHASE_3_ROOT = Path(
    "data/experimental/phase_3"
)

PHASE_4_ROOT = Path(
    "data/experimental/phase_4"
)


# =============================================================================
# PHASE-2 POSITIVE EVENT INPUT
# =============================================================================

TEMPORAL_SPLIT_PATH = (
    PHASE_2_ROOT
    / "model_ready"
    / "interactions_itrs_temporal_split.parquet"
)


# =============================================================================
# PHASE-3 GRAPH INPUTS
# =============================================================================

NODE_INDEX_PATH = (
    PHASE_3_ROOT
    / "model_ready"
    / "node_index.parquet"
)

EDGE_INDEX_PATH = (
    PHASE_3_ROOT
    / "model_ready"
    / "edge_index.npy"
)

EDGE_TYPE_PATH = (
    PHASE_3_ROOT
    / "model_ready"
    / "edge_type.npy"
)

RELATION_INDEX_PATH = (
    PHASE_3_ROOT
    / "model_ready"
    / "relation_index.csv"
)


# =============================================================================
# PHASE-4 STATIC DESCRIPTION INPUTS
# =============================================================================

DOC2VEC_ALL_PATH = (
    PHASE_4_ROOT
    / "doc2vec"
    / "vectors"
    / "doc2vec_vectors_all.npy"
)

DOC2VEC_MANIFEST_PATH = (
    PHASE_4_ROOT
    / "doc2vec"
    / "vectors"
    / "doc2vec_vector_manifest.parquet"
)

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


# =============================================================================
# PHASE-4 TREND RUNTIME
# =============================================================================

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


# =============================================================================
# PHASE-4 FULL-MODEL CONTRACTS
# =============================================================================

STATIC_INPUT_CONTRACT_PATH = (
    PHASE_4_ROOT
    / "full_model_contract"
    / "full_model_static_input_contract.json"
)

FULL_MODEL_TOPOLOGY_CONTRACT_PATH = (
    PHASE_4_ROOT
    / "full_model_contract"
    / "full_itrs_model_topology_contract.json"
)

PHASE_4_6_1B_HASHES_PATH = (
    PHASE_4_ROOT
    / "full_model_contract"
    / "phase_4_6_1b_contract_hashes.csv"
)


# =============================================================================
# OUTPUTS
# =============================================================================

OUT_DIR = (
    PHASE_4_ROOT
    / "full_model_forward_audit"
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

NUM_HISTORY_PERIODS = 60


# =============================================================================
# FROZEN DIMENSIONS
# =============================================================================

LATENT_DIM = 40

DOC2VEC_DIM = 32
LABEL_DIM = 802

DESCRIPTION_TEXT_DIM = 20
DESCRIPTION_LABEL_DIM = 20
DESCRIPTION_DIM = 40

TREND_QUERY_DIM = 80
TREND_ITEM_DIM = 80
TREND_HIDDEN_DIM = 40
TREND_DIM = 40

STRUCTURAL_DIM = 40

INVESTOR_SCORING_DIM = 160
STARTUP_SCORING_DIM = 120

PAIR_DIM = 280


# =============================================================================
# FROZEN MODEL PARAMETER COUNTS
# =============================================================================

EXPECTED_PARAMETER_COUNTS = {

    "investor_embedding":
        6_639_000,

    "startup_embedding":
        12_463_560,

    "description":
        16_720,

    "trend":
        32_480,

    "rgcn":
        19_320,

    "scoring":
        46_849,
}

EXPECTED_FULL_PARAMETERS = 19_217_929


# =============================================================================
# NUMERICAL TOLERANCES
# =============================================================================

ATOL = 1e-6
RTOL = 1e-6


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
        f"Missing required contract: {path}",
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


def count_parameters(module):

    return sum(
        parameter.numel()
        for parameter
        in module.parameters()
    )


def gradient_stats(
    name,
    parameter,
):

    gradient = parameter.grad

    exists = (
        gradient
        is not None
    )

    finite = (
        exists
        and bool(
            torch.isfinite(
                gradient
            ).all()
        )
    )

    abs_sum = (
        float(
            gradient
            .detach()
            .abs()
            .sum()
        )
        if exists
        else 0.0
    )

    nonzero = (
        abs_sum
        > 0.0
    )

    return {

        "parameter":
            name,

        "gradient_exists":
            exists,

        "gradient_finite":
            finite,

        "gradient_abs_sum":
            abs_sum,

        "gradient_nonzero":
            nonzero,
    }


def initialize_tensor_pattern(
    tensor,
    base,
    amplitude,
    phase,
):

    with torch.no_grad():

        flat_index = torch.arange(
            tensor.numel(),
            dtype=tensor.dtype,
            device=tensor.device,
        )

        values = (
            base
            +
            amplitude
            * torch.sin(
                flat_index
                * 0.017
                + phase
            )
        )

        tensor.copy_(
            values.reshape_as(
                tensor
            )
        )


def initialize_embedding_pattern(
    embedding,
    phase,
    chunk_size=8192,
):

    num_rows = (
        embedding.num_embeddings
    )

    dim = (
        embedding.embedding_dim
    )

    column_axis = torch.arange(
        dim,
        dtype=embedding.weight.dtype,
        device=embedding.weight.device,
    ).unsqueeze(
        0
    )

    with torch.no_grad():

        for start in range(
            0,
            num_rows,
            chunk_size,
        ):

            end = min(
                start
                + chunk_size,
                num_rows,
            )

            row_axis = torch.arange(
                start,
                end,
                dtype=embedding.weight.dtype,
                device=embedding.weight.device,
            ).unsqueeze(
                1
            )

            values = (
                0.030
                +
                0.004
                * torch.sin(
                    row_axis
                    * 0.000173
                    +
                    column_axis
                    * 0.137
                    +
                    phase
                )
                +
                0.002
                * torch.cos(
                    row_axis
                    * 0.000071
                    +
                    column_axis
                    * 0.091
                    +
                    phase
                )
            )

            embedding.weight[
                start:end
            ].copy_(
                values
            )


# =============================================================================
# DESCRIPTION ENCODER
# =============================================================================

class DescriptionEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.text_projection = nn.Linear(
            DOC2VEC_DIM,
            DESCRIPTION_TEXT_DIM,
            bias=True,
        )

        self.label_projection = nn.Linear(
            LABEL_DIM,
            DESCRIPTION_LABEL_DIM,
            bias=True,
        )


    def forward(
        self,
        doc2vec,
        labels,
    ):

        text_feature = F.relu(
            self.text_projection(
                doc2vec
            )
        )

        label_feature = F.relu(
            self.label_projection(
                labels
            )
        )

        return torch.cat(
            [
                text_feature,
                label_feature,
            ],
            dim=1,
        )


# =============================================================================
# TREND EXTRACTOR
# =============================================================================

class TrendExtractor(nn.Module):

    def __init__(self):

        super().__init__()

        self.attention_weight = nn.Parameter(
            torch.empty(
                TREND_QUERY_DIM,
                TREND_ITEM_DIM,
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
        )

        self.output_projection = nn.Linear(
            TREND_HIDDEN_DIM,
            TREND_DIM,
            bias=False,
        )


    def attend_period(
        self,
        query,
        items,
    ):

        require(
            query.shape
            == (
                TREND_QUERY_DIM,
            ),
            "Trend query shape mismatch.",
        )

        require(
            items.ndim
            == 2,
            "Trend items must be rank-2.",
        )

        require(
            items.shape[
                1
            ]
            == TREND_ITEM_DIM,
            "Trend item dimension mismatch.",
        )

        projected_query = (
            query
            @ self.attention_weight
        )

        scores = (
            projected_query
            @ items.T
        )

        alpha = F.softmax(
            scores,
            dim=0,
        )

        period_vector = (
            alpha
            @ items
        )

        return (
            period_vector,
            alpha,
        )


    def encode_sequence(
        self,
        sequence,
    ):

        require(
            sequence.shape[
                1
            ]
            == NUM_HISTORY_PERIODS,
            (
                "T60 audit sequence must contain "
                "T0 through T59."
            ),
        )

        gru_output, _ = self.gru(
            sequence
        )

        last_hidden_output = (
            gru_output[
                :,
                -1,
                :
            ]
        )

        trend = torch.sigmoid(
            self.output_projection(
                last_hidden_output
            )
        )

        return (
            trend,
            gru_output,
        )


# =============================================================================
# BASIS R-GCN
# =============================================================================

class BasisRGCNLayer(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
    ):

        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.bases = nn.Parameter(
            torch.empty(
                5,
                in_dim,
                out_dim,
            )
        )

        self.coefficients = nn.Parameter(
            torch.empty(
                NUM_RELATIONS,
                5,
            )
        )

        self.root_weight = nn.Parameter(
            torch.empty(
                in_dim,
                out_dim,
            )
        )


    def effective_weight(
        self,
        relation_id,
    ):

        return torch.einsum(
            "b,bio->io",
            self.coefficients[
                relation_id
            ],
            self.bases,
        )


    def forward(
        self,
        x,
        edge_index,
        edge_type,
    ):

        num_nodes = (
            x.shape[
                0
            ]
        )

        aggregated = x.new_zeros(
            (
                num_nodes,
                self.out_dim,
            )
        )

        source_all = (
            edge_index[
                0
            ]
        )

        destination_all = (
            edge_index[
                1
            ]
        )

        for relation_id in range(
            NUM_RELATIONS
        ):

            mask = (
                edge_type
                == relation_id
            )

            relation_sources = (
                source_all[
                    mask
                ]
            )

            relation_destinations = (
                destination_all[
                    mask
                ]
            )

            require(
                relation_sources.numel()
                > 0,
                (
                    "Frozen relation unexpectedly "
                    f"contains zero edges: {relation_id}"
                ),
            )

            weight = self.effective_weight(
                relation_id
            )

            messages = (
                x[
                    relation_sources
                ]
                @ weight
            )

            relation_degree = torch.bincount(
                relation_destinations,
                minlength=num_nodes,
            )

            normalization = (
                relation_degree[
                    relation_destinations
                ]
                .to(
                    dtype=messages.dtype
                )
                .unsqueeze(
                    1
                )
            )

            require(
                bool(
                    torch.all(
                        normalization
                        > 0
                    )
                ),
                (
                    "Encountered zero incoming "
                    "normalization denominator."
                ),
            )

            messages = (
                messages
                /
                normalization
            )

            aggregated.index_add_(
                0,
                relation_destinations,
                messages,
            )

        root = (
            x
            @ self.root_weight
        )

        return (
            aggregated
            + root
        )


class PreferencePropagation(nn.Module):

    def __init__(self):

        super().__init__()

        self.layer_1 = BasisRGCNLayer(
            LATENT_DIM,
            STRUCTURAL_DIM,
        )

        self.layer_2 = BasisRGCNLayer(
            STRUCTURAL_DIM,
            STRUCTURAL_DIM,
        )


    def forward(
        self,
        latent_all,
        edge_index,
        edge_type,
    ):

        h1_pre = self.layer_1(
            latent_all,
            edge_index,
            edge_type,
        )

        h1 = F.relu(
            h1_pre
        )

        h2_pre = self.layer_2(
            h1,
            edge_index,
            edge_type,
        )

        h2 = F.relu(
            h2_pre
        )

        return {
            "h1_pre":
                h1_pre,

            "h1":
                h1,

            "h2_pre":
                h2_pre,

            "F_s":
                h2,
        }


# =============================================================================
# SCORING MLP
# =============================================================================

class ScoringMLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.hidden_1 = nn.Linear(
            280,
            128,
            bias=True,
        )

        self.hidden_2 = nn.Linear(
            128,
            64,
            bias=True,
        )

        self.hidden_3 = nn.Linear(
            64,
            32,
            bias=True,
        )

        self.hidden_4 = nn.Linear(
            32,
            16,
            bias=True,
        )

        self.output = nn.Linear(
            16,
            1,
            bias=True,
        )


    def forward(
        self,
        pair_features,
    ):

        h1 = F.relu(
            self.hidden_1(
                pair_features
            )
        )

        h2 = F.relu(
            self.hidden_2(
                h1
            )
        )

        h3 = F.relu(
            self.hidden_3(
                h2
            )
        )

        h4 = F.relu(
            self.hidden_4(
                h3
            )
        )

        logit = self.output(
            h4
        )

        return {
            "hidden_1":
                h1,

            "hidden_2":
                h2,

            "hidden_3":
                h3,

            "hidden_4":
                h4,

            "logit":
                logit,
        }


# =============================================================================
# COMPLETE RUNTIME AUDIT MODEL
# =============================================================================

class ITRSRuntimeAuditModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.investor_embedding = nn.Embedding(
            NUM_INVESTORS,
            LATENT_DIM,
        )

        self.startup_embedding = nn.Embedding(
            NUM_STARTUPS,
            LATENT_DIM,
        )

        self.description_encoder = (
            DescriptionEncoder()
        )

        self.trend_extractor = (
            TrendExtractor()
        )

        self.preference_propagation = (
            PreferencePropagation()
        )

        self.scoring_mlp = (
            ScoringMLP()
        )


# =============================================================================
# DETERMINISTIC AUDIT-ONLY INITIALIZATION
# =============================================================================

def initialize_audit_model(
    model,
):

    # -------------------------------------------------------------------------
    # Shared latent tables
    # -------------------------------------------------------------------------

    initialize_embedding_pattern(
        model.investor_embedding,
        phase=0.3,
    )

    initialize_embedding_pattern(
        model.startup_embedding,
        phase=1.1,
    )


    # -------------------------------------------------------------------------
    # Description projections
    # -------------------------------------------------------------------------

    initialize_tensor_pattern(
        model
        .description_encoder
        .text_projection
        .weight,
        base=0.0030,
        amplitude=0.0015,
        phase=0.2,
    )

    initialize_tensor_pattern(
        model
        .description_encoder
        .text_projection
        .bias,
        base=0.0120,
        amplitude=0.0010,
        phase=0.4,
    )

    initialize_tensor_pattern(
        model
        .description_encoder
        .label_projection
        .weight,
        base=0.0015,
        amplitude=0.0007,
        phase=0.6,
    )

    initialize_tensor_pattern(
        model
        .description_encoder
        .label_projection
        .bias,
        base=0.0100,
        amplitude=0.0010,
        phase=0.8,
    )


    # -------------------------------------------------------------------------
    # Trend attention
    # -------------------------------------------------------------------------

    initialize_tensor_pattern(
        model
        .trend_extractor
        .attention_weight,
        base=0.0020,
        amplitude=0.0010,
        phase=1.0,
    )


    # -------------------------------------------------------------------------
    # GRU
    # -------------------------------------------------------------------------

    for parameter_index, (
        parameter_name,
        parameter,
    ) in enumerate(
        model
        .trend_extractor
        .gru
        .named_parameters()
    ):

        if "bias" in parameter_name:

            initialize_tensor_pattern(
                parameter,
                base=0.0080,
                amplitude=0.0020,
                phase=1.2
                + parameter_index
                * 0.1,
            )

        else:

            initialize_tensor_pattern(
                parameter,
                base=0.0040,
                amplitude=0.0020,
                phase=1.2
                + parameter_index
                * 0.1,
            )


    initialize_tensor_pattern(
        model
        .trend_extractor
        .output_projection
        .weight,
        base=0.0060,
        amplitude=0.0020,
        phase=2.0,
    )


    # -------------------------------------------------------------------------
    # R-GCN
    # -------------------------------------------------------------------------

    for layer_index, layer in enumerate(
        [
            model
            .preference_propagation
            .layer_1,

            model
            .preference_propagation
            .layer_2,
        ],
        start=1,
    ):

        initialize_tensor_pattern(
            layer.bases,
            base=0.0040,
            amplitude=0.0015,
            phase=2.4
            + layer_index,
        )

        initialize_tensor_pattern(
            layer.coefficients,
            base=0.1000,
            amplitude=0.0200,
            phase=3.1
            + layer_index,
        )

        initialize_tensor_pattern(
            layer.root_weight,
            base=0.0040,
            amplitude=0.0015,
            phase=3.8
            + layer_index,
        )


    # -------------------------------------------------------------------------
    # Scoring MLP
    # -------------------------------------------------------------------------

    scoring_layers = [

        model
        .scoring_mlp
        .hidden_1,

        model
        .scoring_mlp
        .hidden_2,

        model
        .scoring_mlp
        .hidden_3,

        model
        .scoring_mlp
        .hidden_4,

        model
        .scoring_mlp
        .output,
    ]


    for layer_index, layer in enumerate(
        scoring_layers,
        start=1,
    ):

        initialize_tensor_pattern(
            layer.weight,
            base=0.0020,
            amplitude=0.0008,
            phase=5.0
            + layer_index
            * 0.3,
        )

        initialize_tensor_pattern(
            layer.bias,
            base=0.0060,
            amplitude=0.0010,
            phase=6.0
            + layer_index
            * 0.3,
        )


# =============================================================================
# START
# =============================================================================

banner(
    "PHASE 4.6.2 — "
    "COMPLETE END-TO-END ITRS FORWARD + BCE AUDIT"
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
    "Device:  CPU"
)


# =============================================================================
# 2. UPSTREAM PHASE 4.6.1b CONTRACT
# =============================================================================

banner(
    "UPSTREAM FULL-MODEL CONTRACT"
)


static_contract = load_json(
    STATIC_INPUT_CONTRACT_PATH
)

topology_contract = load_json(
    FULL_MODEL_TOPOLOGY_CONTRACT_PATH
)


require(
    static_contract.get(
        "status"
    )
    == "FROZEN",
    "Phase 4.6.1b static contract is not frozen.",
)

require(
    topology_contract.get(
        "status"
    )
    == "FROZEN",
    "Phase 4.6.1b topology contract is not frozen.",
)

require(
    topology_contract[
        "parameter_budget"
    ][
        "total"
    ]
    == EXPECTED_FULL_PARAMETERS,
    "Full parameter budget changed.",
)

require(
    topology_contract[
        "latent_embeddings"
    ][
        "number_of_tables"
    ]
    == 2,
    "Shared latent-table count changed.",
)


print(
    "Static input contract:    PASS"
)

print(
    "Full topology contract:   PASS"
)

print(
    f"Full trainable parameters: "
    f"{EXPECTED_FULL_PARAMETERS:,}"
)


# =============================================================================
# 3. LOAD FROZEN STATIC INPUTS
# =============================================================================

banner(
    "LOADING FROZEN STATIC INPUTS"
)


doc2vec_all = np.load(
    DOC2VEC_ALL_PATH,
    mmap_mode="r",
)

labels_sparse = sparse.load_npz(
    LABEL_MATRIX_PATH
).tocsr()

node_index = pd.read_parquet(
    NODE_INDEX_PATH
)

doc2vec_manifest = pd.read_parquet(
    DOC2VEC_MANIFEST_PATH
)

label_manifest = pd.read_parquet(
    LABEL_MANIFEST_PATH
)

edge_index_np = np.load(
    EDGE_INDEX_PATH,
    mmap_mode="r",
)

edge_type_np = np.load(
    EDGE_TYPE_PATH,
    mmap_mode="r",
)

relation_index = pd.read_csv(
    RELATION_INDEX_PATH
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


require(
    doc2vec_all.shape
    == (
        NUM_NODES,
        DOC2VEC_DIM,
    ),
    "Doc2Vec shape changed.",
)

require(
    labels_sparse.shape
    == (
        NUM_NODES,
        LABEL_DIM,
    ),
    "Label shape changed.",
)

require(
    edge_index_np.shape
    == (
        2,
        NUM_EDGES,
    ),
    "edge_index shape changed.",
)

require(
    edge_type_np.shape
    == (
        NUM_EDGES,
    ),
    "edge_type shape changed.",
)

require(
    len(
        relation_index
    )
    == NUM_RELATIONS,
    "Relation vocabulary changed.",
)

require(
    trend_period_counts.shape
    == (
        NUM_INVESTORS
        * NUM_HISTORY_PERIODS,
    ),
    "Trend count-array shape changed.",
)


print(
    f"Doc2Vec:          "
    f"{doc2vec_all.shape}"
)

print(
    f"Labels:           "
    f"{labels_sparse.shape}"
)

print(
    f"Graph:            "
    f"{edge_index_np.shape}"
)

print(
    f"Trend memberships:"
    f" {trend_startup_indices.shape}"
)


# =============================================================================
# 4. VALIDATE STATIC ROW ALIGNMENT AGAIN
# =============================================================================

banner(
    "STATIC ROW ALIGNMENT RECHECK"
)


expected_node_rows = np.arange(
    NUM_NODES,
    dtype=np.int64,
)


phase3_rows = (
    node_index[
        "node_index"
    ]
    .to_numpy(
        dtype=np.int64
    )
)

doc_rows = (
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


require(
    np.array_equal(
        phase3_rows,
        expected_node_rows,
    ),
    "Phase-3 row ordering changed.",
)

require(
    np.array_equal(
        phase3_rows,
        doc_rows,
    ),
    "Doc2Vec row ordering changed.",
)

require(
    np.array_equal(
        phase3_rows,
        label_rows,
    ),
    "Label row ordering changed.",
)


print(
    "Phase-3 / Doc2Vec / labels: PASS"
)


# =============================================================================
# 5. BUILD ROLE-SPECIFIC NODE LOOKUPS
# =============================================================================

banner(
    "ROLE-SPECIFIC NODE LOOKUPS"
)


investor_nodes = node_index[
    node_index[
        "node_type"
    ]
    == "investor"
].copy()


startup_nodes = node_index[
    node_index[
        "node_type"
    ]
    == "startup"
].copy()


require(
    len(
        investor_nodes
    )
    == NUM_INVESTORS,
    "Investor node population changed.",
)

require(
    len(
        startup_nodes
    )
    == NUM_STARTUPS,
    "Startup node population changed.",
)


investor_raw_ids = (
    investor_nodes[
        "raw_entity_id"
    ]
    .astype(str)
)

startup_raw_ids = (
    startup_nodes[
        "raw_entity_id"
    ]
    .astype(str)
)


require(
    investor_raw_ids.nunique()
    == NUM_INVESTORS,
    "Investor raw IDs are not unique by role.",
)

require(
    startup_raw_ids.nunique()
    == NUM_STARTUPS,
    "Startup raw IDs are not unique by role.",
)


investor_lookup = pd.Series(
    investor_nodes[
        "node_index"
    ]
    .to_numpy(
        dtype=np.int64
    ),
    index=investor_raw_ids,
)

startup_lookup = pd.Series(
    startup_nodes[
        "node_index"
    ]
    .to_numpy(
        dtype=np.int64
    ),
    index=startup_raw_ids,
)


print(
    f"Investor lookup: "
    f"{len(investor_lookup):,}"
)

print(
    f"Startup lookup:  "
    f"{len(startup_lookup):,}"
)


# =============================================================================
# 6. GRAPH INCOMING DEGREE
# =============================================================================

banner(
    "STRUCTURAL INCOMING DEGREE"
)


graph_destination = np.asarray(
    edge_index_np[
        1
    ],
    dtype=np.int64,
)


incoming_degree = np.bincount(
    graph_destination,
    minlength=NUM_NODES,
)


print(
    f"Nodes with incoming structural degree > 0: "
    f"{int((incoming_degree > 0).sum()):,}"
)


# =============================================================================
# 7. LOAD REAL T60 VALIDATION POSITIVES
# =============================================================================

banner(
    "T60 VALIDATION POSITIVE POOL"
)


validation_columns = [

    "interaction_id",
    "funding_round_id",
    "investor_id",
    "startup_id",
    "segment_number",
    "segment_label",
    "experiment_split",
]


validation_events = pd.read_parquet(
    TEMPORAL_SPLIT_PATH,
    columns=validation_columns,
    filters=[
        (
            "segment_number",
            "==",
            60,
        ),
        (
            "experiment_split",
            "==",
            "validation",
        ),
    ],
)


require(
    len(
        validation_events
    )
    > 0,
    "No T60 validation events found.",
)


require(
    validation_events[
        "segment_number"
    ]
    .eq(
        60
    )
    .all(),
    "Validation pool contains non-T60 event.",
)


require(
    validation_events[
        "experiment_split"
    ]
    .eq(
        "validation"
    )
    .all(),
    "Validation pool contains non-validation event.",
)


print(
    f"T60 validation events: "
    f"{len(validation_events):,}"
)


# =============================================================================
# 8. MAP VALIDATION EVENTS TO FROZEN GLOBAL NODES
# =============================================================================

banner(
    "VALIDATION EVENT -> GLOBAL NODE MAPPING"
)


validation_events[
    "investor_id_string"
] = (
    validation_events[
        "investor_id"
    ]
    .astype(str)
)


validation_events[
    "startup_id_string"
] = (
    validation_events[
        "startup_id"
    ]
    .astype(str)
)


validation_events[
    "investor_global"
] = (
    validation_events[
        "investor_id_string"
    ]
    .map(
        investor_lookup
    )
)


validation_events[
    "startup_global"
] = (
    validation_events[
        "startup_id_string"
    ]
    .map(
        startup_lookup
    )
)


mapped = (
    validation_events[
        "investor_global"
    ]
    .notna()
    &
    validation_events[
        "startup_global"
    ]
    .notna()
)


print(
    f"Mapped validation events: "
    f"{int(mapped.sum()):,}"
)

print(
    f"Unmapped validation events: "
    f"{int((~mapped).sum()):,}"
)


require(
    bool(
        mapped.all()
    ),
    (
        "At least one frozen validation event "
        "failed role-node mapping."
    ),
)


validation_events[
    "investor_global"
] = (
    validation_events[
        "investor_global"
    ]
    .astype(
        np.int64
    )
)


validation_events[
    "startup_global"
] = (
    validation_events[
        "startup_global"
    ]
    .astype(
        np.int64
    )
)


# =============================================================================
# 9. TREND HISTORY DIAGNOSTICS FOR VALIDATION INVESTORS
# =============================================================================

banner(
    "VALIDATION INVESTOR HISTORY DIAGNOSTICS"
)


trend_counts_2d = (
    trend_period_counts
    .reshape(
        NUM_INVESTORS,
        NUM_HISTORY_PERIODS,
    )
)


validation_investor_indices = (
    validation_events[
        "investor_global"
    ]
    .to_numpy(
        dtype=np.int64
    )
)


validation_history_counts = np.asarray(
    trend_counts_2d[
        validation_investor_indices
    ],
    dtype=np.int32,
)


validation_events[
    "history_memberships"
] = (
    validation_history_counts
    .sum(
        axis=1
    )
)


validation_events[
    "active_history_periods"
] = (
    (
        validation_history_counts
        > 0
    )
    .sum(
        axis=1
    )
)


validation_events[
    "max_period_items"
] = (
    validation_history_counts
    .max(
        axis=1
    )
)


print(
    "Validation events with prior history:"
)

print(
    f"  "
    f"{int((validation_events['history_memberships'] > 0).sum()):,}"
)


print(
    "Validation events with at least one "
    "multi-item history period:"
)

print(
    f"  "
    f"{int((validation_events['max_period_items'] >= 2).sum()):,}"
)


# =============================================================================
# 10. ADD STRUCTURAL / STATIC FEATURE DIAGNOSTICS
# =============================================================================

banner(
    "VALIDATION PAIR AUDIT-CASE ELIGIBILITY"
)


doc_zero_flags = (
    doc2vec_manifest[
        "doc2vec_zero_vector"
    ]
    .astype(bool)
    .to_numpy()
)


label_counts = (
    label_manifest[
        "label_count"
    ]
    .fillna(
        0
    )
    .astype(
        np.int64
    )
    .to_numpy()
)


investor_global_array = (
    validation_events[
        "investor_global"
    ]
    .to_numpy(
        dtype=np.int64
    )
)

startup_global_array = (
    validation_events[
        "startup_global"
    ]
    .to_numpy(
        dtype=np.int64
    )
)


validation_events[
    "investor_incoming_degree"
] = incoming_degree[
    investor_global_array
]

validation_events[
    "startup_incoming_degree"
] = incoming_degree[
    startup_global_array
]

validation_events[
    "investor_doc2vec_nonzero"
] = ~doc_zero_flags[
    investor_global_array
]

validation_events[
    "startup_doc2vec_nonzero"
] = ~doc_zero_flags[
    startup_global_array
]

validation_events[
    "startup_label_count"
] = label_counts[
    startup_global_array
]


eligible_mask = (

    (
        validation_events[
            "history_memberships"
        ]
        > 0
    )

    &

    (
        validation_events[
            "max_period_items"
        ]
        >= 2
    )

    &

    (
        validation_events[
            "investor_incoming_degree"
        ]
        > 0
    )

    &

    (
        validation_events[
            "startup_incoming_degree"
        ]
        > 0
    )

    &

    validation_events[
        "investor_doc2vec_nonzero"
    ]

    &

    validation_events[
        "startup_doc2vec_nonzero"
    ]

    &

    (
        validation_events[
            "startup_label_count"
        ]
        > 0
    )
)


eligible_events = (
    validation_events[
        eligible_mask
    ]
    .copy()
)


print(
    f"Audit-eligible validation events: "
    f"{len(eligible_events):,}"
)


require(
    len(
        eligible_events
    )
    > 0,
    (
        "No T60 validation event satisfies "
        "the audit-only branch-exercising criteria."
    ),
)


# =============================================================================
# 11. DETERMINISTIC AUDIT-CASE SELECTION
#
# Prefer the eligible investor with the SMALLEST history footprint.
#
# This keeps the audit computationally compact while still requiring:
#
#   - prior history,
#   - at least one multi-item period,
#   - structural connectivity.
#
# This is NOT a model-selection criterion.
# =============================================================================

banner(
    "SELECTING DETERMINISTIC REAL VALIDATION PAIR"
)


eligible_events = (
    eligible_events
    .sort_values(
        by=[
            "history_memberships",
            "active_history_periods",
            "interaction_id",
        ],
        ascending=[
            True,
            True,
            True,
        ],
        kind="mergesort",
    )
    .reset_index(
        drop=True
    )
)


selected = (
    eligible_events
    .iloc[
        0
    ]
)


selected_interaction_id = str(
    selected[
        "interaction_id"
    ]
)

selected_funding_round_id = str(
    selected[
        "funding_round_id"
    ]
)

selected_investor_id = str(
    selected[
        "investor_id"
    ]
)

selected_startup_id = str(
    selected[
        "startup_id"
    ]
)

investor_global = int(
    selected[
        "investor_global"
    ]
)

startup_global = int(
    selected[
        "startup_global"
    ]
)

startup_local = (
    startup_global
    - NUM_INVESTORS
)


require(
    0
    <= investor_global
    < NUM_INVESTORS,
    "Selected Investor global index invalid.",
)


require(
    0
    <= startup_local
    < NUM_STARTUPS,
    "Selected Startup local index invalid.",
)


print(
    f"interaction_id:          "
    f"{selected_interaction_id}"
)

print(
    f"funding_round_id:        "
    f"{selected_funding_round_id}"
)

print(
    f"investor_id:             "
    f"{selected_investor_id}"
)

print(
    f"startup_id:              "
    f"{selected_startup_id}"
)

print(
    f"investor_global:         "
    f"{investor_global}"
)

print(
    f"startup_global:          "
    f"{startup_global}"
)

print(
    f"startup_local:           "
    f"{startup_local}"
)

print(
    f"history memberships:     "
    f"{int(selected['history_memberships'])}"
)

print(
    f"active history periods:  "
    f"{int(selected['active_history_periods'])}"
)

print(
    f"max period items:        "
    f"{int(selected['max_period_items'])}"
)

print(
    f"Investor incoming degree:"
    f" {int(selected['investor_incoming_degree'])}"
)

print(
    f"Startup incoming degree: "
    f"{int(selected['startup_incoming_degree'])}"
)

print(
    f"Startup label count:     "
    f"{int(selected['startup_label_count'])}"
)


# =============================================================================
# 12. EXTRACT EXACT T0..T59 HISTORY FOR SELECTED INVESTOR
# =============================================================================

banner(
    "SELECTED INVESTOR T0..T59 HISTORY"
)


period_base = (
    investor_global
    * NUM_HISTORY_PERIODS
)


history_period_startups = []

history_period_counts_python = []


for period_index in range(
    NUM_HISTORY_PERIODS
):

    flattened_period_index = (
        period_base
        + period_index
    )

    start = int(
        trend_period_ptr[
            flattened_period_index
        ]
    )

    end = int(
        trend_period_ptr[
            flattened_period_index
            + 1
        ]
    )

    startup_nodes_period = np.asarray(
        trend_startup_indices[
            start:end
        ],
        dtype=np.int64,
    )

    expected_count = int(
        trend_period_counts[
            flattened_period_index
        ]
    )

    require(
        len(
            startup_nodes_period
        )
        == expected_count,
        (
            "Trend CSR count mismatch in "
            f"period index {period_index}."
        ),
    )

    if len(
        startup_nodes_period
    ) > 0:

        require(
            bool(
                np.all(
                    (
                        startup_nodes_period
                        >= NUM_INVESTORS
                    )
                    &
                    (
                        startup_nodes_period
                        < NUM_NODES
                    )
                )
            ),
            "History contains non-Startup node index.",
        )

    history_period_startups.append(
        startup_nodes_period
    )

    history_period_counts_python.append(
        expected_count
    )


history_total = int(
    sum(
        history_period_counts_python
    )
)

active_periods = int(
    sum(
        count
        > 0
        for count
        in history_period_counts_python
    )
)

multi_item_periods = [

    index

    for index, count
    in enumerate(
        history_period_counts_python
    )

    if count >= 2
]


print(
    f"History memberships: "
    f"{history_total}"
)

print(
    f"Active periods:       "
    f"{active_periods}"
)

print(
    f"Empty periods:        "
    f"{NUM_HISTORY_PERIODS - active_periods}"
)

print(
    f"Multi-item periods:   "
    f"{len(multi_item_periods)}"
)

print(
    f"Max period items:     "
    f"{max(history_period_counts_python)}"
)


require(
    history_total
    == int(
        selected[
            "history_memberships"
        ]
    ),
    "Selected history count changed after CSR extraction.",
)


require(
    len(
        multi_item_periods
    )
    >= 1,
    "Selected audit case does not exercise attention.",
)


# =============================================================================
# 13. BUILD REQUIRED DESCRIPTION-NODE SET
# =============================================================================

banner(
    "DESCRIPTION SUBSET FOR REAL FORWARD"
)


nonempty_history_arrays = [

    array

    for array
    in history_period_startups

    if len(
        array
    )
    > 0
]


history_flat = np.concatenate(
    nonempty_history_arrays
)


history_unique = np.unique(
    history_flat
)


required_global_nodes = np.unique(
    np.concatenate(
        [
            np.array(
                [
                    investor_global,
                    startup_global,
                ],
                dtype=np.int64,
            ),

            history_unique,
        ]
    )
)


print(
    f"Unique historical Startups: "
    f"{len(history_unique):,}"
)

print(
    f"Required description nodes: "
    f"{len(required_global_nodes):,}"
)


# =============================================================================
# 14. LOAD REAL DESCRIPTION ROWS ONLY
# =============================================================================

banner(
    "REAL DOC2VEC + LABEL ROW EXTRACTION"
)


doc2vec_subset_np = np.asarray(
    doc2vec_all[
        required_global_nodes
    ],
    dtype=np.float32,
)


label_subset_np = (
    labels_sparse[
        required_global_nodes
    ]
    .toarray()
    .astype(
        np.float32,
        copy=False,
    )
)


print(
    f"Doc2Vec subset: "
    f"{doc2vec_subset_np.shape}"
)

print(
    f"Label subset:   "
    f"{label_subset_np.shape}"
)


require(
    doc2vec_subset_np.shape
    == (
        len(
            required_global_nodes
        ),
        DOC2VEC_DIM,
    ),
    "Description Doc2Vec subset shape invalid.",
)

require(
    label_subset_np.shape
    == (
        len(
            required_global_nodes
        ),
        LABEL_DIM,
    ),
    "Description label subset shape invalid.",
)


node_to_subset_position = {

    int(
        global_node
    ):
        position

    for position, global_node
    in enumerate(
        required_global_nodes
    )
}


investor_description_position = (
    node_to_subset_position[
        investor_global
    ]
)

startup_description_position = (
    node_to_subset_position[
        startup_global
    ]
)


# =============================================================================
# 15. INSTANTIATE COMPLETE REAL MODEL
# =============================================================================

banner(
    "INSTANTIATING COMPLETE REAL ITRS AUDIT MODEL"
)


model = ITRSRuntimeAuditModel()


actual_parameter_count = count_parameters(
    model
)


print(
    f"Actual trainable parameters: "
    f"{actual_parameter_count:,}"
)


require(
    actual_parameter_count
    == EXPECTED_FULL_PARAMETERS,
    "Runtime model parameter count differs from frozen topology.",
)


# =============================================================================
# 16. AUDIT-ONLY DETERMINISTIC INITIALIZATION
# =============================================================================

banner(
    "AUDIT-ONLY PARAMETER INITIALIZATION"
)


initialization_start = time.perf_counter()


initialize_audit_model(
    model
)


initialization_seconds = (
    time.perf_counter()
    - initialization_start
)


print(
    "Initialization family:"
)

print(
    "  DETERMINISTIC_NON_KAIMING_AUDIT_ONLY"
)

print(
    f"Initialization time: "
    f"{initialization_seconds:.3f} s"
)

print(
    "Model state persisted: NO"
)

print(
    "Final Kaiming variant frozen: NO"
)


# =============================================================================
# 17. TORCH STATIC GRAPH
# =============================================================================

banner(
    "TORCH GRAPH INPUT"
)


edge_index = torch.from_numpy(
    np.asarray(
        edge_index_np,
        dtype=np.int64,
    )
)

edge_type = torch.from_numpy(
    np.asarray(
        edge_type_np,
        dtype=np.int64,
    )
)


print(
    f"edge_index: "
    f"{tuple(edge_index.shape)} "
    f"{edge_index.dtype}"
)

print(
    f"edge_type:  "
    f"{tuple(edge_type.shape)} "
    f"{edge_type.dtype}"
)


# =============================================================================
# 18. FULL STRUCTURAL FORWARD
# =============================================================================

banner(
    "FULL R-GCN STRUCTURAL FORWARD"
)


structural_start = time.perf_counter()


latent_all = torch.cat(
    [
        model
        .investor_embedding
        .weight,

        model
        .startup_embedding
        .weight,
    ],
    dim=0,
)


require(
    latent_all.shape
    == (
        NUM_NODES,
        LATENT_DIM,
    ),
    "Combined latent R-GCN input shape invalid.",
)


structural = (
    model
    .preference_propagation(
        latent_all,
        edge_index,
        edge_type,
    )
)


F_s_all = structural[
    "F_s"
]


structural_seconds = (
    time.perf_counter()
    - structural_start
)


print(
    f"latent_all: "
    f"{tuple(latent_all.shape)}"
)

print(
    f"R-GCN h1:   "
    f"{tuple(structural['h1'].shape)}"
)

print(
    f"F_s_all:    "
    f"{tuple(F_s_all.shape)}"
)

print(
    f"finite:     "
    f"{bool(torch.isfinite(F_s_all).all())}"
)

print(
    f"runtime:    "
    f"{structural_seconds:.3f} s"
)


require(
    F_s_all.shape
    == (
        NUM_NODES,
        STRUCTURAL_DIM,
    ),
    "Structural output shape invalid.",
)

require(
    bool(
        torch.isfinite(
            F_s_all
        ).all()
    ),
    "Structural output contains non-finite values.",
)


# =============================================================================
# 19. REAL DESCRIPTION FORWARD
# =============================================================================

banner(
    "REAL DESCRIPTION FORWARD"
)


doc2vec_subset = torch.from_numpy(
    doc2vec_subset_np
)

label_subset = torch.from_numpy(
    label_subset_np
)


description_subset = (
    model
    .description_encoder(
        doc2vec_subset,
        label_subset,
    )
)


print(
    f"F_d subset: "
    f"{tuple(description_subset.shape)}"
)

print(
    f"finite:     "
    f"{bool(torch.isfinite(description_subset).all())}"
)


require(
    description_subset.shape
    == (
        len(
            required_global_nodes
        ),
        DESCRIPTION_DIM,
    ),
    "Description output shape invalid.",
)


F_d_investor = (
    description_subset[
        investor_description_position
        :
        investor_description_position
        + 1
    ]
)


F_d_startup = (
    description_subset[
        startup_description_position
        :
        startup_description_position
        + 1
    ]
)


require(
    F_d_investor.shape
    == (
        1,
        DESCRIPTION_DIM,
    ),
    "Investor F_d shape invalid.",
)

require(
    F_d_startup.shape
    == (
        1,
        DESCRIPTION_DIM,
    ),
    "Startup F_d shape invalid.",
)


# =============================================================================
# 20. SHARED LATENT LOOKUPS
# =============================================================================

banner(
    "SHARED LATENT LOOKUPS"
)


investor_index_tensor = torch.tensor(
    [
        investor_global,
    ],
    dtype=torch.long,
)

startup_local_tensor = torch.tensor(
    [
        startup_local,
    ],
    dtype=torch.long,
)


L_o = (
    model
    .investor_embedding(
        investor_index_tensor
    )
)


L_b = (
    model
    .startup_embedding(
        startup_local_tensor
    )
)


print(
    f"L_o: "
    f"{tuple(L_o.shape)}"
)

print(
    f"L_b: "
    f"{tuple(L_b.shape)}"
)


require(
    L_o.shape
    == (
        1,
        LATENT_DIM,
    ),
    "Investor latent shape invalid.",
)

require(
    L_b.shape
    == (
        1,
        LATENT_DIM,
    ),
    "Startup latent shape invalid.",
)


# =============================================================================
# 21. TREND QUERY
# =============================================================================

banner(
    "TREND QUERY"
)


trend_query = torch.cat(
    [
        L_o[
            0
        ],
        F_d_investor[
            0
        ],
    ],
    dim=0,
)


print(
    f"query shape: "
    f"{tuple(trend_query.shape)}"
)


require(
    trend_query.shape
    == (
        TREND_QUERY_DIM,
    ),
    "Trend query dimension invalid.",
)


# =============================================================================
# 22. BUILD 60 REAL ATTENDED PERIOD VECTORS
# =============================================================================

banner(
    "REAL T0..T59 ATTENTION FORWARD"
)


period_vectors = []

attention_diagnostics = []


for period_index in range(
    NUM_HISTORY_PERIODS
):

    startup_globals_period = (
        history_period_startups[
            period_index
        ]
    )

    item_count = len(
        startup_globals_period
    )


    if item_count == 0:

        period_vector = torch.zeros(
            TREND_ITEM_DIM,
            dtype=L_o.dtype,
        )

        period_vectors.append(
            period_vector
        )

        attention_diagnostics.append(
            {
                "period_index":
                    period_index,

                "item_count":
                    0,

                "attention_sum":
                    None,

                "attention_min":
                    None,

                "attention_max":
                    None,
            }
        )

        continue


    startup_locals_period_np = (
        startup_globals_period
        - NUM_INVESTORS
    )


    startup_locals_period = torch.from_numpy(
        startup_locals_period_np
        .astype(
            np.int64,
            copy=False,
        )
    )


    period_latents = (
        model
        .startup_embedding(
            startup_locals_period
        )
    )


    description_positions = torch.tensor(
        [
            node_to_subset_position[
                int(
                    global_node
                )
            ]

            for global_node
            in startup_globals_period
        ],
        dtype=torch.long,
    )


    period_descriptions = (
        description_subset[
            description_positions
        ]
    )


    period_items = torch.cat(
        [
            period_latents,
            period_descriptions,
        ],
        dim=1,
    )


    require(
        period_items.shape
        == (
            item_count,
            TREND_ITEM_DIM,
        ),
        (
            "Trend item representation shape "
            f"invalid for period {period_index}."
        ),
    )


    (
        period_vector,
        alpha,
    ) = (
        model
        .trend_extractor
        .attend_period(
            trend_query,
            period_items,
        )
    )


    require(
        period_vector.shape
        == (
            TREND_ITEM_DIM,
        ),
        (
            "Attended period-vector shape "
            f"invalid for period {period_index}."
        ),
    )


    attention_sum = float(
        alpha
        .detach()
        .sum()
    )


    require(
        abs(
            attention_sum
            - 1.0
        )
        <= 1e-5,
        (
            "Attention weights do not sum "
            f"to 1 in period {period_index}."
        ),
    )


    period_vectors.append(
        period_vector
    )


    attention_diagnostics.append(
        {
            "period_index":
                period_index,

            "item_count":
                item_count,

            "attention_sum":
                attention_sum,

            "attention_min":
                float(
                    alpha
                    .detach()
                    .min()
                ),

            "attention_max":
                float(
                    alpha
                    .detach()
                    .max()
                ),
        }
    )


trend_sequence = torch.stack(
    period_vectors,
    dim=0,
).unsqueeze(
    0
)


print(
    f"Trend sequence: "
    f"{tuple(trend_sequence.shape)}"
)

print(
    f"Active periods: "
    f"{active_periods}"
)

print(
    f"Multi-item periods: "
    f"{len(multi_item_periods)}"
)


require(
    trend_sequence.shape
    == (
        1,
        NUM_HISTORY_PERIODS,
        TREND_ITEM_DIM,
    ),
    "Complete trend sequence shape invalid.",
)


# =============================================================================
# 23. GRU TREND FORWARD
# =============================================================================

banner(
    "GRU TREND FORWARD"
)


(
    F_t,
    gru_output,
) = (
    model
    .trend_extractor
    .encode_sequence(
        trend_sequence
    )
)


print(
    f"GRU output: "
    f"{tuple(gru_output.shape)}"
)

print(
    f"F_t:        "
    f"{tuple(F_t.shape)}"
)

print(
    f"F_t finite: "
    f"{bool(torch.isfinite(F_t).all())}"
)


require(
    F_t.shape
    == (
        1,
        TREND_DIM,
    ),
    "Final trend feature shape invalid.",
)

require(
    bool(
        torch.isfinite(
            F_t
        ).all()
    ),
    "Trend output contains non-finite values.",
)


# =============================================================================
# 24. STRUCTURAL ROW LOOKUPS
# =============================================================================

banner(
    "STRUCTURAL FEATURE LOOKUPS"
)


F_s_investor = F_s_all[
    investor_global
    :
    investor_global
    + 1
]


F_s_startup = F_s_all[
    startup_global
    :
    startup_global
    + 1
]


print(
    f"F_s,o: "
    f"{tuple(F_s_investor.shape)}"
)

print(
    f"F_s,b: "
    f"{tuple(F_s_startup.shape)}"
)


require(
    F_s_investor.shape
    == (
        1,
        STRUCTURAL_DIM,
    ),
    "Investor structural shape invalid.",
)

require(
    F_s_startup.shape
    == (
        1,
        STRUCTURAL_DIM,
    ),
    "Startup structural shape invalid.",
)


# =============================================================================
# 25. EXACT SCORING REPRESENTATIONS
# =============================================================================

banner(
    "COMPLETE SCORING REPRESENTATION"
)


investor_representation = torch.cat(
    [
        F_t,
        L_o,
        F_d_investor,
        F_s_investor,
    ],
    dim=1,
)


startup_representation = torch.cat(
    [
        L_b,
        F_d_startup,
        F_s_startup,
    ],
    dim=1,
)


pair_representation = torch.cat(
    [
        investor_representation,
        startup_representation,
    ],
    dim=1,
)


print(
    f"Investor R_o: "
    f"{tuple(investor_representation.shape)}"
)

print(
    f"Startup R_b:  "
    f"{tuple(startup_representation.shape)}"
)

print(
    f"Pair input:   "
    f"{tuple(pair_representation.shape)}"
)


require(
    investor_representation.shape
    == (
        1,
        INVESTOR_SCORING_DIM,
    ),
    "Investor scoring representation invalid.",
)

require(
    startup_representation.shape
    == (
        1,
        STARTUP_SCORING_DIM,
    ),
    "Startup scoring representation invalid.",
)

require(
    pair_representation.shape
    == (
        1,
        PAIR_DIM,
    ),
    "Pair representation invalid.",
)


# =============================================================================
# 26. SCORING FORWARD
# =============================================================================

banner(
    "SCORING MLP FORWARD"
)


scoring = (
    model
    .scoring_mlp(
        pair_representation
    )
)


logit = scoring[
    "logit"
]


probability = torch.sigmoid(
    logit
)


print(
    f"Hidden 1:    "
    f"{tuple(scoring['hidden_1'].shape)}"
)

print(
    f"Hidden 2:    "
    f"{tuple(scoring['hidden_2'].shape)}"
)

print(
    f"Hidden 3:    "
    f"{tuple(scoring['hidden_3'].shape)}"
)

print(
    f"Hidden 4:    "
    f"{tuple(scoring['hidden_4'].shape)}"
)

print(
    f"Logit:       "
    f"{tuple(logit.shape)}"
)

print(
    f"Probability: "
    f"{tuple(probability.shape)}"
)


print()
print(
    f"Audit logit:       "
    f"{float(logit.detach()):.10f}"
)

print(
    f"Audit probability: "
    f"{float(probability.detach()):.10f}"
)


require(
    logit.shape
    == (
        1,
        1,
    ),
    "Scoring logit shape invalid.",
)

require(
    bool(
        torch.isfinite(
            logit
        ).all()
    ),
    "Scoring logit non-finite.",
)


# =============================================================================
# 27. REAL POSITIVE BCE
#
# This is an observed investment event:
#
#   y = 1
#
# No negative candidate is created.
# =============================================================================

banner(
    "REAL VALIDATION POSITIVE BCE"
)


target = torch.ones(
    (
        1,
        1,
    ),
    dtype=logit.dtype,
)


criterion = nn.BCEWithLogitsLoss()


loss = criterion(
    logit,
    target,
)


manual_stable_bce = (
    F.softplus(
        logit
    )
    -
    target
    * logit
).mean()


bce_equivalent = torch.allclose(
    loss,
    manual_stable_bce,
    atol=ATOL,
    rtol=RTOL,
)


print(
    f"Target:               "
    f"{float(target.item()):.1f}"
)

print(
    f"BCEWithLogitsLoss:    "
    f"{float(loss.detach()):.10f}"
)

print(
    f"Manual stable BCE:    "
    f"{float(manual_stable_bce.detach()):.10f}"
)

print(
    f"Exact within tolerance:"
    f" {bce_equivalent}"
)


require(
    bool(
        torch.isfinite(
            loss
        )
    ),
    "End-to-end BCE loss is non-finite.",
)

require(
    bce_equivalent,
    "End-to-end BCE implementation mismatch.",
)


# =============================================================================
# 28. BACKWARD PASS
# =============================================================================

banner(
    "COMPLETE END-TO-END BACKWARD PASS"
)


model.zero_grad(
    set_to_none=True
)


backward_start = time.perf_counter()


loss.backward()


backward_seconds = (
    time.perf_counter()
    - backward_start
)


print(
    f"Backward runtime: "
    f"{backward_seconds:.3f} s"
)


# =============================================================================
# 29. GLOBAL PARAMETER GRADIENT AUDIT
# =============================================================================

banner(
    "GLOBAL PARAMETER GRADIENT AUDIT"
)


gradient_records = []


for (
    parameter_name,
    parameter,
) in model.named_parameters():

    record = gradient_stats(
        parameter_name,
        parameter,
    )

    gradient_records.append(
        record
    )

    print(
        f"{parameter_name:<58} "
        f"exists={str(record['gradient_exists']):<5} "
        f"finite={str(record['gradient_finite']):<5} "
        f"nonzero={str(record['gradient_nonzero']):<5} "
        f"abs_sum={record['gradient_abs_sum']:.10e}"
    )


# =============================================================================
# 30. REQUIRED MODULE-LEVEL GRADIENT PATHS
# =============================================================================

banner(
    "REQUIRED END-TO-END GRADIENT PATHS"
)


gradient_df = pd.DataFrame(
    gradient_records
)


def require_parameter_nonzero(
    parameter_name,
):

    rows = gradient_df[
        gradient_df[
            "parameter"
        ]
        == parameter_name
    ]

    require(
        len(
            rows
        )
        == 1,
        (
            "Parameter missing from gradient "
            f"audit: {parameter_name}"
        ),
    )

    row = rows.iloc[
        0
    ]

    require(
        bool(
            row[
                "gradient_exists"
            ]
        ),
        f"Gradient missing: {parameter_name}",
    )

    require(
        bool(
            row[
                "gradient_finite"
            ]
        ),
        f"Gradient non-finite: {parameter_name}",
    )

    require(
        bool(
            row[
                "gradient_nonzero"
            ]
        ),
        f"Gradient is zero: {parameter_name}",
    )


# -----------------------------------------------------------------------------
# Description
# -----------------------------------------------------------------------------

for parameter_name in [

    "description_encoder.text_projection.weight",
    "description_encoder.text_projection.bias",
    "description_encoder.label_projection.weight",
    "description_encoder.label_projection.bias",
]:

    require_parameter_nonzero(
        parameter_name
    )


print(
    "Description branch gradients: PASS"
)


# -----------------------------------------------------------------------------
# Trend
# -----------------------------------------------------------------------------

for parameter_name in [

    "trend_extractor.attention_weight",

    "trend_extractor.gru.weight_ih_l0",
    "trend_extractor.gru.weight_hh_l0",
    "trend_extractor.gru.bias_ih_l0",
    "trend_extractor.gru.bias_hh_l0",

    "trend_extractor.gru.weight_ih_l1",
    "trend_extractor.gru.weight_hh_l1",
    "trend_extractor.gru.bias_ih_l1",
    "trend_extractor.gru.bias_hh_l1",

    "trend_extractor.output_projection.weight",
]:

    require_parameter_nonzero(
        parameter_name
    )


print(
    "Trend branch gradients:       PASS"
)


# -----------------------------------------------------------------------------
# R-GCN
# -----------------------------------------------------------------------------

for parameter_name in [

    "preference_propagation.layer_1.bases",
    "preference_propagation.layer_1.coefficients",
    "preference_propagation.layer_1.root_weight",

    "preference_propagation.layer_2.bases",
    "preference_propagation.layer_2.coefficients",
    "preference_propagation.layer_2.root_weight",
]:

    require_parameter_nonzero(
        parameter_name
    )


print(
    "R-GCN branch gradients:        PASS"
)


# -----------------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------------

for parameter_name in [

    "scoring_mlp.hidden_1.weight",
    "scoring_mlp.hidden_1.bias",

    "scoring_mlp.hidden_2.weight",
    "scoring_mlp.hidden_2.bias",

    "scoring_mlp.hidden_3.weight",
    "scoring_mlp.hidden_3.bias",

    "scoring_mlp.hidden_4.weight",
    "scoring_mlp.hidden_4.bias",

    "scoring_mlp.output.weight",
    "scoring_mlp.output.bias",
]:

    require_parameter_nonzero(
        parameter_name
    )


print(
    "Scoring branch gradients:      PASS"
)


# =============================================================================
# 31. SHARED LATENT ROW GRADIENT AUDIT
# =============================================================================

banner(
    "SHARED LATENT ROW GRADIENT AUDIT"
)


investor_embedding_gradient = (
    model
    .investor_embedding
    .weight
    .grad
)


startup_embedding_gradient = (
    model
    .startup_embedding
    .weight
    .grad
)


require(
    investor_embedding_gradient
    is not None,
    "Investor embedding gradient missing.",
)

require(
    startup_embedding_gradient
    is not None,
    "Startup embedding gradient missing.",
)


selected_investor_gradient_abs_sum = float(
    investor_embedding_gradient[
        investor_global
    ]
    .abs()
    .sum()
)


selected_candidate_gradient_abs_sum = float(
    startup_embedding_gradient[
        startup_local
    ]
    .abs()
    .sum()
)


history_unique_locals = (
    history_unique
    - NUM_INVESTORS
)


history_gradient_abs_sums = (
    startup_embedding_gradient[
        torch.from_numpy(
            history_unique_locals
            .astype(
                np.int64,
                copy=False,
            )
        )
    ]
    .abs()
    .sum(
        dim=1
    )
)


max_history_gradient_value = float(
    history_gradient_abs_sums
    .max()
)


max_history_gradient_position = int(
    torch.argmax(
        history_gradient_abs_sums
    )
)


max_history_startup_global = int(
    history_unique[
        max_history_gradient_position
    ]
)


print(
    f"Selected Investor latent row gradient:"
)

print(
    f"  {selected_investor_gradient_abs_sum:.10e}"
)

print()
print(
    f"Candidate Startup latent row gradient:"
)

print(
    f"  {selected_candidate_gradient_abs_sum:.10e}"
)

print()
print(
    "Historical Startup with largest "
    "latent-row gradient:"
)

print(
    f"  global node: "
    f"{max_history_startup_global}"
)

print(
    f"  abs_sum:     "
    f"{max_history_gradient_value:.10e}"
)


require(
    selected_investor_gradient_abs_sum
    > 0.0,
    "Selected Investor latent row gradient is zero.",
)

require(
    selected_candidate_gradient_abs_sum
    > 0.0,
    "Candidate Startup latent row gradient is zero.",
)

require(
    max_history_gradient_value
    > 0.0,
    "Historical Startup latent gradients are all zero.",
)


print()
print(
    "Shared latent direct/scoring path: PASS"
)

print(
    "Shared latent trend path:          PASS"
)

print(
    "Shared latent structural path:     PASS"
)


# =============================================================================
# 32. ATTENTION-SPECIFIC GRADIENT CHECK
# =============================================================================

banner(
    "ATTENTION-SPECIFIC END-TO-END CHECK"
)


attention_gradient_abs_sum = float(
    model
    .trend_extractor
    .attention_weight
    .grad
    .abs()
    .sum()
)


print(
    f"Multi-item historical periods: "
    f"{len(multi_item_periods)}"
)

print(
    f"Attention W gradient abs_sum:  "
    f"{attention_gradient_abs_sum:.10e}"
)


require(
    attention_gradient_abs_sum
    > 0.0,
    (
        "Trend attention matrix received "
        "zero gradient despite multi-item history."
    ),
)


# =============================================================================
# 33. NO OPTIMIZER / NO TRAINING
# =============================================================================

banner(
    "TRAINING BOUNDARY"
)


print(
    "Optimizer created:             NO"
)

print(
    "optimizer.step() called:       NO"
)

print(
    "Training epoch executed:       NO"
)

print(
    "Negative candidate generated: NO"
)

print(
    "Test event consumed:           NO"
)

print(
    "Model state saved:             NO"
)


# =============================================================================
# 34. FORWARD SHAPE AUDIT TABLE
# =============================================================================

shape_records = [

    {
        "feature":
            "Doc2Vec subset",

        "shape":
            str(
                tuple(
                    doc2vec_subset.shape
                )
            ),

        "expected":
            (
                f"({len(required_global_nodes)}, "
                f"{DOC2VEC_DIM})"
            ),

        "status":
            "PASS",
    },

    {
        "feature":
            "Label subset",

        "shape":
            str(
                tuple(
                    label_subset.shape
                )
            ),

        "expected":
            (
                f"({len(required_global_nodes)}, "
                f"{LABEL_DIM})"
            ),

        "status":
            "PASS",
    },

    {
        "feature":
            "F_d subset",

        "shape":
            str(
                tuple(
                    description_subset.shape
                )
            ),

        "expected":
            (
                f"({len(required_global_nodes)}, "
                f"{DESCRIPTION_DIM})"
            ),

        "status":
            "PASS",
    },

    {
        "feature":
            "latent_all",

        "shape":
            str(
                tuple(
                    latent_all.shape
                )
            ),

        "expected":
            "(477564, 40)",

        "status":
            "PASS",
    },

    {
        "feature":
            "F_s_all",

        "shape":
            str(
                tuple(
                    F_s_all.shape
                )
            ),

        "expected":
            "(477564, 40)",

        "status":
            "PASS",
    },

    {
        "feature":
            "trend query",

        "shape":
            str(
                tuple(
                    trend_query.shape
                )
            ),

        "expected":
            "(80,)",

        "status":
            "PASS",
    },

    {
        "feature":
            "trend sequence",

        "shape":
            str(
                tuple(
                    trend_sequence.shape
                )
            ),

        "expected":
            "(1, 60, 80)",

        "status":
            "PASS",
    },

    {
        "feature":
            "F_t",

        "shape":
            str(
                tuple(
                    F_t.shape
                )
            ),

        "expected":
            "(1, 40)",

        "status":
            "PASS",
    },

    {
        "feature":
            "Investor scoring representation",

        "shape":
            str(
                tuple(
                    investor_representation.shape
                )
            ),

        "expected":
            "(1, 160)",

        "status":
            "PASS",
    },

    {
        "feature":
            "Startup scoring representation",

        "shape":
            str(
                tuple(
                    startup_representation.shape
                )
            ),

        "expected":
            "(1, 120)",

        "status":
            "PASS",
    },

    {
        "feature":
            "pair representation",

        "shape":
            str(
                tuple(
                    pair_representation.shape
                )
            ),

        "expected":
            "(1, 280)",

        "status":
            "PASS",
    },

    {
        "feature":
            "logit",

        "shape":
            str(
                tuple(
                    logit.shape
                )
            ),

        "expected":
            "(1, 1)",

        "status":
            "PASS",
    },
]


shape_df = pd.DataFrame(
    shape_records
)


shape_path = (
    OUT_DIR
    / "phase_4_6_2_forward_shape_audit.csv"
)


shape_df.to_csv(
    shape_path,
    index=False,
)


# =============================================================================
# 35. SAVE ATTENTION DIAGNOSTICS
# =============================================================================

attention_df = pd.DataFrame(
    attention_diagnostics
)


attention_path = (
    OUT_DIR
    / "phase_4_6_2_attention_audit.csv"
)


attention_df.to_csv(
    attention_path,
    index=False,
)


# =============================================================================
# 36. SAVE GRADIENT AUDIT
# =============================================================================

gradient_path = (
    OUT_DIR
    / "phase_4_6_2_gradient_audit.csv"
)


gradient_df.to_csv(
    gradient_path,
    index=False,
)


# =============================================================================
# 37. SAVE SELECTED REAL VALIDATION EVENT
# =============================================================================

selected_pair_record = {

    "phase":
        "4.6.2",

    "selection_purpose":
        "end_to_end_integration_audit_only",

    "split":
        "validation",

    "segment":
        "T60",

    "target":
        1,

    "interaction_id":
        selected_interaction_id,

    "funding_round_id":
        selected_funding_round_id,

    "investor_id":
        selected_investor_id,

    "startup_id":
        selected_startup_id,

    "investor_global_node_index":
        investor_global,

    "startup_global_node_index":
        startup_global,

    "startup_local_index":
        startup_local,

    "history_memberships":
        history_total,

    "active_history_periods":
        active_periods,

    "multi_item_history_periods":
        len(
            multi_item_periods
        ),

    "max_period_items":
        max(
            history_period_counts_python
        ),

    "investor_incoming_degree":
        int(
            selected[
                "investor_incoming_degree"
            ]
        ),

    "startup_incoming_degree":
        int(
            selected[
                "startup_incoming_degree"
            ]
        ),

    "startup_label_count":
        int(
            selected[
                "startup_label_count"
            ]
        ),

    "candidate_negative_generated":
        False,

    "test_data_used":
        False,

    "selection_changes_model_policy":
        False,
}


selected_pair_path = (
    OUT_DIR
    / "phase_4_6_2_selected_validation_pair.json"
)


with open(
    selected_pair_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        selected_pair_record,
        f,
        indent=2,
        ensure_ascii=False,
    )


# =============================================================================
# 38. SAVE END-TO-END CONTRACT
# =============================================================================

banner(
    "FREEZING PHASE 4.6.2 INTEGRATION CONTRACT"
)


integration_contract = {

    "phase":
        "4.6.2",

    "status":
        "FROZEN",

    "component":
        (
            "Complete numerical ITRS "
            "forward + BCE + backward audit"
        ),

    "audit_case":
        {

            "split":
                "validation",

            "segment":
                "T60",

            "positive_observation":
                True,

            "negative_generated":
                False,

            "test_data_used":
                False,
        },

    "forward":
        {

            "real_doc2vec":
                True,

            "real_labels":
                True,

            "real_trend_history":
                True,

            "full_phase_3_graph":
                True,

            "shared_latent_tables":
                True,

            "description_computed":
                True,

            "trend_computed":
                True,

            "structural_computed":
                True,

            "pair_dimension":
                280,

            "logit_shape":
                [
                    1,
                    1,
                ],

            "probability":
                "sigmoid(logit)",
        },

    "trend":
        {

            "history":
                "T0 through T59",

            "target":
                "T60",

            "candidate_affects_history":
                False,

            "empty_period_vector":
                "zero80",

            "attention_exercised":
                True,

            "multi_item_periods":
                len(
                    multi_item_periods
                ),
        },

    "loss":
        {

            "target":
                1,

            "implementation":
                "BCEWithLogitsLoss",

            "manual_stable_equivalence":
                True,

            "optimizer_used":
                False,
        },

    "autograd":
        {

            "description_branch":
                True,

            "trend_attention":
                True,

            "trend_gru":
                True,

            "trend_output_projection":
                True,

            "rgcn_layer_1":
                True,

            "rgcn_layer_2":
                True,

            "scoring_mlp":
                True,

            "investor_latent_row":
                True,

            "candidate_startup_latent_row":
                True,

            "historical_startup_latent_row":
                True,
        },

    "initialization":
        {

            "audit_initialization":
                "DETERMINISTIC_NON_KAIMING_AUDIT_ONLY",

            "model_state_saved":
                False,

            "paper_family":
                "Kaiming",

            "exact_variant":
                "NOT_YET_FROZEN",
        },

    "training":
        {

            "performed":
                False,

            "optimizer_created":
                False,

            "optimizer_step":
                False,

            "negative_sampling":
                "NOT_YET_FROZEN",
        },

    "runtime_seconds":
        {

            "initialization":
                initialization_seconds,

            "structural_forward":
                structural_seconds,

            "backward":
                backward_seconds,
        },

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

            "phase_4_6_1a":
                False,

            "phase_4_6_1b":
                False,
        },
}


integration_contract_path = (
    OUT_DIR
    / "phase_4_6_2_end_to_end_contract.json"
)


with open(
    integration_contract_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        integration_contract,
        f,
        indent=2,
        ensure_ascii=False,
    )


# =============================================================================
# 39. PHASE 4.6 CLOSURE MANIFEST
# =============================================================================

phase_4_6_closure = {

    "phase":
        "4.6",

    "name":
        "Complete ITRS Forward and BCE Integration",

    "status":
        "COMPLETE",

    "subphases":
        {

            "4.6.1a":
                "COMPLETE_AUDIT_ONLY",

            "4.6.1b":
                "FROZEN",

            "4.6.2":
                "FROZEN",
        },

    "verified":
        {

            "authoritative_static_inputs":
                True,

            "full_model_topology":
                True,

            "full_parameter_budget":
                True,

            "real_validation_forward":
                True,

            "real_description_forward":
                True,

            "real_trend_forward":
                True,

            "full_graph_rgcn_forward":
                True,

            "280_dim_scoring_forward":
                True,

            "positive_bce":
                True,

            "complete_backward":
                True,

            "shared_latent_gradient_paths":
                True,
        },

    "still_open":
        [

            "exact global Kaiming initialization variant",

            "global neural seed policy",

            "training negative:positive ratio",

            "training negative candidate eligibility",

            "training historical negative exclusion",

            "training epoch count",

            "early stopping",

            "weight decay",

            "evaluation candidate-generation runtime contract",
        ],

    "training_performed":
        False,

    "next_phase":
        {

            "phase":
                "4.7",

            "name":
                "Complete Model Integrity Audit",
        },
}


phase_4_6_closure_path = (
    OUT_DIR
    / "phase_4_6_closure_manifest.json"
)


with open(
    phase_4_6_closure_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        phase_4_6_closure,
        f,
        indent=2,
        ensure_ascii=False,
    )


# =============================================================================
# 40. ARTIFACT HASHES
# =============================================================================

artifact_paths = [

    selected_pair_path,
    shape_path,
    attention_path,
    gradient_path,
    integration_contract_path,
    phase_4_6_closure_path,
]


artifact_hash_records = []


for path in artifact_paths:

    artifact_hash_records.append(
        {

            "artifact":
                path.stem,

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


artifact_hash_df = pd.DataFrame(
    artifact_hash_records
)


artifact_hash_path = (
    OUT_DIR
    / "phase_4_6_artifact_hashes.csv"
)


artifact_hash_df.to_csv(
    artifact_hash_path,
    index=False,
)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

banner(
    "PHASE 4.6.2 FINAL SUMMARY"
)


print(
    "Audit case:"
)

print(
    "  source                         "
    "real Phase-2 validation positive"
)

print(
    "  target segment                 T60"
)

print(
    "  target                         y = 1"
)

print(
    "  test data used                 NO"
)

print(
    "  negative generated             NO"
)


print()
print(
    "Real static inputs:"
)

print(
    "  Doc2Vec                       PASS"
)

print(
    "  labels                        PASS"
)

print(
    "  T0..T59 history               PASS"
)

print(
    "  Phase-3 full graph            PASS"
)


print()
print(
    "Complete forward:"
)

print(
    "  F_d                            PASS"
)

print(
    "  L_o / L_b                     PASS"
)

print(
    "  F_t                            PASS"
)

print(
    "  F_s                            PASS"
)

print(
    "  R_o [1,160]                   PASS"
)

print(
    "  R_b [1,120]                   PASS"
)

print(
    "  pair [1,280]                  PASS"
)

print(
    "  scorer                         PASS"
)

print(
    "  raw logit                      PASS"
)

print(
    "  sigmoid probability            PASS"
)

print(
    "  BCEWithLogitsLoss              PASS"
)


print()
print(
    "Autograd:"
)

print(
    "  Description encoder            PASS"
)

print(
    "  Trend attention                PASS"
)

print(
    "  Trend GRU                      PASS"
)

print(
    "  Trend output projection        PASS"
)

print(
    "  R-GCN layer 1                  PASS"
)

print(
    "  R-GCN layer 2                  PASS"
)

print(
    "  Scoring MLP                    PASS"
)

print(
    "  Investor latent row            PASS"
)

print(
    "  candidate Startup latent row   PASS"
)

print(
    "  historical Startup latent row  PASS"
)


print()
print(
    "Training boundary:"
)

print(
    "  optimizer                      NONE"
)

print(
    "  optimizer.step                 NO"
)

print(
    "  epoch                          NO"
)

print(
    "  state persisted                NO"
)


print()
print(
    "Still open:"
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
    "  epochs / early stopping"
)

print(
    "  weight decay"
)

print(
    "  evaluation candidate generation"
)


print()
print(
    "Outputs:"
)


for path in [

    selected_pair_path,
    shape_path,
    attention_path,
    gradient_path,
    integration_contract_path,
    phase_4_6_closure_path,
    artifact_hash_path,
]:

    print(
        f"  {path}"
    )


print()
print(
    "PHASE 4.6.2 STATUS: COMPLETE — "
    "END-TO-END ITRS FORWARD / BCE / "
    "BACKWARD VERIFIED"
)


print()
print("=" * 120)

print(
    "PHASE 4.6 STATUS: COMPLETE — "
    "FULL ITRS FORWARD + BCE INTEGRATION CLOSED"
)

print("=" * 120)


print()
print(
    "NEXT:"
)

print(
    "PHASE 4.7 — "
    "COMPLETE MODEL INTEGRITY AUDIT"
)