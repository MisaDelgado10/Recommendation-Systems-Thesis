from pathlib import Path
import gc
import json
import sys
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# PHASE 4.4.2 — R-GCN FORWARD IMPLEMENTATION AUDIT
#
# PURPOSE
# -------
# Verify that the explicit PyTorch implementation of the frozen ITRS
# preference-propagation contract performs:
#
#   1. basis-decomposed typed-relation transformations,
#   2. source -> destination message passing,
#   3. relation-specific destination-side mean normalization,
#   4. summation across typed relations,
#   5. separate root/self transformation,
#   6. ReLU after both R-GCN layers,
#   7. correct isolate handling,
#   8. correct two-layer propagation,
#   9. correct Investor / Startup structural-feature slicing.
#
# IMPORTANT
# ---------
# NO TRAINING OCCURS.
#
# Synthetic deterministic 40-D latent node vectors are used ONLY to audit
# the forward contract because the final trainable latent embedding tables
# have not yet been initialized/trained.
#
# Audit-only deterministic R-GCN weights are also used.
#
# Neither synthetic node features nor audit model parameters are persisted.
#
# Final Kaiming initialization remains NOT YET FROZEN.
# =============================================================================


# =============================================================================
# INPUTS
# =============================================================================

NEURAL_CONTRACT_PATH = Path(
    "data/experimental/phase_4/"
    "rgcn_neural_contract/"
    "rgcn_neural_contract.json"
)

INPUT_AUDIT_METADATA_PATH = Path(
    "data/experimental/phase_4/"
    "rgcn_input_audit/"
    "rgcn_structural_input_audit_metadata.json"
)

RELATION_DEGREE_AUDIT_PATH = Path(
    "data/experimental/phase_4/"
    "rgcn_input_audit/"
    "rgcn_relation_incoming_degree_audit.csv"
)

RELATION_CONTRACT_AUDIT_PATH = Path(
    "data/experimental/phase_4/"
    "rgcn_input_audit/"
    "rgcn_typed_relation_contract_audit.csv"
)

NODE_INDEX_PATH = Path(
    "data/experimental/phase_3/"
    "model_ready/"
    "node_index.parquet"
)

EDGE_INDEX_PATH = Path(
    "data/experimental/phase_3/"
    "model_ready/"
    "edge_index.npy"
)

EDGE_TYPE_PATH = Path(
    "data/experimental/phase_3/"
    "model_ready/"
    "edge_type.npy"
)


# =============================================================================
# OUTPUTS
# =============================================================================

OUT_DIR = Path(
    "data/experimental/phase_4/"
    "rgcn_module"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# FROZEN GRAPH CONSTANTS
# =============================================================================

NUM_INVESTORS = 165_975
NUM_STARTUPS = 311_589

NUM_NODES = 477_564
NUM_EDGES = 158_818

NUM_RELATIONS = 12

EXPECTED_CONNECTED_NODES = 74_757
EXPECTED_ISOLATES = 402_807


# =============================================================================
# FROZEN R-GCN CONSTANTS
# =============================================================================

INPUT_DIM = 40
HIDDEN_DIM = 40
OUTPUT_DIM = 40

NUM_LAYERS = 2
NUM_BASES = 5

EXPECTED_PARAMETERS = 19_320


# =============================================================================
# NUMERICAL AUDIT TOLERANCES
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


def max_abs_difference(
    a,
    b,
):

    return float(
        torch.max(
            torch.abs(
                a - b
            )
        )
    )


# =============================================================================
# BASIS-DECOMPOSED R-GCN LAYER
#
# Row-vector PyTorch convention:
#
#     message_j->i = x_j @ W_r
#
# with:
#
#     W_r = sum_b a[r,b] V_b
#
# For each destination i and relation r:
#
#     m_i,r =
#         mean(
#             x_j @ W_r
#             for j in N_i^r
#         )
#
# Final pre-activation:
#
#     z_i =
#         x_i @ W_root
#         + sum_r m_i,r
#
# No additive bias.
# No structural self-loop edge.
# =============================================================================

class BasisRGCNLayer(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        num_relations,
        num_bases,
    ):

        super().__init__()


        self.in_dim = int(
            in_dim
        )

        self.out_dim = int(
            out_dim
        )

        self.num_relations = int(
            num_relations
        )

        self.num_bases = int(
            num_bases
        )


        self.bases = nn.Parameter(
            torch.empty(
                self.num_bases,
                self.in_dim,
                self.out_dim,
            )
        )


        self.coefficients = nn.Parameter(
            torch.empty(
                self.num_relations,
                self.num_bases,
            )
        )


        self.root_weight = nn.Parameter(
            torch.empty(
                self.in_dim,
                self.out_dim,
            )
        )


    def effective_relation_weights(
        self,
    ):

        weights = (
            self.coefficients
            @ self.bases.reshape(
                self.num_bases,
                -1,
            )
        )


        return weights.reshape(
            self.num_relations,
            self.in_dim,
            self.out_dim,
        )


    def propagate(
        self,
        x,
        edge_index,
        edge_type,
        relation_positions,
    ):

        require(
            x.ndim == 2,
            "Node matrix must be rank 2.",
        )


        require(
            x.shape[1] == self.in_dim,
            "Node input dimension mismatch.",
        )


        require(
            edge_index.shape[0] == 2,
            "edge_index must have shape [2,E].",
        )


        require(
            edge_index.shape[1]
            == edge_type.shape[0],
            "edge_index / edge_type length mismatch.",
        )


        num_nodes = int(
            x.shape[0]
        )


        relation_weights = (
            self.effective_relation_weights()
        )


        # ---------------------------------------------------------------------
        # Separate root/self transformation:
        #
        #     x_i @ W_root
        # ---------------------------------------------------------------------

        out = (
            x
            @ self.root_weight
        )


        # ---------------------------------------------------------------------
        # Relation-specific message passing.
        #
        # We process one typed relation at a time so that normalization is:
        #
        #     1 / |N_i^r|
        #
        # rather than total node degree.
        # ---------------------------------------------------------------------

        for relation_id in range(
            self.num_relations
        ):

            positions = relation_positions[
                relation_id
            ]


            if positions.numel() == 0:

                continue


            src = edge_index[
                0,
                positions,
            ]


            dst = edge_index[
                1,
                positions,
            ]


            # Relation-specific incoming degree for each destination.
            relation_degree = torch.bincount(
                dst,
                minlength=num_nodes,
            ).to(
                dtype=x.dtype
            )


            # Every edge destination in this relation must have degree >= 1.
            edge_destination_degree = (
                relation_degree[
                    dst
                ]
            )


            require(
                bool(
                    torch.all(
                        edge_destination_degree
                        >= 1
                    )
                ),
                (
                    "Relation-specific destination "
                    "degree contains zero."
                ),
            )


            messages = (
                x[
                    src
                ]
                @ relation_weights[
                    relation_id
                ]
            )


            # -------------------------------------------------------------
            # Mean aggregation within relation:
            #
            # Divide EACH edge message by the relation-specific incoming
            # degree of its destination, then sum all normalized messages.
            # -------------------------------------------------------------

            normalized_messages = (
                messages
                /
                edge_destination_degree.unsqueeze(
                    1
                )
            )


            out.index_add_(
                0,
                dst,
                normalized_messages,
            )


        return out


# =============================================================================
# TWO-LAYER PREFERENCE-PROPAGATION MODULE
# =============================================================================

class ITRSPreferencePropagation(nn.Module):

    def __init__(self):

        super().__init__()


        self.layer_1 = BasisRGCNLayer(
            in_dim=INPUT_DIM,
            out_dim=HIDDEN_DIM,
            num_relations=NUM_RELATIONS,
            num_bases=NUM_BASES,
        )


        self.layer_2 = BasisRGCNLayer(
            in_dim=HIDDEN_DIM,
            out_dim=OUTPUT_DIM,
            num_relations=NUM_RELATIONS,
            num_bases=NUM_BASES,
        )


        self.activation = nn.ReLU()


# =============================================================================
# DETERMINISTIC AUDIT-ONLY INITIAL NODE FEATURES
#
# These are NOT final latent embeddings.
#
# They exist only so the frozen graph/message-passing contract can be
# exercised before model training.
# =============================================================================

def make_audit_node_features(
    num_nodes,
    dim,
):

    node_axis = torch.arange(
        num_nodes,
        dtype=torch.float32,
    ).unsqueeze(
        1
    )


    dim_axis = torch.arange(
        dim,
        dtype=torch.float32,
    ).unsqueeze(
        0
    )


    values = (
        0.35
        * torch.sin(
            node_axis * 0.013
            + dim_axis * 0.173
        )
        +
        0.15
        * torch.cos(
            node_axis * 0.007
            - dim_axis * 0.119
        )
    )


    require(
        tuple(
            values.shape
        )
        == (
            num_nodes,
            dim,
        ),
        "Audit node feature shape mismatch.",
    )


    return values


# =============================================================================
# DETERMINISTIC AUDIT-ONLY R-GCN INITIALIZATION
#
# IMPORTANT:
#
# This is NOT Kaiming.
# This is NOT persisted.
# It exists only to make the forward audit deterministic and nontrivial.
# =============================================================================

def initialize_audit_layer(
    layer,
    layer_number,
):

    eye = torch.eye(
        layer.in_dim,
        layer.out_dim,
        dtype=torch.float32,
    )


    with torch.no_grad():

        # ---------------------------------------------------------------------
        # Distinct basis matrices.
        #
        # Each basis contains a diagonal term plus a shifted term so the
        # transformation is not simply scalar multiplication.
        # ---------------------------------------------------------------------

        for basis_id in range(
            layer.num_bases
        ):

            diagonal_scale = (
                0.010
                * (basis_id + 1)
                * (
                    1.0
                    if layer_number == 1
                    else 0.85
                )
            )


            shifted_scale = (
                0.003
                * (basis_id + 1)
                * (
                    1.0
                    if layer_number == 1
                    else 0.75
                )
            )


            shifted = torch.roll(
                eye,
                shifts=basis_id + 1,
                dims=1,
            )


            layer.bases[
                basis_id
            ].copy_(
                diagonal_scale
                * eye
                +
                shifted_scale
                * shifted
            )


        # ---------------------------------------------------------------------
        # Deterministic relation-specific basis coefficients.
        # ---------------------------------------------------------------------

        for relation_id in range(
            layer.num_relations
        ):

            for basis_id in range(
                layer.num_bases
            ):

                value = (
                    (
                        relation_id + 1
                    )
                    *
                    (
                        basis_id + 1
                    )
                    /
                    (
                        layer.num_relations
                        * layer.num_bases
                        * 2.0
                    )
                )


                if layer_number == 2:

                    value *= 0.90


                layer.coefficients[
                    relation_id,
                    basis_id,
                ] = value


        # ---------------------------------------------------------------------
        # Root/self transform.
        # ---------------------------------------------------------------------

        root_diagonal = (
            0.22
            if layer_number == 1
            else 0.18
        )


        root_shift = (
            0.035
            if layer_number == 1
            else 0.025
        )


        shifted_root = torch.roll(
            eye,
            shifts=layer_number,
            dims=1,
        )


        layer.root_weight.copy_(
            root_diagonal
            * eye
            +
            root_shift
            * shifted_root
        )


# =============================================================================
# MANUAL SINGLE-NODE UPDATE
#
# Used to independently reconstruct an actual graph node's update from:
#
#   root transform
#   +
#   mean messages from every incoming typed relation.
# =============================================================================

def manual_node_preactivation(
    x,
    node_index,
    layer,
    edge_index,
    edge_type,
):

    relation_weights = (
        layer.effective_relation_weights()
    )


    z = (
        x[
            node_index
        ]
        @ layer.root_weight
    ).clone()


    sources = edge_index[
        0
    ]


    destinations = edge_index[
        1
    ]


    for relation_id in range(
        NUM_RELATIONS
    ):

        mask = (
            (
                destinations
                == node_index
            )
            &
            (
                edge_type
                == relation_id
            )
        )


        incoming_sources = (
            sources[
                mask
            ]
        )


        if incoming_sources.numel() == 0:

            continue


        relation_messages = (
            x[
                incoming_sources
            ]
            @ relation_weights[
                relation_id
            ]
        )


        relation_mean = (
            relation_messages.mean(
                dim=0
            )
        )


        z = (
            z
            + relation_mean
        )


    return z


# =============================================================================
# START
# =============================================================================

banner(
    "PHASE 4.4.2 — "
    "R-GCN FORWARD IMPLEMENTATION AUDIT"
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


torch.set_grad_enabled(
    False
)


# =============================================================================
# 2. CONTRACT INTEGRITY
# =============================================================================

banner(
    "FROZEN CONTRACT INTEGRITY"
)


with open(
    NEURAL_CONTRACT_PATH,
    "r",
    encoding="utf-8",
) as f:

    neural_contract = json.load(f)


with open(
    INPUT_AUDIT_METADATA_PATH,
    "r",
    encoding="utf-8",
) as f:

    input_audit = json.load(f)


require(
    neural_contract.get(
        "status"
    )
    == "FROZEN",
    "R-GCN neural contract is not frozen.",
)


require(
    input_audit.get(
        "status"
    )
    == "COMPLETE_AUDIT_ONLY",
    "R-GCN structural input audit is incomplete.",
)


require(
    neural_contract[
        "architecture"
    ][
        "layers"
    ]
    == NUM_LAYERS,
    "Frozen R-GCN layer count changed.",
)


require(
    neural_contract[
        "architecture"
    ][
        "num_relations"
    ]
    == NUM_RELATIONS,
    "Frozen relation count changed.",
)


require(
    neural_contract[
        "architecture"
    ][
        "num_bases"
    ]
    == NUM_BASES,
    "Frozen basis count changed.",
)


require(
    neural_contract[
        "extras"
    ][
        "additive_bias"
    ]
    is False,
    "Frozen additive-bias policy changed.",
)


require(
    neural_contract[
        "graph_input"
    ][
        "explicit_self_loop_edges"
    ]
    is False,
    "Frozen self-loop policy changed.",
)


print(
    "R-GCN neural contract:    PASS"
)

print(
    "Structural input audit:   PASS"
)


# =============================================================================
# 3. LOAD RELATION AUDITS
# =============================================================================

banner(
    "RELATION AUDIT INTEGRITY"
)


relation_degree_audit = pd.read_csv(
    RELATION_DEGREE_AUDIT_PATH
)


relation_contract_audit = pd.read_csv(
    RELATION_CONTRACT_AUDIT_PATH
)


require(
    len(
        relation_degree_audit
    )
    == NUM_RELATIONS,
    "Relation degree audit changed.",
)


require(
    len(
        relation_contract_audit
    )
    == NUM_RELATIONS,
    "Relation contract audit changed.",
)


require(
    relation_contract_audit[
        "relation_id"
    ]
    .astype(int)
    .tolist()
    == list(
        range(
            NUM_RELATIONS
        )
    ),
    "Relation ID audit changed.",
)


print(
    "Relation degree audit:    PASS"
)

print(
    "Relation semantic audit:  PASS"
)


# =============================================================================
# 4. LOAD FROZEN GRAPH
# =============================================================================

banner(
    "LOADING FROZEN PHASE-3 GRAPH"
)


edge_index_np = np.load(
    EDGE_INDEX_PATH,
    mmap_mode="r",
)


edge_type_np = np.load(
    EDGE_TYPE_PATH,
    mmap_mode="r",
)


nodes = pd.read_parquet(
    NODE_INDEX_PATH,
    columns=[
        "node_index",
        "node_type",
    ],
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
        nodes
    )
    == NUM_NODES,
    "Node population changed.",
)


# Copy into writable CPU tensors.
edge_index = torch.tensor(
    np.asarray(
        edge_index_np
    ),
    dtype=torch.long,
)


edge_type = torch.tensor(
    np.asarray(
        edge_type_np
    ),
    dtype=torch.long,
)


print(
    f"Nodes:      "
    f"{NUM_NODES:,}"
)

print(
    f"Edges:      "
    f"{NUM_EDGES:,}"
)

print(
    f"Relations:  "
    f"{NUM_RELATIONS}"
)

print(
    f"edge_index: "
    f"{tuple(edge_index.shape)}"
)

print(
    f"edge_type:  "
    f"{tuple(edge_type.shape)}"
)


# =============================================================================
# 5. PRECOMPUTE RELATION EDGE POSITIONS
# =============================================================================

banner(
    "RELATION EDGE POSITION INDEX"
)


relation_positions = []


for relation_id in range(
    NUM_RELATIONS
):

    positions = torch.nonzero(
        edge_type
        == relation_id,
        as_tuple=False,
    ).flatten()


    relation_positions.append(
        positions
    )


    expected_count = int(
        relation_contract_audit.loc[
            relation_contract_audit[
                "relation_id"
            ]
            .eq(
                relation_id
            ),
            "actual_edge_count_from_edge_type",
        ].iloc[0]
    )


    print(
        f"Relation {relation_id:>2}: "
        f"{positions.numel():>8,} edges"
    )


    require(
        positions.numel()
        == expected_count,
        (
            "Relation edge position count "
            f"changed for relation {relation_id}."
        ),
    )


# =============================================================================
# 6. CONNECTED / ISOLATE POPULATION
# =============================================================================

banner(
    "CONNECTED / ISOLATE POPULATION"
)


sources = edge_index[
    0
]


destinations = edge_index[
    1
]


connected_mask = torch.zeros(
    NUM_NODES,
    dtype=torch.bool,
)


connected_mask[
    sources
] = True


connected_mask[
    destinations
] = True


connected_count = int(
    connected_mask.sum()
)


isolate_indices = torch.nonzero(
    ~connected_mask,
    as_tuple=False,
).flatten()


isolate_count = int(
    isolate_indices.numel()
)


print(
    f"Connected nodes: "
    f"{connected_count:,}"
)

print(
    f"Isolates:        "
    f"{isolate_count:,}"
)


require(
    connected_count
    == EXPECTED_CONNECTED_NODES,
    "Connected-node count changed.",
)


require(
    isolate_count
    == EXPECTED_ISOLATES,
    "Isolate count changed.",
)


representative_isolate = int(
    isolate_indices[
        0
    ]
)


print(
    f"Representative isolate node: "
    f"{representative_isolate}"
)


# =============================================================================
# 7. INITIALIZE MODULE
# =============================================================================

banner(
    "R-GCN MODULE INSTANTIATION"
)


model = ITRSPreferencePropagation()


initialize_audit_layer(
    model.layer_1,
    layer_number=1,
)


initialize_audit_layer(
    model.layer_2,
    layer_number=2,
)


model.eval()


parameter_count = sum(
    parameter.numel()

    for parameter
    in model.parameters()
)


print(
    model
)

print()
print(
    f"Trainable parameters: "
    f"{parameter_count:,}"
)


require(
    parameter_count
    == EXPECTED_PARAMETERS,
    "R-GCN parameter count changed.",
)


print()
print(
    "Audit-only deterministic weights: INSTALLED"
)

print(
    "Final Kaiming initialization:     UNCHANGED / UNFROZEN"
)

print(
    "Audit model state will be saved:  NO"
)


# =============================================================================
# 8. BASIS RECONSTRUCTION AUDIT
# =============================================================================

banner(
    "BASIS RECONSTRUCTION AUDIT"
)


for layer_name, layer in [
    (
        "layer_1",
        model.layer_1,
    ),
    (
        "layer_2",
        model.layer_2,
    ),
]:

    effective = (
        layer.effective_relation_weights()
    )


    require(
        tuple(
            effective.shape
        )
        == (
            NUM_RELATIONS,
            40,
            40,
        ),
        (
            f"{layer_name} effective relation "
            "weight shape changed."
        ),
    )


    max_diff = 0.0


    for relation_id in range(
        NUM_RELATIONS
    ):

        manual = torch.zeros(
            40,
            40,
            dtype=torch.float32,
        )


        for basis_id in range(
            NUM_BASES
        ):

            manual += (
                layer.coefficients[
                    relation_id,
                    basis_id,
                ]
                *
                layer.bases[
                    basis_id
                ]
            )


        diff = max_abs_difference(
            manual,
            effective[
                relation_id
            ],
        )


        max_diff = max(
            max_diff,
            diff,
        )


        require(
            torch.allclose(
                manual,
                effective[
                    relation_id
                ],
                atol=ATOL,
                rtol=RTOL,
            ),
            (
                f"{layer_name} basis decomposition "
                f"failed for relation {relation_id}."
            ),
        )


    print(
        f"{layer_name}:"
    )

    print(
        f"  effective shape: "
        f"{tuple(effective.shape)}"
    )

    print(
        f"  max manual reconstruction diff: "
        f"{max_diff:.10f}"
    )

    print(
        "  basis reconstruction: PASS"
    )


# =============================================================================
# 9. SYNTHETIC DIRECTIONALITY + MEAN-NORMALIZATION MICRO-AUDIT
#
# Tiny graph:
#
#   node 0 ----\
#               >---- node 2
#   node 1 ----/
#
# Both edges use relation 0.
#
# Expected:
#
#   neighbor delta node 0 = 0
#   neighbor delta node 1 = 0
#
#   neighbor delta node 2 =
#       mean(
#           x0 @ W_0,
#           x1 @ W_0
#       )
#
# This independently verifies:
#
#   source -> destination
#   mean within relation
# =============================================================================

banner(
    "SYNTHETIC DIRECTIONALITY / NORMALIZATION MICRO-AUDIT"
)


micro_x = torch.zeros(
    3,
    40,
    dtype=torch.float32,
)


micro_x[
    0,
    0
] = 1.0


micro_x[
    1,
    1
] = 2.0


micro_x[
    2,
    2
] = 3.0


micro_edge_index = torch.tensor(
    [
        [
            0,
            1,
        ],
        [
            2,
            2,
        ],
    ],
    dtype=torch.long,
)


micro_edge_type = torch.tensor(
    [
        0,
        0,
    ],
    dtype=torch.long,
)


micro_relation_positions = []


for relation_id in range(
    NUM_RELATIONS
):

    micro_relation_positions.append(
        torch.nonzero(
            micro_edge_type
            == relation_id,
            as_tuple=False,
        ).flatten()
    )


micro_preactivation = (
    model.layer_1.propagate(
        micro_x,
        micro_edge_index,
        micro_edge_type,
        micro_relation_positions,
    )
)


micro_root_only = (
    micro_x
    @ model.layer_1.root_weight
)


micro_neighbor_delta = (
    micro_preactivation
    - micro_root_only
)


effective_layer_1 = (
    model.layer_1
    .effective_relation_weights()
)


expected_destination_delta = (
    torch.stack(
        [
            micro_x[
                0
            ]
            @ effective_layer_1[
                0
            ],

            micro_x[
                1
            ]
            @ effective_layer_1[
                0
            ],
        ],
        dim=0,
    )
    .mean(
        dim=0
    )
)


source_0_delta_zero = torch.allclose(
    micro_neighbor_delta[
        0
    ],
    torch.zeros(
        40
    ),
    atol=ATOL,
    rtol=RTOL,
)


source_1_delta_zero = torch.allclose(
    micro_neighbor_delta[
        1
    ],
    torch.zeros(
        40
    ),
    atol=ATOL,
    rtol=RTOL,
)


destination_delta_match = torch.allclose(
    micro_neighbor_delta[
        2
    ],
    expected_destination_delta,
    atol=ATOL,
    rtol=RTOL,
)


print(
    f"Source node 0 neighbor delta zero: "
    f"{source_0_delta_zero}"
)

print(
    f"Source node 1 neighbor delta zero: "
    f"{source_1_delta_zero}"
)

print(
    f"Destination mean message exact:    "
    f"{destination_delta_match}"
)


require(
    source_0_delta_zero
    and source_1_delta_zero,
    (
        "Synthetic graph indicates reverse or "
        "incorrect message direction."
    ),
)


require(
    destination_delta_match,
    (
        "Synthetic relation mean normalization "
        "does not match contract."
    ),
)


# =============================================================================
# 10. CREATE AUDIT-ONLY FULL NODE INPUT MATRIX
# =============================================================================

banner(
    "AUDIT-ONLY INITIAL NODE REPRESENTATIONS"
)


feature_start = time.perf_counter()


x0 = make_audit_node_features(
    NUM_NODES,
    INPUT_DIM,
)


feature_seconds = (
    time.perf_counter()
    - feature_start
)


print(
    f"x0 shape: "
    f"{tuple(x0.shape)}"
)

print(
    f"x0 dtype: "
    f"{x0.dtype}"
)

print(
    f"x0 finite: "
    f"{bool(torch.isfinite(x0).all())}"
)

print(
    f"Generation time: "
    f"{feature_seconds:.3f} s"
)


require(
    tuple(
        x0.shape
    )
    == (
        NUM_NODES,
        40,
    ),
    "Initial node representation shape mismatch.",
)


require(
    bool(
        torch.isfinite(
            x0
        ).all()
    ),
    "Initial audit node features are non-finite.",
)


# =============================================================================
# 11. SELECT REAL HIGH-DEGREE RELATION CASE
#
# Use the relation with the largest audited relation-specific incoming degree.
#
# The current frozen Phase-4.4.1a audit should select relation 6 first,
# because relation 6 and 7 both reach max degree 123 and relation ID 6 is
# used as deterministic tie-break.
# =============================================================================

banner(
    "REAL HIGH-DEGREE RELATION CASE"
)


high_degree_row = (
    relation_degree_audit
    .sort_values(
        [
            "max_in_degree",
            "relation_id",
        ],
        ascending=[
            False,
            True,
        ],
    )
    .iloc[0]
)


high_degree_relation = int(
    high_degree_row[
        "relation_id"
    ]
)


audited_max_degree = int(
    high_degree_row[
        "max_in_degree"
    ]
)


print(
    f"Selected relation: "
    f"{high_degree_relation}"
)

print(
    f"Typed relation:    "
    f"{high_degree_row['typed_relation_key']}"
)

print(
    f"Audited max degree:"
    f" {audited_max_degree}"
)


require(
    audited_max_degree == 123,
    (
        "Expected frozen maximum relation-specific "
        "incoming degree of 123."
    ),
)


high_positions = relation_positions[
    high_degree_relation
]


high_destinations = edge_index[
    1,
    high_positions,
]


unique_high_destinations, high_counts = torch.unique(
    high_destinations,
    return_counts=True,
)


actual_max_degree = int(
    high_counts.max()
)


require(
    actual_max_degree
    == audited_max_degree,
    (
        "Actual high-degree relation does not "
        "match Phase 4.4.1a degree audit."
    ),
)


candidate_destinations = (
    unique_high_destinations[
        high_counts
        == actual_max_degree
    ]
)


representative_destination = int(
    candidate_destinations.min()
)


representative_mask = (
    high_destinations
    == representative_destination
)


representative_edge_positions = (
    high_positions[
        representative_mask
    ]
)


representative_sources = edge_index[
    0,
    representative_edge_positions,
]


print(
    f"Representative destination node: "
    f"{representative_destination}"
)

print(
    f"Incoming neighbors in relation:  "
    f"{representative_sources.numel()}"
)


require(
    representative_sources.numel()
    == audited_max_degree,
    (
        "Representative high-degree destination "
        "does not contain expected neighbors."
    ),
)


# =============================================================================
# 12. REAL RELATION-SPECIFIC NORMALIZATION AUDIT
#
# Verify actual Crunchbase relation neighborhood:
#
#     aggregate =
#         sum(messages) / degree
#
#               =
#         mean(messages)
# =============================================================================

banner(
    "REAL RELATION-SPECIFIC NORMALIZATION AUDIT"
)


layer_1_weights = (
    model.layer_1
    .effective_relation_weights()
)


real_messages = (
    x0[
        representative_sources
    ]
    @ layer_1_weights[
        high_degree_relation
    ]
)


manual_relation_mean = (
    real_messages.mean(
        dim=0
    )
)


manual_relation_sum_div_degree = (
    real_messages.sum(
        dim=0
    )
    /
    float(
        representative_sources.numel()
    )
)


normalization_exact = torch.allclose(
    manual_relation_mean,
    manual_relation_sum_div_degree,
    atol=ATOL,
    rtol=RTOL,
)


normalization_diff = max_abs_difference(
    manual_relation_mean,
    manual_relation_sum_div_degree,
)


print(
    f"Relation ID:               "
    f"{high_degree_relation}"
)

print(
    f"Destination:               "
    f"{representative_destination}"
)

print(
    f"Incoming relation degree:  "
    f"{representative_sources.numel()}"
)

print(
    f"mean(messages) == "
    f"sum(messages)/degree: "
    f"{normalization_exact}"
)

print(
    f"Max numerical diff:        "
    f"{normalization_diff:.10f}"
)


require(
    normalization_exact,
    (
        "Actual relation-specific normalization "
        "does not equal mean aggregation."
    ),
)


# =============================================================================
# 13. FULL GRAPH — LAYER 1
# =============================================================================

banner(
    "FULL-GRAPH R-GCN LAYER 1"
)


layer_1_start = time.perf_counter()


z1 = model.layer_1.propagate(
    x0,
    edge_index,
    edge_type,
    relation_positions,
)


h1 = F.relu(
    z1
)


layer_1_seconds = (
    time.perf_counter()
    - layer_1_start
)


print(
    f"Preactivation shape: "
    f"{tuple(z1.shape)}"
)

print(
    f"Layer-1 output shape:"
    f" {tuple(h1.shape)}"
)

print(
    f"Finite:              "
    f"{bool(torch.isfinite(h1).all())}"
)

print(
    f"Nonnegative:         "
    f"{bool(torch.all(h1 >= 0))}"
)

print(
    f"Runtime:             "
    f"{layer_1_seconds:.3f} s"
)


require(
    tuple(
        z1.shape
    )
    == (
        NUM_NODES,
        40,
    ),
    "Layer-1 preactivation shape mismatch.",
)


require(
    tuple(
        h1.shape
    )
    == (
        NUM_NODES,
        40,
    ),
    "Layer-1 output shape mismatch.",
)


require(
    bool(
        torch.isfinite(
            h1
        ).all()
    ),
    "Layer-1 output contains non-finite values.",
)


require(
    bool(
        torch.all(
            h1 >= 0
        )
    ),
    "Layer-1 ReLU output contains negative values.",
)


relu_1_exact = torch.equal(
    h1,
    F.relu(
        z1
    ),
)


print(
    f"ReLU placement exact: "
    f"{relu_1_exact}"
)


require(
    relu_1_exact,
    "Layer-1 activation placement changed.",
)


# =============================================================================
# 14. ACTUAL NODE LAYER-1 MANUAL RECONSTRUCTION
# =============================================================================

banner(
    "REAL NODE LAYER-1 MANUAL RECONSTRUCTION"
)


manual_z1 = manual_node_preactivation(
    x=x0,
    node_index=representative_destination,
    layer=model.layer_1,
    edge_index=edge_index,
    edge_type=edge_type,
)


vectorized_z1 = z1[
    representative_destination
]


layer_1_manual_match = torch.allclose(
    manual_z1,
    vectorized_z1,
    atol=ATOL,
    rtol=RTOL,
)


layer_1_manual_diff = max_abs_difference(
    manual_z1,
    vectorized_z1,
)


print(
    f"Node:                   "
    f"{representative_destination}"
)

print(
    f"Manual/vectorized match:"
    f" {layer_1_manual_match}"
)

print(
    f"Max absolute diff:       "
    f"{layer_1_manual_diff:.10f}"
)


require(
    layer_1_manual_match,
    (
        "Full layer-1 implementation does not "
        "match independent manual node update."
    ),
)


# =============================================================================
# 15. VERIFY NEIGHBOR INFORMATION ACTUALLY CHANGES CONNECTED NODE
# =============================================================================

banner(
    "CONNECTED-NODE MESSAGE CONTRIBUTION"
)


root_only_layer_1 = (
    x0[
        representative_destination
    ]
    @ model.layer_1.root_weight
)


neighbor_contribution = (
    vectorized_z1
    - root_only_layer_1
)


neighbor_contribution_magnitude = float(
    torch.max(
        torch.abs(
            neighbor_contribution
        )
    )
)


print(
    f"Max absolute neighbor contribution: "
    f"{neighbor_contribution_magnitude:.10f}"
)


require(
    neighbor_contribution_magnitude
    > 1e-8,
    (
        "Representative connected node received "
        "no measurable neighbor contribution."
    ),
)


# =============================================================================
# 16. ISOLATE — LAYER 1
#
# For isolate i:
#
#     z_i = x_i @ W_root
#
#     h_i = ReLU(z_i)
# =============================================================================

banner(
    "STRUCTURAL ISOLATE — LAYER 1"
)


expected_isolate_z1 = (
    x0[
        representative_isolate
    ]
    @ model.layer_1.root_weight
)


expected_isolate_h1 = F.relu(
    expected_isolate_z1
)


actual_isolate_z1 = z1[
    representative_isolate
]


actual_isolate_h1 = h1[
    representative_isolate
]


isolate_z1_match = torch.allclose(
    expected_isolate_z1,
    actual_isolate_z1,
    atol=ATOL,
    rtol=RTOL,
)


isolate_h1_match = torch.allclose(
    expected_isolate_h1,
    actual_isolate_h1,
    atol=ATOL,
    rtol=RTOL,
)


print(
    f"Isolate node:              "
    f"{representative_isolate}"
)

print(
    f"Root-only preactivation:   "
    f"{isolate_z1_match}"
)

print(
    f"Root-only + ReLU output:   "
    f"{isolate_h1_match}"
)


require(
    isolate_z1_match,
    (
        "Isolate layer-1 preactivation contains "
        "unexpected neighbor information."
    ),
)


require(
    isolate_h1_match,
    "Isolate layer-1 update violates contract.",
)


# z1 is no longer needed after all layer-1 preactivation audits.
del z1

gc.collect()


# =============================================================================
# 17. FULL GRAPH — LAYER 2
# =============================================================================

banner(
    "FULL-GRAPH R-GCN LAYER 2"
)


layer_2_start = time.perf_counter()


z2 = model.layer_2.propagate(
    h1,
    edge_index,
    edge_type,
    relation_positions,
)


h2 = F.relu(
    z2
)


layer_2_seconds = (
    time.perf_counter()
    - layer_2_start
)


print(
    f"Preactivation shape: "
    f"{tuple(z2.shape)}"
)

print(
    f"Layer-2 output shape:"
    f" {tuple(h2.shape)}"
)

print(
    f"Finite:              "
    f"{bool(torch.isfinite(h2).all())}"
)

print(
    f"Nonnegative:         "
    f"{bool(torch.all(h2 >= 0))}"
)

print(
    f"Runtime:             "
    f"{layer_2_seconds:.3f} s"
)


require(
    tuple(
        z2.shape
    )
    == (
        NUM_NODES,
        40,
    ),
    "Layer-2 preactivation shape mismatch.",
)


require(
    tuple(
        h2.shape
    )
    == (
        NUM_NODES,
        40,
    ),
    "Layer-2 output shape mismatch.",
)


require(
    bool(
        torch.isfinite(
            h2
        ).all()
    ),
    "Layer-2 output contains non-finite values.",
)


require(
    bool(
        torch.all(
            h2 >= 0
        )
    ),
    "Layer-2 ReLU output contains negative values.",
)


relu_2_exact = torch.equal(
    h2,
    F.relu(
        z2
    ),
)


print(
    f"ReLU placement exact: "
    f"{relu_2_exact}"
)


require(
    relu_2_exact,
    "Layer-2 activation placement changed.",
)


# =============================================================================
# 18. ACTUAL NODE LAYER-2 MANUAL RECONSTRUCTION
#
# Uses FULL h1 features of all incoming neighbors.
#
# Therefore this also validates two-hop propagation:
#
#   layer-2 neighbors already contain layer-1 propagated information.
# =============================================================================

banner(
    "REAL NODE LAYER-2 MANUAL RECONSTRUCTION"
)


manual_z2 = manual_node_preactivation(
    x=h1,
    node_index=representative_destination,
    layer=model.layer_2,
    edge_index=edge_index,
    edge_type=edge_type,
)


vectorized_z2 = z2[
    representative_destination
]


layer_2_manual_match = torch.allclose(
    manual_z2,
    vectorized_z2,
    atol=ATOL,
    rtol=RTOL,
)


layer_2_manual_diff = max_abs_difference(
    manual_z2,
    vectorized_z2,
)


print(
    f"Node:                   "
    f"{representative_destination}"
)

print(
    f"Manual/vectorized match:"
    f" {layer_2_manual_match}"
)

print(
    f"Max absolute diff:       "
    f"{layer_2_manual_diff:.10f}"
)


require(
    layer_2_manual_match,
    (
        "Full layer-2 implementation does not "
        "match independent manual node update."
    ),
)


# =============================================================================
# 19. ISOLATE — LAYER 2
#
# Isolate remains relation-neighbor-free at every layer:
#
#     h1_i = ReLU(x0_i @ root1)
#
#     h2_i = ReLU(h1_i @ root2)
# =============================================================================

banner(
    "STRUCTURAL ISOLATE — LAYER 2"
)


expected_isolate_z2 = (
    h1[
        representative_isolate
    ]
    @ model.layer_2.root_weight
)


expected_isolate_h2 = F.relu(
    expected_isolate_z2
)


actual_isolate_z2 = z2[
    representative_isolate
]


actual_isolate_h2 = h2[
    representative_isolate
]


isolate_z2_match = torch.allclose(
    expected_isolate_z2,
    actual_isolate_z2,
    atol=ATOL,
    rtol=RTOL,
)


isolate_h2_match = torch.allclose(
    expected_isolate_h2,
    actual_isolate_h2,
    atol=ATOL,
    rtol=RTOL,
)


print(
    f"Root-only preactivation:   "
    f"{isolate_z2_match}"
)

print(
    f"Root-only + ReLU output:   "
    f"{isolate_h2_match}"
)

print(
    f"Final isolate forced zero: "
    f"{bool(torch.equal(actual_isolate_h2, torch.zeros_like(actual_isolate_h2)))}"
)


require(
    isolate_z2_match,
    (
        "Isolate layer-2 preactivation contains "
        "unexpected neighbor information."
    ),
)


require(
    isolate_h2_match,
    "Isolate layer-2 update violates contract.",
)


# z2 no longer needed.
del z2

gc.collect()


# =============================================================================
# 20. FINAL STRUCTURAL FEATURE MATRIX
# =============================================================================

banner(
    "FINAL STRUCTURAL FEATURE MATRIX"
)


structural_features = h2


print(
    f"F_s all nodes shape: "
    f"{tuple(structural_features.shape)}"
)

print(
    f"Finite:              "
    f"{bool(torch.isfinite(structural_features).all())}"
)

print(
    f"Nonnegative:         "
    f"{bool(torch.all(structural_features >= 0))}"
)


require(
    tuple(
        structural_features.shape
    )
    == (
        NUM_NODES,
        OUTPUT_DIM,
    ),
    "Final structural feature shape mismatch.",
)


# =============================================================================
# 21. ROLE SLICING
#
# Frozen node layout:
#
#   Investors: 0 ... 165974
#
#   Startups:  165975 ... 477563
#
# Paper:
#
#   slice n^(k) back to Fs,o and Fs,b.
# =============================================================================

banner(
    "INVESTOR / STARTUP STRUCTURAL FEATURE SLICING"
)


F_s_o = structural_features[
    :NUM_INVESTORS
]


F_s_b = structural_features[
    NUM_INVESTORS:
]


print(
    f"Investor structural features:"
)

print(
    f"  {tuple(F_s_o.shape)}"
)

print(
    f"Startup structural features:"
)

print(
    f"  {tuple(F_s_b.shape)}"
)


require(
    tuple(
        F_s_o.shape
    )
    == (
        NUM_INVESTORS,
        40,
    ),
    "Investor structural feature shape mismatch.",
)


require(
    tuple(
        F_s_b.shape
    )
    == (
        NUM_STARTUPS,
        40,
    ),
    "Startup structural feature shape mismatch.",
)


recombined = torch.cat(
    [
        F_s_o,
        F_s_b,
    ],
    dim=0,
)


slice_roundtrip_exact = torch.equal(
    recombined,
    structural_features,
)


print(
    f"Slice / recombine exact: "
    f"{slice_roundtrip_exact}"
)


require(
    slice_roundtrip_exact,
    "Role slicing changed structural features.",
)


# =============================================================================
# 22. NODE-TYPE RANGE CROSS-CHECK
# =============================================================================

banner(
    "ROLE INDEX RANGE CROSS-CHECK"
)


investor_role_correct = bool(
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


startup_role_correct = bool(
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
    f"Investor slice role-correct: "
    f"{investor_role_correct}"
)

print(
    f"Startup slice role-correct:  "
    f"{startup_role_correct}"
)


require(
    investor_role_correct,
    "Investor structural slice role mismatch.",
)


require(
    startup_role_correct,
    "Startup structural slice role mismatch.",
)


# =============================================================================
# 23. OUTPUT MAGNITUDE DIAGNOSTICS
#
# Diagnostic only.
#
# No thresholds are frozen from these values because they depend on the
# deliberately synthetic audit initialization.
# =============================================================================

banner(
    "AUDIT-ONLY OUTPUT MAGNITUDE DIAGNOSTICS"
)


all_abs = torch.abs(
    structural_features
)


print(
    f"Mean absolute value: "
    f"{float(all_abs.mean()):.8f}"
)

print(
    f"Maximum value:       "
    f"{float(structural_features.max()):.8f}"
)

print(
    f"Zero-value share:    "
    f"{float((structural_features == 0).float().mean()):.8%}"
)


# =============================================================================
# 24. TOTAL AUDIT RUNTIME
# =============================================================================

total_forward_seconds = (
    layer_1_seconds
    + layer_2_seconds
)


print()
print(
    f"Layer-1 propagation time: "
    f"{layer_1_seconds:.3f} s"
)

print(
    f"Layer-2 propagation time: "
    f"{layer_2_seconds:.3f} s"
)

print(
    f"Two-layer propagation:    "
    f"{total_forward_seconds:.3f} s"
)


# =============================================================================
# 25. SAVE FORWARD AUDIT TABLE
# =============================================================================

audit_records = [

    {
        "audit":
            "basis_reconstruction_layer_1",

        "status":
            "PASS",

        "detail":
            (
                "All 12 effective relation matrices "
                "equal sum_b a_rb V_b."
            ),
    },

    {
        "audit":
            "basis_reconstruction_layer_2",

        "status":
            "PASS",

        "detail":
            (
                "All 12 effective relation matrices "
                "equal sum_b a_rb V_b."
            ),
    },

    {
        "audit":
            "synthetic_directionality",

        "status":
            "PASS",

        "detail":
            (
                "Messages propagate source -> "
                "destination only."
            ),
    },

    {
        "audit":
            "synthetic_relation_mean",

        "status":
            "PASS",

        "detail":
            (
                "Two same-relation incoming messages "
                "aggregate by mean."
            ),
    },

    {
        "audit":
            "real_high_degree_normalization",

        "status":
            "PASS",

        "detail":
            (
                f"Relation {high_degree_relation}, "
                f"degree {audited_max_degree}: "
                "mean(messages) equals "
                "sum(messages)/degree."
            ),
    },

    {
        "audit":
            "layer_1_manual_node",

        "status":
            "PASS",

        "detail":
            (
                "Full vectorized layer-1 update "
                "matches independent manual node "
                "reconstruction."
            ),
    },

    {
        "audit":
            "layer_2_manual_node",

        "status":
            "PASS",

        "detail":
            (
                "Full vectorized layer-2 update "
                "matches independent manual node "
                "reconstruction."
            ),
    },

    {
        "audit":
            "isolate_layer_1",

        "status":
            "PASS",

        "detail":
            (
                "Isolate uses root transform only."
            ),
    },

    {
        "audit":
            "isolate_layer_2",

        "status":
            "PASS",

        "detail":
            (
                "Isolate remains root-only after "
                "second layer."
            ),
    },

    {
        "audit":
            "relu_layer_1",

        "status":
            "PASS",

        "detail":
            "ReLU applied after layer 1.",
    },

    {
        "audit":
            "relu_layer_2",

        "status":
            "PASS",

        "detail":
            "ReLU applied after layer 2.",
    },

    {
        "audit":
            "full_graph_shape",

        "status":
            "PASS",

        "detail":
            (
                f"[{NUM_NODES},40] -> "
                f"[{NUM_NODES},40]"
            ),
    },

    {
        "audit":
            "role_slicing",

        "status":
            "PASS",

        "detail":
            (
                f"Investor [{NUM_INVESTORS},40], "
                f"Startup [{NUM_STARTUPS},40]."
            ),
    },
]


audit_df = pd.DataFrame(
    audit_records
)


audit_path = (
    OUT_DIR
    / "rgcn_forward_audit.csv"
)


audit_df.to_csv(
    audit_path,
    index=False,
)


# =============================================================================
# 26. SAVE METADATA
# =============================================================================

metadata = {

    "phase":
        "4.4.2",

    "status":
        "COMPLETE",

    "component":
        (
            "ITRS preference-propagation "
            "R-GCN forward implementation audit"
        ),

    "environment":
        {

            "python":
                sys.version.splitlines()[0],

            "pytorch":
                torch.__version__,

            "device":
                "cpu",
        },

    "graph":
        {

            "nodes":
                NUM_NODES,

            "edges":
                NUM_EDGES,

            "relations":
                NUM_RELATIONS,

            "connected_nodes":
                connected_count,

            "isolates":
                isolate_count,
        },

    "architecture":
        {

            "layers":
                2,

            "dimensions":
                [
                    40,
                    40,
                    40,
                ],

            "relations":
                12,

            "bases":
                5,

            "parameters":
                parameter_count,
        },

    "forward_semantics_verified":
        {

            "basis_decomposition":
                True,

            "source_to_destination":
                True,

            "mean_within_relation":
                True,

            "sum_across_relations":
                True,

            "separate_root_transform":
                True,

            "relu_after_layer_1":
                True,

            "relu_after_layer_2":
                True,

            "isolate_root_only":
                True,

            "two_layer_message_passing":
                True,

            "role_slicing":
                True,
        },

    "real_high_degree_case":
        {

            "relation_id":
                high_degree_relation,

            "typed_relation_key":
                str(
                    high_degree_row[
                        "typed_relation_key"
                    ]
                ),

            "destination_node":
                representative_destination,

            "relation_specific_in_degree":
                audited_max_degree,

            "normalization":
                "1 / relation-specific incoming degree",
        },

    "representative_isolate":
        {

            "node_index":
                representative_isolate,

            "layer_1_behavior":
                "ReLU(x0 @ root_1)",

            "layer_2_behavior":
                "ReLU(h1 @ root_2)",

            "forced_zero":
                False,
        },

    "final_structural_features":
        {

            "all_nodes_shape":
                [
                    NUM_NODES,
                    40,
                ],

            "investor_shape":
                [
                    NUM_INVESTORS,
                    40,
                ],

            "startup_shape":
                [
                    NUM_STARTUPS,
                    40,
                ],

            "persisted":
                False,

            "reason":
                (
                    "Structural features depend on "
                    "trainable latent embeddings and "
                    "trainable R-GCN parameters and "
                    "must be recomputed from current "
                    "model parameters."
                ),
        },

    "audit_initialization":
        {

            "synthetic_node_features":
                True,

            "deterministic_rgcn_weights":
                True,

            "saved":
                False,

            "final_kaiming_variant_frozen":
                False,
        },

    "runtime_seconds":
        {

            "node_feature_generation":
                feature_seconds,

            "layer_1":
                layer_1_seconds,

            "layer_2":
                layer_2_seconds,

            "two_layer_total":
                total_forward_seconds,
        },

    "training_performed":
        False,

    "model_state_saved":
        False,

    "structural_features_saved":
        False,

    "phase_3_graph_modified":
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

            "phase_4_4_1":
                False,
        },
}


metadata_path = (
    OUT_DIR
    / "rgcn_forward_audit_metadata.json"
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
# FINAL SUMMARY
# =============================================================================

banner(
    "PHASE 4.4.2 FINAL SUMMARY"
)


print(
    "Basis decomposition:"
)

print(
    "  layer 1                       PASS"
)

print(
    "  layer 2                       PASS"
)


print()
print(
    "Message passing:"
)

print(
    "  source -> destination          PASS"
)

print(
    "  mean within typed relation     PASS"
)

print(
    "  sum across relations           PASS"
)

print(
    "  separate root transform        PASS"
)


print()
print(
    "Real relation normalization:"
)

print(
    f"  relation ID                    "
    f"{high_degree_relation}"
)

print(
    f"  incoming degree                "
    f"{audited_max_degree}"
)

print(
    "  normalized mean                PASS"
)


print()
print(
    "Manual full-node reconstruction:"
)

print(
    "  layer 1                       PASS"
)

print(
    "  layer 2                       PASS"
)


print()
print(
    "Activation:"
)

print(
    "  ReLU after layer 1             PASS"
)

print(
    "  ReLU after layer 2             PASS"
)


print()
print(
    "Structural isolate:"
)

print(
    "  root-only layer 1              PASS"
)

print(
    "  root-only layer 2              PASS"
)

print(
    "  forced zero                    NO"
)


print()
print(
    "Final structural representation:"
)

print(
    f"  all nodes      "
    f"[{NUM_NODES:,}, 40]"
)

print(
    f"  Investors      "
    f"[{NUM_INVESTORS:,}, 40]"
)

print(
    f"  Startups       "
    f"[{NUM_STARTUPS:,}, 40]"
)

print(
    "  role slice roundtrip           PASS"
)


print()
print(
    f"R-GCN parameters:                "
    f"{parameter_count:,}"
)


print()
print(
    f"Layer-1 runtime:                 "
    f"{layer_1_seconds:.3f} s"
)

print(
    f"Layer-2 runtime:                 "
    f"{layer_2_seconds:.3f} s"
)

print(
    f"Total two-layer runtime:         "
    f"{total_forward_seconds:.3f} s"
)


print()
print(
    "Training performed:              NO"
)

print(
    "Audit model persisted:           NO"
)

print(
    "Structural features persisted:   NO"
)

print(
    "Final Kaiming variant frozen:    NO"
)

print(
    "Phase-3 graph modified:          NO"
)


print()
print("Outputs:")

for path in [
    audit_path,
    metadata_path,
]:

    print(
        f"  {path}"
    )


print()
print(
    "PHASE 4.4.2 STATUS: COMPLETE — "
    "R-GCN FORWARD CONTRACT VERIFIED"
)