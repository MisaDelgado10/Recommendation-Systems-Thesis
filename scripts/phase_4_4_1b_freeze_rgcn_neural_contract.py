from pathlib import Path
import json
import sys

import pandas as pd
import torch
import torch.nn as nn


# =============================================================================
# PHASE 4.4.1b — FREEZE R-GCN NEURAL CONTRACT
#
# PURPOSE
# -------
# Freeze the neural architecture of the ITRS preference-propagation module
# after the Phase-4.4.1a structural input audit.
#
# IMPORTANT
# ---------
# This script:
#   - does NOT train,
#   - does NOT modify the Phase-3 graph,
#   - does NOT add structural self-loop edges,
#   - does NOT add investment-event edges,
#   - does NOT save an initialized model state,
#   - does NOT freeze the exact Kaiming initialization variant.
#
# It freezes:
#   - layer dimensions,
#   - relation/basis parameterization,
#   - relation-specific normalization semantics,
#   - root/self transform semantics,
#   - activation,
#   - bias/dropout policies,
#   - isolate behavior,
#   - exact parameter shapes/counts.
# =============================================================================


# =============================================================================
# INPUTS
# =============================================================================

INPUT_AUDIT_METADATA_PATH = Path(
    "data/experimental/phase_4/"
    "rgcn_input_audit/"
    "rgcn_structural_input_audit_metadata.json"
)

RELATION_AUDIT_PATH = Path(
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
    "rgcn_neural_contract"
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


# =============================================================================
# PAPER-SPECIFIED MODEL CONSTANTS
# =============================================================================

INPUT_DIM = 40
HIDDEN_DIM = 40
OUTPUT_DIM = 40

NUM_LAYERS = 2
NUM_BASES = 5


# =============================================================================
# REPRODUCTION CHOICES
# =============================================================================

ACTIVATION = "relu"

USE_LAYER_BIAS = False

USE_ROOT_TRANSFORM = True

ADD_EXPLICIT_SELF_LOOPS = False

USE_DROPOUT = False
DROPOUT = 0.0

USE_BATCH_NORM = False

USE_LAYER_NORM = False

USE_RESIDUAL = False

MASK_ISOLATES_TO_ZERO = False


# =============================================================================
# EXPECTED PARAMETER COUNTS
# =============================================================================

EXPECTED_BASIS_PARAMETERS_PER_LAYER = (
    NUM_BASES
    * INPUT_DIM
    * OUTPUT_DIM
)

EXPECTED_COEFFICIENT_PARAMETERS_PER_LAYER = (
    NUM_RELATIONS
    * NUM_BASES
)

EXPECTED_ROOT_PARAMETERS_PER_LAYER = (
    INPUT_DIM
    * OUTPUT_DIM
)

EXPECTED_PARAMETERS_PER_LAYER = (
    EXPECTED_BASIS_PARAMETERS_PER_LAYER
    + EXPECTED_COEFFICIENT_PARAMETERS_PER_LAYER
    + EXPECTED_ROOT_PARAMETERS_PER_LAYER
)

EXPECTED_TOTAL_PARAMETERS = (
    NUM_LAYERS
    * EXPECTED_PARAMETERS_PER_LAYER
)


assert (
    EXPECTED_BASIS_PARAMETERS_PER_LAYER
    == 8_000
)

assert (
    EXPECTED_COEFFICIENT_PARAMETERS_PER_LAYER
    == 60
)

assert (
    EXPECTED_ROOT_PARAMETERS_PER_LAYER
    == 1_600
)

assert (
    EXPECTED_PARAMETERS_PER_LAYER
    == 9_660
)

assert (
    EXPECTED_TOTAL_PARAMETERS
    == 19_320
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


# =============================================================================
# CUSTOM CONTRACT LAYER
#
# This class defines PARAMETERS ONLY.
#
# Full graph message passing is deliberately deferred to the forward audit.
#
# Row-vector PyTorch convention:
#
#     message = x_source @ W_r
#
# Basis matrices therefore have:
#
#     [num_bases, in_dim, out_dim]
#
# This is the row-vector equivalent of the column-vector mathematical form:
#
#     W_r h_j
# =============================================================================

class BasisRGCNLayerContract(nn.Module):

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


        # ---------------------------------------------------------------------
        # Basis matrices:
        #
        # V_b
        #
        # Shape:
        #   [B, in_dim, out_dim]
        # ---------------------------------------------------------------------

        self.bases = nn.Parameter(
            torch.empty(
                self.num_bases,
                self.in_dim,
                self.out_dim,
            )
        )


        # ---------------------------------------------------------------------
        # Relation coefficients:
        #
        # a_rb
        #
        # Shape:
        #   [R, B]
        #
        # Effective relation matrix:
        #
        #   W_r = sum_b a_rb V_b
        # ---------------------------------------------------------------------

        self.coefficients = nn.Parameter(
            torch.empty(
                self.num_relations,
                self.num_bases,
            )
        )


        # ---------------------------------------------------------------------
        # Root/self transformation:
        #
        # ITRS Eq. (9) includes:
        #
        #   + W^(k-1) n_e^(k-1)
        #
        # separately from relation-neighbor aggregation.
        #
        # No explicit self-loop graph edges are required.
        # ---------------------------------------------------------------------

        self.root_weight = nn.Parameter(
            torch.empty(
                self.in_dim,
                self.out_dim,
            )
        )


    def effective_relation_weights(
        self,
    ):

        # ---------------------------------------------------------------------
        # Reconstruct:
        #
        #   W_r = sum_b a_rb V_b
        #
        # coefficients:
        #   [R, B]
        #
        # bases flattened:
        #   [B, in_dim * out_dim]
        #
        # result:
        #   [R, in_dim, out_dim]
        # ---------------------------------------------------------------------

        weights = (
            self.coefficients
            @ self.bases.reshape(
                self.num_bases,
                -1,
            )
        )


        weights = weights.reshape(
            self.num_relations,
            self.in_dim,
            self.out_dim,
        )


        return weights


# =============================================================================
# TWO-LAYER CONTRACT MODULE
# =============================================================================

class ITRSPreferencePropagationContract(
    nn.Module
):

    def __init__(self):

        super().__init__()


        self.layer_1 = (
            BasisRGCNLayerContract(
                in_dim=INPUT_DIM,
                out_dim=HIDDEN_DIM,
                num_relations=NUM_RELATIONS,
                num_bases=NUM_BASES,
            )
        )


        self.layer_2 = (
            BasisRGCNLayerContract(
                in_dim=HIDDEN_DIM,
                out_dim=OUTPUT_DIM,
                num_relations=NUM_RELATIONS,
                num_bases=NUM_BASES,
            )
        )


        self.activation = nn.ReLU()


# =============================================================================
# START
# =============================================================================

banner(
    "PHASE 4.4.1b — "
    "FREEZE R-GCN NEURAL CONTRACT"
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
    "Implementation dependency:"
)

print(
    "  PyTorch only"
)

print(
    "  PyTorch Geometric required: NO"
)


# =============================================================================
# 2. VERIFY UPSTREAM 4.4.1a AUDIT
# =============================================================================

banner(
    "UPSTREAM STRUCTURAL INPUT CONTRACT"
)


with open(
    INPUT_AUDIT_METADATA_PATH,
    "r",
    encoding="utf-8",
) as f:

    input_audit = json.load(f)


require(
    input_audit.get(
        "status"
    )
    == "COMPLETE_AUDIT_ONLY",
    (
        "Phase 4.4.1a structural input "
        "audit is not complete."
    ),
)


frozen_graph = (
    input_audit[
        "frozen_graph"
    ]
)


require(
    frozen_graph[
        "node_count"
    ]
    == NUM_NODES,
    "Frozen node count changed.",
)


require(
    frozen_graph[
        "edge_count"
    ]
    == NUM_EDGES,
    "Frozen edge count changed.",
)


require(
    frozen_graph[
        "typed_relation_count"
    ]
    == NUM_RELATIONS,
    "Frozen relation count changed.",
)


require(
    input_audit[
        "edge_integrity"
    ][
        "explicit_self_loops"
    ]
    == 0,
    (
        "Frozen graph unexpectedly "
        "contains self-loop edges."
    ),
)


require(
    input_audit[
        "edge_integrity"
    ][
        "duplicate_typed_edges"
    ]
    == 0,
    (
        "Frozen graph unexpectedly "
        "contains duplicate typed edges."
    ),
)


require(
    input_audit[
        "edge_integrity"
    ][
        "source_target_role_mismatches"
    ]
    == 0,
    (
        "Frozen relation-role mapping "
        "is inconsistent."
    ),
)


print(
    "Phase 4.4.1a audit: PASS"
)

print(
    f"Frozen nodes:         "
    f"{NUM_NODES:,}"
)

print(
    f"Frozen edges:         "
    f"{NUM_EDGES:,}"
)

print(
    f"Frozen relations:     "
    f"{NUM_RELATIONS}"
)


# =============================================================================
# 3. VERIFY RELATION CONTRACT AUDIT
# =============================================================================

banner(
    "TYPED RELATION CONTRACT INTEGRITY"
)


relations = pd.read_csv(
    RELATION_AUDIT_PATH
)


require(
    len(
        relations
    )
    == NUM_RELATIONS,
    "Relation audit row count changed.",
)


required_relation_columns = [
    "relation_id",
    "source_type",
    "relation",
    "target_type",
    "typed_relation_key",
    "expected_edge_count",
    "actual_edge_count_from_edge_type",
    "edge_count_match",
]


missing_columns = [
    column

    for column
    in required_relation_columns

    if column not in relations.columns
]


require(
    len(
        missing_columns
    )
    == 0,
    (
        "Missing relation audit fields: "
        f"{missing_columns}"
    ),
)


require(
    relations[
        "edge_count_match"
    ]
    .astype(bool)
    .all(),
    "A frozen relation edge count differs.",
)


require(
    relations[
        "relation_id"
    ]
    .astype(int)
    .tolist()
    == list(
        range(
            NUM_RELATIONS
        )
    ),
    "Relation IDs are not 0..11.",
)


print(
    "Typed relation IDs:    PASS"
)

print(
    "Typed relation counts: PASS"
)


# =============================================================================
# 4. VERIFY MODEL-READY GRAPH SHAPES
# =============================================================================

banner(
    "MODEL-READY GRAPH SHAPE CONTRACT"
)


import numpy as np


edge_index = np.load(
    EDGE_INDEX_PATH,
    mmap_mode="r",
)


edge_type = np.load(
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
    edge_index.shape
    == (
        2,
        NUM_EDGES,
    ),
    "edge_index shape changed.",
)


require(
    edge_type.shape
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
    "node_index population changed.",
)


print(
    f"Initial node representation:"
)

print(
    f"  [{NUM_NODES:,}, {INPUT_DIM}]"
)

print(
    f"edge_index:"
)

print(
    f"  {edge_index.shape}"
)

print(
    f"edge_type:"
)

print(
    f"  {edge_type.shape}"
)


# =============================================================================
# 5. FREEZE DIMENSION CONTRACT
# =============================================================================

banner(
    "R-GCN DIMENSION CONTRACT"
)


print(
    f"Input node dimension:      "
    f"{INPUT_DIM}"
)

print(
    f"Layer-1 output dimension:  "
    f"{HIDDEN_DIM}"
)

print(
    f"Layer-2 output dimension:  "
    f"{OUTPUT_DIM}"
)

print(
    f"Structural feature dim:    "
    f"{OUTPUT_DIM}"
)


print()
print(
    f"R-GCN layers:              "
    f"{NUM_LAYERS}"
)

print(
    f"Typed relations:           "
    f"{NUM_RELATIONS}"
)

print(
    f"Bases per layer:           "
    f"{NUM_BASES}"
)


require(
    INPUT_DIM
    == HIDDEN_DIM
    == OUTPUT_DIM
    == 40,
    (
        "ITRS structural dimension "
        "contract must remain 40."
    ),
)


# =============================================================================
# 6. FREEZE MESSAGE-PASSING SEMANTICS
# =============================================================================

banner(
    "MESSAGE-PASSING CONTRACT"
)


print(
    "For destination node i "
    "and typed relation r:"
)

print()

print(
    "  m_i,r = mean("
    "x_j @ W_r for j in N_i^r)"
)

print()

print(
    "Layer output before activation:"
)

print()

print(
    "  z_i = sum_r m_i,r "
    "+ x_i @ W_root"
)

print()

print(
    "Layer output:"
)

print()

print(
    "  x'_i = ReLU(z_i)"
)


print()
print(
    "Relation-specific normalization:"
)

print(
    "  denominator = incoming "
    "neighbor count for that"
)

print(
    "                destination "
    "node AND typed relation"
)


print()
print(
    "Cross-relation normalization: NO"
)

print(
    "Global-degree normalization:  NO"
)

print(
    "Symmetric GCN normalization:  NO"
)


# =============================================================================
# 7. FREEZE BASIS-DECOMPOSITION SEMANTICS
# =============================================================================

banner(
    "BASIS-DECOMPOSITION CONTRACT"
)


print(
    "Per layer:"
)

print(
    f"  bases:        "
    f"[{NUM_BASES}, 40, 40]"
)

print(
    f"  coefficients: "
    f"[{NUM_RELATIONS}, "
    f"{NUM_BASES}]"
)

print(
    f"  effective W:  "
    f"[{NUM_RELATIONS}, 40, 40]"
)


print()
print(
    "Effective relation matrix:"
)

print()

print(
    "  W_r = sum_b a[r,b] * V_b"
)


print()
print(
    "Basis sharing scope:"
)

print(
    "  all 12 frozen typed "
    "relation channels in a layer"
)

print(
    "  layer 1 and layer 2 "
    "have separate basis sets"
)


# =============================================================================
# 8. FREEZE SELF/ROOT SEMANTICS
# =============================================================================

banner(
    "SELF / ROOT TRANSFORMATION CONTRACT"
)


print(
    "Explicit graph self-loop edges: NO"
)

print(
    "Special self relation ID:       NO"
)

print(
    "Separate neural root matrix:    YES"
)

print(
    "Root matrix per layer:          "
    "[40, 40]"
)

print(
    "Root matrix shared across "
    "node roles: YES"
)


print()
print(
    "Paper equation:"
)

print(
    "  neighbor aggregation "
    "+ self transformation"
)


# =============================================================================
# 9. FREEZE ACTIVATION / BIAS / REGULARIZATION
# =============================================================================

banner(
    "ACTIVATION AND LAYER EXTRAS"
)


print(
    f"Activation after layer 1: "
    f"{ACTIVATION}"
)

print(
    f"Activation after layer 2: "
    f"{ACTIVATION}"
)


print()
print(
    f"Additive layer bias:       "
    f"{USE_LAYER_BIAS}"
)

print(
    f"Dropout:                   "
    f"{DROPOUT}"
)

print(
    f"BatchNorm:                 "
    f"{USE_BATCH_NORM}"
)

print(
    f"LayerNorm:                 "
    f"{USE_LAYER_NORM}"
)

print(
    f"Residual connection:       "
    f"{USE_RESIDUAL}"
)


# =============================================================================
# 10. ISOLATE SEMANTICS
# =============================================================================

banner(
    "STRUCTURAL ISOLATE SEMANTICS"
)


isolate_count = int(
    frozen_graph[
        "isolates"
    ]
)


connected_count = int(
    frozen_graph[
        "connected_nodes"
    ]
)


print(
    f"Connected nodes: "
    f"{connected_count:,}"
)

print(
    f"Isolates:        "
    f"{isolate_count:,}"
)


print()
print(
    "For an isolate:"
)

print()

print(
    "  all relation-neighbor "
    "aggregates = zero"
)

print()

print(
    "  layer output = "
    "ReLU(x_i @ W_root)"
)


print()
print(
    "Force structural feature "
    "to zero: NO"
)

print(
    "Remove isolate from graph: "
    "NO"
)

print(
    "Skip root transformation:   "
    "NO"
)


# =============================================================================
# 11. INSTANTIATE CONTRACT MODULE
#
# Audit-only instantiation.
#
# Default torch.empty values have no semantic meaning.
# No state is saved.
# =============================================================================

banner(
    "MODULE INSTANTIATION"
)


module = (
    ITRSPreferencePropagationContract()
)


print(
    module
)


# =============================================================================
# 12. EFFECTIVE RELATION-WEIGHT SHAPE AUDIT
# =============================================================================

banner(
    "EFFECTIVE RELATION-WEIGHT SHAPES"
)


layer_1_effective = (
    module
    .layer_1
    .effective_relation_weights()
)


layer_2_effective = (
    module
    .layer_2
    .effective_relation_weights()
)


print(
    f"Layer 1 effective relation weights:"
)

print(
    f"  {tuple(layer_1_effective.shape)}"
)

print(
    f"Layer 2 effective relation weights:"
)

print(
    f"  {tuple(layer_2_effective.shape)}"
)


require(
    tuple(
        layer_1_effective.shape
    )
    == (
        NUM_RELATIONS,
        40,
        40,
    ),
    (
        "Layer-1 effective relation "
        "shape mismatch."
    ),
)


require(
    tuple(
        layer_2_effective.shape
    )
    == (
        NUM_RELATIONS,
        40,
        40,
    ),
    (
        "Layer-2 effective relation "
        "shape mismatch."
    ),
)


# =============================================================================
# 13. EXACT PARAMETER SHAPE AUDIT
# =============================================================================

banner(
    "PARAMETER SHAPE AUDIT"
)


parameter_shapes = {
    name:
        tuple(
            parameter.shape
        )

    for name, parameter
    in module.named_parameters()
}


for name in sorted(
    parameter_shapes
):

    print(
        f"{name:<40} "
        f"{parameter_shapes[name]}"
    )


expected_shapes = {

    "layer_1.bases":
        (
            NUM_BASES,
            40,
            40,
        ),

    "layer_1.coefficients":
        (
            NUM_RELATIONS,
            NUM_BASES,
        ),

    "layer_1.root_weight":
        (
            40,
            40,
        ),

    "layer_2.bases":
        (
            NUM_BASES,
            40,
            40,
        ),

    "layer_2.coefficients":
        (
            NUM_RELATIONS,
            NUM_BASES,
        ),

    "layer_2.root_weight":
        (
            40,
            40,
        ),
}


require(
    parameter_shapes
    == expected_shapes,
    (
        "R-GCN parameter shape contract "
        "does not match expectation."
    ),
)


print()
print(
    "Exact parameter shapes: PASS"
)


# =============================================================================
# 14. PARAMETER COUNT AUDIT
# =============================================================================

banner(
    "PARAMETER COUNT AUDIT"
)


layer_1_basis = (
    module
    .layer_1
    .bases
    .numel()
)


layer_1_coefficients = (
    module
    .layer_1
    .coefficients
    .numel()
)


layer_1_root = (
    module
    .layer_1
    .root_weight
    .numel()
)


layer_2_basis = (
    module
    .layer_2
    .bases
    .numel()
)


layer_2_coefficients = (
    module
    .layer_2
    .coefficients
    .numel()
)


layer_2_root = (
    module
    .layer_2
    .root_weight
    .numel()
)


layer_1_total = (
    layer_1_basis
    + layer_1_coefficients
    + layer_1_root
)


layer_2_total = (
    layer_2_basis
    + layer_2_coefficients
    + layer_2_root
)


total_parameters = (
    layer_1_total
    + layer_2_total
)


print(
    "Layer 1:"
)

print(
    f"  bases:        "
    f"{layer_1_basis:,}"
)

print(
    f"  coefficients: "
    f"{layer_1_coefficients:,}"
)

print(
    f"  root:         "
    f"{layer_1_root:,}"
)

print(
    f"  total:        "
    f"{layer_1_total:,}"
)


print()
print(
    "Layer 2:"
)

print(
    f"  bases:        "
    f"{layer_2_basis:,}"
)

print(
    f"  coefficients: "
    f"{layer_2_coefficients:,}"
)

print(
    f"  root:         "
    f"{layer_2_root:,}"
)

print(
    f"  total:        "
    f"{layer_2_total:,}"
)


print()
print(
    f"TOTAL R-GCN PARAMETERS: "
    f"{total_parameters:,}"
)


require(
    layer_1_basis
    == EXPECTED_BASIS_PARAMETERS_PER_LAYER,
    "Layer-1 basis parameter count mismatch.",
)


require(
    layer_1_coefficients
    == EXPECTED_COEFFICIENT_PARAMETERS_PER_LAYER,
    (
        "Layer-1 coefficient parameter "
        "count mismatch."
    ),
)


require(
    layer_1_root
    == EXPECTED_ROOT_PARAMETERS_PER_LAYER,
    "Layer-1 root parameter count mismatch.",
)


require(
    layer_1_total
    == EXPECTED_PARAMETERS_PER_LAYER,
    "Layer-1 total parameter count mismatch.",
)


require(
    layer_2_total
    == EXPECTED_PARAMETERS_PER_LAYER,
    "Layer-2 total parameter count mismatch.",
)


require(
    total_parameters
    == EXPECTED_TOTAL_PARAMETERS,
    "Total R-GCN parameter count mismatch.",
)


# =============================================================================
# 15. PARAMETER-SHARING CONTRACT
# =============================================================================

banner(
    "PARAMETER-SHARING CONTRACT"
)


print(
    "Within one layer:"
)

print(
    "  relation matrices share "
    "5 basis matrices"
)

print(
    "  each typed relation has "
    "its own 5 coefficients"
)


print()
print(
    "Across layers:"
)

print(
    "  basis matrices shared:       NO"
)

print(
    "  relation coefficients shared:NO"
)

print(
    "  root matrices shared:        NO"
)


print()
print(
    "Across node roles:"
)

print(
    "  root matrix is common to "
    "Investor and Startup nodes"
)

print(
    "  relation typing itself "
    "distinguishes source/target roles"
)


# =============================================================================
# 16. PAPER / REPRODUCTION CLASSIFICATION
# =============================================================================

banner(
    "DECISION CLASSIFICATION"
)


decision_records = [

    {
        "decision":
            "initial node features",

        "value":
            "latent embeddings only",

        "classification":
            "PAPER_SPECIFIED",
    },

    {
        "decision":
            "latent input dimension",

        "value":
            "40",

        "classification":
            "PAPER_SPECIFIED",
    },

    {
        "decision":
            "structural output dimension",

        "value":
            "40",

        "classification":
            "PAPER_SPECIFIED",
    },

    {
        "decision":
            "R-GCN layers",

        "value":
            "2",

        "classification":
            "PAPER_SPECIFIED",
    },

    {
        "decision":
            "basis count",

        "value":
            "5",

        "classification":
            "PAPER_SPECIFIED",
    },

    {
        "decision":
            "neighbor normalization",

        "value":
            (
                "mean per destination "
                "and relation"
            ),

        "classification":
            "PAPER_SPECIFIED",
    },

    {
        "decision":
            "separate root/self transform",

        "value":
            "yes",

        "classification":
            "PAPER_SPECIFIED",
    },

    {
        "decision":
            "explicit graph self loops",

        "value":
            "no",

        "classification":
            (
                "PAPER_GROUNDED_"
                "IMPLEMENTATION"
            ),
    },

    {
        "decision":
            "basis decomposition formula",

        "value":
            "W_r=sum_b a_rb V_b",

        "classification":
            (
                "PAPER_SPECIFIED_VIA_"
                "RGCN_REFERENCE"
            ),
    },

    {
        "decision":
            "typed relation basis scope",

        "value":
            "12 Phase-3 typed channels",

        "classification":
            (
                "CRUNCHBASE_GRAPH_"
                "ADAPTATION_FROZEN_PHASE3"
            ),
    },

    {
        "decision":
            "activation",

        "value":
            "ReLU after both layers",

        "classification":
            (
                "PAPER_UNSPECIFIED_"
                "REPRODUCTION_CHOICE"
            ),
    },

    {
        "decision":
            "additive layer bias",

        "value":
            "none",

        "classification":
            (
                "PAPER_GROUNDED_"
                "IMPLEMENTATION"
            ),
    },

    {
        "decision":
            "dropout",

        "value":
            "0",

        "classification":
            (
                "PAPER_UNSPECIFIED_"
                "REPRODUCTION_CHOICE"
            ),
    },

    {
        "decision":
            "normalization layers",

        "value":
            "none",

        "classification":
            (
                "PAPER_UNSPECIFIED_"
                "REPRODUCTION_CHOICE"
            ),
    },

    {
        "decision":
            "residual connection",

        "value":
            "none",

        "classification":
            (
                "PAPER_UNSPECIFIED_"
                "REPRODUCTION_CHOICE"
            ),
    },

    {
        "decision":
            "isolate structural masking",

        "value":
            "none",

        "classification":
            (
                "PAPER_GROUNDED_"
                "IMPLEMENTATION"
            ),
    },

    {
        "decision":
            "implementation library",

        "value":
            "explicit PyTorch",

        "classification":
            (
                "REPRODUCTION_"
                "IMPLEMENTATION_CHOICE"
            ),
    },
]


decision_df = pd.DataFrame(
    decision_records
)


decision_path = (
    OUT_DIR
    / "rgcn_neural_decision_audit.csv"
)


decision_df.to_csv(
    decision_path,
    index=False,
)


# =============================================================================
# 17. SAVE PARAMETER AUDIT
# =============================================================================

parameter_records = []


for name, parameter in (
    module.named_parameters()
):

    parameter_records.append(
        {
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

            "trainable":
                bool(
                    parameter.requires_grad
                ),
        }
    )


parameter_df = pd.DataFrame(
    parameter_records
)


parameter_path = (
    OUT_DIR
    / "rgcn_neural_parameter_audit.csv"
)


parameter_df.to_csv(
    parameter_path,
    index=False,
)


# =============================================================================
# 18. FREEZE CONTRACT
# =============================================================================

banner(
    "FREEZING R-GCN NEURAL CONTRACT"
)


contract = {

    "phase":
        "4.4.1b",

    "status":
        "FROZEN",

    "component":
        (
            "ITRS preference propagation "
            "R-GCN neural architecture"
        ),

    "implementation":
        {

            "library":
                "torch",

            "torch_geometric_required":
                False,

            "custom_layer":
                True,

            "reason":
                (
                    "Explicit implementation makes "
                    "ITRS Eq. (9), relation-specific "
                    "normalization, basis decomposition, "
                    "root transform and bias policy "
                    "auditable without relying on "
                    "library defaults."
                ),
        },

    "graph_input":
        {

            "node_count":
                NUM_NODES,

            "edge_count":
                NUM_EDGES,

            "typed_relation_count":
                NUM_RELATIONS,

            "initial_node_features":
                "latent embeddings only",

            "initial_node_matrix_shape":
                [
                    NUM_NODES,
                    INPUT_DIM,
                ],

            "investment_event_edges":
                False,

            "explicit_self_loop_edges":
                False,
        },

    "architecture":
        {

            "layers":
                NUM_LAYERS,

            "input_dim":
                INPUT_DIM,

            "hidden_dim":
                HIDDEN_DIM,

            "output_dim":
                OUTPUT_DIM,

            "layer_dimensions":
                [
                    [
                        40,
                        40,
                    ],

                    [
                        40,
                        40,
                    ],
                ],

            "num_relations":
                NUM_RELATIONS,

            "num_bases":
                NUM_BASES,
        },

    "basis_decomposition":
        {

            "formula":
                "W_r = sum_b a[r,b] * V_b",

            "basis_shape_per_layer":
                [
                    NUM_BASES,
                    40,
                    40,
                ],

            "coefficient_shape_per_layer":
                [
                    NUM_RELATIONS,
                    NUM_BASES,
                ],

            "effective_weight_shape":
                [
                    NUM_RELATIONS,
                    40,
                    40,
                ],

            "basis_scope":
                (
                    "all 12 typed relation "
                    "channels within each layer"
                ),

            "basis_shared_between_layers":
                False,

            "coefficient_shared_between_layers":
                False,
        },

    "message_passing":
        {

            "edge_direction":
                "source_to_destination",

            "normalization":
                (
                    "1 / incoming neighbor count "
                    "for each destination node "
                    "and typed relation"
                ),

            "aggregation_within_relation":
                "mean",

            "aggregation_across_relations":
                "sum",

            "global_degree_normalization":
                False,

            "symmetric_gcn_normalization":
                False,
        },

    "root_transform":
        {

            "enabled":
                True,

            "shape_per_layer":
                [
                    40,
                    40,
                ],

            "separate_from_neighbor_edges":
                True,

            "shared_between_node_roles":
                True,

            "shared_between_layers":
                False,
        },

    "activation":
        {

            "function":
                "ReLU",

            "after_layer_1":
                True,

            "after_layer_2":
                True,

            "classification":
                (
                    "PAPER_UNSPECIFIED_"
                    "REPRODUCTION_CHOICE"
                ),

            "rationale":
                (
                    "ITRS Eq. (9) uses sigma without "
                    "identifying it. The referenced "
                    "R-GCN formulation describes an "
                    "element-wise activation such as "
                    "ReLU. ReLU is therefore frozen "
                    "as the explicit reproduction "
                    "choice."
                ),
        },

    "extras":
        {

            "additive_bias":
                False,

            "dropout":
                0.0,

            "batch_norm":
                False,

            "layer_norm":
                False,

            "residual":
                False,
        },

    "isolate_behavior":
        {

            "mask_to_zero":
                False,

            "remove_from_graph":
                False,

            "relation_aggregate":
                "zero",

            "layer_update":
                "ReLU(x_i @ W_root)",

            "rationale":
                (
                    "ITRS Eq. (9) retains the "
                    "self/root transformation even "
                    "when a node has no relation "
                    "neighbors."
                ),
        },

    "parameter_count":
        {

            "basis_per_layer":
                EXPECTED_BASIS_PARAMETERS_PER_LAYER,

            "coefficients_per_layer":
                EXPECTED_COEFFICIENT_PARAMETERS_PER_LAYER,

            "root_per_layer":
                EXPECTED_ROOT_PARAMETERS_PER_LAYER,

            "total_per_layer":
                EXPECTED_PARAMETERS_PER_LAYER,

            "total_two_layer_rgcn":
                EXPECTED_TOTAL_PARAMETERS,

            "excludes":
                [
                    "Investor latent embedding table",
                    "Startup latent embedding table",
                    "Description encoder",
                    "Trend module",
                    "Recommendation scoring MLP",
                ],
        },

    "initialization":
        {

            "paper_specified_family":
                "Kaiming",

            "exact_kaiming_variant":
                "NOT_YET_FROZEN",

            "contract_module_default_values":
                "AUDIT_ONLY_UNINITIALIZED",

            "state_saved":
                False,
        },

    "training_performed":
        False,

    "forward_message_passing_audited":
        False,

    "next_phase":
        (
            "Phase 4.4.2 — "
            "R-GCN forward implementation audit"
        ),

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

            "phase_4_4_1a":
                False,
        },
}


contract_path = (
    OUT_DIR
    / "rgcn_neural_contract.json"
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
# FINAL SUMMARY
# =============================================================================

banner(
    "PHASE 4.4.1b FINAL SUMMARY"
)


print(
    "Implementation:                 "
    "explicit PyTorch"
)

print(
    "PyTorch Geometric required:     NO"
)


print()
print(
    "R-GCN architecture:"
)

print(
    "  layers                        2"
)

print(
    "  dimensions                   "
    "40 -> 40 -> 40"
)

print(
    "  typed relations              12"
)

print(
    "  bases per layer               5"
)


print()
print(
    "Message aggregation:"
)

print(
    "  source -> destination"
)

print(
    "  mean within typed relation"
)

print(
    "  sum across relations"
)

print(
    "  separate root/self transform"
)


print()
print(
    "Activation:"
)

print(
    "  ReLU after layer 1"
)

print(
    "  ReLU after layer 2"
)


print()
print(
    "Additive layer bias:            NO"
)

print(
    "Dropout:                       0.0"
)

print(
    "Batch/Layer normalization:      NO"
)

print(
    "Residual connections:           NO"
)


print()
print(
    "Explicit graph self-loop edges: NO"
)

print(
    "Isolates forced to zero:        NO"
)

print(
    "Isolate update:"
)

print(
    "  ReLU(x @ W_root)"
)


print()
print(
    "Parameter counts:"
)

print(
    f"  bases / layer:        "
    f"{EXPECTED_BASIS_PARAMETERS_PER_LAYER:,}"
)

print(
    f"  coefficients / layer: "
    f"{EXPECTED_COEFFICIENT_PARAMETERS_PER_LAYER:,}"
)

print(
    f"  root / layer:         "
    f"{EXPECTED_ROOT_PARAMETERS_PER_LAYER:,}"
)

print(
    f"  total / layer:        "
    f"{EXPECTED_PARAMETERS_PER_LAYER:,}"
)

print(
    f"  total R-GCN:          "
    f"{EXPECTED_TOTAL_PARAMETERS:,}"
)


print()
print(
    "Final Kaiming variant frozen:   NO"
)

print(
    "Training performed:             NO"
)

print(
    "Model state persisted:          NO"
)

print(
    "Forward propagation verified:   NO"
)


print()
print("Outputs:")

for path in [
    contract_path,
    parameter_path,
    decision_path,
]:

    print(
        f"  {path}"
    )


print()
print(
    "PHASE 4.4.1b STATUS: COMPLETE — "
    "R-GCN NEURAL CONTRACT FROZEN"
)