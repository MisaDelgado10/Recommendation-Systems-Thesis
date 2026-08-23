from pathlib import Path
import gc
import hashlib
import json
import math
import sys

import pandas as pd

import torch
import torch.nn as nn


# =============================================================================
# PHASE 4.7.1b — FREEZE GLOBAL NEURAL INITIALIZATION AND SEED CONTRACT
#
# PURPOSE
# -------
# Resolve the final two Phase-4 model-reconstruction blockers:
#
#   1. exact Kaiming initialization variant
#   2. global neural seed policy
#
#
# PAPER-GROUNDED FACT
# -------------------
# ITRS explicitly reports:
#
#   "The parameters were all initialized by Kaiming initialization."
#
#
# PAPER-UNSPECIFIED DETAILS
# -------------------------
# The paper does NOT report:
#
#   - Kaiming normal vs uniform
#   - fan_in vs fan_out
#   - exact nonlinearity/gain
#   - bias initialization
#   - random seed
#
#
# FROZEN REPRODUCTION CHOICE
# --------------------------
#
# MATRIX-LIKE PARAMETERS:
#
#   torch.nn.init.kaiming_normal_(
#       tensor,
#       a=0.0,
#       mode="fan_in",
#       nonlinearity="relu",
#       generator=canonical_cpu_generator,
#   )
#
# BIASES:
#
#   zeros_
#
# GLOBAL NEURAL SEED:
#
#   42
#
#
# CANONICAL INITIALIZATION DEVICE
# -------------------------------
# CPU.
#
# A dedicated CPU torch.Generator seeded with 42 is used to initialize every
# trainable matrix in an explicit frozen order.
#
# This avoids making the initial parameter state depend on the eventual
# training device.
#
#
# IMPORTANT
# ---------
# This script:
#
#   - DOES initialize temporary audit models,
#   - DOES verify same-seed byte-for-byte reproducibility,
#   - DOES verify a different seed changes the state,
#   - DOES compute a canonical model-state SHA-256,
#
# but:
#
#   - DOES NOT train,
#   - DOES NOT create an optimizer,
#   - DOES NOT call optimizer.step(),
#   - DOES NOT save model weights,
#   - DOES NOT generate negatives,
#   - DOES NOT freeze training epochs,
#   - DOES NOT freeze early stopping,
#   - DOES NOT freeze weight decay,
#   - DOES NOT freeze evaluation candidate generation,
#   - DOES NOT reopen Phase 2,
#   - DOES NOT reopen Phase 3.
# =============================================================================


# =============================================================================
# ROOTS
# =============================================================================

PHASE_4_ROOT = Path(
    "data/experimental/phase_4"
)


# =============================================================================
# UPSTREAM CONTRACTS
# =============================================================================

PHASE_4_7_1A_METADATA_PATH = (
    PHASE_4_ROOT
    / "model_integrity_audit"
    / "phase_4_7_1a_integrity_metadata.json"
)

FULL_MODEL_TOPOLOGY_CONTRACT_PATH = (
    PHASE_4_ROOT
    / "full_model_contract"
    / "full_itrs_model_topology_contract.json"
)

PARAMETER_NAMESPACE_PATH = (
    PHASE_4_ROOT
    / "full_model_contract"
    / "full_model_parameter_namespace.csv"
)


# =============================================================================
# OUTPUTS
# =============================================================================

OUT_DIR = (
    PHASE_4_ROOT
    / "initialization_contract"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# REFERENCE ENVIRONMENT
# =============================================================================

REFERENCE_TORCH_VERSION = "2.7.0"


# =============================================================================
# FROZEN GLOBAL NEURAL SEED
# =============================================================================

GLOBAL_NEURAL_SEED = 42

CONTROL_DIFFERENT_SEED = 43


# =============================================================================
# FROZEN POPULATION
# =============================================================================

NUM_INVESTORS = 165_975
NUM_STARTUPS = 311_589

NUM_NODES = 477_564
NUM_RELATIONS = 12


# =============================================================================
# FROZEN DIMENSIONS
# =============================================================================

LATENT_DIM = 40

DOC2VEC_DIM = 32
LABEL_DIM = 802

DESCRIPTION_TEXT_DIM = 20
DESCRIPTION_LABEL_DIM = 20

TREND_QUERY_DIM = 80
TREND_ITEM_DIM = 80
TREND_HIDDEN_DIM = 40
TREND_DIM = 40

STRUCTURAL_DIM = 40


# =============================================================================
# FROZEN PARAMETER TOTAL
# =============================================================================

EXPECTED_FULL_PARAMETERS = 19_217_929

EXPECTED_PARAMETER_TENSORS = 32


# =============================================================================
# INITIALIZATION CONTRACT
# =============================================================================

KAIMING_DISTRIBUTION = "normal"

KAIMING_A = 0.0

KAIMING_MODE = "fan_in"

KAIMING_NONLINEARITY = "relu"

BIAS_INITIALIZATION = "zeros"


# =============================================================================
# EXPECTED PHASE-4 CLOSURE BLOCKERS BEFORE THIS SUBPHASE
# =============================================================================

EXPECTED_UPSTREAM_BLOCKERS = {

    "exact global Kaiming initialization variant",

    "global neural seed policy",
}


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
        f"Missing required JSON: {path}",
    )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def count_parameters(module):

    return sum(
        parameter.numel()
        for parameter
        in module.parameters()
    )


def tensor_sha256(
    parameter,
):

    tensor = (
        parameter
        .detach()
        .cpu()
        .contiguous()
    )

    array = tensor.numpy()

    digest = hashlib.sha256()

    digest.update(
        str(
            tuple(
                tensor.shape
            )
        ).encode(
            "utf-8"
        )
    )

    digest.update(
        str(
            tensor.dtype
        ).encode(
            "utf-8"
        )
    )

    digest.update(
        array.tobytes(
            order="C"
        )
    )

    return digest.hexdigest()


def model_parameter_state_sha256(
    model,
):

    digest = hashlib.sha256()

    for (
        name,
        parameter,
    ) in model.named_parameters():

        tensor = (
            parameter
            .detach()
            .cpu()
            .contiguous()
        )

        array = tensor.numpy()

        digest.update(
            name.encode(
                "utf-8"
            )
        )

        digest.update(
            str(
                tuple(
                    tensor.shape
                )
            ).encode(
                "utf-8"
            )
        )

        digest.update(
            str(
                tensor.dtype
            ).encode(
                "utf-8"
            )
        )

        digest.update(
            array.tobytes(
                order="C"
            )
        )

    return digest.hexdigest()


def expected_kaiming_std(
    fan_in,
):

    # ReLU gain = sqrt(2)
    #
    # Kaiming normal:
    #
    #   std = gain / sqrt(fan_in)
    #
    #       = sqrt(2 / fan_in)

    return math.sqrt(
        2.0
        / float(
            fan_in
        )
    )


# =============================================================================
# MODEL TOPOLOGY
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


class BasisRGCNLayer(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
    ):

        super().__init__()

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


class ITRSModel(nn.Module):

    def __init__(self):

        super().__init__()

        # ---------------------------------------------------------------------
        # Shared latent tables
        # ---------------------------------------------------------------------

        self.investor_embedding = nn.Embedding(
            NUM_INVESTORS,
            LATENT_DIM,
        )

        self.startup_embedding = nn.Embedding(
            NUM_STARTUPS,
            LATENT_DIM,
        )


        # ---------------------------------------------------------------------
        # Reconstructed modules
        # ---------------------------------------------------------------------

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
# EXPLICIT FROZEN INITIALIZATION ORDER
#
# Why explicit?
#
# We do NOT want initialization to depend silently on module traversal order.
#
# Each trainable parameter is named here exactly once.
#
# Kinds:
#
#   "kaiming"
#       one matrix-like tensor
#
#   "kaiming_basis_stack"
#       R-GCN bases [5,in,out];
#       each 2-D basis matrix initialized independently
#
#   "zero_bias"
#       exact zeros
# =============================================================================

INITIALIZATION_SPECS = [

    # -------------------------------------------------------------------------
    # Shared latent embeddings
    # -------------------------------------------------------------------------

    {
        "name":
            "investor_embedding.weight",

        "kind":
            "kaiming",

        "fan_in":
            40,
    },

    {
        "name":
            "startup_embedding.weight",

        "kind":
            "kaiming",

        "fan_in":
            40,
    },


    # -------------------------------------------------------------------------
    # Description encoder
    # -------------------------------------------------------------------------

    {
        "name":
            "description_encoder.text_projection.weight",

        "kind":
            "kaiming",

        "fan_in":
            32,
    },

    {
        "name":
            "description_encoder.text_projection.bias",

        "kind":
            "zero_bias",

        "fan_in":
            None,
    },

    {
        "name":
            "description_encoder.label_projection.weight",

        "kind":
            "kaiming",

        "fan_in":
            802,
    },

    {
        "name":
            "description_encoder.label_projection.bias",

        "kind":
            "zero_bias",

        "fan_in":
            None,
    },


    # -------------------------------------------------------------------------
    # Trend attention
    # -------------------------------------------------------------------------

    {
        "name":
            "trend_extractor.attention_weight",

        "kind":
            "kaiming",

        "fan_in":
            80,
    },


    # -------------------------------------------------------------------------
    # GRU layer 0
    # -------------------------------------------------------------------------

    {
        "name":
            "trend_extractor.gru.weight_ih_l0",

        "kind":
            "kaiming",

        "fan_in":
            80,
    },

    {
        "name":
            "trend_extractor.gru.weight_hh_l0",

        "kind":
            "kaiming",

        "fan_in":
            40,
    },

    {
        "name":
            "trend_extractor.gru.bias_ih_l0",

        "kind":
            "zero_bias",

        "fan_in":
            None,
    },

    {
        "name":
            "trend_extractor.gru.bias_hh_l0",

        "kind":
            "zero_bias",

        "fan_in":
            None,
    },


    # -------------------------------------------------------------------------
    # GRU layer 1
    # -------------------------------------------------------------------------

    {
        "name":
            "trend_extractor.gru.weight_ih_l1",

        "kind":
            "kaiming",

        "fan_in":
            40,
    },

    {
        "name":
            "trend_extractor.gru.weight_hh_l1",

        "kind":
            "kaiming",

        "fan_in":
            40,
    },

    {
        "name":
            "trend_extractor.gru.bias_ih_l1",

        "kind":
            "zero_bias",

        "fan_in":
            None,
    },

    {
        "name":
            "trend_extractor.gru.bias_hh_l1",

        "kind":
            "zero_bias",

        "fan_in":
            None,
    },


    # -------------------------------------------------------------------------
    # Trend output projection
    # -------------------------------------------------------------------------

    {
        "name":
            "trend_extractor.output_projection.weight",

        "kind":
            "kaiming",

        "fan_in":
            40,
    },


    # -------------------------------------------------------------------------
    # R-GCN layer 1
    # -------------------------------------------------------------------------

    {
        "name":
            "preference_propagation.layer_1.bases",

        "kind":
            "kaiming_basis_stack",

        "fan_in":
            40,
    },

    {
        "name":
            "preference_propagation.layer_1.coefficients",

        "kind":
            "kaiming",

        "fan_in":
            5,
    },

    {
        "name":
            "preference_propagation.layer_1.root_weight",

        "kind":
            "kaiming",

        "fan_in":
            40,
    },


    # -------------------------------------------------------------------------
    # R-GCN layer 2
    # -------------------------------------------------------------------------

    {
        "name":
            "preference_propagation.layer_2.bases",

        "kind":
            "kaiming_basis_stack",

        "fan_in":
            40,
    },

    {
        "name":
            "preference_propagation.layer_2.coefficients",

        "kind":
            "kaiming",

        "fan_in":
            5,
    },

    {
        "name":
            "preference_propagation.layer_2.root_weight",

        "kind":
            "kaiming",

        "fan_in":
            40,
    },


    # -------------------------------------------------------------------------
    # Scoring hidden layer 1
    # -------------------------------------------------------------------------

    {
        "name":
            "scoring_mlp.hidden_1.weight",

        "kind":
            "kaiming",

        "fan_in":
            280,
    },

    {
        "name":
            "scoring_mlp.hidden_1.bias",

        "kind":
            "zero_bias",

        "fan_in":
            None,
    },


    # -------------------------------------------------------------------------
    # Scoring hidden layer 2
    # -------------------------------------------------------------------------

    {
        "name":
            "scoring_mlp.hidden_2.weight",

        "kind":
            "kaiming",

        "fan_in":
            128,
    },

    {
        "name":
            "scoring_mlp.hidden_2.bias",

        "kind":
            "zero_bias",

        "fan_in":
            None,
    },


    # -------------------------------------------------------------------------
    # Scoring hidden layer 3
    # -------------------------------------------------------------------------

    {
        "name":
            "scoring_mlp.hidden_3.weight",

        "kind":
            "kaiming",

        "fan_in":
            64,
    },

    {
        "name":
            "scoring_mlp.hidden_3.bias",

        "kind":
            "zero_bias",

        "fan_in":
            None,
    },


    # -------------------------------------------------------------------------
    # Scoring hidden layer 4
    # -------------------------------------------------------------------------

    {
        "name":
            "scoring_mlp.hidden_4.weight",

        "kind":
            "kaiming",

        "fan_in":
            32,
    },

    {
        "name":
            "scoring_mlp.hidden_4.bias",

        "kind":
            "zero_bias",

        "fan_in":
            None,
    },


    # -------------------------------------------------------------------------
    # Scoring output
    # -------------------------------------------------------------------------

    {
        "name":
            "scoring_mlp.output.weight",

        "kind":
            "kaiming",

        "fan_in":
            16,
    },

    {
        "name":
            "scoring_mlp.output.bias",

        "kind":
            "zero_bias",

        "fan_in":
            None,
    },
]


# =============================================================================
# CANONICAL INITIALIZER
# =============================================================================

def apply_canonical_initialization(
    model,
    seed,
):

    parameters = dict(
        model.named_parameters()
    )

    specification_names = [
        specification[
            "name"
        ]
        for specification
        in INITIALIZATION_SPECS
    ]


    # -------------------------------------------------------------------------
    # Every model parameter must appear exactly once.
    # -------------------------------------------------------------------------

    require(
        len(
            specification_names
        )
        == len(
            set(
                specification_names
            )
        ),
        "Initialization specification contains duplicate names.",
    )


    require(
        set(
            specification_names
        )
        == set(
            parameters.keys()
        ),
        (
            "Initialization specification does not "
            "exactly cover model parameter namespace."
        ),
    )


    require(
        len(
            parameters
        )
        == EXPECTED_PARAMETER_TENSORS,
        "Unexpected integrated parameter tensor count.",
    )


    # -------------------------------------------------------------------------
    # Canonical initialization generator.
    #
    # This generator is independent of:
    #
    #   - module constructor RNG consumption,
    #   - training-device RNG,
    #   - future negative-sampling RNG.
    # -------------------------------------------------------------------------

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        seed
    )


    with torch.no_grad():

        for specification in INITIALIZATION_SPECS:

            parameter_name = (
                specification[
                    "name"
                ]
            )

            initialization_kind = (
                specification[
                    "kind"
                ]
            )

            parameter = parameters[
                parameter_name
            ]


            require(
                parameter.device.type
                == "cpu",
                (
                    "Canonical initialization must "
                    "occur on CPU."
                ),
            )


            if initialization_kind == "kaiming":

                require(
                    parameter.ndim
                    >= 2,
                    (
                        "Kaiming parameter must be "
                        f"matrix-like: {parameter_name}"
                    ),
                )


                nn.init.kaiming_normal_(
                    parameter,
                    a=KAIMING_A,
                    mode=KAIMING_MODE,
                    nonlinearity=KAIMING_NONLINEARITY,
                    generator=generator,
                )


            elif (
                initialization_kind
                == "kaiming_basis_stack"
            ):

                require(
                    parameter.ndim
                    == 3,
                    (
                        "Basis-stack parameter must "
                        f"be rank-3: {parameter_name}"
                    ),
                )


                require(
                    parameter.shape[
                        0
                    ]
                    == 5,
                    (
                        "Frozen R-GCN basis count "
                        "must equal 5."
                    ),
                )


                # -------------------------------------------------------------
                # CRITICAL:
                #
                # Initialize each [in,out] basis independently.
                #
                # Do NOT call Kaiming on the complete [5,in,out] tensor,
                # because that would interpret the leading basis dimension
                # like an additional fan structure.
                # -------------------------------------------------------------

                for basis_index in range(
                    parameter.shape[
                        0
                    ]
                ):

                    nn.init.kaiming_normal_(
                        parameter[
                            basis_index
                        ],
                        a=KAIMING_A,
                        mode=KAIMING_MODE,
                        nonlinearity=KAIMING_NONLINEARITY,
                        generator=generator,
                    )


            elif initialization_kind == "zero_bias":

                require(
                    parameter.ndim
                    == 1,
                    (
                        "Bias-zero parameter must "
                        f"be rank-1: {parameter_name}"
                    ),
                )


                nn.init.zeros_(
                    parameter
                )


            else:

                raise AssertionError(
                    (
                        "Unknown initialization kind: "
                        f"{initialization_kind}"
                    )
                )


# =============================================================================
# BUILD CANONICAL MODEL
# =============================================================================

def build_canonical_model(
    seed,
):

    # -------------------------------------------------------------------------
    # Also seed PyTorch's global neural RNG.
    #
    # Module constructors consume the default RNG, but every trainable
    # parameter is then explicitly overwritten by the dedicated canonical
    # generator.
    #
    # Therefore constructor defaults cannot leak into the final state.
    # -------------------------------------------------------------------------

    torch.manual_seed(
        seed
    )


    model = ITRSModel()


    apply_canonical_initialization(
        model,
        seed=seed,
    )


    return model


# =============================================================================
# START
# =============================================================================

banner(
    "PHASE 4.7.1b — "
    "FREEZE GLOBAL NEURAL INITIALIZATION "
    "AND SEED CONTRACT"
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
    "Canonical initialization device: CPU"
)


torch_version_base = (
    torch.__version__
    .split(
        "+"
    )[
        0
    ]
)


torch_reference_match = (
    torch_version_base
    == REFERENCE_TORCH_VERSION
)


print()
print(
    f"Reference PyTorch version: "
    f"{REFERENCE_TORCH_VERSION}"
)

print(
    f"Current PyTorch version:   "
    f"{torch_version_base}"
)

print(
    f"Reference version match:   "
    f"{torch_reference_match}"
)


require(
    torch_reference_match,
    (
        "Exact initialization fingerprint must "
        "be established under PyTorch 2.7.0."
    ),
)


# =============================================================================
# 2. UPSTREAM PHASE 4.7.1a
# =============================================================================

banner(
    "UPSTREAM PHASE 4.7.1a INTEGRITY"
)


integrity_metadata = load_json(
    PHASE_4_7_1A_METADATA_PATH
)


require(
    integrity_metadata.get(
        "status"
    )
    == "COMPLETE_AUDIT_ONLY",
    "Phase 4.7.1a integrity audit is not complete.",
)


upstream_blockers = set(
    integrity_metadata[
        "phase_4_closure_blockers"
    ]
)


print(
    "Phase 4.7.1a status: PASS"
)

print()
print(
    "Upstream Phase-4 blockers:"
)


for blocker in sorted(
    upstream_blockers
):

    print(
        f"  - {blocker}"
    )


require(
    upstream_blockers
    == EXPECTED_UPSTREAM_BLOCKERS,
    (
        "Unexpected Phase-4 closure blocker set "
        "before initialization freeze."
    ),
)


# =============================================================================
# 3. UPSTREAM FULL MODEL TOPOLOGY
# =============================================================================

banner(
    "FROZEN FULL MODEL TOPOLOGY"
)


topology_contract = load_json(
    FULL_MODEL_TOPOLOGY_CONTRACT_PATH
)


require(
    topology_contract.get(
        "status"
    )
    == "FROZEN",
    "Full-model topology contract is not frozen.",
)


require(
    topology_contract[
        "parameter_budget"
    ][
        "total"
    ]
    == EXPECTED_FULL_PARAMETERS,
    "Frozen full-model parameter total changed.",
)


require(
    topology_contract[
        "latent_embeddings"
    ][
        "number_of_tables"
    ]
    == 2,
    "Frozen latent embedding-table count changed.",
)


print(
    f"Trainable parameters: "
    f"{EXPECTED_FULL_PARAMETERS:,}"
)

print(
    "Shared latent tables: 2"
)

print(
    "Topology contract:    PASS"
)


# =============================================================================
# 4. INITIALIZATION SPECIFICATION COVERAGE
# =============================================================================

banner(
    "INITIALIZATION SPECIFICATION COVERAGE"
)


namespace = pd.read_csv(
    PARAMETER_NAMESPACE_PATH
)


require(
    {
        "parameter",
        "numel",
    }
    .issubset(
        namespace.columns
    ),
    "Parameter namespace schema changed.",
)


frozen_parameter_names = set(
    namespace[
        "parameter"
    ]
    .astype(str)
)


specification_names = [
    specification[
        "name"
    ]
    for specification
    in INITIALIZATION_SPECS
]


specification_name_set = set(
    specification_names
)


print(
    f"Frozen parameter tensors:        "
    f"{len(frozen_parameter_names)}"
)

print(
    f"Initialization specifications:   "
    f"{len(INITIALIZATION_SPECS)}"
)

print(
    f"Unique specification names:      "
    f"{len(specification_name_set)}"
)


missing_initialization_specs = (
    frozen_parameter_names
    - specification_name_set
)


unexpected_initialization_specs = (
    specification_name_set
    - frozen_parameter_names
)


print(
    f"Missing initialization specs:    "
    f"{len(missing_initialization_specs)}"
)

print(
    f"Unexpected initialization specs: "
    f"{len(unexpected_initialization_specs)}"
)


require(
    len(
        missing_initialization_specs
    )
    == 0,
    (
        "A trainable parameter lacks an "
        "initialization specification."
    ),
)


require(
    len(
        unexpected_initialization_specs
    )
    == 0,
    (
        "Initialization contract contains "
        "an unexpected parameter."
    ),
)


require(
    len(
        specification_names
    )
    == len(
        specification_name_set
    ),
    "Initialization specification contains duplicates.",
)


kaiming_spec_count = sum(
    specification[
        "kind"
    ]
    in {
        "kaiming",
        "kaiming_basis_stack",
    }
    for specification
    in INITIALIZATION_SPECS
)


zero_bias_spec_count = sum(
    specification[
        "kind"
    ]
    == "zero_bias"
    for specification
    in INITIALIZATION_SPECS
)


print()
print(
    f"Kaiming parameter tensors: "
    f"{kaiming_spec_count}"
)

print(
    f"Zero-bias tensors:         "
    f"{zero_bias_spec_count}"
)


require(
    (
        kaiming_spec_count
        +
        zero_bias_spec_count
    )
    == EXPECTED_PARAMETER_TENSORS,
    "Initialization categories do not cover all parameters.",
)


# =============================================================================
# 5. FREEZE INITIALIZATION MATHEMATICS
# =============================================================================

banner(
    "FROZEN KAIMING MATHEMATICS"
)


relu_gain = nn.init.calculate_gain(
    "relu"
)


print(
    f"Distribution:    "
    f"Kaiming {KAIMING_DISTRIBUTION}"
)

print(
    f"a:               "
    f"{KAIMING_A}"
)

print(
    f"mode:            "
    f"{KAIMING_MODE}"
)

print(
    f"nonlinearity:    "
    f"{KAIMING_NONLINEARITY}"
)

print(
    f"ReLU gain:       "
    f"{relu_gain:.12f}"
)

print(
    f"Expected sqrt(2):"
    f" {math.sqrt(2.0):.12f}"
)


require(
    abs(
        relu_gain
        - math.sqrt(
            2.0
        )
    )
    < 1e-12,
    "PyTorch ReLU gain differs from expected sqrt(2).",
)


print()
print(
    "For matrix-like parameters:"
)

print(
    "  std = sqrt(2 / fan_in)"
)


print()
print(
    "For one-dimensional bias parameters:"
)

print(
    "  value = 0"
)


# =============================================================================
# 6. BUILD CANONICAL SEED-42 MODEL
# =============================================================================

banner(
    "CANONICAL SEED-42 INITIALIZATION"
)


model_a = build_canonical_model(
    GLOBAL_NEURAL_SEED
)


require(
    count_parameters(
        model_a
    )
    == EXPECTED_FULL_PARAMETERS,
    "Canonical initialized model parameter count changed.",
)


model_a_parameters = dict(
    model_a.named_parameters()
)


require(
    set(
        model_a_parameters.keys()
    )
    == specification_name_set,
    "Canonical model namespace differs from initialization contract.",
)


reference_state_hash = (
    model_parameter_state_sha256(
        model_a
    )
)


print(
    f"Global neural seed: "
    f"{GLOBAL_NEURAL_SEED}"
)

print(
    f"Model-state SHA256:"
)

print(
    f"  {reference_state_hash}"
)


# =============================================================================
# 7. PER-PARAMETER INITIALIZATION AUDIT
# =============================================================================

banner(
    "PER-PARAMETER INITIALIZATION AUDIT"
)


parameter_audit_records = []

reference_parameter_hashes = {}


for specification in INITIALIZATION_SPECS:

    parameter_name = specification[
        "name"
    ]

    initialization_kind = specification[
        "kind"
    ]

    fan_in = specification[
        "fan_in"
    ]

    parameter = model_a_parameters[
        parameter_name
    ]


    require(
        parameter.device.type
        == "cpu",
        (
            "Canonical parameter unexpectedly "
            f"left CPU: {parameter_name}"
        ),
    )


    finite = bool(
        torch.isfinite(
            parameter
        ).all()
    )


    require(
        finite,
        (
            "Initialized parameter contains "
            f"non-finite values: {parameter_name}"
        ),
    )


    parameter_hash = tensor_sha256(
        parameter
    )


    reference_parameter_hashes[
        parameter_name
    ] = parameter_hash


    mean = float(
        parameter
        .detach()
        .mean()
    )


    std = float(
        parameter
        .detach()
        .std(
            unbiased=False
        )
    )


    minimum = float(
        parameter
        .detach()
        .min()
    )


    maximum = float(
        parameter
        .detach()
        .max()
    )


    abs_sum = float(
        parameter
        .detach()
        .abs()
        .sum()
    )


    zero_fraction = float(
        (
            parameter
            .detach()
            == 0
        )
        .float()
        .mean()
    )


    if initialization_kind == "zero_bias":

        expected_std = 0.0

        exact_bias_zero = bool(
            torch.count_nonzero(
                parameter
            )
            == 0
        )


        require(
            exact_bias_zero,
            (
                "Frozen zero-bias contract failed: "
                f"{parameter_name}"
            ),
        )


        initialization_status = "PASS"


    else:

        expected_std = expected_kaiming_std(
            fan_in
        )


        require(
            parameter.ndim
            >= 2,
            (
                "Kaiming parameter is not "
                f"matrix-like: {parameter_name}"
            ),
        )


        require(
            abs_sum
            > 0.0,
            (
                "Kaiming parameter unexpectedly "
                f"all-zero: {parameter_name}"
            ),
        )


        require(
            std
            > 0.0,
            (
                "Kaiming parameter has zero "
                f"standard deviation: {parameter_name}"
            ),
        )


        exact_bias_zero = None

        initialization_status = "PASS"


    print(
        f"{parameter_name:<58} "
        f"{initialization_kind:<20} "
        f"fan_in={str(fan_in):<4} "
        f"mean={mean:+.6e} "
        f"std={std:.6e} "
        f"PASS"
    )


    parameter_audit_records.append(
        {

            "parameter":
                parameter_name,

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

            "initialization_kind":
                initialization_kind,

            "fan_in":
                fan_in,

            "kaiming_distribution":
                (
                    KAIMING_DISTRIBUTION
                    if initialization_kind
                    != "zero_bias"
                    else None
                ),

            "kaiming_a":
                (
                    KAIMING_A
                    if initialization_kind
                    != "zero_bias"
                    else None
                ),

            "kaiming_mode":
                (
                    KAIMING_MODE
                    if initialization_kind
                    != "zero_bias"
                    else None
                ),

            "kaiming_nonlinearity":
                (
                    KAIMING_NONLINEARITY
                    if initialization_kind
                    != "zero_bias"
                    else None
                ),

            "expected_kaiming_std":
                expected_std,

            "observed_mean":
                mean,

            "observed_std":
                std,

            "minimum":
                minimum,

            "maximum":
                maximum,

            "absolute_sum":
                abs_sum,

            "zero_fraction":
                zero_fraction,

            "bias_exact_zero":
                exact_bias_zero,

            "finite":
                finite,

            "sha256":
                parameter_hash,

            "status":
                initialization_status,
        }
    )


# =============================================================================
# 8. SAME-SEED REPRODUCIBILITY AUDIT
# =============================================================================

banner(
    "SAME-SEED BYTE-FOR-BYTE REPRODUCIBILITY"
)


# -----------------------------------------------------------------------------
# Release model A before building model B.
# -----------------------------------------------------------------------------

del model_a
del model_a_parameters

gc.collect()


model_b = build_canonical_model(
    GLOBAL_NEURAL_SEED
)


repeat_state_hash = (
    model_parameter_state_sha256(
        model_b
    )
)


same_seed_state_exact = (
    repeat_state_hash
    == reference_state_hash
)


print(
    f"Seed A: "
    f"{GLOBAL_NEURAL_SEED}"
)

print(
    f"Seed B: "
    f"{GLOBAL_NEURAL_SEED}"
)

print()
print(
    "Reference state SHA256:"
)

print(
    f"  {reference_state_hash}"
)

print(
    "Repeated state SHA256:"
)

print(
    f"  {repeat_state_hash}"
)

print()
print(
    f"Whole-model exact match: "
    f"{same_seed_state_exact}"
)


require(
    same_seed_state_exact,
    (
        "Canonical seed-42 initialization is "
        "not byte-for-byte reproducible."
    ),
)


model_b_parameters = dict(
    model_b.named_parameters()
)


same_seed_parameter_mismatches = []


for parameter_name in specification_names:

    repeated_hash = tensor_sha256(
        model_b_parameters[
            parameter_name
        ]
    )

    if (
        repeated_hash
        != reference_parameter_hashes[
            parameter_name
        ]
    ):

        same_seed_parameter_mismatches.append(
            parameter_name
        )


print(
    f"Per-parameter hash mismatches: "
    f"{len(same_seed_parameter_mismatches)}"
)


require(
    len(
        same_seed_parameter_mismatches
    )
    == 0,
    (
        "At least one parameter differs between "
        "two canonical seed-42 initializations."
    ),
)


# =============================================================================
# 9. DIFFERENT-SEED SENSITIVITY AUDIT
# =============================================================================

banner(
    "DIFFERENT-SEED SENSITIVITY"
)


del model_b
del model_b_parameters

gc.collect()


model_control = build_canonical_model(
    CONTROL_DIFFERENT_SEED
)


control_state_hash = (
    model_parameter_state_sha256(
        model_control
    )
)


different_seed_changes_state = (
    control_state_hash
    != reference_state_hash
)


print(
    f"Canonical seed: "
    f"{GLOBAL_NEURAL_SEED}"
)

print(
    f"Control seed:   "
    f"{CONTROL_DIFFERENT_SEED}"
)

print()
print(
    "Canonical SHA256:"
)

print(
    f"  {reference_state_hash}"
)

print(
    "Control SHA256:"
)

print(
    f"  {control_state_hash}"
)

print()
print(
    f"Different seed changes state: "
    f"{different_seed_changes_state}"
)


require(
    different_seed_changes_state,
    (
        "Different neural seed unexpectedly "
        "produced identical full model state."
    ),
)


del model_control

gc.collect()


# =============================================================================
# 10. INITIALIZATION CLASSIFICATION
# =============================================================================

banner(
    "PAPER / REPRODUCTION-CHOICE CLASSIFICATION"
)


classification_records = [

    {
        "decision":
            "Kaiming initialization family",

        "value":
            "Kaiming",

        "classification":
            "PAPER_SPECIFIED",

        "reason":
            (
                "ITRS explicitly reports that "
                "all parameters were initialized "
                "using Kaiming initialization."
            ),
    },

    {
        "decision":
            "Kaiming distribution",

        "value":
            "normal",

        "classification":
            "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE",

        "reason":
            (
                "ITRS does not report normal "
                "versus uniform Kaiming."
            ),
    },

    {
        "decision":
            "Kaiming mode",

        "value":
            "fan_in",

        "classification":
            "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE",

        "reason":
            (
                "ITRS does not report fan_in "
                "versus fan_out."
            ),
    },

    {
        "decision":
            "Kaiming nonlinearity",

        "value":
            "relu",

        "classification":
            "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE",

        "reason":
            (
                "ReLU is the paper-defined MLP "
                "activation and the canonical "
                "Kaiming rectifier setting."
            ),
    },

    {
        "decision":
            "Kaiming negative slope a",

        "value":
            "0.0",

        "classification":
            "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE",

        "reason":
            (
                "No LeakyReLU is present in the "
                "frozen reconstructed model."
            ),
    },

    {
        "decision":
            "Bias initialization",

        "value":
            "zero",

        "classification":
            "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE",

        "reason":
            (
                "Kaiming fan initialization is "
                "not defined for rank-1 biases."
            ),
    },

    {
        "decision":
            "Global neural seed",

        "value":
            str(
                GLOBAL_NEURAL_SEED
            ),

        "classification":
            "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE",

        "reason":
            (
                "ITRS does not report a neural "
                "random seed; seed 42 is frozen "
                "for this reproduction."
            ),
    },

    {
        "decision":
            "Canonical initialization device",

        "value":
            "CPU",

        "classification":
            "REPRODUCTION_ENVIRONMENT_CHOICE",

        "reason":
            (
                "Canonical CPU initialization "
                "prevents parameter initialization "
                "from depending on training device."
            ),
    },

    {
        "decision":
            "Initialization RNG",

        "value":
            "dedicated torch.Generator(device='cpu')",

        "classification":
            "REPRODUCTION_ENVIRONMENT_CHOICE",

        "reason":
            (
                "Separates parameter initialization "
                "RNG from module constructors and "
                "future training/evaluation RNGs."
            ),
    },

    {
        "decision":
            "Reference PyTorch version",

        "value":
            REFERENCE_TORCH_VERSION,

        "classification":
            "REPRODUCTION_ENVIRONMENT_LOCK",

        "reason":
            (
                "Exact random-number streams are "
                "not guaranteed across PyTorch "
                "versions/platforms."
            ),
    },

    {
        "decision":
            "Deterministic training algorithms",

        "value":
            "NOT_FROZEN_IN_PHASE_4_7_1B",

        "classification":
            "DEFER_TO_TRAINING_RUNTIME",

        "reason":
            (
                "This subphase freezes the initial "
                "parameter state, not runtime kernel "
                "determinism."
            ),
    },
]


classification_df = pd.DataFrame(
    classification_records
)


classification_path = (
    OUT_DIR
    / "phase_4_7_1b_initialization_decision_audit.csv"
)


classification_df.to_csv(
    classification_path,
    index=False,
)


# =============================================================================
# 11. SAVE PARAMETER INITIALIZATION AUDIT
# =============================================================================

parameter_audit_df = pd.DataFrame(
    parameter_audit_records
)


parameter_audit_path = (
    OUT_DIR
    / "phase_4_7_1b_parameter_initialization_audit.csv"
)


parameter_audit_df.to_csv(
    parameter_audit_path,
    index=False,
)


# =============================================================================
# 12. SAVE SEED REPRODUCIBILITY AUDIT
# =============================================================================

seed_audit_records = [

    {
        "check":
            "canonical_global_neural_seed",

        "expected":
            GLOBAL_NEURAL_SEED,

        "actual":
            GLOBAL_NEURAL_SEED,

        "status":
            "PASS",
    },

    {
        "check":
            "same_seed_whole_state_exact",

        "expected":
            True,

        "actual":
            same_seed_state_exact,

        "status":
            "PASS",
    },

    {
        "check":
            "same_seed_parameter_hash_mismatches",

        "expected":
            0,

        "actual":
            len(
                same_seed_parameter_mismatches
            ),

        "status":
            "PASS",
    },

    {
        "check":
            "different_seed_changes_state",

        "expected":
            True,

        "actual":
            different_seed_changes_state,

        "status":
            "PASS",
    },

    {
        "check":
            "reference_torch_version",

        "expected":
            REFERENCE_TORCH_VERSION,

        "actual":
            torch_version_base,

        "status":
            "PASS",
    },
]


seed_audit_df = pd.DataFrame(
    seed_audit_records
)


seed_audit_path = (
    OUT_DIR
    / "phase_4_7_1b_seed_reproducibility_audit.csv"
)


seed_audit_df.to_csv(
    seed_audit_path,
    index=False,
)


# =============================================================================
# 13. SAVE INITIALIZATION STATE HASH MANIFEST
# =============================================================================

state_hash_manifest = {

    "phase":
        "4.7.1b",

    "status":
        "FROZEN",

    "reference_environment":
        {

            "pytorch":
                REFERENCE_TORCH_VERSION,

            "initialization_device":
                "cpu",
        },

    "global_neural_seed":
        GLOBAL_NEURAL_SEED,

    "canonical_state_sha256":
        reference_state_hash,

    "repeat_same_seed_sha256":
        repeat_state_hash,

    "control_different_seed":
        CONTROL_DIFFERENT_SEED,

    "control_state_sha256":
        control_state_hash,

    "same_seed_exact":
        same_seed_state_exact,

    "different_seed_changes_state":
        different_seed_changes_state,

    "model_state_persisted":
        False,
}


state_hash_path = (
    OUT_DIR
    / "phase_4_7_1b_initialization_state_hash.json"
)


with open(
    state_hash_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        state_hash_manifest,
        f,
        indent=2,
        ensure_ascii=False,
    )


# =============================================================================
# 14. FREEZE COMPLETE INITIALIZATION CONTRACT
# =============================================================================

banner(
    "FREEZING GLOBAL NEURAL INITIALIZATION CONTRACT"
)


initialization_contract = {

    "phase":
        "4.7.1b",

    "status":
        "FROZEN",

    "component":
        "Global neural parameter initialization and seed",

    "paper_basis":
        {

            "initialization_family":
                "Kaiming",

            "paper_reports_all_parameters_initialized_by_kaiming":
                True,

            "paper_reports_exact_variant":
                False,

            "paper_reports_seed":
                False,
        },

    "global_neural_seed":
        {

            "value":
                GLOBAL_NEURAL_SEED,

            "classification":
                "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE",

            "scope":
                (
                    "neural model initialization "
                    "and neural RNG base seed"
                ),

            "negative_sampling_rng_frozen_here":
                False,

            "evaluation_sampling_rng_frozen_here":
                False,
        },

    "canonical_rng":
        {

            "library":
                "torch",

            "type":
                "torch.Generator",

            "device":
                "cpu",

            "seed":
                GLOBAL_NEURAL_SEED,

            "independent_of_module_constructor_rng":
                True,

            "independent_of_future_negative_sampling_rng":
                True,
        },

    "kaiming":
        {

            "distribution":
                KAIMING_DISTRIBUTION,

            "function":
                "torch.nn.init.kaiming_normal_",

            "a":
                KAIMING_A,

            "mode":
                KAIMING_MODE,

            "nonlinearity":
                KAIMING_NONLINEARITY,

            "relu_gain":
                relu_gain,

            "standard_deviation":
                "sqrt(2 / fan_in)",

            "classification":
                "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE",
        },

    "bias":
        {

            "initialization":
                BIAS_INITIALIZATION,

            "value":
                0.0,

            "classification":
                "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE",
        },

    "rgcn_basis_initialization":
        {

            "basis_tensor_shape":
                [
                    5,
                    40,
                    40,
                ],

            "basis_matrices_initialized_independently":
                True,

            "individual_basis_shape":
                [
                    40,
                    40,
                ],

            "reason":
                (
                    "Avoid treating basis count as "
                    "a convolution-like fan dimension."
                ),
        },

    "coverage":
        {

            "parameter_tensors":
                EXPECTED_PARAMETER_TENSORS,

            "kaiming_parameter_tensors":
                kaiming_spec_count,

            "zero_bias_tensors":
                zero_bias_spec_count,

            "missing_parameters":
                0,

            "unexpected_parameters":
                0,

            "duplicate_initialization_specs":
                0,
        },

    "reference_environment":
        {

            "pytorch":
                REFERENCE_TORCH_VERSION,

            "canonical_initialization_device":
                "cpu",

            "training_device":
                "NOT_FROZEN_BY_THIS_CONTRACT",
        },

    "reproducibility":
        {

            "canonical_state_sha256":
                reference_state_hash,

            "repeat_state_sha256":
                repeat_state_hash,

            "same_seed_byte_exact":
                True,

            "different_seed_control":
                CONTROL_DIFFERENT_SEED,

            "different_seed_state_sha256":
                control_state_hash,

            "different_seed_changes_state":
                True,
        },

    "backend_determinism":
        {

            "torch_use_deterministic_algorithms":
                "NOT_FROZEN_IN_THIS_SUBPHASE",

            "classification":
                "DEFER_TO_TRAINING_RUNTIME",

            "reason":
                (
                    "Initialization reproducibility "
                    "and training-kernel determinism "
                    "are separate concerns."
                ),
        },

    "training":
        {

            "performed":
                False,

            "optimizer_created":
                False,

            "optimizer_step":
                False,

            "model_state_persisted":
                False,
        },

    "still_deferred_to_training_evaluation":
        [

            "training negative:positive ratio",

            "training negative candidate eligibility",

            "training historical negative exclusion",

            "training epoch count",

            "early stopping",

            "weight decay",

            "evaluation candidate-generation runtime contract",
        ],

    "phase_4_model_reconstruction_blockers_remaining":
        [],

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

            "phase_4_6":
                False,

            "phase_4_7_1a":
                False,
        },

    "next_phase":
        {

            "phase":
                "4.7.2",

            "name":
                (
                    "Final Post-Initialization "
                    "Model Integrity Audit"
                ),
        },
}


initialization_contract_path = (
    OUT_DIR
    / "phase_4_7_1b_neural_initialization_contract.json"
)


with open(
    initialization_contract_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        initialization_contract,
        f,
        indent=2,
        ensure_ascii=False,
    )


# =============================================================================
# 15. ARTIFACT HASHES
# =============================================================================

def file_sha256(path):

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


artifact_paths = [

    classification_path,

    parameter_audit_path,

    seed_audit_path,

    state_hash_path,

    initialization_contract_path,
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
                file_sha256(
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
    / "phase_4_7_1b_artifact_hashes.csv"
)


artifact_hash_df.to_csv(
    artifact_hash_path,
    index=False,
)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

banner(
    "PHASE 4.7.1b FINAL SUMMARY"
)


print(
    "Paper-specified:"
)

print(
    "  initialization family          Kaiming"
)


print()
print(
    "Frozen reproduction variant:"
)

print(
    "  function                       "
    "torch.nn.init.kaiming_normal_"
)

print(
    "  distribution                   normal"
)

print(
    "  a                              0.0"
)

print(
    "  mode                           fan_in"
)

print(
    "  nonlinearity                   relu"
)

print(
    "  matrix std                     sqrt(2 / fan_in)"
)

print(
    "  bias                           zero"
)


print()
print(
    "Global neural seed:"
)

print(
    f"  value                          "
    f"{GLOBAL_NEURAL_SEED}"
)

print(
    "  canonical RNG                  "
    "dedicated CPU torch.Generator"
)

print(
    "  canonical init device          CPU"
)

print(
    f"  reference PyTorch              "
    f"{REFERENCE_TORCH_VERSION}"
)


print()
print(
    "Initialization coverage:"
)

print(
    f"  total parameter tensors        "
    f"{EXPECTED_PARAMETER_TENSORS}"
)

print(
    f"  Kaiming tensors                "
    f"{kaiming_spec_count}"
)

print(
    f"  zero-bias tensors              "
    f"{zero_bias_spec_count}"
)

print(
    "  missing initialization specs   0"
)

print(
    "  unexpected initialization specs 0"
)


print()
print(
    "Seed reproducibility:"
)

print(
    "  same seed whole-state exact    PASS"
)

print(
    "  same seed tensor hashes exact  PASS"
)

print(
    "  different seed changes state   PASS"
)


print()
print(
    "Canonical initialized-state SHA256:"
)

print(
    f"  {reference_state_hash}"
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
    "  training epoch                 NO"
)

print(
    "  negative generation            NO"
)

print(
    "  model weights persisted        NO"
)


print()
print(
    "Still deferred:"
)

print(
    "  training negative:positive ratio"
)

print(
    "  negative candidate eligibility"
)

print(
    "  historical negative exclusion"
)

print(
    "  training epoch count"
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
    "Phase-4 MODEL-RECONSTRUCTION blockers remaining:"
)

print(
    "  NONE"
)


print()
print(
    "Outputs:"
)


for path in [

    classification_path,

    parameter_audit_path,

    seed_audit_path,

    state_hash_path,

    initialization_contract_path,

    artifact_hash_path,
]:

    print(
        f"  {path}"
    )


print()
print(
    "PHASE 4.7.1b STATUS: COMPLETE — "
    "GLOBAL NEURAL INITIALIZATION AND "
    "SEED CONTRACT FROZEN"
)


print()
print(
    "NEXT:"
)

print(
    "PHASE 4.7.2 — "
    "FINAL POST-INITIALIZATION "
    "MODEL INTEGRITY AUDIT"
)