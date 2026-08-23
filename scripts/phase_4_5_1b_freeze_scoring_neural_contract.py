from pathlib import Path
import json
import sys

import pandas as pd
import torch
import torch.nn as nn


# =============================================================================
# PHASE 4.5.1b — FREEZE RECOMMENDATION SCORING NEURAL CONTRACT
#
# PURPOSE
# -------
# Resolve and explicitly freeze the paper-unspecified neural implementation
# choices for the final ITRS recommendation-scoring MLP.
#
# UPSTREAM PAPER-SPECIFIED CONTRACT
# ---------------------------------
#
# Pair representation:
#
#   [ F_t
#     || L_o
#     || F_d,o
#     || F_s,o
#     || L_b
#     || F_d,b
#     || F_s,b ]
#
# Input dimension:
#
#   280
#
# Paper specifies:
#
#   - four hidden layers,
#   - ReLU MLP,
#   - binary logistic prediction,
#   - BCE objective.
#
# Paper does NOT specify:
#
#   - the four hidden widths,
#   - hidden/output bias policy,
#   - whether sigmoid+BCE or logits+BCEWithLogits is used in code,
#   - scoring dropout,
#   - normalization layers,
#   - residual connections.
#
# FROZEN REPRODUCTION CHOICES
# ---------------------------
#
# Architecture:
#
#   280 -> 128 -> 64 -> 32 -> 16 -> 1
#
# Hidden widths:
#
#   [128, 64, 32, 16]
#
# Rationale:
#
#   ITRS explicitly describes the scoring stage as NCF-based.
#   The canonical NCF implementation uses a progressively decreasing /
#   approximately halving MLP topology. Since the ITRS scoring input is
#   280-D and the paper explicitly states FOUR hidden layers, this
#   reproduction adopts:
#
#       128 -> 64 -> 32 -> 16
#
#   This is NOT claimed to be the unpublished original ITRS width setting.
#   It is explicitly recorded as a paper-unspecified reproduction choice
#   guided by the referenced NCF architecture.
#
# Output/loss implementation:
#
#   training:
#
#       raw logit -> BCEWithLogitsLoss
#
#   prediction/reporting:
#
#       probability = sigmoid(logit)
#
# This is mathematically equivalent to the paper's logistic+BCE objective
# while being numerically more stable.
#
# IMPORTANT
# ---------
# This phase does NOT:
#
#   - train the model,
#   - freeze negative-sampling semantics,
#   - freeze optimizer epochs,
#   - freeze early stopping,
#   - freeze weight decay,
#   - freeze the exact Kaiming variant,
#   - save initialized model parameters.
# =============================================================================


# =============================================================================
# INPUTS
# =============================================================================

SCORING_INPUT_CONTRACT_PATH = Path(
    "data/experimental/phase_4/"
    "scoring_contract/"
    "scoring_input_contract.json"
)

SCORING_REPRESENTATION_PATH = Path(
    "data/experimental/phase_4/"
    "scoring_contract/"
    "scoring_pair_representation_contract.csv"
)

SCORING_PAPER_AUDIT_PATH = Path(
    "data/experimental/phase_4/"
    "scoring_contract/"
    "scoring_paper_contract_audit.csv"
)


# =============================================================================
# OUTPUTS
# =============================================================================

OUT_DIR = Path(
    "data/experimental/phase_4/"
    "scoring_neural_contract"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# FROZEN INPUT / OUTPUT DIMENSIONS
# =============================================================================

INPUT_DIM = 280

HIDDEN_WIDTHS = [
    128,
    64,
    32,
    16,
]

HIDDEN_LAYER_COUNT = 4

OUTPUT_DIM = 1


# =============================================================================
# FROZEN SCORING IMPLEMENTATION CHOICES
# =============================================================================

HIDDEN_ACTIVATION = "ReLU"

HIDDEN_BIAS = True

OUTPUT_BIAS = True

DROPOUT = 0.0

USE_BATCH_NORM = False

USE_LAYER_NORM = False

USE_RESIDUAL = False

TRAINING_OUTPUT = "logit"

TRAINING_LOSS = "BCEWithLogitsLoss"

PROBABILITY_TRANSFORM = "sigmoid"


# =============================================================================
# EXPECTED PARAMETER COUNTS
# =============================================================================

EXPECTED_LAYER_1_PARAMETERS = (
    INPUT_DIM
    * HIDDEN_WIDTHS[0]
    + HIDDEN_WIDTHS[0]
)

EXPECTED_LAYER_2_PARAMETERS = (
    HIDDEN_WIDTHS[0]
    * HIDDEN_WIDTHS[1]
    + HIDDEN_WIDTHS[1]
)

EXPECTED_LAYER_3_PARAMETERS = (
    HIDDEN_WIDTHS[1]
    * HIDDEN_WIDTHS[2]
    + HIDDEN_WIDTHS[2]
)

EXPECTED_LAYER_4_PARAMETERS = (
    HIDDEN_WIDTHS[2]
    * HIDDEN_WIDTHS[3]
    + HIDDEN_WIDTHS[3]
)

EXPECTED_OUTPUT_PARAMETERS = (
    HIDDEN_WIDTHS[3]
    * OUTPUT_DIM
    + OUTPUT_DIM
)

EXPECTED_TOTAL_PARAMETERS = (
    EXPECTED_LAYER_1_PARAMETERS
    + EXPECTED_LAYER_2_PARAMETERS
    + EXPECTED_LAYER_3_PARAMETERS
    + EXPECTED_LAYER_4_PARAMETERS
    + EXPECTED_OUTPUT_PARAMETERS
)


assert (
    EXPECTED_LAYER_1_PARAMETERS
    == 35_968
)

assert (
    EXPECTED_LAYER_2_PARAMETERS
    == 8_256
)

assert (
    EXPECTED_LAYER_3_PARAMETERS
    == 2_080
)

assert (
    EXPECTED_LAYER_4_PARAMETERS
    == 528
)

assert (
    EXPECTED_OUTPUT_PARAMETERS
    == 17
)

assert (
    EXPECTED_TOTAL_PARAMETERS
    == 46_849
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
        f"Missing contract: {path}",
    )


    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# =============================================================================
# SCORING MODULE
#
# NOTE:
# reset_parameters() from nn.Linear is NOT accepted as final initialization.
#
# The module is instantiated only to audit parameter topology.
#
# Final global Kaiming initialization remains deferred.
# =============================================================================

class ITRSScoringMLP(nn.Module):

    def __init__(self):

        super().__init__()


        self.hidden_1 = nn.Linear(
            INPUT_DIM,
            HIDDEN_WIDTHS[0],
            bias=HIDDEN_BIAS,
        )


        self.hidden_2 = nn.Linear(
            HIDDEN_WIDTHS[0],
            HIDDEN_WIDTHS[1],
            bias=HIDDEN_BIAS,
        )


        self.hidden_3 = nn.Linear(
            HIDDEN_WIDTHS[1],
            HIDDEN_WIDTHS[2],
            bias=HIDDEN_BIAS,
        )


        self.hidden_4 = nn.Linear(
            HIDDEN_WIDTHS[2],
            HIDDEN_WIDTHS[3],
            bias=HIDDEN_BIAS,
        )


        self.output = nn.Linear(
            HIDDEN_WIDTHS[3],
            OUTPUT_DIM,
            bias=OUTPUT_BIAS,
        )


        self.activation = nn.ReLU()


    def forward_logits(
        self,
        pair_features,
    ):

        x = self.activation(
            self.hidden_1(
                pair_features
            )
        )


        x = self.activation(
            self.hidden_2(
                x
            )
        )


        x = self.activation(
            self.hidden_3(
                x
            )
        )


        x = self.activation(
            self.hidden_4(
                x
            )
        )


        logits = self.output(
            x
        )


        return logits


    def forward(
        self,
        pair_features,
    ):

        # Training-facing output.
        #
        # Return logits, NOT probabilities.
        #
        # BCEWithLogitsLoss applies the logistic transform
        # internally in a numerically stable way.

        return self.forward_logits(
            pair_features
        )


    def predict_probability(
        self,
        pair_features,
    ):

        return torch.sigmoid(
            self.forward_logits(
                pair_features
            )
        )


# =============================================================================
# START
# =============================================================================

banner(
    "PHASE 4.5.1b — "
    "FREEZE RECOMMENDATION SCORING NEURAL CONTRACT"
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
# 2. UPSTREAM CONTRACT INTEGRITY
# =============================================================================

banner(
    "UPSTREAM SCORING INPUT CONTRACT"
)


input_contract = load_json(
    SCORING_INPUT_CONTRACT_PATH
)


representation = pd.read_csv(
    SCORING_REPRESENTATION_PATH
)


paper_audit = pd.read_csv(
    SCORING_PAPER_AUDIT_PATH
)


require(
    input_contract.get(
        "status"
    )
    == "FROZEN_INPUT_CONTRACT",
    (
        "Phase 4.5.1a scoring input "
        "contract is not frozen."
    ),
)


require(
    input_contract[
        "pair_representation"
    ][
        "dimension"
    ]
    == INPUT_DIM,
    "Frozen scoring input dimension changed.",
)


require(
    input_contract[
        "investor_representation"
    ][
        "dimension"
    ]
    == 160,
    "Investor scoring dimension changed.",
)


require(
    input_contract[
        "startup_representation"
    ][
        "dimension"
    ]
    == 120,
    "Startup scoring dimension changed.",
)


require(
    input_contract[
        "paper_specified_scoring"
    ][
        "hidden_layer_count"
    ]
    == HIDDEN_LAYER_COUNT,
    "Paper-specified hidden-layer count changed.",
)


require(
    input_contract[
        "paper_specified_scoring"
    ][
        "hidden_activation"
    ]
    == "ReLU",
    "Paper-specified activation changed.",
)


require(
    input_contract[
        "paper_specified_scoring"
    ][
        "output_dim"
    ]
    == OUTPUT_DIM,
    "Paper scoring output dimension changed.",
)


require(
    len(
        representation
    )
    == 7,
    "Scoring feature component table changed.",
)


print(
    "Phase 4.5.1a input contract: PASS"
)

print(
    "Pair input dimension:          280"
)

print(
    "Hidden-layer count:            4"
)

print(
    "Hidden activation:             ReLU"
)

print(
    "Output dimension:              1"
)


# =============================================================================
# 3. EXACT FEATURE ORDER INTEGRITY
# =============================================================================

banner(
    "SCORING FEATURE ORDER INTEGRITY"
)


expected_component_order = [
    "F_t",
    "L_o",
    "F_d,o",
    "F_s,o",
    "L_b",
    "F_d,b",
    "F_s,b",
]


actual_component_order = (
    representation[
        "paper_symbol"
    ]
    .astype(str)
    .tolist()
)


print(
    f"Expected: "
    f"{expected_component_order}"
)

print(
    f"Actual:   "
    f"{actual_component_order}"
)


require(
    actual_component_order
    == expected_component_order,
    "Scoring pair feature order changed.",
)


print()
print(
    "Feature order: PASS"
)


# =============================================================================
# 4. HIDDEN-WIDTH REPRODUCTION CHOICE
# =============================================================================

banner(
    "HIDDEN-WIDTH CONTRACT"
)


print(
    "ITRS paper:"
)

print(
    "  hidden layers = 4"
)

print(
    "  exact widths  = NOT REPORTED"
)


print()
print(
    "Frozen reproduction choice:"
)

print(
    f"  {HIDDEN_WIDTHS}"
)


print()
print(
    "Architecture:"
)

print(
    "  280 -> 128 -> 64 -> 32 -> 16 -> 1"
)


print()
print(
    "Classification:"
)

print(
    "  PAPER_UNSPECIFIED_REPRODUCTION_CHOICE_GUIDED_BY_NCF"
)


print()
print(
    "Rationale:"
)

print(
    "  - ITRS explicitly uses an NCF-style scorer."
)

print(
    "  - canonical NCF uses a progressively shrinking MLP."
)

print(
    "  - ITRS requires four hidden layers."
)

print(
    "  - 128 is a practical approximately-half compression"
)

print(
    "    from the frozen 280-D ITRS pair representation."
)

print(
    "  - subsequent widths halve monotonically."
)


require(
    len(
        HIDDEN_WIDTHS
    )
    == HIDDEN_LAYER_COUNT,
    "Hidden-width count does not equal four.",
)


require(
    all(
        HIDDEN_WIDTHS[
            index
        ]
        >
        HIDDEN_WIDTHS[
            index + 1
        ]

        for index
        in range(
            len(
                HIDDEN_WIDTHS
            )
            - 1
        )
    ),
    "Scoring widths are not monotonically decreasing.",
)


# =============================================================================
# 5. BIAS CONTRACT
# =============================================================================

banner(
    "LINEAR BIAS CONTRACT"
)


print(
    f"Hidden layer biases: "
    f"{HIDDEN_BIAS}"
)

print(
    f"Output layer bias:   "
    f"{OUTPUT_BIAS}"
)


print()
print(
    "Classification:"
)

print(
    "  PAPER_UNSPECIFIED_REPRODUCTION_CHOICE_GUIDED_BY_STANDARD_NCF_MLP"
)


print()
print(
    "No extra standalone bias outside nn.Linear."
)


# =============================================================================
# 6. OUTPUT / LOSS IMPLEMENTATION CONTRACT
# =============================================================================

banner(
    "OUTPUT AND BCE CONTRACT"
)


print(
    "Paper mathematical semantics:"
)

print(
    "  logistic probability + BCE"
)


print()
print(
    "Frozen training implementation:"
)

print(
    "  scorer returns raw logit"
)

print(
    "  criterion = BCEWithLogitsLoss"
)


print()
print(
    "Prediction / ranking probability:"
)

print(
    "  sigmoid(logit)"
)


print()
print(
    "Classification:"
)

print(
    "  MATHEMATICALLY_EQUIVALENT_NUMERICALLY_STABLE_IMPLEMENTATION"
)


# =============================================================================
# 7. REGULARIZATION / EXTRA-LAYER CONTRACT
# =============================================================================

banner(
    "SCORING REGULARIZATION / EXTRAS"
)


print(
    f"Dropout:                 "
    f"{DROPOUT}"
)

print(
    f"BatchNorm:               "
    f"{USE_BATCH_NORM}"
)

print(
    f"LayerNorm:               "
    f"{USE_LAYER_NORM}"
)

print(
    f"Residual connections:    "
    f"{USE_RESIDUAL}"
)


print()
print(
    "Classification:"
)

print(
    "  PAPER_UNSPECIFIED_NO_ADDED_COMPONENT"
)


# =============================================================================
# 8. MODULE INSTANTIATION
# =============================================================================

banner(
    "SCORING MODULE INSTANTIATION"
)


module = ITRSScoringMLP()


print(
    module
)


# =============================================================================
# 9. EXACT MODULE TOPOLOGY
# =============================================================================

banner(
    "LINEAR LAYER TOPOLOGY"
)


layer_contract = [

    (
        "hidden_1",
        module.hidden_1,
        INPUT_DIM,
        128,
    ),

    (
        "hidden_2",
        module.hidden_2,
        128,
        64,
    ),

    (
        "hidden_3",
        module.hidden_3,
        64,
        32,
    ),

    (
        "hidden_4",
        module.hidden_4,
        32,
        16,
    ),

    (
        "output",
        module.output,
        16,
        1,
    ),
]


for (
    name,
    layer,
    expected_in,
    expected_out,
) in layer_contract:

    print(
        f"{name:<12} "
        f"{layer.in_features:>3} -> "
        f"{layer.out_features:<3} "
        f"bias={layer.bias is not None}"
    )


    require(
        layer.in_features
        == expected_in,
        f"{name} input width mismatch.",
    )


    require(
        layer.out_features
        == expected_out,
        f"{name} output width mismatch.",
    )


# =============================================================================
# 10. PARAMETER SHAPE AUDIT
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
        f"{name:<30} "
        f"{parameter_shapes[name]}"
    )


expected_shapes = {

    "hidden_1.weight":
        (
            128,
            280,
        ),

    "hidden_1.bias":
        (
            128,
        ),

    "hidden_2.weight":
        (
            64,
            128,
        ),

    "hidden_2.bias":
        (
            64,
        ),

    "hidden_3.weight":
        (
            32,
            64,
        ),

    "hidden_3.bias":
        (
            32,
        ),

    "hidden_4.weight":
        (
            16,
            32,
        ),

    "hidden_4.bias":
        (
            16,
        ),

    "output.weight":
        (
            1,
            16,
        ),

    "output.bias":
        (
            1,
        ),
}


require(
    parameter_shapes
    == expected_shapes,
    "Scoring parameter shapes changed.",
)


print()
print(
    "Exact parameter shapes: PASS"
)


# =============================================================================
# 11. PARAMETER COUNT AUDIT
# =============================================================================

banner(
    "PARAMETER COUNT AUDIT"
)


parameter_counts = {

    "hidden_1":
        (
            module.hidden_1.weight.numel()
            +
            module.hidden_1.bias.numel()
        ),

    "hidden_2":
        (
            module.hidden_2.weight.numel()
            +
            module.hidden_2.bias.numel()
        ),

    "hidden_3":
        (
            module.hidden_3.weight.numel()
            +
            module.hidden_3.bias.numel()
        ),

    "hidden_4":
        (
            module.hidden_4.weight.numel()
            +
            module.hidden_4.bias.numel()
        ),

    "output":
        (
            module.output.weight.numel()
            +
            module.output.bias.numel()
        ),
}


for (
    name,
    count,
) in parameter_counts.items():

    print(
        f"{name:<12} "
        f"{count:>8,}"
    )


total_parameters = sum(
    parameter_counts.values()
)


print()
print(
    f"TOTAL SCORING PARAMETERS: "
    f"{total_parameters:,}"
)


require(
    parameter_counts[
        "hidden_1"
    ]
    == EXPECTED_LAYER_1_PARAMETERS,
    "Hidden-1 parameter count mismatch.",
)


require(
    parameter_counts[
        "hidden_2"
    ]
    == EXPECTED_LAYER_2_PARAMETERS,
    "Hidden-2 parameter count mismatch.",
)


require(
    parameter_counts[
        "hidden_3"
    ]
    == EXPECTED_LAYER_3_PARAMETERS,
    "Hidden-3 parameter count mismatch.",
)


require(
    parameter_counts[
        "hidden_4"
    ]
    == EXPECTED_LAYER_4_PARAMETERS,
    "Hidden-4 parameter count mismatch.",
)


require(
    parameter_counts[
        "output"
    ]
    == EXPECTED_OUTPUT_PARAMETERS,
    "Output parameter count mismatch.",
)


require(
    total_parameters
    == EXPECTED_TOTAL_PARAMETERS,
    "Total scoring parameter count mismatch.",
)


# =============================================================================
# 12. TEMPORARY FORWARD SHAPE AUDIT
#
# No final initialization is being tested here.
#
# We only verify topology.
# =============================================================================

banner(
    "FORWARD SHAPE SMOKE AUDIT"
)


batch_size = 7


audit_input = torch.zeros(
    batch_size,
    INPUT_DIM,
    dtype=torch.float32,
)


logits = module(
    audit_input
)


probabilities = (
    module.predict_probability(
        audit_input
    )
)


print(
    f"Input shape:        "
    f"{tuple(audit_input.shape)}"
)

print(
    f"Logit shape:        "
    f"{tuple(logits.shape)}"
)

print(
    f"Probability shape:  "
    f"{tuple(probabilities.shape)}"
)

print(
    f"Probabilities finite:"
    f" {bool(torch.isfinite(probabilities).all())}"
)

print(
    f"Probabilities [0,1]:"
    f" {bool(torch.all((probabilities >= 0) & (probabilities <= 1)))}"
)


require(
    tuple(
        logits.shape
    )
    == (
        batch_size,
        1,
    ),
    "Scoring logit shape mismatch.",
)


require(
    tuple(
        probabilities.shape
    )
    == (
        batch_size,
        1,
    ),
    "Scoring probability shape mismatch.",
)


require(
    bool(
        torch.isfinite(
            probabilities
        ).all()
    ),
    "Probability output contains non-finite values.",
)


require(
    bool(
        torch.all(
            (
                probabilities
                >= 0
            )
            &
            (
                probabilities
                <= 1
            )
        )
    ),
    "Probability output outside [0,1].",
)


# =============================================================================
# 13. BCE IMPLEMENTATION OBJECT
# =============================================================================

banner(
    "BCE IMPLEMENTATION"
)


criterion = nn.BCEWithLogitsLoss()


print(
    criterion
)


audit_targets = torch.tensor(
    [
        [1.0],
        [0.0],
        [1.0],
        [0.0],
        [1.0],
        [0.0],
        [1.0],
    ],
    dtype=torch.float32,
)


audit_loss = criterion(
    logits,
    audit_targets,
)


print(
    f"Audit loss scalar: "
    f"{audit_loss.ndim == 0}"
)

print(
    f"Audit loss finite: "
    f"{bool(torch.isfinite(audit_loss))}"
)


require(
    audit_loss.ndim == 0,
    "BCE loss must be scalar.",
)


require(
    bool(
        torch.isfinite(
            audit_loss
        )
    ),
    "BCE audit loss is non-finite.",
)


# =============================================================================
# 14. ITEMS STILL NOT FROZEN
# =============================================================================

banner(
    "ITEMS STILL NOT FROZEN"
)


still_not_frozen = [

    "exact global Kaiming initialization variant",

    "global model seed policy",

    "training negative:positive ratio",

    "training negative candidate eligibility",

    "training historical negative exclusion policy",

    "neural training epoch count",

    "early-stopping rule",

    "weight decay",

    "evaluation candidate sampling runtime contract",
]


for item in still_not_frozen:

    print(
        f"  - {item}"
    )


print()
print(
    "None of these decisions are changed "
    "by Phase 4.5.1b."
)


# =============================================================================
# 15. DECISION CLASSIFICATION
# =============================================================================

decision_records = [

    {
        "decision":
            "pair input dimension",

        "value":
            "280",

        "classification":
            "PAPER_SPECIFIED_DIMENSIONAL_CONSEQUENCE",
    },

    {
        "decision":
            "hidden-layer count",

        "value":
            "4",

        "classification":
            "PAPER_SPECIFIED",
    },

    {
        "decision":
            "hidden widths",

        "value":
            "[128,64,32,16]",

        "classification":
            (
                "PAPER_UNSPECIFIED_"
                "REPRODUCTION_CHOICE_"
                "GUIDED_BY_NCF"
            ),
    },

    {
        "decision":
            "hidden activation",

        "value":
            "ReLU",

        "classification":
            "PAPER_SPECIFIED",
    },

    {
        "decision":
            "hidden-layer bias",

        "value":
            "True",

        "classification":
            (
                "PAPER_UNSPECIFIED_"
                "REPRODUCTION_CHOICE_"
                "GUIDED_BY_STANDARD_NCF_MLP"
            ),
    },

    {
        "decision":
            "output bias",

        "value":
            "True",

        "classification":
            (
                "PAPER_UNSPECIFIED_"
                "REPRODUCTION_CHOICE_"
                "GUIDED_BY_STANDARD_NCF_MLP"
            ),
    },

    {
        "decision":
            "training output",

        "value":
            "raw logits",

        "classification":
            (
                "MATHEMATICALLY_EQUIVALENT_"
                "NUMERICALLY_STABLE_IMPLEMENTATION"
            ),
    },

    {
        "decision":
            "training loss",

        "value":
            "BCEWithLogitsLoss",

        "classification":
            (
                "MATHEMATICALLY_EQUIVALENT_"
                "NUMERICALLY_STABLE_IMPLEMENTATION"
            ),
    },

    {
        "decision":
            "reported probability",

        "value":
            "sigmoid(logit)",

        "classification":
            "PAPER_SPECIFIED_SEMANTICS",
    },

    {
        "decision":
            "dropout",

        "value":
            "0.0",

        "classification":
            "PAPER_UNSPECIFIED_NO_ADDED_COMPONENT",
    },

    {
        "decision":
            "batch normalization",

        "value":
            "none",

        "classification":
            "PAPER_UNSPECIFIED_NO_ADDED_COMPONENT",
    },

    {
        "decision":
            "layer normalization",

        "value":
            "none",

        "classification":
            "PAPER_UNSPECIFIED_NO_ADDED_COMPONENT",
    },

    {
        "decision":
            "residual connections",

        "value":
            "none",

        "classification":
            "PAPER_UNSPECIFIED_NO_ADDED_COMPONENT",
    },

    {
        "decision":
            "exact Kaiming variant",

        "value":
            "NOT_YET_FROZEN",

        "classification":
            (
                "KAIMING_FAMILY_SPECIFIED_"
                "VARIANT_UNSPECIFIED"
            ),
    },

    {
        "decision":
            "training negative sampling",

        "value":
            "NOT_YET_FROZEN",

        "classification":
            "PHASE2_DEFERRED",
    },
]


decision_df = pd.DataFrame(
    decision_records
)


decision_path = (
    OUT_DIR
    / "scoring_neural_decision_audit.csv"
)


decision_df.to_csv(
    decision_path,
    index=False,
)


# =============================================================================
# 16. PARAMETER AUDIT
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
    / "scoring_neural_parameter_audit.csv"
)


parameter_df.to_csv(
    parameter_path,
    index=False,
)


# =============================================================================
# 17. FREEZE NEURAL CONTRACT
# =============================================================================

banner(
    "FREEZING SCORING NEURAL CONTRACT"
)


contract = {

    "phase":
        "4.5.1b",

    "status":
        "FROZEN",

    "component":
        "ITRS recommendation scoring neural architecture",

    "input":
        {

            "dimension":
                INPUT_DIM,

            "source":
                "Phase 4.5.1a frozen pair representation",
        },

    "architecture":
        {

            "hidden_layer_count":
                HIDDEN_LAYER_COUNT,

            "hidden_widths":
                HIDDEN_WIDTHS,

            "full_dimensions":
                [
                    INPUT_DIM,
                    *HIDDEN_WIDTHS,
                    OUTPUT_DIM,
                ],

            "hidden_activation":
                HIDDEN_ACTIVATION,

            "output_dim":
                OUTPUT_DIM,
        },

    "linear_layers":
        {

            "hidden_bias":
                HIDDEN_BIAS,

            "output_bias":
                OUTPUT_BIAS,

            "additive_external_bias":
                False,
        },

    "regularization":
        {

            "dropout":
                DROPOUT,

            "batch_norm":
                USE_BATCH_NORM,

            "layer_norm":
                USE_LAYER_NORM,

            "residual":
                USE_RESIDUAL,
        },

    "output_and_loss":
        {

            "training_output":
                TRAINING_OUTPUT,

            "training_loss":
                TRAINING_LOSS,

            "probability_transform":
                PROBABILITY_TRANSFORM,

            "paper_semantics":
                "logistic probability + BCE",

            "implementation_equivalence":
                True,
        },

    "parameter_count":
        {

            "hidden_1":
                EXPECTED_LAYER_1_PARAMETERS,

            "hidden_2":
                EXPECTED_LAYER_2_PARAMETERS,

            "hidden_3":
                EXPECTED_LAYER_3_PARAMETERS,

            "hidden_4":
                EXPECTED_LAYER_4_PARAMETERS,

            "output":
                EXPECTED_OUTPUT_PARAMETERS,

            "total":
                EXPECTED_TOTAL_PARAMETERS,
        },

    "hidden_width_rationale":
        {

            "paper_reports_exact_widths":
                False,

            "classification":
                (
                    "PAPER_UNSPECIFIED_"
                    "REPRODUCTION_CHOICE_GUIDED_BY_NCF"
                ),

            "reference_pattern":
                (
                    "Canonical NCF uses a progressively "
                    "shrinking MLP configuration."
                ),

            "adaptation":
                (
                    "With frozen 280-D ITRS pair input "
                    "and four paper-specified hidden "
                    "layers, use 128 -> 64 -> 32 -> 16."
                ),
        },

    "initialization":
        {

            "paper_specified_family":
                "Kaiming",

            "exact_variant":
                "NOT_YET_FROZEN",

            "current_module_default_initialization":
                "AUDIT_ONLY",

            "model_state_saved":
                False,
        },

    "negative_sampling":
        {

            "changed_in_phase_4_5_1b":
                False,

            "status":
                "NOT_YET_FROZEN",

            "phase_2_deferral_preserved":
                True,
        },

    "training_performed":
        False,

    "forward_scoring_semantics_fully_audited":
        False,

    "next_phase":
        {

            "phase":
                "4.5.2",

            "purpose":
                (
                    "Scoring forward, sigmoid/BCE, "
                    "feature-order and autograd audit."
                ),
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

            "phase_4_5_1a":
                False,
        },
}


contract_path = (
    OUT_DIR
    / "scoring_neural_contract.json"
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
    "PHASE 4.5.1b FINAL SUMMARY"
)


print(
    "Scoring architecture:"
)

print(
    "  280 -> 128 -> 64 -> 32 -> 16 -> 1"
)


print()
print(
    "Hidden layers:"
)

print(
    "  count                         4"
)

print(
    "  widths                        [128, 64, 32, 16]"
)

print(
    "  activation                    ReLU"
)

print(
    "  biases                        YES"
)


print()
print(
    "Output:"
)

print(
    "  Linear(16,1,bias=True)"
)

print(
    "  training output                logit"
)

print(
    "  training loss                  BCEWithLogitsLoss"
)

print(
    "  probability                    sigmoid(logit)"
)


print()
print(
    "Extras:"
)

print(
    "  dropout                        0.0"
)

print(
    "  normalization                  NONE"
)

print(
    "  residual                       NONE"
)


print()
print(
    "Parameter counts:"
)

print(
    f"  hidden 1                      "
    f"{EXPECTED_LAYER_1_PARAMETERS:,}"
)

print(
    f"  hidden 2                      "
    f"{EXPECTED_LAYER_2_PARAMETERS:,}"
)

print(
    f"  hidden 3                      "
    f"{EXPECTED_LAYER_3_PARAMETERS:,}"
)

print(
    f"  hidden 4                      "
    f"{EXPECTED_LAYER_4_PARAMETERS:,}"
)

print(
    f"  output                        "
    f"{EXPECTED_OUTPUT_PARAMETERS:,}"
)

print(
    f"  TOTAL                         "
    f"{EXPECTED_TOTAL_PARAMETERS:,}"
)


print()
print(
    "Width classification:"
)

print(
    "  PAPER_UNSPECIFIED_REPRODUCTION_CHOICE_GUIDED_BY_NCF"
)


print()
print(
    "Exact Kaiming variant frozen:   NO"
)

print(
    "Negative sampling frozen:       NO"
)

print(
    "Training performed:             NO"
)

print(
    "Model state persisted:          NO"
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
    "PHASE 4.5.1b STATUS: COMPLETE — "
    "SCORING NEURAL CONTRACT FROZEN"
)