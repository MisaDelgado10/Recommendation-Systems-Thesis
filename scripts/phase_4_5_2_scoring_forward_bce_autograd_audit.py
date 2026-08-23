from pathlib import Path
import hashlib
import json
import sys

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# PHASE 4.5.2 — SCORING FORWARD, BCE, AUTOGRAD, AND PHASE-4.5 CLOSURE
#
# PURPOSE
# -------
# Verify the complete standalone recommendation-scoring contract after:
#
#   Phase 4.5.1a:
#       scoring input / paper contract
#
#   Phase 4.5.1b:
#       scoring neural architecture
#
# This audit verifies:
#
#   1. Exact feature concatenation:
#
#        [ F_t
#          || L_o
#          || F_d,o
#          || F_s,o
#          || L_b
#          || F_d,b
#          || F_s,b ]
#
#   2. Exact input dimension = 280
#
#   3. Forward topology:
#
#        280 -> 128 -> 64 -> 32 -> 16 -> 1
#
#   4. ReLU after every hidden layer.
#
#   5. Raw logit training output.
#
#   6. Probability:
#
#        sigmoid(logit)
#
#   7. BCEWithLogitsLoss equivalence to logistic BCE.
#
#   8. Ranking equivalence between logits and sigmoid probabilities.
#
#   9. Feature-order sensitivity.
#
#  10. Autograd reaches:
#
#        - all seven input feature blocks,
#        - all scoring MLP weights,
#        - all scoring MLP biases.
#
#  11. Batch-size 512 forward behavior.
#
# IMPORTANT
# ---------
# NO TRAINING OCCURS.
#
# NO negative-sampling choice is made.
#
# NO final Kaiming initialization is frozen.
#
# Audit-only deterministic feature values and audit-only deterministic MLP
# weights are used and are NOT persisted.
#
# CHANGE FROM THE FIRST 4.5.2 AUDIT ATTEMPT
# -----------------------------------------
#
# The first audit-only initialization used very small periodic weights.
# Although the scorer architecture itself was correct, swapping F_t and L_o
# produced an identical final float32 logit under that particular synthetic
# parameterization.
#
# That was an AUDIT INITIALIZATION degeneracy, not a model-contract failure.
#
# This version uses:
#
#   - non-periodic,
#   - explicitly column-position-sensitive,
#   - positive,
#   - larger-magnitude
#
# temporary audit weights.
#
# The feature-order audit now checks BOTH:
#
#   1. hidden-1 preactivation sensitivity
#   2. final-logit sensitivity
#
# No frozen Phase-4.5.1a / 4.5.1b decision is changed.
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

SCORING_NEURAL_CONTRACT_PATH = Path(
    "data/experimental/phase_4/"
    "scoring_neural_contract/"
    "scoring_neural_contract.json"
)

SCORING_PARAMETER_AUDIT_PATH = Path(
    "data/experimental/phase_4/"
    "scoring_neural_contract/"
    "scoring_neural_parameter_audit.csv"
)

SCORING_DECISION_AUDIT_PATH = Path(
    "data/experimental/phase_4/"
    "scoring_neural_contract/"
    "scoring_neural_decision_audit.csv"
)


# =============================================================================
# OUTPUTS
# =============================================================================

OUT_DIR = Path(
    "data/experimental/phase_4/"
    "scoring_module"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# FROZEN DIMENSIONS
# =============================================================================

FEATURE_DIM = 40

INVESTOR_DIM = 160
STARTUP_DIM = 120

PAIR_DIM = 280

HIDDEN_WIDTHS = [
    128,
    64,
    32,
    16,
]

OUTPUT_DIM = 1

EXPECTED_PARAMETER_COUNT = 46_849

PAPER_BATCH_SIZE = 512


# =============================================================================
# NUMERICAL TOLERANCES
# =============================================================================

ATOL = 1e-6
RTOL = 1e-6

FEATURE_ORDER_HIDDEN_THRESHOLD = 1e-7
FEATURE_ORDER_LOGIT_THRESHOLD = 1e-8


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


# =============================================================================
# SCORING MODULE
# =============================================================================

class ITRSScoringMLP(nn.Module):

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


    def forward_with_intermediates(
        self,
        pair_features,
    ):

        z1 = self.hidden_1(
            pair_features
        )

        h1 = F.relu(
            z1
        )


        z2 = self.hidden_2(
            h1
        )

        h2 = F.relu(
            z2
        )


        z3 = self.hidden_3(
            h2
        )

        h3 = F.relu(
            z3
        )


        z4 = self.hidden_4(
            h3
        )

        h4 = F.relu(
            z4
        )


        logits = self.output(
            h4
        )


        return {
            "z1":
                z1,

            "h1":
                h1,

            "z2":
                z2,

            "h2":
                h2,

            "z3":
                z3,

            "h3":
                h3,

            "z4":
                z4,

            "h4":
                h4,

            "logits":
                logits,
        }


    def forward(
        self,
        pair_features,
    ):

        return (
            self.forward_with_intermediates(
                pair_features
            )[
                "logits"
            ]
        )


    def predict_probability(
        self,
        pair_features,
    ):

        return torch.sigmoid(
            self.forward(
                pair_features
            )
        )


# =============================================================================
# UPDATED AUDIT-ONLY DETERMINISTIC INITIALIZATION
#
# IMPORTANT
# ---------
# This is NOT Kaiming.
#
# It is used ONLY for forward/autograd auditing.
#
# Design goals:
#
#   1. Explicit column-position sensitivity.
#
#      The value assigned to an input coordinate changes smoothly with its
#      absolute column position. Therefore swapping the 0:40 block with the
#      40:80 block changes hidden-1 preactivations.
#
#   2. Non-periodic structure.
#
#      The old modulo pattern could create accidental symmetries.
#      This version has no modulo / repeating column pattern.
#
#   3. Positive weights and biases.
#
#      This keeps hidden activations away from a completely dead-ReLU
#      configuration during the autograd audit.
#
#   4. Sufficient magnitude.
#
#      The first version's tiny weights caused positional differences to
#      shrink through four hidden layers until they disappeared at float32
#      precision.
#
# No state is saved.
# =============================================================================

def initialize_audit_linear(
    layer,
    layer_number,
):

    # -------------------------------------------------------------------------
    # Normalized row coordinate:
    #
    #   [0, 1]
    #
    # Shape:
    #
    #   [out_features, 1]
    # -------------------------------------------------------------------------

    if layer.out_features > 1:

        row_coordinate = torch.linspace(
            0.0,
            1.0,
            steps=layer.out_features,
            dtype=torch.float32,
        ).unsqueeze(
            1
        )

    else:

        row_coordinate = torch.zeros(
            1,
            1,
            dtype=torch.float32,
        )


    # -------------------------------------------------------------------------
    # Normalized ABSOLUTE column coordinate:
    #
    #   [0, 1]
    #
    # Shape:
    #
    #   [1, in_features]
    #
    # This is the key position-sensitive term.
    # -------------------------------------------------------------------------

    if layer.in_features > 1:

        column_coordinate = torch.linspace(
            0.0,
            1.0,
            steps=layer.in_features,
            dtype=torch.float32,
        ).unsqueeze(
            0
        )

    else:

        column_coordinate = torch.zeros(
            1,
            1,
            dtype=torch.float32,
        )


    # -------------------------------------------------------------------------
    # Layer-specific positive base.
    # -------------------------------------------------------------------------

    base = (
        0.0040
        +
        0.0004
        * float(
            layer_number
        )
    )


    # -------------------------------------------------------------------------
    # Non-periodic position-sensitive matrix.
    #
    # All terms are positive.
    #
    # Approximate range:
    #
    #   ~0.0044 .. ~0.010+
    #
    # depending on layer and coordinates.
    # -------------------------------------------------------------------------

    weights = (
        base

        +
        0.0030
        * column_coordinate

        +
        0.0015
        * row_coordinate

        +
        0.0010
        * (
            row_coordinate
            * column_coordinate
        )
    )


    # -------------------------------------------------------------------------
    # Small monotonic positive bias.
    # -------------------------------------------------------------------------

    if layer.out_features > 1:

        bias_coordinate = torch.linspace(
            0.0,
            1.0,
            steps=layer.out_features,
            dtype=torch.float32,
        )

    else:

        bias_coordinate = torch.zeros(
            1,
            dtype=torch.float32,
        )


    biases = (
        0.020
        +
        0.001
        * float(
            layer_number
        )
        +
        0.002
        * bias_coordinate
    )


    require(
        weights.shape
        == layer.weight.shape,
        "Audit weight shape mismatch.",
    )


    require(
        biases.shape
        == layer.bias.shape,
        "Audit bias shape mismatch.",
    )


    require(
        bool(
            torch.all(
                weights > 0
            )
        ),
        "Audit weights must remain positive.",
    )


    require(
        bool(
            torch.all(
                biases > 0
            )
        ),
        "Audit biases must remain positive.",
    )


    with torch.no_grad():

        layer.weight.copy_(
            weights
        )

        layer.bias.copy_(
            biases
        )


# =============================================================================
# AUDIT FEATURE GENERATOR
#
# Each semantic feature block receives a distinct deterministic numerical
# signature.
#
# This makes it possible to verify:
#
#   - exact block ordering,
#   - slice recovery,
#   - permutation sensitivity.
# =============================================================================

def make_feature_block(
    batch_size,
    block_id,
    requires_grad=False,
):

    row_axis = torch.arange(
        batch_size,
        dtype=torch.float32,
    ).unsqueeze(
        1
    )


    dim_axis = torch.arange(
        FEATURE_DIM,
        dtype=torch.float32,
    ).unsqueeze(
        0
    )


    values = (
        0.10
        +
        block_id * 0.035
        +
        row_axis * 0.0007
        +
        dim_axis * 0.0003
        +
        0.002
        * torch.sin(
            row_axis * 0.37
            + dim_axis * 0.19
            + block_id
        )
    )


    values = values.clone().detach()


    values.requires_grad_(
        requires_grad
    )


    return values


# =============================================================================
# PAIR CONCATENATION
#
# Frozen exact order:
#
#   0 F_t
#   1 L_o
#   2 F_d,o
#   3 F_s,o
#   4 L_b
#   5 F_d,b
#   6 F_s,b
# =============================================================================

def concatenate_pair_features(
    trend,
    investor_latent,
    investor_description,
    investor_structural,
    startup_latent,
    startup_description,
    startup_structural,
):

    return torch.cat(
        [
            trend,
            investor_latent,
            investor_description,
            investor_structural,
            startup_latent,
            startup_description,
            startup_structural,
        ],
        dim=1,
    )


# =============================================================================
# START
# =============================================================================

banner(
    "PHASE 4.5.2 — "
    "SCORING FORWARD, BCE, AUTOGRAD, "
    "AND PHASE-4.5 CLOSURE"
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
    "UPSTREAM SCORING CONTRACT INTEGRITY"
)


input_contract = load_json(
    SCORING_INPUT_CONTRACT_PATH
)


neural_contract = load_json(
    SCORING_NEURAL_CONTRACT_PATH
)


representation_contract = pd.read_csv(
    SCORING_REPRESENTATION_PATH
)


parameter_audit = pd.read_csv(
    SCORING_PARAMETER_AUDIT_PATH
)


decision_audit = pd.read_csv(
    SCORING_DECISION_AUDIT_PATH
)


require(
    input_contract.get(
        "status"
    )
    == "FROZEN_INPUT_CONTRACT",
    "Phase 4.5.1a input contract is not frozen.",
)


require(
    neural_contract.get(
        "status"
    )
    == "FROZEN",
    "Phase 4.5.1b neural contract is not frozen.",
)


require(
    input_contract[
        "pair_representation"
    ][
        "dimension"
    ]
    == PAIR_DIM,
    "Frozen pair dimension changed.",
)


require(
    neural_contract[
        "architecture"
    ][
        "hidden_widths"
    ]
    == HIDDEN_WIDTHS,
    "Frozen scoring hidden widths changed.",
)


require(
    neural_contract[
        "parameter_count"
    ][
        "total"
    ]
    == EXPECTED_PARAMETER_COUNT,
    "Frozen scoring parameter count changed.",
)


require(
    len(
        representation_contract
    )
    == 7,
    "Scoring representation table changed.",
)


require(
    len(
        parameter_audit
    )
    == 10,
    "Scoring parameter audit changed.",
)


print(
    "Phase 4.5.1a input contract: PASS"
)

print(
    "Phase 4.5.1b neural contract: PASS"
)

print(
    "Pair dimension:                  280"
)

print(
    "Architecture:                    "
    "280 -> 128 -> 64 -> 32 -> 16 -> 1"
)

print(
    f"Scoring parameters:              "
    f"{EXPECTED_PARAMETER_COUNT:,}"
)


# =============================================================================
# 3. TREND INDEXING RECONCILIATION
# =============================================================================

banner(
    "TREND INDEXING RECONCILIATION"
)


print(
    "Frozen operational scoring rule:"
)

print(
    "  target T_h consumes the Phase-4.3 "
    "trend representation"
)

print(
    "  built from T0 ... T(h-1)"
)


print()
print(
    "Additional h-1 shift inside scorer:"
)

print(
    "  NO"
)


print()
print(
    "Phase-4.3 temporal semantics reopened:"
)

print(
    "  NO"
)


trend_indexing_classification = (
    "PAPER_INDEXING_NOTATION_RECONCILIATION"
)


# =============================================================================
# 4. EXACT FEATURE ORDER / SLICE CONTRACT
# =============================================================================

banner(
    "PAIR FEATURE ORDER / SLICE CONTRACT"
)


expected_symbols = [
    "F_t",
    "L_o",
    "F_d,o",
    "F_s,o",
    "L_b",
    "F_d,b",
    "F_s,b",
]


expected_starts = [
    0,
    40,
    80,
    120,
    160,
    200,
    240,
]


expected_ends = [
    40,
    80,
    120,
    160,
    200,
    240,
    280,
]


actual_symbols = (
    representation_contract[
        "paper_symbol"
    ]
    .astype(str)
    .tolist()
)


actual_starts = (
    representation_contract[
        "slice_start"
    ]
    .astype(int)
    .tolist()
)


actual_ends = (
    representation_contract[
        "slice_end_exclusive"
    ]
    .astype(int)
    .tolist()
)


require(
    actual_symbols
    == expected_symbols,
    "Pair feature symbol order changed.",
)


require(
    actual_starts
    == expected_starts,
    "Pair feature start slices changed.",
)


require(
    actual_ends
    == expected_ends,
    "Pair feature end slices changed.",
)


for index in range(
    7
):

    print(
        f"{index}: "
        f"{actual_symbols[index]:<8} "
        f"[{actual_starts[index]}:"
        f"{actual_ends[index]}]"
    )


print()
print(
    "Feature order / slices: PASS"
)


# =============================================================================
# 5. INSTANTIATE SCORER
# =============================================================================

banner(
    "SCORING MODULE"
)


model = ITRSScoringMLP()


initialize_audit_linear(
    model.hidden_1,
    1,
)

initialize_audit_linear(
    model.hidden_2,
    2,
)

initialize_audit_linear(
    model.hidden_3,
    3,
)

initialize_audit_linear(
    model.hidden_4,
    4,
)

initialize_audit_linear(
    model.output,
    5,
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
    == EXPECTED_PARAMETER_COUNT,
    "Scoring parameter count changed.",
)


print()
print(
    "Audit-only deterministic initialization:"
)

print(
    "  non-periodic                  YES"
)

print(
    "  explicit column sensitivity  YES"
)

print(
    "  positive weights             YES"
)

print(
    "  persisted                    NO"
)


print()
print(
    "Final Kaiming variant:         UNFROZEN"
)


# =============================================================================
# 6. BUILD REPRESENTATIVE FEATURE BLOCKS
# =============================================================================

banner(
    "REPRESENTATIVE FEATURE BLOCKS"
)


BATCH_SIZE = 8


F_t = make_feature_block(
    BATCH_SIZE,
    block_id=1,
)


L_o = make_feature_block(
    BATCH_SIZE,
    block_id=2,
)


F_d_o = make_feature_block(
    BATCH_SIZE,
    block_id=3,
)


F_s_o = make_feature_block(
    BATCH_SIZE,
    block_id=4,
)


L_b = make_feature_block(
    BATCH_SIZE,
    block_id=5,
)


F_d_b = make_feature_block(
    BATCH_SIZE,
    block_id=6,
)


F_s_b = make_feature_block(
    BATCH_SIZE,
    block_id=7,
)


feature_blocks = [
    F_t,
    L_o,
    F_d_o,
    F_s_o,
    L_b,
    F_d_b,
    F_s_b,
]


for symbol, block in zip(
    expected_symbols,
    feature_blocks,
):

    print(
        f"{symbol:<8} "
        f"{tuple(block.shape)} "
        f"finite={bool(torch.isfinite(block).all())}"
    )


require(
    all(
        tuple(
            block.shape
        )
        == (
            BATCH_SIZE,
            FEATURE_DIM,
        )

        for block
        in feature_blocks
    ),
    "One or more scoring feature blocks have wrong shape.",
)


# =============================================================================
# 7. CONCATENATION FORWARD CONTRACT
# =============================================================================

banner(
    "PAIR CONCATENATION AUDIT"
)


pair_features = concatenate_pair_features(
    trend=F_t,
    investor_latent=L_o,
    investor_description=F_d_o,
    investor_structural=F_s_o,
    startup_latent=L_b,
    startup_description=F_d_b,
    startup_structural=F_s_b,
)


print(
    f"Pair feature shape: "
    f"{tuple(pair_features.shape)}"
)


require(
    tuple(
        pair_features.shape
    )
    == (
        BATCH_SIZE,
        PAIR_DIM,
    ),
    "Pair feature shape mismatch.",
)


slice_matches = []


for (
    symbol,
    original,
    start,
    end,
) in zip(
    expected_symbols,
    feature_blocks,
    expected_starts,
    expected_ends,
):

    recovered = pair_features[
        :,
        start:end,
    ]


    exact = torch.equal(
        recovered,
        original,
    )


    slice_matches.append(
        exact
    )


    print(
        f"{symbol:<8} "
        f"slice exact: {exact}"
    )


require(
    all(
        slice_matches
    ),
    "Pair concatenation failed exact slice recovery.",
)


print()
print(
    "Exact concatenation roundtrip: PASS"
)


# =============================================================================
# 8. FULL HIDDEN-LAYER FORWARD AUDIT
# =============================================================================

banner(
    "FOUR-HIDDEN-LAYER FORWARD AUDIT"
)


forward = (
    model.forward_with_intermediates(
        pair_features
    )
)


expected_shapes = {

    "z1":
        (
            BATCH_SIZE,
            128,
        ),

    "h1":
        (
            BATCH_SIZE,
            128,
        ),

    "z2":
        (
            BATCH_SIZE,
            64,
        ),

    "h2":
        (
            BATCH_SIZE,
            64,
        ),

    "z3":
        (
            BATCH_SIZE,
            32,
        ),

    "h3":
        (
            BATCH_SIZE,
            32,
        ),

    "z4":
        (
            BATCH_SIZE,
            16,
        ),

    "h4":
        (
            BATCH_SIZE,
            16,
        ),

    "logits":
        (
            BATCH_SIZE,
            1,
        ),
}


for name in [
    "z1",
    "h1",
    "z2",
    "h2",
    "z3",
    "h3",
    "z4",
    "h4",
    "logits",
]:

    tensor = forward[
        name
    ]


    print(
        f"{name:<8} "
        f"shape={tuple(tensor.shape)} "
        f"finite={bool(torch.isfinite(tensor).all())}"
    )


    require(
        tuple(
            tensor.shape
        )
        == expected_shapes[
            name
        ],
        f"{name} shape mismatch.",
    )


    require(
        bool(
            torch.isfinite(
                tensor
            ).all()
        ),
        f"{name} contains non-finite values.",
    )


# =============================================================================
# 9. RELU PLACEMENT AUDIT
# =============================================================================

banner(
    "RELU PLACEMENT AUDIT"
)


relu_checks = {

    "hidden_1":
        torch.equal(
            forward[
                "h1"
            ],
            F.relu(
                forward[
                    "z1"
                ]
            ),
        ),

    "hidden_2":
        torch.equal(
            forward[
                "h2"
            ],
            F.relu(
                forward[
                    "z2"
                ]
            ),
        ),

    "hidden_3":
        torch.equal(
            forward[
                "h3"
            ],
            F.relu(
                forward[
                    "z3"
                ]
            ),
        ),

    "hidden_4":
        torch.equal(
            forward[
                "h4"
            ],
            F.relu(
                forward[
                    "z4"
                ]
            ),
        ),
}


for name, exact in (
    relu_checks.items()
):

    print(
        f"{name:<10} "
        f"ReLU exact: {exact}"
    )


require(
    all(
        relu_checks.values()
    ),
    "Hidden activation placement changed.",
)


# =============================================================================
# 10. LOGIT / PROBABILITY SEMANTICS
# =============================================================================

banner(
    "LOGIT / PROBABILITY SEMANTICS"
)


logits = forward[
    "logits"
]


probabilities = torch.sigmoid(
    logits
)


model_probabilities = (
    model.predict_probability(
        pair_features
    )
)


probability_exact = torch.equal(
    probabilities,
    model_probabilities,
)


print(
    f"Logit shape:              "
    f"{tuple(logits.shape)}"
)

print(
    f"Probability shape:        "
    f"{tuple(probabilities.shape)}"
)

print(
    f"Probability finite:       "
    f"{bool(torch.isfinite(probabilities).all())}"
)

print(
    f"Probability within [0,1]: "
    f"{bool(torch.all((probabilities >= 0) & (probabilities <= 1)))}"
)

print(
    f"predict_probability exact:"
    f" {probability_exact}"
)


require(
    probability_exact,
    "Probability helper differs from sigmoid(logit).",
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
    "Probability outside [0,1].",
)


# =============================================================================
# 11. BCE EQUIVALENCE AUDIT
# =============================================================================

banner(
    "BCE / BCE-WITH-LOGITS EQUIVALENCE"
)


targets = torch.tensor(
    [
        [1.0],
        [0.0],
        [1.0],
        [1.0],
        [0.0],
        [0.0],
        [1.0],
        [0.0],
    ],
    dtype=torch.float32,
)


criterion = nn.BCEWithLogitsLoss()


loss_logits = criterion(
    logits,
    targets,
)


manual_stable_loss = (
    F.softplus(
        logits
    )
    -
    targets
    * logits
).mean()


probability_bce = F.binary_cross_entropy(
    probabilities,
    targets,
)


stable_equivalence = torch.allclose(
    loss_logits,
    manual_stable_loss,
    atol=ATOL,
    rtol=RTOL,
)


probability_equivalence = torch.allclose(
    loss_logits,
    probability_bce,
    atol=ATOL,
    rtol=RTOL,
)


print(
    f"BCEWithLogitsLoss:        "
    f"{float(loss_logits):.10f}"
)

print(
    f"Manual stable BCE:        "
    f"{float(manual_stable_loss):.10f}"
)

print(
    f"Probability BCE:          "
    f"{float(probability_bce):.10f}"
)


print()
print(
    f"Logits / stable BCE match:"
    f" {stable_equivalence}"
)

print(
    f"Logits / probability BCE: "
    f"{probability_equivalence}"
)


require(
    stable_equivalence,
    (
        "BCEWithLogitsLoss does not match "
        "stable logistic BCE."
    ),
)


require(
    probability_equivalence,
    (
        "Logit BCE implementation does not "
        "match paper probability-BCE semantics."
    ),
)


# =============================================================================
# 12. LOGIT / PROBABILITY RANKING EQUIVALENCE
# =============================================================================

banner(
    "RANKING EQUIVALENCE"
)


logit_order = torch.argsort(
    logits.flatten(),
    descending=True,
)


probability_order = torch.argsort(
    probabilities.flatten(),
    descending=True,
)


ranking_exact = torch.equal(
    logit_order,
    probability_order,
)


print(
    f"Logit rank == probability rank: "
    f"{ranking_exact}"
)


require(
    ranking_exact,
    (
        "Sigmoid unexpectedly changed "
        "candidate ranking order."
    ),
)


# =============================================================================
# 13. FEATURE-ORDER SENSITIVITY — UPDATED
#
# Swap:
#
#   F_t  [0:40]
#   L_o  [40:80]
#
# while leaving every other block unchanged.
#
# We verify sensitivity at TWO levels:
#
#   A. hidden_1 preactivation
#
#      This directly verifies that the first learned transformation sees
#      these blocks at distinct coordinate locations.
#
#   B. final logit
#
#      This verifies that the positional change propagates through all four
#      hidden layers to the prediction.
#
# The original failed audit used tiny periodic audit weights whose difference
# vanished after deep propagation at float32 precision.
#
# The frozen model contract itself has NOT changed.
# =============================================================================

banner(
    "FEATURE-ORDER SENSITIVITY"
)


permuted_pair_features = torch.cat(
    [
        L_o,       # intentionally wrong position
        F_t,       # intentionally wrong position
        F_d_o,
        F_s_o,
        L_b,
        F_d_b,
        F_s_b,
    ],
    dim=1,
)


require(
    tuple(
        permuted_pair_features.shape
    )
    == (
        BATCH_SIZE,
        PAIR_DIM,
    ),
    "Permuted pair input shape changed.",
)


permuted_forward = (
    model.forward_with_intermediates(
        permuted_pair_features
    )
)


hidden_1_difference = float(
    torch.max(
        torch.abs(
            forward[
                "z1"
            ]
            -
            permuted_forward[
                "z1"
            ]
        )
    )
)


final_logit_difference = float(
    torch.max(
        torch.abs(
            forward[
                "logits"
            ]
            -
            permuted_forward[
                "logits"
            ]
        )
    )
)


hidden_1_order_sensitive = (
    hidden_1_difference
    > FEATURE_ORDER_HIDDEN_THRESHOLD
)


final_output_order_sensitive = (
    final_logit_difference
    > FEATURE_ORDER_LOGIT_THRESHOLD
)


print(
    f"Correct input shape:       "
    f"{tuple(pair_features.shape)}"
)

print(
    f"Permuted input shape:      "
    f"{tuple(permuted_pair_features.shape)}"
)


print()
print(
    f"Hidden-1 max difference:   "
    f"{hidden_1_difference:.10f}"
)

print(
    f"Hidden-1 order sensitive:  "
    f"{hidden_1_order_sensitive}"
)


print()
print(
    f"Final logit max difference:"
    f" {final_logit_difference:.10f}"
)

print(
    f"Final output order sensitive:"
    f" {final_output_order_sensitive}"
)


require(
    hidden_1_order_sensitive,
    (
        "Feature permutation did not alter "
        "hidden-1 preactivation."
    ),
)


require(
    final_output_order_sensitive,
    (
        "Feature permutation changed hidden-1 "
        "but did not propagate to final logit."
    ),
)


print()
print(
    "Feature-order sensitivity: PASS"
)


# =============================================================================
# 14. PAPER BATCH-SIZE 512 FORWARD AUDIT
# =============================================================================

banner(
    "BATCH-SIZE 512 FORWARD AUDIT"
)


batch_blocks = [

    make_feature_block(
        PAPER_BATCH_SIZE,
        block_id=block_id,
    )

    for block_id in range(
        1,
        8
    )
]


batch_pair_features = concatenate_pair_features(
    *batch_blocks
)


batch_logits = model(
    batch_pair_features
)


batch_probabilities = torch.sigmoid(
    batch_logits
)


print(
    f"Batch pair shape:        "
    f"{tuple(batch_pair_features.shape)}"
)

print(
    f"Batch logit shape:       "
    f"{tuple(batch_logits.shape)}"
)

print(
    f"Batch probability shape: "
    f"{tuple(batch_probabilities.shape)}"
)

print(
    f"Batch finite:            "
    f"{bool(torch.isfinite(batch_logits).all())}"
)


require(
    tuple(
        batch_pair_features.shape
    )
    == (
        PAPER_BATCH_SIZE,
        280,
    ),
    "Batch-512 pair shape mismatch.",
)


require(
    tuple(
        batch_logits.shape
    )
    == (
        PAPER_BATCH_SIZE,
        1,
    ),
    "Batch-512 logit shape mismatch.",
)


require(
    bool(
        torch.isfinite(
            batch_logits
        ).all()
    ),
    "Batch-512 logits contain non-finite values.",
)


# =============================================================================
# 15. DUPLICATE INPUT DETERMINISM
# =============================================================================

banner(
    "DUPLICATE INPUT DETERMINISM"
)


duplicate_pair = pair_features[
    0:1
].repeat(
    2,
    1,
)


duplicate_logits = model(
    duplicate_pair
)


duplicate_exact = torch.equal(
    duplicate_logits[
        0
    ],
    duplicate_logits[
        1
    ],
)


print(
    f"Duplicate pair logits exact: "
    f"{duplicate_exact}"
)


require(
    duplicate_exact,
    "Identical scoring inputs produced different logits.",
)


# =============================================================================
# 16. AUTOGRAD FEATURE-BLOCK AUDIT
# =============================================================================

banner(
    "AUTOGRAD — SEVEN INPUT FEATURE BLOCKS"
)


GRAD_BATCH = 6


grad_F_t = make_feature_block(
    GRAD_BATCH,
    1,
    requires_grad=True,
)


grad_L_o = make_feature_block(
    GRAD_BATCH,
    2,
    requires_grad=True,
)


grad_F_d_o = make_feature_block(
    GRAD_BATCH,
    3,
    requires_grad=True,
)


grad_F_s_o = make_feature_block(
    GRAD_BATCH,
    4,
    requires_grad=True,
)


grad_L_b = make_feature_block(
    GRAD_BATCH,
    5,
    requires_grad=True,
)


grad_F_d_b = make_feature_block(
    GRAD_BATCH,
    6,
    requires_grad=True,
)


grad_F_s_b = make_feature_block(
    GRAD_BATCH,
    7,
    requires_grad=True,
)


grad_blocks = {

    "F_t":
        grad_F_t,

    "L_o":
        grad_L_o,

    "F_d,o":
        grad_F_d_o,

    "F_s,o":
        grad_F_s_o,

    "L_b":
        grad_L_b,

    "F_d,b":
        grad_F_d_b,

    "F_s,b":
        grad_F_s_b,
}


grad_pair_features = concatenate_pair_features(
    grad_F_t,
    grad_L_o,
    grad_F_d_o,
    grad_F_s_o,
    grad_L_b,
    grad_F_d_b,
    grad_F_s_b,
)


grad_logits = model(
    grad_pair_features
)


grad_targets = torch.tensor(
    [
        [1.0],
        [0.0],
        [1.0],
        [0.0],
        [0.0],
        [1.0],
    ],
    dtype=torch.float32,
)


grad_loss = F.binary_cross_entropy_with_logits(
    grad_logits,
    grad_targets,
)


model.zero_grad(
    set_to_none=True
)


for block in (
    grad_blocks.values()
):

    if block.grad is not None:

        block.grad = None


grad_loss.backward()


input_gradient_records = []


for name, tensor in (
    grad_blocks.items()
):

    exists = (
        tensor.grad
        is not None
    )


    finite = (
        exists
        and bool(
            torch.isfinite(
                tensor.grad
            ).all()
        )
    )


    abs_sum = (
        float(
            tensor.grad
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


    print(
        f"{name:<8} "
        f"exists={str(exists):<5} "
        f"finite={str(finite):<5} "
        f"abs_sum={abs_sum:.10f}"
    )


    require(
        exists,
        f"{name} input gradient missing.",
    )


    require(
        finite,
        f"{name} input gradient non-finite.",
    )


    require(
        nonzero,
        f"{name} input gradient is zero.",
    )


    input_gradient_records.append(
        {
            "input_feature":
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
    )


# =============================================================================
# 17. AUTOGRAD — ALL SCORING PARAMETERS
# =============================================================================

banner(
    "AUTOGRAD — SCORING PARAMETERS"
)


parameter_gradient_records = []


for name, parameter in (
    model.named_parameters()
):

    exists = (
        parameter.grad
        is not None
    )


    finite = (
        exists
        and bool(
            torch.isfinite(
                parameter.grad
            ).all()
        )
    )


    abs_sum = (
        float(
            parameter.grad
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


    print(
        f"{name:<24} "
        f"exists={str(exists):<5} "
        f"finite={str(finite):<5} "
        f"abs_sum={abs_sum:.10f}"
    )


    require(
        exists,
        f"{name} parameter gradient missing.",
    )


    require(
        finite,
        f"{name} parameter gradient non-finite.",
    )


    require(
        nonzero,
        f"{name} parameter gradient is zero.",
    )


    parameter_gradient_records.append(
        {
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
    )


require(
    len(
        parameter_gradient_records
    )
    == 10,
    "Expected gradients for 10 scoring parameters.",
)


print()
print(
    "All seven feature-block gradients: PASS"
)

print(
    "All ten scoring-parameter gradients: PASS"
)


# =============================================================================
# 18. NEGATIVE-SAMPLING BOUNDARY REMAINS OPEN
# =============================================================================

banner(
    "NEGATIVE-SAMPLING BOUNDARY"
)


print(
    "Scorer accepts:"
)

print(
    "  positive pair -> y=1"
)

print(
    "  negative pair -> y=0"
)


print()
print(
    "Training negative ratio:"
)

print(
    "  NOT FROZEN"
)


print()
print(
    "Negative candidate eligibility:"
)

print(
    "  NOT FROZEN"
)


print()
print(
    "Historical exclusion rule:"
)

print(
    "  NOT FROZEN"
)


print()
print(
    "Phase-2 deferral preserved:"
)

print(
    "  YES"
)


# =============================================================================
# 19. ITEMS STILL OPEN AFTER PHASE 4.5
# =============================================================================

banner(
    "OPEN GLOBAL / TRAINING DECISIONS"
)


still_open = [

    "exact global Kaiming initialization variant",

    "global neural seed policy",

    "training negative:positive ratio",

    "training negative candidate eligibility",

    "training negative historical exclusion",

    "neural training epoch count",

    "early-stopping rule",

    "weight decay",

    "evaluation candidate-generation runtime contract",
]


for item in still_open:

    print(
        f"  - {item}"
    )


# =============================================================================
# 20. SAVE INPUT-GRADIENT AUDIT
# =============================================================================

input_gradient_df = pd.DataFrame(
    input_gradient_records
)


input_gradient_path = (
    OUT_DIR
    / "scoring_input_feature_autograd_audit.csv"
)


input_gradient_df.to_csv(
    input_gradient_path,
    index=False,
)


# =============================================================================
# 21. SAVE PARAMETER-GRADIENT AUDIT
# =============================================================================

parameter_gradient_df = pd.DataFrame(
    parameter_gradient_records
)


parameter_gradient_path = (
    OUT_DIR
    / "scoring_parameter_autograd_audit.csv"
)


parameter_gradient_df.to_csv(
    parameter_gradient_path,
    index=False,
)


# =============================================================================
# 22. SAVE FORWARD AUDIT
# =============================================================================

forward_records = [

    {
        "check":
            "feature_order",

        "status":
            "PASS",
    },

    {
        "check":
            "feature_slice_roundtrip",

        "status":
            "PASS",
    },

    {
        "check":
            "pair_dimension_280",

        "status":
            "PASS",
    },

    {
        "check":
            "hidden_1_shape",

        "status":
            "PASS",
    },

    {
        "check":
            "hidden_2_shape",

        "status":
            "PASS",
    },

    {
        "check":
            "hidden_3_shape",

        "status":
            "PASS",
    },

    {
        "check":
            "hidden_4_shape",

        "status":
            "PASS",
    },

    {
        "check":
            "relu_all_hidden_layers",

        "status":
            "PASS",
    },

    {
        "check":
            "logit_output",

        "status":
            "PASS",
    },

    {
        "check":
            "sigmoid_probability",

        "status":
            "PASS",
    },

    {
        "check":
            "bce_logits_equivalence",

        "status":
            "PASS",
    },

    {
        "check":
            "ranking_equivalence",

        "status":
            "PASS",
    },

    {
        "check":
            "feature_order_sensitivity_hidden_1",

        "status":
            "PASS",
    },

    {
        "check":
            "feature_order_sensitivity_final_logit",

        "status":
            "PASS",
    },

    {
        "check":
            "batch_512",

        "status":
            "PASS",
    },

    {
        "check":
            "duplicate_input_determinism",

        "status":
            "PASS",
    },

    {
        "check":
            "input_feature_autograd",

        "status":
            "PASS",
    },

    {
        "check":
            "scoring_parameter_autograd",

        "status":
            "PASS",
    },
]


forward_df = pd.DataFrame(
    forward_records
)


forward_audit_path = (
    OUT_DIR
    / "scoring_forward_bce_audit.csv"
)


forward_df.to_csv(
    forward_audit_path,
    index=False,
)


# =============================================================================
# 23. FREEZE PHASE-4.5 FORWARD CONTRACT
# =============================================================================

banner(
    "FREEZING SCORING FORWARD CONTRACT"
)


forward_contract = {

    "phase":
        "4.5.2",

    "status":
        "FROZEN",

    "component":
        (
            "ITRS recommendation scoring "
            "forward and BCE contract"
        ),

    "pair_input":
        {

            "dimension":
                280,

            "exact_order":
                [
                    "F_t",
                    "L_o",
                    "F_d,o",
                    "F_s,o",
                    "L_b",
                    "F_d,b",
                    "F_s,b",
                ],

            "component_dimension":
                40,
        },

    "trend_indexing":
        {

            "runtime_name":
                "trend_for_target_h",

            "history_consumed":
                "T0 through T(h-1)",

            "additional_shift_in_scorer":
                False,

            "classification":
                trend_indexing_classification,

            "phase_4_3_reopened":
                False,
        },

    "forward":
        {

            "architecture":
                [
                    280,
                    128,
                    64,
                    32,
                    16,
                    1,
                ],

            "hidden_activation":
                "ReLU",

            "training_output":
                "raw_logit",

            "probability":
                "sigmoid(logit)",

            "parameter_count":
                EXPECTED_PARAMETER_COUNT,
        },

    "loss":
        {

            "implementation":
                "BCEWithLogitsLoss",

            "paper_semantics":
                "logistic probability + BCE",

            "verified_equivalent":
                True,

            "manual_stable_formula":
                "mean(softplus(logit) - y*logit)",
        },

    "ranking":
        {

            "logit_probability_order_equivalent":
                True,

            "reason":
                "sigmoid is strictly monotonic",
        },

    "feature_order_sensitivity":
        {

            "verified":
                True,

            "hidden_1_difference":
                hidden_1_difference,

            "final_logit_difference":
                final_logit_difference,

            "audit_initialization":
                (
                    "non-periodic explicit "
                    "column-position-sensitive"
                ),

            "first_failed_attempt_classification":
                (
                    "AUDIT_INITIALIZATION_DEGENERACY"
                ),

            "frozen_model_contract_changed":
                False,
        },

    "batch":
        {

            "paper_batch_size":
                PAPER_BATCH_SIZE,

            "batch_512_forward_verified":
                True,
        },

    "autograd":
        {

            "all_seven_input_features":
                True,

            "all_scoring_parameters":
                True,

            "training_performed":
                False,
        },

    "negative_sampling":
        {

            "status":
                "NOT_YET_FROZEN",

            "phase_2_deferral_preserved":
                True,

            "changed_in_phase_4_5_2":
                False,
        },

    "initialization":
        {

            "audit_initialization":
                (
                    "deterministic_non_kaiming_"
                    "position_sensitive"
                ),

            "audit_state_saved":
                False,

            "paper_family":
                "Kaiming",

            "exact_variant":
                "NOT_YET_FROZEN",
        },

    "training_performed":
        False,

    "model_state_saved":
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

            "phase_4_5_1a":
                False,

            "phase_4_5_1b":
                False,
        },
}


forward_contract_path = (
    OUT_DIR
    / "scoring_forward_contract.json"
)


with open(
    forward_contract_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        forward_contract,
        f,
        indent=2,
        ensure_ascii=False,
    )


# =============================================================================
# 24. PHASE 4.5 CLOSURE MANIFEST
# =============================================================================

banner(
    "FREEZING PHASE 4.5 CLOSURE MANIFEST"
)


closure_manifest = {

    "phase":
        "4.5",

    "name":
        "Recommendation Scoring Reconstruction",

    "status":
        "COMPLETE",

    "subphases":
        {

            "4.5.1a":
                {

                    "status":
                        "FROZEN",

                    "purpose":
                        (
                            "Scoring input and "
                            "paper-specified contract"
                        ),
                },

            "4.5.1b":
                {

                    "status":
                        "FROZEN",

                    "purpose":
                        (
                            "Scoring MLP neural "
                            "architecture"
                        ),
                },

            "4.5.2":
                {

                    "status":
                        "FROZEN",

                    "purpose":
                        (
                            "Forward, BCE and "
                            "autograd verification"
                        ),
                },
        },

    "final_scoring_input":
        {

            "investor":
                {

                    "formula":
                        (
                            "F_t || L_o || "
                            "F_d,o || F_s,o"
                        ),

                    "dimension":
                        160,
                },

            "startup":
                {

                    "formula":
                        (
                            "L_b || F_d,b || F_s,b"
                        ),

                    "dimension":
                        120,
                },

            "pair":
                {

                    "dimension":
                        280,

                    "exact_order":
                        [
                            "F_t",
                            "L_o",
                            "F_d,o",
                            "F_s,o",
                            "L_b",
                            "F_d,b",
                            "F_s,b",
                        ],
                },
        },

    "final_scorer":
        {

            "architecture":
                [
                    280,
                    128,
                    64,
                    32,
                    16,
                    1,
                ],

            "hidden_layers":
                4,

            "hidden_widths":
                [
                    128,
                    64,
                    32,
                    16,
                ],

            "activation":
                "ReLU",

            "hidden_bias":
                True,

            "output_bias":
                True,

            "dropout":
                0.0,

            "normalization":
                False,

            "residual":
                False,

            "parameter_count":
                EXPECTED_PARAMETER_COUNT,

            "width_classification":
                (
                    "PAPER_UNSPECIFIED_"
                    "REPRODUCTION_CHOICE_GUIDED_BY_NCF"
                ),
        },

    "output_and_loss":
        {

            "training_output":
                "raw_logit",

            "probability":
                "sigmoid(logit)",

            "training_loss":
                "BCEWithLogitsLoss",

            "paper_bce_semantics_preserved":
                True,

            "ranking_can_use_logits":
                True,
        },

    "trend_indexing":
        {

            "target_T_h":
                (
                    "use frozen Phase-4.3 trend "
                    "built from T0..T(h-1)"
                ),

            "additional_shift":
                False,

            "classification":
                trend_indexing_classification,
        },

    "feature_order_audit":
        {

            "verified":
                True,

            "hidden_1_difference":
                hidden_1_difference,

            "final_logit_difference":
                final_logit_difference,

            "failed_prior_attempt":
                (
                    "audit-only initialization "
                    "degeneracy"
                ),

            "architecture_reopened":
                False,
        },

    "verified":
        {

            "feature_order":
                True,

            "feature_slice_roundtrip":
                True,

            "forward_shapes":
                True,

            "relu_placement":
                True,

            "logit_probability_semantics":
                True,

            "bce_equivalence":
                True,

            "ranking_equivalence":
                True,

            "feature_order_sensitivity":
                True,

            "batch_512":
                True,

            "input_autograd":
                True,

            "parameter_autograd":
                True,
        },

    "still_open":
        still_open,

    "negative_sampling":
        {

            "status":
                "DEFERRED",

            "phase_2_decision_preserved":
                True,
        },

    "next_phase":
        {

            "phase":
                "4.6",

            "name":
                (
                    "Complete ITRS Forward "
                    "and BCE Integration"
                ),
        },
}


closure_manifest_path = (
    OUT_DIR
    / "phase_4_5_closure_manifest.json"
)


with open(
    closure_manifest_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        closure_manifest,
        f,
        indent=2,
        ensure_ascii=False,
    )


# =============================================================================
# 25. CLOSURE AUDIT
# =============================================================================

closure_records = [

    {
        "check":
            "scoring_input_contract",

        "status":
            "PASS",
    },

    {
        "check":
            "scoring_neural_contract",

        "status":
            "PASS",
    },

    {
        "check":
            "feature_order",

        "status":
            "PASS",
    },

    {
        "check":
            "280_dim_pair_input",

        "status":
            "PASS",
    },

    {
        "check":
            "four_hidden_layers",

        "status":
            "PASS",
    },

    {
        "check":
            "relu_hidden_layers",

        "status":
            "PASS",
    },

    {
        "check":
            "logistic_probability",

        "status":
            "PASS",
    },

    {
        "check":
            "bce_equivalence",

        "status":
            "PASS",
    },

    {
        "check":
            "ranking_equivalence",

        "status":
            "PASS",
    },

    {
        "check":
            "feature_order_sensitivity_hidden_1",

        "status":
            "PASS",
    },

    {
        "check":
            "feature_order_sensitivity_final_logit",

        "status":
            "PASS",
    },

    {
        "check":
            "feature_input_autograd",

        "status":
            "PASS",
    },

    {
        "check":
            "parameter_autograd",

        "status":
            "PASS",
    },

    {
        "check":
            "negative_sampling_deferral_preserved",

        "status":
            "PASS",
    },
]


closure_df = pd.DataFrame(
    closure_records
)


closure_audit_path = (
    OUT_DIR
    / "phase_4_5_closure_audit.csv"
)


closure_df.to_csv(
    closure_audit_path,
    index=False,
)


# =============================================================================
# 26. ARTIFACT HASHES
# =============================================================================

hash_records = []


for artifact, path in [

    (
        "scoring_forward_contract",
        forward_contract_path,
    ),

    (
        "scoring_forward_bce_audit",
        forward_audit_path,
    ),

    (
        "input_feature_autograd_audit",
        input_gradient_path,
    ),

    (
        "parameter_autograd_audit",
        parameter_gradient_path,
    ),

    (
        "phase_4_5_closure_manifest",
        closure_manifest_path,
    ),

    (
        "phase_4_5_closure_audit",
        closure_audit_path,
    ),
]:

    hash_records.append(
        {
            "artifact":
                artifact,

            "path":
                str(path),

            "sha256":
                sha256_file(
                    path
                ),

            "bytes":
                path.stat().st_size,
        }
    )


hash_df = pd.DataFrame(
    hash_records
)


hash_path = (
    OUT_DIR
    / "phase_4_5_artifact_hashes.csv"
)


hash_df.to_csv(
    hash_path,
    index=False,
)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

banner(
    "PHASE 4.5.2 FINAL SUMMARY"
)


print(
    "Pair representation:"
)

print(
    "  F_t || L_o || F_d,o || F_s,o "
    "|| L_b || F_d,b || F_s,b"
)

print(
    "  dimension                      280"
)


print()
print(
    "Scoring network:"
)

print(
    "  280 -> 128 -> 64 -> 32 -> 16 -> 1"
)

print(
    "  four hidden ReLU layers        PASS"
)

print(
    f"  parameters                     "
    f"{EXPECTED_PARAMETER_COUNT:,}"
)


print()
print(
    "Output / loss:"
)

print(
    "  raw training logit             PASS"
)

print(
    "  sigmoid probability            PASS"
)

print(
    "  BCEWithLogits equivalence      PASS"
)

print(
    "  logit/probability ranking      PASS"
)


print()
print(
    "Feature behavior:"
)

print(
    "  exact slice roundtrip          PASS"
)

print(
    "  hidden-1 order sensitivity     PASS"
)

print(
    "  final-logit order sensitivity  PASS"
)

print(
    "  duplicate input determinism    PASS"
)


print()
print(
    "Batch:"
)

print(
    "  batch size 512                 PASS"
)


print()
print(
    "Autograd:"
)

print(
    "  F_t                             PASS"
)

print(
    "  L_o                             PASS"
)

print(
    "  F_d,o                           PASS"
)

print(
    "  F_s,o                           PASS"
)

print(
    "  L_b                             PASS"
)

print(
    "  F_d,b                           PASS"
)

print(
    "  F_s,b                           PASS"
)

print(
    "  all scoring parameters          PASS"
)


print()
print(
    "Trend indexing:"
)

print(
    "  target T_h uses history "
    "T0..T(h-1)"
)

print(
    "  additional scorer shift        NO"
)

print(
    "  Phase-4.3 reopened             NO"
)


print()
print(
    "Negative sampling:"
)

print(
    "  random family from paper"
)

print(
    "  training contract frozen       NO"
)

print(
    "  Phase-2 deferral preserved     YES"
)


print()
print(
    "Failed first order-sensitivity attempt:"
)

print(
    "  classification                 "
    "AUDIT_INITIALIZATION_DEGENERACY"
)

print(
    "  frozen architecture changed    NO"
)


print()
print(
    "Final Kaiming variant frozen:    NO"
)

print(
    "Training performed:              NO"
)

print(
    "Audit model state persisted:     NO"
)


print()
print("Outputs:")

for path in [

    forward_contract_path,

    forward_audit_path,

    input_gradient_path,

    parameter_gradient_path,

    closure_manifest_path,

    closure_audit_path,

    hash_path,
]:

    print(
        f"  {path}"
    )


print()
print(
    "PHASE 4.5.2 STATUS: COMPLETE — "
    "SCORING FORWARD / BCE CONTRACT VERIFIED"
)


print()
print("=" * 120)

print(
    "PHASE 4.5 STATUS: COMPLETE — "
    "RECOMMENDATION SCORING RECONSTRUCTION CLOSED"
)

print("=" * 120)


print()
print(
    "NEXT:"
)

print(
    "PHASE 4.6 — "
    "COMPLETE ITRS FORWARD + BCE INTEGRATION"
)