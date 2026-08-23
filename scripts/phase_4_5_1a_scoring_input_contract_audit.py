from pathlib import Path
import json
import sys

import pandas as pd


# =============================================================================
# PHASE 4.5.1a — RECOMMENDATION SCORING INPUT / PAPER-CONTRACT AUDIT
#
# PURPOSE
# -------
# Establish the exact feature-composition and dimensional contract for the
# final ITRS recommendation-scoring module before choosing any paper-
# unspecified hidden-layer widths.
#
# PAPER SCORING EQUATIONS
# -----------------------
#
# For an Investor o_i and candidate Startup b_j in target period T_h:
#
#   R_o,i,h =
#       [
#           F_t,i,h-1
#           || L_o,i
#           || F_d,o,i
#           || F_s,o,i
#       ]
#
#   R_b,j =
#       [
#           L_b,j
#           || F_d,b,j
#           || F_s,b,j
#       ]
#
#   y_hat_i,j,h =
#       MLP(
#           [
#               R_o,i,h
#               || R_b,j
#           ]
#       )
#
# FROZEN FEATURE DIMENSIONS
# -------------------------
#
#   latent embedding         = 40
#   description feature      = 40
#   trend feature            = 40
#   structural feature       = 40
#
# Therefore:
#
#   Investor representation:
#
#       40 + 40 + 40 + 40 = 160
#
#   Startup representation:
#
#       40 + 40 + 40 = 120
#
#   Final pair representation:
#
#       160 + 120 = 280
#
# IMPORTANT
# ---------
# This audit DOES NOT:
#
#   - choose scoring hidden-layer widths,
#   - freeze negative-sampling ratio,
#   - freeze train-negative eligibility,
#   - train anything,
#   - initialize scoring weights,
#   - reopen Phase 2,
#   - reopen Phase 3,
#   - reopen description/trend/R-GCN contracts.
#
# The paper specifies four hidden layers but does not specify their widths.
# That ambiguity will remain explicit after Phase 4.5.1a.
# =============================================================================


# =============================================================================
# INPUTS
# =============================================================================

DESCRIPTION_CONTRACT_PATH = Path(
    "data/experimental/phase_4/"
    "description_neural_contract/"
    "description_neural_contract.json"
)

TREND_NEURAL_CONTRACT_PATH = Path(
    "data/experimental/phase_4/"
    "trend_neural_contract/"
    "trend_neural_contract.json"
)

TREND_RUNTIME_CONTRACT_PATH = Path(
    "data/experimental/phase_4/"
    "trend_runtime/"
    "trend_runtime_contract.json"
)

RGCN_NEURAL_CONTRACT_PATH = Path(
    "data/experimental/phase_4/"
    "rgcn_neural_contract/"
    "rgcn_neural_contract.json"
)

RGCN_INTEGRATION_CONTRACT_PATH = Path(
    "data/experimental/phase_4/"
    "rgcn_integration/"
    "rgcn_integration_contract.json"
)

PHASE_4_4_CLOSURE_PATH = Path(
    "data/experimental/phase_4/"
    "rgcn_integration/"
    "phase_4_4_closure_manifest.json"
)


# =============================================================================
# OUTPUTS
# =============================================================================

OUT_DIR = Path(
    "data/experimental/phase_4/"
    "scoring_contract"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# FROZEN FEATURE DIMENSIONS
# =============================================================================

LATENT_DIM = 40
DESCRIPTION_DIM = 40
TREND_DIM = 40
STRUCTURAL_DIM = 40


INVESTOR_REPRESENTATION_DIM = (
    TREND_DIM
    + LATENT_DIM
    + DESCRIPTION_DIM
    + STRUCTURAL_DIM
)


STARTUP_REPRESENTATION_DIM = (
    LATENT_DIM
    + DESCRIPTION_DIM
    + STRUCTURAL_DIM
)


PAIR_REPRESENTATION_DIM = (
    INVESTOR_REPRESENTATION_DIM
    + STARTUP_REPRESENTATION_DIM
)


assert (
    INVESTOR_REPRESENTATION_DIM
    == 160
)

assert (
    STARTUP_REPRESENTATION_DIM
    == 120
)

assert (
    PAIR_REPRESENTATION_DIM
    == 280
)


# =============================================================================
# PAPER-SPECIFIED SCORING CONSTANTS
# =============================================================================

SCORING_HIDDEN_LAYER_COUNT = 4

SCORING_HIDDEN_ACTIVATION = "ReLU"

OUTPUT_DIM = 1

TARGET_TYPE = "binary"

LOSS_FAMILY = "binary_cross_entropy"

OUTPUT_SEMANTICS = "logistic_probability"


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


# =============================================================================
# START
# =============================================================================

banner(
    "PHASE 4.5.1a — "
    "RECOMMENDATION SCORING INPUT / PAPER-CONTRACT AUDIT"
)


# =============================================================================
# 1. ENVIRONMENT
# =============================================================================

banner(
    "ENVIRONMENT"
)


print(
    f"Python: "
    f"{sys.version.splitlines()[0]}"
)


# =============================================================================
# 2. LOAD UPSTREAM FROZEN CONTRACTS
# =============================================================================

banner(
    "UPSTREAM PHASE-4 CONTRACT INTEGRITY"
)


description_contract = load_json(
    DESCRIPTION_CONTRACT_PATH
)


trend_neural_contract = load_json(
    TREND_NEURAL_CONTRACT_PATH
)


trend_runtime_contract = load_json(
    TREND_RUNTIME_CONTRACT_PATH
)


rgcn_neural_contract = load_json(
    RGCN_NEURAL_CONTRACT_PATH
)


rgcn_integration_contract = load_json(
    RGCN_INTEGRATION_CONTRACT_PATH
)


phase_4_4_closure = load_json(
    PHASE_4_4_CLOSURE_PATH
)


require(
    description_contract.get(
        "status"
    )
    == "FROZEN",
    "Description neural contract is not frozen.",
)


require(
    trend_neural_contract.get(
        "status"
    )
    == "FROZEN",
    "Trend neural contract is not frozen.",
)


require(
    trend_runtime_contract.get(
        "status"
    )
    == "FROZEN",
    "Trend runtime contract is not frozen.",
)


require(
    rgcn_neural_contract.get(
        "status"
    )
    == "FROZEN",
    "R-GCN neural contract is not frozen.",
)


require(
    rgcn_integration_contract.get(
        "status"
    )
    == "FROZEN",
    "R-GCN integration contract is not frozen.",
)


require(
    phase_4_4_closure.get(
        "status"
    )
    == "COMPLETE",
    "Phase 4.4 is not closed.",
)


print(
    "Description contract:       PASS"
)

print(
    "Trend neural contract:      PASS"
)

print(
    "Trend runtime contract:     PASS"
)

print(
    "R-GCN neural contract:      PASS"
)

print(
    "R-GCN integration contract: PASS"
)

print(
    "Phase 4.4 closure:          PASS"
)


# =============================================================================
# 3. VERIFY DESCRIPTION DIMENSION
# =============================================================================

banner(
    "DESCRIPTION FEATURE DIMENSION"
)


actual_description_dim = int(
    description_contract[
        "paper_specified"
    ][
        "final_description_dim"
    ]
)


print(
    f"Frozen description dimension: "
    f"{actual_description_dim}"
)


require(
    actual_description_dim
    == DESCRIPTION_DIM,
    "Description feature dimension changed.",
)


# =============================================================================
# 4. VERIFY TREND DIMENSION
# =============================================================================

banner(
    "TREND FEATURE DIMENSION"
)


actual_trend_dim = int(
    trend_neural_contract[
        "paper_specified"
    ][
        "trend_output_dim"
    ]
)


print(
    f"Frozen trend dimension: "
    f"{actual_trend_dim}"
)


require(
    actual_trend_dim
    == TREND_DIM,
    "Trend feature dimension changed.",
)


# =============================================================================
# 5. VERIFY STRUCTURAL DIMENSION
# =============================================================================

banner(
    "STRUCTURAL FEATURE DIMENSION"
)


actual_structural_dim = int(
    rgcn_neural_contract[
        "architecture"
    ][
        "output_dim"
    ]
)


print(
    f"Frozen structural dimension: "
    f"{actual_structural_dim}"
)


require(
    actual_structural_dim
    == STRUCTURAL_DIM,
    "Structural feature dimension changed.",
)


# =============================================================================
# 6. VERIFY SHARED LATENT DIMENSION
#
# Description/trend/R-GCN/scoring all use the same 40-D Investor and
# Startup latent embedding tables.
# =============================================================================

banner(
    "SHARED LATENT EMBEDDING DIMENSION"
)


rgcn_latent_dim = int(
    rgcn_integration_contract[
        "shared_latent_embeddings"
    ][
        "combined_shape"
    ][
        1
    ]
)


print(
    f"Frozen latent dimension: "
    f"{rgcn_latent_dim}"
)


require(
    rgcn_latent_dim
    == LATENT_DIM,
    "Shared latent dimension changed.",
)


# =============================================================================
# 7. INVESTOR SCORING REPRESENTATION
#
# Paper:
#
#   R_o,i,h =
#       [
#           F_t,i,h-1
#           || L_o,i
#           || F_d,o,i
#           || F_s,o,i
#       ]
#
# Exact order matters and is frozen here.
# =============================================================================

banner(
    "INVESTOR SCORING REPRESENTATION"
)


investor_components = [

    {
        "position":
            0,

        "component":
            "trend",

        "symbol":
            "F_t",

        "dimension":
            TREND_DIM,
    },

    {
        "position":
            1,

        "component":
            "latent",

        "symbol":
            "L_o",

        "dimension":
            LATENT_DIM,
    },

    {
        "position":
            2,

        "component":
            "description",

        "symbol":
            "F_d,o",

        "dimension":
            DESCRIPTION_DIM,
    },

    {
        "position":
            3,

        "component":
            "structural",

        "symbol":
            "F_s,o",

        "dimension":
            STRUCTURAL_DIM,
    },
]


running_offset = 0


for component in investor_components:

    start = running_offset

    end = (
        start
        + component[
            "dimension"
        ]
    )


    component[
        "slice_start"
    ] = start


    component[
        "slice_end_exclusive"
    ] = end


    running_offset = end


    print(
        f"{component['position']}: "
        f"{component['symbol']:<10} "
        f"dim={component['dimension']:<3} "
        f"slice=[{start}:{end}]"
    )


require(
    running_offset
    == INVESTOR_REPRESENTATION_DIM,
    "Investor representation dimension mismatch.",
)


print()
print(
    f"Investor representation dimension: "
    f"{INVESTOR_REPRESENTATION_DIM}"
)


# =============================================================================
# 8. STARTUP SCORING REPRESENTATION
#
# Paper:
#
#   R_b,j =
#       [
#           L_b,j
#           || F_d,b,j
#           || F_s,b,j
#       ]
# =============================================================================

banner(
    "STARTUP SCORING REPRESENTATION"
)


startup_components = [

    {
        "position":
            0,

        "component":
            "latent",

        "symbol":
            "L_b",

        "dimension":
            LATENT_DIM,
    },

    {
        "position":
            1,

        "component":
            "description",

        "symbol":
            "F_d,b",

        "dimension":
            DESCRIPTION_DIM,
    },

    {
        "position":
            2,

        "component":
            "structural",

        "symbol":
            "F_s,b",

        "dimension":
            STRUCTURAL_DIM,
    },
]


running_offset = 0


for component in startup_components:

    start = running_offset

    end = (
        start
        + component[
            "dimension"
        ]
    )


    component[
        "slice_start"
    ] = start


    component[
        "slice_end_exclusive"
    ] = end


    running_offset = end


    print(
        f"{component['position']}: "
        f"{component['symbol']:<10} "
        f"dim={component['dimension']:<3} "
        f"slice=[{start}:{end}]"
    )


require(
    running_offset
    == STARTUP_REPRESENTATION_DIM,
    "Startup representation dimension mismatch.",
)


print()
print(
    f"Startup representation dimension: "
    f"{STARTUP_REPRESENTATION_DIM}"
)


# =============================================================================
# 9. FINAL PAIR REPRESENTATION
#
# Paper:
#
#   [R_o || R_b]
#
# Exact ordering:
#
#   [ F_t
#     || L_o
#     || F_d,o
#     || F_s,o
#     || L_b
#     || F_d,b
#     || F_s,b ]
#
# =============================================================================

banner(
    "FINAL PAIR REPRESENTATION"
)


pair_components = [

    {
        "position":
            0,

        "component":
            "investor_trend",

        "symbol":
            "F_t",

        "dimension":
            TREND_DIM,
    },

    {
        "position":
            1,

        "component":
            "investor_latent",

        "symbol":
            "L_o",

        "dimension":
            LATENT_DIM,
    },

    {
        "position":
            2,

        "component":
            "investor_description",

        "symbol":
            "F_d,o",

        "dimension":
            DESCRIPTION_DIM,
    },

    {
        "position":
            3,

        "component":
            "investor_structural",

        "symbol":
            "F_s,o",

        "dimension":
            STRUCTURAL_DIM,
    },

    {
        "position":
            4,

        "component":
            "startup_latent",

        "symbol":
            "L_b",

        "dimension":
            LATENT_DIM,
    },

    {
        "position":
            5,

        "component":
            "startup_description",

        "symbol":
            "F_d,b",

        "dimension":
            DESCRIPTION_DIM,
    },

    {
        "position":
            6,

        "component":
            "startup_structural",

        "symbol":
            "F_s,b",

        "dimension":
            STRUCTURAL_DIM,
    },
]


running_offset = 0


for component in pair_components:

    start = running_offset

    end = (
        start
        + component[
            "dimension"
        ]
    )


    component[
        "slice_start"
    ] = start


    component[
        "slice_end_exclusive"
    ] = end


    running_offset = end


    print(
        f"{component['position']}: "
        f"{component['symbol']:<10} "
        f"dim={component['dimension']:<3} "
        f"slice=[{start}:{end}]"
    )


require(
    running_offset
    == PAIR_REPRESENTATION_DIM,
    "Pair representation dimension mismatch.",
)


print()
print(
    f"Investor side: "
    f"{INVESTOR_REPRESENTATION_DIM}"
)

print(
    f"Startup side:  "
    f"{STARTUP_REPRESENTATION_DIM}"
)

print(
    f"Pair input:    "
    f"{PAIR_REPRESENTATION_DIM}"
)


# =============================================================================
# 10. TEMPORAL FEATURE SEMANTICS
#
# The scoring candidate does not alter the frozen historical trend.
#
# For a target in T_h:
#
#   - T0 target -> frozen zero_40 trend
#   - T1 target -> history T0
#   - ...
#   - T60 target -> history T0..T59
#
# The scoring module merely consumes the already-defined F_t for the
# Investor/target-period request.
# =============================================================================

banner(
    "TREND INPUT TO SCORING"
)


print(
    "Trend feature source:"
)

print(
    "  frozen Phase-4.3 trend encoder"
)


print()
print(
    "Candidate Startup modifies trend history:"
)

print(
    "  NO"
)


print()
print(
    "Current target-period investment consumed by trend:"
)

print(
    "  NO"
)


print()
print(
    "T60 trend history:"
)

print(
    "  T0 through T59 only"
)


# =============================================================================
# 11. STRUCTURAL FEATURE SEMANTICS
#
# The scoring module consumes F_s rows from the current full-graph R-GCN
# result.
#
# F_s is independent of:
#
#   target period
#   label
#   specific pair
#
# Candidate merely selects its Startup structural row.
# =============================================================================

banner(
    "STRUCTURAL INPUT TO SCORING"
)


print(
    "Structural feature source:"
)

print(
    "  current full-graph R-GCN forward"
)


print()
print(
    "Investor structural row:"
)

print(
    "  F_s_all[investor_node_index]"
)


print()
print(
    "Startup structural row:"
)

print(
    "  F_s_all[startup_global_node_index]"
)


print()
print(
    "Target segment changes F_s:"
)

print(
    "  NO"
)


print()
print(
    "Positive/negative label changes F_s:"
)

print(
    "  NO"
)


# =============================================================================
# 12. PAPER-SPECIFIED SCORING NETWORK FACTS
# =============================================================================

banner(
    "PAPER-SPECIFIED SCORING NETWORK"
)


print(
    f"MLP input dimension:        "
    f"{PAIR_REPRESENTATION_DIM}"
)

print(
    f"Hidden layer count:         "
    f"{SCORING_HIDDEN_LAYER_COUNT}"
)

print(
    f"Hidden activation family:   "
    f"{SCORING_HIDDEN_ACTIVATION}"
)

print(
    f"Output dimension:           "
    f"{OUTPUT_DIM}"
)

print(
    f"Prediction target:          "
    f"{TARGET_TYPE}"
)

print(
    f"Output semantics:           "
    f"{OUTPUT_SEMANTICS}"
)

print(
    f"Loss family:                "
    f"{LOSS_FAMILY}"
)


# =============================================================================
# 13. PAPER-UNSPECIFIED SCORING DETAILS
# =============================================================================

banner(
    "PAPER-UNSPECIFIED SCORING DETAILS"
)


paper_unspecified = [

    "hidden layer 1 width",

    "hidden layer 2 width",

    "hidden layer 3 width",

    "hidden layer 4 width",

    "presence/absence of hidden-layer biases",

    (
        "whether probability sigmoid is implemented "
        "inside the model or BCE-with-logits is used"
    ),

    "dropout in scoring MLP",

    "batch normalization",

    "layer normalization",

    "residual connections",

    "exact Kaiming variant",

    "bias initialization",

    "neural-model epoch count",

    "early stopping",

    "weight decay",
]


for item in paper_unspecified:

    print(
        f"  - {item}"
    )


print()
print(
    "These are NOT silently frozen in Phase 4.5.1a."
)


# =============================================================================
# 14. NEGATIVE-SAMPLING BOUNDARY
#
# The paper trains over E union E-minus and states negatives are random.
#
# However, the exact training negative ratio / eligibility / exclusion
# procedure was not reported sufficiently to freeze it.
#
# Phase 2 explicitly deferred this decision.
#
# Phase 4.5.1a preserves that deferral.
# =============================================================================

banner(
    "NEGATIVE-SAMPLING BOUNDARY"
)


print(
    "Positive target:"
)

print(
    "  observed investment event -> y = 1"
)


print()
print(
    "Negative target:"
)

print(
    "  sampled non-positive candidate -> y = 0"
)


print()
print(
    "Paper negative family:"
)

print(
    "  random negatives"
)


print()
print(
    "Training negative:positive ratio:"
)

print(
    "  NOT FROZEN"
)


print()
print(
    "Exact negative candidate eligibility:"
)

print(
    "  NOT FROZEN"
)


print()
print(
    "Exact historical exclusion rules:"
)

print(
    "  NOT FROZEN"
)


print()
print(
    "Phase-2 negative-sampling deferral preserved:"
)

print(
    "  YES"
)


# =============================================================================
# 15. REPRESENTATION CONTRACT TABLE
# =============================================================================

representation_records = []


for component in pair_components:

    representation_records.append(
        {
            "position":
                component[
                    "position"
                ],

            "component":
                component[
                    "component"
                ],

            "paper_symbol":
                component[
                    "symbol"
                ],

            "dimension":
                component[
                    "dimension"
                ],

            "slice_start":
                component[
                    "slice_start"
                ],

            "slice_end_exclusive":
                component[
                    "slice_end_exclusive"
                ],
        }
    )


representation_df = pd.DataFrame(
    representation_records
)


representation_path = (
    OUT_DIR
    / "scoring_pair_representation_contract.csv"
)


representation_df.to_csv(
    representation_path,
    index=False,
)


# =============================================================================
# 16. DECISION CLASSIFICATION
# =============================================================================

decision_records = [

    {
        "decision":
            "Investor representation",

        "value":
            (
                "F_t || L_o || F_d,o || F_s,o"
            ),

        "classification":
            "PAPER_SPECIFIED",
    },

    {
        "decision":
            "Startup representation",

        "value":
            (
                "L_b || F_d,b || F_s,b"
            ),

        "classification":
            "PAPER_SPECIFIED",
    },

    {
        "decision":
            "Investor representation dimension",

        "value":
            "160",

        "classification":
            (
                "PAPER_SPECIFIED_DIMENSIONAL_"
                "CONSEQUENCE"
            ),
    },

    {
        "decision":
            "Startup representation dimension",

        "value":
            "120",

        "classification":
            (
                "PAPER_SPECIFIED_DIMENSIONAL_"
                "CONSEQUENCE"
            ),
    },

    {
        "decision":
            "Pair scoring input dimension",

        "value":
            "280",

        "classification":
            (
                "PAPER_SPECIFIED_DIMENSIONAL_"
                "CONSEQUENCE"
            ),
    },

    {
        "decision":
            "Scoring hidden-layer count",

        "value":
            "4",

        "classification":
            "PAPER_SPECIFIED",
    },

    {
        "decision":
            "MLP hidden activation",

        "value":
            "ReLU",

        "classification":
            "PAPER_SPECIFIED",
    },

    {
        "decision":
            "Prediction dimensionality",

        "value":
            "1",

        "classification":
            (
                "PAPER_SPECIFIED_BY_BINARY_"
                "SCORING_OBJECTIVE"
            ),
    },

    {
        "decision":
            "Prediction semantics",

        "value":
            "binary logistic probability",

        "classification":
            "PAPER_SPECIFIED",
    },

    {
        "decision":
            "Loss family",

        "value":
            "binary cross entropy",

        "classification":
            "PAPER_SPECIFIED",
    },

    {
        "decision":
            "Hidden widths",

        "value":
            "NOT_YET_FROZEN",

        "classification":
            "PAPER_UNSPECIFIED",
    },

    {
        "decision":
            "Training negative sampling",

        "value":
            "NOT_YET_FROZEN",

        "classification":
            (
                "PAPER_PARTIALLY_SPECIFIED_"
                "PHASE2_DEFERRED"
            ),
    },

    {
        "decision":
            "Exact initialization",

        "value":
            "NOT_YET_FROZEN",

        "classification":
            (
                "KAIMING_FAMILY_SPECIFIED_"
                "VARIANT_UNSPECIFIED"
            ),
    },
]


decision_df = pd.DataFrame(
    decision_records
)


decision_path = (
    OUT_DIR
    / "scoring_paper_contract_audit.csv"
)


decision_df.to_csv(
    decision_path,
    index=False,
)


# =============================================================================
# 17. FREEZE SCORING INPUT CONTRACT
# =============================================================================

banner(
    "FREEZING SCORING INPUT CONTRACT"
)


contract = {

    "phase":
        "4.5.1a",

    "status":
        "FROZEN_INPUT_CONTRACT",

    "component":
        (
            "ITRS recommendation scoring "
            "input and paper-specified architecture"
        ),

    "feature_dimensions":
        {

            "latent":
                LATENT_DIM,

            "description":
                DESCRIPTION_DIM,

            "trend":
                TREND_DIM,

            "structural":
                STRUCTURAL_DIM,
        },

    "investor_representation":
        {

            "formula":
                (
                    "F_t || L_o || "
                    "F_d,o || F_s,o"
                ),

            "component_order":
                [
                    "trend",
                    "latent",
                    "description",
                    "structural",
                ],

            "dimension":
                INVESTOR_REPRESENTATION_DIM,
        },

    "startup_representation":
        {

            "formula":
                (
                    "L_b || F_d,b || F_s,b"
                ),

            "component_order":
                [
                    "latent",
                    "description",
                    "structural",
                ],

            "dimension":
                STARTUP_REPRESENTATION_DIM,
        },

    "pair_representation":
        {

            "formula":
                "R_o || R_b",

            "dimension":
                PAIR_REPRESENTATION_DIM,

            "exact_component_order":
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

    "paper_specified_scoring":
        {

            "hidden_layer_count":
                SCORING_HIDDEN_LAYER_COUNT,

            "hidden_activation":
                SCORING_HIDDEN_ACTIVATION,

            "output_dim":
                OUTPUT_DIM,

            "target_type":
                TARGET_TYPE,

            "output_semantics":
                OUTPUT_SEMANTICS,

            "loss":
                LOSS_FAMILY,
        },

    "paper_unspecified":
        {

            "hidden_widths":
                [
                    "NOT_YET_FROZEN",
                    "NOT_YET_FROZEN",
                    "NOT_YET_FROZEN",
                    "NOT_YET_FROZEN",
                ],

            "hidden_biases":
                "NOT_YET_FROZEN",

            "sigmoid_vs_bce_with_logits_implementation":
                "NOT_YET_FROZEN",

            "dropout":
                "NOT_YET_FROZEN",

            "normalization":
                "NOT_YET_FROZEN",

            "residual":
                "NOT_YET_FROZEN",

            "exact_kaiming_variant":
                "NOT_YET_FROZEN",
        },

    "negative_sampling":
        {

            "paper_family":
                "random",

            "training_ratio":
                "NOT_YET_FROZEN",

            "candidate_eligibility":
                "NOT_YET_FROZEN",

            "historical_exclusion":
                "NOT_YET_FROZEN",

            "phase_2_deferral_preserved":
                True,
        },

    "feature_runtime_sources":
        {

            "latent":
                (
                    "shared trainable Investor / "
                    "Startup embedding tables"
                ),

            "description":
                (
                    "current trainable description "
                    "encoder output"
                ),

            "trend":
                (
                    "current trainable trend "
                    "encoder output"
                ),

            "structural":
                (
                    "current trainable full-graph "
                    "R-GCN output"
                ),
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
        },

    "next_phase":
        {

            "phase":
                "4.5.1b",

            "purpose":
                (
                    "Resolve and freeze paper-"
                    "unspecified scoring MLP "
                    "implementation choices."
                ),
        },
}


contract_path = (
    OUT_DIR
    / "scoring_input_contract.json"
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
    "PHASE 4.5.1a FINAL SUMMARY"
)


print(
    "Investor representation:"
)

print(
    "  F_t || L_o || F_d,o || F_s,o"
)

print(
    f"  dimension = "
    f"{INVESTOR_REPRESENTATION_DIM}"
)


print()
print(
    "Startup representation:"
)

print(
    "  L_b || F_d,b || F_s,b"
)

print(
    f"  dimension = "
    f"{STARTUP_REPRESENTATION_DIM}"
)


print()
print(
    "Final scoring input:"
)

print(
    "  R_o || R_b"
)

print(
    f"  dimension = "
    f"{PAIR_REPRESENTATION_DIM}"
)


print()
print(
    "Paper-specified scoring:"
)

print(
    "  hidden layers         4"
)

print(
    "  hidden activation     ReLU"
)

print(
    "  output dimension      1"
)

print(
    "  binary logistic score"
)

print(
    "  BCE objective"
)


print()
print(
    "Not yet frozen:"
)

print(
    "  four hidden widths"
)

print(
    "  hidden biases"
)

print(
    "  sigmoid vs BCEWithLogits implementation"
)

print(
    "  dropout / normalization / residual"
)

print(
    "  exact Kaiming variant"
)

print(
    "  training negative-sampling contract"
)


print()
print(
    "Training performed:             NO"
)

print(
    "Model state persisted:          NO"
)

print(
    "Phase-2 negative deferral kept: YES"
)


print()
print("Outputs:")

for path in [
    representation_path,
    decision_path,
    contract_path,
]:

    print(
        f"  {path}"
    )


print()
print(
    "PHASE 4.5.1a STATUS: COMPLETE — "
    "SCORING INPUT / PAPER CONTRACT FROZEN"
)