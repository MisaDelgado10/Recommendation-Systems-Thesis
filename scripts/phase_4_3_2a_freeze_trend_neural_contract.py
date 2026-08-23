from pathlib import Path
import json
import sys

import torch
import torch.nn as nn


# =============================================================================
# PHASE 4.3.2a — FREEZE TREND ATTENTION + GRU NEURAL CONTRACT
# =============================================================================

HISTORY_CONTRACT_PATH = Path(
    "data/experimental/phase_4/"
    "trend_contract/"
    "trend_history_semantics_contract.json"
)

DESCRIPTION_CONTRACT_PATH = Path(
    "data/experimental/phase_4/"
    "description_neural_contract/"
    "description_neural_contract.json"
)

OUT_DIR = Path(
    "data/experimental/phase_4/"
    "trend_neural_contract"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# PAPER-SPECIFIED / UPSTREAM-FROZEN DIMENSIONS
# =============================================================================

LATENT_DIM = 40
DESCRIPTION_DIM = 40

ATTENTION_ITEM_DIM = (
    LATENT_DIM
    + DESCRIPTION_DIM
)

ATTENTION_QUERY_DIM = (
    LATENT_DIM
    + DESCRIPTION_DIM
)

GRU_INPUT_DIM = 80
GRU_HIDDEN_DIM = 40
GRU_NUM_LAYERS = 2

TREND_OUTPUT_DIM = 40


assert ATTENTION_ITEM_DIM == 80
assert ATTENTION_QUERY_DIM == 80
assert GRU_INPUT_DIM == 80


# =============================================================================
# EXPLICIT REPRODUCTION CHOICES
# =============================================================================

GRU_BIAS = True
GRU_DROPOUT = 0.0
GRU_BIDIRECTIONAL = False
GRU_BATCH_FIRST = True

OUTPUT_PROJECTION_BIAS = False


# =============================================================================
# EXPECTED PARAMETER COUNTS
# =============================================================================

EXPECTED_ATTENTION_PARAMETERS = 6_400
EXPECTED_GRU_PARAMETERS = 24_480
EXPECTED_OUTPUT_PARAMETERS = 1_600

EXPECTED_TOTAL_PARAMETERS = (
    EXPECTED_ATTENTION_PARAMETERS
    + EXPECTED_GRU_PARAMETERS
    + EXPECTED_OUTPUT_PARAMETERS
)

assert EXPECTED_TOTAL_PARAMETERS == 32_480


# =============================================================================
# Helpers
# =============================================================================

def banner(title):

    print()
    print("=" * 120)
    print(title)
    print("=" * 120)


# =============================================================================
# Minimal contract module
#
# IMPORTANT:
# This is instantiated for architecture auditing only.
# No final initialization is frozen here.
# =============================================================================

class TrendNeuralContractModule(nn.Module):

    def __init__(self):

        super().__init__()

        # -------------------------------------------------------------
        # Bilinear attention parameter:
        #
        # score = (u @ W) @ V.T
        #
        # No attention bias exists in the paper equation.
        # -------------------------------------------------------------

        self.attention_weight = nn.Parameter(
            torch.empty(
                ATTENTION_QUERY_DIM,
                ATTENTION_ITEM_DIM,
            )
        )


        # -------------------------------------------------------------
        # Recurrent trend extractor.
        #
        # Standard PyTorch GRU chosen as explicit reproduction policy.
        # -------------------------------------------------------------

        self.gru = nn.GRU(
            input_size=GRU_INPUT_DIM,
            hidden_size=GRU_HIDDEN_DIM,
            num_layers=GRU_NUM_LAYERS,
            bias=GRU_BIAS,
            batch_first=GRU_BATCH_FIRST,
            dropout=GRU_DROPOUT,
            bidirectional=GRU_BIDIRECTIONAL,
        )


        # -------------------------------------------------------------
        # Paper:
        #
        #     y_t = sigmoid(W_o h_t)
        #
        # We therefore use no output-projection bias.
        # -------------------------------------------------------------

        self.output_projection = nn.Linear(
            GRU_HIDDEN_DIM,
            TREND_OUTPUT_DIM,
            bias=OUTPUT_PROJECTION_BIAS,
        )


        self.output_activation = nn.Sigmoid()


# =============================================================================
# 1. Environment
# =============================================================================

banner(
    "PHASE 4.3.2a — "
    "FREEZE TREND ATTENTION + GRU NEURAL CONTRACT"
)


print("\nENVIRONMENT")
print("-" * 120)

print(
    f"Python:  "
    f"{sys.version.splitlines()[0]}"
)

print(
    f"PyTorch: "
    f"{torch.__version__}"
)

print(
    "Audit device: CPU"
)


# =============================================================================
# 2. Verify frozen upstream contracts
# =============================================================================

banner(
    "UPSTREAM CONTRACT INTEGRITY"
)


with open(
    HISTORY_CONTRACT_PATH,
    "r",
    encoding="utf-8",
) as f:

    history_contract = json.load(f)


with open(
    DESCRIPTION_CONTRACT_PATH,
    "r",
    encoding="utf-8",
) as f:

    description_contract = json.load(f)


if (
    history_contract.get("status")
    != "FROZEN"
):

    raise AssertionError(
        "Trend-history semantic contract "
        "is not frozen."
    )


if (
    description_contract.get("status")
    != "FROZEN"
):

    raise AssertionError(
        "Description neural contract "
        "is not frozen."
    )


if (
    history_contract[
        "dimensions"
    ][
        "startup_attention_item"
    ]
    != ATTENTION_ITEM_DIM
):

    raise AssertionError(
        "Frozen period-vector dimension "
        "changed."
    )


if (
    history_contract[
        "dimensions"
    ][
        "period_attention_output"
    ]
    != GRU_INPUT_DIM
):

    raise AssertionError(
        "Frozen GRU input dimension changed."
    )


if (
    description_contract[
        "paper_specified"
    ][
        "final_description_dim"
    ]
    != DESCRIPTION_DIM
):

    raise AssertionError(
        "Frozen description dimension changed."
    )


print(
    "Trend-history contract: PASS"
)

print(
    "Description contract:   PASS"
)


# =============================================================================
# 3. Dimension contract
# =============================================================================

banner(
    "TREND DIMENSION CONTRACT"
)


print(
    f"Investor latent:                 "
    f"{LATENT_DIM}"
)

print(
    f"Investor description:            "
    f"{DESCRIPTION_DIM}"
)

print(
    f"Investor attention query:        "
    f"{ATTENTION_QUERY_DIM}"
)


print()
print(
    f"Startup latent:                  "
    f"{LATENT_DIM}"
)

print(
    f"Startup description:             "
    f"{DESCRIPTION_DIM}"
)

print(
    f"Startup attention item:          "
    f"{ATTENTION_ITEM_DIM}"
)


print()
print(
    f"Attention period output:         "
    f"{GRU_INPUT_DIM}"
)

print(
    f"GRU hidden dimension:            "
    f"{GRU_HIDDEN_DIM}"
)

print(
    f"GRU recurrent layers:            "
    f"{GRU_NUM_LAYERS}"
)

print(
    f"Trend output dimension:          "
    f"{TREND_OUTPUT_DIM}"
)


# =============================================================================
# 4. Instantiate neural contract
# =============================================================================

banner(
    "MODULE INSTANTIATION"
)


module = TrendNeuralContractModule()


print(
    module
)


# =============================================================================
# 5. Exact parameter-shape audit
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

    "attention_weight":
        (80, 80),

    "gru.weight_ih_l0":
        (120, 80),

    "gru.weight_hh_l0":
        (120, 40),

    "gru.bias_ih_l0":
        (120,),

    "gru.bias_hh_l0":
        (120,),

    "gru.weight_ih_l1":
        (120, 40),

    "gru.weight_hh_l1":
        (120, 40),

    "gru.bias_ih_l1":
        (120,),

    "gru.bias_hh_l1":
        (120,),

    "output_projection.weight":
        (40, 40),
}


if (
    parameter_shapes
    != expected_shapes
):

    missing = sorted(
        set(expected_shapes)
        - set(parameter_shapes)
    )

    extra = sorted(
        set(parameter_shapes)
        - set(expected_shapes)
    )

    mismatched = {
        key: {
            "expected":
                expected_shapes[key],

            "actual":
                parameter_shapes.get(key),
        }

        for key in expected_shapes

        if (
            key in parameter_shapes
            and parameter_shapes[key]
            != expected_shapes[key]
        )
    }


    raise AssertionError(
        "Trend parameter-shape contract "
        "does not match expectation.\n"
        f"Missing: {missing}\n"
        f"Extra: {extra}\n"
        f"Mismatched: {mismatched}"
    )


print()
print(
    "Exact parameter shapes: PASS"
)


# =============================================================================
# 6. Parameter-count audit by component
# =============================================================================

banner(
    "PARAMETER COUNT AUDIT"
)


attention_parameters = (
    module.attention_weight.numel()
)


gru_parameters = sum(
    parameter.numel()
    for parameter
    in module.gru.parameters()
)


output_parameters = sum(
    parameter.numel()
    for parameter
    in module.output_projection.parameters()
)


total_parameters = (
    attention_parameters
    + gru_parameters
    + output_parameters
)


print(
    f"Bilinear attention:   "
    f"{attention_parameters:,}"
)

print(
    f"2-layer GRU:          "
    f"{gru_parameters:,}"
)

print(
    f"Output projection:    "
    f"{output_parameters:,}"
)

print(
    f"TOTAL:                "
    f"{total_parameters:,}"
)


assert (
    attention_parameters
    == EXPECTED_ATTENTION_PARAMETERS
)


assert (
    gru_parameters
    == EXPECTED_GRU_PARAMETERS
)


assert (
    output_parameters
    == EXPECTED_OUTPUT_PARAMETERS
)


assert (
    total_parameters
    == EXPECTED_TOTAL_PARAMETERS
)


# =============================================================================
# 7. Explicit GRU implementation settings
# =============================================================================

banner(
    "GRU REPRODUCTION SETTINGS"
)


print(
    f"input_size:       "
    f"{module.gru.input_size}"
)

print(
    f"hidden_size:      "
    f"{module.gru.hidden_size}"
)

print(
    f"num_layers:       "
    f"{module.gru.num_layers}"
)

print(
    f"bias:             "
    f"{module.gru.bias}"
)

print(
    f"batch_first:      "
    f"{module.gru.batch_first}"
)

print(
    f"dropout:          "
    f"{module.gru.dropout}"
)

print(
    f"bidirectional:    "
    f"{module.gru.bidirectional}"
)


assert module.gru.input_size == 80
assert module.gru.hidden_size == 40
assert module.gru.num_layers == 2
assert module.gru.bias is True
assert module.gru.batch_first is True
assert module.gru.dropout == 0.0
assert module.gru.bidirectional is False


# =============================================================================
# 8. Initial-hidden-state contract
#
# PyTorch GRU defaults to h0=zeros when h0 is omitted.
#
# We freeze that semantic explicitly rather than relying silently on the API.
# =============================================================================

banner(
    "INITIAL HIDDEN-STATE CONTRACT"
)


initial_hidden_policy = (
    "zero tensor with shape "
    "[2, batch_size, 40]"
)


print(
    f"Policy: {initial_hidden_policy}"
)

print(
    "Learned initial hidden state: NO"
)


# =============================================================================
# 9. Attention semantics
# =============================================================================

banner(
    "ATTENTION SEMANTICS"
)


print(
    "Active period:"
)

print(
    "  query        [80]"
)

print(
    "  items        [n_startups, 80]"
)

print(
    "  W            [80, 80]"
)

print(
    "  scores       [n_startups]"
)

print(
    "  softmax dim  startups"
)

print(
    "  x_i,h        [80]"
)


print()
print(
    "Empty period:"
)

print(
    "  attention evaluated: NO"
)

print(
    "  x_i,h = exact zero_80"
)


# =============================================================================
# 10. Trend target semantics
# =============================================================================

banner(
    "TREND TARGET SEMANTICS"
)


print(
    "Target T0:"
)

print(
    "  no GRU sequence"
)

print(
    "  F_t = exact zero_40"
)


print()
print(
    "Target T1:"
)

print(
    "  sequence = [x_T0]"
)


print()
print(
    "Target T_h:"
)

print(
    "  sequence = "
    "[x_T0, ..., x_T(h-1)]"
)


print()
print(
    "Target T60:"
)

print(
    "  sequence length = 60"
)

print(
    "  periods = T0 ... T59"
)


# =============================================================================
# 11. Freeze contract
# =============================================================================

contract = {

    "phase":
        "4.3.2a",

    "status":
        "FROZEN",

    "component":
        "ITRS trend attention and recurrent architecture",

    "paper_specified": {

        "attention_query":
            (
                "[Investor latent embedding || "
                "Investor description feature]"
            ),

        "attention_item":
            (
                "[Startup latent embedding || "
                "Startup description feature]"
            ),

        "attention_equation":
            "score = u_trend @ W @ V.T",

        "attention_normalization":
            "softmax over startups within the period",

        "attention_output":
            "alpha @ V",

        "recurrent_family":
            "GRU",

        "gru_hidden_dim":
            GRU_HIDDEN_DIM,

        "gru_num_layers":
            GRU_NUM_LAYERS,

        "trend_output_dim":
            TREND_OUTPUT_DIM,

        "trend_output_activation":
            "sigmoid",

        "trend_output_equation":
            "y_t = sigmoid(W_o @ h_t)",

        "trend_for_target_period":
            (
                "last output from the sequence "
                "ending at the preceding period"
            ),
    },

    "upstream_frozen_dimensions": {

        "latent_dim":
            LATENT_DIM,

        "description_dim":
            DESCRIPTION_DIM,

        "attention_query_dim":
            ATTENTION_QUERY_DIM,

        "attention_item_dim":
            ATTENTION_ITEM_DIM,

        "period_vector_dim":
            GRU_INPUT_DIM,
    },

    "paper_grounded_interpretations": {

        "attention_bias":
            False,

        "output_projection_bias":
            False,

        "bidirectional_gru":
            False,

        "reason_bidirectional":
            (
                "Trend is constructed causally from "
                "preceding temporal periods only."
            ),
    },

    "paper_unspecified_reproduction_choices": {

        "gru_implementation":
            "torch.nn.GRU",

        "gru_equation_discrepancy_note":
            (
                "The displayed ITRS GRU candidate-state "
                "equation follows the conventional form "
                "with reset gating before the hidden "
                "transformation. PyTorch nn.GRU applies "
                "the reset gate after its hidden linear "
                "transformation. The reproduction uses "
                "standard torch.nn.GRU because the paper "
                "reports a PyTorch implementation and "
                "does not document a custom GRU."
            ),

        "gru_bias":
            GRU_BIAS,

        "gru_dropout":
            GRU_DROPOUT,

        "batch_first":
            GRU_BATCH_FIRST,

        "initial_hidden_state":
            "zeros",

        "learned_initial_hidden_state":
            False,
    },

    "empty_period_behavior": {

        "attention_called":
            False,

        "period_vector":
            "exact zero_80",

        "period_slot_retained":
            True,
    },

    "t0_behavior": {

        "gru_called":
            False,

        "trend_feature":
            "exact zero_40",
    },

    "parameter_count": {

        "attention":
            EXPECTED_ATTENTION_PARAMETERS,

        "gru":
            EXPECTED_GRU_PARAMETERS,

        "output_projection":
            EXPECTED_OUTPUT_PARAMETERS,

        "total_trend_neural":
            EXPECTED_TOTAL_PARAMETERS,

        "excludes":
            [
                "Investor latent embeddings",
                "Startup latent embeddings",
                "DescriptionEncoder parameters",
            ],
    },

    "initialization": {

        "paper_specified_family":
            "Kaiming",

        "exact_kaiming_variant":
            "NOT_YET_FROZEN",

        "current_torch_default_initialization_accepted_as_model_contract":
            False,

        "status":
            (
                "Exact initialization remains deferred "
                "to the Phase-4 global initialization "
                "contract."
            ),
    },

    "training_performed":
        False,

    "module_state_saved":
        False,

    "upstream_reopened": {

        "phase_2":
            False,

        "phase_3":
            False,

        "phase_4_2":
            False,

        "phase_4_3_1":
            False,
    },
}


contract_path = (
    OUT_DIR
    / "trend_neural_contract.json"
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
# 12. Save parameter audit
# =============================================================================

audit_records = []


for name, parameter in (
    module.named_parameters()
):

    audit_records.append(
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
                parameter.numel(),

            "trainable":
                parameter.requires_grad,
        }
    )


import pandas as pd


audit_df = pd.DataFrame(
    audit_records
)


audit_path = (
    OUT_DIR
    / "trend_neural_parameter_audit.csv"
)


audit_df.to_csv(
    audit_path,
    index=False,
)


# =============================================================================
# Final summary
# =============================================================================

banner(
    "PHASE 4.3.2a FINAL SUMMARY"
)


print(
    "Attention:"
)

print(
    "  query              80"
)

print(
    "  item               80"
)

print(
    "  W                 80 × 80"
)

print(
    "  softmax             over startups"
)

print(
    "  period output       80"
)


print()
print(
    "GRU:"
)

print(
    "  implementation      torch.nn.GRU"
)

print(
    "  input_size          80"
)

print(
    "  hidden_size         40"
)

print(
    "  num_layers           2"
)

print(
    "  bias                YES"
)

print(
    "  dropout             0.0"
)

print(
    "  bidirectional       NO"
)

print(
    "  h0                  zeros"
)


print()
print(
    "Trend projection:"
)

print(
    "  Linear(40,40,bias=False)"
)

print(
    "  Sigmoid"
)

print(
    "  output               40"
)


print()
print(
    f"Attention parameters:   "
    f"{attention_parameters:,}"
)

print(
    f"GRU parameters:         "
    f"{gru_parameters:,}"
)

print(
    f"Output parameters:      "
    f"{output_parameters:,}"
)

print(
    f"Trend module total:     "
    f"{total_parameters:,}"
)


print()
print(
    "PyTorch-vs-equation GRU discrepancy:"
)

print(
    "  explicitly documented"
)

print(
    "  standard torch.nn.GRU chosen"
)


print()
print(
    "Final Kaiming variant frozen: NO"
)

print(
    "Training performed:           NO"
)

print(
    "Model state persisted:        NO"
)


print()
print("Outputs:")

for path in [
    contract_path,
    audit_path,
]:

    print(
        f"  {path}"
    )


print()
print(
    "PHASE 4.3.2a STATUS: COMPLETE — "
    "TREND NEURAL CONTRACT FROZEN"
)