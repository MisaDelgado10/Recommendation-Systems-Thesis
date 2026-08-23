from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import sparse


# =============================================================================
# PHASE 4.2.3b — FREEZE DESCRIPTION NEURAL ARCHITECTURE
# =============================================================================

DOC2VEC_VECTOR_PATH = Path(
    "data/experimental/phase_4/"
    "doc2vec/vectors/"
    "doc2vec_vectors_all.npy"
)

DOC2VEC_MANIFEST_PATH = Path(
    "data/experimental/phase_4/"
    "doc2vec/vectors/"
    "doc2vec_vector_manifest.parquet"
)

LABEL_MATRIX_PATH = Path(
    "data/experimental/phase_4/"
    "description_labels/"
    "description_label_multihot.npz"
)

LABEL_MANIFEST_PATH = Path(
    "data/experimental/phase_4/"
    "description_labels/"
    "description_label_vector_manifest.parquet"
)

OUT_DIR = Path(
    "data/experimental/phase_4/"
    "description_neural_contract"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# Frozen upstream dimensions
# =============================================================================

ROLE_NODES = 477_564

DOC2VEC_INPUT_DIM = 32
LABEL_INPUT_DIM = 802

DESCRIPTION_DIM = 40


# =============================================================================
# Phase-4.2.3b reproduction choice
# =============================================================================

TEXT_OUTPUT_DIM = 20
LABEL_OUTPUT_DIM = 20


assert (
    TEXT_OUTPUT_DIM
    + LABEL_OUTPUT_DIM
    == DESCRIPTION_DIM
)


# =============================================================================
# Helpers
# =============================================================================

def banner(title):

    print()
    print("=" * 118)
    print(title)
    print("=" * 118)


# =============================================================================
# 1. Load upstream feature inputs
# =============================================================================

banner(
    "PHASE 4.2.3b — "
    "FREEZE DESCRIPTION NEURAL ARCHITECTURE"
)


doc2vec = np.load(
    DOC2VEC_VECTOR_PATH,
    mmap_mode="r",
)

labels = sparse.load_npz(
    LABEL_MATRIX_PATH
)


print(
    f"Doc2Vec shape: "
    f"{doc2vec.shape}"
)

print(
    f"Label shape:   "
    f"{labels.shape}"
)


if doc2vec.shape != (
    ROLE_NODES,
    DOC2VEC_INPUT_DIM,
):

    raise AssertionError(
        "Frozen Doc2Vec input shape changed."
    )


if labels.shape != (
    ROLE_NODES,
    LABEL_INPUT_DIM,
):

    raise AssertionError(
        "Frozen label input shape changed."
    )


# =============================================================================
# 2. Verify exact feature-row alignment
# =============================================================================

banner(
    "UPSTREAM FEATURE ROW ALIGNMENT"
)


text_manifest = pd.read_parquet(
    DOC2VEC_MANIFEST_PATH,
    columns=[
        "doc2vec_feature_row",
        "node_id",
        "node_type",
        "raw_entity_id",
    ],
)


label_manifest = pd.read_parquet(
    LABEL_MANIFEST_PATH,
    columns=[
        "label_feature_row",
        "node_id",
        "node_type",
        "raw_entity_id",
    ],
)


if len(text_manifest) != ROLE_NODES:
    raise AssertionError(
        "Doc2Vec manifest population changed."
    )


if len(label_manifest) != ROLE_NODES:
    raise AssertionError(
        "Label manifest population changed."
    )


same_node_order = np.array_equal(
    text_manifest[
        "node_id"
    ].astype(str).to_numpy(),

    label_manifest[
        "node_id"
    ].astype(str).to_numpy(),
)


same_type_order = np.array_equal(
    text_manifest[
        "node_type"
    ].astype(str).to_numpy(),

    label_manifest[
        "node_type"
    ].astype(str).to_numpy(),
)


same_raw_order = np.array_equal(
    text_manifest[
        "raw_entity_id"
    ].astype(str).to_numpy(),

    label_manifest[
        "raw_entity_id"
    ].astype(str).to_numpy(),
)


text_rows_contiguous = np.array_equal(
    text_manifest[
        "doc2vec_feature_row"
    ].to_numpy(
        dtype=np.int64
    ),

    np.arange(
        ROLE_NODES,
        dtype=np.int64,
    ),
)


label_rows_contiguous = np.array_equal(
    label_manifest[
        "label_feature_row"
    ].to_numpy(
        dtype=np.int64
    ),

    np.arange(
        ROLE_NODES,
        dtype=np.int64,
    ),
)


print(
    f"Same node order:          "
    f"{same_node_order}"
)

print(
    f"Same node-type order:     "
    f"{same_type_order}"
)

print(
    f"Same raw-ID order:        "
    f"{same_raw_order}"
)

print(
    f"Doc2Vec rows contiguous:  "
    f"{text_rows_contiguous}"
)

print(
    f"Label rows contiguous:    "
    f"{label_rows_contiguous}"
)


if not (
    same_node_order
    and same_type_order
    and same_raw_order
    and text_rows_contiguous
    and label_rows_contiguous
):

    raise AssertionError(
        "Description text and label "
        "inputs are not exactly aligned."
    )


# =============================================================================
# 3. Architecture dimension audit
# =============================================================================

banner(
    "DESCRIPTION ARCHITECTURE DIMENSIONS"
)


print(
    "Text branch:"
)

print(
    f"  {DOC2VEC_INPUT_DIM} "
    f"-> Linear -> "
    f"{TEXT_OUTPUT_DIM} "
    f"-> ReLU"
)


print()
print(
    "Label branch:"
)

print(
    f"  {LABEL_INPUT_DIM} "
    f"-> Linear -> "
    f"{LABEL_OUTPUT_DIM} "
    f"-> ReLU"
)


print()
print(
    "Concatenation:"
)

print(
    f"  {TEXT_OUTPUT_DIM} "
    f"+ {LABEL_OUTPUT_DIM} "
    f"= {DESCRIPTION_DIM}"
)


if (
    TEXT_OUTPUT_DIM
    + LABEL_OUTPUT_DIM
    != DESCRIPTION_DIM
):

    raise AssertionError(
        "Description branches do not "
        "sum to frozen 40 dimensions."
    )


# =============================================================================
# 4. Parameter-count audit
#
# Shared across Investor and Startup roles.
# Each branch uses one affine projection with bias.
# =============================================================================

banner(
    "TRAINABLE PARAMETER COUNT"
)


text_weight_parameters = (
    DOC2VEC_INPUT_DIM
    * TEXT_OUTPUT_DIM
)

text_bias_parameters = (
    TEXT_OUTPUT_DIM
)

text_parameters = (
    text_weight_parameters
    + text_bias_parameters
)


label_weight_parameters = (
    LABEL_INPUT_DIM
    * LABEL_OUTPUT_DIM
)

label_bias_parameters = (
    LABEL_OUTPUT_DIM
)

label_parameters = (
    label_weight_parameters
    + label_bias_parameters
)


total_parameters = (
    text_parameters
    + label_parameters
)


print(
    f"MLP_text parameters:   "
    f"{text_parameters:,}"
)

print(
    f"MLP_labels parameters: "
    f"{label_parameters:,}"
)

print(
    f"Total description-MLP "
    f"parameters: "
    f"{total_parameters:,}"
)


assert text_parameters == 660
assert label_parameters == 16_060
assert total_parameters == 16_720


# =============================================================================
# 5. Freeze contract
# =============================================================================

contract = {
    "phase":
        "4.2.3b",

    "status":
        "FROZEN",

    "component":
        "ITRS description neural architecture",

    "paper_specified": {
        "text_input":
            "pretrained Doc2Vec representation",

        "labels_input":
            "binary label indicator representation",

        "text_transformation":
            "MLP_text",

        "labels_transformation":
            "MLP_labels",

        "concatenation_order": [
            "text_branch",
            "label_branch",
        ],

        "final_description_dim":
            DESCRIPTION_DIM,

        "mlp_activation":
            "ReLU",
    },

    "paper_grounded_architecture_interpretation": {
        "shared_text_mlp_across_roles":
            True,

        "shared_label_mlp_across_roles":
            True,

        "reason":
            (
                "The ITRS equations denote the same "
                "MLP_text function for organization and "
                "brand text, and the same MLP_labels "
                "function for both label inputs; no "
                "role-specific description MLPs are "
                "defined."
            ),
    },

    "paper_unspecified_reproduction_choices": {
        "text_branch": {
            "input_dim":
                DOC2VEC_INPUT_DIM,

            "output_dim":
                TEXT_OUTPUT_DIM,

            "architecture": [
                {
                    "layer":
                        "Linear",

                    "in_features":
                        DOC2VEC_INPUT_DIM,

                    "out_features":
                        TEXT_OUTPUT_DIM,

                    "bias":
                        True,
                },
                {
                    "layer":
                        "ReLU",
                },
            ],
        },

        "label_branch": {
            "input_dim":
                LABEL_INPUT_DIM,

            "output_dim":
                LABEL_OUTPUT_DIM,

            "architecture": [
                {
                    "layer":
                        "Linear",

                    "in_features":
                        LABEL_INPUT_DIM,

                    "out_features":
                        LABEL_OUTPUT_DIM,

                    "bias":
                        True,
                },
                {
                    "layer":
                        "ReLU",
                },
            ],
        },

        "branch_dimension_split":
            "20 text + 20 labels",

        "dropout":
            False,

        "batch_normalization":
            False,

        "residual_connection":
            False,
    },

    "parameter_sharing": {
        "text_mlp":
            (
                "one shared trainable module "
                "for Investor and Startup"
            ),

        "label_mlp":
            (
                "one shared trainable module "
                "for Investor and Startup"
            ),

        "text_and_label_modules_shared_with_each_other":
            False,
    },

    "trainability": {
        "doc2vec_vectors":
            False,

        "label_inputs":
            False,

        "text_mlp":
            True,

        "label_mlp":
            True,

        "description_features_precomputed":
            False,

        "training_mode":
            (
                "MLP_text and MLP_labels are trained "
                "jointly with the downstream ITRS "
                "recommendation objective."
            ),
    },

    "missing_input_behavior": {
        "text":
            (
                "Entities without usable text retain "
                "the frozen zero 32-dimensional "
                "Doc2Vec input. No post-MLP mask is "
                "applied."
            ),

        "labels":
            (
                "Entities without labels retain the "
                "frozen all-zero 802-dimensional "
                "multi-hot input. No post-MLP mask is "
                "applied."
            ),

        "bias_implication":
            (
                "Because the trainable projections use "
                "bias, a zero input may acquire a common "
                "learned baseline representation after "
                "the MLP."
            ),
    },

    "parameter_count": {
        "text_mlp":
            text_parameters,

        "label_mlp":
            label_parameters,

        "total":
            total_parameters,
    },

    "initialization": {
        "paper_specified_family":
            "Kaiming",

        "exact_kaiming_variant":
            "NOT_YET_FROZEN",

        "status":
            (
                "Exact PyTorch Kaiming variant and "
                "bias initialization will be frozen "
                "with the global Phase-4 model "
                "initialization contract."
            ),
    },

    "upstream_inputs": {
        "doc2vec_shape": [
            ROLE_NODES,
            DOC2VEC_INPUT_DIM,
        ],

        "label_shape": [
            ROLE_NODES,
            LABEL_INPUT_DIM,
        ],

        "row_alignment":
            "exact",
    },

    "not_reopened": {
        "phase_2":
            True,

        "phase_3":
            True,

        "phase_4_1_3":
            True,

        "doc2vec_contract":
            True,

        "label_input_contract":
            True,
    },
}


contract_path = (
    OUT_DIR
    / "description_neural_contract.json"
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
# 6. Dimension audit
# =============================================================================

audit_df = pd.DataFrame(
    [
        {
            "component":
                "text_input",
            "dimension":
                DOC2VEC_INPUT_DIM,
        },
        {
            "component":
                "text_branch_output",
            "dimension":
                TEXT_OUTPUT_DIM,
        },
        {
            "component":
                "label_input",
            "dimension":
                LABEL_INPUT_DIM,
        },
        {
            "component":
                "label_branch_output",
            "dimension":
                LABEL_OUTPUT_DIM,
        },
        {
            "component":
                "final_description",
            "dimension":
                DESCRIPTION_DIM,
        },
        {
            "component":
                "trainable_description_parameters",
            "dimension":
                total_parameters,
        },
    ]
)


audit_path = (
    OUT_DIR
    / "description_neural_dimension_audit.csv"
)


audit_df.to_csv(
    audit_path,
    index=False,
)


# =============================================================================
# Final
# =============================================================================

banner(
    "PHASE 4.2.3b FINAL SUMMARY"
)


print(
    "MLP_text:"
)

print(
    "  shared across roles"
)

print(
    "  Linear(32, 20)"
)

print(
    "  ReLU"
)


print()
print(
    "MLP_labels:"
)

print(
    "  shared across roles"
)

print(
    "  Linear(802, 20)"
)

print(
    "  ReLU"
)


print()
print(
    "Final description feature:"
)

print(
    "  [text_20 || labels_20]"
)

print(
    "  dimension = 40"
)


print()
print(
    f"Trainable description "
    f"parameters: "
    f"{total_parameters:,}"
)


print()
print(
    "Dropout:                  NO"
)

print(
    "BatchNorm:                NO"
)

print(
    "Residual connections:     NO"
)


print()
print(
    "Description MLP weights "
    "trained now:             NO"
)

print(
    "Final description "
    "features materialized:   NO"
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
    "PHASE 4.2.3b STATUS: COMPLETE — "
    "DESCRIPTION NEURAL CONTRACT FROZEN"
)