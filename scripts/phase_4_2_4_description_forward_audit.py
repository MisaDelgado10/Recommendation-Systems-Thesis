from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from scipy import sparse

import torch
import torch.nn as nn


# =============================================================================
# PHASE 4.2.4 — DESCRIPTION MODULE IMPLEMENTATION AND FORWARD AUDIT
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

NEURAL_CONTRACT_PATH = Path(
    "data/experimental/phase_4/"
    "description_neural_contract/"
    "description_neural_contract.json"
)

OUT_DIR = Path(
    "data/experimental/phase_4/"
    "description_module"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# Frozen dimensions
# =============================================================================

ROLE_NODES = 477_564

TEXT_INPUT_DIM = 32
LABEL_INPUT_DIM = 802

TEXT_OUTPUT_DIM = 20
LABEL_OUTPUT_DIM = 20

DESCRIPTION_DIM = 40

EXPECTED_PARAMETERS = 16_720


# =============================================================================
# Helpers
# =============================================================================

def banner(title):

    print()
    print("=" * 120)
    print(title)
    print("=" * 120)


# =============================================================================
# Description module
# =============================================================================

class DescriptionEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.text_projection = nn.Linear(
            TEXT_INPUT_DIM,
            TEXT_OUTPUT_DIM,
            bias=True,
        )

        self.label_projection = nn.Linear(
            LABEL_INPUT_DIM,
            LABEL_OUTPUT_DIM,
            bias=True,
        )

        self.activation = nn.ReLU()


    def forward(
        self,
        text_features,
        label_features,
    ):

        if text_features.ndim != 2:
            raise ValueError(
                "text_features must be rank 2."
            )

        if label_features.ndim != 2:
            raise ValueError(
                "label_features must be rank 2."
            )

        if (
            text_features.shape[0]
            != label_features.shape[0]
        ):
            raise ValueError(
                "Text and label batch sizes differ."
            )

        if (
            text_features.shape[1]
            != TEXT_INPUT_DIM
        ):
            raise ValueError(
                "Unexpected text input dimension."
            )

        if (
            label_features.shape[1]
            != LABEL_INPUT_DIM
        ):
            raise ValueError(
                "Unexpected label input dimension."
            )


        text_output = self.activation(
            self.text_projection(
                text_features
            )
        )


        label_output = self.activation(
            self.label_projection(
                label_features
            )
        )


        description_output = torch.cat(
            [
                text_output,
                label_output,
            ],
            dim=1,
        )


        return {
            "text": text_output,
            "labels": label_output,
            "description": description_output,
        }


# =============================================================================
# 1. Environment
# =============================================================================

banner(
    "PHASE 4.2.4 — "
    "DESCRIPTION MODULE IMPLEMENTATION AND FORWARD AUDIT"
)


print("\nENVIRONMENT")
print("-" * 120)

print(
    f"Python:   "
    f"{sys.version.splitlines()[0]}"
)

print(
    f"PyTorch:  "
    f"{torch.__version__}"
)

print(
    f"Device:   CPU"
)


# We intentionally perform this audit on CPU.
device = torch.device(
    "cpu"
)


# =============================================================================
# 2. Verify frozen neural contract
# =============================================================================

banner(
    "LOADING FROZEN DESCRIPTION NEURAL CONTRACT"
)


with open(
    NEURAL_CONTRACT_PATH,
    "r",
    encoding="utf-8",
) as f:

    contract = json.load(f)


if (
    contract.get("status")
    != "FROZEN"
):

    raise AssertionError(
        "Description neural contract "
        "is not frozen."
    )


paper_specified = contract[
    "paper_specified"
]

reproduction = contract[
    "paper_unspecified_reproduction_choices"
]


assert (
    paper_specified[
        "final_description_dim"
    ]
    == DESCRIPTION_DIM
)


assert (
    reproduction[
        "text_branch"
    ][
        "input_dim"
    ]
    == TEXT_INPUT_DIM
)


assert (
    reproduction[
        "text_branch"
    ][
        "output_dim"
    ]
    == TEXT_OUTPUT_DIM
)


assert (
    reproduction[
        "label_branch"
    ][
        "input_dim"
    ]
    == LABEL_INPUT_DIM
)


assert (
    reproduction[
        "label_branch"
    ][
        "output_dim"
    ]
    == LABEL_OUTPUT_DIM
)


print(
    "Frozen architecture contract: PASS"
)


# =============================================================================
# 3. Load fixed description inputs
# =============================================================================

banner(
    "LOADING FIXED DESCRIPTION INPUTS"
)


doc2vec = np.load(
    DOC2VEC_VECTOR_PATH,
    mmap_mode="r",
)


labels = sparse.load_npz(
    LABEL_MATRIX_PATH
)


if doc2vec.shape != (
    ROLE_NODES,
    TEXT_INPUT_DIM,
):

    raise AssertionError(
        "Doc2Vec matrix shape changed."
    )


if labels.shape != (
    ROLE_NODES,
    LABEL_INPUT_DIM,
):

    raise AssertionError(
        "Label matrix shape changed."
    )


print(
    f"Doc2Vec shape: "
    f"{doc2vec.shape}"
)

print(
    f"Label shape:   "
    f"{labels.shape}"
)


# =============================================================================
# 4. Verify feature-row alignment again
# =============================================================================

banner(
    "FEATURE ROW ALIGNMENT"
)


text_manifest = pd.read_parquet(
    DOC2VEC_MANIFEST_PATH,
)


label_manifest = pd.read_parquet(
    LABEL_MANIFEST_PATH,
)


if len(text_manifest) != ROLE_NODES:
    raise AssertionError(
        "Doc2Vec manifest size changed."
    )


if len(label_manifest) != ROLE_NODES:
    raise AssertionError(
        "Label manifest size changed."
    )


same_nodes = np.array_equal(
    text_manifest[
        "node_id"
    ].astype(str).to_numpy(),

    label_manifest[
        "node_id"
    ].astype(str).to_numpy(),
)


same_types = np.array_equal(
    text_manifest[
        "node_type"
    ].astype(str).to_numpy(),

    label_manifest[
        "node_type"
    ].astype(str).to_numpy(),
)


if not (
    same_nodes
    and same_types
):

    raise AssertionError(
        "Description input row alignment changed."
    )


print(
    f"Same node order: "
    f"{same_nodes}"
)

print(
    f"Same role order: "
    f"{same_types}"
)


# =============================================================================
# 5. Instantiate ONE shared DescriptionEncoder
#
# The same object will be used for Investor and Startup batches.
# =============================================================================

banner(
    "DESCRIPTION ENCODER INSTANTIATION"
)


model = DescriptionEncoder().to(
    device
)


parameter_count = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


print(model)

print()
print(
    f"Trainable parameters: "
    f"{parameter_count:,}"
)


if (
    parameter_count
    != EXPECTED_PARAMETERS
):

    raise AssertionError(
        "Description parameter count changed."
    )


# =============================================================================
# 6. AUDIT-ONLY deterministic initialization
#
# IMPORTANT:
# These weights are NOT the final ITRS initialization.
# They are not saved.
# Kaiming initialization remains unfrozen.
# =============================================================================

banner(
    "AUDIT-ONLY DETERMINISTIC WEIGHTS"
)


with torch.no_grad():

    model.text_projection.weight.fill_(
        0.01
    )

    model.text_projection.bias.fill_(
        0.001
    )

    model.label_projection.weight.fill_(
        0.01
    )

    model.label_projection.bias.fill_(
        0.001
    )


print(
    "Temporary deterministic audit "
    "weights installed."
)

print(
    "State dict will NOT be saved."
)


# =============================================================================
# 7. Batch helper
# =============================================================================

def build_batch(
    rows,
):

    rows = np.asarray(
        rows,
        dtype=np.int64,
    )


    text_np = np.asarray(
        doc2vec[
            rows
        ],
        dtype=np.float32,
    )


    # Densify only selected CSR rows.
    label_np = (
        labels[
            rows
        ]
        .toarray()
        .astype(
            np.float32,
            copy=False,
        )
    )


    text_tensor = torch.from_numpy(
        text_np
    ).to(
        device
    )


    label_tensor = torch.from_numpy(
        label_np
    ).to(
        device
    )


    return (
        text_tensor,
        label_tensor,
    )


# =============================================================================
# 8. Build representative audit batches
# =============================================================================

banner(
    "REPRESENTATIVE BATCH SELECTION"
)


node_types = (
    text_manifest[
        "node_type"
    ]
    .astype(str)
    .to_numpy()
)


text_zero = (
    text_manifest[
        "doc2vec_zero_vector"
    ]
    .to_numpy(
        dtype=bool
    )
)


label_zero = (
    label_manifest[
        "label_count"
    ]
    .to_numpy(
        dtype=np.int64
    )
    == 0
)


investor_rows = np.flatnonzero(
    node_types
    == "investor"
)


startup_rows = np.flatnonzero(
    node_types
    == "startup"
)


text_zero_rows = np.flatnonzero(
    text_zero
)


label_zero_rows = np.flatnonzero(
    label_zero
)


both_present_rows = np.flatnonzero(
    (~text_zero)
    & (~label_zero)
)


both_zero_rows = np.flatnonzero(
    text_zero
    & label_zero
)


print(
    f"Investor rows available: "
    f"{len(investor_rows):,}"
)

print(
    f"Startup rows available:  "
    f"{len(startup_rows):,}"
)

print(
    f"Zero-text rows:           "
    f"{len(text_zero_rows):,}"
)

print(
    f"Zero-label rows:          "
    f"{len(label_zero_rows):,}"
)

print(
    f"Both inputs present:      "
    f"{len(both_present_rows):,}"
)

print(
    f"Both inputs zero:         "
    f"{len(both_zero_rows):,}"
)


# Deterministic row selections.
audit_batches = {
    "investor_only":
        investor_rows[:32],

    "startup_only":
        startup_rows[:32],

    "text_zero":
        text_zero_rows[:32],

    "label_zero":
        label_zero_rows[:32],

    "both_present":
        both_present_rows[:32],
}


# Both-zero may theoretically be scarce.
if len(both_zero_rows) > 0:

    audit_batches[
        "both_zero"
    ] = (
        both_zero_rows[
            :min(
                32,
                len(
                    both_zero_rows
                ),
            )
        ]
    )


# =============================================================================
# 9. Forward-shape audit
# =============================================================================

banner(
    "FORWARD-SHAPE AUDIT"
)


audit_records = []


model.eval()


with torch.no_grad():

    for batch_name, rows in (
        audit_batches.items()
    ):

        text_batch, label_batch = (
            build_batch(
                rows
            )
        )


        outputs = model(
            text_batch,
            label_batch,
        )


        text_output = outputs[
            "text"
        ]

        label_output = outputs[
            "labels"
        ]

        description_output = outputs[
            "description"
        ]


        batch_size = len(
            rows
        )


        expected_text_shape = (
            batch_size,
            TEXT_OUTPUT_DIM,
        )

        expected_label_shape = (
            batch_size,
            LABEL_OUTPUT_DIM,
        )

        expected_description_shape = (
            batch_size,
            DESCRIPTION_DIM,
        )


        if (
            tuple(
                text_output.shape
            )
            != expected_text_shape
        ):

            raise AssertionError(
                f"{batch_name}: "
                "text output shape mismatch."
            )


        if (
            tuple(
                label_output.shape
            )
            != expected_label_shape
        ):

            raise AssertionError(
                f"{batch_name}: "
                "label output shape mismatch."
            )


        if (
            tuple(
                description_output.shape
            )
            != expected_description_shape
        ):

            raise AssertionError(
                f"{batch_name}: "
                "description output shape mismatch."
            )


        if not torch.isfinite(
            description_output
        ).all():

            raise AssertionError(
                f"{batch_name}: "
                "non-finite output."
            )


        if torch.any(
            description_output
            < 0
        ):

            raise AssertionError(
                f"{batch_name}: "
                "ReLU output contains "
                "negative values."
            )


        concat_exact = torch.equal(
            description_output,

            torch.cat(
                [
                    text_output,
                    label_output,
                ],
                dim=1,
            ),
        )


        if not concat_exact:

            raise AssertionError(
                f"{batch_name}: "
                "concatenation mismatch."
            )


        audit_records.append(
            {
                "batch":
                    batch_name,

                "batch_size":
                    batch_size,

                "text_input_shape":
                    str(
                        tuple(
                            text_batch.shape
                        )
                    ),

                "label_input_shape":
                    str(
                        tuple(
                            label_batch.shape
                        )
                    ),

                "text_output_shape":
                    str(
                        tuple(
                            text_output.shape
                        )
                    ),

                "label_output_shape":
                    str(
                        tuple(
                            label_output.shape
                        )
                    ),

                "description_shape":
                    str(
                        tuple(
                            description_output.shape
                        )
                    ),

                "finite":
                    True,

                "relu_nonnegative":
                    True,

                "concat_exact":
                    True,
            }
        )


        print()
        print(
            f"{batch_name}:"
        )

        print(
            f"  text input:   "
            f"{tuple(text_batch.shape)}"
        )

        print(
            f"  label input:  "
            f"{tuple(label_batch.shape)}"
        )

        print(
            f"  text output:  "
            f"{tuple(text_output.shape)}"
        )

        print(
            f"  label output: "
            f"{tuple(label_output.shape)}"
        )

        print(
            f"  description:  "
            f"{tuple(description_output.shape)}"
        )

        print(
            f"  finite:       True"
        )

        print(
            f"  concat exact: True"
        )


# =============================================================================
# 10. Verify shared-module behavior across roles
#
# We use one module object. There are no role-specific parameters.
# =============================================================================

banner(
    "PARAMETER-SHARING AUDIT"
)


named_parameters = dict(
    model.named_parameters()
)


expected_parameter_names = {
    "text_projection.weight",
    "text_projection.bias",
    "label_projection.weight",
    "label_projection.bias",
}


actual_parameter_names = set(
    named_parameters
)


print(
    "Parameter names:"
)

for name in sorted(
    actual_parameter_names
):

    print(
        f"  {name}"
    )


if (
    actual_parameter_names
    != expected_parameter_names
):

    raise AssertionError(
        "Unexpected description parameters."
    )


print()
print(
    "Role-specific description "
    "parameters found: NO"
)

print(
    "Shared encoder object: PASS"
)


# =============================================================================
# 11. Zero-input behavior audit
#
# With bias=True, zero inputs are allowed to produce a common learned baseline.
# We verify this behavior instead of masking the MLP output.
# =============================================================================

banner(
    "ZERO-INPUT BEHAVIOR AUDIT"
)


zero_text_input = torch.zeros(
    (
        4,
        TEXT_INPUT_DIM,
    ),
    dtype=torch.float32,
)


zero_label_input = torch.zeros(
    (
        4,
        LABEL_INPUT_DIM,
    ),
    dtype=torch.float32,
)


with torch.no_grad():

    zero_outputs = model(
        zero_text_input,
        zero_label_input,
    )


print(
    "Zero-input description shape:",
    tuple(
        zero_outputs[
            "description"
        ].shape
    ),
)


all_rows_equal = torch.allclose(
    zero_outputs[
        "description"
    ][0:1].expand_as(
        zero_outputs[
            "description"
        ]
    ),

    zero_outputs[
        "description"
    ],
)


print(
    "All zero-input rows share "
    "same baseline:",
    all_rows_equal,
)


if not all_rows_equal:

    raise AssertionError(
        "Zero-input baseline behavior "
        "is inconsistent."
    )


# With temporary positive bias,
# output should be non-zero.
zero_output_nonzero = bool(
    torch.any(
        zero_outputs[
            "description"
        ]
        != 0
    )
)


print(
    "Zero input may yield nonzero "
    "bias baseline:",
    zero_output_nonzero,
)


if not zero_output_nonzero:

    raise AssertionError(
        "Audit-only bias baseline "
        "was unexpectedly zero."
    )


# =============================================================================
# 12. Save implementation audit
# =============================================================================

audit_df = pd.DataFrame(
    audit_records
)


audit_path = (
    OUT_DIR
    / "description_forward_audit.csv"
)


audit_df.to_csv(
    audit_path,
    index=False,
)


metadata = {
    "phase":
        "4.2.4",

    "status":
        "COMPLETE",

    "component":
        "DescriptionEncoder forward implementation",

    "architecture": {
        "text_branch":
            "Linear(32,20)+ReLU",

        "label_branch":
            "Linear(802,20)+ReLU",

        "concatenation":
            "[text || labels]",

        "output_dim":
            40,

        "trainable_parameters":
            EXPECTED_PARAMETERS,
    },

    "parameter_sharing": {
        "one_description_encoder":
            True,

        "role_specific_parameters":
            False,
    },

    "input_storage": {
        "doc2vec":
            "dense float32 NumPy",

        "labels":
            "CSR sparse float32",

        "label_batch_conversion":
            (
                "selected CSR rows are "
                "densified per batch only"
            ),
    },

    "initialization": {
        "audit_weights":
            (
                "temporary deterministic "
                "constant values"
            ),

        "audit_weights_saved":
            False,

        "final_kaiming_variant_frozen":
            False,
    },

    "training_performed":
        False,

    "model_state_saved":
        False,

    "description_features_materialized":
        False,

    "upstream_contracts_reopened":
        False,
}


metadata_path = (
    OUT_DIR
    / "description_forward_audit_metadata.json"
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
# Final summary
# =============================================================================

banner(
    "PHASE 4.2.4 FINAL SUMMARY"
)


print(
    "DescriptionEncoder:"
)

print(
    "  text:   Linear(32,20) + ReLU"
)

print(
    "  labels: Linear(802,20) + ReLU"
)

print(
    "  concat: 20 + 20 = 40"
)


print()
print(
    f"Trainable parameters: "
    f"{parameter_count:,}"
)


print()
print(
    "Text/label row alignment:     PASS"
)

print(
    "Sparse batch materialization: PASS"
)

print(
    "Forward shapes:               PASS"
)

print(
    "ReLU semantics:               PASS"
)

print(
    "Concatenation order:          PASS"
)

print(
    "Role parameter sharing:       PASS"
)

print(
    "Zero-input behavior:          PASS"
)


print()
print(
    "Training performed:           NO"
)

print(
    "Audit weights persisted:      NO"
)

print(
    "Final Kaiming variant frozen: NO"
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
    "PHASE 4.2.4 STATUS: COMPLETE — "
    "DESCRIPTION FORWARD CONTRACT VERIFIED"
)