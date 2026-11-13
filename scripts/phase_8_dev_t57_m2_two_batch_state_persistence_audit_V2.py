#!/usr/bin/env python3

from __future__ import annotations

import ast
import hashlib
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path(
    "scripts/"
    "phase_8_2_7_dev_t57_frozen_batch_one_step_preflight_V1.py"
)

REFERENCE = Path(
    "data/experimental/phase_8/"
    "phase_8_2_framework_development/"
    "phase_8_2_7_autograd_optimizer_preflight/"
    "phase_8_2_7_dev_t57_frozen_batch_one_step_results_V1.json"
)

STREAM_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_2_framework_development/"
    "phase_8_2_6_training_stream_dry_run/"
    "DEV_T57/epoch_0"
)

POSITIVE_ORDER = (
    STREAM_DIR
    / "canonical_training_positive_event_order.parquet"
)

NEGATIVE_MATRIX = (
    STREAM_DIR
    / "epoch_0_training_negative_startup_local.npy"
)

EPOCH_ORDER = (
    STREAM_DIR
    / "epoch_0_training_example_order.npy"
)

FROZEN_FIRST_BATCH = (
    STREAM_DIR
    / "epoch_0_first_batch_manifest.parquet"
)

OUTPUT_DIR = Path(
    "data/experimental/phase_8/"
    "comparative_development_training_preflight/"
    "dev_t57_m2_two_batch_state_persistence_V2"
)

OUTPUT_RESULT = (
    OUTPUT_DIR
    / "phase_8_dev_t57_m2_two_batch_state_persistence_V2.json"
)

OUTPUT_HASHES = (
    OUTPUT_DIR
    / "phase_8_dev_t57_m2_two_batch_state_persistence_sha256_V2.json"
)

EXPECTED_SOURCE_SHA256 = (
    "06cfea074a447a11cbdc83c7c53fa213"
    "f3f870730440d69c84b25039b19696a3"
)

FAMILY = "M2"
BATCH_SIZE = 512
SLOTS = 5
NUM_INVESTORS = 165_975


def require(condition, message):

    if not bool(condition):
        raise RuntimeError(message)


def sha256_file(path):

    h = hashlib.sha256()

    with Path(path).open("rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


print("=" * 120)
print(
    "PHASE 8 — DEV_T57 M2 TWO-BATCH "
    "STATE-PERSISTENCE AUDIT V2"
)
print("=" * 120)

print()
print("Version history:")
print(
    "  V1: failed safely during generated-code compilation; "
    "no model was instantiated and no optimizer was created."
)
print(
    "  Cause: fixed source-line slicing bisected "
    "a multiline Python statement."
)
print(
    "  V2: complete-statement AST extraction."
)
print()

print("Scientific role:")
print(
    "  Verify the ONLY new retained-training transition:"
)
print(
    "  batch 0 -> retain model/Adam -> batch 1."
)
print()
print(
    "  NO full epoch, NO ranking metrics, "
    "NO model selection, NO T60, NO April."
)


# =============================================================================
# 1. AUTHORITATIVE INPUTS
# =============================================================================

print()
print("1. AUTHORITATIVE INPUTS")

for path in [
    SOURCE,
    REFERENCE,
    POSITIVE_ORDER,
    NEGATIVE_MATRIX,
    EPOCH_ORDER,
    FROZEN_FIRST_BATCH,
]:

    require(
        path.exists(),
        f"Missing authoritative input: {path}",
    )


source_sha = sha256_file(
    SOURCE
)

require(
    source_sha
    == EXPECTED_SOURCE_SHA256,
    (
        "Authoritative 8.2.7 source changed.\n"
        f"expected={EXPECTED_SOURCE_SHA256}\n"
        f"observed={source_sha}"
    ),
)

reference = json.loads(
    REFERENCE.read_text(
        encoding="utf-8"
    )
)

reference_m2 = reference[
    "families"
][
    FAMILY
]

print(
    f"  8.2.7 source SHA:       {source_sha} PASS"
)
print(
    "  Frozen M2 reference:    PASS"
)


# =============================================================================
# 2. LOAD FROZEN EPOCH-0 STREAM
# =============================================================================

print()
print("2. FROZEN DEV_T57 EPOCH-0 STREAM")

positives = pd.read_parquet(
    POSITIVE_ORDER
).sort_values(
    "positive_order_index",
    kind="mergesort",
).reset_index(
    drop=True
)

negative_matrix = np.load(
    NEGATIVE_MATRIX,
    mmap_mode="r",
)

epoch_order = np.load(
    EPOCH_ORDER,
    mmap_mode="r",
)

require(
    negative_matrix.shape
    == (
        len(positives),
        4,
    ),
    "Negative matrix shape mismatch.",
)

require(
    len(epoch_order)
    == len(positives) * SLOTS,
    "Epoch example-order length mismatch.",
)

require(
    np.unique(
        np.asarray(
            epoch_order,
            dtype=np.int64,
        )
    ).size
    == len(epoch_order),
    "Epoch order is not a permutation.",
)

print(
    f"  Positives:               {len(positives):,}"
)
print(
    f"  Negatives:               {negative_matrix.size:,}"
)
print(
    f"  Conceptual examples:     {len(epoch_order):,}"
)


# =============================================================================
# 3. GENERALIZED FROZEN BATCH BUILDER
# =============================================================================

def build_batch(batch_index):

    start = (
        int(batch_index)
        * BATCH_SIZE
    )

    stop = min(
        start + BATCH_SIZE,
        len(epoch_order),
    )

    serialized = np.asarray(
        epoch_order[
            start:
            stop
        ],
        dtype=np.int64,
    )

    rows = []

    for batch_position, serialized_index in enumerate(
        serialized.tolist()
    ):

        positive_index = (
            int(serialized_index)
            // SLOTS
        )

        slot = (
            int(serialized_index)
            % SLOTS
        )

        positive_row = positives.iloc[
            positive_index
        ]

        investor_global = int(
            positive_row[
                "investor_global"
            ]
        )

        focal_startup_local = int(
            positive_row[
                "startup_local"
            ]
        )

        h = int(
            positive_row[
                "segment_number"
            ]
        )

        if slot == 0:

            label = 1

            startup_local = (
                focal_startup_local
            )

            negative_draw_index = (
                np.nan
            )

        else:

            label = 0

            negative_draw_index = (
                slot - 1
            )

            startup_local = int(
                negative_matrix[
                    positive_index,
                    negative_draw_index,
                ]
            )

        startup_global = (
            NUM_INVESTORS
            + startup_local
        )

        rows.append(
            {
                "batch_position":
                    int(
                        batch_position
                    ),

                "serialized_example_index":
                    int(
                        serialized_index
                    ),

                "positive_order_index":
                    int(
                        positive_index
                    ),

                "example_slot":
                    int(
                        slot
                    ),

                "negative_draw_index":
                    negative_draw_index,

                "label":
                    int(
                        label
                    ),

                "investor_global":
                    investor_global,

                "startup_local":
                    startup_local,

                "startup_global":
                    startup_global,

                "focal_positive_startup_local":
                    focal_startup_local,

                "segment_number":
                    h,

                "segment_label":
                    f"T{h}",

                "trend_history_period_count":
                    h,
            }
        )

    return pd.DataFrame(
        rows
    )


batch0 = build_batch(
    0
)

batch1 = build_batch(
    1
)

require(
    len(batch0)
    == BATCH_SIZE,
    "Batch 0 size mismatch.",
)

require(
    len(batch1)
    == BATCH_SIZE,
    "Batch 1 size mismatch.",
)


# =============================================================================
# 4. PROVE GENERALIZED BUILDER RECONSTRUCTS FROZEN FIRST BATCH
# =============================================================================

print()
print("3. BATCH-BUILDER EQUIVALENCE")

frozen_batch0 = pd.read_parquet(
    FROZEN_FIRST_BATCH
).sort_values(
    "batch_position",
    kind="mergesort",
).reset_index(
    drop=True
)

comparison_columns = [
    "batch_position",
    "serialized_example_index",
    "positive_order_index",
    "example_slot",
    "label",
    "investor_global",
    "startup_local",
    "startup_global",
    "focal_positive_startup_local",
    "segment_number",
    "trend_history_period_count",
]

for column in comparison_columns:

    require(
        np.array_equal(
            batch0[
                column
            ].to_numpy(),
            frozen_batch0[
                column
            ].to_numpy(),
        ),
        (
            "Generalized batch builder differs "
            f"from frozen first batch: {column}"
        ),
    )

print(
    "  Frozen batch-0 reconstruction: EXACT MATCH"
)
print(
    "  Batch-1 materialization:        PASS"
)


# =============================================================================
# 5. LOAD AUTHORITATIVE 8.2.7 RUNTIME DEFINITIONS WITHOUT EXECUTING PREFLIGHT
# =============================================================================

source_text = SOURCE.read_text(
    encoding="utf-8"
)

source_lines = source_text.splitlines()

runtime = {
    "__file__":
        str(
            SOURCE.resolve()
        ),

    "__name__":
        "__phase8_two_batch_persistence__",
}

# Imports, constants and helper functions only.
base_source = (
    "\n".join(
        source_lines[
            :549
        ]
    )
    + "\n"
)

exec(
    compile(
        base_source,
        str(
            SOURCE
        ),
        "exec",
    ),
    runtime,
)


# =============================================================================
# 6. MECHANICALLY EXTRACT BATCH PREPARATION — AST SAFE
#
# V1 failed safely before model instantiation because fixed source-line
# slicing bisected a multiline statement.
#
# V2 selects complete top-level AST statements from the authoritative
# Phase-8.2.7 source.
#
# The exact frozen-batch assertions:
#     positive_count == EXPECTED_POSITIVES
#     negative_count == EXPECTED_NEGATIVES
# are batch-0 fixture checks, not arbitrary-batch invariants, so only those
# two assertions are excluded.
# =============================================================================

authoritative_tree = ast.parse(
    source_text,
    filename=str(
        SOURCE
    ),
)


def complete_source_for_nodes(nodes):

    blocks = []

    for node in nodes:

        block = ast.get_source_segment(
            source_text,
            node,
        )

        require(
            block is not None,
            (
                "Could not recover authoritative "
                f"statement at line {node.lineno}."
            ),
        )

        blocks.append(
            block
        )

    return "\n\n".join(
        blocks
    )


prepare_nodes = []

for node in authoritative_tree.body:

    node_end = getattr(
        node,
        "end_lineno",
        node.lineno,
    )

    if (
        node.lineno >= 622
        and node_end <= 1462
    ):

        # Frozen first-batch composition assertions.
        if node.lineno in {
            657,
            663,
        }:
            continue

        prepare_nodes.append(
            node
        )


require(
    len(
        prepare_nodes
    )
    > 0,
    "No batch-preparation AST statements found.",
)

first_prepare_line = min(
    node.lineno
    for node in prepare_nodes
)

last_prepare_line = max(
    getattr(
        node,
        "end_lineno",
        node.lineno,
    )
    for node in prepare_nodes
)

require(
    first_prepare_line
    == 622,
    (
        "Batch-preparation AST start changed: "
        f"{first_prepare_line}"
    ),
)

require(
    last_prepare_line
    == 1462,
    (
        "Batch-preparation AST end changed: "
        f"{last_prepare_line}"
    ),
)


prepare_body = complete_source_for_nodes(
    prepare_nodes
)


prepare_function_source = (
    "def prepare_batch_runtime(batch):\n"
    + textwrap.indent(
        prepare_body,
        "    ",
    )
    + "\n"
    + textwrap.indent(
        """
require(
    len(batch)
    == BATCH_SIZE,
    "Development batch size changed.",
)

require(
    positive_count
    + negative_count
    == len(batch),
    "Batch label accounting mismatch.",
)

return locals()
""".strip(),
        "    ",
    )
    + "\n"
)


compile(
    prepare_function_source,
    "<phase8_prepare_batch_runtime_ast_safe>",
    "exec",
)

exec(
    compile(
        prepare_function_source,
        "<phase8_prepare_batch_runtime_ast_safe>",
        "exec",
    ),
    runtime,
)

prepare_batch_runtime = runtime[
    "prepare_batch_runtime"
]

print()
print(
    "  AST-safe preparation statements: "
    f"{len(prepare_nodes)}"
)
print(
    f"  Authoritative source span:       "
    f"{first_prepare_line}..{last_prepare_line}"
)

# 7. MECHANICALLY EXTRACT PERSISTENT FAMILY STEP — AST SAFE
#
# Complete statements are selected from the already-audited authoritative
# `for family in FAMILIES` body.
#
# Preserved:
#   structured/content encoding
#   Investor encoding
#   strict mixed-h trend
#   two-hop graph branch
#   exact 7x40 pair assembly
#   raw-logit scoring
#   BCEWithLogitsLoss
#   zero_grad
#   backward
#   gradient integrity
#   Adam step
#   Adam-state integrity
#
# Externalized intentionally:
#   model instantiation/canonical initialization
#   optimizer instantiation
#   model/optimizer destruction
#
# This is precisely what allows state to persist batch0 -> batch1.
# =============================================================================

family_loops = []

for node in authoritative_tree.body:

    if not isinstance(
        node,
        ast.For,
    ):
        continue

    if (
        ast.unparse(
            node.target
        )
        == "family"
        and ast.unparse(
            node.iter
        )
        == "FAMILIES"
        and node.lineno == 1478
    ):

        family_loops.append(
            node
        )


require(
    len(
        family_loops
    )
    == 1,
    (
        "Could not identify exactly one "
        "authoritative family loop."
    ),
)

family_loop = family_loops[
    0
]


compute_nodes = []

update_nodes = []

for node in family_loop.body:

    node_end = getattr(
        node,
        "end_lineno",
        node.lineno,
    )

    if (
        node.lineno >= 1524
        and node_end <= 2114
    ):

        compute_nodes.append(
            node
        )

    if (
        node.lineno >= 2148
        and node_end <= 2410
    ):

        update_nodes.append(
            node
        )


require(
    len(
        compute_nodes
    )
    > 0,
    "No forward/BCE AST statements found.",
)

require(
    len(
        update_nodes
    )
    > 0,
    "No backward/Adam AST statements found.",
)


compute_body = complete_source_for_nodes(
    compute_nodes
)

update_body = complete_source_for_nodes(
    update_nodes
)


step_function_source = (
    "def persistent_train_step(family, model, optimizer):\n"
    "    initial_full_hash = full_parameter_hash(model)\n"
    "    initial_hashes = parameter_hashes(model)\n"
    + textwrap.indent(
        compute_body,
        "    ",
    )
    + "\n"
    + textwrap.indent(
        update_body,
        "    ",
    )
    + "\n"
    + textwrap.indent(
        """
return {
    "loss_bce_with_logits":
        float(
            loss.detach().item()
        ),

    "initial_parameter_sha256":
        initial_full_hash,

    "post_step_parameter_sha256":
        post_step_full_hash,

    "changed_parameter_tensors":
        int(
            len(
                changed_parameters
            )
        ),

    "optimizer_state_entries":
        int(
            len(
                optimizer.state
            )
        ),

    "gradient_components":
        {
            key:
                float(value)
            for key, value
            in sorted(
                component_grad_abs_sum.items()
            )
        },

    "all_gradients_present":
        len(
            missing_gradients
        ) == 0,

    "all_gradients_finite":
        len(
            nonfinite_gradients
        ) == 0,

    "all_active_components_nonzero_gradient":
        len(
            zero_components
        ) == 0,
}
""".strip(),
        "    ",
    )
    + "\n"
)


compile(
    step_function_source,
    "<phase8_persistent_train_step_ast_safe>",
    "exec",
)

exec(
    compile(
        step_function_source,
        "<phase8_persistent_train_step_ast_safe>",
        "exec",
    ),
    runtime,
)

persistent_train_step = runtime[
    "persistent_train_step"
]

print(
    "  AST-safe forward/BCE statements:   "
    f"{len(compute_nodes)}"
)

print(
    "  AST-safe backward/Adam statements: "
    f"{len(update_nodes)}"
)

# 8. PREPARE FIRST BATCH
# =============================================================================

print()
print("4. PREPARE BATCH 0")

ctx0 = prepare_batch_runtime(
    batch0
)

# The mechanically extracted family computation resolves batch inputs
# through its module namespace, just as the original top-level code did.
runtime.update(
    ctx0
)

dimensions = ctx0[
    "dimensions"
]

namespace_lookup = ctx0[
    "namespace_lookup"
]

init_module = runtime[
    "init_module"
]

model_module = runtime[
    "model_module"
]

torch = runtime[
    "torch"
]


# =============================================================================
# 9. ONE CANONICAL M2 MODEL + ONE PERSISTENT ADAM
# =============================================================================

print()
print("5. INITIALIZE ONE PERSISTENT M2 TRAINING STATE")

model = (
    init_module
    .instantiate_model(
        model_module,
        FAMILY,
        dimensions,
    )
)

init_module.initialize_model(
    model=model,
    family=FAMILY,
    scope="DEV_T57",
    namespace_lookup=namespace_lookup,
    reverse_order=False,
)

model.train()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    betas=(
        0.9,
        0.999,
    ),
    eps=1e-8,
    weight_decay=0.0,
    amsgrad=False,
    foreach=False,
    maximize=False,
    capturable=False,
    differentiable=False,
    fused=False,
)

require(
    len(
        optimizer.state
    )
    == 0,
    "Adam state exists before first retained step.",
)

canonical_initial_hash = (
    runtime[
        "full_parameter_hash"
    ](
        model
    )
)

require(
    canonical_initial_hash
    ==
    reference_m2[
        "state_audit"
    ][
        "initial_parameter_sha256"
    ],
    "Persistent M2 canonical initialization differs from 8.2.7.",
)

print(
    "  Canonical initialization: PASS"
)
print(
    "  Adam initial state:       EMPTY PASS"
)


# =============================================================================
# 10. RETAINED STEP 1 — MUST EXACTLY MATCH FROZEN 8.2.7
# =============================================================================

print()
print("=" * 120)
print("6. RETAINED STEP 1 — FROZEN BATCH 0")
print("=" * 120)

step1 = persistent_train_step(
    FAMILY,
    model,
    optimizer,
)

require(
    step1[
        "loss_bce_with_logits"
    ]
    ==
    reference_m2[
        "numerical_audit"
    ][
        "loss_bce_with_logits"
    ],
    "Step-1 BCE differs from frozen 8.2.7 M2.",
)

require(
    step1[
        "gradient_components"
    ]
    ==
    reference_m2[
        "numerical_audit"
    ][
        "gradient_components"
    ],
    "Step-1 gradient components differ from frozen 8.2.7 M2.",
)

require(
    step1[
        "post_step_parameter_sha256"
    ]
    ==
    reference_m2[
        "state_audit"
    ][
        "post_step_parameter_sha256"
    ],
    "Step-1 post-Adam model hash differs from frozen 8.2.7 M2.",
)

require(
    step1[
        "changed_parameter_tensors"
    ]
    ==
    reference_m2[
        "state_audit"
    ][
        "changed_parameter_tensors"
    ],
    "Step-1 changed-parameter count differs from frozen 8.2.7 M2.",
)

step1_post_hash = (
    step1[
        "post_step_parameter_sha256"
    ]
)

print(
    "  BCE:                     EXACT MATCH"
)
print(
    "  Gradient components:     EXACT MATCH"
)
print(
    "  Post-step model SHA:     EXACT MATCH"
)
print(
    "  Model retained:          YES"
)
print(
    "  Adam state retained:     YES"
)


# =============================================================================
# 11. PREPARE SECOND BATCH
# =============================================================================

print()
print("7. PREPARE BATCH 1")

ctx1 = prepare_batch_runtime(
    batch1
)

runtime.update(
    ctx1
)

pre_step2_hash = (
    runtime[
        "full_parameter_hash"
    ](
        model
    )
)

require(
    pre_step2_hash
    == step1_post_hash,
    (
        "Model state was not preserved exactly "
        "between batch 0 and batch 1."
    ),
)

print(
    "  Step-1 state -> Step-2 input: EXACT MATCH"
)


# =============================================================================
# 12. RETAINED STEP 2
# =============================================================================

print()
print("=" * 120)
print("8. RETAINED STEP 2 — FROZEN BATCH 1")
print("=" * 120)

step2 = persistent_train_step(
    FAMILY,
    model,
    optimizer,
)

require(
    step2[
        "initial_parameter_sha256"
    ]
    == step1_post_hash,
    "Step 2 did not begin from the retained step-1 model.",
)

require(
    np.isfinite(
        step2[
            "loss_bce_with_logits"
        ]
    ),
    "Step-2 BCE is non-finite.",
)

require(
    step2[
        "all_gradients_present"
    ],
    "Step-2 has missing gradients.",
)

require(
    step2[
        "all_gradients_finite"
    ],
    "Step-2 has non-finite gradients.",
)

require(
    step2[
        "all_active_components_nonzero_gradient"
    ],
    "Step-2 has an active component with zero aggregate gradient.",
)

require(
    step2[
        "post_step_parameter_sha256"
    ]
    != step2[
        "initial_parameter_sha256"
    ],
    "Step-2 Adam update changed no model state.",
)


# =============================================================================
# 13. ADAM STEP COUNTERS MUST NOW ALL BE 2
# =============================================================================

adam_steps = []

for parameter, state in optimizer.state.items():

    require(
        "step" in state,
        "Adam state missing step counter.",
    )

    step_value = state[
        "step"
    ]

    if torch.is_tensor(
        step_value
    ):

        step_value = float(
            step_value.detach()
            .cpu()
            .item()
        )

    else:

        step_value = float(
            step_value
        )

    adam_steps.append(
        step_value
    )

require(
    len(
        adam_steps
    )
    > 0,
    "Adam has no state after two retained steps.",
)

require(
    all(
        value == 2.0
        for value in adam_steps
    ),
    (
        "Not all Adam parameter states advanced "
        f"to step 2: {sorted(set(adam_steps))}"
    ),
)

print(
    f"  BCE finite:              PASS ({step2['loss_bce_with_logits']:.9f})"
)
print(
    "  Gradients finite:        PASS"
)
print(
    "  Active components > 0:  PASS"
)
print(
    "  Model state changed:     PASS"
)
print(
    "  Adam step counters:      ALL 2 PASS"
)


# =============================================================================
# 14. SAVE AUDIT ONLY — NEVER CHECKPOINT MODEL/OPTIMIZER
# =============================================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

result = {
    "schema_version":
        (
            "PHASE_8_DEV_T57_M2_TWO_BATCH_"
            "STATE_PERSISTENCE_AUDIT_V2"
        ),

    "status":
        "TWO_BATCH_STATE_PERSISTENCE_AUDITED",

    "scientific_role":
        (
            "POST_8_2_8_DEVELOPMENT_TRAINING_"
            "RUNTIME_PERSISTENCE_PREFLIGHT"
        ),

    "scope":
        "DEV_T57",

    "family":
        FAMILY,

    "epoch_index":
        0,

    "batch_indices":
        [
            0,
            1,
        ],

    "batch_builder": {
        "frozen_batch0_reconstructed_exactly":
            True,

        "serialization":
            [
                "positive",
                "negative_0",
                "negative_1",
                "negative_2",
                "negative_3",
            ],

        "batch_size":
            BATCH_SIZE,
    },

    "step_1": {
        **step1,

        "matches_frozen_8_2_7":
            True,
    },

    "state_transition": {
        "step1_post_sha_equals_step2_pre_sha":
            True,

        "model_reinitialized_between_batches":
            False,

        "optimizer_reinitialized_between_batches":
            False,

        "optimizer_state_retained":
            True,
    },

    "step_2":
        step2,

    "adam_after_step_2": {
        "state_entries":
            int(
                len(
                    optimizer.state
                )
            ),

        "unique_step_values":
            sorted(
                set(
                    adam_steps
                )
            ),

        "all_step_counters_equal_2":
            True,
    },

    "execution_boundary": {
        "optimizer_steps_executed":
            2,

        "full_epoch_executed":
            False,

        "ranking_metrics_computed":
            False,

        "model_selection_performed":
            False,

        "checkpoint_written":
            False,

        "model_state_persisted_to_disk":
            False,

        "optimizer_state_persisted_to_disk":
            False,

        "T60_scored":
            False,

        "April_analyzed":
            False,
    },

    "interpretation":
        (
            "The exact Phase-8 candidate computation path "
            "reproduces the frozen 8.2.7 M2 first step and "
            "supports a second optimizer step while retaining "
            "the exact post-step-1 model and Adam state. "
            "This validates the batch-to-batch state transition "
            "required by full development training."
        ),
}

OUTPUT_RESULT.write_text(
    json.dumps(
        result,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

hash_registry = {
    str(
        SOURCE
    ):
        sha256_file(
            SOURCE
        ),

    str(
        REFERENCE
    ):
        sha256_file(
            REFERENCE
        ),

    str(
        POSITIVE_ORDER
    ):
        sha256_file(
            POSITIVE_ORDER
        ),

    str(
        NEGATIVE_MATRIX
    ):
        sha256_file(
            NEGATIVE_MATRIX
        ),

    str(
        EPOCH_ORDER
    ):
        sha256_file(
            EPOCH_ORDER
        ),

    str(
        FROZEN_FIRST_BATCH
    ):
        sha256_file(
            FROZEN_FIRST_BATCH
        ),

    str(
        OUTPUT_RESULT
    ):
        sha256_file(
            OUTPUT_RESULT
        ),
}

OUTPUT_HASHES.write_text(
    json.dumps(
        hash_registry,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)


# Destroy temporary updated state after the persistence audit.
del optimizer
del model


print()
print("=" * 120)
print("TWO-BATCH STATE-PERSISTENCE AUDIT")
print("=" * 120)

print(
    "Frozen batch-0 reconstruction:       EXACT MATCH"
)
print(
    "M2 retained step-1 vs 8.2.7:         EXACT MATCH"
)
print(
    "Step1-post == Step2-pre state:        PASS"
)
print(
    "Model reinitialized between batches:  NO"
)
print(
    "Optimizer reinitialized between:      NO"
)
print(
    "Second forward/BCE/backward:           PASS"
)
print(
    "Second Adam step:                     PASS"
)
print(
    "Adam counters after second step:       ALL 2"
)
print(
    "Full epoch executed:                  NO"
)
print(
    "Ranking metrics computed:             NO"
)
print(
    "Checkpoint written:                   NO"
)
print(
    "T60 scored:                           NO"
)
print(
    "April analyzed:                       NO"
)
print(
    "Temporary updated state discarded:    YES"
)

print()
print(
    f"Audit:  {OUTPUT_RESULT}"
)
print(
    f"Hashes: {OUTPUT_HASHES}"
)

print()
print(
    "STATUS: TWO_BATCH_STATE_PERSISTENCE_AUDITED"
)
print(
    "NEXT:   Run the same persistent runtime across "
    "the complete DEV_T57 epoch-0 stream for one family."
)
print("=" * 120)
