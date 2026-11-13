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
    "dev_t57_m2_two_batch_state_persistence_V3"
)

OUTPUT_RESULT = (
    OUTPUT_DIR
    / "phase_8_dev_t57_m2_two_batch_state_persistence_V3.json"
)

OUTPUT_HASHES = (
    OUTPUT_DIR
    / "phase_8_dev_t57_m2_two_batch_state_persistence_sha256_V3.json"
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
    "STATE-PERSISTENCE AUDIT V3"
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
    "  V2: failed safely during generated-step compilation; "
    "AST node selection was correct but ast.get_source_segment() "
    "produced inconsistent indentation for nested blocks."
)
print(
    "  V3: AST node selection + full-line extraction + "
    "uniform textwrap.dedent()."
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

        node_end = getattr(
            node,
            "end_lineno",
            node.lineno,
        )

        block = "\n".join(
            source_lines[
                node.lineno - 1:
                node_end
            ]
        )

        block = textwrap.dedent(
            block
        )

        require(
            bool(
                block.strip()
            ),
            (
                "Could not recover authoritative "
                f"statement at line {node.lineno}."
            ),
        )

        # Every recovered AST statement must itself be valid Python
        # before being embedded in the generated function.
        compile(
            block,
            (
                "<authoritative_statement_"
                f"{node.lineno}_{node_end}>"
            ),
            "exec",
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


# =============================================================================
# 8. CPU THROUGHPUT BENCHMARK
# =============================================================================

import contextlib
import math
import os
import platform
import statistics
import time


BENCHMARK_BATCHES = 100
WARMUP_BATCHES = 5

BENCHMARK_DIR = Path(
    "data/experimental/phase_8/"
    "comparative_development_training_preflight/"
    "dev_t57_m2_cpu_100batch_benchmark_V1"
)

BENCHMARK_JSON = (
    BENCHMARK_DIR
    / "phase_8_dev_t57_m2_cpu_100batch_benchmark_V1.json"
)


print()
print("=" * 120)
print(
    "PHASE 8 — DEV_T57 M2 CPU 100-BATCH THROUGHPUT BENCHMARK V1"
)
print("=" * 120)

print()
print("Scientific role:")
print(
    "  Performance characterization only."
)
print(
    "  Same validated M2 persistent runtime; "
    "NO architecture/protocol changes."
)
print()
print(
    "  NO full epoch, NO ranking metrics, NO model selection,"
)
print(
    "  NO checkpoint retention, NO T60, NO April."
)


# =============================================================================
# Environment
# =============================================================================

torch = runtime[
    "torch"
]

print()
print("1. EXECUTION ENVIRONMENT")

print(
    f"  Platform:                  {platform.platform()}"
)

print(
    f"  Machine:                   {platform.machine()}"
)

print(
    f"  Processor:                 {platform.processor()}"
)

print(
    f"  Python:                    {platform.python_version()}"
)

print(
    f"  PyTorch:                   {torch.__version__}"
)

print(
    f"  torch num threads:         {torch.get_num_threads()}"
)

print(
    f"  torch interop threads:     {torch.get_num_interop_threads()}"
)

print(
    "  Training device:           CPU"
)

print(
    f"  MPS available:             "
    f"{bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())}"
)

print(
    f"  CUDA available:            {torch.cuda.is_available()}"
)


# =============================================================================
# Stream geometry
# =============================================================================

epoch_examples = int(
    len(
        epoch_order
    )
)

epoch_batches = int(
    math.ceil(
        epoch_examples
        / BATCH_SIZE
    )
)

final_batch_size = int(
    epoch_examples
    - (
        epoch_batches - 1
    )
    * BATCH_SIZE
)


require(
    BENCHMARK_BATCHES
    < epoch_batches,
    "Benchmark unexpectedly spans full epoch.",
)


print()
print("2. FROZEN STREAM GEOMETRY")

print(
    f"  Epoch examples:            {epoch_examples:,}"
)

print(
    f"  Epoch batches:             {epoch_batches:,}"
)

print(
    f"  Normal batch size:         {BATCH_SIZE}"
)

print(
    f"  Final batch size:          {final_batch_size}"
)

print(
    f"  Benchmark batches:         {BENCHMARK_BATCHES}"
)

print(
    f"  Warm-up batches:           {WARMUP_BATCHES}"
)

print(
    f"  Measured batches:          "
    f"{BENCHMARK_BATCHES - WARMUP_BATCHES}"
)


# =============================================================================
# Prepare batch 0 once to instantiate canonical state
# =============================================================================

print()
print("3. INITIALIZE CANONICAL M2 STATE")

with open(
    os.devnull,
    "w",
) as devnull:

    with contextlib.redirect_stdout(
        devnull
    ):

        ctx0 = prepare_batch_runtime(
            build_batch(
                0
            )
        )


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
    "Adam unexpectedly has state before first step.",
)


initial_model_sha = (
    runtime[
        "full_parameter_hash"
    ](
        model
    )
)


require(
    initial_model_sha
    ==
    reference_m2[
        "state_audit"
    ][
        "initial_parameter_sha256"
    ],
    "Canonical M2 initialization differs from frozen 8.2.7.",
)


print(
    "  Canonical initialization:     PASS"
)

print(
    "  Initial model SHA:            EXACT MATCH"
)

print(
    "  Adam initial state:           EMPTY PASS"
)


# =============================================================================
# Sequential 100-batch benchmark
# =============================================================================

print()
print("=" * 120)
print("4. SEQUENTIAL BENCHMARK")
print("=" * 120)

prepare_times = []
train_times = []
total_times = []
losses = []

first_step_exact = False


for batch_index in range(
    BENCHMARK_BATCHES
):

    batch_wall_start = time.perf_counter()

    batch_frame = build_batch(
        batch_index
    )

    prep_start = time.perf_counter()

    with open(
        os.devnull,
        "w",
    ) as devnull:

        with contextlib.redirect_stdout(
            devnull
        ):

            ctx = prepare_batch_runtime(
                batch_frame
            )

    prep_seconds = (
        time.perf_counter()
        - prep_start
    )

    runtime.update(
        ctx
    )

    train_start = time.perf_counter()

    with open(
        os.devnull,
        "w",
    ) as devnull:

        with contextlib.redirect_stdout(
            devnull
        ):

            step_result = (
                persistent_train_step(
                    FAMILY,
                    model,
                    optimizer,
                )
            )

    train_seconds = (
        time.perf_counter()
        - train_start
    )

    total_seconds = (
        time.perf_counter()
        - batch_wall_start
    )

    loss_value = float(
        step_result[
            "loss_bce_with_logits"
        ]
    )


    # Strong correctness gate on first benchmark step.
    if batch_index == 0:

        require(
            loss_value
            ==
            reference_m2[
                "numerical_audit"
            ][
                "loss_bce_with_logits"
            ],
            (
                "Benchmark batch-0 BCE differs "
                "from frozen 8.2.7."
            ),
        )

        require(
            step_result[
                "gradient_components"
            ]
            ==
            reference_m2[
                "numerical_audit"
            ][
                "gradient_components"
            ],
            (
                "Benchmark batch-0 gradients differ "
                "from frozen 8.2.7."
            ),
        )

        require(
            step_result[
                "post_step_parameter_sha256"
            ]
            ==
            reference_m2[
                "state_audit"
            ][
                "post_step_parameter_sha256"
            ],
            (
                "Benchmark batch-0 post-step SHA differs "
                "from frozen 8.2.7."
            ),
        )

        first_step_exact = True


    require(
        np.isfinite(
            loss_value
        ),
        (
            f"Non-finite BCE at batch "
            f"{batch_index}."
        ),
    )

    require(
        step_result[
            "all_gradients_finite"
        ],
        (
            f"Non-finite gradient at batch "
            f"{batch_index}."
        ),
    )


    prepare_times.append(
        float(
            prep_seconds
        )
    )

    train_times.append(
        float(
            train_seconds
        )
    )

    total_times.append(
        float(
            total_seconds
        )
    )

    losses.append(
        loss_value
    )


    if (
        batch_index < 5
        or (
            batch_index + 1
        )
        % 10
        == 0
    ):

        phase = (
            "WARMUP"
            if batch_index
            < WARMUP_BATCHES
            else "MEASURED"
        )

        print(
            f"  Batch {batch_index:03d} "
            f"[{phase}] "
            f"prep={prep_seconds:8.3f}s "
            f"train={train_seconds:8.3f}s "
            f"total={total_seconds:8.3f}s "
            f"loss={loss_value:.6f}"
        )


# =============================================================================
# Validate persistent optimizer trajectory
# =============================================================================

adam_steps = []

for state in optimizer.state.values():

    require(
        "step" in state,
        "Adam parameter state missing step counter.",
    )

    value = state[
        "step"
    ]

    if torch.is_tensor(
        value
    ):

        value = float(
            value.detach()
            .cpu()
            .item()
        )

    else:

        value = float(
            value
        )

    adam_steps.append(
        value
    )


require(
    len(
        adam_steps
    )
    > 0,
    "Adam state missing after benchmark.",
)

require(
    all(
        value
        == float(
            BENCHMARK_BATCHES
        )
        for value in adam_steps
    ),
    (
        "Adam counters do not equal benchmark "
        f"batch count: {sorted(set(adam_steps))}"
    ),
)


# =============================================================================
# Statistics
# =============================================================================

measured_prepare = (
    prepare_times[
        WARMUP_BATCHES:
    ]
)

measured_train = (
    train_times[
        WARMUP_BATCHES:
    ]
)

measured_total = (
    total_times[
        WARMUP_BATCHES:
    ]
)


def percentile(
    values,
    q,
):

    return float(
        np.percentile(
            np.asarray(
                values,
                dtype=np.float64,
            ),
            q,
        )
    )


def describe(
    values,
):

    return {
        "count":
            int(
                len(
                    values
                )
            ),

        "mean_seconds":
            float(
                statistics.fmean(
                    values
                )
            ),

        "median_seconds":
            float(
                statistics.median(
                    values
                )
            ),

        "p95_seconds":
            percentile(
                values,
                95,
            ),

        "min_seconds":
            float(
                min(
                    values
                )
            ),

        "max_seconds":
            float(
                max(
                    values
                )
            ),
    }


prepare_stats = describe(
    measured_prepare
)

train_stats = describe(
    measured_train
)

total_stats = describe(
    measured_total
)


mean_total_seconds = float(
    total_stats[
        "mean_seconds"
    ]
)

mean_prepare_seconds = float(
    prepare_stats[
        "mean_seconds"
    ]
)

mean_train_seconds = float(
    train_stats[
        "mean_seconds"
    ]
)


projected_epoch_seconds = (
    mean_total_seconds
    * epoch_batches
)

projected_20_epoch_seconds = (
    projected_epoch_seconds
    * 20
)

projected_60_m2_equivalent_seconds = (
    projected_epoch_seconds
    * 60
)

projected_240_m2_equivalent_seconds = (
    projected_epoch_seconds
    * 240
)


throughput_batches_per_second = (
    1.0
    / mean_total_seconds
)

throughput_examples_per_second = (
    BATCH_SIZE
    / mean_total_seconds
)


prepare_share = (
    mean_prepare_seconds
    / mean_total_seconds
)

train_share = (
    mean_train_seconds
    / mean_total_seconds
)


def human_duration(
    seconds,
):

    seconds = float(
        seconds
    )

    minutes = (
        seconds
        / 60.0
    )

    if minutes < 60:

        return (
            f"{minutes:.2f} min"
        )

    hours = (
        minutes
        / 60.0
    )

    if hours < 48:

        return (
            f"{hours:.2f} h"
        )

    days = (
        hours
        / 24.0
    )

    return (
        f"{days:.2f} days"
    )


print()
print("=" * 120)
print("5. BENCHMARK RESULT")
print("=" * 120)

print(
    f"  First-step frozen equivalence:  "
    f"{'PASS' if first_step_exact else 'FAIL'}"
)

print(
    f"  Adam counters after benchmark:  "
    f"ALL {BENCHMARK_BATCHES} PASS"
)

print()

print(
    f"  Mean preparation / batch:       "
    f"{mean_prepare_seconds:.3f} s"
)

print(
    f"  Mean train-step / batch:        "
    f"{mean_train_seconds:.3f} s"
)

print(
    f"  Mean TOTAL / batch:             "
    f"{mean_total_seconds:.3f} s"
)

print(
    f"  Median TOTAL / batch:           "
    f"{total_stats['median_seconds']:.3f} s"
)

print(
    f"  p95 TOTAL / batch:              "
    f"{total_stats['p95_seconds']:.3f} s"
)

print()

print(
    f"  Preparation share:              "
    f"{prepare_share * 100.0:.1f}%"
)

print(
    f"  Forward/backward/Adam share:    "
    f"{train_share * 100.0:.1f}%"
)

print()

print(
    f"  Throughput:                     "
    f"{throughput_batches_per_second:.4f} batches/s"
)

print(
    f"  Throughput:                     "
    f"{throughput_examples_per_second:.1f} examples/s"
)

print()

print(
    f"  Projected DEV_T57 M2 epoch:     "
    f"{human_duration(projected_epoch_seconds)}"
)

print(
    f"  Projected T57 M2 × 20 epochs:   "
    f"{human_duration(projected_20_epoch_seconds)}"
)

print(
    f"  Projected M2-equivalent ×60:    "
    f"{human_duration(projected_60_m2_equivalent_seconds)}"
)

print(
    f"  Projected M2-equivalent ×240:   "
    f"{human_duration(projected_240_m2_equivalent_seconds)}"
)

print()
print(
    "  NOTE: ×60 and ×240 are throughput-scale projections,"
)
print(
    "  not final experiment-time estimates. T58/T59 differ in"
)
print(
    "  size and B1a/B1b/M1 have different computational costs."
)


# =============================================================================
# Save benchmark evidence
# =============================================================================

BENCHMARK_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


result = {
    "schema_version":
        "PHASE_8_DEV_T57_M2_CPU_100BATCH_BENCHMARK_V1",

    "status":
        "CPU_BASELINE_THROUGHPUT_MEASURED",

    "scientific_boundary": {
        "scope":
            "DEV_T57",

        "family":
            "M2",

        "epoch_index":
            0,

        "full_epoch_executed":
            False,

        "ranking_metrics_computed":
            False,

        "model_selection_performed":
            False,

        "checkpoint_written":
            False,

        "T60_scored":
            False,

        "April_analyzed":
            False,
    },

    "environment": {
        "platform":
            platform.platform(),

        "machine":
            platform.machine(),

        "processor":
            platform.processor(),

        "python":
            platform.python_version(),

        "pytorch":
            str(
                torch.__version__
            ),

        "torch_num_threads":
            int(
                torch.get_num_threads()
            ),

        "torch_num_interop_threads":
            int(
                torch.get_num_interop_threads()
            ),

        "training_device":
            "cpu",

        "cuda_available":
            bool(
                torch.cuda.is_available()
            ),
    },

    "stream": {
        "epoch_examples":
            epoch_examples,

        "epoch_batches":
            epoch_batches,

        "batch_size":
            BATCH_SIZE,

        "final_batch_size":
            final_batch_size,

        "benchmark_batches":
            BENCHMARK_BATCHES,

        "warmup_batches":
            WARMUP_BATCHES,

        "measured_batches":
            int(
                BENCHMARK_BATCHES
                - WARMUP_BATCHES
            ),
    },

    "correctness_gates": {
        "batch0_exact_8_2_7":
            bool(
                first_step_exact
            ),

        "adam_steps_after_benchmark":
            sorted(
                set(
                    adam_steps
                )
            ),

        "expected_adam_step":
            BENCHMARK_BATCHES,
    },

    "timing": {
        "prepare":
            prepare_stats,

        "train_step":
            train_stats,

        "total":
            total_stats,

        "prepare_fraction":
            float(
                prepare_share
            ),

        "train_fraction":
            float(
                train_share
            ),

        "batches_per_second":
            float(
                throughput_batches_per_second
            ),

        "examples_per_second":
            float(
                throughput_examples_per_second
            ),
    },

    "projections": {
        "DEV_T57_M2_one_epoch_seconds":
            float(
                projected_epoch_seconds
            ),

        "DEV_T57_M2_20_epochs_seconds":
            float(
                projected_20_epoch_seconds
            ),

        "M2_equivalent_60_epochs_seconds":
            float(
                projected_60_m2_equivalent_seconds
            ),

        "M2_equivalent_240_epochs_seconds":
            float(
                projected_240_m2_equivalent_seconds
            ),

        "caveat":
            (
                "Scaling-only projections from DEV_T57/M2. "
                "T58/T59 have different stream sizes and "
                "B1a/B1b/M1 have different runtime costs."
            ),
    },

    "per_batch": [
        {
            "batch_index":
                int(
                    i
                ),

            "warmup":
                bool(
                    i
                    < WARMUP_BATCHES
                ),

            "prepare_seconds":
                float(
                    prepare_times[
                        i
                    ]
                ),

            "train_seconds":
                float(
                    train_times[
                        i
                    ]
                ),

            "total_seconds":
                float(
                    total_times[
                        i
                    ]
                ),

            "loss":
                float(
                    losses[
                        i
                    ]
                ),
        }
        for i in range(
            BENCHMARK_BATCHES
        )
    ],
}


BENCHMARK_JSON.write_text(
    json.dumps(
        result,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)


# Temporary benchmark training state is intentionally discarded.
del optimizer
del model


print()
print(
    f"Benchmark artifact: {BENCHMARK_JSON}"
)

print()
print(
    "Temporary updated model/Adam state: DISCARDED"
)

print(
    "STATUS: CPU_BASELINE_THROUGHPUT_MEASURED"
)

print(
    "NEXT: Compare this baseline with the same bounded "
    "benchmark on RTX 4090 before purchasing long RunPod time."
)

print("=" * 120)
