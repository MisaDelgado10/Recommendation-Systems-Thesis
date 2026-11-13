#!/usr/bin/env python3

from __future__ import annotations

import ast
import hashlib
import json
import textwrap
import argparse
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

parser = argparse.ArgumentParser(
    description=(
        "Phase 8 DEV_T57 clean CUDA throughput benchmark "
        "for one frozen candidate family."
    )
)

parser.add_argument(
    "--family",
    required=True,
    choices=["B1a", "B1b", "M1", "M2"],
)

args = parser.parse_args()

FAMILY = args.family
FAMILY_SLUG = FAMILY.lower()
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

reference_family = reference[
    "families"
][
    FAMILY
]

print(
    f"  8.2.7 source SHA:       {source_sha} PASS"
)
print(
    f"  Frozen {FAMILY} reference:    PASS"
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


# -------------------------------------------------------------------------
# PRODUCTION VARIABLE-BATCH GENERALIZATION
# -------------------------------------------------------------------------
#
# The authoritative 8.2.7 source was written for the frozen first batch and
# therefore contains two fixture-only BATCH_SIZE assertions inside the batch
# preparation span. Production training has drop_last=False and the frozen
# DEV_T57 stream ends with 30 examples, so only those size guards are
# generalized here. No feature, graph, trend, model, loss, or optimizer math
# is changed.

_frozen_first_batch_guard = """require(
    len(batch) == BATCH_SIZE,
    "Frozen first batch no longer has 512 rows.",
)"""

require(
    prepare_body.count(_frozen_first_batch_guard) == 1,
    "Frozen first-batch size guard inventory changed.",
)

prepare_body = prepare_body.replace(
    _frozen_first_batch_guard,
    "",
)

require(
    prepare_body.count("BATCH_SIZE") == 1,
    "Unexpected BATCH_SIZE inventory in batch preparation.",
)

prepare_body = prepare_body.replace(
    "BATCH_SIZE",
    "len(batch)",
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
    1
    <= len(batch)
    <= BATCH_SIZE,
    "Development batch size outside 1..512.",
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


compute_line_starts = {
    1524,
    1547,
    1558,
    1599,
    1606,
    1614,
    1620,
    1626,
    1632,
    1644,
    1869,
    2007,
    2056,
    2098,
    2102,
}

update_line_starts = {
    2148,
    2156,
    2305,
}

compute_nodes = [
    node
    for node in family_loop.body
    if node.lineno in compute_line_starts
]

update_nodes = [
    node
    for node in family_loop.body
    if node.lineno in update_line_starts
]

require(
    {
        node.lineno
        for node in compute_nodes
    }
    == compute_line_starts,
    "Clean forward/BCE node inventory changed.",
)

require(
    {
        node.lineno
        for node in update_nodes
    }
    == update_line_starts,
    "Clean update node inventory changed.",
)

compute_body = complete_source_for_nodes(
    compute_nodes
)


def remove_attention_finite_sync_audit(source):

    lines = source.splitlines(
        keepends=True
    )

    hits = [
        i
        for i, line
        in enumerate(lines)
        if 'f"{family}: non-finite "' in line
        and i + 1 < len(lines)
    ]

    require(
        len(hits) == 1,
        (
            "Expected exactly one trend-attention "
            f"finite audit; found {len(hits)}."
        ),
    )

    hit = hits[0]

    start = hit

    while (
        start >= 0
        and lines[start].strip()
        != "require("
    ):
        start -= 1

    require(
        start >= 0,
        "Could not locate start of attention require().",
    )

    depth = 0
    end = None

    for j in range(
        start,
        len(lines),
    ):

        depth += (
            lines[j].count("(")
            - lines[j].count(")")
        )

        if (
            j > start
            and depth == 0
        ):
            end = j + 1
            break

    require(
        end is not None,
        "Could not locate end of attention require().",
    )

    removed = "".join(
        lines[start:end]
    )

    require(
        "torch.isfinite" in removed
        and "attention" in removed
        and ".all()" in removed,
        (
            "Located block is not the expected "
            "attention finite audit."
        ),
    )

    cleaned = "".join(
        lines[:start]
        + lines[end:]
    )

    require(
        "non-finite "
        '"trend attention."'
        not in cleaned,
        (
            "Trend-attention audit text unexpectedly "
            "remains after removal."
        ),
    )

    return cleaned, removed


compute_body, removed_attention_audit = (
    remove_attention_finite_sync_audit(
        compute_body
    )
)

print(
    "  Trend attention finite-sync audit: REMOVED"
)


def apply_history_item_hoist_candidate(
    source,
):

    lines = source.splitlines(
        keepends=True
    )

    targets = (
        "history_content",
        "history_structured",
        "history_items",
    )

    ranges = {}

    for variable in targets:

        hits = [
            i
            for i, line
            in enumerate(lines)
            if line.strip()
            == f"{variable} = ("
        ]

        require(
            len(hits) == 1,
            (
                f"Expected exactly one {variable} "
                f"assignment; found {len(hits)}."
            ),
        )

        start = hits[0]

        depth = 0
        end = None

        for j in range(
            start,
            len(lines),
        ):

            depth += (
                lines[j].count("(")
                - lines[j].count(")")
            )

            if (
                j > start
                and depth == 0
            ):
                end = j + 1
                break

        require(
            end is not None,
            f"Could not locate end of {variable}.",
        )

        ranges[variable] = (
            start,
            end,
        )

    require(
        ranges["history_content"][0]
        < ranges["history_structured"][0]
        < ranges["history_items"][0],
        "Historical-item assignment ordering changed.",
    )

    first = ranges[
        "history_content"
    ][0]

    last = ranges[
        "history_items"
    ][1]

    indent = (
        lines[first][
            :len(lines[first])
            - len(lines[first].lstrip())
        ]
    )

    replacement = [
        (
            f"{indent}"
            "history_items = "
            "history_items_all[item_positions]\n"
        )
    ]

    lines = (
        lines[:first]
        + replacement
        + lines[last:]
    )

    query_hits = [
        i
        for i, line
        in enumerate(lines)
        if line.strip()
        == "trend_query_unique = ("
    ]

    require(
        len(query_hits) == 1,
        (
            "Expected exactly one trend_query_unique; "
            f"found {len(query_hits)}."
        ),
    )

    q = query_hits[0]

    qindent = (
        lines[q][
            :len(lines[q])
            - len(lines[q].lstrip())
        ]
    )

    hoist = [
        f"{qindent}history_items_all = (\n",
        f"{qindent}    model.compose_history_items(\n",
        f"{qindent}        startup_structured_all,\n",
        f"{qindent}        startup_content_all,\n",
        f"{qindent}    )\n",
        f"{qindent})\n",
        "\n",
    ]

    lines = (
        lines[:q]
        + hoist
        + lines[q:]
    )

    cleaned = "".join(lines)

    require(
        cleaned.count(
            "history_items_all = ("
        ) == 1,
        "History hoist insertion count mismatch.",
    )

    require(
        cleaned.count(
            "history_items_all[item_positions]"
        ) == 1,
        "Combined-history gather count mismatch.",
    )

    require(
        "history_content = (" not in cleaned,
        "history_content gather still present.",
    )

    require(
        "history_structured = (" not in cleaned,
        "history_structured gather still present.",
    )

    return cleaned


compute_body = (
    apply_history_item_hoist_candidate(
        compute_body
    )
)


# Generalize ONLY the three semantic batch-length dependencies inherited
# from the frozen 512-row preflight. Do not replace arbitrary BATCH_SIZE text:
# every mathematical substitution below must match exactly once.
_variable_batch_replacements = [
    (
        """for i
                in range(
                    BATCH_SIZE
                )""",
        """for i
                in range(
                    len(batch)
                )""",
        "trend batch assembly loop",
    ),
    (
        """pair.shape
        == (
            BATCH_SIZE,
            280,
        )""",
        """pair.shape
        == (
            len(batch),
            280,
        )""",
        "pair shape audit",
    ),
    (
        """logits.shape
        == (
            BATCH_SIZE,
            1,
        )""",
        """logits.shape
        == (
            len(batch),
            1,
        )""",
        "logit shape audit",
    ),
]

for old_text, new_text, label in _variable_batch_replacements:

    require(
        compute_body.count(old_text) == 1,
        (
            "Variable-batch semantic pattern inventory changed: "
            f"{label}; matches={compute_body.count(old_text)}"
        ),
    )

    compute_body = compute_body.replace(
        old_text,
        new_text,
    )

# Any remaining occurrence is permitted only if it is not one of the three
# batch-length semantics above. Record it for audit instead of rewriting it.
remaining_batch_size_mentions = [
    line.strip()
    for line in compute_body.splitlines()
    if "BATCH_SIZE" in line
]

print(
    "  Residual BATCH_SIZE mentions after semantic patch: "
    f"{len(remaining_batch_size_mentions)}"
)

for line in remaining_batch_size_mentions:
    print(
        f"    residual: {line}"
    )


print(
    "  History-item optimization:     "
    "CAT-ALL ONCE -> GATHER COMBINED"
)


update_body = complete_source_for_nodes(
    update_nodes
)

# Factory tensors created inside the authoritative computation inherit
# the model device. The structured-input bridge in CUDA V4 is retained.
compute_body = (
    "with torch.device(next(model.parameters()).device):\n"
    + textwrap.indent(
        compute_body,
        "    ",
    )
)

step_function_source = (
    "def persistent_train_step(family, model, optimizer):\n"
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
}
""".strip(),
        "    ",
    )
    + "\n"
)

compile(
    step_function_source,
    "<phase8_clean_persistent_train_step>",
    "exec",
)

exec(
    compile(
        step_function_source,
        "<phase8_clean_persistent_train_step>",
        "exec",
    ),
    runtime,
)

persistent_train_step = runtime[
    "persistent_train_step"
]

print(
    "  Clean forward/BCE statements:      "
    f"{len(compute_nodes)}"
)

print(
    "  Clean update statements:           "
    f"{len(update_nodes)}"
)

print(
    "  Audit-only per-batch statements:   REMOVED"
)

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

BENCHMARK_DIR = (
    Path(
        "data/experimental/phase_8/"
        "comparative_development_training_preflight"
    )
    / f"dev_t57_{FAMILY_SLUG}_cuda_history_hoist_100batch_abba_V3"
)

BENCHMARK_JSON = (
    BENCHMARK_DIR
    / (
        f"phase_8_dev_t57_{FAMILY_SLUG}_"
        "cuda_history_hoist_100batch_abba_V3.json"
    )
)


print()
print("=" * 120)
print(
    f"PHASE 8 — DEV_T57 {FAMILY} CUDA HISTORY-HOIST 100-BATCH THROUGHPUT BENCHMARK V3"
)
print("=" * 120)

print()
print("Scientific role:")
print(
    "  Performance characterization only."
)
print(
    f"  Same validated {FAMILY} persistent runtime; "
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

# =============================================================================
# CUDA EXECUTION CONTRACT
# =============================================================================

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

require(
    torch.cuda.is_available(),
    "CUDA is required for CUDA benchmark.",
)

CUDA_DEVICE = torch.device("cuda:0")


# -------------------------------------------------------------------------
# Structured-input CUDA bridge.
#
# make_structured_tensor() is authoritative and intentionally unchanged.
# It materializes the exact frozen structured representation on CPU using
# NumPy and torch.from_numpy(). For CUDA execution only, move its completed
# tensor to the model device immediately after construction.
# -------------------------------------------------------------------------

_original_make_structured_tensor = runtime[
    "make_structured_tensor"
]


def make_structured_tensor_cuda_bridge(
    *args,
    **kwargs,
):

    value = _original_make_structured_tensor(
        *args,
        **kwargs,
    )

    if value is None:
        return None

    return value.to(
        CUDA_DEVICE,
        non_blocking=False,
    )


runtime[
    "make_structured_tensor"
] = make_structured_tensor_cuda_bridge


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)


def move_runtime_value_to_cuda(value):

    if torch.is_tensor(value):
        return value.to(
            CUDA_DEVICE,
            non_blocking=False,
        )

    if isinstance(value, dict):
        return {
            key: move_runtime_value_to_cuda(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            move_runtime_value_to_cuda(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            move_runtime_value_to_cuda(item)
            for item in value
        )

    return value


def move_runtime_to_cuda(ctx):

    return {
        key: move_runtime_value_to_cuda(value)
        for key, value in ctx.items()
    }

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
    "  Training device:           CUDA:0"
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
print(f"3. INITIALIZE CANONICAL {FAMILY} STATE")

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


ctx0 = move_runtime_to_cuda(
    ctx0
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

# -------------------------------------------------------------------------
# Cross-platform canonical-state bridge.
#
# The semantic initializer is still executed above as part of the frozen
# model construction path, but the authoritative parameter values for
# cross-platform execution are loaded from the exact Mac/8.2.7-approved
# canonical state artifact before any training or CUDA transfer.
# -------------------------------------------------------------------------

CANONICAL_STATE_PATH = (
    Path(
        "data/experimental/phase_8/"
        "comparative_development_training_preflight/"
        "canonical_initial_state_bridge_V2/"
        "DEV_T57"
    )
    / f"{FAMILY}_canonical_initial_state_V2.pt"
)

require(
    CANONICAL_STATE_PATH.exists(),
    f"Canonical {FAMILY} initial-state artifact is missing.",
)

canonical_state = torch.load(
    CANONICAL_STATE_PATH,
    map_location="cpu",
    weights_only=True,
)

load_result = model.load_state_dict(
    canonical_state,
    strict=True,
)

require(
    len(load_result.missing_keys) == 0,
    f"Canonical state missing keys: {load_result.missing_keys}",
)

require(
    len(load_result.unexpected_keys) == 0,
    f"Canonical state unexpected keys: {load_result.unexpected_keys}",
)

model.train()


# -------------------------------------------------------------------------
# Canonical initialization is qualified on CPU BEFORE device transfer.
# This preserves the exact frozen 8.2.7 initialization audit.
# -------------------------------------------------------------------------

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
    reference_family[
        "state_audit"
    ][
        "initial_parameter_sha256"
    ],
    f"Canonical {FAMILY} initialization differs from frozen 8.2.7.",
)


print(
    "  Canonical CPU initialization: PASS"
)

print(
    "  Initial CPU model SHA:        EXACT MATCH"
)


# -------------------------------------------------------------------------
# Only after canonical qualification is the verified model transferred
# to CUDA. Adam is instantiated on the CUDA-resident parameters.
# -------------------------------------------------------------------------

model = model.to(
    CUDA_DEVICE
)

torch.cuda.synchronize()


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


print(
    "  CUDA transfer:                PASS"
)

print(
    "  Adam CUDA initial state:      EMPTY PASS"
)



# =============================================================================
# VARIABLE-BATCH QUALIFICATION
# =============================================================================

import gc

print()
print("=" * 120)
print(
    f"5. {FAMILY} VARIABLE-BATCH PRODUCTION QUALIFICATION"
)
print("=" * 120)

require(
    FAMILY == "M1",
    "V1 qualification is intentionally scoped to M1 only.",
)

require(
    epoch_batches == 9801,
    f"DEV_T57 epoch batch count drift: {epoch_batches}",
)

require(
    final_batch_size == 30,
    f"DEV_T57 final batch size drift: {final_batch_size}",
)


def adam_step_values(optimizer):
    values = []

    for state in optimizer.state.values():
        require(
            "step" in state,
            "Adam parameter state missing step counter.",
        )

        value = state["step"]

        if torch.is_tensor(value):
            value = int(
                value.detach().cpu().item()
            )
        else:
            value = int(value)

        values.append(value)

    return values


def fresh_cuda_model_optimizer():
    fresh_model = (
        init_module
        .instantiate_model(
            model_module,
            FAMILY,
            dimensions,
        )
    )

    init_module.initialize_model(
        model=fresh_model,
        family=FAMILY,
        scope="DEV_T57",
        namespace_lookup=namespace_lookup,
        reverse_order=False,
    )

    fresh_state = torch.load(
        CANONICAL_STATE_PATH,
        map_location="cpu",
        weights_only=True,
    )

    load_result = fresh_model.load_state_dict(
        fresh_state,
        strict=True,
    )

    require(
        len(load_result.missing_keys) == 0
        and len(load_result.unexpected_keys) == 0,
        "Canonical state load drift.",
    )

    require(
        runtime["full_parameter_hash"](
            fresh_model
        )
        == reference_family[
            "state_audit"
        ][
            "initial_parameter_sha256"
        ],
        "Fresh canonical CPU state differs from frozen 8.2.7.",
    )

    fresh_model = fresh_model.to(
        CUDA_DEVICE
    )

    torch.cuda.synchronize()

    fresh_optimizer = torch.optim.Adam(
        fresh_model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
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
        len(fresh_optimizer.state) == 0,
        "Fresh Adam state must be empty.",
    )

    return fresh_model, fresh_optimizer


# -------------------------------------------------------------------------
# A. 512-row equivalence qualification
# -------------------------------------------------------------------------

batch_512 = build_batch(0)

require(
    len(batch_512) == 512,
    "Qualification batch 0 is not 512 rows.",
)

with open(os.devnull, "w") as devnull:
    with contextlib.redirect_stdout(devnull):
        ctx_512 = prepare_batch_runtime(
            batch_512
        )
        ctx_512 = move_runtime_to_cuda(
            ctx_512
        )

runtime.update(
    ctx_512
)

torch.cuda.synchronize()

with open(os.devnull, "w") as devnull:
    with contextlib.redirect_stdout(devnull):
        step_512 = persistent_train_step(
            FAMILY,
            model,
            optimizer,
        )

torch.cuda.synchronize()

loss_512 = float(
    step_512[
        "loss_bce_with_logits"
    ]
)

reference_loss = float(
    reference_family[
        "numerical_audit"
    ][
        "loss_bce_with_logits"
    ]
)

loss_abs_diff = abs(
    loss_512
    - reference_loss
)

require(
    bool(
        np.isclose(
            loss_512,
            reference_loss,
            rtol=1e-5,
            atol=1e-6,
        )
    ),
    (
        "Variable-batch 512-row loss is not numerically "
        "equivalent to frozen CPU 8.2.7."
    ),
)

steps_512 = adam_step_values(
    optimizer
)

require(
    len(steps_512) > 0
    and all(
        value == 1
        for value in steps_512
    ),
    "512-row Adam counters are not all 1.",
)

print(
    "  512-row batch:                         PASS"
)
print(
    f"  512-row BCE abs diff vs frozen CPU:    {loss_abs_diff:.12g}"
)
print(
    "  512-row Adam counters:                 ALL 1 PASS"
)


# -------------------------------------------------------------------------
# B. Frozen final 30-row batch execution qualification
# -------------------------------------------------------------------------

del model
del optimizer
del ctx_512
gc.collect()
torch.cuda.empty_cache()

model, optimizer = fresh_cuda_model_optimizer()

final_batch_index = (
    epoch_batches
    - 1
)

batch_30 = build_batch(
    final_batch_index
)

require(
    len(batch_30) == 30,
    (
        "Frozen DEV_T57 final batch does not contain "
        f"30 rows: {len(batch_30)}"
    ),
)

require(
    int(
        batch_30[
            "serialized_example_index"
        ].nunique()
    )
    == 30,
    "Final batch serialized indices are not unique.",
)

with open(os.devnull, "w") as devnull:
    with contextlib.redirect_stdout(devnull):
        ctx_30 = prepare_batch_runtime(
            batch_30
        )
        ctx_30 = move_runtime_to_cuda(
            ctx_30
        )

runtime.update(
    ctx_30
)

torch.cuda.synchronize()

with open(os.devnull, "w") as devnull:
    with contextlib.redirect_stdout(devnull):
        step_30 = persistent_train_step(
            FAMILY,
            model,
            optimizer,
        )

torch.cuda.synchronize()

loss_30 = float(
    step_30[
        "loss_bce_with_logits"
    ]
)

require(
    np.isfinite(
        loss_30
    ),
    "Final 30-row batch produced non-finite BCE.",
)

steps_30 = adam_step_values(
    optimizer
)

require(
    len(steps_30) > 0
    and all(
        value == 1
        for value in steps_30
    ),
    "30-row Adam counters are not all 1.",
)

print(
    "  Frozen final batch index:              9800"
)
print(
    "  Frozen final batch rows:               30 PASS"
)
print(
    f"  Final-batch BCE:                       {loss_30:.12f}"
)
print(
    "  Final-batch Adam counters:             ALL 1 PASS"
)


# -------------------------------------------------------------------------
# C. Persist qualification evidence only; updated states are discarded.
# -------------------------------------------------------------------------

QUAL_DIR = Path(
    "data/experimental/phase_8/"
    "comparative_development_training_preflight/"
    "dev_t57_m1_variable_batch_qualification_V1"
)

QUAL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

QUAL_JSON = (
    QUAL_DIR
    / "phase_8_dev_t57_m1_variable_batch_qualification_V1.json"
)

qualification = {
    "schema_version":
        "PHASE8_DEV_T57_M1_VARIABLE_BATCH_QUALIFICATION_V1",

    "status":
        "VARIABLE_BATCH_RUNTIME_QUALIFIED",

    "family":
        FAMILY,

    "scope":
        "DEV_T57",

    "authoritative_preflight_source":
        str(SOURCE),

    "authoritative_preflight_source_sha256":
        source_sha,

    "canonical_state":
        str(CANONICAL_STATE_PATH),

    "epoch_geometry": {
        "examples":
            int(epoch_examples),

        "batches":
            int(epoch_batches),

        "batch_size":
            int(BATCH_SIZE),

        "final_batch_size":
            int(final_batch_size),
    },

    "qualification": {
        "batch_512": {
            "batch_index":
                0,

            "rows":
                512,

            "loss":
                loss_512,

            "frozen_cpu_reference_loss":
                reference_loss,

            "absolute_difference":
                loss_abs_diff,

            "numerically_equivalent":
                True,

            "adam_steps_all_one":
                True,
        },

        "final_batch_30": {
            "batch_index":
                int(final_batch_index),

            "rows":
                30,

            "loss":
                loss_30,

            "finite_loss":
                True,

            "adam_steps_all_one":
                True,
        },
    },

    "protocol_changes":
        False,

    "mathematical_operations_changed":
        False,

    "variable_batch_change_only":
        (
            "Fixture-only BATCH_SIZE guards and shape/loop audits "
            "generalized to len(batch)."
        ),

    "training_state_retained":
        False,

    "T60_scored":
        False,

    "April_analyzed":
        False,
}

QUAL_JSON.write_text(
    json.dumps(
        qualification,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

del model
del optimizer
del ctx_30
gc.collect()
torch.cuda.empty_cache()

print()
print(
    f"Qualification artifact: {QUAL_JSON}"
)
print()
print(
    "Temporary updated model/Adam state:       DISCARDED"
)
print(
    "T60 scored:                               NO"
)
print(
    "April analyzed:                           NO"
)
print(
    "STATUS: VARIABLE_BATCH_RUNTIME_QUALIFIED"
)
print(
    "NEXT: Build resumable M1 DEV_T57 production runner."
)
print("=" * 120)
