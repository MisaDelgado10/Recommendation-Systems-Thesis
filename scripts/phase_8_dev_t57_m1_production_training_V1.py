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

mode = parser.add_mutually_exclusive_group(
    required=True
)

mode.add_argument(
    "--preflight-only",
    action="store_true",
)

mode.add_argument(
    "--fresh",
    action="store_true",
)

mode.add_argument(
    "--resume",
    action="store_true",
)

parser.add_argument(
    "--stop-after-epoch",
    type=int,
    default=None,
    help=(
        "Safe stop after this many completed + validated display epochs."
    ),
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
# from the frozen 512-row preflight. Use an AST transformation so formatting
# cannot affect the qualification. The persistent step intentionally excludes
# audit-only pair/logit shape assertions, leaving one executable BATCH_SIZE
# occurrence in the trend batch assembly loop.

class _VariableBatchTransformer(ast.NodeTransformer):

    def __init__(self):
        super().__init__()
        self.replacements = 0

    def visit_Name(self, node):

        if (
            isinstance(node.ctx, ast.Load)
            and node.id == "BATCH_SIZE"
        ):
            self.replacements += 1

            return ast.copy_location(
                ast.Call(
                    func=ast.Name(
                        id="len",
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.Name(
                            id="batch",
                            ctx=ast.Load(),
                        )
                    ],
                    keywords=[],
                ),
                node,
            )

        return node


_compute_tree = ast.parse(
    compute_body,
    filename="<phase8_variable_batch_compute_body>",
    mode="exec",
)

_variable_batch_transformer = (
    _VariableBatchTransformer()
)

_compute_tree = (
    _variable_batch_transformer
    .visit(
        _compute_tree
    )
)

ast.fix_missing_locations(
    _compute_tree
)

require(
    _variable_batch_transformer.replacements == 1,
    (
        "Unexpected AST BATCH_SIZE inventory in persistent forward/BCE body: "
        f"{_variable_batch_transformer.replacements}"
    ),
)

compute_body = ast.unparse(
    _compute_tree
)

require(
    not any(
        isinstance(node, ast.Name)
        and node.id == "BATCH_SIZE"
        for node in ast.walk(
            ast.parse(
                compute_body,
                filename="<phase8_variable_batch_compute_body_post>",
                mode="exec",
            )
        )
    ),
    "Variable-batch forward still contains executable BATCH_SIZE names.",
)

print(
    "  Variable-batch AST replacements: 1 PASS"
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
    "  Production training runtime assembled from frozen/qualified components."
)
print(
    f"  Same validated {FAMILY} persistent runtime; "
    "NO architecture/protocol changes."
)
print()
print(
    "  Development-only training. T60 and April remain forbidden."
)
print(
    "  Checkpoint/resume enabled; frozen T57 validation only."
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
# PRODUCTION CONTROLLER — M1 / DEV_T57
# =============================================================================

import gc
import importlib
import math
import os
import random
import subprocess
import sys


require(
    FAMILY == "M1",
    (
        "Production V1 is intentionally qualified only for M1. "
        f"Received {FAMILY}."
    ),
)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    require(
        torch.get_num_interop_threads() == 1,
        (
            "torch interop threads were already initialized "
            f"to {torch.get_num_interop_threads()}, expected 1."
        ),
    )


NUM_EPOCHS = 20
NEGATIVES_PER_POSITIVE = 4
EXAMPLES_PER_EPOCH = len(positives) * SLOTS
BATCHES_PER_EPOCH = int(
    math.ceil(
        EXAMPLES_PER_EPOCH
        / BATCH_SIZE
    )
)
FINAL_BATCH_SIZE = int(
    EXAMPLES_PER_EPOCH
    - (
        BATCHES_PER_EPOCH - 1
    )
    * BATCH_SIZE
)
TOTAL_OPTIMIZER_STEPS = (
    NUM_EPOCHS
    * BATCHES_PER_EPOCH
)

CHECKPOINT_INTERVAL_BATCHES = 250
PROGRESS_PRINT_INTERVAL_BATCHES = 100
VALIDATION_CASES_PER_CHUNK = 5
VALIDATION_ROWS_PER_FULL_CHUNK = (
    VALIDATION_CASES_PER_CHUNK
    * 100
)

require(
    EXAMPLES_PER_EPOCH == 5_017_630,
    "DEV_T57 epoch example count drift.",
)

require(
    BATCHES_PER_EPOCH == 9_801,
    "DEV_T57 batch count drift.",
)

require(
    FINAL_BATCH_SIZE == 30,
    "DEV_T57 final batch size drift.",
)


REPO_ROOT = Path(
    __file__
).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )


PHASE8_ROOT = Path(
    "data/experimental/phase_8"
)

FRAMEWORK_ROOT = (
    PHASE8_ROOT
    / "phase_8_2_framework_development"
)

RUNTIME_CONTRACT_PATH = (
    FRAMEWORK_ROOT
    / "phase_8_2_8_framework_integration_closure"
    / "phase_8_2_8_training_runtime_contract_V6.json"
)

FINGERPRINT_REGISTRY_PATH = (
    FRAMEWORK_ROOT
    / "phase_8_2_6_training_stream_dry_run"
    / "phase_8_2_6_development_epoch_fingerprints_V2.csv"
)

QUALIFICATION_PATH = (
    PHASE8_ROOT
    / "comparative_development_training_preflight"
    / "dev_t57_m1_variable_batch_qualification_V1"
    / "phase_8_dev_t57_m1_variable_batch_qualification_V1.json"
)

CANDIDATE_DIR = (
    PHASE8_ROOT
    / "phase_8_0_evaluation_reset"
    / "phase_8_0_11_evaluation_candidates"
)

VALIDATION_CASE_MANIFEST_PATH = (
    CANDIDATE_DIR
    / "phase_8_0_11_evaluation_case_manifest_V1.parquet"
)

VALIDATION_NEGATIVE_MATRIX_PATH = (
    CANDIDATE_DIR
    / "phase_8_0_11_t57_negative_node_indices_V1.npy"
)

CANDIDATE_HASH_REGISTRY_PATH = (
    CANDIDATE_DIR
    / "phase_8_0_11_derived_artifact_sha256_V1.json"
)

NODE_INDEX_PATH = Path(
    "data/experimental/phase_3/model_ready/"
    "node_index.parquet"
)

TEMPORAL_SPLIT_PATH = Path(
    "data/experimental/phase_2/model_ready/"
    "interactions_itrs_temporal_split.parquet"
)

ELIGIBLE_LOCAL_PATH = (
    STREAM_DIR
    / "origin_eligible_startup_local.npy"
)

RANKING_RUNTIME_PATH = Path(
    "scripts/"
    "phase_5_3_3a_validation_ranking_metric_semantics_audit.py"
)

GRAPH_EDGE_INDEX_PATH = (
    GRAPH_DIR
    / "phase_8_2_2_t57_V2_edge_index.npy"
)

GRAPH_EDGE_TYPE_PATH = (
    GRAPH_DIR
    / "phase_8_2_2_t57_V2_edge_type.npy"
)

GRAPH_FIT_MASK_PATH = (
    GRAPH_DIR
    / "phase_8_2_2_t57_V2_fit_edge_mask.npy"
)

GRAPH_INFERENCE_MASK_PATH = (
    GRAPH_DIR
    / "phase_8_2_2_t57_V2_inference_edge_mask.npy"
)


RUN_DIR = (
    PHASE8_ROOT
    / "comparative_development_training"
    / "M1"
    / "DEV_T57"
)

CHECKPOINT_DIR = (
    RUN_DIR
    / "checkpoints"
)

VALIDATION_DIR = (
    RUN_DIR
    / "validation"
)

LATEST_CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "latest.pt"
)

PROGRESS_PATH = (
    RUN_DIR
    / "progress.json"
)

EPOCH_METRICS_PATH = (
    RUN_DIR
    / "epoch_metrics.csv"
)

FINAL_MANIFEST_PATH = (
    RUN_DIR
    / "phase_8_m1_dev_t57_training_manifest_V1.json"
)


STREAM_MODULE_NAME = (
    "scripts."
    "phase_8_2_6_dev_t57_epoch0_stream_materialization_V1"
)

COVERAGE_MODULE_NAME = (
    "scripts."
    "phase_8_2_6_development_epoch_fingerprint_coverage_V2"
)


def load_json(path):

    return json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


def git_head():

    return (
        subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            text=True,
        )
        .strip()
    )


def require_clean_tracked_tree():

    result = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    require(
        result.stdout.strip() == "",
        (
            "Tracked Git working tree is not clean.\n"
            + result.stdout
        ),
    )


def recursive_cpu_copy(value):

    if torch.is_tensor(value):
        return (
            value
            .detach()
            .cpu()
            .clone()
        )

    if isinstance(
        value,
        dict,
    ):
        return {
            key:
                recursive_cpu_copy(
                    item
                )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        list,
    ):
        return [
            recursive_cpu_copy(
                item
            )
            for item
            in value
        ]

    if isinstance(
        value,
        tuple,
    ):
        return tuple(
            recursive_cpu_copy(
                item
            )
            for item
            in value
        )

    return value


def clone_model_state_dict(
    model,
):

    return {
        key:
            value
            .detach()
            .cpu()
            .clone()
        for key, value
        in model.state_dict().items()
    }


def atomic_torch_save(
    payload,
    path,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = Path(
        str(path)
        + ".tmp"
    )

    torch.save(
        payload,
        temporary,
    )

    os.replace(
        temporary,
        path,
    )


def atomic_json_write(
    payload,
    path,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = Path(
        str(path)
        + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


def atomic_csv_write(
    frame,
    path,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = Path(
        str(path)
        + ".tmp"
    )

    frame.to_csv(
        temporary,
        index=False,
    )

    os.replace(
        temporary,
        path,
    )


# =============================================================================
# FINAL AUTHORIZATION + FROZEN ARTIFACT GATE
# =============================================================================

required_production_paths = [
    RUNTIME_CONTRACT_PATH,
    FINGERPRINT_REGISTRY_PATH,
    QUALIFICATION_PATH,
    VALIDATION_CASE_MANIFEST_PATH,
    VALIDATION_NEGATIVE_MATRIX_PATH,
    CANDIDATE_HASH_REGISTRY_PATH,
    NODE_INDEX_PATH,
    TEMPORAL_SPLIT_PATH,
    ELIGIBLE_LOCAL_PATH,
    RANKING_RUNTIME_PATH,
    GRAPH_EDGE_INDEX_PATH,
    GRAPH_EDGE_TYPE_PATH,
    GRAPH_FIT_MASK_PATH,
    GRAPH_INFERENCE_MASK_PATH,
    CANONICAL_STATE_PATH,
]

for path in required_production_paths:

    require(
        path.exists(),
        f"Missing production prerequisite: {path}",
    )


training_contract = load_json(
    RUNTIME_CONTRACT_PATH
)

require(
    training_contract[
        "status"
    ]
    ==
    "FROZEN_FOR_COMPARATIVE_DEVELOPMENT_TRAINING",
    "8.2.8 runtime contract is not frozen for development training.",
)

require(
    training_contract[
        "comparative_development_training_authorized"
    ]
    is True,
    "8.2.8 does not authorize comparative development training.",
)

require(
    "M1"
    in
    training_contract[
        "authorized_models"
    ],
    "M1 is not authorized by 8.2.8.",
)

require(
    "DEV_T57"
    in
    training_contract[
        "development_folds"
    ],
    "DEV_T57 is not authorized by 8.2.8.",
)

require(
    training_contract[
        "epoch_policy"
    ][
        "maximum_epochs"
    ]
    == NUM_EPOCHS,
    "Maximum epoch contract drift.",
)

require(
    training_contract[
        "epoch_policy"
    ][
        "early_stopping"
    ]
    is False,
    "Early stopping unexpectedly enabled.",
)

require(
    training_contract[
        "training_stream"
    ][
        "batch_size"
    ]
    == BATCH_SIZE,
    "Batch-size contract drift.",
)

require(
    training_contract[
        "training_stream"
    ][
        "drop_last"
    ]
    is False,
    "drop_last contract drift.",
)

require(
    training_contract[
        "development_performance_policy"
    ][
        "T60_may_select_epoch"
    ]
    is False
    and
    training_contract[
        "development_performance_policy"
    ][
        "April_may_select_epoch"
    ]
    is False,
    "T60/April selection boundary drift.",
)


qualification = load_json(
    QUALIFICATION_PATH
)

require(
    qualification[
        "status"
    ]
    == "VARIABLE_BATCH_RUNTIME_QUALIFIED",
    "Variable-batch runtime is not qualified.",
)

require(
    qualification[
        "family"
    ]
    == "M1"
    and qualification[
        "scope"
    ]
    == "DEV_T57",
    "Variable-batch qualification scope drift.",
)

require(
    qualification[
        "epoch_geometry"
    ][
        "final_batch_size"
    ]
    == 30,
    "Variable-batch qualification final batch drift.",
)


candidate_hashes = load_json(
    CANDIDATE_HASH_REGISTRY_PATH
)

for path in [
    VALIDATION_CASE_MANIFEST_PATH,
    VALIDATION_NEGATIVE_MATRIX_PATH,
]:

    require(
        str(path)
        in candidate_hashes,
        (
            "Candidate hash registry is missing "
            f"{path}."
        ),
    )

    require(
        sha256_file(
            path
        )
        ==
        candidate_hashes[
            str(path)
        ],
        (
            "Frozen candidate artifact SHA drift: "
            f"{path}"
        ),
    )


# =============================================================================
# FROZEN T57 VALIDATION BINDING
# =============================================================================

validation_cases = (
    pd.read_parquet(
        VALIDATION_CASE_MANIFEST_PATH
    )
    .loc[
        lambda frame:
            frame[
                "origin"
            ]
            ==
            "T57"
    ]
    .sort_values(
        "case_index",
        kind="mergesort",
    )
    .reset_index(
        drop=True
    )
)

validation_negatives = np.load(
    VALIDATION_NEGATIVE_MATRIX_PATH,
    mmap_mode="r",
)

require(
    len(
        validation_cases
    )
    == 21_924,
    "T57 validation case count drift.",
)

require(
    validation_cases[
        "case_index"
    ].tolist()
    ==
    list(
        range(
            21_924
        )
    ),
    "T57 validation case_index ordering drift.",
)

require(
    int(
        validation_cases[
            "is_primary_stratum"
        ].sum()
    )
    == 6_850,
    "T57 PRIMARY case count drift.",
)

require(
    validation_negatives.shape
    ==
    (
        21_924,
        99,
    ),
    "T57 validation negative matrix shape drift.",
)

require(
    validation_negatives.dtype
    ==
    np.dtype(
        np.int64
    ),
    "T57 validation negative dtype drift.",
)


node_index = pd.read_parquet(
    NODE_INDEX_PATH,
    columns=[
        "node_index",
        "node_type",
        "raw_entity_id",
    ],
)

investor_nodes = (
    node_index.loc[
        node_index[
            "node_type"
        ]
        .astype(str)
        .str.lower()
        .eq(
            "investor"
        ),
        [
            "node_index",
            "raw_entity_id",
        ],
    ]
)

investor_id_to_global = dict(
    zip(
        investor_nodes[
            "raw_entity_id"
        ].astype(str),
        investor_nodes[
            "node_index"
        ].astype(
            np.int64
        ),
    )
)

validation_investor_global = (
    validation_cases[
        "investor_id"
    ]
    .astype(str)
    .map(
        investor_id_to_global
    )
)

require(
    validation_investor_global
    .notna()
    .all(),
    "At least one T57 validation Investor is absent from role registry.",
)

validation_cases[
    "investor_global"
] = (
    validation_investor_global
    .to_numpy(
        dtype=np.int64
    )
)


# =============================================================================
# INFERENCE PREPARATION DERIVED FROM AUTHORITATIVE 8.2.7 AST
# =============================================================================

inference_prepare_body = str(
    prepare_body
)

_fit_mask_filename = (
    '"phase_8_2_2_t57_V2_fit_edge_mask.npy"'
)

_inference_mask_filename = (
    '"phase_8_2_2_t57_V2_inference_edge_mask.npy"'
)

require(
    inference_prepare_body.count(
        _fit_mask_filename
    )
    == 1,
    (
        "Inference-preparation fit-mask filename inventory changed: "
        f"{inference_prepare_body.count(_fit_mask_filename)}"
    ),
)

inference_prepare_body = (
    inference_prepare_body.replace(
        _fit_mask_filename,
        _inference_mask_filename,
    )
)


class _InferenceVisibilityTransformer(
    ast.NodeTransformer
):

    def __init__(
        self,
    ):

        super().__init__()

        self.replacements = 0

    def visit_Assign(
        self,
        node,
    ):

        if (
            len(
                node.targets
            )
            == 1
            and isinstance(
                node.targets[
                    0
                ],
                ast.Name,
            )
            and node.targets[
                0
            ].id
            ==
            "feature_visibility_np"
        ):

            self.replacements += 1

            node.value = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(
                        id="np",
                        ctx=ast.Load(),
                    ),
                    attr="ones",
                    ctx=ast.Load(),
                ),
                args=[
                    ast.Call(
                        func=ast.Name(
                            id="len",
                            ctx=ast.Load(),
                        ),
                        args=[
                            ast.Name(
                                id="needed_feature_rows",
                                ctx=ast.Load(),
                            )
                        ],
                        keywords=[],
                    )
                ],
                keywords=[
                    ast.keyword(
                        arg="dtype",
                        value=ast.Name(
                            id="bool",
                            ctx=ast.Load(),
                        ),
                    )
                ],
            )

        return self.generic_visit(
            node
        )


_inference_prepare_tree = ast.parse(
    inference_prepare_body,
    filename="<phase8_m1_t57_inference_prepare>",
    mode="exec",
)

_visibility_transformer = (
    _InferenceVisibilityTransformer()
)

_inference_prepare_tree = (
    _visibility_transformer
    .visit(
        _inference_prepare_tree
    )
)

ast.fix_missing_locations(
    _inference_prepare_tree
)

require(
    _visibility_transformer.replacements
    == 1,
    (
        "Expected exactly one COLD_t inference-visibility "
        "restoration assignment."
    ),
)

inference_prepare_body = ast.unparse(
    _inference_prepare_tree
)

prepare_inference_function_source = (
    "def prepare_inference_runtime(batch):\n"
    + textwrap.indent(
        inference_prepare_body,
        "    ",
    )
    + "\n"
    + textwrap.indent(
        """
require(
    1 <= len(batch) <= 500,
    "Validation chunk must contain 1..500 candidate rows.",
)

require(
    positive_count
    + negative_count
    == len(batch),
    "Validation candidate label accounting mismatch.",
)

require(
    bool(
        feature_visibility.all()
    ),
    "Inference did not restore all materialized Startup features.",
)

return locals()
""".strip(),
        "    ",
    )
    + "\n"
)

compile(
    prepare_inference_function_source,
    "<phase8_m1_t57_prepare_inference_runtime>",
    "exec",
)

exec(
    compile(
        prepare_inference_function_source,
        "<phase8_m1_t57_prepare_inference_runtime>",
        "exec",
    ),
    runtime,
)

prepare_inference_runtime = (
    runtime[
        "prepare_inference_runtime"
    ]
)


# =============================================================================
# SCORE-ONLY FORWARD DERIVED FROM AUTHORITATIVE COMPUTE NODES
# =============================================================================

score_line_starts = {
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
}

score_nodes = [
    node
    for node
    in family_loop.body
    if node.lineno
    in score_line_starts
]

require(
    {
        node.lineno
        for node
        in score_nodes
    }
    == score_line_starts,
    "Inference score AST node inventory changed.",
)

score_body = complete_source_for_nodes(
    score_nodes
)

score_body, _removed_inference_attention_audit = (
    remove_attention_finite_sync_audit(
        score_body
    )
)

score_body = (
    apply_history_item_hoist_candidate(
        score_body
    )
)

_score_tree = ast.parse(
    score_body,
    filename="<phase8_m1_t57_score_body>",
    mode="exec",
)

_score_variable_batch_transformer = (
    _VariableBatchTransformer()
)

_score_tree = (
    _score_variable_batch_transformer
    .visit(
        _score_tree
    )
)

ast.fix_missing_locations(
    _score_tree
)

require(
    _score_variable_batch_transformer.replacements
    == 1,
    (
        "Unexpected score-only BATCH_SIZE AST inventory: "
        f"{_score_variable_batch_transformer.replacements}"
    ),
)

score_body = ast.unparse(
    _score_tree
)

score_body = (
    "with torch.device(next(model.parameters()).device):\n"
    + textwrap.indent(
        score_body,
        "    ",
    )
)

score_function_source = (
    "def score_inference_step(family, model):\n"
    + textwrap.indent(
        score_body,
        "    ",
    )
    + "\n"
    + textwrap.indent(
        """
return logits
""".strip(),
        "    ",
    )
    + "\n"
)

compile(
    score_function_source,
    "<phase8_m1_t57_score_inference_step>",
    "exec",
)

exec(
    compile(
        score_function_source,
        "<phase8_m1_t57_score_inference_step>",
        "exec",
    ),
    runtime,
)

score_inference_step = (
    runtime[
        "score_inference_step"
    ]
)


# =============================================================================
# RANKING RUNTIME
# =============================================================================

ranking_runtime = importlib.import_module(
    "scripts."
    "phase_5_3_3a_validation_ranking_metric_semantics_audit"
)


# =============================================================================
# GRAPH MASK STATIC CHECK
# =============================================================================

_graph_edge_index = np.load(
    GRAPH_EDGE_INDEX_PATH,
    mmap_mode="r",
)

_graph_edge_type = np.load(
    GRAPH_EDGE_TYPE_PATH,
    mmap_mode="r",
)

_graph_fit_mask = np.load(
    GRAPH_FIT_MASK_PATH,
    mmap_mode="r",
)

_graph_inference_mask = np.load(
    GRAPH_INFERENCE_MASK_PATH,
    mmap_mode="r",
)

require(
    _graph_edge_index.ndim == 2
    and _graph_edge_index.shape[0] == 2,
    "T57 graph edge_index shape drift.",
)

require(
    _graph_edge_index.shape[1]
    == len(
        _graph_edge_type
    )
    == len(
        _graph_fit_mask
    )
    == len(
        _graph_inference_mask
    ),
    "T57 graph edge/mask length mismatch.",
)

require(
    int(
        np.asarray(
            _graph_inference_mask,
            dtype=bool,
        ).sum()
    )
    >=
    int(
        np.asarray(
            _graph_fit_mask,
            dtype=bool,
        ).sum()
    ),
    "Inference graph unexpectedly has fewer edges than fit graph.",
)

del _graph_edge_index
del _graph_edge_type
del _graph_fit_mask
del _graph_inference_mask


# =============================================================================
# STREAM REGISTRY + GENERATION
# =============================================================================

fingerprints = (
    pd.read_csv(
        FINGERPRINT_REGISTRY_PATH
    )
    .loc[
        lambda frame:
            frame[
                "scope_id"
            ]
            ==
            "DEV_T57"
    ]
    .sort_values(
        "epoch_index",
        kind="mergesort",
    )
    .reset_index(
        drop=True
    )
)

require(
    fingerprints[
        "epoch_index"
    ].tolist()
    ==
    list(
        range(
            NUM_EPOCHS
        )
    ),
    "DEV_T57 fingerprint registry epoch coverage drift.",
)

require(
    fingerprints[
        "order_regeneration_exact"
    ]
    .astype(bool)
    .all(),
    "DEV_T57 order-regeneration registry contains a failure.",
)


stream_module = importlib.import_module(
    STREAM_MODULE_NAME
)

positive_logical_sha = (
    stream_module
    .positive_stream_logical_sha256(
        positives
    )
)

require(
    (
        fingerprints[
            "positive_order_sha256"
        ]
        ==
        positive_logical_sha
    ).all(),
    "DEV_T57 positive stream logical SHA drift.",
)

require(
    stream_module
    .array_logical_sha256(
        np.asarray(
            negative_matrix
        )
    )
    ==
    str(
        fingerprints
        .iloc[
            0
        ][
            "negative_matrix_sha256"
        ]
    ),
    "DEV_T57 epoch-0 negative matrix logical SHA drift.",
)

require(
    stream_module
    .array_logical_sha256(
        np.asarray(
            epoch_order
        )
    )
    ==
    str(
        fingerprints
        .iloc[
            0
        ][
            "epoch_order_sha256"
        ]
    ),
    "DEV_T57 epoch-0 example order logical SHA drift.",
)


_generation_context = None


def get_generation_context():

    global _generation_context

    if _generation_context is not None:
        return _generation_context

    coverage_module = importlib.import_module(
        COVERAGE_MODULE_NAME
    )

    (
        investor_map,
        startup_map,
    ) = (
        coverage_module
        .build_role_maps()
    )

    temporal = pd.read_parquet(
        TEMPORAL_SPLIT_PATH,
        columns=[
            "investor_id",
            "startup_id",
            "segment_number",
        ],
    )

    temporal[
        "segment_number"
    ] = (
        temporal[
            "segment_number"
        ]
        .astype(
            np.int64
        )
    )

    (
        pair_first_segment,
        history_count,
        pair_count,
    ) = (
        coverage_module
        .build_pair_first_segment(
            temporal=temporal,
            investor_map=investor_map,
            startup_map=startup_map,
            max_segment=56,
        )
    )

    eligible_startup_locals = np.load(
        ELIGIBLE_LOCAL_PATH,
        allow_pickle=False,
    ).astype(
        np.int64,
        copy=False,
    )

    require(
        len(
            eligible_startup_locals
        )
        == 308_027,
        "DEV_T57 eligible Startup count drift.",
    )

    _generation_context = {
        "pair_first_segment":
            pair_first_segment,

        "eligible_startup_locals":
            eligible_startup_locals,

        "history_count":
            int(
                history_count
            ),

        "pair_count":
            int(
                pair_count
            ),
    }

    return _generation_context


def load_epoch_stream(
    epoch_index,
):

    global negative_matrix
    global epoch_order

    require(
        0
        <= epoch_index
        < NUM_EPOCHS,
        "Epoch index outside 0..19.",
    )

    row = (
        fingerprints
        .iloc[
            epoch_index
        ]
    )

    if epoch_index == 0:

        current_negative = np.load(
            NEGATIVE_MATRIX,
            mmap_mode="r",
        )

        current_order = np.load(
            EPOCH_ORDER,
            mmap_mode="r",
        )

    else:

        ctx = get_generation_context()

        (
            negative_seed,
            _negative_payload,
        ) = (
            stream_module
            .derive_seed(
                stream_module
                .NEGATIVE_NAMESPACE,
                "DEV_T57",
                epoch_index,
            )
        )

        (
            order_seed,
            _order_payload,
        ) = (
            stream_module
            .derive_seed(
                stream_module
                .ORDER_NAMESPACE,
                "DEV_T57",
                epoch_index,
            )
        )

        require(
            int(
                negative_seed
            )
            ==
            int(
                row[
                    "negative_seed_u64"
                ]
            ),
            (
                f"Epoch {epoch_index}: "
                "negative seed drift."
            ),
        )

        require(
            int(
                order_seed
            )
            ==
            int(
                row[
                    "order_seed_u64"
                ]
            ),
            (
                f"Epoch {epoch_index}: "
                "order seed drift."
            ),
        )

        investor_globals = (
            positives[
                "investor_global"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        target_segments = (
            positives[
                "segment_number"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        (
            current_negative,
            _stats,
        ) = (
            stream_module
            .generate_epoch_negatives(
                investor_globals=(
                    investor_globals
                ),
                target_segments=(
                    target_segments
                ),
                pair_first_segment=(
                    ctx[
                        "pair_first_segment"
                    ]
                ),
                eligible_startup_locals=(
                    ctx[
                        "eligible_startup_locals"
                    ]
                ),
                num_startups=311_589,
                seed=(
                    negative_seed
                ),
                max_rows=None,
                progress=False,
            )
        )

        current_order = np.arange(
            EXAMPLES_PER_EPOCH,
            dtype=np.int64,
        )

        rng = np.random.Generator(
            np.random.PCG64(
                order_seed
            )
        )

        rng.shuffle(
            current_order
        )

    require(
        current_negative.shape
        ==
        (
            len(
                positives
            ),
            NEGATIVES_PER_POSITIVE,
        ),
        (
            f"Epoch {epoch_index}: "
            "negative matrix shape drift."
        ),
    )

    require(
        len(
            current_order
        )
        == EXAMPLES_PER_EPOCH,
        (
            f"Epoch {epoch_index}: "
            "example order length drift."
        ),
    )

    negative_sha = (
        stream_module
        .array_logical_sha256(
            np.asarray(
                current_negative
            )
        )
    )

    order_sha = (
        stream_module
        .array_logical_sha256(
            np.asarray(
                current_order
            )
        )
    )

    require(
        negative_sha
        ==
        str(
            row[
                "negative_matrix_sha256"
            ]
        ),
        (
            f"Epoch {epoch_index}: "
            "negative logical SHA drift."
        ),
    )

    require(
        order_sha
        ==
        str(
            row[
                "epoch_order_sha256"
            ]
        ),
        (
            f"Epoch {epoch_index}: "
            "order logical SHA drift."
        ),
    )

    negative_matrix = current_negative
    epoch_order = current_order

    return {
        "negative":
            current_negative,

        "order":
            current_order,

        "negative_sha256":
            negative_sha,

        "order_sha256":
            order_sha,
    }


# =============================================================================
# VALIDATION BATCH CONSTRUCTION + FULL FROZEN T57 VALIDATION
# =============================================================================

def build_validation_candidate_frame(
    start_case,
    end_case,
):

    rows = []

    for position in range(
        start_case,
        end_case,
    ):

        case = (
            validation_cases
            .iloc[
                position
            ]
        )

        case_index = int(
            case[
                "case_index"
            ]
        )

        investor_global = int(
            case[
                "investor_global"
            ]
        )

        positive_global = int(
            case[
                "positive_node_index"
            ]
        )

        candidates_global = np.concatenate(
            [
                np.asarray(
                    [
                        positive_global
                    ],
                    dtype=np.int64,
                ),
                np.asarray(
                    validation_negatives[
                        case_index
                    ],
                    dtype=np.int64,
                ),
            ]
        )

        require(
            candidates_global.shape
            ==
            (
                100,
            ),
            "Validation candidate row is not 100.",
        )

        require(
            np.unique(
                candidates_global
            ).size
            == 100,
            (
                f"Validation case {case_index}: "
                "candidate duplication."
            ),
        )

        for candidate_position, startup_global in enumerate(
            candidates_global.tolist()
        ):

            rows.append(
                {
                    "label":
                        int(
                            candidate_position
                            == 0
                        ),

                    "investor_global":
                        investor_global,

                    "startup_global":
                        int(
                            startup_global
                        ),

                    "segment_number":
                        57,

                    "trend_history_period_count":
                        57,

                    "validation_case_index":
                        case_index,

                    "validation_candidate_position":
                        int(
                            candidate_position
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


def full_t57_validation(
    *,
    epoch_index,
    model,
    optimizer,
):

    print()
    print("=" * 120)
    print(
        f"EPOCH {epoch_index + 1} — "
        "FROZEN DEV_T57 VALIDATION"
    )
    print("=" * 120)

    optimizer.zero_grad(
        set_to_none=True
    )

    model.eval()

    state_before = (
        runtime[
            "full_parameter_hash"
        ](
            model
        )
    )

    metric_rows = []

    validation_start = time.perf_counter()

    with torch.no_grad():

        for start_case in range(
            0,
            len(
                validation_cases
            ),
            VALIDATION_CASES_PER_CHUNK,
        ):

            end_case = min(
                start_case
                + VALIDATION_CASES_PER_CHUNK,
                len(
                    validation_cases
                ),
            )

            batch = (
                build_validation_candidate_frame(
                    start_case,
                    end_case,
                )
            )

            require(
                len(
                    batch
                )
                ==
                (
                    end_case
                    - start_case
                )
                * 100,
                "Validation chunk row geometry drift.",
            )

            with open(
                os.devnull,
                "w",
            ) as devnull:

                with contextlib.redirect_stdout(
                    devnull
                ):

                    ctx = (
                        prepare_inference_runtime(
                            batch
                        )
                    )

                    ctx = (
                        move_runtime_to_cuda(
                            ctx
                        )
                    )

            runtime.update(
                ctx
            )

            torch.cuda.synchronize()

            with open(
                os.devnull,
                "w",
            ) as devnull:

                with contextlib.redirect_stdout(
                    devnull
                ):

                    logits = (
                        score_inference_step(
                            FAMILY,
                            model,
                        )
                    )

            torch.cuda.synchronize()

            logits_np = (
                logits
                .detach()
                .cpu()
                .numpy()
                .reshape(
                    -1
                )
                .astype(
                    np.float32,
                    copy=False,
                )
            )

            require(
                np.isfinite(
                    logits_np
                ).all(),
                "Validation logits contain NaN/inf.",
            )

            expected_rows = (
                end_case
                - start_case
            ) * 100

            require(
                len(
                    logits_np
                )
                == expected_rows,
                "Validation logit count drift.",
            )

            for local_case_position, case_position in enumerate(
                range(
                    start_case,
                    end_case,
                )
            ):

                case = (
                    validation_cases
                    .iloc[
                        case_position
                    ]
                )

                case_index = int(
                    case[
                        "case_index"
                    ]
                )

                positive_global = int(
                    case[
                        "positive_node_index"
                    ]
                )

                candidates_global = np.concatenate(
                    [
                        np.asarray(
                            [
                                positive_global
                            ],
                            dtype=np.int64,
                        ),
                        np.asarray(
                            validation_negatives[
                                case_index
                            ],
                            dtype=np.int64,
                        ),
                    ]
                )

                candidates_local = (
                    candidates_global
                    - NUM_INVESTORS
                )

                positive_local = (
                    positive_global
                    - NUM_INVESTORS
                )

                start_row = (
                    local_case_position
                    * 100
                )

                end_row = (
                    start_row
                    + 100
                )

                case_logits = (
                    logits_np[
                        start_row:
                        end_row
                    ]
                )

                positive_rank = (
                    ranking_runtime
                    .rank_candidates(
                        case_logits,
                        candidates_local,
                        int(
                            positive_local
                        ),
                    )
                )

                (
                    hr10,
                    ndcg10,
                ) = (
                    ranking_runtime
                    .metrics_from_positive_rank(
                        int(
                            positive_rank
                        ),
                        k=10,
                    )
                )

                metric_rows.append(
                    {
                        "epoch_index":
                            int(
                                epoch_index
                            ),

                        "display_epoch":
                            int(
                                epoch_index
                                + 1
                            ),

                        "case_index":
                            case_index,

                        "interaction_id":
                            str(
                                case[
                                    "interaction_id"
                                ]
                            ),

                        "is_primary_stratum":
                            bool(
                                case[
                                    "is_primary_stratum"
                                ]
                            ),

                        "positive_rank":
                            int(
                                positive_rank
                            ),

                        "HR@10":
                            float(
                                hr10
                            ),

                        "NDCG@10":
                            float(
                                ndcg10
                            ),
                    }
                )

            del ctx
            del logits
            del batch

            if (
                start_case == 0
                or end_case
                % 2_500
                == 0
                or end_case
                ==
                len(
                    validation_cases
                )
            ):

                print(
                    f"  validation cases: "
                    f"{end_case:,}/"
                    f"{len(validation_cases):,}"
                )

    validation_seconds = (
        time.perf_counter()
        - validation_start
    )

    metrics = pd.DataFrame(
        metric_rows
    )

    require(
        len(
            metrics
        )
        == 21_924,
        "Validation did not score exactly 21,924 cases.",
    )

    require(
        bool(
            metrics[
                "positive_rank"
            ]
            .between(
                1,
                100,
            )
            .all()
        ),
        "Validation positive rank outside 1..100.",
    )

    primary = (
        metrics.loc[
            metrics[
                "is_primary_stratum"
            ]
        ]
        .copy()
    )

    require(
        len(
            primary
        )
        == 6_850,
        "Validation PRIMARY count drift.",
    )

    (
        primary_hr10,
        primary_ndcg10,
    ) = (
        ranking_runtime
        .aggregate_event_level_metrics(
            primary
        )
    )

    (
        aggregate_hr10,
        aggregate_ndcg10,
    ) = (
        ranking_runtime
        .aggregate_event_level_metrics(
            metrics
        )
    )

    state_after = (
        runtime[
            "full_parameter_hash"
        ](
            model
        )
    )

    require(
        state_after
        == state_before,
        "Validation changed model parameter bytes.",
    )

    validation_path = (
        VALIDATION_DIR
        / (
            f"epoch_{epoch_index:02d}_"
            "t57_validation_cases.parquet"
        )
    )

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics.to_parquet(
        validation_path,
        index=False,
    )

    model.train()

    print()
    print(
        f"  PRIMARY HR@10:              "
        f"{primary_hr10:.12f}"
    )

    print(
        f"  PRIMARY NDCG@10:            "
        f"{primary_ndcg10:.12f}"
    )

    print(
        f"  Aggregate eligible HR@10:   "
        f"{aggregate_hr10:.12f}"
    )

    print(
        f"  Aggregate eligible NDCG@10: "
        f"{aggregate_ndcg10:.12f}"
    )

    print(
        f"  Validation seconds:         "
        f"{validation_seconds:.2f}"
    )

    print(
        "  T60 cases scored:           0"
    )

    print(
        "  April cases scored:         0"
    )

    return {
        "primary_HR@10":
            float(
                primary_hr10
            ),

        "primary_NDCG@10":
            float(
                primary_ndcg10
            ),

        "aggregate_HR@10":
            float(
                aggregate_hr10
            ),

        "aggregate_NDCG@10":
            float(
                aggregate_ndcg10
            ),

        "validation_seconds":
            float(
                validation_seconds
            ),

        "case_metrics_path":
            str(
                validation_path
            ),
    }


# =============================================================================
# RUNTIME BINDINGS + CHECKPOINT CONTROLLER
# =============================================================================

repository_commit = git_head()

runtime_bindings = {
    "repository_commit":
        repository_commit,

    "trainer_script_sha256":
        sha256_file(
            Path(
                __file__
            )
        ),

    "phase_8_2_8_runtime_contract_sha256":
        sha256_file(
            RUNTIME_CONTRACT_PATH
        ),

    "variable_batch_qualification_sha256":
        sha256_file(
            QUALIFICATION_PATH
        ),

    "development_fingerprint_registry_sha256":
        sha256_file(
            FINGERPRINT_REGISTRY_PATH
        ),

    "candidate_hash_registry_sha256":
        sha256_file(
            CANDIDATE_HASH_REGISTRY_PATH
        ),

    "canonical_initial_state_sha256":
        sha256_file(
            CANONICAL_STATE_PATH
        ),

    "authoritative_8_2_7_source_sha256":
        source_sha,
}


def make_fresh_state():

    return {
        "epoch_index":
            0,

        "next_batch_index":
            0,

        "global_optimizer_step":
            0,

        "epoch_loss_weighted_sum":
            0.0,

        "epoch_example_count":
            0,

        "epoch_training_seconds":
            0.0,

        "validation_pending":
            False,

        "validation_history":
            [],

        "training_complete":
            False,

        "T60_scored":
            False,

        "April_scored":
            False,
    }


def progress_payload(
    state,
):

    return {
        "phase":
            "8_COMPARATIVE_DEVELOPMENT",

        "family":
            "M1",

        "scope":
            "DEV_T57",

        "runtime_bindings":
            runtime_bindings,

        "epoch_index":
            int(
                state[
                    "epoch_index"
                ]
            ),

        "display_epoch":
            int(
                state[
                    "epoch_index"
                ]
            )
            + 1,

        "next_batch_index":
            int(
                state[
                    "next_batch_index"
                ]
            ),

        "batches_per_epoch":
            BATCHES_PER_EPOCH,

        "global_optimizer_step":
            int(
                state[
                    "global_optimizer_step"
                ]
            ),

        "total_optimizer_steps":
            TOTAL_OPTIMIZER_STEPS,

        "completed_validations":
            len(
                state[
                    "validation_history"
                ]
            ),

        "validation_pending":
            bool(
                state[
                    "validation_pending"
                ]
            ),

        "training_complete":
            bool(
                state[
                    "training_complete"
                ]
            ),

        "T60_scored":
            False,

        "April_scored":
            False,
    }


def checkpoint_payload(
    state,
):

    return {
        "schema_version":
            "PHASE8_M1_DEV_T57_PRODUCTION_V1",

        "family":
            "M1",

        "scope":
            "DEV_T57",

        "runtime_bindings":
            runtime_bindings,

        "controller_state":
            state,

        "model_state_dict":
            clone_model_state_dict(
                model
            ),

        "optimizer_state_dict":
            recursive_cpu_copy(
                optimizer.state_dict()
            ),

        "python_rng_state":
            random.getstate(),

        "numpy_rng_state":
            np.random.get_state(),

        "torch_rng_state":
            torch.get_rng_state()
            .clone(),

        "cuda_rng_state":
            torch.cuda.get_rng_state(
                device=0
            )
            .cpu()
            .clone(),

        "training_schedule": {
            "epochs":
                NUM_EPOCHS,

            "examples_per_epoch":
                EXAMPLES_PER_EPOCH,

            "batches_per_epoch":
                BATCHES_PER_EPOCH,

            "batch_size":
                BATCH_SIZE,

            "final_batch_size":
                FINAL_BATCH_SIZE,

            "checkpoint_interval_batches":
                CHECKPOINT_INTERVAL_BATCHES,
        },

        "forbidden_boundaries": {
            "T60_scored":
                False,

            "April_scored":
                False,
        },
    }


def save_latest(
    state,
):

    atomic_torch_save(
        checkpoint_payload(
            state
        ),
        LATEST_CHECKPOINT_PATH,
    )

    atomic_json_write(
        progress_payload(
            state
        ),
        PROGRESS_PATH,
    )


def restore_checkpoint(
    checkpoint,
):

    require(
        checkpoint[
            "schema_version"
        ]
        ==
        "PHASE8_M1_DEV_T57_PRODUCTION_V1",
        "Checkpoint schema drift.",
    )

    require(
        checkpoint[
            "family"
        ]
        == "M1"
        and checkpoint[
            "scope"
        ]
        == "DEV_T57",
        "Checkpoint family/scope drift.",
    )

    require(
        checkpoint[
            "runtime_bindings"
        ]
        ==
        runtime_bindings,
        (
            "Resume runtime bindings differ from checkpoint. "
            "Do not resume under changed code/contracts."
        ),
    )

    require(
        checkpoint[
            "forbidden_boundaries"
        ][
            "T60_scored"
        ]
        is False
        and checkpoint[
            "forbidden_boundaries"
        ][
            "April_scored"
        ]
        is False,
        "Checkpoint crossed forbidden T60/April boundary.",
    )

    load_result = model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    require(
        len(
            load_result.missing_keys
        )
        == 0
        and len(
            load_result.unexpected_keys
        )
        == 0,
        "Checkpoint model-state load drift.",
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    random.setstate(
        checkpoint[
            "python_rng_state"
        ]
    )

    np.random.set_state(
        checkpoint[
            "numpy_rng_state"
        ]
    )

    torch.set_rng_state(
        checkpoint[
            "torch_rng_state"
        ]
    )

    torch.cuda.set_rng_state(
        checkpoint[
            "cuda_rng_state"
        ],
        device=0,
    )

    model.train()

    return checkpoint[
        "controller_state"
    ]


# =============================================================================
# PRE-FLIGHT-ONLY BOUNDARY
# =============================================================================

print()
print("=" * 120)
print("PHASE 8 M1 / DEV_T57 PRODUCTION PREREQUISITE GATE")
print("=" * 120)

print(
    "  8.2.8 development authorization:       PASS"
)

print(
    "  Variable final-batch qualification:     PASS"
)

print(
    "  Frozen 20-epoch stream registry:        PASS"
)

print(
    "  Frozen T57 21,924 x 100 candidates:     PASS"
)

print(
    "  T57 fit + sink-only inference masks:    PASS"
)

print(
    "  Canonical M1 initialization:            PASS"
)

print(
    f"  Repository HEAD:                        "
    f"{repository_commit}"
)

print(
    f"  GPU:                                    "
    f"{torch.cuda.get_device_name(0)}"
)

print(
    f"  PyTorch / CUDA:                         "
    f"{torch.__version__} / {torch.version.cuda}"
)


if args.preflight_only:

    smoke_batch = (
        build_validation_candidate_frame(
            0,
            VALIDATION_CASES_PER_CHUNK,
        )
    )

    require(
        len(
            smoke_batch
        )
        == VALIDATION_ROWS_PER_FULL_CHUNK,
        "Validation preflight chunk is not 500 rows.",
    )

    with open(
        os.devnull,
        "w",
    ) as devnull:

        with contextlib.redirect_stdout(
            devnull
        ):

            smoke_ctx = (
                prepare_inference_runtime(
                    smoke_batch
                )
            )

            smoke_ctx = (
                move_runtime_to_cuda(
                    smoke_ctx
                )
            )

    require(
        bool(
            smoke_ctx[
                "feature_visibility"
            ]
            .all()
            .item()
        ),
        "Validation preflight did not restore all Startup features.",
    )

    inference_edges_observed = int(
        smoke_ctx[
            "fit_edges"
        ].shape[
            1
        ]
    )

    expected_inference_edges = int(
        np.asarray(
            np.load(
                GRAPH_INFERENCE_MASK_PATH,
                mmap_mode="r",
            ),
            dtype=bool,
        ).sum()
    )

    # The prepared context contains the two-hop closure, not the full graph.
    require(
        inference_edges_observed
        > 0
        and inference_edges_observed
        <= expected_inference_edges,
        "Validation inference closure edge count invalid.",
    )

    del smoke_ctx
    del smoke_batch

    gc.collect()
    torch.cuda.empty_cache()

    print()
    print(
        "  Validation 5-case / 500-row materialization: PASS"
    )

    print(
        "  Inference COLD_t feature restoration:        PASS"
    )

    print(
        "  Inference graph closure:                     PASS"
    )

    print(
        "  Model forward executed:                      NO"
    )

    print(
        "  Optimizer steps executed:                    0"
    )

    print(
        "  Development metrics inspected:               NO"
    )

    print(
        "  T60 scored:                                  NO"
    )

    print(
        "  April scored:                                NO"
    )

    print()
    print(
        "STATUS: M1_DEV_T57_PRODUCTION_PREFLIGHT_PASS"
    )

    sys.exit(
        0
    )


# =============================================================================
# ACTUAL OPTIMIZER-BEARING RUN
# =============================================================================

require_clean_tracked_tree()

if args.fresh:

    for path in [
        LATEST_CHECKPOINT_PATH,
        PROGRESS_PATH,
        EPOCH_METRICS_PATH,
        FINAL_MANIFEST_PATH,
    ]:

        require(
            not path.exists(),
            (
                "Fresh run refuses existing production artifact: "
                f"{path}"
            ),
        )

    require(
        not VALIDATION_DIR.exists()
        or not any(
            VALIDATION_DIR.glob(
                "*.parquet"
            )
        ),
        (
            "Fresh run refuses existing validation "
            f"artifacts in {VALIDATION_DIR}."
        ),
    )

    RUN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    state = make_fresh_state()

    print()
    print(
        "Run mode:                                  FRESH"
    )

else:

    require(
        LATEST_CHECKPOINT_PATH.exists(),
        (
            "Cannot --resume: latest checkpoint "
            f"missing at {LATEST_CHECKPOINT_PATH}"
        ),
    )

    checkpoint = torch.load(
        LATEST_CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    state = restore_checkpoint(
        checkpoint
    )

    del checkpoint

    print()
    print(
        "Run mode:                                  RESUME"
    )

    print(
        f"Resume display epoch:                      "
        f"{int(state['epoch_index']) + 1}"
    )

    print(
        f"Resume next batch:                         "
        f"{int(state['next_batch_index'])}"
    )

    print(
        f"Global optimizer step:                     "
        f"{int(state['global_optimizer_step']):,}"
    )


# =============================================================================
# 20-EPOCH DEVELOPMENT TRAINING
# =============================================================================

while not state[
    "training_complete"
]:

    epoch_index = int(
        state[
            "epoch_index"
        ]
    )

    require(
        0
        <= epoch_index
        < NUM_EPOCHS,
        "Controller epoch outside 0..19.",
    )

    print()
    print("=" * 120)
    print(
        f"M1 / DEV_T57 — EPOCH "
        f"{epoch_index + 1}/{NUM_EPOCHS}"
    )
    print("=" * 120)

    stream = load_epoch_stream(
        epoch_index
    )

    print(
        f"  negative_sha256: "
        f"{stream['negative_sha256']}"
    )

    print(
        f"  order_sha256:    "
        f"{stream['order_sha256']}"
    )

    if not state[
        "validation_pending"
    ]:

        start_batch = int(
            state[
                "next_batch_index"
            ]
        )

        require(
            0
            <= start_batch
            <= BATCHES_PER_EPOCH,
            "Resume batch outside epoch.",
        )

        for batch_index in range(
            start_batch,
            BATCHES_PER_EPOCH,
        ):

            batch_wall_start = (
                time.perf_counter()
            )

            batch = build_batch(
                batch_index
            )

            with open(
                os.devnull,
                "w",
            ) as devnull:

                with contextlib.redirect_stdout(
                    devnull
                ):

                    ctx = (
                        prepare_batch_runtime(
                            batch
                        )
                    )

                    ctx = (
                        move_runtime_to_cuda(
                            ctx
                        )
                    )

            runtime.update(
                ctx
            )

            torch.cuda.synchronize()

            train_start = (
                time.perf_counter()
            )

            with open(
                os.devnull,
                "w",
            ) as devnull:

                with contextlib.redirect_stdout(
                    devnull
                ):

                    result = (
                        persistent_train_step(
                            FAMILY,
                            model,
                            optimizer,
                        )
                    )

            torch.cuda.synchronize()

            train_seconds = (
                time.perf_counter()
                - train_start
            )

            total_seconds = (
                time.perf_counter()
                - batch_wall_start
            )

            loss_value = float(
                result[
                    "loss_bce_with_logits"
                ]
            )

            require(
                np.isfinite(
                    loss_value
                ),
                (
                    f"Non-finite training BCE at "
                    f"epoch={epoch_index} "
                    f"batch={batch_index}."
                ),
            )

            batch_size = len(
                batch
            )

            state[
                "epoch_loss_weighted_sum"
            ] += (
                loss_value
                * batch_size
            )

            state[
                "epoch_example_count"
            ] += (
                batch_size
            )

            state[
                "epoch_training_seconds"
            ] += (
                float(
                    total_seconds
                )
            )

            state[
                "global_optimizer_step"
            ] += 1

            state[
                "next_batch_index"
            ] = (
                batch_index
                + 1
            )

            if (
                batch_index == 0
                or (
                    batch_index + 1
                )
                % PROGRESS_PRINT_INTERVAL_BATCHES
                == 0
                or (
                    batch_index + 1
                )
                == BATCHES_PER_EPOCH
            ):

                mean_loss = (
                    state[
                        "epoch_loss_weighted_sum"
                    ]
                    /
                    state[
                        "epoch_example_count"
                    ]
                )

                print(
                    f"  epoch={epoch_index + 1:02d} "
                    f"batch={batch_index + 1:05d}/"
                    f"{BATCHES_PER_EPOCH} "
                    f"step="
                    f"{state['global_optimizer_step']:06d}/"
                    f"{TOTAL_OPTIMIZER_STEPS} "
                    f"rows={batch_size:03d} "
                    f"loss={loss_value:.8f} "
                    f"mean={mean_loss:.8f} "
                    f"train={train_seconds:.3f}s "
                    f"total={total_seconds:.3f}s"
                )

            if (
                (
                    batch_index + 1
                )
                % CHECKPOINT_INTERVAL_BATCHES
                == 0
                and (
                    batch_index + 1
                )
                < BATCHES_PER_EPOCH
            ):

                save_latest(
                    state
                )

                print(
                    "    checkpoint: latest.pt "
                    f"after batch {batch_index + 1}"
                )

            del ctx
            del batch

        require(
            state[
                "next_batch_index"
            ]
            == BATCHES_PER_EPOCH,
            "Epoch ended before all 9,801 batches.",
        )

        require(
            state[
                "epoch_example_count"
            ]
            == EXAMPLES_PER_EPOCH,
            (
                "Epoch example count drift: "
                f"{state['epoch_example_count']}"
            ),
        )

        state[
            "validation_pending"
        ] = True

        save_latest(
            state
        )

        print(
            "  checkpoint: end-of-epoch / "
            "pre-validation restart boundary"
        )

    require(
        state[
            "validation_pending"
        ]
        is True,
        "Validation requires validation_pending=True.",
    )

    epoch_training_loss = (
        state[
            "epoch_loss_weighted_sum"
        ]
        /
        state[
            "epoch_example_count"
        ]
    )

    validation = full_t57_validation(
        epoch_index=epoch_index,
        model=model,
        optimizer=optimizer,
    )

    epoch_parameter_sha = (
        runtime[
            "full_parameter_hash"
        ](
            model
        )
    )

    history_row = {
        "epoch_index":
            int(
                epoch_index
            ),

        "display_epoch":
            int(
                epoch_index
                + 1
            ),

        "training_loss":
            float(
                epoch_training_loss
            ),

        "training_seconds":
            float(
                state[
                    "epoch_training_seconds"
                ]
            ),

        "primary_HR@10":
            validation[
                "primary_HR@10"
            ],

        "primary_NDCG@10":
            validation[
                "primary_NDCG@10"
            ],

        "aggregate_HR@10":
            validation[
                "aggregate_HR@10"
            ],

        "aggregate_NDCG@10":
            validation[
                "aggregate_NDCG@10"
            ],

        "validation_seconds":
            validation[
                "validation_seconds"
            ],

        "negative_sha256":
            stream[
                "negative_sha256"
            ],

        "order_sha256":
            stream[
                "order_sha256"
            ],

        "model_parameter_sha256":
            epoch_parameter_sha,

        "global_optimizer_step":
            int(
                state[
                    "global_optimizer_step"
                ]
            ),

        "T60_cases_scored":
            0,

        "April_cases_scored":
            0,
    }

    state[
        "validation_history"
    ] = (
        list(
            state[
                "validation_history"
            ]
        )
        + [
            history_row
        ]
    )

    state[
        "validation_pending"
    ] = False

    atomic_csv_write(
        pd.DataFrame(
            state[
                "validation_history"
            ]
        ),
        EPOCH_METRICS_PATH,
    )

    print()
    print("=" * 120)
    print(
        f"M1 / DEV_T57 — EPOCH "
        f"{epoch_index + 1} COMMITTED"
    )
    print("=" * 120)

    print(
        f"  Training BCE:               "
        f"{epoch_training_loss:.10f}"
    )

    print(
        f"  PRIMARY NDCG@10:            "
        f"{validation['primary_NDCG@10']:.12f}"
    )

    print(
        f"  PRIMARY HR@10:              "
        f"{validation['primary_HR@10']:.12f}"
    )

    print(
        f"  Completed validations:      "
        f"{len(state['validation_history'])}/20"
    )

    print(
        "  Fold-local best selection:  NOT PERFORMED"
    )

    print(
        "  Reason: final epoch selection requires "
        "equal-fold T57/T58/T59 means."
    )

    print(
        "  T60/April scored:           0 / 0"
    )

    if epoch_index < (
        NUM_EPOCHS
        - 1
    ):

        state[
            "epoch_index"
        ] = (
            epoch_index
            + 1
        )

        state[
            "next_batch_index"
        ] = 0

        state[
            "epoch_loss_weighted_sum"
        ] = 0.0

        state[
            "epoch_example_count"
        ] = 0

        state[
            "epoch_training_seconds"
        ] = 0.0

        state[
            "validation_pending"
        ] = False

        state[
            "training_complete"
        ] = False

    else:

        state[
            "epoch_index"
        ] = epoch_index

        state[
            "next_batch_index"
        ] = BATCHES_PER_EPOCH

        state[
            "training_complete"
        ] = True

    save_latest(
        state
    )

    del stream
    gc.collect()
    torch.cuda.empty_cache()

    if (
        args.stop_after_epoch
        is not None
        and len(
            state[
                "validation_history"
            ]
        )
        >= args.stop_after_epoch
        and not state[
            "training_complete"
        ]
    ):

        print()
        print("=" * 120)
        print("SAFE USER-REQUESTED STOP")
        print("=" * 120)

        print(
            f"Completed + validated epochs: "
            f"{len(state['validation_history'])}"
        )

        print(
            f"Next display epoch:           "
            f"{int(state['epoch_index']) + 1}"
        )

        print(
            "Next batch:                   0"
        )

        print(
            "Resume with:                  --resume"
        )

        print(
            "T60/April scored:             0 / 0"
        )

        sys.exit(
            0
        )


# =============================================================================
# DEV_T57 COMPLETION MANIFEST — NO FOLD-LOCAL EPOCH SELECTION
# =============================================================================

final_manifest = {
    "schema_version":
        "PHASE8_M1_DEV_T57_TRAINING_COMPLETE_V1",

    "status":
        "M1_DEV_T57_20_EPOCHS_TRAINED_AND_VALIDATED",

    "family":
        "M1",

    "scope":
        "DEV_T57",

    "epochs_completed":
        len(
            state[
                "validation_history"
            ]
        ),

    "optimizer_steps_completed":
        int(
            state[
                "global_optimizer_step"
            ]
        ),

    "runtime_bindings":
        runtime_bindings,

    "epoch_metrics":
        str(
            EPOCH_METRICS_PATH
        ),

    "latest_checkpoint":
        str(
            LATEST_CHECKPOINT_PATH
        ),

    "selection": {
        "performed":
            False,

        "reason":
            (
                "Frozen Phase 8.2.8 selection requires "
                "equal-fold mean across DEV_T57, DEV_T58, DEV_T59 "
                "for each candidate epoch."
            ),
    },

    "T60_scored":
        False,

    "April_scored":
        False,
}

atomic_json_write(
    final_manifest,
    FINAL_MANIFEST_PATH,
)

print()
print("=" * 120)
print("M1 / DEV_T57 TRAINING COMPLETE")
print("=" * 120)

print(
    "Training epochs:                         20 / 20"
)

print(
    f"Optimizer steps:                         "
    f"{state['global_optimizer_step']:,} / "
    f"{TOTAL_OPTIMIZER_STEPS:,}"
)

print(
    "Frozen T57 validations:                  20 / 20"
)

print(
    "Fold-local epoch selection:              NOT PERFORMED"
)

print(
    "T60 scored:                              NO"
)

print(
    "April scored:                            NO"
)

print(
    f"WROTE  {EPOCH_METRICS_PATH}"
)

print(
    f"WROTE  {LATEST_CHECKPOINT_PATH}"
)

print(
    f"WROTE  {FINAL_MANIFEST_PATH}"
)

print(
    "NEXT: complete M1 DEV_T58 and DEV_T59, "
    "then perform frozen equal-fold epoch selection."
)
