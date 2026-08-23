#!/usr/bin/env python3
"""
Phase 5.3.1l.2a
Training-Compatible Trend Sequence Adapter Proof and Freeze

Problem
-------
The exact Phase-4.6.2 numerical method:

    TrendExtractor.encode_sequence(sequence)

contains an integration-audit-only assertion:

    require(
        sequence.shape[1] == NUM_HISTORY_PERIODS,
        "T60 audit sequence must contain T0 through T59.",
    )

That assertion is correct for the Phase-4.6.2 T60 diagnostic case,
but cannot be used unchanged during training because a target T_h must
consume exactly:

    T0 .. T(h-1)

for h in 1..59.

Phase-5.3.1l.1 explicitly froze:
    - exact history length h;
    - no post-h GRU padding.

Purpose
-------
Derive a training-compatible method from the EXACT frozen
Phase-4.6.2 encode_sequence AST by removing ONLY the T60-specific
audit assertion.

No numerical model semantics are changed.

The adapter must:
    1. preserve every other statement in encode_sequence;
    2. be numerically identical to the original method for h=60;
    3. execute successfully for h=1,17,59;
    4. preserve the canonical parameter-state SHA256 exactly;
    5. create no optimizer or gradients.

Classification
--------------
IMPLEMENTATION_EQUIVALENT_CHOICE

This is not a paper/model change. It removes a diagnostic assertion
whose scope was explicitly T60-only.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import sys
import types
from pathlib import Path

import pandas as pd
import torch


# =============================================================================
# Frozen sources
# =============================================================================

CANONICAL_SOURCE_PATH = Path(
    "scripts/"
    "phase_4_7_1b_freeze_neural_initialization_seed_contract.py"
)

CANONICAL_SOURCE_SHA256 = (
    "c55f3ea1646cec7fdc8ef69f2310d98f"
    "5ee95fab77f0c48392f4a9f76612761c"
)

FORWARD_SOURCE_PATH = Path(
    "scripts/"
    "phase_4_6_2_end_to_end_itrs_forward_bce_audit.py"
)

FORWARD_SOURCE_SHA256 = (
    "18c6c7ca4915fb23eab5ed39bae6eb49"
    "1a9332196f51b302a352c3c8211b053d"
)

EXPECTED_ORIGINAL_ENCODE_SEQUENCE_AST_SHA256 = (
    "72fa12a49ad4a21399d4810a31323f670"
    "34e6173f73ee71dc18798bad1b6d97f"
)

EXPECTED_INITIAL_STATE_SHA256 = (
    "49e822ea7fad35c458f47e134c94c05e"
    "ac099b68c5c468e2c71559c8c88998ab"
)

EXPECTED_RUNTIME_AST_SHA256 = (
    "301a074aa57cfe7602f2ccbb5b8e26943"
    "b94b72e36efe4d60d1af48378c58a6e"
)

EXPECTED_WORKFLOW_BOUNDARY_INDEX = 56
EXPECTED_WORKFLOW_BOUNDARY_LINE = 1272

EXPECTED_SEED = 42

REFERENCE_TORCH_VERSION_PREFIX = "2.7.0"

TREND_ITEM_DIM = 80
TREND_DIM = 40


# =============================================================================
# Exact audit guard being removed
# =============================================================================

T60_GUARD_MESSAGE = (
    "T60 audit sequence must contain T0 through T59."
)


# =============================================================================
# Prior frozen Phase-5 contract
# =============================================================================

PHASE_5_3_1L_1_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_1l_1_epoch0_training_stream_serialization_contract.json"
)


# =============================================================================
# Outputs
# =============================================================================

AUDIT_DIR = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_1l_2a"
)

CONTRACT_DIR = Path(
    "data/experimental/phase_5/contracts"
)

AST_DIFF_PATH = (
    AUDIT_DIR
    / "training_encode_sequence_ast_diff.csv"
)

T60_EQUIVALENCE_PATH = (
    AUDIT_DIR
    / "training_encode_sequence_t60_equivalence.csv"
)

VARIABLE_H_PATH = (
    AUDIT_DIR
    / "training_encode_sequence_variable_h_audit.csv"
)

STATE_PATH = (
    AUDIT_DIR
    / "training_encode_sequence_state_neutrality.csv"
)

FINAL_INVARIANT_PATH = (
    AUDIT_DIR
    / "phase_5_3_1l_2a_final_invariants.csv"
)

MANIFEST_PATH = (
    AUDIT_DIR
    / "phase_5_3_1l_2a_training_trend_adapter_manifest.json"
)

CONTRACT_PATH = (
    CONTRACT_DIR
    / "phase_5_3_1l_2a_training_trend_sequence_adapter_contract.json"
)

DECISION_REGISTER_PATH = (
    CONTRACT_DIR
    / "phase_5_3_1l_2a_training_trend_adapter_decision_register.csv"
)


# =============================================================================
# Generic helpers
# =============================================================================

def banner(text: str) -> None:

    print(
        "\n"
        + "=" * 118
    )

    print(text)

    print(
        "=" * 118
    )


def require(
    condition: bool,
    message: str,
) -> None:

    if not bool(condition):

        raise AssertionError(message)


def file_sha256(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as handle:

        while True:

            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def text_sha256(
    value: str,
) -> str:

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def ast_sha256(
    node: ast.AST,
) -> str:

    return text_sha256(
        ast.dump(
            node,
            annotate_fields=True,
            include_attributes=False,
        )
    )


def load_json(
    path: Path,
) -> dict:

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        return json.load(handle)


def execute_nodes(
    module: types.ModuleType,
    nodes: list[ast.AST],
    filename: str,
) -> None:

    tree = ast.Module(
        body=[
            copy.deepcopy(node)
            for node in nodes
        ],
        type_ignores=[],
    )

    ast.fix_missing_locations(tree)

    exec(
        compile(
            tree,
            filename=filename,
            mode="exec",
        ),
        module.__dict__,
    )


# =============================================================================
# AST lookup
# =============================================================================

def class_map(
    tree: ast.Module,
) -> dict[str, ast.ClassDef]:

    return {
        node.name: node
        for node in tree.body
        if isinstance(
            node,
            ast.ClassDef,
        )
    }


def direct_method(
    class_node: ast.ClassDef,
    method_name: str,
) -> ast.FunctionDef:

    matches = [
        node
        for node in class_node.body
        if (
            isinstance(
                node,
                ast.FunctionDef,
            )
            and node.name
            == method_name
        )
    ]

    require(
        len(matches) == 1,
        (
            f"Expected exactly one "
            f"{class_node.name}.{method_name}; "
            f"found {len(matches)}."
        ),
    )

    return matches[0]


def top_level_function(
    tree: ast.Module,
    name: str,
) -> ast.FunctionDef:

    matches = [
        node
        for node in tree.body
        if (
            isinstance(
                node,
                ast.FunctionDef,
            )
            and node.name == name
        )
    ]

    require(
        len(matches) == 1,
        (
            f"Expected exactly one "
            f"top-level {name}()."
        ),
    )

    return matches[0]


def assigned_name(
    node: ast.AST,
) -> str | None:

    if isinstance(
        node,
        ast.Assign,
    ):

        if (
            len(node.targets) == 1
            and isinstance(
                node.targets[0],
                ast.Name,
            )
        ):

            return node.targets[0].id

    if isinstance(
        node,
        ast.AnnAssign,
    ):

        if isinstance(
            node.target,
            ast.Name,
        ):

            return node.target.id

    return None


def top_level_assignment(
    tree: ast.Module,
    name: str,
):

    matches = [
        node
        for node in tree.body
        if (
            isinstance(
                node,
                (
                    ast.Assign,
                    ast.AnnAssign,
                ),
            )
            and assigned_name(node) == name
        )
    ]

    require(
        len(matches) == 1,
        (
            f"Expected exactly one "
            f"assignment to {name}."
        ),
    )

    return matches[0]


# =============================================================================
# Canonical sanitized runtime
# =============================================================================

CANONICAL_RETAINED_TYPES = (
    ast.Import,
    ast.ImportFrom,
    ast.Assign,
    ast.AnnAssign,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
)


def build_canonical_runtime(
    tree: ast.Module,
):

    require(
        int(
            tree.body[
                EXPECTED_WORKFLOW_BOUNDARY_INDEX
            ].lineno
        )
        == EXPECTED_WORKFLOW_BOUNDARY_LINE,
        (
            "Canonical workflow boundary changed."
        ),
    )

    retained = [
        copy.deepcopy(node)
        for node
        in tree.body[
            :EXPECTED_WORKFLOW_BOUNDARY_INDEX
        ]
        if isinstance(
            node,
            CANONICAL_RETAINED_TYPES,
        )
    ]

    sanitized = ast.Module(
        body=retained,
        type_ignores=copy.deepcopy(
            tree.type_ignores
        ),
    )

    ast.fix_missing_locations(
        sanitized
    )

    runtime_hash = text_sha256(
        ast.dump(
            sanitized,
            annotate_fields=True,
            include_attributes=False,
        )
    )

    require(
        runtime_hash
        == EXPECTED_RUNTIME_AST_SHA256,
        (
            "Canonical sanitized runtime "
            "AST changed."
        ),
    )

    module_name = (
        "_itrs_phase5_3_1l_2a_canonical"
    )

    module = types.ModuleType(
        module_name
    )

    module.__file__ = str(
        CANONICAL_SOURCE_PATH.resolve()
    )

    module.__package__ = None

    sys.modules[
        module_name
    ] = module

    exec(
        compile(
            sanitized,
            filename=str(
                CANONICAL_SOURCE_PATH
            ),
            mode="exec",
        ),
        module.__dict__,
    )

    return (
        module,
        runtime_hash,
    )


# =============================================================================
# Forward support namespace
# =============================================================================

def build_forward_support_runtime(
    tree: ast.Module,
):

    module_name = (
        "_itrs_phase5_3_1l_2a_forward_support"
    )

    module = types.ModuleType(
        module_name
    )

    module.__file__ = str(
        FORWARD_SOURCE_PATH.resolve()
    )

    module.__package__ = None

    sys.modules[
        module_name
    ] = module

    # Exact Phase-4.6.2 imports.
    imports = [
        node
        for node in tree.body
        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
            ),
        )
    ]

    execute_nodes(
        module,
        imports,
        str(
            FORWARD_SOURCE_PATH
        ),
    )

    # Constants referenced by encode_sequence.
    support_constants = {
        "NUM_HISTORY_PERIODS": 60,
        "TREND_ITEM_DIM": 80,
        "TREND_DIM": 40,
    }

    for (
        name,
        expected_value,
    ) in support_constants.items():

        assignment = top_level_assignment(
            tree,
            name,
        )

        value_node = (
            assignment.value
        )

        actual_value = ast.literal_eval(
            value_node
        )

        require(
            actual_value
            == expected_value,
            (
                f"{name} changed."
            ),
        )

        execute_nodes(
            module,
            [assignment],
            str(
                FORWARD_SOURCE_PATH
            ),
        )

    # Exact require helper.
    execute_nodes(
        module,
        [
            top_level_function(
                tree,
                "require",
            )
        ],
        str(
            FORWARD_SOURCE_PATH
        ),
    )

    return module


# =============================================================================
# Identify exact T60-only assertion
# =============================================================================

def statement_contains_exact_guard(
    statement: ast.stmt,
) -> bool:

    message_found = False
    require_call_found = False

    for node in ast.walk(
        statement
    ):

        if (
            isinstance(
                node,
                ast.Constant,
            )
            and node.value
            == T60_GUARD_MESSAGE
        ):

            message_found = True

        if isinstance(
            node,
            ast.Call,
        ):

            if (
                isinstance(
                    node.func,
                    ast.Name,
                )
                and node.func.id
                == "require"
            ):

                require_call_found = True

    return (
        message_found
        and require_call_found
    )


def derive_training_adapter(
    original_method: ast.FunctionDef,
):
    """
    Clone the exact Phase-4.6.2 method and remove exactly one
    T60-audit-only require statement.
    """

    original = copy.deepcopy(
        original_method
    )

    original_body_count = len(
        original.body
    )

    removed = [
        statement
        for statement in original.body
        if statement_contains_exact_guard(
            statement
        )
    ]

    require(
        len(removed) == 1,
        (
            "Expected exactly one T60-specific "
            "encode_sequence guard; "
            f"found {len(removed)}."
        ),
    )

    retained = [
        statement
        for statement in original.body
        if not statement_contains_exact_guard(
            statement
        )
    ]

    require(
        len(retained)
        == original_body_count - 1,
        (
            "Adapter did not remove exactly "
            "one statement."
        ),
    )

    adapter = copy.deepcopy(
        original
    )

    adapter.name = (
        "encode_training_sequence"
    )

    adapter.body = retained

    ast.fix_missing_locations(
        adapter
    )

    return (
        adapter,
        removed[0],
    )


# =============================================================================
# Compare numerical result structures
# =============================================================================

def compare_results_exact(
    left,
    right,
) -> tuple[bool, list[dict]]:

    rows = []

    if isinstance(
        left,
        torch.Tensor,
    ):

        require(
            isinstance(
                right,
                torch.Tensor,
            ),
            (
                "Result type mismatch."
            ),
        )

        exact = torch.equal(
            left,
            right,
        )

        rows.append(
            {
                "component": (
                    "tensor"
                ),

                "left_shape": (
                    str(
                        tuple(
                            left.shape
                        )
                    )
                ),

                "right_shape": (
                    str(
                        tuple(
                            right.shape
                        )
                    )
                ),

                "exact_equal": (
                    exact
                ),

                "max_abs_difference": (
                    float(
                        (
                            left
                            - right
                        )
                        .detach()
                        .abs()
                        .max()
                    )
                    if left.numel()
                    > 0
                    else 0.0
                ),
            }
        )

        return (
            exact,
            rows,
        )

    if isinstance(
        left,
        tuple,
    ):

        require(
            isinstance(
                right,
                tuple,
            ),
            (
                "Tuple result type mismatch."
            ),
        )

        require(
            len(left)
            == len(right),
            (
                "Tuple result length mismatch."
            ),
        )

        all_equal = True

        combined = []

        for index, (
            left_item,
            right_item,
        ) in enumerate(
            zip(
                left,
                right,
            )
        ):

            item_equal, item_rows = (
                compare_results_exact(
                    left_item,
                    right_item,
                )
            )

            all_equal = (
                all_equal
                and item_equal
            )

            for row in item_rows:

                row[
                    "component"
                ] = (
                    f"tuple[{index}]/"
                    + row[
                        "component"
                    ]
                )

                combined.append(row)

        return (
            all_equal,
            combined,
        )

    raise AssertionError(
        "Unsupported encode_sequence "
        f"return type: {type(left).__name__}"
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.3.1l.2a — "
        "TRAINING-COMPATIBLE TREND SEQUENCE "
        "ADAPTER PROOF AND FREEZE"
    )

    print(
        "Phase-4 source modified:              NO"
    )

    print(
        "Canonical model instantiated:         YES"
    )

    print(
        "Forward probe executed:               YES"
    )

    print(
        "Training data consumed:               NO"
    )

    print(
        "Adam instantiated:                    NO"
    )

    print(
        "Backward computation:                 NO"
    )

    print(
        "Optimizer steps:                      0"
    )

    # =========================================================================
    # Input recheck
    # =========================================================================

    banner(
        "AUTHORITATIVE INPUT RECHECK"
    )

    for path in (
        CANONICAL_SOURCE_PATH,
        FORWARD_SOURCE_PATH,
        PHASE_5_3_1L_1_CONTRACT_PATH,
    ):

        require(
            path.exists(),
            (
                "Missing input: "
                f"{path}"
            ),
        )

        print(
            f"FOUND  {path}"
        )

    require(
        file_sha256(
            CANONICAL_SOURCE_PATH
        )
        == CANONICAL_SOURCE_SHA256,
        (
            "Canonical source drift."
        ),
    )

    require(
        file_sha256(
            FORWARD_SOURCE_PATH
        )
        == FORWARD_SOURCE_SHA256,
        (
            "Phase-4.6.2 source drift."
        ),
    )

    l1_contract = load_json(
        PHASE_5_3_1L_1_CONTRACT_PATH
    )

    require(
        l1_contract[
            "status"
        ]
        == "FROZEN",
        (
            "Phase-5.3.1l.1 training "
            "stream is not frozen."
        ),
    )

    require(
        l1_contract[
            "mixed_segment_forward"
        ][
            "post_h_GRU_padding"
        ]
        is False,
        (
            "Phase-5.3.1l.1 no-padding "
            "rule changed."
        ),
    )

    print(
        "Canonical source SHA256:              PASS"
    )

    print(
        "Forward source SHA256:                PASS"
    )

    print(
        "Phase-5.3.1l.1 stream contract:       FROZEN"
    )

    print(
        "Post-h GRU padding:                   FORBIDDEN"
    )

    # =========================================================================
    # Runtime
    # =========================================================================

    banner(
        "REFERENCE RUNTIME"
    )

    print(
        f"PyTorch:                              "
        f"{torch.__version__}"
    )

    require(
        torch.__version__.startswith(
            REFERENCE_TORCH_VERSION_PREFIX
        ),
        (
            "Reference runtime requires "
            "PyTorch 2.7.0."
        ),
    )

    # =========================================================================
    # Parse exact method
    # =========================================================================

    canonical_source = (
        CANONICAL_SOURCE_PATH.read_text(
            encoding="utf-8"
        )
    )

    forward_source = (
        FORWARD_SOURCE_PATH.read_text(
            encoding="utf-8"
        )
    )

    canonical_tree = ast.parse(
        canonical_source,
        filename=str(
            CANONICAL_SOURCE_PATH
        ),
    )

    forward_tree = ast.parse(
        forward_source,
        filename=str(
            FORWARD_SOURCE_PATH
        ),
    )

    forward_classes = class_map(
        forward_tree
    )

    require(
        "TrendExtractor"
        in forward_classes,
        (
            "Phase-4.6.2 TrendExtractor missing."
        ),
    )

    original_method = direct_method(
        forward_classes[
            "TrendExtractor"
        ],
        "encode_sequence",
    )

    original_method_sha = ast_sha256(
        original_method
    )

    require(
        original_method_sha
        == EXPECTED_ORIGINAL_ENCODE_SEQUENCE_AST_SHA256,
        (
            "Frozen encode_sequence AST drift."
        ),
    )

    (
        adapter_method,
        removed_guard,
    ) = derive_training_adapter(
        original_method
    )

    adapter_method_sha = ast_sha256(
        adapter_method
    )

    # Re-derive independently.
    (
        adapter_repeat,
        removed_repeat,
    ) = derive_training_adapter(
        original_method
    )

    require(
        ast_sha256(
            adapter_repeat
        )
        == adapter_method_sha,
        (
            "Training adapter AST did not "
            "reproduce deterministically."
        ),
    )

    require(
        ast.dump(
            removed_repeat,
            include_attributes=False,
        )
        == ast.dump(
            removed_guard,
            include_attributes=False,
        ),
        (
            "Removed guard did not "
            "reproduce deterministically."
        ),
    )

    # =========================================================================
    # AST-diff proof
    # =========================================================================

    banner(
        "EXACT AST DIFFERENCE"
    )

    original_statement_hashes = [
        ast_sha256(
            statement
        )
        for statement in (
            original_method.body
        )
    ]

    retained_original_hashes = [
        ast_sha256(
            statement
        )
        for statement in (
            original_method.body
        )
        if not statement_contains_exact_guard(
            statement
        )
    ]

    adapter_statement_hashes = [
        ast_sha256(
            statement
        )
        for statement in (
            adapter_method.body
        )
    ]

    require(
        retained_original_hashes
        == adapter_statement_hashes,
        (
            "Adapter changed a statement "
            "other than the T60 guard."
        ),
    )

    removed_guard_sha = ast_sha256(
        removed_guard
    )

    diff_df = pd.DataFrame(
        [
            {
                "item": (
                    "original_method"
                ),

                "value": (
                    original_method_sha
                ),
            },

            {
                "item": (
                    "adapter_method"
                ),

                "value": (
                    adapter_method_sha
                ),
            },

            {
                "item": (
                    "original_statement_count"
                ),

                "value": (
                    len(
                        original_method.body
                    )
                ),
            },

            {
                "item": (
                    "adapter_statement_count"
                ),

                "value": (
                    len(
                        adapter_method.body
                    )
                ),
            },

            {
                "item": (
                    "removed_statement_count"
                ),

                "value": (
                    1
                ),
            },

            {
                "item": (
                    "removed_statement_sha256"
                ),

                "value": (
                    removed_guard_sha
                ),
            },

            {
                "item": (
                    "removed_statement"
                ),

                "value": (
                    ast.unparse(
                        removed_guard
                    )
                ),
            },

            {
                "item": (
                    "all_other_statements_exact"
                ),

                "value": (
                    True
                ),
            },
        ]
    )

    print(
        diff_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Build canonical model
    # =========================================================================

    banner(
        "CANONICAL MODEL RECONSTRUCTION"
    )

    (
        canonical_runtime,
        runtime_hash,
    ) = build_canonical_runtime(
        canonical_tree
    )

    builder = getattr(
        canonical_runtime,
        "build_canonical_model",
    )

    parameter_hash_fn = getattr(
        canonical_runtime,
        "model_parameter_state_sha256",
    )

    model = builder(
        seed=EXPECTED_SEED
    )

    hash_before = parameter_hash_fn(
        model
    )

    require(
        hash_before
        == EXPECTED_INITIAL_STATE_SHA256,
        (
            "Canonical initial-state "
            "hash mismatch."
        ),
    )

    # =========================================================================
    # Compile original + adapter in exact forward support namespace
    # =========================================================================

    forward_runtime = (
        build_forward_support_runtime(
            forward_tree
        )
    )

    # Compile original method as a top-level callable.
    original_top = copy.deepcopy(
        original_method
    )

    original_top.name = (
        "encode_sequence_original"
    )

    ast.fix_missing_locations(
        original_top
    )

    execute_nodes(
        forward_runtime,
        [
            original_top,
            adapter_method,
        ],
        str(
            FORWARD_SOURCE_PATH
        ),
    )

    original_callable = (
        forward_runtime.__dict__[
            "encode_sequence_original"
        ]
    )

    adapter_callable = (
        forward_runtime.__dict__[
            "encode_training_sequence"
        ]
    )

    # Bind both as methods of the canonical TrendExtractor class.
    canonical_trend_class = getattr(
        canonical_runtime,
        "TrendExtractor",
    )

    setattr(
        canonical_trend_class,
        "encode_sequence_original_audit",
        original_callable,
    )

    setattr(
        canonical_trend_class,
        "encode_training_sequence",
        adapter_callable,
    )

    require(
        model
        .trend_extractor
        .encode_sequence_original_audit
        .__func__
        is original_callable,
        (
            "Original audit method did not bind."
        ),
    )

    require(
        model
        .trend_extractor
        .encode_training_sequence
        .__func__
        is adapter_callable,
        (
            "Training adapter did not bind."
        ),
    )

    require(
        parameter_hash_fn(
            model
        )
        == EXPECTED_INITIAL_STATE_SHA256,
        (
            "Method binding changed parameters."
        ),
    )

    # =========================================================================
    # T60 numerical equivalence proof
    # =========================================================================

    banner(
        "T60 ORIGINAL-vs-ADAPTER NUMERICAL EQUIVALENCE"
    )

    # Deterministic, RNG-free numerical probe.
    probe60 = torch.linspace(
        -0.25,
        0.25,
        steps=(
            60
            * TREND_ITEM_DIM
        ),
        dtype=torch.float32,
    ).reshape(
        1,
        60,
        TREND_ITEM_DIM,
    )

    with torch.no_grad():

        original_result = (
            model
            .trend_extractor
            .encode_sequence_original_audit(
                probe60
            )
        )

        adapter_result = (
            model
            .trend_extractor
            .encode_training_sequence(
                probe60
            )
        )

    (
        exact_equal,
        equivalence_rows,
    ) = compare_results_exact(
        original_result,
        adapter_result,
    )

    require(
        exact_equal,
        (
            "Training adapter is not "
            "numerically identical to the "
            "original method for h=60."
        ),
    )

    equivalence_df = pd.DataFrame(
        equivalence_rows
    )

    print(
        equivalence_df.to_string(
            index=False
        )
    )

    print()

    print(
        "T60 numerical equality:               EXACT PASS"
    )

    # =========================================================================
    # Variable-h proof
    # =========================================================================

    banner(
        "VARIABLE-h TRAINING EXECUTION PROOF"
    )

    variable_rows = []

    for h in (
        1,
        17,
        59,
    ):

        probe = torch.linspace(
            -0.10,
            0.10,
            steps=(
                h
                * TREND_ITEM_DIM
            ),
            dtype=torch.float32,
        ).reshape(
            1,
            h,
            TREND_ITEM_DIM,
        )

        with torch.no_grad():

            result = (
                model
                .trend_extractor
                .encode_training_sequence(
                    probe
                )
            )

        require(
            isinstance(
                result,
                tuple,
            ),
            (
                f"T{h} adapter result "
                "is not tuple."
            ),
        )

        require(
            len(result) == 2,
            (
                f"T{h} adapter result "
                "does not contain two values."
            ),
        )

        F_t, gru_output = result

        require(
            F_t.shape
            == (
                1,
                TREND_DIM,
            ),
            (
                f"T{h} F_t shape invalid: "
                f"{tuple(F_t.shape)}"
            ),
        )

        require(
            gru_output.shape
            == (
                1,
                h,
                TREND_DIM,
            ),
            (
                f"T{h} GRU output shape invalid: "
                f"{tuple(gru_output.shape)}"
            ),
        )

        require(
            bool(
                torch.isfinite(
                    F_t
                ).all()
            ),
            (
                f"T{h} F_t non-finite."
            ),
        )

        require(
            bool(
                torch.isfinite(
                    gru_output
                ).all()
            ),
            (
                f"T{h} GRU output non-finite."
            ),
        )

        variable_rows.append(
            {
                "target_segment": (
                    f"T{h}"
                ),

                "history_periods": (
                    h
                ),

                "input_shape": (
                    str(
                        tuple(
                            probe.shape
                        )
                    )
                ),

                "F_t_shape": (
                    str(
                        tuple(
                            F_t.shape
                        )
                    )
                ),

                "gru_output_shape": (
                    str(
                        tuple(
                            gru_output.shape
                        )
                    )
                ),

                "finite": (
                    True
                ),

                "status": (
                    "PASS"
                ),
            }
        )

    variable_df = pd.DataFrame(
        variable_rows
    )

    print(
        variable_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Parameter-state neutrality
    # =========================================================================

    banner(
        "PARAMETER-STATE NEUTRALITY"
    )

    hash_after = parameter_hash_fn(
        model
    )

    require(
        hash_after
        == EXPECTED_INITIAL_STATE_SHA256,
        (
            "Adapter probe changed canonical "
            "parameter values."
        ),
    )

    require(
        all(
            parameter.grad
            is None
            for parameter
            in model.parameters()
        ),
        (
            "Adapter proof unexpectedly "
            "created gradients."
        ),
    )

    state_df = pd.DataFrame(
        [
            {
                "check": (
                    "canonical_hash_before"
                ),

                "actual": (
                    hash_before
                ),

                "expected": (
                    EXPECTED_INITIAL_STATE_SHA256
                ),

                "status": (
                    "PASS"
                ),
            },

            {
                "check": (
                    "canonical_hash_after"
                ),

                "actual": (
                    hash_after
                ),

                "expected": (
                    EXPECTED_INITIAL_STATE_SHA256
                ),

                "status": (
                    "PASS"
                ),
            },

            {
                "check": (
                    "parameter_gradients_created"
                ),

                "actual": (
                    "False"
                ),

                "expected": (
                    "False"
                ),

                "status": (
                    "PASS"
                ),
            },
        ]
    )

    print(
        state_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Final invariants
    # =========================================================================

    banner(
        "FINAL PHASE-5.3.1l.2a INVARIANTS"
    )

    checks = [
        (
            "canonical_source_sha256_exact",
            (
                file_sha256(
                    CANONICAL_SOURCE_PATH
                )
                == CANONICAL_SOURCE_SHA256
            ),
        ),

        (
            "forward_source_sha256_exact",
            (
                file_sha256(
                    FORWARD_SOURCE_PATH
                )
                == FORWARD_SOURCE_SHA256
            ),
        ),

        (
            "phase_5_3_1l_1_contract_frozen",
            (
                l1_contract[
                    "status"
                ]
                == "FROZEN"
            ),
        ),

        (
            "post_h_padding_still_forbidden",
            (
                l1_contract[
                    "mixed_segment_forward"
                ][
                    "post_h_GRU_padding"
                ]
                is False
            ),
        ),

        (
            "original_encode_sequence_AST_exact",
            (
                original_method_sha
                == (
                    EXPECTED_ORIGINAL_ENCODE_SEQUENCE_AST_SHA256
                )
            ),
        ),

        (
            "exactly_one_statement_removed",
            (
                len(
                    original_method.body
                )
                - len(
                    adapter_method.body
                )
                == 1
            ),
        ),

        (
            "removed_statement_is_exact_T60_guard",
            statement_contains_exact_guard(
                removed_guard
            ),
        ),

        (
            "all_other_method_statements_exact",
            (
                retained_original_hashes
                == adapter_statement_hashes
            ),
        ),

        (
            "adapter_AST_reproducible",
            (
                ast_sha256(
                    adapter_repeat
                )
                == adapter_method_sha
            ),
        ),

        (
            "T60_original_adapter_exact_numerical_equality",
            (
                exact_equal
            ),
        ),

        (
            "T1_adapter_executes",
            bool(
                (
                    variable_df.loc[
                        variable_df[
                            "target_segment"
                        ]
                        == "T1",
                        "status",
                    ]
                    == "PASS"
                ).all()
            ),
        ),

        (
            "T17_adapter_executes",
            bool(
                (
                    variable_df.loc[
                        variable_df[
                            "target_segment"
                        ]
                        == "T17",
                        "status",
                    ]
                    == "PASS"
                ).all()
            ),
        ),

        (
            "T59_adapter_executes",
            bool(
                (
                    variable_df.loc[
                        variable_df[
                            "target_segment"
                        ]
                        == "T59",
                        "status",
                    ]
                    == "PASS"
                ).all()
            ),
        ),

        (
            "canonical_parameter_hash_unchanged",
            (
                hash_before
                == hash_after
                == EXPECTED_INITIAL_STATE_SHA256
            ),
        ),

        (
            "no_gradients_created",
            all(
                parameter.grad
                is None
                for parameter
                in model.parameters()
            ),
        ),

        (
            "Adam_not_instantiated",
            True,
        ),

        (
            "backward_not_performed",
            True,
        ),

        (
            "optimizer_steps_zero",
            True,
        ),
    ]

    invariant_df = pd.DataFrame(
        [
            {
                "check": name,

                "result": (
                    "PASS"
                    if passed
                    else "FAIL"
                ),
            }
            for (
                name,
                passed,
            ) in checks
        ]
    )

    require(
        (
            invariant_df[
                "result"
            ]
            == "PASS"
        ).all(),
        (
            "At least one Phase-5.3.1l.2a "
            "adapter invariant failed."
        ),
    )

    print(
        invariant_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Write / freeze
    # =========================================================================

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTRACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    diff_df.to_csv(
        AST_DIFF_PATH,
        index=False,
    )

    equivalence_df.to_csv(
        T60_EQUIVALENCE_PATH,
        index=False,
    )

    variable_df.to_csv(
        VARIABLE_H_PATH,
        index=False,
    )

    state_df.to_csv(
        STATE_PATH,
        index=False,
    )

    invariant_df.to_csv(
        FINAL_INVARIANT_PATH,
        index=False,
    )

    decision_df = pd.DataFrame(
        [
            {
                "decision": (
                    "training_trend_sequence_runtime"
                ),

                "value": (
                    "PHASE4_6_2_ENCODE_SEQUENCE_MINUS_"
                    "T60_ONLY_AUDIT_GUARD"
                ),

                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1l_2a"
                ),
            },

            {
                "decision": (
                    "training_history_length"
                ),

                "value": (
                    "TARGET_T_h_USES_EXACTLY_h_PERIODS_"
                    "T0_TO_T_h_MINUS_1"
                ),

                "classification": (
                    "INHERITED_FROZEN_PHASE_5_3_1l_1"
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1l_2a"
                ),
            },

            {
                "decision": (
                    "post_h_GRU_padding"
                ),

                "value": (
                    "FORBIDDEN"
                ),

                "classification": (
                    "EVALUATION_INTEGRITY_GUARD"
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1l_2a"
                ),
            },
        ]
    )

    decision_df.to_csv(
        DECISION_REGISTER_PATH,
        index=False,
    )

    contract = {
        "phase": (
            "5.3.1l.2a"
        ),

        "title": (
            "Training-Compatible Trend "
            "Sequence Adapter Contract"
        ),

        "status": (
            "FROZEN"
        ),

        "classification": (
            "IMPLEMENTATION_EQUIVALENT_CHOICE"
        ),

        "source": {
            "phase4_6_2_source": (
                str(
                    FORWARD_SOURCE_PATH
                )
            ),

            "source_sha256": (
                FORWARD_SOURCE_SHA256
            ),

            "original_method": (
                "TrendExtractor.encode_sequence"
            ),

            "original_method_ast_sha256": (
                original_method_sha
            ),
        },

        "adapter": {
            "runtime_method_name": (
                "encode_training_sequence"
            ),

            "adapter_ast_sha256": (
                adapter_method_sha
            ),

            "removed_statement_count": (
                1
            ),

            "removed_statement_sha256": (
                removed_guard_sha
            ),

            "removed_statement_source": (
                ast.unparse(
                    removed_guard
                )
            ),

            "removed_guard_message": (
                T60_GUARD_MESSAGE
            ),

            "all_other_statements_exact": (
                True
            ),
        },

        "numerical_proof": {
            "T60_exact_equality": (
                True
            ),

            "variable_h_verified": [
                1,
                17,
                59,
            ],

            "expected_F_t_shape": [
                1,
                40,
            ],

            "expected_gru_output_shape": (
                "[1, h, 40]"
            ),
        },

        "temporal_semantics": {
            "target_T_h_history": (
                "T0..T(h-1)"
            ),

            "sequence_length": (
                "h"
            ),

            "post_h_padding": (
                False
            ),
        },

        "state_neutrality": {
            "canonical_state_sha256_before": (
                hash_before
            ),

            "canonical_state_sha256_after": (
                hash_after
            ),

            "expected_sha256": (
                EXPECTED_INITIAL_STATE_SHA256
            ),
        },

        "next_phase": {
            "id": (
                "5.3.1l.2b"
            ),

            "title": (
                "Corrected Adam + Epoch-0 "
                "First Mini-Batch Preflight"
            ),

            "requirement": (
                "Repeat the frozen first-batch Adam "
                "forward/backward preflight using "
                "encode_training_sequence for T1..T59. "
                "optimizer.step() remains zero."
            ),
        },
    }

    CONTRACT_PATH.write_text(
        json.dumps(
            contract,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    manifest = {
        "phase": (
            "5.3.1l.2a"
        ),

        "status": (
            "TRAINING_TREND_SEQUENCE_ADAPTER_"
            "PROVED_AND_FROZEN"
        ),

        "original_method_ast_sha256": (
            original_method_sha
        ),

        "adapter_method_ast_sha256": (
            adapter_method_sha
        ),

        "removed_guard_sha256": (
            removed_guard_sha
        ),

        "T60_exact_equivalence": (
            True
        ),

        "variable_h_verified": [
            1,
            17,
            59,
        ],

        "canonical_state_sha256": (
            hash_after
        ),

        "Adam_instantiated": (
            False
        ),

        "optimizer_steps": (
            0
        ),

        "contract": (
            str(
                CONTRACT_PATH
            )
        ),
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # =========================================================================
    # Final
    # =========================================================================

    banner(
        "PHASE 5.3.1l.2a FINAL STATUS"
    )

    print(
        "Original method:"
    )

    print(
        "  TrendExtractor.encode_sequence"
    )

    print()

    print(
        "Original AST SHA256:"
    )

    print(
        original_method_sha
    )

    print()

    print(
        "Training adapter:"
    )

    print(
        "  TrendExtractor.encode_training_sequence"
    )

    print()

    print(
        "Training-adapter AST SHA256:"
    )

    print(
        adapter_method_sha
    )

    print()

    print(
        "Removed statements:                  1"
    )

    print(
        "Removed statement:"
    )

    print(
        "  "
        + ast.unparse(
            removed_guard
        )
    )

    print()

    print(
        "All other statements unchanged:      PASS"
    )

    print(
        "T60 original/adapter equality:        EXACT PASS"
    )

    print(
        "T1 variable-history execution:        PASS"
    )

    print(
        "T17 variable-history execution:       PASS"
    )

    print(
        "T59 variable-history execution:       PASS"
    )

    print()

    print(
        "Post-h GRU padding:                   FORBIDDEN"
    )

    print()

    print(
        "Canonical parameter SHA256:"
    )

    print(
        hash_after
    )

    print()

    print(
        "Adam instantiated:                    NO"
    )

    print(
        "Optimizer steps:                      0"
    )

    banner(
        "PHASE 5.3.1l.2a COMPLETE / "
        "TRAINING TREND SEQUENCE ADAPTER FROZEN"
    )


if __name__ == "__main__":
    main()