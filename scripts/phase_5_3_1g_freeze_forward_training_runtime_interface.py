
#!/usr/bin/env python3
"""
Phase 5.3.1g — Canonical Forward Implementation Provenance Audit
CORRECTED RERUN

IMPORTANT
---------
This subphase is now an AUDIT, not a freeze.

Why?
----
Phase 5.3.1f established that:

    phase_4_7_1b_freeze_neural_initialization_seed_contract.py

is authoritative for:
- ITRSModel topology;
- canonical parameter namespace;
- canonical Kaiming initialization;
- seed 42;
- canonical initial-state SHA256.

However, Phase 5.3.1g initially assumed that the same source also
implemented every module's forward() method.

That assumption was false.

In particular, DescriptionEncoder.forward() is absent from the
Phase-4.7.1b initialization-freeze source.

This corrected audit therefore answers a different and necessary
question:

    WHICH frozen Phase-4 source actually contains the canonical
    end-to-end forward implementation that Phase 5 should execute?

The audit statically scans all scripts/phase_4*.py files and inventories:

- classes;
- methods;
- forward() methods;
- forward signatures;
- forward AST hashes;
- forward source;
- calls made inside forward();
- return expressions;
- model-level ITRS classes;
- BCEWithLogitsLoss evidence;
- branch-module forward coverage.

It then ranks forward-runtime provenance candidates.

NO forward implementation is frozen by this script.

THIS SCRIPT DOES NOT:
- import any Phase-4 Python module;
- execute any Phase-4 workflow;
- instantiate a model;
- instantiate torch.Generator;
- instantiate NumPy Generator;
- instantiate Adam;
- generate training negatives;
- generate training order;
- perform forward computation;
- perform backward computation;
- call optimizer.step();
- modify Phase-4 artifacts.

All inspection is static AST/source analysis.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


# =============================================================================
# Frozen initialization/runtime-loader provenance
# =============================================================================

CANONICAL_INITIALIZATION_SOURCE_PATH = Path(
    "scripts/"
    "phase_4_7_1b_freeze_neural_initialization_seed_contract.py"
)

CANONICAL_INITIALIZATION_SOURCE_SHA256 = (
    "c55f3ea1646cec7fdc8ef69f2310d98f"
    "5ee95fab77f0c48392f4a9f76612761c"
)

FROZEN_RUNTIME_AST_SHA256 = (
    "301a074aa57cfe7602f2ccbb5b8e26943"
    "b94b72e36efe4d60d1af48378c58a6e"
)

CANONICAL_INITIAL_STATE_SHA256 = (
    "49e822ea7fad35c458f47e134c94c05e"
    "ac099b68c5c468e2c71559c8c88998ab"
)

EXPECTED_PARAMETER_COUNT = 19_217_929
EXPECTED_PARAMETER_TENSORS = 32


# =============================================================================
# Prior Phase-5 contracts
# =============================================================================

PHASE_5_3_1F_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_1f_side_effect_free_runtime_loading_contract.json"
)

PHASE_5_3_1F_MANIFEST_PATH = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_1f/"
    "phase_5_3_1f_runtime_loader_manifest.json"
)


# =============================================================================
# Phase-4 scripts
# =============================================================================

SCRIPT_DIR = Path(
    "scripts"
)

EXPECTED_FORWARD_AUDIT_SCRIPT_NAME = (
    "phase_4_6_2_end_to_end_itrs_forward_bce_audit.py"
)


# =============================================================================
# Canonical conceptual module names
#
# Presence of these names is evidence only.
# We DO NOT require Phase-4.7.1b to implement forward() for all of them.
# =============================================================================

CANONICAL_BRANCH_CLASS_NAMES = (
    "DescriptionEncoder",
    "TrendExtractor",
    "BasisRGCNLayer",
    "PreferencePropagation",
    "ScoringMLP",
)

INITIALIZATION_MODEL_CLASS = (
    "ITRSModel"
)


# =============================================================================
# Outputs
# =============================================================================

OUT_DIR = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_1g"
)

SCRIPT_SUMMARY_PATH = (
    OUT_DIR
    / "phase4_forward_provenance_script_summary.csv"
)

CLASS_INVENTORY_PATH = (
    OUT_DIR
    / "phase4_forward_provenance_class_inventory.csv"
)

METHOD_INVENTORY_PATH = (
    OUT_DIR
    / "phase4_method_inventory.csv"
)

FORWARD_INVENTORY_PATH = (
    OUT_DIR
    / "phase4_forward_method_inventory.csv"
)

FORWARD_CALL_PATH = (
    OUT_DIR
    / "phase4_forward_call_inventory.csv"
)

FORWARD_RETURN_PATH = (
    OUT_DIR
    / "phase4_forward_return_inventory.csv"
)

MODEL_FORWARD_CANDIDATE_PATH = (
    OUT_DIR
    / "phase4_model_forward_candidate_inventory.csv"
)

COMMON_CLASS_COMPARISON_PATH = (
    OUT_DIR
    / "phase4_common_class_forward_comparison.csv"
)

FINAL_INVARIANT_PATH = (
    OUT_DIR
    / "phase_5_3_1g_final_audit_invariants.csv"
)

MANIFEST_PATH = (
    OUT_DIR
    / "phase_5_3_1g_forward_provenance_manifest.json"
)


# =============================================================================
# Generic helpers
# =============================================================================

def banner(
    text: str,
) -> None:

    print(
        "\n"
        + "=" * 118
    )

    print(
        text
    )

    print(
        "=" * 118
    )


def require(
    condition: bool,
    message: str,
) -> None:

    if not bool(
        condition
    ):

        raise AssertionError(
            message
        )


def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:

        while True:

            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def sha256_text(
    text: str,
) -> str:

    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


def load_json(
    path: Path,
) -> dict:

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        return json.load(
            handle
        )


def safe_read_text(
    path: Path,
) -> str:

    return path.read_text(
        encoding="utf-8",
        errors="replace",
    )


def dotted_name(
    node: ast.AST,
) -> str:

    if isinstance(
        node,
        ast.Name,
    ):

        return (
            node.id
        )

    if isinstance(
        node,
        ast.Attribute,
    ):

        prefix = dotted_name(
            node.value
        )

        if prefix:

            return (
                f"{prefix}.{node.attr}"
            )

        return (
            node.attr
        )

    return ""


def ast_sha256(
    node: ast.AST,
) -> str:

    payload = ast.dump(
        node,
        annotate_fields=True,
        include_attributes=False,
    )

    return sha256_text(
        payload
    )


def node_source(
    source: str,
    node: ast.AST,
) -> str:

    value = ast.get_source_segment(
        source,
        node,
    )

    if value is not None:

        return (
            value
        )

    return ast.unparse(
        node
    )


def compact_source(
    text: str,
    limit: int = 1200,
) -> str:

    text = (
        " ".join(
            text.split()
        )
    )

    if len(
        text
    ) <= limit:

        return (
            text
        )

    return (
        text[
            :limit
        ]
        + " ..."
    )


# =============================================================================
# Static signature extraction
# =============================================================================

def argument_text(
    argument: ast.arg,
) -> str:

    if (
        argument.annotation
        is None
    ):

        return (
            argument.arg
        )

    return (
        f"{argument.arg}: "
        f"{ast.unparse(argument.annotation)}"
    )


def ast_function_signature(
    function_node: ast.FunctionDef,
    drop_self: bool = False,
) -> str:
    """
    Build a readable signature from AST only.

    This avoids importing any Phase-4 script.
    """

    args = (
        function_node.args
    )

    positional = (
        list(
            args.posonlyargs
        )
        + list(
            args.args
        )
    )

    defaults = (
        list(
            args.defaults
        )
    )

    default_start = (
        len(
            positional
        )
        - len(
            defaults
        )
    )

    parts = []

    for index, argument in enumerate(
        positional
    ):

        if (
            drop_self
            and argument.arg
            == "self"
            and index
            == 0
        ):

            continue

        value = (
            argument_text(
                argument
            )
        )

        if (
            index
            >= default_start
        ):

            default = (
                defaults[
                    index
                    - default_start
                ]
            )

            value += (
                "="
                + ast.unparse(
                    default
                )
            )

        parts.append(
            value
        )

        if (
            len(
                args.posonlyargs
            )
            > 0
            and index
            == (
                len(
                    args.posonlyargs
                )
                - 1
            )
        ):

            parts.append(
                "/"
            )

    if (
        args.vararg
        is not None
    ):

        parts.append(
            "*"
            + argument_text(
                args.vararg
            )
        )

    elif (
        len(
            args.kwonlyargs
        )
        > 0
    ):

        parts.append(
            "*"
        )

    for (
        argument,
        default,
    ) in zip(
        args.kwonlyargs,
        args.kw_defaults,
    ):

        value = (
            argument_text(
                argument
            )
        )

        if (
            default
            is not None
        ):

            value += (
                "="
                + ast.unparse(
                    default
                )
            )

        parts.append(
            value
        )

    if (
        args.kwarg
        is not None
    ):

        parts.append(
            "**"
            + argument_text(
                args.kwarg
            )
        )

    return_annotation = ""

    if (
        function_node.returns
        is not None
    ):

        return_annotation = (
            " -> "
            + ast.unparse(
                function_node.returns
            )
        )

    return (
        "("
        + ", ".join(
            parts
        )
        + ")"
        + return_annotation
    )


# =============================================================================
# AST class / method helpers
# =============================================================================

def top_level_classes(
    tree: ast.Module,
) -> list[
    ast.ClassDef
]:

    return [
        node
        for node
        in tree.body
        if isinstance(
            node,
            ast.ClassDef,
        )
    ]


def direct_class_methods(
    class_node: ast.ClassDef,
) -> list[
    ast.FunctionDef
]:

    return [
        node
        for node
        in class_node.body
        if isinstance(
            node,
            ast.FunctionDef,
        )
    ]


def forward_method_optional(
    class_node: ast.ClassDef,
) -> ast.FunctionDef | None:
    """
    IMPORTANT CORRECTION:

    Missing forward() is VALID evidence.

    Phase-4.7.1b classes may exist only for topology /
    initialization reconstruction.
    """

    matches = [
        node
        for node
        in direct_class_methods(
            class_node
        )
        if node.name
        == "forward"
    ]

    require(
        len(
            matches
        )
        <= 1,
        (
            f"Class {class_node.name} contains multiple "
            "direct forward() definitions"
        ),
    )

    if not matches:

        return None

    return (
        matches[
            0
        ]
    )


def loaded_names(
    node: ast.AST,
) -> list[str]:

    values = set()

    for candidate in ast.walk(
        node
    ):

        if (
            isinstance(
                candidate,
                ast.Name,
            )
            and isinstance(
                candidate.ctx,
                ast.Load,
            )
        ):

            values.add(
                candidate.id
            )

    return sorted(
        values
    )


def call_inventory(
    node: ast.AST,
) -> list[dict]:

    rows = []

    for candidate in ast.walk(
        node
    ):

        if not isinstance(
            candidate,
            ast.Call,
        ):

            continue

        rows.append(
            {
                "line_number": (
                    int(
                        candidate.lineno
                    )
                ),
                "call_name": (
                    dotted_name(
                        candidate.func
                    )
                ),
                "call_source": (
                    compact_source(
                        ast.unparse(
                            candidate
                        ),
                        limit=600,
                    )
                ),
            }
        )

    return (
        rows
    )


def return_inventory(
    node: ast.AST,
) -> list[dict]:

    rows = []

    for candidate in ast.walk(
        node
    ):

        if not isinstance(
            candidate,
            ast.Return,
        ):

            continue

        rows.append(
            {
                "line_number": (
                    int(
                        candidate.lineno
                    )
                ),
                "return_expression": (
                    "None"
                    if candidate.value
                    is None
                    else ast.unparse(
                        candidate.value
                    )
                ),
            }
        )

    return (
        rows
    )


def is_nn_module_class(
    class_node: ast.ClassDef,
) -> bool:

    base_names = [
        dotted_name(
            base
        )
        for base
        in class_node.bases
    ]

    return any(
        name.endswith(
            "Module"
        )
        for name
        in base_names
    )


def is_model_level_class_name(
    class_name: str,
) -> bool:

    lower = (
        class_name.lower()
    )

    return (
        "itrs"
        in lower
        and (
            "model"
            in lower
            or class_name
            == "ITRSModel"
        )
    )


# =============================================================================
# Script-wide semantic evidence
# =============================================================================

def has_call_suffix(
    tree: ast.Module,
    suffix: str,
) -> bool:

    for node in ast.walk(
        tree
    ):

        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        name = (
            dotted_name(
                node.func
            )
        )

        if name.endswith(
            suffix
        ):

            return True

    return False


def contains_text_token(
    source: str,
    token: str,
) -> bool:

    return (
        token.lower()
        in source.lower()
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.3.1g — "
        "CANONICAL FORWARD IMPLEMENTATION PROVENANCE AUDIT "
        "(CORRECTED RERUN)"
    )

    print(
        "Normal Phase-4 import:                NO"
    )

    print(
        "Sanitized runtime execution:          NO"
    )

    print(
        "Model instantiated:                   NO"
    )

    print(
        "Training-negative RNG instantiated:  NO"
    )

    print(
        "Training-order RNG instantiated:     NO"
    )

    print(
        "Optimizer instantiated:               NO"
    )

    print(
        "Forward computation performed:        NO"
    )

    print(
        "Backward computation performed:       NO"
    )

    print(
        "Optimizer steps:                      0"
    )

    # =========================================================================
    # Frozen prerequisite
    # =========================================================================

    banner(
        "PHASE 5.3.1f RUNTIME-LOADER CONTRACT RECHECK"
    )

    for path in (
        CANONICAL_INITIALIZATION_SOURCE_PATH,
        PHASE_5_3_1F_CONTRACT_PATH,
        PHASE_5_3_1F_MANIFEST_PATH,
    ):

        require(
            path.exists(),
            (
                "Missing authoritative input: "
                f"{path}"
            ),
        )

        print(
            f"FOUND  {path}"
        )

    initialization_source_hash = (
        sha256_file(
            CANONICAL_INITIALIZATION_SOURCE_PATH
        )
    )

    require(
        initialization_source_hash
        == CANONICAL_INITIALIZATION_SOURCE_SHA256,
        (
            "Canonical initialization source SHA256 drift.\n"
            f"Expected: "
            f"{CANONICAL_INITIALIZATION_SOURCE_SHA256}\n"
            f"Actual:   {initialization_source_hash}"
        ),
    )

    loader_contract = (
        load_json(
            PHASE_5_3_1F_CONTRACT_PATH
        )
    )

    loader_manifest = (
        load_json(
            PHASE_5_3_1F_MANIFEST_PATH
        )
    )

    require(
        loader_contract[
            "status"
        ]
        == "FROZEN",
        (
            "Phase-5.3.1f runtime loader is not frozen"
        ),
    )

    require(
        loader_contract[
            "loading_policy"
        ][
            "mechanism"
        ]
        == "SANITIZED_PREFIX_AST",
        (
            "Unexpected frozen runtime loading mechanism"
        ),
    )

    require(
        loader_contract[
            "loading_policy"
        ][
            "sanitized_ast_sha256"
        ]
        == FROZEN_RUNTIME_AST_SHA256,
        (
            "Frozen sanitized runtime AST hash drift"
        ),
    )

    require(
        loader_manifest[
            "optimizer_instantiated"
        ]
        is False,
        (
            "Optimizer unexpectedly instantiated "
            "before Phase-5.3.1g"
        ),
    )

    require(
        int(
            loader_manifest[
                "optimizer_steps"
            ]
        )
        == 0,
        (
            "Optimizer step occurred before "
            "Phase-5.3.1g"
        ),
    )

    require(
        loader_manifest[
            "canonical_state_sha256"
        ]
        == CANONICAL_INITIAL_STATE_SHA256,
        (
            "Canonical initial-state hash drift "
            "before forward-provenance audit"
        ),
    )

    print(
        "Canonical initialization source:      PASS"
    )

    print(
        "Side-effect-free loader:               FROZEN  PASS"
    )

    print(
        "Canonical initial-state oracle:        PASS"
    )

    print(
        "Optimizer steps:                       0  PASS"
    )

    # =========================================================================
    # Find all Phase-4 scripts
    # =========================================================================

    banner(
        "PHASE-4 SCRIPT DISCOVERY"
    )

    script_paths = sorted(
        SCRIPT_DIR.glob(
            "phase_4*.py"
        )
    )

    require(
        len(
            script_paths
        )
        > 0,
        (
            "No Phase-4 Python scripts found"
        ),
    )

    expected_forward_script = (
        SCRIPT_DIR
        / EXPECTED_FORWARD_AUDIT_SCRIPT_NAME
    )

    require(
        expected_forward_script.exists(),
        (
            "Expected Phase-4 forward/BCE audit script missing: "
            f"{expected_forward_script}"
        ),
    )

    print(
        f"Phase-4 Python scripts found:          "
        f"{len(script_paths)}"
    )

    print(
        f"Expected forward-audit source exists: "
        f"YES"
    )

    # =========================================================================
    # Static AST inventory
    # =========================================================================

    banner(
        "STATIC CLASS / METHOD / FORWARD INVENTORY"
    )

    script_rows = []
    class_rows = []
    method_rows = []
    forward_rows = []
    forward_call_rows = []
    forward_return_rows = []
    model_candidate_rows = []

    parsed_script_count = 0

    for script_path in (
        script_paths
    ):

        source = (
            safe_read_text(
                script_path
            )
        )

        source_hash = (
            sha256_file(
                script_path
            )
        )

        try:

            tree = ast.parse(
                source,
                filename=str(
                    script_path
                ),
            )

            parse_ok = True
            parse_error = None

        except SyntaxError as exc:

            tree = None
            parse_ok = False
            parse_error = str(
                exc
            )

        if not parse_ok:

            script_rows.append(
                {
                    "script": (
                        str(
                            script_path
                        )
                    ),

                    "script_sha256": (
                        source_hash
                    ),

                    "ast_parse_ok": (
                        False
                    ),

                    "ast_parse_error": (
                        parse_error
                    ),

                    "class_count": (
                        0
                    ),

                    "nn_module_class_count": (
                        0
                    ),

                    "forward_method_count": (
                        0
                    ),

                    "model_level_forward_count": (
                        0
                    ),

                    "canonical_branch_classes_present": (
                        0
                    ),

                    "canonical_branch_forwards_present": (
                        0
                    ),

                    "contains_BCEWithLogitsLoss": (
                        False
                    ),

                    "contains_sigmoid": (
                        False
                    ),

                    "contains_torch_cat": (
                        False
                    ),

                    "forward_provenance_score": (
                        0
                    ),
                }
            )

            continue

        parsed_script_count += 1

        classes = (
            top_level_classes(
                tree
            )
        )

        class_names = {
            class_node.name
            for class_node
            in classes
        }

        nn_module_classes = [
            class_node
            for class_node
            in classes
            if is_nn_module_class(
                class_node
            )
        ]

        forward_count = 0
        model_level_forward_count = 0

        branch_classes_present = sum(
            class_name
            in class_names
            for class_name
            in CANONICAL_BRANCH_CLASS_NAMES
        )

        branch_forwards_present = 0

        model_class_names_with_forward = []

        for class_node in (
            classes
        ):

            methods = (
                direct_class_methods(
                    class_node
                )
            )

            forward_node = (
                forward_method_optional(
                    class_node
                )
            )

            method_names = [
                method.name
                for method
                in methods
            ]

            class_rows.append(
                {
                    "script": (
                        str(
                            script_path
                        )
                    ),

                    "class_name": (
                        class_node.name
                    ),

                    "line_number": (
                        int(
                            class_node.lineno
                        )
                    ),

                    "is_nn_module": (
                        is_nn_module_class(
                            class_node
                        )
                    ),

                    "is_model_level_class": (
                        is_model_level_class_name(
                            class_node.name
                        )
                    ),

                    "class_ast_sha256": (
                        ast_sha256(
                            class_node
                        )
                    ),

                    "method_count": (
                        len(
                            methods
                        )
                    ),

                    "method_names": (
                        ";".join(
                            method_names
                        )
                    ),

                    "has_forward": (
                        forward_node
                        is not None
                    ),
                }
            )

            for method in (
                methods
            ):

                method_rows.append(
                    {
                        "script": (
                            str(
                                script_path
                            )
                        ),

                        "class_name": (
                            class_node.name
                        ),

                        "method_name": (
                            method.name
                        ),

                        "line_number": (
                            int(
                                method.lineno
                            )
                        ),

                        "signature_without_self": (
                            ast_function_signature(
                                method,
                                drop_self=True,
                            )
                        ),

                        "method_ast_sha256": (
                            ast_sha256(
                                method
                            )
                        ),
                    }
                )

            if (
                forward_node
                is None
            ):

                continue

            forward_count += 1

            if (
                class_node.name
                in CANONICAL_BRANCH_CLASS_NAMES
            ):

                branch_forwards_present += 1

            model_level = (
                is_model_level_class_name(
                    class_node.name
                )
            )

            if (
                model_level
            ):

                model_level_forward_count += 1

                model_class_names_with_forward.append(
                    class_node.name
                )

            signature = (
                ast_function_signature(
                    forward_node,
                    drop_self=True,
                )
            )

            returns = (
                return_inventory(
                    forward_node
                )
            )

            calls = (
                call_inventory(
                    forward_node
                )
            )

            loaded = (
                loaded_names(
                    forward_node
                )
            )

            forward_rows.append(
                {
                    "script": (
                        str(
                            script_path
                        )
                    ),

                    "script_sha256": (
                        source_hash
                    ),

                    "class_name": (
                        class_node.name
                    ),

                    "is_model_level_class": (
                        model_level
                    ),

                    "forward_line_number": (
                        int(
                            forward_node.lineno
                        )
                    ),

                    "signature_without_self": (
                        signature
                    ),

                    "forward_ast_sha256": (
                        ast_sha256(
                            forward_node
                        )
                    ),

                    "return_expression_count": (
                        len(
                            returns
                        )
                    ),

                    "return_expressions": (
                        " || ".join(
                            row[
                                "return_expression"
                            ]
                            for row
                            in returns
                        )
                    ),

                    "loaded_names": (
                        ";".join(
                            loaded
                        )
                    ),

                    "forward_source": (
                        node_source(
                            source,
                            forward_node,
                        )
                    ),
                }
            )

            for call in (
                calls
            ):

                forward_call_rows.append(
                    {
                        "script": (
                            str(
                                script_path
                            )
                        ),

                        "class_name": (
                            class_node.name
                        ),

                        **call,
                    }
                )

            for return_row in (
                returns
            ):

                forward_return_rows.append(
                    {
                        "script": (
                            str(
                                script_path
                            )
                        ),

                        "class_name": (
                            class_node.name
                        ),

                        **return_row,
                    }
                )

        contains_bce = (
            has_call_suffix(
                tree,
                "BCEWithLogitsLoss",
            )
            or contains_text_token(
                source,
                "BCEWithLogitsLoss",
            )
        )

        contains_sigmoid = (
            contains_text_token(
                source,
                "sigmoid",
            )
        )

        contains_torch_cat = (
            contains_text_token(
                source,
                "torch.cat",
            )
        )

        # ---------------------------------------------------------------------
        # Diagnostic provenance score.
        #
        # This does NOT freeze anything.
        # ---------------------------------------------------------------------

        score = 0

        if (
            model_level_forward_count
            > 0
        ):

            score += 300

        score += (
            branch_forwards_present
            * 40
        )

        if contains_bce:

            score += 80

        if contains_torch_cat:

            score += 30

        if contains_sigmoid:

            score += 10

        if (
            script_path.name
            == EXPECTED_FORWARD_AUDIT_SCRIPT_NAME
        ):

            score += 50

        script_rows.append(
            {
                "script": (
                    str(
                        script_path
                    )
                ),

                "script_sha256": (
                    source_hash
                ),

                "ast_parse_ok": (
                    True
                ),

                "ast_parse_error": (
                    None
                ),

                "class_count": (
                    len(
                        classes
                    )
                ),

                "nn_module_class_count": (
                    len(
                        nn_module_classes
                    )
                ),

                "forward_method_count": (
                    forward_count
                ),

                "model_level_forward_count": (
                    model_level_forward_count
                ),

                "model_classes_with_forward": (
                    ";".join(
                        model_class_names_with_forward
                    )
                ),

                "canonical_branch_classes_present": (
                    branch_classes_present
                ),

                "canonical_branch_forwards_present": (
                    branch_forwards_present
                ),

                "contains_BCEWithLogitsLoss": (
                    contains_bce
                ),

                "contains_sigmoid": (
                    contains_sigmoid
                ),

                "contains_torch_cat": (
                    contains_torch_cat
                ),

                "forward_provenance_score": (
                    score
                ),
            }
        )

        # ---------------------------------------------------------------------
        # Model-level candidate details
        # ---------------------------------------------------------------------

        for class_node in (
            classes
        ):

            forward_node = (
                forward_method_optional(
                    class_node
                )
            )

            if (
                forward_node
                is None
                or not is_model_level_class_name(
                    class_node.name
                )
            ):

                continue

            model_candidate_rows.append(
                {
                    "script": (
                        str(
                            script_path
                        )
                    ),

                    "script_sha256": (
                        source_hash
                    ),

                    "model_class_name": (
                        class_node.name
                    ),

                    "model_class_ast_sha256": (
                        ast_sha256(
                            class_node
                        )
                    ),

                    "forward_line_number": (
                        int(
                            forward_node.lineno
                        )
                    ),

                    "forward_signature_without_self": (
                        ast_function_signature(
                            forward_node,
                            drop_self=True,
                        )
                    ),

                    "forward_ast_sha256": (
                        ast_sha256(
                            forward_node
                        )
                    ),

                    "contains_BCEWithLogitsLoss_in_script": (
                        contains_bce
                    ),

                    "canonical_branch_forwards_present": (
                        branch_forwards_present
                    ),

                    "forward_source": (
                        node_source(
                            source,
                            forward_node,
                        )
                    ),
                }
            )

    script_df = pd.DataFrame(
        script_rows
    )

    class_df = pd.DataFrame(
        class_rows
    )

    method_df = pd.DataFrame(
        method_rows
    )

    forward_df = pd.DataFrame(
        forward_rows
    )

    forward_call_df = pd.DataFrame(
        forward_call_rows
    )

    forward_return_df = pd.DataFrame(
        forward_return_rows
    )

    model_candidate_df = pd.DataFrame(
        model_candidate_rows
    )

    require(
        parsed_script_count
        > 0,
        (
            "No Phase-4 script parsed successfully"
        ),
    )

    # =========================================================================
    # Phase-4.7.1b initialization source diagnostic
    # =========================================================================

    banner(
        "PHASE-4.7.1b INITIALIZATION SOURCE METHOD DIAGNOSTIC"
    )

    init_classes = (
        class_df.loc[
            class_df[
                "script"
            ]
            == str(
                CANONICAL_INITIALIZATION_SOURCE_PATH
            )
        ].copy()
    )

    require(
        not init_classes.empty,
        (
            "Could not locate canonical Phase-4.7.1b classes "
            "inside the AST inventory"
        ),
    )

    print(
        init_classes[
            [
                "class_name",
                "is_nn_module",
                "method_count",
                "method_names",
                "has_forward",
            ]
        ].to_string(
            index=False
        )
    )

    description_rows = (
        init_classes.loc[
            init_classes[
                "class_name"
            ]
            == "DescriptionEncoder"
        ]
    )

    require(
        len(
            description_rows
        )
        == 1,
        (
            "Expected exactly one DescriptionEncoder "
            "in canonical initialization source"
        ),
    )

    description_forward_present = bool(
        description_rows.iloc[
            0
        ][
            "has_forward"
        ]
    )

    print()

    print(
        "DescriptionEncoder.forward in "
        "Phase-4.7.1b:"
    )

    print(
        "  PRESENT"
        if description_forward_present
        else "  ABSENT"
    )

    print()

    print(
        "Interpretation:"
    )

    print(
        "  Phase-4.7.1b is authoritative for topology and"
    )

    print(
        "  initialization, but forward-runtime provenance must"
    )

    print(
        "  be resolved independently."
    )

    # =========================================================================
    # Rank scripts by forward-runtime evidence
    # =========================================================================

    banner(
        "PHASE-4 FORWARD-RUNTIME PROVENANCE RANKING"
    )

    ranked_script_df = (
        script_df.loc[
            script_df[
                "ast_parse_ok"
            ]
        ]
        .sort_values(
            [
                "forward_provenance_score",
                "model_level_forward_count",
                "canonical_branch_forwards_present",
                "script",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    print(
        ranked_script_df[
            [
                "script",
                "class_count",
                "nn_module_class_count",
                "forward_method_count",
                "model_level_forward_count",
                "model_classes_with_forward",
                "canonical_branch_forwards_present",
                "contains_BCEWithLogitsLoss",
                "contains_torch_cat",
                "forward_provenance_score",
            ]
        ]
        .head(
            20
        )
        .to_string(
            index=False
        )
    )

    # =========================================================================
    # Model-level forward candidates
    # =========================================================================

    banner(
        "MODEL-LEVEL FORWARD IMPLEMENTATION CANDIDATES"
    )

    if (
        model_candidate_df.empty
    ):

        print(
            "No model-level ITRS forward implementation "
            "was found."
        )

    else:

        model_candidate_ranked = (
            model_candidate_df
            .merge(
                ranked_script_df[
                    [
                        "script",
                        "forward_provenance_score",
                    ]
                ],
                on="script",
                how="left",
            )
            .sort_values(
                [
                    "forward_provenance_score",
                    "canonical_branch_forwards_present",
                    "script",
                    "model_class_name",
                ],
                ascending=[
                    False,
                    False,
                    True,
                    True,
                ],
                kind="mergesort",
            )
            .reset_index(
                drop=True
            )
        )

        print(
            model_candidate_ranked[
                [
                    "script",
                    "model_class_name",
                    "forward_line_number",
                    "forward_signature_without_self",
                    "forward_ast_sha256",
                    "canonical_branch_forwards_present",
                    "contains_BCEWithLogitsLoss_in_script",
                    "forward_provenance_score",
                ]
            ].to_string(
                index=False
            )
        )

    # =========================================================================
    # Expected Phase-4.6.2 forward source
    # =========================================================================

    banner(
        "PHASE-4.6.2 END-TO-END FORWARD/BCE SOURCE AUDIT"
    )

    phase_4_6_2_path = (
        SCRIPT_DIR
        / EXPECTED_FORWARD_AUDIT_SCRIPT_NAME
    )

    phase_4_6_2_script_row = (
        ranked_script_df.loc[
            ranked_script_df[
                "script"
            ]
            == str(
                phase_4_6_2_path
            )
        ]
    )

    require(
        len(
            phase_4_6_2_script_row
        )
        == 1,
        (
            "Could not uniquely resolve Phase-4.6.2 "
            "inside script inventory"
        ),
    )

    phase_4_6_2_script_row = (
        phase_4_6_2_script_row.iloc[
            0
        ]
    )

    print(
        f"Script:"
    )

    print(
        f"  {phase_4_6_2_path}"
    )

    print()

    print(
        f"SHA256:"
    )

    print(
        f"  "
        f"{phase_4_6_2_script_row['script_sha256']}"
    )

    print()

    print(
        f"Forward methods:                    "
        f"{phase_4_6_2_script_row['forward_method_count']}"
    )

    print(
        f"Model-level forward methods:        "
        f"{phase_4_6_2_script_row['model_level_forward_count']}"
    )

    print(
        f"Canonical branch forwards present:  "
        f"{phase_4_6_2_script_row['canonical_branch_forwards_present']}"
        f"/{len(CANONICAL_BRANCH_CLASS_NAMES)}"
    )

    print(
        f"BCEWithLogitsLoss evidence:          "
        f"{phase_4_6_2_script_row['contains_BCEWithLogitsLoss']}"
    )

    phase_4_6_2_forwards = (
        forward_df.loc[
            forward_df[
                "script"
            ]
            == str(
                phase_4_6_2_path
            )
        ].copy()
    )

    if (
        phase_4_6_2_forwards.empty
    ):

        print()

        print(
            "No forward methods found in Phase-4.6.2."
        )

    else:

        print()

        print(
            phase_4_6_2_forwards[
                [
                    "class_name",
                    "is_model_level_class",
                    "forward_line_number",
                    "signature_without_self",
                    "forward_ast_sha256",
                    "return_expressions",
                ]
            ].to_string(
                index=False
            )
        )

    # =========================================================================
    # Show source of model-level forward candidates
    # =========================================================================

    banner(
        "MODEL-LEVEL FORWARD SOURCE DETAILS"
    )

    if (
        model_candidate_df.empty
    ):

        print(
            "NONE"
        )

    else:

        for (
            row_index,
            row,
        ) in model_candidate_df.iterrows():

            print(
                "-" * 118
            )

            print(
                f"SCRIPT: "
                f"{row['script']}"
            )

            print(
                f"CLASS:  "
                f"{row['model_class_name']}"
            )

            print(
                f"SIGNATURE: "
                f"{row['forward_signature_without_self']}"
            )

            print(
                f"FORWARD AST SHA256: "
                f"{row['forward_ast_sha256']}"
            )

            print()

            print(
                row[
                    "forward_source"
                ]
            )

            print()

    # =========================================================================
    # Direct call graph for model-level forwards
    # =========================================================================

    banner(
        "MODEL-LEVEL FORWARD DIRECT CALL INVENTORY"
    )

    if (
        forward_call_df.empty
        or class_df.empty
    ):

        model_forward_call_df = (
            pd.DataFrame()
        )

    else:

        model_class_lookup = (
            class_df[
                [
                    "script",
                    "class_name",
                    "is_model_level_class",
                ]
            ]
            .drop_duplicates()
        )

        model_forward_call_df = (
            forward_call_df
            .merge(
                model_class_lookup,
                on=[
                    "script",
                    "class_name",
                ],
                how="left",
            )
        )

        model_forward_call_df = (
            model_forward_call_df.loc[
                model_forward_call_df[
                    "is_model_level_class"
                ]
                == True
            ].copy()
        )

    if (
        model_forward_call_df.empty
    ):

        print(
            "No model-level forward calls found."
        )

    else:

        print(
            model_forward_call_df[
                [
                    "script",
                    "class_name",
                    "line_number",
                    "call_name",
                    "call_source",
                ]
            ].to_string(
                index=False
            )
        )

    # =========================================================================
    # Common-class comparison
    # =========================================================================

    banner(
        "COMMON CLASS FORWARD COMPARISON"
    )

    comparison_rows = []

    comparison_scripts = (
        str(
            CANONICAL_INITIALIZATION_SOURCE_PATH
        ),
        str(
            phase_4_6_2_path
        ),
    )

    for class_name in (
        CANONICAL_BRANCH_CLASS_NAMES
        + (
            INITIALIZATION_MODEL_CLASS,
        )
    ):

        for script_name in (
            comparison_scripts
        ):

            matching_class = (
                class_df.loc[
                    (
                        class_df[
                            "script"
                        ]
                        == script_name
                    )
                    & (
                        class_df[
                            "class_name"
                        ]
                        == class_name
                    )
                ]
            )

            matching_forward = (
                forward_df.loc[
                    (
                        forward_df[
                            "script"
                        ]
                        == script_name
                    )
                    & (
                        forward_df[
                            "class_name"
                        ]
                        == class_name
                    )
                ]
            )

            comparison_rows.append(
                {
                    "class_name": (
                        class_name
                    ),

                    "script": (
                        script_name
                    ),

                    "class_present": (
                        len(
                            matching_class
                        )
                        == 1
                    ),

                    "forward_present": (
                        len(
                            matching_forward
                        )
                        == 1
                    ),

                    "forward_signature": (
                        matching_forward.iloc[
                            0
                        ][
                            "signature_without_self"
                        ]
                        if len(
                            matching_forward
                        )
                        == 1
                        else None
                    ),

                    "forward_ast_sha256": (
                        matching_forward.iloc[
                            0
                        ][
                            "forward_ast_sha256"
                        ]
                        if len(
                            matching_forward
                        )
                        == 1
                        else None
                    ),
                }
            )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    print(
        comparison_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Decision-facing interpretation
    # =========================================================================

    banner(
        "FORWARD PROVENANCE DECISION-FACING INTERPRETATION"
    )

    # We intentionally do not freeze yet.
    #
    # Strong candidate = highest score.
    # Unique strongest candidate is useful evidence, but still requires
    # one follow-up freeze after inspecting its actual interface and
    # matching it to Phase-4 model contracts.

    highest_score = int(
        ranked_script_df.iloc[
            0
        ][
            "forward_provenance_score"
        ]
    )

    strongest = (
        ranked_script_df.loc[
            ranked_script_df[
                "forward_provenance_score"
            ]
            == highest_score
        ].copy()
    )

    unique_strongest = (
        len(
            strongest
        )
        == 1
    )

    strongest_script = (
        str(
            strongest.iloc[
                0
            ][
                "script"
            ]
        )
        if unique_strongest
        else None
    )

    print(
        f"Highest provenance evidence score:   "
        f"{highest_score}"
    )

    print(
        f"Scripts tied at highest score:       "
        f"{len(strongest)}"
    )

    if (
        unique_strongest
    ):

        print(
            "Unique strongest forward-runtime source:"
        )

        print(
            f"  {strongest_script}"
        )

    else:

        print(
            "No unique strongest forward-runtime source."
        )

        print(
            "Candidate ambiguity must be resolved before freezing."
        )

    print()

    print(
        "Canonical initialization source remains:"
    )

    print(
        f"  {CANONICAL_INITIALIZATION_SOURCE_PATH}"
    )

    print()

    print(
        "Forward implementation frozen by this audit: NO"
    )

    print(
        "Reason:"
    )

    print(
        "  Initialization provenance and forward-runtime provenance"
    )

    print(
        "  are separate and must be joined explicitly rather than"
    )

    print(
        "  assumed to come from the same Phase-4 script."
    )

    # =========================================================================
    # Hard audit invariants
    # =========================================================================

    banner(
        "FINAL PHASE-5.3.1g AUDIT INVARIANTS"
    )

    checks = [
        (
            "canonical_initialization_source_sha256_exact",
            (
                initialization_source_hash
                == CANONICAL_INITIALIZATION_SOURCE_SHA256
            ),
        ),

        (
            "phase_5_3_1f_loader_frozen",
            (
                loader_contract[
                    "status"
                ]
                == "FROZEN"
            ),
        ),

        (
            "phase_5_3_1f_sanitized_ast_exact",
            (
                loader_contract[
                    "loading_policy"
                ][
                    "sanitized_ast_sha256"
                ]
                == FROZEN_RUNTIME_AST_SHA256
            ),
        ),

        (
            "canonical_state_hash_unchanged",
            (
                loader_manifest[
                    "canonical_state_sha256"
                ]
                == CANONICAL_INITIAL_STATE_SHA256
            ),
        ),

        (
            "phase4_scripts_discovered",
            (
                len(
                    script_paths
                )
                > 0
            ),
        ),

        (
            "phase4_6_2_forward_audit_script_exists",
            (
                phase_4_6_2_path.exists()
            ),
        ),

        (
            "phase4_7_1b_DescriptionEncoder_audited_without_forward_assumption",
            (
                len(
                    description_rows
                )
                == 1
            ),
        ),

        (
            "static_forward_inventory_completed",
            (
                not forward_df.empty
            ),
        ),

        (
            "model_level_forward_candidates_audited",
            (
                model_candidate_df
                is not None
            ),
        ),

        (
            "no_phase4_module_import",
            True,
        ),

        (
            "no_model_instantiation",
            True,
        ),

        (
            "no_training_negative_rng",
            True,
        ),

        (
            "no_training_order_rng",
            True,
        ),

        (
            "no_optimizer_instantiation",
            True,
        ),

        (
            "no_forward_computation",
            True,
        ),

        (
            "no_backward_computation",
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
                "check": (
                    name
                ),

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
            "At least one corrected Phase-5.3.1g "
            "forward-provenance audit invariant failed"
        ),
    )

    print(
        invariant_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Write audit outputs
    # =========================================================================

    banner(
        "WRITE PHASE-5.3.1g AUDIT OUTPUTS"
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ranked_script_df.to_csv(
        SCRIPT_SUMMARY_PATH,
        index=False,
    )

    class_df.to_csv(
        CLASS_INVENTORY_PATH,
        index=False,
    )

    method_df.to_csv(
        METHOD_INVENTORY_PATH,
        index=False,
    )

    forward_df.to_csv(
        FORWARD_INVENTORY_PATH,
        index=False,
    )

    forward_call_df.to_csv(
        FORWARD_CALL_PATH,
        index=False,
    )

    forward_return_df.to_csv(
        FORWARD_RETURN_PATH,
        index=False,
    )

    model_candidate_df.to_csv(
        MODEL_FORWARD_CANDIDATE_PATH,
        index=False,
    )

    comparison_df.to_csv(
        COMMON_CLASS_COMPARISON_PATH,
        index=False,
    )

    invariant_df.to_csv(
        FINAL_INVARIANT_PATH,
        index=False,
    )

    manifest = {
        "phase": (
            "5.3.1g"
        ),

        "title": (
            "Canonical Forward Implementation Provenance Audit"
        ),

        "status": (
            "AUDIT_COMPLETE_"
            "FORWARD_IMPLEMENTATION_PROVENANCE_NOT_YET_FROZEN"
        ),

        "correction": {
            "previous_false_assumption": (
                "Every canonical Phase-4.7.1b neural class "
                "must implement forward()."
            ),

            "observed_counterexample": (
                "DescriptionEncoder.forward absent from "
                "Phase-4.7.1b initialization-freeze source."
            ),

            "interpretation": (
                "Initialization provenance and forward-runtime "
                "provenance must be resolved independently."
            ),
        },

        "frozen_initialization_provenance": {
            "source": (
                str(
                    CANONICAL_INITIALIZATION_SOURCE_PATH
                )
            ),

            "source_sha256": (
                initialization_source_hash
            ),

            "sanitized_runtime_ast_sha256": (
                FROZEN_RUNTIME_AST_SHA256
            ),

            "canonical_initial_state_sha256": (
                CANONICAL_INITIAL_STATE_SHA256
            ),
        },

        "forward_provenance_audit": {
            "phase4_script_count": (
                len(
                    script_paths
                )
            ),

            "parsed_phase4_script_count": (
                parsed_script_count
            ),

            "forward_method_count": (
                len(
                    forward_df
                )
            ),

            "model_level_forward_candidate_count": (
                len(
                    model_candidate_df
                )
            ),

            "highest_evidence_score": (
                highest_score
            ),

            "strongest_candidate_count": (
                len(
                    strongest
                )
            ),

            "unique_strongest_candidate": (
                strongest_script
            ),

            "forward_implementation_frozen": (
                False
            ),
        },

        "phase4_6_2": {
            "path": (
                str(
                    phase_4_6_2_path
                )
            ),

            "sha256": (
                str(
                    phase_4_6_2_script_row[
                        "script_sha256"
                    ]
                )
            ),

            "forward_method_count": int(
                phase_4_6_2_script_row[
                    "forward_method_count"
                ]
            ),

            "model_level_forward_count": int(
                phase_4_6_2_script_row[
                    "model_level_forward_count"
                ]
            ),

            "canonical_branch_forwards_present": int(
                phase_4_6_2_script_row[
                    "canonical_branch_forwards_present"
                ]
            ),

            "contains_BCEWithLogitsLoss": bool(
                phase_4_6_2_script_row[
                    "contains_BCEWithLogitsLoss"
                ]
            ),
        },

        "training_boundary": {
            "phase4_module_imported": (
                False
            ),

            "model_instantiated": (
                False
            ),

            "training_negative_rng_instantiated": (
                False
            ),

            "training_order_rng_instantiated": (
                False
            ),

            "optimizer_instantiated": (
                False
            ),

            "forward_computation_performed": (
                False
            ),

            "backward_computation_performed": (
                False
            ),

            "optimizer_steps": (
                0
            ),
        },

        "next_phase_requirement": (
            "Review the strongest forward-runtime candidate, "
            "freeze its exact model-level forward signature/AST and "
            "its required branch-method implementations, and prove "
            "that they are compatible with the already-frozen "
            "Phase-4.7.1b parameter namespace and initialization state."
        ),

        "outputs": [
            str(
                SCRIPT_SUMMARY_PATH
            ),

            str(
                CLASS_INVENTORY_PATH
            ),

            str(
                METHOD_INVENTORY_PATH
            ),

            str(
                FORWARD_INVENTORY_PATH
            ),

            str(
                FORWARD_CALL_PATH
            ),

            str(
                FORWARD_RETURN_PATH
            ),

            str(
                MODEL_FORWARD_CANDIDATE_PATH
            ),

            str(
                COMMON_CLASS_COMPARISON_PATH
            ),

            str(
                FINAL_INVARIANT_PATH
            ),
        ],
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    for path in (
        SCRIPT_SUMMARY_PATH,
        CLASS_INVENTORY_PATH,
        METHOD_INVENTORY_PATH,
        FORWARD_INVENTORY_PATH,
        FORWARD_CALL_PATH,
        FORWARD_RETURN_PATH,
        MODEL_FORWARD_CANDIDATE_PATH,
        COMMON_CLASS_COMPARISON_PATH,
        FINAL_INVARIANT_PATH,
        MANIFEST_PATH,
    ):

        print(
            f"WROTE  {path}"
        )

    # =========================================================================
    # Final status
    # =========================================================================

    banner(
        "PHASE 5.3.1g FINAL STATUS"
    )

    print(
        "Canonical initialization provenance:  UNCHANGED / FROZEN"
    )

    print(
        "Canonical initialization source:"
    )

    print(
        f"  {CANONICAL_INITIALIZATION_SOURCE_PATH}"
    )

    print()

    print(
        "Forward provenance audit:              COMPLETE"
    )

    print(
        f"Forward methods discovered:           "
        f"{len(forward_df)}"
    )

    print(
        f"Model-level forward candidates:       "
        f"{len(model_candidate_df)}"
    )

    print()

    if (
        unique_strongest
    ):

        print(
            "Unique strongest forward candidate:"
        )

        print(
            f"  {strongest_script}"
        )

    else:

        print(
            "Unique strongest forward candidate: NO"
        )

    print()

    print(
        "Forward implementation frozen:        NO"
    )

    print()

    print(
        "Phase-4 module imported:              NO"
    )

    print(
        "Model instantiated:                   NO"
    )

    print(
        "Training-negative RNG instantiated:  NO"
    )

    print(
        "Training-order RNG instantiated:     NO"
    )

    print(
        "Optimizer instantiated:               NO"
    )

    print(
        "Forward computation performed:        NO"
    )

    print(
        "Backward computation performed:       NO"
    )

    print(
        "Optimizer steps:                      0"
    )

    banner(
        "PHASE 5.3.1g COMPLETE / "
        "FORWARD PROVENANCE EVIDENCE COLLECTED — "
        "NOT YET FROZEN"
    )


if __name__ == "__main__":
    main()