#!/usr/bin/env python3
"""
Phase 5.3.1i.1 — Missing Canonical self.* Dependency Classification Audit

STATIC AUDIT ONLY.

Background
----------
Phase 5.3.1i established a REAL compatibility failure:

    Methods audited:                    7 / 7
    Runtime globals missing:            0
    Canonical self dependencies:        FAIL
    BasisRGCNLayer compatibility:       FAIL

We must NOT patch this blindly.

Purpose
-------
For every self.<attribute> consumed by the exact Phase-4.6.2
runtime methods but absent from the canonical Phase-4.7.1b
constructor topology, determine:

1. the exact missing attribute;
2. which method consumes it;
3. whether Phase-4.6.2 __init__ creates it;
4. the exact Phase-4.6.2 constructor expression;
5. whether it is:
       - immutable runtime metadata;
       - a parameter;
       - a buffer;
       - a submodule;
       - executable/stateful runtime construction;
       - unresolved.

This decides whether a parameter-state-neutral runtime metadata bridge
is even permissible.

THIS SCRIPT DOES NOT:
- import either Phase-4 script;
- execute either Phase-4 script;
- instantiate a model;
- attach forward methods;
- construct missing attributes;
- instantiate RNG;
- instantiate Adam;
- perform forward/backward;
- call optimizer.step();
- modify any Phase-4 artifact.

No compatibility decision is forced to PASS.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pandas as pd


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


# =============================================================================
# Exact proposed numerical-method provenance
# =============================================================================

METHOD_GRAFT_SPEC = {
    "DescriptionEncoder": [
        "forward",
    ],

    "TrendExtractor": [
        "attend_period",
        "encode_sequence",
    ],

    "BasisRGCNLayer": [
        "effective_weight",
        "forward",
    ],

    "PreferencePropagation": [
        "forward",
    ],

    "ScoringMLP": [
        "forward",
    ],
}


# =============================================================================
# Prior evidence
# =============================================================================

PHASE_5_3_1H_MANIFEST_PATH = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_1h/"
    "phase_5_3_1h_procedural_forward_manifest.json"
)

PHASE_5_3_1I_MANIFEST_PATH = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_1i/"
    "phase_5_3_1i_method_compatibility_manifest.json"
)


# =============================================================================
# Outputs
# =============================================================================

OUT_DIR = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_1i_1"
)

CLASS_ATTRIBUTE_COMPARISON_PATH = (
    OUT_DIR
    / "class_constructor_attribute_comparison.csv"
)

METHOD_MISSING_ATTRIBUTE_PATH = (
    OUT_DIR
    / "method_missing_self_attribute_summary.csv"
)

MISSING_ATTRIBUTE_CLASSIFICATION_PATH = (
    OUT_DIR
    / "missing_self_attribute_classification.csv"
)

BASIS_RGCN_DETAIL_PATH = (
    OUT_DIR
    / "BasisRGCNLayer_missing_runtime_state_detail.csv"
)

FINAL_INVARIANT_PATH = (
    OUT_DIR
    / "phase_5_3_1i_1_audit_invariants.csv"
)

MANIFEST_PATH = (
    OUT_DIR
    / "phase_5_3_1i_1_missing_dependency_manifest.json"
)


# =============================================================================
# Helpers
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


def sha256_file(
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


def ast_sha256(
    node: ast.AST,
) -> str:

    payload = ast.dump(
        node,
        annotate_fields=True,
        include_attributes=False,
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def load_json(
    path: Path,
) -> dict:

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        return json.load(handle)


def read_text(
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

        return node.id

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

        return node.attr

    return ""


def node_source(
    source: str,
    node: ast.AST,
) -> str:

    value = ast.get_source_segment(
        source,
        node,
    )

    if value is not None:

        return value

    return ast.unparse(node)


def compact(
    value: str,
    limit: int = 1000,
) -> str:

    value = " ".join(
        value.split()
    )

    if len(value) <= limit:

        return value

    return (
        value[:limit]
        + " ..."
    )


# =============================================================================
# Class/method utilities
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
) -> ast.FunctionDef | None:

    matches = [
        node
        for node in class_node.body
        if (
            isinstance(
                node,
                ast.FunctionDef,
            )
            and node.name == method_name
        )
    ]

    require(
        len(matches) <= 1,
        (
            f"Multiple {class_node.name}.{method_name} "
            "definitions found."
        ),
    )

    if not matches:

        return None

    return matches[0]


# =============================================================================
# self.* utilities
# =============================================================================

def direct_self_attribute(
    target: ast.AST,
) -> str | None:
    """
    Match exactly:

        self.foo

    rather than nested:
        self.foo.bar
    """

    if not isinstance(
        target,
        ast.Attribute,
    ):

        return None

    if not (
        isinstance(
            target.value,
            ast.Name,
        )
        and target.value.id == "self"
    ):

        return None

    return target.attr


def self_root_attribute(
    node: ast.Attribute,
) -> str | None:

    current = node

    path = []

    while isinstance(
        current,
        ast.Attribute,
    ):

        path.append(
            current.attr
        )

        current = current.value

    if not (
        isinstance(
            current,
            ast.Name,
        )
        and current.id == "self"
    ):

        return None

    if not path:

        return None

    path.reverse()

    return path[0]


def loaded_self_attributes(
    method: ast.FunctionDef,
) -> set[str]:

    result = set()

    for node in ast.walk(
        method
    ):

        if not (
            isinstance(
                node,
                ast.Attribute,
            )
            and isinstance(
                node.ctx,
                ast.Load,
            )
        ):

            continue

        root = self_root_attribute(
            node
        )

        if root is not None:

            result.add(root)

    return result


# =============================================================================
# Constructor attribute extraction
# =============================================================================

def constructor_attribute_inventory(
    init_method: ast.FunctionDef,
) -> dict[str, dict]:
    """
    Map self.<attribute> created by __init__ to its static provenance.

    Includes:
        self.foo = expression
        self.register_parameter("foo", ...)
        self.register_buffer("foo", ...)
        self.add_module("foo", ...)
    """

    result = {}

    for node in ast.walk(
        init_method
    ):

        # ---------------------------------------------------------------------
        # self.foo = expression
        # ---------------------------------------------------------------------

        if isinstance(
            node,
            ast.Assign,
        ):

            for target in node.targets:

                attribute = (
                    direct_self_attribute(
                        target
                    )
                )

                if attribute is None:
                    continue

                result[
                    attribute
                ] = {
                    "creation_kind": (
                        "ASSIGN"
                    ),

                    "expression_node": (
                        node.value
                    ),

                    "expression": (
                        ast.unparse(
                            node.value
                        )
                    ),

                    "statement_ast_sha256": (
                        ast_sha256(node)
                    ),

                    "line_number": (
                        int(node.lineno)
                    ),
                }

        elif isinstance(
            node,
            ast.AnnAssign,
        ):

            attribute = (
                direct_self_attribute(
                    node.target
                )
            )

            if (
                attribute is not None
                and node.value is not None
            ):

                result[
                    attribute
                ] = {
                    "creation_kind": (
                        "ANN_ASSIGN"
                    ),

                    "expression_node": (
                        node.value
                    ),

                    "expression": (
                        ast.unparse(
                            node.value
                        )
                    ),

                    "statement_ast_sha256": (
                        ast_sha256(node)
                    ),

                    "line_number": (
                        int(node.lineno)
                    ),
                }

        # ---------------------------------------------------------------------
        # register_parameter / register_buffer / add_module
        # ---------------------------------------------------------------------

        elif isinstance(
            node,
            ast.Call,
        ):

            call_name = (
                dotted_name(
                    node.func
                )
            )

            creation_kind = {
                "self.register_parameter": (
                    "REGISTER_PARAMETER"
                ),

                "self.register_buffer": (
                    "REGISTER_BUFFER"
                ),

                "self.add_module": (
                    "ADD_MODULE"
                ),
            }.get(
                call_name
            )

            if creation_kind is None:
                continue

            if not (
                len(node.args) >= 1
                and isinstance(
                    node.args[0],
                    ast.Constant,
                )
                and isinstance(
                    node.args[0].value,
                    str,
                )
            ):

                continue

            attribute = (
                node.args[0].value
            )

            expression_node = (
                node.args[1]
                if len(node.args) >= 2
                else None
            )

            result[
                attribute
            ] = {
                "creation_kind": (
                    creation_kind
                ),

                "expression_node": (
                    expression_node
                ),

                "expression": (
                    ast.unparse(
                        expression_node
                    )
                    if expression_node
                    is not None
                    else None
                ),

                "statement_ast_sha256": (
                    ast_sha256(node)
                ),

                "line_number": (
                    int(node.lineno)
                ),
            }

    return result


# =============================================================================
# Constructor argument / global-symbol resolution
# =============================================================================

def function_argument_names(
    function: ast.FunctionDef,
) -> set[str]:

    args = function.args

    result = {
        item.arg
        for item
        in (
            list(args.posonlyargs)
            + list(args.args)
            + list(args.kwonlyargs)
        )
    }

    if args.vararg is not None:

        result.add(
            args.vararg.arg
        )

    if args.kwarg is not None:

        result.add(
            args.kwarg.arg
        )

    result.discard(
        "self"
    )

    return result


def top_level_literal_names(
    tree: ast.Module,
) -> set[str]:

    result = set()

    for node in tree.body:

        if isinstance(
            node,
            ast.Assign,
        ):

            try:

                ast.literal_eval(
                    node.value
                )

            except Exception:

                continue

            for target in node.targets:

                if isinstance(
                    target,
                    ast.Name,
                ):

                    result.add(
                        target.id
                    )

        elif isinstance(
            node,
            ast.AnnAssign,
        ):

            if not (
                isinstance(
                    node.target,
                    ast.Name,
                )
                and node.value
                is not None
            ):

                continue

            try:

                ast.literal_eval(
                    node.value
                )

            except Exception:

                continue

            result.add(
                node.target.id
            )

    return result


def loaded_expression_names(
    node: ast.AST | None,
) -> set[str]:

    if node is None:

        return set()

    return {
        candidate.id
        for candidate
        in ast.walk(node)
        if (
            isinstance(
                candidate,
                ast.Name,
            )
            and isinstance(
                candidate.ctx,
                ast.Load,
            )
        )
    }


def expression_calls(
    node: ast.AST | None,
) -> list[str]:

    if node is None:

        return []

    return [
        dotted_name(
            candidate.func
        )
        for candidate
        in ast.walk(node)
        if isinstance(
            candidate,
            ast.Call,
        )
    ]


# =============================================================================
# Attribute classification
# =============================================================================

STATE_PARAMETER_MARKERS = (
    "Parameter",
    "nn.Parameter",
    "torch.nn.Parameter",
)

MODULE_PREFIXES = (
    "nn.Linear",
    "nn.GRU",
    "nn.Embedding",
    "nn.ModuleList",
    "nn.Sequential",
    "BasisRGCNLayer",
)


def classify_missing_attribute(
    attribute: str,
    forward_record: dict | None,
    forward_constructor_args: set[str],
    forward_literal_names: set[str],
) -> tuple[str, bool, str]:
    """
    Returns:
        classification
        metadata_bridge_candidate
        explanation
    """

    if forward_record is None:

        return (
            "MISSING_FROM_FORWARD_CONSTRUCTOR",
            False,
            (
                "The forward method consumes this self attribute, "
                "but it is not constructed by Phase-4.6.2 __init__."
            ),
        )

    creation_kind = (
        forward_record[
            "creation_kind"
        ]
    )

    expression = (
        forward_record[
            "expression"
        ]
        or ""
    )

    expression_node = (
        forward_record[
            "expression_node"
        ]
    )

    # -------------------------------------------------------------------------
    # Explicit state registration
    # -------------------------------------------------------------------------

    if creation_kind == "REGISTER_PARAMETER":

        return (
            "STATE_BEARING_PARAMETER",
            False,
            (
                "Explicit register_parameter() state."
            ),
        )

    if creation_kind == "REGISTER_BUFFER":

        return (
            "STATE_BEARING_BUFFER",
            False,
            (
                "Explicit register_buffer() state."
            ),
        )

    if creation_kind == "ADD_MODULE":

        return (
            "STATE_BEARING_MODULE",
            False,
            (
                "Explicit add_module() state."
            ),
        )

    # -------------------------------------------------------------------------
    # Assignment expression
    # -------------------------------------------------------------------------

    calls = (
        expression_calls(
            expression_node
        )
    )

    for marker in (
        STATE_PARAMETER_MARKERS
    ):

        if marker in expression:

            return (
                "STATE_BEARING_PARAMETER",
                False,
                (
                    f"Assignment constructs {marker}."
                ),
            )

    for module_prefix in (
        MODULE_PREFIXES
    ):

        if (
            expression.startswith(
                module_prefix
            )
            or module_prefix
            in calls
        ):

            return (
                "STATE_BEARING_MODULE",
                False,
                (
                    "Assignment constructs an nn.Module/"
                    "canonical neural submodule."
                ),
            )

    # -------------------------------------------------------------------------
    # Pure literal
    # -------------------------------------------------------------------------

    try:

        literal_value = (
            ast.literal_eval(
                expression_node
            )
        )

        return (
            "PURE_LITERAL_METADATA",
            True,
            (
                "Pure literal runtime metadata: "
                f"{repr(literal_value)}"
            ),
        )

    except Exception:
        pass

    # -------------------------------------------------------------------------
    # Pure expression based only on constructor args / frozen literals
    # -------------------------------------------------------------------------

    loaded_names = (
        loaded_expression_names(
            expression_node
        )
    )

    permitted_names = (
        forward_constructor_args
        | forward_literal_names
    )

    if (
        not calls
        and loaded_names.issubset(
            permitted_names
        )
    ):

        return (
            "PURE_DERIVED_METADATA",
            True,
            (
                "Stateless expression derived only from "
                "constructor arguments and/or top-level "
                "literal constants."
            ),
        )

    # -------------------------------------------------------------------------
    # Calls imply runtime construction, not safe metadata by default
    # -------------------------------------------------------------------------

    if calls:

        return (
            "EXECUTABLE_EXPRESSION",
            False,
            (
                "Constructor expression invokes: "
                + ", ".join(
                    calls
                )
            ),
        )

    return (
        "UNRESOLVED_EXPRESSION",
        False,
        (
            "Expression is not obviously state-bearing, "
            "but cannot be proven to be pure metadata."
        ),
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.3.1i.1 — "
        "MISSING CANONICAL self.* DEPENDENCY CLASSIFICATION AUDIT"
    )

    print(
        "Phase-4 source imported:              NO"
    )

    print(
        "Runtime executed:                     NO"
    )

    print(
        "Methods grafted:                      NO"
    )

    print(
        "Missing attributes constructed:       NO"
    )

    print(
        "Model instantiated:                   NO"
    )

    print(
        "Optimizer instantiated:               NO"
    )

    print(
        "Forward computation:                  NO"
    )

    print(
        "Backward computation:                 NO"
    )

    print(
        "Optimizer steps:                      0"
    )

    # =========================================================================
    # Integrity
    # =========================================================================

    banner(
        "SOURCE INTEGRITY RECHECK"
    )

    for path in (
        CANONICAL_SOURCE_PATH,
        FORWARD_SOURCE_PATH,
        PHASE_5_3_1H_MANIFEST_PATH,
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

    canonical_hash = (
        sha256_file(
            CANONICAL_SOURCE_PATH
        )
    )

    forward_hash = (
        sha256_file(
            FORWARD_SOURCE_PATH
        )
    )

    require(
        canonical_hash
        == CANONICAL_SOURCE_SHA256,
        (
            "Canonical Phase-4.7.1b source changed."
        ),
    )

    require(
        forward_hash
        == FORWARD_SOURCE_SHA256,
        (
            "Phase-4.6.2 source changed."
        ),
    )

    phase_5_3_1h = (
        load_json(
            PHASE_5_3_1H_MANIFEST_PATH
        )
    )

    require(
        phase_5_3_1h[
            "status"
        ]
        == (
            "AUDIT_COMPLETE_"
            "PROCEDURAL_FORWARD_ORCHESTRATION_NOT_YET_FROZEN"
        ),
        (
            "Unexpected Phase-5.3.1h status."
        ),
    )

    print(
        "Canonical source SHA256:              PASS"
    )

    print(
        "Forward source SHA256:                PASS"
    )

    # Optional: record prior failed 5.3.1i status.
    previous_i_status = None

    if (
        PHASE_5_3_1I_MANIFEST_PATH.exists()
    ):

        previous_i = (
            load_json(
                PHASE_5_3_1I_MANIFEST_PATH
            )
        )

        previous_i_status = (
            previous_i.get(
                "status"
            )
        )

        print(
            f"Prior Phase-5.3.1i status:           "
            f"{previous_i_status}"
        )

    # =========================================================================
    # Parse exact sources
    # =========================================================================

    canonical_source = (
        read_text(
            CANONICAL_SOURCE_PATH
        )
    )

    forward_source = (
        read_text(
            FORWARD_SOURCE_PATH
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

    canonical_classes = (
        class_map(
            canonical_tree
        )
    )

    forward_classes = (
        class_map(
            forward_tree
        )
    )

    forward_literal_names = (
        top_level_literal_names(
            forward_tree
        )
    )

    # =========================================================================
    # Constructor attribute comparison
    # =========================================================================

    banner(
        "CONSTRUCTOR ATTRIBUTE COMPARISON"
    )

    comparison_rows = []

    canonical_constructor_attributes = {}
    forward_constructor_attributes = {}
    forward_constructor_arguments = {}

    for class_name in (
        METHOD_GRAFT_SPEC
    ):

        require(
            class_name
            in canonical_classes,
            (
                f"Canonical class missing: {class_name}"
            ),
        )

        require(
            class_name
            in forward_classes,
            (
                f"Forward class missing: {class_name}"
            ),
        )

        canonical_init = (
            direct_method(
                canonical_classes[
                    class_name
                ],
                "__init__",
            )
        )

        forward_init = (
            direct_method(
                forward_classes[
                    class_name
                ],
                "__init__",
            )
        )

        require(
            canonical_init
            is not None,
            (
                f"Canonical {class_name} has no __init__."
            ),
        )

        require(
            forward_init
            is not None,
            (
                f"Forward {class_name} has no __init__."
            ),
        )

        canonical_attributes = (
            constructor_attribute_inventory(
                canonical_init
            )
        )

        forward_attributes = (
            constructor_attribute_inventory(
                forward_init
            )
        )

        canonical_constructor_attributes[
            class_name
        ] = canonical_attributes

        forward_constructor_attributes[
            class_name
        ] = forward_attributes

        forward_constructor_arguments[
            class_name
        ] = function_argument_names(
            forward_init
        )

        union = (
            set(
                canonical_attributes
            )
            | set(
                forward_attributes
            )
        )

        for attribute in sorted(
            union
        ):

            comparison_rows.append(
                {
                    "class_name": (
                        class_name
                    ),

                    "attribute": (
                        attribute
                    ),

                    "canonical_present": (
                        attribute
                        in canonical_attributes
                    ),

                    "forward_present": (
                        attribute
                        in forward_attributes
                    ),

                    "canonical_creation_kind": (
                        canonical_attributes[
                            attribute
                        ][
                            "creation_kind"
                        ]
                        if attribute
                        in canonical_attributes
                        else None
                    ),

                    "forward_creation_kind": (
                        forward_attributes[
                            attribute
                        ][
                            "creation_kind"
                        ]
                        if attribute
                        in forward_attributes
                        else None
                    ),

                    "canonical_expression": (
                        canonical_attributes[
                            attribute
                        ][
                            "expression"
                        ]
                        if attribute
                        in canonical_attributes
                        else None
                    ),

                    "forward_expression": (
                        forward_attributes[
                            attribute
                        ][
                            "expression"
                        ]
                        if attribute
                        in forward_attributes
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
    # Missing method dependencies
    # =========================================================================

    banner(
        "EXACT MISSING self.* DEPENDENCIES"
    )

    method_missing_rows = []
    classification_rows = []

    for (
        class_name,
        method_names,
    ) in METHOD_GRAFT_SPEC.items():

        canonical_attribute_names = set(
            canonical_constructor_attributes[
                class_name
            ]
        )

        forward_attribute_map = (
            forward_constructor_attributes[
                class_name
            ]
        )

        grafted_methods = set(
            method_names
        )

        for method_name in (
            method_names
        ):

            method = (
                direct_method(
                    forward_classes[
                        class_name
                    ],
                    method_name,
                )
            )

            require(
                method
                is not None,
                (
                    f"Missing runtime method: "
                    f"{class_name}.{method_name}"
                ),
            )

            required_self = (
                loaded_self_attributes(
                    method
                )
            )

            available = (
                canonical_attribute_names
                | grafted_methods
            )

            missing = (
                required_self
                - available
            )

            method_missing_rows.append(
                {
                    "class_name": (
                        class_name
                    ),

                    "method_name": (
                        method_name
                    ),

                    "method_ast_sha256": (
                        ast_sha256(
                            method
                        )
                    ),

                    "required_self_attributes": (
                        ";".join(
                            sorted(
                                required_self
                            )
                        )
                    ),

                    "canonical_constructor_attributes": (
                        ";".join(
                            sorted(
                                canonical_attribute_names
                            )
                        )
                    ),

                    "missing_self_attributes": (
                        ";".join(
                            sorted(
                                missing
                            )
                        )
                    ),

                    "missing_count": (
                        len(
                            missing
                        )
                    ),
                }
            )

            for attribute in sorted(
                missing
            ):

                forward_record = (
                    forward_attribute_map.get(
                        attribute
                    )
                )

                (
                    classification,
                    metadata_bridge_candidate,
                    explanation,
                ) = classify_missing_attribute(
                    attribute=attribute,
                    forward_record=forward_record,
                    forward_constructor_args=(
                        forward_constructor_arguments[
                            class_name
                        ]
                    ),
                    forward_literal_names=(
                        forward_literal_names
                    ),
                )

                classification_rows.append(
                    {
                        "class_name": (
                            class_name
                        ),

                        "method_name": (
                            method_name
                        ),

                        "missing_attribute": (
                            attribute
                        ),

                        "forward_constructor_defines_attribute": (
                            forward_record
                            is not None
                        ),

                        "forward_creation_kind": (
                            forward_record[
                                "creation_kind"
                            ]
                            if forward_record
                            is not None
                            else None
                        ),

                        "forward_expression": (
                            forward_record[
                                "expression"
                            ]
                            if forward_record
                            is not None
                            else None
                        ),

                        "forward_assignment_line": (
                            forward_record[
                                "line_number"
                            ]
                            if forward_record
                            is not None
                            else None
                        ),

                        "classification": (
                            classification
                        ),

                        "metadata_bridge_candidate": (
                            metadata_bridge_candidate
                        ),

                        "explanation": (
                            explanation
                        ),
                    }
                )

    method_missing_df = pd.DataFrame(
        method_missing_rows
    )

    classification_df = pd.DataFrame(
        classification_rows
    )

    print(
        method_missing_df[
            [
                "class_name",
                "method_name",
                "missing_self_attributes",
                "missing_count",
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # Classification
    # =========================================================================

    banner(
        "MISSING ATTRIBUTE CLASSIFICATION"
    )

    require(
        not classification_df.empty,
        (
            "No missing self dependencies were recovered, "
            "which contradicts the reported Phase-5.3.1i failure."
        ),
    )

    print(
        classification_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # BasisRGCN focus
    # =========================================================================

    banner(
        "BasisRGCNLayer — EXACT MISSING STATE"
    )

    basis_df = (
        classification_df.loc[
            classification_df[
                "class_name"
            ]
            == "BasisRGCNLayer"
        ].copy()
    )

    require(
        not basis_df.empty,
        (
            "No BasisRGCNLayer missing dependencies recovered, "
            "contradicting the reported Phase-5.3.1i result."
        ),
    )

    print(
        basis_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Decision classification
    # =========================================================================

    banner(
        "DECISION CLASSIFICATION"
    )

    unique_missing = (
        classification_df[
            [
                "class_name",
                "missing_attribute",
                "classification",
                "metadata_bridge_candidate",
                "forward_creation_kind",
                "forward_expression",
            ]
        ]
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )

    all_metadata_bridgeable = bool(
        unique_missing[
            "metadata_bridge_candidate"
        ].all()
    )

    state_bearing = (
        unique_missing.loc[
            unique_missing[
                "classification"
            ].isin(
                [
                    "STATE_BEARING_PARAMETER",
                    "STATE_BEARING_BUFFER",
                    "STATE_BEARING_MODULE",
                ]
            )
        ].copy()
    )

    unresolved = (
        unique_missing.loc[
            unique_missing[
                "classification"
            ].isin(
                [
                    "EXECUTABLE_EXPRESSION",
                    "UNRESOLVED_EXPRESSION",
                    "MISSING_FROM_FORWARD_CONSTRUCTOR",
                ]
            )
        ].copy()
    )

    print(
        f"Unique missing attributes:            "
        f"{len(unique_missing)}"
    )

    print(
        f"State-bearing missing attributes:     "
        f"{len(state_bearing)}"
    )

    print(
        f"Unresolved missing attributes:        "
        f"{len(unresolved)}"
    )

    print(
        f"All missing attrs pure metadata:      "
        f"{all_metadata_bridgeable}"
    )

    print()

    if all_metadata_bridgeable:

        decision = (
            "RUNTIME_METADATA_BRIDGE_CANDIDATE"
        )

        print(
            "Interpretation:"
        )

        print(
            "  Every missing self.* dependency is stateless "
            "constructor metadata."
        )

        print(
            "  A Phase-5 runtime metadata bridge MAY be audited "
            "next, but is NOT frozen here."
        )

    else:

        decision = (
            "CANONICAL_FORWARD_COMPOSITION_REMAINS_BLOCKED"
        )

        print(
            "Interpretation:"
        )

        print(
            "  At least one missing self.* dependency is state-bearing "
            "or cannot yet be proven parameter-state-neutral."
        )

        print(
            "  Forward-method composition remains BLOCKED."
        )

    # =========================================================================
    # Audit invariants
    # =========================================================================

    banner(
        "PHASE-5.3.1i.1 AUDIT INVARIANTS"
    )

    checks = [
        (
            "canonical_source_sha256_exact",
            (
                canonical_hash
                == CANONICAL_SOURCE_SHA256
            ),
        ),

        (
            "forward_source_sha256_exact",
            (
                forward_hash
                == FORWARD_SOURCE_SHA256
            ),
        ),

        (
            "phase_5_3_1h_complete",
            (
                phase_5_3_1h[
                    "status"
                ]
                == (
                    "AUDIT_COMPLETE_"
                    "PROCEDURAL_FORWARD_ORCHESTRATION_NOT_YET_FROZEN"
                )
            ),
        ),

        (
            "seven_runtime_methods_reaudited",
            (
                len(
                    method_missing_df
                )
                == 7
            ),
        ),

        (
            "missing_self_dependency_reproduced",
            (
                int(
                    method_missing_df[
                        "missing_count"
                    ].sum()
                )
                > 0
            ),
        ),

        (
            "BasisRGCNLayer_missing_dependency_reproduced",
            (
                len(
                    basis_df
                )
                > 0
            ),
        ),

        (
            "no_phase4_import",
            True,
        ),

        (
            "no_runtime_execution",
            True,
        ),

        (
            "no_method_graft",
            True,
        ),

        (
            "no_missing_attribute_construction",
            True,
        ),

        (
            "no_model_instantiation",
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
                "check": name,
                "result": (
                    "PASS"
                    if result
                    else "FAIL"
                ),
            }
            for (
                name,
                result,
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
            "Phase-5.3.1i.1 diagnostic audit itself failed."
        ),
    )

    print(
        invariant_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Write
    # =========================================================================

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_df.to_csv(
        CLASS_ATTRIBUTE_COMPARISON_PATH,
        index=False,
    )

    method_missing_df.to_csv(
        METHOD_MISSING_ATTRIBUTE_PATH,
        index=False,
    )

    classification_df.to_csv(
        MISSING_ATTRIBUTE_CLASSIFICATION_PATH,
        index=False,
    )

    basis_df.to_csv(
        BASIS_RGCN_DETAIL_PATH,
        index=False,
    )

    invariant_df.to_csv(
        FINAL_INVARIANT_PATH,
        index=False,
    )

    manifest = {
        "phase": (
            "5.3.1i.1"
        ),

        "title": (
            "Missing Canonical self.* "
            "Dependency Classification Audit"
        ),

        "status": (
            "AUDIT_COMPLETE"
        ),

        "decision": (
            decision
        ),

        "previous_phase_5_3_1i_status": (
            previous_i_status
        ),

        "missing_dependencies": {
            "unique_missing_attribute_count": (
                len(
                    unique_missing
                )
            ),

            "state_bearing_missing_count": (
                len(
                    state_bearing
                )
            ),

            "unresolved_missing_count": (
                len(
                    unresolved
                )
            ),

            "all_missing_attributes_metadata_bridgeable": (
                all_metadata_bridgeable
            ),
        },

        "training_boundary": {
            "phase4_imported": False,
            "runtime_executed": False,
            "methods_grafted": False,
            "missing_attributes_constructed": False,
            "model_instantiated": False,
            "optimizer_instantiated": False,
            "forward_performed": False,
            "backward_performed": False,
            "optimizer_steps": 0,
        },

        "next_step": (
            "If and only if every missing attribute is proven "
            "PURE_LITERAL_METADATA or PURE_DERIVED_METADATA, "
            "audit a parameter-state-neutral runtime metadata bridge. "
            "Otherwise stop forward composition and reconcile the "
            "Phase-4 implementation inconsistency."
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

    for path in (
        CLASS_ATTRIBUTE_COMPARISON_PATH,
        METHOD_MISSING_ATTRIBUTE_PATH,
        MISSING_ATTRIBUTE_CLASSIFICATION_PATH,
        BASIS_RGCN_DETAIL_PATH,
        FINAL_INVARIANT_PATH,
        MANIFEST_PATH,
    ):

        print(
            f"WROTE  {path}"
        )

    # =========================================================================
    # Final
    # =========================================================================

    banner(
        "PHASE 5.3.1i.1 FINAL STATUS"
    )

    print(
        f"Unique missing attributes:            "
        f"{len(unique_missing)}"
    )

    print(
        f"State-bearing missing attributes:     "
        f"{len(state_bearing)}"
    )

    print(
        f"Unresolved missing attributes:        "
        f"{len(unresolved)}"
    )

    print(
        f"All missing attributes metadata-only: "
        f"{all_metadata_bridgeable}"
    )

    print()

    print(
        f"Decision:                             "
        f"{decision}"
    )

    print()

    print(
        "Methods grafted:                      NO"
    )

    print(
        "Missing attrs constructed:            NO"
    )

    print(
        "Model instantiated:                   NO"
    )

    print(
        "Optimizer instantiated:               NO"
    )

    print(
        "Forward computation:                  NO"
    )

    print(
        "Backward computation:                 NO"
    )

    print(
        "Optimizer steps:                      0"
    )

    banner(
        "PHASE 5.3.1i.1 COMPLETE / "
        "MISSING SELF-STATE CLASSIFIED"
    )


if __name__ == "__main__":
    main()