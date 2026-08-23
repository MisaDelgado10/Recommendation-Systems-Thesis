#!/usr/bin/env python3
"""
Phase 5.3.1i — Forward-Method / Canonical-Topology Compatibility Audit
CORRECTED RERUN

STATIC AUDIT ONLY.

Correction
----------
The first 5.3.1i audit incorrectly required every global symbol used
by a Phase-4.6.2 numerical method to already exist in the Phase-4.7.1b
initialization source.

That is too strict because Phase-4.7.1b is authoritative for model
topology / parameter state / initialization, not necessarily for
runtime-only imports or stateless helpers.

This corrected audit separates:

HARD MODEL-STATE COMPATIBILITY
    Every self.<attribute> consumed by the Phase-4.6.2 numerical
    methods must be supplied by the canonical Phase-4.7.1b constructor
    or by another explicitly grafted method.

RUNTIME NAMESPACE COMPATIBILITY
    A non-self global may be supplied by:
        1. canonical Phase-4.7.1b runtime namespace;
        2. an exact import from Phase-4.6.2;
        3. an exact stateless helper function from Phase-4.6.2;
        4. an exact top-level literal constant from Phase-4.6.2.

    Unknown/missing globals remain a hard failure.

NO method graft is executed.
NO model is instantiated.
NO optimizer/RNG/forward/backward is executed.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import json
from pathlib import Path

import pandas as pd


# =============================================================================
# Frozen sources
# =============================================================================

INITIALIZATION_SOURCE_PATH = Path(
    "scripts/phase_4_7_1b_freeze_neural_initialization_seed_contract.py"
)

INITIALIZATION_SOURCE_SHA256 = (
    "c55f3ea1646cec7fdc8ef69f2310d98f"
    "5ee95fab77f0c48392f4a9f76612761c"
)

FORWARD_SOURCE_PATH = Path(
    "scripts/phase_4_6_2_end_to_end_itrs_forward_bce_audit.py"
)

FORWARD_SOURCE_SHA256 = (
    "18c6c7ca4915fb23eab5ed39bae6eb49"
    "1a9332196f51b302a352c3c8211b053d"
)

CANONICAL_INITIAL_STATE_SHA256 = (
    "49e822ea7fad35c458f47e134c94c05e"
    "ac099b68c5c468e2c71559c8c88998ab"
)

FROZEN_RUNTIME_AST_SHA256 = (
    "301a074aa57cfe7602f2ccbb5b8e26943"
    "b94b72e36efe4d60d1af48378c58a6e"
)


# =============================================================================
# Prior frozen evidence
# =============================================================================

PHASE_5_3_1H_MANIFEST_PATH = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_1h/"
    "phase_5_3_1h_procedural_forward_manifest.json"
)

PHASE_5_3_1F_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_1f_side_effect_free_runtime_loading_contract.json"
)


# =============================================================================
# Exact Phase-4.6.2 numerical methods proposed for composition
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
# Outputs
# =============================================================================

OUT_DIR = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_1i"
)

CANONICAL_ATTRIBUTE_PATH = (
    OUT_DIR
    / "canonical_constructor_attribute_inventory.csv"
)

METHOD_PATH = (
    OUT_DIR
    / "forward_method_dependency_inventory.csv"
)

SELF_DEPENDENCY_PATH = (
    OUT_DIR
    / "forward_method_self_attribute_dependencies.csv"
)

GLOBAL_DEPENDENCY_PATH = (
    OUT_DIR
    / "forward_method_global_dependencies.csv"
)

GLOBAL_PROVENANCE_PATH = (
    OUT_DIR
    / "forward_runtime_global_provenance.csv"
)

COMPATIBILITY_PATH = (
    OUT_DIR
    / "forward_method_canonical_topology_compatibility.csv"
)

METHOD_HASH_PATH = (
    OUT_DIR
    / "forward_method_ast_hash_registry.csv"
)

FAILED_INVARIANT_PATH = (
    OUT_DIR
    / "phase_5_3_1i_failed_invariants.csv"
)

FINAL_INVARIANT_PATH = (
    OUT_DIR
    / "phase_5_3_1i_final_invariants.csv"
)

MANIFEST_PATH = (
    OUT_DIR
    / "phase_5_3_1i_method_compatibility_manifest.json"
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


def sha256_text(
    text: str,
) -> str:

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def ast_sha256(
    node: ast.AST,
) -> str:

    return sha256_text(
        ast.dump(
            node,
            annotate_fields=True,
            include_attributes=False,
        )
    )


def read_text(
    path: Path,
) -> str:

    return path.read_text(
        encoding="utf-8",
        errors="replace",
    )


def load_json(
    path: Path,
) -> dict:

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        return json.load(handle)


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


def compact_source(
    text: str,
    limit: int = 1000,
) -> str:

    value = " ".join(
        text.split()
    )

    if len(value) <= limit:

        return value

    return (
        value[:limit]
        + " ..."
    )


# =============================================================================
# AST structure helpers
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
        for node
        in class_node.body
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
# Canonical constructor attributes
# =============================================================================

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

        current = (
            current.value
        )

    if not (
        isinstance(
            current,
            ast.Name,
        )
        and current.id
        == "self"
    ):

        return None

    if not path:

        return None

    path.reverse()

    return path[0]


def constructor_self_attributes(
    init_method: ast.FunctionDef,
) -> set[str]:

    attributes = set()

    for node in ast.walk(
        init_method
    ):

        if (
            isinstance(
                node,
                ast.Attribute,
            )
            and isinstance(
                node.ctx,
                ast.Store,
            )
        ):

            root = self_root_attribute(
                node
            )

            if root is not None:

                attributes.add(root)

        elif isinstance(
            node,
            ast.Call,
        ):

            name = dotted_name(
                node.func
            )

            if name in {
                "self.register_parameter",
                "self.register_buffer",
                "self.add_module",
            }:

                if (
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

                    attributes.add(
                        node.args[0].value
                    )

    return attributes


def loaded_self_attributes(
    method: ast.FunctionDef,
) -> set[str]:

    attributes = set()

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

            attributes.add(root)

    return attributes


# =============================================================================
# Local/global name dependency analysis
# =============================================================================

BUILTIN_NAMES = set(
    dir(builtins)
)


def argument_names(
    method: ast.FunctionDef,
) -> set[str]:

    args = method.args

    result = {
        arg.arg
        for arg in (
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

    return result


def stored_local_names(
    method: ast.FunctionDef,
) -> set[str]:

    values = set()

    for statement in method.body:

        for node in ast.walk(
            statement
        ):

            if (
                isinstance(
                    node,
                    ast.Name,
                )
                and isinstance(
                    node.ctx,
                    ast.Store,
                )
            ):

                values.add(
                    node.id
                )

    return values


def loaded_names(
    method: ast.FunctionDef,
) -> set[str]:

    values = set()

    for statement in method.body:

        for node in ast.walk(
            statement
        ):

            if (
                isinstance(
                    node,
                    ast.Name,
                )
                and isinstance(
                    node.ctx,
                    ast.Load,
                )
            ):

                values.add(
                    node.id
                )

    return values


def global_dependencies(
    method: ast.FunctionDef,
) -> set[str]:

    locals_ = (
        argument_names(method)
        | stored_local_names(method)
        | {
            "self",
        }
    )

    return {
        name
        for name
        in loaded_names(method)
        if (
            name not in locals_
            and name not in BUILTIN_NAMES
        )
    }


# =============================================================================
# Top-level namespace inventories
# =============================================================================

def import_alias_inventory(
    tree: ast.Module,
) -> dict[str, dict]:

    result = {}

    for node in tree.body:

        if isinstance(
            node,
            ast.Import,
        ):

            for alias in node.names:

                local_name = (
                    alias.asname
                    or alias.name.split(".")[0]
                )

                result[
                    local_name
                ] = {
                    "kind": (
                        "IMPORT"
                    ),
                    "ast_sha256": (
                        ast_sha256(node)
                    ),
                    "source": (
                        ast.unparse(node)
                    ),
                }

        elif isinstance(
            node,
            ast.ImportFrom,
        ):

            for alias in node.names:

                local_name = (
                    alias.asname
                    or alias.name
                )

                result[
                    local_name
                ] = {
                    "kind": (
                        "IMPORT_FROM"
                    ),
                    "ast_sha256": (
                        ast_sha256(node)
                    ),
                    "source": (
                        ast.unparse(node)
                    ),
                }

    return result


def function_inventory(
    tree: ast.Module,
) -> dict[str, ast.FunctionDef]:

    return {
        node.name: node
        for node in tree.body
        if isinstance(
            node,
            ast.FunctionDef,
        )
    }


def literal_assignment_inventory(
    tree: ast.Module,
) -> dict[str, dict]:

    result = {}

    for node in tree.body:

        if isinstance(
            node,
            ast.Assign,
        ):

            targets = [
                target
                for target in node.targets
                if isinstance(
                    target,
                    ast.Name,
                )
            ]

            if len(targets) != 1:

                continue

            target = (
                targets[0].id
            )

            try:

                value = ast.literal_eval(
                    node.value
                )

            except Exception:

                continue

            result[
                target
            ] = {
                "value": value,
                "value_repr": repr(value),
                "assignment_ast_sha256": (
                    ast_sha256(node)
                ),
                "source": (
                    ast.unparse(node)
                ),
            }

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

                value = ast.literal_eval(
                    node.value
                )

            except Exception:

                continue

            result[
                node.target.id
            ] = {
                "value": value,
                "value_repr": repr(value),
                "assignment_ast_sha256": (
                    ast_sha256(node)
                ),
                "source": (
                    ast.unparse(node)
                ),
            }

    return result


def top_level_symbol_names(
    tree: ast.Module,
) -> set[str]:

    result = set()

    result.update(
        import_alias_inventory(
            tree
        ).keys()
    )

    result.update(
        function_inventory(
            tree
        ).keys()
    )

    result.update(
        class_map(
            tree
        ).keys()
    )

    for node in tree.body:

        if isinstance(
            node,
            ast.Assign,
        ):

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

            if isinstance(
                node.target,
                ast.Name,
            ):

                result.add(
                    node.target.id
                )

    return result


# =============================================================================
# Forward-runtime global provenance
# =============================================================================

def classify_runtime_global(
    name: str,
    canonical_symbols: set[str],
    forward_imports: dict[str, dict],
    forward_helpers: dict[str, ast.FunctionDef],
    forward_literals: dict[str, dict],
) -> dict:

    if name in canonical_symbols:

        return {
            "available": True,
            "provenance": (
                "CANONICAL_RUNTIME_SYMBOL"
            ),
            "detail": (
                "Available in frozen Phase-4.7.1b "
                "sanitized canonical namespace."
            ),
            "ast_sha256": None,
        }

    if name in forward_imports:

        evidence = (
            forward_imports[
                name
            ]
        )

        return {
            "available": True,
            "provenance": (
                "FORWARD_RUNTIME_IMPORT"
            ),
            "detail": (
                evidence[
                    "source"
                ]
            ),
            "ast_sha256": (
                evidence[
                    "ast_sha256"
                ]
            ),
        }

    if name in forward_helpers:

        helper = (
            forward_helpers[
                name
            ]
        )

        return {
            "available": True,
            "provenance": (
                "FORWARD_RUNTIME_HELPER"
            ),
            "detail": (
                compact_source(
                    ast.unparse(
                        helper
                    )
                )
            ),
            "ast_sha256": (
                ast_sha256(
                    helper
                )
            ),
        }

    if name in forward_literals:

        literal = (
            forward_literals[
                name
            ]
        )

        return {
            "available": True,
            "provenance": (
                "FORWARD_RUNTIME_LITERAL"
            ),
            "detail": (
                literal[
                    "value_repr"
                ]
            ),
            "ast_sha256": (
                literal[
                    "assignment_ast_sha256"
                ]
            ),
        }

    return {
        "available": False,
        "provenance": (
            "MISSING"
        ),
        "detail": (
            "No canonical or frozen forward-runtime "
            "provenance found."
        ),
        "ast_sha256": None,
    }


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.3.1i — "
        "FORWARD-METHOD / CANONICAL-TOPOLOGY "
        "COMPATIBILITY AUDIT — CORRECTED RERUN"
    )

    print(
        "Phase-4 module imported:              NO"
    )

    print(
        "Sanitized runtime executed:           NO"
    )

    print(
        "Methods attached:                     NO"
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
    # Source / prior-phase integrity
    # =========================================================================

    banner(
        "AUTHORITATIVE SOURCE / PRIOR-PHASE RECHECK"
    )

    for path in (
        INITIALIZATION_SOURCE_PATH,
        FORWARD_SOURCE_PATH,
        PHASE_5_3_1H_MANIFEST_PATH,
        PHASE_5_3_1F_CONTRACT_PATH,
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

    initialization_hash = (
        sha256_file(
            INITIALIZATION_SOURCE_PATH
        )
    )

    forward_hash = (
        sha256_file(
            FORWARD_SOURCE_PATH
        )
    )

    require(
        initialization_hash
        == INITIALIZATION_SOURCE_SHA256,
        (
            "Canonical initialization source changed."
        ),
    )

    require(
        forward_hash
        == FORWARD_SOURCE_SHA256,
        (
            "Forward source changed."
        ),
    )

    phase_5_3_1h = load_json(
        PHASE_5_3_1H_MANIFEST_PATH
    )

    loader_contract = load_json(
        PHASE_5_3_1F_CONTRACT_PATH
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

    require(
        loader_contract[
            "status"
        ]
        == "FROZEN",
        (
            "Phase-5.3.1f loader is not frozen."
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
            "Frozen sanitized-runtime AST changed."
        ),
    )

    print(
        "Canonical initialization SHA256:      PASS"
    )

    print(
        "Forward source SHA256:                 PASS"
    )

    print(
        "Phase-5.3.1h:                          PASS"
    )

    print(
        "Phase-5.3.1f loader:                   FROZEN  PASS"
    )

    # =========================================================================
    # Parse
    # =========================================================================

    initialization_source = (
        read_text(
            INITIALIZATION_SOURCE_PATH
        )
    )

    forward_source = (
        read_text(
            FORWARD_SOURCE_PATH
        )
    )

    initialization_tree = ast.parse(
        initialization_source,
        filename=str(
            INITIALIZATION_SOURCE_PATH
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
            initialization_tree
        )
    )

    forward_classes = (
        class_map(
            forward_tree
        )
    )

    canonical_symbols = (
        top_level_symbol_names(
            initialization_tree
        )
    )

    forward_imports = (
        import_alias_inventory(
            forward_tree
        )
    )

    forward_helpers = (
        function_inventory(
            forward_tree
        )
    )

    forward_literals = (
        literal_assignment_inventory(
            forward_tree
        )
    )

    # =========================================================================
    # Canonical constructor topology
    # =========================================================================

    banner(
        "CANONICAL CONSTRUCTOR ATTRIBUTE INVENTORY"
    )

    canonical_attribute_rows = []
    canonical_attributes = {}

    for class_name in (
        METHOD_GRAFT_SPEC
    ):

        require(
            class_name
            in canonical_classes,
            (
                f"Canonical class missing: "
                f"{class_name}"
            ),
        )

        class_node = (
            canonical_classes[
                class_name
            ]
        )

        init_method = (
            direct_method(
                class_node,
                "__init__",
            )
        )

        require(
            init_method
            is not None,
            (
                f"{class_name} has no canonical __init__."
            ),
        )

        attributes = (
            constructor_self_attributes(
                init_method
            )
        )

        canonical_attributes[
            class_name
        ] = (
            attributes
        )

        for attribute in sorted(
            attributes
        ):

            canonical_attribute_rows.append(
                {
                    "class_name": (
                        class_name
                    ),
                    "attribute": (
                        attribute
                    ),
                    "constructor_ast_sha256": (
                        ast_sha256(
                            init_method
                        )
                    ),
                }
            )

    canonical_attribute_df = pd.DataFrame(
        canonical_attribute_rows
    )

    print(
        canonical_attribute_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Method dependency audit
    # =========================================================================

    banner(
        "EXACT PHASE-4.6.2 METHOD COMPATIBILITY"
    )

    method_rows = []
    self_rows = []
    global_rows = []
    provenance_rows = []
    compatibility_rows = []
    hash_rows = []

    for (
        class_name,
        method_names,
    ) in METHOD_GRAFT_SPEC.items():

        require(
            class_name
            in forward_classes,
            (
                f"Forward source class missing: "
                f"{class_name}"
            ),
        )

        forward_class = (
            forward_classes[
                class_name
            ]
        )

        constructor_attributes = (
            canonical_attributes[
                class_name
            ]
        )

        grafted_method_names = set(
            method_names
        )

        for method_name in (
            method_names
        ):

            method = (
                direct_method(
                    forward_class,
                    method_name,
                )
            )

            require(
                method
                is not None,
                (
                    f"Missing forward-runtime method: "
                    f"{class_name}.{method_name}"
                ),
            )

            method_hash = (
                ast_sha256(
                    method
                )
            )

            self_dependencies = (
                loaded_self_attributes(
                    method
                )
            )

            available_self_names = (
                constructor_attributes
                | grafted_method_names
            )

            missing_self = (
                self_dependencies
                - available_self_names
            )

            globals_ = (
                global_dependencies(
                    method
                )
            )

            missing_globals = []

            global_classifications = []

            for global_name in sorted(
                globals_
            ):

                provenance = (
                    classify_runtime_global(
                        global_name,
                        canonical_symbols,
                        forward_imports,
                        forward_helpers,
                        forward_literals,
                    )
                )

                if not (
                    provenance[
                        "available"
                    ]
                ):

                    missing_globals.append(
                        global_name
                    )

                global_classifications.append(
                    (
                        global_name,
                        provenance[
                            "provenance"
                        ],
                    )
                )

                global_rows.append(
                    {
                        "class_name": (
                            class_name
                        ),
                        "method_name": (
                            method_name
                        ),
                        "global_name": (
                            global_name
                        ),
                        "available": (
                            provenance[
                                "available"
                            ]
                        ),
                        "provenance": (
                            provenance[
                                "provenance"
                            ]
                        ),
                    }
                )

                provenance_rows.append(
                    {
                        "global_name": (
                            global_name
                        ),
                        "provenance": (
                            provenance[
                                "provenance"
                            ]
                        ),
                        "available": (
                            provenance[
                                "available"
                            ]
                        ),
                        "detail": (
                            provenance[
                                "detail"
                            ]
                        ),
                        "ast_sha256": (
                            provenance[
                                "ast_sha256"
                            ]
                        ),
                    }
                )

            self_compatible = (
                len(
                    missing_self
                )
                == 0
            )

            global_compatible = (
                len(
                    missing_globals
                )
                == 0
            )

            compatible = (
                self_compatible
                and global_compatible
            )

            method_rows.append(
                {
                    "class_name": (
                        class_name
                    ),
                    "method_name": (
                        method_name
                    ),
                    "line_number": (
                        int(
                            method.lineno
                        )
                    ),
                    "method_ast_sha256": (
                        method_hash
                    ),
                    "source": (
                        compact_source(
                            node_source(
                                forward_source,
                                method,
                            )
                        )
                    ),
                }
            )

            for attribute in sorted(
                self_dependencies
            ):

                self_rows.append(
                    {
                        "class_name": (
                            class_name
                        ),
                        "method_name": (
                            method_name
                        ),
                        "self_attribute": (
                            attribute
                        ),
                        "canonical_constructor_attribute": (
                            attribute
                            in constructor_attributes
                        ),
                        "grafted_method": (
                            attribute
                            in grafted_method_names
                        ),
                        "satisfied": (
                            attribute
                            in available_self_names
                        ),
                    }
                )

            compatibility_rows.append(
                {
                    "class_name": (
                        class_name
                    ),
                    "method_name": (
                        method_name
                    ),
                    "method_ast_sha256": (
                        method_hash
                    ),
                    "required_self_attributes": (
                        ";".join(
                            sorted(
                                self_dependencies
                            )
                        )
                    ),
                    "missing_self_attributes": (
                        ";".join(
                            sorted(
                                missing_self
                            )
                        )
                    ),
                    "required_globals": (
                        ";".join(
                            sorted(
                                globals_
                            )
                        )
                    ),
                    "missing_globals": (
                        ";".join(
                            sorted(
                                missing_globals
                            )
                        )
                    ),
                    "global_provenance": (
                        ";".join(
                            (
                                f"{name}={kind}"
                            )
                            for (
                                name,
                                kind,
                            )
                            in global_classifications
                        )
                    ),
                    "self_compatible": (
                        self_compatible
                    ),
                    "runtime_globals_compatible": (
                        global_compatible
                    ),
                    "method_compatible": (
                        compatible
                    ),
                }
            )

            hash_rows.append(
                {
                    "class_name": (
                        class_name
                    ),
                    "method_name": (
                        method_name
                    ),
                    "source_sha256": (
                        forward_hash
                    ),
                    "method_ast_sha256": (
                        method_hash
                    ),
                }
            )

    method_df = pd.DataFrame(
        method_rows
    )

    self_df = pd.DataFrame(
        self_rows
    )

    global_df = pd.DataFrame(
        global_rows
    )

    provenance_df = (
        pd.DataFrame(
            provenance_rows
        )
        .drop_duplicates(
            subset=[
                "global_name",
                "provenance",
                "detail",
                "ast_sha256",
            ]
        )
        .sort_values(
            [
                "global_name",
                "provenance",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    compatibility_df = pd.DataFrame(
        compatibility_rows
    )

    method_hash_df = pd.DataFrame(
        hash_rows
    )

    print(
        compatibility_df[
            [
                "class_name",
                "method_name",
                "missing_self_attributes",
                "missing_globals",
                "global_provenance",
                "method_compatible",
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # Runtime-only namespace additions
    # =========================================================================

    banner(
        "FORWARD-RUNTIME NAMESPACE ADDITIONS"
    )

    runtime_additions = (
        provenance_df.loc[
            provenance_df[
                "provenance"
            ]
            != "CANONICAL_RUNTIME_SYMBOL"
        ].copy()
    )

    if runtime_additions.empty:

        print(
            "No additional forward-runtime globals required."
        )

    else:

        print(
            runtime_additions.to_string(
                index=False
            )
        )

    missing_runtime_globals = (
        provenance_df.loc[
            provenance_df[
                "provenance"
            ]
            == "MISSING"
        ].copy()
    )

    print()

    print(
        f"Missing runtime globals:              "
        f"{len(missing_runtime_globals)}"
    )

    # =========================================================================
    # BasisRGCN hard check
    # =========================================================================

    banner(
        "BasisRGCNLayer COMPATIBILITY — HARD MODEL-STATE CHECK"
    )

    basis_df = (
        compatibility_df.loc[
            compatibility_df[
                "class_name"
            ]
            == "BasisRGCNLayer"
        ].copy()
    )

    require(
        len(
            basis_df
        )
        == 2,
        (
            "Expected two BasisRGCNLayer runtime methods."
        ),
    )

    print(
        basis_df[
            [
                "method_name",
                "required_self_attributes",
                "missing_self_attributes",
                "missing_globals",
                "method_compatible",
            ]
        ].to_string(
            index=False
        )
    )

    basis_self_compatible = bool(
        basis_df[
            "self_compatible"
        ].all()
    )

    print()

    print(
        "BasisRGCNLayer canonical-state dependencies:"
    )

    print(
        "  PASS"
        if basis_self_compatible
        else "  FAIL"
    )

    # =========================================================================
    # Explicit Phase-4.6.2 initialization exclusion
    # =========================================================================

    banner(
        "PHASE-4.6.2 AUDIT INITIALIZATION EXCLUSION"
    )

    non_kaiming_marker = (
        "DETERMINISTIC_NON_KAIMING_AUDIT_ONLY"
        in forward_source
    )

    require(
        non_kaiming_marker,
        (
            "Phase-4.6.2 audit-only initialization "
            "marker disappeared."
        ),
    )

    print(
        "Phase-4.6.2 initialization:"
    )

    print(
        "  DETERMINISTIC_NON_KAIMING_AUDIT_ONLY"
    )

    print()

    print(
        "Allowed for Phase-5 initialization:   NO"
    )

    print(
        "Canonical Phase-5 initialization:     Phase-4.7.1b only"
    )

    # =========================================================================
    # Final invariants
    # =========================================================================

    banner(
        "FINAL PHASE-5.3.1i COMPATIBILITY INVARIANTS"
    )

    expected_method_count = sum(
        len(value)
        for value
        in METHOD_GRAFT_SPEC.values()
    )

    checks = [
        (
            "initialization_source_sha256_exact",
            (
                initialization_hash
                == INITIALIZATION_SOURCE_SHA256
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
            "runtime_loader_still_frozen",
            (
                loader_contract[
                    "status"
                ]
                == "FROZEN"
            ),
        ),

        (
            "all_required_methods_found",
            (
                len(
                    method_df
                )
                == expected_method_count
            ),
        ),

        (
            "all_canonical_self_dependencies_satisfied",
            bool(
                compatibility_df[
                    "self_compatible"
                ].all()
            ),
        ),

        (
            "all_runtime_globals_have_explicit_provenance",
            (
                len(
                    missing_runtime_globals
                )
                == 0
            ),
        ),

        (
            "all_methods_statically_compatible",
            bool(
                compatibility_df[
                    "method_compatible"
                ].all()
            ),
        ),

        (
            "BasisRGCNLayer_canonical_state_dependencies_satisfied",
            (
                basis_self_compatible
            ),
        ),

        (
            "phase4_6_2_non_kaiming_initialization_excluded",
            (
                non_kaiming_marker
            ),
        ),

        (
            "no_phase4_module_import",
            True,
        ),

        (
            "no_runtime_execution",
            True,
        ),

        (
            "no_method_attachment",
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
                    if passed
                    else "FAIL"
                ),
            }
            for (
                name,
                passed,
            )
            in checks
        ]
    )

    failed_df = (
        invariant_df.loc[
            invariant_df[
                "result"
            ]
            == "FAIL"
        ].copy()
    )

    # =========================================================================
    # WRITE BEFORE FAILING
    # =========================================================================

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    canonical_attribute_df.to_csv(
        CANONICAL_ATTRIBUTE_PATH,
        index=False,
    )

    method_df.to_csv(
        METHOD_PATH,
        index=False,
    )

    self_df.to_csv(
        SELF_DEPENDENCY_PATH,
        index=False,
    )

    global_df.to_csv(
        GLOBAL_DEPENDENCY_PATH,
        index=False,
    )

    provenance_df.to_csv(
        GLOBAL_PROVENANCE_PATH,
        index=False,
    )

    compatibility_df.to_csv(
        COMPATIBILITY_PATH,
        index=False,
    )

    method_hash_df.to_csv(
        METHOD_HASH_PATH,
        index=False,
    )

    invariant_df.to_csv(
        FINAL_INVARIANT_PATH,
        index=False,
    )

    failed_df.to_csv(
        FAILED_INVARIANT_PATH,
        index=False,
    )

    print(
        invariant_df.to_string(
            index=False
        )
    )

    if not failed_df.empty:

        banner(
            "FAILED PHASE-5.3.1i INVARIANTS — DIAGNOSTIC"
        )

        print(
            failed_df.to_string(
                index=False
            )
        )

        failed_methods = (
            compatibility_df.loc[
                ~compatibility_df[
                    "method_compatible"
                ]
            ]
        )

        if not failed_methods.empty:

            print()

            print(
                "INCOMPATIBLE METHOD DETAIL"
            )

            print(
                failed_methods[
                    [
                        "class_name",
                        "method_name",
                        "missing_self_attributes",
                        "missing_globals",
                        "global_provenance",
                    ]
                ].to_string(
                    index=False
                )
            )

        if not missing_runtime_globals.empty:

            print()

            print(
                "MISSING RUNTIME GLOBAL DETAIL"
            )

            print(
                missing_runtime_globals.to_string(
                    index=False
                )
            )

    # =========================================================================
    # Manifest
    # =========================================================================

    all_compatible = (
        failed_df.empty
    )

    manifest = {
        "phase": (
            "5.3.1i"
        ),

        "title": (
            "Forward-Method / Canonical-Topology "
            "Compatibility Audit — Corrected Rerun"
        ),

        "status": (
            "AUDIT_COMPLETE_"
            "FORWARD_METHOD_GRAFT_STATICALLY_COMPATIBLE_"
            "NOT_YET_EXECUTED"
            if all_compatible
            else
            "AUDIT_COMPLETE_COMPATIBILITY_FAILURE_DIAGNOSED"
        ),

        "canonical_model_authority": {
            "source": (
                str(
                    INITIALIZATION_SOURCE_PATH
                )
            ),
            "sha256": (
                initialization_hash
            ),
            "initial_state_sha256": (
                CANONICAL_INITIAL_STATE_SHA256
            ),
        },

        "forward_method_authority": {
            "source": (
                str(
                    FORWARD_SOURCE_PATH
                )
            ),
            "sha256": (
                forward_hash
            ),
        },

        "compatibility": {
            "expected_method_count": (
                expected_method_count
            ),
            "methods_audited": (
                len(
                    method_df
                )
            ),
            "self_state_dependencies_satisfied": bool(
                compatibility_df[
                    "self_compatible"
                ].all()
            ),
            "missing_runtime_global_count": (
                len(
                    missing_runtime_globals
                )
            ),
            "runtime_globals_have_explicit_provenance": (
                len(
                    missing_runtime_globals
                )
                == 0
            ),
            "BasisRGCNLayer_self_dependencies_satisfied": (
                basis_self_compatible
            ),
            "all_methods_compatible": (
                all_compatible
            ),
            "method_graft_executed": (
                False
            ),
        },

        "training_boundary": {
            "phase4_module_imported": False,
            "runtime_executed": False,
            "methods_attached": False,
            "model_instantiated": False,
            "optimizer_instantiated": False,
            "forward_computation_performed": False,
            "backward_computation_performed": False,
            "optimizer_steps": 0,
        },
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    banner(
        "PHASE 5.3.1i FINAL STATUS"
    )

    print(
        f"Methods audited:                     "
        f"{len(method_df)} / {expected_method_count}"
    )

    print(
        f"Canonical self dependencies:          "
        f"{'PASS' if bool(compatibility_df['self_compatible'].all()) else 'FAIL'}"
    )

    print(
        f"Missing runtime globals:              "
        f"{len(missing_runtime_globals)}"
    )

    print(
        f"BasisRGCNLayer state compatibility:   "
        f"{'PASS' if basis_self_compatible else 'FAIL'}"
    )

    print(
        f"Overall static compatibility:         "
        f"{'PASS' if all_compatible else 'FAIL'}"
    )

    print()

    print(
        "Method graft executed:                NO"
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

    if not all_compatible:

        raise AssertionError(
            "Phase-5.3.1i compatibility audit found "
            "one or more unresolved dependencies. "
            "See FAILED PHASE-5.3.1i INVARIANTS above and "
            f"{FAILED_INVARIANT_PATH}."
        )

    banner(
        "PHASE 5.3.1i COMPLETE / "
        "FORWARD METHODS STATICALLY COMPATIBLE — "
        "NOT YET EXECUTED"
    )


if __name__ == "__main__":
    main()