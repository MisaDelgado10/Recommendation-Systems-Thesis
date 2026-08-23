#!/usr/bin/env python3
"""
Phase 5.3.1h — Procedural End-to-End Forward Orchestration Audit

STATIC AUDIT ONLY.

Background
----------
Phase 5.3.1g established:

1. Phase-4.7.1b is authoritative for parameter topology and
   canonical initialization.

2. Phase-4.6.2 is the unique strongest forward-runtime provenance
   source.

3. There is NO model-level ITRSModel.forward() implementation.

4. Phase-4.6.2 contains four branch forward methods:

       DescriptionEncoder.forward
       BasisRGCNLayer.forward
       PreferencePropagation.forward
       ScoringMLP.forward

5. TrendExtractor has no forward() method.

Therefore Phase 4's end-to-end forward path is at least partly
PROCEDURAL rather than encapsulated in a monolithic nn.Module.forward.

Purpose
-------
Statically recover the Phase-4.6.2 end-to-end orchestration:

- component construction;
- component calls;
- procedural trend computation;
- graph/description branch use;
- pair-feature construction;
- torch.cat ordering;
- ScoringMLP invocation;
- BCEWithLogitsLoss invocation;
- backward/autograd evidence;
- orchestration scope (top-level versus helper function).

THIS SCRIPT DOES NOT:
- import Phase-4 modules;
- execute Phase-4 source;
- instantiate a neural model;
- instantiate RNG;
- instantiate Adam;
- generate negatives;
- perform forward propagation;
- perform backward propagation;
- call optimizer.step();
- modify frozen Phase-4 artifacts.

No forward implementation is frozen here.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pandas as pd


# =============================================================================
# Frozen provenance
# =============================================================================

FORWARD_SOURCE_PATH = Path(
    "scripts/"
    "phase_4_6_2_end_to_end_itrs_forward_bce_audit.py"
)

FORWARD_SOURCE_SHA256 = (
    "18c6c7ca4915fb23eab5ed39bae6eb49"
    "1a9332196f51b302a352c3c8211b053d"
)

INITIALIZATION_SOURCE_PATH = Path(
    "scripts/"
    "phase_4_7_1b_freeze_neural_initialization_seed_contract.py"
)

INITIALIZATION_SOURCE_SHA256 = (
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


# =============================================================================
# Prior Phase-5 evidence
# =============================================================================

PHASE_5_3_1G_MANIFEST_PATH = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_1g/"
    "phase_5_3_1g_forward_provenance_manifest.json"
)

PHASE_5_3_1F_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_1f_side_effect_free_runtime_loading_contract.json"
)


# =============================================================================
# Canonical conceptual components
# =============================================================================

CANONICAL_COMPONENT_CLASSES = (
    "DescriptionEncoder",
    "TrendExtractor",
    "BasisRGCNLayer",
    "PreferencePropagation",
    "ScoringMLP",
)

EXPECTED_IMPLEMENTED_FORWARD_CLASSES = {
    "DescriptionEncoder",
    "BasisRGCNLayer",
    "PreferencePropagation",
    "ScoringMLP",
}

EXPECTED_FORWARD_SIGNATURES = {
    "DescriptionEncoder": (
        "(doc2vec, labels)"
    ),
    "BasisRGCNLayer": (
        "(x, edge_index, edge_type)"
    ),
    "PreferencePropagation": (
        "(latent_all, edge_index, edge_type)"
    ),
    "ScoringMLP": (
        "(pair_features)"
    ),
}

# Frozen semantic pair-feature order from Phase 4.
FROZEN_PAIR_FEATURE_ORDER = [
    "F_t",
    "L_o",
    "F_d,o",
    "F_s,o",
    "L_b",
    "F_d,b",
    "F_s,b",
]


# =============================================================================
# Output paths
# =============================================================================

OUT_DIR = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_1h"
)

CONSTRUCTOR_BINDING_PATH = (
    OUT_DIR
    / "phase4_6_2_component_constructor_bindings.csv"
)

COMPONENT_CALL_PATH = (
    OUT_DIR
    / "phase4_6_2_component_call_inventory.csv"
)

TOP_LEVEL_STATEMENT_PATH = (
    OUT_DIR
    / "phase4_6_2_top_level_executable_statements.csv"
)

FUNCTION_SUMMARY_PATH = (
    OUT_DIR
    / "phase4_6_2_function_orchestration_summary.csv"
)

TORCH_CAT_PATH = (
    OUT_DIR
    / "phase4_6_2_torch_cat_inventory.csv"
)

TREND_EVIDENCE_PATH = (
    OUT_DIR
    / "phase4_6_2_trend_runtime_evidence.csv"
)

LOSS_EVIDENCE_PATH = (
    OUT_DIR
    / "phase4_6_2_loss_autograd_evidence.csv"
)

PROCEDURAL_FORWARD_PATH = (
    OUT_DIR
    / "phase4_6_2_procedural_forward_candidate_statements.csv"
)

CONSTRUCTOR_COMPARISON_PATH = (
    OUT_DIR
    / "phase4_6_2_vs_7_1b_constructor_comparison.csv"
)

FINAL_INVARIANT_PATH = (
    OUT_DIR
    / "phase_5_3_1h_final_invariants.csv"
)

MANIFEST_PATH = (
    OUT_DIR
    / "phase_5_3_1h_procedural_forward_manifest.json"
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

    if (
        value
        is not None
    ):

        return (
            value
        )

    return ast.unparse(
        node
    )


def compact_source(
    text: str,
    limit: int = 1000,
) -> str:

    text = (
        " ".join(
            text.split()
        )
    )

    if (
        len(
            text
        )
        <= limit
    ):

        return (
            text
        )

    return (
        text[
            :limit
        ]
        + " ..."
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


def stored_names(
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
                ast.Store,
            )
        ):

            values.add(
                candidate.id
            )

    return sorted(
        values
    )


def call_nodes(
    node: ast.AST,
) -> list[
    ast.Call
]:

    return [
        candidate
        for candidate
        in ast.walk(
            node
        )
        if isinstance(
            candidate,
            ast.Call,
        )
    ]


def call_names(
    node: ast.AST,
) -> list[str]:

    return [
        dotted_name(
            candidate.func
        )
        for candidate
        in call_nodes(
            node
        )
    ]


# =============================================================================
# Static function signature
# =============================================================================

def function_signature(
    function: ast.FunctionDef,
    drop_self: bool = False,
) -> str:

    arguments = (
        function.args
    )

    positional = (
        list(
            arguments.posonlyargs
        )
        + list(
            arguments.args
        )
    )

    defaults = list(
        arguments.defaults
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

    for (
        index,
        argument,
    ) in enumerate(
        positional
    ):

        if (
            drop_self
            and index
            == 0
            and argument.arg
            == "self"
        ):

            continue

        value = (
            argument.arg
        )

        if (
            index
            >= default_start
        ):

            value += (
                "="
                + ast.unparse(
                    defaults[
                        index
                        - default_start
                    ]
                )
            )

        parts.append(
            value
        )

    if (
        arguments.vararg
        is not None
    ):

        parts.append(
            "*"
            + arguments.vararg.arg
        )

    elif (
        arguments.kwonlyargs
    ):

        parts.append(
            "*"
        )

    for (
        argument,
        default,
    ) in zip(
        arguments.kwonlyargs,
        arguments.kw_defaults,
    ):

        value = (
            argument.arg
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
        arguments.kwarg
        is not None
    ):

        parts.append(
            "**"
            + arguments.kwarg.arg
        )

    return (
        "("
        + ", ".join(
            parts
        )
        + ")"
    )


# =============================================================================
# Class helpers
# =============================================================================

def class_map(
    tree: ast.Module,
) -> dict[
    str,
    ast.ClassDef,
]:

    return {
        node.name: node
        for node
        in tree.body
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
        len(
            matches
        )
        <= 1,
        (
            f"Multiple {class_node.name}.{method_name} "
            "definitions found"
        ),
    )

    if not matches:

        return None

    return (
        matches[
            0
        ]
    )


# =============================================================================
# Assignment / constructor binding helpers
# =============================================================================

def assignment_targets(
    node: ast.AST,
) -> list[str]:

    targets = []

    raw_targets = []

    if isinstance(
        node,
        ast.Assign,
    ):

        raw_targets = list(
            node.targets
        )

    elif isinstance(
        node,
        ast.AnnAssign,
    ):

        raw_targets = [
            node.target
        ]

    else:

        return (
            targets
        )

    for raw_target in (
        raw_targets
    ):

        for candidate in ast.walk(
            raw_target
        ):

            if isinstance(
                candidate,
                ast.Name,
            ):

                targets.append(
                    candidate.id
                )

    return sorted(
        set(
            targets
        )
    )


def assignment_value(
    node: ast.AST,
) -> ast.AST | None:

    if isinstance(
        node,
        ast.Assign,
    ):

        return (
            node.value
        )

    if isinstance(
        node,
        ast.AnnAssign,
    ):

        return (
            node.value
        )

    return None


def direct_constructor_call(
    node: ast.AST,
) -> tuple[
    str,
    ast.Call,
] | None:

    value = assignment_value(
        node
    )

    if not isinstance(
        value,
        ast.Call,
    ):

        return None

    name = dotted_name(
        value.func
    )

    if (
        name
        in CANONICAL_COMPONENT_CLASSES
    ):

        return (
            name,
            value,
        )

    return None


# =============================================================================
# Scope inventory
# =============================================================================

class ScopeCallVisitor(
    ast.NodeVisitor
):
    """
    Collect calls while preserving whether they occur:
    - top-level
    - inside a top-level function
    - inside a class method

    Nested function/class definitions are given their own scope.
    """

    def __init__(
        self,
    ) -> None:

        self.scope_stack = [
            "__TOP_LEVEL__"
        ]

        self.rows = []

    def current_scope(
        self,
    ) -> str:

        return (
            "::".join(
                self.scope_stack
            )
        )

    def visit_FunctionDef(
        self,
        node: ast.FunctionDef,
    ) -> None:

        self.scope_stack.append(
            node.name
        )

        for statement in (
            node.body
        ):

            self.visit(
                statement
            )

        self.scope_stack.pop()

    def visit_AsyncFunctionDef(
        self,
        node: ast.AsyncFunctionDef,
    ) -> None:

        self.scope_stack.append(
            node.name
        )

        for statement in (
            node.body
        ):

            self.visit(
                statement
            )

        self.scope_stack.pop()

    def visit_ClassDef(
        self,
        node: ast.ClassDef,
    ) -> None:

        self.scope_stack.append(
            node.name
        )

        for statement in (
            node.body
        ):

            self.visit(
                statement
            )

        self.scope_stack.pop()

    def visit_Call(
        self,
        node: ast.Call,
    ) -> None:

        self.rows.append(
            {
                "scope": (
                    self.current_scope()
                ),

                "line_number": (
                    int(
                        node.lineno
                    )
                ),

                "call_name": (
                    dotted_name(
                        node.func
                    )
                ),

                "call_source": (
                    compact_source(
                        ast.unparse(
                            node
                        ),
                        limit=700,
                    )
                ),
            }
        )

        self.generic_visit(
            node
        )


# =============================================================================
# Semantic evidence
# =============================================================================

TREND_TOKENS = (
    "trend",
    "gru",
    "attention",
    "period",
    "history",
    "period_ptr",
    "startup_count",
    "startup_node",
)

PAIR_TOKENS = (
    "pair",
    "feature",
    "investor",
    "startup",
    "latent",
    "description",
    "structural",
    "trend",
)

LOSS_TOKENS = (
    "bce",
    "binary_cross_entropy",
    "withlogits",
    "loss",
    "backward",
    "autograd",
)

STRUCTURAL_TOKENS = (
    "edge_index",
    "edge_type",
    "rgcn",
    "structural",
    "propagation",
)

DESCRIPTION_TOKENS = (
    "doc2vec",
    "label",
    "description",
    "encoder",
)


def text_contains_any(
    text: str,
    tokens: tuple[str, ...],
) -> bool:

    lower = (
        text.lower()
    )

    return any(
        token.lower()
        in lower
        for token
        in tokens
    )


def statement_semantic_tags(
    text: str,
) -> list[str]:

    tags = []

    if text_contains_any(
        text,
        TREND_TOKENS,
    ):

        tags.append(
            "TREND"
        )

    if text_contains_any(
        text,
        PAIR_TOKENS,
    ):

        tags.append(
            "PAIR"
        )

    if text_contains_any(
        text,
        LOSS_TOKENS,
    ):

        tags.append(
            "LOSS_AUTOGRAD"
        )

    if text_contains_any(
        text,
        STRUCTURAL_TOKENS,
    ):

        tags.append(
            "STRUCTURAL"
        )

    if text_contains_any(
        text,
        DESCRIPTION_TOKENS,
    ):

        tags.append(
            "DESCRIPTION"
        )

    return (
        tags
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.3.1h — "
        "PROCEDURAL END-TO-END FORWARD ORCHESTRATION AUDIT"
    )

    print(
        "Phase-4 module import:               NO"
    )

    print(
        "Sanitized runtime execution:         NO"
    )

    print(
        "Model instantiated:                  NO"
    )

    print(
        "Training-negative RNG:               NO"
    )

    print(
        "Training-order RNG:                  NO"
    )

    print(
        "Optimizer instantiated:              NO"
    )

    print(
        "Forward computation:                 NO"
    )

    print(
        "Backward computation:                NO"
    )

    print(
        "Optimizer steps:                     0"
    )

    # =========================================================================
    # Prerequisite integrity
    # =========================================================================

    banner(
        "PRIOR-PHASE AND SOURCE INTEGRITY"
    )

    for path in (
        FORWARD_SOURCE_PATH,
        INITIALIZATION_SOURCE_PATH,
        PHASE_5_3_1G_MANIFEST_PATH,
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

    forward_source_hash = (
        sha256_file(
            FORWARD_SOURCE_PATH
        )
    )

    initialization_source_hash = (
        sha256_file(
            INITIALIZATION_SOURCE_PATH
        )
    )

    require(
        forward_source_hash
        == FORWARD_SOURCE_SHA256,
        (
            "Phase-4.6.2 source SHA256 changed.\n"
            f"Expected: {FORWARD_SOURCE_SHA256}\n"
            f"Actual:   {forward_source_hash}"
        ),
    )

    require(
        initialization_source_hash
        == INITIALIZATION_SOURCE_SHA256,
        (
            "Phase-4.7.1b source SHA256 changed.\n"
            f"Expected: {INITIALIZATION_SOURCE_SHA256}\n"
            f"Actual:   {initialization_source_hash}"
        ),
    )

    phase_5_3_1g = load_json(
        PHASE_5_3_1G_MANIFEST_PATH
    )

    loader_contract = load_json(
        PHASE_5_3_1F_CONTRACT_PATH
    )

    require(
        phase_5_3_1g[
            "status"
        ]
        == (
            "AUDIT_COMPLETE_"
            "FORWARD_IMPLEMENTATION_PROVENANCE_NOT_YET_FROZEN"
        ),
        (
            "Unexpected Phase-5.3.1g status"
        ),
    )

    require(
        phase_5_3_1g[
            "forward_provenance_audit"
        ][
            "unique_strongest_candidate"
        ]
        == str(
            FORWARD_SOURCE_PATH
        ),
        (
            "Phase-5.3.1g strongest forward provenance "
            "candidate changed"
        ),
    )

    require(
        loader_contract[
            "status"
        ]
        == "FROZEN",
        (
            "Phase-5.3.1f loader contract is not frozen"
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
            "Frozen runtime-loader AST hash changed"
        ),
    )

    print(
        "Phase-4.6.2 source SHA256:           PASS"
    )

    print(
        "Phase-4.7.1b source SHA256:          PASS"
    )

    print(
        "Phase-5.3.1g provenance audit:       PASS"
    )

    print(
        "Phase-5.3.1f loader contract:        FROZEN  PASS"
    )

    # =========================================================================
    # Parse both Phase-4 sources
    # =========================================================================

    forward_source = read_text(
        FORWARD_SOURCE_PATH
    )

    initialization_source = read_text(
        INITIALIZATION_SOURCE_PATH
    )

    forward_tree = ast.parse(
        forward_source,
        filename=str(
            FORWARD_SOURCE_PATH
        ),
    )

    initialization_tree = ast.parse(
        initialization_source,
        filename=str(
            INITIALIZATION_SOURCE_PATH
        ),
    )

    # =========================================================================
    # Constructor compatibility diagnostic
    # =========================================================================

    banner(
        "PHASE-4.6.2 vs PHASE-4.7.1b CONSTRUCTOR DIAGNOSTIC"
    )

    forward_classes = class_map(
        forward_tree
    )

    initialization_classes = class_map(
        initialization_tree
    )

    constructor_rows = []

    for class_name in (
        CANONICAL_COMPONENT_CLASSES
    ):

        forward_class = (
            forward_classes.get(
                class_name
            )
        )

        initialization_class = (
            initialization_classes.get(
                class_name
            )
        )

        forward_init = (
            direct_method(
                forward_class,
                "__init__",
            )
            if forward_class
            is not None
            else None
        )

        initialization_init = (
            direct_method(
                initialization_class,
                "__init__",
            )
            if initialization_class
            is not None
            else None
        )

        constructor_rows.append(
            {
                "class_name": (
                    class_name
                ),

                "present_phase4_6_2": (
                    forward_class
                    is not None
                ),

                "present_phase4_7_1b": (
                    initialization_class
                    is not None
                ),

                "phase4_6_2_init_signature": (
                    function_signature(
                        forward_init,
                        drop_self=True,
                    )
                    if forward_init
                    is not None
                    else None
                ),

                "phase4_7_1b_init_signature": (
                    function_signature(
                        initialization_init,
                        drop_self=True,
                    )
                    if initialization_init
                    is not None
                    else None
                ),

                "phase4_6_2_init_ast_sha256": (
                    ast_sha256(
                        forward_init
                    )
                    if forward_init
                    is not None
                    else None
                ),

                "phase4_7_1b_init_ast_sha256": (
                    ast_sha256(
                        initialization_init
                    )
                    if initialization_init
                    is not None
                    else None
                ),

                "constructor_ast_exact_match": (
                    (
                        ast_sha256(
                            forward_init
                        )
                        == ast_sha256(
                            initialization_init
                        )
                    )
                    if (
                        forward_init
                        is not None
                        and initialization_init
                        is not None
                    )
                    else False
                ),
            }
        )

    constructor_df = pd.DataFrame(
        constructor_rows
    )

    print(
        constructor_df.to_string(
            index=False
        )
    )

    print()

    print(
        "Constructor AST equality is diagnostic only."
    )

    print(
        "The frozen Phase-4.7.1b parameter namespace/state remains"
    )

    print(
        "the numerical authority."
    )

    # =========================================================================
    # Constructor bindings inside Phase-4.6.2
    # =========================================================================

    banner(
        "PHASE-4.6.2 COMPONENT CONSTRUCTOR BINDINGS"
    )

    constructor_binding_rows = []

    instance_bindings = {}

    for node in (
        forward_tree.body
    ):

        result = (
            direct_constructor_call(
                node
            )
        )

        if (
            result
            is None
        ):

            continue

        (
            class_name,
            constructor_call,
        ) = result

        targets = assignment_targets(
            node
        )

        for target in (
            targets
        ):

            instance_bindings[
                target
            ] = (
                class_name
            )

            constructor_binding_rows.append(
                {
                    "scope": (
                        "__TOP_LEVEL__"
                    ),

                    "line_number": (
                        int(
                            node.lineno
                        )
                    ),

                    "variable": (
                        target
                    ),

                    "class_name": (
                        class_name
                    ),

                    "constructor_source": (
                        compact_source(
                            ast.unparse(
                                constructor_call
                            ),
                            limit=900,
                        )
                    ),
                }
            )

    constructor_binding_df = pd.DataFrame(
        constructor_binding_rows
    )

    if (
        constructor_binding_df.empty
    ):

        print(
            "No direct top-level canonical component "
            "constructor bindings found."
        )

    else:

        print(
            constructor_binding_df.to_string(
                index=False
            )
        )

    # =========================================================================
    # Top-level executable statements
    # =========================================================================

    banner(
        "PHASE-4.6.2 TOP-LEVEL EXECUTABLE STATEMENT INVENTORY"
    )

    executable_rows = []

    definition_types = (
        ast.Import,
        ast.ImportFrom,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
    )

    for (
        statement_index,
        node,
    ) in enumerate(
        forward_tree.body
    ):

        if isinstance(
            node,
            definition_types,
        ):

            continue

        source_text = (
            node_source(
                forward_source,
                node,
            )
        )

        tags = (
            statement_semantic_tags(
                source_text
            )
        )

        executable_rows.append(
            {
                "statement_index": (
                    statement_index
                ),

                "line_number": (
                    int(
                        getattr(
                            node,
                            "lineno",
                            -1,
                        )
                    )
                ),

                "statement_type": (
                    type(
                        node
                    ).__name__
                ),

                "stored_names": (
                    ";".join(
                        stored_names(
                            node
                        )
                    )
                ),

                "loaded_names": (
                    ";".join(
                        loaded_names(
                            node
                        )
                    )
                ),

                "calls": (
                    ";".join(
                        call_names(
                            node
                        )
                    )
                ),

                "semantic_tags": (
                    ";".join(
                        tags
                    )
                ),

                "source": (
                    compact_source(
                        source_text,
                        limit=1400,
                    )
                ),
            }
        )

    executable_df = pd.DataFrame(
        executable_rows
    )

    print(
        f"Top-level executable statements:     "
        f"{len(executable_df)}"
    )

    # Print only semantically relevant rows.
    relevant_top_level = (
        executable_df.loc[
            executable_df[
                "semantic_tags"
            ]
            != ""
        ].copy()
    )

    print(
        f"Forward-relevant top-level statements:"
        f" {len(relevant_top_level)}"
    )

    if not (
        relevant_top_level.empty
    ):

        print()

        print(
            relevant_top_level[
                [
                    "statement_index",
                    "line_number",
                    "statement_type",
                    "semantic_tags",
                    "calls",
                    "source",
                ]
            ].to_string(
                index=False
            )
        )

    # =========================================================================
    # Scope call inventory
    # =========================================================================

    banner(
        "PROCEDURAL ORCHESTRATION SCOPE AUDIT"
    )

    scope_visitor = (
        ScopeCallVisitor()
    )

    scope_visitor.visit(
        forward_tree
    )

    scope_call_df = pd.DataFrame(
        scope_visitor.rows
    )

    scope_summary_rows = []

    if not (
        scope_call_df.empty
    ):

        for (
            scope,
            group,
        ) in scope_call_df.groupby(
            "scope",
            sort=True,
        ):

            joined = (
                " ".join(
                    group[
                        "call_source"
                    ].astype(
                        str
                    )
                )
            )

            constructor_mentions = sum(
                class_name
                in joined
                for class_name
                in CANONICAL_COMPONENT_CLASSES
            )

            component_variable_mentions = sum(
                variable
                in joined
                for variable
                in instance_bindings
            )

            contains_cat = bool(
                group[
                    "call_name"
                ]
                .str.endswith(
                    "torch.cat"
                )
                .any()
                or (
                    group[
                        "call_name"
                    ]
                    == "torch.cat"
                ).any()
            )

            contains_bce = (
                text_contains_any(
                    joined,
                    (
                        "BCEWithLogitsLoss",
                        "binary_cross_entropy_with_logits",
                    ),
                )
            )

            contains_backward = (
                text_contains_any(
                    joined,
                    (
                        ".backward",
                        "autograd.grad",
                    ),
                )
            )

            contains_trend = (
                text_contains_any(
                    joined,
                    TREND_TOKENS,
                )
            )

            score = (
                constructor_mentions
                * 10
                + component_variable_mentions
                * 15
                + int(
                    contains_cat
                )
                * 40
                + int(
                    contains_bce
                )
                * 50
                + int(
                    contains_backward
                )
                * 20
                + int(
                    contains_trend
                )
                * 30
            )

            scope_summary_rows.append(
                {
                    "scope": (
                        scope
                    ),

                    "call_count": (
                        len(
                            group
                        )
                    ),

                    "canonical_constructor_mentions": (
                        constructor_mentions
                    ),

                    "bound_component_variable_mentions": (
                        component_variable_mentions
                    ),

                    "contains_torch_cat": (
                        contains_cat
                    ),

                    "contains_BCEWithLogitsLoss": (
                        contains_bce
                    ),

                    "contains_backward_or_autograd": (
                        contains_backward
                    ),

                    "contains_trend_evidence": (
                        contains_trend
                    ),

                    "orchestration_score": (
                        score
                    ),
                }
            )

    function_summary_df = pd.DataFrame(
        scope_summary_rows
    )

    if (
        function_summary_df.empty
    ):

        print(
            "No call scopes found."
        )

    else:

        function_summary_df = (
            function_summary_df
            .sort_values(
                [
                    "orchestration_score",
                    "scope",
                ],
                ascending=[
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
            function_summary_df.to_string(
                index=False
            )
        )

    # =========================================================================
    # Component calls
    # =========================================================================

    banner(
        "CANONICAL COMPONENT CALL INVENTORY"
    )

    component_call_rows = []

    if not (
        scope_call_df.empty
    ):

        for _, row in (
            scope_call_df.iterrows()
        ):

            call_name = str(
                row[
                    "call_name"
                ]
            )

            matched_variable = None
            matched_class = None

            for (
                variable,
                class_name,
            ) in instance_bindings.items():

                if (
                    call_name
                    == variable
                    or call_name.startswith(
                        variable
                        + "."
                    )
                ):

                    matched_variable = (
                        variable
                    )

                    matched_class = (
                        class_name
                    )

                    break

            if (
                matched_variable
                is None
            ):

                continue

            component_call_rows.append(
                {
                    "scope": (
                        row[
                            "scope"
                        ]
                    ),

                    "line_number": (
                        row[
                            "line_number"
                        ]
                    ),

                    "component_variable": (
                        matched_variable
                    ),

                    "component_class": (
                        matched_class
                    ),

                    "call_name": (
                        call_name
                    ),

                    "call_source": (
                        row[
                            "call_source"
                        ]
                    ),
                }
            )

    component_call_df = pd.DataFrame(
        component_call_rows
    )

    if (
        component_call_df.empty
    ):

        print(
            "No calls through directly detected component "
            "bindings were found."
        )

    else:

        print(
            component_call_df.to_string(
                index=False
            )
        )

    # =========================================================================
    # torch.cat inventory
    # =========================================================================

    banner(
        "torch.cat / FEATURE-ASSEMBLY INVENTORY"
    )

    cat_rows = []

    for node in ast.walk(
        forward_tree
    ):

        if not isinstance(
            node,
            ast.Call,
        ):

            continue

        name = dotted_name(
            node.func
        )

        if (
            name
            != "torch.cat"
            and not name.endswith(
                ".cat"
            )
        ):

            continue

        source_text = (
            ast.unparse(
                node
            )
        )

        first_argument = (
            ast.unparse(
                node.args[
                    0
                ]
            )
            if (
                len(
                    node.args
                )
                >= 1
            )
            else None
        )

        cat_rows.append(
            {
                "line_number": (
                    int(
                        node.lineno
                    )
                ),

                "call_name": (
                    name
                ),

                "first_argument": (
                    first_argument
                ),

                "loaded_names": (
                    ";".join(
                        loaded_names(
                            node
                        )
                    )
                ),

                "semantic_tags": (
                    ";".join(
                        statement_semantic_tags(
                            source_text
                        )
                    )
                ),

                "call_source": (
                    compact_source(
                        source_text,
                        limit=1600,
                    )
                ),
            }
        )

    cat_df = pd.DataFrame(
        cat_rows
    )

    require(
        not cat_df.empty,
        (
            "Phase-4.6.2 contains no torch.cat calls, "
            "contradicting Phase-5.3.1g evidence"
        ),
    )

    print(
        cat_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Trend evidence
    # =========================================================================

    banner(
        "PROCEDURAL TREND-RUNTIME EVIDENCE"
    )

    trend_rows = []

    # Top-level executable statements.
    for _, row in (
        executable_df.iterrows()
    ):

        source_text = str(
            row[
                "source"
            ]
        )

        if not text_contains_any(
            source_text,
            TREND_TOKENS,
        ):

            continue

        trend_rows.append(
            {
                "scope": (
                    "__TOP_LEVEL__"
                ),

                "line_number": (
                    row[
                        "line_number"
                    ]
                ),

                "statement_type": (
                    row[
                        "statement_type"
                    ]
                ),

                "calls": (
                    row[
                        "calls"
                    ]
                ),

                "source": (
                    source_text
                ),
            }
        )

    # Function/method calls.
    if not (
        scope_call_df.empty
    ):

        for _, row in (
            scope_call_df.iterrows()
        ):

            source_text = str(
                row[
                    "call_source"
                ]
            )

            if not text_contains_any(
                source_text,
                TREND_TOKENS,
            ):

                continue

            trend_rows.append(
                {
                    "scope": (
                        row[
                            "scope"
                        ]
                    ),

                    "line_number": (
                        row[
                            "line_number"
                        ]
                    ),

                    "statement_type": (
                        "Call"
                    ),

                    "calls": (
                        row[
                            "call_name"
                        ]
                    ),

                    "source": (
                        source_text
                    ),
                }
            )

    trend_df = (
        pd.DataFrame(
            trend_rows
        )
        .drop_duplicates()
        .reset_index(
            drop=True
        )
        if trend_rows
        else pd.DataFrame(
            columns=[
                "scope",
                "line_number",
                "statement_type",
                "calls",
                "source",
            ]
        )
    )

    print(
        f"Trend-related evidence rows:         "
        f"{len(trend_df)}"
    )

    if not (
        trend_df.empty
    ):

        print()

        print(
            trend_df.to_string(
                index=False
            )
        )

    # =========================================================================
    # BCE / loss / autograd evidence
    # =========================================================================

    banner(
        "BCEWithLogitsLoss / AUTOGRAD EVIDENCE"
    )

    loss_rows = []

    for node in ast.walk(
        forward_tree
    ):

        if not isinstance(
            node,
            ast.Call,
        ):

            continue

        call_name = (
            dotted_name(
                node.func
            )
        )

        call_text = (
            ast.unparse(
                node
            )
        )

        lower = (
            call_text.lower()
        )

        relevant = (
            "bcewithlogitsloss"
            in lower
            or "binary_cross_entropy_with_logits"
            in lower
            or call_name.endswith(
                ".backward"
            )
            or "autograd.grad"
            in lower
        )

        if not relevant:

            continue

        loss_rows.append(
            {
                "line_number": (
                    int(
                        node.lineno
                    )
                ),

                "call_name": (
                    call_name
                ),

                "call_source": (
                    compact_source(
                        call_text,
                        limit=1200,
                    )
                ),
            }
        )

    loss_df = pd.DataFrame(
        loss_rows
    )

    require(
        not loss_df.empty,
        (
            "No BCE/autograd evidence recovered "
            "from Phase-4.6.2"
        ),
    )

    print(
        loss_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Procedural forward candidate statements
    # =========================================================================

    banner(
        "PROCEDURAL END-TO-END FORWARD CANDIDATE STATEMENTS"
    )

    procedural_rows = []

    for _, row in (
        executable_df.iterrows()
    ):

        tags = {
            tag
            for tag
            in str(
                row[
                    "semantic_tags"
                ]
            ).split(
                ";"
            )
            if tag
        }

        if not (
            tags
            & {
                "TREND",
                "PAIR",
                "LOSS_AUTOGRAD",
                "STRUCTURAL",
                "DESCRIPTION",
            }
        ):

            continue

        procedural_rows.append(
            row.to_dict()
        )

    procedural_df = pd.DataFrame(
        procedural_rows
    )

    if (
        procedural_df.empty
    ):

        print(
            "No top-level procedural candidate statements."
        )

    else:

        print(
            procedural_df[
                [
                    "statement_index",
                    "line_number",
                    "semantic_tags",
                    "stored_names",
                    "loaded_names",
                    "calls",
                    "source",
                ]
            ].to_string(
                index=False
            )
        )

    # =========================================================================
    # Decision-facing summary
    # =========================================================================

    banner(
        "DECISION-FACING ORCHESTRATION SUMMARY"
    )

    top_scope_row = None

    if not (
        function_summary_df.empty
    ):

        top_scope_row = (
            function_summary_df.iloc[
                0
            ]
        )

    strongest_scope = (
        str(
            top_scope_row[
                "scope"
            ]
        )
        if top_scope_row
        is not None
        else None
    )

    strongest_scope_score = (
        int(
            top_scope_row[
                "orchestration_score"
            ]
        )
        if top_scope_row
        is not None
        else None
    )

    print(
        "Initialization authority:"
    )

    print(
        f"  {INITIALIZATION_SOURCE_PATH}"
    )

    print()

    print(
        "Forward procedural authority candidate:"
    )

    print(
        f"  {FORWARD_SOURCE_PATH}"
    )

    print()

    print(
        f"Strongest orchestration scope:       "
        f"{strongest_scope}"
    )

    print(
        f"Orchestration evidence score:        "
        f"{strongest_scope_score}"
    )

    print(
        f"torch.cat calls:                     "
        f"{len(cat_df)}"
    )

    print(
        f"Trend evidence rows:                 "
        f"{len(trend_df)}"
    )

    print(
        f"BCE/autograd evidence rows:          "
        f"{len(loss_df)}"
    )

    print()

    print(
        "Forward orchestration frozen:        NO"
    )

    print(
        "Reason:"
    )

    print(
        "  Exact procedural trend path, pair-feature assembly,"
    )

    print(
        "  component calls, and BCE path must first be reviewed."
    )

    # =========================================================================
    # Hard invariants
    # =========================================================================

    banner(
        "FINAL PHASE-5.3.1h AUDIT INVARIANTS"
    )

    checks = [
        (
            "phase4_6_2_source_sha256_exact",
            (
                forward_source_hash
                == FORWARD_SOURCE_SHA256
            ),
        ),

        (
            "phase4_7_1b_source_sha256_exact",
            (
                initialization_source_hash
                == INITIALIZATION_SOURCE_SHA256
            ),
        ),

        (
            "phase_5_3_1g_unique_forward_candidate_preserved",
            (
                phase_5_3_1g[
                    "forward_provenance_audit"
                ][
                    "unique_strongest_candidate"
                ]
                == str(
                    FORWARD_SOURCE_PATH
                )
            ),
        ),

        (
            "phase_5_3_1f_loader_still_frozen",
            (
                loader_contract[
                    "status"
                ]
                == "FROZEN"
            ),
        ),

        (
            "phase4_6_2_contains_torch_cat",
            (
                len(
                    cat_df
                )
                >= 1
            ),
        ),

        (
            "phase4_6_2_contains_trend_runtime_evidence",
            (
                len(
                    trend_df
                )
                >= 1
            ),
        ),

        (
            "phase4_6_2_contains_BCE_or_autograd_evidence",
            (
                len(
                    loss_df
                )
                >= 1
            ),
        ),

        (
            "procedural_forward_statements_audited",
            (
                procedural_df
                is not None
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
            "At least one Phase-5.3.1h "
            "procedural-forward audit invariant failed"
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
        "WRITE PHASE-5.3.1h AUDIT OUTPUTS"
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    constructor_binding_df.to_csv(
        CONSTRUCTOR_BINDING_PATH,
        index=False,
    )

    component_call_df.to_csv(
        COMPONENT_CALL_PATH,
        index=False,
    )

    executable_df.to_csv(
        TOP_LEVEL_STATEMENT_PATH,
        index=False,
    )

    function_summary_df.to_csv(
        FUNCTION_SUMMARY_PATH,
        index=False,
    )

    cat_df.to_csv(
        TORCH_CAT_PATH,
        index=False,
    )

    trend_df.to_csv(
        TREND_EVIDENCE_PATH,
        index=False,
    )

    loss_df.to_csv(
        LOSS_EVIDENCE_PATH,
        index=False,
    )

    procedural_df.to_csv(
        PROCEDURAL_FORWARD_PATH,
        index=False,
    )

    constructor_df.to_csv(
        CONSTRUCTOR_COMPARISON_PATH,
        index=False,
    )

    invariant_df.to_csv(
        FINAL_INVARIANT_PATH,
        index=False,
    )

    manifest = {
        "phase": (
            "5.3.1h"
        ),

        "title": (
            "Procedural End-to-End Forward "
            "Orchestration Audit"
        ),

        "status": (
            "AUDIT_COMPLETE_"
            "PROCEDURAL_FORWARD_ORCHESTRATION_NOT_YET_FROZEN"
        ),

        "initialization_provenance": {
            "source": (
                str(
                    INITIALIZATION_SOURCE_PATH
                )
            ),

            "source_sha256": (
                initialization_source_hash
            ),

            "canonical_state_sha256": (
                CANONICAL_INITIAL_STATE_SHA256
            ),
        },

        "forward_provenance": {
            "source": (
                str(
                    FORWARD_SOURCE_PATH
                )
            ),

            "source_sha256": (
                forward_source_hash
            ),

            "classification": (
                "UNIQUE_STRONGEST_PHASE4_FORWARD_RUNTIME_EVIDENCE"
            ),
        },

        "orchestration_audit": {
            "strongest_scope": (
                strongest_scope
            ),

            "strongest_scope_score": (
                strongest_scope_score
            ),

            "component_constructor_binding_count": (
                len(
                    constructor_binding_df
                )
            ),

            "component_call_count": (
                len(
                    component_call_df
                )
            ),

            "torch_cat_call_count": (
                len(
                    cat_df
                )
            ),

            "trend_evidence_row_count": (
                len(
                    trend_df
                )
            ),

            "loss_autograd_evidence_row_count": (
                len(
                    loss_df
                )
            ),

            "procedural_candidate_statement_count": (
                len(
                    procedural_df
                )
            ),

            "forward_orchestration_frozen": (
                False
            ),
        },

        "frozen_pair_feature_semantics": {
            "order": (
                FROZEN_PAIR_FEATURE_ORDER
            ),

            "note": (
                "Phase-5.3.1h does not assume source variable names "
                "match these semantic labels; torch.cat source is "
                "audited before mapping semantics."
            ),
        },

        "training_boundary": {
            "phase4_module_imported": (
                False
            ),

            "runtime_executed": (
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
            "Map the audited Phase-4.6.2 procedural statements "
            "to the frozen semantic branches F_t, L_o, F_d,o, "
            "F_s,o, L_b, F_d,b, F_s,b; freeze exact procedural "
            "trend extraction, pair-feature assembly, scoring call, "
            "and BCE runtime path; then prove compatibility with the "
            "Phase-4.7.1b initialized parameter namespace."
        ),

        "outputs": [
            str(
                CONSTRUCTOR_BINDING_PATH
            ),

            str(
                COMPONENT_CALL_PATH
            ),

            str(
                TOP_LEVEL_STATEMENT_PATH
            ),

            str(
                FUNCTION_SUMMARY_PATH
            ),

            str(
                TORCH_CAT_PATH
            ),

            str(
                TREND_EVIDENCE_PATH
            ),

            str(
                LOSS_EVIDENCE_PATH
            ),

            str(
                PROCEDURAL_FORWARD_PATH
            ),

            str(
                CONSTRUCTOR_COMPARISON_PATH
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
        CONSTRUCTOR_BINDING_PATH,
        COMPONENT_CALL_PATH,
        TOP_LEVEL_STATEMENT_PATH,
        FUNCTION_SUMMARY_PATH,
        TORCH_CAT_PATH,
        TREND_EVIDENCE_PATH,
        LOSS_EVIDENCE_PATH,
        PROCEDURAL_FORWARD_PATH,
        CONSTRUCTOR_COMPARISON_PATH,
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
        "PHASE 5.3.1h FINAL STATUS"
    )

    print(
        "Initialization authority:            "
    )

    print(
        f"  {INITIALIZATION_SOURCE_PATH}"
    )

    print()

    print(
        "Procedural forward authority candidate:"
    )

    print(
        f"  {FORWARD_SOURCE_PATH}"
    )

    print()

    print(
        f"Strongest orchestration scope:       "
        f"{strongest_scope}"
    )

    print(
        f"torch.cat calls:                     "
        f"{len(cat_df)}"
    )

    print(
        f"Trend evidence rows:                 "
        f"{len(trend_df)}"
    )

    print(
        f"BCE/autograd evidence rows:          "
        f"{len(loss_df)}"
    )

    print()

    print(
        "Procedural forward frozen:           NO"
    )

    print()

    print(
        "Phase-4 module imported:             NO"
    )

    print(
        "Model instantiated:                  NO"
    )

    print(
        "Training-negative RNG instantiated: NO"
    )

    print(
        "Training-order RNG instantiated:    NO"
    )

    print(
        "Optimizer instantiated:             NO"
    )

    print(
        "Forward computation performed:      NO"
    )

    print(
        "Backward computation performed:     NO"
    )

    print(
        "Optimizer steps:                    0"
    )

    banner(
        "PHASE 5.3.1h COMPLETE / "
        "PROCEDURAL FORWARD ORCHESTRATION AUDITED — "
        "NOT YET FROZEN"
    )


if __name__ == "__main__":
    main()