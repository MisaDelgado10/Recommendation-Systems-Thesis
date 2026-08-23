#!/usr/bin/env python3
"""
Phase 5.3.1j — Canonical Numerical Method Composition Proof and Freeze

Purpose
-------
Phase 5 has established two complementary Phase-4 implementation
authorities:

CANONICAL PARAMETER / INITIALIZATION AUTHORITY
----------------------------------------------
scripts/phase_4_7_1b_freeze_neural_initialization_seed_contract.py

This source determines:
- canonical model topology;
- canonical parameter namespace;
- 32 trainable parameter tensors;
- 19,217,929 trainable parameters;
- exact Kaiming initialization;
- neural seed 42;
- canonical initial-state SHA256.

FORWARD NUMERICAL METHOD AUTHORITY
----------------------------------
scripts/phase_4_6_2_end_to_end_itrs_forward_bce_audit.py

This source provides the exact numerical methods:

    DescriptionEncoder.forward
    TrendExtractor.attend_period
    TrendExtractor.encode_sequence
    BasisRGCNLayer.effective_weight
    BasisRGCNLayer.forward
    PreferencePropagation.forward
    ScoringMLP.forward

Phase 5.3.1i.1 proved that the only missing self.* runtime dependency
was:

    BasisRGCNLayer.out_dim

and classified it as PURE_DERIVED_METADATA.

Phase 5.3.1i.2 then froze the state-neutral bridge:

    layer.out_dim = int(layer.root_weight.shape[1])

for:

    preference_propagation.layer_1
    preference_propagation.layer_2

with exact proof that the bridge changes no neural state.

This phase now proves the METHOD COMPOSITION itself.

Method-composition policy
-------------------------
1. Load Phase-4.7.1b using the frozen SANITIZED_PREFIX_AST loader.
2. Instantiate canonical model using:
       build_canonical_model(seed=42)
3. Verify exact canonical state SHA256.
4. Apply the frozen R-GCN out_dim metadata bridge.
5. Build a definition-only Phase-4.6.2 support namespace.
6. Extract the seven exact Phase-4.6.2 method function objects.
7. Attach those exact methods to the canonical Phase-4.7.1b classes.
8. Verify all model instances resolve the attached methods.
9. Prove that no neural parameter/buffer/module/state_dict/tensor bytes
   changed.

NO forward call is made in this phase.

This separates:

    METHOD COMPOSITION

from:

    PROCEDURAL END-TO-END FORWARD EXECUTION

so any later numerical failure can be attributed cleanly.

THIS SCRIPT DOES:
- instantiate one canonical Phase-4.7.1b model;
- apply the already-frozen metadata bridge;
- construct a definition-only Phase-4.6.2 method namespace;
- attach exact Phase-4.6.2 numerical method objects;
- prove exact neural-state neutrality;
- freeze the composed numerical-method runtime.

THIS SCRIPT DOES NOT:
- normally import either Phase-4 executable script;
- execute either Phase-4 audit workflow;
- use Phase-4.6.2 audit-only initialization;
- instantiate Adam;
- instantiate Phase-5 training-negative RNG;
- instantiate Phase-5 training-order RNG;
- generate negatives;
- generate training order;
- perform a model forward computation;
- compute BCE;
- perform backward;
- call optimizer.step();
- save a model checkpoint;
- modify Phase-4 artifacts.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import random
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# =============================================================================
# Frozen canonical model source
# =============================================================================

CANONICAL_SOURCE_PATH = Path(
    "scripts/"
    "phase_4_7_1b_freeze_neural_initialization_seed_contract.py"
)

CANONICAL_SOURCE_SHA256 = (
    "c55f3ea1646cec7fdc8ef69f2310d98f"
    "5ee95fab77f0c48392f4a9f76612761c"
)


# =============================================================================
# Frozen forward-method source
# =============================================================================

FORWARD_SOURCE_PATH = Path(
    "scripts/"
    "phase_4_6_2_end_to_end_itrs_forward_bce_audit.py"
)

FORWARD_SOURCE_SHA256 = (
    "18c6c7ca4915fb23eab5ed39bae6eb49"
    "1a9332196f51b302a352c3c8211b053d"
)


# =============================================================================
# Frozen canonical numerical oracle
# =============================================================================

EXPECTED_INITIAL_STATE_SHA256 = (
    "49e822ea7fad35c458f47e134c94c05e"
    "ac099b68c5c468e2c71559c8c88998ab"
)

EXPECTED_PARAMETER_COUNT = 19_217_929
EXPECTED_PARAMETER_TENSORS = 32
EXPECTED_SEED = 42

REFERENCE_TORCH_VERSION_PREFIX = (
    "2.7.0"
)


# =============================================================================
# Frozen sanitized canonical runtime
# =============================================================================

EXPECTED_RUNTIME_AST_SHA256 = (
    "301a074aa57cfe7602f2ccbb5b8e26943"
    "b94b72e36efe4d60d1af48378c58a6e"
)

EXPECTED_WORKFLOW_BOUNDARY_INDEX = 56
EXPECTED_WORKFLOW_BOUNDARY_LINE = 1272


# =============================================================================
# Exact method composition
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

EXPECTED_METHOD_COUNT = 7


# =============================================================================
# Metadata bridge
# =============================================================================

EXPECTED_BRIDGE_MODULES = (
    "preference_propagation.layer_1",
    "preference_propagation.layer_2",
)

EXPECTED_OUT_DIM = 40


# =============================================================================
# Prior Phase-5 contracts
# =============================================================================

PHASE_5_3_1F_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_1f_side_effect_free_runtime_loading_contract.json"
)

PHASE_5_3_1I_2_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_1i_2_rgcn_runtime_metadata_bridge_contract.json"
)

PHASE_5_3_1I_2_MANIFEST_PATH = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_1i_2/"
    "phase_5_3_1i_2_metadata_bridge_manifest.json"
)


# =============================================================================
# Frozen Phase-4 artifacts
# =============================================================================

PHASE4_INIT_DIR = Path(
    "data/experimental/phase_4/"
    "initialization_contract"
)

PHASE4_FROZEN_ARTIFACTS = (
    PHASE4_INIT_DIR
    / "phase_4_7_1b_initialization_decision_audit.csv",

    PHASE4_INIT_DIR
    / "phase_4_7_1b_parameter_initialization_audit.csv",

    PHASE4_INIT_DIR
    / "phase_4_7_1b_seed_reproducibility_audit.csv",

    PHASE4_INIT_DIR
    / "phase_4_7_1b_initialization_state_hash.json",

    PHASE4_INIT_DIR
    / "phase_4_7_1b_neural_initialization_contract.json",

    PHASE4_INIT_DIR
    / "phase_4_7_1b_artifact_hashes.csv",
)


# =============================================================================
# Outputs
# =============================================================================

AUDIT_DIR = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_1j"
)

CONTRACT_DIR = Path(
    "data/experimental/phase_5/contracts"
)

METHOD_AST_REGISTRY_PATH = (
    AUDIT_DIR
    / "canonical_forward_method_ast_registry.csv"
)

FORWARD_SUPPORT_IMPORT_PATH = (
    AUDIT_DIR
    / "forward_runtime_support_import_registry.csv"
)

FORWARD_SUPPORT_STATIC_PATH = (
    AUDIT_DIR
    / "forward_runtime_support_static_registry.csv"
)

FORWARD_SUPPORT_HELPER_PATH = (
    AUDIT_DIR
    / "forward_runtime_support_helper_registry.csv"
)

METHOD_ATTACHMENT_PATH = (
    AUDIT_DIR
    / "canonical_method_attachment_audit.csv"
)

BOUND_METHOD_PATH = (
    AUDIT_DIR
    / "canonical_bound_method_resolution_audit.csv"
)

STATE_NEUTRALITY_PATH = (
    AUDIT_DIR
    / "method_composition_state_neutrality.csv"
)

PHASE4_IMMUTABILITY_PATH = (
    AUDIT_DIR
    / "phase4_artifact_immutability_audit.csv"
)

FINAL_INVARIANT_PATH = (
    AUDIT_DIR
    / "phase_5_3_1j_final_invariants.csv"
)

MANIFEST_PATH = (
    AUDIT_DIR
    / "phase_5_3_1j_method_composition_manifest.json"
)

CONTRACT_PATH = (
    CONTRACT_DIR
    / "phase_5_3_1j_canonical_numerical_method_composition_contract.json"
)

DECISION_REGISTER_PATH = (
    CONTRACT_DIR
    / "phase_5_3_1j_method_composition_decision_register.csv"
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

    print(text)

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
    value: str,
) -> str:

    return hashlib.sha256(
        value.encode(
            "utf-8"
        )
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


def tensor_sha256(
    tensor: torch.Tensor,
) -> str:

    value = (
        tensor
        .detach()
        .cpu()
        .contiguous()
    )

    digest = hashlib.sha256()

    digest.update(
        str(
            value.dtype
        ).encode(
            "utf-8"
        )
    )

    digest.update(
        str(
            tuple(
                value.shape
            )
        ).encode(
            "utf-8"
        )
    )

    digest.update(
        value.numpy().tobytes(
            order="C"
        )
    )

    return digest.hexdigest()


def parameter_tensor_hashes(
    model: torch.nn.Module,
) -> dict[str, str]:

    return {
        name: tensor_sha256(
            parameter
        )
        for name, parameter
        in model.named_parameters()
    }


def logical_state_dict_sha256(
    model: torch.nn.Module,
) -> str:

    digest = hashlib.sha256()

    for (
        name,
        tensor,
    ) in model.state_dict().items():

        digest.update(
            name.encode(
                "utf-8"
            )
        )

        digest.update(
            b"\0"
        )

        digest.update(
            tensor_sha256(
                tensor
            ).encode(
                "ascii"
            )
        )

        digest.update(
            b"\0"
        )

    return digest.hexdigest()


def snapshot_phase4_artifacts() -> dict[str, str]:

    result = {}

    for path in (
        PHASE4_FROZEN_ARTIFACTS
    ):

        require(
            path.exists(),
            (
                "Missing frozen Phase-4 artifact: "
                f"{path}"
            ),
        )

        result[
            str(path)
        ] = (
            sha256_file(
                path
            )
        )

    return result


def numpy_rng_state_equal(
    left,
    right,
) -> bool:

    return (
        left[0]
        == right[0]
        and np.array_equal(
            left[1],
            right[1],
        )
        and left[2:]
        == right[2:]
    )


# =============================================================================
# Source structure
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


def function_map(
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
            f"Multiple {class_node.name}."
            f"{method_name} definitions found."
        ),
    )

    if not matches:

        return None

    return matches[0]


# =============================================================================
# Canonical sanitized runtime loader
# =============================================================================

CANONICAL_RETAINED_NODE_TYPES = (
    ast.Import,
    ast.ImportFrom,
    ast.Assign,
    ast.AnnAssign,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
)


def canonical_retain_node(
    node: ast.AST,
) -> bool:

    return isinstance(
        node,
        CANONICAL_RETAINED_NODE_TYPES,
    )


def build_canonical_runtime_module(
    tree: ast.Module,
):
    """
    Reconstruct exactly the Phase-5.3.1f frozen canonical runtime
    definition payload.
    """

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

    prefix = (
        tree.body[
            :EXPECTED_WORKFLOW_BOUNDARY_INDEX
        ]
    )

    retained = [
        copy.deepcopy(
            node
        )
        for node
        in prefix
        if canonical_retain_node(
            node
        )
    ]

    sanitized_tree = ast.Module(
        body=retained,
        type_ignores=copy.deepcopy(
            tree.type_ignores
        ),
    )

    ast.fix_missing_locations(
        sanitized_tree
    )

    payload = ast.dump(
        sanitized_tree,
        annotate_fields=True,
        include_attributes=False,
    )

    runtime_ast_sha = (
        sha256_text(
            payload
        )
    )

    require(
        runtime_ast_sha
        == EXPECTED_RUNTIME_AST_SHA256,
        (
            "Frozen canonical runtime AST drift.\n"
            f"Expected: {EXPECTED_RUNTIME_AST_SHA256}\n"
            f"Actual:   {runtime_ast_sha}"
        ),
    )

    module_name = (
        "_itrs_phase5_3_1j_canonical_runtime"
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

    compiled = compile(
        sanitized_tree,
        filename=str(
            CANONICAL_SOURCE_PATH
        ),
        mode="exec",
    )

    exec(
        compiled,
        module.__dict__,
    )

    return (
        module,
        runtime_ast_sha,
    )


# =============================================================================
# Static assignment support for forward-method namespace
# =============================================================================

def top_level_assignment_map(
    tree: ast.Module,
) -> dict[str, ast.Assign | ast.AnnAssign]:

    result = {}

    for node in (
        tree.body
    ):

        if isinstance(
            node,
            ast.Assign,
        ):

            for target in (
                node.targets
            ):

                if isinstance(
                    target,
                    ast.Name,
                ):

                    result[
                        target.id
                    ] = node

        elif isinstance(
            node,
            ast.AnnAssign,
        ):

            if isinstance(
                node.target,
                ast.Name,
            ):

                result[
                    node.target.id
                ] = node

    return result


def assignment_value(
    node: ast.Assign | ast.AnnAssign,
) -> ast.AST | None:

    if isinstance(
        node,
        ast.Assign,
    ):

        return node.value

    return node.value


def loaded_names_from_expression(
    node: ast.AST | None,
) -> set[str]:

    if node is None:

        return set()

    return {
        candidate.id
        for candidate
        in ast.walk(
            node
        )
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


def contains_call(
    node: ast.AST | None,
) -> bool:

    if node is None:

        return False

    return any(
        isinstance(
            candidate,
            ast.Call,
        )
        for candidate
        in ast.walk(
            node
        )
    )


def safe_runtime_static_value(
    value,
) -> bool:
    """
    Only allow immutable/stateless configuration values.
    """

    if isinstance(
        value,
        (
            type(None),
            bool,
            int,
            float,
            str,
        ),
    ):

        return True

    if isinstance(
        value,
        tuple,
    ):

        return all(
            safe_runtime_static_value(
                item
            )
            for item in value
        )

    return False


# =============================================================================
# Forward-method global dependency analysis
# =============================================================================

BUILTIN_NAMES = set(
    dir(
        __builtins__
    )
)


def function_argument_names(
    function: ast.FunctionDef,
) -> set[str]:

    args = (
        function.args
    )

    result = {
        item.arg
        for item
        in (
            list(
                args.posonlyargs
            )
            + list(
                args.args
            )
            + list(
                args.kwonlyargs
            )
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


def function_stored_names(
    function: ast.FunctionDef,
) -> set[str]:

    result = set()

    for statement in (
        function.body
    ):

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

                result.add(
                    node.id
                )

    return result


def function_loaded_names(
    function: ast.FunctionDef,
) -> set[str]:

    result = set()

    for statement in (
        function.body
    ):

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

                result.add(
                    node.id
                )

    return result


def function_global_dependencies(
    function: ast.FunctionDef,
) -> set[str]:

    local_names = (
        function_argument_names(
            function
        )
        | function_stored_names(
            function
        )
        | {
            "self",
        }
    )

    return {
        name
        for name
        in function_loaded_names(
            function
        )
        if (
            name
            not in local_names
            and name
            not in BUILTIN_NAMES
        )
    }


# =============================================================================
# Forward support module
# =============================================================================

def execute_ast_nodes(
    module: types.ModuleType,
    nodes: list[ast.AST],
    filename: str,
) -> None:

    tree = ast.Module(
        body=[
            copy.deepcopy(
                node
            )
            for node in nodes
        ],
        type_ignores=[],
    )

    ast.fix_missing_locations(
        tree
    )

    compiled = compile(
        tree,
        filename=filename,
        mode="exec",
    )

    exec(
        compiled,
        module.__dict__,
    )


def build_forward_method_module(
    forward_tree: ast.Module,
):
    """
    Build a definition-only module containing:

    - exact Phase-4.6.2 imports;
    - only statically required immutable configuration assignments;
    - only required top-level helper functions;
    - exact selected Phase-4.6.2 class definitions.

    No top-level Phase-4.6.2 audit workflow is executed.
    No forward method is called.
    No Phase-4.6.2 class is instantiated.
    """

    module_name = (
        "_itrs_phase4_6_2_methods_phase5_3_1j"
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

    # -------------------------------------------------------------------------
    # Imports
    #
    # Import statements are definitions/runtime dependencies only.
    # We execute exact source import nodes, never the audit workflow.
    # -------------------------------------------------------------------------

    import_nodes = [
        node
        for node
        in forward_tree.body
        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
            ),
        )
    ]

    execute_ast_nodes(
        module=module,
        nodes=import_nodes,
        filename=str(
            FORWARD_SOURCE_PATH
        ),
    )

    import_rows = [
        {
            "line_number": (
                int(
                    node.lineno
                )
            ),
            "statement_type": (
                type(
                    node
                ).__name__
            ),
            "ast_sha256": (
                ast_sha256(
                    node
                )
            ),
            "source": (
                ast.unparse(
                    node
                )
            ),
        }
        for node in (
            import_nodes
        )
    ]

    # -------------------------------------------------------------------------
    # Source maps
    # -------------------------------------------------------------------------

    classes = (
        class_map(
            forward_tree
        )
    )

    helpers = (
        function_map(
            forward_tree
        )
    )

    assignments = (
        top_level_assignment_map(
            forward_tree
        )
    )

    # -------------------------------------------------------------------------
    # Verify all selected classes / methods.
    # -------------------------------------------------------------------------

    selected_method_nodes = []

    method_rows = []

    for (
        class_name,
        method_names,
    ) in METHOD_GRAFT_SPEC.items():

        require(
            class_name
            in classes,
            (
                "Forward source missing class "
                f"{class_name}."
            ),
        )

        class_node = (
            classes[
                class_name
            ]
        )

        for method_name in (
            method_names
        ):

            method = (
                direct_method(
                    class_node,
                    method_name,
                )
            )

            require(
                method
                is not None,
                (
                    "Forward source missing "
                    f"{class_name}.{method_name}."
                ),
            )

            selected_method_nodes.append(
                (
                    class_name,
                    method_name,
                    method,
                )
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
                        ast_sha256(
                            method
                        )
                    ),
                    "global_dependencies": (
                        ";".join(
                            sorted(
                                function_global_dependencies(
                                    method
                                )
                            )
                        )
                    ),
                }
            )

    require(
        len(
            selected_method_nodes
        )
        == EXPECTED_METHOD_COUNT,
        (
            "Unexpected exact-method count."
        ),
    )

    # -------------------------------------------------------------------------
    # Runtime-support dependency closure.
    #
    # Imports are already available.
    # Resolve only any globals not supplied by those imports.
    # -------------------------------------------------------------------------

    executed_static_names = set()

    static_rows = []

    helper_rows = []

    helper_compile_stack = set()

    def ensure_static_assignment(
        name: str,
    ) -> bool:

        if name in module.__dict__:

            return True

        if name not in assignments:

            return False

        if name in executed_static_names:

            return True

        assignment = (
            assignments[
                name
            ]
        )

        value_node = (
            assignment_value(
                assignment
            )
        )

        require(
            value_node
            is not None,
            (
                f"Forward runtime assignment "
                f"{name} has no value."
            ),
        )

        require(
            not contains_call(
                value_node
            ),
            (
                "Required forward-runtime static "
                f"assignment {name} contains a Call and "
                "cannot be executed as configuration-only "
                "support."
            ),
        )

        dependencies = (
            loaded_names_from_expression(
                value_node
            )
        )

        for dependency in sorted(
            dependencies
        ):

            if dependency in module.__dict__:

                continue

            resolved = (
                ensure_static_assignment(
                    dependency
                )
            )

            require(
                resolved,
                (
                    "Cannot statically resolve dependency "
                    f"{dependency} required by "
                    f"forward-runtime assignment {name}."
                ),
            )

        execute_ast_nodes(
            module=module,
            nodes=[
                assignment,
            ],
            filename=str(
                FORWARD_SOURCE_PATH
            ),
        )

        require(
            name
            in module.__dict__,
            (
                f"Static assignment {name} did "
                "not create runtime symbol."
            ),
        )

        value = (
            module.__dict__[
                name
            ]
        )

        require(
            safe_runtime_static_value(
                value
            ),
            (
                f"Forward-runtime static symbol {name} "
                "resolved to non-immutable value of type "
                f"{type(value).__name__}."
            ),
        )

        executed_static_names.add(
            name
        )

        static_rows.append(
            {
                "name": (
                    name
                ),
                "line_number": (
                    int(
                        assignment.lineno
                    )
                ),
                "value_repr": (
                    repr(
                        value
                    )
                ),
                "python_type": (
                    type(
                        value
                    ).__name__
                ),
                "assignment_ast_sha256": (
                    ast_sha256(
                        assignment
                    )
                ),
                "status": (
                    "PASS"
                ),
            }
        )

        return True

    compiled_helpers = set()

    def ensure_helper(
        name: str,
    ) -> bool:

        if name in module.__dict__:

            return True

        if name not in helpers:

            return False

        if name in compiled_helpers:

            return True

        require(
            name
            not in helper_compile_stack,
            (
                "Recursive helper dependency detected: "
                f"{name}"
            ),
        )

        helper_compile_stack.add(
            name
        )

        helper = (
            helpers[
                name
            ]
        )

        dependencies = (
            function_global_dependencies(
                helper
            )
        )

        for dependency in sorted(
            dependencies
        ):

            if dependency in module.__dict__:

                continue

            if ensure_static_assignment(
                dependency
            ):

                continue

            if ensure_helper(
                dependency
            ):

                continue

            raise AssertionError(
                "Cannot resolve forward-runtime helper "
                f"dependency {dependency} required by "
                f"{name}."
            )

        execute_ast_nodes(
            module=module,
            nodes=[
                helper,
            ],
            filename=str(
                FORWARD_SOURCE_PATH
            ),
        )

        require(
            name
            in module.__dict__,
            (
                f"Helper {name} failed to compile."
            ),
        )

        compiled_helpers.add(
            name
        )

        helper_compile_stack.remove(
            name
        )

        helper_rows.append(
            {
                "name": (
                    name
                ),
                "line_number": (
                    int(
                        helper.lineno
                    )
                ),
                "helper_ast_sha256": (
                    ast_sha256(
                        helper
                    )
                ),
                "global_dependencies": (
                    ";".join(
                        sorted(
                            dependencies
                        )
                    )
                ),
                "status": (
                    "PASS"
                ),
            }
        )

        return True

    # -------------------------------------------------------------------------
    # Resolve body globals for all selected methods.
    # -------------------------------------------------------------------------

    unresolved = []

    for (
        class_name,
        method_name,
        method,
    ) in selected_method_nodes:

        for dependency in sorted(
            function_global_dependencies(
                method
            )
        ):

            if dependency in module.__dict__:

                continue

            if ensure_static_assignment(
                dependency
            ):

                continue

            if ensure_helper(
                dependency
            ):

                continue

            unresolved.append(
                (
                    class_name,
                    method_name,
                    dependency,
                )
            )

    require(
        len(
            unresolved
        )
        == 0,
        (
            "Unresolved Phase-4.6.2 method globals:\n"
            + "\n".join(
                (
                    f"  {class_name}.{method_name}: "
                    f"{dependency}"
                )
                for (
                    class_name,
                    method_name,
                    dependency,
                ) in unresolved
            )
        ),
    )

    # -------------------------------------------------------------------------
    # Compile EXACT selected Phase-4.6.2 class definitions.
    #
    # We instantiate NONE of these classes.
    #
    # Extracted methods retain their original class qualname and exact
    # function body while their globals point to this definition-only
    # forward-runtime namespace.
    # -------------------------------------------------------------------------

    selected_class_nodes = [
        classes[
            class_name
        ]
        for class_name
        in METHOD_GRAFT_SPEC
    ]

    execute_ast_nodes(
        module=module,
        nodes=selected_class_nodes,
        filename=str(
            FORWARD_SOURCE_PATH
        ),
    )

    # -------------------------------------------------------------------------
    # Extract exact method objects.
    # -------------------------------------------------------------------------

    methods = {}

    for (
        class_name,
        method_names,
    ) in METHOD_GRAFT_SPEC.items():

        require(
            class_name
            in module.__dict__,
            (
                "Definition-only forward namespace "
                f"did not create class {class_name}."
            ),
        )

        support_class = (
            module.__dict__[
                class_name
            ]
        )

        for method_name in (
            method_names
        ):

            require(
                method_name
                in support_class.__dict__,
                (
                    "Definition-only forward class "
                    f"{class_name} does not expose "
                    f"{method_name}."
                ),
            )

            method_object = (
                support_class.__dict__[
                    method_name
                ]
            )

            require(
                callable(
                    method_object
                ),
                (
                    f"{class_name}.{method_name} "
                    "is not callable."
                ),
            )

            methods[
                (
                    class_name,
                    method_name,
                )
            ] = (
                method_object
            )

    return (
        module,
        methods,
        pd.DataFrame(
            method_rows
        ),
        pd.DataFrame(
            import_rows
        ),
        pd.DataFrame(
            static_rows
        ),
        pd.DataFrame(
            helper_rows
        ),
    )


# =============================================================================
# Apply already-frozen metadata bridge
# =============================================================================

def apply_frozen_metadata_bridge(
    model: torch.nn.Module,
    runtime_module,
) -> pd.DataFrame:

    basis_class = getattr(
        runtime_module,
        "BasisRGCNLayer",
    )

    basis_instances = [
        (
            name,
            module,
        )
        for (
            name,
            module,
        ) in model.named_modules()
        if isinstance(
            module,
            basis_class,
        )
    ]

    require(
        len(
            basis_instances
        )
        == 2,
        (
            "Expected exactly two canonical "
            "BasisRGCNLayer instances."
        ),
    )

    actual_names = tuple(
        name
        for name, _
        in basis_instances
    )

    require(
        actual_names
        == EXPECTED_BRIDGE_MODULES,
        (
            "Canonical R-GCN module path drift.\n"
            f"Expected: {EXPECTED_BRIDGE_MODULES}\n"
            f"Actual:   {actual_names}"
        ),
    )

    rows = []

    for (
        name,
        layer,
    ) in basis_instances:

        require(
            not hasattr(
                layer,
                "out_dim",
            ),
            (
                f"{name} unexpectedly already "
                "contains out_dim."
            ),
        )

        from_root = int(
            layer.root_weight.shape[
                1
            ]
        )

        from_bases = int(
            layer.bases.shape[
                2
            ]
        )

        require(
            from_root
            == from_bases
            == EXPECTED_OUT_DIM,
            (
                f"{name} out_dim derivation drift."
            ),
        )

        setattr(
            layer,
            "out_dim",
            from_root,
        )

        require(
            isinstance(
                layer.out_dim,
                int,
            ),
            (
                f"{name}.out_dim is not Python int."
            ),
        )

        require(
            "out_dim"
            not in layer._parameters,
            (
                f"{name}.out_dim registered as parameter."
            ),
        )

        require(
            "out_dim"
            not in layer._buffers,
            (
                f"{name}.out_dim registered as buffer."
            ),
        )

        require(
            "out_dim"
            not in layer._modules,
            (
                f"{name}.out_dim registered as module."
            ),
        )

        rows.append(
            {
                "module_name": (
                    name
                ),
                "out_dim": (
                    int(
                        layer.out_dim
                    )
                ),
                "status": (
                    "PASS"
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.3.1j — "
        "CANONICAL NUMERICAL METHOD COMPOSITION PROOF AND FREEZE"
    )

    print(
        "Canonical model instantiated:         YES"
    )

    print(
        "Frozen metadata bridge applied:       YES"
    )

    print(
        "Exact Phase-4.6.2 methods attached:   YES — proof target"
    )

    print(
        "Forward computation performed:        NO"
    )

    print(
        "BCE computed:                         NO"
    )

    print(
        "Backward computation performed:       NO"
    )

    print(
        "Optimizer instantiated:               NO"
    )

    print(
        "Optimizer steps:                      0"
    )

    # =========================================================================
    # Inputs
    # =========================================================================

    banner(
        "AUTHORITATIVE CONTRACT / SOURCE RECHECK"
    )

    for path in (
        CANONICAL_SOURCE_PATH,
        FORWARD_SOURCE_PATH,
        PHASE_5_3_1F_CONTRACT_PATH,
        PHASE_5_3_1I_2_CONTRACT_PATH,
        PHASE_5_3_1I_2_MANIFEST_PATH,
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

    canonical_sha = (
        sha256_file(
            CANONICAL_SOURCE_PATH
        )
    )

    forward_sha = (
        sha256_file(
            FORWARD_SOURCE_PATH
        )
    )

    require(
        canonical_sha
        == CANONICAL_SOURCE_SHA256,
        (
            "Canonical model source SHA drift."
        ),
    )

    require(
        forward_sha
        == FORWARD_SOURCE_SHA256,
        (
            "Forward method source SHA drift."
        ),
    )

    loader_contract = (
        load_json(
            PHASE_5_3_1F_CONTRACT_PATH
        )
    )

    bridge_contract = (
        load_json(
            PHASE_5_3_1I_2_CONTRACT_PATH
        )
    )

    bridge_manifest = (
        load_json(
            PHASE_5_3_1I_2_MANIFEST_PATH
        )
    )

    require(
        loader_contract[
            "status"
        ]
        == "FROZEN",
        (
            "Phase-5.3.1f loader not frozen."
        ),
    )

    require(
        bridge_contract[
            "status"
        ]
        == "FROZEN",
        (
            "Phase-5.3.1i.2 metadata bridge "
            "not frozen."
        ),
    )

    require(
        bridge_manifest[
            "status"
        ]
        == (
            "RGCN_RUNTIME_METADATA_BRIDGE_"
            "PROVED_AND_FROZEN"
        ),
        (
            "Unexpected Phase-5.3.1i.2 manifest status."
        ),
    )

    require(
        bridge_manifest[
            "canonical_state_sha256_after"
        ]
        == EXPECTED_INITIAL_STATE_SHA256,
        (
            "Phase-5.3.1i.2 canonical state "
            "hash drift."
        ),
    )

    require(
        bridge_manifest[
            "optimizer_instantiated"
        ]
        is False,
        (
            "Optimizer unexpectedly instantiated "
            "before 5.3.1j."
        ),
    )

    require(
        int(
            bridge_manifest[
                "optimizer_steps"
            ]
        )
        == 0,
        (
            "Optimizer step occurred before 5.3.1j."
        ),
    )

    print(
        "Canonical source SHA256:              PASS"
    )

    print(
        "Forward source SHA256:                PASS"
    )

    print(
        "Runtime loader:                       FROZEN"
    )

    print(
        "R-GCN metadata bridge:                FROZEN"
    )

    print(
        "Optimizer steps entering phase:       0"
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
    # Parse sources
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

    # =========================================================================
    # Frozen Phase-4 artifact snapshot
    # =========================================================================

    phase4_before = (
        snapshot_phase4_artifacts()
    )

    # =========================================================================
    # Load canonical runtime and model
    # =========================================================================

    banner(
        "LOAD AND INSTANTIATE CANONICAL MODEL"
    )

    (
        canonical_runtime,
        canonical_runtime_ast_sha,
    ) = build_canonical_runtime_module(
        canonical_tree
    )

    builder = getattr(
        canonical_runtime,
        "build_canonical_model",
    )

    canonical_hash_fn = getattr(
        canonical_runtime,
        "model_parameter_state_sha256",
    )

    model = builder(
        seed=EXPECTED_SEED
    )

    require(
        canonical_hash_fn(
            model
        )
        == EXPECTED_INITIAL_STATE_SHA256,
        (
            "Canonical model state hash mismatch."
        ),
    )

    bridge_df = (
        apply_frozen_metadata_bridge(
            model,
            canonical_runtime,
        )
    )

    require(
        canonical_hash_fn(
            model
        )
        == EXPECTED_INITIAL_STATE_SHA256,
        (
            "Metadata bridge altered canonical "
            "state before method composition."
        ),
    )

    print(
        "Canonical model SHA256:"
    )

    print(
        canonical_hash_fn(
            model
        )
    )

    print()

    print(
        "Frozen metadata bridge:"
    )

    print(
        bridge_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # State snapshot before method composition
    # =========================================================================

    before_hash = (
        canonical_hash_fn(
            model
        )
    )

    before_state_dict_hash = (
        logical_state_dict_sha256(
            model
        )
    )

    before_parameter_names = tuple(
        name
        for name, _
        in model.named_parameters()
    )

    before_parameter_hashes = (
        parameter_tensor_hashes(
            model
        )
    )

    before_buffer_names = tuple(
        name
        for name, _
        in model.named_buffers()
    )

    before_module_names = tuple(
        name
        for name, _
        in model.named_modules()
    )

    before_state_keys = tuple(
        model.state_dict().keys()
    )

    before_parameter_tensor_count = sum(
        1
        for _
        in model.parameters()
    )

    before_trainable_count = sum(
        int(
            parameter.numel()
        )
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    require(
        before_parameter_tensor_count
        == EXPECTED_PARAMETER_TENSORS,
        (
            "Canonical parameter tensor count drift."
        ),
    )

    require(
        before_trainable_count
        == EXPECTED_PARAMETER_COUNT,
        (
            "Canonical parameter count drift."
        ),
    )

    # =========================================================================
    # RNG snapshot around support-definition + method graft
    # =========================================================================

    python_rng_before = (
        random.getstate()
    )

    numpy_rng_before = (
        np.random.get_state()
    )

    torch_rng_before = (
        torch.get_rng_state().clone()
    )

    # =========================================================================
    # Build exact Phase-4.6.2 method-definition namespace
    # =========================================================================

    banner(
        "BUILD DEFINITION-ONLY PHASE-4.6.2 METHOD NAMESPACE"
    )

    (
        forward_runtime,
        exact_methods,
        method_registry_df,
        import_registry_df,
        static_registry_df,
        helper_registry_df,
    ) = build_forward_method_module(
        forward_tree
    )

    require(
        len(
            exact_methods
        )
        == EXPECTED_METHOD_COUNT,
        (
            "Expected exactly seven exact "
            "Phase-4.6.2 numerical methods."
        ),
    )

    print(
        "Exact numerical methods:"
    )

    print(
        method_registry_df[
            [
                "class_name",
                "method_name",
                "line_number",
                "method_ast_sha256",
                "global_dependencies",
            ]
        ].to_string(
            index=False
        )
    )

    if not (
        static_registry_df.empty
    ):

        print()

        print(
            "Additional immutable forward-runtime configuration:"
        )

        print(
            static_registry_df.to_string(
                index=False
            )
        )

    if not (
        helper_registry_df.empty
    ):

        print()

        print(
            "Additional exact forward-runtime helpers:"
        )

        print(
            helper_registry_df.to_string(
                index=False
            )
        )

    # =========================================================================
    # Attach exact methods to canonical classes
    # =========================================================================

    banner(
        "ATTACH EXACT PHASE-4.6.2 NUMERICAL METHODS"
    )

    attachment_rows = []

    canonical_classes = (
        class_map(
            canonical_tree
        )
    )

    for (
        class_name,
        method_names,
    ) in METHOD_GRAFT_SPEC.items():

        require(
            class_name
            in canonical_classes,
            (
                "Canonical source missing class "
                f"{class_name}."
            ),
        )

        canonical_class = getattr(
            canonical_runtime,
            class_name,
        )

        for method_name in (
            method_names
        ):

            method_object = (
                exact_methods[
                    (
                        class_name,
                        method_name,
                    )
                ]
            )

            source_class = (
                class_map(
                    forward_tree
                )[
                    class_name
                ]
            )

            source_method = (
                direct_method(
                    source_class,
                    method_name,
                )
            )

            require(
                source_method
                is not None,
                (
                    "Unexpected missing source method."
                ),
            )

            source_method_hash = (
                ast_sha256(
                    source_method
                )
            )

            setattr(
                canonical_class,
                method_name,
                method_object,
            )

            resolved = getattr(
                canonical_class,
                method_name,
            )

            require(
                resolved
                is method_object,
                (
                    "Canonical class did not retain exact "
                    f"method object for "
                    f"{class_name}.{method_name}."
                ),
            )

            require(
                resolved.__module__
                == forward_runtime.__name__,
                (
                    "Attached method global-module provenance "
                    f"drift for {class_name}.{method_name}."
                ),
            )

            attachment_rows.append(
                {
                    "class_name": (
                        class_name
                    ),

                    "method_name": (
                        method_name
                    ),

                    "source_method_ast_sha256": (
                        source_method_hash
                    ),

                    "attached_function_module": (
                        resolved.__module__
                    ),

                    "attached_function_qualname": (
                        resolved.__qualname__
                    ),

                    "same_exact_function_object": (
                        resolved
                        is method_object
                    ),

                    "status": (
                        "PASS"
                    ),
                }
            )

    attachment_df = pd.DataFrame(
        attachment_rows
    )

    require(
        len(
            attachment_df
        )
        == EXPECTED_METHOD_COUNT,
        (
            "Unexpected attached-method count."
        ),
    )

    print(
        attachment_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Bound method resolution against actual canonical model instance
    # =========================================================================

    banner(
        "CANONICAL MODEL BOUND-METHOD RESOLUTION"
    )

    bound_checks = [
        (
            "description_encoder",
            model.description_encoder,
            "DescriptionEncoder",
            "forward",
        ),

        (
            "trend_extractor.attend_period",
            model.trend_extractor,
            "TrendExtractor",
            "attend_period",
        ),

        (
            "trend_extractor.encode_sequence",
            model.trend_extractor,
            "TrendExtractor",
            "encode_sequence",
        ),

        (
            "preference_propagation.layer_1.effective_weight",
            model.preference_propagation.layer_1,
            "BasisRGCNLayer",
            "effective_weight",
        ),

        (
            "preference_propagation.layer_1.forward",
            model.preference_propagation.layer_1,
            "BasisRGCNLayer",
            "forward",
        ),

        (
            "preference_propagation.layer_2.effective_weight",
            model.preference_propagation.layer_2,
            "BasisRGCNLayer",
            "effective_weight",
        ),

        (
            "preference_propagation.layer_2.forward",
            model.preference_propagation.layer_2,
            "BasisRGCNLayer",
            "forward",
        ),

        (
            "preference_propagation.forward",
            model.preference_propagation,
            "PreferencePropagation",
            "forward",
        ),

        (
            "scoring_mlp.forward",
            model.scoring_mlp,
            "ScoringMLP",
            "forward",
        ),
    ]

    bound_rows = []

    for (
        runtime_path,
        instance,
        class_name,
        method_name,
    ) in bound_checks:

        bound_method = getattr(
            instance,
            method_name,
        )

        expected_function = (
            exact_methods[
                (
                    class_name,
                    method_name,
                )
            ]
        )

        require(
            hasattr(
                bound_method,
                "__self__",
            ),
            (
                f"{runtime_path} did not resolve "
                "as bound method."
            ),
        )

        require(
            hasattr(
                bound_method,
                "__func__",
            ),
            (
                f"{runtime_path} lacks __func__."
            ),
        )

        require(
            bound_method.__self__
            is instance,
            (
                f"{runtime_path} bound-self mismatch."
            ),
        )

        require(
            bound_method.__func__
            is expected_function,
            (
                f"{runtime_path} does not resolve the "
                "exact frozen Phase-4.6.2 function."
            ),
        )

        bound_rows.append(
            {
                "runtime_path": (
                    runtime_path
                ),

                "class_name": (
                    class_name
                ),

                "method_name": (
                    method_name
                ),

                "bound_self_exact": (
                    True
                ),

                "exact_function_object": (
                    True
                ),

                "function_module": (
                    bound_method.__func__.__module__
                ),

                "function_qualname": (
                    bound_method.__func__.__qualname__
                ),

                "status": (
                    "PASS"
                ),
            }
        )

    bound_df = pd.DataFrame(
        bound_rows
    )

    print(
        bound_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # RNG neutrality after definition + graft
    # =========================================================================

    python_rng_after = (
        random.getstate()
    )

    numpy_rng_after = (
        np.random.get_state()
    )

    torch_rng_after = (
        torch.get_rng_state().clone()
    )

    require(
        python_rng_before
        == python_rng_after,
        (
            "Method composition altered Python RNG."
        ),
    )

    require(
        numpy_rng_state_equal(
            numpy_rng_before,
            numpy_rng_after,
        ),
        (
            "Method composition altered NumPy RNG."
        ),
    )

    require(
        torch.equal(
            torch_rng_before,
            torch_rng_after,
        ),
        (
            "Method composition altered torch global RNG."
        ),
    )

    # =========================================================================
    # Post-graft neural state neutrality
    # =========================================================================

    banner(
        "POST-COMPOSITION NEURAL STATE NEUTRALITY"
    )

    after_hash = (
        canonical_hash_fn(
            model
        )
    )

    after_state_dict_hash = (
        logical_state_dict_sha256(
            model
        )
    )

    after_parameter_names = tuple(
        name
        for name, _
        in model.named_parameters()
    )

    after_parameter_hashes = (
        parameter_tensor_hashes(
            model
        )
    )

    after_buffer_names = tuple(
        name
        for name, _
        in model.named_buffers()
    )

    after_module_names = tuple(
        name
        for name, _
        in model.named_modules()
    )

    after_state_keys = tuple(
        model.state_dict().keys()
    )

    after_parameter_tensor_count = sum(
        1
        for _
        in model.parameters()
    )

    after_trainable_count = sum(
        int(
            parameter.numel()
        )
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    neutrality_rows = [
        {
            "check": (
                "canonical_parameter_state_sha256"
            ),

            "before": (
                before_hash
            ),

            "after": (
                after_hash
            ),

            "unchanged": (
                before_hash
                == after_hash
                == EXPECTED_INITIAL_STATE_SHA256
            ),
        },

        {
            "check": (
                "logical_state_dict_sha256"
            ),

            "before": (
                before_state_dict_hash
            ),

            "after": (
                after_state_dict_hash
            ),

            "unchanged": (
                before_state_dict_hash
                == after_state_dict_hash
            ),
        },

        {
            "check": (
                "parameter_names"
            ),

            "before": (
                str(
                    before_parameter_names
                )
            ),

            "after": (
                str(
                    after_parameter_names
                )
            ),

            "unchanged": (
                before_parameter_names
                == after_parameter_names
            ),
        },

        {
            "check": (
                "parameter_tensor_bytes"
            ),

            "before": (
                "per-parameter SHA256 registry"
            ),

            "after": (
                "per-parameter SHA256 registry"
            ),

            "unchanged": (
                before_parameter_hashes
                == after_parameter_hashes
            ),
        },

        {
            "check": (
                "buffer_names"
            ),

            "before": (
                str(
                    before_buffer_names
                )
            ),

            "after": (
                str(
                    after_buffer_names
                )
            ),

            "unchanged": (
                before_buffer_names
                == after_buffer_names
            ),
        },

        {
            "check": (
                "module_names"
            ),

            "before": (
                str(
                    before_module_names
                )
            ),

            "after": (
                str(
                    after_module_names
                )
            ),

            "unchanged": (
                before_module_names
                == after_module_names
            ),
        },

        {
            "check": (
                "state_dict_keys"
            ),

            "before": (
                str(
                    before_state_keys
                )
            ),

            "after": (
                str(
                    after_state_keys
                )
            ),

            "unchanged": (
                before_state_keys
                == after_state_keys
            ),
        },

        {
            "check": (
                "parameter_tensor_count"
            ),

            "before": (
                str(
                    before_parameter_tensor_count
                )
            ),

            "after": (
                str(
                    after_parameter_tensor_count
                )
            ),

            "unchanged": (
                before_parameter_tensor_count
                == after_parameter_tensor_count
                == EXPECTED_PARAMETER_TENSORS
            ),
        },

        {
            "check": (
                "trainable_parameter_count"
            ),

            "before": (
                str(
                    before_trainable_count
                )
            ),

            "after": (
                str(
                    after_trainable_count
                )
            ),

            "unchanged": (
                before_trainable_count
                == after_trainable_count
                == EXPECTED_PARAMETER_COUNT
            ),
        },
    ]

    neutrality_df = pd.DataFrame(
        neutrality_rows
    )

    require(
        neutrality_df[
            "unchanged"
        ].all(),
        (
            "Method composition changed canonical "
            "neural state."
        ),
    )

    print(
        neutrality_df.to_string(
            index=False
        )
    )

    print()

    print(
        "Canonical SHA256 after method composition:"
    )

    print(
        after_hash
    )

    # =========================================================================
    # Phase-4 artifact immutability
    # =========================================================================

    banner(
        "PHASE-4 ARTIFACT IMMUTABILITY"
    )

    phase4_after = (
        snapshot_phase4_artifacts()
    )

    artifact_rows = []

    for path in (
        phase4_before
    ):

        before_sha = (
            phase4_before[
                path
            ]
        )

        after_sha = (
            phase4_after[
                path
            ]
        )

        artifact_rows.append(
            {
                "path": (
                    path
                ),

                "before_sha256": (
                    before_sha
                ),

                "after_sha256": (
                    after_sha
                ),

                "unchanged": (
                    before_sha
                    == after_sha
                ),
            }
        )

    artifact_df = pd.DataFrame(
        artifact_rows
    )

    require(
        artifact_df[
            "unchanged"
        ].all(),
        (
            "Method composition modified frozen "
            "Phase-4 artifacts."
        ),
    )

    print(
        artifact_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Final invariants
    # =========================================================================

    banner(
        "FINAL PHASE-5.3.1j FREEZE INVARIANTS"
    )

    checks = [
        (
            "canonical_source_sha256_exact",
            (
                canonical_sha
                == CANONICAL_SOURCE_SHA256
            ),
        ),

        (
            "forward_source_sha256_exact",
            (
                forward_sha
                == FORWARD_SOURCE_SHA256
            ),
        ),

        (
            "canonical_runtime_ast_exact",
            (
                canonical_runtime_ast_sha
                == EXPECTED_RUNTIME_AST_SHA256
            ),
        ),

        (
            "metadata_bridge_contract_frozen",
            (
                bridge_contract[
                    "status"
                ]
                == "FROZEN"
            ),
        ),

        (
            "canonical_hash_before_composition_exact",
            (
                before_hash
                == EXPECTED_INITIAL_STATE_SHA256
            ),
        ),

        (
            "seven_exact_forward_methods_resolved",
            (
                len(
                    exact_methods
                )
                == EXPECTED_METHOD_COUNT
            ),
        ),

        (
            "seven_exact_forward_methods_attached",
            (
                len(
                    attachment_df
                )
                == EXPECTED_METHOD_COUNT
            ),
        ),

        (
            "all_attached_functions_are_exact_objects",
            bool(
                attachment_df[
                    "same_exact_function_object"
                ].all()
            ),
        ),

        (
            "all_expected_bound_methods_resolve",
            bool(
                (
                    bound_df[
                        "status"
                    ]
                    == "PASS"
                ).all()
            ),
        ),

        (
            "method_composition_python_rng_neutral",
            (
                python_rng_before
                == python_rng_after
            ),
        ),

        (
            "method_composition_numpy_rng_neutral",
            numpy_rng_state_equal(
                numpy_rng_before,
                numpy_rng_after,
            ),
        ),

        (
            "method_composition_torch_rng_neutral",
            torch.equal(
                torch_rng_before,
                torch_rng_after,
            ),
        ),

        (
            "canonical_hash_after_composition_exact",
            (
                after_hash
                == EXPECTED_INITIAL_STATE_SHA256
            ),
        ),

        (
            "all_neural_state_neutrality_checks_pass",
            bool(
                neutrality_df[
                    "unchanged"
                ].all()
            ),
        ),

        (
            "phase4_frozen_artifacts_unchanged",
            bool(
                artifact_df[
                    "unchanged"
                ].all()
            ),
        ),

        (
            "no_forward_computation",
            True,
        ),

        (
            "no_BCE_computation",
            True,
        ),

        (
            "no_backward_computation",
            True,
        ),

        (
            "optimizer_not_instantiated",
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
            "At least one Phase-5.3.1j "
            "method-composition invariant failed."
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

    banner(
        "WRITE AND FREEZE CANONICAL METHOD COMPOSITION"
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTRACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    method_registry_df.to_csv(
        METHOD_AST_REGISTRY_PATH,
        index=False,
    )

    import_registry_df.to_csv(
        FORWARD_SUPPORT_IMPORT_PATH,
        index=False,
    )

    static_registry_df.to_csv(
        FORWARD_SUPPORT_STATIC_PATH,
        index=False,
    )

    helper_registry_df.to_csv(
        FORWARD_SUPPORT_HELPER_PATH,
        index=False,
    )

    attachment_df.to_csv(
        METHOD_ATTACHMENT_PATH,
        index=False,
    )

    bound_df.to_csv(
        BOUND_METHOD_PATH,
        index=False,
    )

    neutrality_df.to_csv(
        STATE_NEUTRALITY_PATH,
        index=False,
    )

    artifact_df.to_csv(
        PHASE4_IMMUTABILITY_PATH,
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
                    "canonical_parameter_initialization_runtime"
                ),

                "value": (
                    "PHASE_4_7_1B_SANITIZED_PREFIX_AST"
                ),

                "classification": (
                    "INHERITED_FROZEN_PHASE4_IMPLEMENTATION"
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1j"
                ),
            },

            {
                "decision": (
                    "BasisRGCNLayer_out_dim_metadata_bridge"
                ),

                "value": (
                    "int(layer.root_weight.shape[1])"
                ),

                "classification": (
                    "INHERITED_PHASE_5_3_1i_2_"
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1j"
                ),
            },

            {
                "decision": (
                    "canonical_numerical_method_source"
                ),

                "value": (
                    str(
                        FORWARD_SOURCE_PATH
                    )
                ),

                "classification": (
                    "INHERITED_PHASE4_FORWARD_RUNTIME_PROVENANCE"
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1j"
                ),
            },

            {
                "decision": (
                    "canonical_numerical_method_count"
                ),

                "value": (
                    EXPECTED_METHOD_COUNT
                ),

                "classification": (
                    "IMPLEMENTATION_INTEGRITY_FINGERPRINT"
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1j"
                ),
            },

            {
                "decision": (
                    "method_composition_state_oracle"
                ),

                "value": (
                    EXPECTED_INITIAL_STATE_SHA256
                ),

                "classification": (
                    "INHERITED_FROZEN_PHASE4_NUMERICAL_ORACLE"
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1j"
                ),
            },
        ]
    )

    decision_df.to_csv(
        DECISION_REGISTER_PATH,
        index=False,
    )

    method_contract_records = (
        method_registry_df[
            [
                "class_name",
                "method_name",
                "line_number",
                "method_ast_sha256",
                "global_dependencies",
            ]
        ]
        .to_dict(
            orient="records"
        )
    )

    contract = {
        "phase": (
            "5.3.1j"
        ),

        "title": (
            "Canonical Numerical Method "
            "Composition Contract"
        ),

        "status": (
            "FROZEN"
        ),

        "canonical_model": {
            "source": (
                str(
                    CANONICAL_SOURCE_PATH
                )
            ),

            "source_sha256": (
                canonical_sha
            ),

            "runtime_loader": (
                "SANITIZED_PREFIX_AST"
            ),

            "runtime_ast_sha256": (
                canonical_runtime_ast_sha
            ),

            "builder": (
                "build_canonical_model(seed=42)"
            ),

            "canonical_initial_state_sha256": (
                EXPECTED_INITIAL_STATE_SHA256
            ),
        },

        "metadata_bridge": {
            "contract": (
                str(
                    PHASE_5_3_1I_2_CONTRACT_PATH
                )
            ),

            "modules": (
                list(
                    EXPECTED_BRIDGE_MODULES
                )
            ),

            "attribute": (
                "out_dim"
            ),

            "derivation": (
                "int(layer.root_weight.shape[1])"
            ),

            "expected_value": (
                EXPECTED_OUT_DIM
            ),
        },

        "forward_methods": {
            "source": (
                str(
                    FORWARD_SOURCE_PATH
                )
            ),

            "source_sha256": (
                forward_sha
            ),

            "method_count": (
                EXPECTED_METHOD_COUNT
            ),

            "methods": (
                method_contract_records
            ),

            "composition": (
                "Attach exact Phase-4.6.2 function "
                "objects to canonical Phase-4.7.1b "
                "classes."
            ),
        },

        "state_neutrality": {
            "parameter_tensor_count": (
                after_parameter_tensor_count
            ),

            "trainable_parameter_count": (
                after_trainable_count
            ),

            "canonical_state_sha256_before": (
                before_hash
            ),

            "canonical_state_sha256_after": (
                after_hash
            ),

            "logical_state_dict_sha256_before": (
                before_state_dict_hash
            ),

            "logical_state_dict_sha256_after": (
                after_state_dict_hash
            ),

            "parameter_names_unchanged": (
                before_parameter_names
                == after_parameter_names
            ),

            "parameter_tensor_bytes_unchanged": (
                before_parameter_hashes
                == after_parameter_hashes
            ),

            "buffer_names_unchanged": (
                before_buffer_names
                == after_buffer_names
            ),

            "module_names_unchanged": (
                before_module_names
                == after_module_names
            ),

            "state_dict_keys_unchanged": (
                before_state_keys
                == after_state_keys
            ),
        },

        "training_boundary": {
            "model_instantiated": (
                True
            ),

            "metadata_bridge_applied": (
                True
            ),

            "forward_methods_attached": (
                True
            ),

            "forward_computation_performed": (
                False
            ),

            "BCE_computed": (
                False
            ),

            "backward_computation_performed": (
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

            "optimizer_steps": (
                0
            ),
        },

        "next_phase": {
            "id": (
                "5.3.1k"
            ),

            "title": (
                "Procedural Forward Runtime Input-Binding "
                "and Execution Preflight"
            ),

            "requirement": (
                "Bind the exact frozen Phase-4.6.2 procedural "
                "trend, description, structural, pair-feature and "
                "scoring path to authoritative Phase-3/Phase-4 "
                "runtime artifacts, then execute the first composed "
                "forward/BCE/backward preflight while keeping "
                "optimizer.step() forbidden."
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
            "5.3.1j"
        ),

        "status": (
            "CANONICAL_NUMERICAL_METHOD_COMPOSITION_"
            "PROVED_AND_FROZEN"
        ),

        "canonical_source_sha256": (
            canonical_sha
        ),

        "forward_source_sha256": (
            forward_sha
        ),

        "method_count": (
            EXPECTED_METHOD_COUNT
        ),

        "canonical_state_sha256_before": (
            before_hash
        ),

        "canonical_state_sha256_after": (
            after_hash
        ),

        "neural_state_unchanged": (
            True
        ),

        "forward_computation_performed": (
            False
        ),

        "optimizer_instantiated": (
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

    for path in (
        METHOD_AST_REGISTRY_PATH,
        FORWARD_SUPPORT_IMPORT_PATH,
        FORWARD_SUPPORT_STATIC_PATH,
        FORWARD_SUPPORT_HELPER_PATH,
        METHOD_ATTACHMENT_PATH,
        BOUND_METHOD_PATH,
        STATE_NEUTRALITY_PATH,
        PHASE4_IMMUTABILITY_PATH,
        FINAL_INVARIANT_PATH,
        DECISION_REGISTER_PATH,
        CONTRACT_PATH,
        MANIFEST_PATH,
    ):

        print(
            f"WROTE  {path}"
        )

    # =========================================================================
    # Final
    # =========================================================================

    banner(
        "PHASE 5.3.1j FINAL STATUS"
    )

    print(
        "Canonical parameter authority:        Phase-4.7.1b"
    )

    print(
        "Canonical method authority:           Phase-4.6.2"
    )

    print(
        "Runtime metadata bridge:              FROZEN / APPLIED"
    )

    print()

    print(
        f"Exact numerical methods attached:     "
        f"{len(attachment_df)} / {EXPECTED_METHOD_COUNT}"
    )

    print(
        f"Bound runtime method checks:          "
        f"{len(bound_df)}"
    )

    print()

    print(
        f"Parameter tensors:                    "
        f"{after_parameter_tensor_count}"
    )

    print(
        f"Trainable parameters:                 "
        f"{after_trainable_count:,}"
    )

    print()

    print(
        "Canonical SHA256 BEFORE composition:"
    )

    print(
        before_hash
    )

    print()

    print(
        "Canonical SHA256 AFTER composition:"
    )

    print(
        after_hash
    )

    print()

    print(
        "Canonical neural state unchanged:     PASS"
    )

    print(
        "Parameter tensor bytes unchanged:     PASS"
    )

    print(
        "state_dict unchanged:                 PASS"
    )

    print(
        "Phase-4 artifacts unchanged:          PASS"
    )

    print(
        "Method composition RNG neutrality:    PASS"
    )

    print()

    print(
        "Numerical method composition:         FROZEN"
    )

    print()

    print(
        "Forward computation performed:        NO"
    )

    print(
        "BCE computed:                         NO"
    )

    print(
        "Backward computation performed:       NO"
    )

    print(
        "Optimizer instantiated:               NO"
    )

    print(
        "Optimizer steps:                      0"
    )

    banner(
        "PHASE 5.3.1j COMPLETE / "
        "CANONICAL NUMERICAL METHOD COMPOSITION FROZEN"
    )


if __name__ == "__main__":
    main()