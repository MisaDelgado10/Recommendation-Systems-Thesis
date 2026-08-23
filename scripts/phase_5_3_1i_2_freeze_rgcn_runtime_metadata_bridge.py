#!/usr/bin/env python3
"""
Phase 5.3.1i.2 — Freeze Parameter-State-Neutral R-GCN Runtime Metadata Bridge

Purpose
-------
Phase 5.3.1i.1 proved that the only unresolved self.* dependency
between:

    canonical Phase-4.7.1b parameter topology

and

    exact Phase-4.6.2 numerical forward methods

is:

    BasisRGCNLayer.forward -> self.out_dim

Phase-4.6.2 creates that attribute as:

    self.out_dim = out_dim

The attribute was classified as:

    PURE_DERIVED_METADATA

with:
    state-bearing missing attributes = 0
    unresolved missing attributes    = 0

This phase proves that adding the missing integer metadata to the
canonical Phase-4.7.1b BasisRGCNLayer instances is completely neutral
with respect to neural parameter/buffer/module state.

Bridge policy
-------------
For every canonical BasisRGCNLayer:

    derived_out_dim_from_root =
        int(layer.root_weight.shape[1])

    derived_out_dim_from_bases =
        int(layer.bases.shape[2])

Require equality, then:

    layer.out_dim = derived_out_dim_from_root

This reproduces the semantic value originally stored by
Phase-4.6.2's:

    self.out_dim = out_dim

without introducing:
- a parameter;
- a buffer;
- a module;
- a tensor;
- an optimizer state;
- a new neural degree of freedom.

Hard neutrality proof
---------------------
Before versus after bridge must preserve exactly:

- canonical model parameter-state SHA256;
- logical state_dict SHA256;
- state_dict keys;
- parameter names;
- parameter tensor count;
- trainable parameter count;
- buffer names;
- module names;
- all existing parameter tensor bytes.

The bridge itself must not modify Python, NumPy, or torch global RNG
state.

THIS SCRIPT DOES:
- load the canonical Phase-4.7.1b definition namespace using the
  already-frozen SANITIZED_PREFIX_AST mechanism;
- instantiate one canonical model with seed 42;
- verify the frozen canonical state hash;
- attach ONLY the non-state integer `out_dim` metadata;
- prove exact state neutrality.

THIS SCRIPT DOES NOT:
- normally import the Phase-4.7.1b script;
- execute the Phase-4.7.1b audit workflow;
- execute Phase-4.6.2;
- attach Phase-4.6.2 forward methods;
- instantiate Adam;
- instantiate Phase-5 training-negative RNG;
- instantiate Phase-5 training-order RNG;
- perform a forward pass;
- perform backward;
- call optimizer.step();
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
# Frozen canonical source
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

EXPECTED_INITIAL_STATE_SHA256 = (
    "49e822ea7fad35c458f47e134c94c05e"
    "ac099b68c5c468e2c71559c8c88998ab"
)

EXPECTED_PARAMETER_COUNT = 19_217_929
EXPECTED_PARAMETER_TENSORS = 32
EXPECTED_GLOBAL_RELATIONS = 12
EXPECTED_RGCN_BASES = 5
EXPECTED_STRUCTURAL_DIM = 40
EXPECTED_SEED = 42

REFERENCE_TORCH_VERSION_PREFIX = (
    "2.7.0"
)


# =============================================================================
# Frozen runtime-loader contract
# =============================================================================

PHASE_5_3_1F_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_1f_side_effect_free_runtime_loading_contract.json"
)

EXPECTED_RUNTIME_AST_SHA256 = (
    "301a074aa57cfe7602f2ccbb5b8e26943"
    "b94b72e36efe4d60d1af48378c58a6e"
)

EXPECTED_WORKFLOW_BOUNDARY_INDEX = 56
EXPECTED_WORKFLOW_BOUNDARY_LINE = 1272


# =============================================================================
# Phase-5.3.1i.1 evidence
# =============================================================================

PHASE_5_3_1I_1_MANIFEST_PATH = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_1i_1/"
    "phase_5_3_1i_1_missing_dependency_manifest.json"
)


# =============================================================================
# Frozen Phase-4 artifacts — immutability guard
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
    "phase_5_3_1i_2"
)

CONTRACT_DIR = Path(
    "data/experimental/phase_5/contracts"
)

SOURCE_METADATA_PROOF_PATH = (
    AUDIT_DIR
    / "rgcn_out_dim_source_provenance.csv"
)

MODULE_BRIDGE_PROOF_PATH = (
    AUDIT_DIR
    / "rgcn_runtime_metadata_bridge_module_proof.csv"
)

STATE_NEUTRALITY_PATH = (
    AUDIT_DIR
    / "rgcn_metadata_bridge_state_neutrality.csv"
)

PHASE4_IMMUTABILITY_PATH = (
    AUDIT_DIR
    / "phase4_artifact_immutability_audit.csv"
)

FINAL_INVARIANT_PATH = (
    AUDIT_DIR
    / "phase_5_3_1i_2_final_invariants.csv"
)

MANIFEST_PATH = (
    AUDIT_DIR
    / "phase_5_3_1i_2_metadata_bridge_manifest.json"
)

CONTRACT_PATH = (
    CONTRACT_DIR
    / "phase_5_3_1i_2_rgcn_runtime_metadata_bridge_contract.json"
)

DECISION_REGISTER_PATH = (
    CONTRACT_DIR
    / "phase_5_3_1i_2_runtime_metadata_bridge_decision_register.csv"
)


# =============================================================================
# Helpers
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

    if not bool(condition):

        raise AssertionError(
            message
        )


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

            digest.update(
                block
            )

    return digest.hexdigest()


def sha256_text(
    value: str,
) -> str:

    return hashlib.sha256(
        value.encode("utf-8")
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


def logical_state_dict_sha256(
    model: torch.nn.Module,
) -> str:

    digest = hashlib.sha256()

    state = (
        model.state_dict()
    )

    for name, tensor in (
        state.items()
    ):

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


# =============================================================================
# AST/source provenance helpers
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


def direct_self_assignment(
    function: ast.FunctionDef,
    attribute: str,
) -> ast.Assign | ast.AnnAssign | None:

    matches = []

    for statement in function.body:

        if isinstance(
            statement,
            ast.Assign,
        ):

            for target in (
                statement.targets
            ):

                if (
                    isinstance(
                        target,
                        ast.Attribute,
                    )
                    and isinstance(
                        target.value,
                        ast.Name,
                    )
                    and target.value.id
                    == "self"
                    and target.attr
                    == attribute
                ):

                    matches.append(
                        statement
                    )

        elif isinstance(
            statement,
            ast.AnnAssign,
        ):

            target = (
                statement.target
            )

            if (
                isinstance(
                    target,
                    ast.Attribute,
                )
                and isinstance(
                    target.value,
                    ast.Name,
                )
                and target.value.id
                == "self"
                and target.attr
                == attribute
            ):

                matches.append(
                    statement
                )

    require(
        len(matches) <= 1,
        (
            f"Multiple direct assignments to "
            f"self.{attribute} found."
        ),
    )

    if not matches:

        return None

    return matches[0]


def assignment_value(
    statement: ast.Assign | ast.AnnAssign,
) -> ast.AST:

    if isinstance(
        statement,
        ast.Assign,
    ):

        return statement.value

    require(
        statement.value
        is not None,
        (
            "Annotated assignment has no value."
        ),
    )

    return statement.value


def loaded_self_attribute(
    function: ast.FunctionDef,
    attribute: str,
) -> bool:

    for node in ast.walk(
        function
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

            continue

        path.reverse()

        if (
            path
            and path[0]
            == attribute
        ):

            return True

    return False


# =============================================================================
# Frozen sanitized runtime loader
# =============================================================================

RETAINED_NODE_TYPES = (
    ast.Import,
    ast.ImportFrom,
    ast.Assign,
    ast.AnnAssign,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
)


def retain_prefix_node(
    node: ast.AST,
) -> bool:

    return isinstance(
        node,
        RETAINED_NODE_TYPES,
    )


def build_sanitized_runtime_module(
    tree: ast.Module,
):
    """
    Reconstruct the exact Phase-5.3.1f loading payload.
    """

    prefix = (
        tree.body[
            :EXPECTED_WORKFLOW_BOUNDARY_INDEX
        ]
    )

    retained = [
        copy.deepcopy(
            node
        )
        for node in prefix
        if retain_prefix_node(
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
            "Frozen sanitized runtime AST drift.\n"
            f"Expected: {EXPECTED_RUNTIME_AST_SHA256}\n"
            f"Actual:   {runtime_ast_sha}"
        ),
    )

    module_name = (
        "_itrs_phase5_3_1i_2_canonical_runtime"
    )

    runtime_module = types.ModuleType(
        module_name
    )

    runtime_module.__file__ = str(
        CANONICAL_SOURCE_PATH.resolve()
    )

    runtime_module.__package__ = None

    sys.modules[
        module_name
    ] = runtime_module

    compiled = compile(
        sanitized_tree,
        filename=str(
            CANONICAL_SOURCE_PATH
        ),
        mode="exec",
    )

    exec(
        compiled,
        runtime_module.__dict__,
    )

    return (
        runtime_module,
        runtime_ast_sha,
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.3.1i.2 — "
        "PARAMETER-STATE-NEUTRAL R-GCN METADATA BRIDGE PROOF AND FREEZE"
    )

    print(
        "Normal Phase-4 import:                NO"
    )

    print(
        "Canonical sanitized runtime:          YES"
    )

    print(
        "Canonical model instantiation:        YES"
    )

    print(
        "Phase-4.6.2 methods grafted:          NO"
    )

    print(
        "Metadata bridge tested:               YES"
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
    # Input integrity
    # =========================================================================

    banner(
        "AUTHORITATIVE INPUT RECHECK"
    )

    for path in (
        CANONICAL_SOURCE_PATH,
        FORWARD_SOURCE_PATH,
        PHASE_5_3_1F_CONTRACT_PATH,
        PHASE_5_3_1I_1_MANIFEST_PATH,
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
            "Canonical Phase-4.7.1b source drift."
        ),
    )

    require(
        forward_sha
        == FORWARD_SOURCE_SHA256,
        (
            "Phase-4.6.2 source drift."
        ),
    )

    loader_contract = (
        load_json(
            PHASE_5_3_1F_CONTRACT_PATH
        )
    )

    dependency_manifest = (
        load_json(
            PHASE_5_3_1I_1_MANIFEST_PATH
        )
    )

    require(
        loader_contract[
            "status"
        ]
        == "FROZEN",
        (
            "Phase-5.3.1f runtime loader "
            "is not frozen."
        ),
    )

    require(
        loader_contract[
            "loading_policy"
        ][
            "sanitized_ast_sha256"
        ]
        == EXPECTED_RUNTIME_AST_SHA256,
        (
            "Frozen Phase-5 runtime AST changed."
        ),
    )

    require(
        dependency_manifest[
            "status"
        ]
        == "AUDIT_COMPLETE",
        (
            "Phase-5.3.1i.1 did not complete."
        ),
    )

    require(
        dependency_manifest[
            "decision"
        ]
        == "RUNTIME_METADATA_BRIDGE_CANDIDATE",
        (
            "Phase-5.3.1i.1 did not approve "
            "metadata-bridge auditing."
        ),
    )

    missing = (
        dependency_manifest[
            "missing_dependencies"
        ]
    )

    require(
        int(
            missing[
                "unique_missing_attribute_count"
            ]
        )
        == 1,
        (
            "Expected exactly one missing runtime attribute."
        ),
    )

    require(
        int(
            missing[
                "state_bearing_missing_count"
            ]
        )
        == 0,
        (
            "State-bearing missing dependencies exist."
        ),
    )

    require(
        int(
            missing[
                "unresolved_missing_count"
            ]
        )
        == 0,
        (
            "Unresolved missing dependencies exist."
        ),
    )

    require(
        missing[
            "all_missing_attributes_metadata_bridgeable"
        ]
        is True,
        (
            "Phase-5.3.1i.1 did not classify all "
            "missing attributes as metadata-only."
        ),
    )

    print(
        "Canonical source SHA256:              PASS"
    )

    print(
        "Forward source SHA256:                PASS"
    )

    print(
        "Runtime loader contract:              PASS"
    )

    print(
        "Phase-5.3.1i.1 metadata classification: PASS"
    )

    # =========================================================================
    # Static source proof
    # =========================================================================

    banner(
        "STATIC self.out_dim SOURCE PROVENANCE"
    )

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

    require(
        int(
            canonical_tree.body[
                EXPECTED_WORKFLOW_BOUNDARY_INDEX
            ].lineno
        )
        == EXPECTED_WORKFLOW_BOUNDARY_LINE,
        (
            "Canonical workflow boundary changed."
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

    require(
        "BasisRGCNLayer"
        in canonical_classes,
        (
            "Canonical BasisRGCNLayer missing."
        ),
    )

    require(
        "BasisRGCNLayer"
        in forward_classes,
        (
            "Forward BasisRGCNLayer missing."
        ),
    )

    canonical_init = (
        direct_method(
            canonical_classes[
                "BasisRGCNLayer"
            ],
            "__init__",
        )
    )

    forward_init = (
        direct_method(
            forward_classes[
                "BasisRGCNLayer"
            ],
            "__init__",
        )
    )

    forward_forward = (
        direct_method(
            forward_classes[
                "BasisRGCNLayer"
            ],
            "forward",
        )
    )

    require(
        canonical_init
        is not None,
        (
            "Canonical BasisRGCNLayer.__init__ missing."
        ),
    )

    require(
        forward_init
        is not None,
        (
            "Forward BasisRGCNLayer.__init__ missing."
        ),
    )

    require(
        forward_forward
        is not None,
        (
            "Phase-4.6.2 BasisRGCNLayer.forward missing."
        ),
    )

    canonical_out_dim_assignment = (
        direct_self_assignment(
            canonical_init,
            "out_dim",
        )
    )

    forward_out_dim_assignment = (
        direct_self_assignment(
            forward_init,
            "out_dim",
        )
    )

    require(
        canonical_out_dim_assignment
        is None,
        (
            "Canonical source unexpectedly already "
            "defines self.out_dim."
        ),
    )

    require(
        forward_out_dim_assignment
        is not None,
        (
            "Phase-4.6.2 no longer defines self.out_dim."
        ),
    )

    forward_out_dim_value = (
        assignment_value(
            forward_out_dim_assignment
        )
    )

    require(
        (
            isinstance(
                forward_out_dim_value,
                ast.Name,
            )
            and forward_out_dim_value.id
            == "out_dim"
        ),
        (
            "Phase-4.6.2 self.out_dim is no longer "
            "a direct copy of constructor argument out_dim."
        ),
    )

    require(
        loaded_self_attribute(
            forward_forward,
            "out_dim",
        ),
        (
            "Phase-4.6.2 BasisRGCNLayer.forward "
            "no longer consumes self.out_dim."
        ),
    )

    source_proof_df = pd.DataFrame(
        [
            {
                "attribute": (
                    "out_dim"
                ),

                "canonical_constructor_defines": (
                    False
                ),

                "phase4_6_2_constructor_defines": (
                    True
                ),

                "phase4_6_2_assignment": (
                    ast.unparse(
                        forward_out_dim_assignment
                    )
                ),

                "assignment_value": (
                    ast.unparse(
                        forward_out_dim_value
                    )
                ),

                "phase4_6_2_forward_consumes": (
                    True
                ),

                "classification": (
                    "PURE_DERIVED_METADATA"
                ),
            }
        ]
    )

    print(
        source_proof_df.to_string(
            index=False
        )
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
            "Canonical fingerprint reference runtime "
            "requires PyTorch 2.7.0."
        ),
    )

    # =========================================================================
    # Frozen Phase-4 artifact snapshot
    # =========================================================================

    phase4_before = (
        snapshot_phase4_artifacts()
    )

    # =========================================================================
    # Sanitized module
    # =========================================================================

    banner(
        "LOAD CANONICAL SANITIZED RUNTIME"
    )

    (
        runtime_module,
        runtime_ast_sha,
    ) = (
        build_sanitized_runtime_module(
            canonical_tree
        )
    )

    require(
        runtime_ast_sha
        == EXPECTED_RUNTIME_AST_SHA256,
        (
            "Sanitized runtime AST mismatch."
        ),
    )

    print(
        "Sanitized runtime AST SHA256:"
    )

    print(
        runtime_ast_sha
    )

    print()

    print(
        "Normal Phase-4 import used:           NO"
    )

    # =========================================================================
    # Canonical model construction
    # =========================================================================

    banner(
        "CANONICAL MODEL BEFORE METADATA BRIDGE"
    )

    builder = getattr(
        runtime_module,
        "build_canonical_model",
    )

    canonical_state_hash_fn = getattr(
        runtime_module,
        "model_parameter_state_sha256",
    )

    model = builder(
        seed=EXPECTED_SEED
    )

    before_canonical_hash = (
        canonical_state_hash_fn(
            model
        )
    )

    require(
        before_canonical_hash
        == EXPECTED_INITIAL_STATE_SHA256,
        (
            "Canonical model hash mismatch BEFORE bridge.\n"
            f"Expected: {EXPECTED_INITIAL_STATE_SHA256}\n"
            f"Actual:   {before_canonical_hash}"
        ),
    )

    before_parameter_names = tuple(
        name
        for name, _
        in model.named_parameters()
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

    before_parameter_hashes = (
        parameter_tensor_hashes(
            model
        )
    )

    before_state_dict_hash = (
        logical_state_dict_sha256(
            model
        )
    )

    before_parameter_tensor_count = sum(
        1
        for _
        in model.parameters()
    )

    before_trainable_parameter_count = sum(
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
        before_trainable_parameter_count
        == EXPECTED_PARAMETER_COUNT,
        (
            "Canonical trainable parameter count drift."
        ),
    )

    print(
        f"Parameter tensors:                    "
        f"{before_parameter_tensor_count}"
    )

    print(
        f"Trainable parameters:                 "
        f"{before_trainable_parameter_count:,}"
    )

    print(
        "Canonical parameter-state SHA256:"
    )

    print(
        before_canonical_hash
    )

    # =========================================================================
    # Locate the exact canonical BasisRGCNLayer instances
    # =========================================================================

    banner(
        "CANONICAL BasisRGCNLayer INSTANCE AUDIT"
    )

    basis_class = getattr(
        runtime_module,
        "BasisRGCNLayer",
    )

    basis_instances = [
        (
            name,
            module,
        )
        for name, module
        in model.named_modules()
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
            f"BasisRGCNLayer instances, found "
            f"{len(basis_instances)}."
        ),
    )

    expected_names = (
        "preference_propagation.layer_1",
        "preference_propagation.layer_2",
    )

    actual_names = tuple(
        name
        for name, _
        in basis_instances
    )

    require(
        actual_names
        == expected_names,
        (
            "Canonical R-GCN module paths changed.\n"
            f"Expected: {expected_names}\n"
            f"Actual:   {actual_names}"
        ),
    )

    bridge_rows = []

    # =========================================================================
    # Snapshot RNG immediately before bridge itself
    # =========================================================================

    python_rng_before_bridge = (
        random.getstate()
    )

    numpy_rng_before_bridge = (
        np.random.get_state()
    )

    torch_rng_before_bridge = (
        torch.get_rng_state().clone()
    )

    # =========================================================================
    # Apply ONLY the candidate metadata bridge
    # =========================================================================

    banner(
        "APPLY PARAMETER-STATE-NEUTRAL out_dim METADATA BRIDGE"
    )

    for (
        module_name,
        layer,
    ) in basis_instances:

        require(
            not hasattr(
                layer,
                "out_dim",
            ),
            (
                f"{module_name} already has out_dim "
                "before bridge."
            ),
        )

        require(
            hasattr(
                layer,
                "root_weight",
            ),
            (
                f"{module_name} lacks root_weight."
            ),
        )

        require(
            hasattr(
                layer,
                "bases",
            ),
            (
                f"{module_name} lacks bases."
            ),
        )

        require(
            hasattr(
                layer,
                "coefficients",
            ),
            (
                f"{module_name} lacks coefficients."
            ),
        )

        root_shape = tuple(
            layer.root_weight.shape
        )

        bases_shape = tuple(
            layer.bases.shape
        )

        coefficients_shape = tuple(
            layer.coefficients.shape
        )

        require(
            len(
                root_shape
            )
            == 2,
            (
                f"{module_name}.root_weight rank changed."
            ),
        )

        require(
            len(
                bases_shape
            )
            == 3,
            (
                f"{module_name}.bases rank changed."
            ),
        )

        require(
            len(
                coefficients_shape
            )
            == 2,
            (
                f"{module_name}.coefficients rank changed."
            ),
        )

        derived_from_root = int(
            root_shape[
                1
            ]
        )

        derived_from_bases = int(
            bases_shape[
                2
            ]
        )

        require(
            derived_from_root
            == derived_from_bases,
            (
                f"{module_name}: root_weight and bases "
                "disagree on out_dim."
            ),
        )

        require(
            int(
                bases_shape[
                    0
                ]
            )
            == EXPECTED_RGCN_BASES,
            (
                f"{module_name}: number of bases changed."
            ),
        )

        require(
            int(
                coefficients_shape[
                    0
                ]
            )
            == EXPECTED_GLOBAL_RELATIONS,
            (
                f"{module_name}: relation count changed."
            ),
        )

        require(
            int(
                coefficients_shape[
                    1
                ]
            )
            == EXPECTED_RGCN_BASES,
            (
                f"{module_name}: coefficient basis dimension changed."
            ),
        )

        require(
            derived_from_root
            == EXPECTED_STRUCTURAL_DIM,
            (
                f"{module_name}: expected out_dim "
                f"{EXPECTED_STRUCTURAL_DIM}, "
                f"found {derived_from_root}."
            ),
        )

        # ---------------------------------------------------------------------
        # EXACT BRIDGE.
        #
        # Plain Python int on nn.Module.
        # This must not register as parameter/buffer/module.
        # ---------------------------------------------------------------------

        setattr(
            layer,
            "out_dim",
            derived_from_root,
        )

        require(
            hasattr(
                layer,
                "out_dim",
            ),
            (
                f"{module_name}: out_dim bridge failed."
            ),
        )

        require(
            isinstance(
                layer.out_dim,
                int,
            ),
            (
                f"{module_name}: bridged out_dim "
                "is not Python int."
            ),
        )

        require(
            layer.out_dim
            == derived_from_root,
            (
                f"{module_name}: bridged out_dim "
                "value mismatch."
            ),
        )

        require(
            "out_dim"
            not in layer._parameters,
            (
                f"{module_name}: out_dim became a parameter."
            ),
        )

        require(
            "out_dim"
            not in layer._buffers,
            (
                f"{module_name}: out_dim became a buffer."
            ),
        )

        require(
            "out_dim"
            not in layer._modules,
            (
                f"{module_name}: out_dim became a module."
            ),
        )

        bridge_rows.append(
            {
                "module_name": (
                    module_name
                ),

                "root_weight_shape": (
                    str(
                        root_shape
                    )
                ),

                "bases_shape": (
                    str(
                        bases_shape
                    )
                ),

                "coefficients_shape": (
                    str(
                        coefficients_shape
                    )
                ),

                "out_dim_from_root_weight": (
                    derived_from_root
                ),

                "out_dim_from_bases": (
                    derived_from_bases
                ),

                "bridged_out_dim": (
                    int(
                        layer.out_dim
                    )
                ),

                "python_type": (
                    type(
                        layer.out_dim
                    ).__name__
                ),

                "registered_parameter": (
                    "out_dim"
                    in layer._parameters
                ),

                "registered_buffer": (
                    "out_dim"
                    in layer._buffers
                ),

                "registered_module": (
                    "out_dim"
                    in layer._modules
                ),

                "status": (
                    "PASS"
                ),
            }
        )

    bridge_df = pd.DataFrame(
        bridge_rows
    )

    print(
        bridge_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # RNG after bridge
    # =========================================================================

    python_rng_after_bridge = (
        random.getstate()
    )

    numpy_rng_after_bridge = (
        np.random.get_state()
    )

    torch_rng_after_bridge = (
        torch.get_rng_state().clone()
    )

    require(
        python_rng_before_bridge
        == python_rng_after_bridge,
        (
            "Metadata bridge altered Python RNG."
        ),
    )

    require(
        numpy_rng_state_equal(
            numpy_rng_before_bridge,
            numpy_rng_after_bridge,
        ),
        (
            "Metadata bridge altered NumPy global RNG."
        ),
    )

    require(
        torch.equal(
            torch_rng_before_bridge,
            torch_rng_after_bridge,
        ),
        (
            "Metadata bridge altered torch global RNG."
        ),
    )

    # =========================================================================
    # Exact post-bridge neural-state proof
    # =========================================================================

    banner(
        "POST-BRIDGE NEURAL STATE NEUTRALITY"
    )

    after_canonical_hash = (
        canonical_state_hash_fn(
            model
        )
    )

    after_parameter_names = tuple(
        name
        for name, _
        in model.named_parameters()
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

    after_parameter_hashes = (
        parameter_tensor_hashes(
            model
        )
    )

    after_state_dict_hash = (
        logical_state_dict_sha256(
            model
        )
    )

    after_parameter_tensor_count = sum(
        1
        for _
        in model.parameters()
    )

    after_trainable_parameter_count = sum(
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
                before_canonical_hash
            ),
            "after": (
                after_canonical_hash
            ),
            "unchanged": (
                before_canonical_hash
                == after_canonical_hash
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
                    before_trainable_parameter_count
                )
            ),
            "after": (
                str(
                    after_trainable_parameter_count
                )
            ),
            "unchanged": (
                before_trainable_parameter_count
                == after_trainable_parameter_count
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
            "Metadata bridge altered canonical neural state."
        ),
    )

    print(
        neutrality_df.to_string(
            index=False
        )
    )

    print()

    print(
        "Canonical parameter-state SHA256:"
    )

    print(
        after_canonical_hash
    )

    print()

    print(
        "Exact frozen hash preserved:          PASS"
    )

    # =========================================================================
    # Frozen Phase-4 artifact immutability
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
            "Metadata bridge proof modified "
            "a frozen Phase-4 artifact."
        ),
    )

    print(
        artifact_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Freeze invariants
    # =========================================================================

    banner(
        "FINAL PHASE-5.3.1i.2 FREEZE INVARIANTS"
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
            "phase_5_3_1i_1_metadata_bridge_candidate",
            (
                dependency_manifest[
                    "decision"
                ]
                == "RUNTIME_METADATA_BRIDGE_CANDIDATE"
            ),
        ),

        (
            "exactly_one_missing_attribute",
            (
                int(
                    missing[
                        "unique_missing_attribute_count"
                    ]
                )
                == 1
            ),
        ),

        (
            "missing_attribute_state_bearing_count_zero",
            (
                int(
                    missing[
                        "state_bearing_missing_count"
                    ]
                )
                == 0
            ),
        ),

        (
            "missing_attribute_unresolved_count_zero",
            (
                int(
                    missing[
                        "unresolved_missing_count"
                    ]
                )
                == 0
            ),
        ),

        (
            "forward_source_assigns_self_out_dim_from_out_dim",
            (
                isinstance(
                    forward_out_dim_value,
                    ast.Name,
                )
                and forward_out_dim_value.id
                == "out_dim"
            ),
        ),

        (
            "canonical_source_omits_self_out_dim",
            (
                canonical_out_dim_assignment
                is None
            ),
        ),

        (
            "canonical_model_hash_before_bridge_exact",
            (
                before_canonical_hash
                == EXPECTED_INITIAL_STATE_SHA256
            ),
        ),

        (
            "exactly_two_canonical_rgcn_layers",
            (
                len(
                    basis_instances
                )
                == 2
            ),
        ),

        (
            "bridge_values_derived_from_existing_parameter_shapes",
            bool(
                (
                    bridge_df[
                        "out_dim_from_root_weight"
                    ]
                    == bridge_df[
                        "out_dim_from_bases"
                    ]
                ).all()
            ),
        ),

        (
            "bridged_out_dim_equals_structural_dim",
            bool(
                (
                    bridge_df[
                        "bridged_out_dim"
                    ]
                    == EXPECTED_STRUCTURAL_DIM
                ).all()
            ),
        ),

        (
            "bridge_registers_no_parameters",
            bool(
                (
                    bridge_df[
                        "registered_parameter"
                    ]
                    == False
                ).all()
            ),
        ),

        (
            "bridge_registers_no_buffers",
            bool(
                (
                    bridge_df[
                        "registered_buffer"
                    ]
                    == False
                ).all()
            ),
        ),

        (
            "bridge_registers_no_modules",
            bool(
                (
                    bridge_df[
                        "registered_module"
                    ]
                    == False
                ).all()
            ),
        ),

        (
            "canonical_parameter_hash_after_bridge_exact",
            (
                after_canonical_hash
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
            "bridge_does_not_change_python_rng",
            (
                python_rng_before_bridge
                == python_rng_after_bridge
            ),
        ),

        (
            "bridge_does_not_change_numpy_rng",
            numpy_rng_state_equal(
                numpy_rng_before_bridge,
                numpy_rng_after_bridge,
            ),
        ),

        (
            "bridge_does_not_change_torch_rng",
            torch.equal(
                torch_rng_before_bridge,
                torch_rng_after_bridge,
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
            "phase4_6_2_methods_not_grafted",
            True,
        ),

        (
            "optimizer_not_instantiated",
            True,
        ),

        (
            "forward_computation_not_performed",
            True,
        ),

        (
            "backward_computation_not_performed",
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
            for name, passed
            in checks
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
            "At least one Phase-5.3.1i.2 "
            "metadata-bridge freeze invariant failed."
        ),
    )

    print(
        invariant_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Write outputs + freeze
    # =========================================================================

    banner(
        "WRITE AND FREEZE RUNTIME METADATA BRIDGE"
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTRACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_proof_df.to_csv(
        SOURCE_METADATA_PROOF_PATH,
        index=False,
    )

    bridge_df.to_csv(
        MODULE_BRIDGE_PROOF_PATH,
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
                    "BasisRGCNLayer_runtime_out_dim_bridge"
                ),

                "value": (
                    "layer.out_dim = "
                    "int(layer.root_weight.shape[1])"
                ),

                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),

                "reason": (
                    "Phase-4.6.2 forward requires self.out_dim; "
                    "Phase-4.7.1b omits only this non-state metadata. "
                    "Value is derived from existing canonical parameter "
                    "shape and cross-checked against bases.shape[2]."
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1i_2"
                ),
            },

            {
                "decision": (
                    "metadata_bridge_state_policy"
                ),

                "value": (
                    "PYTHON_INT_ONLY_NO_PARAMETER_BUFFER_MODULE"
                ),

                "classification": (
                    "EVALUATION_INTEGRITY_GUARD"
                ),

                "reason": (
                    "Bridge must remain invisible to neural state_dict "
                    "and optimizer parameter enumeration."
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1i_2"
                ),
            },

            {
                "decision": (
                    "bridge_validation_oracle"
                ),

                "value": (
                    EXPECTED_INITIAL_STATE_SHA256
                ),

                "classification": (
                    "INHERITED_FROZEN_PHASE4_NUMERICAL_ORACLE"
                ),

                "reason": (
                    "Canonical parameter-state hash must be exact "
                    "before and after bridge."
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1i_2"
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
            "5.3.1i.2"
        ),

        "title": (
            "Parameter-State-Neutral R-GCN "
            "Runtime Metadata Bridge Contract"
        ),

        "status": (
            "FROZEN"
        ),

        "classification": (
            "IMPLEMENTATION_EQUIVALENT_CHOICE"
        ),

        "provenance": {
            "canonical_parameter_topology": {
                "source": (
                    str(
                        CANONICAL_SOURCE_PATH
                    )
                ),
                "sha256": (
                    canonical_sha
                ),
            },

            "forward_runtime": {
                "source": (
                    str(
                        FORWARD_SOURCE_PATH
                    )
                ),
                "sha256": (
                    forward_sha
                ),
            },
        },

        "missing_runtime_metadata": {
            "class": (
                "BasisRGCNLayer"
            ),

            "attribute": (
                "out_dim"
            ),

            "phase4_6_2_constructor_semantics": (
                "self.out_dim = out_dim"
            ),

            "classification": (
                "PURE_DERIVED_METADATA"
            ),

            "state_bearing": (
                False
            ),
        },

        "bridge": {
            "target_modules": [
                (
                    "preference_propagation.layer_1"
                ),
                (
                    "preference_propagation.layer_2"
                ),
            ],

            "derivation": (
                "int(layer.root_weight.shape[1])"
            ),

            "cross_check": (
                "int(layer.bases.shape[2])"
            ),

            "expected_value": (
                EXPECTED_STRUCTURAL_DIM
            ),

            "assignment": (
                "setattr(layer, 'out_dim', derived_out_dim)"
            ),

            "python_type": (
                "int"
            ),

            "registered_as_parameter": (
                False
            ),

            "registered_as_buffer": (
                False
            ),

            "registered_as_module": (
                False
            ),
        },

        "state_neutrality": {
            "parameter_tensor_count": (
                EXPECTED_PARAMETER_TENSORS
            ),

            "trainable_parameter_count": (
                EXPECTED_PARAMETER_COUNT
            ),

            "canonical_state_sha256_before": (
                before_canonical_hash
            ),

            "canonical_state_sha256_after": (
                after_canonical_hash
            ),

            "expected_canonical_state_sha256": (
                EXPECTED_INITIAL_STATE_SHA256
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
            "phase4_6_2_methods_grafted": (
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

        "next_phase": {
            "id": (
                "5.3.1j"
            ),

            "requirement": (
                "Compose exact Phase-4.6.2 numerical methods "
                "with the canonical Phase-4.7.1b initialized model "
                "using this frozen metadata bridge, prove the "
                "parameter-state SHA256 remains exact, and perform "
                "the first real composed forward/BCE/backward "
                "preflight without optimizer.step()."
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
            "5.3.1i.2"
        ),

        "status": (
            "RGCN_RUNTIME_METADATA_BRIDGE_PROVED_AND_FROZEN"
        ),

        "bridge_attribute": (
            "BasisRGCNLayer.out_dim"
        ),

        "bridge_value": (
            EXPECTED_STRUCTURAL_DIM
        ),

        "canonical_state_sha256_before": (
            before_canonical_hash
        ),

        "canonical_state_sha256_after": (
            after_canonical_hash
        ),

        "neural_state_unchanged": (
            True
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
        SOURCE_METADATA_PROOF_PATH,
        MODULE_BRIDGE_PROOF_PATH,
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
        "PHASE 5.3.1i.2 FINAL STATUS"
    )

    print(
        "Missing runtime metadata:             BasisRGCNLayer.out_dim"
    )

    print(
        "Classification:                       PURE_DERIVED_METADATA"
    )

    print(
        "Bridge:"
    )

    print(
        "  layer.out_dim = "
        "int(layer.root_weight.shape[1])"
    )

    print()

    print(
        f"Canonical BasisRGCNLayer instances:   "
        f"{len(basis_instances)}"
    )

    print(
        f"Bridged value:                        "
        f"{EXPECTED_STRUCTURAL_DIM}"
    )

    print()

    print(
        f"Parameter tensors:                    "
        f"{after_parameter_tensor_count}"
    )

    print(
        f"Trainable parameters:                 "
        f"{after_trainable_parameter_count:,}"
    )

    print()

    print(
        "Canonical state SHA256 BEFORE:"
    )

    print(
        before_canonical_hash
    )

    print()

    print(
        "Canonical state SHA256 AFTER:"
    )

    print(
        after_canonical_hash
    )

    print()

    print(
        "Canonical state unchanged:            PASS"
    )

    print(
        "state_dict unchanged:                 PASS"
    )

    print(
        "Parameter tensor bytes unchanged:     PASS"
    )

    print(
        "Phase-4 artifacts unchanged:          PASS"
    )

    print(
        "Bridge RNG neutrality:                PASS"
    )

    print()

    print(
        "Runtime metadata bridge:              FROZEN"
    )

    print()

    print(
        "Phase-4.6.2 methods grafted:          NO"
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
        "PHASE 5.3.1i.2 COMPLETE / "
        "R-GCN RUNTIME METADATA BRIDGE FROZEN"
    )


if __name__ == "__main__":
    main()