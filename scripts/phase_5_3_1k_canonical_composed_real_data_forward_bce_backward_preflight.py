#!/usr/bin/env python3
"""
Phase 5.3.1k — Canonical Composed Real-Data Forward / BCE / Backward Preflight

Purpose
-------
This is the FIRST real numerical execution of the fully resolved
canonical Phase-5 runtime:

    Phase-4.7.1b canonical topology + Kaiming(seed=42)
        +
    Phase-5.3.1i.2 frozen BasisRGCNLayer.out_dim metadata bridge
        +
    Phase-5.3.1j exact Phase-4.6.2 numerical methods

The preflight reuses the exact frozen real T60 validation positive
selected by Phase 4.6.2 for integration diagnostics.

It performs:

    real static input binding
        ->
    description forward
        ->
    full-graph R-GCN
        ->
    real T0..T59 trend attention + GRU
        ->
    exact 280-D pair construction
        ->
    scoring MLP
        ->
    raw logit
        ->
    BCEWithLogitsLoss(target=1)
        ->
    backward

CRITICAL TRAINING BOUNDARY
--------------------------
This is NOT an optimizer update.

NO Adam is instantiated.
NO training negatives are generated.
NO training-order RNG is instantiated.
NO optimizer.step() occurs.
NO checkpoint is written.

The parameter-state SHA256 MUST remain exactly unchanged:
before forward, after forward, and after backward.

Backward is allowed because gradients do not modify parameter values.
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
from scipy import sparse


# =============================================================================
# Frozen canonical implementation
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

EXPECTED_RUNTIME_AST_SHA256 = (
    "301a074aa57cfe7602f2ccbb5b8e26943"
    "b94b72e36efe4d60d1af48378c58a6e"
)

EXPECTED_WORKFLOW_BOUNDARY_INDEX = 56
EXPECTED_WORKFLOW_BOUNDARY_LINE = 1272

EXPECTED_SEED = 42
EXPECTED_PARAMETER_TENSORS = 32
EXPECTED_PARAMETER_COUNT = 19_217_929

REFERENCE_TORCH_VERSION_PREFIX = "2.7.0"


# =============================================================================
# Prior Phase-5 contracts
# =============================================================================

PHASE_5_3_1J_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_1j_canonical_numerical_method_composition_contract.json"
)

PHASE_5_3_1J_MANIFEST_PATH = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_1j/"
    "phase_5_3_1j_method_composition_manifest.json"
)

PHASE_5_3_1I_2_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_1i_2_rgcn_runtime_metadata_bridge_contract.json"
)


# =============================================================================
# Exact Phase-4.6.2 numerical methods
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

EXPECTED_METHOD_AST_SHA256 = {
    (
        "DescriptionEncoder",
        "forward",
    ): (
        "b15f98801c6acc1c8d46da3e6ce136a2"
        "e23ba560892f7250a50279f4d0b7243c"
    ),

    (
        "TrendExtractor",
        "attend_period",
    ): (
        "f284661d0d89280b9c2e2f082441cf9e"
        "a52fee99fd4379d67f5b50e93804dc5d"
    ),

    (
        "TrendExtractor",
        "encode_sequence",
    ): (
        "72fa12a49ad4a21399d4810a31323f670"
        "34e6173f73ee71dc18798bad1b6d97f"
    ),

    (
        "BasisRGCNLayer",
        "effective_weight",
    ): (
        "0d499494ea8ddd2b4acec70dacd69f2a3"
        "af36a5e71d358806a903546ad14c134"
    ),

    (
        "BasisRGCNLayer",
        "forward",
    ): (
        "c4199e84dafb278fe61d66732792d8258c"
        "57b4bb0295cd5950e8e51f4be06e56"
    ),

    (
        "PreferencePropagation",
        "forward",
    ): (
        "cc842ec360a214f1cdafb30b4264c0c8c"
        "95978efa6a5836b231824cdc3b86256"
    ),

    (
        "ScoringMLP",
        "forward",
    ): (
        "1f5b79f7ea6839c6f2d9ee92b596f9c6"
        "a2d7967c56254130bfc215c3ca636967"
    ),
}

FORWARD_SUPPORT_CONSTANTS = {
    "TREND_ITEM_DIM": 80,
    "TREND_QUERY_DIM": 80,
    "NUM_HISTORY_PERIODS": 60,
    "NUM_RELATIONS": 12,
}


# =============================================================================
# Frozen dimensions
# =============================================================================

NUM_INVESTORS = 165_975
NUM_STARTUPS = 311_589
NUM_NODES = 477_564

NUM_RELATIONS = 12
NUM_EDGES = 158_818

NUM_HISTORY_PERIODS = 60

LATENT_DIM = 40
DOC2VEC_DIM = 32
LABEL_DIM = 802
DESCRIPTION_DIM = 40
TREND_ITEM_DIM = 80
TREND_QUERY_DIM = 80
TREND_DIM = 40
STRUCTURAL_DIM = 40

INVESTOR_SCORING_DIM = 160
STARTUP_SCORING_DIM = 120
PAIR_DIM = 280


# =============================================================================
# Frozen authoritative data
# =============================================================================

TEMPORAL_SPLIT_PATH = Path(
    "data/experimental/phase_2/model_ready/"
    "interactions_itrs_temporal_split.parquet"
)

NODE_INDEX_PATH = Path(
    "data/experimental/phase_3/model_ready/"
    "node_index.parquet"
)

EDGE_INDEX_PATH = Path(
    "data/experimental/phase_3/model_ready/"
    "edge_index.npy"
)

EDGE_TYPE_PATH = Path(
    "data/experimental/phase_3/model_ready/"
    "edge_type.npy"
)

DOC2VEC_PATH = Path(
    "data/experimental/phase_4/doc2vec/"
    "vectors/doc2vec_vectors_all.npy"
)

DOC2VEC_MANIFEST_PATH = Path(
    "data/experimental/phase_4/doc2vec/"
    "vectors/doc2vec_vector_manifest.parquet"
)

LABEL_MATRIX_PATH = Path(
    "data/experimental/phase_4/"
    "description_labels/"
    "description_label_multihot.npz"
)

LABEL_MANIFEST_PATH = Path(
    "data/experimental/phase_4/"
    "description_labels/"
    "description_label_vector_manifest.parquet"
)

TREND_PERIOD_PTR_PATH = Path(
    "data/experimental/phase_4/"
    "trend_runtime/trend_period_ptr.npy"
)

TREND_STARTUP_INDICES_PATH = Path(
    "data/experimental/phase_4/"
    "trend_runtime/trend_startup_node_indices.npy"
)

TREND_PERIOD_COUNTS_PATH = Path(
    "data/experimental/phase_4/"
    "trend_runtime/trend_period_startup_counts.npy"
)

SELECTED_PAIR_PATH = Path(
    "data/experimental/phase_4/"
    "full_model_forward_audit/"
    "phase_4_6_2_selected_validation_pair.json"
)


# =============================================================================
# Frozen static artifact hashes
# =============================================================================

EXPECTED_STATIC_SHA256 = {
    str(
        EDGE_INDEX_PATH
    ): (
        "9aa2628ee0f68ceb7739dde165c782509"
        "a8054ab26ce8c8ee6488665ca57cdbd"
    ),

    str(
        EDGE_TYPE_PATH
    ): (
        "b9798d441cb965cd405f6c5c0e45701"
        "fb82a39e71997c4bc33d989dfbefb121e"
    ),

    str(
        DOC2VEC_PATH
    ): (
        "5b7e413789a547a3aae214488cbe31e18"
        "196e7d9fdc62df35cd96b61d9c3ad4e"
    ),

    str(
        LABEL_MATRIX_PATH
    ): (
        "db542b3e24f2e21fc69c4a0dcdb435b"
        "367e5c56a68439c8ed27518c1ef65b053"
    ),

    str(
        TREND_PERIOD_PTR_PATH
    ): (
        "4d97968bbbb2e78ef4f49702a66c459d"
        "301c7e644e18e8c10735d490275e8cc6"
    ),

    str(
        TREND_STARTUP_INDICES_PATH
    ): (
        "c51d6f68b1780226bd38b680a3d6a761"
        "dc3cb5c89bfb3b1f6115c72fe6beae06"
    ),

    str(
        TREND_PERIOD_COUNTS_PATH
    ): (
        "6148b97d4e3e2a7c8ba2401ac1b2f7c"
        "9ba86c6672a767f4749fc9d7a2f1f817d"
    ),
}


# =============================================================================
# Exact frozen Phase-4.6.2 diagnostic event
# =============================================================================

EXPECTED_SELECTED_CASE = {
    "phase": (
        "4.6.2"
    ),

    "selection_purpose": (
        "end_to_end_integration_audit_only"
    ),

    "split": (
        "validation"
    ),

    "segment": (
        "T60"
    ),

    "target": (
        1
    ),

    "interaction_id": (
        "eed46b22-a77f-4ef6-b882-"
        "2bc592d93c38::"
        "8614b833-7a0d-41ca-8a76-9a637ee1691f"
    ),

    "funding_round_id": (
        "eed46b22-a77f-4ef6-b882-2bc592d93c38"
    ),

    "investor_id": (
        "8614b833-7a0d-41ca-8a76-9a637ee1691f"
    ),

    "startup_id": (
        "c37eb826-45a6-9800-c1e6-c8fe6bfb60c5"
    ),

    "investor_global_node_index": (
        87_207
    ),

    "startup_global_node_index": (
        403_772
    ),

    "startup_local_index": (
        237_797
    ),

    "history_memberships": (
        7
    ),

    "active_history_periods": (
        4
    ),

    "multi_item_history_periods": (
        1
    ),

    "max_period_items": (
        4
    ),

    "investor_incoming_degree": (
        5
    ),

    "startup_incoming_degree": (
        3
    ),

    "startup_label_count": (
        5
    ),

    "candidate_negative_generated": (
        False
    ),

    "test_data_used": (
        False
    ),

    "selection_changes_model_policy": (
        False
    ),
}


# =============================================================================
# Outputs
# =============================================================================

AUDIT_DIR = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_1k"
)

CONTRACT_DIR = Path(
    "data/experimental/phase_5/contracts"
)

STATIC_INTEGRITY_PATH = (
    AUDIT_DIR
    / "canonical_real_forward_static_artifact_integrity.csv"
)

SELECTED_CASE_PATH = (
    AUDIT_DIR
    / "canonical_real_forward_selected_case_integrity.csv"
)

SOURCE_ASSIGNMENT_PATH = (
    AUDIT_DIR
    / "canonical_loss_source_assignment_provenance.csv"
)

FORWARD_SHAPE_PATH = (
    AUDIT_DIR
    / "canonical_composed_real_forward_shape_audit.csv"
)

ATTENTION_PATH = (
    AUDIT_DIR
    / "canonical_composed_attention_audit.csv"
)

GRADIENT_PATH = (
    AUDIT_DIR
    / "canonical_composed_gradient_audit.csv"
)

STATE_NEUTRALITY_PATH = (
    AUDIT_DIR
    / "canonical_composed_forward_state_neutrality.csv"
)

FINAL_INVARIANT_PATH = (
    AUDIT_DIR
    / "phase_5_3_1k_final_invariants.csv"
)

MANIFEST_PATH = (
    AUDIT_DIR
    / "phase_5_3_1k_real_forward_preflight_manifest.json"
)

CONTRACT_PATH = (
    CONTRACT_DIR
    / "phase_5_3_1k_canonical_real_forward_bce_backward_contract.json"
)

DECISION_REGISTER_PATH = (
    CONTRACT_DIR
    / "phase_5_3_1k_real_forward_decision_register.csv"
)


# =============================================================================
# Basic helpers
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


def file_sha256(
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


def text_sha256(
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


def parameter_hashes(
    model: torch.nn.Module,
) -> dict[str, str]:

    return {
        name: tensor_sha256(
            parameter
        )
        for (
            name,
            parameter,
        ) in model.named_parameters()
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


# =============================================================================
# AST helpers
# =============================================================================

def class_map(
    tree: ast.Module,
) -> dict[str, ast.ClassDef]:

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
            f"Multiple "
            f"{class_node.name}.{method_name} "
            "definitions."
        ),
    )

    if not matches:

        return None

    return matches[
        0
    ]


def top_level_function(
    tree: ast.Module,
    name: str,
) -> ast.FunctionDef:

    matches = [
        node
        for node
        in tree.body
        if (
            isinstance(
                node,
                ast.FunctionDef,
            )
            and node.name
            == name
        )
    ]

    require(
        len(
            matches
        )
        == 1,
        (
            f"Expected one top-level function "
            f"{name}; found {len(matches)}."
        ),
    )

    return matches[
        0
    ]


def assigned_name(
    node: ast.AST,
) -> str | None:

    if isinstance(
        node,
        ast.Assign,
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
        ):

            return (
                node.targets[
                    0
                ].id
            )

    elif isinstance(
        node,
        ast.AnnAssign,
    ):

        if isinstance(
            node.target,
            ast.Name,
        ):

            return (
                node.target.id
            )

    return None


def assignment_value(
    node: ast.Assign | ast.AnnAssign,
) -> ast.AST:

    if isinstance(
        node,
        ast.Assign,
    ):

        return (
            node.value
        )

    require(
        node.value
        is not None,
        (
            "Annotated assignment has "
            "no value."
        ),
    )

    return (
        node.value
    )


def top_level_assignment(
    tree: ast.Module,
    name: str,
) -> ast.Assign | ast.AnnAssign:

    matches = [
        node
        for node
        in tree.body
        if (
            isinstance(
                node,
                (
                    ast.Assign,
                    ast.AnnAssign,
                ),
            )
            and assigned_name(
                node
            )
            == name
        )
    ]

    require(
        len(
            matches
        )
        == 1,
        (
            f"Expected exactly one top-level "
            f"assignment to {name}; "
            f"found {len(matches)}."
        ),
    )

    return matches[
        0
    ]


def expression_loaded_names(
    node: ast.AST,
) -> set[str]:

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


def execute_nodes(
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


# =============================================================================
# Canonical Phase-4.7.1b sanitized runtime
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


def build_canonical_runtime(
    tree: ast.Module,
):
    """
    Exact frozen Phase-5.3.1f SANITIZED_PREFIX_AST loader.
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
        for node in prefix
        if isinstance(
            node,
            CANONICAL_RETAINED_NODE_TYPES,
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

    payload = ast.dump(
        sanitized,
        annotate_fields=True,
        include_attributes=False,
    )

    runtime_sha = (
        text_sha256(
            payload
        )
    )

    require(
        runtime_sha
        == EXPECTED_RUNTIME_AST_SHA256,
        (
            "Canonical sanitized runtime AST drift.\n"
            f"Expected: {EXPECTED_RUNTIME_AST_SHA256}\n"
            f"Actual:   {runtime_sha}"
        ),
    )

    module_name = (
        "_itrs_phase5_3_1k_canonical_runtime"
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
        sanitized,
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
        runtime_sha,
    )


# =============================================================================
# Exact Phase-4.6.2 numerical-method namespace
# =============================================================================

def build_forward_runtime(
    tree: ast.Module,
):
    """
    Definition-only Phase-4.6.2 runtime.

    Executes:
      - imports;
      - four frozen immutable constants;
      - exact require() helper;
      - five selected class definitions.

    Does NOT execute Phase-4.6.2's top-level audit workflow.
    Does NOT instantiate Phase-4.6.2 classes.
    """

    module_name = (
        "_itrs_phase4_6_2_methods_phase5_3_1k"
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
    # Exact imports
    # -------------------------------------------------------------------------

    import_nodes = [
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
        import_nodes,
        str(
            FORWARD_SOURCE_PATH
        ),
    )

    # -------------------------------------------------------------------------
    # Exact four frozen support constants
    # -------------------------------------------------------------------------

    for (
        name,
        expected_value,
    ) in FORWARD_SUPPORT_CONSTANTS.items():

        assignment = (
            top_level_assignment(
                tree,
                name,
            )
        )

        value_node = (
            assignment_value(
                assignment
            )
        )

        try:

            literal = ast.literal_eval(
                value_node
            )

        except Exception as exc:

            raise AssertionError(
                f"{name} is no longer a literal."
            ) from exc

        require(
            literal
            == expected_value,
            (
                f"{name} changed.\n"
                f"Expected: {expected_value}\n"
                f"Actual:   {literal}"
            ),
        )

        execute_nodes(
            module,
            [
                assignment,
            ],
            str(
                FORWARD_SOURCE_PATH
            ),
        )

    # -------------------------------------------------------------------------
    # Exact require helper
    # -------------------------------------------------------------------------

    require_helper = (
        top_level_function(
            tree,
            "require",
        )
    )

    execute_nodes(
        module,
        [
            require_helper,
        ],
        str(
            FORWARD_SOURCE_PATH
        ),
    )

    # -------------------------------------------------------------------------
    # Exact selected class definitions
    # -------------------------------------------------------------------------

    source_classes = (
        class_map(
            tree
        )
    )

    selected_class_nodes = []

    method_rows = []

    for (
        class_name,
        method_names,
    ) in METHOD_GRAFT_SPEC.items():

        require(
            class_name
            in source_classes,
            (
                f"Forward source missing "
                f"{class_name}."
            ),
        )

        class_node = (
            source_classes[
                class_name
            ]
        )

        selected_class_nodes.append(
            class_node
        )

        for method_name in (
            method_names
        ):

            method = direct_method(
                class_node,
                method_name,
            )

            require(
                method
                is not None,
                (
                    f"Missing "
                    f"{class_name}.{method_name}."
                ),
            )

            actual_sha = (
                ast_sha256(
                    method
                )
            )

            expected_sha = (
                EXPECTED_METHOD_AST_SHA256[
                    (
                        class_name,
                        method_name,
                    )
                ]
            )

            require(
                actual_sha
                == expected_sha,
                (
                    f"Method AST drift: "
                    f"{class_name}.{method_name}\n"
                    f"Expected: {expected_sha}\n"
                    f"Actual:   {actual_sha}"
                ),
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
                        actual_sha
                    ),

                    "status": (
                        "PASS"
                    ),
                }
            )

    execute_nodes(
        module,
        selected_class_nodes,
        str(
            FORWARD_SOURCE_PATH
        ),
    )

    methods = {}

    for (
        class_name,
        method_names,
    ) in METHOD_GRAFT_SPEC.items():

        runtime_class = (
            module.__dict__[
                class_name
            ]
        )

        for method_name in (
            method_names
        ):

            methods[
                (
                    class_name,
                    method_name,
                )
            ] = (
                runtime_class.__dict__[
                    method_name
                ]
            )

    require(
        len(
            methods
        )
        == 7,
        (
            "Expected exactly seven "
            "Phase-4.6.2 numerical methods."
        ),
    )

    return (
        module,
        methods,
        pd.DataFrame(
            method_rows
        ),
    )


# =============================================================================
# Compose canonical runtime
# =============================================================================

def compose_canonical_model(
    canonical_runtime,
    exact_methods: dict,
):
    """
    Build canonical seed-42 model, apply frozen out_dim bridge,
    and attach exact Phase-4.6.2 methods.
    """

    builder = getattr(
        canonical_runtime,
        "build_canonical_model",
    )

    hash_fn = getattr(
        canonical_runtime,
        "model_parameter_state_sha256",
    )

    model = builder(
        seed=EXPECTED_SEED
    )

    require(
        hash_fn(
            model
        )
        == EXPECTED_INITIAL_STATE_SHA256,
        (
            "Canonical initial model hash mismatch."
        ),
    )

    # -------------------------------------------------------------------------
    # Frozen 5.3.1i.2 metadata bridge
    # -------------------------------------------------------------------------

    basis_class = getattr(
        canonical_runtime,
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

    expected_names = (
        "preference_propagation.layer_1",
        "preference_propagation.layer_2",
    )

    require(
        tuple(
            name
            for name, _
            in basis_instances
        )
        == expected_names,
        (
            "Canonical BasisRGCNLayer "
            "instance paths changed."
        ),
    )

    for (
        name,
        layer,
    ) in basis_instances:

        out_dim_root = int(
            layer.root_weight.shape[
                1
            ]
        )

        out_dim_bases = int(
            layer.bases.shape[
                2
            ]
        )

        require(
            out_dim_root
            == out_dim_bases
            == STRUCTURAL_DIM,
            (
                f"{name}: out_dim derivation drift."
            ),
        )

        setattr(
            layer,
            "out_dim",
            out_dim_root,
        )

        require(
            isinstance(
                layer.out_dim,
                int,
            ),
            (
                f"{name}.out_dim "
                "is not Python int."
            ),
        )

        require(
            "out_dim"
            not in layer._parameters,
            (
                f"{name}.out_dim "
                "registered as parameter."
            ),
        )

        require(
            "out_dim"
            not in layer._buffers,
            (
                f"{name}.out_dim "
                "registered as buffer."
            ),
        )

        require(
            "out_dim"
            not in layer._modules,
            (
                f"{name}.out_dim "
                "registered as module."
            ),
        )

    require(
        hash_fn(
            model
        )
        == EXPECTED_INITIAL_STATE_SHA256,
        (
            "Metadata bridge changed "
            "canonical state."
        ),
    )

    # -------------------------------------------------------------------------
    # Frozen 5.3.1j exact method composition
    # -------------------------------------------------------------------------

    for (
        class_name,
        method_names,
    ) in METHOD_GRAFT_SPEC.items():

        canonical_class = getattr(
            canonical_runtime,
            class_name,
        )

        for method_name in (
            method_names
        ):

            function_object = (
                exact_methods[
                    (
                        class_name,
                        method_name,
                    )
                ]
            )

            setattr(
                canonical_class,
                method_name,
                function_object,
            )

            require(
                getattr(
                    canonical_class,
                    method_name,
                )
                is function_object,
                (
                    f"Exact method attachment failed: "
                    f"{class_name}.{method_name}"
                ),
            )

    require(
        hash_fn(
            model
        )
        == EXPECTED_INITIAL_STATE_SHA256,
        (
            "Method composition changed "
            "canonical state."
        ),
    )

    return (
        model,
        hash_fn,
    )


# =============================================================================
# Exact Phase-4.6.2 loss/scoring assignment execution
# =============================================================================

LOSS_ASSIGNMENT_NAMES = (
    "logit",
    "probability",
    "target",
    "criterion",
    "loss",
    "manual_stable_bce",
    "bce_equivalent",
)


def inject_literal_dependency(
    name: str,
    environment: dict,
    source_tree: ast.Module,
    stack: set[str] | None = None,
) -> bool:

    if name in environment:

        return True

    if stack is None:

        stack = set()

    require(
        name
        not in stack,
        (
            f"Recursive literal dependency: "
            f"{name}"
        ),
    )

    stack.add(
        name
    )

    try:

        assignment = (
            top_level_assignment(
                source_tree,
                name,
            )
        )

    except AssertionError:

        stack.remove(
            name
        )

        return False

    value_node = (
        assignment_value(
            assignment
        )
    )

    try:

        value = ast.literal_eval(
            value_node
        )

    except Exception:

        stack.remove(
            name
        )

        return False

    environment[
        name
    ] = value

    stack.remove(
        name
    )

    return True


def execute_exact_loss_assignments(
    forward_tree: ast.Module,
    forward_runtime,
    scoring: dict,
):
    """
    Execute the exact frozen Phase-4.6.2 top-level assignments for:

      logit
      probability
      target
      criterion
      loss
      manual_stable_bce
      bce_equivalent

    rather than reconstructing the target/loss expressions by hand.
    """

    assignment_nodes = []

    provenance_rows = []

    for name in (
        LOSS_ASSIGNMENT_NAMES
    ):

        node = (
            top_level_assignment(
                forward_tree,
                name,
            )
        )

        assignment_nodes.append(
            (
                int(
                    node.lineno
                ),
                name,
                node,
            )
        )

        provenance_rows.append(
            {
                "name": (
                    name
                ),

                "line_number": (
                    int(
                        node.lineno
                    )
                ),

                "assignment_ast_sha256": (
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
        )

    assignment_nodes.sort(
        key=lambda item: item[
            0
        ]
    )

    environment = dict(
        forward_runtime.__dict__
    )

    environment[
        "scoring"
    ] = scoring

    for (
        _,
        name,
        node,
    ) in assignment_nodes:

        value_node = (
            assignment_value(
                node
            )
        )

        dependencies = (
            expression_loaded_names(
                value_node
            )
        )

        for dependency in sorted(
            dependencies
        ):

            if dependency in environment:

                continue

            resolved = (
                inject_literal_dependency(
                    dependency,
                    environment,
                    forward_tree,
                )
            )

            require(
                resolved,
                (
                    f"Cannot resolve dependency "
                    f"{dependency} needed by "
                    f"exact frozen assignment "
                    f"{name}."
                ),
            )

        temp_tree = ast.Module(
            body=[
                copy.deepcopy(
                    node
                ),
            ],
            type_ignores=[],
        )

        ast.fix_missing_locations(
            temp_tree
        )

        compiled = compile(
            temp_tree,
            filename=str(
                FORWARD_SOURCE_PATH
            ),
            mode="exec",
        )

        exec(
            compiled,
            environment,
        )

        require(
            name in environment,
            (
                f"Exact assignment did not create "
                f"{name}."
            ),
        )

    return (
        environment,
        pd.DataFrame(
            provenance_rows
        ).sort_values(
            "line_number",
            kind="mergesort",
        ),
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.3.1k — "
        "CANONICAL COMPOSED REAL-DATA "
        "FORWARD / BCE / BACKWARD PREFLIGHT"
    )

    print(
        "Canonical model instantiated:         YES"
    )

    print(
        "Frozen metadata bridge applied:       YES"
    )

    print(
        "Frozen Phase-4.6.2 methods attached:  YES"
    )

    print(
        "Real Phase-4 data consumed:           YES"
    )

    print(
        "Forward computation:                  YES — preflight"
    )

    print(
        "BCE computation:                      YES — preflight"
    )

    print(
        "Backward computation:                 YES — preflight"
    )

    print(
        "Training negatives generated:         NO"
    )

    print(
        "Training-order RNG instantiated:      NO"
    )

    print(
        "Optimizer instantiated:               NO"
    )

    print(
        "Optimizer steps:                      0"
    )

    # =========================================================================
    # Authoritative prerequisites
    # =========================================================================

    banner(
        "AUTHORITATIVE SOURCE / CONTRACT RECHECK"
    )

    required_paths = (
        CANONICAL_SOURCE_PATH,
        FORWARD_SOURCE_PATH,
        PHASE_5_3_1J_CONTRACT_PATH,
        PHASE_5_3_1J_MANIFEST_PATH,
        PHASE_5_3_1I_2_CONTRACT_PATH,
        TEMPORAL_SPLIT_PATH,
        NODE_INDEX_PATH,
        EDGE_INDEX_PATH,
        EDGE_TYPE_PATH,
        DOC2VEC_PATH,
        DOC2VEC_MANIFEST_PATH,
        LABEL_MATRIX_PATH,
        LABEL_MANIFEST_PATH,
        TREND_PERIOD_PTR_PATH,
        TREND_STARTUP_INDICES_PATH,
        TREND_PERIOD_COUNTS_PATH,
        SELECTED_PAIR_PATH,
    )

    for path in (
        required_paths
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

    require(
        file_sha256(
            CANONICAL_SOURCE_PATH
        )
        == CANONICAL_SOURCE_SHA256,
        (
            "Canonical source SHA256 drift."
        ),
    )

    require(
        file_sha256(
            FORWARD_SOURCE_PATH
        )
        == FORWARD_SOURCE_SHA256,
        (
            "Forward source SHA256 drift."
        ),
    )

    phase_5_3_1j_contract = (
        load_json(
            PHASE_5_3_1J_CONTRACT_PATH
        )
    )

    phase_5_3_1j_manifest = (
        load_json(
            PHASE_5_3_1J_MANIFEST_PATH
        )
    )

    bridge_contract = (
        load_json(
            PHASE_5_3_1I_2_CONTRACT_PATH
        )
    )

    require(
        phase_5_3_1j_contract[
            "status"
        ]
        == "FROZEN",
        (
            "Phase-5.3.1j method composition "
            "contract is not frozen."
        ),
    )

    require(
        phase_5_3_1j_manifest[
            "status"
        ]
        == (
            "CANONICAL_NUMERICAL_METHOD_COMPOSITION_"
            "PROVED_AND_FROZEN"
        ),
        (
            "Unexpected Phase-5.3.1j manifest status."
        ),
    )

    require(
        phase_5_3_1j_manifest[
            "canonical_state_sha256_after"
        ]
        == EXPECTED_INITIAL_STATE_SHA256,
        (
            "5.3.1j canonical hash drift."
        ),
    )

    require(
        phase_5_3_1j_manifest[
            "optimizer_instantiated"
        ]
        is False,
        (
            "Optimizer unexpectedly existed in 5.3.1j."
        ),
    )

    require(
        int(
            phase_5_3_1j_manifest[
                "optimizer_steps"
            ]
        )
        == 0,
        (
            "Optimizer step occurred before 5.3.1k."
        ),
    )

    require(
        bridge_contract[
            "status"
        ]
        == "FROZEN",
        (
            "R-GCN metadata bridge is not frozen."
        ),
    )

    print(
        "Canonical source integrity:           PASS"
    )

    print(
        "Forward source integrity:             PASS"
    )

    print(
        "Phase-5.3.1j composition:             FROZEN  PASS"
    )

    print(
        "Phase-5.3.1i.2 metadata bridge:       FROZEN  PASS"
    )

    print(
        "Optimizer steps entering preflight:   0"
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
    # Frozen static artifact hashes
    # =========================================================================

    banner(
        "FROZEN STATIC ARTIFACT INTEGRITY"
    )

    static_rows = []

    for (
        path_string,
        expected_sha,
    ) in EXPECTED_STATIC_SHA256.items():

        path = Path(
            path_string
        )

        actual_sha = (
            file_sha256(
                path
            )
        )

        passed = (
            actual_sha
            == expected_sha
        )

        static_rows.append(
            {
                "path": (
                    path_string
                ),

                "expected_sha256": (
                    expected_sha
                ),

                "actual_sha256": (
                    actual_sha
                ),

                "status": (
                    "PASS"
                    if passed
                    else "FAIL"
                ),
            }
        )

    static_df = pd.DataFrame(
        static_rows
    )

    require(
        (
            static_df[
                "status"
            ]
            == "PASS"
        ).all(),
        (
            "At least one frozen Phase-3/4 "
            "static artifact hash changed."
        ),
    )

    print(
        static_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Parse exact implementation sources
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
    # Build composed canonical model
    # =========================================================================

    banner(
        "RECONSTRUCT FROZEN COMPOSED CANONICAL MODEL"
    )

    (
        canonical_runtime,
        runtime_ast_sha,
    ) = build_canonical_runtime(
        canonical_tree
    )

    (
        forward_runtime,
        exact_methods,
        method_registry_df,
    ) = build_forward_runtime(
        forward_tree
    )

    (
        model,
        canonical_hash_fn,
    ) = compose_canonical_model(
        canonical_runtime,
        exact_methods,
    )

    model.train()

    parameter_count = sum(
        int(
            parameter.numel()
        )
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    parameter_tensor_count = sum(
        1
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    require(
        parameter_count
        == EXPECTED_PARAMETER_COUNT,
        (
            "Canonical trainable parameter "
            "count changed."
        ),
    )

    require(
        parameter_tensor_count
        == EXPECTED_PARAMETER_TENSORS,
        (
            "Canonical trainable tensor "
            "count changed."
        ),
    )

    canonical_hash_before_data = (
        canonical_hash_fn(
            model
        )
    )

    require(
        canonical_hash_before_data
        == EXPECTED_INITIAL_STATE_SHA256,
        (
            "Composed canonical model hash "
            "mismatch before data binding."
        ),
    )

    print(
        f"Exact numerical methods:              "
        f"{len(exact_methods)} / 7"
    )

    print(
        f"Parameter tensors:                    "
        f"{parameter_tensor_count}"
    )

    print(
        f"Trainable parameters:                 "
        f"{parameter_count:,}"
    )

    print()

    print(
        "Canonical SHA256:"
    )

    print(
        canonical_hash_before_data
    )

    # =========================================================================
    # Frozen selected Phase-4.6.2 validation case
    # =========================================================================

    banner(
        "FROZEN PHASE-4.6.2 VALIDATION CASE RECHECK"
    )

    selected_case = (
        load_json(
            SELECTED_PAIR_PATH
        )
    )

    selected_case_rows = []

    for (
        field,
        expected_value,
    ) in EXPECTED_SELECTED_CASE.items():

        actual_value = (
            selected_case.get(
                field
            )
        )

        passed = (
            actual_value
            == expected_value
        )

        selected_case_rows.append(
            {
                "field": (
                    field
                ),

                "expected": (
                    repr(
                        expected_value
                    )
                ),

                "actual": (
                    repr(
                        actual_value
                    )
                ),

                "status": (
                    "PASS"
                    if passed
                    else "FAIL"
                ),
            }
        )

    selected_case_df = pd.DataFrame(
        selected_case_rows
    )

    require(
        (
            selected_case_df[
                "status"
            ]
            == "PASS"
        ).all(),
        (
            "Frozen Phase-4.6.2 selected "
            "validation case changed."
        ),
    )

    print(
        selected_case_df.to_string(
            index=False
        )
    )

    interaction_id = str(
        selected_case[
            "interaction_id"
        ]
    )

    investor_id = str(
        selected_case[
            "investor_id"
        ]
    )

    startup_id = str(
        selected_case[
            "startup_id"
        ]
    )

    investor_global = int(
        selected_case[
            "investor_global_node_index"
        ]
    )

    startup_global = int(
        selected_case[
            "startup_global_node_index"
        ]
    )

    startup_local = int(
        selected_case[
            "startup_local_index"
        ]
    )

    # =========================================================================
    # Phase-2 event identity recheck
    # =========================================================================

    banner(
        "PHASE-2 EVENT IDENTITY / HOLDOUT SAFETY"
    )

    temporal_columns = [
        "interaction_id",
        "funding_round_id",
        "investor_id",
        "startup_id",
        "segment_number",
        "segment_label",
        "experiment_split",
    ]

    event = pd.read_parquet(
        TEMPORAL_SPLIT_PATH,
        columns=temporal_columns,
        filters=[
            (
                "interaction_id",
                "==",
                interaction_id,
            ),
        ],
    )

    require(
        len(
            event
        )
        == 1,
        (
            "Selected interaction is not uniquely "
            "present in frozen Phase-2 split."
        ),
    )

    event_row = (
        event.iloc[
            0
        ]
    )

    require(
        str(
            event_row[
                "funding_round_id"
            ]
        )
        == str(
            selected_case[
                "funding_round_id"
            ]
        ),
        (
            "Funding-round identity mismatch."
        ),
    )

    require(
        str(
            event_row[
                "investor_id"
            ]
        )
        == investor_id,
        (
            "Investor identity mismatch."
        ),
    )

    require(
        str(
            event_row[
                "startup_id"
            ]
        )
        == startup_id,
        (
            "Startup identity mismatch."
        ),
    )

    require(
        int(
            event_row[
                "segment_number"
            ]
        )
        == 60,
        (
            "Selected event is not T60."
        ),
    )

    require(
        str(
            event_row[
                "segment_label"
            ]
        )
        == "T60",
        (
            "Selected event segment label "
            "is not T60."
        ),
    )

    require(
        str(
            event_row[
                "experiment_split"
            ]
        )
        == "validation",
        (
            "Selected event is not validation."
        ),
    )

    print(
        "Selected Phase-2 row:                 1  PASS"
    )

    print(
        "Segment:                              T60  PASS"
    )

    print(
        "Split:                                validation  PASS"
    )

    print(
        "Target:                               positive / y=1"
    )

    print(
        "Test data consumed:                   NO"
    )

    # =========================================================================
    # Load static arrays / row alignment
    # =========================================================================

    banner(
        "LOAD AUTHORITATIVE REAL STATIC INPUTS"
    )

    node_index = pd.read_parquet(
        NODE_INDEX_PATH
    )

    doc_manifest = pd.read_parquet(
        DOC2VEC_MANIFEST_PATH
    )

    label_manifest = pd.read_parquet(
        LABEL_MANIFEST_PATH
    )

    doc2vec_all = np.load(
        DOC2VEC_PATH,
        mmap_mode="r",
    )

    labels_sparse = (
        sparse.load_npz(
            LABEL_MATRIX_PATH
        )
        .tocsr()
    )

    edge_index_np = np.load(
        EDGE_INDEX_PATH,
        mmap_mode="r",
    )

    edge_type_np = np.load(
        EDGE_TYPE_PATH,
        mmap_mode="r",
    )

    trend_period_ptr = np.load(
        TREND_PERIOD_PTR_PATH,
        mmap_mode="r",
    )

    trend_startup_indices = np.load(
        TREND_STARTUP_INDICES_PATH,
        mmap_mode="r",
    )

    trend_period_counts = np.load(
        TREND_PERIOD_COUNTS_PATH,
        mmap_mode="r",
    )

    require(
        len(
            node_index
        )
        == NUM_NODES,
        (
            "Node-index row count changed."
        ),
    )

    require(
        doc2vec_all.shape
        == (
            NUM_NODES,
            DOC2VEC_DIM,
        ),
        (
            "Doc2Vec shape changed."
        ),
    )

    require(
        labels_sparse.shape
        == (
            NUM_NODES,
            LABEL_DIM,
        ),
        (
            "Label matrix shape changed."
        ),
    )

    require(
        edge_index_np.shape
        == (
            2,
            NUM_EDGES,
        ),
        (
            "edge_index shape changed."
        ),
    )

    require(
        edge_type_np.shape
        == (
            NUM_EDGES,
        ),
        (
            "edge_type shape changed."
        ),
    )

    require(
        trend_period_ptr.shape
        == (
            NUM_INVESTORS
            * NUM_HISTORY_PERIODS
            + 1,
        ),
        (
            "Trend pointer shape changed."
        ),
    )

    require(
        trend_period_counts.shape
        == (
            NUM_INVESTORS
            * NUM_HISTORY_PERIODS,
        ),
        (
            "Trend count-array shape changed."
        ),
    )

    require(
        int(
            trend_period_ptr[
                -1
            ]
        )
        == len(
            trend_startup_indices
        ),
        (
            "Trend CSR pointer terminal "
            "does not match membership count."
        ),
    )

    require(
        int(
            np.asarray(
                trend_period_counts,
                dtype=np.int64,
            ).sum()
        )
        == len(
            trend_startup_indices
        ),
        (
            "Trend period counts do not sum "
            "to membership count."
        ),
    )

    # -------------------------------------------------------------------------
    # Global row identity
    # -------------------------------------------------------------------------

    phase3_rows = (
        node_index[
            "node_index"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    doc_rows = (
        doc_manifest[
            "doc2vec_feature_row"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    label_rows = (
        label_manifest[
            "label_feature_row"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    expected_rows = np.arange(
        NUM_NODES,
        dtype=np.int64,
    )

    require(
        np.array_equal(
            phase3_rows,
            expected_rows,
        ),
        (
            "Phase-3 node_index ordering changed."
        ),
    )

    require(
        np.array_equal(
            phase3_rows,
            doc_rows,
        ),
        (
            "Doc2Vec row alignment changed."
        ),
    )

    require(
        np.array_equal(
            phase3_rows,
            label_rows,
        ),
        (
            "Label row alignment changed."
        ),
    )

    print(
        f"Node index:                           "
        f"{len(node_index):,}"
    )

    print(
        f"Doc2Vec:                              "
        f"{doc2vec_all.shape}"
    )

    print(
        f"Labels:                               "
        f"{labels_sparse.shape}"
    )

    print(
        f"Graph:                                "
        f"{edge_index_np.shape}"
    )

    print(
        f"Trend memberships:                    "
        f"{len(trend_startup_indices):,}"
    )

    print(
        "Static row alignment:                 PASS"
    )

    # =========================================================================
    # Role/global-node identity
    # =========================================================================

    banner(
        "ROLE-SPECIFIC GLOBAL NODE RECHECK"
    )

    node_types = (
        node_index[
            "node_type"
        ]
        .astype(
            str
        )
        .str.lower()
    )

    raw_ids = (
        node_index[
            "raw_entity_id"
        ]
        .astype(
            str
        )
    )

    investor_match = (
        node_index.loc[
            (
                node_types
                == "investor"
            )
            & (
                raw_ids
                == investor_id
            )
        ]
    )

    startup_match = (
        node_index.loc[
            (
                node_types
                == "startup"
            )
            & (
                raw_ids
                == startup_id
            )
        ]
    )

    require(
        len(
            investor_match
        )
        == 1,
        (
            "Selected Investor does not map "
            "uniquely to Phase-3 node index."
        ),
    )

    require(
        len(
            startup_match
        )
        == 1,
        (
            "Selected Startup does not map "
            "uniquely to Phase-3 node index."
        ),
    )

    investor_global_actual = int(
        investor_match.iloc[
            0
        ][
            "node_index"
        ]
    )

    startup_global_actual = int(
        startup_match.iloc[
            0
        ][
            "node_index"
        ]
    )

    require(
        investor_global_actual
        == investor_global,
        (
            "Investor global node index changed."
        ),
    )

    require(
        startup_global_actual
        == startup_global,
        (
            "Startup global node index changed."
        ),
    )

    require(
        startup_global
        - NUM_INVESTORS
        == startup_local,
        (
            "Startup local role index changed."
        ),
    )

    require(
        0
        <= investor_global
        < NUM_INVESTORS,
        (
            "Investor global index outside "
            "Investor slice."
        ),
    )

    require(
        NUM_INVESTORS
        <= startup_global
        < NUM_NODES,
        (
            "Startup global index outside "
            "Startup slice."
        ),
    )

    print(
        f"Investor global:                      "
        f"{investor_global}"
    )

    print(
        f"Startup global:                       "
        f"{startup_global}"
    )

    print(
        f"Startup local:                        "
        f"{startup_local}"
    )

    print(
        "Role-node identity:                   PASS"
    )

    # =========================================================================
    # Structural incoming-degree recheck
    # =========================================================================

    graph_destination = np.asarray(
        edge_index_np[
            1
        ],
        dtype=np.int64,
    )

    incoming_degree = np.bincount(
        graph_destination,
        minlength=NUM_NODES,
    )

    require(
        int(
            incoming_degree[
                investor_global
            ]
        )
        == int(
            selected_case[
                "investor_incoming_degree"
            ]
        )
        == 5,
        (
            "Selected Investor incoming degree changed."
        ),
    )

    require(
        int(
            incoming_degree[
                startup_global
            ]
        )
        == int(
            selected_case[
                "startup_incoming_degree"
            ]
        )
        == 3,
        (
            "Selected Startup incoming degree changed."
        ),
    )

    # =========================================================================
    # Exact selected Investor T0..T59 history
    # =========================================================================

    banner(
        "SELECTED INVESTOR T0..T59 HISTORY RECONSTRUCTION"
    )

    period_base = (
        investor_global
        * NUM_HISTORY_PERIODS
    )

    period_memberships = []

    history_period_counts = []

    active_periods = 0
    multi_item_periods = 0

    for period in range(
        NUM_HISTORY_PERIODS
    ):

        slot = (
            period_base
            + period
        )

        start = int(
            trend_period_ptr[
                slot
            ]
        )

        end = int(
            trend_period_ptr[
                slot
                + 1
            ]
        )

        count = int(
            trend_period_counts[
                slot
            ]
        )

        require(
            end
            - start
            == count,
            (
                f"Trend CSR count mismatch "
                f"at period {period}."
            ),
        )

        startups = np.asarray(
            trend_startup_indices[
                start:end
            ],
            dtype=np.int64,
        )

        require(
            len(
                startups
            )
            == count,
            (
                f"Trend membership-length mismatch "
                f"at period {period}."
            ),
        )

        if len(
            startups
        ) > 0:

            require(
                bool(
                    (
                        startups
                        >= NUM_INVESTORS
                    ).all()
                ),
                (
                    f"Historical non-Startup node "
                    f"in period {period}."
                ),
            )

            require(
                bool(
                    (
                        startups
                        < NUM_NODES
                    ).all()
                ),
                (
                    f"Historical Startup index "
                    f"outside node universe "
                    f"in period {period}."
                ),
            )

            active_periods += 1

            if len(
                startups
            ) >= 2:

                multi_item_periods += 1

        period_memberships.append(
            startups
        )

        history_period_counts.append(
            count
        )

    nonempty_history = [
        values
        for values
        in period_memberships
        if len(
            values
        )
        > 0
    ]

    require(
        len(
            nonempty_history
        )
        > 0,
        (
            "Frozen selected Investor unexpectedly "
            "has no T0..T59 history."
        ),
    )

    history_all = np.concatenate(
        nonempty_history
    )

    history_unique = np.unique(
        history_all
    )

    history_total = int(
        len(
            history_all
        )
    )

    max_period_items = int(
        max(
            history_period_counts
        )
    )

    require(
        history_total
        == int(
            selected_case[
                "history_memberships"
            ]
        )
        == 7,
        (
            "History membership count changed."
        ),
    )

    require(
        active_periods
        == int(
            selected_case[
                "active_history_periods"
            ]
        )
        == 4,
        (
            "Active history-period count changed."
        ),
    )

    require(
        multi_item_periods
        == int(
            selected_case[
                "multi_item_history_periods"
            ]
        )
        == 1,
        (
            "Multi-item history-period count changed."
        ),
    )

    require(
        max_period_items
        == int(
            selected_case[
                "max_period_items"
            ]
        )
        == 4,
        (
            "Max period-item count changed."
        ),
    )

    require(
        len(
            history_unique
        )
        == 5,
        (
            "Unique historical Startup count changed."
        ),
    )

    print(
        f"History memberships:                  "
        f"{history_total}"
    )

    print(
        f"Unique historical Startups:           "
        f"{len(history_unique)}"
    )

    print(
        f"Active periods:                       "
        f"{active_periods}"
    )

    print(
        f"Empty periods:                        "
        f"{NUM_HISTORY_PERIODS - active_periods}"
    )

    print(
        f"Multi-item periods:                   "
        f"{multi_item_periods}"
    )

    print(
        f"Max period items:                     "
        f"{max_period_items}"
    )

    # =========================================================================
    # Required description rows
    # =========================================================================

    banner(
        "REAL DESCRIPTION SUBSET"
    )

    required_global_nodes = np.unique(
        np.concatenate(
            [
                np.array(
                    [
                        investor_global,
                        startup_global,
                    ],
                    dtype=np.int64,
                ),

                history_unique,
            ]
        )
    )

    require(
        len(
            required_global_nodes
        )
        == 7,
        (
            "Required description subset "
            "size changed."
        ),
    )

    doc2vec_subset_np = np.array(
        doc2vec_all[
            required_global_nodes
        ],
        dtype=np.float32,
        copy=True,
    )

    label_subset_np = (
        labels_sparse[
            required_global_nodes
        ]
        .toarray()
        .astype(
            np.float32,
            copy=False,
        )
    )

    require(
        doc2vec_subset_np.shape
        == (
            7,
            DOC2VEC_DIM,
        ),
        (
            "Doc2Vec subset shape invalid."
        ),
    )

    require(
        label_subset_np.shape
        == (
            7,
            LABEL_DIM,
        ),
        (
            "Label subset shape invalid."
        ),
    )

    require(
        bool(
            np.isfinite(
                doc2vec_subset_np
            ).all()
        ),
        (
            "Doc2Vec subset contains non-finite "
            "values."
        ),
    )

    require(
        bool(
            np.isfinite(
                label_subset_np
            ).all()
        ),
        (
            "Label subset contains non-finite "
            "values."
        ),
    )

    # Frozen selected endpoints must have non-zero Doc2Vec.
    node_to_subset_position = {
        int(
            node
        ): int(
            position
        )
        for (
            position,
            node,
        ) in enumerate(
            required_global_nodes
        )
    }

    investor_description_position = (
        node_to_subset_position[
            investor_global
        ]
    )

    startup_description_position = (
        node_to_subset_position[
            startup_global
        ]
    )

    require(
        bool(
            np.any(
                doc2vec_subset_np[
                    investor_description_position
                ]
                != 0
            )
        ),
        (
            "Selected Investor Doc2Vec became zero."
        ),
    )

    require(
        bool(
            np.any(
                doc2vec_subset_np[
                    startup_description_position
                ]
                != 0
            )
        ),
        (
            "Selected Startup Doc2Vec became zero."
        ),
    )

    selected_startup_label_count = int(
        np.count_nonzero(
            label_subset_np[
                startup_description_position
            ]
        )
    )

    require(
        selected_startup_label_count
        == int(
            selected_case[
                "startup_label_count"
            ]
        )
        == 5,
        (
            "Selected Startup label count changed."
        ),
    )

    print(
        f"Required global description nodes:    "
        f"{len(required_global_nodes)}"
    )

    print(
        f"Doc2Vec subset:                       "
        f"{doc2vec_subset_np.shape}"
    )

    print(
        f"Label subset:                         "
        f"{label_subset_np.shape}"
    )

    print(
        f"Startup labels:                       "
        f"{selected_startup_label_count}"
    )

    # =========================================================================
    # Torch input construction
    # =========================================================================

    banner(
        "TORCH RUNTIME INPUT BINDING"
    )

    edge_index = torch.from_numpy(
        np.array(
            edge_index_np,
            dtype=np.int64,
            copy=True,
        )
    )

    edge_type = torch.from_numpy(
        np.array(
            edge_type_np,
            dtype=np.int64,
            copy=True,
        )
    )

    doc2vec_subset = torch.from_numpy(
        doc2vec_subset_np
    )

    label_subset = torch.from_numpy(
        label_subset_np
    )

    require(
        edge_index.dtype
        == torch.int64,
        (
            "edge_index dtype is not int64."
        ),
    )

    require(
        edge_type.dtype
        == torch.int64,
        (
            "edge_type dtype is not int64."
        ),
    )

    require(
        doc2vec_subset.dtype
        == torch.float32,
        (
            "Doc2Vec runtime dtype changed."
        ),
    )

    require(
        label_subset.dtype
        == torch.float32,
        (
            "Label runtime dtype changed."
        ),
    )

    print(
        f"edge_index:                           "
        f"{tuple(edge_index.shape)} "
        f"{edge_index.dtype}"
    )

    print(
        f"edge_type:                            "
        f"{tuple(edge_type.shape)} "
        f"{edge_type.dtype}"
    )

    print(
        f"Doc2Vec subset:                       "
        f"{tuple(doc2vec_subset.shape)} "
        f"{doc2vec_subset.dtype}"
    )

    print(
        f"Label subset:                         "
        f"{tuple(label_subset.shape)} "
        f"{label_subset.dtype}"
    )

    # =========================================================================
    # Parameter-state snapshot before numerical execution
    # =========================================================================

    parameter_hashes_before = (
        parameter_hashes(
            model
        )
    )

    state_dict_hash_before = (
        logical_state_dict_sha256(
            model
        )
    )

    canonical_hash_before_forward = (
        canonical_hash_fn(
            model
        )
    )

    require(
        canonical_hash_before_forward
        == EXPECTED_INITIAL_STATE_SHA256,
        (
            "Canonical state changed during "
            "data binding."
        ),
    )

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
    # FULL CANONICAL COMPOSED FORWARD
    # =========================================================================

    banner(
        "FIRST CANONICAL COMPOSED REAL-DATA FORWARD"
    )

    # -------------------------------------------------------------------------
    # Shared latent matrix -> full graph structural forward
    # -------------------------------------------------------------------------

    latent_all = torch.cat(
        [
            model.investor_embedding.weight,
            model.startup_embedding.weight,
        ],
        dim=0,
    )

    require(
        latent_all.shape
        == (
            NUM_NODES,
            LATENT_DIM,
        ),
        (
            "latent_all shape invalid."
        ),
    )

    structural = (
        model.preference_propagation(
            latent_all,
            edge_index,
            edge_type,
        )
    )

    require(
        isinstance(
            structural,
            dict,
        ),
        (
            "PreferencePropagation output "
            "is not dict."
        ),
    )

    require(
        "F_s"
        in structural,
        (
            "PreferencePropagation lacks F_s."
        ),
    )

    F_s_all = (
        structural[
            "F_s"
        ]
    )

    require(
        F_s_all.shape
        == (
            NUM_NODES,
            STRUCTURAL_DIM,
        ),
        (
            "F_s_all shape invalid."
        ),
    )

    require(
        bool(
            torch.isfinite(
                F_s_all
            ).all()
        ),
        (
            "F_s_all contains non-finite values."
        ),
    )

    # -------------------------------------------------------------------------
    # Description branch
    # -------------------------------------------------------------------------

    description_subset = (
        model.description_encoder(
            doc2vec_subset,
            label_subset,
        )
    )

    require(
        description_subset.shape
        == (
            len(
                required_global_nodes
            ),
            DESCRIPTION_DIM,
        ),
        (
            "Description subset output "
            "shape invalid."
        ),
    )

    require(
        bool(
            torch.isfinite(
                description_subset
            ).all()
        ),
        (
            "Description output contains "
            "non-finite values."
        ),
    )

    F_d_investor = (
        description_subset[
            investor_description_position
            :
            investor_description_position
            + 1
        ]
    )

    F_d_startup = (
        description_subset[
            startup_description_position
            :
            startup_description_position
            + 1
        ]
    )

    # -------------------------------------------------------------------------
    # Direct shared latent lookups
    # -------------------------------------------------------------------------

    investor_index_tensor = torch.tensor(
        [
            investor_global,
        ],
        dtype=torch.int64,
    )

    startup_index_tensor = torch.tensor(
        [
            startup_local,
        ],
        dtype=torch.int64,
    )

    L_o = (
        model.investor_embedding(
            investor_index_tensor
        )
    )

    L_b = (
        model.startup_embedding(
            startup_index_tensor
        )
    )

    require(
        L_o.shape
        == (
            1,
            LATENT_DIM,
        ),
        (
            "L_o shape invalid."
        ),
    )

    require(
        L_b.shape
        == (
            1,
            LATENT_DIM,
        ),
        (
            "L_b shape invalid."
        ),
    )

    # -------------------------------------------------------------------------
    # Frozen trend Investor query
    # -------------------------------------------------------------------------

    trend_query = torch.cat(
        [
            L_o[
                0
            ],
            F_d_investor[
                0
            ],
        ],
        dim=0,
    )

    require(
        trend_query.shape
        == (
            TREND_QUERY_DIM,
        ),
        (
            "Trend query shape invalid."
        ),
    )

    # -------------------------------------------------------------------------
    # Exact T0..T59 period attention
    # -------------------------------------------------------------------------

    period_vectors = []

    attention_rows = []

    for (
        period,
        startup_globals,
    ) in enumerate(
        period_memberships
    ):

        item_count = int(
            len(
                startup_globals
            )
        )

        if item_count == 0:

            period_vector = torch.zeros(
                TREND_ITEM_DIM,
                dtype=L_o.dtype,
                device=L_o.device,
            )

            alpha = None

            attention_rows.append(
                {
                    "period": (
                        period
                    ),

                    "item_count": (
                        0
                    ),

                    "empty_period": (
                        True
                    ),

                    "attention_sum": (
                        None
                    ),

                    "finite": (
                        True
                    ),
                }
            )

        else:

            startup_locals = (
                startup_globals
                - NUM_INVESTORS
            )

            startup_locals_tensor = torch.from_numpy(
                np.array(
                    startup_locals,
                    dtype=np.int64,
                    copy=True,
                )
            )

            period_latents = (
                model.startup_embedding(
                    startup_locals_tensor
                )
            )

            period_positions = torch.tensor(
                [
                    node_to_subset_position[
                        int(
                            global_node
                        )
                    ]
                    for global_node
                    in startup_globals
                ],
                dtype=torch.int64,
            )

            period_descriptions = (
                description_subset[
                    period_positions
                ]
            )

            period_items = torch.cat(
                [
                    period_latents,
                    period_descriptions,
                ],
                dim=1,
            )

            require(
                period_items.shape
                == (
                    item_count,
                    TREND_ITEM_DIM,
                ),
                (
                    f"Trend item shape invalid "
                    f"at period {period}."
                ),
            )

            (
                period_vector,
                alpha,
            ) = (
                model.trend_extractor.attend_period(
                    trend_query,
                    period_items,
                )
            )

            require(
                period_vector.shape
                == (
                    TREND_ITEM_DIM,
                ),
                (
                    f"Period vector shape invalid "
                    f"at period {period}."
                ),
            )

            require(
                alpha.shape
                == (
                    item_count,
                ),
                (
                    f"Attention alpha shape invalid "
                    f"at period {period}."
                ),
            )

            require(
                bool(
                    torch.isfinite(
                        alpha
                    ).all()
                ),
                (
                    f"Non-finite attention "
                    f"at period {period}."
                ),
            )

            attention_sum = float(
                alpha.detach().sum()
            )

            require(
                abs(
                    attention_sum
                    - 1.0
                )
                <= 1e-6,
                (
                    f"Attention does not sum to 1 "
                    f"at period {period}."
                ),
            )

            attention_rows.append(
                {
                    "period": (
                        period
                    ),

                    "item_count": (
                        item_count
                    ),

                    "empty_period": (
                        False
                    ),

                    "attention_sum": (
                        attention_sum
                    ),

                    "finite": (
                        True
                    ),
                }
            )

        require(
            bool(
                torch.isfinite(
                    period_vector
                ).all()
            ),
            (
                f"Period vector non-finite "
                f"at period {period}."
            ),
        )

        period_vectors.append(
            period_vector
        )

    trend_sequence = torch.stack(
        period_vectors,
        dim=0,
    ).unsqueeze(
        0
    )

    require(
        trend_sequence.shape
        == (
            1,
            NUM_HISTORY_PERIODS,
            TREND_ITEM_DIM,
        ),
        (
            "Trend sequence shape invalid."
        ),
    )

    (
        F_t,
        gru_output,
    ) = (
        model.trend_extractor.encode_sequence(
            trend_sequence
        )
    )

    require(
        gru_output.shape
        == (
            1,
            NUM_HISTORY_PERIODS,
            TREND_DIM,
        ),
        (
            "GRU output shape invalid."
        ),
    )

    require(
        F_t.shape
        == (
            1,
            TREND_DIM,
        ),
        (
            "F_t shape invalid."
        ),
    )

    require(
        bool(
            torch.isfinite(
                F_t
            ).all()
        ),
        (
            "F_t contains non-finite values."
        ),
    )

    # -------------------------------------------------------------------------
    # Structural endpoint features
    # -------------------------------------------------------------------------

    F_s_investor = (
        F_s_all[
            investor_global
            :
            investor_global
            + 1
        ]
    )

    F_s_startup = (
        F_s_all[
            startup_global
            :
            startup_global
            + 1
        ]
    )

    # -------------------------------------------------------------------------
    # Exact frozen scoring representation order
    # -------------------------------------------------------------------------

    investor_representation = torch.cat(
        [
            F_t,
            L_o,
            F_d_investor,
            F_s_investor,
        ],
        dim=1,
    )

    startup_representation = torch.cat(
        [
            L_b,
            F_d_startup,
            F_s_startup,
        ],
        dim=1,
    )

    pair_representation = torch.cat(
        [
            investor_representation,
            startup_representation,
        ],
        dim=1,
    )

    require(
        investor_representation.shape
        == (
            1,
            INVESTOR_SCORING_DIM,
        ),
        (
            "Investor representation "
            "shape invalid."
        ),
    )

    require(
        startup_representation.shape
        == (
            1,
            STARTUP_SCORING_DIM,
        ),
        (
            "Startup representation "
            "shape invalid."
        ),
    )

    require(
        pair_representation.shape
        == (
            1,
            PAIR_DIM,
        ),
        (
            "Pair representation "
            "shape invalid."
        ),
    )

    # -------------------------------------------------------------------------
    # Exact ScoringMLP.forward
    # -------------------------------------------------------------------------

    scoring = (
        model.scoring_mlp(
            pair_representation
        )
    )

    require(
        isinstance(
            scoring,
            dict,
        ),
        (
            "ScoringMLP output is not dict."
        ),
    )

    require(
        "logit"
        in scoring,
        (
            "Scoring output lacks logit."
        ),
    )

    # -------------------------------------------------------------------------
    # Execute exact frozen Phase-4.6.2 logit/probability/target/BCE
    # assignments directly from the frozen source AST.
    # -------------------------------------------------------------------------

    (
        loss_environment,
        source_assignment_df,
    ) = execute_exact_loss_assignments(
        forward_tree,
        forward_runtime,
        scoring,
    )

    logit = (
        loss_environment[
            "logit"
        ]
    )

    probability = (
        loss_environment[
            "probability"
        ]
    )

    target = (
        loss_environment[
            "target"
        ]
    )

    criterion = (
        loss_environment[
            "criterion"
        ]
    )

    loss = (
        loss_environment[
            "loss"
        ]
    )

    manual_stable_bce = (
        loss_environment[
            "manual_stable_bce"
        ]
    )

    bce_equivalent = (
        loss_environment[
            "bce_equivalent"
        ]
    )

    require(
        logit.shape
        == (
            1,
            1,
        ),
        (
            "Logit shape invalid."
        ),
    )

    require(
        probability.shape
        == (
            1,
            1,
        ),
        (
            "Probability shape invalid."
        ),
    )

    require(
        target.shape
        == logit.shape,
        (
            "Target shape does not match logit."
        ),
    )

    require(
        bool(
            torch.isfinite(
                logit
            ).all()
        ),
        (
            "Logit non-finite."
        ),
    )

    require(
        bool(
            torch.isfinite(
                probability
            ).all()
        ),
        (
            "Probability non-finite."
        ),
    )

    require(
        bool(
            (
                probability
                >= 0.0
            ).all()
        )
        and bool(
            (
                probability
                <= 1.0
            ).all()
        ),
        (
            "Probability outside [0,1]."
        ),
    )

    require(
        torch.equal(
            probability,
            torch.sigmoid(
                logit
            ),
        ),
        (
            "Probability is not exact "
            "sigmoid(logit)."
        ),
    )

    require(
        bool(
            (
                target
                == 1.0
            ).all()
        ),
        (
            "Frozen validation-positive target "
            "is not exactly 1."
        ),
    )

    require(
        isinstance(
            criterion,
            torch.nn.BCEWithLogitsLoss,
        ),
        (
            "Criterion is not BCEWithLogitsLoss."
        ),
    )

    require(
        bool(
            torch.isfinite(
                loss
            )
        ),
        (
            "BCE loss is non-finite."
        ),
    )

    require(
        bool(
            torch.isfinite(
                manual_stable_bce
            )
        ),
        (
            "Manual stable BCE is non-finite."
        ),
    )

    require(
        bool(
            bce_equivalent
        ),
        (
            "BCEWithLogitsLoss no longer "
            "matches manual stable BCE."
        ),
    )

    # =========================================================================
    # Shape audit before backward
    # =========================================================================

    shape_rows = [
        {
            "feature": (
                "Doc2Vec subset"
            ),
            "actual": (
                str(
                    tuple(
                        doc2vec_subset.shape
                    )
                )
            ),
            "expected": (
                "(7, 32)"
            ),
        },

        {
            "feature": (
                "Label subset"
            ),
            "actual": (
                str(
                    tuple(
                        label_subset.shape
                    )
                )
            ),
            "expected": (
                "(7, 802)"
            ),
        },

        {
            "feature": (
                "Description subset"
            ),
            "actual": (
                str(
                    tuple(
                        description_subset.shape
                    )
                )
            ),
            "expected": (
                "(7, 40)"
            ),
        },

        {
            "feature": (
                "latent_all"
            ),
            "actual": (
                str(
                    tuple(
                        latent_all.shape
                    )
                )
            ),
            "expected": (
                "(477564, 40)"
            ),
        },

        {
            "feature": (
                "F_s_all"
            ),
            "actual": (
                str(
                    tuple(
                        F_s_all.shape
                    )
                )
            ),
            "expected": (
                "(477564, 40)"
            ),
        },

        {
            "feature": (
                "trend_query"
            ),
            "actual": (
                str(
                    tuple(
                        trend_query.shape
                    )
                )
            ),
            "expected": (
                "(80,)"
            ),
        },

        {
            "feature": (
                "trend_sequence"
            ),
            "actual": (
                str(
                    tuple(
                        trend_sequence.shape
                    )
                )
            ),
            "expected": (
                "(1, 60, 80)"
            ),
        },

        {
            "feature": (
                "F_t"
            ),
            "actual": (
                str(
                    tuple(
                        F_t.shape
                    )
                )
            ),
            "expected": (
                "(1, 40)"
            ),
        },

        {
            "feature": (
                "Investor representation"
            ),
            "actual": (
                str(
                    tuple(
                        investor_representation.shape
                    )
                )
            ),
            "expected": (
                "(1, 160)"
            ),
        },

        {
            "feature": (
                "Startup representation"
            ),
            "actual": (
                str(
                    tuple(
                        startup_representation.shape
                    )
                )
            ),
            "expected": (
                "(1, 120)"
            ),
        },

        {
            "feature": (
                "Pair representation"
            ),
            "actual": (
                str(
                    tuple(
                        pair_representation.shape
                    )
                )
            ),
            "expected": (
                "(1, 280)"
            ),
        },

        {
            "feature": (
                "logit"
            ),
            "actual": (
                str(
                    tuple(
                        logit.shape
                    )
                )
            ),
            "expected": (
                "(1, 1)"
            ),
        },
    ]

    shape_df = pd.DataFrame(
        shape_rows
    )

    shape_df[
        "status"
    ] = np.where(
        shape_df[
            "actual"
        ]
        == shape_df[
            "expected"
        ],
        "PASS",
        "FAIL",
    )

    require(
        (
            shape_df[
                "status"
            ]
            == "PASS"
        ).all(),
        (
            "At least one canonical forward "
            "shape check failed."
        ),
    )

    banner(
        "CANONICAL FORWARD RESULT"
    )

    print(
        shape_df.to_string(
            index=False
        )
    )

    print()

    print(
        f"Canonical Kaiming logit:              "
        f"{float(logit.detach().item()):.10f}"
    )

    print(
        f"Canonical probability:                "
        f"{float(probability.detach().item()):.10f}"
    )

    print(
        f"Target:                               "
        f"{float(target.detach().item()):.1f}"
    )

    print(
        f"BCEWithLogitsLoss:                    "
        f"{float(loss.detach().item()):.10f}"
    )

    print(
        f"Manual stable BCE:                    "
        f"{float(manual_stable_bce.detach().item()):.10f}"
    )

    print(
        f"BCE equivalence:                      "
        f"{bool(bce_equivalent)}"
    )

    # =========================================================================
    # Parameter values MUST still be canonical after forward
    # =========================================================================

    canonical_hash_after_forward = (
        canonical_hash_fn(
            model
        )
    )

    require(
        canonical_hash_after_forward
        == EXPECTED_INITIAL_STATE_SHA256,
        (
            "Forward pass modified canonical "
            "parameter values."
        ),
    )

    # =========================================================================
    # FIRST CANONICAL BACKWARD
    # =========================================================================

    banner(
        "FIRST CANONICAL COMPOSED BACKWARD PASS"
    )

    loss.backward()

    # =========================================================================
    # Complete parameter-gradient audit
    # =========================================================================

    gradient_rows = []

    for (
        name,
        parameter,
    ) in model.named_parameters():

        gradient = (
            parameter.grad
        )

        exists = (
            gradient
            is not None
        )

        finite = (
            exists
            and bool(
                torch.isfinite(
                    gradient
                ).all()
            )
        )

        abs_sum = (
            float(
                gradient
                .detach()
                .abs()
                .sum()
            )
            if exists
            else 0.0
        )

        nonzero = (
            exists
            and abs_sum
            > 0.0
        )

        gradient_rows.append(
            {
                "parameter": (
                    name
                ),

                "gradient_exists": (
                    exists
                ),

                "gradient_finite": (
                    finite
                ),

                "gradient_abs_sum": (
                    abs_sum
                ),

                "gradient_nonzero": (
                    nonzero
                ),

                "status": (
                    "PASS"
                    if (
                        exists
                        and finite
                        and nonzero
                    )
                    else "FAIL"
                ),
            }
        )

    gradient_df = pd.DataFrame(
        gradient_rows
    )

    require(
        len(
            gradient_df
        )
        == EXPECTED_PARAMETER_TENSORS,
        (
            "Gradient audit did not cover "
            "all 32 trainable tensors."
        ),
    )

    require(
        (
            gradient_df[
                "status"
            ]
            == "PASS"
        ).all(),
        (
            "At least one canonical parameter "
            "tensor has missing, non-finite, "
            "or zero gradient."
        ),
    )

    # -------------------------------------------------------------------------
    # Shared latent row paths
    # -------------------------------------------------------------------------

    investor_embedding_gradient = (
        model
        .investor_embedding
        .weight
        .grad
    )

    startup_embedding_gradient = (
        model
        .startup_embedding
        .weight
        .grad
    )

    require(
        investor_embedding_gradient
        is not None,
        (
            "Investor embedding gradient missing."
        ),
    )

    require(
        startup_embedding_gradient
        is not None,
        (
            "Startup embedding gradient missing."
        ),
    )

    investor_row_gradient = float(
        investor_embedding_gradient[
            investor_global
        ]
        .abs()
        .sum()
    )

    candidate_row_gradient = float(
        startup_embedding_gradient[
            startup_local
        ]
        .abs()
        .sum()
    )

    history_unique_locals = (
        history_unique
        - NUM_INVESTORS
    )

    history_gradient_abs_sums = (
        startup_embedding_gradient[
            torch.from_numpy(
                np.array(
                    history_unique_locals,
                    dtype=np.int64,
                    copy=True,
                )
            )
        ]
        .abs()
        .sum(
            dim=1
        )
    )

    max_history_gradient = float(
        history_gradient_abs_sums.max()
    )

    max_history_position = int(
        torch.argmax(
            history_gradient_abs_sums
        )
    )

    max_history_global = int(
        history_unique[
            max_history_position
        ]
    )

    require(
        investor_row_gradient
        > 0.0,
        (
            "Selected Investor embedding row "
            "received zero gradient."
        ),
    )

    require(
        candidate_row_gradient
        > 0.0,
        (
            "Candidate Startup embedding row "
            "received zero gradient."
        ),
    )

    require(
        max_history_gradient
        > 0.0,
        (
            "All historical Startup embedding "
            "rows received zero gradient."
        ),
    )

    attention_gradient_abs_sum = float(
        model
        .trend_extractor
        .attention_weight
        .grad
        .abs()
        .sum()
    )

    require(
        attention_gradient_abs_sum
        > 0.0,
        (
            "Attention weight received zero "
            "gradient despite multi-item history."
        ),
    )

    print(
        f"Parameter tensors with gradients:     "
        f"{int(gradient_df['gradient_exists'].sum())}"
    )

    print(
        f"Finite parameter gradients:           "
        f"{int(gradient_df['gradient_finite'].sum())}"
    )

    print(
        f"Non-zero parameter gradients:         "
        f"{int(gradient_df['gradient_nonzero'].sum())}"
    )

    print()

    print(
        f"Selected Investor latent row abs_sum: "
        f"{investor_row_gradient:.10e}"
    )

    print(
        f"Candidate Startup row abs_sum:        "
        f"{candidate_row_gradient:.10e}"
    )

    print(
        f"Historical Startup max abs_sum:       "
        f"{max_history_gradient:.10e}"
    )

    print(
        f"Historical Startup global node:       "
        f"{max_history_global}"
    )

    print(
        f"Attention-weight gradient abs_sum:    "
        f"{attention_gradient_abs_sum:.10e}"
    )

    # =========================================================================
    # Parameter-state MUST remain byte-exact after backward
    # =========================================================================

    banner(
        "POST-BACKWARD CANONICAL STATE NEUTRALITY"
    )

    canonical_hash_after_backward = (
        canonical_hash_fn(
            model
        )
    )

    parameter_hashes_after = (
        parameter_hashes(
            model
        )
    )

    state_dict_hash_after = (
        logical_state_dict_sha256(
            model
        )
    )

    require(
        canonical_hash_after_backward
        == EXPECTED_INITIAL_STATE_SHA256,
        (
            "Backward pass modified canonical "
            "parameter values."
        ),
    )

    require(
        parameter_hashes_before
        == parameter_hashes_after,
        (
            "At least one parameter tensor "
            "changed during forward/backward."
        ),
    )

    require(
        state_dict_hash_before
        == state_dict_hash_after,
        (
            "state_dict changed during "
            "forward/backward."
        ),
    )

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
            "Forward/backward changed "
            "Python RNG state."
        ),
    )

    require(
        numpy_rng_state_equal(
            numpy_rng_before,
            numpy_rng_after,
        ),
        (
            "Forward/backward changed "
            "NumPy global RNG state."
        ),
    )

    require(
        torch.equal(
            torch_rng_before,
            torch_rng_after,
        ),
        (
            "Forward/backward changed "
            "torch global RNG state."
        ),
    )

    state_rows = [
        {
            "check": (
                "canonical_hash_before_forward"
            ),

            "value": (
                canonical_hash_before_forward
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
                "canonical_hash_after_forward"
            ),

            "value": (
                canonical_hash_after_forward
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
                "canonical_hash_after_backward"
            ),

            "value": (
                canonical_hash_after_backward
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
                "parameter_tensor_bytes"
            ),

            "value": (
                str(
                    parameter_hashes_before
                    == parameter_hashes_after
                )
            ),

            "expected": (
                "True"
            ),

            "status": (
                "PASS"
            ),
        },

        {
            "check": (
                "logical_state_dict_sha256"
            ),

            "value": (
                state_dict_hash_after
            ),

            "expected": (
                state_dict_hash_before
            ),

            "status": (
                "PASS"
            ),
        },

        {
            "check": (
                "python_rng_neutral"
            ),

            "value": (
                str(
                    python_rng_before
                    == python_rng_after
                )
            ),

            "expected": (
                "True"
            ),

            "status": (
                "PASS"
            ),
        },

        {
            "check": (
                "numpy_rng_neutral"
            ),

            "value": (
                str(
                    numpy_rng_state_equal(
                        numpy_rng_before,
                        numpy_rng_after,
                    )
                )
            ),

            "expected": (
                "True"
            ),

            "status": (
                "PASS"
            ),
        },

        {
            "check": (
                "torch_rng_neutral"
            ),

            "value": (
                str(
                    torch.equal(
                        torch_rng_before,
                        torch_rng_after,
                    )
                )
            ),

            "expected": (
                "True"
            ),

            "status": (
                "PASS"
            ),
        },
    ]

    state_df = pd.DataFrame(
        state_rows
    )

    print(
        state_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Attention diagnostics
    # =========================================================================

    attention_df = pd.DataFrame(
        attention_rows
    )

    require(
        int(
            (
                attention_df[
                    "empty_period"
                ]
                == False
            ).sum()
        )
        == active_periods
        == 4,
        (
            "Active attention-period count changed."
        ),
    )

    require(
        int(
            (
                attention_df[
                    "item_count"
                ]
                >= 2
            ).sum()
        )
        == multi_item_periods
        == 1,
        (
            "Multi-item attention-period "
            "count changed."
        ),
    )

    # =========================================================================
    # Final invariants
    # =========================================================================

    banner(
        "FINAL PHASE-5.3.1k PREFLIGHT INVARIANTS"
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
            "phase_5_3_1j_contract_frozen",
            (
                phase_5_3_1j_contract[
                    "status"
                ]
                == "FROZEN"
            ),
        ),

        (
            "canonical_runtime_ast_exact",
            (
                runtime_ast_sha
                == EXPECTED_RUNTIME_AST_SHA256
            ),
        ),

        (
            "seven_exact_methods_composed",
            (
                len(
                    exact_methods
                )
                == 7
            ),
        ),

        (
            "canonical_hash_before_forward_exact",
            (
                canonical_hash_before_forward
                == EXPECTED_INITIAL_STATE_SHA256
            ),
        ),

        (
            "frozen_selected_validation_case_exact",
            bool(
                (
                    selected_case_df[
                        "status"
                    ]
                    == "PASS"
                ).all()
            ),
        ),

        (
            "selected_case_phase2_validation_t60_exact",
            (
                len(
                    event
                )
                == 1
                and int(
                    event_row[
                        "segment_number"
                    ]
                )
                == 60
                and str(
                    event_row[
                        "experiment_split"
                    ]
                )
                == "validation"
            ),
        ),

        (
            "no_test_event_consumed",
            (
                str(
                    event_row[
                        "experiment_split"
                    ]
                )
                != "test"
            ),
        ),

        (
            "all_frozen_static_hashes_exact",
            bool(
                (
                    static_df[
                        "status"
                    ]
                    == "PASS"
                ).all()
            ),
        ),

        (
            "static_feature_row_alignment_exact",
            (
                np.array_equal(
                    phase3_rows,
                    doc_rows,
                )
                and np.array_equal(
                    phase3_rows,
                    label_rows,
                )
            ),
        ),

        (
            "selected_history_memberships_exact",
            (
                history_total
                == 7
            ),
        ),

        (
            "selected_history_active_periods_exact",
            (
                active_periods
                == 4
            ),
        ),

        (
            "selected_history_multi_item_period_exact",
            (
                multi_item_periods
                == 1
            ),
        ),

        (
            "canonical_forward_shapes_exact",
            bool(
                (
                    shape_df[
                        "status"
                    ]
                    == "PASS"
                ).all()
            ),
        ),

        (
            "logit_finite",
            bool(
                torch.isfinite(
                    logit
                ).all()
            ),
        ),

        (
            "target_exact_positive_one",
            bool(
                (
                    target
                    == 1.0
                ).all()
            ),
        ),

        (
            "BCEWithLogitsLoss_finite",
            bool(
                torch.isfinite(
                    loss
                )
            ),
        ),

        (
            "manual_stable_BCE_equivalent",
            bool(
                bce_equivalent
            ),
        ),

        (
            "all_32_parameter_gradients_exist",
            (
                int(
                    gradient_df[
                        "gradient_exists"
                    ].sum()
                )
                == EXPECTED_PARAMETER_TENSORS
            ),
        ),

        (
            "all_32_parameter_gradients_finite",
            (
                int(
                    gradient_df[
                        "gradient_finite"
                    ].sum()
                )
                == EXPECTED_PARAMETER_TENSORS
            ),
        ),

        (
            "all_32_parameter_gradients_nonzero",
            (
                int(
                    gradient_df[
                        "gradient_nonzero"
                    ].sum()
                )
                == EXPECTED_PARAMETER_TENSORS
            ),
        ),

        (
            "selected_investor_latent_row_gradient_nonzero",
            (
                investor_row_gradient
                > 0.0
            ),
        ),

        (
            "candidate_startup_latent_row_gradient_nonzero",
            (
                candidate_row_gradient
                > 0.0
            ),
        ),

        (
            "historical_startup_latent_gradient_nonzero",
            (
                max_history_gradient
                > 0.0
            ),
        ),

        (
            "trend_attention_gradient_nonzero",
            (
                attention_gradient_abs_sum
                > 0.0
            ),
        ),

        (
            "canonical_hash_after_forward_exact",
            (
                canonical_hash_after_forward
                == EXPECTED_INITIAL_STATE_SHA256
            ),
        ),

        (
            "canonical_hash_after_backward_exact",
            (
                canonical_hash_after_backward
                == EXPECTED_INITIAL_STATE_SHA256
            ),
        ),

        (
            "parameter_bytes_unchanged_after_backward",
            (
                parameter_hashes_before
                == parameter_hashes_after
            ),
        ),

        (
            "state_dict_unchanged_after_backward",
            (
                state_dict_hash_before
                == state_dict_hash_after
            ),
        ),

        (
            "forward_backward_rng_neutral",
            (
                python_rng_before
                == python_rng_after
                and numpy_rng_state_equal(
                    numpy_rng_before,
                    numpy_rng_after,
                )
                and torch.equal(
                    torch_rng_before,
                    torch_rng_after,
                )
            ),
        ),

        (
            "training_negative_rng_not_instantiated",
            True,
        ),

        (
            "training_order_rng_not_instantiated",
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

        (
            "checkpoint_not_written",
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
            "At least one Phase-5.3.1k "
            "real-forward preflight invariant failed."
        ),
    )

    print(
        invariant_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Write audit / freeze
    # =========================================================================

    banner(
        "WRITE PHASE-5.3.1k PREFLIGHT CONTRACT"
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTRACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    static_df.to_csv(
        STATIC_INTEGRITY_PATH,
        index=False,
    )

    selected_case_df.to_csv(
        SELECTED_CASE_PATH,
        index=False,
    )

    source_assignment_df.to_csv(
        SOURCE_ASSIGNMENT_PATH,
        index=False,
    )

    shape_df.to_csv(
        FORWARD_SHAPE_PATH,
        index=False,
    )

    attention_df.to_csv(
        ATTENTION_PATH,
        index=False,
    )

    gradient_df.to_csv(
        GRADIENT_PATH,
        index=False,
    )

    state_df.to_csv(
        STATE_NEUTRALITY_PATH,
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
                    "real_data_forward_preflight_case"
                ),

                "value": (
                    interaction_id
                ),

                "classification": (
                    "INHERITED_PHASE4_DIAGNOSTIC_CASE"
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1k"
                ),
            },

            {
                "decision": (
                    "real_data_forward_runtime"
                ),

                "value": (
                    "PHASE4_7_1B_KAIMING_MODEL"
                    "+PHASE5_3_1I_2_METADATA_BRIDGE"
                    "+PHASE4_6_2_EXACT_METHODS"
                ),

                "classification": (
                    "INHERITED_FROZEN_IMPLEMENTATION"
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1k"
                ),
            },

            {
                "decision": (
                    "loss_target_runtime"
                ),

                "value": (
                    "EXECUTE_EXACT_PHASE4_6_2_SOURCE_ASSIGNMENT"
                ),

                "classification": (
                    "INHERITED_PHASE4_IMPLEMENTATION"
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1k"
                ),
            },

            {
                "decision": (
                    "preflight_optimizer_policy"
                ),

                "value": (
                    "NO_OPTIMIZER_NO_STEP"
                ),

                "classification": (
                    "EVALUATION_INTEGRITY_GUARD"
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1k"
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
            "5.3.1k"
        ),

        "title": (
            "Canonical Composed Real-Data "
            "Forward / BCE / Backward Preflight Contract"
        ),

        "status": (
            "FROZEN"
        ),

        "runtime": {
            "canonical_parameter_source": (
                str(
                    CANONICAL_SOURCE_PATH
                )
            ),

            "canonical_parameter_source_sha256": (
                CANONICAL_SOURCE_SHA256
            ),

            "forward_method_source": (
                str(
                    FORWARD_SOURCE_PATH
                )
            ),

            "forward_method_source_sha256": (
                FORWARD_SOURCE_SHA256
            ),

            "metadata_bridge": (
                "BasisRGCNLayer.out_dim = "
                "int(root_weight.shape[1])"
            ),

            "neural_seed": (
                EXPECTED_SEED
            ),

            "reference_torch": (
                torch.__version__
            ),
        },

        "audit_case": {
            "interaction_id": (
                interaction_id
            ),

            "split": (
                "validation"
            ),

            "segment": (
                "T60"
            ),

            "target": (
                1
            ),

            "investor_global_node_index": (
                investor_global
            ),

            "startup_global_node_index": (
                startup_global
            ),

            "startup_local_index": (
                startup_local
            ),

            "history_memberships": (
                history_total
            ),

            "active_periods": (
                active_periods
            ),

            "multi_item_periods": (
                multi_item_periods
            ),

            "negative_generated": (
                False
            ),

            "test_data_used": (
                False
            ),
        },

        "forward": {
            "real_doc2vec": (
                True
            ),

            "real_labels": (
                True
            ),

            "real_T0_T59_history": (
                True
            ),

            "full_phase3_graph": (
                True
            ),

            "description_shape": (
                list(
                    description_subset.shape
                )
            ),

            "trend_shape": (
                list(
                    F_t.shape
                )
            ),

            "structural_shape": (
                list(
                    F_s_all.shape
                )
            ),

            "investor_representation_shape": (
                list(
                    investor_representation.shape
                )
            ),

            "startup_representation_shape": (
                list(
                    startup_representation.shape
                )
            ),

            "pair_shape": (
                list(
                    pair_representation.shape
                )
            ),

            "logit_shape": (
                list(
                    logit.shape
                )
            ),

            "logit": (
                float(
                    logit.detach().item()
                )
            ),

            "probability": (
                float(
                    probability.detach().item()
                )
            ),
        },

        "loss": {
            "implementation": (
                "BCEWithLogitsLoss"
            ),

            "target": (
                float(
                    target.detach().item()
                )
            ),

            "loss": (
                float(
                    loss.detach().item()
                )
            ),

            "manual_stable_bce": (
                float(
                    manual_stable_bce.detach().item()
                )
            ),

            "equivalent": (
                bool(
                    bce_equivalent
                )
            ),
        },

        "autograd": {
            "parameter_tensors": (
                EXPECTED_PARAMETER_TENSORS
            ),

            "all_gradients_exist": (
                True
            ),

            "all_gradients_finite": (
                True
            ),

            "all_parameter_tensor_gradients_nonzero": (
                True
            ),

            "selected_investor_row_abs_sum": (
                investor_row_gradient
            ),

            "candidate_startup_row_abs_sum": (
                candidate_row_gradient
            ),

            "max_historical_startup_row_abs_sum": (
                max_history_gradient
            ),

            "attention_weight_abs_sum": (
                attention_gradient_abs_sum
            ),
        },

        "parameter_state": {
            "before_forward_sha256": (
                canonical_hash_before_forward
            ),

            "after_forward_sha256": (
                canonical_hash_after_forward
            ),

            "after_backward_sha256": (
                canonical_hash_after_backward
            ),

            "expected_sha256": (
                EXPECTED_INITIAL_STATE_SHA256
            ),

            "parameter_bytes_unchanged": (
                True
            ),

            "state_dict_unchanged": (
                True
            ),
        },

        "training_boundary": {
            "training_negatives_generated": (
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

            "checkpoint_written": (
                False
            ),

            "training_epoch_executed": (
                False
            ),
        },

        "next_phase": {
            "id": (
                "5.3.1l"
            ),

            "title": (
                "Adam + Epoch-0 Canonical "
                "Training Mini-Batch Preflight"
            ),

            "requirement": (
                "Instantiate the exact frozen Adam runtime only "
                "after reconstructing and re-verifying this canonical "
                "composed model. Generate epoch-0 training negatives "
                "with the frozen training-negative seed, generate "
                "epoch-0 training order with the frozen order seed, "
                "construct the first real 512-example training batch, "
                "perform forward/BCE/backward, and keep "
                "optimizer.step() at zero."
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
            "5.3.1k"
        ),

        "status": (
            "CANONICAL_COMPOSED_REAL_DATA_"
            "FORWARD_BCE_BACKWARD_PREFLIGHT_PASSED"
        ),

        "interaction_id": (
            interaction_id
        ),

        "canonical_state_sha256_before_forward": (
            canonical_hash_before_forward
        ),

        "canonical_state_sha256_after_forward": (
            canonical_hash_after_forward
        ),

        "canonical_state_sha256_after_backward": (
            canonical_hash_after_backward
        ),

        "loss": (
            float(
                loss.detach().item()
            )
        ),

        "all_32_gradients_finite_nonzero": (
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
        STATIC_INTEGRITY_PATH,
        SELECTED_CASE_PATH,
        SOURCE_ASSIGNMENT_PATH,
        FORWARD_SHAPE_PATH,
        ATTENTION_PATH,
        GRADIENT_PATH,
        STATE_NEUTRALITY_PATH,
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
        "PHASE 5.3.1k FINAL STATUS"
    )

    print(
        "Canonical runtime:                    COMPOSED / FROZEN"
    )

    print(
        "Real validation case:                 VERIFIED"
    )

    print(
        "Test data consumed:                   NO"
    )

    print()

    print(
        f"History memberships:                  "
        f"{history_total}"
    )

    print(
        f"Active periods:                       "
        f"{active_periods}"
    )

    print(
        f"Multi-item periods:                   "
        f"{multi_item_periods}"
    )

    print()

    print(
        f"Pair shape:                           "
        f"{tuple(pair_representation.shape)}"
    )

    print(
        f"Logit:                                "
        f"{float(logit.detach().item()):.10f}"
    )

    print(
        f"Probability:                          "
        f"{float(probability.detach().item()):.10f}"
    )

    print(
        f"BCEWithLogitsLoss:                    "
        f"{float(loss.detach().item()):.10f}"
    )

    print()

    print(
        "Parameter tensors with gradients:     32 / 32"
    )

    print(
        "Finite gradients:                     32 / 32"
    )

    print(
        "Non-zero gradients:                   32 / 32"
    )

    print()

    print(
        "Canonical state SHA256 before forward:"
    )

    print(
        canonical_hash_before_forward
    )

    print()

    print(
        "Canonical state SHA256 after forward:"
    )

    print(
        canonical_hash_after_forward
    )

    print()

    print(
        "Canonical state SHA256 after backward:"
    )

    print(
        canonical_hash_after_backward
    )

    print()

    print(
        "Parameter values unchanged:           PASS"
    )

    print(
        "Forward/backward RNG neutrality:      PASS"
    )

    print()

    print(
        "Training negatives generated:         NO"
    )

    print(
        "Training-order RNG instantiated:      NO"
    )

    print(
        "Optimizer instantiated:               NO"
    )

    print(
        "Optimizer steps:                      0"
    )

    print(
        "Checkpoint written:                   NO"
    )

    banner(
        "PHASE 5.3.1k COMPLETE / "
        "CANONICAL REAL-DATA FORWARD + "
        "BCE + BACKWARD PREFLIGHT PASSED"
    )


if __name__ == "__main__":
    main()