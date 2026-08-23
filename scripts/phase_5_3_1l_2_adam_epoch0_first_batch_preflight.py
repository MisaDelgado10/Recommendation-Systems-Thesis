#!/usr/bin/env python3
"""
Phase 5.3.1l.2
Adam + Canonical Epoch-0 First Mini-Batch Forward/Backward Preflight

Purpose
-------
Phase 5.3.1l.1 froze the exact epoch-0 training stream:

    1,073,249 positive events
    4 negatives per positive
    4,292,996 negative examples
    5,366,245 total epoch examples
    10,481 batches
    batch size 512
    final batch 485

It also froze:

    canonical positive-event order
    epoch-0 negative matrix
    epoch-0 example permutation
    canonical first 512-example batch

This phase performs the FIRST actual TRAINING-BATCH numerical preflight.

Runtime
-------
Canonical neural model:

    Phase-4.7.1b
        canonical topology
        canonical Kaiming(seed=42)
        canonical parameter state

        +

    Phase-5.3.1i.2
        BasisRGCNLayer.out_dim metadata bridge

        +

    Phase-5.3.1j
        exact seven Phase-4.6.2 numerical methods

Training batch:

    frozen Phase-5.3.1l.1 epoch-0 first 512 examples

Mixed temporal targets
----------------------
For every example targeting T_h:

    trend history = T0 .. T(h-1)

No post-h zero padding is allowed through the GRU.

The batch is grouped by h for numerical execution. Results are then
restored to the original frozen batch_position before scoring/loss.

Optimizer
---------
This phase is the FIRST point where Adam may be instantiated.

Exact frozen configuration:

    torch.optim.Adam
        lr=0.001
        betas=(0.9, 0.999)
        eps=1e-8
        weight_decay=0.0
        amsgrad=False
        foreach=False
        fused=False
        maximize=False
        capturable=False
        differentiable=False

Scheduler:
    NONE

CRITICAL BOUNDARY
-----------------
optimizer.step() remains FORBIDDEN.

Adam state is expected to remain empty because Adam initializes its
per-parameter moment state lazily on optimizer.step().

This script performs:

    canonical model construction
    metadata bridge
    exact method composition
    exact Adam instantiation
    frozen first-batch loading
    real mixed-h forward
    BCEWithLogitsLoss
    backward
    gradient audit

This script DOES NOT:

    generate new negatives
    generate a new epoch permutation
    consume T60 labels
    use validation/test candidates
    call optimizer.step()
    update any parameter value
    write a model checkpoint
    run a complete training epoch
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
# Frozen canonical model / method provenance
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

EXPECTED_NEURAL_SEED = 42

EXPECTED_PARAMETER_TENSORS = 32
EXPECTED_PARAMETER_COUNT = 19_217_929

REFERENCE_TORCH_VERSION_PREFIX = "2.7.0"


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
# Frozen model dimensions
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

BATCH_SIZE = 512


# =============================================================================
# Frozen epoch-0 stream fingerprints from Phase 5.3.1l.1
# =============================================================================

EXPECTED_POSITIVE_ORDER_LOGICAL_SHA256 = (
    "73b074a80675793b811fbdc8a0609883"
    "c857fb2a687a2e01c31865ade5b509d1"
)

EXPECTED_NEGATIVE_MATRIX_LOGICAL_SHA256 = (
    "47015b147b1949562c0f6737a6f3a3f2"
    "d7cabd2d2202e4e57456d884a1e23fe6"
)

EXPECTED_EPOCH_ORDER_LOGICAL_SHA256 = (
    "0156be3ee623ade1ae696557337bfb324"
    "e9011adb7df8be9648ecb0a426c134e"
)

EXPECTED_FIRST_BATCH_LOGICAL_SHA256 = (
    "8408432b944bcd0805af9c34ff1b2db3"
    "ea938e0649a75d381b7839b86cd280ea"
)

EXPECTED_FIRST_BATCH_POSITIVES = 105
EXPECTED_FIRST_BATCH_NEGATIVES = 407
EXPECTED_FIRST_BATCH_SEGMENTS = 59

EXPECTED_TRAIN_POSITIVES = 1_073_249
EXPECTED_EPOCH_NEGATIVES = 4_292_996
EXPECTED_EPOCH_EXAMPLES = 5_366_245


# =============================================================================
# Exact frozen Adam runtime
# =============================================================================

ADAM_LR = 0.001
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
ADAM_WEIGHT_DECAY = 0.0
ADAM_AMSGRAD = False
ADAM_FOREACH = False
ADAM_FUSED = False
ADAM_MAXIMIZE = False
ADAM_CAPTURABLE = False
ADAM_DIFFERENTIABLE = False


# =============================================================================
# Phase-5 contracts
# =============================================================================

PHASE_5_3_1L_1_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_1l_1_epoch0_training_stream_serialization_contract.json"
)

PHASE_5_3_1L_1_MANIFEST_PATH = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_1l_1/"
    "phase_5_3_1l_1_training_stream_manifest.json"
)

PHASE_5_3_1K_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_1k_canonical_real_forward_bce_backward_contract.json"
)

PHASE_5_3_1J_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_1j_canonical_numerical_method_composition_contract.json"
)

PHASE_5_3_1I_2_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_1i_2_rgcn_runtime_metadata_bridge_contract.json"
)


# =============================================================================
# Frozen epoch-0 runtime artifacts
# =============================================================================

EPOCH0_DIR = Path(
    "data/experimental/phase_5/training_runtime/"
    "epoch_0"
)

POSITIVE_ORDER_PATH = (
    EPOCH0_DIR
    / "canonical_training_positive_event_order.parquet"
)

NEGATIVE_MATRIX_PATH = (
    EPOCH0_DIR
    / "epoch_0_training_negative_startup_local.npy"
)

EPOCH_ORDER_PATH = (
    EPOCH0_DIR
    / "epoch_0_training_example_order.npy"
)

FIRST_BATCH_PATH = (
    EPOCH0_DIR
    / "epoch_0_first_batch_manifest.parquet"
)

FIRST_BATCH_SEGMENT_GROUP_PATH = (
    EPOCH0_DIR
    / "epoch_0_first_batch_segment_groups.csv"
)


# =============================================================================
# Frozen graph / feature runtime
# =============================================================================

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


# =============================================================================
# Frozen graph / feature hashes
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
# Outputs
# =============================================================================

AUDIT_DIR = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_1l_2"
)

CONTRACT_DIR = Path(
    "data/experimental/phase_5/contracts"
)

STREAM_INTEGRITY_PATH = (
    AUDIT_DIR
    / "epoch0_first_batch_stream_integrity.csv"
)

STATIC_INTEGRITY_PATH = (
    AUDIT_DIR
    / "epoch0_first_batch_static_artifact_integrity.csv"
)

ADAM_CONFIG_PATH = (
    AUDIT_DIR
    / "adam_runtime_configuration_audit.csv"
)

INPUT_BINDING_PATH = (
    AUDIT_DIR
    / "epoch0_first_batch_input_binding_audit.csv"
)

TREND_GROUP_PATH = (
    AUDIT_DIR
    / "epoch0_first_batch_mixed_h_trend_audit.csv"
)

FORWARD_SHAPE_PATH = (
    AUDIT_DIR
    / "epoch0_first_batch_forward_shape_audit.csv"
)

GRADIENT_PATH = (
    AUDIT_DIR
    / "epoch0_first_batch_gradient_audit.csv"
)

STATE_NEUTRALITY_PATH = (
    AUDIT_DIR
    / "epoch0_first_batch_state_neutrality.csv"
)

FINAL_INVARIANT_PATH = (
    AUDIT_DIR
    / "phase_5_3_1l_2_final_invariants.csv"
)

MANIFEST_PATH = (
    AUDIT_DIR
    / "phase_5_3_1l_2_adam_batch_preflight_manifest.json"
)

CONTRACT_PATH = (
    CONTRACT_DIR
    / "phase_5_3_1l_2_adam_epoch0_first_batch_preflight_contract.json"
)

DECISION_REGISTER_PATH = (
    CONTRACT_DIR
    / "phase_5_3_1l_2_adam_batch_preflight_decision_register.csv"
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


def numpy_rng_state_equal(
    left,
    right,
) -> bool:

    return (
        left[
            0
        ]
        == right[
            0
        ]
        and np.array_equal(
            left[
                1
            ],
            right[
                1
            ],
        )
        and left[
            2:
        ]
        == right[
            2:
        ]
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


def gradient_logical_sha256(
    model: torch.nn.Module,
) -> str:

    digest = hashlib.sha256()

    for (
        name,
        parameter,
    ) in model.named_parameters():

        require(
            parameter.grad
            is not None,
            (
                f"Cannot hash missing gradient: "
                f"{name}"
            ),
        )

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
                parameter.grad
            ).encode(
                "ascii"
            )
        )

        digest.update(
            b"\0"
        )

    return digest.hexdigest()


# =============================================================================
# Frozen stream logical hash helpers
# =============================================================================

def array_logical_sha256(
    array: np.ndarray,
) -> str:

    value = np.ascontiguousarray(
        array
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
        value.tobytes(
            order="C"
        )
    )

    return digest.hexdigest()


def positive_stream_logical_sha256(
    frame: pd.DataFrame,
) -> str:

    digest = hashlib.sha256()

    ids = (
        frame[
            "interaction_id"
        ]
        .astype(
            str
        )
        .tolist()
    )

    chunk_size = 10_000

    for start in range(
        0,
        len(
            ids
        ),
        chunk_size,
    ):

        end = min(
            start
            + chunk_size,
            len(
                ids
            ),
        )

        text = "\n".join(
            ids[
                start:end
            ]
        )

        digest.update(
            text.encode(
                "utf-8"
            )
        )

        digest.update(
            b"\0"
        )

    for column in (
        "investor_global",
        "startup_local",
        "segment_number",
    ):

        values = np.ascontiguousarray(
            frame[
                column
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        digest.update(
            column.encode(
                "utf-8"
            )
        )

        digest.update(
            b"\0"
        )

        digest.update(
            values.tobytes(
                order="C"
            )
        )

        digest.update(
            b"\0"
        )

    return digest.hexdigest()


def dataframe_logical_sha256(
    frame: pd.DataFrame,
    columns: list[str],
) -> str:

    digest = hashlib.sha256()

    for column in (
        columns
    ):

        digest.update(
            column.encode(
                "utf-8"
            )
        )

        digest.update(
            b"\0"
        )

        series = (
            frame[
                column
            ]
        )

        if pd.api.types.is_integer_dtype(
            series.dtype
        ):

            values = np.ascontiguousarray(
                series.to_numpy(
                    dtype=np.int64
                )
            )

            digest.update(
                values.tobytes(
                    order="C"
                )
            )

        else:

            for value in (
                series.astype(
                    str
                ).tolist()
            ):

                digest.update(
                    value.encode(
                        "utf-8"
                    )
                )

                digest.update(
                    b"\0"
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
            "definitions found."
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
        for node in tree.body
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
            f"Expected exactly one top-level "
            f"function {name}."
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


def assignment_value(
    node,
):

    if isinstance(
        node,
        ast.Assign,
    ):

        return (
            node.value
        )

    return (
        node.value
    )


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
        copy.deepcopy(
            node
        )
        for node in tree.body[
            :EXPECTED_WORKFLOW_BOUNDARY_INDEX
        ]
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

    runtime_sha = (
        text_sha256(
            ast.dump(
                sanitized,
                annotate_fields=True,
                include_attributes=False,
            )
        )
    )

    require(
        runtime_sha
        == EXPECTED_RUNTIME_AST_SHA256,
        (
            "Canonical sanitized-runtime "
            "AST drift."
        ),
    )

    module_name = (
        "_itrs_phase5_3_1l_2_canonical"
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
        runtime_sha,
    )


# =============================================================================
# Exact Phase-4.6.2 numerical-method runtime
# =============================================================================

def build_forward_runtime(
    tree: ast.Module,
):

    module_name = (
        "_itrs_phase5_3_1l_2_forward_methods"
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

    # Exact import statements only.
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

    # Exact required immutable constants.
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

            value = ast.literal_eval(
                value_node
            )

        except Exception as exc:

            raise AssertionError(
                f"{name} is no longer literal."
            ) from exc

        require(
            value
            == expected_value,
            (
                f"Forward support constant "
                f"{name} changed."
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

    # Exact require() helper.
    execute_nodes(
        module,
        [
            top_level_function(
                tree,
                "require",
            ),
        ],
        str(
            FORWARD_SOURCE_PATH
        ),
    )

    classes = (
        class_map(
            tree
        )
    )

    selected_classes = []

    methods = {}

    for (
        class_name,
        method_names,
    ) in METHOD_GRAFT_SPEC.items():

        require(
            class_name
            in classes,
            (
                f"Forward source missing "
                f"{class_name}."
            ),
        )

        class_node = (
            classes[
                class_name
            ]
        )

        selected_classes.append(
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
                    f"Forward source missing "
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
                    f"{class_name}.{method_name}"
                ),
            )

    execute_nodes(
        module,
        selected_classes,
        str(
            FORWARD_SOURCE_PATH
        ),
    )

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
            "Expected exactly seven numerical methods."
        ),
    )

    return (
        module,
        methods,
    )


# =============================================================================
# Canonical model composition
# =============================================================================

def compose_canonical_model(
    canonical_runtime,
    methods: dict,
):

    builder = getattr(
        canonical_runtime,
        "build_canonical_model",
    )

    hash_fn = getattr(
        canonical_runtime,
        "model_parameter_state_sha256",
    )

    model = builder(
        seed=EXPECTED_NEURAL_SEED
    )

    require(
        hash_fn(
            model
        )
        == EXPECTED_INITIAL_STATE_SHA256,
        (
            "Canonical initial-state hash mismatch."
        ),
    )

    # -------------------------------------------------------------------------
    # Frozen 5.3.1i.2 metadata bridge.
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

    expected_basis_names = (
        "preference_propagation.layer_1",
        "preference_propagation.layer_2",
    )

    require(
        tuple(
            name
            for (
                name,
                _
            ) in basis_instances
        )
        == expected_basis_names,
        (
            "Canonical R-GCN layer paths changed."
        ),
    )

    for (
        name,
        layer,
    ) in basis_instances:

        out_from_root = int(
            layer.root_weight.shape[
                1
            ]
        )

        out_from_bases = int(
            layer.bases.shape[
                2
            ]
        )

        require(
            out_from_root
            == out_from_bases
            == STRUCTURAL_DIM,
            (
                f"{name}: out_dim derivation changed."
            ),
        )

        setattr(
            layer,
            "out_dim",
            out_from_root,
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
                f"{name}.out_dim became parameter."
            ),
        )

        require(
            "out_dim"
            not in layer._buffers,
            (
                f"{name}.out_dim became buffer."
            ),
        )

        require(
            "out_dim"
            not in layer._modules,
            (
                f"{name}.out_dim became module."
            ),
        )

    # -------------------------------------------------------------------------
    # Frozen 5.3.1j method composition.
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

            method = (
                methods[
                    (
                        class_name,
                        method_name,
                    )
                ]
            )

            setattr(
                canonical_class,
                method_name,
                method,
            )

            require(
                getattr(
                    canonical_class,
                    method_name,
                )
                is method,
                (
                    f"Method graft failed: "
                    f"{class_name}.{method_name}"
                ),
            )

    require(
        hash_fn(
            model
        )
        == EXPECTED_INITIAL_STATE_SHA256,
        (
            "Runtime composition changed "
            "canonical parameters."
        ),
    )

    return (
        model,
        hash_fn,
    )


# =============================================================================
# Adam instantiation
# =============================================================================

def build_frozen_adam(
    model: torch.nn.Module,
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=ADAM_LR,
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
        weight_decay=ADAM_WEIGHT_DECAY,
        amsgrad=ADAM_AMSGRAD,
        foreach=ADAM_FOREACH,
        maximize=ADAM_MAXIMIZE,
        capturable=ADAM_CAPTURABLE,
        differentiable=ADAM_DIFFERENTIABLE,
        fused=ADAM_FUSED,
    )

    return optimizer


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.3.1l.2 — "
        "ADAM + CANONICAL EPOCH-0 FIRST "
        "MINI-BATCH FORWARD/BACKWARD PREFLIGHT"
    )

    print(
        "Frozen epoch-0 negatives regenerated: NO"
    )

    print(
        "Frozen epoch-0 order regenerated:     NO"
    )

    print(
        "Canonical model instantiated:         YES"
    )

    print(
        "Adam instantiated:                    YES"
    )

    print(
        "Forward computation:                  YES"
    )

    print(
        "BCE computation:                      YES"
    )

    print(
        "Backward computation:                 YES"
    )

    print(
        "optimizer.step():                     FORBIDDEN / 0"
    )

    print(
        "Checkpoint written:                   NO"
    )

    # =========================================================================
    # Authoritative prerequisites
    # =========================================================================

    banner(
        "AUTHORITATIVE CONTRACT / ARTIFACT RECHECK"
    )

    required_paths = (
        CANONICAL_SOURCE_PATH,
        FORWARD_SOURCE_PATH,
        PHASE_5_3_1L_1_CONTRACT_PATH,
        PHASE_5_3_1L_1_MANIFEST_PATH,
        PHASE_5_3_1K_CONTRACT_PATH,
        PHASE_5_3_1J_CONTRACT_PATH,
        PHASE_5_3_1I_2_CONTRACT_PATH,
        POSITIVE_ORDER_PATH,
        NEGATIVE_MATRIX_PATH,
        EPOCH_ORDER_PATH,
        FIRST_BATCH_PATH,
        FIRST_BATCH_SEGMENT_GROUP_PATH,
        EDGE_INDEX_PATH,
        EDGE_TYPE_PATH,
        DOC2VEC_PATH,
        DOC2VEC_MANIFEST_PATH,
        LABEL_MATRIX_PATH,
        LABEL_MANIFEST_PATH,
        TREND_PERIOD_PTR_PATH,
        TREND_STARTUP_INDICES_PATH,
        TREND_PERIOD_COUNTS_PATH,
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
            "Canonical source SHA drift."
        ),
    )

    require(
        file_sha256(
            FORWARD_SOURCE_PATH
        )
        == FORWARD_SOURCE_SHA256,
        (
            "Forward source SHA drift."
        ),
    )

    l1_contract = (
        load_json(
            PHASE_5_3_1L_1_CONTRACT_PATH
        )
    )

    l1_manifest = (
        load_json(
            PHASE_5_3_1L_1_MANIFEST_PATH
        )
    )

    k_contract = (
        load_json(
            PHASE_5_3_1K_CONTRACT_PATH
        )
    )

    j_contract = (
        load_json(
            PHASE_5_3_1J_CONTRACT_PATH
        )
    )

    bridge_contract = (
        load_json(
            PHASE_5_3_1I_2_CONTRACT_PATH
        )
    )

    require(
        l1_contract[
            "status"
        ]
        == "FROZEN",
        (
            "Phase-5.3.1l.1 stream contract "
            "is not frozen."
        ),
    )

    require(
        l1_manifest[
            "status"
        ]
        == (
            "EPOCH0_TRAINING_STREAM_SERIALIZATION_"
            "PROVED_AND_FROZEN"
        ),
        (
            "Unexpected Phase-5.3.1l.1 status."
        ),
    )

    require(
        k_contract[
            "status"
        ]
        == "FROZEN",
        (
            "Phase-5.3.1k contract not frozen."
        ),
    )

    require(
        j_contract[
            "status"
        ]
        == "FROZEN",
        (
            "Phase-5.3.1j contract not frozen."
        ),
    )

    require(
        bridge_contract[
            "status"
        ]
        == "FROZEN",
        (
            "Phase-5.3.1i.2 contract not frozen."
        ),
    )

    require(
        l1_manifest[
            "Adam_instantiated"
        ]
        is False,
        (
            "Adam existed before Phase-5.3.1l.2."
        ),
    )

    require(
        int(
            l1_manifest[
                "optimizer_steps"
            ]
        )
        == 0,
        (
            "Optimizer step occurred before "
            "Phase-5.3.1l.2."
        ),
    )

    print(
        "Phase-5.3.1l.1 stream:                FROZEN"
    )

    print(
        "Phase-5.3.1k numerical runtime:       FROZEN"
    )

    print(
        "Phase-5.3.1j method composition:      FROZEN"
    )

    print(
        "Phase-5.3.1i.2 metadata bridge:       FROZEN"
    )

    print(
        "Optimizer steps entering phase:       0"
    )

    # =========================================================================
    # Reference runtime
    # =========================================================================

    banner(
        "REFERENCE RUNTIME"
    )

    print(
        f"PyTorch:                              "
        f"{torch.__version__}"
    )

    print(
        f"NumPy:                                "
        f"{np.__version__}"
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
    # Load and verify frozen training stream
    # =========================================================================

    banner(
        "FROZEN EPOCH-0 TRAINING STREAM INTEGRITY"
    )

    positive_order = pd.read_parquet(
        POSITIVE_ORDER_PATH
    )

    negative_matrix = np.load(
        NEGATIVE_MATRIX_PATH,
        mmap_mode="r",
    )

    epoch_order = np.load(
        EPOCH_ORDER_PATH,
        mmap_mode="r",
    )

    first_batch = pd.read_parquet(
        FIRST_BATCH_PATH
    )

    first_batch_groups = pd.read_csv(
        FIRST_BATCH_SEGMENT_GROUP_PATH
    )

    require(
        len(
            positive_order
        )
        == EXPECTED_TRAIN_POSITIVES,
        (
            "Positive-order row count changed."
        ),
    )

    require(
        negative_matrix.shape
        == (
            EXPECTED_TRAIN_POSITIVES,
            4,
        ),
        (
            "Negative matrix shape changed."
        ),
    )

    require(
        negative_matrix.size
        == EXPECTED_EPOCH_NEGATIVES,
        (
            "Negative example count changed."
        ),
    )

    require(
        epoch_order.shape
        == (
            EXPECTED_EPOCH_EXAMPLES,
        ),
        (
            "Epoch-order shape changed."
        ),
    )

    require(
        len(
            first_batch
        )
        == BATCH_SIZE,
        (
            "First batch is not 512 examples."
        ),
    )

    positive_hash = (
        positive_stream_logical_sha256(
            positive_order
        )
    )

    negative_hash = (
        array_logical_sha256(
            np.asarray(
                negative_matrix
            )
        )
    )

    epoch_order_hash = (
        array_logical_sha256(
            np.asarray(
                epoch_order
            )
        )
    )

    first_batch_hash = (
        dataframe_logical_sha256(
            first_batch,
            columns=[
                "batch_position",
                "serialized_example_index",
                "positive_order_index",
                "example_slot",
                "label",
                "source_interaction_id",
                "investor_global",
                "startup_local",
                "segment_number",
            ],
        )
    )

    require(
        positive_hash
        == EXPECTED_POSITIVE_ORDER_LOGICAL_SHA256,
        (
            "Positive-order logical SHA drift."
        ),
    )

    require(
        negative_hash
        == EXPECTED_NEGATIVE_MATRIX_LOGICAL_SHA256,
        (
            "Negative-matrix logical SHA drift."
        ),
    )

    require(
        epoch_order_hash
        == EXPECTED_EPOCH_ORDER_LOGICAL_SHA256,
        (
            "Epoch-order logical SHA drift."
        ),
    )

    require(
        first_batch_hash
        == EXPECTED_FIRST_BATCH_LOGICAL_SHA256,
        (
            "First-batch logical SHA drift."
        ),
    )

    # -------------------------------------------------------------------------
    # Decode first batch AGAIN from the frozen parent artifacts.
    # -------------------------------------------------------------------------

    decode_mismatch_count = 0

    for batch_position in range(
        BATCH_SIZE
    ):

        stored = (
            first_batch.iloc[
                batch_position
            ]
        )

        serialized_index = int(
            epoch_order[
                batch_position
            ]
        )

        positive_index = (
            serialized_index
            // 5
        )

        slot = (
            serialized_index
            % 5
        )

        positive_row = (
            positive_order.iloc[
                positive_index
            ]
        )

        if slot == 0:

            expected_label = 1

            expected_startup_local = int(
                positive_row[
                    "startup_local"
                ]
            )

        else:

            expected_label = 0

            expected_startup_local = int(
                negative_matrix[
                    positive_index,
                    slot
                    - 1,
                ]
            )

        expected_values = {
            "batch_position": (
                batch_position
            ),

            "serialized_example_index": (
                serialized_index
            ),

            "positive_order_index": (
                positive_index
            ),

            "example_slot": (
                slot
            ),

            "label": (
                expected_label
            ),

            "source_interaction_id": (
                str(
                    positive_row[
                        "interaction_id"
                    ]
                )
            ),

            "investor_global": (
                int(
                    positive_row[
                        "investor_global"
                    ]
                )
            ),

            "startup_local": (
                expected_startup_local
            ),

            "segment_number": (
                int(
                    positive_row[
                        "segment_number"
                    ]
                )
            ),
        }

        for (
            key,
            expected_value,
        ) in expected_values.items():

            actual_value = (
                stored[
                    key
                ]
            )

            if key == "source_interaction_id":

                actual_value = str(
                    actual_value
                )

            else:

                actual_value = int(
                    actual_value
                )

            if (
                actual_value
                != expected_value
            ):

                decode_mismatch_count += 1

    require(
        decode_mismatch_count
        == 0,
        (
            "Frozen first-batch manifest does not "
            "decode exactly from epoch-order / "
            "negative-matrix / positive-order artifacts."
        ),
    )

    positive_count = int(
        (
            first_batch[
                "label"
            ]
            == 1
        ).sum()
    )

    negative_count = int(
        (
            first_batch[
                "label"
            ]
            == 0
        ).sum()
    )

    distinct_segments = int(
        first_batch[
            "segment_number"
        ].nunique()
    )

    require(
        positive_count
        == EXPECTED_FIRST_BATCH_POSITIVES,
        (
            "First-batch positive count changed."
        ),
    )

    require(
        negative_count
        == EXPECTED_FIRST_BATCH_NEGATIVES,
        (
            "First-batch negative count changed."
        ),
    )

    require(
        distinct_segments
        == EXPECTED_FIRST_BATCH_SEGMENTS,
        (
            "First batch no longer contains "
            "all T1..T59."
        ),
    )

    require(
        int(
            first_batch[
                "segment_number"
            ].min()
        )
        == 1,
        (
            "First-batch minimum target "
            "segment changed."
        ),
    )

    require(
        int(
            first_batch[
                "segment_number"
            ].max()
        )
        == 59,
        (
            "First-batch maximum target "
            "segment changed."
        ),
    )

    stream_rows = [
        {
            "artifact": (
                "positive_order"
            ),

            "logical_sha256": (
                positive_hash
            ),

            "expected_sha256": (
                EXPECTED_POSITIVE_ORDER_LOGICAL_SHA256
            ),

            "status": (
                "PASS"
            ),
        },

        {
            "artifact": (
                "negative_matrix"
            ),

            "logical_sha256": (
                negative_hash
            ),

            "expected_sha256": (
                EXPECTED_NEGATIVE_MATRIX_LOGICAL_SHA256
            ),

            "status": (
                "PASS"
            ),
        },

        {
            "artifact": (
                "epoch_order"
            ),

            "logical_sha256": (
                epoch_order_hash
            ),

            "expected_sha256": (
                EXPECTED_EPOCH_ORDER_LOGICAL_SHA256
            ),

            "status": (
                "PASS"
            ),
        },

        {
            "artifact": (
                "first_batch"
            ),

            "logical_sha256": (
                first_batch_hash
            ),

            "expected_sha256": (
                EXPECTED_FIRST_BATCH_LOGICAL_SHA256
            ),

            "status": (
                "PASS"
            ),
        },
    ]

    stream_df = pd.DataFrame(
        stream_rows
    )

    print(
        stream_df.to_string(
            index=False
        )
    )

    print()

    print(
        f"First-batch positives:                "
        f"{positive_count}"
    )

    print(
        f"First-batch negatives:                "
        f"{negative_count}"
    )

    print(
        f"Distinct target segments:             "
        f"{distinct_segments}"
    )

    print(
        "Exact parent-artifact decode:         PASS"
    )

    print(
        "New negative sampling performed:      NO"
    )

    print(
        "New epoch shuffle performed:          NO"
    )

    # =========================================================================
    # Frozen static feature integrity
    # =========================================================================

    banner(
        "STATIC GRAPH / FEATURE INTEGRITY"
    )

    static_rows = []

    for (
        path_string,
        expected_hash,
    ) in EXPECTED_STATIC_SHA256.items():

        actual_hash = (
            file_sha256(
                Path(
                    path_string
                )
            )
        )

        static_rows.append(
            {
                "path": (
                    path_string
                ),

                "expected_sha256": (
                    expected_hash
                ),

                "actual_sha256": (
                    actual_hash
                ),

                "status": (
                    "PASS"
                    if (
                        actual_hash
                        == expected_hash
                    )
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
            "Frozen graph/feature artifact "
            "integrity failed."
        ),
    )

    print(
        static_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Parse frozen model sources
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

    # =========================================================================
    # Reconstruct exact canonical model
    # =========================================================================

    banner(
        "RECONSTRUCT CANONICAL COMPOSED MODEL"
    )

    (
        canonical_runtime,
        canonical_runtime_sha,
    ) = build_canonical_runtime(
        canonical_tree
    )

    (
        forward_runtime,
        forward_methods,
    ) = build_forward_runtime(
        forward_tree
    )

    (
        model,
        canonical_hash_fn,
    ) = compose_canonical_model(
        canonical_runtime,
        forward_methods,
    )

    model.train()

    trainable_parameters = [
        parameter
        for parameter
        in model.parameters()
        if parameter.requires_grad
    ]

    require(
        len(
            trainable_parameters
        )
        == EXPECTED_PARAMETER_TENSORS,
        (
            "Canonical trainable tensor "
            "count changed."
        ),
    )

    require(
        sum(
            int(
                parameter.numel()
            )
            for parameter
            in trainable_parameters
        )
        == EXPECTED_PARAMETER_COUNT,
        (
            "Canonical trainable parameter "
            "count changed."
        ),
    )

    canonical_hash_before_adam = (
        canonical_hash_fn(
            model
        )
    )

    require(
        canonical_hash_before_adam
        == EXPECTED_INITIAL_STATE_SHA256,
        (
            "Canonical state mismatch "
            "before Adam."
        ),
    )

    print(
        f"Parameter tensors:                    "
        f"{len(trainable_parameters)}"
    )

    print(
        f"Trainable parameters:                 "
        f"{EXPECTED_PARAMETER_COUNT:,}"
    )

    print()

    print(
        "Canonical SHA256 before Adam:"
    )

    print(
        canonical_hash_before_adam
    )

    # =========================================================================
    # FIRST ADAM INSTANTIATION
    # =========================================================================

    banner(
        "FIRST FROZEN ADAM INSTANTIATION"
    )

    python_rng_before_adam = (
        random.getstate()
    )

    numpy_rng_before_adam = (
        np.random.get_state()
    )

    torch_rng_before_adam = (
        torch.get_rng_state().clone()
    )

    parameter_hashes_before_adam = (
        parameter_hashes(
            model
        )
    )

    optimizer = (
        build_frozen_adam(
            model
        )
    )

    scheduler = None
    optimizer_step_count = 0

    canonical_hash_after_adam = (
        canonical_hash_fn(
            model
        )
    )

    parameter_hashes_after_adam = (
        parameter_hashes(
            model
        )
    )

    python_rng_after_adam = (
        random.getstate()
    )

    numpy_rng_after_adam = (
        np.random.get_state()
    )

    torch_rng_after_adam = (
        torch.get_rng_state().clone()
    )

    require(
        canonical_hash_after_adam
        == EXPECTED_INITIAL_STATE_SHA256,
        (
            "Adam construction changed "
            "canonical parameters."
        ),
    )

    require(
        parameter_hashes_before_adam
        == parameter_hashes_after_adam,
        (
            "Adam construction changed "
            "parameter tensor bytes."
        ),
    )

    require(
        python_rng_before_adam
        == python_rng_after_adam,
        (
            "Adam construction changed "
            "Python RNG."
        ),
    )

    require(
        numpy_rng_state_equal(
            numpy_rng_before_adam,
            numpy_rng_after_adam,
        ),
        (
            "Adam construction changed "
            "NumPy RNG."
        ),
    )

    require(
        torch.equal(
            torch_rng_before_adam,
            torch_rng_after_adam,
        ),
        (
            "Adam construction changed "
            "torch RNG."
        ),
    )

    require(
        len(
            optimizer.param_groups
        )
        == 1,
        (
            "Expected exactly one Adam "
            "parameter group."
        ),
    )

    group = (
        optimizer.param_groups[
            0
        ]
    )

    optimizer_parameters = (
        group[
            "params"
        ]
    )

    require(
        len(
            optimizer_parameters
        )
        == EXPECTED_PARAMETER_TENSORS,
        (
            "Adam parameter-group tensor "
            "count changed."
        ),
    )

    require(
        all(
            left
            is right
            for (
                left,
                right,
            ) in zip(
                optimizer_parameters,
                trainable_parameters,
            )
        ),
        (
            "Adam parameter identity/order "
            "does not match model.parameters()."
        ),
    )

    require(
        sum(
            int(
                parameter.numel()
            )
            for parameter in (
                optimizer_parameters
            )
        )
        == EXPECTED_PARAMETER_COUNT,
        (
            "Adam parameter-group element "
            "count changed."
        ),
    )

    require(
        len(
            optimizer.state
        )
        == 0,
        (
            "Adam state should be empty "
            "before optimizer.step()."
        ),
    )

    require(
        float(
            group[
                "lr"
            ]
        )
        == ADAM_LR,
        (
            "Adam lr changed."
        ),
    )

    require(
        tuple(
            group[
                "betas"
            ]
        )
        == ADAM_BETAS,
        (
            "Adam betas changed."
        ),
    )

    require(
        float(
            group[
                "eps"
            ]
        )
        == ADAM_EPS,
        (
            "Adam eps changed."
        ),
    )

    require(
        float(
            group[
                "weight_decay"
            ]
        )
        == ADAM_WEIGHT_DECAY,
        (
            "Adam weight_decay changed."
        ),
    )

    require(
        bool(
            group[
                "amsgrad"
            ]
        )
        == ADAM_AMSGRAD,
        (
            "Adam amsgrad changed."
        ),
    )

    require(
        group.get(
            "foreach"
        )
        is ADAM_FOREACH,
        (
            "Adam foreach changed."
        ),
    )

    require(
        group.get(
            "fused"
        )
        is ADAM_FUSED,
        (
            "Adam fused changed."
        ),
    )

    require(
        bool(
            group.get(
                "maximize"
            )
        )
        == ADAM_MAXIMIZE,
        (
            "Adam maximize changed."
        ),
    )

    require(
        bool(
            group.get(
                "capturable"
            )
        )
        == ADAM_CAPTURABLE,
        (
            "Adam capturable changed."
        ),
    )

    require(
        bool(
            group.get(
                "differentiable"
            )
        )
        == ADAM_DIFFERENTIABLE,
        (
            "Adam differentiable changed."
        ),
    )

    require(
        scheduler
        is None,
        (
            "Scheduler must remain NONE."
        ),
    )

    adam_df = pd.DataFrame(
        [
            {
                "setting": (
                    "optimizer"
                ),
                "expected": (
                    "torch.optim.Adam"
                ),
                "actual": (
                    type(
                        optimizer
                    ).__module__
                    + "."
                    + type(
                        optimizer
                    ).__name__
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "setting": (
                    "lr"
                ),
                "expected": (
                    ADAM_LR
                ),
                "actual": (
                    group[
                        "lr"
                    ]
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "setting": (
                    "betas"
                ),
                "expected": (
                    str(
                        ADAM_BETAS
                    )
                ),
                "actual": (
                    str(
                        group[
                            "betas"
                        ]
                    )
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "setting": (
                    "eps"
                ),
                "expected": (
                    ADAM_EPS
                ),
                "actual": (
                    group[
                        "eps"
                    ]
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "setting": (
                    "weight_decay"
                ),
                "expected": (
                    ADAM_WEIGHT_DECAY
                ),
                "actual": (
                    group[
                        "weight_decay"
                    ]
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "setting": (
                    "amsgrad"
                ),
                "expected": (
                    ADAM_AMSGRAD
                ),
                "actual": (
                    group[
                        "amsgrad"
                    ]
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "setting": (
                    "foreach"
                ),
                "expected": (
                    ADAM_FOREACH
                ),
                "actual": (
                    group.get(
                        "foreach"
                    )
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "setting": (
                    "fused"
                ),
                "expected": (
                    ADAM_FUSED
                ),
                "actual": (
                    group.get(
                        "fused"
                    )
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "setting": (
                    "maximize"
                ),
                "expected": (
                    ADAM_MAXIMIZE
                ),
                "actual": (
                    group.get(
                        "maximize"
                    )
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "setting": (
                    "capturable"
                ),
                "expected": (
                    ADAM_CAPTURABLE
                ),
                "actual": (
                    group.get(
                        "capturable"
                    )
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "setting": (
                    "differentiable"
                ),
                "expected": (
                    ADAM_DIFFERENTIABLE
                ),
                "actual": (
                    group.get(
                        "differentiable"
                    )
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "setting": (
                    "scheduler"
                ),
                "expected": (
                    "NONE"
                ),
                "actual": (
                    "NONE"
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "setting": (
                    "optimizer_state_entries_before_step"
                ),
                "expected": (
                    0
                ),
                "actual": (
                    len(
                        optimizer.state
                    )
                ),
                "status": (
                    "PASS"
                ),
            },
        ]
    )

    print(
        adam_df.to_string(
            index=False
        )
    )

    print()

    print(
        "Canonical SHA256 after Adam:"
    )

    print(
        canonical_hash_after_adam
    )

    print()

    print(
        "optimizer.step() calls:               0"
    )

    # =========================================================================
    # Load real graph / feature runtime
    # =========================================================================

    banner(
        "LOAD REAL GRAPH / DESCRIPTION / TREND INPUTS"
    )

    edge_index_np = np.load(
        EDGE_INDEX_PATH,
        mmap_mode="r",
    )

    edge_type_np = np.load(
        EDGE_TYPE_PATH,
        mmap_mode="r",
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
            "Trend count shape changed."
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
            "Trend pointer terminal mismatch."
        ),
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

    # =========================================================================
    # Build exact mixed-h history requirements for frozen first batch
    # =========================================================================

    banner(
        "BIND EXACT T0..T(h-1) HISTORIES"
    )

    batch_investors = (
        first_batch[
            "investor_global"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    batch_startup_locals = (
        first_batch[
            "startup_local"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    batch_startup_globals = (
        first_batch[
            "startup_global"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    batch_segments = (
        first_batch[
            "segment_number"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    batch_labels = (
        first_batch[
            "label"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    require(
        bool(
            (
                batch_segments
                >= 1
            ).all()
        )
        and bool(
            (
                batch_segments
                <= 59
            ).all()
        ),
        (
            "First batch contains target "
            "outside T1..T59."
        ),
    )

    unique_trend_keys = sorted(
        {
            (
                int(
                    investor
                ),
                int(
                    segment
                ),
            )
            for (
                investor,
                segment,
            ) in zip(
                batch_investors,
                batch_segments,
            )
        },
        key=lambda item: (
            item[
                1
            ],
            item[
                0
            ],
        ),
    )

    history_by_key = {}

    historical_node_arrays = []

    for (
        investor_global,
        h,
    ) in unique_trend_keys:

        require(
            0
            <= investor_global
            < NUM_INVESTORS,
            (
                "Batch Investor global index "
                "outside Investor slice."
            ),
        )

        require(
            1
            <= h
            <= 59,
            (
                "Invalid training target h."
            ),
        )

        period_arrays = []

        for period in range(
            h
        ):

            slot = (
                investor_global
                * NUM_HISTORY_PERIODS
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
                    f"Trend CSR mismatch "
                    f"for Investor {investor_global}, "
                    f"period {period}."
                ),
            )

            values = np.array(
                trend_startup_indices[
                    start:end
                ],
                dtype=np.int64,
                copy=True,
            )

            if len(
                values
            ) > 0:

                require(
                    bool(
                        (
                            values
                            >= NUM_INVESTORS
                        ).all()
                    )
                    and bool(
                        (
                            values
                            < NUM_NODES
                        ).all()
                    ),
                    (
                        "Historical trend membership "
                        "contains non-Startup node."
                    ),
                )

                historical_node_arrays.append(
                    values
                )

            period_arrays.append(
                values
            )

        require(
            len(
                period_arrays
            )
            == h,
            (
                "History period count != h."
            ),
        )

        history_by_key[
            (
                investor_global,
                h,
            )
        ] = (
            period_arrays
        )

    # No post-h periods were read into history_by_key.
    require(
        all(
            len(
                periods
            )
            == h
            for (
                (
                    _,
                    h,
                ),
                periods,
            ) in history_by_key.items()
        ),
        (
            "At least one trend key contains "
            "post-h periods."
        ),
    )

    # =========================================================================
    # Build required description-node subset
    # =========================================================================

    description_node_parts = [
        np.asarray(
            batch_investors,
            dtype=np.int64,
        ),

        np.asarray(
            batch_startup_globals,
            dtype=np.int64,
        ),
    ]

    description_node_parts.extend(
        historical_node_arrays
    )

    required_nodes = np.unique(
        np.concatenate(
            description_node_parts
        )
    )

    global_to_subset = np.full(
        NUM_NODES,
        -1,
        dtype=np.int64,
    )

    global_to_subset[
        required_nodes
    ] = np.arange(
        len(
            required_nodes
        ),
        dtype=np.int64,
    )

    require(
        bool(
            (
                global_to_subset[
                    batch_investors
                ]
                >= 0
            ).all()
        ),
        (
            "Missing Investor description row."
        ),
    )

    require(
        bool(
            (
                global_to_subset[
                    batch_startup_globals
                ]
                >= 0
            ).all()
        ),
        (
            "Missing candidate Startup "
            "description row."
        ),
    )

    doc_subset_np = np.array(
        doc2vec_all[
            required_nodes
        ],
        dtype=np.float32,
        copy=True,
    )

    label_subset_np = (
        labels_sparse[
            required_nodes
        ]
        .toarray()
        .astype(
            np.float32,
            copy=False,
        )
    )

    require(
        doc_subset_np.shape
        == (
            len(
                required_nodes
            ),
            DOC2VEC_DIM,
        ),
        (
            "Description Doc2Vec subset "
            "shape changed."
        ),
    )

    require(
        label_subset_np.shape
        == (
            len(
                required_nodes
            ),
            LABEL_DIM,
        ),
        (
            "Description label subset "
            "shape changed."
        ),
    )

    require(
        bool(
            np.isfinite(
                doc_subset_np
            ).all()
        ),
        (
            "Doc2Vec subset contains "
            "non-finite values."
        ),
    )

    require(
        bool(
            np.isfinite(
                label_subset_np
            ).all()
        ),
        (
            "Label subset contains "
            "non-finite values."
        ),
    )

    doc_subset = torch.from_numpy(
        doc_subset_np
    )

    label_subset = torch.from_numpy(
        label_subset_np
    )

    input_binding_df = pd.DataFrame(
        [
            {
                "metric": (
                    "batch_examples"
                ),
                "value": (
                    BATCH_SIZE
                ),
            },

            {
                "metric": (
                    "positive_examples"
                ),
                "value": (
                    positive_count
                ),
            },

            {
                "metric": (
                    "negative_examples"
                ),
                "value": (
                    negative_count
                ),
            },

            {
                "metric": (
                    "distinct_target_segments"
                ),
                "value": (
                    distinct_segments
                ),
            },

            {
                "metric": (
                    "unique_investor_h_keys"
                ),
                "value": (
                    len(
                        unique_trend_keys
                    )
                ),
            },

            {
                "metric": (
                    "required_description_nodes"
                ),
                "value": (
                    len(
                        required_nodes
                    )
                ),
            },

            {
                "metric": (
                    "post_h_GRU_padding"
                ),
                "value": (
                    "FORBIDDEN"
                ),
            },
        ]
    )

    print(
        input_binding_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Parameter state snapshot before forward
    # =========================================================================

    canonical_hash_before_forward = (
        canonical_hash_fn(
            model
        )
    )

    parameter_hashes_before_forward = (
        parameter_hashes(
            model
        )
    )

    state_dict_hash_before_forward = (
        logical_state_dict_sha256(
            model
        )
    )

    require(
        canonical_hash_before_forward
        == EXPECTED_INITIAL_STATE_SHA256,
        (
            "Canonical parameters changed "
            "before training-batch forward."
        ),
    )

    require(
        all(
            parameter.grad
            is None
            for parameter in (
                model.parameters()
            )
        ),
        (
            "Canonical model unexpectedly "
            "has gradients before preflight."
        ),
    )

    python_rng_before_forward = (
        random.getstate()
    )

    numpy_rng_before_forward = (
        np.random.get_state()
    )

    torch_rng_before_forward = (
        torch.get_rng_state().clone()
    )

    # =========================================================================
    # REAL TRAINING-BATCH FORWARD
    # =========================================================================

    banner(
        "CANONICAL EPOCH-0 FIRST-BATCH FORWARD"
    )

    # -------------------------------------------------------------------------
    # Full shared latent graph branch
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
            "PreferencePropagation output "
            "lacks F_s."
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
            "Structural representation "
            "contains non-finite values."
        ),
    )

    # -------------------------------------------------------------------------
    # Description branch for exact required nodes
    # -------------------------------------------------------------------------

    description_subset = (
        model.description_encoder(
            doc_subset,
            label_subset,
        )
    )

    require(
        description_subset.shape
        == (
            len(
                required_nodes
            ),
            DESCRIPTION_DIM,
        ),
        (
            "Description output shape invalid."
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

    # -------------------------------------------------------------------------
    # Mixed-h trend.
    #
    # Group by exact target h.
    # No post-h padding.
    # -------------------------------------------------------------------------

    trend_output_by_key = {}

    trend_group_rows = []

    keys_by_h = {}

    for key in (
        unique_trend_keys
    ):

        h = int(
            key[
                1
            ]
        )

        keys_by_h.setdefault(
            h,
            [],
        ).append(
            key
        )

    total_nonempty_periods = 0
    total_multi_item_periods = 0

    for h in sorted(
        keys_by_h
    ):

        keys = (
            keys_by_h[
                h
            ]
        )

        sequence_list = []

        group_nonempty = 0
        group_multi_item = 0

        for (
            investor_global,
            _,
        ) in keys:

            investor_local_tensor = torch.tensor(
                [
                    investor_global,
                ],
                dtype=torch.int64,
            )

            L_o_single = (
                model.investor_embedding(
                    investor_local_tensor
                )[
                    0
                ]
            )

            investor_description_position = int(
                global_to_subset[
                    investor_global
                ]
            )

            require(
                investor_description_position
                >= 0,
                (
                    "Investor missing from "
                    "description subset."
                ),
            )

            F_d_o_single = (
                description_subset[
                    investor_description_position
                ]
            )

            query = torch.cat(
                [
                    L_o_single,
                    F_d_o_single,
                ],
                dim=0,
            )

            require(
                query.shape
                == (
                    TREND_QUERY_DIM,
                ),
                (
                    "Trend query shape invalid."
                ),
            )

            period_vectors = []

            periods = (
                history_by_key[
                    (
                        investor_global,
                        h,
                    )
                ]
            )

            require(
                len(
                    periods
                )
                == h,
                (
                    "Training trend history "
                    "length is not exactly h."
                ),
            )

            for startup_globals in (
                periods
            ):

                item_count = int(
                    len(
                        startup_globals
                    )
                )

                if item_count == 0:

                    period_vector = torch.zeros(
                        TREND_ITEM_DIM,
                        dtype=query.dtype,
                        device=query.device,
                    )

                else:

                    group_nonempty += 1
                    total_nonempty_periods += 1

                    if item_count >= 2:

                        group_multi_item += 1
                        total_multi_item_periods += 1

                    startup_locals = (
                        startup_globals
                        - NUM_INVESTORS
                    )

                    startup_local_tensor = (
                        torch.from_numpy(
                            np.asarray(
                                startup_locals,
                                dtype=np.int64,
                            )
                        )
                    )

                    history_latent = (
                        model.startup_embedding(
                            startup_local_tensor
                        )
                    )

                    description_positions = (
                        global_to_subset[
                            startup_globals
                        ]
                    )

                    require(
                        bool(
                            (
                                description_positions
                                >= 0
                            ).all()
                        ),
                        (
                            "Historical Startup "
                            "missing description."
                        ),
                    )

                    description_position_tensor = (
                        torch.from_numpy(
                            np.asarray(
                                description_positions,
                                dtype=np.int64,
                            )
                        )
                    )

                    history_description = (
                        description_subset[
                            description_position_tensor
                        ]
                    )

                    items = torch.cat(
                        [
                            history_latent,
                            history_description,
                        ],
                        dim=1,
                    )

                    require(
                        items.shape
                        == (
                            item_count,
                            TREND_ITEM_DIM,
                        ),
                        (
                            "Trend item matrix "
                            "shape invalid."
                        ),
                    )

                    (
                        period_vector,
                        alpha,
                    ) = (
                        model
                        .trend_extractor
                        .attend_period(
                            query,
                            items,
                        )
                    )

                    require(
                        period_vector.shape
                        == (
                            TREND_ITEM_DIM,
                        ),
                        (
                            "Period trend vector "
                            "shape invalid."
                        ),
                    )

                    require(
                        alpha.shape
                        == (
                            item_count,
                        ),
                        (
                            "Attention alpha shape "
                            "invalid."
                        ),
                    )

                    require(
                        bool(
                            torch.isfinite(
                                alpha
                            ).all()
                        ),
                        (
                            "Attention contains "
                            "non-finite value."
                        ),
                    )

                    require(
                        abs(
                            float(
                                alpha
                                .detach()
                                .sum()
                            )
                            - 1.0
                        )
                        <= 1e-6,
                        (
                            "Attention weights "
                            "do not sum to one."
                        ),
                    )

                require(
                    bool(
                        torch.isfinite(
                            period_vector
                        ).all()
                    ),
                    (
                        "Trend period vector "
                        "contains non-finite value."
                    ),
                )

                period_vectors.append(
                    period_vector
                )

            sequence = torch.stack(
                period_vectors,
                dim=0,
            )

            require(
                sequence.shape
                == (
                    h,
                    TREND_ITEM_DIM,
                ),
                (
                    "Trend sequence does not "
                    "have exact h length."
                ),
            )

            sequence_list.append(
                sequence
            )

        group_sequence = torch.stack(
            sequence_list,
            dim=0,
        )

        require(
            group_sequence.shape
            == (
                len(
                    keys
                ),
                h,
                TREND_ITEM_DIM,
            ),
            (
                f"T{h} grouped sequence "
                "shape invalid."
            ),
        )

        (
            group_F_t,
            group_gru_output,
        ) = (
            model
            .trend_extractor
            .encode_sequence(
                group_sequence
            )
        )

        require(
            group_F_t.shape
            == (
                len(
                    keys
                ),
                TREND_DIM,
            ),
            (
                f"T{h} F_t shape invalid."
            ),
        )

        require(
            group_gru_output.shape
            == (
                len(
                    keys
                ),
                h,
                TREND_DIM,
            ),
            (
                f"T{h} GRU output shape invalid."
            ),
        )

        require(
            bool(
                torch.isfinite(
                    group_F_t
                ).all()
            ),
            (
                f"T{h} F_t non-finite."
            ),
        )

        for (
            key_index,
            key,
        ) in enumerate(
            keys
        ):

            trend_output_by_key[
                key
            ] = (
                group_F_t[
                    key_index
                ]
            )

        trend_group_rows.append(
            {
                "segment_number": (
                    h
                ),

                "history_periods_consumed": (
                    h
                ),

                "unique_investor_h_keys": (
                    len(
                        keys
                    )
                ),

                "sequence_shape": (
                    str(
                        tuple(
                            group_sequence.shape
                        )
                    )
                ),

                "F_t_shape": (
                    str(
                        tuple(
                            group_F_t.shape
                        )
                    )
                ),

                "nonempty_history_periods": (
                    group_nonempty
                ),

                "multi_item_history_periods": (
                    group_multi_item
                ),

                "post_h_padding": (
                    False
                ),

                "status": (
                    "PASS"
                ),
            }
        )

    require(
        len(
            trend_output_by_key
        )
        == len(
            unique_trend_keys
        ),
        (
            "Not every unique (Investor,h) "
            "trend key received F_t."
        ),
    )

    trend_group_df = pd.DataFrame(
        trend_group_rows
    )

    require(
        len(
            trend_group_df
        )
        == EXPECTED_FIRST_BATCH_SEGMENTS,
        (
            "Expected one mixed-h trend group "
            "for every T1..T59."
        ),
    )

    require(
        bool(
            (
                trend_group_df[
                    "history_periods_consumed"
                ]
                == trend_group_df[
                    "segment_number"
                ]
            ).all()
        ),
        (
            "At least one segment consumed "
            "wrong history length."
        ),
    )

    require(
        bool(
            (
                trend_group_df[
                    "post_h_padding"
                ]
                == False
            ).all()
        ),
        (
            "Post-h GRU padding detected."
        ),
    )

    # Restore exact frozen batch_position.
    F_t_batch = torch.stack(
        [
            trend_output_by_key[
                (
                    int(
                        batch_investors[
                            row
                        ]
                    ),
                    int(
                        batch_segments[
                            row
                        ]
                    ),
                )
            ]
            for row in range(
                BATCH_SIZE
            )
        ],
        dim=0,
    )

    require(
        F_t_batch.shape
        == (
            BATCH_SIZE,
            TREND_DIM,
        ),
        (
            "Restored batch F_t shape invalid."
        ),
    )

    # -------------------------------------------------------------------------
    # Batch endpoint representations
    # -------------------------------------------------------------------------

    investor_tensor = torch.from_numpy(
        np.asarray(
            batch_investors,
            dtype=np.int64,
        )
    )

    startup_local_tensor = torch.from_numpy(
        np.asarray(
            batch_startup_locals,
            dtype=np.int64,
        )
    )

    investor_description_positions = torch.from_numpy(
        np.asarray(
            global_to_subset[
                batch_investors
            ],
            dtype=np.int64,
        )
    )

    startup_description_positions = torch.from_numpy(
        np.asarray(
            global_to_subset[
                batch_startup_globals
            ],
            dtype=np.int64,
        )
    )

    L_o_batch = (
        model.investor_embedding(
            investor_tensor
        )
    )

    L_b_batch = (
        model.startup_embedding(
            startup_local_tensor
        )
    )

    F_d_o_batch = (
        description_subset[
            investor_description_positions
        ]
    )

    F_d_b_batch = (
        description_subset[
            startup_description_positions
        ]
    )

    F_s_o_batch = (
        F_s_all[
            investor_tensor
        ]
    )

    startup_global_tensor = torch.from_numpy(
        np.asarray(
            batch_startup_globals,
            dtype=np.int64,
        )
    )

    F_s_b_batch = (
        F_s_all[
            startup_global_tensor
        ]
    )

    require(
        L_o_batch.shape
        == (
            BATCH_SIZE,
            LATENT_DIM,
        ),
        (
            "Batch L_o shape invalid."
        ),
    )

    require(
        L_b_batch.shape
        == (
            BATCH_SIZE,
            LATENT_DIM,
        ),
        (
            "Batch L_b shape invalid."
        ),
    )

    # Frozen semantic order:
    # F_t, L_o, F_d,o, F_s,o
    investor_representation = torch.cat(
        [
            F_t_batch,
            L_o_batch,
            F_d_o_batch,
            F_s_o_batch,
        ],
        dim=1,
    )

    # Frozen semantic order:
    # L_b, F_d,b, F_s,b
    startup_representation = torch.cat(
        [
            L_b_batch,
            F_d_b_batch,
            F_s_b_batch,
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
            BATCH_SIZE,
            INVESTOR_SCORING_DIM,
        ),
        (
            "Investor batch representation "
            "shape invalid."
        ),
    )

    require(
        startup_representation.shape
        == (
            BATCH_SIZE,
            STARTUP_SCORING_DIM,
        ),
        (
            "Startup batch representation "
            "shape invalid."
        ),
    )

    require(
        pair_representation.shape
        == (
            BATCH_SIZE,
            PAIR_DIM,
        ),
        (
            "Pair batch representation "
            "shape invalid."
        ),
    )

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
            "Scoring output is not dict."
        ),
    )

    require(
        "logit"
        in scoring,
        (
            "Scoring output lacks logit."
        ),
    )

    logits = (
        scoring[
            "logit"
        ]
    )

    require(
        logits.shape
        == (
            BATCH_SIZE,
            1,
        ),
        (
            "Training-batch logits "
            "shape invalid."
        ),
    )

    require(
        bool(
            torch.isfinite(
                logits
            ).all()
        ),
        (
            "Training-batch logits "
            "contain non-finite values."
        ),
    )

    targets = torch.from_numpy(
        batch_labels.astype(
            np.float32,
            copy=True,
        )
    ).reshape(
        BATCH_SIZE,
        1,
    )

    require(
        targets.shape
        == logits.shape,
        (
            "Target/logit shapes differ."
        ),
    )

    require(
        int(
            (
                targets
                == 1.0
            ).sum()
        )
        == EXPECTED_FIRST_BATCH_POSITIVES,
        (
            "Target positive count changed."
        ),
    )

    require(
        int(
            (
                targets
                == 0.0
            ).sum()
        )
        == EXPECTED_FIRST_BATCH_NEGATIVES,
        (
            "Target negative count changed."
        ),
    )

    criterion = (
        torch.nn.BCEWithLogitsLoss()
    )

    loss = criterion(
        logits,
        targets,
    )

    require(
        bool(
            torch.isfinite(
                loss
            )
        ),
        (
            "Training-batch BCE loss "
            "is non-finite."
        ),
    )

    probabilities = torch.sigmoid(
        logits
    )

    require(
        bool(
            torch.isfinite(
                probabilities
            ).all()
        ),
        (
            "Batch probability contains "
            "non-finite value."
        ),
    )

    require(
        bool(
            (
                probabilities
                >= 0.0
            ).all()
        )
        and bool(
            (
                probabilities
                <= 1.0
            ).all()
        ),
        (
            "Batch probabilities outside [0,1]."
        ),
    )

    logits_sha = (
        tensor_sha256(
            logits
        )
    )

    # =========================================================================
    # Forward shape / result audit
    # =========================================================================

    shape_rows = [
        (
            "latent_all",
            tuple(
                latent_all.shape
            ),
            (
                NUM_NODES,
                LATENT_DIM,
            ),
        ),

        (
            "F_s_all",
            tuple(
                F_s_all.shape
            ),
            (
                NUM_NODES,
                STRUCTURAL_DIM,
            ),
        ),

        (
            "description_subset",
            tuple(
                description_subset.shape
            ),
            (
                len(
                    required_nodes
                ),
                DESCRIPTION_DIM,
            ),
        ),

        (
            "F_t_batch",
            tuple(
                F_t_batch.shape
            ),
            (
                BATCH_SIZE,
                TREND_DIM,
            ),
        ),

        (
            "L_o_batch",
            tuple(
                L_o_batch.shape
            ),
            (
                BATCH_SIZE,
                LATENT_DIM,
            ),
        ),

        (
            "L_b_batch",
            tuple(
                L_b_batch.shape
            ),
            (
                BATCH_SIZE,
                LATENT_DIM,
            ),
        ),

        (
            "investor_representation",
            tuple(
                investor_representation.shape
            ),
            (
                BATCH_SIZE,
                INVESTOR_SCORING_DIM,
            ),
        ),

        (
            "startup_representation",
            tuple(
                startup_representation.shape
            ),
            (
                BATCH_SIZE,
                STARTUP_SCORING_DIM,
            ),
        ),

        (
            "pair_representation",
            tuple(
                pair_representation.shape
            ),
            (
                BATCH_SIZE,
                PAIR_DIM,
            ),
        ),

        (
            "logits",
            tuple(
                logits.shape
            ),
            (
                BATCH_SIZE,
                1,
            ),
        ),

        (
            "targets",
            tuple(
                targets.shape
            ),
            (
                BATCH_SIZE,
                1,
            ),
        ),
    ]

    shape_df = pd.DataFrame(
        [
            {
                "feature": (
                    name
                ),

                "actual_shape": (
                    str(
                        actual
                    )
                ),

                "expected_shape": (
                    str(
                        expected
                    )
                ),

                "status": (
                    "PASS"
                    if (
                        actual
                        == expected
                    )
                    else "FAIL"
                ),
            }
            for (
                name,
                actual,
                expected,
            ) in shape_rows
        ]
    )

    require(
        (
            shape_df[
                "status"
            ]
            == "PASS"
        ).all(),
        (
            "Training-batch forward "
            "shape audit failed."
        ),
    )

    print(
        shape_df.to_string(
            index=False
        )
    )

    print()

    print(
        f"Batch BCEWithLogitsLoss:              "
        f"{float(loss.detach().item()):.10f}"
    )

    print(
        f"Mean logit:                           "
        f"{float(logits.detach().mean()):.10f}"
    )

    print(
        f"Min logit:                            "
        f"{float(logits.detach().min()):.10f}"
    )

    print(
        f"Max logit:                            "
        f"{float(logits.detach().max()):.10f}"
    )

    print(
        f"Mean probability:                     "
        f"{float(probabilities.detach().mean()):.10f}"
    )

    print()

    print(
        "Logit tensor logical SHA256:"
    )

    print(
        logits_sha
    )

    # =========================================================================
    # Parameters MUST remain exact after forward
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
            "Forward modified canonical "
            "parameter values."
        ),
    )

    # =========================================================================
    # FIRST REAL TRAINING-BATCH BACKWARD
    # =========================================================================

    banner(
        "FIRST EPOCH-0 TRAINING-BATCH BACKWARD"
    )

    loss.backward()

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
            "Gradient audit does not cover "
            "all 32 trainable tensors."
        ),
    )

    require(
        bool(
            gradient_df[
                "gradient_exists"
            ].all()
        ),
        (
            "At least one trainable tensor "
            "has no gradient."
        ),
    )

    require(
        bool(
            gradient_df[
                "gradient_finite"
            ].all()
        ),
        (
            "At least one trainable tensor "
            "has non-finite gradient."
        ),
    )

    require(
        bool(
            gradient_df[
                "gradient_nonzero"
            ].all()
        ),
        (
            "At least one trainable tensor "
            "has zero gradient."
        ),
    )

    gradient_sha = (
        gradient_logical_sha256(
            model
        )
    )

    print(
        f"Parameter tensors with gradients:     "
        f"{int(gradient_df['gradient_exists'].sum())}"
        f" / {EXPECTED_PARAMETER_TENSORS}"
    )

    print(
        f"Finite parameter gradients:           "
        f"{int(gradient_df['gradient_finite'].sum())}"
        f" / {EXPECTED_PARAMETER_TENSORS}"
    )

    print(
        f"Non-zero parameter gradients:         "
        f"{int(gradient_df['gradient_nonzero'].sum())}"
        f" / {EXPECTED_PARAMETER_TENSORS}"
    )

    print()

    print(
        "Gradient logical SHA256:"
    )

    print(
        gradient_sha
    )

    # =========================================================================
    # Adam must STILL have no per-parameter state
    # =========================================================================

    require(
        len(
            optimizer.state
        )
        == 0,
        (
            "Adam per-parameter state was created "
            "before optimizer.step()."
        ),
    )

    require(
        optimizer_step_count
        == 0,
        (
            "optimizer.step() count changed."
        ),
    )

    # =========================================================================
    # Parameter/state neutrality after backward
    # =========================================================================

    banner(
        "POST-BACKWARD / PRE-STEP STATE NEUTRALITY"
    )

    canonical_hash_after_backward = (
        canonical_hash_fn(
            model
        )
    )

    parameter_hashes_after_backward = (
        parameter_hashes(
            model
        )
    )

    state_dict_hash_after_backward = (
        logical_state_dict_sha256(
            model
        )
    )

    python_rng_after_forward = (
        random.getstate()
    )

    numpy_rng_after_forward = (
        np.random.get_state()
    )

    torch_rng_after_forward = (
        torch.get_rng_state().clone()
    )

    require(
        canonical_hash_after_backward
        == EXPECTED_INITIAL_STATE_SHA256,
        (
            "Backward changed canonical "
            "parameter values."
        ),
    )

    require(
        parameter_hashes_before_forward
        == parameter_hashes_after_backward,
        (
            "Parameter tensor bytes changed "
            "before optimizer.step()."
        ),
    )

    require(
        state_dict_hash_before_forward
        == state_dict_hash_after_backward,
        (
            "state_dict changed before "
            "optimizer.step()."
        ),
    )

    require(
        python_rng_before_forward
        == python_rng_after_forward,
        (
            "Forward/backward changed "
            "Python RNG."
        ),
    )

    require(
        numpy_rng_state_equal(
            numpy_rng_before_forward,
            numpy_rng_after_forward,
        ),
        (
            "Forward/backward changed "
            "NumPy global RNG."
        ),
    )

    require(
        torch.equal(
            torch_rng_before_forward,
            torch_rng_after_forward,
        ),
        (
            "Forward/backward changed "
            "torch global RNG."
        ),
    )

    state_df = pd.DataFrame(
        [
            {
                "check": (
                    "canonical_hash_before_adam"
                ),
                "actual": (
                    canonical_hash_before_adam
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
                    "canonical_hash_after_adam"
                ),
                "actual": (
                    canonical_hash_after_adam
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
                    "canonical_hash_before_forward"
                ),
                "actual": (
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
                "actual": (
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
                "actual": (
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
                    "parameter_bytes_unchanged"
                ),
                "actual": (
                    str(
                        parameter_hashes_before_forward
                        == parameter_hashes_after_backward
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
                    "state_dict_unchanged"
                ),
                "actual": (
                    str(
                        state_dict_hash_before_forward
                        == state_dict_hash_after_backward
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
                    "Adam_state_entries_before_step"
                ),
                "actual": (
                    str(
                        len(
                            optimizer.state
                        )
                    )
                ),
                "expected": (
                    "0"
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "check": (
                    "optimizer_step_count"
                ),
                "actual": (
                    str(
                        optimizer_step_count
                    )
                ),
                "expected": (
                    "0"
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
        "FINAL PHASE-5.3.1l.2 PREFLIGHT INVARIANTS"
    )

    checks = [
        (
            "phase_5_3_1l_1_stream_contract_frozen",
            (
                l1_contract[
                    "status"
                ]
                == "FROZEN"
            ),
        ),

        (
            "optimizer_steps_entering_phase_zero",
            (
                int(
                    l1_manifest[
                        "optimizer_steps"
                    ]
                )
                == 0
            ),
        ),

        (
            "positive_order_hash_exact",
            (
                positive_hash
                == EXPECTED_POSITIVE_ORDER_LOGICAL_SHA256
            ),
        ),

        (
            "negative_matrix_hash_exact",
            (
                negative_hash
                == EXPECTED_NEGATIVE_MATRIX_LOGICAL_SHA256
            ),
        ),

        (
            "epoch_order_hash_exact",
            (
                epoch_order_hash
                == EXPECTED_EPOCH_ORDER_LOGICAL_SHA256
            ),
        ),

        (
            "first_batch_hash_exact",
            (
                first_batch_hash
                == EXPECTED_FIRST_BATCH_LOGICAL_SHA256
            ),
        ),

        (
            "first_batch_decodes_exactly_from_parent_artifacts",
            (
                decode_mismatch_count
                == 0
            ),
        ),

        (
            "first_batch_positive_count_exact",
            (
                positive_count
                == EXPECTED_FIRST_BATCH_POSITIVES
            ),
        ),

        (
            "first_batch_negative_count_exact",
            (
                negative_count
                == EXPECTED_FIRST_BATCH_NEGATIVES
            ),
        ),

        (
            "first_batch_contains_all_T1_to_T59",
            (
                distinct_segments
                == 59
            ),
        ),

        (
            "no_new_negative_sampling",
            True,
        ),

        (
            "no_new_epoch_shuffle",
            True,
        ),

        (
            "all_static_artifact_hashes_exact",
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
            "canonical_runtime_ast_exact",
            (
                canonical_runtime_sha
                == EXPECTED_RUNTIME_AST_SHA256
            ),
        ),

        (
            "canonical_hash_before_Adam_exact",
            (
                canonical_hash_before_adam
                == EXPECTED_INITIAL_STATE_SHA256
            ),
        ),

        (
            "Adam_exact_class",
            (
                type(
                    optimizer
                )
                is torch.optim.Adam
            ),
        ),

        (
            "Adam_lr_exact",
            (
                float(
                    group[
                        "lr"
                    ]
                )
                == ADAM_LR
            ),
        ),

        (
            "Adam_betas_exact",
            (
                tuple(
                    group[
                        "betas"
                    ]
                )
                == ADAM_BETAS
            ),
        ),

        (
            "Adam_eps_exact",
            (
                float(
                    group[
                        "eps"
                    ]
                )
                == ADAM_EPS
            ),
        ),

        (
            "Adam_weight_decay_zero",
            (
                float(
                    group[
                        "weight_decay"
                    ]
                )
                == 0.0
            ),
        ),

        (
            "Adam_amsgrad_false",
            (
                bool(
                    group[
                        "amsgrad"
                    ]
                )
                is False
            ),
        ),

        (
            "Adam_foreach_false",
            (
                group.get(
                    "foreach"
                )
                is False
            ),
        ),

        (
            "Adam_fused_false",
            (
                group.get(
                    "fused"
                )
                is False
            ),
        ),

        (
            "Adam_maximize_false",
            (
                bool(
                    group.get(
                        "maximize"
                    )
                )
                is False
            ),
        ),

        (
            "Adam_capturable_false",
            (
                bool(
                    group.get(
                        "capturable"
                    )
                )
                is False
            ),
        ),

        (
            "Adam_differentiable_false",
            (
                bool(
                    group.get(
                        "differentiable"
                    )
                )
                is False
            ),
        ),

        (
            "Adam_all_32_parameter_tensors",
            (
                len(
                    optimizer_parameters
                )
                == EXPECTED_PARAMETER_TENSORS
            ),
        ),

        (
            "Adam_parameter_identity_order_exact",
            all(
                left
                is right
                for (
                    left,
                    right,
                ) in zip(
                    optimizer_parameters,
                    trainable_parameters,
                )
            ),
        ),

        (
            "Adam_state_empty_before_step",
            (
                len(
                    optimizer.state
                )
                == 0
            ),
        ),

        (
            "scheduler_none",
            (
                scheduler
                is None
            ),
        ),

        (
            "Adam_instantiation_parameter_neutral",
            (
                canonical_hash_after_adam
                == EXPECTED_INITIAL_STATE_SHA256
            ),
        ),

        (
            "Adam_instantiation_rng_neutral",
            (
                python_rng_before_adam
                == python_rng_after_adam
                and numpy_rng_state_equal(
                    numpy_rng_before_adam,
                    numpy_rng_after_adam,
                )
                and torch.equal(
                    torch_rng_before_adam,
                    torch_rng_after_adam,
                )
            ),
        ),

        (
            "all_trend_groups_use_history_length_h",
            bool(
                (
                    trend_group_df[
                        "history_periods_consumed"
                    ]
                    == trend_group_df[
                        "segment_number"
                    ]
                ).all()
            ),
        ),

        (
            "post_h_GRU_padding_forbidden_and_absent",
            bool(
                (
                    trend_group_df[
                        "post_h_padding"
                    ]
                    == False
                ).all()
            ),
        ),

        (
            "all_59_target_segment_groups_executed",
            (
                len(
                    trend_group_df
                )
                == 59
            ),
        ),

        (
            "F_t_restored_to_all_512_batch_positions",
            (
                F_t_batch.shape
                == (
                    BATCH_SIZE,
                    TREND_DIM,
                )
            ),
        ),

        (
            "pair_representation_shape_exact",
            (
                pair_representation.shape
                == (
                    BATCH_SIZE,
                    PAIR_DIM,
                )
            ),
        ),

        (
            "logit_shape_exact",
            (
                logits.shape
                == (
                    BATCH_SIZE,
                    1,
                )
            ),
        ),

        (
            "logits_finite",
            bool(
                torch.isfinite(
                    logits
                ).all()
            ),
        ),

        (
            "targets_match_frozen_105_407_labels",
            (
                int(
                    (
                        targets
                        == 1.0
                    ).sum()
                )
                == EXPECTED_FIRST_BATCH_POSITIVES
                and int(
                    (
                        targets
                        == 0.0
                    ).sum()
                )
                == EXPECTED_FIRST_BATCH_NEGATIVES
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
            "parameter_bytes_unchanged_before_step",
            (
                parameter_hashes_before_forward
                == parameter_hashes_after_backward
            ),
        ),

        (
            "state_dict_unchanged_before_step",
            (
                state_dict_hash_before_forward
                == state_dict_hash_after_backward
            ),
        ),

        (
            "forward_backward_rng_neutral",
            (
                python_rng_before_forward
                == python_rng_after_forward
                and numpy_rng_state_equal(
                    numpy_rng_before_forward,
                    numpy_rng_after_forward,
                )
                and torch.equal(
                    torch_rng_before_forward,
                    torch_rng_after_forward,
                )
            ),
        ),

        (
            "Adam_state_still_empty_after_backward",
            (
                len(
                    optimizer.state
                )
                == 0
            ),
        ),

        (
            "optimizer_step_count_zero",
            (
                optimizer_step_count
                == 0
            ),
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
            "At least one Phase-5.3.1l.2 "
            "Adam/batch preflight invariant failed."
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
        "WRITE PHASE-5.3.1l.2 PREFLIGHT OUTPUTS"
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTRACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stream_df.to_csv(
        STREAM_INTEGRITY_PATH,
        index=False,
    )

    static_df.to_csv(
        STATIC_INTEGRITY_PATH,
        index=False,
    )

    adam_df.to_csv(
        ADAM_CONFIG_PATH,
        index=False,
    )

    input_binding_df.to_csv(
        INPUT_BINDING_PATH,
        index=False,
    )

    trend_group_df.to_csv(
        TREND_GROUP_PATH,
        index=False,
    )

    shape_df.to_csv(
        FORWARD_SHAPE_PATH,
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

    # =========================================================================
    # Decision register
    # =========================================================================

    decision_df = pd.DataFrame(
        [
            {
                "decision": (
                    "optimizer_runtime"
                ),

                "value": (
                    "torch.optim.Adam"
                ),

                "classification": (
                    "INHERITED_FROZEN_PHASE_5_2_2"
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1l_2"
                ),
            },

            {
                "decision": (
                    "optimizer_configuration"
                ),

                "value": (
                    "lr=.001,betas=(.9,.999),eps=1e-8,"
                    "weight_decay=0,amsgrad=False,"
                    "foreach=False,fused=False,maximize=False,"
                    "capturable=False,differentiable=False"
                ),

                "classification": (
                    "INHERITED_FROZEN_PHASE_5_2_2"
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1l_2"
                ),
            },

            {
                "decision": (
                    "first_training_batch"
                ),

                "value": (
                    EXPECTED_FIRST_BATCH_LOGICAL_SHA256
                ),

                "classification": (
                    "INHERITED_FROZEN_PHASE_5_3_1l_1"
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1l_2"
                ),
            },

            {
                "decision": (
                    "mixed_h_runtime"
                ),

                "value": (
                    "GROUP_BY_h_USE_EXACT_T0_TO_T_h_MINUS_1_"
                    "RESTORE_BATCH_POSITION"
                ),

                "classification": (
                    "INHERITED_FROZEN_PHASE_5_3_1l_1"
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1l_2"
                ),
            },

            {
                "decision": (
                    "optimizer_step_preflight_policy"
                ),

                "value": (
                    "FORBIDDEN"
                ),

                "classification": (
                    "EVALUATION_INTEGRITY_GUARD"
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1l_2"
                ),
            },
        ]
    )

    decision_df.to_csv(
        DECISION_REGISTER_PATH,
        index=False,
    )

    # =========================================================================
    # Contract
    # =========================================================================

    contract = {
        "phase": (
            "5.3.1l.2"
        ),

        "title": (
            "Adam + Canonical Epoch-0 First "
            "Mini-Batch Forward/Backward Preflight Contract"
        ),

        "status": (
            "FROZEN"
        ),

        "canonical_model": {
            "initial_state_sha256": (
                EXPECTED_INITIAL_STATE_SHA256
            ),

            "parameter_tensors": (
                EXPECTED_PARAMETER_TENSORS
            ),

            "trainable_parameters": (
                EXPECTED_PARAMETER_COUNT
            ),

            "state_sha256_before_adam": (
                canonical_hash_before_adam
            ),

            "state_sha256_after_adam": (
                canonical_hash_after_adam
            ),

            "state_sha256_after_forward": (
                canonical_hash_after_forward
            ),

            "state_sha256_after_backward": (
                canonical_hash_after_backward
            ),
        },

        "training_stream": {
            "positive_order_sha256": (
                positive_hash
            ),

            "negative_matrix_sha256": (
                negative_hash
            ),

            "epoch_order_sha256": (
                epoch_order_hash
            ),

            "first_batch_sha256": (
                first_batch_hash
            ),

            "batch_size": (
                BATCH_SIZE
            ),

            "positive_count": (
                positive_count
            ),

            "negative_count": (
                negative_count
            ),

            "target_segments": (
                "T1..T59"
            ),

            "new_negative_sampling": (
                False
            ),

            "new_epoch_shuffle": (
                False
            ),
        },

        "optimizer": {
            "class": (
                "torch.optim.Adam"
            ),

            "lr": (
                ADAM_LR
            ),

            "betas": (
                list(
                    ADAM_BETAS
                )
            ),

            "eps": (
                ADAM_EPS
            ),

            "weight_decay": (
                ADAM_WEIGHT_DECAY
            ),

            "amsgrad": (
                ADAM_AMSGRAD
            ),

            "foreach": (
                ADAM_FOREACH
            ),

            "fused": (
                ADAM_FUSED
            ),

            "maximize": (
                ADAM_MAXIMIZE
            ),

            "capturable": (
                ADAM_CAPTURABLE
            ),

            "differentiable": (
                ADAM_DIFFERENTIABLE
            ),

            "scheduler": (
                None
            ),

            "parameter_tensors": (
                len(
                    optimizer_parameters
                )
            ),

            "state_entries_before_step": (
                len(
                    optimizer.state
                )
            ),

            "optimizer_steps": (
                optimizer_step_count
            ),
        },

        "mixed_h_forward": {
            "target_segment_groups": (
                len(
                    trend_group_df
                )
            ),

            "history_for_target_T_h": (
                "T0..T(h-1)"
            ),

            "post_h_padding": (
                False
            ),

            "restore_original_batch_positions": (
                True
            ),

            "unique_investor_h_keys": (
                len(
                    unique_trend_keys
                )
            ),

            "total_nonempty_history_periods": (
                total_nonempty_periods
            ),

            "total_multi_item_history_periods": (
                total_multi_item_periods
            ),
        },

        "forward": {
            "pair_shape": (
                list(
                    pair_representation.shape
                )
            ),

            "logit_shape": (
                list(
                    logits.shape
                )
            ),

            "logit_sha256": (
                logits_sha
            ),

            "mean_logit": (
                float(
                    logits
                    .detach()
                    .mean()
                )
            ),

            "minimum_logit": (
                float(
                    logits
                    .detach()
                    .min()
                )
            ),

            "maximum_logit": (
                float(
                    logits
                    .detach()
                    .max()
                )
            ),

            "mean_probability": (
                float(
                    probabilities
                    .detach()
                    .mean()
                )
            ),
        },

        "loss": {
            "implementation": (
                "torch.nn.BCEWithLogitsLoss"
            ),

            "batch_loss": (
                float(
                    loss
                    .detach()
                    .item()
                )
            ),
        },

        "backward": {
            "parameter_tensors": (
                EXPECTED_PARAMETER_TENSORS
            ),

            "all_gradients_exist": (
                True
            ),

            "all_gradients_finite": (
                True
            ),

            "all_gradients_nonzero": (
                True
            ),

            "gradient_logical_sha256": (
                gradient_sha
            ),
        },

        "pre_step_state": {
            "parameter_bytes_unchanged": (
                parameter_hashes_before_forward
                == parameter_hashes_after_backward
            ),

            "state_dict_unchanged": (
                state_dict_hash_before_forward
                == state_dict_hash_after_backward
            ),

            "Adam_state_entries": (
                len(
                    optimizer.state
                )
            ),

            "optimizer_steps": (
                optimizer_step_count
            ),
        },

        "training_boundary": {
            "Adam_instantiated": (
                True
            ),

            "forward_performed": (
                True
            ),

            "BCE_computed": (
                True
            ),

            "backward_performed": (
                True
            ),

            "optimizer_step_performed": (
                False
            ),

            "optimizer_steps": (
                0
            ),

            "checkpoint_written": (
                False
            ),

            "full_epoch_executed": (
                False
            ),
        },

        "next_phase": {
            "id": (
                "5.3.1m"
            ),

            "title": (
                "First Adam Weight-Update Proof"
            ),

            "requirement": (
                "Reconstruct this exact pre-step state and first batch; "
                "verify the frozen batch/logit/gradient fingerprints; "
                "execute exactly one optimizer.step(); prove Adam state "
                "is initialized correctly, parameter values change "
                "finitely, parameter topology remains unchanged, and "
                "the update can be reproduced exactly from the same "
                "canonical initial state and batch."
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
            "5.3.1l.2"
        ),

        "status": (
            "ADAM_AND_EPOCH0_FIRST_BATCH_"
            "FORWARD_BACKWARD_PREFLIGHT_PASSED"
        ),

        "first_batch_sha256": (
            first_batch_hash
        ),

        "canonical_state_sha256_before_adam": (
            canonical_hash_before_adam
        ),

        "canonical_state_sha256_after_backward": (
            canonical_hash_after_backward
        ),

        "batch_loss": (
            float(
                loss.detach().item()
            )
        ),

        "logit_sha256": (
            logits_sha
        ),

        "gradient_sha256": (
            gradient_sha
        ),

        "Adam_instantiated": (
            True
        ),

        "Adam_state_entries": (
            len(
                optimizer.state
            )
        ),

        "optimizer_steps": (
            optimizer_step_count
        ),

        "checkpoint_written": (
            False
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
        STREAM_INTEGRITY_PATH,
        STATIC_INTEGRITY_PATH,
        ADAM_CONFIG_PATH,
        INPUT_BINDING_PATH,
        TREND_GROUP_PATH,
        FORWARD_SHAPE_PATH,
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
        "PHASE 5.3.1l.2 FINAL STATUS"
    )

    print(
        "Canonical model:                      VERIFIED"
    )

    print(
        "Frozen epoch-0 first batch:           VERIFIED"
    )

    print(
        "New negatives generated:              NO"
    )

    print(
        "New epoch order generated:            NO"
    )

    print()

    print(
        f"Batch size:                           "
        f"{BATCH_SIZE}"
    )

    print(
        f"Positives:                            "
        f"{positive_count}"
    )

    print(
        f"Negatives:                            "
        f"{negative_count}"
    )

    print(
        f"Target segments:                      "
        f"T1..T59"
    )

    print()

    print(
        "Mixed-h runtime:"
    )

    print(
        "  target T_h consumes exactly T0..T(h-1)"
    )

    print(
        "  post-h GRU padding: NONE"
    )

    print(
        "  original batch positions restored: YES"
    )

    print()

    print(
        "Adam:                                 INSTANTIATED"
    )

    print(
        f"Adam state entries:                   "
        f"{len(optimizer.state)}"
    )

    print(
        "Scheduler:                            NONE"
    )

    print()

    print(
        f"Batch BCEWithLogitsLoss:              "
        f"{float(loss.detach().item()):.10f}"
    )

    print()

    print(
        "Logit SHA256:"
    )

    print(
        logits_sha
    )

    print()

    print(
        "Gradient SHA256:"
    )

    print(
        gradient_sha
    )

    print()

    print(
        "Parameter gradients:                  32 / 32"
    )

    print(
        "Finite gradients:                     32 / 32"
    )

    print(
        "Non-zero gradients:                   32 / 32"
    )

    print()

    print(
        "Canonical state before Adam:"
    )

    print(
        canonical_hash_before_adam
    )

    print()

    print(
        "Canonical state after backward:"
    )

    print(
        canonical_hash_after_backward
    )

    print()

    print(
        "Parameter values changed:             NO"
    )

    print(
        "optimizer.step():                     0"
    )

    print(
        "Checkpoint written:                   NO"
    )

    banner(
        "PHASE 5.3.1l.2 COMPLETE / "
        "ADAM + EPOCH-0 FIRST-BATCH "
        "FORWARD/BACKWARD PREFLIGHT PASSED"
    )


if __name__ == "__main__":
    main()