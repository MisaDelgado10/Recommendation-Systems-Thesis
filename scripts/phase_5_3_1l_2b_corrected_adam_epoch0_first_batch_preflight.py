#!/usr/bin/env python3
"""
Phase 5.3.1l.2b
Corrected Adam + Canonical Epoch-0 First Mini-Batch Forward/Backward Preflight

Purpose
-------
Resume the Phase-5.3.1l.2 training-batch preflight after Phase 5.3.1l.2a
proved and froze the training-compatible TrendExtractor sequence adapter.

The original frozen Phase-4.6.2 method:

    TrendExtractor.encode_sequence()

contains a T60-integration-audit-only assertion requiring exactly
60 history periods.

Phase 5.3.1l.2a froze:

    TrendExtractor.encode_training_sequence()

which differs from the original method by exactly one removed statement:

    require(
        sequence.shape[1] == NUM_HISTORY_PERIODS,
        "T60 audit sequence must contain T0 through T59.",
    )

Everything else is exact.

The adapter was proven:
    - bit-exact to original encode_sequence for h=60;
    - valid at h=1,17,59;
    - parameter-state neutral.

This phase therefore executes the exact frozen epoch-0 first training batch:

    512 examples
    105 positives
    407 negatives
    targets spanning T1..T59

For every target T_h:

    history = T0..T(h-1)
    sequence length = h

No post-h GRU padding is allowed.

Optimizer
---------
Exact frozen Adam:

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
    scheduler=None

CRITICAL BOUNDARY
-----------------
optimizer.step() remains FORBIDDEN.

This script performs:
    canonical model reconstruction
    frozen metadata bridge
    exact Phase-4.6.2 numerical method composition
    frozen training trend adapter attachment
    exact Adam construction
    frozen first-batch loading
    mixed-h forward
    BCEWithLogitsLoss
    backward
    full gradient audit

This script does NOT:
    generate negatives
    regenerate epoch order
    use T60 labels
    consume validation/test candidates
    call optimizer.step()
    modify parameter values
    write a model checkpoint
    execute a full training epoch
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

EXPECTED_NEURAL_SEED = 42

EXPECTED_PARAMETER_TENSORS = 32
EXPECTED_PARAMETER_COUNT = 19_217_929

REFERENCE_TORCH_VERSION_PREFIX = "2.7.0"


# =============================================================================
# Frozen Phase-4.6.2 numerical methods
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

EXPECTED_ORIGINAL_ENCODE_SEQUENCE_AST_SHA256 = (
    "72fa12a49ad4a21399d4810a31323f670"
    "34e6173f73ee71dc18798bad1b6d97f"
)

EXPECTED_TRAINING_ADAPTER_AST_SHA256 = (
    "32e1eb9cba61d67cfd3737317d755173"
    "e1c8317b38ca3b18ea4e02df2576dc45"
)

EXPECTED_REMOVED_GUARD_SHA256 = (
    "1c2ff52815b09d4693446f39b90789f3"
    "aa2c7c8d083199ab59c263ef486b4919"
)

T60_GUARD_MESSAGE = (
    "T60 audit sequence must contain T0 through T59."
)

FORWARD_SUPPORT_CONSTANTS = {
    "TREND_ITEM_DIM": 80,
    "TREND_QUERY_DIM": 80,
    "TREND_DIM": 40,
    "NUM_HISTORY_PERIODS": 60,
    "NUM_RELATIONS": 12,
}


# =============================================================================
# Frozen model dimensions
# =============================================================================

NUM_INVESTORS = 165_975
NUM_STARTUPS = 311_589
NUM_NODES = 477_564

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
# Frozen epoch-0 stream fingerprints
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

EXPECTED_TRAIN_POSITIVES = 1_073_249
EXPECTED_EPOCH_NEGATIVES = 4_292_996
EXPECTED_EPOCH_EXAMPLES = 5_366_245

EXPECTED_FIRST_BATCH_POSITIVES = 105
EXPECTED_FIRST_BATCH_NEGATIVES = 407
EXPECTED_FIRST_BATCH_SEGMENTS = 59


# =============================================================================
# Exact frozen Adam
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
# Frozen Phase-5 contracts
# =============================================================================

PHASE_5_3_1L_2A_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_1l_2a_training_trend_sequence_adapter_contract.json"
)

PHASE_5_3_1L_2A_MANIFEST_PATH = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_1l_2a/"
    "phase_5_3_1l_2a_training_trend_adapter_manifest.json"
)

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
# Frozen Phase-3 / Phase-4 static inputs
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

LABEL_MATRIX_PATH = Path(
    "data/experimental/phase_4/"
    "description_labels/"
    "description_label_multihot.npz"
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
    "phase_5_3_1l_2b"
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
    / "epoch0_first_batch_static_integrity.csv"
)

ADAPTER_INTEGRITY_PATH = (
    AUDIT_DIR
    / "training_trend_adapter_integrity.csv"
)

ADAM_CONFIG_PATH = (
    AUDIT_DIR
    / "adam_configuration_audit.csv"
)

TREND_GROUP_PATH = (
    AUDIT_DIR
    / "mixed_h_training_trend_audit.csv"
)

FORWARD_PATH = (
    AUDIT_DIR
    / "epoch0_first_batch_forward_audit.csv"
)

GRADIENT_PATH = (
    AUDIT_DIR
    / "epoch0_first_batch_gradient_audit.csv"
)

STATE_PATH = (
    AUDIT_DIR
    / "pre_step_state_neutrality.csv"
)

FINAL_INVARIANT_PATH = (
    AUDIT_DIR
    / "phase_5_3_1l_2b_final_invariants.csv"
)

MANIFEST_PATH = (
    AUDIT_DIR
    / "phase_5_3_1l_2b_adam_batch_preflight_manifest.json"
)

CONTRACT_PATH = (
    CONTRACT_DIR
    / "phase_5_3_1l_2b_corrected_adam_first_batch_preflight_contract.json"
)

DECISION_REGISTER_PATH = (
    CONTRACT_DIR
    / "phase_5_3_1l_2b_decision_register.csv"
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

    if not bool(condition):

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
                "Missing gradient while "
                f"hashing {name}."
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
# Frozen stream logical hashes
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

    for start in range(
        0,
        len(
            ids
        ),
        10_000,
    ):

        text = "\n".join(
            ids[
                start:
                min(
                    start + 10_000,
                    len(ids),
                )
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
            ].to_numpy(
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

    for column in columns:

        digest.update(
            column.encode(
                "utf-8"
            )
        )

        digest.update(
            b"\0"
        )

        series = frame[
            column
        ]

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
                series
                .astype(
                    str
                )
                .tolist()
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
# AST utilities
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
        len(matches)
        == 1,
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
            and node.name
            == name
        )
    ]

    require(
        len(matches)
        == 1,
        (
            f"Expected one top-level "
            f"{name}()."
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
            len(
                node.targets
            )
            == 1
            and isinstance(
                node.targets[0],
                ast.Name,
            )
        ):

            return (
                node.targets[0].id
            )

    if isinstance(
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
            and assigned_name(node)
            == name
        )
    ]

    require(
        len(matches)
        == 1,
        (
            "Expected one top-level "
            f"assignment to {name}."
        ),
    )

    return matches[0]


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

    exec(
        compile(
            tree,
            filename=filename,
            mode="exec",
        ),
        module.__dict__,
    )


# =============================================================================
# T60 guard / adapter derivation
# =============================================================================

def statement_contains_exact_guard(
    statement: ast.stmt,
) -> bool:

    message_found = False
    require_found = False

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

                require_found = True

    return (
        message_found
        and require_found
    )


def derive_training_adapter(
    original_method: ast.FunctionDef,
):

    removed = [
        statement
        for statement in original_method.body
        if statement_contains_exact_guard(
            statement
        )
    ]

    require(
        len(removed)
        == 1,
        (
            "Expected exactly one "
            "T60-only guard."
        ),
    )

    adapter = copy.deepcopy(
        original_method
    )

    adapter.name = (
        "encode_training_sequence"
    )

    adapter.body = [
        copy.deepcopy(
            statement
        )
        for statement in original_method.body
        if not statement_contains_exact_guard(
            statement
        )
    ]

    ast.fix_missing_locations(
        adapter
    )

    require(
        len(
            adapter.body
        )
        == (
            len(
                original_method.body
            )
            - 1
        ),
        (
            "Adapter did not remove exactly "
            "one statement."
        ),
    )

    return (
        adapter,
        removed[0],
    )


# =============================================================================
# Canonical side-effect-free runtime
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
            "Canonical workflow "
            "boundary drift."
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

    runtime_sha = text_sha256(
        ast.dump(
            sanitized,
            annotate_fields=True,
            include_attributes=False,
        )
    )

    require(
        runtime_sha
        == EXPECTED_RUNTIME_AST_SHA256,
        (
            "Canonical sanitized runtime "
            "AST drift."
        ),
    )

    module_name = (
        "_itrs_phase5_3_1l_2b_canonical"
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
# Exact Phase-4.6.2 numerical method namespace + adapter
# =============================================================================

def build_forward_runtime(
    tree: ast.Module,
):

    module_name = (
        "_itrs_phase5_3_1l_2b_forward"
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

    for (
        name,
        expected,
    ) in FORWARD_SUPPORT_CONSTANTS.items():

        assignment = top_level_assignment(
            tree,
            name,
        )

        actual = ast.literal_eval(
            assignment.value
        )

        require(
            actual == expected,
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

    classes = class_map(
        tree
    )

    class_nodes = []

    for (
        class_name,
        method_names,
    ) in METHOD_GRAFT_SPEC.items():

        require(
            class_name in classes,
            (
                f"Missing class "
                f"{class_name}."
            ),
        )

        class_node = classes[
            class_name
        ]

        class_nodes.append(
            class_node
        )

        for method_name in (
            method_names
        ):

            method = direct_method(
                class_node,
                method_name,
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
                ast_sha256(
                    method
                )
                == expected_sha,
                (
                    "Frozen method AST drift: "
                    f"{class_name}.{method_name}"
                ),
            )

    execute_nodes(
        module,
        class_nodes,
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
                runtime_class
                .__dict__[
                    method_name
                ]
            )

    require(
        len(methods)
        == 7,
        (
            "Expected seven exact "
            "numerical methods."
        ),
    )

    # -------------------------------------------------------------------------
    # Re-derive the frozen Phase-5.3.1l.2a training adapter.
    # -------------------------------------------------------------------------

    trend_class_node = (
        classes[
            "TrendExtractor"
        ]
    )

    original_encode = direct_method(
        trend_class_node,
        "encode_sequence",
    )

    require(
        ast_sha256(
            original_encode
        )
        == (
            EXPECTED_ORIGINAL_ENCODE_SEQUENCE_AST_SHA256
        ),
        (
            "Original encode_sequence "
            "AST changed."
        ),
    )

    (
        adapter_node,
        removed_guard,
    ) = derive_training_adapter(
        original_encode
    )

    adapter_sha = ast_sha256(
        adapter_node
    )

    removed_guard_sha = ast_sha256(
        removed_guard
    )

    require(
        adapter_sha
        == EXPECTED_TRAINING_ADAPTER_AST_SHA256,
        (
            "Frozen training-adapter "
            "AST drift."
        ),
    )

    require(
        removed_guard_sha
        == EXPECTED_REMOVED_GUARD_SHA256,
        (
            "Removed T60 guard "
            "AST drift."
        ),
    )

    execute_nodes(
        module,
        [
            adapter_node,
        ],
        str(
            FORWARD_SOURCE_PATH
        ),
    )

    adapter_callable = (
        module.__dict__[
            "encode_training_sequence"
        ]
    )

    return (
        module,
        methods,
        adapter_callable,
        adapter_sha,
        removed_guard_sha,
    )


# =============================================================================
# Compose canonical model
# =============================================================================

def compose_canonical_model(
    canonical_runtime,
    methods: dict,
    adapter_callable,
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
        hash_fn(model)
        == EXPECTED_INITIAL_STATE_SHA256,
        (
            "Canonical initial-state "
            "hash mismatch."
        ),
    )

    # -------------------------------------------------------------------------
    # Frozen R-GCN metadata bridge.
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

    require(
        tuple(
            name
            for (
                name,
                _
            ) in basis_instances
        )
        == (
            "preference_propagation.layer_1",
            "preference_propagation.layer_2",
        ),
        (
            "Canonical R-GCN paths changed."
        ),
    )

    for (
        name,
        layer,
    ) in basis_instances:

        root_out = int(
            layer.root_weight.shape[
                1
            ]
        )

        basis_out = int(
            layer.bases.shape[
                2
            ]
        )

        require(
            root_out
            == basis_out
            == STRUCTURAL_DIM,
            (
                f"{name}: out_dim "
                "derivation drift."
            ),
        )

        setattr(
            layer,
            "out_dim",
            root_out,
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

    # -------------------------------------------------------------------------
    # Exact frozen Phase-4.6.2 numerical methods.
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

            method = methods[
                (
                    class_name,
                    method_name,
                )
            ]

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
                    "Method attachment failed: "
                    f"{class_name}.{method_name}"
                ),
            )

    # -------------------------------------------------------------------------
    # Frozen Phase-5.3.1l.2a training-only adapter.
    # Original encode_sequence remains present and unchanged.
    # -------------------------------------------------------------------------

    canonical_trend_class = getattr(
        canonical_runtime,
        "TrendExtractor",
    )

    setattr(
        canonical_trend_class,
        "encode_training_sequence",
        adapter_callable,
    )

    require(
        model
        .trend_extractor
        .encode_training_sequence
        .__func__
        is adapter_callable,
        (
            "Training adapter binding failed."
        ),
    )

    require(
        hash_fn(model)
        == EXPECTED_INITIAL_STATE_SHA256,
        (
            "Runtime method composition "
            "changed parameters."
        ),
    )

    return (
        model,
        hash_fn,
    )


# =============================================================================
# Adam
# =============================================================================

def build_frozen_adam(
    model: torch.nn.Module,
):

    return torch.optim.Adam(
        model.parameters(),
        lr=ADAM_LR,
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
        weight_decay=ADAM_WEIGHT_DECAY,
        amsgrad=ADAM_AMSGRAD,
        foreach=ADAM_FOREACH,
        fused=ADAM_FUSED,
        maximize=ADAM_MAXIMIZE,
        capturable=ADAM_CAPTURABLE,
        differentiable=ADAM_DIFFERENTIABLE,
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.3.1l.2b — "
        "CORRECTED ADAM + EPOCH-0 FIRST-BATCH "
        "FORWARD/BACKWARD PREFLIGHT"
    )

    print(
        "New negatives generated:              NO"
    )

    print(
        "New epoch permutation generated:      NO"
    )

    print(
        "Canonical model instantiated:         YES"
    )

    print(
        "Training trend adapter used:          YES"
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
    # Contract / artifact prerequisites
    # =========================================================================

    banner(
        "AUTHORITATIVE CONTRACT / ARTIFACT RECHECK"
    )

    required_paths = (
        CANONICAL_SOURCE_PATH,
        FORWARD_SOURCE_PATH,
        PHASE_5_3_1L_2A_CONTRACT_PATH,
        PHASE_5_3_1L_2A_MANIFEST_PATH,
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
        LABEL_MATRIX_PATH,
        TREND_PERIOD_PTR_PATH,
        TREND_STARTUP_INDICES_PATH,
        TREND_PERIOD_COUNTS_PATH,
    )

    for path in required_paths:

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

    adapter_contract = load_json(
        PHASE_5_3_1L_2A_CONTRACT_PATH
    )

    adapter_manifest = load_json(
        PHASE_5_3_1L_2A_MANIFEST_PATH
    )

    stream_contract = load_json(
        PHASE_5_3_1L_1_CONTRACT_PATH
    )

    stream_manifest = load_json(
        PHASE_5_3_1L_1_MANIFEST_PATH
    )

    k_contract = load_json(
        PHASE_5_3_1K_CONTRACT_PATH
    )

    j_contract = load_json(
        PHASE_5_3_1J_CONTRACT_PATH
    )

    bridge_contract = load_json(
        PHASE_5_3_1I_2_CONTRACT_PATH
    )

    require(
        adapter_contract[
            "status"
        ]
        == "FROZEN",
        (
            "5.3.1l.2a adapter "
            "contract not frozen."
        ),
    )

    require(
        adapter_manifest[
            "status"
        ]
        == (
            "TRAINING_TREND_SEQUENCE_ADAPTER_"
            "PROVED_AND_FROZEN"
        ),
        (
            "Unexpected adapter "
            "manifest status."
        ),
    )

    require(
        adapter_manifest[
            "adapter_method_ast_sha256"
        ]
        == EXPECTED_TRAINING_ADAPTER_AST_SHA256,
        (
            "Adapter manifest "
            "AST mismatch."
        ),
    )

    require(
        adapter_manifest[
            "T60_exact_equivalence"
        ]
        is True,
        (
            "Adapter T60 equivalence "
            "not proven."
        ),
    )

    require(
        stream_contract[
            "status"
        ]
        == "FROZEN",
        (
            "5.3.1l.1 stream "
            "contract not frozen."
        ),
    )

    require(
        stream_contract[
            "mixed_segment_forward"
        ][
            "post_h_GRU_padding"
        ]
        is False,
        (
            "Frozen no-padding "
            "rule changed."
        ),
    )

    require(
        int(
            stream_manifest[
                "optimizer_steps"
            ]
        )
        == 0,
        (
            "Optimizer step occurred "
            "before this phase."
        ),
    )

    require(
        k_contract[
            "status"
        ]
        == "FROZEN",
        (
            "5.3.1k not frozen."
        ),
    )

    require(
        j_contract[
            "status"
        ]
        == "FROZEN",
        (
            "5.3.1j not frozen."
        ),
    )

    require(
        bridge_contract[
            "status"
        ]
        == "FROZEN",
        (
            "5.3.1i.2 not frozen."
        ),
    )

    print(
        "5.3.1l.2a training adapter:           FROZEN"
    )

    print(
        "5.3.1l.1 training stream:             FROZEN"
    )

    print(
        "5.3.1k numerical runtime:             FROZEN"
    )

    print(
        "5.3.1j numerical methods:             FROZEN"
    )

    print(
        "5.3.1i.2 R-GCN metadata bridge:       FROZEN"
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
    # Frozen stream verification
    # =========================================================================

    banner(
        "FROZEN EPOCH-0 STREAM VERIFICATION"
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

    segment_groups = pd.read_csv(
        FIRST_BATCH_SEGMENT_GROUP_PATH
    )

    require(
        len(
            positive_order
        )
        == EXPECTED_TRAIN_POSITIVES,
        (
            "Training-positive count drift."
        ),
    )

    require(
        negative_matrix.shape
        == (
            EXPECTED_TRAIN_POSITIVES,
            4,
        ),
        (
            "Negative matrix shape drift."
        ),
    )

    require(
        negative_matrix.size
        == EXPECTED_EPOCH_NEGATIVES,
        (
            "Epoch-negative count drift."
        ),
    )

    require(
        epoch_order.shape
        == (
            EXPECTED_EPOCH_EXAMPLES,
        ),
        (
            "Epoch-order shape drift."
        ),
    )

    require(
        len(
            first_batch
        )
        == BATCH_SIZE,
        (
            "First batch is not 512."
        ),
    )

    positive_sha = (
        positive_stream_logical_sha256(
            positive_order
        )
    )

    negative_sha = (
        array_logical_sha256(
            np.asarray(
                negative_matrix
            )
        )
    )

    epoch_order_sha = (
        array_logical_sha256(
            np.asarray(
                epoch_order
            )
        )
    )

    first_batch_sha = (
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
        positive_sha
        == EXPECTED_POSITIVE_ORDER_LOGICAL_SHA256,
        (
            "Positive-order SHA drift."
        ),
    )

    require(
        negative_sha
        == EXPECTED_NEGATIVE_MATRIX_LOGICAL_SHA256,
        (
            "Negative-matrix SHA drift."
        ),
    )

    require(
        epoch_order_sha
        == EXPECTED_EPOCH_ORDER_LOGICAL_SHA256,
        (
            "Epoch-order SHA drift."
        ),
    )

    require(
        first_batch_sha
        == EXPECTED_FIRST_BATCH_LOGICAL_SHA256,
        (
            "First-batch SHA drift."
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
            "First-batch positive "
            "count drift."
        ),
    )

    require(
        negative_count
        == EXPECTED_FIRST_BATCH_NEGATIVES,
        (
            "First-batch negative "
            "count drift."
        ),
    )

    require(
        distinct_segments
        == EXPECTED_FIRST_BATCH_SEGMENTS,
        (
            "First batch no longer "
            "contains all T1..T59."
        ),
    )

    require(
        set(
            first_batch[
                "segment_number"
            ].astype(int)
        )
        == set(
            range(
                1,
                60,
            )
        ),
        (
            "First-batch segment set "
            "is not exactly T1..T59."
        ),
    )

    require(
        set(
            segment_groups[
                "segment_number"
            ].astype(int)
        )
        == set(
            range(
                1,
                60,
            )
        ),
        (
            "Frozen segment-group "
            "manifest is not T1..T59."
        ),
    )

    require(
        bool(
            (
                segment_groups[
                    "history_periods_consumed"
                ]
                == segment_groups[
                    "segment_number"
                ]
            ).all()
        ),
        (
            "Segment-group history "
            "length != h."
        ),
    )

    stream_df = pd.DataFrame(
        [
            {
                "artifact": (
                    "positive_order"
                ),
                "actual_sha256": (
                    positive_sha
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
                "actual_sha256": (
                    negative_sha
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
                "actual_sha256": (
                    epoch_order_sha
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
                "actual_sha256": (
                    first_batch_sha
                ),
                "expected_sha256": (
                    EXPECTED_FIRST_BATCH_LOGICAL_SHA256
                ),
                "status": (
                    "PASS"
                ),
            },
        ]
    )

    print(
        stream_df.to_string(
            index=False
        )
    )

    print()

    print(
        f"First batch:                          "
        f"{positive_count} positive + "
        f"{negative_count} negative"
    )

    print(
        "Target segments:                      T1..T59"
    )

    print(
        "New negative sampling:                NO"
    )

    print(
        "New epoch shuffle:                    NO"
    )

    # =========================================================================
    # Frozen static artifacts
    # =========================================================================

    banner(
        "STATIC GRAPH / FEATURE INTEGRITY"
    )

    static_rows = []

    for (
        path_string,
        expected_sha,
    ) in EXPECTED_STATIC_SHA256.items():

        actual_sha = file_sha256(
            Path(
                path_string
            )
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
                    if actual_sha
                    == expected_sha
                    else "FAIL"
                ),
            }
        )

    static_df = pd.DataFrame(
        static_rows
    )

    require(
        bool(
            (
                static_df[
                    "status"
                ]
                == "PASS"
            ).all()
        ),
        (
            "Static artifact integrity "
            "failure."
        ),
    )

    print(
        static_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Parse and reconstruct canonical numerical runtime
    # =========================================================================

    banner(
        "RECONSTRUCT CANONICAL TRAINING RUNTIME"
    )

    canonical_source = (
        CANONICAL_SOURCE_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    forward_source = (
        FORWARD_SOURCE_PATH
        .read_text(
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

    (
        canonical_runtime,
        runtime_ast_sha,
    ) = build_canonical_runtime(
        canonical_tree
    )

    (
        forward_runtime,
        exact_methods,
        training_adapter,
        adapter_sha,
        removed_guard_sha,
    ) = build_forward_runtime(
        forward_tree
    )

    (
        model,
        canonical_hash_fn,
    ) = compose_canonical_model(
        canonical_runtime,
        exact_methods,
        training_adapter,
    )

    model.train()

    require(
        adapter_sha
        == adapter_manifest[
            "adapter_method_ast_sha256"
        ],
        (
            "Re-derived adapter differs "
            "from frozen 5.3.1l.2a."
        ),
    )

    adapter_df = pd.DataFrame(
        [
            {
                "check": (
                    "adapter_AST"
                ),
                "actual": (
                    adapter_sha
                ),
                "expected": (
                    EXPECTED_TRAINING_ADAPTER_AST_SHA256
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "check": (
                    "removed_guard_AST"
                ),
                "actual": (
                    removed_guard_sha
                ),
                "expected": (
                    EXPECTED_REMOVED_GUARD_SHA256
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "check": (
                    "T60_equivalence_prior_proof"
                ),
                "actual": (
                    str(
                        adapter_manifest[
                            "T60_exact_equivalence"
                        ]
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
                    "post_h_padding"
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
        adapter_df.to_string(
            index=False
        )
    )

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
            "Parameter-count drift."
        ),
    )

    require(
        parameter_tensor_count
        == EXPECTED_PARAMETER_TENSORS,
        (
            "Parameter-tensor-count drift."
        ),
    )

    initial_hash = canonical_hash_fn(
        model
    )

    require(
        initial_hash
        == EXPECTED_INITIAL_STATE_SHA256,
        (
            "Canonical model hash "
            "mismatch before Adam."
        ),
    )

    print()

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
        initial_hash
    )

    # =========================================================================
    # Exact Adam construction
    # =========================================================================

    banner(
        "FIRST SUCCESSFUL FROZEN ADAM INSTANTIATION"
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

    parameters_before_adam = (
        parameter_hashes(
            model
        )
    )

    optimizer = build_frozen_adam(
        model
    )

    scheduler = None
    optimizer_step_count = 0

    hash_after_adam = canonical_hash_fn(
        model
    )

    require(
        hash_after_adam
        == EXPECTED_INITIAL_STATE_SHA256,
        (
            "Adam construction modified "
            "canonical parameters."
        ),
    )

    require(
        parameters_before_adam
        == parameter_hashes(
            model
        ),
        (
            "Adam construction changed "
            "parameter bytes."
        ),
    )

    require(
        python_rng_before_adam
        == random.getstate(),
        (
            "Adam changed Python RNG."
        ),
    )

    require(
        numpy_rng_state_equal(
            numpy_rng_before_adam,
            np.random.get_state(),
        ),
        (
            "Adam changed NumPy RNG."
        ),
    )

    require(
        torch.equal(
            torch_rng_before_adam,
            torch.get_rng_state(),
        ),
        (
            "Adam changed torch RNG."
        ),
    )

    require(
        len(
            optimizer.param_groups
        )
        == 1,
        (
            "Expected one Adam "
            "parameter group."
        ),
    )

    group = optimizer.param_groups[
        0
    ]

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer_parameters = group[
        "params"
    ]

    require(
        len(
            optimizer_parameters
        )
        == EXPECTED_PARAMETER_TENSORS,
        (
            "Adam tensor count drift."
        ),
    )

    require(
        all(
            left is right
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
            "mismatch."
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

    adam_checks = {
        "lr": (
            float(
                group[
                    "lr"
                ]
            )
            == ADAM_LR
        ),

        "betas": (
            tuple(
                group[
                    "betas"
                ]
            )
            == ADAM_BETAS
        ),

        "eps": (
            float(
                group[
                    "eps"
                ]
            )
            == ADAM_EPS
        ),

        "weight_decay": (
            float(
                group[
                    "weight_decay"
                ]
            )
            == ADAM_WEIGHT_DECAY
        ),

        "amsgrad": (
            bool(
                group[
                    "amsgrad"
                ]
            )
            == ADAM_AMSGRAD
        ),

        "foreach": (
            group.get(
                "foreach"
            )
            is ADAM_FOREACH
        ),

        "fused": (
            group.get(
                "fused"
            )
            is ADAM_FUSED
        ),

        "maximize": (
            bool(
                group.get(
                    "maximize"
                )
            )
            == ADAM_MAXIMIZE
        ),

        "capturable": (
            bool(
                group.get(
                    "capturable"
                )
            )
            == ADAM_CAPTURABLE
        ),

        "differentiable": (
            bool(
                group.get(
                    "differentiable"
                )
            )
            == ADAM_DIFFERENTIABLE
        ),
    }

    require(
        all(
            adam_checks.values()
        ),
        (
            "Frozen Adam configuration "
            "mismatch."
        ),
    )

    adam_df = pd.DataFrame(
        [
            {
                "setting": (
                    key
                ),
                "result": (
                    "PASS"
                    if value
                    else "FAIL"
                ),
            }
            for (
                key,
                value,
            ) in adam_checks.items()
        ]
        + [
            {
                "setting": (
                    "scheduler_none"
                ),
                "result": (
                    "PASS"
                ),
            },

            {
                "setting": (
                    "Adam_state_empty_pre_step"
                ),
                "result": (
                    "PASS"
                ),
            },

            {
                "setting": (
                    "optimizer_steps_zero"
                ),
                "result": (
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

    # =========================================================================
    # Load real feature / graph arrays
    # =========================================================================

    banner(
        "LOAD REAL TRAINING INPUTS"
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
            "edge_index shape drift."
        ),
    )

    require(
        edge_type_np.shape
        == (
            NUM_EDGES,
        ),
        (
            "edge_type shape drift."
        ),
    )

    require(
        doc2vec_all.shape
        == (
            NUM_NODES,
            DOC2VEC_DIM,
        ),
        (
            "Doc2Vec shape drift."
        ),
    )

    require(
        labels_sparse.shape
        == (
            NUM_NODES,
            LABEL_DIM,
        ),
        (
            "Label matrix shape drift."
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
            "Trend pointer shape drift."
        ),
    )

    require(
        trend_period_counts.shape
        == (
            NUM_INVESTORS
            * NUM_HISTORY_PERIODS,
        ),
        (
            "Trend-count shape drift."
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
    # Frozen batch arrays
    # =========================================================================

    batch_investors = (
        first_batch[
            "investor_global"
        ].to_numpy(
            dtype=np.int64
        )
    )

    batch_startup_locals = (
        first_batch[
            "startup_local"
        ].to_numpy(
            dtype=np.int64
        )
    )

    batch_startup_globals = (
        first_batch[
            "startup_global"
        ].to_numpy(
            dtype=np.int64
        )
    )

    batch_segments = (
        first_batch[
            "segment_number"
        ].to_numpy(
            dtype=np.int64
        )
    )

    batch_labels = (
        first_batch[
            "label"
        ].to_numpy(
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
            "Batch target outside "
            "T1..T59."
        ),
    )

    # =========================================================================
    # Build exact T0..T(h-1) histories
    # =========================================================================

    banner(
        "BIND EXACT VARIABLE-h TREND HISTORIES"
    )

    unique_keys = sorted(
        {
            (
                int(
                    investor
                ),
                int(
                    h
                ),
            )
            for (
                investor,
                h,
            ) in zip(
                batch_investors,
                batch_segments,
            )
        },
        key=lambda value: (
            value[1],
            value[0],
        ),
    )

    history_by_key = {}
    historical_nodes = []

    for (
        investor_global,
        h,
    ) in unique_keys:

        require(
            0
            <= investor_global
            < NUM_INVESTORS,
            (
                "Investor index outside "
                "role universe."
            ),
        )

        require(
            1
            <= h
            <= 59,
            (
                "Invalid target h."
            ),
        )

        periods = []

        for period in range(
            h
        ):

            flattened = (
                investor_global
                * NUM_HISTORY_PERIODS
                + period
            )

            start = int(
                trend_period_ptr[
                    flattened
                ]
            )

            end = int(
                trend_period_ptr[
                    flattened + 1
                ]
            )

            count = int(
                trend_period_counts[
                    flattened
                ]
            )

            require(
                end
                - start
                == count,
                (
                    "Trend CSR mismatch: "
                    f"Investor={investor_global}, "
                    f"period={period}"
                ),
            )

            values = np.array(
                trend_startup_indices[
                    start:end
                ],
                dtype=np.int64,
                copy=True,
            )

            if len(values) > 0:

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
                        "Historical trend "
                        "contains non-Startup node."
                    ),
                )

                historical_nodes.append(
                    values
                )

            periods.append(
                values
            )

        require(
            len(periods)
            == h,
            (
                "History length is not "
                "exactly h."
            ),
        )

        history_by_key[
            (
                investor_global,
                h,
            )
        ] = periods

    require(
        all(
            len(periods)
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
            "Post-h periods detected."
        ),
    )

    # =========================================================================
    # Description-node subset
    # =========================================================================

    node_parts = [
        np.asarray(
            batch_investors,
            dtype=np.int64,
        ),

        np.asarray(
            batch_startup_globals,
            dtype=np.int64,
        ),
    ]

    node_parts.extend(
        historical_nodes
    )

    required_nodes = np.unique(
        np.concatenate(
            node_parts
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
        bool(
            np.isfinite(
                doc_subset_np
            ).all()
        ),
        (
            "Non-finite Doc2Vec "
            "training subset."
        ),
    )

    require(
        bool(
            np.isfinite(
                label_subset_np
            ).all()
        ),
        (
            "Non-finite label "
            "training subset."
        ),
    )

    doc_subset = torch.from_numpy(
        doc_subset_np
    )

    label_subset = torch.from_numpy(
        label_subset_np
    )

    print(
        f"Batch examples:                       "
        f"{BATCH_SIZE}"
    )

    print(
        f"Unique (Investor,h) keys:             "
        f"{len(unique_keys)}"
    )

    print(
        f"Required description nodes:           "
        f"{len(required_nodes):,}"
    )

    print(
        "Post-h periods loaded:                0"
    )

    # =========================================================================
    # Pre-forward state
    # =========================================================================

    hash_before_forward = (
        canonical_hash_fn(
            model
        )
    )

    parameter_bytes_before = (
        parameter_hashes(
            model
        )
    )

    state_dict_before = (
        logical_state_dict_sha256(
            model
        )
    )

    require(
        hash_before_forward
        == EXPECTED_INITIAL_STATE_SHA256,
        (
            "Canonical state changed "
            "before forward."
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
            "Unexpected pre-existing "
            "gradients."
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
    # Full canonical batch forward
    # =========================================================================

    banner(
        "CORRECTED CANONICAL EPOCH-0 FIRST-BATCH FORWARD"
    )

    latent_all = torch.cat(
        [
            model.investor_embedding.weight,
            model.startup_embedding.weight,
        ],
        dim=0,
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
        )
        and "F_s"
        in structural,
        (
            "Invalid structural "
            "forward output."
        ),
    )

    F_s_all = structural[
        "F_s"
    ]

    require(
        F_s_all.shape
        == (
            NUM_NODES,
            STRUCTURAL_DIM,
        ),
        (
            "Structural output "
            "shape invalid."
        ),
    )

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
            "Description output "
            "shape invalid."
        ),
    )

    # -------------------------------------------------------------------------
    # Group exact trend sequences by h.
    # This time use the frozen training adapter.
    # -------------------------------------------------------------------------

    keys_by_h = {}

    for key in unique_keys:

        h = int(
            key[1]
        )

        keys_by_h.setdefault(
            h,
            [],
        ).append(
            key
        )

    require(
        set(
            keys_by_h.keys()
        )
        == set(
            range(
                1,
                60,
            )
        ),
        (
            "Expected mixed-h execution "
            "for all T1..T59."
        ),
    )

    trend_output_by_key = {}
    trend_audit_rows = []

    total_nonempty_periods = 0
    total_multi_item_periods = 0

    for h in range(
        1,
        60,
    ):

        keys = keys_by_h[
            h
        ]

        sequences = []

        h_nonempty = 0
        h_multi_item = 0

        for (
            investor_global,
            _,
        ) in keys:

            investor_tensor = torch.tensor(
                [
                    investor_global,
                ],
                dtype=torch.int64,
            )

            L_o_single = (
                model.investor_embedding(
                    investor_tensor
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
                    "Investor description "
                    "row missing."
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
                    "Trend query "
                    "shape invalid."
                ),
            )

            period_vectors = []

            periods = history_by_key[
                (
                    investor_global,
                    h,
                )
            ]

            require(
                len(periods)
                == h,
                (
                    "History periods != h."
                ),
            )

            for startup_globals in periods:

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

                    h_nonempty += 1
                    total_nonempty_periods += 1

                    if item_count >= 2:

                        h_multi_item += 1
                        total_multi_item_periods += 1

                    startup_locals = (
                        startup_globals
                        - NUM_INVESTORS
                    )

                    startup_tensor = (
                        torch.from_numpy(
                            np.array(
                                startup_locals,
                                dtype=np.int64,
                                copy=True,
                            )
                        )
                    )

                    history_latent = (
                        model.startup_embedding(
                            startup_tensor
                        )
                    )

                    subset_positions_np = (
                        global_to_subset[
                            startup_globals
                        ]
                    )

                    require(
                        bool(
                            (
                                subset_positions_np
                                >= 0
                            ).all()
                        ),
                        (
                            "Historical description "
                            "row missing."
                        ),
                    )

                    subset_positions = (
                        torch.from_numpy(
                            np.array(
                                subset_positions_np,
                                dtype=np.int64,
                                copy=True,
                            )
                        )
                    )

                    history_description = (
                        description_subset[
                            subset_positions
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
                            "Trend item "
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
                        alpha.shape
                        == (
                            item_count,
                        ),
                        (
                            "Attention alpha "
                            "shape invalid."
                        ),
                    )

                    require(
                        bool(
                            torch.isfinite(
                                alpha
                            ).all()
                        ),
                        (
                            "Non-finite "
                            "attention."
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
                    period_vector.shape
                    == (
                        TREND_ITEM_DIM,
                    ),
                    (
                        "Period-vector "
                        "shape invalid."
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
                    f"T{h}: sequence shape "
                    "is not exactly (h,80)."
                ),
            )

            sequences.append(
                sequence
            )

        grouped_sequence = torch.stack(
            sequences,
            dim=0,
        )

        require(
            grouped_sequence.shape
            == (
                len(keys),
                h,
                TREND_ITEM_DIM,
            ),
            (
                f"T{h}: grouped trend "
                "shape invalid."
            ),
        )

        # ================================================================
        # CRITICAL CORRECTION FROM FAILED 5.3.1l.2:
        #
        # Use the frozen training adapter, not the T60-only audit method.
        # ================================================================

        (
            group_F_t,
            group_gru_output,
        ) = (
            model
            .trend_extractor
            .encode_training_sequence(
                grouped_sequence
            )
        )

        require(
            group_F_t.shape
            == (
                len(keys),
                TREND_DIM,
            ),
            (
                f"T{h}: F_t shape invalid."
            ),
        )

        require(
            group_gru_output.shape
            == (
                len(keys),
                h,
                TREND_DIM,
            ),
            (
                f"T{h}: GRU output "
                "shape invalid."
            ),
        )

        require(
            bool(
                torch.isfinite(
                    group_F_t
                ).all()
            ),
            (
                f"T{h}: non-finite F_t."
            ),
        )

        for (
            position,
            key,
        ) in enumerate(
            keys
        ):

            trend_output_by_key[
                key
            ] = (
                group_F_t[
                    position
                ]
            )

        trend_audit_rows.append(
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
                            grouped_sequence.shape
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
                "GRU_output_shape": (
                    str(
                        tuple(
                            group_gru_output.shape
                        )
                    )
                ),
                "nonempty_periods": (
                    h_nonempty
                ),
                "multi_item_periods": (
                    h_multi_item
                ),
                "post_h_padding": (
                    False
                ),
                "runtime_method": (
                    "encode_training_sequence"
                ),
                "status": (
                    "PASS"
                ),
            }
        )

    trend_df = pd.DataFrame(
        trend_audit_rows
    )

    require(
        len(
            trend_df
        )
        == 59,
        (
            "Expected 59 trend "
            "target groups."
        ),
    )

    require(
        bool(
            (
                trend_df[
                    "history_periods_consumed"
                ]
                == trend_df[
                    "segment_number"
                ]
            ).all()
        ),
        (
            "At least one T_h group "
            "used wrong history length."
        ),
    )

    require(
        bool(
            (
                trend_df[
                    "post_h_padding"
                ]
                == False
            ).all()
        ),
        (
            "Post-h padding detected."
        ),
    )

    require(
        set(
            trend_df[
                "runtime_method"
            ]
        )
        == {
            "encode_training_sequence"
        },
        (
            "Training group used "
            "wrong trend method."
        ),
    )

    # Restore exact frozen batch order.
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
            "Restored F_t batch "
            "shape invalid."
        ),
    )

    # -------------------------------------------------------------------------
    # Endpoint representations
    # -------------------------------------------------------------------------

    investor_tensor = torch.from_numpy(
        np.array(
            batch_investors,
            dtype=np.int64,
            copy=True,
        )
    )

    startup_local_tensor = torch.from_numpy(
        np.array(
            batch_startup_locals,
            dtype=np.int64,
            copy=True,
        )
    )

    startup_global_tensor = torch.from_numpy(
        np.array(
            batch_startup_globals,
            dtype=np.int64,
            copy=True,
        )
    )

    investor_description_positions = (
        torch.from_numpy(
            np.array(
                global_to_subset[
                    batch_investors
                ],
                dtype=np.int64,
                copy=True,
            )
        )
    )

    startup_description_positions = (
        torch.from_numpy(
            np.array(
                global_to_subset[
                    batch_startup_globals
                ],
                dtype=np.int64,
                copy=True,
            )
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

    F_s_b_batch = (
        F_s_all[
            startup_global_tensor
        ]
    )

    investor_representation = torch.cat(
        [
            F_t_batch,
            L_o_batch,
            F_d_o_batch,
            F_s_o_batch,
        ],
        dim=1,
    )

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
            "Investor representation "
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
            "Startup representation "
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
            "Pair representation "
            "shape invalid."
        ),
    )

    scoring = model.scoring_mlp(
        pair_representation
    )

    require(
        isinstance(
            scoring,
            dict,
        )
        and "logit"
        in scoring,
        (
            "Scoring output invalid."
        ),
    )

    logits = scoring[
        "logit"
    ]

    require(
        logits.shape
        == (
            BATCH_SIZE,
            1,
        ),
        (
            "Logit shape invalid."
        ),
    )

    require(
        bool(
            torch.isfinite(
                logits
            ).all()
        ),
        (
            "Non-finite logits."
        ),
    )

    targets = torch.from_numpy(
        np.array(
            batch_labels,
            dtype=np.float32,
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
            "Target/logit "
            "shape mismatch."
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
            "Positive target-count drift."
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
            "Negative target-count drift."
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
            "Non-finite batch BCE."
        ),
    )

    probabilities = torch.sigmoid(
        logits
    )

    logits_sha = tensor_sha256(
        logits
    )

    forward_df = pd.DataFrame(
        [
            {
                "feature": (
                    "latent_all"
                ),
                "shape": (
                    str(
                        tuple(
                            latent_all.shape
                        )
                    )
                ),
                "expected": (
                    "(477564, 40)"
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "feature": (
                    "F_s_all"
                ),
                "shape": (
                    str(
                        tuple(
                            F_s_all.shape
                        )
                    )
                ),
                "expected": (
                    "(477564, 40)"
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "feature": (
                    "F_t_batch"
                ),
                "shape": (
                    str(
                        tuple(
                            F_t_batch.shape
                        )
                    )
                ),
                "expected": (
                    "(512, 40)"
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "feature": (
                    "investor_representation"
                ),
                "shape": (
                    str(
                        tuple(
                            investor_representation.shape
                        )
                    )
                ),
                "expected": (
                    "(512, 160)"
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "feature": (
                    "startup_representation"
                ),
                "shape": (
                    str(
                        tuple(
                            startup_representation.shape
                        )
                    )
                ),
                "expected": (
                    "(512, 120)"
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "feature": (
                    "pair_representation"
                ),
                "shape": (
                    str(
                        tuple(
                            pair_representation.shape
                        )
                    )
                ),
                "expected": (
                    "(512, 280)"
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "feature": (
                    "logits"
                ),
                "shape": (
                    str(
                        tuple(
                            logits.shape
                        )
                    )
                ),
                "expected": (
                    "(512, 1)"
                ),
                "status": (
                    "PASS"
                ),
            },
        ]
    )

    print(
        forward_df.to_string(
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
        f"Minimum logit:                        "
        f"{float(logits.detach().min()):.10f}"
    )

    print(
        f"Maximum logit:                        "
        f"{float(logits.detach().max()):.10f}"
    )

    print(
        f"Mean probability:                     "
        f"{float(probabilities.detach().mean()):.10f}"
    )

    print()

    print(
        "Logit SHA256:"
    )

    print(
        logits_sha
    )

    # =========================================================================
    # Forward must not modify parameters
    # =========================================================================

    hash_after_forward = (
        canonical_hash_fn(
            model
        )
    )

    require(
        hash_after_forward
        == EXPECTED_INITIAL_STATE_SHA256,
        (
            "Forward modified "
            "parameter values."
        ),
    )

    # =========================================================================
    # Backward
    # =========================================================================

    banner(
        "FIRST SUCCESSFUL EPOCH-0 TRAINING-BATCH BACKWARD"
    )

    loss.backward()

    gradient_rows = []

    for (
        name,
        parameter,
    ) in model.named_parameters():

        gradient = parameter.grad

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
            "Gradient audit does not "
            "cover all tensors."
        ),
    )

    require(
        bool(
            gradient_df[
                "gradient_exists"
            ].all()
        ),
        (
            "Missing parameter gradient."
        ),
    )

    require(
        bool(
            gradient_df[
                "gradient_finite"
            ].all()
        ),
        (
            "Non-finite parameter "
            "gradient."
        ),
    )

    require(
        bool(
            gradient_df[
                "gradient_nonzero"
            ].all()
        ),
        (
            "Zero gradient detected "
            "for trainable tensor."
        ),
    )

    gradient_sha = (
        gradient_logical_sha256(
            model
        )
    )

    print(
        "Parameter gradients:                  "
        "32 / 32"
    )

    print(
        "Finite gradients:                     "
        "32 / 32"
    )

    print(
        "Non-zero gradients:                   "
        "32 / 32"
    )

    print()

    print(
        "Gradient SHA256:"
    )

    print(
        gradient_sha
    )

    # =========================================================================
    # Adam state MUST still be empty
    # =========================================================================

    require(
        len(
            optimizer.state
        )
        == 0,
        (
            "Adam state appeared before "
            "optimizer.step()."
        ),
    )

    require(
        optimizer_step_count
        == 0,
        (
            "optimizer.step() count "
            "is not zero."
        ),
    )

    # =========================================================================
    # State neutrality
    # =========================================================================

    banner(
        "POST-BACKWARD / PRE-STEP STATE NEUTRALITY"
    )

    hash_after_backward = (
        canonical_hash_fn(
            model
        )
    )

    parameter_bytes_after = (
        parameter_hashes(
            model
        )
    )

    state_dict_after = (
        logical_state_dict_sha256(
            model
        )
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
        hash_after_backward
        == EXPECTED_INITIAL_STATE_SHA256,
        (
            "Backward modified "
            "parameter values."
        ),
    )

    require(
        parameter_bytes_before
        == parameter_bytes_after,
        (
            "Parameter bytes changed "
            "before optimizer.step()."
        ),
    )

    require(
        state_dict_before
        == state_dict_after,
        (
            "state_dict changed "
            "before optimizer.step()."
        ),
    )

    require(
        python_rng_before_forward
        == python_rng_after,
        (
            "Forward/backward changed "
            "Python RNG."
        ),
    )

    require(
        numpy_rng_state_equal(
            numpy_rng_before_forward,
            numpy_rng_after,
        ),
        (
            "Forward/backward changed "
            "NumPy RNG."
        ),
    )

    require(
        torch.equal(
            torch_rng_before_forward,
            torch_rng_after,
        ),
        (
            "Forward/backward changed "
            "torch RNG."
        ),
    )

    state_df = pd.DataFrame(
        [
            {
                "check": (
                    "initial_hash"
                ),
                "actual": (
                    initial_hash
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
                    "hash_after_Adam"
                ),
                "actual": (
                    hash_after_adam
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
                    "hash_after_forward"
                ),
                "actual": (
                    hash_after_forward
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
                    "hash_after_backward"
                ),
                "actual": (
                    hash_after_backward
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
                        parameter_bytes_before
                        == parameter_bytes_after
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
                    "Adam_state_entries"
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
                    "optimizer_steps"
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
        "FINAL PHASE-5.3.1l.2b PREFLIGHT INVARIANTS"
    )

    checks = [
        (
            "phase_5_3_1l_2a_adapter_frozen",
            (
                adapter_contract[
                    "status"
                ]
                == "FROZEN"
            ),
        ),

        (
            "training_adapter_AST_exact",
            (
                adapter_sha
                == EXPECTED_TRAINING_ADAPTER_AST_SHA256
            ),
        ),

        (
            "removed_T60_guard_AST_exact",
            (
                removed_guard_sha
                == EXPECTED_REMOVED_GUARD_SHA256
            ),
        ),

        (
            "prior_T60_adapter_equivalence_proven",
            (
                adapter_manifest[
                    "T60_exact_equivalence"
                ]
                is True
            ),
        ),

        (
            "post_h_padding_forbidden",
            (
                stream_contract[
                    "mixed_segment_forward"
                ][
                    "post_h_GRU_padding"
                ]
                is False
            ),
        ),

        (
            "positive_order_hash_exact",
            (
                positive_sha
                == EXPECTED_POSITIVE_ORDER_LOGICAL_SHA256
            ),
        ),

        (
            "negative_matrix_hash_exact",
            (
                negative_sha
                == EXPECTED_NEGATIVE_MATRIX_LOGICAL_SHA256
            ),
        ),

        (
            "epoch_order_hash_exact",
            (
                epoch_order_sha
                == EXPECTED_EPOCH_ORDER_LOGICAL_SHA256
            ),
        ),

        (
            "first_batch_hash_exact",
            (
                first_batch_sha
                == EXPECTED_FIRST_BATCH_LOGICAL_SHA256
            ),
        ),

        (
            "first_batch_105_positives",
            (
                positive_count
                == 105
            ),
        ),

        (
            "first_batch_407_negatives",
            (
                negative_count
                == 407
            ),
        ),

        (
            "first_batch_exact_T1_to_T59",
            (
                set(
                    first_batch[
                        "segment_number"
                    ].astype(int)
                )
                == set(
                    range(
                        1,
                        60,
                    )
                )
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
            "canonical_runtime_AST_exact",
            (
                runtime_ast_sha
                == EXPECTED_RUNTIME_AST_SHA256
            ),
        ),

        (
            "canonical_initial_hash_exact",
            (
                initial_hash
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
            "Adam_exact_configuration",
            all(
                adam_checks.values()
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
            "Adam_state_empty_pre_step",
            (
                len(
                    optimizer.state
                )
                == 0
            ),
        ),

        (
            "all_59_mixed_h_groups_executed",
            (
                len(
                    trend_df
                )
                == 59
            ),
        ),

        (
            "training_adapter_used_for_all_groups",
            (
                set(
                    trend_df[
                        "runtime_method"
                    ]
                )
                == {
                    "encode_training_sequence"
                }
            ),
        ),

        (
            "each_T_h_consumes_exactly_h_periods",
            bool(
                (
                    trend_df[
                        "history_periods_consumed"
                    ]
                    == trend_df[
                        "segment_number"
                    ]
                ).all()
            ),
        ),

        (
            "no_post_h_padding",
            bool(
                (
                    trend_df[
                        "post_h_padding"
                    ]
                    == False
                ).all()
            ),
        ),

        (
            "F_t_restored_to_512_examples",
            (
                F_t_batch.shape
                == (
                    512,
                    40,
                )
            ),
        ),

        (
            "pair_shape_exact",
            (
                pair_representation.shape
                == (
                    512,
                    280,
                )
            ),
        ),

        (
            "logit_shape_exact",
            (
                logits.shape
                == (
                    512,
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
            "batch_BCE_finite",
            bool(
                torch.isfinite(
                    loss
                )
            ),
        ),

        (
            "all_32_gradients_exist",
            (
                int(
                    gradient_df[
                        "gradient_exists"
                    ].sum()
                )
                == 32
            ),
        ),

        (
            "all_32_gradients_finite",
            (
                int(
                    gradient_df[
                        "gradient_finite"
                    ].sum()
                )
                == 32
            ),
        ),

        (
            "all_32_gradients_nonzero",
            (
                int(
                    gradient_df[
                        "gradient_nonzero"
                    ].sum()
                )
                == 32
            ),
        ),

        (
            "hash_after_forward_exact",
            (
                hash_after_forward
                == EXPECTED_INITIAL_STATE_SHA256
            ),
        ),

        (
            "hash_after_backward_exact",
            (
                hash_after_backward
                == EXPECTED_INITIAL_STATE_SHA256
            ),
        ),

        (
            "parameter_bytes_unchanged",
            (
                parameter_bytes_before
                == parameter_bytes_after
            ),
        ),

        (
            "state_dict_unchanged",
            (
                state_dict_before
                == state_dict_after
            ),
        ),

        (
            "forward_backward_RNG_neutral",
            (
                python_rng_before_forward
                == python_rng_after
                and numpy_rng_state_equal(
                    numpy_rng_before_forward,
                    numpy_rng_after,
                )
                and torch.equal(
                    torch_rng_before_forward,
                    torch_rng_after,
                )
            ),
        ),

        (
            "Adam_state_still_empty",
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
        bool(
            (
                invariant_df[
                    "result"
                ]
                == "PASS"
            ).all()
        ),
        (
            "At least one Phase-5.3.1l.2b "
            "preflight invariant failed."
        ),
    )

    print(
        invariant_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Write freeze outputs
    # =========================================================================

    banner(
        "WRITE PHASE-5.3.1l.2b OUTPUTS"
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

    adapter_df.to_csv(
        ADAPTER_INTEGRITY_PATH,
        index=False,
    )

    adam_df.to_csv(
        ADAM_CONFIG_PATH,
        index=False,
    )

    trend_df.to_csv(
        TREND_GROUP_PATH,
        index=False,
    )

    forward_df.to_csv(
        FORWARD_PATH,
        index=False,
    )

    gradient_df.to_csv(
        GRADIENT_PATH,
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

    # =========================================================================
    # Decision register
    # =========================================================================

    decision_df = pd.DataFrame(
        [
            {
                "decision": (
                    "training_trend_encoder"
                ),

                "value": (
                    "encode_training_sequence"
                ),

                "classification": (
                    "INHERITED_FROZEN_PHASE_5_3_1l_2a"
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1l_2b"
                ),
            },

            {
                "decision": (
                    "training_target_history"
                ),

                "value": (
                    "T_h_USES_T0_TO_T_h_MINUS_1"
                ),

                "classification": (
                    "INHERITED_FROZEN_PHASE_5_3_1l_1"
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1l_2b"
                ),
            },

            {
                "decision": (
                    "Adam_runtime"
                ),

                "value": (
                    "torch.optim.Adam_lr=.001_"
                    "betas=.9,.999_eps=1e-8_"
                    "weight_decay=0"
                ),

                "classification": (
                    "INHERITED_FROZEN_PHASE_5_2_2"
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1l_2b"
                ),
            },

            {
                "decision": (
                    "optimizer_step_policy"
                ),

                "value": (
                    "FORBIDDEN_IN_PREFLIGHT"
                ),

                "classification": (
                    "EVALUATION_INTEGRITY_GUARD"
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1l_2b"
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
            "5.3.1l.2b"
        ),

        "title": (
            "Corrected Adam + Epoch-0 First-Batch "
            "Forward/Backward Preflight Contract"
        ),

        "status": (
            "FROZEN"
        ),

        "training_adapter": {
            "method": (
                "TrendExtractor.encode_training_sequence"
            ),

            "adapter_ast_sha256": (
                adapter_sha
            ),

            "removed_guard_ast_sha256": (
                removed_guard_sha
            ),

            "T60_exact_equivalence_proven": (
                True
            ),

            "history_for_T_h": (
                "T0..T(h-1)"
            ),

            "post_h_padding": (
                False
            ),
        },

        "training_stream": {
            "positive_order_sha256": (
                positive_sha
            ),

            "negative_matrix_sha256": (
                negative_sha
            ),

            "epoch_order_sha256": (
                epoch_order_sha
            ),

            "first_batch_sha256": (
                first_batch_sha
            ),

            "batch_size": (
                BATCH_SIZE
            ),

            "positives": (
                positive_count
            ),

            "negatives": (
                negative_count
            ),

            "target_segments": (
                "T1..T59"
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

            "state_entries_before_step": (
                len(
                    optimizer.state
                )
            ),

            "optimizer_steps": (
                optimizer_step_count
            ),
        },

        "numerical_result": {
            "batch_loss": (
                float(
                    loss
                    .detach()
                    .item()
                )
            ),

            "logit_sha256": (
                logits_sha
            ),

            "gradient_sha256": (
                gradient_sha
            ),

            "mean_logit": (
                float(
                    logits
                    .detach()
                    .mean()
                )
            ),

            "mean_probability": (
                float(
                    probabilities
                    .detach()
                    .mean()
                )
            ),

            "all_32_gradients_exist": (
                True
            ),

            "all_32_gradients_finite": (
                True
            ),

            "all_32_gradients_nonzero": (
                True
            ),
        },

        "parameter_state": {
            "initial_sha256": (
                initial_hash
            ),

            "after_Adam_sha256": (
                hash_after_adam
            ),

            "after_forward_sha256": (
                hash_after_forward
            ),

            "after_backward_sha256": (
                hash_after_backward
            ),

            "expected_pre_step_sha256": (
                EXPECTED_INITIAL_STATE_SHA256
            ),

            "parameter_values_changed": (
                False
            ),
        },

        "training_boundary": {
            "Adam_instantiated": (
                True
            ),

            "forward_performed": (
                True
            ),

            "BCE_performed": (
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
                "Reconstruct this exact canonical pre-step "
                "state and exact first batch, reproduce the "
                "frozen pre-step logit and gradient fingerprints, "
                "then execute exactly one optimizer.step(). "
                "The canonical parameter-state hash must change "
                "for the first time while all values remain finite."
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
            "5.3.1l.2b"
        ),

        "status": (
            "CORRECTED_ADAM_EPOCH0_FIRST_BATCH_"
            "FORWARD_BACKWARD_PREFLIGHT_PASSED"
        ),

        "training_adapter_ast_sha256": (
            adapter_sha
        ),

        "first_batch_sha256": (
            first_batch_sha
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

        "canonical_state_sha256_after_backward": (
            hash_after_backward
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
        ADAPTER_INTEGRITY_PATH,
        ADAM_CONFIG_PATH,
        TREND_GROUP_PATH,
        FORWARD_PATH,
        GRADIENT_PATH,
        STATE_PATH,
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
        "PHASE 5.3.1l.2b FINAL STATUS"
    )

    print(
        "Training adapter:                     VERIFIED / FROZEN"
    )

    print(
        "Training adapter AST:"
    )

    print(
        adapter_sha
    )

    print()

    print(
        "Canonical first batch:                VERIFIED / FROZEN"
    )

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
        "Target segments:                      T1..T59"
    )

    print()

    print(
        "Trend runtime:"
    )

    print(
        "  target T_h -> T0..T(h-1)"
    )

    print(
        "  runtime method -> encode_training_sequence"
    )

    print(
        "  post-h GRU padding -> NONE"
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
        "Canonical state before training batch:"
    )

    print(
        hash_before_forward
    )

    print()

    print(
        "Canonical state after backward:"
    )

    print(
        hash_after_backward
    )

    print()

    print(
        "Parameter values changed:             NO"
    )

    print(
        "Adam state initialized:               NO"
    )

    print(
        "optimizer.step():                     0"
    )

    print(
        "Checkpoint written:                   NO"
    )

    banner(
        "PHASE 5.3.1l.2b COMPLETE / "
        "CORRECTED ADAM + EPOCH-0 FIRST-BATCH "
        "FORWARD/BACKWARD PREFLIGHT PASSED"
    )


if __name__ == "__main__":
    main()