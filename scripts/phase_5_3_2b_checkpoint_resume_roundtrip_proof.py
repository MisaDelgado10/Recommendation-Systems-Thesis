"""
Phase 5.3.2b — Checkpoint / Resume Round-Trip Numerical Proof

Goal
----
Prove that the frozen Phase-5 checkpoint/resume semantics are numerically exact.

Two paths are executed from the same canonical seed-42 model and the same
frozen epoch-0 training stream.

Path A — uninterrupted
----------------------
    canonical initial state
        -> batch 0 -> Adam step 1
        -> batch 1 -> Adam step 2

Path B — checkpoint / resume
----------------------------
    canonical initial state
        -> batch 0 -> Adam step 1
        -> write checkpoint
        -> destroy live model / optimizer
        -> reconstruct fresh canonical model / Adam
        -> load checkpoint
        -> restore RNG states and counters
        -> batch 1 -> Adam step 2

Required equality after batch 1
-------------------------------
    model parameter-state SHA256
    every individual parameter tensor SHA256
    Adam logical-state SHA256
    batch-1 logit SHA256
    batch-1 gradient SHA256
    batch-1 BCE loss

The checkpoint payload must conform exactly to the frozen Phase-5.3.2a
required-field schema.

Training/runtime invariants
---------------------------
- Frozen epoch-0 positive order / negative matrix / epoch order are reused.
- NO new negative sampling.
- NO new epoch shuffle.
- T_h consumes exactly T0..T(h-1).
- Frozen encode_training_sequence adapter is used.
- No post-h GRU padding.
- Adam configuration remains frozen.
- Path A performs exactly two optimizer.step() calls.
- Path B performs one step before save and one step after reload.
- No validation.
- No test.
- No production checkpoint is created.
- The written .pt file is an audit round-trip probe only.

This phase creates NEW permanent fingerprints for:
    epoch-0 batch 1 logical manifest
    batch-1 logits
    batch-1 gradients
    model state after two optimizer steps
    Adam state after two optimizer steps
"""

from __future__ import annotations

import ast
import copy
import gc
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse


# =============================================================================
# Frozen source/runtime dependencies
# =============================================================================

PREFLIGHT_SOURCE_PATH = Path(
    "scripts/"
    "phase_5_3_1l_2b_corrected_adam_epoch0_first_batch_preflight.py"
)

PHASE_5_3_1M_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_1m_first_adam_weight_update_contract.json"
)

PHASE_5_3_2A_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_2a_training_execution_state_contract.json"
)

PHASE_5_3_2A_MANIFEST_PATH = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_2a/"
    "phase_5_3_2a_training_execution_state_manifest.json"
)


# =============================================================================
# Frozen numerical fingerprints
# =============================================================================

EXPECTED_INITIAL_MODEL_SHA256 = (
    "49e822ea7fad35c458f47e134c94c05e"
    "ac099b68c5c468e2c71559c8c88998ab"
)

EXPECTED_FIRST_BATCH_SHA256 = (
    "8408432b944bcd0805af9c34ff1b2db3"
    "ea938e0649a75d381b7839b86cd280ea"
)

EXPECTED_FIRST_BATCH_LOGIT_SHA256 = (
    "35b89aaed29d51d2ebb7ba1cadf2dc4b"
    "b5e8f81cf3aa78bc216b3cc6fed13845"
)

EXPECTED_FIRST_BATCH_GRADIENT_SHA256 = (
    "8c542430813d8ca91b8397409954ea92"
    "295a2b55bcc420661783fb865010845d"
)

EXPECTED_FIRST_STEP_MODEL_SHA256 = (
    "42a521f11d8f24e4144d0215d6e1b34d"
    "5f8bf0c2d8848624e4f7c3130699035d"
)

EXPECTED_FIRST_STEP_OPTIMIZER_SHA256 = (
    "5ce2683c21f456b9d5d15eb876b049c5"
    "e6db1215db5a026630f093f7f9d49891"
)

EXPECTED_EPOCH_SEED_REGISTRY_SHA256 = (
    "96a4e2c52526ec7d7ca48d3d7cd1eee3"
    "893b0f8c35df9717df107a874583f956"
)

EXPECTED_PARAMETER_TENSORS = 32
EXPECTED_PARAMETER_COUNT = 19_217_929

REFERENCE_TORCH_VERSION_PREFIX = "2.7.0"


# =============================================================================
# Frozen stream sizes
# =============================================================================

NUM_INVESTORS = 165_975
NUM_STARTUPS = 311_589
NUM_NODES = 477_564
NUM_HISTORY_PERIODS = 60

LATENT_DIM = 40
DESCRIPTION_DIM = 40
TREND_ITEM_DIM = 80
TREND_QUERY_DIM = 80
TREND_DIM = 40
STRUCTURAL_DIM = 40

INVESTOR_SCORING_DIM = 160
STARTUP_SCORING_DIM = 120
PAIR_DIM = 280

BATCH_SIZE = 512
EXAMPLES_PER_EPOCH = 5_366_245
BATCHES_PER_EPOCH = 10_481


# =============================================================================
# Outputs
# =============================================================================

AUDIT_DIR = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_2b"
)

CONTRACT_DIR = Path(
    "data/experimental/phase_5/contracts"
)

ROUNDTRIP_CHECKPOINT_PATH = (
    AUDIT_DIR
    / "epoch0_after_batch0_roundtrip_probe.pt"
)

BATCH1_MANIFEST_PATH = (
    AUDIT_DIR
    / "epoch0_batch1_manifest.parquet"
)

BATCH1_AUDIT_PATH = (
    AUDIT_DIR
    / "epoch0_batch1_integrity_audit.csv"
)

PATH_A_AUDIT_PATH = (
    AUDIT_DIR
    / "uninterrupted_two_batch_path_audit.csv"
)

CHECKPOINT_AUDIT_PATH = (
    AUDIT_DIR
    / "roundtrip_checkpoint_payload_audit.csv"
)

RELOAD_AUDIT_PATH = (
    AUDIT_DIR
    / "roundtrip_checkpoint_reload_audit.csv"
)

PATH_B_AUDIT_PATH = (
    AUDIT_DIR
    / "resumed_two_batch_path_audit.csv"
)

PATH_COMPARISON_PATH = (
    AUDIT_DIR
    / "uninterrupted_vs_resumed_comparison.csv"
)

PARAMETER_COMPARISON_PATH = (
    AUDIT_DIR
    / "post_batch1_parameter_hash_comparison.csv"
)

FINAL_INVARIANT_PATH = (
    AUDIT_DIR
    / "phase_5_3_2b_final_invariants.csv"
)

CONTRACT_PATH = (
    CONTRACT_DIR
    / "phase_5_3_2b_checkpoint_resume_roundtrip_contract.json"
)

DECISION_REGISTER_PATH = (
    CONTRACT_DIR
    / "phase_5_3_2b_checkpoint_resume_decision_register.csv"
)

MANIFEST_PATH = (
    AUDIT_DIR
    / "phase_5_3_2b_checkpoint_resume_roundtrip_manifest.json"
)


# =============================================================================
# Helpers
# =============================================================================

def banner(text: str) -> None:
    print("\n" + "=" * 118)
    print(text)
    print("=" * 118)


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def load_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def file_sha256(path: Path) -> str:
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
        ).encode("utf-8")
    )

    digest.update(
        str(
            tuple(
                value.shape
            )
        ).encode("utf-8")
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


def numpy_rng_state_equal(
    left,
    right,
) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(
            left[1],
            right[1],
        )
        and left[2:] == right[2:]
    )


def rng_snapshot() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
    }


def rng_equal(
    left: dict,
    right: dict,
) -> bool:
    return (
        left["python"]
        == right["python"]
        and numpy_rng_state_equal(
            left["numpy"],
            right["numpy"],
        )
        and torch.equal(
            left["torch"],
            right["torch"],
        )
    )


def restore_rng(
    state: dict,
) -> None:
    random.setstate(
        state["python"]
    )

    np.random.set_state(
        state["numpy"]
    )

    torch.set_rng_state(
        state["torch"]
    )


def dataframe_logical_sha256(
    frame: pd.DataFrame,
    columns: list[str],
) -> str:
    digest = hashlib.sha256()

    for column in columns:
        digest.update(
            column.encode("utf-8")
        )
        digest.update(b"\0")

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
                .astype(str)
                .tolist()
            ):
                digest.update(
                    value.encode("utf-8")
                )
                digest.update(b"\0")

        digest.update(b"\0")

    return digest.hexdigest()


def optimizer_state_logical_sha256(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> str:
    """
    Same logical optimizer-state convention frozen in Phase 5.3.1m.
    """

    digest = hashlib.sha256()

    for (
        name,
        parameter,
    ) in model.named_parameters():

        require(
            parameter in optimizer.state,
            (
                "Missing Adam state for "
                f"{name}."
            ),
        )

        state = optimizer.state[
            parameter
        ]

        digest.update(
            name.encode("utf-8")
        )
        digest.update(b"\0")

        for key in (
            "step",
            "exp_avg",
            "exp_avg_sq",
        ):
            require(
                key in state,
                (
                    f"Adam state for {name} "
                    f"missing {key}."
                ),
            )

            value = state[
                key
            ]

            require(
                isinstance(
                    value,
                    torch.Tensor,
                ),
                (
                    f"Adam state {name}.{key} "
                    "is not Tensor."
                ),
            )

            digest.update(
                key.encode("utf-8")
            )
            digest.update(b"\0")

            digest.update(
                tensor_sha256(
                    value
                ).encode("ascii")
            )
            digest.update(b"\0")

    group = optimizer.param_groups[
        0
    ]

    for key in (
        "lr",
        "betas",
        "eps",
        "weight_decay",
        "amsgrad",
        "foreach",
        "maximize",
        "capturable",
        "differentiable",
        "fused",
    ):
        digest.update(
            key.encode("utf-8")
        )
        digest.update(b"\0")

        digest.update(
            repr(
                group.get(key)
            ).encode("utf-8")
        )
        digest.update(b"\0")

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
            parameter.grad is not None,
            (
                "Cannot hash missing "
                f"gradient: {name}"
            ),
        )

        digest.update(
            name.encode("utf-8")
        )
        digest.update(b"\0")

        digest.update(
            tensor_sha256(
                parameter.grad
            ).encode("ascii")
        )
        digest.update(b"\0")

    return digest.hexdigest()


def clone_model_state_dict(
    model: torch.nn.Module,
):
    return {
        key: (
            value
            .detach()
            .cpu()
            .clone()
        )
        for (
            key,
            value,
        ) in model.state_dict().items()
    }


def clone_optimizer_state_dict(
    optimizer: torch.optim.Optimizer,
):
    return copy.deepcopy(
        optimizer.state_dict()
    )


# =============================================================================
# Load the already-proven Phase-5.3.1l.2b runtime library
# =============================================================================

def load_preflight_runtime():
    require(
        PREFLIGHT_SOURCE_PATH.exists(),
        (
            "Missing frozen preflight source: "
            f"{PREFLIGHT_SOURCE_PATH}"
        ),
    )

    source = PREFLIGHT_SOURCE_PATH.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(
            PREFLIGHT_SOURCE_PATH
        ),
    )

    # Import safety: exactly one normal __main__ guard and no top-level main().
    main_guard_count = 0
    unguarded_main_call_count = 0

    for node in tree.body:
        if isinstance(
            node,
            ast.If,
        ):
            test = node.test

            if (
                isinstance(
                    test,
                    ast.Compare,
                )
                and isinstance(
                    test.left,
                    ast.Name,
                )
                and test.left.id
                == "__name__"
                and len(
                    test.ops
                )
                == 1
                and isinstance(
                    test.ops[0],
                    ast.Eq,
                )
                and len(
                    test.comparators
                )
                == 1
                and isinstance(
                    test.comparators[0],
                    ast.Constant,
                )
                and test.comparators[0].value
                == "__main__"
            ):
                main_guard_count += 1
                continue

        if isinstance(
            node,
            ast.Expr,
        ) and isinstance(
            node.value,
            ast.Call,
        ):
            call = node.value

            if (
                isinstance(
                    call.func,
                    ast.Name,
                )
                and call.func.id
                == "main"
            ):
                unguarded_main_call_count += 1

    require(
        main_guard_count == 1,
        (
            "Expected exactly one __main__ "
            "guard in Phase-5.3.1l.2b."
        ),
    )

    require(
        unguarded_main_call_count == 0,
        (
            "Unguarded main() call detected "
            "in Phase-5.3.1l.2b."
        ),
    )

    module_name = (
        "_itrs_phase5_3_2b_preflight_runtime"
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        PREFLIGHT_SOURCE_PATH,
    )

    require(
        spec is not None
        and spec.loader is not None,
        (
            "Could not create import spec for "
            "Phase-5.3.1l.2b."
        ),
    )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[
        module_name
    ] = module

    spec.loader.exec_module(
        module
    )

    required_symbols = (
        "build_canonical_runtime",
        "build_forward_runtime",
        "compose_canonical_model",
        "build_frozen_adam",
        "positive_stream_logical_sha256",
        "array_logical_sha256",
    )

    for symbol in required_symbols:
        require(
            hasattr(
                module,
                symbol,
            ),
            (
                "Frozen preflight runtime "
                f"missing symbol {symbol}."
            ),
        )

    return module


# =============================================================================
# Construct a fresh canonical training model + Adam
# =============================================================================

def construct_fresh_training_state(
    runtime,
):
    canonical_source = (
        runtime
        .CANONICAL_SOURCE_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    forward_source = (
        runtime
        .FORWARD_SOURCE_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    canonical_tree = ast.parse(
        canonical_source,
        filename=str(
            runtime.CANONICAL_SOURCE_PATH
        ),
    )

    forward_tree = ast.parse(
        forward_source,
        filename=str(
            runtime.FORWARD_SOURCE_PATH
        ),
    )

    (
        canonical_runtime,
        runtime_ast_sha,
    ) = runtime.build_canonical_runtime(
        canonical_tree
    )

    (
        _,
        exact_methods,
        training_adapter,
        adapter_sha,
        removed_guard_sha,
    ) = runtime.build_forward_runtime(
        forward_tree
    )

    (
        model,
        canonical_hash_fn,
    ) = runtime.compose_canonical_model(
        canonical_runtime,
        exact_methods,
        training_adapter,
    )

    model.train()

    optimizer = runtime.build_frozen_adam(
        model
    )

    require(
        canonical_hash_fn(
            model
        )
        == EXPECTED_INITIAL_MODEL_SHA256,
        (
            "Fresh canonical model does not "
            "match frozen initial state."
        ),
    )

    require(
        len(
            optimizer.state
        )
        == 0,
        (
            "Fresh Adam state is not empty."
        ),
    )

    require(
        sum(
            1
            for parameter
            in model.parameters()
            if parameter.requires_grad
        )
        == EXPECTED_PARAMETER_TENSORS,
        (
            "Fresh trainable tensor count drift."
        ),
    )

    require(
        sum(
            int(
                parameter.numel()
            )
            for parameter
            in model.parameters()
            if parameter.requires_grad
        )
        == EXPECTED_PARAMETER_COUNT,
        (
            "Fresh trainable parameter count drift."
        ),
    )

    return (
        model,
        optimizer,
        canonical_hash_fn,
        runtime_ast_sha,
        adapter_sha,
        removed_guard_sha,
    )


# =============================================================================
# Frozen epoch-0 stream
# =============================================================================

def load_epoch0_stream(
    runtime,
):
    positive_order = pd.read_parquet(
        runtime.POSITIVE_ORDER_PATH
    )

    negative_matrix = np.load(
        runtime.NEGATIVE_MATRIX_PATH,
        mmap_mode="r",
    )

    epoch_order = np.load(
        runtime.EPOCH_ORDER_PATH,
        mmap_mode="r",
    )

    first_batch = pd.read_parquet(
        runtime.FIRST_BATCH_PATH
    )

    positive_sha = (
        runtime
        .positive_stream_logical_sha256(
            positive_order
        )
    )

    negative_sha = (
        runtime
        .array_logical_sha256(
            np.asarray(
                negative_matrix
            )
        )
    )

    epoch_order_sha = (
        runtime
        .array_logical_sha256(
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
        first_batch_sha
        == EXPECTED_FIRST_BATCH_SHA256,
        (
            "Frozen epoch-0 first-batch "
            "fingerprint drift."
        ),
    )

    require(
        len(
            epoch_order
        )
        == EXAMPLES_PER_EPOCH,
        (
            "Epoch-0 order length drift."
        ),
    )

    return {
        "positive_order": (
            positive_order
        ),
        "negative_matrix": (
            negative_matrix
        ),
        "epoch_order": (
            epoch_order
        ),
        "positive_sha": (
            positive_sha
        ),
        "negative_sha": (
            negative_sha
        ),
        "epoch_order_sha": (
            epoch_order_sha
        ),
        "first_batch_sha": (
            first_batch_sha
        ),
    }


def decode_batch(
    stream: dict,
    batch_index: int,
) -> pd.DataFrame:
    require(
        0
        <= batch_index
        < BATCHES_PER_EPOCH,
        (
            "Batch index outside epoch."
        ),
    )

    start = (
        batch_index
        * BATCH_SIZE
    )

    end = min(
        start
        + BATCH_SIZE,
        EXAMPLES_PER_EPOCH,
    )

    serialized_indices = (
        stream[
            "epoch_order"
        ][
            start:end
        ]
    )

    rows = []

    for local_position, serialized_value in enumerate(
        serialized_indices
    ):
        serialized_index = int(
            serialized_value
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
            stream[
                "positive_order"
            ].iloc[
                positive_index
            ]
        )

        focal_startup_local = int(
            positive_row[
                "startup_local"
            ]
        )

        if slot == 0:
            label = 1
            startup_local = (
                focal_startup_local
            )
            negative_draw_index = -1

        else:
            label = 0
            negative_draw_index = (
                slot - 1
            )

            startup_local = int(
                stream[
                    "negative_matrix"
                ][
                    positive_index,
                    negative_draw_index,
                ]
            )

        investor_global = int(
            positive_row[
                "investor_global"
            ]
        )

        segment_number = int(
            positive_row[
                "segment_number"
            ]
        )

        startup_global = (
            NUM_INVESTORS
            + startup_local
        )

        rows.append(
            {
                "batch_index": (
                    batch_index
                ),
                "batch_position": (
                    local_position
                ),
                "epoch_example_position": (
                    start
                    + local_position
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
                "negative_draw_index": (
                    negative_draw_index
                ),
                "label": (
                    label
                ),
                "source_interaction_id": (
                    str(
                        positive_row[
                            "interaction_id"
                        ]
                    )
                ),
                "investor_global": (
                    investor_global
                ),
                "startup_local": (
                    startup_local
                ),
                "startup_global": (
                    startup_global
                ),
                "focal_positive_startup_local": (
                    focal_startup_local
                ),
                "segment_number": (
                    segment_number
                ),
                "trend_history_period_count": (
                    segment_number
                ),
            }
        )

    frame = pd.DataFrame(
        rows
    )

    require(
        len(
            frame
        )
        == (
            BATCH_SIZE
            if batch_index
            < (
                BATCHES_PER_EPOCH
                - 1
            )
            else 485
        ),
        (
            f"Decoded batch {batch_index} "
            "size drift."
        ),
    )

    require(
        bool(
            frame[
                "segment_number"
            ]
            .between(
                1,
                59,
            )
            .all()
        ),
        (
            "Decoded training batch contains "
            "target outside T1..T59."
        ),
    )

    return frame


def batch_logical_sha256(
    frame: pd.DataFrame,
) -> str:
    return dataframe_logical_sha256(
        frame,
        columns=[
            "batch_index",
            "batch_position",
            "epoch_example_position",
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


# =============================================================================
# Shared immutable feature runtime
# =============================================================================

def load_shared_inputs(
    runtime,
):
    edge_index_np = np.load(
        runtime.EDGE_INDEX_PATH,
        mmap_mode="r",
    )

    edge_type_np = np.load(
        runtime.EDGE_TYPE_PATH,
        mmap_mode="r",
    )

    doc2vec_all = np.load(
        runtime.DOC2VEC_PATH,
        mmap_mode="r",
    )

    labels_sparse = (
        sparse.load_npz(
            runtime.LABEL_MATRIX_PATH
        )
        .tocsr()
    )

    trend_period_ptr = np.load(
        runtime.TREND_PERIOD_PTR_PATH,
        mmap_mode="r",
    )

    trend_startup_indices = np.load(
        runtime.TREND_STARTUP_INDICES_PATH,
        mmap_mode="r",
    )

    trend_period_counts = np.load(
        runtime.TREND_PERIOD_COUNTS_PATH,
        mmap_mode="r",
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

    return {
        "edge_index": (
            edge_index
        ),
        "edge_type": (
            edge_type
        ),
        "doc2vec_all": (
            doc2vec_all
        ),
        "labels_sparse": (
            labels_sparse
        ),
        "trend_period_ptr": (
            trend_period_ptr
        ),
        "trend_startup_indices": (
            trend_startup_indices
        ),
        "trend_period_counts": (
            trend_period_counts
        ),
    }


# =============================================================================
# Exact generic batch forward / backward / step
# =============================================================================

def execute_training_batch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    canonical_hash_fn,
    batch: pd.DataFrame,
    shared: dict,
) -> dict:
    """
    Execute one exact frozen Phase-5 training batch.

    Atomic order:
        zero_grad(set_to_none=True)
        forward
        BCEWithLogitsLoss
        backward
        optimizer.step

    No checkpointing occurs inside this function.
    """

    require(
        len(
            batch
        )
        > 0,
        "Cannot execute empty batch.",
    )

    batch_size = len(
        batch
    )

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    require(
        all(
            parameter.grad
            is None
            for parameter
            in model.parameters()
        ),
        (
            "zero_grad(set_to_none=True) "
            "did not clear all gradients."
        ),
    )

    pre_step_model_sha = (
        canonical_hash_fn(
            model
        )
    )

    # -------------------------------------------------------------------------
    # Batch arrays.
    # -------------------------------------------------------------------------

    batch_investors = (
        batch[
            "investor_global"
        ].to_numpy(
            dtype=np.int64
        )
    )

    batch_startup_locals = (
        batch[
            "startup_local"
        ].to_numpy(
            dtype=np.int64
        )
    )

    batch_startup_globals = (
        batch[
            "startup_global"
        ].to_numpy(
            dtype=np.int64
        )
    )

    batch_segments = (
        batch[
            "segment_number"
        ].to_numpy(
            dtype=np.int64
        )
    )

    batch_labels = (
        batch[
            "label"
        ].to_numpy(
            dtype=np.int64
        )
    )

    # -------------------------------------------------------------------------
    # Exact history T0..T(h-1).
    # -------------------------------------------------------------------------

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
                shared[
                    "trend_period_ptr"
                ][
                    flattened
                ]
            )

            end = int(
                shared[
                    "trend_period_ptr"
                ][
                    flattened
                    + 1
                ]
            )

            count = int(
                shared[
                    "trend_period_counts"
                ][
                    flattened
                ]
            )

            require(
                end - start
                == count,
                (
                    "Trend CSR mismatch: "
                    f"Investor={investor_global}, "
                    f"period={period}"
                ),
            )

            values = np.array(
                shared[
                    "trend_startup_indices"
                ][
                    start:end
                ],
                dtype=np.int64,
                copy=True,
            )

            if len(
                values
            ) > 0:
                historical_nodes.append(
                    values
                )

            periods.append(
                values
            )

        require(
            len(
                periods
            )
            == h,
            (
                "History length != target h."
            ),
        )

        history_by_key[
            (
                investor_global,
                h,
            )
        ] = periods

    # -------------------------------------------------------------------------
    # Description subset.
    # -------------------------------------------------------------------------

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
        shared[
            "doc2vec_all"
        ][
            required_nodes
        ],
        dtype=np.float32,
        copy=True,
    )

    label_subset_np = (
        shared[
            "labels_sparse"
        ][
            required_nodes
        ]
        .toarray()
        .astype(
            np.float32,
            copy=False,
        )
    )

    doc_subset = torch.from_numpy(
        doc_subset_np
    )

    label_subset = torch.from_numpy(
        label_subset_np
    )

    # -------------------------------------------------------------------------
    # Structural branch.
    # -------------------------------------------------------------------------

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
            shared[
                "edge_index"
            ],
            shared[
                "edge_type"
            ],
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
            "Invalid structural output."
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
            "Structural representation "
            "shape drift."
        ),
    )

    # -------------------------------------------------------------------------
    # Description branch.
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
            "Description representation "
            "shape drift."
        ),
    )

    # -------------------------------------------------------------------------
    # Mixed-h trend branch.
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

    trend_output_by_key = {}

    for h in sorted(
        keys_by_h
    ):
        keys = keys_by_h[
            h
        ]

        sequences = []

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
                )[0]
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
                    "Trend query shape drift."
                ),
            )

            period_vectors = []

            periods = history_by_key[
                (
                    investor_global,
                    h,
                )
            ]

            for startup_globals in periods:
                item_count = len(
                    startup_globals
                )

                if item_count == 0:
                    period_vector = torch.zeros(
                        TREND_ITEM_DIM,
                        dtype=query.dtype,
                        device=query.device,
                    )

                else:
                    startup_locals = (
                        startup_globals
                        - NUM_INVESTORS
                    )

                    startup_tensor = torch.from_numpy(
                        np.array(
                            startup_locals,
                            dtype=np.int64,
                            copy=True,
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

                    subset_positions = torch.from_numpy(
                        np.array(
                            subset_positions_np,
                            dtype=np.int64,
                            copy=True,
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
                        bool(
                            torch.isfinite(
                                alpha
                            ).all()
                        ),
                        (
                            "Non-finite trend "
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
                    "Trend sequence length "
                    "is not exactly h."
                ),
            )

            sequences.append(
                sequence
            )

        grouped_sequence = torch.stack(
            sequences,
            dim=0,
        )

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
            group_gru_output.shape
            == (
                len(
                    keys
                ),
                h,
                TREND_DIM,
            ),
            (
                "Variable-h GRU output "
                "shape drift."
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
                batch_size
            )
        ],
        dim=0,
    )

    # -------------------------------------------------------------------------
    # Pair representation.
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

    investor_description_positions = torch.from_numpy(
        np.array(
            global_to_subset[
                batch_investors
            ],
            dtype=np.int64,
            copy=True,
        )
    )

    startup_description_positions = torch.from_numpy(
        np.array(
            global_to_subset[
                batch_startup_globals
            ],
            dtype=np.int64,
            copy=True,
        )
    )

    L_o_batch = model.investor_embedding(
        investor_tensor
    )

    L_b_batch = model.startup_embedding(
        startup_local_tensor
    )

    F_d_o_batch = description_subset[
        investor_description_positions
    ]

    F_d_b_batch = description_subset[
        startup_description_positions
    ]

    F_s_o_batch = F_s_all[
        investor_tensor
    ]

    F_s_b_batch = F_s_all[
        startup_global_tensor
    ]

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
            batch_size,
            INVESTOR_SCORING_DIM,
        ),
        (
            "Investor scoring representation "
            "shape drift."
        ),
    )

    require(
        startup_representation.shape
        == (
            batch_size,
            STARTUP_SCORING_DIM,
        ),
        (
            "Startup scoring representation "
            "shape drift."
        ),
    )

    require(
        pair_representation.shape
        == (
            batch_size,
            PAIR_DIM,
        ),
        (
            "Pair representation shape drift."
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
            "Invalid scoring output."
        ),
    )

    logits = scoring[
        "logit"
    ]

    require(
        logits.shape
        == (
            batch_size,
            1,
        ),
        (
            "Logit shape drift."
        ),
    )

    targets = torch.from_numpy(
        np.array(
            batch_labels,
            dtype=np.float32,
            copy=True,
        )
    ).reshape(
        batch_size,
        1,
    )

    criterion = torch.nn.BCEWithLogitsLoss()

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
            "Non-finite batch loss."
        ),
    )

    logit_sha = tensor_sha256(
        logits
    )

    # -------------------------------------------------------------------------
    # Backward.
    # -------------------------------------------------------------------------

    loss.backward()

    gradient_sha = (
        gradient_logical_sha256(
            model
        )
    )

    for (
        name,
        parameter,
    ) in model.named_parameters():
        require(
            parameter.grad
            is not None,
            (
                f"Missing gradient: {name}"
            ),
        )

        require(
            bool(
                torch.isfinite(
                    parameter.grad
                ).all()
            ),
            (
                f"Non-finite gradient: {name}"
            ),
        )

    pre_step_rng = rng_snapshot()

    optimizer.step()

    post_step_rng = rng_snapshot()

    require(
        rng_equal(
            pre_step_rng,
            post_step_rng,
        ),
        (
            "Adam step consumed global RNG."
        ),
    )

    post_step_model_sha = (
        canonical_hash_fn(
            model
        )
    )

    optimizer_sha = (
        optimizer_state_logical_sha256(
            model,
            optimizer,
        )
    )

    step_values = []

    for parameter in model.parameters():
        if parameter not in optimizer.state:
            continue

        step_values.append(
            float(
                optimizer.state[
                    parameter
                ][
                    "step"
                ].item()
            )
        )

    require(
        len(
            step_values
        )
        == EXPECTED_PARAMETER_TENSORS,
        (
            "Adam state does not cover "
            "all 32 parameters."
        ),
    )

    require(
        len(
            set(
                step_values
            )
        )
        == 1,
        (
            "Adam parameter step counters "
            "are not synchronized."
        ),
    )

    result = {
        "batch_index": int(
            batch[
                "batch_index"
            ].iloc[0]
        ),
        "batch_size": (
            batch_size
        ),
        "positive_count": int(
            (
                batch[
                    "label"
                ]
                == 1
            ).sum()
        ),
        "negative_count": int(
            (
                batch[
                    "label"
                ]
                == 0
            ).sum()
        ),
        "distinct_target_segments": int(
            batch[
                "segment_number"
            ].nunique()
        ),
        "pre_step_model_sha256": (
            pre_step_model_sha
        ),
        "loss": float(
            loss
            .detach()
            .item()
        ),
        "logit_sha256": (
            logit_sha
        ),
        "gradient_sha256": (
            gradient_sha
        ),
        "post_step_model_sha256": (
            post_step_model_sha
        ),
        "optimizer_state_sha256": (
            optimizer_sha
        ),
        "Adam_state_entries": len(
            optimizer.state
        ),
        "Adam_step_counter": (
            step_values[0]
        ),
    }

    return result


# =============================================================================
# Checkpoint payload
# =============================================================================

def build_checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    execution_contract: dict,
    first_update_contract: dict,
    batch0_result: dict,
    batch0_size: int,
    training_fingerprints: dict,
) -> dict:
    required_fields = set(
        execution_contract[
            "checkpoint"
        ][
            "required_fields"
        ]
    )

    payload = {
        "schema_version": (
            "ITRS_PHASE5_CHECKPOINT_V1"
        ),
        "phase": (
            "5.3.2b_roundtrip_probe"
        ),
        "checkpoint_kind": (
            "latest"
        ),
        "model_state_dict": (
            clone_model_state_dict(
                model
            )
        ),
        "optimizer_state_dict": (
            clone_optimizer_state_dict(
                optimizer
            )
        ),
        "epoch_index": (
            0
        ),
        "next_batch_index": (
            1
        ),
        "global_optimizer_step": (
            1
        ),
        "epoch_loss_weighted_sum": (
            float(
                batch0_result[
                    "loss"
                ]
            )
            * batch0_size
        ),
        "epoch_example_count": (
            batch0_size
        ),
        "validation_pending": (
            False
        ),
        "validation_history": (
            []
        ),
        "best_validation_epoch": (
            None
        ),
        "best_validation_ndcg10": (
            None
        ),
        "best_validation_hr10": (
            None
        ),
        "python_rng_state": (
            random.getstate()
        ),
        "numpy_rng_state": (
            np.random.get_state()
        ),
        "torch_rng_state": (
            torch.get_rng_state().clone()
        ),
        "training_contract_fingerprints": (
            training_fingerprints
        ),
        "training_complete": (
            False
        ),
    }

    actual_fields = set(
        payload.keys()
    )

    require(
        actual_fields
        == required_fields,
        (
            "Round-trip checkpoint payload fields "
            "do not exactly match frozen 5.3.2a schema.\n"
            f"Missing: {sorted(required_fields - actual_fields)}\n"
            f"Extra:   {sorted(actual_fields - required_fields)}"
        ),
    )

    require(
        payload[
            "epoch_index"
        ]
        == 0,
        (
            "Checkpoint epoch_index "
            "must be 0."
        ),
    )

    require(
        payload[
            "next_batch_index"
        ]
        == 1,
        (
            "Checkpoint next_batch_index "
            "must be 1 after batch 0."
        ),
    )

    require(
        payload[
            "global_optimizer_step"
        ]
        == 1,
        (
            "Checkpoint global_optimizer_step "
            "must be 1 after batch 0."
        ),
    )

    require(
        payload[
            "epoch_example_count"
        ]
        == batch0_size,
        (
            "Checkpoint epoch_example_count "
            "drift."
        ),
    )

    require(
        payload[
            "validation_pending"
        ]
        is False,
        (
            "Validation cannot be pending "
            "after only batch 0."
        ),
    )

    require(
        payload[
            "training_complete"
        ]
        is False,
        (
            "Training cannot be complete "
            "after batch 0."
        ),
    )

    # The round-trip probe must start from the already-proven first step.
    require(
        batch0_result[
            "post_step_model_sha256"
        ]
        == EXPECTED_FIRST_STEP_MODEL_SHA256,
        (
            "Checkpoint source model is not "
            "the frozen first-step state."
        ),
    )

    require(
        batch0_result[
            "optimizer_state_sha256"
        ]
        == EXPECTED_FIRST_STEP_OPTIMIZER_SHA256,
        (
            "Checkpoint source optimizer is not "
            "the frozen first-step Adam state."
        ),
    )

    require(
        first_update_contract[
            "first_update"
        ][
            "post_step_parameter_sha256"
        ]
        == EXPECTED_FIRST_STEP_MODEL_SHA256,
        (
            "Phase-5.3.1m contract drift."
        ),
    )

    return payload


def restore_checkpoint(
    runtime,
    checkpoint: dict,
):
    (
        model,
        optimizer,
        canonical_hash_fn,
        runtime_ast_sha,
        adapter_sha,
        removed_guard_sha,
    ) = construct_fresh_training_state(
        runtime
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    restore_rng(
        {
            "python": (
                checkpoint[
                    "python_rng_state"
                ]
            ),
            "numpy": (
                checkpoint[
                    "numpy_rng_state"
                ]
            ),
            "torch": (
                checkpoint[
                    "torch_rng_state"
                ]
            ),
        }
    )

    return (
        model,
        optimizer,
        canonical_hash_fn,
        runtime_ast_sha,
        adapter_sha,
        removed_guard_sha,
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.3.2b — "
        "CHECKPOINT / RESUME ROUND-TRIP NUMERICAL PROOF"
    )

    print(
        "Training paths:                       2"
    )
    print(
        "Path A:                               batch0 -> batch1"
    )
    print(
        "Path B:                               batch0 -> checkpoint -> reload -> batch1"
    )
    print(
        "Validation performed:                 NO"
    )
    print(
        "Test performed:                       NO"
    )
    print(
        "Production checkpoint written:        NO"
    )

    # =========================================================================
    # Prior contract recheck
    # =========================================================================

    banner(
        "AUTHORITATIVE CONTRACT RECHECK"
    )

    for path in (
        PREFLIGHT_SOURCE_PATH,
        PHASE_5_3_1M_CONTRACT_PATH,
        PHASE_5_3_2A_CONTRACT_PATH,
        PHASE_5_3_2A_MANIFEST_PATH,
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

    first_update_contract = load_json(
        PHASE_5_3_1M_CONTRACT_PATH
    )

    execution_contract = load_json(
        PHASE_5_3_2A_CONTRACT_PATH
    )

    execution_manifest = load_json(
        PHASE_5_3_2A_MANIFEST_PATH
    )

    require(
        first_update_contract[
            "status"
        ]
        == "FROZEN",
        (
            "Phase-5.3.1m contract "
            "is not frozen."
        ),
    )

    require(
        execution_contract[
            "status"
        ]
        == "FROZEN",
        (
            "Phase-5.3.2a contract "
            "is not frozen."
        ),
    )

    require(
        execution_manifest[
            "status"
        ]
        == (
            "FULL_TRAINING_EXECUTION_STATE_"
            "CONTRACT_FROZEN"
        ),
        (
            "Unexpected Phase-5.3.2a "
            "manifest status."
        ),
    )

    require(
        execution_manifest[
            "epoch_seed_registry_sha256"
        ]
        == EXPECTED_EPOCH_SEED_REGISTRY_SHA256,
        (
            "20-epoch seed registry "
            "fingerprint drift."
        ),
    )

    require(
        execution_contract[
            "training_control"
        ][
            "batches_per_epoch"
        ]
        == BATCHES_PER_EPOCH,
        (
            "Batches-per-epoch contract drift."
        ),
    )

    require(
        execution_contract[
            "resume"
        ][
            "repeat_completed_batch"
        ]
        is False,
        (
            "Resume contract unexpectedly "
            "allows repeated completed batch."
        ),
    )

    require(
        execution_contract[
            "batch_atomicity"
        ][
            "next_batch_index_semantics"
        ]
        == "next batch not yet executed",
        (
            "next_batch_index semantics drift."
        ),
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

    print(
        "Phase-5.3.1m first update:            FROZEN / PASS"
    )
    print(
        "Phase-5.3.2a execution contract:      FROZEN / PASS"
    )
    print(
        "20-epoch seed registry:               VERIFIED"
    )

    # =========================================================================
    # Load frozen runtime / stream / immutable inputs
    # =========================================================================

    banner(
        "LOAD FROZEN TRAINING RUNTIME"
    )

    runtime = load_preflight_runtime()

    stream = load_epoch0_stream(
        runtime
    )

    shared = load_shared_inputs(
        runtime
    )

    batch0 = decode_batch(
        stream,
        batch_index=0,
    )

    batch1 = decode_batch(
        stream,
        batch_index=1,
    )

    batch0_sha = (
        batch_logical_sha256(
            batch0
        )
    )

    batch1_sha = (
        batch_logical_sha256(
            batch1
        )
    )

    # Batch-0 decode must match the already-frozen first batch at the
    # semantic core. Compare against the stored first batch separately.
    stored_first_batch = pd.read_parquet(
        runtime.FIRST_BATCH_PATH
    )

    stored_core = stored_first_batch[
        [
            "batch_position",
            "serialized_example_index",
            "positive_order_index",
            "example_slot",
            "label",
            "source_interaction_id",
            "investor_global",
            "startup_local",
            "segment_number",
        ]
    ].reset_index(
        drop=True
    )

    decoded_core = batch0[
        [
            "batch_position",
            "serialized_example_index",
            "positive_order_index",
            "example_slot",
            "label",
            "source_interaction_id",
            "investor_global",
            "startup_local",
            "segment_number",
        ]
    ].reset_index(
        drop=True
    )

    require(
        stored_core.equals(
            decoded_core
        ),
        (
            "Generic batch decoder does not "
            "reproduce frozen batch 0."
        ),
    )

    batch1.to_parquet(
        BATCH1_MANIFEST_PATH.parent
        / "__deferred_batch1_manifest.parquet",
        index=False,
    ) if False else None

    batch1_audit_df = pd.DataFrame(
        [
            {
                "metric": (
                    "batch_index"
                ),
                "value": (
                    1
                ),
            },
            {
                "metric": (
                    "batch_size"
                ),
                "value": (
                    len(
                        batch1
                    )
                ),
            },
            {
                "metric": (
                    "positive_count"
                ),
                "value": (
                    int(
                        (
                            batch1[
                                "label"
                            ]
                            == 1
                        ).sum()
                    )
                ),
            },
            {
                "metric": (
                    "negative_count"
                ),
                "value": (
                    int(
                        (
                            batch1[
                                "label"
                            ]
                            == 0
                        ).sum()
                    )
                ),
            },
            {
                "metric": (
                    "distinct_target_segments"
                ),
                "value": (
                    int(
                        batch1[
                            "segment_number"
                        ].nunique()
                    )
                ),
            },
            {
                "metric": (
                    "minimum_target_segment"
                ),
                "value": (
                    int(
                        batch1[
                            "segment_number"
                        ].min()
                    )
                ),
            },
            {
                "metric": (
                    "maximum_target_segment"
                ),
                "value": (
                    int(
                        batch1[
                            "segment_number"
                        ].max()
                    )
                ),
            },
            {
                "metric": (
                    "batch1_logical_sha256"
                ),
                "value": (
                    batch1_sha
                ),
            },
        ]
    )

    print(
        batch1_audit_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # PATH A — uninterrupted batch 0 -> batch 1
    # =========================================================================

    banner(
        "PATH A — UNINTERRUPTED BATCH 0 -> BATCH 1"
    )

    (
        model_a,
        optimizer_a,
        hash_fn_a,
        runtime_ast_a,
        adapter_sha_a,
        removed_guard_sha_a,
    ) = construct_fresh_training_state(
        runtime
    )

    path_a_rng_before = rng_snapshot()

    a_batch0 = execute_training_batch(
        model_a,
        optimizer_a,
        hash_fn_a,
        batch0,
        shared,
    )

    require(
        a_batch0[
            "logit_sha256"
        ]
        == EXPECTED_FIRST_BATCH_LOGIT_SHA256,
        (
            "Path A batch-0 logit "
            "fingerprint drift."
        ),
    )

    require(
        a_batch0[
            "gradient_sha256"
        ]
        == EXPECTED_FIRST_BATCH_GRADIENT_SHA256,
        (
            "Path A batch-0 gradient "
            "fingerprint drift."
        ),
    )

    require(
        a_batch0[
            "post_step_model_sha256"
        ]
        == EXPECTED_FIRST_STEP_MODEL_SHA256,
        (
            "Path A first-step model "
            "fingerprint drift."
        ),
    )

    require(
        a_batch0[
            "optimizer_state_sha256"
        ]
        == EXPECTED_FIRST_STEP_OPTIMIZER_SHA256,
        (
            "Path A first-step optimizer "
            "fingerprint drift."
        ),
    )

    a_batch1 = execute_training_batch(
        model_a,
        optimizer_a,
        hash_fn_a,
        batch1,
        shared,
    )

    path_a_rng_after = rng_snapshot()

    require(
        rng_equal(
            path_a_rng_before,
            path_a_rng_after,
        ),
        (
            "Two uninterrupted training batches "
            "changed global RNG state."
        ),
    )

    require(
        a_batch1[
            "Adam_step_counter"
        ]
        == 2.0,
        (
            "Path A Adam step counter "
            "after batch 1 is not 2."
        ),
    )

    path_a_parameter_hashes = (
        parameter_hashes(
            model_a
        )
    )

    path_a_model_sha = (
        a_batch1[
            "post_step_model_sha256"
        ]
    )

    path_a_optimizer_sha = (
        a_batch1[
            "optimizer_state_sha256"
        ]
    )

    path_a_df = pd.DataFrame(
        [
            {
                "metric": (
                    "batch0_loss"
                ),
                "value": (
                    a_batch0[
                        "loss"
                    ]
                ),
            },
            {
                "metric": (
                    "batch0_logit_sha256"
                ),
                "value": (
                    a_batch0[
                        "logit_sha256"
                    ]
                ),
            },
            {
                "metric": (
                    "batch0_gradient_sha256"
                ),
                "value": (
                    a_batch0[
                        "gradient_sha256"
                    ]
                ),
            },
            {
                "metric": (
                    "batch0_post_step_model_sha256"
                ),
                "value": (
                    a_batch0[
                        "post_step_model_sha256"
                    ]
                ),
            },
            {
                "metric": (
                    "batch0_optimizer_sha256"
                ),
                "value": (
                    a_batch0[
                        "optimizer_state_sha256"
                    ]
                ),
            },
            {
                "metric": (
                    "batch1_loss"
                ),
                "value": (
                    a_batch1[
                        "loss"
                    ]
                ),
            },
            {
                "metric": (
                    "batch1_logit_sha256"
                ),
                "value": (
                    a_batch1[
                        "logit_sha256"
                    ]
                ),
            },
            {
                "metric": (
                    "batch1_gradient_sha256"
                ),
                "value": (
                    a_batch1[
                        "gradient_sha256"
                    ]
                ),
            },
            {
                "metric": (
                    "post_batch1_model_sha256"
                ),
                "value": (
                    path_a_model_sha
                ),
            },
            {
                "metric": (
                    "post_batch1_optimizer_sha256"
                ),
                "value": (
                    path_a_optimizer_sha
                ),
            },
            {
                "metric": (
                    "Adam_step_counter"
                ),
                "value": (
                    a_batch1[
                        "Adam_step_counter"
                    ]
                ),
            },
        ]
    )

    print(
        path_a_df.to_string(
            index=False
        )
    )

    # Keep only fingerprints; free the actual Path-A neural state.
    del model_a
    del optimizer_a
    gc.collect()

    # =========================================================================
    # PATH B — batch 0 -> checkpoint -> reload -> batch 1
    # =========================================================================

    banner(
        "PATH B — BATCH 0 -> CHECKPOINT"
    )

    (
        model_b,
        optimizer_b,
        hash_fn_b,
        runtime_ast_b,
        adapter_sha_b,
        removed_guard_sha_b,
    ) = construct_fresh_training_state(
        runtime
    )

    b_batch0 = execute_training_batch(
        model_b,
        optimizer_b,
        hash_fn_b,
        batch0,
        shared,
    )

    require(
        b_batch0[
            "logit_sha256"
        ]
        == EXPECTED_FIRST_BATCH_LOGIT_SHA256,
        (
            "Path B batch-0 logit "
            "fingerprint drift."
        ),
    )

    require(
        b_batch0[
            "gradient_sha256"
        ]
        == EXPECTED_FIRST_BATCH_GRADIENT_SHA256,
        (
            "Path B batch-0 gradient "
            "fingerprint drift."
        ),
    )

    require(
        b_batch0[
            "post_step_model_sha256"
        ]
        == EXPECTED_FIRST_STEP_MODEL_SHA256,
        (
            "Path B checkpoint source "
            "model fingerprint drift."
        ),
    )

    require(
        b_batch0[
            "optimizer_state_sha256"
        ]
        == EXPECTED_FIRST_STEP_OPTIMIZER_SHA256,
        (
            "Path B checkpoint source "
            "optimizer fingerprint drift."
        ),
    )

    training_fingerprints = {
        "canonical_initial_model_sha256": (
            EXPECTED_INITIAL_MODEL_SHA256
        ),
        "epoch0_positive_order_sha256": (
            stream[
                "positive_sha"
            ]
        ),
        "epoch0_negative_matrix_sha256": (
            stream[
                "negative_sha"
            ]
        ),
        "epoch0_order_sha256": (
            stream[
                "epoch_order_sha"
            ]
        ),
        "epoch0_first_batch_sha256": (
            EXPECTED_FIRST_BATCH_SHA256
        ),
        "epoch0_batch1_sha256": (
            batch1_sha
        ),
        "first_step_model_sha256": (
            EXPECTED_FIRST_STEP_MODEL_SHA256
        ),
        "first_step_optimizer_sha256": (
            EXPECTED_FIRST_STEP_OPTIMIZER_SHA256
        ),
        "epoch_seed_registry_sha256": (
            EXPECTED_EPOCH_SEED_REGISTRY_SHA256
        ),
        "phase_5_3_2a_contract_file_sha256": (
            file_sha256(
                PHASE_5_3_2A_CONTRACT_PATH
            )
        ),
    }

    checkpoint_payload = build_checkpoint_payload(
        model=model_b,
        optimizer=optimizer_b,
        execution_contract=execution_contract,
        first_update_contract=first_update_contract,
        batch0_result=b_batch0,
        batch0_size=len(
            batch0
        ),
        training_fingerprints=(
            training_fingerprints
        ),
    )

    checkpoint_rng_expected = {
        "python": (
            checkpoint_payload[
                "python_rng_state"
            ]
        ),
        "numpy": (
            checkpoint_payload[
                "numpy_rng_state"
            ]
        ),
        "torch": (
            checkpoint_payload[
                "torch_rng_state"
            ]
        ),
    }

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        checkpoint_payload,
        ROUNDTRIP_CHECKPOINT_PATH,
    )

    checkpoint_file_sha = (
        file_sha256(
            ROUNDTRIP_CHECKPOINT_PATH
        )
    )

    checkpoint_df = pd.DataFrame(
        [
            {
                "check": (
                    "required_field_count"
                ),
                "actual": (
                    len(
                        checkpoint_payload
                    )
                ),
                "expected": (
                    len(
                        execution_contract[
                            "checkpoint"
                        ][
                            "required_fields"
                        ]
                    )
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "check": (
                    "epoch_index"
                ),
                "actual": (
                    checkpoint_payload[
                        "epoch_index"
                    ]
                ),
                "expected": (
                    0
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "check": (
                    "next_batch_index"
                ),
                "actual": (
                    checkpoint_payload[
                        "next_batch_index"
                    ]
                ),
                "expected": (
                    1
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "check": (
                    "global_optimizer_step"
                ),
                "actual": (
                    checkpoint_payload[
                        "global_optimizer_step"
                    ]
                ),
                "expected": (
                    1
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "check": (
                    "epoch_example_count"
                ),
                "actual": (
                    checkpoint_payload[
                        "epoch_example_count"
                    ]
                ),
                "expected": (
                    512
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "check": (
                    "checkpoint_file_sha256"
                ),
                "actual": (
                    checkpoint_file_sha
                ),
                "expected": (
                    "RECORDED_NOT_CROSS_RUN_FROZEN"
                ),
                "status": (
                    "PASS"
                ),
            },
        ]
    )

    print(
        checkpoint_df.to_string(
            index=False
        )
    )

    # Destroy source state. The resumed path must come from disk.
    del checkpoint_payload
    del model_b
    del optimizer_b
    gc.collect()

    # Perturb global RNGs deliberately; checkpoint restore must overwrite them.
    random.random()
    np.random.random()
    torch.rand(
        1
    )

    banner(
        "PATH B — RELOAD CHECKPOINT INTO FRESH MODEL / ADAM"
    )

    loaded_checkpoint = torch.load(
        ROUNDTRIP_CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    require(
        set(
            loaded_checkpoint.keys()
        )
        == set(
            execution_contract[
                "checkpoint"
            ][
                "required_fields"
            ]
        ),
        (
            "Reloaded checkpoint schema "
            "does not match frozen contract."
        ),
    )

    (
        model_r,
        optimizer_r,
        hash_fn_r,
        runtime_ast_r,
        adapter_sha_r,
        removed_guard_sha_r,
    ) = restore_checkpoint(
        runtime,
        loaded_checkpoint,
    )

    reload_model_sha = (
        hash_fn_r(
            model_r
        )
    )

    reload_optimizer_sha = (
        optimizer_state_logical_sha256(
            model_r,
            optimizer_r,
        )
    )

    reload_rng = rng_snapshot()

    require(
        reload_model_sha
        == EXPECTED_FIRST_STEP_MODEL_SHA256,
        (
            "Reloaded model does not match "
            "frozen first-step state."
        ),
    )

    require(
        reload_optimizer_sha
        == EXPECTED_FIRST_STEP_OPTIMIZER_SHA256,
        (
            "Reloaded optimizer does not match "
            "frozen first-step state."
        ),
    )

    require(
        rng_equal(
            reload_rng,
            checkpoint_rng_expected,
        ),
        (
            "Checkpoint RNG states were not "
            "restored exactly."
        ),
    )

    require(
        int(
            loaded_checkpoint[
                "epoch_index"
            ]
        )
        == 0,
        (
            "Reloaded epoch_index drift."
        ),
    )

    require(
        int(
            loaded_checkpoint[
                "next_batch_index"
            ]
        )
        == 1,
        (
            "Reloaded next_batch_index drift."
        ),
    )

    require(
        int(
            loaded_checkpoint[
                "global_optimizer_step"
            ]
        )
        == 1,
        (
            "Reloaded optimizer-step counter drift."
        ),
    )

    reload_df = pd.DataFrame(
        [
            {
                "check": (
                    "reloaded_model_sha256"
                ),
                "actual": (
                    reload_model_sha
                ),
                "expected": (
                    EXPECTED_FIRST_STEP_MODEL_SHA256
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "check": (
                    "reloaded_optimizer_sha256"
                ),
                "actual": (
                    reload_optimizer_sha
                ),
                "expected": (
                    EXPECTED_FIRST_STEP_OPTIMIZER_SHA256
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "check": (
                    "RNG_states_restored"
                ),
                "actual": (
                    "True"
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
                    "next_batch_index"
                ),
                "actual": (
                    loaded_checkpoint[
                        "next_batch_index"
                    ]
                ),
                "expected": (
                    1
                ),
                "status": (
                    "PASS"
                ),
            },
        ]
    )

    print(
        reload_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Resume at exactly batch 1.
    # =========================================================================

    banner(
        "PATH B — RESUME EXACTLY AT BATCH 1"
    )

    require(
        int(
            loaded_checkpoint[
                "next_batch_index"
            ]
        )
        == int(
            batch1[
                "batch_index"
            ].iloc[0]
        ),
        (
            "Resume is not starting at "
            "checkpoint next_batch_index."
        ),
    )

    b_batch1 = execute_training_batch(
        model_r,
        optimizer_r,
        hash_fn_r,
        batch1,
        shared,
    )

    require(
        b_batch1[
            "Adam_step_counter"
        ]
        == 2.0,
        (
            "Resumed Adam step counter "
            "after batch 1 is not 2."
        ),
    )

    path_b_parameter_hashes = (
        parameter_hashes(
            model_r
        )
    )

    path_b_model_sha = (
        b_batch1[
            "post_step_model_sha256"
        ]
    )

    path_b_optimizer_sha = (
        b_batch1[
            "optimizer_state_sha256"
        ]
    )

    path_b_df = pd.DataFrame(
        [
            {
                "metric": (
                    "batch1_loss"
                ),
                "value": (
                    b_batch1[
                        "loss"
                    ]
                ),
            },
            {
                "metric": (
                    "batch1_logit_sha256"
                ),
                "value": (
                    b_batch1[
                        "logit_sha256"
                    ]
                ),
            },
            {
                "metric": (
                    "batch1_gradient_sha256"
                ),
                "value": (
                    b_batch1[
                        "gradient_sha256"
                    ]
                ),
            },
            {
                "metric": (
                    "post_batch1_model_sha256"
                ),
                "value": (
                    path_b_model_sha
                ),
            },
            {
                "metric": (
                    "post_batch1_optimizer_sha256"
                ),
                "value": (
                    path_b_optimizer_sha
                ),
            },
            {
                "metric": (
                    "Adam_step_counter"
                ),
                "value": (
                    b_batch1[
                        "Adam_step_counter"
                    ]
                ),
            },
        ]
    )

    print(
        path_b_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Exact uninterrupted-vs-resumed comparison
    # =========================================================================

    banner(
        "UNINTERRUPTED vs RESUMED EXACT COMPARISON"
    )

    require(
        a_batch1[
            "logit_sha256"
        ]
        == b_batch1[
            "logit_sha256"
        ],
        (
            "Batch-1 logits differ between "
            "uninterrupted and resumed paths."
        ),
    )

    require(
        a_batch1[
            "gradient_sha256"
        ]
        == b_batch1[
            "gradient_sha256"
        ],
        (
            "Batch-1 gradients differ between "
            "uninterrupted and resumed paths."
        ),
    )

    require(
        a_batch1[
            "loss"
        ]
        == b_batch1[
            "loss"
        ],
        (
            "Batch-1 BCE differs between "
            "uninterrupted and resumed paths."
        ),
    )

    require(
        path_a_model_sha
        == path_b_model_sha,
        (
            "Post-batch-1 model SHA differs "
            "between uninterrupted and resumed paths."
        ),
    )

    require(
        path_a_optimizer_sha
        == path_b_optimizer_sha,
        (
            "Post-batch-1 Adam SHA differs "
            "between uninterrupted and resumed paths."
        ),
    )

    require(
        path_a_parameter_hashes
        == path_b_parameter_hashes,
        (
            "At least one post-batch-1 parameter "
            "tensor differs after resume."
        ),
    )

    parameter_rows = []

    for name in sorted(
        path_a_parameter_hashes
    ):
        left = (
            path_a_parameter_hashes[
                name
            ]
        )

        right = (
            path_b_parameter_hashes[
                name
            ]
        )

        parameter_rows.append(
            {
                "parameter": (
                    name
                ),
                "uninterrupted_sha256": (
                    left
                ),
                "resumed_sha256": (
                    right
                ),
                "exact_match": (
                    left == right
                ),
                "status": (
                    "PASS"
                    if left == right
                    else "FAIL"
                ),
            }
        )

    parameter_comparison_df = pd.DataFrame(
        parameter_rows
    )

    require(
        len(
            parameter_comparison_df
        )
        == EXPECTED_PARAMETER_TENSORS,
        (
            "Parameter comparison does not "
            "cover all 32 tensors."
        ),
    )

    require(
        bool(
            parameter_comparison_df[
                "exact_match"
            ].all()
        ),
        (
            "Post-resume parameter "
            "comparison failed."
        ),
    )

    comparison_df = pd.DataFrame(
        [
            {
                "check": (
                    "batch1_loss_exact"
                ),
                "uninterrupted": (
                    a_batch1[
                        "loss"
                    ]
                ),
                "resumed": (
                    b_batch1[
                        "loss"
                    ]
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "check": (
                    "batch1_logit_sha256"
                ),
                "uninterrupted": (
                    a_batch1[
                        "logit_sha256"
                    ]
                ),
                "resumed": (
                    b_batch1[
                        "logit_sha256"
                    ]
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "check": (
                    "batch1_gradient_sha256"
                ),
                "uninterrupted": (
                    a_batch1[
                        "gradient_sha256"
                    ]
                ),
                "resumed": (
                    b_batch1[
                        "gradient_sha256"
                    ]
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "check": (
                    "post_batch1_model_sha256"
                ),
                "uninterrupted": (
                    path_a_model_sha
                ),
                "resumed": (
                    path_b_model_sha
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "check": (
                    "post_batch1_optimizer_sha256"
                ),
                "uninterrupted": (
                    path_a_optimizer_sha
                ),
                "resumed": (
                    path_b_optimizer_sha
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "check": (
                    "all_32_parameter_hashes"
                ),
                "uninterrupted": (
                    "32"
                ),
                "resumed": (
                    "32"
                ),
                "status": (
                    "PASS"
                ),
            },
        ]
    )

    print(
        comparison_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Final invariants
    # =========================================================================

    banner(
        "FINAL PHASE-5.3.2b INVARIANTS"
    )

    checks = [
        (
            "phase_5_3_2a_contract_frozen",
            (
                execution_contract[
                    "status"
                ]
                == "FROZEN"
            ),
        ),
        (
            "epoch_seed_registry_hash_exact",
            (
                execution_manifest[
                    "epoch_seed_registry_sha256"
                ]
                == EXPECTED_EPOCH_SEED_REGISTRY_SHA256
            ),
        ),
        (
            "generic_batch0_decode_matches_frozen_batch0",
            stored_core.equals(
                decoded_core
            ),
        ),
        (
            "path_A_batch0_logit_exact",
            (
                a_batch0[
                    "logit_sha256"
                ]
                == EXPECTED_FIRST_BATCH_LOGIT_SHA256
            ),
        ),
        (
            "path_A_batch0_gradient_exact",
            (
                a_batch0[
                    "gradient_sha256"
                ]
                == EXPECTED_FIRST_BATCH_GRADIENT_SHA256
            ),
        ),
        (
            "path_A_first_step_model_exact",
            (
                a_batch0[
                    "post_step_model_sha256"
                ]
                == EXPECTED_FIRST_STEP_MODEL_SHA256
            ),
        ),
        (
            "path_A_first_step_optimizer_exact",
            (
                a_batch0[
                    "optimizer_state_sha256"
                ]
                == EXPECTED_FIRST_STEP_OPTIMIZER_SHA256
            ),
        ),
        (
            "path_B_checkpoint_source_model_exact",
            (
                b_batch0[
                    "post_step_model_sha256"
                ]
                == EXPECTED_FIRST_STEP_MODEL_SHA256
            ),
        ),
        (
            "path_B_checkpoint_source_optimizer_exact",
            (
                b_batch0[
                    "optimizer_state_sha256"
                ]
                == EXPECTED_FIRST_STEP_OPTIMIZER_SHA256
            ),
        ),
        (
            "checkpoint_fields_exact",
            (
                set(
                    loaded_checkpoint.keys()
                )
                == set(
                    execution_contract[
                        "checkpoint"
                    ][
                        "required_fields"
                    ]
                )
            ),
        ),
        (
            "checkpoint_epoch_index_zero",
            (
                int(
                    loaded_checkpoint[
                        "epoch_index"
                    ]
                )
                == 0
            ),
        ),
        (
            "checkpoint_next_batch_index_one",
            (
                int(
                    loaded_checkpoint[
                        "next_batch_index"
                    ]
                )
                == 1
            ),
        ),
        (
            "checkpoint_global_step_one",
            (
                int(
                    loaded_checkpoint[
                        "global_optimizer_step"
                    ]
                )
                == 1
            ),
        ),
        (
            "checkpoint_reload_model_exact",
            (
                reload_model_sha
                == EXPECTED_FIRST_STEP_MODEL_SHA256
            ),
        ),
        (
            "checkpoint_reload_optimizer_exact",
            (
                reload_optimizer_sha
                == EXPECTED_FIRST_STEP_OPTIMIZER_SHA256
            ),
        ),
        (
            "checkpoint_reload_RNG_exact",
            rng_equal(
                reload_rng,
                checkpoint_rng_expected,
            ),
        ),
        (
            "resume_starts_at_next_batch_index",
            (
                int(
                    loaded_checkpoint[
                        "next_batch_index"
                    ]
                )
                == 1
            ),
        ),
        (
            "batch1_loss_exact_after_resume",
            (
                a_batch1[
                    "loss"
                ]
                == b_batch1[
                    "loss"
                ]
            ),
        ),
        (
            "batch1_logits_exact_after_resume",
            (
                a_batch1[
                    "logit_sha256"
                ]
                == b_batch1[
                    "logit_sha256"
                ]
            ),
        ),
        (
            "batch1_gradients_exact_after_resume",
            (
                a_batch1[
                    "gradient_sha256"
                ]
                == b_batch1[
                    "gradient_sha256"
                ]
            ),
        ),
        (
            "post_batch1_model_exact_after_resume",
            (
                path_a_model_sha
                == path_b_model_sha
            ),
        ),
        (
            "post_batch1_optimizer_exact_after_resume",
            (
                path_a_optimizer_sha
                == path_b_optimizer_sha
            ),
        ),
        (
            "all_32_parameter_tensors_exact_after_resume",
            (
                path_a_parameter_hashes
                == path_b_parameter_hashes
            ),
        ),
        (
            "path_A_Adam_step_counter_two",
            (
                a_batch1[
                    "Adam_step_counter"
                ]
                == 2.0
            ),
        ),
        (
            "path_B_Adam_step_counter_two",
            (
                b_batch1[
                    "Adam_step_counter"
                ]
                == 2.0
            ),
        ),
        (
            "no_validation_performed",
            True,
        ),
        (
            "no_test_performed",
            True,
        ),
        (
            "no_production_checkpoint_written",
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
            "At least one Phase-5.3.2b "
            "checkpoint/resume invariant failed."
        ),
    )

    print(
        invariant_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Write outputs
    # =========================================================================

    banner(
        "WRITE PHASE-5.3.2b ROUND-TRIP OUTPUTS"
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTRACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    batch1.to_parquet(
        BATCH1_MANIFEST_PATH,
        index=False,
    )

    batch1_audit_df.to_csv(
        BATCH1_AUDIT_PATH,
        index=False,
    )

    path_a_df.to_csv(
        PATH_A_AUDIT_PATH,
        index=False,
    )

    checkpoint_df.to_csv(
        CHECKPOINT_AUDIT_PATH,
        index=False,
    )

    reload_df.to_csv(
        RELOAD_AUDIT_PATH,
        index=False,
    )

    path_b_df.to_csv(
        PATH_B_AUDIT_PATH,
        index=False,
    )

    comparison_df.to_csv(
        PATH_COMPARISON_PATH,
        index=False,
    )

    parameter_comparison_df.to_csv(
        PARAMETER_COMPARISON_PATH,
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
                    "checkpoint_roundtrip_proof"
                ),
                "value": (
                    "UNINTERRUPTED_BATCH0_BATCH1_EQUALS_"
                    "BATCH0_SAVE_RELOAD_BATCH1"
                ),
                "classification": (
                    "EVALUATION_INTEGRITY_GUARD"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_2b"
                ),
            },
            {
                "decision": (
                    "resume_batch_semantics"
                ),
                "value": (
                    "NEXT_BATCH_INDEX_IS_FIRST_NOT_YET_EXECUTED"
                ),
                "classification": (
                    "INHERITED_FROZEN_PHASE_5_3_2a"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_2b"
                ),
            },
            {
                "decision": (
                    "checkpoint_serialization_runtime"
                ),
                "value": (
                    "TORCH_SAVE_TORCH_LOAD_WEIGHTS_ONLY_FALSE_CPU"
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_2b"
                ),
            },
            {
                "decision": (
                    "checkpoint_RNG_restore"
                ),
                "value": (
                    "PYTHON_NUMPY_TORCH_CPU_EXACT"
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_2b"
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
            "5.3.2b"
        ),
        "title": (
            "Checkpoint / Resume Round-Trip Numerical Proof Contract"
        ),
        "status": (
            "FROZEN"
        ),
        "checkpoint_source_state": {
            "epoch_index": (
                0
            ),
            "next_batch_index": (
                1
            ),
            "global_optimizer_step": (
                1
            ),
            "model_sha256": (
                EXPECTED_FIRST_STEP_MODEL_SHA256
            ),
            "optimizer_sha256": (
                EXPECTED_FIRST_STEP_OPTIMIZER_SHA256
            ),
        },
        "batch1": {
            "logical_sha256": (
                batch1_sha
            ),
            "batch_size": (
                len(
                    batch1
                )
            ),
            "positive_count": int(
                (
                    batch1[
                        "label"
                    ]
                    == 1
                ).sum()
            ),
            "negative_count": int(
                (
                    batch1[
                        "label"
                    ]
                    == 0
                ).sum()
            ),
            "distinct_target_segments": int(
                batch1[
                    "segment_number"
                ].nunique()
            ),
            "loss": (
                a_batch1[
                    "loss"
                ]
            ),
            "logit_sha256": (
                a_batch1[
                    "logit_sha256"
                ]
            ),
            "gradient_sha256": (
                a_batch1[
                    "gradient_sha256"
                ]
            ),
        },
        "after_two_optimizer_steps": {
            "model_sha256": (
                path_a_model_sha
            ),
            "optimizer_sha256": (
                path_a_optimizer_sha
            ),
            "Adam_step_counter": (
                2
            ),
            "parameter_tensors": (
                EXPECTED_PARAMETER_TENSORS
            ),
        },
        "roundtrip": {
            "checkpoint_file": (
                str(
                    ROUNDTRIP_CHECKPOINT_PATH
                )
            ),
            "checkpoint_file_sha256": (
                checkpoint_file_sha
            ),
            "checkpoint_physical_sha_cross_run_frozen": (
                False
            ),
            "required_payload_schema_exact": (
                True
            ),
            "RNG_restored_exactly": (
                True
            ),
            "batch1_loss_exact": (
                True
            ),
            "batch1_logits_exact": (
                True
            ),
            "batch1_gradients_exact": (
                True
            ),
            "post_batch1_model_exact": (
                True
            ),
            "post_batch1_optimizer_exact": (
                True
            ),
            "all_32_parameter_tensors_exact": (
                True
            ),
        },
        "boundary": {
            "path_A_optimizer_steps": (
                2
            ),
            "path_B_steps_before_checkpoint": (
                1
            ),
            "path_B_steps_after_reload": (
                1
            ),
            "validation_performed": (
                False
            ),
            "test_performed": (
                False
            ),
            "production_checkpoint_written": (
                False
            ),
        },
        "next_phase": {
            "id": (
                "5.3.2c"
            ),
            "title": (
                "Production Training Driver Dry-Run and Freeze"
            ),
            "requirement": (
                "Build the resumable 20-epoch production driver using "
                "the now-proven checkpoint/resume mechanism. Execute a "
                "bounded dry-run that exercises checkpoint write/reload, "
                "batch counters, epoch state transitions, and validation "
                "entry without launching the full 20-epoch run."
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
            "5.3.2b"
        ),
        "status": (
            "CHECKPOINT_RESUME_ROUNDTRIP_"
            "NUMERICALLY_EXACT_AND_FROZEN"
        ),
        "batch1_logical_sha256": (
            batch1_sha
        ),
        "batch1_logit_sha256": (
            a_batch1[
                "logit_sha256"
            ]
        ),
        "batch1_gradient_sha256": (
            a_batch1[
                "gradient_sha256"
            ]
        ),
        "post_two_step_model_sha256": (
            path_a_model_sha
        ),
        "post_two_step_optimizer_sha256": (
            path_a_optimizer_sha
        ),
        "checkpoint_file_sha256": (
            checkpoint_file_sha
        ),
        "parameter_tensors_exact_after_resume": (
            EXPECTED_PARAMETER_TENSORS
        ),
        "validation_performed": (
            False
        ),
        "test_performed": (
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
        BATCH1_MANIFEST_PATH,
        BATCH1_AUDIT_PATH,
        PATH_A_AUDIT_PATH,
        CHECKPOINT_AUDIT_PATH,
        RELOAD_AUDIT_PATH,
        PATH_B_AUDIT_PATH,
        PATH_COMPARISON_PATH,
        PARAMETER_COMPARISON_PATH,
        FINAL_INVARIANT_PATH,
        ROUNDTRIP_CHECKPOINT_PATH,
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
        "PHASE 5.3.2b FINAL STATUS"
    )

    print(
        "Path A:"
    )
    print(
        "  initial -> batch 0 -> batch 1"
    )
    print()
    print(
        "Path B:"
    )
    print(
        "  initial -> batch 0 -> checkpoint -> reload -> batch 1"
    )
    print()

    print(
        "Checkpoint counters:"
    )
    print(
        "  epoch_index:                        0"
    )
    print(
        "  next_batch_index:                   1"
    )
    print(
        "  global_optimizer_step:              1"
    )
    print()

    print(
        "Reloaded first-step model SHA256:"
    )
    print(
        reload_model_sha
    )
    print()

    print(
        "Reloaded first-step optimizer SHA256:"
    )
    print(
        reload_optimizer_sha
    )
    print()

    print(
        "Epoch-0 batch-1 logical SHA256:"
    )
    print(
        batch1_sha
    )
    print()

    print(
        f"Batch-1 BCEWithLogitsLoss:            "
        f"{a_batch1['loss']:.10f}"
    )
    print()

    print(
        "Batch-1 logit SHA256:"
    )
    print(
        a_batch1[
            "logit_sha256"
        ]
    )
    print()

    print(
        "Batch-1 gradient SHA256:"
    )
    print(
        a_batch1[
            "gradient_sha256"
        ]
    )
    print()

    print(
        "POST-BATCH-1 / TWO-STEP MODEL SHA256:"
    )
    print(
        path_a_model_sha
    )
    print()

    print(
        "POST-BATCH-1 / TWO-STEP OPTIMIZER SHA256:"
    )
    print(
        path_a_optimizer_sha
    )
    print()

    print(
        "Uninterrupted vs resumed:"
    )
    print(
        "  batch-1 loss:                       EXACT"
    )
    print(
        "  batch-1 logits:                     EXACT"
    )
    print(
        "  batch-1 gradients:                  EXACT"
    )
    print(
        "  model parameters:                   32 / 32 EXACT"
    )
    print(
        "  Adam state:                         EXACT"
    )
    print()

    print(
        "Validation performed:                 NO"
    )
    print(
        "Test performed:                       NO"
    )

    banner(
        "PHASE 5.3.2b COMPLETE / "
        "CHECKPOINT-RESUME ROUND TRIP NUMERICALLY EXACT AND FROZEN"
    )


if __name__ == "__main__":
    main()