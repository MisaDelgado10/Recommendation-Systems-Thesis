#!/usr/bin/env python3

from __future__ import annotations

import gc
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# =============================================================================
# Configuration
# =============================================================================

BATCH_SIZE = 512
REDUCED_POSITIVES = 10_732
SLOTS_PER_POSITIVE = 5
REDUCED_EXAMPLES = REDUCED_POSITIVES * SLOTS_PER_POSITIVE
REDUCED_BATCHES = math.ceil(REDUCED_EXAMPLES / BATCH_SIZE)

EXPECTED_SUBSET_SHA = (
    "b433a3472422ca1587eee73241346221"
    "aa884dbded396e18a5ebc5f9cd96e9c3"
)

EXPECTED_INITIAL_MODEL_SHA = (
    "49e822ea7fad35c458f47e134c94c05e"
    "ac099b68c5c468e2c71559c8c88998ab"
)

EXPECTED_VALIDATION16_CANDIDATE_SHA = (
    "a6a1b63954f3065d7c748fab55886af2"
    "8e440d0029626ed9db1809a389665514"
)

PHASE6_BASE_PATH = Path(
    "scripts/phase_6_1_cuda_mac_reference_qualification.py"
)

PACKED_RUNTIME_PATH = Path(
    "scripts/"
    "phase_5_4_8b_packed_mps_"
    "numerical_equivalence_feasibility_audit_V2.py"
)

VALIDATION_RUNTIME_PATH = Path(
    "scripts/"
    "phase_5_3_3b_canonical_real_validation_scoring_preflight.py"
)

RANKING_RUNTIME_PATH = Path(
    "scripts/"
    "phase_5_3_3a_validation_ranking_metric_semantics_audit.py"
)

PHASE6_2_CONTRACT_PATH = Path(
    "data/experimental/phase_6/contracts/"
    "phase_6_2_packed_sparse_cuda_strict_fp32_qualification.json"
)

PHASE6_4_CONTRACT_PATH = Path(
    "data/experimental/phase_6/contracts/"
    "phase_6_4_reduced_training_1pct_contract.json"
)

PHASE6_5A_CONTRACT_PATH = Path(
    "data/experimental/phase_6/contracts/"
    "phase_6_5a_1pct_filtered_epoch_stream_contract.json"
)

POSITIVE_PATH = Path(
    "data/experimental/phase_6/reduced_training/"
    "1pct/positive_order_1pct.parquet"
)

EPOCH0_NEGATIVE_PATH = Path(
    "data/experimental/phase_6/reduced_training/"
    "1pct/epoch_streams/"
    "epoch_00_negative_startup_local.npy"
)

EPOCH0_ORDER_PATH = Path(
    "data/experimental/phase_6/reduced_training/"
    "1pct/epoch_streams/"
    "epoch_00_example_order.npy"
)

VALIDATION_CASES_PATH = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_3c/"
    "full_validation_case_binding.parquet"
)

VALIDATION_CANDIDATES_PATH = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_3c/"
    "full_validation_candidate_startup_local.npy"
)

OUT_DIR = Path(
    "data/experimental/phase_6/audits/"
    "phase_6_5b_1pct_bounded_trainer"
)

CHECKPOINT_PATH = (
    OUT_DIR
    / "bounded_after_batch0_checkpoint.pt"
)

PATH_COMPARISON_PATH = (
    OUT_DIR
    / "uninterrupted_vs_resume.csv"
)

VALIDATION_METRIC_PATH = (
    OUT_DIR
    / "validation16_metrics.csv"
)

CONTRACT_PATH = Path(
    "data/experimental/phase_6/contracts/"
    "phase_6_5b_1pct_bounded_trainer_integration.json"
)


# =============================================================================
# Helpers
# =============================================================================

def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def banner(text):
    print()
    print("=" * 112)
    print(text)
    print("=" * 112)


def load_module(path, name):
    require(
        path.exists(),
        f"Missing source: {path}",
    )

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    require(
        spec is not None
        and spec.loader is not None,
        f"Cannot import {path}",
    )

    module = importlib.util.module_from_spec(spec)

    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def reduced_decode_batch(
    *,
    positive_order,
    negative_matrix,
    epoch_order,
    batch_index,
    num_investors,
):

    require(
        0 <= batch_index < REDUCED_BATCHES,
        "Reduced batch index outside epoch.",
    )

    start = batch_index * BATCH_SIZE

    end = min(
        start + BATCH_SIZE,
        REDUCED_EXAMPLES,
    )

    serialized_indices = (
        epoch_order[start:end]
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
            // SLOTS_PER_POSITIVE
        )

        slot = (
            serialized_index
            % SLOTS_PER_POSITIVE
        )

        positive_row = (
            positive_order.iloc[
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
                negative_matrix[
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
            num_investors
            + startup_local
        )

        rows.append(
            {
                "batch_index": batch_index,
                "batch_position": local_position,
                "epoch_example_position": (
                    start + local_position
                ),
                "serialized_example_index": (
                    serialized_index
                ),
                "positive_order_index": (
                    positive_index
                ),
                "example_slot": slot,
                "negative_draw_index": (
                    negative_draw_index
                ),
                "label": label,
                "source_interaction_id": str(
                    positive_row[
                        "interaction_id"
                    ]
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

    frame = pd.DataFrame(rows)

    expected_size = (
        BATCH_SIZE
        if batch_index
        < REDUCED_BATCHES - 1
        else (
            REDUCED_EXAMPLES
            - (
                REDUCED_BATCHES - 1
            )
            * BATCH_SIZE
        )
    )

    require(
        len(frame) == expected_size,
        (
            f"Reduced batch size drift: "
            f"{len(frame)} != {expected_size}"
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
        "Reduced batch contains non-training segment.",
    )

    return frame


def configure_cuda():

    require(
        os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        )
        == ":4096:8",
        (
            "CUBLAS_WORKSPACE_CONFIG must be "
            "set to :4096:8 before launch."
        ),
    )

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    torch.set_float32_matmul_precision(
        "highest"
    )

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.use_deterministic_algorithms(
        True
    )

    require(
        torch.cuda.is_available(),
        "CUDA unavailable.",
    )

    require(
        torch.__version__.startswith(
            "2.7.0"
        ),
        (
            f"Unexpected PyTorch: "
            f"{torch.__version__}"
        ),
    )


def build_cuda_candidate(
    *,
    roundtrip,
    preflight,
    shared_cpu,
    cuda,
):

    (
        model,
        old_optimizer,
        hash_fn,
        *_,
    ) = (
        roundtrip
        .construct_fresh_training_state(
            preflight
        )
    )

    require(
        hash_fn(model)
        == EXPECTED_INITIAL_MODEL_SHA,
        "Fresh portable model SHA drift.",
    )

    del old_optimizer

    model = model.to(cuda)

    model.startup_embedding.sparse = True
    model.investor_embedding.sparse = True

    model.train()

    optimizer = (
        preflight.build_frozen_adam(
            model
        )
    )

    require(
        len(optimizer.state) == 0,
        "Fresh CUDA Adam must be empty.",
    )

    require(
        hash_fn(model)
        == EXPECTED_INITIAL_MODEL_SHA,
        (
            "Initial parameter bytes "
            "changed during CPU->CUDA."
        ),
    )

    shared_cuda = dict(
        shared_cpu
    )

    shared_cuda[
        "edge_index"
    ] = (
        shared_cpu[
            "edge_index"
        ]
        .to(cuda)
    )

    shared_cuda[
        "edge_type"
    ] = (
        shared_cpu[
            "edge_type"
        ]
        .to(cuda)
    )

    return (
        model,
        optimizer,
        hash_fn,
        shared_cuda,
    )


def execute_packed_batch(
    *,
    model,
    optimizer,
    batch,
    shared_cpu,
    shared_cuda,
    lean_execute,
    packed,
    cuda,
    roundtrip,
    capture,
):

    plan = (
        packed.build_embedding_call_plan(
            batch,
            shared_cpu,
            num_history_periods=(
                roundtrip.NUM_HISTORY_PERIODS
            ),
            num_investors=(
                roundtrip.NUM_INVESTORS
            ),
        )
    )

    startup_calls = (
        list(
            plan[
                "history_startup_calls"
            ]
        )
        + [
            plan[
                "pair_startup_call"
            ]
        ]
    )

    investor_calls = (
        list(
            plan[
                "trend_investor_calls"
            ]
        )
        + [
            plan[
                "pair_investor_call"
            ]
        ]
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    capture.clear()

    startup_dispatch = (
        packed.DevicePackedEmbeddingDispatch(
            model.startup_embedding,
            startup_calls,
            device=cuda,
        )
    )

    investor_dispatch = (
        packed.DevicePackedEmbeddingDispatch(
            model.investor_embedding,
            investor_calls,
            device=cuda,
        )
    )

    startup_dispatch.install()
    investor_dispatch.install()

    torch.cuda.synchronize()

    start = time.perf_counter()

    try:

        result = lean_execute(
            model,
            optimizer,
            lambda _model: (
                "LEAN_RUNTIME_HASH_SKIPPED"
            ),
            batch,
            shared_cuda,
        )

        startup_dispatch.assert_consumed()
        investor_dispatch.assert_consumed()

        torch.cuda.synchronize()

    finally:

        startup_dispatch.restore()
        investor_dispatch.restore()

    elapsed = (
        time.perf_counter()
        - start
    )

    logits = (
        capture.cpu_copy()
    )

    return {
        "loss": float(
            result[
                "loss"
            ]
        ),
        "logits": logits,
        "seconds": elapsed,
        "startup_calls": len(
            startup_calls
        ),
        "investor_calls": len(
            investor_calls
        ),
    }


def save_probe_checkpoint(
    *,
    model,
    optimizer,
    roundtrip,
    epoch0_negative_sha,
    epoch0_order_sha,
):

    payload = {
        "schema_version": (
            "ITRS_PHASE6_1PCT_CHECKPOINT_V1"
        ),

        "experiment": (
            "PHASE6_1PCT_PILOT"
        ),

        "subset_sha256": (
            EXPECTED_SUBSET_SHA
        ),

        "epoch_index": 0,
        "next_batch_index": 1,
        "global_optimizer_step": 1,
        "epoch_example_count": 512,

        "epoch0_reduced_negative_sha256": (
            epoch0_negative_sha
        ),

        "epoch0_reduced_order_sha256": (
            epoch0_order_sha
        ),

        "model_state_dict": (
            roundtrip.clone_model_state_dict(
                model
            )
        ),

        "optimizer_state_dict": (
            roundtrip.clone_optimizer_state_dict(
                optimizer
            )
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

        "cuda_rng_state": (
            torch.cuda.get_rng_state(
                device=0
            )
            .cpu()
            .clone()
        ),
    }

    torch.save(
        payload,
        CHECKPOINT_PATH,
    )


def restore_probe_checkpoint(
    *,
    payload,
    roundtrip,
    preflight,
    shared_cpu,
    cuda,
):

    require(
        payload[
            "schema_version"
        ]
        == "ITRS_PHASE6_1PCT_CHECKPOINT_V1",
        "Checkpoint schema drift.",
    )

    require(
        payload[
            "subset_sha256"
        ]
        == EXPECTED_SUBSET_SHA,
        "Checkpoint subset SHA drift.",
    )

    require(
        int(
            payload[
                "next_batch_index"
            ]
        )
        == 1,
        "Checkpoint resume batch drift.",
    )

    (
        model,
        old_optimizer,
        hash_fn,
        *_,
    ) = (
        roundtrip
        .construct_fresh_training_state(
            preflight
        )
    )

    del old_optimizer

    model.load_state_dict(
        payload[
            "model_state_dict"
        ],
        strict=True,
    )

    model = model.to(cuda)

    model.startup_embedding.sparse = True
    model.investor_embedding.sparse = True

    model.train()

    optimizer = (
        preflight.build_frozen_adam(
            model
        )
    )

    optimizer.load_state_dict(
        payload[
            "optimizer_state_dict"
        ]
    )

    shared_cuda = dict(
        shared_cpu
    )

    shared_cuda[
        "edge_index"
    ] = (
        shared_cpu[
            "edge_index"
        ].to(cuda)
    )

    shared_cuda[
        "edge_type"
    ] = (
        shared_cpu[
            "edge_type"
        ].to(cuda)
    )

    random.setstate(
        payload[
            "python_rng_state"
        ]
    )

    np.random.set_state(
        payload[
            "numpy_rng_state"
        ]
    )

    torch.set_rng_state(
        payload[
            "torch_rng_state"
        ]
    )

    torch.cuda.set_rng_state(
        payload[
            "cuda_rng_state"
        ],
        device=0,
    )

    return (
        model,
        optimizer,
        hash_fn,
        shared_cuda,
    )


# =============================================================================
# Main
# =============================================================================

def main():

    banner(
        "PHASE 6.5b — 1% BOUNDED CUDA "
        "TRAINER / CHECKPOINT / VALIDATION INTEGRATION"
    )

    print(
        "Full epoch launched:          NO"
    )
    print(
        "Production run launched:      NO"
    )
    print(
        "Optimizer steps Path A:       2"
    )
    print(
        "Optimizer steps Path B:       2"
    )
    print(
        "Validation cases scored:      16"
    )
    print(
        "Test cases accessed/scored:   0"
    )

    # =========================================================================
    # Contracts / runtime configuration
    # =========================================================================

    configure_cuda()

    cuda = torch.device(
        "cuda:0"
    )

    for path in (
        PHASE6_2_CONTRACT_PATH,
        PHASE6_4_CONTRACT_PATH,
        PHASE6_5A_CONTRACT_PATH,
        POSITIVE_PATH,
        EPOCH0_NEGATIVE_PATH,
        EPOCH0_ORDER_PATH,
        VALIDATION_CASES_PATH,
        VALIDATION_CANDIDATES_PATH,
    ):
        require(
            path.exists(),
            f"Missing prerequisite: {path}",
        )

    q2 = json.loads(
        PHASE6_2_CONTRACT_PATH.read_text(
            encoding="utf-8"
        )
    )

    q4 = json.loads(
        PHASE6_4_CONTRACT_PATH.read_text(
            encoding="utf-8"
        )
    )

    q5a = json.loads(
        PHASE6_5A_CONTRACT_PATH.read_text(
            encoding="utf-8"
        )
    )

    require(
        q2[
            "status"
        ]
        == "QUALIFIED",
        "Phase 6.2 not qualified.",
    )

    require(
        int(
            q2[
                "policy_checks_passed"
            ]
        )
        == 22,
        "Phase 6.2 is not 22/22.",
    )

    require(
        q4[
            "selection"
        ][
            "subset_logical_sha256"
        ]
        == EXPECTED_SUBSET_SHA,
        "Phase 6.4 subset SHA drift.",
    )

    require(
        q5a[
            "status"
        ]
        == "FROZEN_FOR_1PCT_PILOT",
        "Phase 6.5a not frozen.",
    )

    print()
    print(
        "Phase 6.2 numerical qualification: PASS / 22/22"
    )
    print(
        "Phase 6.4 subset gate:             PASS"
    )
    print(
        "Phase 6.5a epoch-stream gate:      PASS"
    )

    # =========================================================================
    # Load runtime infrastructure
    # =========================================================================

    q = load_module(
        PHASE6_BASE_PATH,
        "_phase6_5b_base",
    )

    packed = load_module(
        PACKED_RUNTIME_PATH,
        "_phase6_5b_packed",
    )

    validation_runtime = load_module(
        VALIDATION_RUNTIME_PATH,
        "_phase6_5b_validation",
    )

    ranking_runtime = load_module(
        RANKING_RUNTIME_PATH,
        "_phase6_5b_ranking",
    )

    require(
        q.file_sha256(
            q.INITIAL_STATE_PATH
        )
        == q.EXPECTED_INITIAL_FILE_SHA,
        "Portable initial file SHA drift.",
    )

    initial_payload = torch.load(
        q.INITIAL_STATE_PATH,
        map_location="cpu",
        weights_only=False,
    )

    roundtrip = q.load_module(
        q.ROUNDTRIP_PATH,
        "_phase6_5b_roundtrip",
    )

    preflight = (
        roundtrip
        .load_preflight_runtime()
    )

    q.install_portable_initialization_bridge(
        preflight,
        initial_payload[
            "model_state_dict"
        ],
    )

    (
        lean_execute,
        set_executor_device,
        executor_metadata,
    ) = (
        packed
        .build_corrected_lean_device_executor(
            roundtrip
        )
    )

    require(
        executor_metadata[
            "require_statements_removed"
        ]
        == 25,
        "Lean executor guard count drift.",
    )

    require(
        executor_metadata[
            "torch_from_numpy_rewrites"
        ]
        == 10,
        "Lean executor device rewrite drift.",
    )

    set_executor_device(
        "cuda:0"
    )

    shared_cpu = (
        roundtrip.load_shared_inputs(
            preflight
        )
    )

    # =========================================================================
    # Reduced epoch-0 stream
    # =========================================================================

    banner(
        "LOAD REDUCED EPOCH-0 STREAM"
    )

    positive_order = pd.read_parquet(
        POSITIVE_PATH
    )

    negative_matrix = np.load(
        EPOCH0_NEGATIVE_PATH,
        mmap_mode="r",
    )

    epoch_order = np.load(
        EPOCH0_ORDER_PATH,
        mmap_mode="r",
    )

    require(
        len(positive_order)
        == REDUCED_POSITIVES,
        "Reduced positive count drift.",
    )

    require(
        negative_matrix.shape
        == (
            REDUCED_POSITIVES,
            4,
        ),
        "Reduced negative shape drift.",
    )

    require(
        len(epoch_order)
        == REDUCED_EXAMPLES,
        "Reduced epoch-order length drift.",
    )

    subset_sha = (
        roundtrip
        .dataframe_logical_sha256(
            positive_order,
            columns=[
                "interaction_id",
                "investor_global",
                "startup_local",
                "segment_number",
            ],
        )
    )

    require(
        subset_sha
        == EXPECTED_SUBSET_SHA,
        "Reduced positive SHA drift.",
    )

    epoch0_negative_sha = (
        preflight
        .array_logical_sha256(
            np.asarray(
                negative_matrix
            )
        )
    )

    epoch0_order_sha = (
        preflight
        .array_logical_sha256(
            np.asarray(
                epoch_order
            )
        )
    )

    batch0 = reduced_decode_batch(
        positive_order=positive_order,
        negative_matrix=negative_matrix,
        epoch_order=epoch_order,
        batch_index=0,
        num_investors=(
            roundtrip.NUM_INVESTORS
        ),
    )

    batch1 = reduced_decode_batch(
        positive_order=positive_order,
        negative_matrix=negative_matrix,
        epoch_order=epoch_order,
        batch_index=1,
        num_investors=(
            roundtrip.NUM_INVESTORS
        ),
    )

    require(
        len(batch0) == 512,
        "Batch 0 is not 512.",
    )

    require(
        len(batch1) == 512,
        "Batch 1 is not 512.",
    )

    print(
        f"Positive groups:               "
        f"{len(positive_order):,}"
    )

    print(
        f"Examples / epoch:              "
        f"{len(epoch_order):,}"
    )

    print(
        f"Batches / epoch:               "
        f"{REDUCED_BATCHES}"
    )

    print(
        "Batch 0 decoded:               512"
    )
    print(
        "Batch 1 decoded:               512"
    )

    # =========================================================================
    # Path A — uninterrupted
    # =========================================================================

    banner(
        "PATH A — TWO UNINTERRUPTED REDUCED CUDA BATCHES"
    )

    (
        model_a,
        optimizer_a,
        hash_fn_a,
        shared_cuda_a,
    ) = build_cuda_candidate(
        roundtrip=roundtrip,
        preflight=preflight,
        shared_cpu=shared_cpu,
        cuda=cuda,
    )

    capture_a = (
        packed.LogitCapture(
            model_a
        )
    )

    capture_a.install()

    a0 = execute_packed_batch(
        model=model_a,
        optimizer=optimizer_a,
        batch=batch0,
        shared_cpu=shared_cpu,
        shared_cuda=shared_cuda_a,
        lean_execute=lean_execute,
        packed=packed,
        cuda=cuda,
        roundtrip=roundtrip,
        capture=capture_a,
    )

    a1 = execute_packed_batch(
        model=model_a,
        optimizer=optimizer_a,
        batch=batch1,
        shared_cpu=shared_cpu,
        shared_cuda=shared_cuda_a,
        lean_execute=lean_execute,
        packed=packed,
        cuda=cuda,
        roundtrip=roundtrip,
        capture=capture_a,
    )

    capture_a.restore()

    path_a_model_sha = (
        hash_fn_a(
            model_a
        )
    )

    path_a_optimizer_sha = (
        roundtrip
        .optimizer_state_logical_sha256(
            model_a,
            optimizer_a,
        )
    )

    path_a_parameter_hashes = (
        roundtrip.parameter_hashes(
            model_a
        )
    )

    print(
        f"Batch 0: loss={a0['loss']:.10f} "
        f"time={a0['seconds']:.3f}s"
    )

    print(
        f"Batch 1: loss={a1['loss']:.10f} "
        f"time={a1['seconds']:.3f}s"
    )

    print(
        "Path A complete:               PASS"
    )

    # =========================================================================
    # Path B — checkpoint / destroy / reload / resume
    # =========================================================================

    banner(
        "PATH B — BATCH 0 -> CHECKPOINT -> RELOAD -> BATCH 1"
    )

    (
        model_b,
        optimizer_b,
        hash_fn_b,
        shared_cuda_b,
    ) = build_cuda_candidate(
        roundtrip=roundtrip,
        preflight=preflight,
        shared_cpu=shared_cpu,
        cuda=cuda,
    )

    capture_b = (
        packed.LogitCapture(
            model_b
        )
    )

    capture_b.install()

    b0 = execute_packed_batch(
        model=model_b,
        optimizer=optimizer_b,
        batch=batch0,
        shared_cpu=shared_cpu,
        shared_cuda=shared_cuda_b,
        lean_execute=lean_execute,
        packed=packed,
        cuda=cuda,
        roundtrip=roundtrip,
        capture=capture_b,
    )

    batch0_model_sha = (
        hash_fn_b(
            model_b
        )
    )

    batch0_optimizer_sha = (
        roundtrip
        .optimizer_state_logical_sha256(
            model_b,
            optimizer_b,
        )
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_probe_checkpoint(
        model=model_b,
        optimizer=optimizer_b,
        roundtrip=roundtrip,
        epoch0_negative_sha=(
            epoch0_negative_sha
        ),
        epoch0_order_sha=(
            epoch0_order_sha
        ),
    )

    capture_b.restore()

    del model_b
    del optimizer_b
    del shared_cuda_b

    gc.collect()
    torch.cuda.empty_cache()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    require(
        checkpoint[
            "epoch0_reduced_negative_sha256"
        ]
        == epoch0_negative_sha,
        "Checkpoint negative-stream SHA drift.",
    )

    require(
        checkpoint[
            "epoch0_reduced_order_sha256"
        ]
        == epoch0_order_sha,
        "Checkpoint epoch-order SHA drift.",
    )

    (
        model_r,
        optimizer_r,
        hash_fn_r,
        shared_cuda_r,
    ) = restore_probe_checkpoint(
        payload=checkpoint,
        roundtrip=roundtrip,
        preflight=preflight,
        shared_cpu=shared_cpu,
        cuda=cuda,
    )

    reload_model_sha = (
        hash_fn_r(
            model_r
        )
    )

    reload_optimizer_sha = (
        roundtrip
        .optimizer_state_logical_sha256(
            model_r,
            optimizer_r,
        )
    )

    require(
        reload_model_sha
        == batch0_model_sha,
        (
            "Checkpoint reload model "
            "does not equal post-batch0 state."
        ),
    )

    require(
        reload_optimizer_sha
        == batch0_optimizer_sha,
        (
            "Checkpoint reload Adam "
            "does not equal post-batch0 state."
        ),
    )

    capture_r = (
        packed.LogitCapture(
            model_r
        )
    )

    capture_r.install()

    b1 = execute_packed_batch(
        model=model_r,
        optimizer=optimizer_r,
        batch=batch1,
        shared_cpu=shared_cpu,
        shared_cuda=shared_cuda_r,
        lean_execute=lean_execute,
        packed=packed,
        cuda=cuda,
        roundtrip=roundtrip,
        capture=capture_r,
    )

    capture_r.restore()

    path_b_model_sha = (
        hash_fn_r(
            model_r
        )
    )

    path_b_optimizer_sha = (
        roundtrip
        .optimizer_state_logical_sha256(
            model_r,
            optimizer_r,
        )
    )

    path_b_parameter_hashes = (
        roundtrip.parameter_hashes(
            model_r
        )
    )

    # =========================================================================
    # Exact uninterrupted vs resume comparison
    # =========================================================================

    banner(
        "UNINTERRUPTED vs CHECKPOINT/RESUME"
    )

    batch0_loss_exact = (
        a0[
            "loss"
        ]
        == b0[
            "loss"
        ]
    )

    batch1_loss_exact = (
        a1[
            "loss"
        ]
        == b1[
            "loss"
        ]
    )

    batch0_logits_exact = (
        torch.equal(
            a0[
                "logits"
            ],
            b0[
                "logits"
            ],
        )
    )

    batch1_logits_exact = (
        torch.equal(
            a1[
                "logits"
            ],
            b1[
                "logits"
            ],
        )
    )

    model_exact = (
        path_a_model_sha
        == path_b_model_sha
    )

    optimizer_exact = (
        path_a_optimizer_sha
        == path_b_optimizer_sha
    )

    parameter_hashes_exact = (
        path_a_parameter_hashes
        == path_b_parameter_hashes
    )

    checks = {
        "batch0_loss_exact": (
            batch0_loss_exact
        ),
        "batch0_logits_exact": (
            batch0_logits_exact
        ),
        "checkpoint_model_reload_exact": (
            reload_model_sha
            == batch0_model_sha
        ),
        "checkpoint_optimizer_reload_exact": (
            reload_optimizer_sha
            == batch0_optimizer_sha
        ),
        "batch1_loss_exact": (
            batch1_loss_exact
        ),
        "batch1_logits_exact": (
            batch1_logits_exact
        ),
        "two_step_model_exact": (
            model_exact
        ),
        "two_step_optimizer_exact": (
            optimizer_exact
        ),
        "all_parameter_hashes_exact": (
            parameter_hashes_exact
        ),
    }

    require(
        all(
            checks.values()
        ),
        (
            "At least one checkpoint/resume "
            "equivalence check failed."
        ),
    )

    comparison_df = pd.DataFrame(
        [
            {
                "check": key,
                "result": (
                    "PASS"
                    if value
                    else "FAIL"
                ),
            }
            for key, value
            in checks.items()
        ]
    )

    print(
        comparison_df.to_string(
            index=False
        )
    )

    PATH_COMPARISON_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_df.to_csv(
        PATH_COMPARISON_PATH,
        index=False,
    )

    # =========================================================================
    # Capture exact two-step state for validation
    # =========================================================================

    post_two_step_state = (
        roundtrip.clone_model_state_dict(
            model_r
        )
    )

    del model_a
    del optimizer_a
    del shared_cuda_a

    del optimizer_r
    del shared_cuda_r
    del model_r

    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # Real validation integration — FIRST 16 VALIDATION CASES ONLY
    # =========================================================================

    banner(
        "REAL VALIDATION INTEGRATION — FIRST 16 FROZEN CASES"
    )

    validation_cases_full = pd.read_parquet(
        VALIDATION_CASES_PATH
    )

    validation_candidates_full = np.load(
        VALIDATION_CANDIDATES_PATH,
        mmap_mode="r",
    )

    require(
        len(
            validation_cases_full
        )
        == 2_251,
        (
            "Frozen validation case count "
            "is not 2,251."
        ),
    )

    require(
        validation_candidates_full.shape
        == (
            2_251,
            100,
        ),
        (
            "Frozen validation candidate "
            "matrix shape drift."
        ),
    )

    selected_cases = (
        validation_cases_full.iloc[
            :16
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    selected_candidates = np.array(
        validation_candidates_full[
            :16
        ],
        dtype=np.int64,
        copy=True,
    )

    candidate_sha = (
        validation_runtime
        .array_logical_sha256(
            selected_candidates
        )
    )

    require(
        candidate_sha
        == EXPECTED_VALIDATION16_CANDIDATE_SHA,
        (
            "First-16 validation candidate "
            "fingerprint drift."
        ),
    )

    # -------------------------------------------------------------------------
    # Reconstruct exact architecture on CPU and load the REAL trained 2-step
    # state produced by the reduced CUDA trainer.
    # -------------------------------------------------------------------------

    (
        validation_model,
        validation_old_optimizer,
        validation_hash_fn,
        *_,
    ) = (
        roundtrip
        .construct_fresh_training_state(
            preflight
        )
    )

    del validation_old_optimizer

    validation_model.load_state_dict(
        post_two_step_state,
        strict=True,
    )

    validation_model.eval()

    validation_state_before = (
        validation_hash_fn(
            validation_model
        )
    )

    require(
        validation_state_before
        == path_b_model_sha,
        (
            "CPU validation model does not "
            "match trained two-step state."
        ),
    )

    validation_start = (
        time.perf_counter()
    )

    with torch.no_grad():

        features = (
            validation_runtime
            .compute_validation_features(
                validation_model,
                selected_cases,
                selected_candidates,
                shared_cpu,
            )
        )

        metric_rows = []

        for case_position, row in (
            selected_cases.iterrows()
        ):

            investor_global = int(
                row[
                    "investor_global"
                ]
            )

            positive_local = int(
                row[
                    "positive_startup_local"
                ]
            )

            candidates_local = (
                selected_candidates[
                    case_position
                ]
            )

            logits = (
                validation_runtime
                .score_validation_case(
                    validation_model,
                    investor_global,
                    candidates_local,
                    features,
                )
            )

            logits_np = (
                logits
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

            require(
                np.isfinite(
                    logits_np
                ).all(),
                (
                    "Validation logits "
                    "contain non-finite values."
                ),
            )

            positive_rank = (
                ranking_runtime
                .rank_candidates(
                    logits_np,
                    candidates_local,
                    positive_local,
                )
            )

            (
                hr10,
                ndcg10,
            ) = (
                ranking_runtime
                .metrics_from_positive_rank(
                    positive_rank,
                    k=10,
                )
            )

            metric_rows.append(
                {
                    "validation_case_position": (
                        int(
                            case_position
                        )
                    ),
                    "investor_global": (
                        investor_global
                    ),
                    "positive_startup_local": (
                        positive_local
                    ),
                    "positive_rank": (
                        int(
                            positive_rank
                        )
                    ),
                    "HR@10": (
                        float(
                            hr10
                        )
                    ),
                    "NDCG@10": (
                        float(
                            ndcg10
                        )
                    ),
                }
            )

    validation_seconds = (
        time.perf_counter()
        - validation_start
    )

    metric_df = pd.DataFrame(
        metric_rows
    )

    require(
        len(metric_df) == 16,
        "Validation did not score 16 cases.",
    )

    require(
        bool(
            metric_df[
                "positive_rank"
            ]
            .between(
                1,
                100,
            )
            .all()
        ),
        "Validation rank outside 1..100.",
    )

    (
        validation_hr10,
        validation_ndcg10,
    ) = (
        ranking_runtime
        .aggregate_event_level_metrics(
            metric_df
        )
    )

    validation_state_after = (
        validation_hash_fn(
            validation_model
        )
    )

    require(
        validation_state_after
        == validation_state_before,
        (
            "Validation changed "
            "model parameters."
        ),
    )

    require(
        all(
            parameter.grad is None
            for parameter
            in validation_model.parameters()
        ),
        (
            "Validation created gradients."
        ),
    )

    VALIDATION_METRIC_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metric_df.to_csv(
        VALIDATION_METRIC_PATH,
        index=False,
    )

    print(
        metric_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Diagnostic 16-case HR@10:      "
        f"{validation_hr10:.12f}"
    )

    print(
        f"Diagnostic 16-case NDCG@10:    "
        f"{validation_ndcg10:.12f}"
    )

    print(
        f"Validation elapsed:            "
        f"{validation_seconds:.2f} s"
    )

    print(
        "Parameter state unchanged:     YES"
    )

    print(
        "Test cases accessed/scored:    0"
    )

    # =========================================================================
    # Contract
    # =========================================================================

    contract = {
        "phase": "6.5b",

        "status": (
            "PASS_BOUNDED_1PCT_TRAINER_INTEGRATION"
        ),

        "experiment": (
            "1PCT_REDUCED_SUPERVISION_PILOT"
        ),

        "cuda_runtime": {
            "gpu": (
                torch.cuda.get_device_name(
                    0
                )
            ),
            "torch": (
                torch.__version__
            ),
            "cuda_runtime": (
                torch.version.cuda
            ),
            "strict_fp32": True,
            "TF32": False,
            "deterministic_algorithms": True,
            "packed_embeddings": True,
            "startup_embedding_sparse": True,
            "investor_embedding_sparse": True,
        },

        "training_probe": {
            "epoch": 0,
            "batches_tested": [
                0,
                1,
            ],
            "batch_size": 512,
            "path_A_optimizer_steps": 2,
            "path_B_optimizer_steps": 2,

            "batch0_loss": (
                a0[
                    "loss"
                ]
            ),

            "batch1_loss": (
                a1[
                    "loss"
                ]
            ),

            "checkpoint_resume_exact": (
                True
            ),

            "two_step_model_sha256": (
                path_b_model_sha
            ),

            "two_step_optimizer_sha256": (
                path_b_optimizer_sha
            ),
        },

        "validation_probe": {
            "cases": 16,
            "candidate_count_per_case": 100,
            "candidate_prefix_sha256": (
                candidate_sha
            ),
            "HR@10": (
                validation_hr10
            ),
            "NDCG@10": (
                validation_ndcg10
            ),
            "elapsed_seconds": (
                validation_seconds
            ),
            "diagnostic_only": True,
            "used_for_checkpoint_selection": False,
        },

        "boundary": {
            "full_epoch_completed": False,
            "production_training_launched": False,
            "full_validation_executed": False,
            "test_cases_accessed": 0,
            "test_cases_scored": 0,
        },
    }

    CONTRACT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTRACT_PATH.write_text(
        json.dumps(
            contract,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # =========================================================================
    # Final
    # =========================================================================

    banner(
        "PHASE 6.5b FINAL STATUS"
    )

    print(
        "Reduced stream decode:                  PASS"
    )

    print(
        "Qualified packed+sparse CUDA runtime:   PASS"
    )

    print(
        "Batch0 uninterrupted/resume equality:   EXACT"
    )

    print(
        "Checkpoint model reload:                EXACT"
    )

    print(
        "Checkpoint Adam reload:                 EXACT"
    )

    print(
        "Batch1 logits after resume:             EXACT"
    )

    print(
        "Two-step parameter state:               EXACT"
    )

    print(
        "Two-step Adam state:                    EXACT"
    )

    print(
        "Real 16-case validation forward:        PASS"
    )

    print(
        "Validation model state unchanged:       YES"
    )

    print(
        "Test cases accessed/scored:             0"
    )

    print()
    print(
        f"Batch 0 CUDA time:                      "
        f"{a0['seconds']:.3f} s"
    )

    print(
        f"Batch 1 CUDA time:                      "
        f"{a1['seconds']:.3f} s"
    )

    print(
        f"16-case validation time:                "
        f"{validation_seconds:.2f} s"
    )

    print()
    print(
        f"WROTE  {CHECKPOINT_PATH}"
    )

    print(
        f"WROTE  {PATH_COMPARISON_PATH}"
    )

    print(
        f"WROTE  {VALIDATION_METRIC_PATH}"
    )

    print(
        f"WROTE  {CONTRACT_PATH}"
    )

    print()
    print(
        "PHASE 6.5b: PASS / "
        "1% TRAINER INTEGRATION QUALIFIED"
    )


if __name__ == "__main__":
    main()
