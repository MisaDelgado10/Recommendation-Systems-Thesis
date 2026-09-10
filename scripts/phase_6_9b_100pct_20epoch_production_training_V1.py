#!/usr/bin/env python3
"""
Phase 6.9b — 100% 20-Epoch CUDA Production Trainer V1

Purpose
-------
Run the full-data ITRS reproduction training schedule using the already-frozen
Phase-6.9a 20-epoch training streams.

Scientific schedule
-------------------
Positive events / epoch:  1,073,249
Negatives / positive:     4
Examples / epoch:         5,366,245
Batch size:               512
Batches / epoch:          10,481
Final batch size:         485
Epochs:                   20
Optimizer steps:          209,620
Validation:               all 2,251 frozen validation cases after every epoch
Checkpoint selection:     max NDCG@10, then HR@10, then earliest epoch
Test:                     strictly disabled

Operational policy
------------------
- Strict deterministic FP32 CUDA runtime already qualified in Phase 6.2.
- Qualified packed embedding dispatch + D_BOTH_SPARSE path.
- Canonical full-data Phase-5.3.2b batch decoder.
- Atomic restart checkpoint every 100 training batches.
- Mandatory restart checkpoint after an epoch finishes and before validation.
- Latest checkpoint stores model, Adam, controller state, Python/NumPy/Torch/CUDA RNG.
- Resume reuses the exact frozen epoch stream and exact next batch.
- Test is not loaded, scored, or used for model selection.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# =============================================================================
# Experimental contract
# =============================================================================

NUM_EPOCHS = 20

BATCH_SIZE = 512
POSITIVE_EVENTS = 1_073_249
NEGATIVES_PER_POSITIVE = 4
SLOTS_PER_POSITIVE = 5

EXAMPLES_PER_EPOCH = (
    POSITIVE_EVENTS
    * SLOTS_PER_POSITIVE
)

BATCHES_PER_EPOCH = math.ceil(
    EXAMPLES_PER_EPOCH
    / BATCH_SIZE
)

FINAL_BATCH_SIZE = (
    EXAMPLES_PER_EPOCH
    - (
        BATCHES_PER_EPOCH - 1
    )
    * BATCH_SIZE
)

TOTAL_OPTIMIZER_STEPS = (
    NUM_EPOCHS
    * BATCHES_PER_EPOCH
)

CHECKPOINT_INTERVAL_BATCHES = 100
PROGRESS_PRINT_INTERVAL_BATCHES = 25

VALIDATION_CASES = 2_251
CANDIDATES_PER_CASE = 100

EXPECTED_REPOSITORY_HEAD = (
    "6c94a4e787d2bc7a27e9c1ebced3ddf41132d915"
)

EXPECTED_INITIAL_MODEL_SHA = (
    "49e822ea7fad35c458f47e134c94c05e"
    "ac099b68c5c468e2c71559c8c88998ab"
)

EXPECTED_POSITIVE_SHA = (
    "73b074a80675793b811fbdc8a0609883"
    "c857fb2a687a2e01c31865ade5b509d1"
)

EXPECTED_STREAM_REGISTRY_FILE_SHA256 = (
    "e6970082a06b3b097e0858e4d0761240"
    "c99c3a255b3227339cea6243821e2eb1"
)

BASELINE_VALIDATION_HR10 = (
    0.09151488227454465
)

BASELINE_VALIDATION_NDCG10 = (
    0.040193099163
)


# =============================================================================
# Frozen sources
# =============================================================================

PHASE6_BASE_PATH = Path(
    "scripts/"
    "phase_6_1_cuda_mac_reference_qualification.py"
)

PACKED_RUNTIME_PATH = Path(
    "scripts/"
    "phase_5_4_8b_packed_mps_"
    "numerical_equivalence_feasibility_audit_V2.py"
)

ROUNDTRIP_PATH = Path(
    "scripts/"
    "phase_5_3_2b_checkpoint_resume_roundtrip_proof.py"
)

VALIDATION_PREFLIGHT_PATH = Path(
    "scripts/"
    "phase_5_3_3b_"
    "canonical_real_validation_scoring_preflight.py"
)

FULL_VALIDATION_RUNTIME_PATH = Path(
    "scripts/"
    "phase_5_3_3c_"
    "full_validation_split_runtime_dry_run.py"
)

RANKING_RUNTIME_PATH = Path(
    "scripts/"
    "phase_5_3_3a_"
    "validation_ranking_metric_semantics_audit.py"
)

SELECTION_RUNTIME_PATH = Path(
    "scripts/"
    "phase_5_3_3d_"
    "validation_checkpoint_selection_integration_proof.py"
)


# =============================================================================
# Frozen prerequisite contracts
# =============================================================================

PHASE6_2_CONTRACT_PATH = Path(
    "data/experimental/phase_6/contracts/"
    "phase_6_2_packed_sparse_cuda_strict_fp32_qualification.json"
)

PHASE6_9A_CONTRACT_PATH = Path(
    "data/experimental/phase_6/contracts/"
    "phase_6_9a_100pct_20epoch_stream_freeze_V2.json"
)


# =============================================================================
# Frozen full-data stream
# =============================================================================

FULL_TRAINING_ROOT = Path(
    "data/experimental/phase_6/"
    "full_training/100pct"
)

POSITIVE_PATH = (
    FULL_TRAINING_ROOT
    / "canonical_training_positive_event_order.parquet"
)

STREAM_DIR = (
    FULL_TRAINING_ROOT
    / "epoch_streams"
)

STREAM_REGISTRY_PATH = (
    STREAM_DIR
    / "full_epoch_stream_registry.csv"
)


# =============================================================================
# Frozen validation artifacts
# =============================================================================

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


# =============================================================================
# Full production outputs
# =============================================================================

RUN_DIR = (
    FULL_TRAINING_ROOT
    / "run_20epoch"
)

CHECKPOINT_DIR = (
    RUN_DIR
    / "checkpoints"
)

VALIDATION_DIR = (
    RUN_DIR
    / "validation"
)

LATEST_CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "latest.pt"
)

BEST_CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "best.pt"
)

EPOCH_METRICS_PATH = (
    RUN_DIR
    / "epoch_metrics.csv"
)

PROGRESS_PATH = (
    RUN_DIR
    / "progress.json"
)

FINAL_CONTRACT_PATH = Path(
    "data/experimental/phase_6/contracts/"
    "phase_6_9b_100pct_20epoch_production_training_V1.json"
)


# =============================================================================
# Helpers
# =============================================================================

def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def banner(text):
    print()
    print("=" * 118)
    print(text)
    print("=" * 118)


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
        f"Could not import {path}",
    )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[
        name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


def load_json(path):
    require(
        path.exists(),
        f"Missing JSON: {path}",
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def git_head():
    return (
        subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            text=True,
        )
        .strip()
    )


def require_clean_tracked_tree():
    result = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    require(
        result.stdout.strip() == "",
        (
            "Tracked Git working tree is not clean.\n"
            + result.stdout
        ),
    )


def file_sha256(
    path,
    chunk_size=8 * 1024 * 1024,
):
    digest = hashlib.sha256()

    with Path(path).open(
        "rb"
    ) as handle:
        while True:
            chunk = handle.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def current_script_path():
    return Path(
        __file__
    ).resolve()


def configure_cuda():
    require(
        os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        )
        == ":4096:8",
        (
            "CUBLAS_WORKSPACE_CONFIG must "
            "equal :4096:8."
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
            "Expected PyTorch 2.7.0, got "
            f"{torch.__version__}"
        ),
    )


def recursive_cpu_copy(value):
    if isinstance(
        value,
        torch.Tensor,
    ):
        return (
            value
            .detach()
            .cpu()
            .clone()
        )

    if isinstance(
        value,
        dict,
    ):
        return {
            key: recursive_cpu_copy(
                item
            )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        list,
    ):
        return [
            recursive_cpu_copy(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        tuple,
    ):
        return tuple(
            recursive_cpu_copy(
                item
            )
            for item in value
        )

    return value


def atomic_torch_save(
    payload,
    path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = Path(
        str(path)
        + ".tmp"
    )

    torch.save(
        payload,
        temporary,
    )

    os.replace(
        temporary,
        path,
    )


def atomic_json_write(
    payload,
    path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = Path(
        str(path)
        + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


def epoch_negative_path(epoch):
    return (
        STREAM_DIR
        / (
            f"epoch_{epoch:02d}_"
            "negative_startup_local.npy"
        )
    )


def epoch_order_path(epoch):
    return (
        STREAM_DIR
        / (
            f"epoch_{epoch:02d}_"
            "example_order.npy"
        )
    )


def make_runtime_bindings(
    *,
    repository_commit,
):
    return {
        "repository_commit": (
            repository_commit
        ),
        "trainer_script_file_sha256": (
            file_sha256(
                current_script_path()
            )
        ),
        "phase6_9a_contract_file_sha256": (
            file_sha256(
                PHASE6_9A_CONTRACT_PATH
            )
        ),
        "stream_registry_file_sha256": (
            file_sha256(
                STREAM_REGISTRY_PATH
            )
        ),
        "positive_file_sha256": (
            file_sha256(
                POSITIVE_PATH
            )
        ),
    }


# =============================================================================
# Controller + checkpoint schema
# =============================================================================

def make_fresh_state():
    return {
        "epoch_index": 0,
        "next_batch_index": 0,
        "global_optimizer_step": 0,

        "epoch_loss_weighted_sum": 0.0,
        "epoch_example_count": 0,
        "epoch_training_seconds": 0.0,

        "validation_pending": False,
        "validation_history": [],

        "best_validation_epoch": None,
        "best_validation_ndcg10": None,
        "best_validation_hr10": None,

        "training_complete": False,

        "test_accessed": False,
        "test_scored": False,
    }


def progress_payload(
    *,
    state,
    runtime_bindings,
):
    return {
        "phase": "6.9b",
        "experiment": (
            "FULL_100PCT_ITRS_PRODUCTION"
        ),
        "runtime_bindings": (
            runtime_bindings
        ),
        "epoch_index": int(
            state[
                "epoch_index"
            ]
        ),
        "display_epoch": (
            int(
                state[
                    "epoch_index"
                ]
            )
            + 1
        ),
        "next_batch_index": int(
            state[
                "next_batch_index"
            ]
        ),
        "batches_per_epoch": (
            BATCHES_PER_EPOCH
        ),
        "global_optimizer_step": int(
            state[
                "global_optimizer_step"
            ]
        ),
        "total_optimizer_steps": (
            TOTAL_OPTIMIZER_STEPS
        ),
        "epoch_example_count": int(
            state[
                "epoch_example_count"
            ]
        ),
        "validation_pending": bool(
            state[
                "validation_pending"
            ]
        ),
        "completed_validations": len(
            state[
                "validation_history"
            ]
        ),
        "best_validation_epoch": (
            state[
                "best_validation_epoch"
            ]
        ),
        "best_validation_ndcg10": (
            state[
                "best_validation_ndcg10"
            ]
        ),
        "best_validation_hr10": (
            state[
                "best_validation_hr10"
            ]
        ),
        "training_complete": bool(
            state[
                "training_complete"
            ]
        ),
        "test_accessed": False,
        "test_scored": False,
    }


def checkpoint_payload(
    *,
    role,
    model,
    optimizer,
    state,
    roundtrip,
    runtime_bindings,
):
    return {
        "schema_version": (
            "ITRS_PHASE6_100PCT_PRODUCTION_V1"
        ),
        "checkpoint_role": (
            role
        ),
        "experiment": (
            "FULL_100PCT_ITRS_PRODUCTION"
        ),
        "runtime_bindings": (
            runtime_bindings
        ),
        "controller_state": (
            state
        ),
        "model_state_dict": (
            roundtrip
            .clone_model_state_dict(
                model
            )
        ),
        "optimizer_state_dict": (
            recursive_cpu_copy(
                optimizer.state_dict()
            )
        ),
        "python_rng_state": (
            random.getstate()
        ),
        "numpy_rng_state": (
            np.random.get_state()
        ),
        "torch_rng_state": (
            torch.get_rng_state()
            .clone()
        ),
        "cuda_rng_state": (
            torch.cuda.get_rng_state(
                device=0
            )
            .cpu()
            .clone()
        ),
        "strict_cuda": {
            "CUBLAS_WORKSPACE_CONFIG": (
                os.environ.get(
                    "CUBLAS_WORKSPACE_CONFIG"
                )
            ),
            "allow_tf32_matmul": False,
            "allow_tf32_cudnn": False,
            "float32_matmul_precision": (
                torch
                .get_float32_matmul_precision()
            ),
            "deterministic_algorithms": (
                torch
                .are_deterministic_algorithms_enabled()
            ),
        },
        "training_schedule": {
            "epochs": (
                NUM_EPOCHS
            ),
            "positive_events_per_epoch": (
                POSITIVE_EVENTS
            ),
            "examples_per_epoch": (
                EXAMPLES_PER_EPOCH
            ),
            "batches_per_epoch": (
                BATCHES_PER_EPOCH
            ),
            "batch_size": (
                BATCH_SIZE
            ),
            "final_batch_size": (
                FINAL_BATCH_SIZE
            ),
            "checkpoint_interval_batches": (
                CHECKPOINT_INTERVAL_BATCHES
            ),
        },
        "test_boundary": {
            "test_accessed": False,
            "test_scored": False,
        },
    }


def save_latest(
    *,
    model,
    optimizer,
    state,
    roundtrip,
    runtime_bindings,
):
    payload = checkpoint_payload(
        role="LATEST_RESUME",
        model=model,
        optimizer=optimizer,
        state=state,
        roundtrip=roundtrip,
        runtime_bindings=(
            runtime_bindings
        ),
    )

    atomic_torch_save(
        payload,
        LATEST_CHECKPOINT_PATH,
    )

    atomic_json_write(
        progress_payload(
            state=state,
            runtime_bindings=(
                runtime_bindings
            ),
        ),
        PROGRESS_PATH,
    )


def save_best(
    *,
    model,
    optimizer,
    state,
    roundtrip,
    runtime_bindings,
):
    payload = checkpoint_payload(
        role="BEST_VALIDATION",
        model=model,
        optimizer=optimizer,
        state=state,
        roundtrip=roundtrip,
        runtime_bindings=(
            runtime_bindings
        ),
    )

    atomic_torch_save(
        payload,
        BEST_CHECKPOINT_PATH,
    )


# =============================================================================
# CUDA model construction / resume
# =============================================================================

def construct_fresh_cuda_state(
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
        hash_fn(
            model
        )
        == EXPECTED_INITIAL_MODEL_SHA,
        (
            "Portable initial model "
            "SHA drift."
        ),
    )

    del old_optimizer

    model = model.to(
        cuda
    )

    model.startup_embedding.sparse = True
    model.investor_embedding.sparse = True

    model.train()

    optimizer = (
        preflight
        .build_frozen_adam(
            model
        )
    )

    require(
        len(
            optimizer.state
        )
        == 0,
        (
            "Fresh Adam state "
            "must be empty."
        ),
    )

    require(
        hash_fn(
            model
        )
        == EXPECTED_INITIAL_MODEL_SHA,
        (
            "CPU->CUDA changed "
            "initial parameter bytes."
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
        .to(
            cuda
        )
    )

    shared_cuda[
        "edge_type"
    ] = (
        shared_cpu[
            "edge_type"
        ]
        .to(
            cuda
        )
    )

    return (
        model,
        optimizer,
        hash_fn,
        shared_cuda,
    )


def restore_cuda_state(
    *,
    checkpoint,
    roundtrip,
    preflight,
    shared_cpu,
    cuda,
    runtime_bindings,
):
    require(
        checkpoint[
            "schema_version"
        ]
        == (
            "ITRS_PHASE6_100PCT_PRODUCTION_V1"
        ),
        (
            "Latest checkpoint "
            "schema drift."
        ),
    )

    require(
        checkpoint[
            "experiment"
        ]
        == (
            "FULL_100PCT_ITRS_PRODUCTION"
        ),
        (
            "Latest checkpoint "
            "experiment drift."
        ),
    )

    require(
        checkpoint[
            "runtime_bindings"
        ]
        == runtime_bindings,
        (
            "Resume runtime bindings differ "
            "from the latest checkpoint."
        ),
    )

    require(
        checkpoint[
            "test_boundary"
        ][
            "test_accessed"
        ]
        is False
        and checkpoint[
            "test_boundary"
        ][
            "test_scored"
        ]
        is False,
        (
            "Checkpoint test boundary drift."
        ),
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
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    model = model.to(
        cuda
    )

    model.startup_embedding.sparse = True
    model.investor_embedding.sparse = True

    model.train()

    optimizer = (
        preflight
        .build_frozen_adam(
            model
        )
    )

    optimizer.load_state_dict(
        checkpoint[
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
        ]
        .to(
            cuda
        )
    )

    shared_cuda[
        "edge_type"
    ] = (
        shared_cpu[
            "edge_type"
        ]
        .to(
            cuda
        )
    )

    random.setstate(
        checkpoint[
            "python_rng_state"
        ]
    )

    np.random.set_state(
        checkpoint[
            "numpy_rng_state"
        ]
    )

    torch.set_rng_state(
        checkpoint[
            "torch_rng_state"
        ]
    )

    torch.cuda.set_rng_state(
        checkpoint[
            "cuda_rng_state"
        ],
        device=0,
    )

    return (
        model,
        optimizer,
        hash_fn,
        shared_cuda,
        checkpoint[
            "controller_state"
        ],
    )


# =============================================================================
# Qualified packed + sparse training batch
# =============================================================================

def execute_packed_training_batch(
    *,
    model,
    optimizer,
    batch,
    shared_cpu,
    shared_cuda,
    lean_execute,
    packed,
    roundtrip,
    cuda,
):
    plan = (
        packed
        .build_embedding_call_plan(
            batch,
            shared_cpu,
            num_history_periods=(
                roundtrip
                .NUM_HISTORY_PERIODS
            ),
            num_investors=(
                roundtrip
                .NUM_INVESTORS
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

    startup_dispatch = (
        packed
        .DevicePackedEmbeddingDispatch(
            model.startup_embedding,
            startup_calls,
            device=cuda,
        )
    )

    investor_dispatch = (
        packed
        .DevicePackedEmbeddingDispatch(
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

    return {
        "loss": float(
            result[
                "loss"
            ]
        ),
        "seconds": (
            elapsed
        ),
    }


# =============================================================================
# Full 2,251-case validation
# =============================================================================

def full_validation(
    *,
    epoch_index,
    trained_state_dict,
    roundtrip,
    preflight,
    shared_cpu,
    validation_preflight,
    full_validation_runtime,
    ranking_runtime,
    validation_cases,
    validation_candidates,
):
    banner(
        f"EPOCH {epoch_index + 1} — "
        "FULL 2,251-CASE VALIDATION"
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
        trained_state_dict,
        strict=True,
    )

    model.eval()

    state_before = (
        hash_fn(
            model
        )
    )

    start_time = (
        time.perf_counter()
    )

    metric_rows = []

    with torch.no_grad():
        latent_all = torch.cat(
            [
                model.investor_embedding.weight,
                model.startup_embedding.weight,
            ],
            dim=0,
        )

        structural = (
            model
            .preference_propagation(
                latent_all,
                shared_cpu[
                    "edge_index"
                ],
                shared_cpu[
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
                "Validation structural "
                "output invalid."
            ),
        )

        F_s_all = (
            structural[
                "F_s"
            ]
        )

        chunks = (
            full_validation_runtime
            .build_chunk_schedule()
        )

        for (
            chunk_index,
            start,
            end,
        ) in chunks:
            chunk_cases = (
                validation_cases
                .iloc[
                    start:end
                ]
                .copy()
                .reset_index(
                    drop=True
                )
            )

            chunk_candidates = (
                validation_candidates[
                    start:end
                ]
            )

            features = (
                full_validation_runtime
                .compute_chunk_features(
                    preflight_runtime=(
                        validation_preflight
                    ),
                    model=model,
                    chunk_cases=(
                        chunk_cases
                    ),
                    chunk_candidate_matrix_local=(
                        chunk_candidates
                    ),
                    shared=shared_cpu,
                    F_s_all=F_s_all,
                )
            )

            for (
                local_position,
                row,
            ) in (
                chunk_cases
                .iterrows()
            ):
                global_position = (
                    start
                    + int(
                        local_position
                    )
                )

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
                    chunk_candidates[
                        local_position
                    ]
                )

                logits = (
                    validation_preflight
                    .score_validation_case(
                        model,
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
                        "non-finite."
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
                        "epoch_index": (
                            epoch_index
                        ),
                        "display_epoch": (
                            epoch_index + 1
                        ),
                        "validation_case_position": (
                            global_position
                        ),
                        "matrix_row_index": int(
                            row[
                                "matrix_row_index"
                            ]
                        ),
                        "interaction_id": str(
                            row[
                                "interaction_id"
                            ]
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
                        "HR@10": float(
                            hr10
                        ),
                        "NDCG@10": float(
                            ndcg10
                        ),
                        "chunk_index": (
                            chunk_index
                        ),
                    }
                )

            del features

    elapsed = (
        time.perf_counter()
        - start_time
    )

    metrics = pd.DataFrame(
        metric_rows
    )

    require(
        len(
            metrics
        )
        == VALIDATION_CASES,
        (
            "Validation did not score "
            "exactly 2,251 cases."
        ),
    )

    require(
        bool(
            metrics[
                "positive_rank"
            ]
            .between(
                1,
                100,
            )
            .all()
        ),
        (
            "Validation rank outside "
            "1..100."
        ),
    )

    (
        hr10,
        ndcg10,
    ) = (
        ranking_runtime
        .aggregate_event_level_metrics(
            metrics
        )
    )

    state_after = (
        hash_fn(
            model
        )
    )

    require(
        state_after
        == state_before,
        (
            "Validation changed "
            "model parameters."
        ),
    )

    require(
        all(
            parameter.grad is None
            for parameter
            in model.parameters()
        ),
        (
            "Validation created "
            "gradients."
        ),
    )

    hit_count = int(
        metrics[
            "HR@10"
        ].sum()
    )

    mean_rank = float(
        metrics[
            "positive_rank"
        ].mean()
    )

    median_rank = float(
        metrics[
            "positive_rank"
        ].median()
    )

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_path = (
        VALIDATION_DIR
        / (
            f"epoch_{epoch_index:02d}_"
            "validation_cases.parquet"
        )
    )

    metrics.to_parquet(
        validation_path,
        index=False,
    )

    del F_s_all
    del model
    gc.collect()

    print(
        f"Validation HR@10:             "
        f"{hr10:.12f}"
    )

    print(
        f"Validation NDCG@10:           "
        f"{ndcg10:.12f}"
    )

    print(
        f"Hits @10:                     "
        f"{hit_count:,} / {VALIDATION_CASES:,}"
    )

    print(
        f"Mean positive rank:            "
        f"{mean_rank:.6f}"
    )

    print(
        f"Median positive rank:          "
        f"{median_rank:.6f}"
    )

    print(
        f"Validation time:               "
        f"{elapsed:.2f} s"
    )

    print(
        "Test cases scored:             0"
    )

    return {
        "HR@10": float(
            hr10
        ),
        "NDCG@10": float(
            ndcg10
        ),
        "hit_count": (
            hit_count
        ),
        "mean_positive_rank": (
            mean_rank
        ),
        "median_positive_rank": (
            median_rank
        ),
        "validation_seconds": (
            elapsed
        ),
        "case_metrics_path": str(
            validation_path
        ),
    }


def write_epoch_history(
    history,
):
    RUN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = Path(
        str(
            EPOCH_METRICS_PATH
        )
        + ".tmp"
    )

    pd.DataFrame(
        history
    ).to_csv(
        temporary,
        index=False,
    )

    os.replace(
        temporary,
        EPOCH_METRICS_PATH,
    )


# =============================================================================
# Phase 6.9a + validation prerequisite gate
# =============================================================================

def prerequisite_gate(
    *,
    preflight,
):
    banner(
        "PHASE 6.9b PREREQUISITE GATE"
    )

    for path in (
        PHASE6_2_CONTRACT_PATH,
        PHASE6_9A_CONTRACT_PATH,
        POSITIVE_PATH,
        STREAM_REGISTRY_PATH,
        VALIDATION_CASES_PATH,
        VALIDATION_CANDIDATES_PATH,
    ):
        require(
            path.exists(),
            (
                "Missing prerequisite: "
                f"{path}"
            ),
        )

    q2 = load_json(
        PHASE6_2_CONTRACT_PATH
    )

    q9a = load_json(
        PHASE6_9A_CONTRACT_PATH
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
        == 22
        and int(
            q2[
                "policy_checks_total"
            ]
        )
        == 22,
        (
            "Phase 6.2 numerical "
            "policy is not 22/22."
        ),
    )

    require(
        q9a[
            "status"
        ]
        == (
            "FROZEN_FOR_100PCT_PRODUCTION"
        ),
        (
            "Phase 6.9a full-data "
            "stream freeze is not production-ready."
        ),
    )

    require(
        q9a[
            "registry"
        ][
            "file_sha256"
        ]
        == EXPECTED_STREAM_REGISTRY_FILE_SHA256,
        (
            "Phase 6.9a contract registry "
            "binding drift."
        ),
    )

    require(
        q9a[
            "positive_stream"
        ][
            "logical_sha256"
        ]
        == EXPECTED_POSITIVE_SHA,
        (
            "Phase 6.9a positive stream "
            "binding drift."
        ),
    )

    require(
        q9a[
            "verification"
        ][
            "binary_stream_files"
        ]
        == "40/40 PASS",
        (
            "Phase 6.9a binary stream "
            "verification not complete."
        ),
    )

    require(
        q9a[
            "verification"
        ][
            "epoch_logical_hashes"
        ]
        == "20/20 PASS",
        (
            "Phase 6.9a logical "
            "fingerprint verification not complete."
        ),
    )

    require(
        q9a[
            "boundary"
        ][
            "optimizer_steps"
        ]
        == 0
        and not bool(
            q9a[
                "boundary"
            ][
                "test_accessed"
            ]
        ),
        (
            "Phase 6.9a pretraining/test "
            "boundary drift."
        ),
    )

    actual_registry_file_sha = (
        file_sha256(
            STREAM_REGISTRY_PATH
        )
    )

    require(
        actual_registry_file_sha
        == EXPECTED_STREAM_REGISTRY_FILE_SHA256,
        (
            "Persisted stream registry "
            "file SHA drift."
        ),
    )

    positive_order = (
        pd.read_parquet(
            POSITIVE_PATH
        )
    )

    require(
        len(
            positive_order
        )
        == POSITIVE_EVENTS,
        (
            "100% positive event "
            "count drift."
        ),
    )

    positive_sha = (
        preflight
        .positive_stream_logical_sha256(
            positive_order
        )
    )

    require(
        positive_sha
        == EXPECTED_POSITIVE_SHA,
        (
            "100% positive-order "
            "logical SHA drift."
        ),
    )

    stream_registry = pd.read_csv(
        STREAM_REGISTRY_PATH
    )

    require(
        len(
            stream_registry
        )
        == NUM_EPOCHS,
        (
            "Stream registry does not "
            "contain 20 epochs."
        ),
    )

    require(
        stream_registry[
            "epoch_index"
        ].tolist()
        == list(
            range(
                NUM_EPOCHS
            )
        ),
        (
            "Stream registry epoch "
            "ordering drift."
        ),
    )

    for epoch in range(
        NUM_EPOCHS
    ):
        row = (
            stream_registry
            .iloc[
                epoch
            ]
        )

        negative_path = (
            epoch_negative_path(
                epoch
            )
        )

        order_path = (
            epoch_order_path(
                epoch
            )
        )

        require(
            negative_path.exists(),
            (
                "Missing epoch negative "
                f"matrix: {epoch}"
            ),
        )

        require(
            order_path.exists(),
            (
                "Missing epoch order: "
                f"{epoch}"
            ),
        )

        negative = np.load(
            negative_path,
            mmap_mode="r",
        )

        order = np.load(
            order_path,
            mmap_mode="r",
        )

        require(
            negative.shape
            == (
                POSITIVE_EVENTS,
                NEGATIVES_PER_POSITIVE,
            ),
            (
                "Negative shape drift "
                f"at epoch {epoch}."
            ),
        )

        require(
            negative.dtype
            == np.dtype(
                np.int32
            ),
            (
                "Negative dtype drift "
                f"at epoch {epoch}."
            ),
        )

        require(
            order.shape
            == (
                EXAMPLES_PER_EPOCH,
            ),
            (
                "Order shape drift "
                f"at epoch {epoch}."
            ),
        )

        require(
            order.dtype
            == np.dtype(
                np.int64
            ),
            (
                "Order dtype drift "
                f"at epoch {epoch}."
            ),
        )

        require(
            preflight
            .array_logical_sha256(
                np.asarray(
                    negative
                )
            )
            == str(
                row[
                    "negative_sha256"
                ]
            ),
            (
                "Negative logical SHA drift "
                f"at epoch {epoch}."
            ),
        )

        require(
            preflight
            .array_logical_sha256(
                np.asarray(
                    order
                )
            )
            == str(
                row[
                    "order_sha256"
                ]
            ),
            (
                "Order logical SHA drift "
                f"at epoch {epoch}."
            ),
        )

    validation_cases = (
        pd.read_parquet(
            VALIDATION_CASES_PATH
        )
    )

    validation_candidates = (
        np.load(
            VALIDATION_CANDIDATES_PATH,
            mmap_mode="r",
        )
    )

    require(
        len(
            validation_cases
        )
        == VALIDATION_CASES,
        (
            "Validation case count "
            "drift."
        ),
    )

    require(
        validation_candidates.shape
        == (
            VALIDATION_CASES,
            CANDIDATES_PER_CASE,
        ),
        (
            "Validation candidate "
            "matrix shape drift."
        ),
    )

    print(
        "Phase 6.2 CUDA qualification:          PASS / 22/22"
    )
    print(
        "Phase 6.9a 100% stream freeze:         PASS"
    )
    print(
        "100% positive logical SHA:             PASS"
    )
    print(
        "Persisted registry byte SHA:           PASS"
    )
    print(
        "20 epoch stream fingerprints:          PASS"
    )
    print(
        "2,251 validation cases:                PASS"
    )
    print(
        "Test split accessed:                   NO"
    )

    return (
        positive_order,
        stream_registry,
        validation_cases,
        validation_candidates,
    )


# =============================================================================
# Full-data stream loader
# =============================================================================

def load_epoch_stream(
    *,
    epoch,
    positive_order,
    stream_registry,
    preflight,
):
    row = (
        stream_registry
        .iloc[
            epoch
        ]
    )

    require(
        int(
            row[
                "epoch_index"
            ]
        )
        == epoch,
        (
            "Registry row / epoch "
            "mismatch."
        ),
    )

    negative_path = (
        epoch_negative_path(
            epoch
        )
    )

    order_path = (
        epoch_order_path(
            epoch
        )
    )

    negative_matrix = np.load(
        negative_path,
        mmap_mode="r",
    )

    epoch_order = np.load(
        order_path,
        mmap_mode="r",
    )

    negative_sha = (
        preflight
        .array_logical_sha256(
            np.asarray(
                negative_matrix
            )
        )
    )

    order_sha = (
        preflight
        .array_logical_sha256(
            np.asarray(
                epoch_order
            )
        )
    )

    require(
        negative_sha
        == str(
            row[
                "negative_sha256"
            ]
        ),
        (
            "Current epoch negative "
            "fingerprint drift."
        ),
    )

    require(
        order_sha
        == str(
            row[
                "order_sha256"
            ]
        ),
        (
            "Current epoch order "
            "fingerprint drift."
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
        "negative_sha": (
            negative_sha
        ),
        "order_sha": (
            order_sha
        ),
    }


# =============================================================================
# Final result writer
# =============================================================================

def write_final_contract(
    *,
    state,
    runtime_bindings,
):
    require(
        len(
            state[
                "validation_history"
            ]
        )
        == NUM_EPOCHS,
        (
            "Training complete without "
            "20 validation commits."
        ),
    )

    require(
        state[
            "global_optimizer_step"
        ]
        == TOTAL_OPTIMIZER_STEPS,
        (
            "Final optimizer-step "
            "count drift."
        ),
    )

    require(
        BEST_CHECKPOINT_PATH.exists(),
        (
            "Best validation checkpoint "
            "missing."
        ),
    )

    require(
        LATEST_CHECKPOINT_PATH.exists(),
        (
            "Latest/final checkpoint "
            "missing."
        ),
    )

    require(
        state[
            "test_accessed"
        ]
        is False
        and state[
            "test_scored"
        ]
        is False,
        (
            "Controller test boundary drift."
        ),
    )

    result_contract = {
        "phase": "6.9b",
        "status": (
            "COMPLETE_100PCT_20EPOCH_"
            "PRODUCTION_VALIDATION_ONLY"
        ),
        "experiment": (
            "FULL_100PCT_ITRS_PRODUCTION"
        ),
        "repository_commit": (
            runtime_bindings[
                "repository_commit"
            ]
        ),
        "runtime_bindings": (
            runtime_bindings
        ),
        "training": {
            "epochs": (
                NUM_EPOCHS
            ),
            "positive_events_per_epoch": (
                POSITIVE_EVENTS
            ),
            "negatives_per_positive": (
                NEGATIVES_PER_POSITIVE
            ),
            "examples_per_epoch": (
                EXAMPLES_PER_EPOCH
            ),
            "batches_per_epoch": (
                BATCHES_PER_EPOCH
            ),
            "batch_size": (
                BATCH_SIZE
            ),
            "final_batch_size": (
                FINAL_BATCH_SIZE
            ),
            "optimizer_steps": (
                TOTAL_OPTIMIZER_STEPS
            ),
            "checkpoint_interval_batches": (
                CHECKPOINT_INTERVAL_BATCHES
            ),
            "decoder": (
                "canonical_phase_5_3_2b_full_decode_batch"
            ),
            "qualified_cuda_path": (
                "packed_embedding_dispatch_plus_D_BOTH_SPARSE"
            ),
        },
        "stream_freeze": {
            "phase6_9a_contract": str(
                PHASE6_9A_CONTRACT_PATH
            ),
            "phase6_9a_contract_file_sha256": (
                runtime_bindings[
                    "phase6_9a_contract_file_sha256"
                ]
            ),
            "stream_registry": str(
                STREAM_REGISTRY_PATH
            ),
            "stream_registry_file_sha256": (
                runtime_bindings[
                    "stream_registry_file_sha256"
                ]
            ),
            "positive_logical_sha256": (
                EXPECTED_POSITIVE_SHA
            ),
        },
        "initial_validation_baseline": {
            "HR@10": (
                BASELINE_VALIDATION_HR10
            ),
            "NDCG@10": (
                BASELINE_VALIDATION_NDCG10
            ),
        },
        "selection": {
            "rule": (
                "maximum validation NDCG@10, "
                "then HR@10, then earliest epoch"
            ),
            "best_epoch_index": (
                state[
                    "best_validation_epoch"
                ]
            ),
            "best_display_epoch": (
                int(
                    state[
                        "best_validation_epoch"
                    ]
                )
                + 1
            ),
            "best_validation_NDCG@10": (
                state[
                    "best_validation_ndcg10"
                ]
            ),
            "best_validation_HR@10": (
                state[
                    "best_validation_hr10"
                ]
            ),
        },
        "validation_history": (
            state[
                "validation_history"
            ]
        ),
        "test_boundary": {
            "test_accessed": False,
            "test_scored": False,
            "test_metrics_reported": False,
            "test_selection_influence": False,
            "reason": (
                "The full 20-epoch run is validation-selected. "
                "Final test remains withheld until a separate "
                "post-training integrity gate authorizes one "
                "evaluation of the frozen best checkpoint."
            ),
        },
        "artifacts": {
            "latest_checkpoint": str(
                LATEST_CHECKPOINT_PATH
            ),
            "best_checkpoint": str(
                BEST_CHECKPOINT_PATH
            ),
            "epoch_metrics": str(
                EPOCH_METRICS_PATH
            ),
            "progress": str(
                PROGRESS_PATH
            ),
        },
        "next_phase": {
            "id": "6.9c",
            "role": (
                "post-training closure and one-shot "
                "final-test authorization"
            ),
        },
    }

    atomic_json_write(
        result_contract,
        FINAL_CONTRACT_PATH,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    mode = (
        parser
        .add_mutually_exclusive_group(
            required=True
        )
    )

    mode.add_argument(
        "--preflight-only",
        action="store_true",
    )

    mode.add_argument(
        "--fresh",
        action="store_true",
    )

    mode.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        default=None,
        help=(
            "Optional safe stop after this many "
            "completed + validated display epochs."
        ),
    )

    args = parser.parse_args()

    banner(
        "PHASE 6.9b — "
        "100% 20-EPOCH CUDA PRODUCTION TRAINER V1"
    )

    print(
        f"Positive events / epoch:       "
        f"{POSITIVE_EVENTS:,}"
    )

    print(
        f"Negatives / positive:          "
        f"{NEGATIVES_PER_POSITIVE}"
    )

    print(
        f"Examples / epoch:              "
        f"{EXAMPLES_PER_EPOCH:,}"
    )

    print(
        f"Batches / epoch:               "
        f"{BATCHES_PER_EPOCH:,}"
    )

    print(
        f"Final batch size:              "
        f"{FINAL_BATCH_SIZE}"
    )

    print(
        f"Epochs:                        "
        f"{NUM_EPOCHS}"
    )

    print(
        f"Total optimizer steps:         "
        f"{TOTAL_OPTIMIZER_STEPS:,}"
    )

    print(
        f"Checkpoint interval:           "
        f"{CHECKPOINT_INTERVAL_BATCHES} batches"
    )

    print(
        f"Validation cases / epoch:      "
        f"{VALIDATION_CASES:,}"
    )

    print(
        "Test evaluation:               DISABLED"
    )

    configure_cuda()

    cuda = torch.device(
        "cuda:0"
    )

    # =========================================================================
    # Load frozen/qualified infrastructure
    # =========================================================================

    q = load_module(
        PHASE6_BASE_PATH,
        "_phase6_9b_base",
    )

    packed = load_module(
        PACKED_RUNTIME_PATH,
        "_phase6_9b_packed",
    )

    validation_preflight = (
        load_module(
            VALIDATION_PREFLIGHT_PATH,
            "_phase6_9b_validation_preflight",
        )
    )

    full_validation_runtime = (
        load_module(
            FULL_VALIDATION_RUNTIME_PATH,
            "_phase6_9b_full_validation",
        )
    )

    ranking_runtime = (
        load_module(
            RANKING_RUNTIME_PATH,
            "_phase6_9b_ranking",
        )
    )

    selection_runtime = (
        load_module(
            SELECTION_RUNTIME_PATH,
            "_phase6_9b_selection",
        )
    )

    require(
        q.file_sha256(
            q.INITIAL_STATE_PATH
        )
        == q.EXPECTED_INITIAL_FILE_SHA,
        (
            "Portable initial-state "
            "file SHA drift."
        ),
    )

    initial_payload = torch.load(
        q.INITIAL_STATE_PATH,
        map_location="cpu",
        weights_only=False,
    )

    roundtrip = q.load_module(
        q.ROUNDTRIP_PATH,
        "_phase6_9b_roundtrip",
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
        (
            "Lean executor guard "
            "count drift."
        ),
    )

    require(
        executor_metadata[
            "torch_from_numpy_rewrites"
        ]
        == 10,
        (
            "Lean executor device "
            "rewrite drift."
        ),
    )

    set_executor_device(
        "cuda:0"
    )

    shared_cpu = (
        roundtrip
        .load_shared_inputs(
            preflight
        )
    )

    (
        positive_order,
        stream_registry,
        validation_cases,
        validation_candidates,
    ) = prerequisite_gate(
        preflight=preflight
    )

    repository_commit = git_head()

    require(
        repository_commit
        == EXPECTED_REPOSITORY_HEAD,
        (
            "Repository HEAD drift. "
            f"Expected {EXPECTED_REPOSITORY_HEAD}, "
            f"got {repository_commit}."
        ),
    )

    runtime_bindings = make_runtime_bindings(
        repository_commit=(
            repository_commit
        )
    )

    require(
        runtime_bindings[
            "stream_registry_file_sha256"
        ]
        == EXPECTED_STREAM_REGISTRY_FILE_SHA256,
        (
            "Runtime stream registry "
            "binding drift."
        ),
    )

    # =========================================================================
    # Portable model smoke gate
    # =========================================================================

    (
        smoke_model,
        smoke_optimizer,
        smoke_hash_fn,
        smoke_shared_cuda,
    ) = (
        construct_fresh_cuda_state(
            roundtrip=roundtrip,
            preflight=preflight,
            shared_cpu=shared_cpu,
            cuda=cuda,
        )
    )

    require(
        smoke_hash_fn(
            smoke_model
        )
        == EXPECTED_INITIAL_MODEL_SHA,
        (
            "Fresh CUDA model "
            "smoke gate failed."
        ),
    )

    del smoke_model
    del smoke_optimizer
    del smoke_shared_cuda

    gc.collect()
    torch.cuda.empty_cache()

    print()
    print(
        f"Repository HEAD:               "
        f"{repository_commit}"
    )

    print(
        f"Trainer script SHA256:         "
        f"{runtime_bindings['trainer_script_file_sha256']}"
    )

    print(
        f"Phase 6.9a contract SHA256:    "
        f"{runtime_bindings['phase6_9a_contract_file_sha256']}"
    )

    print(
        f"Stream registry SHA256:        "
        f"{runtime_bindings['stream_registry_file_sha256']}"
    )

    print(
        f"GPU:                           "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"PyTorch:                       "
        f"{torch.__version__}"
    )

    print(
        f"CUDA runtime:                  "
        f"{torch.version.cuda}"
    )

    # =========================================================================
    # Preflight-only boundary
    # =========================================================================

    if args.preflight_only:
        banner(
            "PHASE 6.9b PREFLIGHT RESULT"
        )

        print(
            "Phase 6.9a full stream freeze: PASS"
        )
        print(
            "All 20 full streams:           PASS"
        )
        print(
            "Canonical full decoder:        READY"
        )
        print(
            "Portable initial model:        PASS"
        )
        print(
            "Qualified CUDA runtime:        PASS"
        )
        print(
            "Validation artifacts:          PASS"
        )
        print(
            "Optimizer steps executed:      0"
        )
        print(
            "Validation cases scored:       0"
        )
        print(
            "Test cases scored:             0"
        )
        print()
        print(
            "PHASE 6.9b PREFLIGHT: PASS"
        )

        return

    # Actual optimizer-bearing run must preserve frozen tracked sources.
    require_clean_tracked_tree()

    # =========================================================================
    # Fresh or resume
    # =========================================================================

    if args.fresh:
        for path in (
            LATEST_CHECKPOINT_PATH,
            BEST_CHECKPOINT_PATH,
            EPOCH_METRICS_PATH,
            PROGRESS_PATH,
            FINAL_CONTRACT_PATH,
        ):
            require(
                not path.exists(),
                (
                    "Fresh production run refuses "
                    f"existing artifact: {path}"
                ),
            )

        RUN_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        CHECKPOINT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        VALIDATION_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            model,
            optimizer,
            hash_fn,
            shared_cuda,
        ) = (
            construct_fresh_cuda_state(
                roundtrip=roundtrip,
                preflight=preflight,
                shared_cpu=shared_cpu,
                cuda=cuda,
            )
        )

        state = make_fresh_state()

        print()
        print(
            "Run mode:                      FRESH"
        )

    else:
        require(
            LATEST_CHECKPOINT_PATH.exists(),
            (
                "Cannot --resume: "
                "latest checkpoint missing."
            ),
        )

        checkpoint = torch.load(
            LATEST_CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=False,
        )

        (
            model,
            optimizer,
            hash_fn,
            shared_cuda,
            state,
        ) = (
            restore_cuda_state(
                checkpoint=checkpoint,
                roundtrip=roundtrip,
                preflight=preflight,
                shared_cpu=shared_cpu,
                cuda=cuda,
                runtime_bindings=(
                    runtime_bindings
                ),
            )
        )

        print()
        print(
            "Run mode:                      RESUME"
        )

        print(
            f"Resume epoch:                  "
            f"{int(state['epoch_index']) + 1}"
        )

        print(
            f"Resume batch:                  "
            f"{state['next_batch_index']}"
        )

        print(
            f"Global optimizer step:         "
            f"{state['global_optimizer_step']:,}"
        )

        print(
            f"Completed validations:         "
            f"{len(state['validation_history'])}"
        )

        print(
            f"Training complete:             "
            f"{state['training_complete']}"
        )

    # =========================================================================
    # Training epochs
    # =========================================================================

    while not state[
        "training_complete"
    ]:
        epoch = int(
            state[
                "epoch_index"
            ]
        )

        require(
            0
            <= epoch
            < NUM_EPOCHS,
            "Epoch state outside 0..19.",
        )

        banner(
            f"EPOCH {epoch + 1}/{NUM_EPOCHS}"
        )

        stream = load_epoch_stream(
            epoch=epoch,
            positive_order=(
                positive_order
            ),
            stream_registry=(
                stream_registry
            ),
            preflight=preflight,
        )

        print(
            f"negative_sha256:               "
            f"{stream['negative_sha']}"
        )

        print(
            f"order_sha256:                  "
            f"{stream['order_sha']}"
        )

        # ---------------------------------------------------------------------
        # Train remaining batches unless validation was already pending.
        # ---------------------------------------------------------------------

        if not state[
            "validation_pending"
        ]:
            start_batch = int(
                state[
                    "next_batch_index"
                ]
            )

            require(
                0
                <= start_batch
                <= BATCHES_PER_EPOCH,
                (
                    "Resume batch outside "
                    "full epoch."
                ),
            )

            for batch_index in range(
                start_batch,
                BATCHES_PER_EPOCH,
            ):
                batch = (
                    roundtrip
                    .decode_batch(
                        stream,
                        batch_index,
                    )
                )

                result = (
                    execute_packed_training_batch(
                        model=model,
                        optimizer=optimizer,
                        batch=batch,
                        shared_cpu=shared_cpu,
                        shared_cuda=shared_cuda,
                        lean_execute=lean_execute,
                        packed=packed,
                        roundtrip=roundtrip,
                        cuda=cuda,
                    )
                )

                batch_size = len(
                    batch
                )

                state[
                    "epoch_loss_weighted_sum"
                ] += (
                    result[
                        "loss"
                    ]
                    * batch_size
                )

                state[
                    "epoch_example_count"
                ] += (
                    batch_size
                )

                state[
                    "epoch_training_seconds"
                ] += (
                    result[
                        "seconds"
                    ]
                )

                state[
                    "global_optimizer_step"
                ] += 1

                state[
                    "next_batch_index"
                ] = (
                    batch_index + 1
                )

                if (
                    batch_index == 0
                    or (
                        batch_index + 1
                    )
                    % PROGRESS_PRINT_INTERVAL_BATCHES
                    == 0
                    or (
                        batch_index + 1
                    )
                    == BATCHES_PER_EPOCH
                ):
                    running_loss = (
                        state[
                            "epoch_loss_weighted_sum"
                        ]
                        / state[
                            "epoch_example_count"
                        ]
                    )

                    print(
                        f"epoch={epoch + 1:02d} "
                        f"batch={batch_index + 1:05d}/"
                        f"{BATCHES_PER_EPOCH} "
                        f"step="
                        f"{state['global_optimizer_step']:06d}/"
                        f"{TOTAL_OPTIMIZER_STEPS} "
                        f"loss={result['loss']:.8f} "
                        f"mean={running_loss:.8f} "
                        f"time={result['seconds']:.3f}s"
                    )

                # Periodic atomic restart point.
                if (
                    (
                        batch_index + 1
                    )
                    % CHECKPOINT_INTERVAL_BATCHES
                    == 0
                    and (
                        batch_index + 1
                    )
                    < BATCHES_PER_EPOCH
                ):
                    save_latest(
                        model=model,
                        optimizer=optimizer,
                        state=state,
                        roundtrip=roundtrip,
                        runtime_bindings=(
                            runtime_bindings
                        ),
                    )

                    print(
                        "  checkpoint: latest.pt "
                        f"after batch {batch_index + 1}"
                    )

            require(
                state[
                    "next_batch_index"
                ]
                == BATCHES_PER_EPOCH,
                (
                    "Epoch ended before "
                    "all epoch batches."
                ),
            )

            require(
                state[
                    "epoch_example_count"
                ]
                == EXAMPLES_PER_EPOCH,
                (
                    "Epoch example count "
                    "drift."
                ),
            )

            state[
                "validation_pending"
            ] = True

            # Critical restart point:
            # interruption during validation reruns validation only.
            save_latest(
                model=model,
                optimizer=optimizer,
                state=state,
                roundtrip=roundtrip,
                runtime_bindings=(
                    runtime_bindings
                ),
            )

            print(
                "  checkpoint: latest.pt "
                "at end-of-epoch / pre-validation boundary"
            )

        # ---------------------------------------------------------------------
        # Full frozen validation
        # ---------------------------------------------------------------------

        require(
            state[
                "validation_pending"
            ]
            is True,
            (
                "Validation requires "
                "validation_pending=True."
            ),
        )

        require(
            state[
                "epoch_example_count"
            ]
            == EXAMPLES_PER_EPOCH,
            (
                "Validation requires "
                "full epoch example count."
            ),
        )

        epoch_train_loss = (
            state[
                "epoch_loss_weighted_sum"
            ]
            / state[
                "epoch_example_count"
            ]
        )

        trained_state_cpu = (
            roundtrip
            .clone_model_state_dict(
                model
            )
        )

        validation = (
            full_validation(
                epoch_index=epoch,
                trained_state_dict=(
                    trained_state_cpu
                ),
                roundtrip=roundtrip,
                preflight=preflight,
                shared_cpu=shared_cpu,
                validation_preflight=(
                    validation_preflight
                ),
                full_validation_runtime=(
                    full_validation_runtime
                ),
                ranking_runtime=(
                    ranking_runtime
                ),
                validation_cases=(
                    validation_cases
                ),
                validation_candidates=(
                    validation_candidates
                ),
            )
        )

        del trained_state_cpu
        gc.collect()

        better = (
            selection_runtime
            .validation_candidate_is_better(
                candidate_ndcg=(
                    validation[
                        "NDCG@10"
                    ]
                ),
                candidate_hr=(
                    validation[
                        "HR@10"
                    ]
                ),
                candidate_epoch=(
                    epoch
                ),
                best_ndcg=(
                    state[
                        "best_validation_ndcg10"
                    ]
                ),
                best_hr=(
                    state[
                        "best_validation_hr10"
                    ]
                ),
                best_epoch=(
                    state[
                        "best_validation_epoch"
                    ]
                ),
            )
        )

        history_row = {
            "epoch_index": epoch,
            "display_epoch": (
                epoch + 1
            ),
            "training_loss": float(
                epoch_train_loss
            ),
            "training_seconds": float(
                state[
                    "epoch_training_seconds"
                ]
            ),
            "validation_HR@10": (
                validation[
                    "HR@10"
                ]
            ),
            "validation_NDCG@10": (
                validation[
                    "NDCG@10"
                ]
            ),
            "validation_hit_count": (
                validation[
                    "hit_count"
                ]
            ),
            "mean_positive_rank": (
                validation[
                    "mean_positive_rank"
                ]
            ),
            "median_positive_rank": (
                validation[
                    "median_positive_rank"
                ]
            ),
            "validation_seconds": (
                validation[
                    "validation_seconds"
                ]
            ),
            "became_best": bool(
                better
            ),
            "global_optimizer_step": int(
                state[
                    "global_optimizer_step"
                ]
            ),
            "negative_sha256": (
                stream[
                    "negative_sha"
                ]
            ),
            "order_sha256": (
                stream[
                    "order_sha"
                ]
            ),
            "test_cases_scored": 0,
        }

        state[
            "validation_history"
        ] = (
            list(
                state[
                    "validation_history"
                ]
            )
            + [
                history_row
            ]
        )

        # Validation metrics are now committed to this epoch.
        state[
            "validation_pending"
        ] = False

        if better:
            state[
                "best_validation_epoch"
            ] = epoch

            state[
                "best_validation_ndcg10"
            ] = (
                validation[
                    "NDCG@10"
                ]
            )

            state[
                "best_validation_hr10"
            ] = (
                validation[
                    "HR@10"
                ]
            )

            save_best(
                model=model,
                optimizer=optimizer,
                state=state,
                roundtrip=roundtrip,
                runtime_bindings=(
                    runtime_bindings
                ),
            )

        write_epoch_history(
            state[
                "validation_history"
            ]
        )

        banner(
            f"EPOCH {epoch + 1} RESULT"
        )

        print(
            f"Training BCE loss:             "
            f"{epoch_train_loss:.10f}"
        )

        print(
            f"Training GPU seconds:          "
            f"{state['epoch_training_seconds']:.2f}"
        )

        print(
            f"Validation HR@10:              "
            f"{validation['HR@10']:.12f}"
        )

        print(
            f"Validation NDCG@10:            "
            f"{validation['NDCG@10']:.12f}"
        )

        print(
            f"Initial baseline HR@10:        "
            f"{BASELINE_VALIDATION_HR10:.12f}"
        )

        print(
            f"Initial baseline NDCG@10:      "
            f"{BASELINE_VALIDATION_NDCG10:.12f}"
        )

        print(
            f"Delta HR@10 vs initialization: "
            f"{validation['HR@10'] - BASELINE_VALIDATION_HR10:+.12f}"
        )

        print(
            f"Delta NDCG vs initialization:  "
            f"{validation['NDCG@10'] - BASELINE_VALIDATION_NDCG10:+.12f}"
        )

        print(
            "New best checkpoint:           "
            + (
                "YES"
                if better
                else "NO"
            )
        )

        print(
            f"Best epoch so far:             "
            f"{int(state['best_validation_epoch']) + 1}"
        )

        print(
            f"Best validation NDCG@10:       "
            f"{state['best_validation_ndcg10']:.12f}"
        )

        print(
            f"Best validation HR@10:         "
            f"{state['best_validation_hr10']:.12f}"
        )

        print(
            "Test cases scored:             0"
        )

        # ---------------------------------------------------------------------
        # Advance controller only AFTER validation commit.
        # ---------------------------------------------------------------------

        if epoch < (
            NUM_EPOCHS - 1
        ):
            state[
                "epoch_index"
            ] = (
                epoch + 1
            )

            state[
                "next_batch_index"
            ] = 0

            state[
                "epoch_loss_weighted_sum"
            ] = 0.0

            state[
                "epoch_example_count"
            ] = 0

            state[
                "epoch_training_seconds"
            ] = 0.0

            state[
                "validation_pending"
            ] = False

            state[
                "training_complete"
            ] = False

        else:
            state[
                "epoch_index"
            ] = epoch

            state[
                "next_batch_index"
            ] = (
                BATCHES_PER_EPOCH
            )

            state[
                "training_complete"
            ] = True

        # Latest checkpoint is the next exact resumable state.
        save_latest(
            model=model,
            optimizer=optimizer,
            state=state,
            roundtrip=roundtrip,
            runtime_bindings=(
                runtime_bindings
            ),
        )

        # Safe requested stop after completed validation only.
        if (
            args.stop_after_epoch
            is not None
            and len(
                state[
                    "validation_history"
                ]
            )
            >= args.stop_after_epoch
            and not state[
                "training_complete"
            ]
        ):
            banner(
                "SAFE USER-REQUESTED STOP"
            )

            print(
                f"Completed + validated epochs:  "
                f"{len(state['validation_history'])}"
            )

            print(
                f"Next epoch:                    "
                f"{int(state['epoch_index']) + 1}"
            )

            print(
                "Next batch:                    0"
            )

            print(
                "Resume mode:                   --resume"
            )

            print(
                "Test cases scored:             0"
            )

            return

        # Explicitly release mmaps from completed epoch before next stream.
        del stream
        gc.collect()

    # =========================================================================
    # Final validation-only experimental result — still NO TEST
    # =========================================================================

    write_final_contract(
        state=state,
        runtime_bindings=(
            runtime_bindings
        ),
    )

    banner(
        "PHASE 6.9b FINAL STATUS"
    )

    print(
        "Training epochs completed:             20 / 20"
    )

    print(
        f"Optimizer steps completed:             "
        f"{state['global_optimizer_step']:,} / "
        f"{TOTAL_OPTIMIZER_STEPS:,}"
    )

    print(
        "Validation evaluations completed:      20 / 20"
    )

    print(
        f"Best epoch:                            "
        f"{int(state['best_validation_epoch']) + 1}"
    )

    print(
        f"Best validation NDCG@10:               "
        f"{state['best_validation_ndcg10']:.12f}"
    )

    print(
        f"Best validation HR@10:                 "
        f"{state['best_validation_hr10']:.12f}"
    )

    print(
        "Best checkpoint saved:                 YES"
    )

    print(
        "Latest/final checkpoint saved:         YES"
    )

    print(
        "Test cases accessed/scored:             0"
    )

    print(
        "Test intentionally withheld:            YES"
    )

    print()
    print(
        f"WROTE  {EPOCH_METRICS_PATH}"
    )

    print(
        f"WROTE  {BEST_CHECKPOINT_PATH}"
    )

    print(
        f"WROTE  {LATEST_CHECKPOINT_PATH}"
    )

    print(
        f"WROTE  {FINAL_CONTRACT_PATH}"
    )

    print()
    print(
        "PHASE 6.9b: COMPLETE / "
        "100% 20-EPOCH PRODUCTION TRAINED + "
        "VALIDATED / TEST UNTOUCHED"
    )


if __name__ == "__main__":
    main()