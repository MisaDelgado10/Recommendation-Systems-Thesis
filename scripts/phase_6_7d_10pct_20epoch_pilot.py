#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
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
POSITIVE_EVENTS = 107_324
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

CHECKPOINT_INTERVAL_BATCHES = 25

EXPECTED_SUBSET_SHA = (
    "bc0aef0b9f840a58c84e32c1a50fc14a"
    "d69a76e688cbe4697dafb7e91d25bddf"
)

EXPECTED_INITIAL_MODEL_SHA = (
    "49e822ea7fad35c458f47e134c94c05e"
    "ac099b68c5c468e2c71559c8c88998ab"
)

BASELINE_VALIDATION_HR10 = (
    0.09151488227454465
)

BASELINE_VALIDATION_NDCG10 = (
    0.040193099163
)

VALIDATION_CASES = 2_251
CANDIDATES_PER_CASE = 100


# =============================================================================
# Sources
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

REDUCED_RUNTIME_PATH = Path(
    "scripts/"
    "phase_6_7c_10pct_bounded_trainer_integration.py"
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
# Prerequisite contracts
# =============================================================================

PHASE6_2_CONTRACT_PATH = Path(
    "data/experimental/phase_6/contracts/"
    "phase_6_2_packed_sparse_cuda_strict_fp32_qualification.json"
)

PHASE6_7A_CONTRACT_PATH = Path(
    "data/experimental/phase_6/contracts/"
    "phase_6_7a_reduced_training_10pct_contract.json"
)

PHASE6_7B_CONTRACT_PATH = Path(
    "data/experimental/phase_6/contracts/"
    "phase_6_7b_10pct_filtered_epoch_stream_contract.json"
)

PHASE6_7C_CONTRACT_PATH = Path(
    "data/experimental/phase_6/contracts/"
    "phase_6_7c_10pct_bounded_trainer_integration.json"
)


# =============================================================================
# Frozen 10% stream
# =============================================================================

POSITIVE_PATH = Path(
    "data/experimental/phase_6/"
    "reduced_training/10pct/"
    "positive_order_10pct.parquet"
)

STREAM_DIR = Path(
    "data/experimental/phase_6/"
    "reduced_training/10pct/"
    "epoch_streams"
)

STREAM_REGISTRY_PATH = (
    STREAM_DIR
    / "reduced_epoch_stream_registry.csv"
)


# =============================================================================
# Validation artifacts
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
# Production pilot outputs
# =============================================================================

RUN_DIR = Path(
    "data/experimental/phase_6/"
    "reduced_training/10pct/"
    "run_20epoch"
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
    "phase_6_7d_10pct_20epoch_pilot_result.json"
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
    repository_commit,
):

    return {
        "phase": "6.7d",
        "experiment": (
            "10PCT_REDUCED_SUPERVISION_PILOT"
        ),

        "repository_commit": (
            repository_commit
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
    repository_commit,
):

    return {
        "schema_version": (
            "ITRS_PHASE6_10PCT_PRODUCTION_PILOT_V1"
        ),

        "checkpoint_role": (
            role
        ),

        "experiment": (
            "10PCT_REDUCED_SUPERVISION_PILOT"
        ),

        "repository_commit": (
            repository_commit
        ),

        "subset_sha256": (
            EXPECTED_SUBSET_SHA
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
    }


def save_latest(
    *,
    model,
    optimizer,
    state,
    roundtrip,
    repository_commit,
):

    payload = checkpoint_payload(
        role="LATEST_RESUME",
        model=model,
        optimizer=optimizer,
        state=state,
        roundtrip=roundtrip,
        repository_commit=(
            repository_commit
        ),
    )

    atomic_torch_save(
        payload,
        LATEST_CHECKPOINT_PATH,
    )

    atomic_json_write(
        progress_payload(
            state=state,
            repository_commit=(
                repository_commit
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
    repository_commit,
):

    payload = checkpoint_payload(
        role="BEST_VALIDATION",
        model=model,
        optimizer=optimizer,
        state=state,
        roundtrip=roundtrip,
        repository_commit=(
            repository_commit
        ),
    )

    atomic_torch_save(
        payload,
        BEST_CHECKPOINT_PATH,
    )


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
):

    require(
        checkpoint[
            "schema_version"
        ]
        == (
            "ITRS_PHASE6_10PCT_"
            "PRODUCTION_PILOT_V1"
        ),
        (
            "Latest checkpoint "
            "schema drift."
        ),
    )

    require(
        checkpoint[
            "subset_sha256"
        ]
        == EXPECTED_SUBSET_SHA,
        (
            "Latest checkpoint "
            "subset SHA drift."
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

    pd.DataFrame(
        history
    ).to_csv(
        EPOCH_METRICS_PATH,
        index=False,
    )


# =============================================================================
# Prerequisite gate
# =============================================================================

def prerequisite_gate(
    *,
    preflight,
):

    banner(
        "PHASE 6.7d PREREQUISITE GATE"
    )

    for path in (
        PHASE6_2_CONTRACT_PATH,
        PHASE6_7A_CONTRACT_PATH,
        PHASE6_7B_CONTRACT_PATH,
        PHASE6_7C_CONTRACT_PATH,
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

    q4 = load_json(
        PHASE6_7A_CONTRACT_PATH
    )

    q5a = load_json(
        PHASE6_7B_CONTRACT_PATH
    )

    q5b = load_json(
        PHASE6_7C_CONTRACT_PATH
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
        q4[
            "selection"
        ][
            "subset_logical_sha256"
        ]
        == EXPECTED_SUBSET_SHA,
        (
            "Phase 6.7a subset "
            "SHA drift."
        ),
    )

    require(
        q5a[
            "status"
        ]
        == "FROZEN_FOR_10PCT_PILOT",
        (
            "Phase 6.7b "
            "not frozen."
        ),
    )

    require(
        q5b[
            "status"
        ]
        == (
            "PASS_BOUNDED_10PCT_"
            "TRAINER_INTEGRATION"
        ),
        (
            "Phase 6.7c integration "
            "not qualified."
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
            "10% positive count "
            "drift."
        ),
    )

    subset_sha = (
        preflight
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
        (
            "10% positive-order "
            "SHA drift."
        ),
    )

    registry = pd.read_csv(
        STREAM_REGISTRY_PATH
    )

    require(
        len(
            registry
        )
        == NUM_EPOCHS,
        (
            "Stream registry does not "
            "contain 20 epochs."
        ),
    )

    for epoch in range(
        NUM_EPOCHS
    ):

        row = (
            registry.loc[
                registry[
                    "epoch_index"
                ]
                == epoch
            ]
            .iloc[
                0
            ]
        )

        negative_path = (
            STREAM_DIR
            / (
                f"epoch_{epoch:02d}_"
                "negative_startup_local.npy"
            )
        )

        order_path = (
            STREAM_DIR
            / (
                f"epoch_{epoch:02d}_"
                "example_order.npy"
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
            len(
                order
            )
            == EXAMPLES_PER_EPOCH,
            (
                "Order length drift "
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
                "Negative SHA drift "
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
                "Order SHA drift "
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
        "Phase 6.7a subset:                      PASS"
    )
    print(
        "Phase 6.7b 20 epoch streams:           PASS"
    )
    print(
        "Phase 6.7c bounded integration:        PASS"
    )
    print(
        "10% positive SHA:                       PASS"
    )
    print(
        "20 stream fingerprints:                PASS"
    )
    print(
        "2,251 validation cases:                PASS"
    )
    print(
        "Test split accessed:                   NO"
    )

    return (
        positive_order,
        registry,
        validation_cases,
        validation_candidates,
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
            "Stop safely after this many "
            "completed/validated display epochs."
        ),
    )

    args = parser.parse_args()

    banner(
        "PHASE 6.7d — "
        "10% REDUCED-SUPERVISION "
        "20-EPOCH PILOT"
    )

    print(
        f"Positive events / epoch:       "
        f"{POSITIVE_EVENTS:,}"
    )

    print(
        f"Examples / epoch:              "
        f"{EXAMPLES_PER_EPOCH:,}"
    )

    print(
        f"Batches / epoch:               "
        f"{BATCHES_PER_EPOCH}"
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
    # Load frozen infrastructure
    # =========================================================================

    q = load_module(
        PHASE6_BASE_PATH,
        "_phase6_7d_base",
    )

    packed = load_module(
        PACKED_RUNTIME_PATH,
        "_phase6_7d_packed",
    )

    reduced_runtime = load_module(
        REDUCED_RUNTIME_PATH,
        "_phase6_7d_reduced",
    )

    validation_preflight = (
        load_module(
            VALIDATION_PREFLIGHT_PATH,
            "_phase6_7d_validation_preflight",
        )
    )

    full_validation_runtime = (
        load_module(
            FULL_VALIDATION_RUNTIME_PATH,
            "_phase6_7d_full_validation",
        )
    )

    ranking_runtime = (
        load_module(
            RANKING_RUNTIME_PATH,
            "_phase6_7d_ranking",
        )
    )

    selection_runtime = (
        load_module(
            SELECTION_RUNTIME_PATH,
            "_phase6_7d_selection",
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
        "_phase6_7d_roundtrip",
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

    repository_commit = (
        git_head()
    )

    print()
    print(
        f"Repository HEAD:               "
        f"{repository_commit}"
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
            "PHASE 6.7d PREFLIGHT RESULT"
        )

        print(
            "All 20 reduced streams:        PASS"
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
            "PHASE 6.7d PREFLIGHT: PASS"
        )

        return

    # Actual run must correspond to committed code.
    require_clean_tracked_tree()

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

    # =========================================================================
    # Fresh or resume
    # =========================================================================

    if args.fresh:

        require(
            not LATEST_CHECKPOINT_PATH.exists(),
            (
                "latest.pt already exists. "
                "Use --resume rather than "
                "overwriting a scientific run."
            ),
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

        state = (
            make_fresh_state()
        )

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

        require(
            checkpoint[
                "repository_commit"
            ]
            == repository_commit,
            (
                "Resume repository commit "
                "differs from checkpoint."
            ),
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

    # =========================================================================
    # Already complete guard
    # =========================================================================

    if state[
        "training_complete"
    ]:

        banner(
            "RUN ALREADY COMPLETE"
        )

        print(
            f"Best epoch:                    "
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

        return

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

        negative_path = (
            STREAM_DIR
            / (
                f"epoch_{epoch:02d}_"
                "negative_startup_local.npy"
            )
        )

        order_path = (
            STREAM_DIR
            / (
                f"epoch_{epoch:02d}_"
                "example_order.npy"
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

        registry_row = (
            stream_registry.loc[
                stream_registry[
                    "epoch_index"
                ]
                == epoch
            ]
            .iloc[
                0
            ]
        )

        require(
            preflight
            .array_logical_sha256(
                np.asarray(
                    negative_matrix
                )
            )
            == str(
                registry_row[
                    "negative_sha256"
                ]
            ),
            (
                "Current epoch negative "
                "fingerprint drift."
            ),
        )

        require(
            preflight
            .array_logical_sha256(
                np.asarray(
                    epoch_order
                )
            )
            == str(
                registry_row[
                    "order_sha256"
                ]
            ),
            (
                "Current epoch order "
                "fingerprint drift."
            ),
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
                    "reduced epoch."
                ),
            )

            for batch_index in range(
                start_batch,
                BATCHES_PER_EPOCH,
            ):

                batch = (
                    reduced_runtime
                    .reduced_decode_batch(
                        positive_order=(
                            positive_order
                        ),
                        negative_matrix=(
                            negative_matrix
                        ),
                        epoch_order=(
                            epoch_order
                        ),
                        batch_index=(
                            batch_index
                        ),
                        num_investors=(
                            roundtrip
                            .NUM_INVESTORS
                        ),
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
                    % 5
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
                        f"batch={batch_index + 1:03d}/"
                        f"{BATCHES_PER_EPOCH} "
                        f"step="
                        f"{state['global_optimizer_step']:04d}/"
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
                        repository_commit=(
                            repository_commit
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
            # if validation is interrupted, resume redoes only validation.
            save_latest(
                model=model,
                optimizer=optimizer,
                state=state,
                roundtrip=roundtrip,
                repository_commit=(
                    repository_commit
                ),
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
        # BEST checkpoints must therefore record validation_pending=False.
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

            # BEST is associated with the model state
            # that produced this epoch's validation metrics.
            save_best(
                model=model,
                optimizer=optimizer,
                state=state,
                roundtrip=roundtrip,
                repository_commit=(
                    repository_commit
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

        # Latest checkpoint is the next resumable state.
        save_latest(
            model=model,
            optimizer=optimizer,
            state=state,
            roundtrip=roundtrip,
            repository_commit=(
                repository_commit
            ),
        )

        # ---------------------------------------------------------------------
        # Safe requested stop after a completed validation.
        # ---------------------------------------------------------------------

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
                "Resume command:                "
                "python scripts/"
                "phase_6_7d_10pct_20epoch_pilot.py "
                "--resume"
            )

            print(
                "Test cases scored:             0"
            )

            return

    # =========================================================================
    # Final experimental result — still NO TEST
    # =========================================================================

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

    result_contract = {
        "phase": "6.7d",

        "status": (
            "COMPLETE_10PCT_20EPOCH_"
            "PILOT_VALIDATION_ONLY"
        ),

        "experiment": (
            "10PCT_REDUCED_SUPERVISION_PILOT"
        ),

        "not_claimed_as": (
            "FULL_DATASET_ITRS_REPRODUCTION"
        ),

        "repository_commit": (
            repository_commit
        ),

        "training": {
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
            "optimizer_steps": (
                TOTAL_OPTIMIZER_STEPS
            ),
            "subset_sha256": (
                EXPECTED_SUBSET_SHA
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

            "reason": (
                "10% is a model-selection/data-efficiency "
                "pilot. Test remains withheld until "
                "training-budget comparison is complete "
                "and one configuration is selected."
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
        },
    }

    atomic_json_write(
        result_contract,
        FINAL_CONTRACT_PATH,
    )

    banner(
        "PHASE 6.7d FINAL STATUS"
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
        "PHASE 6.7d: COMPLETE / "
        "10% 20-EPOCH PILOT TRAINED + "
        "VALIDATED / TEST UNTOUCHED"
    )


if __name__ == "__main__":
    main()
