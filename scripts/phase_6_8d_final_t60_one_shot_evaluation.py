#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# =============================================================================
# Frozen final-test contract
# =============================================================================

TEST_CASES = 20_264
CANDIDATES_PER_CASE = 100

PREFIX_CHUNK_SIZE = 16
MAIN_CHUNK_SIZE = 64

EXPECTED_CHECKPOINT_SHA = (
    "a86d1e3ecb8058be747d2f289414dbf0"
    "bddf779feb2fb7fdbc1cba8cfd3bd4b2"
)

EXPECTED_BINDING_LOGICAL_SHA = (
    "951bf06814aee2e6db6f3a43c099a757"
    "15ab1ee8f76f493e989b21bf7ebcc54c"
)

EXPECTED_CANDIDATE_LOGICAL_SHA = (
    "7c16db4253a347471f0e33c0dd3c8333"
    "4191adf0166dc74f95c8e2aab5e0abfe"
)

EXPECTED_VALIDATION_NDCG10 = 0.188732390716
EXPECTED_VALIDATION_HR10 = 0.350955131053


# =============================================================================
# Paths
# =============================================================================

SELF_PATH = Path(
    "scripts/"
    "phase_6_8d_final_t60_one_shot_evaluation.py"
)

TRAIN_RUNTIME_PATH = Path(
    "scripts/"
    "phase_6_7d_10pct_20epoch_pilot.py"
)

SELECTION_CONTRACT_PATH = Path(
    "data/experimental/phase_6/contracts/"
    "phase_6_8a_final_training_budget_selection.json"
)

BINDING_CONTRACT_PATH = Path(
    "data/experimental/phase_6/contracts/"
    "phase_6_8c_final_t60_test_binding_contract.json"
)

CHECKPOINT_PATH = Path(
    "/workspace/phase6_10pct_best_epoch2.pt"
)

FINAL_TEST_DIR = Path(
    "data/experimental/phase_6/final_test"
)

TEST_BINDING_PATH = (
    FINAL_TEST_DIR
    / "final_t60_test_case_binding.parquet"
)

TEST_CANDIDATE_PATH = (
    FINAL_TEST_DIR
    / "final_t60_test_candidate_startup_local.npy"
)

SCORING_MARKER_PATH = (
    FINAL_TEST_DIR
    / "FINAL_T60_SCORING_STARTED.json"
)

CASE_METRICS_PATH = (
    FINAL_TEST_DIR
    / "final_t60_test_case_metrics.parquet"
)

RESULT_CONTRACT_PATH = Path(
    "data/experimental/phase_6/contracts/"
    "phase_6_8d_final_t60_test_result.json"
)


# =============================================================================
# Helpers
# =============================================================================

def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def banner(text):
    print()
    print("=" * 110)
    print(text)
    print("=" * 110)


def file_sha256(path):
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


def load_json(path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_module(path, name):
    require(
        path.exists(),
        f"Missing module: {path}",
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

    sys.modules[name] = module

    spec.loader.exec_module(
        module
    )

    return module


def git_head():
    return subprocess.check_output(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()


def tracked_tree_is_clean():
    result = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    return (
        result.stdout.strip()
        == ""
    )


def self_is_tracked():
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--error-unmatch",
            str(SELF_PATH),
        ],
        capture_output=True,
        text=True,
    )

    return result.returncode == 0


def numpy_rng_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(
            left[1],
            right[1],
        )
        and left[2:] == right[2:]
    )


def rng_snapshot():
    return {
        "python":
            random.getstate(),

        "numpy":
            np.random.get_state(),

        "torch":
            torch.get_rng_state().clone(),
    }


def rng_equal(left, right):
    return (
        left["python"]
        == right["python"]

        and numpy_rng_equal(
            left["numpy"],
            right["numpy"],
        )

        and torch.equal(
            left["torch"],
            right["torch"],
        )
    )


def atomic_json_write(payload, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = path.with_name(
        path.name + ".tmp"
    )

    temp.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )

    temp.replace(path)


def atomic_parquet_write(frame, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = path.with_name(
        path.name + ".tmp"
    )

    frame.to_parquet(
        temp,
        index=False,
    )

    temp.replace(path)


def build_test_chunk_schedule():
    chunks = []

    start = 0
    end = min(
        PREFIX_CHUNK_SIZE,
        TEST_CASES,
    )

    chunks.append(
        (
            0,
            start,
            end,
        )
    )

    chunk_index = 1
    start = end

    while start < TEST_CASES:

        end = min(
            start + MAIN_CHUNK_SIZE,
            TEST_CASES,
        )

        chunks.append(
            (
                chunk_index,
                start,
                end,
            )
        )

        chunk_index += 1
        start = end

    covered = []

    for _, start, end in chunks:
        covered.extend(
            range(
                start,
                end,
            )
        )

    require(
        covered
        == list(
            range(
                TEST_CASES
            )
        ),
        (
            "Test chunk schedule "
            "contains gap/overlap/reordering."
        ),
    )

    return chunks


# =============================================================================
# Portable CPU evaluation model — NO optimizer
# =============================================================================

def construct_evaluation_model(
    *,
    runtime,
    roundtrip,
    checkpoint,
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
        _,
    ) = (
        runtime
        .build_canonical_runtime(
            canonical_tree
        )
    )

    (
        _,
        exact_methods,
        training_adapter,
        _,
        _,
    ) = (
        runtime
        .build_forward_runtime(
            forward_tree
        )
    )

    (
        model,
        hash_fn,
    ) = (
        runtime
        .compose_canonical_model(
            canonical_runtime,
            exact_methods,
            training_adapter,
        )
    )

    require(
        hash_fn(model)
        == roundtrip.EXPECTED_INITIAL_MODEL_SHA256,
        (
            "Portable canonical initial "
            "model SHA drift."
        ),
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    model.eval()

    require(
        all(
            parameter.device.type
            == "cpu"
            for parameter
            in model.parameters()
        ),
        (
            "Final evaluation model "
            "must remain on CPU."
        ),
    )

    require(
        all(
            parameter.grad is None
            for parameter
            in model.parameters()
        ),
        (
            "Evaluation model unexpectedly "
            "contains gradients."
        ),
    )

    return (
        model,
        hash_fn,
    )


# =============================================================================
# Full preflight — ZERO test scoring
# =============================================================================

def build_context():

    banner(
        "PHASE 6.8d — FINAL T60 "
        "ONE-SHOT EVALUATION PREFLIGHT"
    )

    for path in (
        SELF_PATH,
        TRAIN_RUNTIME_PATH,
        SELECTION_CONTRACT_PATH,
        BINDING_CONTRACT_PATH,
        CHECKPOINT_PATH,
        TEST_BINDING_PATH,
        TEST_CANDIDATE_PATH,
    ):
        require(
            path.exists(),
            f"Missing prerequisite: {path}",
        )

    selection = load_json(
        SELECTION_CONTRACT_PATH
    )

    binding_contract = load_json(
        BINDING_CONTRACT_PATH
    )

    require(
        selection["status"]
        == "FINAL_TRAINING_BUDGET_SELECTION_FROZEN",
        "Phase 6.8a is not frozen.",
    )

    selected = (
        selection[
            "selected_configuration"
        ]
    )

    require(
        selected["training_budget"]
        == "10pct",
        "Selected budget drift.",
    )

    require(
        int(
            selected[
                "best_epoch"
            ]
        )
        == 2,
        "Selected epoch drift.",
    )

    require(
        abs(
            float(
                selected[
                    "validation_NDCG@10"
                ]
            )
            - EXPECTED_VALIDATION_NDCG10
        )
        < 1e-12,
        "Selected validation NDCG drift.",
    )

    require(
        abs(
            float(
                selected[
                    "validation_HR@10"
                ]
            )
            - EXPECTED_VALIDATION_HR10
        )
        < 1e-12,
        "Selected validation HR drift.",
    )

    require(
        file_sha256(
            CHECKPOINT_PATH
        )
        == EXPECTED_CHECKPOINT_SHA,
        "Final checkpoint SHA drift.",
    )

    require(
        binding_contract["status"]
        == "FINAL_T60_TEST_BINDING_FROZEN",
        "Phase 6.8c is not frozen.",
    )

    final_binding = (
        binding_contract[
            "final_test_binding"
        ]
    )

    require(
        int(
            final_binding[
                "test_cases"
            ]
        )
        == TEST_CASES,
        "Frozen test-case count drift.",
    )

    require(
        final_binding[
            "binding_logical_sha256"
        ]
        == EXPECTED_BINDING_LOGICAL_SHA,
        "Frozen test binding SHA drift.",
    )

    require(
        final_binding[
            "candidate_matrix_logical_sha256"
        ]
        == EXPECTED_CANDIDATE_LOGICAL_SHA,
        "Frozen candidate SHA drift.",
    )

    require(
        file_sha256(
            TEST_BINDING_PATH
        )
        == final_binding[
            "binding_file_sha256"
        ],
        "Test binding physical SHA drift.",
    )

    require(
        file_sha256(
            TEST_CANDIDATE_PATH
        )
        == final_binding[
            "candidate_matrix_file_sha256"
        ],
        "Test candidate physical SHA drift.",
    )

    # -------------------------------------------------------------------------
    # Load the already-qualified Phase-6 / Phase-5 infrastructure.
    # -------------------------------------------------------------------------

    train_runtime = load_module(
        TRAIN_RUNTIME_PATH,
        "_phase6_8d_train_runtime",
    )

    q = load_module(
        train_runtime.PHASE6_BASE_PATH,
        "_phase6_8d_base",
    )

    require(
        q.file_sha256(
            q.INITIAL_STATE_PATH
        )
        == q.EXPECTED_INITIAL_FILE_SHA,
        "Portable initial-state file SHA drift.",
    )

    initial_payload = torch.load(
        q.INITIAL_STATE_PATH,
        map_location="cpu",
        weights_only=False,
    )

    roundtrip = q.load_module(
        q.ROUNDTRIP_PATH,
        "_phase6_8d_roundtrip",
    )

    runtime = (
        roundtrip
        .load_preflight_runtime()
    )

    q.install_portable_initialization_bridge(
        runtime,
        initial_payload[
            "model_state_dict"
        ],
    )

    validation_preflight = load_module(
        train_runtime
        .VALIDATION_PREFLIGHT_PATH,
        "_phase6_8d_validation_preflight",
    )

    full_validation_runtime = load_module(
        train_runtime
        .FULL_VALIDATION_RUNTIME_PATH,
        "_phase6_8d_full_validation",
    )

    ranking_runtime = load_module(
        train_runtime
        .RANKING_RUNTIME_PATH,
        "_phase6_8d_ranking",
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    require(
        checkpoint[
            "checkpoint_role"
        ]
        == "BEST_VALIDATION",
        "Frozen checkpoint is not BEST_VALIDATION.",
    )

    require(
        checkpoint[
            "schema_version"
        ]
        == "ITRS_PHASE6_10PCT_PRODUCTION_PILOT_V1",
        "Checkpoint schema drift.",
    )

    require(
        checkpoint[
            "experiment"
        ]
        == "10PCT_REDUCED_SUPERVISION_PILOT",
        "Checkpoint experiment identity drift.",
    )

    (
        model,
        hash_fn,
    ) = construct_evaluation_model(
        runtime=runtime,
        roundtrip=roundtrip,
        checkpoint=checkpoint,
    )

    trained_model_sha = (
        hash_fn(
            model
        )
    )

    shared_cpu = (
        roundtrip
        .load_shared_inputs(
            runtime
        )
    )

    test_cases = pd.read_parquet(
        TEST_BINDING_PATH
    )

    candidates = np.load(
        TEST_CANDIDATE_PATH,
        mmap_mode="r",
    )

    require(
        len(test_cases)
        == TEST_CASES,
        "Final test binding row-count drift.",
    )

    require(
        candidates.shape
        == (
            TEST_CASES,
            CANDIDATES_PER_CASE,
        ),
        "Final candidate matrix shape drift.",
    )

    require(
        bool(
            np.array_equal(
                candidates[:, 0],
                test_cases[
                    "positive_startup_local"
                ].to_numpy(
                    dtype=np.int64
                ),
            )
        ),
        (
            "Candidate position zero "
            "does not match positive startup."
        ),
    )

    binding_logical_sha = (
        validation_preflight
        .dataframe_logical_sha256(
            test_cases,
            columns=[
                "test_case_position",
                "matrix_row_index",
                "interaction_id",
                "investor_global",
                "positive_startup_local",
                "candidate_count",
                "negative_count",
            ],
        )
    )

    candidate_logical_sha = (
        validation_preflight
        .array_logical_sha256(
            np.asarray(
                candidates
            )
        )
    )

    require(
        binding_logical_sha
        == EXPECTED_BINDING_LOGICAL_SHA,
        "Runtime test binding logical SHA drift.",
    )

    require(
        candidate_logical_sha
        == EXPECTED_CANDIDATE_LOGICAL_SHA,
        "Runtime candidate logical SHA drift.",
    )

    chunks = build_test_chunk_schedule()

    print(
        "Selected training budget:          10%"
    )
    print(
        "Selected checkpoint epoch:         2"
    )
    print(
        "Checkpoint physical SHA:           PASS"
    )
    print(
        "Checkpoint role:                   BEST_VALIDATION"
    )
    print(
        "Frozen test cases:                 20,264"
    )
    print(
        "Candidates per case:               100"
    )
    print(
        f"Test chunks:                       {len(chunks)}"
    )
    print(
        "Test binding logical SHA:          PASS"
    )
    print(
        "Candidate matrix logical SHA:      PASS"
    )
    print(
        "Evaluation device:                 CPU"
    )
    print(
        "Optimizer instantiated:            NO"
    )
    print(
        "Backward computation:              NO"
    )
    print(
        "Test model inference performed:    NO"
    )
    print(
        "Test metrics computed:             NO"
    )

    print()
    print(
        "Trained-model logical SHA256:"
    )
    print(
        trained_model_sha
    )

    print()
    print(
        "PHASE 6.8d PREFLIGHT: PASS / "
        "ZERO TEST SCORING"
    )

    return {
        "train_runtime":
            train_runtime,

        "roundtrip":
            roundtrip,

        "runtime":
            runtime,

        "validation_preflight":
            validation_preflight,

        "full_validation_runtime":
            full_validation_runtime,

        "ranking_runtime":
            ranking_runtime,

        "model":
            model,

        "hash_fn":
            hash_fn,

        "trained_model_sha":
            trained_model_sha,

        "shared_cpu":
            shared_cpu,

        "test_cases":
            test_cases,

        "candidates":
            candidates,

        "chunks":
            chunks,
    }


# =============================================================================
# ONE AND ONLY final T60 scoring pass
# =============================================================================

def score_once(context):

    banner(
        "PHASE 6.8d — "
        "ONE-SHOT FINAL T60 TEST EVALUATION"
    )

    require(
        self_is_tracked(),
        (
            "REFUSING TEST SCORING: "
            "evaluator is not tracked by Git."
        ),
    )

    require(
        tracked_tree_is_clean(),
        (
            "REFUSING TEST SCORING: "
            "tracked repository tree is dirty."
        ),
    )

    require(
        not SCORING_MARKER_PATH.exists(),
        (
            "REFUSING TEST SCORING: "
            "a prior final-test scoring "
            "attempt has already started."
        ),
    )

    require(
        not CASE_METRICS_PATH.exists(),
        (
            "REFUSING TEST SCORING: "
            "final test metrics already exist."
        ),
    )

    require(
        not RESULT_CONTRACT_PATH.exists(),
        (
            "REFUSING TEST SCORING: "
            "final test result contract "
            "already exists."
        ),
    )

    repository_commit = git_head()

    marker = {
        "phase":
            "6.8d",

        "status":
            "FINAL_T60_SCORING_STARTED",

        "repository_commit":
            repository_commit,

        "evaluator_file_sha256":
            file_sha256(
                SELF_PATH
            ),

        "checkpoint_sha256":
            EXPECTED_CHECKPOINT_SHA,

        "binding_logical_sha256":
            EXPECTED_BINDING_LOGICAL_SHA,

        "candidate_matrix_logical_sha256":
            EXPECTED_CANDIDATE_LOGICAL_SHA,

        "planned_test_cases":
            TEST_CASES,
    }

    # Exclusive create. Once this exists, this script will
    # never silently begin another final-test pass.
    SCORING_MARKER_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SCORING_MARKER_PATH.open(
        "x",
        encoding="utf-8",
    ) as handle:
        json.dump(
            marker,
            handle,
            indent=2,
            sort_keys=True,
        )

    model = context[
        "model"
    ]

    hash_fn = context[
        "hash_fn"
    ]

    shared_cpu = context[
        "shared_cpu"
    ]

    test_cases = context[
        "test_cases"
    ]

    candidates = context[
        "candidates"
    ]

    chunks = context[
        "chunks"
    ]

    validation_preflight = context[
        "validation_preflight"
    ]

    full_validation_runtime = context[
        "full_validation_runtime"
    ]

    ranking_runtime = context[
        "ranking_runtime"
    ]

    state_before = (
        hash_fn(
            model
        )
    )

    rng_before = (
        rng_snapshot()
    )

    metric_rows = []

    start_time = (
        time.perf_counter()
    )

    with torch.no_grad():

        latent_all = torch.cat(
            [
                model
                .investor_embedding
                .weight,

                model
                .startup_embedding
                .weight,
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
            "Final-test structural output invalid.",
        )

        F_s_all = (
            structural[
                "F_s"
            ]
        )

        for (
            chunk_index,
            start,
            end,
        ) in chunks:

            chunk_cases = (
                test_cases
                .iloc[
                    start:end
                ]
                .copy()
                .reset_index(
                    drop=True
                )
            )

            chunk_candidates = (
                candidates[
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
                    bool(
                        np.isfinite(
                            logits_np
                        ).all()
                    ),
                    "Final-test logits non-finite.",
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
                        "test_case_position":
                            global_position,

                        "matrix_row_index":
                            int(
                                row[
                                    "matrix_row_index"
                                ]
                            ),

                        "interaction_id":
                            str(
                                row[
                                    "interaction_id"
                                ]
                            ),

                        "investor_global":
                            investor_global,

                        "positive_startup_local":
                            positive_local,

                        "positive_rank":
                            int(
                                positive_rank
                            ),

                        "HR@10":
                            float(
                                hr10
                            ),

                        "NDCG@10":
                            float(
                                ndcg10
                            ),

                        "chunk_index":
                            int(
                                chunk_index
                            ),
                    }
                )

            del features

            if (
                chunk_index % 25 == 0
                or end == TEST_CASES
            ):
                print(
                    f"Scored cases: "
                    f"{end:,} / "
                    f"{TEST_CASES:,}"
                )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    metrics = pd.DataFrame(
        metric_rows
    )

    require(
        len(metrics)
        == TEST_CASES,
        (
            "Final test did not score "
            "exactly 20,264 cases."
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
        "Final-test rank outside 1..100.",
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

    rng_after = (
        rng_snapshot()
    )

    require(
        state_after
        == state_before,
        "Final test changed model parameters.",
    )

    require(
        all(
            parameter.grad is None
            for parameter
            in model.parameters()
        ),
        "Final test created gradients.",
    )

    require(
        rng_equal(
            rng_before,
            rng_after,
        ),
        "Final test changed CPU RNG state.",
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

    atomic_parquet_write(
        metrics,
        CASE_METRICS_PATH,
    )

    result = {
        "phase":
            "6.8d",

        "status":
            "FINAL_T60_TEST_EVALUATION_COMPLETE",

        "repository_commit":
            repository_commit,

        "evaluator_file_sha256":
            file_sha256(
                SELF_PATH
            ),

        "selected_configuration": {
            "training_budget":
                "10pct",

            "best_epoch":
                2,

            "checkpoint_sha256":
                EXPECTED_CHECKPOINT_SHA,

            "trained_model_logical_sha256":
                context[
                    "trained_model_sha"
                ],
        },

        "frozen_test_inputs": {
            "test_cases":
                TEST_CASES,

            "candidates_per_case":
                CANDIDATES_PER_CASE,

            "binding_logical_sha256":
                EXPECTED_BINDING_LOGICAL_SHA,

            "candidate_matrix_logical_sha256":
                EXPECTED_CANDIDATE_LOGICAL_SHA,
        },

        "test_results": {
            "HR@10":
                float(
                    hr10
                ),

            "NDCG@10":
                float(
                    ndcg10
                ),

            "hits_at_10":
                hit_count,

            "mean_positive_rank":
                mean_rank,

            "median_positive_rank":
                median_rank,

            "evaluation_seconds":
                elapsed,
        },

        "execution_invariants": {
            "device":
                "cpu",

            "optimizer_instantiated":
                False,

            "backward_performed":
                False,

            "optimizer_steps":
                0,

            "model_state_unchanged":
                True,

            "gradients_created":
                False,

            "cpu_rng_state_unchanged":
                True,

            "test_cases_scored":
                TEST_CASES,

            "final_test_pass":
                1,
        },

        "case_metrics": {
            "path":
                str(
                    CASE_METRICS_PATH
                ),

            "file_sha256":
                file_sha256(
                    CASE_METRICS_PATH
                ),
        },
    }

    atomic_json_write(
        result,
        RESULT_CONTRACT_PATH,
    )

    banner(
        "PHASE 6.8d FINAL T60 RESULT"
    )

    print(
        f"Test cases scored:             "
        f"{TEST_CASES:,} / {TEST_CASES:,}"
    )

    print(
        f"Test HR@10:                    "
        f"{hr10:.12f}"
    )

    print(
        f"Test NDCG@10:                  "
        f"{ndcg10:.12f}"
    )

    print(
        f"Hits @10:                      "
        f"{hit_count:,} / {TEST_CASES:,}"
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
        f"Evaluation time:               "
        f"{elapsed:.2f} s"
    )

    print(
        "Model state unchanged:          YES"
    )

    print(
        "Gradients created:              NO"
    )

    print(
        "Optimizer instantiated:         NO"
    )

    print(
        "Optimizer steps:                0"
    )

    print(
        "CPU RNG state unchanged:        YES"
    )

    print()
    print(
        f"WROTE  {CASE_METRICS_PATH}"
    )

    print(
        f"WROTE  {RESULT_CONTRACT_PATH}"
    )

    print()
    print(
        "PHASE 6.8d: COMPLETE / "
        "FINAL T60 TEST SCORED ONCE"
    )


# =============================================================================
# Entry point
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
        "--score-once",
        action="store_true",
    )

    args = parser.parse_args()

    context = build_context()

    if args.preflight_only:
        return

    score_once(
        context
    )


if __name__ == "__main__":
    main()
