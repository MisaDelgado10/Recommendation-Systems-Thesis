#!/usr/bin/env python3
"""
Phase 5.4.2 — Lean Exact CPU Runtime + Threading Equivalence Benchmark

Purpose
-------
Separate audit-instrumentation overhead from actual training computation and
test whether a different PyTorch intra-op CPU thread count can accelerate the
same canonical CPU training path while preserving BYTE-EXACT frozen numerical
anchors.

Scientific boundary
-------------------
This phase changes NO:
    - model architecture
    - parameters / initialization
    - training examples
    - batch size
    - negative sampling
    - epoch order
    - loss
    - optimizer
    - learning rate
    - validation/test protocol

The only candidate runtime changes are:
    A) remove proof-only full-state hashing from INSIDE the timed batch path;
       the hashes are recomputed OUTSIDE the timer to prove exactness.
    B) vary torch intra-op CPU thread count.

A candidate thread count is ELIGIBLE only if batch 0 reproduces exactly:
    - BCE loss
    - logit SHA256
    - gradient SHA256
    - post-step model SHA256
    - optimizer-state SHA256

The fastest eligible thread count is then required to reproduce the exact
two-step batch0 -> batch1 trajectory.

No production checkpoint is written.
No validation is scored.
No test is accessed.
The benchmark model is discarded.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# =============================================================================
# Paths
# =============================================================================

ROUNDTRIP_SOURCE_PATH = Path(
    "scripts/"
    "phase_5_3_2b_checkpoint_resume_roundtrip_proof.py"
)

PHASE_5_4_1_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_4_1_cpu_training_feasibility_benchmark_contract.json"
)

PHASE_5_4_AUTHORIZATION_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_4_production_training_launch_authorization.json"
)

AUDIT_DIR = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_4_2"
)

THREAD_AUDIT_PATH = (
    AUDIT_DIR
    / "exact_cpu_thread_candidate_audit.csv"
)

SELECTED_TWO_STEP_PATH = (
    AUDIT_DIR
    / "selected_exact_cpu_two_step_timing.csv"
)

PROJECTION_PATH = (
    AUDIT_DIR
    / "lean_exact_cpu_runtime_projection.csv"
)

FINAL_INVARIANT_PATH = (
    AUDIT_DIR
    / "phase_5_4_2_final_invariants.csv"
)

CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_4_2_lean_exact_cpu_runtime_contract.json"
)

MANIFEST_PATH = (
    AUDIT_DIR
    / "phase_5_4_2_lean_exact_cpu_runtime_manifest.json"
)


# =============================================================================
# Frozen anchors
# =============================================================================

EXPECTED_PYTORCH = "2.7.0"

EXPECTED_INITIAL_MODEL_SHA256 = (
    "49e822ea7fad35c458f47e134c94c05e"
    "ac099b68c5c468e2c71559c8c88998ab"
)

EXPECTED_BATCH0_LOSS = 0.7080879807
EXPECTED_BATCH1_LOSS = 0.6636360884

EXPECTED_BATCH0_LOGIT_SHA256 = (
    "35b89aaed29d51d2ebb7ba1cadf2dc4b"
    "b5e8f81cf3aa78bc216b3cc6fed13845"
)

EXPECTED_BATCH0_GRADIENT_SHA256 = (
    "8c542430813d8ca91b8397409954ea92"
    "295a2b55bcc420661783fb865010845d"
)

EXPECTED_BATCH0_POST_MODEL_SHA256 = (
    "42a521f11d8f24e4144d0215d6e1b34d"
    "5f8bf0c2d8848624e4f7c3130699035d"
)

EXPECTED_BATCH0_OPTIMIZER_SHA256 = (
    "5ce2683c21f456b9d5d15eb876b049c5"
    "e6db1215db5a026630f093f7f9d49891"
)

EXPECTED_BATCH1_LOGIT_SHA256 = (
    "cfbc4106103abf9478b8f04f0e0d909b"
    "ed37659e5ee7e29257bce0a7dd4beb26"
)

EXPECTED_BATCH1_GRADIENT_SHA256 = (
    "8c066fd5f8002e1edd0a282f4ac549a3"
    "903f590716b38ca060b0f01088594f22"
)

EXPECTED_BATCH1_POST_MODEL_SHA256 = (
    "c41702cda99092a7fb63bb0a8227e658"
    "851b3ac4cbc373d90cdd6816eccdd196"
)

EXPECTED_BATCH1_OPTIMIZER_SHA256 = (
    "569a6691424ac32d0f252728750281cffd"
    "175a2b6b6c6ea1913f5f497200b00d"
)

BATCHES_PER_EPOCH = 10_481
TOTAL_OPTIMIZER_STEPS = 209_620


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


def load_module(
    path: Path,
    module_name: str,
):
    require(
        path.exists(),
        f"Missing source: {path}",
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    require(
        spec is not None
        and spec.loader is not None,
        f"Could not import {path}.",
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

    return module


def close_float(
    actual: float,
    expected: float,
    tolerance: float = 5e-10,
) -> bool:
    return (
        math.isfinite(
            float(actual)
        )
        and abs(
            float(actual)
            - float(expected)
        )
        <= tolerance
    )


def human_duration(
    seconds: float,
) -> str:
    seconds = float(seconds)

    if seconds < 60:
        return f"{seconds:.2f} seconds"

    minutes = seconds / 60.0

    if minutes < 60:
        return f"{minutes:.2f} minutes"

    hours = minutes / 60.0

    if hours < 48:
        return f"{hours:.2f} hours"

    return f"{hours / 24.0:.2f} days"


def candidate_thread_counts(
    current_threads: int,
) -> list[int]:
    cpu_count = (
        os.cpu_count()
        or current_threads
    )

    raw = {
        int(current_threads),
        min(
            int(cpu_count),
            6,
        ),
        min(
            int(cpu_count),
            8,
        ),
        int(cpu_count),
    }

    return sorted(
        value
        for value in raw
        if value >= 1
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.4.2 — "
        "LEAN EXACT CPU RUNTIME + THREADING EQUIVALENCE BENCHMARK"
    )

    print(
        "Production training launched:         NO"
    )
    print(
        "Production checkpoint written:        NO"
    )
    print(
        "Validation cases scored:              0"
    )
    print(
        "Test cases scored:                    0"
    )

    # =========================================================================
    # Gates
    # =========================================================================

    banner(
        "PREREQUISITE GATE"
    )

    for path in (
        ROUNDTRIP_SOURCE_PATH,
        PHASE_5_4_1_CONTRACT_PATH,
        PHASE_5_4_AUTHORIZATION_PATH,
    ):
        require(
            path.exists(),
            f"Missing prerequisite: {path}",
        )

        print(
            f"FOUND  {path}"
        )

    benchmark_5_4_1 = load_json(
        PHASE_5_4_1_CONTRACT_PATH
    )

    authorization = load_json(
        PHASE_5_4_AUTHORIZATION_PATH
    )

    require(
        benchmark_5_4_1.get(
            "status"
        )
        == "COMPLETE",
        (
            "Phase-5.4.1 benchmark is not COMPLETE."
        ),
    )

    require(
        benchmark_5_4_1.get(
            "numerical_anchor_status"
        )
        == "EXACT_FROZEN_BATCH0_AND_BATCH1",
        (
            "Phase-5.4.1 exact numerical anchor drift."
        ),
    )

    require(
        authorization.get(
            "training_allowed"
        )
        is True,
        (
            "Production training is not authorized."
        ),
    )

    audited_mean_seconds = float(
        benchmark_5_4_1[
            "batch_seconds"
        ][
            "mean"
        ]
    )

    print(
        "Phase-5.4.1 exact proof path:         PASS"
    )
    print(
        f"Audited mean seconds / batch:         "
        f"{audited_mean_seconds:.3f}"
    )

    # =========================================================================
    # Runtime
    # =========================================================================

    banner(
        "LOAD FROZEN TRAINING RUNTIME"
    )

    roundtrip = load_module(
        ROUNDTRIP_SOURCE_PATH,
        "_itrs_phase5_4_2_roundtrip",
    )

    preflight = (
        roundtrip
        .load_preflight_runtime()
    )

    stream = (
        roundtrip
        .load_epoch0_stream(
            preflight
        )
    )

    shared = (
        roundtrip
        .load_shared_inputs(
            preflight
        )
    )

    batch0 = (
        roundtrip
        .decode_batch(
            stream,
            0,
        )
    )

    batch1 = (
        roundtrip
        .decode_batch(
            stream,
            1,
        )
    )

    require(
        torch.__version__
        == EXPECTED_PYTORCH,
        (
            "Reference PyTorch runtime drift."
        ),
    )

    # Preserve true expensive audit functions.
    true_gradient_hash_fn = (
        roundtrip
        .gradient_logical_sha256
    )

    true_optimizer_hash_fn = (
        roundtrip
        .optimizer_state_logical_sha256
    )

    # Proof-only expensive hashes are removed from the TIMED section.
    # They are recomputed immediately AFTER timing to prove exactness.
    roundtrip.gradient_logical_sha256 = (
        lambda model: "SKIPPED_INSIDE_TIMED_PATH"
    )

    roundtrip.optimizer_state_logical_sha256 = (
        lambda model, optimizer: "SKIPPED_INSIDE_TIMED_PATH"
    )

    current_threads = int(
        torch.get_num_threads()
    )

    thread_candidates = (
        candidate_thread_counts(
            current_threads
        )
    )

    print(
        f"Platform:                              "
        f"{platform.platform()}"
    )
    print(
        f"os.cpu_count():                        "
        f"{os.cpu_count()}"
    )
    print(
        f"Current torch intra-op threads:        "
        f"{current_threads}"
    )
    print(
        f"Thread candidates:                     "
        f"{thread_candidates}"
    )

    # =========================================================================
    # Batch-0 thread candidate audit
    # =========================================================================

    banner(
        "BATCH-0 EXACT THREAD-CANDIDATE AUDIT"
    )

    candidate_rows = []

    for threads in thread_candidates:

        torch.set_num_threads(
            int(threads)
        )

        (
            model,
            optimizer,
            canonical_hash_fn,
            _runtime_ast_sha,
            _adapter_sha,
            _removed_guard_sha,
        ) = (
            roundtrip
            .construct_fresh_training_state(
                preflight
            )
        )

        require(
            canonical_hash_fn(
                model
            )
            == EXPECTED_INITIAL_MODEL_SHA256,
            (
                "Fresh model does not match "
                "canonical initialization."
            ),
        )

        # Disable full model hashing only inside timed executor.
        dummy_model_hash_fn = (
            lambda model: "SKIPPED_INSIDE_TIMED_PATH"
        )

        start = time.perf_counter()

        result = (
            roundtrip
            .execute_training_batch(
                model,
                optimizer,
                dummy_model_hash_fn,
                batch0,
                shared,
            )
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        # Exactness checks happen OUTSIDE timer.
        gradient_sha = (
            true_gradient_hash_fn(
                model
            )
        )

        post_model_sha = (
            canonical_hash_fn(
                model
            )
        )

        optimizer_sha = (
            true_optimizer_hash_fn(
                model,
                optimizer,
            )
        )

        loss_exact = close_float(
            result[
                "loss"
            ],
            EXPECTED_BATCH0_LOSS,
        )

        logit_exact = (
            result[
                "logit_sha256"
            ]
            == EXPECTED_BATCH0_LOGIT_SHA256
        )

        gradient_exact = (
            gradient_sha
            == EXPECTED_BATCH0_GRADIENT_SHA256
        )

        model_exact = (
            post_model_sha
            == EXPECTED_BATCH0_POST_MODEL_SHA256
        )

        optimizer_exact = (
            optimizer_sha
            == EXPECTED_BATCH0_OPTIMIZER_SHA256
        )

        exact_all = bool(
            loss_exact
            and logit_exact
            and gradient_exact
            and model_exact
            and optimizer_exact
        )

        candidate_rows.append(
            {
                "threads": (
                    int(threads)
                ),
                "wall_seconds_timed_compute": (
                    float(elapsed)
                ),
                "loss": (
                    float(
                        result[
                            "loss"
                        ]
                    )
                ),
                "loss_exact": (
                    loss_exact
                ),
                "logit_exact": (
                    logit_exact
                ),
                "gradient_exact": (
                    gradient_exact
                ),
                "post_model_exact": (
                    model_exact
                ),
                "optimizer_exact": (
                    optimizer_exact
                ),
                "byte_exact_eligible": (
                    exact_all
                ),
            }
        )

        print(
            f"threads={threads:2d} | "
            f"timed={elapsed:8.3f}s | "
            f"exact={'YES' if exact_all else 'NO'}"
        )

        del model
        del optimizer

        gc.collect()

    candidate_df = pd.DataFrame(
        candidate_rows
    )

    eligible_df = candidate_df.loc[
        candidate_df[
            "byte_exact_eligible"
        ]
        == True
    ].copy()

    require(
        len(
            eligible_df
        )
        >= 1,
        (
            "No CPU thread configuration preserves "
            "the exact batch-0 frozen trajectory."
        ),
    )

    selected_row = (
        eligible_df.sort_values(
            [
                "wall_seconds_timed_compute",
                "threads",
            ],
            ascending=[
                True,
                True,
            ],
            kind="mergesort",
        )
        .iloc[
            0
        ]
    )

    selected_threads = int(
        selected_row[
            "threads"
        ]
    )

    print()
    print(
        f"Selected exact thread count:           "
        f"{selected_threads}"
    )
    print(
        f"Selected batch-0 timed compute:        "
        f"{float(selected_row['wall_seconds_timed_compute']):.3f} s"
    )

    # =========================================================================
    # Selected exact two-step proof
    # =========================================================================

    banner(
        "SELECTED THREAD COUNT — EXACT TWO-STEP TIMING PROOF"
    )

    torch.set_num_threads(
        selected_threads
    )

    (
        model,
        optimizer,
        canonical_hash_fn,
        runtime_ast_sha,
        adapter_sha,
        removed_guard_sha,
    ) = (
        roundtrip
        .construct_fresh_training_state(
            preflight
        )
    )

    require(
        canonical_hash_fn(
            model
        )
        == EXPECTED_INITIAL_MODEL_SHA256,
        (
            "Selected-runtime fresh model initial hash drift."
        ),
    )

    dummy_model_hash_fn = (
        lambda model: "SKIPPED_INSIDE_TIMED_PATH"
    )

    selected_rows = []

    for (
        batch_index,
        batch,
        expected_loss,
        expected_logit_sha,
        expected_gradient_sha,
        expected_model_sha,
        expected_optimizer_sha,
    ) in (
        (
            0,
            batch0,
            EXPECTED_BATCH0_LOSS,
            EXPECTED_BATCH0_LOGIT_SHA256,
            EXPECTED_BATCH0_GRADIENT_SHA256,
            EXPECTED_BATCH0_POST_MODEL_SHA256,
            EXPECTED_BATCH0_OPTIMIZER_SHA256,
        ),
        (
            1,
            batch1,
            EXPECTED_BATCH1_LOSS,
            EXPECTED_BATCH1_LOGIT_SHA256,
            EXPECTED_BATCH1_GRADIENT_SHA256,
            EXPECTED_BATCH1_POST_MODEL_SHA256,
            EXPECTED_BATCH1_OPTIMIZER_SHA256,
        ),
    ):

        start = time.perf_counter()

        result = (
            roundtrip
            .execute_training_batch(
                model,
                optimizer,
                dummy_model_hash_fn,
                batch,
                shared,
            )
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        gradient_sha = (
            true_gradient_hash_fn(
                model
            )
        )

        post_model_sha = (
            canonical_hash_fn(
                model
            )
        )

        optimizer_sha = (
            true_optimizer_hash_fn(
                model,
                optimizer,
            )
        )

        exact_all = bool(
            close_float(
                result[
                    "loss"
                ],
                expected_loss,
            )
            and result[
                "logit_sha256"
            ]
            == expected_logit_sha
            and gradient_sha
            == expected_gradient_sha
            and post_model_sha
            == expected_model_sha
            and optimizer_sha
            == expected_optimizer_sha
        )

        require(
            exact_all,
            (
                f"Selected thread count {selected_threads} "
                f"failed exact two-step anchor at batch {batch_index}."
            ),
        )

        selected_rows.append(
            {
                "batch_index": (
                    batch_index
                ),
                "threads": (
                    selected_threads
                ),
                "wall_seconds_timed_compute": (
                    float(elapsed)
                ),
                "loss": (
                    float(
                        result[
                            "loss"
                        ]
                    )
                ),
                "logit_sha256": (
                    result[
                        "logit_sha256"
                    ]
                ),
                "gradient_sha256": (
                    gradient_sha
                ),
                "post_model_sha256": (
                    post_model_sha
                ),
                "optimizer_state_sha256": (
                    optimizer_sha
                ),
                "exact_frozen": (
                    True
                ),
            }
        )

        print(
            f"batch={batch_index} | "
            f"threads={selected_threads} | "
            f"timed={elapsed:.3f}s | EXACT"
        )

    selected_df = pd.DataFrame(
        selected_rows
    )

    # =========================================================================
    # Projection
    # =========================================================================

    banner(
        "LEAN EXACT CPU PROJECTION"
    )

    lean_mean_seconds = float(
        selected_df[
            "wall_seconds_timed_compute"
        ].mean()
    )

    lean_max_seconds = float(
        selected_df[
            "wall_seconds_timed_compute"
        ].max()
    )

    instrumentation_speedup = (
        audited_mean_seconds
        / lean_mean_seconds
    )

    projected_epoch_seconds = (
        lean_mean_seconds
        * BATCHES_PER_EPOCH
    )

    projected_full_seconds = (
        lean_mean_seconds
        * TOTAL_OPTIMIZER_STEPS
    )

    projected_epoch_conservative_seconds = (
        lean_max_seconds
        * BATCHES_PER_EPOCH
    )

    projected_full_conservative_seconds = (
        lean_max_seconds
        * TOTAL_OPTIMIZER_STEPS
    )

    print(
        f"Audited proof-path mean / batch:       "
        f"{audited_mean_seconds:.3f} s"
    )
    print(
        f"Lean exact mean / batch:               "
        f"{lean_mean_seconds:.3f} s"
    )
    print(
        f"Proof-instrumentation speedup:         "
        f"{instrumentation_speedup:.2f}x"
    )
    print()
    print(
        "Projected lean one epoch:             "
        f"{human_duration(projected_epoch_seconds)}"
    )
    print(
        "Projected lean 20 epochs:             "
        f"{human_duration(projected_full_seconds)}"
    )
    print(
        "Conservative lean 20 epochs:          "
        f"{human_duration(projected_full_conservative_seconds)}"
    )

    projection_df = pd.DataFrame(
        [
            {
                "metric": (
                    "audited_proof_mean_seconds_per_batch"
                ),
                "value": (
                    audited_mean_seconds
                ),
                "human": (
                    f"{audited_mean_seconds:.3f} s"
                ),
            },
            {
                "metric": (
                    "lean_exact_mean_seconds_per_batch"
                ),
                "value": (
                    lean_mean_seconds
                ),
                "human": (
                    f"{lean_mean_seconds:.3f} s"
                ),
            },
            {
                "metric": (
                    "audit_instrumentation_speedup"
                ),
                "value": (
                    instrumentation_speedup
                ),
                "human": (
                    f"{instrumentation_speedup:.2f}x"
                ),
            },
            {
                "metric": (
                    "lean_exact_one_epoch"
                ),
                "value": (
                    projected_epoch_seconds
                ),
                "human": (
                    human_duration(
                        projected_epoch_seconds
                    )
                ),
            },
            {
                "metric": (
                    "lean_exact_20_epochs"
                ),
                "value": (
                    projected_full_seconds
                ),
                "human": (
                    human_duration(
                        projected_full_seconds
                    )
                ),
            },
            {
                "metric": (
                    "lean_exact_20_epochs_conservative"
                ),
                "value": (
                    projected_full_conservative_seconds
                ),
                "human": (
                    human_duration(
                        projected_full_conservative_seconds
                    )
                ),
            },
        ]
    )

    # =========================================================================
    # Final invariants
    # =========================================================================

    banner(
        "FINAL PHASE-5.4.2 INVARIANTS"
    )

    checks = [
        (
            "production_training_not_launched",
            True,
        ),
        (
            "validation_cases_scored_zero",
            True,
        ),
        (
            "test_cases_scored_zero",
            True,
        ),
        (
            "at_least_one_exact_thread_candidate",
            (
                len(
                    eligible_df
                )
                >= 1
            ),
        ),
        (
            "selected_thread_batch0_exact",
            bool(
                selected_df.loc[
                    selected_df[
                        "batch_index"
                    ]
                    == 0,
                    "exact_frozen",
                ].iloc[
                    0
                ]
            ),
        ),
        (
            "selected_thread_batch1_exact",
            bool(
                selected_df.loc[
                    selected_df[
                        "batch_index"
                    ]
                    == 1,
                    "exact_frozen",
                ].iloc[
                    0
                ]
            ),
        ),
        (
            "selected_two_step_final_model_exact",
            (
                selected_df.iloc[
                    -1
                ][
                    "post_model_sha256"
                ]
                == EXPECTED_BATCH1_POST_MODEL_SHA256
            ),
        ),
        (
            "selected_two_step_final_optimizer_exact",
            (
                selected_df.iloc[
                    -1
                ][
                    "optimizer_state_sha256"
                ]
                == EXPECTED_BATCH1_OPTIMIZER_SHA256
            ),
        ),
        (
            "lean_timing_positive_finite",
            (
                math.isfinite(
                    lean_mean_seconds
                )
                and lean_mean_seconds > 0
            ),
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
            "At least one Phase-5.4.2 invariant failed."
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
        "WRITE PHASE-5.4.2 OUTPUTS"
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTRACT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_df.to_csv(
        THREAD_AUDIT_PATH,
        index=False,
    )

    selected_df.to_csv(
        SELECTED_TWO_STEP_PATH,
        index=False,
    )

    projection_df.to_csv(
        PROJECTION_PATH,
        index=False,
    )

    invariant_df.to_csv(
        FINAL_INVARIANT_PATH,
        index=False,
    )

    contract = {
        "phase": (
            "5.4.2"
        ),
        "title": (
            "Lean Exact CPU Runtime + Threading Equivalence Benchmark"
        ),
        "status": (
            "COMPLETE"
        ),
        "classification": (
            "IMPLEMENTATION_EQUIVALENT_RUNTIME_AUDIT"
        ),
        "production_training_launched": (
            False
        ),
        "validation_cases_scored": (
            0
        ),
        "test_cases_scored": (
            0
        ),
        "audit_instrumentation_removed_from_timed_path": [
            (
                "pre-step full canonical model SHA"
            ),
            (
                "full gradient logical SHA"
            ),
            (
                "post-step full canonical model SHA"
            ),
            (
                "full Adam-state logical SHA"
            ),
        ],
        "numerical_verification": (
            "ALL_REMOVED_HASHES_RECOMPUTED_OUTSIDE_TIMER"
        ),
        "thread_candidates": (
            candidate_df.to_dict(
                orient="records"
            )
        ),
        "selected_threads": (
            selected_threads
        ),
        "selected_two_step_exact": (
            True
        ),
        "audited_proof_mean_seconds_per_batch": (
            audited_mean_seconds
        ),
        "lean_exact_mean_seconds_per_batch": (
            lean_mean_seconds
        ),
        "instrumentation_speedup": (
            instrumentation_speedup
        ),
        "projected_seconds": {
            "one_epoch_mean": (
                projected_epoch_seconds
            ),
            "20_epochs_mean": (
                projected_full_seconds
            ),
            "20_epochs_conservative": (
                projected_full_conservative_seconds
            ),
        },
        "next_phase": (
            "5.4.3_RUNTIME_DECISION"
        ),
        "next_phase_rule": (
            "Use exact lean CPU runtime if practical; "
            "otherwise audit a further implementation-equivalent "
            "acceleration without changing scientific semantics."
        ),
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
            "5.4.2"
        ),
        "status": (
            "LEAN_EXACT_CPU_RUNTIME_BENCHMARK_COMPLETE"
        ),
        "selected_threads": (
            selected_threads
        ),
        "lean_exact_mean_seconds_per_batch": (
            lean_mean_seconds
        ),
        "projected_one_epoch_human": (
            human_duration(
                projected_epoch_seconds
            )
        ),
        "projected_20_epochs_human": (
            human_duration(
                projected_full_seconds
            )
        ),
        "production_training_steps": (
            0
        ),
        "production_checkpoint_written": (
            False
        ),
        "validation_cases_scored": (
            0
        ),
        "test_cases_scored": (
            0
        ),
        "contract": str(
            CONTRACT_PATH
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
        THREAD_AUDIT_PATH,
        SELECTED_TWO_STEP_PATH,
        PROJECTION_PATH,
        FINAL_INVARIANT_PATH,
        CONTRACT_PATH,
        MANIFEST_PATH,
    ):
        print(
            f"WROTE  {path}"
        )

    # Restore runtime module helpers for cleanliness.
    roundtrip.gradient_logical_sha256 = (
        true_gradient_hash_fn
    )

    roundtrip.optimizer_state_logical_sha256 = (
        true_optimizer_hash_fn
    )

    del model
    del optimizer
    del batch0
    del batch1
    del stream
    del shared

    gc.collect()

    banner(
        "PHASE 5.4.2 FINAL STATUS"
    )

    print(
        f"Selected exact CPU threads:           "
        f"{selected_threads}"
    )
    print(
        f"Audited proof mean / batch:           "
        f"{audited_mean_seconds:.3f} s"
    )
    print(
        f"Lean exact mean / batch:              "
        f"{lean_mean_seconds:.3f} s"
    )
    print(
        f"Speedup from removing audit hashes:   "
        f"{instrumentation_speedup:.2f}x"
    )
    print(
        "Projected lean exact one epoch:       "
        f"{human_duration(projected_epoch_seconds)}"
    )
    print(
        "Projected lean exact 20 epochs:       "
        f"{human_duration(projected_full_seconds)}"
    )
    print()
    print(
        "Production training launched:         NO"
    )
    print(
        "Production checkpoint written:        NO"
    )
    print(
        "Validation cases scored:              0"
    )
    print(
        "Test cases scored:                    0"
    )

    banner(
        "PHASE 5.4.2 COMPLETE / "
        "LEAN EXACT CPU RUNTIME + THREADING EQUIVALENCE PROVED"
    )


if __name__ == "__main__":
    main()