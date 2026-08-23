#!/usr/bin/env python3
"""
Phase 5.4.1 — Exact CPU Training-Path Feasibility Benchmark

Purpose
-------
Measure the wall-clock cost of the EXACT already-frozen canonical CPU training
batch path before launching all 209,620 optimizer steps.

This is a RUNTIME-FEASIBILITY audit, not a scientific/model-selection phase.

What it does
------------
- requires Phase-5.3.7 launch authorization == ALLOWED
- imports the already-proved Phase-5.3.2b runtime
- constructs a fresh canonical seed-42 model + frozen Adam
- loads the frozen epoch-0 stream and immutable model inputs
- executes exactly TWO real training batches (batch 0 then batch 1)
- verifies those two batches reproduce the already-frozen numerical anchors
- measures wall-clock duration
- projects approximate compute for one epoch and 20 epochs

What it does NOT do
-------------------
- no production checkpoint is written
- no production training state is retained
- no validation is scored
- no test is accessed
- no model selection occurs
- no Phase-5.4 production checkpoint/state is created

Why only two batches
--------------------
Batch 0 and batch 1 already have exact frozen fingerprints from the
checkpoint/resume proof. Reusing them lets runtime feasibility be measured
without introducing an unanchored numerical path.

The benchmark model is discarded after the script exits.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# =============================================================================
# Frozen paths
# =============================================================================

ROUNDTRIP_SOURCE_PATH = Path(
    "scripts/"
    "phase_5_3_2b_checkpoint_resume_roundtrip_proof.py"
)

PHASE_5_3_7_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_7_full_reproduction_integrity_closure_contract.json"
)

PHASE_5_4_AUTHORIZATION_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_4_production_training_launch_authorization.json"
)

PHASE_5_3_6_LAUNCH_CONFIG_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_4_production_training_launch_config.json"
)

AUDIT_DIR = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_4_1"
)

BATCH_TIMING_PATH = (
    AUDIT_DIR
    / "exact_cpu_training_batch_timing.csv"
)

PROJECTION_PATH = (
    AUDIT_DIR
    / "cpu_training_runtime_projection.csv"
)

FINAL_INVARIANT_PATH = (
    AUDIT_DIR
    / "phase_5_4_1_final_invariants.csv"
)

CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_4_1_cpu_training_feasibility_benchmark_contract.json"
)

MANIFEST_PATH = (
    AUDIT_DIR
    / "phase_5_4_1_cpu_training_feasibility_benchmark_manifest.json"
)


# =============================================================================
# Frozen numerical anchors
# =============================================================================

EXPECTED_PYTORCH = "2.7.0"

EXPECTED_INITIAL_MODEL_SHA256 = (
    "49e822ea7fad35c458f47e134c94c05e"
    "ac099b68c5c468e2c71559c8c88998ab"
)

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

EXPECTED_BATCH0_LOSS = 0.7080879807
EXPECTED_BATCH1_LOSS = 0.6636360884

BATCHES_PER_EPOCH = 10_481
TOTAL_EPOCHS = 20
TOTAL_OPTIMIZER_STEPS = 209_620


# =============================================================================
# Helpers
# =============================================================================

def banner(text: str) -> None:
    print("\n" + "=" * 118)
    print(text)
    print("=" * 118)


def require(
    condition: bool,
    message: str,
) -> None:
    if not bool(condition):
        raise AssertionError(message)


def load_json(
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def load_guarded_module(
    path: Path,
    module_name: str,
):
    require(
        path.exists(),
        f"Missing runtime source: {path}",
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    require(
        spec is not None
        and spec.loader is not None,
        f"Could not construct import spec for {path}.",
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
            float(
                actual
            )
        )
        and abs(
            float(
                actual
            )
            - float(
                expected
            )
        )
        <= tolerance
    )


def human_duration(
    seconds: float,
) -> str:
    seconds = float(
        seconds
    )

    if seconds < 60:
        return (
            f"{seconds:.2f} seconds"
        )

    minutes = (
        seconds
        / 60.0
    )

    if minutes < 60:
        return (
            f"{minutes:.2f} minutes"
        )

    hours = (
        minutes
        / 60.0
    )

    if hours < 48:
        return (
            f"{hours:.2f} hours"
        )

    days = (
        hours
        / 24.0
    )

    return (
        f"{days:.2f} days"
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.4.1 — "
        "EXACT CPU TRAINING-PATH FEASIBILITY BENCHMARK"
    )

    print(
        "Production training launched:         NO"
    )
    print(
        "Benchmark training batches:           2"
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
    # Launch authorization
    # =========================================================================

    banner(
        "PHASE-5.3.7 LAUNCH AUTHORIZATION GATE"
    )

    for path in (
        PHASE_5_3_7_CONTRACT_PATH,
        PHASE_5_4_AUTHORIZATION_PATH,
        PHASE_5_3_6_LAUNCH_CONFIG_PATH,
        ROUNDTRIP_SOURCE_PATH,
    ):
        require(
            path.exists(),
            f"Missing prerequisite: {path}",
        )

        print(
            f"FOUND  {path}"
        )

    closure = load_json(
        PHASE_5_3_7_CONTRACT_PATH
    )

    authorization = load_json(
        PHASE_5_4_AUTHORIZATION_PATH
    )

    launch_config = load_json(
        PHASE_5_3_6_LAUNCH_CONFIG_PATH
    )

    require(
        closure.get(
            "status"
        )
        == "FROZEN",
        (
            "Phase-5.3.7 closure is not FROZEN."
        ),
    )

    require(
        closure.get(
            "launch_authorization"
        )
        == "ALLOWED",
        (
            "Phase-5.3.7 did not authorize training."
        ),
    )

    require(
        authorization.get(
            "status"
        )
        == "ALLOWED",
        (
            "Phase-5.4 launch authorization is not ALLOWED."
        ),
    )

    require(
        authorization.get(
            "training_allowed"
        )
        is True,
        (
            "Phase-5.4 authorization training_allowed != True."
        ),
    )

    require(
        int(
            authorization.get(
                "critical_checks_failed",
                -1,
            )
        )
        == 0,
        (
            "Phase-5.3.7 has failed critical checks."
        ),
    )

    require(
        launch_config.get(
            "reference_device"
        )
        == "CPU",
        (
            "Frozen reference device is not CPU."
        ),
    )

    require(
        launch_config.get(
            "initial_model_sha256"
        )
        == EXPECTED_INITIAL_MODEL_SHA256,
        (
            "Launch-config initial model SHA drift."
        ),
    )

    print(
        "Phase-5.3.7 closure:                  FROZEN / PASS"
    )
    print(
        "Phase-5.4 launch authorization:       ALLOWED"
    )
    print(
        "Frozen reference device:              CPU"
    )

    # =========================================================================
    # Environment
    # =========================================================================

    banner(
        "BENCHMARK ENVIRONMENT"
    )

    print(
        f"Platform:                              "
        f"{platform.platform()}"
    )
    print(
        f"Processor:                             "
        f"{platform.processor()}"
    )
    print(
        f"Python:                                "
        f"{platform.python_version()}"
    )
    print(
        f"PyTorch:                               "
        f"{torch.__version__}"
    )
    print(
        f"Torch CPU threads:                     "
        f"{torch.get_num_threads()}"
    )
    print(
        f"Torch interop threads:                 "
        f"{torch.get_num_interop_threads()}"
    )

    require(
        torch.__version__
        == EXPECTED_PYTORCH,
        (
            "Reference PyTorch runtime drift: "
            f"{torch.__version__} != {EXPECTED_PYTORCH}"
        ),
    )

    # =========================================================================
    # Load exact frozen runtime
    # =========================================================================

    banner(
        "LOAD EXACT FROZEN TRAINING RUNTIME"
    )

    roundtrip = load_guarded_module(
        ROUNDTRIP_SOURCE_PATH,
        "_itrs_phase5_4_1_roundtrip_runtime",
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

    initial_sha = (
        canonical_hash_fn(
            model
        )
    )

    require(
        initial_sha
        == EXPECTED_INITIAL_MODEL_SHA256,
        (
            "Benchmark model does not begin at "
            "canonical initialization."
        ),
    )

    print(
        "Canonical seed-42 model:              PASS"
    )
    print(
        "Frozen Adam:                         PASS"
    )
    print(
        "Frozen epoch-0 stream:               PASS"
    )
    print(
        "Immutable feature runtime:           LOADED"
    )

    # =========================================================================
    # Decode anchored batches before starting timer
    # =========================================================================

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

    # =========================================================================
    # Exact batch 0
    # =========================================================================

    banner(
        "BENCHMARK EXACT FROZEN BATCH 0"
    )

    start0 = time.perf_counter()

    result0 = (
        roundtrip
        .execute_training_batch(
            model,
            optimizer,
            canonical_hash_fn,
            batch0,
            shared,
        )
    )

    end0 = time.perf_counter()

    seconds0 = (
        end0
        - start0
    )

    require(
        close_float(
            result0[
                "loss"
            ],
            EXPECTED_BATCH0_LOSS,
        ),
        (
            "Batch-0 loss drift."
        ),
    )

    require(
        result0[
            "logit_sha256"
        ]
        == EXPECTED_BATCH0_LOGIT_SHA256,
        (
            "Batch-0 logit SHA drift."
        ),
    )

    require(
        result0[
            "gradient_sha256"
        ]
        == EXPECTED_BATCH0_GRADIENT_SHA256,
        (
            "Batch-0 gradient SHA drift."
        ),
    )

    require(
        result0[
            "post_step_model_sha256"
        ]
        == EXPECTED_BATCH0_POST_MODEL_SHA256,
        (
            "Batch-0 post-step model SHA drift."
        ),
    )

    require(
        result0[
            "optimizer_state_sha256"
        ]
        == EXPECTED_BATCH0_OPTIMIZER_SHA256,
        (
            "Batch-0 optimizer SHA drift."
        ),
    )

    print(
        f"Batch-0 BCE:                          "
        f"{result0['loss']:.10f}"
    )
    print(
        f"Batch-0 wall time:                    "
        f"{seconds0:.3f} s"
    )
    print(
        "Batch-0 numerical anchors:            EXACT"
    )

    # =========================================================================
    # Exact batch 1
    # =========================================================================

    banner(
        "BENCHMARK EXACT FROZEN BATCH 1"
    )

    start1 = time.perf_counter()

    result1 = (
        roundtrip
        .execute_training_batch(
            model,
            optimizer,
            canonical_hash_fn,
            batch1,
            shared,
        )
    )

    end1 = time.perf_counter()

    seconds1 = (
        end1
        - start1
    )

    require(
        close_float(
            result1[
                "loss"
            ],
            EXPECTED_BATCH1_LOSS,
        ),
        (
            "Batch-1 loss drift."
        ),
    )

    require(
        result1[
            "logit_sha256"
        ]
        == EXPECTED_BATCH1_LOGIT_SHA256,
        (
            "Batch-1 logit SHA drift."
        ),
    )

    require(
        result1[
            "gradient_sha256"
        ]
        == EXPECTED_BATCH1_GRADIENT_SHA256,
        (
            "Batch-1 gradient SHA drift."
        ),
    )

    require(
        result1[
            "post_step_model_sha256"
        ]
        == EXPECTED_BATCH1_POST_MODEL_SHA256,
        (
            "Batch-1 post-step model SHA drift."
        ),
    )

    require(
        result1[
            "optimizer_state_sha256"
        ]
        == EXPECTED_BATCH1_OPTIMIZER_SHA256,
        (
            "Batch-1 optimizer SHA drift."
        ),
    )

    print(
        f"Batch-1 BCE:                          "
        f"{result1['loss']:.10f}"
    )
    print(
        f"Batch-1 wall time:                    "
        f"{seconds1:.3f} s"
    )
    print(
        "Batch-1 numerical anchors:            EXACT"
    )

    # =========================================================================
    # Runtime projection
    # =========================================================================

    banner(
        "CPU RUNTIME PROJECTION"
    )

    mean_seconds = float(
        np.mean(
            [
                seconds0,
                seconds1,
            ]
        )
    )

    median_seconds = float(
        np.median(
            [
                seconds0,
                seconds1,
            ]
        )
    )

    conservative_seconds = float(
        max(
            seconds0,
            seconds1,
        )
    )

    epoch_mean_projection = (
        mean_seconds
        * BATCHES_PER_EPOCH
    )

    epoch_conservative_projection = (
        conservative_seconds
        * BATCHES_PER_EPOCH
    )

    full_mean_projection = (
        mean_seconds
        * TOTAL_OPTIMIZER_STEPS
    )

    full_conservative_projection = (
        conservative_seconds
        * TOTAL_OPTIMIZER_STEPS
    )

    print(
        f"Mean seconds / batch:                 "
        f"{mean_seconds:.3f}"
    )
    print(
        f"Median seconds / batch:               "
        f"{median_seconds:.3f}"
    )
    print(
        f"Conservative seconds / batch:         "
        f"{conservative_seconds:.3f}"
    )
    print()
    print(
        "Projected one epoch from mean:        "
        f"{human_duration(epoch_mean_projection)}"
    )
    print(
        "Projected one epoch conservative:     "
        f"{human_duration(epoch_conservative_projection)}"
    )
    print()
    print(
        "Projected 20 epochs from mean:        "
        f"{human_duration(full_mean_projection)}"
    )
    print(
        "Projected 20 epochs conservative:     "
        f"{human_duration(full_conservative_projection)}"
    )

    timing_df = pd.DataFrame(
        [
            {
                "batch_index": (
                    0
                ),
                "batch_size": (
                    int(
                        result0[
                            "batch_size"
                        ]
                    )
                ),
                "loss": (
                    float(
                        result0[
                            "loss"
                        ]
                    )
                ),
                "wall_seconds": (
                    seconds0
                ),
                "logit_sha256": (
                    result0[
                        "logit_sha256"
                    ]
                ),
                "gradient_sha256": (
                    result0[
                        "gradient_sha256"
                    ]
                ),
                "post_step_model_sha256": (
                    result0[
                        "post_step_model_sha256"
                    ]
                ),
                "optimizer_state_sha256": (
                    result0[
                        "optimizer_state_sha256"
                    ]
                ),
                "numerical_anchor": (
                    "EXACT_FROZEN"
                ),
            },
            {
                "batch_index": (
                    1
                ),
                "batch_size": (
                    int(
                        result1[
                            "batch_size"
                        ]
                    )
                ),
                "loss": (
                    float(
                        result1[
                            "loss"
                        ]
                    )
                ),
                "wall_seconds": (
                    seconds1
                ),
                "logit_sha256": (
                    result1[
                        "logit_sha256"
                    ]
                ),
                "gradient_sha256": (
                    result1[
                        "gradient_sha256"
                    ]
                ),
                "post_step_model_sha256": (
                    result1[
                        "post_step_model_sha256"
                    ]
                ),
                "optimizer_state_sha256": (
                    result1[
                        "optimizer_state_sha256"
                    ]
                ),
                "numerical_anchor": (
                    "EXACT_FROZEN"
                ),
            },
        ]
    )

    projection_df = pd.DataFrame(
        [
            {
                "projection": (
                    "mean_seconds_per_batch"
                ),
                "value": (
                    mean_seconds
                ),
                "human": (
                    f"{mean_seconds:.3f} s"
                ),
            },
            {
                "projection": (
                    "conservative_seconds_per_batch"
                ),
                "value": (
                    conservative_seconds
                ),
                "human": (
                    f"{conservative_seconds:.3f} s"
                ),
            },
            {
                "projection": (
                    "one_epoch_mean"
                ),
                "value": (
                    epoch_mean_projection
                ),
                "human": (
                    human_duration(
                        epoch_mean_projection
                    )
                ),
            },
            {
                "projection": (
                    "one_epoch_conservative"
                ),
                "value": (
                    epoch_conservative_projection
                ),
                "human": (
                    human_duration(
                        epoch_conservative_projection
                    )
                ),
            },
            {
                "projection": (
                    "20_epochs_mean"
                ),
                "value": (
                    full_mean_projection
                ),
                "human": (
                    human_duration(
                        full_mean_projection
                    )
                ),
            },
            {
                "projection": (
                    "20_epochs_conservative"
                ),
                "value": (
                    full_conservative_projection
                ),
                "human": (
                    human_duration(
                        full_conservative_projection
                    )
                ),
            },
        ]
    )

    # =========================================================================
    # Final invariants
    # =========================================================================

    banner(
        "FINAL PHASE-5.4.1 INVARIANTS"
    )

    checks = [
        (
            "phase_5_3_7_authorization_allowed",
            (
                authorization.get(
                    "training_allowed"
                )
                is True
            ),
        ),
        (
            "reference_device_cpu",
            (
                launch_config.get(
                    "reference_device"
                )
                == "CPU"
            ),
        ),
        (
            "pytorch_2_7_0",
            (
                torch.__version__
                == EXPECTED_PYTORCH
            ),
        ),
        (
            "initial_model_sha_exact",
            (
                initial_sha
                == EXPECTED_INITIAL_MODEL_SHA256
            ),
        ),
        (
            "batch0_loss_exact",
            close_float(
                result0[
                    "loss"
                ],
                EXPECTED_BATCH0_LOSS,
            ),
        ),
        (
            "batch0_logit_sha_exact",
            (
                result0[
                    "logit_sha256"
                ]
                == EXPECTED_BATCH0_LOGIT_SHA256
            ),
        ),
        (
            "batch0_gradient_sha_exact",
            (
                result0[
                    "gradient_sha256"
                ]
                == EXPECTED_BATCH0_GRADIENT_SHA256
            ),
        ),
        (
            "batch0_post_model_sha_exact",
            (
                result0[
                    "post_step_model_sha256"
                ]
                == EXPECTED_BATCH0_POST_MODEL_SHA256
            ),
        ),
        (
            "batch0_optimizer_sha_exact",
            (
                result0[
                    "optimizer_state_sha256"
                ]
                == EXPECTED_BATCH0_OPTIMIZER_SHA256
            ),
        ),
        (
            "batch1_loss_exact",
            close_float(
                result1[
                    "loss"
                ],
                EXPECTED_BATCH1_LOSS,
            ),
        ),
        (
            "batch1_logit_sha_exact",
            (
                result1[
                    "logit_sha256"
                ]
                == EXPECTED_BATCH1_LOGIT_SHA256
            ),
        ),
        (
            "batch1_gradient_sha_exact",
            (
                result1[
                    "gradient_sha256"
                ]
                == EXPECTED_BATCH1_GRADIENT_SHA256
            ),
        ),
        (
            "batch1_post_model_sha_exact",
            (
                result1[
                    "post_step_model_sha256"
                ]
                == EXPECTED_BATCH1_POST_MODEL_SHA256
            ),
        ),
        (
            "batch1_optimizer_sha_exact",
            (
                result1[
                    "optimizer_state_sha256"
                ]
                == EXPECTED_BATCH1_OPTIMIZER_SHA256
            ),
        ),
        (
            "timings_positive_finite",
            (
                math.isfinite(
                    seconds0
                )
                and math.isfinite(
                    seconds1
                )
                and seconds0 > 0
                and seconds1 > 0
            ),
        ),
        (
            "production_checkpoint_not_written",
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
            "At least one Phase-5.4.1 "
            "feasibility invariant failed."
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
        "WRITE PHASE-5.4.1 OUTPUTS"
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTRACT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    timing_df.to_csv(
        BATCH_TIMING_PATH,
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
            "5.4.1"
        ),
        "title": (
            "Exact CPU Training-Path Feasibility Benchmark"
        ),
        "status": (
            "COMPLETE"
        ),
        "classification": (
            "RUNTIME_FEASIBILITY_ONLY"
        ),
        "reference_device": (
            "CPU"
        ),
        "reference_pytorch": (
            torch.__version__
        ),
        "benchmark_batches": (
            2
        ),
        "production_training_launched": (
            False
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
        "batch_seconds": {
            "batch0": (
                seconds0
            ),
            "batch1": (
                seconds1
            ),
            "mean": (
                mean_seconds
            ),
            "median": (
                median_seconds
            ),
            "conservative_max": (
                conservative_seconds
            ),
        },
        "projection_seconds": {
            "one_epoch_mean": (
                epoch_mean_projection
            ),
            "one_epoch_conservative": (
                epoch_conservative_projection
            ),
            "20_epochs_mean": (
                full_mean_projection
            ),
            "20_epochs_conservative": (
                full_conservative_projection
            ),
        },
        "numerical_anchor_status": (
            "EXACT_FROZEN_BATCH0_AND_BATCH1"
        ),
        "next_action": (
            "ASSESS_PRODUCTION_CPU_FEASIBILITY_AND_"
            "LAUNCH_OR_OPTIMIZE_IMPLEMENTATION_EQUIVALENT_RUNTIME"
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
            "5.4.1"
        ),
        "status": (
            "CPU_TRAINING_FEASIBILITY_BENCHMARK_COMPLETE"
        ),
        "batch0_seconds": (
            seconds0
        ),
        "batch1_seconds": (
            seconds1
        ),
        "mean_seconds_per_batch": (
            mean_seconds
        ),
        "projected_epoch_human": (
            human_duration(
                epoch_mean_projection
            )
        ),
        "projected_20_epochs_human": (
            human_duration(
                full_mean_projection
            )
        ),
        "training_steps_executed_in_benchmark_model": (
            2
        ),
        "production_training_steps_executed": (
            0
        ),
        "production_checkpoint_written": (
            False
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
        BATCH_TIMING_PATH,
        PROJECTION_PATH,
        FINAL_INVARIANT_PATH,
        CONTRACT_PATH,
        MANIFEST_PATH,
    ):
        print(
            f"WROTE  {path}"
        )

    # =========================================================================
    # Explicitly discard benchmark state
    # =========================================================================

    del batch0
    del batch1
    del model
    del optimizer
    del stream
    del shared

    gc.collect()

    banner(
        "PHASE 5.4.1 FINAL STATUS"
    )

    print(
        "Frozen numerical path:                EXACT"
    )
    print(
        f"Mean CPU seconds / batch:             "
        f"{mean_seconds:.3f}"
    )
    print(
        "Projected one epoch:                  "
        f"{human_duration(epoch_mean_projection)}"
    )
    print(
        "Projected full 20 epochs:             "
        f"{human_duration(full_mean_projection)}"
    )
    print()
    print(
        "Production training launched:         NO"
    )
    print(
        "Benchmark model retained:             NO"
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
        "PHASE 5.4.1 COMPLETE / "
        "EXACT CPU TRAINING-PATH FEASIBILITY BENCHMARK PASSED"
    )


if __name__ == "__main__":
    main()