#!/usr/bin/env python3
"""
Phase 5.4.6 — Sparse Embedding Backward Equivalence + Acceleration Audit

Purpose
-------
Test a closer-to-canonical acceleration than packed embedding lookup.

Instead of changing call structure, this audit keeps EVERY canonical
nn.Embedding call separate and in the same order, but toggles the embedding
module's `sparse` backward mode.

Why this targets the bottleneck
-------------------------------
Phase 5.4.4 showed:
    - 7,085 EmbeddingBackward evaluations in one batch
    - huge aten::fill_ + aten::add_ cost from dense embedding gradients

With sparse=True, each lookup can emit only the rows it touched instead of a
full embedding-table gradient. The same embedding weight parameters are still
used by the structural R-GCN branch, which already contributes dense gradients.

Candidate variants
------------------
C_STARTUP_SPARSE
    startup_embedding.sparse = True
    investor_embedding.sparse = False

D_BOTH_SPARSE
    startup_embedding.sparse = True
    investor_embedding.sparse = True

Acceptance rule
---------------
A variant is eligible ONLY if batch 0 reproduces exactly:
    - BCE
    - logit SHA256
    - gradient SHA256
    - post-step model SHA256
    - optimizer-state SHA256

The fastest eligible variant must then reproduce the exact batch0 -> batch1
trajectory.

No production checkpoint is written.
No validation/test is accessed.
No production training state is retained.
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
# Paths
# =============================================================================

ROUNDTRIP_SOURCE_PATH = Path(
    "scripts/"
    "phase_5_3_2b_checkpoint_resume_roundtrip_proof.py"
)

PHASE_5_4_4_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_4_4_exact_autograd_operator_profile_contract.json"
)

PHASE_5_4_5_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_4_5_packed_embedding_acceleration_contract.json"
)

PHASE_5_4_2_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_4_2_lean_exact_cpu_runtime_contract.json"
)

PHASE_5_4_AUTHORIZATION_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_4_production_training_launch_authorization.json"
)

AUDIT_DIR = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_4_6"
)

VARIANT_AUDIT_PATH = (
    AUDIT_DIR
    / "sparse_embedding_variant_equivalence_audit.csv"
)

TWO_STEP_PATH = (
    AUDIT_DIR
    / "selected_sparse_embedding_two_step_proof.csv"
)

PROJECTION_PATH = (
    AUDIT_DIR
    / "sparse_embedding_runtime_projection.csv"
)

FINAL_INVARIANT_PATH = (
    AUDIT_DIR
    / "phase_5_4_6_final_invariants.csv"
)

CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_4_6_sparse_embedding_acceleration_contract.json"
)

MANIFEST_PATH = (
    AUDIT_DIR
    / "phase_5_4_6_sparse_embedding_acceleration_manifest.json"
)


# =============================================================================
# Frozen anchors
# =============================================================================

EXPECTED_PYTORCH = "2.7.0"
EXPECTED_SELECTED_THREADS = 8

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

VARIANT_STARTUP = "C_STARTUP_SPARSE"
VARIANT_BOTH = "D_BOTH_SPARSE"


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


def load_module(path: Path, module_name: str):
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


def human_duration(seconds: float) -> str:
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


def run_sparse_candidate(
    *,
    roundtrip,
    preflight,
    batch,
    shared,
    variant: str,
    expected: dict,
) -> dict:

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

    initial_sha = canonical_hash_fn(
        model
    )

    require(
        initial_sha
        == EXPECTED_INITIAL_MODEL_SHA256,
        (
            "Fresh candidate model initial SHA drift."
        ),
    )

    original_startup_sparse = bool(
        model.startup_embedding.sparse
    )

    original_investor_sparse = bool(
        model.investor_embedding.sparse
    )

    if variant == VARIANT_STARTUP:
        model.startup_embedding.sparse = True
        model.investor_embedding.sparse = False

    elif variant == VARIANT_BOTH:
        model.startup_embedding.sparse = True
        model.investor_embedding.sparse = True

    else:
        raise AssertionError(
            f"Unknown sparse candidate: {variant}"
        )

    true_gradient_hash_fn = (
        roundtrip
        .gradient_logical_sha256
    )

    true_optimizer_hash_fn = (
        roundtrip
        .optimizer_state_logical_sha256
    )

    # Keep proof hashing out of the timed path.
    roundtrip.gradient_logical_sha256 = (
        lambda model: "SKIPPED_INSIDE_SPARSE_TIMER"
    )

    roundtrip.optimizer_state_logical_sha256 = (
        lambda model, optimizer: "SKIPPED_INSIDE_SPARSE_TIMER"
    )

    dummy_model_hash_fn = (
        lambda model: "SKIPPED_INSIDE_SPARSE_TIMER"
    )

    start = time.perf_counter()

    try:
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

        # execute_training_batch has already performed backward and Adam.
        # Gradients remain attached after optimizer.step(), allowing exact
        # post-hoc hashing.
        gradient_layouts = {
            name: str(
                parameter.grad.layout
            )
            if parameter.grad is not None
            else "NONE"
            for name, parameter
            in model.named_parameters()
        }

        sparse_grad_parameters = [
            name
            for name, parameter
            in model.named_parameters()
            if (
                parameter.grad is not None
                and parameter.grad.is_sparse
            )
        ]

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
            expected[
                "loss"
            ],
        )

        logit_exact = (
            result[
                "logit_sha256"
            ]
            == expected[
                "logit_sha256"
            ]
        )

        gradient_exact = (
            gradient_sha
            == expected[
                "gradient_sha256"
            ]
        )

        model_exact = (
            post_model_sha
            == expected[
                "post_model_sha256"
            ]
        )

        optimizer_exact = (
            optimizer_sha
            == expected[
                "optimizer_state_sha256"
            ]
        )

        exact_all = bool(
            loss_exact
            and logit_exact
            and gradient_exact
            and model_exact
            and optimizer_exact
        )

        row = {
            "variant": (
                variant
            ),
            "elapsed_seconds": (
                float(elapsed)
            ),
            "startup_sparse": (
                bool(
                    model.startup_embedding.sparse
                )
            ),
            "investor_sparse": (
                bool(
                    model.investor_embedding.sparse
                )
            ),
            "sparse_grad_parameter_count_after_step": (
                len(
                    sparse_grad_parameters
                )
            ),
            "sparse_grad_parameters_after_step": (
                ";".join(
                    sparse_grad_parameters
                )
            ),
            "startup_grad_layout": (
                gradient_layouts.get(
                    "startup_embedding.weight",
                    "MISSING",
                )
            ),
            "investor_grad_layout": (
                gradient_layouts.get(
                    "investor_embedding.weight",
                    "MISSING",
                )
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
            "runtime_error": (
                ""
            ),
        }

    except Exception as exc:
        elapsed = (
            time.perf_counter()
            - start
        )

        row = {
            "variant": (
                variant
            ),
            "elapsed_seconds": (
                float(elapsed)
            ),
            "startup_sparse": (
                variant
                in (
                    VARIANT_STARTUP,
                    VARIANT_BOTH,
                )
            ),
            "investor_sparse": (
                variant
                == VARIANT_BOTH
            ),
            "sparse_grad_parameter_count_after_step": (
                np.nan
            ),
            "sparse_grad_parameters_after_step": (
                ""
            ),
            "startup_grad_layout": (
                ""
            ),
            "investor_grad_layout": (
                ""
            ),
            "loss": (
                np.nan
            ),
            "loss_exact": (
                False
            ),
            "logit_exact": (
                False
            ),
            "gradient_exact": (
                False
            ),
            "post_model_exact": (
                False
            ),
            "optimizer_exact": (
                False
            ),
            "byte_exact_eligible": (
                False
            ),
            "logit_sha256": (
                ""
            ),
            "gradient_sha256": (
                ""
            ),
            "post_model_sha256": (
                ""
            ),
            "optimizer_state_sha256": (
                ""
            ),
            "runtime_error": (
                f"{type(exc).__name__}: {exc}"
            ),
        }

    finally:
        model.startup_embedding.sparse = (
            original_startup_sparse
        )

        model.investor_embedding.sparse = (
            original_investor_sparse
        )

        roundtrip.gradient_logical_sha256 = (
            true_gradient_hash_fn
        )

        roundtrip.optimizer_state_logical_sha256 = (
            true_optimizer_hash_fn
        )

    return (
        row,
        model,
        optimizer,
        canonical_hash_fn,
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.4.6 — "
        "SPARSE EMBEDDING BACKWARD EQUIVALENCE + ACCELERATION AUDIT"
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
    # Gate
    # =========================================================================

    banner(
        "PREREQUISITE GATE"
    )

    for path in (
        ROUNDTRIP_SOURCE_PATH,
        PHASE_5_4_2_CONTRACT_PATH,
        PHASE_5_4_4_CONTRACT_PATH,
        PHASE_5_4_5_CONTRACT_PATH,
        PHASE_5_4_AUTHORIZATION_PATH,
    ):
        require(
            path.exists(),
            f"Missing prerequisite: {path}",
        )

        print(
            f"FOUND  {path}"
        )

    runtime_contract = load_json(
        PHASE_5_4_2_CONTRACT_PATH
    )

    operator_contract = load_json(
        PHASE_5_4_4_CONTRACT_PATH
    )

    packed_contract = load_json(
        PHASE_5_4_5_CONTRACT_PATH
    )

    authorization = load_json(
        PHASE_5_4_AUTHORIZATION_PATH
    )

    require(
        runtime_contract.get(
            "status"
        )
        == "COMPLETE",
        (
            "Phase-5.4.2 runtime contract is not COMPLETE."
        ),
    )

    require(
        int(
            runtime_contract.get(
                "selected_threads",
                -1,
            )
        )
        == EXPECTED_SELECTED_THREADS,
        (
            "Selected exact CPU thread count drift."
        ),
    )

    require(
        operator_contract.get(
            "status"
        )
        == "COMPLETE",
        (
            "Phase-5.4.4 operator profile is not COMPLETE."
        ),
    )

    require(
        packed_contract.get(
            "status"
        )
        == "COMPLETE",
        (
            "Phase-5.4.5 packed audit is not COMPLETE."
        ),
    )

    require(
        packed_contract.get(
            "acceleration_status"
        )
        == "NO_BYTE_EXACT_PACKED_RUNTIME",
        (
            "Phase-5.4.5 result changed; sparse audit "
            "prerequisite is no longer the expected state."
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

    lean_exact_seconds = float(
        runtime_contract[
            "lean_exact_mean_seconds_per_batch"
        ]
    )

    print(
        "Phase-5.4.4 operator diagnosis:       PASS"
    )
    print(
        "Phase-5.4.5 packed exact runtime:     REJECTED"
    )
    print(
        f"Canonical lean seconds / batch:       "
        f"{lean_exact_seconds:.3f}"
    )

    # =========================================================================
    # Load runtime
    # =========================================================================

    banner(
        "LOAD FROZEN RUNTIME"
    )

    roundtrip = load_module(
        ROUNDTRIP_SOURCE_PATH,
        "_itrs_phase5_4_6_roundtrip",
    )

    require(
        torch.__version__
        == EXPECTED_PYTORCH,
        (
            "Reference PyTorch version drift."
        ),
    )

    torch.set_num_threads(
        EXPECTED_SELECTED_THREADS
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

    print(
        f"Platform:                              "
        f"{platform.platform()}"
    )
    print(
        f"PyTorch:                               "
        f"{torch.__version__}"
    )
    print(
        f"Torch intra-op threads:                "
        f"{torch.get_num_threads()}"
    )

    # =========================================================================
    # Batch-0 candidate screen
    # =========================================================================

    banner(
        "BATCH-0 SPARSE-BACKWARD VARIANT SCREEN"
    )

    expected0 = {
        "loss": (
            EXPECTED_BATCH0_LOSS
        ),
        "logit_sha256": (
            EXPECTED_BATCH0_LOGIT_SHA256
        ),
        "gradient_sha256": (
            EXPECTED_BATCH0_GRADIENT_SHA256
        ),
        "post_model_sha256": (
            EXPECTED_BATCH0_POST_MODEL_SHA256
        ),
        "optimizer_state_sha256": (
            EXPECTED_BATCH0_OPTIMIZER_SHA256
        ),
    }

    candidate_rows = []

    for variant in (
        VARIANT_STARTUP,
        VARIANT_BOTH,
    ):

        (
            row,
            model,
            optimizer,
            _canonical_hash_fn,
        ) = run_sparse_candidate(
            roundtrip=roundtrip,
            preflight=preflight,
            batch=batch0,
            shared=shared,
            variant=variant,
            expected=expected0,
        )

        candidate_rows.append(
            row
        )

        print(
            f"{variant:24s} | "
            f"time={row['elapsed_seconds']:.3f}s | "
            f"logit={'PASS' if row['logit_exact'] else 'FAIL'} | "
            f"grad={'PASS' if row['gradient_exact'] else 'FAIL'} | "
            f"state={'PASS' if row['post_model_exact'] and row['optimizer_exact'] else 'FAIL'} | "
            f"eligible={'YES' if row['byte_exact_eligible'] else 'NO'}"
        )

        if row[
            "runtime_error"
        ]:
            print(
                f"  error: {row['runtime_error']}"
            )
        else:
            print(
                f"  final grad layouts: "
                f"startup={row['startup_grad_layout']}, "
                f"investor={row['investor_grad_layout']}"
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

    # =========================================================================
    # Select exact candidate
    # =========================================================================

    banner(
        "SPARSE-BACKWARD EXACTNESS DECISION"
    )

    if len(
        eligible_df
    ) == 0:

        selected_variant = None
        two_step_df = pd.DataFrame()

        print(
            "No sparse-backward variant preserved the "
            "byte-exact frozen batch-0 trajectory."
        )

    else:
        selected = (
            eligible_df.sort_values(
                [
                    "elapsed_seconds",
                    "variant",
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

        selected_variant = str(
            selected[
                "variant"
            ]
        )

        print(
            f"Selected exact variant:               "
            f"{selected_variant}"
        )
        print(
            f"Batch-0 candidate time:               "
            f"{float(selected['elapsed_seconds']):.3f} s"
        )

        # =====================================================================
        # Two-step proof must use ONE stateful model.
        # Therefore reproduce variant manually instead of fresh-state helper.
        # =====================================================================

        banner(
            "SELECTED SPARSE VARIANT — EXACT TWO-STEP PROOF"
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
                "Selected sparse candidate initial SHA drift."
            ),
        )

        if selected_variant == VARIANT_STARTUP:
            model.startup_embedding.sparse = True
            model.investor_embedding.sparse = False

        elif selected_variant == VARIANT_BOTH:
            model.startup_embedding.sparse = True
            model.investor_embedding.sparse = True

        true_gradient_hash_fn = (
            roundtrip
            .gradient_logical_sha256
        )

        true_optimizer_hash_fn = (
            roundtrip
            .optimizer_state_logical_sha256
        )

        roundtrip.gradient_logical_sha256 = (
            lambda model: "SKIPPED_INSIDE_TWO_STEP_TIMER"
        )

        roundtrip.optimizer_state_logical_sha256 = (
            lambda model, optimizer: "SKIPPED_INSIDE_TWO_STEP_TIMER"
        )

        dummy_model_hash_fn = (
            lambda model: "SKIPPED_INSIDE_TWO_STEP_TIMER"
        )

        expected1 = {
            "loss": (
                EXPECTED_BATCH1_LOSS
            ),
            "logit_sha256": (
                EXPECTED_BATCH1_LOGIT_SHA256
            ),
            "gradient_sha256": (
                EXPECTED_BATCH1_GRADIENT_SHA256
            ),
            "post_model_sha256": (
                EXPECTED_BATCH1_POST_MODEL_SHA256
            ),
            "optimizer_state_sha256": (
                EXPECTED_BATCH1_OPTIMIZER_SHA256
            ),
        }

        two_step_rows = []

        try:
            for (
                batch_index,
                batch,
                expected,
            ) in (
                (
                    0,
                    batch0,
                    expected0,
                ),
                (
                    1,
                    batch1,
                    expected1,
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

                model_sha = (
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

                exact = bool(
                    close_float(
                        result[
                            "loss"
                        ],
                        expected[
                            "loss"
                        ],
                    )
                    and result[
                        "logit_sha256"
                    ]
                    == expected[
                        "logit_sha256"
                    ]
                    and gradient_sha
                    == expected[
                        "gradient_sha256"
                    ]
                    and model_sha
                    == expected[
                        "post_model_sha256"
                    ]
                    and optimizer_sha
                    == expected[
                        "optimizer_state_sha256"
                    ]
                )

                require(
                    exact,
                    (
                        f"Selected sparse runtime failed "
                        f"two-step proof at batch {batch_index}."
                    ),
                )

                two_step_rows.append(
                    {
                        "batch_index": (
                            batch_index
                        ),
                        "variant": (
                            selected_variant
                        ),
                        "elapsed_seconds": (
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
                            model_sha
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
                    f"time={elapsed:.3f}s | EXACT"
                )

        finally:
            roundtrip.gradient_logical_sha256 = (
                true_gradient_hash_fn
            )

            roundtrip.optimizer_state_logical_sha256 = (
                true_optimizer_hash_fn
            )

        two_step_df = pd.DataFrame(
            two_step_rows
        )

        del model
        del optimizer

        gc.collect()

    # =========================================================================
    # Projection
    # =========================================================================

    banner(
        "SPARSE-BACKWARD RUNTIME PROJECTION"
    )

    if (
        selected_variant
        is not None
        and len(
            two_step_df
        )
        == 2
    ):
        selected_mean_seconds = float(
            two_step_df[
                "elapsed_seconds"
            ].mean()
        )

        speedup = (
            lean_exact_seconds
            / selected_mean_seconds
        )

        projected_epoch_seconds = (
            selected_mean_seconds
            * BATCHES_PER_EPOCH
        )

        projected_full_seconds = (
            selected_mean_seconds
            * TOTAL_OPTIMIZER_STEPS
        )

        print(
            f"Selected exact runtime:               "
            f"{selected_variant}"
        )
        print(
            f"Mean seconds / batch:                 "
            f"{selected_mean_seconds:.3f}"
        )
        print(
            f"Speedup vs canonical lean CPU:        "
            f"{speedup:.2f}x"
        )
        print(
            "Projected one epoch:                 "
            f"{human_duration(projected_epoch_seconds)}"
        )
        print(
            "Projected 20 epochs:                 "
            f"{human_duration(projected_full_seconds)}"
        )

        acceleration_status = (
            "EXACT_SPARSE_BACKWARD_RUNTIME_PROVED"
        )

        projection_df = pd.DataFrame(
            [
                {
                    "metric": (
                        "canonical_lean_seconds_per_batch"
                    ),
                    "value": (
                        lean_exact_seconds
                    ),
                    "human": (
                        f"{lean_exact_seconds:.3f} s"
                    ),
                },
                {
                    "metric": (
                        "selected_sparse_seconds_per_batch"
                    ),
                    "value": (
                        selected_mean_seconds
                    ),
                    "human": (
                        f"{selected_mean_seconds:.3f} s"
                    ),
                },
                {
                    "metric": (
                        "speedup"
                    ),
                    "value": (
                        speedup
                    ),
                    "human": (
                        f"{speedup:.2f}x"
                    ),
                },
                {
                    "metric": (
                        "one_epoch"
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
                        "20_epochs"
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
            ]
        )

    else:
        selected_mean_seconds = None
        speedup = None
        projected_epoch_seconds = None
        projected_full_seconds = None

        acceleration_status = (
            "NO_BYTE_EXACT_SPARSE_BACKWARD_RUNTIME"
        )

        projection_df = pd.DataFrame(
            [
                {
                    "metric": (
                        "canonical_lean_seconds_per_batch"
                    ),
                    "value": (
                        lean_exact_seconds
                    ),
                    "human": (
                        f"{lean_exact_seconds:.3f} s"
                    ),
                },
                {
                    "metric": (
                        "selected_sparse_runtime"
                    ),
                    "value": (
                        np.nan
                    ),
                    "human": (
                        "NO_BYTE_EXACT_VARIANT"
                    ),
                },
            ]
        )

        print(
            "No byte-exact sparse-backward runtime accepted."
        )

    # =========================================================================
    # Final invariants
    # =========================================================================

    banner(
        "FINAL PHASE-5.4.6 INVARIANTS"
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
            "startup_sparse_variant_tested",
            (
                VARIANT_STARTUP
                in candidate_df[
                    "variant"
                ].tolist()
            ),
        ),
        (
            "both_sparse_variant_tested",
            (
                VARIANT_BOTH
                in candidate_df[
                    "variant"
                ].tolist()
            ),
        ),
        (
            "no_inexact_variant_accepted",
            True,
        ),
        (
            "selected_two_step_exact_if_any",
            (
                True
                if selected_variant is None
                else (
                    len(
                        two_step_df
                    )
                    == 2
                    and bool(
                        two_step_df[
                            "exact_frozen"
                        ].all()
                    )
                )
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
            for name, passed
            in checks
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
            "At least one Phase-5.4.6 invariant failed."
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
        "WRITE PHASE-5.4.6 OUTPUTS"
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
        VARIANT_AUDIT_PATH,
        index=False,
    )

    two_step_df.to_csv(
        TWO_STEP_PATH,
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
            "5.4.6"
        ),
        "title": (
            "Sparse Embedding Backward Equivalence + Acceleration Audit"
        ),
        "status": (
            "COMPLETE"
        ),
        "classification": (
            "IMPLEMENTATION_EQUIVALENT_ACCELERATION_AUDIT"
        ),
        "canonical_lean_seconds_per_batch": (
            lean_exact_seconds
        ),
        "candidate_variants": (
            candidate_df.to_dict(
                orient="records"
            )
        ),
        "selected_variant": (
            selected_variant
        ),
        "acceleration_status": (
            acceleration_status
        ),
        "selected_two_step_exact": (
            (
                selected_variant
                is not None
            )
            and (
                len(
                    two_step_df
                )
                == 2
            )
            and bool(
                two_step_df[
                    "exact_frozen"
                ].all()
            )
        ),
        "selected_mean_seconds_per_batch": (
            selected_mean_seconds
        ),
        "selected_speedup": (
            speedup
        ),
        "projected_one_epoch_seconds": (
            projected_epoch_seconds
        ),
        "projected_20_epochs_seconds": (
            projected_full_seconds
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
        "next_phase": (
            (
                "5.4.7_FREEZE_EXACT_ACCELERATED_PRODUCTION_RUNTIME"
            )
            if acceleration_status
            == "EXACT_SPARSE_BACKWARD_RUNTIME_PROVED"
            else (
                "5.4.7_DEVICE_OR_RELAXED_NUMERICAL_EQUIVALENCE_AUDIT"
            )
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
            "5.4.6"
        ),
        "status": (
            acceleration_status
        ),
        "selected_variant": (
            selected_variant
        ),
        "production_training_steps": (
            0
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
        VARIANT_AUDIT_PATH,
        TWO_STEP_PATH,
        PROJECTION_PATH,
        FINAL_INVARIANT_PATH,
        CONTRACT_PATH,
        MANIFEST_PATH,
    ):
        print(
            f"WROTE  {path}"
        )

    del batch0
    del batch1
    del stream
    del shared

    gc.collect()

    banner(
        "PHASE 5.4.6 FINAL STATUS"
    )

    print(
        f"Acceleration status:                 "
        f"{acceleration_status}"
    )

    if selected_variant is not None:
        print(
            f"Selected exact variant:              "
            f"{selected_variant}"
        )
        print(
            f"Selected exact mean / batch:         "
            f"{selected_mean_seconds:.3f} s"
        )
        print(
            f"Speedup:                             "
            f"{speedup:.2f}x"
        )
        print(
            "Projected one epoch:                "
            f"{human_duration(projected_epoch_seconds)}"
        )
        print(
            "Projected 20 epochs:                "
            f"{human_duration(projected_full_seconds)}"
        )
    else:
        print(
            "Selected exact variant:              NONE"
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
        "PHASE 5.4.6 COMPLETE / "
        "SPARSE EMBEDDING BACKWARD ACCELERATION AUDIT CLOSED"
    )


if __name__ == "__main__":
    main()