#!/usr/bin/env python3
"""
Phase 5.4.3 — Exact Canonical Batch Runtime Bottleneck Profile

Purpose
-------
Profile the already-frozen exact CPU batch path at a HIGH LEVEL using the
two numerically anchored epoch-0 batches.

This phase changes NO scientific semantics.

It measures:
    - full structural propagation
    - description encoder
    - trend attention calls
    - trend GRU sequence encoding
    - scoring MLP
    - autograd backward
    - Adam optimizer.step()
    - unattributed Python/data/check overhead

The profiler is implemented with lightweight wall-clock wrappers around the
existing frozen methods. It does NOT rewrite model math.

Exactness rule
--------------
Both profiled batches must still reproduce the frozen:
    - BCE
    - logit SHA256
    - gradient SHA256
    - post-step model SHA256
    - optimizer-state SHA256

No production checkpoint is written.
No validation is scored.
No test is accessed.
No model state is retained.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import math
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import MethodType

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
    "phase_5_4_3"
)

BATCH_COMPONENT_PATH = (
    AUDIT_DIR
    / "exact_batch_component_wall_time.csv"
)

AGGREGATE_PATH = (
    AUDIT_DIR
    / "exact_runtime_component_aggregate.csv"
)

BOTTLENECK_PATH = (
    AUDIT_DIR
    / "exact_runtime_bottleneck_ranking.csv"
)

FINAL_INVARIANT_PATH = (
    AUDIT_DIR
    / "phase_5_4_3_final_invariants.csv"
)

CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_4_3_exact_runtime_bottleneck_profile_contract.json"
)

MANIFEST_PATH = (
    AUDIT_DIR
    / "phase_5_4_3_exact_runtime_bottleneck_profile_manifest.json"
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

EXPECTED_SELECTED_THREADS = 8


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


class TimerAccumulator:
    def __init__(self):
        self.total = defaultdict(float)
        self.calls = defaultdict(int)

    def add(
        self,
        name: str,
        seconds: float,
    ) -> None:
        self.total[
            name
        ] += float(seconds)

        self.calls[
            name
        ] += 1

    def reset(self) -> None:
        self.total.clear()
        self.calls.clear()


def wrap_bound_method(
    obj,
    method_name: str,
    component_name: str,
    timer: TimerAccumulator,
):
    original = getattr(
        obj,
        method_name,
    )

    def wrapped(
        self,
        *args,
        **kwargs,
    ):
        start = time.perf_counter()

        try:
            return original(
                *args,
                **kwargs,
            )
        finally:
            timer.add(
                component_name,
                time.perf_counter()
                - start,
            )

    setattr(
        obj,
        method_name,
        MethodType(
            wrapped,
            obj,
        ),
    )

    return original


def build_component_rows(
    *,
    batch_index: int,
    total_seconds: float,
    timer: TimerAccumulator,
) -> list[dict]:
    structural = timer.total.get(
        "structural_forward",
        0.0,
    )

    description = timer.total.get(
        "description_forward",
        0.0,
    )

    trend_attention = timer.total.get(
        "trend_attention",
        0.0,
    )

    trend_gru = timer.total.get(
        "trend_gru",
        0.0,
    )

    scoring = timer.total.get(
        "scoring_forward",
        0.0,
    )

    backward = timer.total.get(
        "backward",
        0.0,
    )

    adam = timer.total.get(
        "adam_step",
        0.0,
    )

    named_sum = (
        structural
        + description
        + trend_attention
        + trend_gru
        + scoring
        + backward
        + adam
    )

    unattributed = max(
        0.0,
        total_seconds
        - named_sum,
    )

    items = [
        (
            "structural_forward",
            structural,
            timer.calls.get(
                "structural_forward",
                0,
            ),
        ),
        (
            "description_forward",
            description,
            timer.calls.get(
                "description_forward",
                0,
            ),
        ),
        (
            "trend_attention",
            trend_attention,
            timer.calls.get(
                "trend_attention",
                0,
            ),
        ),
        (
            "trend_gru",
            trend_gru,
            timer.calls.get(
                "trend_gru",
                0,
            ),
        ),
        (
            "scoring_forward",
            scoring,
            timer.calls.get(
                "scoring_forward",
                0,
            ),
        ),
        (
            "backward",
            backward,
            timer.calls.get(
                "backward",
                0,
            ),
        ),
        (
            "adam_step",
            adam,
            timer.calls.get(
                "adam_step",
                0,
            ),
        ),
        (
            "unattributed_preprocessing_checks",
            unattributed,
            1,
        ),
    ]

    return [
        {
            "batch_index": (
                batch_index
            ),
            "component": (
                name
            ),
            "calls": (
                int(calls)
            ),
            "wall_seconds": (
                float(seconds)
            ),
            "share_of_batch": (
                (
                    float(seconds)
                    / float(total_seconds)
                )
                if total_seconds > 0
                else 0.0
            ),
            "batch_total_seconds": (
                float(total_seconds)
            ),
        }
        for (
            name,
            seconds,
            calls,
        ) in items
    ]


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.4.3 — "
        "EXACT CANONICAL BATCH RUNTIME BOTTLENECK PROFILE"
    )

    print(
        "Production training launched:         NO"
    )
    print(
        "Profiled training batches:            2"
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
    # Prerequisites
    # =========================================================================

    banner(
        "PREREQUISITE GATE"
    )

    for path in (
        ROUNDTRIP_SOURCE_PATH,
        PHASE_5_4_2_CONTRACT_PATH,
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

    authorization = load_json(
        PHASE_5_4_AUTHORIZATION_PATH
    )

    require(
        runtime_contract.get(
            "status"
        )
        == "COMPLETE",
        (
            "Phase-5.4.2 runtime contract "
            "is not COMPLETE."
        ),
    )

    require(
        runtime_contract.get(
            "selected_two_step_exact"
        )
        is True,
        (
            "Phase-5.4.2 selected CPU runtime "
            "is not exact."
        ),
    )

    selected_threads = int(
        runtime_contract[
            "selected_threads"
        ]
    )

    require(
        selected_threads
        == EXPECTED_SELECTED_THREADS,
        (
            "Selected exact CPU thread count "
            "drift from Phase-5.4.2."
        ),
    )

    require(
        authorization.get(
            "training_allowed"
        )
        is True,
        (
            "Phase-5.4 production training "
            "is not authorized."
        ),
    )

    print(
        "Phase-5.4.2 exact runtime:            PASS"
    )
    print(
        f"Selected exact CPU threads:           "
        f"{selected_threads}"
    )

    # =========================================================================
    # Load runtime
    # =========================================================================

    banner(
        "LOAD FROZEN RUNTIME"
    )

    roundtrip = load_module(
        ROUNDTRIP_SOURCE_PATH,
        "_itrs_phase5_4_3_roundtrip",
    )

    require(
        torch.__version__
        == EXPECTED_PYTORCH,
        (
            "Reference PyTorch version drift."
        ),
    )

    torch.set_num_threads(
        selected_threads
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
            "Canonical initial model SHA drift."
        ),
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
    print(
        "Canonical initial model:              PASS"
    )

    # =========================================================================
    # Install lightweight high-level timers
    # =========================================================================

    timer = TimerAccumulator()

    wrap_bound_method(
        model.preference_propagation,
        "forward",
        "structural_forward",
        timer,
    )

    wrap_bound_method(
        model.description_encoder,
        "forward",
        "description_forward",
        timer,
    )

    wrap_bound_method(
        model.trend_extractor,
        "attend_period",
        "trend_attention",
        timer,
    )

    wrap_bound_method(
        model.trend_extractor,
        "encode_training_sequence",
        "trend_gru",
        timer,
    )

    wrap_bound_method(
        model.scoring_mlp,
        "forward",
        "scoring_forward",
        timer,
    )

    original_optimizer_step = (
        optimizer.step
    )

    def timed_optimizer_step(
        *args,
        **kwargs,
    ):
        start = time.perf_counter()

        try:
            return original_optimizer_step(
                *args,
                **kwargs,
            )
        finally:
            timer.add(
                "adam_step",
                time.perf_counter()
                - start,
            )

    optimizer.step = (
        timed_optimizer_step
    )

    original_autograd_backward = (
        torch.autograd.backward
    )

    def timed_autograd_backward(
        *args,
        **kwargs,
    ):
        start = time.perf_counter()

        try:
            return original_autograd_backward(
                *args,
                **kwargs,
            )
        finally:
            timer.add(
                "backward",
                time.perf_counter()
                - start,
            )

    torch.autograd.backward = (
        timed_autograd_backward
    )

    # Keep expensive proof hashes OUTSIDE the timed executor.
    true_gradient_hash_fn = (
        roundtrip
        .gradient_logical_sha256
    )

    true_optimizer_hash_fn = (
        roundtrip
        .optimizer_state_logical_sha256
    )

    roundtrip.gradient_logical_sha256 = (
        lambda model: "SKIPPED_INSIDE_PROFILE_TIMER"
    )

    roundtrip.optimizer_state_logical_sha256 = (
        lambda model, optimizer: "SKIPPED_INSIDE_PROFILE_TIMER"
    )

    dummy_model_hash_fn = (
        lambda model: "SKIPPED_INSIDE_PROFILE_TIMER"
    )

    # =========================================================================
    # Profile exact batch0 and batch1
    # =========================================================================

    banner(
        "PROFILE EXACT TWO-STEP CANONICAL TRAJECTORY"
    )

    all_component_rows = []
    batch_results = []

    batch_specs = [
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
    ]

    for (
        batch_index,
        batch,
        expected_loss,
        expected_logit_sha,
        expected_gradient_sha,
        expected_model_sha,
        expected_optimizer_sha,
    ) in batch_specs:

        timer.reset()

        batch_start = (
            time.perf_counter()
        )

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

        batch_seconds = (
            time.perf_counter()
            - batch_start
        )

        # Recompute exact numerical fingerprints OUTSIDE profiling interval.
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
                expected_loss,
            )
            and result[
                "logit_sha256"
            ]
            == expected_logit_sha
            and gradient_sha
            == expected_gradient_sha
            and model_sha
            == expected_model_sha
            and optimizer_sha
            == expected_optimizer_sha
        )

        require(
            exact,
            (
                f"Profiled batch {batch_index} "
                "did not preserve exact frozen numerics."
            ),
        )

        component_rows = build_component_rows(
            batch_index=batch_index,
            total_seconds=batch_seconds,
            timer=timer,
        )

        all_component_rows.extend(
            component_rows
        )

        batch_results.append(
            {
                "batch_index": (
                    batch_index
                ),
                "total_seconds": (
                    float(batch_seconds)
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

        print()
        print(
            f"Batch {batch_index} total:             "
            f"{batch_seconds:.3f} s"
        )

        print(
            pd.DataFrame(
                component_rows
            )[
                [
                    "component",
                    "calls",
                    "wall_seconds",
                    "share_of_batch",
                ]
            ]
            .assign(
                share_of_batch=lambda frame: (
                    100.0
                    * frame[
                        "share_of_batch"
                    ]
                )
            )
            .rename(
                columns={
                    "share_of_batch": (
                        "share_percent"
                    )
                }
            )
            .sort_values(
                "wall_seconds",
                ascending=False,
            )
            .to_string(
                index=False,
                formatters={
                    "wall_seconds": (
                        lambda value: (
                            f"{value:.3f}"
                        )
                    ),
                    "share_percent": (
                        lambda value: (
                            f"{value:.2f}"
                        )
                    ),
                },
            )
        )

    # Restore patched functions.
    torch.autograd.backward = (
        original_autograd_backward
    )

    roundtrip.gradient_logical_sha256 = (
        true_gradient_hash_fn
    )

    roundtrip.optimizer_state_logical_sha256 = (
        true_optimizer_hash_fn
    )

    # =========================================================================
    # Aggregate bottleneck ranking
    # =========================================================================

    banner(
        "AGGREGATE BOTTLENECK RANKING"
    )

    component_df = pd.DataFrame(
        all_component_rows
    )

    aggregate_df = (
        component_df.groupby(
            "component",
            sort=False,
        )
        .agg(
            mean_wall_seconds=(
                "wall_seconds",
                "mean",
            ),
            median_wall_seconds=(
                "wall_seconds",
                "median",
            ),
            total_calls=(
                "calls",
                "sum",
            ),
            mean_share_of_batch=(
                "share_of_batch",
                "mean",
            ),
        )
        .reset_index()
    )

    aggregate_df[
        "mean_share_percent"
    ] = (
        100.0
        * aggregate_df[
            "mean_share_of_batch"
        ]
    )

    bottleneck_df = (
        aggregate_df.sort_values(
            [
                "mean_wall_seconds",
                "component",
            ],
            ascending=[
                False,
                True,
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    bottleneck_df[
        "rank"
    ] = np.arange(
        1,
        len(
            bottleneck_df
        )
        + 1,
        dtype=np.int64,
    )

    bottleneck_df = bottleneck_df[
        [
            "rank",
            "component",
            "mean_wall_seconds",
            "median_wall_seconds",
            "total_calls",
            "mean_share_percent",
        ]
    ]

    print(
        bottleneck_df.to_string(
            index=False,
            formatters={
                "mean_wall_seconds": (
                    lambda value: (
                        f"{value:.3f}"
                    )
                ),
                "median_wall_seconds": (
                    lambda value: (
                        f"{value:.3f}"
                    )
                ),
                "mean_share_percent": (
                    lambda value: (
                        f"{value:.2f}"
                    )
                ),
            },
        )
    )

    dominant_component = str(
        bottleneck_df.iloc[
            0
        ][
            "component"
        ]
    )

    dominant_seconds = float(
        bottleneck_df.iloc[
            0
        ][
            "mean_wall_seconds"
        ]
    )

    dominant_share = float(
        bottleneck_df.iloc[
            0
        ][
            "mean_share_percent"
        ]
    )

    print()
    print(
        f"Dominant component:                   "
        f"{dominant_component}"
    )
    print(
        f"Dominant mean seconds:                "
        f"{dominant_seconds:.3f}"
    )
    print(
        f"Dominant mean share:                  "
        f"{dominant_share:.2f}%"
    )

    # =========================================================================
    # Final invariants
    # =========================================================================

    banner(
        "FINAL PHASE-5.4.3 INVARIANTS"
    )

    batch_result_df = pd.DataFrame(
        batch_results
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
            "batch0_exact",
            bool(
                batch_result_df.loc[
                    batch_result_df[
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
            "batch1_exact",
            bool(
                batch_result_df.loc[
                    batch_result_df[
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
            "component_profile_nonempty",
            (
                len(
                    component_df
                )
                > 0
            ),
        ),
        (
            "dominant_component_identified",
            (
                len(
                    dominant_component
                )
                > 0
            ),
        ),
        (
            "dominant_seconds_positive",
            (
                math.isfinite(
                    dominant_seconds
                )
                and dominant_seconds > 0
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
            "At least one Phase-5.4.3 invariant failed."
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
        "WRITE PHASE-5.4.3 OUTPUTS"
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTRACT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    component_df.to_csv(
        BATCH_COMPONENT_PATH,
        index=False,
    )

    aggregate_df.to_csv(
        AGGREGATE_PATH,
        index=False,
    )

    bottleneck_df.to_csv(
        BOTTLENECK_PATH,
        index=False,
    )

    invariant_df.to_csv(
        FINAL_INVARIANT_PATH,
        index=False,
    )

    contract = {
        "phase": (
            "5.4.3"
        ),
        "title": (
            "Exact Canonical Batch Runtime Bottleneck Profile"
        ),
        "status": (
            "COMPLETE"
        ),
        "classification": (
            "RUNTIME_PROFILING_ONLY"
        ),
        "selected_exact_cpu_threads": (
            selected_threads
        ),
        "profiled_batches": (
            2
        ),
        "exact_two_step_trajectory_preserved": (
            True
        ),
        "dominant_component": (
            dominant_component
        ),
        "dominant_mean_seconds": (
            dominant_seconds
        ),
        "dominant_mean_share_percent": (
            dominant_share
        ),
        "component_ranking": (
            bottleneck_df.to_dict(
                orient="records"
            )
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
            "5.4.4_IMPLEMENTATION_EQUIVALENT_ACCELERATION_"
            "TARGETING_DOMINANT_COMPONENT"
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
            "5.4.3"
        ),
        "status": (
            "EXACT_RUNTIME_BOTTLENECK_PROFILE_COMPLETE"
        ),
        "dominant_component": (
            dominant_component
        ),
        "dominant_mean_share_percent": (
            dominant_share
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
        BATCH_COMPONENT_PATH,
        AGGREGATE_PATH,
        BOTTLENECK_PATH,
        FINAL_INVARIANT_PATH,
        CONTRACT_PATH,
        MANIFEST_PATH,
    ):
        print(
            f"WROTE  {path}"
        )

    # =========================================================================
    # Cleanup
    # =========================================================================

    del model
    del optimizer
    del batch0
    del batch1
    del stream
    del shared

    gc.collect()

    banner(
        "PHASE 5.4.3 FINAL STATUS"
    )

    print(
        f"Dominant component:                   "
        f"{dominant_component}"
    )
    print(
        f"Dominant mean wall time:              "
        f"{dominant_seconds:.3f} s"
    )
    print(
        f"Dominant mean share:                  "
        f"{dominant_share:.2f}%"
    )
    print()
    print(
        "Exact numerical trajectory:           PRESERVED"
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

    banner(
        "PHASE 5.4.3 COMPLETE / "
        "EXACT CANONICAL RUNTIME BOTTLENECK IDENTIFIED"
    )


if __name__ == "__main__":
    main()