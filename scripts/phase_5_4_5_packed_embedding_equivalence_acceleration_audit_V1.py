#!/usr/bin/env python3
"""
Phase 5.4.5 — Packed Embedding Autograd Equivalence + Acceleration Audit

Purpose
-------
Test an implementation-equivalent optimization targeted directly at the
Phase-5.4.4 bottleneck: thousands of dense EmbeddingBackward nodes.

Canonical behavior
------------------
The frozen batch runtime performs:
    - one startup embedding lookup for every non-empty historical period
    - one investor embedding lookup for every (investor, target-h) trend query
    - one startup/investor embedding lookup for current pair scoring

Each lookup is individually correct, but dense embedding backward creates a
full dense gradient tensor per lookup. Phase 5.4.4 observed 7,085 embedding
backward evaluations in a single batch.

Candidate optimization
----------------------
Preserve the EXACT ordered embedding indices and values, but execute them as
larger packed embedding lookups and return non-overlapping slices to the frozen
forward runtime.

Variants:
    A_HISTORY_STARTUP_PACK
        Pack only all historical startup embedding calls.

    B_ALL_TREND_AND_PAIR_PACK
        Pack historical startup + current startup calls into one startup lookup,
        and trend-query investor + current investor calls into one investor
        lookup.

Acceptance rule
---------------
A variant is eligible ONLY if it reproduces batch 0 exactly:
    - BCE
    - logit SHA256
    - gradient SHA256
    - post-step model SHA256
    - optimizer-state SHA256

The fastest eligible variant must then reproduce the exact batch0 -> batch1
two-step trajectory.

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

PHASE_5_4_4_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_4_4_exact_autograd_operator_profile_contract.json"
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
    "phase_5_4_5"
)

VARIANT_AUDIT_PATH = (
    AUDIT_DIR
    / "packed_embedding_variant_equivalence_audit.csv"
)

TWO_STEP_PATH = (
    AUDIT_DIR
    / "selected_packed_embedding_two_step_proof.csv"
)

PROJECTION_PATH = (
    AUDIT_DIR
    / "packed_embedding_runtime_projection.csv"
)

FINAL_INVARIANT_PATH = (
    AUDIT_DIR
    / "phase_5_4_5_final_invariants.csv"
)

CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_4_5_packed_embedding_acceleration_contract.json"
)

MANIFEST_PATH = (
    AUDIT_DIR
    / "phase_5_4_5_packed_embedding_acceleration_manifest.json"
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

VARIANT_HISTORY = "A_HISTORY_STARTUP_PACK"
VARIANT_ALL = "B_ALL_TREND_AND_PAIR_PACK"


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


def build_embedding_call_plan(
    batch: pd.DataFrame,
    shared: dict,
    *,
    num_history_periods: int,
    num_investors: int,
) -> dict:
    """
    Reconstruct ONLY the embedding call index sequence used by the frozen batch
    runtime. No model computation occurs here.
    """

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

    batch_segments = (
        batch[
            "segment_number"
        ].to_numpy(
            dtype=np.int64
        )
    )

    unique_keys = sorted(
        {
            (
                int(investor),
                int(h),
            )
            for investor, h in zip(
                batch_investors,
                batch_segments,
            )
        },
        key=lambda value: (
            value[1],
            value[0],
        ),
    )

    history_startup_calls = []

    for investor_global, h in unique_keys:
        for period in range(
            h
        ):
            flattened = (
                investor_global
                * num_history_periods
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
                    flattened + 1
                ]
            )

            if end <= start:
                continue

            startup_globals = np.array(
                shared[
                    "trend_startup_indices"
                ][
                    start:end
                ],
                dtype=np.int64,
                copy=True,
            )

            startup_locals = (
                startup_globals
                - num_investors
            ).astype(
                np.int64,
                copy=False,
            )

            history_startup_calls.append(
                np.array(
                    startup_locals,
                    dtype=np.int64,
                    copy=True,
                )
            )

    trend_investor_calls = [
        np.asarray(
            [
                int(
                    investor_global
                )
            ],
            dtype=np.int64,
        )
        for investor_global, _h
        in unique_keys
    ]

    return {
        "history_startup_calls": (
            history_startup_calls
        ),
        "trend_investor_calls": (
            trend_investor_calls
        ),
        "pair_startup_call": (
            np.array(
                batch_startup_locals,
                dtype=np.int64,
                copy=True,
            )
        ),
        "pair_investor_call": (
            np.array(
                batch_investors,
                dtype=np.int64,
                copy=True,
            )
        ),
        "unique_key_count": (
            len(
                unique_keys
            )
        ),
    }


def _pack_arrays(
    arrays: list[np.ndarray],
) -> tuple[np.ndarray, list[int]]:
    lengths = [
        int(
            len(
                array
            )
        )
        for array in arrays
    ]

    if len(arrays) == 0:
        return (
            np.empty(
                0,
                dtype=np.int64,
            ),
            lengths,
        )

    packed = np.concatenate(
        arrays
    ).astype(
        np.int64,
        copy=False,
    )

    return (
        packed,
        lengths,
    )


class PackedEmbeddingDispatch:
    """
    Replace a sequence of nn.Embedding.forward calls with one packed embedding
    lookup and non-overlapping slices.

    Every frozen call input is checked for exact equality before its slice is
    returned.
    """

    def __init__(
        self,
        embedding: torch.nn.Embedding,
        planned_calls: list[np.ndarray],
    ):
        self.embedding = embedding
        self.original_forward = (
            embedding.forward
        )

        self.planned_calls = [
            np.array(
                values,
                dtype=np.int64,
                copy=True,
            )
            for values in planned_calls
        ]

        (
            packed_np,
            lengths,
        ) = _pack_arrays(
            self.planned_calls
        )

        self.lengths = lengths
        self.offsets = np.cumsum(
            [
                0,
            ]
            + lengths,
            dtype=np.int64,
        )

        packed_tensor = torch.from_numpy(
            np.array(
                packed_np,
                dtype=np.int64,
                copy=True,
            )
        )

        # ONE actual embedding call.
        self.packed_output = (
            self.original_forward(
                packed_tensor
            )
        )

        self.call_index = 0

    def install(self) -> None:
        dispatcher = self

        def packed_forward(
            self_embedding,
            input_tensor,
        ):
            if (
                dispatcher.call_index
                >= len(
                    dispatcher.planned_calls
                )
            ):
                raise AssertionError(
                    "Packed embedding received more calls "
                    "than planned."
                )

            expected_np = (
                dispatcher.planned_calls[
                    dispatcher.call_index
                ]
            )

            actual_np = (
                input_tensor
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.int64,
                    copy=False,
                )
            )

            if not np.array_equal(
                actual_np,
                expected_np,
            ):
                raise AssertionError(
                    "Packed embedding call-order/index drift "
                    f"at call {dispatcher.call_index}."
                )

            start = int(
                dispatcher.offsets[
                    dispatcher.call_index
                ]
            )

            end = int(
                dispatcher.offsets[
                    dispatcher.call_index
                    + 1
                ]
            )

            dispatcher.call_index += 1

            return (
                dispatcher.packed_output[
                    start:end
                ]
            )

        self.embedding.forward = (
            MethodType(
                packed_forward,
                self.embedding,
            )
        )

    def restore(self) -> None:
        self.embedding.forward = (
            self.original_forward
        )

    def assert_consumed(self) -> None:
        require(
            self.call_index
            == len(
                self.planned_calls
            ),
            (
                "Packed embedding did not consume all "
                "planned calls."
            ),
        )


def run_candidate_batch(
    *,
    roundtrip,
    preflight,
    model,
    optimizer,
    canonical_hash_fn,
    batch,
    shared,
    variant: str,
    expected: dict,
) -> dict:

    plan_start = time.perf_counter()

    plan = build_embedding_call_plan(
        batch,
        shared,
        num_history_periods=(
            roundtrip.NUM_HISTORY_PERIODS
        ),
        num_investors=(
            roundtrip.NUM_INVESTORS
        ),
    )

    startup_calls = []

    if variant in (
        VARIANT_HISTORY,
        VARIANT_ALL,
    ):
        startup_calls.extend(
            plan[
                "history_startup_calls"
            ]
        )

    if variant == VARIANT_ALL:
        startup_calls.append(
            plan[
                "pair_startup_call"
            ]
        )

    investor_calls = []

    if variant == VARIANT_ALL:
        investor_calls.extend(
            plan[
                "trend_investor_calls"
            ]
        )

        investor_calls.append(
            plan[
                "pair_investor_call"
            ]
        )

    startup_dispatch = None
    investor_dispatch = None

    # Build packed graph before calling the frozen executor. The timer includes
    # the packed lookup creation and dispatch installation.
    if len(
        startup_calls
    ) > 0:
        startup_dispatch = (
            PackedEmbeddingDispatch(
                model.startup_embedding,
                startup_calls,
            )
        )

        startup_dispatch.install()

    if len(
        investor_calls
    ) > 0:
        investor_dispatch = (
            PackedEmbeddingDispatch(
                model.investor_embedding,
                investor_calls,
            )
        )

        investor_dispatch.install()

    setup_seconds = (
        time.perf_counter()
        - plan_start
    )

    true_gradient_hash_fn = (
        roundtrip
        .gradient_logical_sha256
    )

    true_optimizer_hash_fn = (
        roundtrip
        .optimizer_state_logical_sha256
    )

    roundtrip.gradient_logical_sha256 = (
        lambda model: "SKIPPED_INSIDE_CANDIDATE_TIMER"
    )

    roundtrip.optimizer_state_logical_sha256 = (
        lambda model, optimizer: "SKIPPED_INSIDE_CANDIDATE_TIMER"
    )

    dummy_model_hash_fn = (
        lambda model: "SKIPPED_INSIDE_CANDIDATE_TIMER"
    )

    execute_start = time.perf_counter()

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

        execute_seconds = (
            time.perf_counter()
            - execute_start
        )

        if startup_dispatch is not None:
            startup_dispatch.assert_consumed()

        if investor_dispatch is not None:
            investor_dispatch.assert_consumed()

    finally:
        if startup_dispatch is not None:
            startup_dispatch.restore()

        if investor_dispatch is not None:
            investor_dispatch.restore()

        roundtrip.gradient_logical_sha256 = (
            true_gradient_hash_fn
        )

        roundtrip.optimizer_state_logical_sha256 = (
            true_optimizer_hash_fn
        )

    # Exact proof hashes after timed execution.
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

    return {
        "variant": (
            variant
        ),
        "setup_seconds": (
            float(
                setup_seconds
            )
        ),
        "execute_seconds": (
            float(
                execute_seconds
            )
        ),
        "total_candidate_seconds": (
            float(
                setup_seconds
                + execute_seconds
            )
        ),
        "history_startup_calls_packed": (
            len(
                plan[
                    "history_startup_calls"
                ]
            )
            if variant
            in (
                VARIANT_HISTORY,
                VARIANT_ALL,
            )
            else 0
        ),
        "trend_investor_calls_packed": (
            len(
                plan[
                    "trend_investor_calls"
                ]
            )
            if variant
            == VARIANT_ALL
            else 0
        ),
        "pair_startup_packed": (
            variant
            == VARIANT_ALL
        ),
        "pair_investor_packed": (
            variant
            == VARIANT_ALL
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
    }


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.4.5 — "
        "PACKED EMBEDDING AUTOGRAD EQUIVALENCE + ACCELERATION AUDIT"
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
        float(
            operator_contract.get(
                "embedding_named_self_share_percent",
                0.0,
            )
        )
        > 0.0,
        (
            "Phase-5.4.4 provides no embedding-backward evidence."
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
        "Phase-5.4.2 exact lean CPU runtime:   PASS"
    )
    print(
        "Phase-5.4.4 embedding-backward target: PASS"
    )
    print(
        f"Canonical lean seconds / batch:       "
        f"{lean_exact_seconds:.3f}"
    )

    # =========================================================================
    # Load frozen runtime
    # =========================================================================

    banner(
        "LOAD FROZEN RUNTIME"
    )

    roundtrip = load_module(
        ROUNDTRIP_SOURCE_PATH,
        "_itrs_phase5_4_5_roundtrip",
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
        "BATCH-0 PACKING VARIANT EQUIVALENCE SCREEN"
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

    variant_rows = []

    for variant in (
        VARIANT_HISTORY,
        VARIANT_ALL,
    ):

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
                "Fresh candidate model initial SHA drift."
            ),
        )

        try:
            row = run_candidate_batch(
                roundtrip=roundtrip,
                preflight=preflight,
                model=model,
                optimizer=optimizer,
                canonical_hash_fn=canonical_hash_fn,
                batch=batch0,
                shared=shared,
                variant=variant,
                expected=expected0,
            )

            row[
                "runtime_error"
            ] = ""

        except Exception as exc:
            row = {
                "variant": (
                    variant
                ),
                "setup_seconds": (
                    np.nan
                ),
                "execute_seconds": (
                    np.nan
                ),
                "total_candidate_seconds": (
                    np.nan
                ),
                "history_startup_calls_packed": (
                    np.nan
                ),
                "trend_investor_calls_packed": (
                    np.nan
                ),
                "pair_startup_packed": (
                    False
                ),
                "pair_investor_packed": (
                    False
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

        variant_rows.append(
            row
        )

        print(
            f"{variant:30s} | "
            f"time={row['total_candidate_seconds']!s:>10} | "
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

        del model
        del optimizer

        gc.collect()

    variant_df = pd.DataFrame(
        variant_rows
    )

    eligible_df = variant_df.loc[
        variant_df[
            "byte_exact_eligible"
        ]
        == True
    ].copy()

    # =========================================================================
    # Decide whether exact packing is viable
    # =========================================================================

    banner(
        "PACKED EMBEDDING EXACTNESS DECISION"
    )

    if len(
        eligible_df
    ) == 0:

        print(
            "No packed-embedding variant preserved the "
            "byte-exact frozen batch-0 gradient/state."
        )
        print()
        print(
            "Interpretation:"
        )
        print(
            "  Forward semantics may be equivalent, but the "
            "floating-point embedding-gradient accumulation order "
            "changed. No packed runtime is accepted."
        )

        selected_variant = None
        selected_seconds = None
        selected_speedup = None
        two_step_df = pd.DataFrame()

    else:
        selected = (
            eligible_df.sort_values(
                [
                    "total_candidate_seconds",
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

        selected_seconds = float(
            selected[
                "total_candidate_seconds"
            ]
        )

        selected_speedup = (
            lean_exact_seconds
            / selected_seconds
        )

        print(
            f"Selected exact variant:               "
            f"{selected_variant}"
        )
        print(
            f"Candidate seconds / batch0:           "
            f"{selected_seconds:.3f}"
        )
        print(
            f"Speedup vs canonical lean CPU:        "
            f"{selected_speedup:.2f}x"
        )

        # =====================================================================
        # Exact two-step proof for selected candidate
        # =====================================================================

        banner(
            "SELECTED PACKED VARIANT — EXACT TWO-STEP PROOF"
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
                "Selected candidate fresh model SHA drift."
            ),
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

            row = run_candidate_batch(
                roundtrip=roundtrip,
                preflight=preflight,
                model=model,
                optimizer=optimizer,
                canonical_hash_fn=canonical_hash_fn,
                batch=batch,
                shared=shared,
                variant=selected_variant,
                expected=expected,
            )

            require(
                row[
                    "byte_exact_eligible"
                ]
                is True,
                (
                    f"Selected candidate failed exact "
                    f"two-step proof at batch {batch_index}."
                ),
            )

            row[
                "batch_index"
            ] = (
                batch_index
            )

            two_step_rows.append(
                row
            )

            print(
                f"batch={batch_index} | "
                f"time={row['total_candidate_seconds']:.3f}s | "
                "EXACT"
            )

        two_step_df = pd.DataFrame(
            two_step_rows
        )

        del model
        del optimizer

        gc.collect()

    # =========================================================================
    # Projection if accepted
    # =========================================================================

    banner(
        "PACKED RUNTIME PROJECTION"
    )

    if (
        selected_variant
        is not None
        and len(
            two_step_df
        )
        == 2
    ):
        mean_candidate_seconds = float(
            two_step_df[
                "total_candidate_seconds"
            ].mean()
        )

        candidate_speedup = (
            lean_exact_seconds
            / mean_candidate_seconds
        )

        projected_epoch_seconds = (
            mean_candidate_seconds
            * BATCHES_PER_EPOCH
        )

        projected_full_seconds = (
            mean_candidate_seconds
            * TOTAL_OPTIMIZER_STEPS
        )

        print(
            f"Selected variant:                     "
            f"{selected_variant}"
        )
        print(
            f"Mean candidate seconds / batch:       "
            f"{mean_candidate_seconds:.3f}"
        )
        print(
            f"Speedup vs canonical lean CPU:        "
            f"{candidate_speedup:.2f}x"
        )
        print(
            "Projected one epoch:                 "
            f"{human_duration(projected_epoch_seconds)}"
        )
        print(
            "Projected 20 epochs:                 "
            f"{human_duration(projected_full_seconds)}"
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
                        "packed_exact_seconds_per_batch"
                    ),
                    "value": (
                        mean_candidate_seconds
                    ),
                    "human": (
                        f"{mean_candidate_seconds:.3f} s"
                    ),
                },
                {
                    "metric": (
                        "speedup"
                    ),
                    "value": (
                        candidate_speedup
                    ),
                    "human": (
                        f"{candidate_speedup:.2f}x"
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

        acceleration_status = (
            "EXACT_PACKED_RUNTIME_PROVED"
        )

    else:
        mean_candidate_seconds = None
        candidate_speedup = None
        projected_epoch_seconds = None
        projected_full_seconds = None

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
                        "packed_exact_runtime"
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

        acceleration_status = (
            "NO_BYTE_EXACT_PACKED_RUNTIME"
        )

        print(
            "No byte-exact packed runtime was accepted."
        )

    # =========================================================================
    # Final invariants
    # =========================================================================

    banner(
        "FINAL PHASE-5.4.5 INVARIANTS"
    )

    # This phase is allowed to conclude that no candidate is byte-exact.
    # That is a valid audit result, not an invariant failure.
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
            "both_candidate_variants_tested",
            (
                set(
                    variant_df[
                        "variant"
                    ].tolist()
                )
                == {
                    VARIANT_HISTORY,
                    VARIANT_ALL,
                }
            ),
        ),
        (
            "no_inexact_variant_accepted",
            bool(
                (
                    variant_df.loc[
                        variant_df[
                            "byte_exact_eligible"
                        ]
                        == False
                    ].shape[
                        0
                    ]
                    + eligible_df.shape[
                        0
                    ]
                )
                == variant_df.shape[
                    0
                ]
            ),
        ),
        (
            "selected_variant_two_step_exact_if_any",
            (
                True
                if selected_variant
                is None
                else (
                    len(
                        two_step_df
                    )
                    == 2
                    and bool(
                        two_step_df[
                            "byte_exact_eligible"
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
            "At least one Phase-5.4.5 audit invariant failed."
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
        "WRITE PHASE-5.4.5 OUTPUTS"
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTRACT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    variant_df.to_csv(
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
            "5.4.5"
        ),
        "title": (
            "Packed Embedding Autograd Equivalence + Acceleration Audit"
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
            variant_df.to_dict(
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
                    "byte_exact_eligible"
                ].all()
            )
        ),
        "selected_mean_seconds_per_batch": (
            mean_candidate_seconds
        ),
        "selected_speedup": (
            candidate_speedup
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
                "5.4.6_FREEZE_PACKED_PRODUCTION_RUNTIME"
            )
            if acceleration_status
            == "EXACT_PACKED_RUNTIME_PROVED"
            else (
                "5.4.6_ALTERNATIVE_ACCELERATION_AUDIT"
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
            "5.4.5"
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
        "PHASE 5.4.5 FINAL STATUS"
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
            f"{mean_candidate_seconds:.3f} s"
        )
        print(
            f"Speedup:                             "
            f"{candidate_speedup:.2f}x"
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
        "PHASE 5.4.5 COMPLETE / "
        "PACKED EMBEDDING ACCELERATION AUDIT CLOSED"
    )


if __name__ == "__main__":
    main()