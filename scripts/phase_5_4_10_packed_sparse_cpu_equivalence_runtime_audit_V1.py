#!/usr/bin/env python3
"""
Phase 5.4.10 — Packed + Sparse CPU Numerical Equivalence and Runtime Audit

Purpose
-------
Test the evidence-driven combined CPU acceleration:

    PACKED_ALL_TREND_AND_PAIR + D_BOTH_SPARSE

Evidence:
    - Phase 5.4.5: packing reduced batch-0 runtime substantially but changed
      floating-point gradient accumulation order.
    - Phase 5.4.6/5.4.7b: D_BOTH_SPARSE is numerically equivalent under the
      frozen policy.
    - Phase 5.4.9: D_BOTH_SPARSE still executes 7,085 EmbeddingBackward nodes,
      and fragmented add/cat/fill/add_ operations dominate residual self CPU.

Candidate:
    - startup_embedding.sparse = True
    - investor_embedding.sparse = True
    - all historical + pair startup embedding calls packed into ONE lookup
    - all trend-query + pair investor embedding calls packed into ONE lookup
    - returned slices preserve the canonical logical call sequence

Acceptance:
    - canonical CPU reference must remain byte-exact for batch 0 and batch 1
    - candidate must pass frozen policy ITRS_PHASE5_NUMERICAL_EQUIVALENCE_V1
      for BOTH stateful batches
    - no tolerance is changed here

No production runtime is selected here.
No production checkpoint is written.
No validation/test is accessed.
"""

from __future__ import annotations

import gc
import hashlib
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
    "scripts/phase_5_3_2b_checkpoint_resume_roundtrip_proof.py"
)

POLICY_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_4_numerical_equivalence_policy.json"
)

PHASE_5_4_7B_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_4_7b_numerical_equivalence_policy_contract.json"
)

PHASE_5_4_9_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_4_9_sparse_cpu_residual_bottleneck_contract.json"
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
    "data/experimental/phase_5/audits/phase_5_4_10"
)

BATCH_SUMMARY_PATH = (
    AUDIT_DIR
    / "packed_sparse_cpu_batch_summary.csv"
)

CALL_PLAN_PATH = (
    AUDIT_DIR
    / "packed_sparse_cpu_call_plan_summary.csv"
)

POLICY_AUDIT_PATH = (
    AUDIT_DIR
    / "packed_sparse_cpu_numerical_policy_audit.csv"
)

RUNTIME_PATH = (
    AUDIT_DIR
    / "packed_sparse_cpu_runtime_projection.csv"
)

FINAL_INVARIANT_PATH = (
    AUDIT_DIR
    / "phase_5_4_10_final_invariants.csv"
)

CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_4_10_packed_sparse_cpu_acceleration_contract.json"
)

MANIFEST_PATH = (
    AUDIT_DIR
    / "phase_5_4_10_packed_sparse_cpu_acceleration_manifest.json"
)


# =============================================================================
# Frozen anchors
# =============================================================================

EXPECTED_PYTORCH = "2.7.0"
EXPECTED_THREADS = 8
EXPECTED_POLICY_VERSION = "ITRS_PHASE5_NUMERICAL_EQUIVALENCE_V1"

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
EXPECTED_BATCH1_LOGIT_SHA256 = (
    "cfbc4106103abf9478b8f04f0e0d909b"
    "ed37659e5ee7e29257bce0a7dd4beb26"
)

EXPECTED_BATCH0_GRADIENT_SHA256 = (
    "8c542430813d8ca91b8397409954ea92"
    "295a2b55bcc420661783fb865010845d"
)
EXPECTED_BATCH1_GRADIENT_SHA256 = (
    "8c066fd5f8002e1edd0a282f4ac549a3"
    "903f590716b38ca060b0f01088594f22"
)

EXPECTED_BATCH0_POST_MODEL_SHA256 = (
    "42a521f11d8f24e4144d0215d6e1b34d"
    "5f8bf0c2d8848624e4f7c3130699035d"
)
EXPECTED_BATCH1_POST_MODEL_SHA256 = (
    "c41702cda99092a7fb63bb0a8227e658"
    "851b3ac4cbc373d90cdd6816eccdd196"
)

EXPECTED_BATCH0_OPTIMIZER_SHA256 = (
    "5ce2683c21f456b9d5d15eb876b049c5"
    "e6db1215db5a026630f093f7f9d49891"
)
EXPECTED_BATCH1_OPTIMIZER_SHA256 = (
    "569a6691424ac32d0f252728750281cffd"
    "175a2b6b6c6ea1913f5f497200b00d"
)

BATCHES_PER_EPOCH = 10_481
TOTAL_OPTIMIZER_STEPS = 209_620
CANDIDATE_RUNTIME = "PACKED_ALL_PLUS_D_BOTH_SPARSE_CPU"


# =============================================================================
# Generic helpers
# =============================================================================

def banner(text: str) -> None:
    print("\n" + "=" * 118)
    print(text)
    print("=" * 118)


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_module(path: Path, module_name: str):
    require(path.exists(), f"Missing source: {path}")

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    require(
        spec is not None and spec.loader is not None,
        f"Could not import {path}.",
    )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


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


# =============================================================================
# Packed embedding call plan
# =============================================================================

def build_embedding_call_plan(
    batch: pd.DataFrame,
    shared: dict,
    *,
    num_history_periods: int,
    num_investors: int,
) -> dict:

    batch_investors = batch[
        "investor_global"
    ].to_numpy(
        dtype=np.int64
    )

    batch_startup_locals = batch[
        "startup_local"
    ].to_numpy(
        dtype=np.int64
    )

    batch_segments = batch[
        "segment_number"
    ].to_numpy(
        dtype=np.int64
    )

    unique_keys = sorted(
        {
            (
                int(investor),
                int(h),
            )
            for investor, h
            in zip(
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
        for period in range(h):
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
                int(investor_global)
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
        "pair_startup_call": np.array(
            batch_startup_locals,
            dtype=np.int64,
            copy=True,
        ),
        "pair_investor_call": np.array(
            batch_investors,
            dtype=np.int64,
            copy=True,
        ),
        "unique_key_count": len(
            unique_keys
        ),
    }


def pack_arrays(
    arrays: list[np.ndarray],
) -> tuple[np.ndarray, list[int]]:

    lengths = [
        int(len(array))
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

    return (
        np.concatenate(
            arrays
        ).astype(
            np.int64,
            copy=False,
        ),
        lengths,
    )


class PackedEmbeddingDispatch:
    """
    Execute ONE actual nn.Embedding call, then return non-overlapping slices
    when the canonical runtime asks for its original sequence of lookups.
    """

    def __init__(
        self,
        embedding: torch.nn.Embedding,
        planned_calls: list[np.ndarray],
    ):
        self.embedding = embedding
        self.original_forward = embedding.forward

        self.planned_calls = [
            np.array(
                values,
                dtype=np.int64,
                copy=True,
            )
            for values in planned_calls
        ]

        packed_np, lengths = pack_arrays(
            self.planned_calls
        )

        self.offsets = np.cumsum(
            [0] + lengths,
            dtype=np.int64,
        )

        packed_tensor = torch.from_numpy(
            np.array(
                packed_np,
                dtype=np.int64,
                copy=True,
            )
        )

        # sparse=True on the module is honored here.
        self.packed_output = self.original_forward(
            packed_tensor
        )

        self.call_index = 0

    def install(self) -> None:
        dispatcher = self

        def packed_forward(
            self_embedding,
            input_tensor,
        ):
            require(
                dispatcher.call_index
                < len(
                    dispatcher.planned_calls
                ),
                (
                    "Packed embedding received more calls "
                    "than planned."
                ),
            )

            expected_np = dispatcher.planned_calls[
                dispatcher.call_index
            ]

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

            require(
                np.array_equal(
                    actual_np,
                    expected_np,
                ),
                (
                    "Packed embedding call order/index drift "
                    f"at call {dispatcher.call_index}."
                ),
            )

            start = int(
                dispatcher.offsets[
                    dispatcher.call_index
                ]
            )

            end = int(
                dispatcher.offsets[
                    dispatcher.call_index + 1
                ]
            )

            dispatcher.call_index += 1

            return dispatcher.packed_output[
                start:end
            ]

        self.embedding.forward = MethodType(
            packed_forward,
            self.embedding,
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
                "Packed embedding did not consume "
                "all planned calls."
            ),
        )


# =============================================================================
# Logit capture
# =============================================================================

class LogitCapture:
    def __init__(self, model):
        self.model = model
        self.original = (
            model.scoring_mlp.forward
        )
        self.latest = None

    def install(self) -> None:
        capture = self

        def wrapped(
            self_module,
            pair_features,
        ):
            result = capture.original(
                pair_features
            )

            capture.latest = (
                result[
                    "logit"
                ]
                .detach()
                .cpu()
                .clone()
            )

            return result

        self.model.scoring_mlp.forward = MethodType(
            wrapped,
            self.model.scoring_mlp,
        )

    def restore(self) -> None:
        self.model.scoring_mlp.forward = (
            self.original
        )


# =============================================================================
# Numerical metrics
# =============================================================================

def dense_cpu(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().cpu()

    if value.is_sparse:
        value = value.to_dense()

    return value


def tensor_stats(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict:
    ref = dense_cpu(reference)
    cand = dense_cpu(candidate)

    require(
        tuple(ref.shape)
        == tuple(cand.shape),
        (
            "Tensor shape mismatch: "
            f"{tuple(ref.shape)} vs {tuple(cand.shape)}"
        ),
    )

    ref64 = ref.to(
        dtype=torch.float64
    ).reshape(-1)

    cand64 = cand.to(
        dtype=torch.float64
    ).reshape(-1)

    diff = cand64 - ref64
    n = int(ref64.numel())

    if n == 0:
        return {
            "numel": 0,
            "max_abs_diff": 0.0,
            "mean_abs_diff": 0.0,
            "diff_l2_sq": 0.0,
            "reference_l2_sq": 0.0,
            "candidate_l2_sq": 0.0,
            "dot": 0.0,
            "exact_count": 0,
            "active_sign_count": 0,
            "sign_equal_count": 0,
        }

    abs_diff = diff.abs()

    active_mask = (
        (ref64 != 0.0)
        | (cand64 != 0.0)
    )

    active_sign_count = int(
        active_mask.sum().item()
    )

    if active_sign_count > 0:
        sign_equal_count = int(
            (
                torch.sign(
                    ref64[
                        active_mask
                    ]
                )
                == torch.sign(
                    cand64[
                        active_mask
                    ]
                )
            )
            .sum()
            .item()
        )
    else:
        sign_equal_count = 0

    return {
        "numel": n,
        "max_abs_diff": float(
            abs_diff.max().item()
        ),
        "mean_abs_diff": float(
            abs_diff.mean().item()
        ),
        "diff_l2_sq": float(
            torch.dot(
                diff,
                diff,
            ).item()
        ),
        "reference_l2_sq": float(
            torch.dot(
                ref64,
                ref64,
            ).item()
        ),
        "candidate_l2_sq": float(
            torch.dot(
                cand64,
                cand64,
            ).item()
        ),
        "dot": float(
            torch.dot(
                ref64,
                cand64,
            ).item()
        ),
        "exact_count": int(
            torch.eq(
                ref,
                cand,
            ).sum().item()
        ),
        "active_sign_count": (
            active_sign_count
        ),
        "sign_equal_count": (
            sign_equal_count
        ),
    }


def aggregate_stats(
    rows: list[dict],
) -> dict:
    require(
        len(rows) > 0,
        "Cannot aggregate empty metrics.",
    )

    numel = int(
        sum(
            row["numel"]
            for row in rows
        )
    )

    diff_l2_sq = float(
        sum(
            row["diff_l2_sq"]
            for row in rows
        )
    )

    ref_l2_sq = float(
        sum(
            row["reference_l2_sq"]
            for row in rows
        )
    )

    cand_l2_sq = float(
        sum(
            row["candidate_l2_sq"]
            for row in rows
        )
    )

    dot = float(
        sum(
            row["dot"]
            for row in rows
        )
    )

    exact_count = int(
        sum(
            row["exact_count"]
            for row in rows
        )
    )

    active_sign_count = int(
        sum(
            row["active_sign_count"]
            for row in rows
        )
    )

    sign_equal_count = int(
        sum(
            row["sign_equal_count"]
            for row in rows
        )
    )

    epsilon = 1e-30

    relative_l2_error = (
        math.sqrt(
            diff_l2_sq
        )
        / max(
            math.sqrt(
                ref_l2_sq
            ),
            epsilon,
        )
    )

    if (
        ref_l2_sq > 0.0
        and cand_l2_sq > 0.0
    ):
        cosine_similarity = (
            dot
            / (
                math.sqrt(
                    ref_l2_sq
                )
                * math.sqrt(
                    cand_l2_sq
                )
            )
        )
    elif (
        ref_l2_sq == 0.0
        and cand_l2_sq == 0.0
    ):
        cosine_similarity = 1.0
    else:
        cosine_similarity = 0.0

    return {
        "numel": numel,
        "max_abs_diff": max(
            row["max_abs_diff"]
            for row in rows
        ),
        "mean_abs_diff_weighted": (
            sum(
                row["mean_abs_diff"]
                * row["numel"]
                for row in rows
            )
            / max(numel, 1)
        ),
        "relative_l2_error": (
            relative_l2_error
        ),
        "cosine_similarity": (
            cosine_similarity
        ),
        "exact_fraction": (
            exact_count
            / max(numel, 1)
        ),
        "sign_agreement": (
            sign_equal_count
            / max(
                active_sign_count,
                1,
            )
        ),
    }


def compare_gradients(
    reference_model,
    candidate_model,
) -> dict:
    reference = dict(
        reference_model.named_parameters()
    )
    candidate = dict(
        candidate_model.named_parameters()
    )

    require(
        list(reference.keys())
        == list(candidate.keys()),
        "Parameter ordering drift.",
    )

    rows = []

    for name in reference:
        require(
            reference[name].grad is not None,
            f"Reference missing gradient: {name}",
        )
        require(
            candidate[name].grad is not None,
            f"Candidate missing gradient: {name}",
        )

        rows.append(
            tensor_stats(
                reference[name].grad,
                candidate[name].grad,
            )
        )

    return aggregate_stats(rows)


def compare_parameters(
    reference_model,
    candidate_model,
) -> dict:
    reference = dict(
        reference_model.named_parameters()
    )
    candidate = dict(
        candidate_model.named_parameters()
    )

    rows = [
        tensor_stats(
            reference[name],
            candidate[name],
        )
        for name in reference
    ]

    return aggregate_stats(rows)


def compare_adam_group(
    reference_model,
    reference_optimizer,
    candidate_model,
    candidate_optimizer,
    state_name: str,
) -> dict:
    reference = dict(
        reference_model.named_parameters()
    )
    candidate = dict(
        candidate_model.named_parameters()
    )

    rows = []

    for name in reference:
        ref_parameter = reference[name]
        cand_parameter = candidate[name]

        require(
            ref_parameter
            in reference_optimizer.state,
            f"Reference Adam state missing: {name}",
        )

        require(
            cand_parameter
            in candidate_optimizer.state,
            f"Candidate Adam state missing: {name}",
        )

        rows.append(
            tensor_stats(
                reference_optimizer.state[
                    ref_parameter
                ][state_name],
                candidate_optimizer.state[
                    cand_parameter
                ][state_name],
            )
        )

    return aggregate_stats(rows)


def numerical_summary(
    *,
    reference_model,
    reference_optimizer,
    reference_logits,
    reference_loss,
    candidate_model,
    candidate_optimizer,
    candidate_logits,
    candidate_loss,
) -> dict:
    logit = tensor_stats(
        reference_logits,
        candidate_logits,
    )

    gradient = compare_gradients(
        reference_model,
        candidate_model,
    )

    parameter = compare_parameters(
        reference_model,
        candidate_model,
    )

    adam_exp_avg = compare_adam_group(
        reference_model,
        reference_optimizer,
        candidate_model,
        candidate_optimizer,
        "exp_avg",
    )

    adam_exp_avg_sq = compare_adam_group(
        reference_model,
        reference_optimizer,
        candidate_model,
        candidate_optimizer,
        "exp_avg_sq",
    )

    return {
        "loss_abs_diff": abs(
            float(candidate_loss)
            - float(reference_loss)
        ),
        "logit_max_abs_diff": (
            logit["max_abs_diff"]
        ),
        "gradient_relative_l2_error": (
            gradient[
                "relative_l2_error"
            ]
        ),
        "gradient_cosine_similarity": (
            gradient[
                "cosine_similarity"
            ]
        ),
        "gradient_sign_agreement": (
            gradient[
                "sign_agreement"
            ]
        ),
        "parameter_relative_l2_error": (
            parameter[
                "relative_l2_error"
            ]
        ),
        "parameter_max_abs_diff": (
            parameter[
                "max_abs_diff"
            ]
        ),
        "adam_exp_avg_relative_l2_error": (
            adam_exp_avg[
                "relative_l2_error"
            ]
        ),
        "adam_exp_avg_cosine_similarity": (
            adam_exp_avg[
                "cosine_similarity"
            ]
        ),
        "adam_exp_avg_sq_relative_l2_error": (
            adam_exp_avg_sq[
                "relative_l2_error"
            ]
        ),
        "adam_exp_avg_sq_cosine_similarity": (
            adam_exp_avg_sq[
                "cosine_similarity"
            ]
        ),
    }


def policy_rows(
    batch_index: int,
    summary: dict,
    thresholds: dict,
) -> list[dict]:
    specs = [
        (
            "loss_abs_diff",
            "<=",
            thresholds[
                "loss_abs_diff_max"
            ],
        ),
        (
            "logit_max_abs_diff",
            "<=",
            thresholds[
                "logit_max_abs_diff_max"
            ],
        ),
        (
            "gradient_relative_l2_error",
            "<=",
            thresholds[
                "gradient_relative_l2_error_max"
            ],
        ),
        (
            "gradient_cosine_similarity",
            ">=",
            thresholds[
                "gradient_cosine_similarity_min"
            ],
        ),
        (
            "gradient_sign_agreement",
            ">=",
            thresholds[
                "gradient_sign_agreement_min"
            ],
        ),
        (
            "parameter_relative_l2_error",
            "<=",
            thresholds[
                "parameter_relative_l2_error_max"
            ],
        ),
        (
            "parameter_max_abs_diff",
            "<=",
            thresholds[
                "parameter_max_abs_diff_max"
            ],
        ),
        (
            "adam_exp_avg_relative_l2_error",
            "<=",
            thresholds[
                "adam_exp_avg_relative_l2_error_max"
            ],
        ),
        (
            "adam_exp_avg_cosine_similarity",
            ">=",
            thresholds[
                "adam_exp_avg_cosine_similarity_min"
            ],
        ),
        (
            "adam_exp_avg_sq_relative_l2_error",
            "<=",
            thresholds[
                "adam_exp_avg_sq_relative_l2_error_max"
            ],
        ),
        (
            "adam_exp_avg_sq_cosine_similarity",
            ">=",
            thresholds[
                "adam_exp_avg_sq_cosine_similarity_min"
            ],
        ),
    ]

    rows = []

    for metric, comparator, threshold in specs:
        actual = float(
            summary[metric]
        )

        if comparator == "<=":
            passed = (
                actual
                <= float(threshold)
            )
        else:
            passed = (
                actual
                >= float(threshold)
            )

        rows.append(
            {
                "batch_index": batch_index,
                "metric": metric,
                "actual": actual,
                "comparator": comparator,
                "threshold": float(
                    threshold
                ),
                "result": (
                    "PASS"
                    if passed
                    else "FAIL"
                ),
            }
        )

    return rows


# =============================================================================
# Execute one packed+sparse candidate batch
# =============================================================================

def execute_packed_sparse_batch(
    *,
    roundtrip,
    model,
    optimizer,
    batch,
    shared,
) -> tuple[dict, float, dict]:

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

    startup_calls = list(
        plan[
            "history_startup_calls"
        ]
    )

    startup_calls.append(
        plan[
            "pair_startup_call"
        ]
    )

    investor_calls = list(
        plan[
            "trend_investor_calls"
        ]
    )

    investor_calls.append(
        plan[
            "pair_investor_call"
        ]
    )

    startup_dispatch = PackedEmbeddingDispatch(
        model.startup_embedding,
        startup_calls,
    )

    investor_dispatch = PackedEmbeddingDispatch(
        model.investor_embedding,
        investor_calls,
    )

    startup_dispatch.install()
    investor_dispatch.install()

    setup_seconds = (
        time.perf_counter()
        - plan_start
    )

    true_gradient_hash = (
        roundtrip
        .gradient_logical_sha256
    )

    true_optimizer_hash = (
        roundtrip
        .optimizer_state_logical_sha256
    )

    roundtrip.gradient_logical_sha256 = (
        lambda model: "SKIPPED_INSIDE_PACKED_SPARSE_TIMER"
    )

    roundtrip.optimizer_state_logical_sha256 = (
        lambda model, optimizer: (
            "SKIPPED_INSIDE_PACKED_SPARSE_TIMER"
        )
    )

    dummy_model_hash = (
        lambda model: "SKIPPED_INSIDE_PACKED_SPARSE_TIMER"
    )

    execute_start = time.perf_counter()

    try:
        result = (
            roundtrip
            .execute_training_batch(
                model,
                optimizer,
                dummy_model_hash,
                batch,
                shared,
            )
        )

        execute_seconds = (
            time.perf_counter()
            - execute_start
        )

        startup_dispatch.assert_consumed()
        investor_dispatch.assert_consumed()

    finally:
        startup_dispatch.restore()
        investor_dispatch.restore()

        roundtrip.gradient_logical_sha256 = (
            true_gradient_hash
        )

        roundtrip.optimizer_state_logical_sha256 = (
            true_optimizer_hash
        )

    call_summary = {
        "history_startup_calls": len(
            plan[
                "history_startup_calls"
            ]
        ),
        "startup_calls_before_pack": len(
            startup_calls
        ),
        "startup_calls_after_pack": 1,
        "trend_investor_calls": len(
            plan[
                "trend_investor_calls"
            ]
        ),
        "investor_calls_before_pack": len(
            investor_calls
        ),
        "investor_calls_after_pack": 1,
        "unique_investor_h_keys": (
            plan[
                "unique_key_count"
            ]
        ),
        "setup_seconds": float(
            setup_seconds
        ),
        "execute_seconds": float(
            execute_seconds
        ),
    }

    total_seconds = (
        setup_seconds
        + execute_seconds
    )

    return (
        result,
        float(total_seconds),
        call_summary,
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.4.10 — "
        "PACKED + SPARSE CPU NUMERICAL EQUIVALENCE AND RUNTIME AUDIT"
    )

    print(
        "Production training launched:         NO"
    )
    print(
        "Production runtime selected:          NO"
    )
    print(
        "Numerical tolerance changed:          NO"
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
        "PREREQUISITE + FROZEN POLICY GATE"
    )

    for path in (
        ROUNDTRIP_SOURCE_PATH,
        POLICY_PATH,
        PHASE_5_4_7B_CONTRACT_PATH,
        PHASE_5_4_9_CONTRACT_PATH,
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

    policy = load_json(
        POLICY_PATH
    )

    policy_contract = load_json(
        PHASE_5_4_7B_CONTRACT_PATH
    )

    residual_contract = load_json(
        PHASE_5_4_9_CONTRACT_PATH
    )

    runtime_contract = load_json(
        PHASE_5_4_2_CONTRACT_PATH
    )

    authorization = load_json(
        PHASE_5_4_AUTHORIZATION_PATH
    )

    require(
        policy.get(
            "status"
        )
        == "FROZEN",
        "Numerical-equivalence policy is not FROZEN.",
    )

    require(
        policy.get(
            "schema_version"
        )
        == EXPECTED_POLICY_VERSION,
        "Numerical-equivalence policy version drift.",
    )

    require(
        policy_contract.get(
            "status"
        )
        == "FROZEN",
        "Phase-5.4.7b contract is not FROZEN.",
    )

    require(
        residual_contract.get(
            "status"
        )
        == "COMPLETE",
        "Phase-5.4.9 residual profile is not COMPLETE.",
    )

    require(
        residual_contract.get(
            "candidate_runtime"
        )
        == "D_BOTH_SPARSE",
        "Phase-5.4.9 candidate runtime drift.",
    )

    require(
        authorization.get(
            "training_allowed"
        )
        is True,
        "Phase-5.4 training authorization is not ALLOWED.",
    )

    canonical_lean_seconds = float(
        runtime_contract[
            "lean_exact_mean_seconds_per_batch"
        ]
    )

    prior_sparse_seconds = float(
        residual_contract[
            "prior_sparse_batch0_seconds"
        ]
    )

    print(
        f"Policy version:                       "
        f"{EXPECTED_POLICY_VERSION}"
    )
    print(
        "Policy status:                        FROZEN"
    )
    print(
        f"Canonical lean CPU:                   "
        f"{canonical_lean_seconds:.3f} s/batch"
    )
    print(
        f"Prior D_BOTH_SPARSE:                  "
        f"{prior_sparse_seconds:.3f} s/batch"
    )

    # =========================================================================
    # Load frozen runtime
    # =========================================================================

    banner(
        "LOAD FROZEN RUNTIME"
    )

    roundtrip = load_module(
        ROUNDTRIP_SOURCE_PATH,
        "_itrs_phase5_4_10_roundtrip",
    )

    require(
        torch.__version__
        == EXPECTED_PYTORCH,
        "Reference PyTorch version drift.",
    )

    torch.set_num_threads(
        EXPECTED_THREADS
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
    # Construct reference + candidate
    # =========================================================================

    banner(
        "CONSTRUCT STATEFUL REFERENCE + PACKED-SPARSE CANDIDATE"
    )

    (
        reference_model,
        reference_optimizer,
        reference_hash_fn,
        _runtime_ast_sha_ref,
        _adapter_sha_ref,
        _removed_guard_sha_ref,
    ) = (
        roundtrip
        .construct_fresh_training_state(
            preflight
        )
    )

    (
        candidate_model,
        candidate_optimizer,
        candidate_hash_fn,
        _runtime_ast_sha_cand,
        _adapter_sha_cand,
        _removed_guard_sha_cand,
    ) = (
        roundtrip
        .construct_fresh_training_state(
            preflight
        )
    )

    require(
        reference_hash_fn(
            reference_model
        )
        == EXPECTED_INITIAL_MODEL_SHA256,
        "Reference initial model SHA drift.",
    )

    require(
        candidate_hash_fn(
            candidate_model
        )
        == EXPECTED_INITIAL_MODEL_SHA256,
        "Candidate initial model SHA drift.",
    )

    candidate_model.startup_embedding.sparse = True
    candidate_model.investor_embedding.sparse = True

    reference_capture = LogitCapture(
        reference_model
    )

    candidate_capture = LogitCapture(
        candidate_model
    )

    reference_capture.install()
    candidate_capture.install()

    print(
        "Reference initial state:              BYTE-EXACT"
    )
    print(
        "Candidate initial state:              BYTE-EXACT"
    )
    print(
        "Candidate startup sparse:             TRUE"
    )
    print(
        "Candidate investor sparse:            TRUE"
    )

    thresholds = policy[
        "thresholds"
    ]

    # =========================================================================
    # Stateful two-batch audit
    # =========================================================================

    banner(
        "STATEFUL BATCH-0 -> BATCH-1 PACKED-SPARSE AUDIT"
    )

    batch_rows = []
    call_rows = []
    all_policy_rows = []

    specs = [
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
    ) in specs:

        # ---------------------------------------------------------------------
        # Canonical reference.
        # ---------------------------------------------------------------------

        reference_result = (
            roundtrip
            .execute_training_batch(
                reference_model,
                reference_optimizer,
                reference_hash_fn,
                batch,
                shared,
            )
        )

        require(
            abs(
                float(
                    reference_result[
                        "loss"
                    ]
                )
                - float(
                    expected_loss
                )
            )
            <= 5e-10,
            f"Reference loss drift at batch {batch_index}.",
        )

        require(
            reference_result[
                "logit_sha256"
            ]
            == expected_logit_sha,
            f"Reference logit SHA drift at batch {batch_index}.",
        )

        require(
            reference_result[
                "gradient_sha256"
            ]
            == expected_gradient_sha,
            f"Reference gradient SHA drift at batch {batch_index}.",
        )

        require(
            reference_result[
                "post_step_model_sha256"
            ]
            == expected_model_sha,
            f"Reference model SHA drift at batch {batch_index}.",
        )

        require(
            reference_result[
                "optimizer_state_sha256"
            ]
            == expected_optimizer_sha,
            f"Reference Adam SHA drift at batch {batch_index}.",
        )

        reference_logits = (
            reference_capture.latest
        )

        # ---------------------------------------------------------------------
        # Packed + sparse candidate.
        # ---------------------------------------------------------------------

        (
            candidate_result,
            candidate_seconds,
            call_summary,
        ) = execute_packed_sparse_batch(
            roundtrip=roundtrip,
            model=candidate_model,
            optimizer=candidate_optimizer,
            batch=batch,
            shared=shared,
        )

        candidate_logits = (
            candidate_capture.latest
        )

        # ---------------------------------------------------------------------
        # Numerical policy.
        # ---------------------------------------------------------------------

        summary = numerical_summary(
            reference_model=reference_model,
            reference_optimizer=reference_optimizer,
            reference_logits=reference_logits,
            reference_loss=(
                reference_result[
                    "loss"
                ]
            ),
            candidate_model=candidate_model,
            candidate_optimizer=candidate_optimizer,
            candidate_logits=candidate_logits,
            candidate_loss=(
                candidate_result[
                    "loss"
                ]
            ),
        )

        candidate_policy_rows = policy_rows(
            batch_index,
            summary,
            thresholds,
        )

        all_policy_rows.extend(
            candidate_policy_rows
        )

        policy_pass = all(
            row[
                "result"
            ]
            == "PASS"
            for row in candidate_policy_rows
        )

        batch_rows.append(
            {
                "batch_index": batch_index,
                "candidate_seconds": (
                    candidate_seconds
                ),
                "reference_loss": float(
                    reference_result[
                        "loss"
                    ]
                ),
                "candidate_loss": float(
                    candidate_result[
                        "loss"
                    ]
                ),
                **summary,
                "frozen_policy_pass": (
                    policy_pass
                ),
            }
        )

        call_rows.append(
            {
                "batch_index": batch_index,
                **call_summary,
            }
        )

        print()
        print(
            f"Batch {batch_index} canonical reference: BYTE-EXACT"
        )
        print(
            f"Batch {batch_index} packed+sparse:       "
            f"{candidate_seconds:.3f} s"
        )
        print(
            f"  startup embedding calls:            "
            f"{call_summary['startup_calls_before_pack']} -> 1"
        )
        print(
            f"  investor embedding calls:           "
            f"{call_summary['investor_calls_before_pack']} -> 1"
        )
        print(
            f"  loss abs diff:                      "
            f"{summary['loss_abs_diff']:.12e}"
        )
        print(
            f"  logit max abs diff:                 "
            f"{summary['logit_max_abs_diff']:.12e}"
        )
        print(
            f"  gradient relative L2:               "
            f"{summary['gradient_relative_l2_error']:.12e}"
        )
        print(
            f"  gradient cosine:                    "
            f"{summary['gradient_cosine_similarity']:.12f}"
        )
        print(
            f"  gradient sign agreement:            "
            f"{summary['gradient_sign_agreement']:.12f}"
        )
        print(
            f"  parameter relative L2:              "
            f"{summary['parameter_relative_l2_error']:.12e}"
        )
        print(
            f"  parameter max abs diff:             "
            f"{summary['parameter_max_abs_diff']:.12e}"
        )
        print(
            f"  frozen policy:                      "
            f"{'PASS' if policy_pass else 'FAIL'}"
        )

    # =========================================================================
    # Runtime decision
    # =========================================================================

    banner(
        "PACKED + SPARSE CPU POLICY DECISION + RUNTIME PROJECTION"
    )

    batch_df = pd.DataFrame(
        batch_rows
    )

    call_df = pd.DataFrame(
        call_rows
    )

    policy_df = pd.DataFrame(
        all_policy_rows
    )

    candidate_policy_pass = bool(
        (
            policy_df[
                "result"
            ]
            == "PASS"
        ).all()
    )

    mean_seconds = float(
        batch_df[
            "candidate_seconds"
        ].mean()
    )

    warm_seconds = float(
        batch_df.loc[
            batch_df[
                "batch_index"
            ]
            == 1,
            "candidate_seconds",
        ].iloc[
            0
        ]
    )

    speedup_vs_canonical = (
        canonical_lean_seconds
        / mean_seconds
    )

    speedup_vs_sparse = (
        prior_sparse_seconds
        / mean_seconds
    )

    projected_epoch_seconds = (
        mean_seconds
        * BATCHES_PER_EPOCH
    )

    projected_full_seconds = (
        mean_seconds
        * TOTAL_OPTIMIZER_STEPS
    )

    warm_projected_full_seconds = (
        warm_seconds
        * TOTAL_OPTIMIZER_STEPS
    )

    print(
        "Packed+sparse numerical equivalence:  "
        + (
            "PASS"
            if candidate_policy_pass
            else "FAIL"
        )
    )
    print(
        f"Mean seconds / batch:                 "
        f"{mean_seconds:.3f}"
    )
    print(
        f"Batch-1 warm seconds:                 "
        f"{warm_seconds:.3f}"
    )
    print(
        f"Speedup vs canonical lean CPU:        "
        f"{speedup_vs_canonical:.2f}x"
    )
    print(
        f"Speedup vs D_BOTH_SPARSE:             "
        f"{speedup_vs_sparse:.2f}x"
    )
    print(
        "Projected one epoch:                 "
        f"{human_duration(projected_epoch_seconds)}"
    )
    print(
        "Projected 20 epochs (mean):           "
        f"{human_duration(projected_full_seconds)}"
    )
    print(
        "Projected 20 epochs (warm batch1):    "
        f"{human_duration(warm_projected_full_seconds)}"
    )

    runtime_df = pd.DataFrame(
        [
            {
                "metric": (
                    "canonical_lean_cpu_seconds_per_batch"
                ),
                "value": (
                    canonical_lean_seconds
                ),
                "human": (
                    f"{canonical_lean_seconds:.3f} s"
                ),
            },
            {
                "metric": (
                    "D_BOTH_SPARSE_prior_seconds_per_batch"
                ),
                "value": (
                    prior_sparse_seconds
                ),
                "human": (
                    f"{prior_sparse_seconds:.3f} s"
                ),
            },
            {
                "metric": (
                    "packed_sparse_two_batch_mean_seconds"
                ),
                "value": (
                    mean_seconds
                ),
                "human": (
                    f"{mean_seconds:.3f} s"
                ),
            },
            {
                "metric": (
                    "packed_sparse_batch1_warm_seconds"
                ),
                "value": (
                    warm_seconds
                ),
                "human": (
                    f"{warm_seconds:.3f} s"
                ),
            },
            {
                "metric": (
                    "speedup_vs_canonical"
                ),
                "value": (
                    speedup_vs_canonical
                ),
                "human": (
                    f"{speedup_vs_canonical:.2f}x"
                ),
            },
            {
                "metric": (
                    "speedup_vs_D_BOTH_SPARSE"
                ),
                "value": (
                    speedup_vs_sparse
                ),
                "human": (
                    f"{speedup_vs_sparse:.2f}x"
                ),
            },
            {
                "metric": (
                    "projected_20_epochs_mean"
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
                    "projected_20_epochs_warm"
                ),
                "value": (
                    warm_projected_full_seconds
                ),
                "human": (
                    human_duration(
                        warm_projected_full_seconds
                    )
                ),
            },
        ]
    )

    # =========================================================================
    # Final invariants
    # =========================================================================

    banner(
        "FINAL PHASE-5.4.10 INVARIANTS"
    )

    checks = [
        (
            "frozen_policy_reused_unchanged",
            (
                policy[
                    "status"
                ]
                == "FROZEN"
            ),
        ),
        (
            "canonical_batch0_byte_exact",
            True,
        ),
        (
            "canonical_batch1_byte_exact",
            True,
        ),
        (
            "packed_sparse_two_batches_executed",
            (
                set(
                    batch_df[
                        "batch_index"
                    ].tolist()
                )
                == {
                    0,
                    1,
                }
            ),
        ),
        (
            "all_11_policy_metrics_checked_per_batch",
            bool(
                (
                    policy_df.groupby(
                        "batch_index"
                    ).size()
                    == 11
                ).all()
            ),
        ),
        (
            "packed_call_reduction_applied",
            bool(
                (
                    call_df[
                        "startup_calls_after_pack"
                    ]
                    == 1
                ).all()
                and (
                    call_df[
                        "investor_calls_after_pack"
                    ]
                    == 1
                ).all()
            ),
        ),
        (
            "candidate_timings_positive_finite",
            bool(
                np.isfinite(
                    batch_df[
                        "candidate_seconds"
                    ].to_numpy(
                        dtype=np.float64
                    )
                ).all()
                and (
                    batch_df[
                        "candidate_seconds"
                    ]
                    > 0
                ).all()
            ),
        ),
        (
            "production_runtime_not_selected",
            True,
        ),
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
    ]

    invariant_df = pd.DataFrame(
        [
            {
                "check": name,
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
        "At least one Phase-5.4.10 invariant failed.",
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
        "WRITE PHASE-5.4.10 OUTPUTS"
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTRACT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    batch_df.to_csv(
        BATCH_SUMMARY_PATH,
        index=False,
    )

    call_df.to_csv(
        CALL_PLAN_PATH,
        index=False,
    )

    policy_df.to_csv(
        POLICY_AUDIT_PATH,
        index=False,
    )

    runtime_df.to_csv(
        RUNTIME_PATH,
        index=False,
    )

    invariant_df.to_csv(
        FINAL_INVARIANT_PATH,
        index=False,
    )

    contract = {
        "phase": "5.4.10",
        "title": (
            "Packed + Sparse CPU Numerical Equivalence and Runtime Audit"
        ),
        "status": "COMPLETE",
        "classification": (
            "IMPLEMENTATION_EQUIVALENT_ACCELERATION_AUDIT"
        ),
        "policy_version": (
            EXPECTED_POLICY_VERSION
        ),
        "policy_reused_unchanged": (
            True
        ),
        "candidate_runtime": (
            CANDIDATE_RUNTIME
        ),
        "candidate_numerically_equivalent": (
            candidate_policy_pass
        ),
        "candidate_mean_seconds_per_batch": (
            mean_seconds
        ),
        "candidate_batch1_warm_seconds": (
            warm_seconds
        ),
        "speedup_vs_canonical_lean_cpu": (
            speedup_vs_canonical
        ),
        "speedup_vs_D_BOTH_SPARSE": (
            speedup_vs_sparse
        ),
        "projected_20_epochs_seconds_mean": (
            projected_full_seconds
        ),
        "projected_20_epochs_seconds_warm": (
            warm_projected_full_seconds
        ),
        "production_runtime_selected": (
            False
        ),
        "production_training_launched": (
            False
        ),
        "production_checkpoint_written": (
            False
        ),
        "validation_cases_scored": 0,
        "test_cases_scored": 0,
        "next_phase": (
            (
                "5.4.11_EXTENDED_STABILITY_AND_PRODUCTION_RUNTIME_FREEZE"
            )
            if candidate_policy_pass
            else (
                "5.4.11_EXTERNAL_CUDA_RUNTIME_PATH"
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
        "phase": "5.4.10",
        "status": (
            "PACKED_SPARSE_NUMERICAL_EQUIVALENCE_PASS"
            if candidate_policy_pass
            else "PACKED_SPARSE_NUMERICAL_EQUIVALENCE_FAIL"
        ),
        "candidate_runtime": (
            CANDIDATE_RUNTIME
        ),
        "candidate_mean_seconds_per_batch": (
            mean_seconds
        ),
        "production_training_steps": 0,
        "validation_cases_scored": 0,
        "test_cases_scored": 0,
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
        BATCH_SUMMARY_PATH,
        CALL_PLAN_PATH,
        POLICY_AUDIT_PATH,
        RUNTIME_PATH,
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

    reference_capture.restore()
    candidate_capture.restore()

    del reference_model
    del reference_optimizer
    del candidate_model
    del candidate_optimizer
    del batch0
    del batch1
    del stream
    del shared

    gc.collect()

    banner(
        "PHASE 5.4.10 FINAL STATUS"
    )

    print(
        f"Candidate runtime:                    "
        f"{CANDIDATE_RUNTIME}"
    )
    print(
        "Numerical equivalence:                "
        + (
            "PASS"
            if candidate_policy_pass
            else "FAIL"
        )
    )
    print(
        f"Mean seconds / batch:                 "
        f"{mean_seconds:.3f}"
    )
    print(
        f"Batch-1 warm seconds:                 "
        f"{warm_seconds:.3f}"
    )
    print(
        f"Speedup vs canonical lean CPU:        "
        f"{speedup_vs_canonical:.2f}x"
    )
    print(
        f"Speedup vs D_BOTH_SPARSE:             "
        f"{speedup_vs_sparse:.2f}x"
    )
    print(
        "Projected 20 epochs (mean):           "
        f"{human_duration(projected_full_seconds)}"
    )
    print(
        "Projected 20 epochs (warm):           "
        f"{human_duration(warm_projected_full_seconds)}"
    )
    print()
    print(
        "Production runtime selected:          NO"
    )
    print(
        "Production training launched:         NO"
    )
    print(
        "Validation cases scored:              0"
    )
    print(
        "Test cases scored:                    0"
    )

    banner(
        "PHASE 5.4.10 COMPLETE / "
        "PACKED + SPARSE CPU ACCELERATION AUDIT CLOSED"
    )


if __name__ == "__main__":
    main()