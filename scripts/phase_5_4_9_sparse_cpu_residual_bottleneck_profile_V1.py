#!/usr/bin/env python3
"""
Phase 5.4.9 — D_BOTH_SPARSE CPU Residual Bottleneck Profile

Purpose
-------
Profile the FASTEST currently validated numerical-equivalent runtime:

    D_BOTH_SPARSE CPU
        startup_embedding.sparse = True
        investor_embedding.sparse = True

This runtime already passed the frozen policy:
    ITRS_PHASE5_NUMERICAL_EQUIVALENCE_V1

Phase 5.4.9 asks a new question:
    After sparse embedding backward removed the original dense-gradient
    pathology, what now dominates the remaining ~6.8 seconds per batch?

Two independent candidate passes are run from the canonical seed-42 state:

    Pass A — high-level wall-clock profile
        structural forward
        description forward
        trend attention
        trend GRU
        scoring forward
        backward
        Adam step
        unattributed preprocessing/check overhead

    Pass B — PyTorch operator-level profile
        ranked by self CPU time and total CPU time

Both passes are compared against a canonical CPU batch-0 reference and must
satisfy the already-frozen numerical-equivalence policy.

This phase freezes NO new numerical tolerance and makes NO production-runtime
selection.

No production checkpoint is written.
No validation is scored.
No test is accessed.
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
from torch.profiler import ProfilerActivity, profile


# =============================================================================
# Paths
# =============================================================================

ROUNDTRIP_SOURCE_PATH = Path(
    "scripts/"
    "phase_5_3_2b_checkpoint_resume_roundtrip_proof.py"
)

POLICY_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_4_numerical_equivalence_policy.json"
)

PHASE_5_4_7B_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_4_7b_numerical_equivalence_policy_contract.json"
)

PHASE_5_4_6_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_4_6_sparse_embedding_acceleration_contract.json"
)

PHASE_5_4_8B_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_4_8b_packed_mps_numerical_equivalence_feasibility_contract.json"
)

PHASE_5_4_AUTHORIZATION_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_4_production_training_launch_authorization.json"
)

AUDIT_DIR = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_4_9"
)

HIGH_LEVEL_PATH = (
    AUDIT_DIR
    / "sparse_cpu_high_level_component_profile.csv"
)

HIGH_LEVEL_AGGREGATE_PATH = (
    AUDIT_DIR
    / "sparse_cpu_high_level_component_ranking.csv"
)

OPERATOR_SELF_PATH = (
    AUDIT_DIR
    / "sparse_cpu_operator_profile_by_self_cpu.csv"
)

OPERATOR_TOTAL_PATH = (
    AUDIT_DIR
    / "sparse_cpu_operator_profile_by_total_cpu.csv"
)

POLICY_AUDIT_PATH = (
    AUDIT_DIR
    / "sparse_cpu_profile_numerical_policy_audit.csv"
)

SUMMARY_PATH = (
    AUDIT_DIR
    / "sparse_cpu_residual_bottleneck_summary.csv"
)

FINAL_INVARIANT_PATH = (
    AUDIT_DIR
    / "phase_5_4_9_final_invariants.csv"
)

CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_4_9_sparse_cpu_residual_bottleneck_contract.json"
)

MANIFEST_PATH = (
    AUDIT_DIR
    / "phase_5_4_9_sparse_cpu_residual_bottleneck_manifest.json"
)


# =============================================================================
# Frozen anchors
# =============================================================================

EXPECTED_PYTORCH = "2.7.0"
EXPECTED_SELECTED_THREADS = 8
EXPECTED_POLICY_VERSION = "ITRS_PHASE5_NUMERICAL_EQUIVALENCE_V1"
CANDIDATE_RUNTIME = "D_BOTH_SPARSE"

EXPECTED_INITIAL_MODEL_SHA256 = (
    "49e822ea7fad35c458f47e134c94c05e"
    "ac099b68c5c468e2c71559c8c88998ab"
)

EXPECTED_BATCH0_LOSS = 0.7080879807

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

    module = importlib.util.module_from_spec(spec)

    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu()

    if value.is_sparse:
        value = value.to_dense()

    array = value.contiguous().numpy()

    return __import__("hashlib").sha256(
        array.tobytes(order="C")
    ).hexdigest()


class LogitCapture:
    def __init__(
        self,
        model,
        timer=None,
    ):
        self.model = model
        self.timer = timer
        self.original = model.scoring_mlp.forward
        self.latest = None

    def install(self) -> None:
        capture = self

        def wrapped(
            self_module,
            pair_features,
        ):
            start = time.perf_counter()

            try:
                result = capture.original(
                    pair_features
                )

                capture.latest = (
                    result["logit"]
                    .detach()
                    .cpu()
                    .clone()
                )

                return result
            finally:
                if capture.timer is not None:
                    capture.timer.add(
                        "scoring_forward",
                        time.perf_counter() - start,
                    )

        self.model.scoring_mlp.forward = MethodType(
            wrapped,
            self.model.scoring_mlp,
        )

    def restore(self) -> None:
        self.model.scoring_mlp.forward = self.original


class TimerAccumulator:
    def __init__(self):
        self.total = defaultdict(float)
        self.calls = defaultdict(int)

    def add(self, name: str, seconds: float) -> None:
        self.total[name] += float(seconds)
        self.calls[name] += 1


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
                time.perf_counter() - start,
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


def _to_dense_cpu(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().cpu()

    if value.is_sparse:
        value = value.to_dense()

    return value


def tensor_stats(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict:
    ref = _to_dense_cpu(reference)
    cand = _to_dense_cpu(candidate)

    require(
        tuple(ref.shape) == tuple(cand.shape),
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

    ref_l2_sq = float(
        torch.dot(
            ref64,
            ref64,
        ).item()
    )

    cand_l2_sq = float(
        torch.dot(
            cand64,
            cand64,
        ).item()
    )

    diff_l2_sq = float(
        torch.dot(
            diff,
            diff,
        ).item()
    )

    dot = float(
        torch.dot(
            ref64,
            cand64,
        ).item()
    )

    exact_count = int(
        torch.eq(
            ref,
            cand,
        ).sum().item()
    )

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
                    ref64[active_mask]
                )
                == torch.sign(
                    cand64[active_mask]
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
        "diff_l2_sq": diff_l2_sq,
        "reference_l2_sq": ref_l2_sq,
        "candidate_l2_sq": cand_l2_sq,
        "dot": dot,
        "exact_count": exact_count,
        "active_sign_count": active_sign_count,
        "sign_equal_count": sign_equal_count,
    }


def aggregate_stats(rows: list[dict]) -> dict:
    require(
        len(rows) > 0,
        "Cannot aggregate empty metric rows.",
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

    reference_l2_sq = float(
        sum(
            row["reference_l2_sq"]
            for row in rows
        )
    )

    candidate_l2_sq = float(
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
        math.sqrt(diff_l2_sq)
        / max(
            math.sqrt(reference_l2_sq),
            epsilon,
        )
    )

    if (
        reference_l2_sq > 0.0
        and candidate_l2_sq > 0.0
    ):
        cosine_similarity = (
            dot
            / (
                math.sqrt(reference_l2_sq)
                * math.sqrt(candidate_l2_sq)
            )
        )
    elif (
        reference_l2_sq == 0.0
        and candidate_l2_sq == 0.0
    ):
        cosine_similarity = 1.0
    else:
        cosine_similarity = 0.0

    mean_abs_diff_weighted = (
        sum(
            row["mean_abs_diff"]
            * row["numel"]
            for row in rows
        )
        / max(numel, 1)
    )

    return {
        "numel": numel,
        "max_abs_diff": max(
            row["max_abs_diff"]
            for row in rows
        ),
        "mean_abs_diff_weighted": (
            mean_abs_diff_weighted
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
            / max(active_sign_count, 1)
        ),
    }


def compare_model_gradients(
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
        "Parameter-name ordering drift.",
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


def compare_model_parameters(
    reference_model,
    candidate_model,
) -> dict:
    reference = dict(
        reference_model.named_parameters()
    )

    candidate = dict(
        candidate_model.named_parameters()
    )

    rows = []

    for name in reference:
        rows.append(
            tensor_stats(
                reference[name],
                candidate[name],
            )
        )

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


def make_policy_rows(
    *,
    pass_name: str,
    summary: dict,
    thresholds: dict,
) -> list[dict]:
    specifications = [
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

    for (
        metric,
        comparator,
        threshold,
    ) in specifications:
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
                "profile_pass": (
                    pass_name
                ),
                "metric": metric,
                "actual": actual,
                "comparator": (
                    comparator
                ),
                "threshold": (
                    float(threshold)
                ),
                "result": (
                    "PASS"
                    if passed
                    else "FAIL"
                ),
            }
        )

    return rows


def numerical_summary(
    *,
    reference_model,
    reference_optimizer,
    reference_logits,
    reference_loss: float,
    candidate_model,
    candidate_optimizer,
    candidate_logits,
    candidate_loss: float,
) -> dict:
    logit = tensor_stats(
        reference_logits,
        candidate_logits,
    )

    gradient = compare_model_gradients(
        reference_model,
        candidate_model,
    )

    parameter = compare_model_parameters(
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


def operator_rows(profiler) -> pd.DataFrame:
    rows = []

    for event in profiler.key_averages():
        rows.append(
            {
                "operator": str(event.key),
                "count": int(event.count),
                "self_cpu_time_us": float(
                    event.self_cpu_time_total
                ),
                "cpu_time_total_us": float(
                    event.cpu_time_total
                ),
                "self_cpu_time_s": float(
                    event.self_cpu_time_total
                    / 1_000_000.0
                ),
                "cpu_time_total_s": float(
                    event.cpu_time_total
                    / 1_000_000.0
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.4.9 — "
        "D_BOTH_SPARSE CPU RESIDUAL BOTTLENECK PROFILE"
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
        PHASE_5_4_6_CONTRACT_PATH,
        PHASE_5_4_8B_CONTRACT_PATH,
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

    sparse_contract = load_json(
        PHASE_5_4_6_CONTRACT_PATH
    )

    packed_mps_contract = load_json(
        PHASE_5_4_8B_CONTRACT_PATH
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
        sparse_contract.get(
            "status"
        )
        == "COMPLETE",
        "Phase-5.4.6 sparse audit is not COMPLETE.",
    )

    require(
        packed_mps_contract.get(
            "status"
        )
        == "COMPLETE",
        "Phase-5.4.8b contract is not COMPLETE.",
    )

    require(
        packed_mps_contract.get(
            "production_runtime_selected"
        )
        is False,
        (
            "Phase-5.4.8b unexpectedly selected "
            "a production runtime."
        ),
    )

    require(
        authorization.get(
            "training_allowed"
        )
        is True,
        "Phase-5.4 training authorization is not ALLOWED.",
    )

    candidate_rows = sparse_contract.get(
        "candidate_variants",
        []
    )

    d_rows = [
        row
        for row in candidate_rows
        if row.get(
            "variant"
        )
        == CANDIDATE_RUNTIME
    ]

    require(
        len(d_rows) == 1,
        "Could not resolve D_BOTH_SPARSE runtime row.",
    )

    prior_sparse_seconds = float(
        d_rows[0][
            "elapsed_seconds"
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
        f"Candidate runtime:                    "
        f"{CANDIDATE_RUNTIME}"
    )
    print(
        f"Prior sparse batch-0 runtime:         "
        f"{prior_sparse_seconds:.3f} s"
    )

    # =========================================================================
    # Load frozen runtime
    # =========================================================================

    banner(
        "LOAD FROZEN RUNTIME"
    )

    roundtrip = load_module(
        ROUNDTRIP_SOURCE_PATH,
        "_itrs_phase5_4_9_roundtrip",
    )

    require(
        torch.__version__
        == EXPECTED_PYTORCH,
        "Reference PyTorch version drift.",
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
    # Canonical batch-0 reference
    # =========================================================================

    banner(
        "BUILD CANONICAL BATCH-0 REFERENCE"
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

    require(
        reference_hash_fn(
            reference_model
        )
        == EXPECTED_INITIAL_MODEL_SHA256,
        "Canonical initial model SHA drift.",
    )

    reference_capture = LogitCapture(
        reference_model
    )
    reference_capture.install()

    reference_result = (
        roundtrip
        .execute_training_batch(
            reference_model,
            reference_optimizer,
            reference_hash_fn,
            batch0,
            shared,
        )
    )

    reference_capture.restore()

    reference_logits = (
        reference_capture.latest
    )

    require(
        abs(
            float(
                reference_result[
                    "loss"
                ]
            )
            - EXPECTED_BATCH0_LOSS
        )
        <= 5e-10,
        "Reference batch-0 BCE drift.",
    )

    require(
        reference_result[
            "logit_sha256"
        ]
        == EXPECTED_BATCH0_LOGIT_SHA256,
        "Reference batch-0 logit SHA drift.",
    )

    require(
        reference_result[
            "gradient_sha256"
        ]
        == EXPECTED_BATCH0_GRADIENT_SHA256,
        "Reference batch-0 gradient SHA drift.",
    )

    require(
        reference_result[
            "post_step_model_sha256"
        ]
        == EXPECTED_BATCH0_POST_MODEL_SHA256,
        "Reference batch-0 model SHA drift.",
    )

    require(
        reference_result[
            "optimizer_state_sha256"
        ]
        == EXPECTED_BATCH0_OPTIMIZER_SHA256,
        "Reference batch-0 optimizer SHA drift.",
    )

    print(
        "Canonical batch-0 trajectory:         BYTE-EXACT"
    )

    thresholds = policy[
        "thresholds"
    ]

    # =========================================================================
    # PASS A — High-level sparse CPU profile
    # =========================================================================

    banner(
        "PASS A — HIGH-LEVEL D_BOTH_SPARSE PROFILE"
    )

    (
        high_model,
        high_optimizer,
        _high_hash_fn,
        _runtime_ast_sha_high,
        _adapter_sha_high,
        _removed_guard_sha_high,
    ) = (
        roundtrip
        .construct_fresh_training_state(
            preflight
        )
    )

    high_model.startup_embedding.sparse = True
    high_model.investor_embedding.sparse = True

    timer = TimerAccumulator()

    wrap_bound_method(
        high_model.preference_propagation,
        "forward",
        "structural_forward",
        timer,
    )

    wrap_bound_method(
        high_model.description_encoder,
        "forward",
        "description_forward",
        timer,
    )

    wrap_bound_method(
        high_model.trend_extractor,
        "attend_period",
        "trend_attention",
        timer,
    )

    wrap_bound_method(
        high_model.trend_extractor,
        "encode_training_sequence",
        "trend_gru",
        timer,
    )

    high_capture = LogitCapture(
        high_model,
        timer=timer,
    )
    high_capture.install()

    original_optimizer_step = (
        high_optimizer.step
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
                time.perf_counter() - start,
            )

    high_optimizer.step = (
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
                time.perf_counter() - start,
            )

    torch.autograd.backward = (
        timed_autograd_backward
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
        lambda model: "SKIPPED_INSIDE_PROFILE_TIMER"
    )

    roundtrip.optimizer_state_logical_sha256 = (
        lambda model, optimizer: "SKIPPED_INSIDE_PROFILE_TIMER"
    )

    dummy_model_hash_fn = (
        lambda model: "SKIPPED_INSIDE_PROFILE_TIMER"
    )

    high_start = time.perf_counter()

    high_result = (
        roundtrip
        .execute_training_batch(
            high_model,
            high_optimizer,
            dummy_model_hash_fn,
            batch0,
            shared,
        )
    )

    high_total_seconds = (
        time.perf_counter()
        - high_start
    )

    torch.autograd.backward = (
        original_autograd_backward
    )

    roundtrip.gradient_logical_sha256 = (
        true_gradient_hash_fn
    )

    roundtrip.optimizer_state_logical_sha256 = (
        true_optimizer_hash_fn
    )

    high_capture.restore()

    named_components = {
        "structural_forward": (
            timer.total.get(
                "structural_forward",
                0.0,
            )
        ),
        "description_forward": (
            timer.total.get(
                "description_forward",
                0.0,
            )
        ),
        "trend_attention": (
            timer.total.get(
                "trend_attention",
                0.0,
            )
        ),
        "trend_gru": (
            timer.total.get(
                "trend_gru",
                0.0,
            )
        ),
        "scoring_forward": (
            timer.total.get(
                "scoring_forward",
                0.0,
            )
        ),
        "backward": (
            timer.total.get(
                "backward",
                0.0,
            )
        ),
        "adam_step": (
            timer.total.get(
                "adam_step",
                0.0,
            )
        ),
    }

    named_sum = sum(
        named_components.values()
    )

    unattributed = max(
        0.0,
        high_total_seconds
        - named_sum,
    )

    high_rows = []

    for component, seconds in (
        list(
            named_components.items()
        )
        + [
            (
                "unattributed_preprocessing_checks",
                unattributed,
            )
        ]
    ):
        high_rows.append(
            {
                "component": component,
                "calls": (
                    timer.calls.get(
                        component,
                        1
                        if component
                        == (
                            "unattributed_preprocessing_checks"
                        )
                        else 0,
                    )
                ),
                "wall_seconds": (
                    float(seconds)
                ),
                "share_percent": (
                    100.0
                    * float(seconds)
                    / max(
                        high_total_seconds,
                        1e-30,
                    )
                ),
                "batch_total_seconds": (
                    high_total_seconds
                ),
            }
        )

    high_df = pd.DataFrame(
        high_rows
    )

    high_ranking_df = (
        high_df.sort_values(
            [
                "wall_seconds",
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

    high_ranking_df[
        "rank"
    ] = np.arange(
        1,
        len(
            high_ranking_df
        )
        + 1,
        dtype=np.int64,
    )

    print(
        high_ranking_df[
            [
                "rank",
                "component",
                "calls",
                "wall_seconds",
                "share_percent",
            ]
        ]
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

    high_numerical = numerical_summary(
        reference_model=reference_model,
        reference_optimizer=reference_optimizer,
        reference_logits=reference_logits,
        reference_loss=float(
            reference_result[
                "loss"
            ]
        ),
        candidate_model=high_model,
        candidate_optimizer=high_optimizer,
        candidate_logits=high_capture.latest,
        candidate_loss=float(
            high_result[
                "loss"
            ]
        ),
    )

    high_policy_rows = make_policy_rows(
        pass_name="HIGH_LEVEL",
        summary=high_numerical,
        thresholds=thresholds,
    )

    high_policy_pass = all(
        row["result"] == "PASS"
        for row in high_policy_rows
    )

    require(
        high_policy_pass,
        (
            "High-level sparse profile drifted outside "
            "the frozen numerical-equivalence policy."
        ),
    )

    print()
    print(
        f"High-level sparse wall time:          "
        f"{high_total_seconds:.3f} s"
    )
    print(
        "High-level frozen policy:              PASS"
    )

    # =========================================================================
    # PASS B — Operator-level sparse CPU profile
    # =========================================================================

    banner(
        "PASS B — PYTORCH OPERATOR PROFILE FOR D_BOTH_SPARSE"
    )

    (
        op_model,
        op_optimizer,
        _op_hash_fn,
        _runtime_ast_sha_op,
        _adapter_sha_op,
        _removed_guard_sha_op,
    ) = (
        roundtrip
        .construct_fresh_training_state(
            preflight
        )
    )

    op_model.startup_embedding.sparse = True
    op_model.investor_embedding.sparse = True

    op_capture = LogitCapture(
        op_model
    )
    op_capture.install()

    true_gradient_hash_fn_2 = (
        roundtrip
        .gradient_logical_sha256
    )

    true_optimizer_hash_fn_2 = (
        roundtrip
        .optimizer_state_logical_sha256
    )

    roundtrip.gradient_logical_sha256 = (
        lambda model: "SKIPPED_INSIDE_OPERATOR_PROFILE"
    )

    roundtrip.optimizer_state_logical_sha256 = (
        lambda model, optimizer: "SKIPPED_INSIDE_OPERATOR_PROFILE"
    )

    dummy_model_hash_fn_2 = (
        lambda model: "SKIPPED_INSIDE_OPERATOR_PROFILE"
    )

    operator_wall_start = time.perf_counter()

    with profile(
        activities=[
            ProfilerActivity.CPU,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
    ) as prof:

        op_result = (
            roundtrip
            .execute_training_batch(
                op_model,
                op_optimizer,
                dummy_model_hash_fn_2,
                batch0,
                shared,
            )
        )

    operator_wall_seconds = (
        time.perf_counter()
        - operator_wall_start
    )

    roundtrip.gradient_logical_sha256 = (
        true_gradient_hash_fn_2
    )

    roundtrip.optimizer_state_logical_sha256 = (
        true_optimizer_hash_fn_2
    )

    op_capture.restore()

    op_numerical = numerical_summary(
        reference_model=reference_model,
        reference_optimizer=reference_optimizer,
        reference_logits=reference_logits,
        reference_loss=float(
            reference_result[
                "loss"
            ]
        ),
        candidate_model=op_model,
        candidate_optimizer=op_optimizer,
        candidate_logits=op_capture.latest,
        candidate_loss=float(
            op_result[
                "loss"
            ]
        ),
    )

    op_policy_rows = make_policy_rows(
        pass_name="OPERATOR",
        summary=op_numerical,
        thresholds=thresholds,
    )

    op_policy_pass = all(
        row["result"] == "PASS"
        for row in op_policy_rows
    )

    require(
        op_policy_pass,
        (
            "Operator sparse profile drifted outside "
            "the frozen numerical-equivalence policy."
        ),
    )

    operator_df = operator_rows(
        prof
    )

    require(
        len(operator_df) > 0,
        "PyTorch profiler returned no operator rows.",
    )

    by_self_df = (
        operator_df.sort_values(
            [
                "self_cpu_time_us",
                "operator",
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

    total_self_cpu = float(
        by_self_df[
            "self_cpu_time_us"
        ].sum()
    )

    by_self_df[
        "self_cpu_share_percent"
    ] = (
        100.0
        * by_self_df[
            "self_cpu_time_us"
        ]
        / max(
            total_self_cpu,
            1e-30,
        )
    )

    by_self_df[
        "rank"
    ] = np.arange(
        1,
        len(
            by_self_df
        )
        + 1,
        dtype=np.int64,
    )

    by_total_df = (
        operator_df.sort_values(
            [
                "cpu_time_total_us",
                "operator",
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

    by_total_df[
        "rank"
    ] = np.arange(
        1,
        len(
            by_total_df
        )
        + 1,
        dtype=np.int64,
    )

    print(
        by_self_df[
            [
                "rank",
                "operator",
                "count",
                "self_cpu_time_s",
                "cpu_time_total_s",
                "self_cpu_share_percent",
            ]
        ]
        .head(
            30
        )
        .to_string(
            index=False,
            formatters={
                "self_cpu_time_s": (
                    lambda value: (
                        f"{value:.6f}"
                    )
                ),
                "cpu_time_total_s": (
                    lambda value: (
                        f"{value:.6f}"
                    )
                ),
                "self_cpu_share_percent": (
                    lambda value: (
                        f"{value:.2f}"
                    )
                ),
            },
        )
    )

    print()
    print(
        f"Operator-profile wall time:           "
        f"{operator_wall_seconds:.3f} s"
    )
    print(
        "Operator-profile frozen policy:        PASS"
    )

    # =========================================================================
    # Consolidate
    # =========================================================================

    banner(
        "RESIDUAL BOTTLENECK DECISION"
    )

    policy_df = pd.DataFrame(
        high_policy_rows
        + op_policy_rows
    )

    dominant_high_component = str(
        high_ranking_df.iloc[
            0
        ][
            "component"
        ]
    )

    dominant_high_seconds = float(
        high_ranking_df.iloc[
            0
        ][
            "wall_seconds"
        ]
    )

    dominant_high_share = float(
        high_ranking_df.iloc[
            0
        ][
            "share_percent"
        ]
    )

    top_operator = str(
        by_self_df.iloc[
            0
        ][
            "operator"
        ]
    )

    top_operator_seconds = float(
        by_self_df.iloc[
            0
        ][
            "self_cpu_time_s"
        ]
    )

    top_operator_share = float(
        by_self_df.iloc[
            0
        ][
            "self_cpu_share_percent"
        ]
    )

    embedding_named = by_self_df.loc[
        by_self_df[
            "operator"
        ].str.contains(
            "embedding",
            case=False,
            regex=False,
        )
    ]

    embedding_self_seconds = float(
        embedding_named[
            "self_cpu_time_s"
        ].sum()
    )

    embedding_self_share = float(
        embedding_named[
            "self_cpu_share_percent"
        ].sum()
    )

    fill_add_rows = by_self_df.loc[
        by_self_df[
            "operator"
        ].isin(
            [
                "aten::fill_",
                "aten::add_",
            ]
        )
    ]

    fill_add_seconds = float(
        fill_add_rows[
            "self_cpu_time_s"
        ].sum()
    )

    fill_add_share = float(
        fill_add_rows[
            "self_cpu_share_percent"
        ].sum()
    )

    summary_df = pd.DataFrame(
        [
            {
                "candidate_runtime": (
                    CANDIDATE_RUNTIME
                ),
                "prior_sparse_batch0_seconds": (
                    prior_sparse_seconds
                ),
                "high_level_profile_seconds": (
                    high_total_seconds
                ),
                "dominant_high_level_component": (
                    dominant_high_component
                ),
                "dominant_high_level_seconds": (
                    dominant_high_seconds
                ),
                "dominant_high_level_share_percent": (
                    dominant_high_share
                ),
                "top_operator": (
                    top_operator
                ),
                "top_operator_self_seconds": (
                    top_operator_seconds
                ),
                "top_operator_self_share_percent": (
                    top_operator_share
                ),
                "embedding_named_self_seconds": (
                    embedding_self_seconds
                ),
                "embedding_named_self_share_percent": (
                    embedding_self_share
                ),
                "fill_plus_add_self_seconds": (
                    fill_add_seconds
                ),
                "fill_plus_add_self_share_percent": (
                    fill_add_share
                ),
                "high_level_policy_pass": (
                    high_policy_pass
                ),
                "operator_policy_pass": (
                    op_policy_pass
                ),
            }
        ]
    )

    print(
        f"Dominant high-level component:        "
        f"{dominant_high_component}"
    )
    print(
        f"Dominant high-level share:            "
        f"{dominant_high_share:.2f}%"
    )
    print(
        f"Top self-CPU operator:                "
        f"{top_operator}"
    )
    print(
        f"Top operator self share:              "
        f"{top_operator_share:.2f}%"
    )
    print(
        f"Embedding-named self share:           "
        f"{embedding_self_share:.2f}%"
    )
    print(
        f"aten::fill_ + aten::add_ share:       "
        f"{fill_add_share:.2f}%"
    )

    # =========================================================================
    # Final invariants
    # =========================================================================

    banner(
        "FINAL PHASE-5.4.9 INVARIANTS"
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
            "reference_batch0_byte_exact",
            True,
        ),
        (
            "high_level_sparse_policy_pass",
            high_policy_pass,
        ),
        (
            "operator_sparse_policy_pass",
            op_policy_pass,
        ),
        (
            "high_level_profile_nonempty",
            (
                len(high_df) > 0
            ),
        ),
        (
            "operator_profile_nonempty",
            (
                len(operator_df) > 0
            ),
        ),
        (
            "dominant_high_level_component_identified",
            (
                len(
                    dominant_high_component
                )
                > 0
            ),
        ),
        (
            "top_operator_identified",
            (
                len(
                    top_operator
                )
                > 0
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
        "At least one Phase-5.4.9 invariant failed.",
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
        "WRITE PHASE-5.4.9 OUTPUTS"
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTRACT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    high_df.to_csv(
        HIGH_LEVEL_PATH,
        index=False,
    )

    high_ranking_df.to_csv(
        HIGH_LEVEL_AGGREGATE_PATH,
        index=False,
    )

    by_self_df.to_csv(
        OPERATOR_SELF_PATH,
        index=False,
    )

    by_total_df.to_csv(
        OPERATOR_TOTAL_PATH,
        index=False,
    )

    policy_df.to_csv(
        POLICY_AUDIT_PATH,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    invariant_df.to_csv(
        FINAL_INVARIANT_PATH,
        index=False,
    )

    contract = {
        "phase": (
            "5.4.9"
        ),
        "title": (
            "D_BOTH_SPARSE CPU Residual Bottleneck Profile"
        ),
        "status": (
            "COMPLETE"
        ),
        "classification": (
            "RUNTIME_PROFILING_ONLY"
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
        "prior_sparse_batch0_seconds": (
            prior_sparse_seconds
        ),
        "high_level_profile_seconds": (
            high_total_seconds
        ),
        "dominant_high_level_component": (
            dominant_high_component
        ),
        "dominant_high_level_seconds": (
            dominant_high_seconds
        ),
        "dominant_high_level_share_percent": (
            dominant_high_share
        ),
        "top_operator": (
            top_operator
        ),
        "top_operator_self_seconds": (
            top_operator_seconds
        ),
        "top_operator_self_share_percent": (
            top_operator_share
        ),
        "embedding_named_self_seconds": (
            embedding_self_seconds
        ),
        "embedding_named_self_share_percent": (
            embedding_self_share
        ),
        "fill_plus_add_self_seconds": (
            fill_add_seconds
        ),
        "fill_plus_add_self_share_percent": (
            fill_add_share
        ),
        "high_level_policy_pass": (
            high_policy_pass
        ),
        "operator_profile_policy_pass": (
            op_policy_pass
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
        "validation_cases_scored": (
            0
        ),
        "test_cases_scored": (
            0
        ),
        "next_phase": (
            "5.4.10_TARGETED_ACCELERATION_OF_SPARSE_CPU_RESIDUAL_BOTTLENECK"
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
            "5.4.9"
        ),
        "status": (
            "SPARSE_CPU_RESIDUAL_BOTTLENECK_IDENTIFIED"
        ),
        "candidate_runtime": (
            CANDIDATE_RUNTIME
        ),
        "dominant_high_level_component": (
            dominant_high_component
        ),
        "top_operator": (
            top_operator
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
        HIGH_LEVEL_PATH,
        HIGH_LEVEL_AGGREGATE_PATH,
        OPERATOR_SELF_PATH,
        OPERATOR_TOTAL_PATH,
        POLICY_AUDIT_PATH,
        SUMMARY_PATH,
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

    del reference_model
    del reference_optimizer
    del high_model
    del high_optimizer
    del op_model
    del op_optimizer
    del batch0
    del stream
    del shared

    gc.collect()

    banner(
        "PHASE 5.4.9 FINAL STATUS"
    )

    print(
        f"Candidate runtime:                    "
        f"{CANDIDATE_RUNTIME}"
    )
    print(
        f"High-level profile seconds:           "
        f"{high_total_seconds:.3f}"
    )
    print(
        f"Dominant high-level component:        "
        f"{dominant_high_component}"
    )
    print(
        f"Dominant high-level share:            "
        f"{dominant_high_share:.2f}%"
    )
    print(
        f"Top operator:                         "
        f"{top_operator}"
    )
    print(
        f"Top operator share:                   "
        f"{top_operator_share:.2f}%"
    )
    print()
    print(
        "Frozen numerical policy:              PASS / BOTH PROFILE PASSES"
    )
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
        "PHASE 5.4.9 COMPLETE / "
        "SPARSE CPU RESIDUAL BOTTLENECK IDENTIFIED"
    )


if __name__ == "__main__":
    main()