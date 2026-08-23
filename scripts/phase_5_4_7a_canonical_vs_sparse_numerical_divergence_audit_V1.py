#!/usr/bin/env python3
"""
Phase 5.4.7a — Canonical vs Sparse Runtime Numerical Divergence Audit

Purpose
-------
Quantify the ACTUAL floating-point divergence between:

    Reference:
        canonical dense-embedding CPU runtime

    Candidate:
        D_BOTH_SPARSE
        startup_embedding.sparse = True
        investor_embedding.sparse = True

The candidate was rejected under the byte-exact rule in Phase 5.4.6, but its
forward logits remained exact and its runtime was much faster. A hash mismatch
alone does not tell us whether the numerical divergence is 1e-9 or materially
large.

This audit therefore measures, without adopting any acceptance threshold:
    - loss difference
    - logit tensor difference
    - gradient difference across all 32 parameters
    - parameter-state difference after Adam
    - Adam exp_avg difference
    - Adam exp_avg_sq difference
    - sign agreement
    - cosine similarity
    - exact-element fraction

It follows BOTH runtimes statefully through:
    canonical initialization
        -> batch 0 -> Adam step
        -> batch 1 -> Adam step

No model-selection decision is made here.
No numerical-equivalence tolerance is frozen here.
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
from pathlib import Path
from types import MethodType
from typing import Dict, Iterable, Tuple

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

PHASE_5_4_6_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_4_6_sparse_embedding_acceleration_contract.json"
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
    "phase_5_4_7a"
)

BATCH_SUMMARY_PATH = (
    AUDIT_DIR
    / "canonical_vs_sparse_batch_summary.csv"
)

GRADIENT_DETAIL_PATH = (
    AUDIT_DIR
    / "canonical_vs_sparse_gradient_parameter_detail.csv"
)

PARAMETER_DETAIL_PATH = (
    AUDIT_DIR
    / "canonical_vs_sparse_parameter_state_detail.csv"
)

ADAM_DETAIL_PATH = (
    AUDIT_DIR
    / "canonical_vs_sparse_adam_state_detail.csv"
)

AGGREGATE_PATH = (
    AUDIT_DIR
    / "canonical_vs_sparse_numerical_aggregate.csv"
)

FINAL_INVARIANT_PATH = (
    AUDIT_DIR
    / "phase_5_4_7a_final_invariants.csv"
)

CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_4_7a_numerical_divergence_characterization_contract.json"
)

MANIFEST_PATH = (
    AUDIT_DIR
    / "phase_5_4_7a_numerical_divergence_characterization_manifest.json"
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

EXPECTED_PARAMETER_TENSORS = 32
CANDIDATE_VARIANT = "D_BOTH_SPARSE"


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

    sys.modules[module_name] = module

    spec.loader.exec_module(module)

    return module


def close_float(
    actual: float,
    expected: float,
    tolerance: float = 5e-10,
) -> bool:
    return (
        math.isfinite(float(actual))
        and abs(float(actual) - float(expected))
        <= tolerance
    )


class LogitCapture:
    def __init__(self, model):
        self.model = model
        self.original = model.scoring_mlp.forward
        self.latest = None

    def install(self) -> None:
        capture = self

        def wrapped(self_module, pair_features):
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

        self.model.scoring_mlp.forward = MethodType(
            wrapped,
            self.model.scoring_mlp,
        )

    def restore(self) -> None:
        self.model.scoring_mlp.forward = self.original


def tensor_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict:
    """
    Compute divergence metrics in float64 without concatenating model tensors.
    """

    ref = reference.detach().cpu()
    cand = candidate.detach().cpu()

    if ref.is_sparse:
        ref = ref.to_dense()

    if cand.is_sparse:
        cand = cand.to_dense()

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
    abs_diff = diff.abs()

    n = int(
        ref64.numel()
    )

    max_abs = (
        float(
            abs_diff.max().item()
        )
        if n > 0
        else 0.0
    )

    mean_abs = (
        float(
            abs_diff.mean().item()
        )
        if n > 0
        else 0.0
    )

    diff_l2_sq = float(
        torch.dot(
            diff,
            diff,
        ).item()
    )

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

    dot = float(
        torch.dot(
            ref64,
            cand64,
        ).item()
    )

    eps = 1e-30

    rel_l2 = math.sqrt(
        diff_l2_sq
    ) / max(
        math.sqrt(
            ref_l2_sq
        ),
        eps,
    )

    if (
        ref_l2_sq > 0.0
        and cand_l2_sq > 0.0
    ):
        cosine = dot / (
            math.sqrt(
                ref_l2_sq
            )
            * math.sqrt(
                cand_l2_sq
            )
        )
    elif (
        ref_l2_sq == 0.0
        and cand_l2_sq == 0.0
    ):
        cosine = 1.0
    else:
        cosine = 0.0

    exact_count = int(
        torch.eq(
            ref,
            cand,
        )
        .sum()
        .item()
    )

    exact_fraction = (
        exact_count / n
        if n > 0
        else 1.0
    )

    ref_sign = torch.sign(
        ref64
    )

    cand_sign = torch.sign(
        cand64
    )

    active_mask = (
        (ref64 != 0.0)
        | (cand64 != 0.0)
    )

    active_count = int(
        active_mask.sum().item()
    )

    if active_count > 0:
        sign_equal_count = int(
            (
                ref_sign[
                    active_mask
                ]
                == cand_sign[
                    active_mask
                ]
            )
            .sum()
            .item()
        )

        sign_agreement = (
            sign_equal_count
            / active_count
        )
    else:
        sign_agreement = 1.0

    return {
        "numel": n,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "diff_l2_sq": diff_l2_sq,
        "reference_l2_sq": ref_l2_sq,
        "candidate_l2_sq": cand_l2_sq,
        "dot": dot,
        "relative_l2_error": rel_l2,
        "cosine_similarity": cosine,
        "exact_count": exact_count,
        "exact_fraction": exact_fraction,
        "active_sign_count": active_count,
        "sign_agreement": sign_agreement,
    }


def aggregate_metrics(
    detail_df: pd.DataFrame,
) -> dict:
    require(
        len(detail_df) > 0,
        "Cannot aggregate empty tensor detail.",
    )

    numel = int(
        detail_df["numel"].sum()
    )

    diff_l2_sq = float(
        detail_df["diff_l2_sq"].sum()
    )

    reference_l2_sq = float(
        detail_df["reference_l2_sq"].sum()
    )

    candidate_l2_sq = float(
        detail_df["candidate_l2_sq"].sum()
    )

    dot = float(
        detail_df["dot"].sum()
    )

    exact_count = int(
        detail_df["exact_count"].sum()
    )

    active_sign_count = int(
        detail_df["active_sign_count"].sum()
    )

    sign_equal_count = int(
        round(
            float(
                (
                    detail_df[
                        "sign_agreement"
                    ]
                    * detail_df[
                        "active_sign_count"
                    ]
                ).sum()
            )
        )
    )

    eps = 1e-30

    relative_l2_error = (
        math.sqrt(diff_l2_sq)
        / max(
            math.sqrt(
                reference_l2_sq
            ),
            eps,
        )
    )

    if (
        reference_l2_sq > 0.0
        and candidate_l2_sq > 0.0
    ):
        cosine_similarity = dot / (
            math.sqrt(
                reference_l2_sq
            )
            * math.sqrt(
                candidate_l2_sq
            )
        )
    elif (
        reference_l2_sq == 0.0
        and candidate_l2_sq == 0.0
    ):
        cosine_similarity = 1.0
    else:
        cosine_similarity = 0.0

    return {
        "numel": numel,
        "max_abs_diff": float(
            detail_df[
                "max_abs_diff"
            ].max()
        ),
        "mean_abs_diff_weighted": (
            float(
                (
                    detail_df[
                        "mean_abs_diff"
                    ]
                    * detail_df[
                        "numel"
                    ]
                ).sum()
            )
            / max(
                numel,
                1,
            )
        ),
        "relative_l2_error": (
            relative_l2_error
        ),
        "cosine_similarity": (
            cosine_similarity
        ),
        "exact_fraction": (
            exact_count
            / max(
                numel,
                1,
            )
        ),
        "sign_agreement": (
            sign_equal_count
            / max(
                active_sign_count,
                1,
            )
        ),
    }


def compare_named_gradients(
    reference_model,
    candidate_model,
    *,
    batch_index: int,
) -> pd.DataFrame:
    reference = dict(
        reference_model.named_parameters()
    )

    candidate = dict(
        candidate_model.named_parameters()
    )

    require(
        list(reference.keys())
        == list(candidate.keys()),
        (
            "Parameter-name ordering differs "
            "between reference and candidate."
        ),
    )

    rows = []

    for name in reference:
        ref_grad = reference[name].grad
        cand_grad = candidate[name].grad

        require(
            ref_grad is not None
            and cand_grad is not None,
            (
                f"Missing gradient for {name}."
            ),
        )

        metrics = tensor_metrics(
            ref_grad,
            cand_grad,
        )

        rows.append(
            {
                "batch_index": (
                    batch_index
                ),
                "tensor_name": name,
                "tensor_group": (
                    "gradient"
                ),
                **metrics,
            }
        )

    return pd.DataFrame(
        rows
    )


def compare_named_parameters(
    reference_model,
    candidate_model,
    *,
    batch_index: int,
) -> pd.DataFrame:
    reference = dict(
        reference_model.named_parameters()
    )

    candidate = dict(
        candidate_model.named_parameters()
    )

    require(
        list(reference.keys())
        == list(candidate.keys()),
        (
            "Parameter-name ordering differs "
            "between reference and candidate."
        ),
    )

    rows = []

    for name in reference:
        metrics = tensor_metrics(
            reference[name],
            candidate[name],
        )

        rows.append(
            {
                "batch_index": (
                    batch_index
                ),
                "tensor_name": name,
                "tensor_group": (
                    "parameter_state"
                ),
                **metrics,
            }
        )

    return pd.DataFrame(
        rows
    )


def compare_optimizer_state(
    reference_model,
    reference_optimizer,
    candidate_model,
    candidate_optimizer,
    *,
    batch_index: int,
) -> pd.DataFrame:
    ref_named = dict(
        reference_model.named_parameters()
    )

    cand_named = dict(
        candidate_model.named_parameters()
    )

    rows = []

    for name in ref_named:
        ref_parameter = ref_named[name]
        cand_parameter = cand_named[name]

        require(
            ref_parameter
            in reference_optimizer.state,
            (
                f"Reference Adam state missing: {name}"
            ),
        )

        require(
            cand_parameter
            in candidate_optimizer.state,
            (
                f"Candidate Adam state missing: {name}"
            ),
        )

        ref_state = reference_optimizer.state[
            ref_parameter
        ]

        cand_state = candidate_optimizer.state[
            cand_parameter
        ]

        ref_step = float(
            ref_state[
                "step"
            ].item()
        )

        cand_step = float(
            cand_state[
                "step"
            ].item()
        )

        require(
            ref_step == cand_step,
            (
                f"Adam step mismatch for {name}: "
                f"{ref_step} vs {cand_step}"
            ),
        )

        for state_name in (
            "exp_avg",
            "exp_avg_sq",
        ):
            metrics = tensor_metrics(
                ref_state[
                    state_name
                ],
                cand_state[
                    state_name
                ],
            )

            rows.append(
                {
                    "batch_index": (
                        batch_index
                    ),
                    "tensor_name": name,
                    "tensor_group": (
                        f"adam_{state_name}"
                    ),
                    "adam_step": (
                        ref_step
                    ),
                    **metrics,
                }
            )

    return pd.DataFrame(
        rows
    )


def aggregate_group(
    detail_df: pd.DataFrame,
    *,
    batch_index: int,
    group: str,
) -> dict:
    subset = detail_df.loc[
        detail_df[
            "batch_index"
        ]
        == batch_index
    ]

    require(
        len(subset) > 0,
        (
            f"No detail rows for group={group}, "
            f"batch={batch_index}."
        ),
    )

    metrics = aggregate_metrics(
        subset
    )

    return {
        "batch_index": batch_index,
        "comparison_group": group,
        **metrics,
    }


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.4.7a — "
        "CANONICAL VS SPARSE RUNTIME NUMERICAL DIVERGENCE AUDIT"
    )

    print(
        "Production training launched:         NO"
    )
    print(
        "Production checkpoint written:        NO"
    )
    print(
        "Numerical tolerance frozen:           NO"
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
        PHASE_5_4_6_CONTRACT_PATH,
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

    sparse_contract = load_json(
        PHASE_5_4_6_CONTRACT_PATH
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
        int(
            runtime_contract.get(
                "selected_threads",
                -1,
            )
        )
        == EXPECTED_SELECTED_THREADS,
        (
            "Exact CPU thread-count drift."
        ),
    )

    require(
        sparse_contract.get(
            "status"
        )
        == "COMPLETE",
        (
            "Phase-5.4.6 sparse audit "
            "is not COMPLETE."
        ),
    )

    require(
        sparse_contract.get(
            "acceleration_status"
        )
        == (
            "NO_BYTE_EXACT_SPARSE_BACKWARD_RUNTIME"
        ),
        (
            "Phase-5.4.6 prerequisite state drift."
        ),
    )

    require(
        authorization.get(
            "training_allowed"
        )
        is True,
        (
            "Production training authorization "
            "is not ALLOWED."
        ),
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
        == CANDIDATE_VARIANT
    ]

    require(
        len(d_rows) == 1,
        (
            "Could not resolve exactly one "
            "D_BOTH_SPARSE Phase-5.4.6 result."
        ),
    )

    prior_candidate_seconds = float(
        d_rows[0][
            "elapsed_seconds"
        ]
    )

    print(
        "Canonical CPU reference:              FROZEN"
    )
    print(
        f"Candidate runtime:                    "
        f"{CANDIDATE_VARIANT}"
    )
    print(
        f"Prior candidate batch-0 seconds:      "
        f"{prior_candidate_seconds:.3f}"
    )

    # =========================================================================
    # Load runtime and two independent model states
    # =========================================================================

    banner(
        "LOAD CANONICAL AND CANDIDATE STATES"
    )

    roundtrip = load_module(
        ROUNDTRIP_SOURCE_PATH,
        "_itrs_phase5_4_7a_roundtrip",
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
        (
            "Reference initial model SHA drift."
        ),
    )

    require(
        candidate_hash_fn(
            candidate_model
        )
        == EXPECTED_INITIAL_MODEL_SHA256,
        (
            "Candidate initial model SHA drift."
        ),
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
        "Initial parameter state equality:     BYTE-EXACT"
    )

    # =========================================================================
    # Stateful two-batch comparison
    # =========================================================================

    banner(
        "STATEFUL BATCH-0 -> BATCH-1 NUMERICAL COMPARISON"
    )

    batch_summaries = []
    all_gradient_detail = []
    all_parameter_detail = []
    all_adam_detail = []
    aggregate_rows = []

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
        expected_reference_loss,
        expected_reference_logit_sha,
        expected_reference_gradient_sha,
        expected_reference_model_sha,
        expected_reference_optimizer_sha,
    ) in batch_specs:

        # Reference
        ref_start = time.perf_counter()

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

        reference_seconds = (
            time.perf_counter()
            - ref_start
        )

        # Candidate
        cand_start = time.perf_counter()

        candidate_result = (
            roundtrip
            .execute_training_batch(
                candidate_model,
                candidate_optimizer,
                candidate_hash_fn,
                batch,
                shared,
            )
        )

        candidate_seconds = (
            time.perf_counter()
            - cand_start
        )

        # Reference must remain the exact frozen CPU trajectory.
        require(
            close_float(
                reference_result[
                    "loss"
                ],
                expected_reference_loss,
            ),
            (
                f"Reference loss drift at "
                f"batch {batch_index}."
            ),
        )

        require(
            reference_result[
                "logit_sha256"
            ]
            == expected_reference_logit_sha,
            (
                f"Reference logit SHA drift at "
                f"batch {batch_index}."
            ),
        )

        require(
            reference_result[
                "gradient_sha256"
            ]
            == expected_reference_gradient_sha,
            (
                f"Reference gradient SHA drift at "
                f"batch {batch_index}."
            ),
        )

        require(
            reference_result[
                "post_step_model_sha256"
            ]
            == expected_reference_model_sha,
            (
                f"Reference model SHA drift at "
                f"batch {batch_index}."
            ),
        )

        require(
            reference_result[
                "optimizer_state_sha256"
            ]
            == expected_reference_optimizer_sha,
            (
                f"Reference optimizer SHA drift at "
                f"batch {batch_index}."
            ),
        )

        require(
            reference_capture.latest
            is not None
            and candidate_capture.latest
            is not None,
            (
                "Logit capture missing."
            ),
        )

        logit_metrics = tensor_metrics(
            reference_capture.latest,
            candidate_capture.latest,
        )

        gradient_detail = compare_named_gradients(
            reference_model,
            candidate_model,
            batch_index=batch_index,
        )

        parameter_detail = compare_named_parameters(
            reference_model,
            candidate_model,
            batch_index=batch_index,
        )

        adam_detail = compare_optimizer_state(
            reference_model,
            reference_optimizer,
            candidate_model,
            candidate_optimizer,
            batch_index=batch_index,
        )

        all_gradient_detail.append(
            gradient_detail
        )

        all_parameter_detail.append(
            parameter_detail
        )

        all_adam_detail.append(
            adam_detail
        )

        grad_agg = aggregate_metrics(
            gradient_detail
        )

        parameter_agg = aggregate_metrics(
            parameter_detail
        )

        exp_avg_agg = aggregate_metrics(
            adam_detail.loc[
                adam_detail[
                    "tensor_group"
                ]
                == "adam_exp_avg"
            ]
        )

        exp_avg_sq_agg = aggregate_metrics(
            adam_detail.loc[
                adam_detail[
                    "tensor_group"
                ]
                == "adam_exp_avg_sq"
            ]
        )

        aggregate_rows.extend(
            [
                {
                    "batch_index": batch_index,
                    "comparison_group": "logits",
                    **logit_metrics,
                },
                {
                    "batch_index": batch_index,
                    "comparison_group": "gradients",
                    **grad_agg,
                },
                {
                    "batch_index": batch_index,
                    "comparison_group": "parameter_state",
                    **parameter_agg,
                },
                {
                    "batch_index": batch_index,
                    "comparison_group": "adam_exp_avg",
                    **exp_avg_agg,
                },
                {
                    "batch_index": batch_index,
                    "comparison_group": "adam_exp_avg_sq",
                    **exp_avg_sq_agg,
                },
            ]
        )

        loss_abs_diff = abs(
            float(
                candidate_result[
                    "loss"
                ]
            )
            - float(
                reference_result[
                    "loss"
                ]
            )
        )

        batch_summaries.append(
            {
                "batch_index": batch_index,
                "reference_seconds_including_hashes": (
                    reference_seconds
                ),
                "candidate_seconds_including_hashes": (
                    candidate_seconds
                ),
                "reference_loss": (
                    reference_result[
                        "loss"
                    ]
                ),
                "candidate_loss": (
                    candidate_result[
                        "loss"
                    ]
                ),
                "loss_abs_diff": (
                    loss_abs_diff
                ),
                "reference_logit_sha256": (
                    reference_result[
                        "logit_sha256"
                    ]
                ),
                "candidate_logit_sha256": (
                    candidate_result[
                        "logit_sha256"
                    ]
                ),
                "logit_hash_exact": (
                    reference_result[
                        "logit_sha256"
                    ]
                    == candidate_result[
                        "logit_sha256"
                    ]
                ),
                "logit_max_abs_diff": (
                    logit_metrics[
                        "max_abs_diff"
                    ]
                ),
                "logit_mean_abs_diff": (
                    logit_metrics[
                        "mean_abs_diff"
                    ]
                ),
                "logit_relative_l2_error": (
                    logit_metrics[
                        "relative_l2_error"
                    ]
                ),
                "logit_cosine_similarity": (
                    logit_metrics[
                        "cosine_similarity"
                    ]
                ),
                "gradient_hash_exact": (
                    reference_result[
                        "gradient_sha256"
                    ]
                    == candidate_result[
                        "gradient_sha256"
                    ]
                ),
                "gradient_max_abs_diff": (
                    grad_agg[
                        "max_abs_diff"
                    ]
                ),
                "gradient_mean_abs_diff_weighted": (
                    grad_agg[
                        "mean_abs_diff_weighted"
                    ]
                ),
                "gradient_relative_l2_error": (
                    grad_agg[
                        "relative_l2_error"
                    ]
                ),
                "gradient_cosine_similarity": (
                    grad_agg[
                        "cosine_similarity"
                    ]
                ),
                "gradient_exact_fraction": (
                    grad_agg[
                        "exact_fraction"
                    ]
                ),
                "gradient_sign_agreement": (
                    grad_agg[
                        "sign_agreement"
                    ]
                ),
                "post_model_hash_exact": (
                    reference_result[
                        "post_step_model_sha256"
                    ]
                    == candidate_result[
                        "post_step_model_sha256"
                    ]
                ),
                "parameter_max_abs_diff": (
                    parameter_agg[
                        "max_abs_diff"
                    ]
                ),
                "parameter_mean_abs_diff_weighted": (
                    parameter_agg[
                        "mean_abs_diff_weighted"
                    ]
                ),
                "parameter_relative_l2_error": (
                    parameter_agg[
                        "relative_l2_error"
                    ]
                ),
                "parameter_cosine_similarity": (
                    parameter_agg[
                        "cosine_similarity"
                    ]
                ),
                "parameter_exact_fraction": (
                    parameter_agg[
                        "exact_fraction"
                    ]
                ),
                "optimizer_hash_exact": (
                    reference_result[
                        "optimizer_state_sha256"
                    ]
                    == candidate_result[
                        "optimizer_state_sha256"
                    ]
                ),
                "adam_exp_avg_relative_l2_error": (
                    exp_avg_agg[
                        "relative_l2_error"
                    ]
                ),
                "adam_exp_avg_cosine_similarity": (
                    exp_avg_agg[
                        "cosine_similarity"
                    ]
                ),
                "adam_exp_avg_sq_relative_l2_error": (
                    exp_avg_sq_agg[
                        "relative_l2_error"
                    ]
                ),
                "adam_exp_avg_sq_cosine_similarity": (
                    exp_avg_sq_agg[
                        "cosine_similarity"
                    ]
                ),
            }
        )

        print()
        print(
            f"Batch {batch_index}"
        )
        print(
            f"  reference loss:                     "
            f"{reference_result['loss']:.12f}"
        )
        print(
            f"  candidate loss:                     "
            f"{candidate_result['loss']:.12f}"
        )
        print(
            f"  loss abs diff:                      "
            f"{loss_abs_diff:.12e}"
        )
        print(
            f"  logit max abs diff:                 "
            f"{logit_metrics['max_abs_diff']:.12e}"
        )
        print(
            f"  logit relative L2:                  "
            f"{logit_metrics['relative_l2_error']:.12e}"
        )
        print(
            f"  gradient max abs diff:              "
            f"{grad_agg['max_abs_diff']:.12e}"
        )
        print(
            f"  gradient relative L2:               "
            f"{grad_agg['relative_l2_error']:.12e}"
        )
        print(
            f"  gradient cosine:                    "
            f"{grad_agg['cosine_similarity']:.12f}"
        )
        print(
            f"  gradient sign agreement:            "
            f"{grad_agg['sign_agreement']:.12f}"
        )
        print(
            f"  parameter max abs diff after Adam:  "
            f"{parameter_agg['max_abs_diff']:.12e}"
        )
        print(
            f"  parameter relative L2 after Adam:   "
            f"{parameter_agg['relative_l2_error']:.12e}"
        )
        print(
            f"  parameter exact fraction:           "
            f"{parameter_agg['exact_fraction']:.12f}"
        )

    # =========================================================================
    # Consolidate
    # =========================================================================

    gradient_detail_df = pd.concat(
        all_gradient_detail,
        ignore_index=True,
    )

    parameter_detail_df = pd.concat(
        all_parameter_detail,
        ignore_index=True,
    )

    adam_detail_df = pd.concat(
        all_adam_detail,
        ignore_index=True,
    )

    aggregate_df = pd.DataFrame(
        aggregate_rows
    )

    batch_summary_df = pd.DataFrame(
        batch_summaries
    )

    # =========================================================================
    # Decision-facing diagnostic summary
    # =========================================================================

    banner(
        "DECISION-FACING NUMERICAL CHARACTERIZATION"
    )

    print(
        batch_summary_df[
            [
                "batch_index",
                "loss_abs_diff",
                "logit_max_abs_diff",
                "gradient_max_abs_diff",
                "gradient_relative_l2_error",
                "gradient_cosine_similarity",
                "gradient_sign_agreement",
                "parameter_max_abs_diff",
                "parameter_relative_l2_error",
                "parameter_exact_fraction",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.12e}"
            ),
        )
    )

    # =========================================================================
    # Final invariants
    # =========================================================================

    banner(
        "FINAL PHASE-5.4.7a INVARIANTS"
    )

    reference_batch0_exact = bool(
        close_float(
            batch_summary_df.loc[
                batch_summary_df[
                    "batch_index"
                ]
                == 0,
                "reference_loss",
            ].iloc[0],
            EXPECTED_BATCH0_LOSS,
        )
    )

    reference_batch1_exact = bool(
        close_float(
            batch_summary_df.loc[
                batch_summary_df[
                    "batch_index"
                ]
                == 1,
                "reference_loss",
            ].iloc[0],
            EXPECTED_BATCH1_LOSS,
        )
    )

    checks = [
        (
            "production_training_not_launched",
            True,
        ),
        (
            "numerical_tolerance_not_frozen",
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
            "reference_batch0_exact",
            reference_batch0_exact,
        ),
        (
            "reference_batch1_exact",
            reference_batch1_exact,
        ),
        (
            "both_batches_characterized",
            (
                set(
                    batch_summary_df[
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
            "all_32_gradient_tensors_compared_per_batch",
            bool(
                (
                    gradient_detail_df
                    .groupby(
                        "batch_index"
                    )
                    .size()
                    == EXPECTED_PARAMETER_TENSORS
                ).all()
            ),
        ),
        (
            "all_32_parameter_tensors_compared_per_batch",
            bool(
                (
                    parameter_detail_df
                    .groupby(
                        "batch_index"
                    )
                    .size()
                    == EXPECTED_PARAMETER_TENSORS
                ).all()
            ),
        ),
        (
            "adam_exp_avg_and_sq_compared",
            bool(
                set(
                    adam_detail_df[
                        "tensor_group"
                    ].unique()
                )
                == {
                    "adam_exp_avg",
                    "adam_exp_avg_sq",
                }
            ),
        ),
        (
            "all_metrics_finite",
            bool(
                np.isfinite(
                    batch_summary_df.select_dtypes(
                        include=[
                            np.number,
                        ]
                    ).to_numpy(
                        dtype=np.float64
                    )
                ).all()
            ),
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
        (
            "At least one Phase-5.4.7a "
            "diagnostic invariant failed."
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
        "WRITE PHASE-5.4.7a OUTPUTS"
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTRACT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    batch_summary_df.to_csv(
        BATCH_SUMMARY_PATH,
        index=False,
    )

    gradient_detail_df.to_csv(
        GRADIENT_DETAIL_PATH,
        index=False,
    )

    parameter_detail_df.to_csv(
        PARAMETER_DETAIL_PATH,
        index=False,
    )

    adam_detail_df.to_csv(
        ADAM_DETAIL_PATH,
        index=False,
    )

    aggregate_df.to_csv(
        AGGREGATE_PATH,
        index=False,
    )

    invariant_df.to_csv(
        FINAL_INVARIANT_PATH,
        index=False,
    )

    contract = {
        "phase": (
            "5.4.7a"
        ),
        "title": (
            "Canonical vs Sparse Runtime Numerical Divergence Characterization"
        ),
        "status": (
            "DIAGNOSTIC_COMPLETE"
        ),
        "classification": (
            "IMPLEMENTATION_EQUIVALENT_NUMERICAL_CHARACTERIZATION"
        ),
        "reference_runtime": (
            "CANONICAL_DENSE_EMBEDDING_CPU"
        ),
        "candidate_runtime": (
            CANDIDATE_VARIANT
        ),
        "selected_exact_cpu_threads": (
            EXPECTED_SELECTED_THREADS
        ),
        "tolerance_frozen": (
            False
        ),
        "acceptance_decision_made": (
            False
        ),
        "batches_compared": [
            0,
            1,
        ],
        "batch_summary": (
            batch_summary_df.to_dict(
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
            "5.4.7b_NUMERICAL_EQUIVALENCE_DECISION_AND_DEVICE_BENCHMARK_POLICY"
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
            "5.4.7a"
        ),
        "status": (
            "NUMERICAL_DIVERGENCE_CHARACTERIZATION_COMPLETE"
        ),
        "reference_runtime": (
            "CANONICAL_DENSE_EMBEDDING_CPU"
        ),
        "candidate_runtime": (
            CANDIDATE_VARIANT
        ),
        "tolerance_frozen": (
            False
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
        BATCH_SUMMARY_PATH,
        GRADIENT_DETAIL_PATH,
        PARAMETER_DETAIL_PATH,
        ADAM_DETAIL_PATH,
        AGGREGATE_PATH,
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
        "PHASE 5.4.7a FINAL STATUS"
    )

    print(
        "Canonical CPU trajectory:             EXACT / FROZEN"
    )
    print(
        f"Candidate characterized:              "
        f"{CANDIDATE_VARIANT}"
    )
    print(
        "Numerical tolerance frozen:           NO"
    )
    print(
        "Candidate accepted for production:    NO — NOT DECIDED HERE"
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
        "PHASE 5.4.7a COMPLETE / "
        "NUMERICAL DIVERGENCE CHARACTERIZED WITHOUT ACCEPTANCE DECISION"
    )


if __name__ == "__main__":
    main()