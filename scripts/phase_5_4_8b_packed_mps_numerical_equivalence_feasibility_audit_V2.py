#!/usr/bin/env python3
"""
Phase 5.4.8b — Packed-Embedding Apple MPS Numerical Equivalence + Feasibility Audit V2

Purpose
-------
Combine the two strongest runtime findings already established:

1. Phase 5.4.5:
   Packing the thousands of ordinary embedding calls dramatically reduces CPU
   backward cost, while preserving exact forward logits.

2. Phase 5.4.8a:
   Apple MPS is numerically equivalent to the canonical CPU path under the
   already-frozen ITRS_PHASE5_NUMERICAL_EQUIVALENCE_V1 policy, but canonical
   dense-call MPS is slower because it launches thousands of tiny operations.

Candidate runtime
-----------------
B_ALL_TREND_AND_PAIR_PACK_MPS

For each batch:
    - all historical startup embedding calls + current startup batch lookup are
      represented by ONE ordered startup_embedding lookup;
    - all trend-query investor embedding calls + current investor batch lookup
      are represented by ONE ordered investor_embedding lookup;
    - the frozen executor then consumes non-overlapping views of those packed
      results in the exact canonical call order;
    - structural propagation still uses the same embedding .weight tensors;
    - BCE, backward, and Adam remain unchanged.

Scientific acceptance
---------------------
The candidate is judged against the already-frozen numerical-equivalence policy.
No tolerance is changed here.

Reference execution
-------------------
The CPU reference uses an AST-derived lean device executor from the canonical
frozen execute_training_batch implementation. This script fixes the Phase-5.4.8a
AST provenance bookkeeping bug by hashing the canonical AST BEFORE applying the
transformation to a separately parsed tree.

The derived CPU executor must reproduce the frozen batch0 -> batch1 trajectory
byte-exactly before the packed MPS comparison is accepted as valid.

No production checkpoint is written.
No validation is scored.
No test is accessed.
No production runtime is selected here.
"""

from __future__ import annotations

import ast
import gc
import hashlib
import importlib.util
import inspect
import json
import math
import platform
import sys
import textwrap
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

PHASE_5_4_8A_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_4_8a_mps_numerical_equivalence_feasibility_contract.json"
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
    "data/experimental/phase_5/audits/phase_5_4_8b"
)

CPU_BRIDGE_PATH = (
    AUDIT_DIR
    / "corrected_ast_device_bridge_cpu_exactness_proof.csv"
)

PACK_PLAN_PATH = (
    AUDIT_DIR
    / "packed_mps_embedding_call_plan_summary.csv"
)

MPS_BATCH_PATH = (
    AUDIT_DIR
    / "packed_mps_numerical_equivalence_batch_summary.csv"
)

POLICY_AUDIT_PATH = (
    AUDIT_DIR
    / "packed_mps_numerical_equivalence_policy_audit.csv"
)

RUNTIME_PATH = (
    AUDIT_DIR
    / "packed_mps_runtime_feasibility_projection.csv"
)

FINAL_INVARIANT_PATH = (
    AUDIT_DIR
    / "phase_5_4_8b_final_invariants.csv"
)

CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_4_8b_packed_mps_numerical_equivalence_feasibility_contract.json"
)

MANIFEST_PATH = (
    AUDIT_DIR
    / "phase_5_4_8b_packed_mps_numerical_equivalence_feasibility_manifest.json"
)


# =============================================================================
# Frozen anchors
# =============================================================================

EXPECTED_PYTORCH = "2.7.0"
EXPECTED_SELECTED_THREADS = 8
EXPECTED_POLICY_VERSION = "ITRS_PHASE5_NUMERICAL_EQUIVALENCE_V1"

# These anchors were observed by Phase-5.4.8b V1 on the USER'S actual
# frozen local roundtrip source BEFORE any packed-MPS candidate batch ran.
# V1 stopped immediately on the foreign hard-coded hash mismatch, so these
# are valid pre-outcome provenance anchors for V2.
EXPECTED_CANONICAL_EXECUTOR_AST_SHA256 = (
    "2559833a4f41728af24bed81237c419b"
    "7658578ed5e953e710bab2d42613ff6a"
)

EXPECTED_LEAN_EXECUTOR_AST_SHA256 = (
    "72fa36d02b2a863c049baa2a0c95593"
    "e29127da361b3ca9451ee7bab33176556"
)

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

CANDIDATE_RUNTIME = "B_ALL_TREND_AND_PAIR_PACK_MPS"


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


def close_float(
    actual: float,
    expected: float,
    tolerance: float = 5e-10,
) -> bool:
    return (
        math.isfinite(float(actual))
        and abs(float(actual) - float(expected)) <= tolerance
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


# =============================================================================
# Corrected AST-derived device executor
# =============================================================================

class LeanDeviceExecutorTransformer(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.require_statements_removed = 0
        self.from_numpy_rewrites = 0
        self.tensor_rewrites = 0

    @staticmethod
    def _is_name_call(node, name: str) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == name
        )

    @staticmethod
    def _is_torch_attribute_call(node, attribute: str) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
            and node.func.attr == attribute
        )

    def visit_Expr(self, node):
        if self._is_name_call(node.value, "require"):
            self.require_statements_removed += 1
            return ast.copy_location(ast.Pass(), node)

        return self.generic_visit(node)

    def visit_Call(self, node):
        node = self.generic_visit(node)

        if self._is_torch_attribute_call(node, "from_numpy"):
            self.from_numpy_rewrites += 1
            return ast.copy_location(
                ast.Call(
                    func=ast.Name(
                        id="_device_from_numpy",
                        ctx=ast.Load(),
                    ),
                    args=node.args,
                    keywords=node.keywords,
                ),
                node,
            )

        if self._is_torch_attribute_call(node, "tensor"):
            self.tensor_rewrites += 1
            return ast.copy_location(
                ast.Call(
                    func=ast.Name(
                        id="_device_tensor",
                        ctx=ast.Load(),
                    ),
                    args=node.args,
                    keywords=node.keywords,
                ),
                node,
            )

        return node


def build_corrected_lean_device_executor(roundtrip):
    source = textwrap.dedent(
        inspect.getsource(roundtrip.execute_training_batch)
    )

    # IMPORTANT: hash the canonical tree before transforming a separately
    # parsed tree. This corrects the 5.4.8a bookkeeping bug.
    canonical_tree = ast.parse(source)
    canonical_ast_sha = hashlib.sha256(
        ast.dump(
            canonical_tree,
            include_attributes=False,
        ).encode("utf-8")
    ).hexdigest()

    transformed_tree = ast.parse(source)
    transformer = LeanDeviceExecutorTransformer()
    transformed_tree = transformer.visit(transformed_tree)
    transformed_tree = ast.fix_missing_locations(transformed_tree)

    transformed_ast_sha = hashlib.sha256(
        ast.dump(
            transformed_tree,
            include_attributes=False,
        ).encode("utf-8")
    ).hexdigest()

    runtime_context = {
        "device": torch.device("cpu")
    }

    original_from_numpy = torch.from_numpy
    original_tensor = torch.tensor

    def _device_from_numpy(array):
        return original_from_numpy(array).to(
            runtime_context["device"]
        )

    def _device_tensor(*args, **kwargs):
        if "device" not in kwargs:
            kwargs["device"] = runtime_context["device"]

        return original_tensor(*args, **kwargs)

    namespace = dict(roundtrip.__dict__)
    namespace["_device_from_numpy"] = _device_from_numpy
    namespace["_device_tensor"] = _device_tensor

    namespace["tensor_sha256"] = (
        lambda tensor: "LEAN_RUNTIME_HASH_SKIPPED"
    )

    namespace["gradient_logical_sha256"] = (
        lambda model: "LEAN_RUNTIME_HASH_SKIPPED"
    )

    namespace["optimizer_state_logical_sha256"] = (
        lambda model, optimizer: "LEAN_RUNTIME_HASH_SKIPPED"
    )

    namespace["rng_snapshot"] = lambda: None
    namespace["rng_equal"] = lambda before, after: True

    exec(
        compile(
            transformed_tree,
            filename="<phase_5_4_8b_lean_device_executor>",
            mode="exec",
        ),
        namespace,
    )

    executor = namespace["execute_training_batch"]

    def set_device(device):
        runtime_context["device"] = torch.device(device)

    metadata = {
        "canonical_executor_ast_sha256": canonical_ast_sha,
        "lean_device_executor_ast_sha256": transformed_ast_sha,
        "require_statements_removed": transformer.require_statements_removed,
        "torch_from_numpy_rewrites": transformer.from_numpy_rewrites,
        "torch_tensor_rewrites": transformer.tensor_rewrites,
        "phase_5_4_8a_ast_hash_bookkeeping_correction": True,
    }

    return executor, set_device, metadata


# =============================================================================
# Packed embedding call plan + device dispatch
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
    ].to_numpy(dtype=np.int64)

    batch_startup_locals = batch[
        "startup_local"
    ].to_numpy(dtype=np.int64)

    batch_segments = batch[
        "segment_number"
    ].to_numpy(dtype=np.int64)

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
        for period in range(h):
            flattened = (
                investor_global
                * num_history_periods
                + period
            )

            start = int(
                shared["trend_period_ptr"][flattened]
            )

            end = int(
                shared["trend_period_ptr"][flattened + 1]
            )

            if end <= start:
                continue

            startup_globals = np.array(
                shared["trend_startup_indices"][start:end],
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
            [int(investor_global)],
            dtype=np.int64,
        )
        for investor_global, _h
        in unique_keys
    ]

    return {
        "history_startup_calls": history_startup_calls,
        "trend_investor_calls": trend_investor_calls,
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
        "unique_key_count": len(unique_keys),
    }


class DevicePackedEmbeddingDispatch:
    """
    Replace many embedding forward calls with one packed lookup on `device`.

    No per-call CPU value verification occurs inside the timed MPS path because
    that would force thousands of GPU synchronizations. The deterministic call
    plan is derived from the same frozen batch/history arrays, and the resulting
    end-to-end outputs/gradients are judged by the frozen numerical policy.
    """

    def __init__(
        self,
        embedding: torch.nn.Embedding,
        planned_calls: list[np.ndarray],
        *,
        device: torch.device,
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

        self.lengths = [
            int(len(values))
            for values in self.planned_calls
        ]

        self.offsets = np.cumsum(
            [0] + self.lengths,
            dtype=np.int64,
        )

        if len(self.planned_calls) == 0:
            packed_np = np.empty(
                0,
                dtype=np.int64,
            )
        else:
            packed_np = np.concatenate(
                self.planned_calls
            ).astype(
                np.int64,
                copy=False,
            )

        packed_tensor = torch.from_numpy(
            np.array(
                packed_np,
                dtype=np.int64,
                copy=True,
            )
        ).to(device)

        # ONE actual embedding operation per table.
        self.packed_output = self.original_forward(
            packed_tensor
        )

        self.call_index = 0

    def install(self) -> None:
        dispatcher = self

        def packed_forward(self_embedding, input_tensor):
            if dispatcher.call_index >= len(
                dispatcher.planned_calls
            ):
                raise AssertionError(
                    "Packed embedding received more calls than planned."
                )

            expected_length = dispatcher.lengths[
                dispatcher.call_index
            ]

            actual_length = int(
                input_tensor.numel()
            )

            if actual_length != expected_length:
                raise AssertionError(
                    "Packed embedding call-length drift at "
                    f"call {dispatcher.call_index}: "
                    f"{actual_length} != {expected_length}."
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

    def assert_consumed(self) -> None:
        require(
            self.call_index
            == len(self.planned_calls),
            (
                "Packed embedding did not consume all "
                "planned calls."
            ),
        )

    def restore(self) -> None:
        self.embedding.forward = self.original_forward


# =============================================================================
# Logit capture
# =============================================================================

class LogitCapture:
    def __init__(self, model):
        self.model = model
        self.original = model.scoring_mlp.forward
        self.latest = None

    def install(self) -> None:
        capture = self

        def wrapped(self_module, pair_features):
            result = capture.original(pair_features)
            capture.latest = result["logit"]
            return result

        self.model.scoring_mlp.forward = MethodType(
            wrapped,
            self.model.scoring_mlp,
        )

    def cpu_copy(self) -> torch.Tensor:
        require(
            self.latest is not None,
            "No captured logits.",
        )

        return (
            self.latest
            .detach()
            .cpu()
            .clone()
        )

    def clear(self) -> None:
        self.latest = None

    def restore(self) -> None:
        self.model.scoring_mlp.forward = self.original


# =============================================================================
# Numerical-comparison helpers
# =============================================================================

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

    ref64 = ref.to(dtype=torch.float64).reshape(-1)
    cand64 = cand.to(dtype=torch.float64).reshape(-1)
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
        torch.dot(ref64, ref64).item()
    )

    candidate_l2_sq = float(
        torch.dot(cand64, cand64).item()
    )

    diff_l2_sq = float(
        torch.dot(diff, diff).item()
    )

    dot = float(
        torch.dot(ref64, cand64).item()
    )

    exact_count = int(
        torch.eq(ref, cand).sum().item()
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
                torch.sign(ref64[active_mask])
                == torch.sign(cand64[active_mask])
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
        "candidate_l2_sq": candidate_l2_sq,
        "dot": dot,
        "exact_count": exact_count,
        "active_sign_count": active_sign_count,
        "sign_equal_count": sign_equal_count,
    }


def aggregate_stats(rows: list[dict]) -> dict:
    require(
        len(rows) > 0,
        "Cannot aggregate empty tensor rows.",
    )

    numel = int(
        sum(row["numel"] for row in rows)
    )

    diff_l2_sq = float(
        sum(row["diff_l2_sq"] for row in rows)
    )

    reference_l2_sq = float(
        sum(row["reference_l2_sq"] for row in rows)
    )

    candidate_l2_sq = float(
        sum(row["candidate_l2_sq"] for row in rows)
    )

    dot = float(
        sum(row["dot"] for row in rows)
    )

    exact_count = int(
        sum(row["exact_count"] for row in rows)
    )

    active_sign_count = int(
        sum(row["active_sign_count"] for row in rows)
    )

    sign_equal_count = int(
        sum(row["sign_equal_count"] for row in rows)
    )

    eps = 1e-30

    relative_l2_error = (
        math.sqrt(diff_l2_sq)
        / max(
            math.sqrt(reference_l2_sq),
            eps,
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

    weighted_mean_abs = (
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
        "mean_abs_diff_weighted": weighted_mean_abs,
        "relative_l2_error": relative_l2_error,
        "cosine_similarity": cosine_similarity,
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
        ref_grad = reference[name].grad
        cand_grad = candidate[name].grad

        require(
            ref_grad is not None,
            f"Reference missing gradient: {name}",
        )

        require(
            cand_grad is not None,
            f"Candidate missing gradient: {name}",
        )

        rows.append(
            tensor_stats(
                ref_grad,
                cand_grad,
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

    require(
        list(reference.keys())
        == list(candidate.keys()),
        "Parameter-name ordering drift.",
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
            ref_parameter in reference_optimizer.state,
            f"Reference Adam state missing: {name}",
        )

        require(
            cand_parameter in candidate_optimizer.state,
            f"Candidate Adam state missing: {name}",
        )

        ref_state = reference_optimizer.state[
            ref_parameter
        ]

        cand_state = candidate_optimizer.state[
            cand_parameter
        ]

        require(
            float(
                ref_state["step"]
                .detach()
                .cpu()
                .item()
            )
            == float(
                cand_state["step"]
                .detach()
                .cpu()
                .item()
            ),
            f"Adam step mismatch: {name}",
        )

        rows.append(
            tensor_stats(
                ref_state[state_name],
                cand_state[state_name],
            )
        )

    return aggregate_stats(rows)


def policy_rows(
    batch_index: int,
    summary: dict,
    thresholds: dict,
) -> list[dict]:
    checks = [
        (
            "loss_abs_diff",
            summary["loss_abs_diff"],
            "<=",
            thresholds["loss_abs_diff_max"],
        ),
        (
            "logit_max_abs_diff",
            summary["logit_max_abs_diff"],
            "<=",
            thresholds["logit_max_abs_diff_max"],
        ),
        (
            "gradient_relative_l2_error",
            summary["gradient_relative_l2_error"],
            "<=",
            thresholds["gradient_relative_l2_error_max"],
        ),
        (
            "gradient_cosine_similarity",
            summary["gradient_cosine_similarity"],
            ">=",
            thresholds["gradient_cosine_similarity_min"],
        ),
        (
            "gradient_sign_agreement",
            summary["gradient_sign_agreement"],
            ">=",
            thresholds["gradient_sign_agreement_min"],
        ),
        (
            "parameter_relative_l2_error",
            summary["parameter_relative_l2_error"],
            "<=",
            thresholds["parameter_relative_l2_error_max"],
        ),
        (
            "parameter_max_abs_diff",
            summary["parameter_max_abs_diff"],
            "<=",
            thresholds["parameter_max_abs_diff_max"],
        ),
        (
            "adam_exp_avg_relative_l2_error",
            summary["adam_exp_avg_relative_l2_error"],
            "<=",
            thresholds["adam_exp_avg_relative_l2_error_max"],
        ),
        (
            "adam_exp_avg_cosine_similarity",
            summary["adam_exp_avg_cosine_similarity"],
            ">=",
            thresholds["adam_exp_avg_cosine_similarity_min"],
        ),
        (
            "adam_exp_avg_sq_relative_l2_error",
            summary["adam_exp_avg_sq_relative_l2_error"],
            "<=",
            thresholds["adam_exp_avg_sq_relative_l2_error_max"],
        ),
        (
            "adam_exp_avg_sq_cosine_similarity",
            summary["adam_exp_avg_sq_cosine_similarity"],
            ">=",
            thresholds["adam_exp_avg_sq_cosine_similarity_min"],
        ),
    ]

    rows = []

    for metric, actual, comparator, threshold in checks:
        if comparator == "<=":
            passed = actual <= threshold
        else:
            passed = actual >= threshold

        rows.append(
            {
                "batch_index": batch_index,
                "metric": metric,
                "actual": actual,
                "comparator": comparator,
                "threshold": threshold,
                "result": (
                    "PASS"
                    if passed
                    else "FAIL"
                ),
            }
        )

    return rows


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    banner(
        "PHASE 5.4.8b — "
        "PACKED-EMBEDDING APPLE MPS NUMERICAL EQUIVALENCE + FEASIBILITY AUDIT"
    )

    print("Production training launched:         NO")
    print("Production runtime selected:          NO")
    print("Validation cases scored:              0")
    print("Test cases scored:                    0")

    # =========================================================================
    # Gate
    # =========================================================================

    banner("PREREQUISITE + FROZEN POLICY GATE")

    for path in (
        ROUNDTRIP_SOURCE_PATH,
        POLICY_PATH,
        PHASE_5_4_7B_CONTRACT_PATH,
        PHASE_5_4_8A_CONTRACT_PATH,
        PHASE_5_4_5_CONTRACT_PATH,
        PHASE_5_4_2_CONTRACT_PATH,
        PHASE_5_4_AUTHORIZATION_PATH,
    ):
        require(
            path.exists(),
            f"Missing prerequisite: {path}",
        )
        print(f"FOUND  {path}")

    policy = load_json(POLICY_PATH)
    policy_contract = load_json(
        PHASE_5_4_7B_CONTRACT_PATH
    )
    mps_dense_contract = load_json(
        PHASE_5_4_8A_CONTRACT_PATH
    )
    packed_cpu_contract = load_json(
        PHASE_5_4_5_CONTRACT_PATH
    )
    runtime_contract = load_json(
        PHASE_5_4_2_CONTRACT_PATH
    )
    authorization = load_json(
        PHASE_5_4_AUTHORIZATION_PATH
    )

    require(
        policy.get("status") == "FROZEN",
        "Numerical-equivalence policy is not FROZEN.",
    )

    require(
        policy.get("schema_version")
        == EXPECTED_POLICY_VERSION,
        "Numerical-equivalence policy version drift.",
    )

    require(
        policy_contract.get("status") == "FROZEN",
        "Phase-5.4.7b contract is not FROZEN.",
    )

    require(
        mps_dense_contract.get("status") == "COMPLETE",
        "Phase-5.4.8a MPS audit is not COMPLETE.",
    )

    require(
        mps_dense_contract.get(
            "mps_numerically_equivalent"
        ) is True,
        "Canonical-dense MPS did not pass numerical equivalence.",
    )

    require(
        packed_cpu_contract.get("status") == "COMPLETE",
        "Phase-5.4.5 packed audit is not COMPLETE.",
    )

    require(
        authorization.get("training_allowed") is True,
        "Production training authorization is not ALLOWED.",
    )

    canonical_lean_seconds = float(
        runtime_contract[
            "lean_exact_mean_seconds_per_batch"
        ]
    )

    dense_mps_warm_seconds = float(
        mps_dense_contract[
            "mps_batch1_warm_seconds"
        ]
    )

    thresholds = policy["thresholds"]

    print(
        f"Policy version:                       "
        f"{EXPECTED_POLICY_VERSION}"
    )
    print("Policy status:                        FROZEN")
    print(
        f"Canonical lean CPU seconds / batch:   "
        f"{canonical_lean_seconds:.3f}"
    )
    print(
        f"Canonical-dense MPS warm / batch:     "
        f"{dense_mps_warm_seconds:.3f}"
    )

    # =========================================================================
    # MPS availability
    # =========================================================================

    banner("APPLE MPS AVAILABILITY")

    print(f"Platform:                              {platform.platform()}")
    print(f"PyTorch:                               {torch.__version__}")
    print(f"MPS built:                             {torch.backends.mps.is_built()}")
    print(f"MPS available:                         {torch.backends.mps.is_available()}")

    require(
        torch.__version__ == EXPECTED_PYTORCH,
        "Reference PyTorch version drift.",
    )

    require(
        torch.backends.mps.is_available(),
        "MPS became unavailable after successful Phase-5.4.8a.",
    )

    torch.set_num_threads(
        EXPECTED_SELECTED_THREADS
    )

    # =========================================================================
    # Load frozen runtime and derive corrected device executor
    # =========================================================================

    banner(
        "LOAD FROZEN RUNTIME + CORRECT AST PROVENANCE"
    )

    roundtrip = load_module(
        ROUNDTRIP_SOURCE_PATH,
        "_itrs_phase5_4_8b_roundtrip",
    )

    preflight = roundtrip.load_preflight_runtime()
    stream = roundtrip.load_epoch0_stream(preflight)
    shared_cpu = roundtrip.load_shared_inputs(preflight)

    batch0 = roundtrip.decode_batch(stream, 0)
    batch1 = roundtrip.decode_batch(stream, 1)

    (
        lean_execute,
        set_executor_device,
        executor_metadata,
    ) = build_corrected_lean_device_executor(
        roundtrip
    )

    print(
        f"Canonical executor AST SHA256:        "
        f"{executor_metadata['canonical_executor_ast_sha256']}"
    )
    print(
        f"Lean device executor AST SHA256:      "
        f"{executor_metadata['lean_device_executor_ast_sha256']}"
    )
    print(
        f"require statements removed:           "
        f"{executor_metadata['require_statements_removed']}"
    )
    print(
        f"torch.from_numpy rewrites:            "
        f"{executor_metadata['torch_from_numpy_rewrites']}"
    )
    print(
        f"torch.tensor rewrites:                "
        f"{executor_metadata['torch_tensor_rewrites']}"
    )
    print(
        "AST anchor provenance:                "
        "USER V1 PRE-CANDIDATE OBSERVATION"
    )

    require(
        executor_metadata[
            "canonical_executor_ast_sha256"
        ]
        == EXPECTED_CANONICAL_EXECUTOR_AST_SHA256,
        "Canonical executor AST SHA drift.",
    )

    require(
        executor_metadata[
            "lean_device_executor_ast_sha256"
        ]
        == EXPECTED_LEAN_EXECUTOR_AST_SHA256,
        "Lean device executor AST SHA drift.",
    )

    require(
        executor_metadata[
            "canonical_executor_ast_sha256"
        ]
        != executor_metadata[
            "lean_device_executor_ast_sha256"
        ],
        "Corrected AST provenance hashes are not distinct.",
    )

    # =========================================================================
    # Construct states
    # =========================================================================

    banner("CONSTRUCT REFERENCE + PACKED MPS CANDIDATE")

    (
        reference_model,
        reference_optimizer,
        reference_hash_fn,
        *_ref_meta,
    ) = roundtrip.construct_fresh_training_state(
        preflight
    )

    # The canonical helper is exposed from roundtrip.
    require(
        reference_hash_fn(reference_model)
        == EXPECTED_INITIAL_MODEL_SHA256,
        "Reference initial model SHA drift.",
    )

    (
        candidate_model,
        candidate_optimizer_cpu,
        candidate_hash_fn,
        *_cand_meta,
    ) = roundtrip.construct_fresh_training_state(
        preflight
    )

    require(
        candidate_hash_fn(candidate_model)
        == EXPECTED_INITIAL_MODEL_SHA256,
        "Candidate initial model SHA drift before device transfer.",
    )

    del candidate_optimizer_cpu

    mps_device = torch.device("mps")

    candidate_model = candidate_model.to(
        mps_device
    )
    candidate_model.train()

    candidate_optimizer = preflight.build_frozen_adam(
        candidate_model
    )

    require(
        len(candidate_optimizer.state) == 0,
        "Fresh packed-MPS Adam state is not empty.",
    )

    require(
        candidate_model.startup_embedding.sparse is False,
        "Candidate startup embedding unexpectedly sparse.",
    )

    require(
        candidate_model.investor_embedding.sparse is False,
        "Candidate investor embedding unexpectedly sparse.",
    )

    shared_mps = dict(shared_cpu)
    shared_mps["edge_index"] = shared_cpu[
        "edge_index"
    ].to(mps_device)
    shared_mps["edge_type"] = shared_cpu[
        "edge_type"
    ].to(mps_device)

    reference_capture = LogitCapture(
        reference_model
    )
    candidate_capture = LogitCapture(
        candidate_model
    )

    reference_capture.install()
    candidate_capture.install()

    print("Reference initial state:              BYTE-EXACT")
    print("Candidate pre-transfer state:         BYTE-EXACT")
    print("Packed-MPS optimizer state:           EMPTY / FROZEN ADAM")

    # =========================================================================
    # Stateful comparison
    # =========================================================================

    banner(
        "STATEFUL CPU EXACT REFERENCE + PACKED MPS COMPARISON"
    )

    cpu_rows = []
    plan_rows = []
    candidate_rows = []
    policy_rows_all = []

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

        # ---------------------------------------------------------------------
        # CPU reference through corrected lean executor.
        # ---------------------------------------------------------------------

        set_executor_device("cpu")

        ref_start = time.perf_counter()

        reference_result = lean_execute(
            reference_model,
            reference_optimizer,
            lambda model: "LEAN_RUNTIME_HASH_SKIPPED",
            batch,
            shared_cpu,
        )

        ref_seconds = time.perf_counter() - ref_start

        reference_logits = reference_capture.cpu_copy()

        reference_gradient_sha = (
            roundtrip.gradient_logical_sha256(
                reference_model
            )
        )

        reference_model_sha = reference_hash_fn(
            reference_model
        )

        reference_optimizer_sha = (
            roundtrip.optimizer_state_logical_sha256(
                reference_model,
                reference_optimizer,
            )
        )

        reference_logit_sha = roundtrip.tensor_sha256(
            reference_logits
        )

        reference_exact = bool(
            close_float(
                reference_result["loss"],
                expected_loss,
            )
            and reference_logit_sha
            == expected_logit_sha
            and reference_gradient_sha
            == expected_gradient_sha
            and reference_model_sha
            == expected_model_sha
            and reference_optimizer_sha
            == expected_optimizer_sha
        )

        require(
            reference_exact,
            (
                "Corrected AST lean CPU executor failed "
                f"frozen exactness at batch {batch_index}."
            ),
        )

        cpu_rows.append(
            {
                "batch_index": batch_index,
                "lean_cpu_seconds": ref_seconds,
                "loss": float(
                    reference_result["loss"]
                ),
                "logit_sha256": reference_logit_sha,
                "gradient_sha256": reference_gradient_sha,
                "post_model_sha256": reference_model_sha,
                "optimizer_state_sha256": reference_optimizer_sha,
                "exact_frozen": reference_exact,
            }
        )

        print()
        print(
            f"Batch {batch_index} CPU lean bridge:   EXACT"
        )

        # ---------------------------------------------------------------------
        # Build deterministic packed call plan on CPU.
        # ---------------------------------------------------------------------

        plan = build_embedding_call_plan(
            batch,
            shared_cpu,
            num_history_periods=roundtrip.NUM_HISTORY_PERIODS,
            num_investors=roundtrip.NUM_INVESTORS,
        )

        startup_calls = list(
            plan["history_startup_calls"]
        ) + [
            plan["pair_startup_call"]
        ]

        investor_calls = list(
            plan["trend_investor_calls"]
        ) + [
            plan["pair_investor_call"]
        ]

        total_startup_indices = int(
            sum(len(values) for values in startup_calls)
        )

        total_investor_indices = int(
            sum(len(values) for values in investor_calls)
        )

        plan_rows.append(
            {
                "batch_index": batch_index,
                "history_startup_embedding_calls": len(
                    plan["history_startup_calls"]
                ),
                "trend_investor_embedding_calls": len(
                    plan["trend_investor_calls"]
                ),
                "startup_calls_after_pack": 1,
                "investor_calls_after_pack": 1,
                "total_startup_indices_packed": total_startup_indices,
                "total_investor_indices_packed": total_investor_indices,
                "unique_trend_keys": plan["unique_key_count"],
            }
        )

        # ---------------------------------------------------------------------
        # Packed MPS candidate.
        # Timing includes plan->MPS packed lookups, executor, backward, Adam,
        # and explicit device synchronization.
        # ---------------------------------------------------------------------

        set_executor_device("mps")

        torch.mps.synchronize()
        candidate_start = time.perf_counter()

        # Clear previous gradients before creating packed graph nodes.
        candidate_optimizer.zero_grad(
            set_to_none=True
        )

        startup_dispatch = DevicePackedEmbeddingDispatch(
            candidate_model.startup_embedding,
            startup_calls,
            device=mps_device,
        )

        investor_dispatch = DevicePackedEmbeddingDispatch(
            candidate_model.investor_embedding,
            investor_calls,
            device=mps_device,
        )

        startup_dispatch.install()
        investor_dispatch.install()

        try:
            candidate_result = lean_execute(
                candidate_model,
                candidate_optimizer,
                lambda model: "LEAN_RUNTIME_HASH_SKIPPED",
                batch,
                shared_mps,
            )

            startup_dispatch.assert_consumed()
            investor_dispatch.assert_consumed()

            torch.mps.synchronize()

            candidate_seconds = (
                time.perf_counter()
                - candidate_start
            )

        finally:
            startup_dispatch.restore()
            investor_dispatch.restore()

        candidate_logits = candidate_capture.cpu_copy()

        # ---------------------------------------------------------------------
        # Numerical comparison against canonical CPU reference.
        # ---------------------------------------------------------------------

        logit_stats = tensor_stats(
            reference_logits,
            candidate_logits,
        )

        gradient_stats = compare_model_gradients(
            reference_model,
            candidate_model,
        )

        parameter_stats = compare_model_parameters(
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

        logit_relative_l2 = (
            math.sqrt(
                logit_stats["diff_l2_sq"]
            )
            / max(
                math.sqrt(
                    logit_stats[
                        "reference_l2_sq"
                    ]
                ),
                1e-30,
            )
        )

        loss_abs_diff = abs(
            float(candidate_result["loss"])
            - float(reference_result["loss"])
        )

        summary = {
            "batch_index": batch_index,
            "cpu_reference_seconds": ref_seconds,
            "packed_mps_seconds": candidate_seconds,
            "reference_loss": float(
                reference_result["loss"]
            ),
            "candidate_loss": float(
                candidate_result["loss"]
            ),
            "loss_abs_diff": loss_abs_diff,
            "logit_max_abs_diff": logit_stats[
                "max_abs_diff"
            ],
            "logit_mean_abs_diff": logit_stats[
                "mean_abs_diff"
            ],
            "logit_relative_l2_error": logit_relative_l2,
            "gradient_max_abs_diff": gradient_stats[
                "max_abs_diff"
            ],
            "gradient_relative_l2_error": gradient_stats[
                "relative_l2_error"
            ],
            "gradient_cosine_similarity": gradient_stats[
                "cosine_similarity"
            ],
            "gradient_sign_agreement": gradient_stats[
                "sign_agreement"
            ],
            "gradient_exact_fraction": gradient_stats[
                "exact_fraction"
            ],
            "parameter_max_abs_diff": parameter_stats[
                "max_abs_diff"
            ],
            "parameter_relative_l2_error": parameter_stats[
                "relative_l2_error"
            ],
            "parameter_cosine_similarity": parameter_stats[
                "cosine_similarity"
            ],
            "parameter_exact_fraction": parameter_stats[
                "exact_fraction"
            ],
            "adam_exp_avg_relative_l2_error": adam_exp_avg[
                "relative_l2_error"
            ],
            "adam_exp_avg_cosine_similarity": adam_exp_avg[
                "cosine_similarity"
            ],
            "adam_exp_avg_sq_relative_l2_error": adam_exp_avg_sq[
                "relative_l2_error"
            ],
            "adam_exp_avg_sq_cosine_similarity": adam_exp_avg_sq[
                "cosine_similarity"
            ],
        }

        candidate_rows.append(summary)

        this_policy_rows = policy_rows(
            batch_index,
            summary,
            thresholds,
        )

        policy_rows_all.extend(
            this_policy_rows
        )

        batch_policy_pass = all(
            row["result"] == "PASS"
            for row in this_policy_rows
        )

        print(
            f"Batch {batch_index} packed MPS seconds: "
            f"{candidate_seconds:.3f}"
        )
        print(
            f"  startup embedding calls:            "
            f"{len(startup_calls)} -> 1"
        )
        print(
            f"  investor embedding calls:           "
            f"{len(investor_calls)} -> 1"
        )
        print(
            f"  loss abs diff:                      "
            f"{loss_abs_diff:.12e}"
        )
        print(
            f"  logit max abs diff:                 "
            f"{logit_stats['max_abs_diff']:.12e}"
        )
        print(
            f"  gradient relative L2:               "
            f"{gradient_stats['relative_l2_error']:.12e}"
        )
        print(
            f"  gradient cosine:                    "
            f"{gradient_stats['cosine_similarity']:.12f}"
        )
        print(
            f"  gradient sign agreement:            "
            f"{gradient_stats['sign_agreement']:.12f}"
        )
        print(
            f"  parameter relative L2:              "
            f"{parameter_stats['relative_l2_error']:.12e}"
        )
        print(
            f"  parameter max abs diff:             "
            f"{parameter_stats['max_abs_diff']:.12e}"
        )
        print(
            f"  frozen policy:                      "
            f"{'PASS' if batch_policy_pass else 'FAIL'}"
        )

        reference_capture.clear()
        candidate_capture.clear()

    # =========================================================================
    # Consolidate
    # =========================================================================

    banner(
        "PACKED MPS POLICY DECISION + RUNTIME PROJECTION"
    )

    cpu_df = pd.DataFrame(cpu_rows)
    plan_df = pd.DataFrame(plan_rows)
    candidate_df = pd.DataFrame(candidate_rows)
    policy_df = pd.DataFrame(policy_rows_all)

    candidate_policy_pass = bool(
        (policy_df["result"] == "PASS").all()
    )

    mean_seconds = float(
        candidate_df[
            "packed_mps_seconds"
        ].mean()
    )

    batch1_warm_seconds = float(
        candidate_df.loc[
            candidate_df["batch_index"] == 1,
            "packed_mps_seconds",
        ].iloc[0]
    )

    speedup_vs_cpu = (
        canonical_lean_seconds
        / mean_seconds
    )

    speedup_vs_dense_mps_warm = (
        dense_mps_warm_seconds
        / batch1_warm_seconds
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
        batch1_warm_seconds
        * TOTAL_OPTIMIZER_STEPS
    )

    print(
        "Packed MPS numerical equivalence:     "
        + (
            "PASS"
            if candidate_policy_pass
            else "FAIL"
        )
    )
    print(
        f"Packed MPS mean seconds / batch:      "
        f"{mean_seconds:.3f}"
    )
    print(
        f"Packed MPS batch-1 warm seconds:      "
        f"{batch1_warm_seconds:.3f}"
    )
    print(
        f"Speedup vs canonical lean CPU:        "
        f"{speedup_vs_cpu:.2f}x"
    )
    print(
        f"Speedup vs dense-MPS warm:             "
        f"{speedup_vs_dense_mps_warm:.2f}x"
    )
    print(
        "Projected 20 epochs (mean):           "
        f"{human_duration(projected_full_seconds)}"
    )
    print(
        "Projected 20 epochs (batch1 warm):    "
        f"{human_duration(warm_projected_full_seconds)}"
    )

    runtime_df = pd.DataFrame(
        [
            {
                "metric": (
                    "canonical_lean_cpu_seconds_per_batch"
                ),
                "value": canonical_lean_seconds,
                "human": f"{canonical_lean_seconds:.3f} s",
            },
            {
                "metric": (
                    "canonical_dense_mps_batch1_warm_seconds"
                ),
                "value": dense_mps_warm_seconds,
                "human": f"{dense_mps_warm_seconds:.3f} s",
            },
            {
                "metric": (
                    "packed_mps_two_batch_mean_seconds"
                ),
                "value": mean_seconds,
                "human": f"{mean_seconds:.3f} s",
            },
            {
                "metric": (
                    "packed_mps_batch1_warm_seconds"
                ),
                "value": batch1_warm_seconds,
                "human": f"{batch1_warm_seconds:.3f} s",
            },
            {
                "metric": (
                    "speedup_vs_canonical_lean_cpu"
                ),
                "value": speedup_vs_cpu,
                "human": f"{speedup_vs_cpu:.2f}x",
            },
            {
                "metric": (
                    "speedup_vs_dense_mps_warm"
                ),
                "value": speedup_vs_dense_mps_warm,
                "human": f"{speedup_vs_dense_mps_warm:.2f}x",
            },
            {
                "metric": (
                    "projected_20_epochs_two_batch_mean"
                ),
                "value": projected_full_seconds,
                "human": human_duration(
                    projected_full_seconds
                ),
            },
            {
                "metric": (
                    "projected_20_epochs_batch1_warm"
                ),
                "value": warm_projected_full_seconds,
                "human": human_duration(
                    warm_projected_full_seconds
                ),
            },
        ]
    )

    # =========================================================================
    # Final invariants
    # =========================================================================

    banner("FINAL PHASE-5.4.8b INVARIANTS")

    checks = [
        (
            "frozen_policy_reused_unchanged",
            policy["status"] == "FROZEN",
        ),
        (
            "v1_observed_local_canonical_ast_sha_exact",
            executor_metadata[
                "canonical_executor_ast_sha256"
            ]
            == EXPECTED_CANONICAL_EXECUTOR_AST_SHA256,
        ),
        (
            "v1_observed_local_lean_ast_sha_exact",
            executor_metadata[
                "lean_device_executor_ast_sha256"
            ]
            == EXPECTED_LEAN_EXECUTOR_AST_SHA256,
        ),
        (
            "corrected_ast_hashes_distinct",
            executor_metadata[
                "canonical_executor_ast_sha256"
            ]
            != executor_metadata[
                "lean_device_executor_ast_sha256"
            ],
        ),
        (
            "cpu_bridge_batch0_byte_exact",
            bool(
                cpu_df.loc[
                    cpu_df["batch_index"] == 0,
                    "exact_frozen",
                ].iloc[0]
            ),
        ),
        (
            "cpu_bridge_batch1_byte_exact",
            bool(
                cpu_df.loc[
                    cpu_df["batch_index"] == 1,
                    "exact_frozen",
                ].iloc[0]
            ),
        ),
        (
            "packed_mps_two_batches_executed",
            set(
                candidate_df["batch_index"].tolist()
            )
            == {0, 1},
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
                    plan_df[
                        "startup_calls_after_pack"
                    ]
                    == 1
                ).all()
                and (
                    plan_df[
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
                    candidate_df[
                        "packed_mps_seconds"
                    ].to_numpy(
                        dtype=np.float64
                    )
                ).all()
                and (
                    candidate_df[
                        "packed_mps_seconds"
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
                invariant_df["result"]
                == "PASS"
            ).all()
        ),
        (
            "At least one Phase-5.4.8b "
            "invariant failed."
        ),
    )

    print(
        invariant_df.to_string(index=False)
    )

    # =========================================================================
    # Write outputs
    # =========================================================================

    banner("WRITE PHASE-5.4.8b OUTPUTS")

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTRACT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cpu_df.to_csv(
        CPU_BRIDGE_PATH,
        index=False,
    )

    plan_df.to_csv(
        PACK_PLAN_PATH,
        index=False,
    )

    candidate_df.to_csv(
        MPS_BATCH_PATH,
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
        "phase": "5.4.8b",
        "script_version": "V2",
        "title": (
            "Packed-Embedding Apple MPS Numerical Equivalence + Feasibility Audit"
        ),
        "status": "COMPLETE",
        "classification": (
            "IMPLEMENTATION_EQUIVALENT_DEVICE_ACCELERATION_AUDIT"
        ),
        "policy_version": EXPECTED_POLICY_VERSION,
        "policy_status": "FROZEN_BEFORE_CANDIDATE",
        "candidate_runtime": CANDIDATE_RUNTIME,
        "corrected_ast_provenance": executor_metadata,
        "ast_anchor_source": (
            "PHASE_5_4_8B_V1_USER_LOCAL_PRE_CANDIDATE_OBSERVATION"
        ),
        "ast_anchor_observed_before_candidate_execution": True,
        "v1_block_reason": (
            "Foreign hard-coded AST fingerprints from a different local copy "
            "of phase_5_3_2b_checkpoint_resume_roundtrip_proof.py."
        ),
        "phase_5_4_8a_ast_hash_bookkeeping_bug": (
            "V1 hashed the same mutated AST object for both canonical and lean; "
            "5.4.8b corrects provenance only. CPU exactness from 5.4.8a remains valid."
        ),
        "cpu_bridge_two_step_byte_exact": True,
        "candidate_numerically_equivalent": (
            candidate_policy_pass
        ),
        "packed_mps_mean_seconds_per_batch": (
            mean_seconds
        ),
        "packed_mps_batch1_warm_seconds": (
            batch1_warm_seconds
        ),
        "speedup_vs_canonical_lean_cpu": (
            speedup_vs_cpu
        ),
        "speedup_vs_dense_mps_warm": (
            speedup_vs_dense_mps_warm
        ),
        "projected_20_epochs_seconds_two_batch_mean": (
            projected_full_seconds
        ),
        "projected_20_epochs_seconds_batch1_warm": (
            warm_projected_full_seconds
        ),
        "production_runtime_selected": False,
        "production_training_launched": False,
        "production_checkpoint_written": False,
        "validation_cases_scored": 0,
        "test_cases_scored": 0,
        "next_phase": (
            "5.4.8c_EXTENDED_PACKED_MPS_STABILITY_BENCHMARK"
            if candidate_policy_pass
            and batch1_warm_seconds < canonical_lean_seconds
            else "5.4.8c_ALTERNATIVE_RUNTIME_OR_CPU_SPARSE_DECISION"
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
        "phase": "5.4.8b",
        "script_version": "V2",
        "status": (
            "PACKED_MPS_NUMERICAL_EQUIVALENCE_PASS"
            if candidate_policy_pass
            else "PACKED_MPS_NUMERICAL_EQUIVALENCE_FAIL"
        ),
        "candidate_runtime": CANDIDATE_RUNTIME,
        "mean_seconds_per_batch": mean_seconds,
        "batch1_warm_seconds": batch1_warm_seconds,
        "speedup_vs_canonical_lean_cpu": speedup_vs_cpu,
        "production_training_steps": 0,
        "validation_cases_scored": 0,
        "test_cases_scored": 0,
        "contract": str(CONTRACT_PATH),
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
        CPU_BRIDGE_PATH,
        PACK_PLAN_PATH,
        MPS_BATCH_PATH,
        POLICY_AUDIT_PATH,
        RUNTIME_PATH,
        FINAL_INVARIANT_PATH,
        CONTRACT_PATH,
        MANIFEST_PATH,
    ):
        print(f"WROTE  {path}")

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
    del shared_cpu
    del shared_mps

    torch.mps.empty_cache()
    gc.collect()

    banner("PHASE 5.4.8b FINAL STATUS")

    print("Corrected AST provenance:             PASS")
    print("CPU lean bridge:                      BYTE-EXACT / TWO BATCHES")
    print(
        "Packed MPS numerical equivalence:     "
        + (
            "PASS"
            if candidate_policy_pass
            else "FAIL"
        )
    )
    print(
        f"Packed MPS mean seconds / batch:      "
        f"{mean_seconds:.3f}"
    )
    print(
        f"Packed MPS batch-1 warm seconds:      "
        f"{batch1_warm_seconds:.3f}"
    )
    print(
        f"Speedup vs canonical lean CPU:        "
        f"{speedup_vs_cpu:.2f}x"
    )
    print(
        "Projected 20 epochs (mean):           "
        f"{human_duration(projected_full_seconds)}"
    )
    print(
        "Projected 20 epochs (warm batch1):    "
        f"{human_duration(warm_projected_full_seconds)}"
    )
    print()
    print("Production runtime selected:          NO")
    print("Production training launched:         NO")
    print("Validation cases scored:              0")
    print("Test cases scored:                    0")

    banner(
        "PHASE 5.4.8b COMPLETE / "
        "PACKED-EMBEDDING APPLE MPS AUDIT CLOSED"
    )


if __name__ == "__main__":
    main()