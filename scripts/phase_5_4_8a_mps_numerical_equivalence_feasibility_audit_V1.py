#!/usr/bin/env python3
"""
Phase 5.4.8a — Canonical-Dense Apple MPS Numerical Equivalence + Feasibility Audit

Purpose
-------
Test Apple MPS as an accelerated device runtime against the frozen policy:

    ITRS_PHASE5_NUMERICAL_EQUIVALENCE_V1

The candidate keeps the CANONICAL DENSE embedding call structure. It changes
only execution device and removes audit-only runtime assertions/hashes from the
timed numerical path.

Provenance-preserving device bridge
-----------------------------------
Rather than hand-rewriting the frozen batch implementation, this script derives
a lean device-aware executor from the exact frozen `execute_training_batch`
source via AST transformation.

Allowed transformations:
    1. Replace statement-level require(...) audit guards with pass.
       Rationale: on MPS their scalar tensor checks force thousands of device
       synchronizations; they are integrity instrumentation, not model math.

    2. Replace torch.from_numpy(...) with _device_from_numpy(...).

    3. Replace torch.tensor(...) with _device_tensor(...).

    4. Substitute proof-only hash/RNG helpers in the derived function namespace
       with no-op placeholders.

Unchanged:
    - model architecture
    - all forward numerical operations
    - BCEWithLogitsLoss
    - loss.backward()
    - Adam optimizer.step()
    - batch examples/order
    - dense embedding semantics
    - graph
    - trend history
    - parameter shapes
    - optimizer hyperparameters

Critical gate
-------------
BEFORE MPS is evaluated, the AST-derived lean executor must reproduce the
already-frozen CPU batch0 -> batch1 trajectory byte-exactly.

MPS is then judged using the already-frozen numerical-equivalence policy.
No tolerance is changed here.

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
    "phase_5_4_8a"
)

CPU_BRIDGE_PROOF_PATH = (
    AUDIT_DIR
    / "ast_device_bridge_cpu_exactness_proof.csv"
)

MPS_BATCH_SUMMARY_PATH = (
    AUDIT_DIR
    / "mps_numerical_equivalence_batch_summary.csv"
)

MPS_POLICY_AUDIT_PATH = (
    AUDIT_DIR
    / "mps_numerical_equivalence_policy_audit.csv"
)

MPS_RUNTIME_PATH = (
    AUDIT_DIR
    / "mps_runtime_feasibility_projection.csv"
)

FINAL_INVARIANT_PATH = (
    AUDIT_DIR
    / "phase_5_4_8a_final_invariants.csv"
)

CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_4_8a_mps_numerical_equivalence_feasibility_contract.json"
)

MANIFEST_PATH = (
    AUDIT_DIR
    / "phase_5_4_8a_mps_numerical_equivalence_feasibility_manifest.json"
)


# =============================================================================
# Frozen anchors
# =============================================================================

EXPECTED_PYTORCH = "2.7.0"
EXPECTED_SELECTED_THREADS = 8
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
# AST-derived lean device executor
# =============================================================================

class LeanDeviceExecutorTransformer(ast.NodeTransformer):
    """
    Minimal transformation of the frozen executor.

    - statement-level require(...) -> pass
    - torch.from_numpy(...) -> _device_from_numpy(...)
    - torch.tensor(...) -> _device_tensor(...)
    """

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
    def _is_torch_attribute_call(
        node,
        attribute: str,
    ) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
            and node.func.attr == attribute
        )

    def visit_Expr(self, node):
        if self._is_name_call(
            node.value,
            "require",
        ):
            self.require_statements_removed += 1
            return ast.copy_location(
                ast.Pass(),
                node,
            )

        return self.generic_visit(node)

    def visit_Call(self, node):
        node = self.generic_visit(node)

        if self._is_torch_attribute_call(
            node,
            "from_numpy",
        ):
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

        if self._is_torch_attribute_call(
            node,
            "tensor",
        ):
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


def build_lean_device_executor(
    roundtrip,
):
    canonical_source = textwrap.dedent(
        inspect.getsource(
            roundtrip.execute_training_batch
        )
    )

    canonical_tree = ast.parse(
        canonical_source
    )

    transformer = (
        LeanDeviceExecutorTransformer()
    )

    transformed_tree = transformer.visit(
        canonical_tree
    )

    transformed_tree = ast.fix_missing_locations(
        transformed_tree
    )

    canonical_ast_sha = hashlib.sha256(
        ast.dump(
            canonical_tree,
            include_attributes=False,
        ).encode(
            "utf-8"
        )
    ).hexdigest()

    transformed_ast_sha = hashlib.sha256(
        ast.dump(
            transformed_tree,
            include_attributes=False,
        ).encode(
            "utf-8"
        )
    ).hexdigest()

    runtime_context = {
        "device": torch.device(
            "cpu"
        )
    }

    original_from_numpy = torch.from_numpy
    original_tensor = torch.tensor

    def _device_from_numpy(array):
        return original_from_numpy(
            array
        ).to(
            runtime_context[
                "device"
            ]
        )

    def _device_tensor(
        *args,
        **kwargs,
    ):
        if "device" not in kwargs:
            kwargs[
                "device"
            ] = runtime_context[
                "device"
            ]

        return original_tensor(
            *args,
            **kwargs,
        )

    namespace = dict(
        roundtrip.__dict__
    )

    namespace[
        "_device_from_numpy"
    ] = _device_from_numpy

    namespace[
        "_device_tensor"
    ] = _device_tensor

    # Proof-only operations inside the canonical executor are deliberately
    # replaced by constants in this lean numerical runtime.
    namespace[
        "tensor_sha256"
    ] = (
        lambda tensor: "LEAN_RUNTIME_HASH_SKIPPED"
    )

    namespace[
        "gradient_logical_sha256"
    ] = (
        lambda model: "LEAN_RUNTIME_HASH_SKIPPED"
    )

    namespace[
        "optimizer_state_logical_sha256"
    ] = (
        lambda model, optimizer: "LEAN_RUNTIME_HASH_SKIPPED"
    )

    namespace[
        "rng_snapshot"
    ] = (
        lambda: None
    )

    namespace[
        "rng_equal"
    ] = (
        lambda before, after: True
    )

    exec(
        compile(
            transformed_tree,
            filename=(
                "<phase_5_4_8a_lean_device_executor>"
            ),
            mode="exec",
        ),
        namespace,
    )

    executor = namespace[
        "execute_training_batch"
    ]

    def set_device(device):
        runtime_context[
            "device"
        ] = torch.device(
            device
        )

    metadata = {
        "canonical_executor_ast_sha256": (
            canonical_ast_sha
        ),
        "lean_device_executor_ast_sha256": (
            transformed_ast_sha
        ),
        "require_statements_removed": (
            transformer
            .require_statements_removed
        ),
        "torch_from_numpy_rewrites": (
            transformer
            .from_numpy_rewrites
        ),
        "torch_tensor_rewrites": (
            transformer
            .tensor_rewrites
        ),
    }

    return (
        executor,
        set_device,
        metadata,
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

            capture.latest = result[
                "logit"
            ]

            return result

        self.model.scoring_mlp.forward = (
            MethodType(
                wrapped,
                self.model.scoring_mlp,
            )
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
        self.model.scoring_mlp.forward = (
            self.original
        )


# =============================================================================
# Numerical-comparison helpers
# =============================================================================

def _to_dense_cpu(
    tensor: torch.Tensor,
) -> torch.Tensor:
    value = (
        tensor.detach().cpu()
    )

    if value.is_sparse:
        value = value.to_dense()

    return value


def tensor_stats(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict:
    ref = _to_dense_cpu(
        reference
    )

    cand = _to_dense_cpu(
        candidate
    )

    require(
        tuple(
            ref.shape
        )
        == tuple(
            cand.shape
        ),
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

    n = int(
        ref64.numel()
    )

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

    candidate_l2_sq = float(
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
        "diff_l2_sq": (
            diff_l2_sq
        ),
        "reference_l2_sq": (
            ref_l2_sq
        ),
        "candidate_l2_sq": (
            candidate_l2_sq
        ),
        "dot": dot,
        "exact_count": (
            exact_count
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
        "Cannot aggregate empty rows.",
    )

    numel = int(
        sum(
            row[
                "numel"
            ]
            for row in rows
        )
    )

    diff_l2_sq = float(
        sum(
            row[
                "diff_l2_sq"
            ]
            for row in rows
        )
    )

    reference_l2_sq = float(
        sum(
            row[
                "reference_l2_sq"
            ]
            for row in rows
        )
    )

    candidate_l2_sq = float(
        sum(
            row[
                "candidate_l2_sq"
            ]
            for row in rows
        )
    )

    dot = float(
        sum(
            row[
                "dot"
            ]
            for row in rows
        )
    )

    exact_count = int(
        sum(
            row[
                "exact_count"
            ]
            for row in rows
        )
    )

    active_sign_count = int(
        sum(
            row[
                "active_sign_count"
            ]
            for row in rows
        )
    )

    sign_equal_count = int(
        sum(
            row[
                "sign_equal_count"
            ]
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
                reference_l2_sq
            ),
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
                math.sqrt(
                    reference_l2_sq
                )
                * math.sqrt(
                    candidate_l2_sq
                )
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
            row[
                "mean_abs_diff"
            ]
            * row[
                "numel"
            ]
            for row in rows
        )
        / max(
            numel,
            1,
        )
    )

    return {
        "numel": (
            numel
        ),
        "max_abs_diff": max(
            row[
                "max_abs_diff"
            ]
            for row in rows
        ),
        "mean_abs_diff_weighted": (
            weighted_mean_abs
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
        list(
            reference.keys()
        )
        == list(
            candidate.keys()
        ),
        "Parameter-name ordering drift.",
    )

    rows = []

    for name in reference:
        ref_grad = reference[
            name
        ].grad

        cand_grad = candidate[
            name
        ].grad

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

    return aggregate_stats(
        rows
    )


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
        list(
            reference.keys()
        )
        == list(
            candidate.keys()
        ),
        "Parameter-name ordering drift.",
    )

    rows = []

    for name in reference:
        rows.append(
            tensor_stats(
                reference[
                    name
                ],
                candidate[
                    name
                ],
            )
        )

    return aggregate_stats(
        rows
    )


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
        ref_parameter = reference[
            name
        ]

        cand_parameter = candidate[
            name
        ]

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

        ref_state = reference_optimizer.state[
            ref_parameter
        ]

        cand_state = candidate_optimizer.state[
            cand_parameter
        ]

        require(
            float(
                ref_state[
                    "step"
                ].detach().cpu().item()
            )
            == float(
                cand_state[
                    "step"
                ].detach().cpu().item()
            ),
            f"Adam step mismatch: {name}",
        )

        rows.append(
            tensor_stats(
                ref_state[
                    state_name
                ],
                cand_state[
                    state_name
                ],
            )
        )

    return aggregate_stats(
        rows
    )


def policy_rows(
    batch_index: int,
    summary: dict,
    thresholds: dict,
) -> list[dict]:
    checks = [
        (
            "loss_abs_diff",
            summary[
                "loss_abs_diff"
            ],
            "<=",
            thresholds[
                "loss_abs_diff_max"
            ],
        ),
        (
            "logit_max_abs_diff",
            summary[
                "logit_max_abs_diff"
            ],
            "<=",
            thresholds[
                "logit_max_abs_diff_max"
            ],
        ),
        (
            "gradient_relative_l2_error",
            summary[
                "gradient_relative_l2_error"
            ],
            "<=",
            thresholds[
                "gradient_relative_l2_error_max"
            ],
        ),
        (
            "gradient_cosine_similarity",
            summary[
                "gradient_cosine_similarity"
            ],
            ">=",
            thresholds[
                "gradient_cosine_similarity_min"
            ],
        ),
        (
            "gradient_sign_agreement",
            summary[
                "gradient_sign_agreement"
            ],
            ">=",
            thresholds[
                "gradient_sign_agreement_min"
            ],
        ),
        (
            "parameter_relative_l2_error",
            summary[
                "parameter_relative_l2_error"
            ],
            "<=",
            thresholds[
                "parameter_relative_l2_error_max"
            ],
        ),
        (
            "parameter_max_abs_diff",
            summary[
                "parameter_max_abs_diff"
            ],
            "<=",
            thresholds[
                "parameter_max_abs_diff_max"
            ],
        ),
        (
            "adam_exp_avg_relative_l2_error",
            summary[
                "adam_exp_avg_relative_l2_error"
            ],
            "<=",
            thresholds[
                "adam_exp_avg_relative_l2_error_max"
            ],
        ),
        (
            "adam_exp_avg_cosine_similarity",
            summary[
                "adam_exp_avg_cosine_similarity"
            ],
            ">=",
            thresholds[
                "adam_exp_avg_cosine_similarity_min"
            ],
        ),
        (
            "adam_exp_avg_sq_relative_l2_error",
            summary[
                "adam_exp_avg_sq_relative_l2_error"
            ],
            "<=",
            thresholds[
                "adam_exp_avg_sq_relative_l2_error_max"
            ],
        ),
        (
            "adam_exp_avg_sq_cosine_similarity",
            summary[
                "adam_exp_avg_sq_cosine_similarity"
            ],
            ">=",
            thresholds[
                "adam_exp_avg_sq_cosine_similarity_min"
            ],
        ),
    ]

    rows = []

    for (
        metric,
        actual,
        comparator,
        threshold,
    ) in checks:
        if comparator == "<=":
            passed = (
                actual
                <= threshold
            )
        else:
            passed = (
                actual
                >= threshold
            )

        rows.append(
            {
                "batch_index": (
                    batch_index
                ),
                "metric": (
                    metric
                ),
                "actual": (
                    actual
                ),
                "comparator": (
                    comparator
                ),
                "threshold": (
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
# Output helper for unavailable/unsupported MPS
# =============================================================================

def write_unavailable_outputs(
    *,
    status: str,
    reason: str,
    executor_metadata: dict | None,
) -> None:
    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTRACT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        [
            {
                "check": (
                    "mps_runtime_available"
                ),
                "result": (
                    "FAIL"
                ),
                "reason": (
                    reason
                ),
            }
        ]
    ).to_csv(
        FINAL_INVARIANT_PATH,
        index=False,
    )

    contract = {
        "phase": (
            "5.4.8a"
        ),
        "title": (
            "Canonical-Dense Apple MPS Numerical Equivalence + Feasibility Audit"
        ),
        "status": (
            status
        ),
        "mps_available": (
            bool(
                torch.backends.mps.is_available()
            )
        ),
        "reason": (
            reason
        ),
        "lean_executor_metadata": (
            executor_metadata
        ),
        "production_training_launched": (
            False
        ),
        "production_runtime_selected": (
            False
        ),
        "validation_cases_scored": (
            0
        ),
        "test_cases_scored": (
            0
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

    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "phase": "5.4.8a",
                "status": status,
                "reason": reason,
                "production_training_steps": 0,
                "validation_cases_scored": 0,
                "test_cases_scored": 0,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(
        f"WROTE  {FINAL_INVARIANT_PATH}"
    )
    print(
        f"WROTE  {CONTRACT_PATH}"
    )
    print(
        f"WROTE  {MANIFEST_PATH}"
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.4.8a — "
        "CANONICAL-DENSE APPLE MPS NUMERICAL EQUIVALENCE + FEASIBILITY AUDIT"
    )

    print(
        "Production training launched:         NO"
    )
    print(
        "Production runtime selected:          NO"
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
        policy.get(
            "policy_frozen_before_device_benchmark"
        )
        is True,
        "Policy was not frozen before device benchmark.",
    )

    require(
        policy_contract.get(
            "status"
        )
        == "FROZEN",
        "Phase-5.4.7b contract is not FROZEN.",
    )

    require(
        authorization.get(
            "training_allowed"
        )
        is True,
        "Production training authorization is not ALLOWED.",
    )

    canonical_lean_seconds = float(
        runtime_contract[
            "lean_exact_mean_seconds_per_batch"
        ]
    )

    thresholds = policy[
        "thresholds"
    ]

    print(
        f"Policy version:                       "
        f"{EXPECTED_POLICY_VERSION}"
    )
    print(
        "Policy status:                        FROZEN"
    )
    print(
        f"Canonical lean CPU seconds / batch:   "
        f"{canonical_lean_seconds:.3f}"
    )

    # =========================================================================
    # MPS availability
    # =========================================================================

    banner(
        "APPLE MPS AVAILABILITY"
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
        f"MPS built:                             "
        f"{torch.backends.mps.is_built()}"
    )
    print(
        f"MPS available:                         "
        f"{torch.backends.mps.is_available()}"
    )

    require(
        torch.__version__
        == EXPECTED_PYTORCH,
        "Reference PyTorch version drift.",
    )

    if not torch.backends.mps.is_available():
        write_unavailable_outputs(
            status="MPS_UNAVAILABLE",
            reason=(
                "torch.backends.mps.is_available() == False"
            ),
            executor_metadata=None,
        )

        banner(
            "PHASE 5.4.8a COMPLETE / MPS UNAVAILABLE"
        )

        return

    torch.set_num_threads(
        EXPECTED_SELECTED_THREADS
    )

    # =========================================================================
    # Load runtime
    # =========================================================================

    banner(
        "LOAD FROZEN RUNTIME + DERIVE LEAN DEVICE EXECUTOR"
    )

    roundtrip = load_module(
        ROUNDTRIP_SOURCE_PATH,
        "_itrs_phase5_4_8a_roundtrip",
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

    shared_cpu = (
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
        lean_execute,
        set_executor_device,
        executor_metadata,
    ) = build_lean_device_executor(
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

    require(
        executor_metadata[
            "require_statements_removed"
        ]
        > 0,
        "Lean executor removed no require guards.",
    )

    require(
        executor_metadata[
            "torch_from_numpy_rewrites"
        ]
        > 0,
        "Lean executor rewrote no torch.from_numpy calls.",
    )

    # =========================================================================
    # Construct CPU reference and MPS candidate
    # =========================================================================

    banner(
        "CONSTRUCT BYTE-IDENTICAL INITIAL STATES"
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
        candidate_optimizer_cpu,
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
        "Candidate initial model SHA drift before device transfer.",
    )

    # Candidate optimizer was built on CPU parameters. Build a fresh frozen Adam
    # after moving the model to MPS, so optimizer references are guaranteed to
    # target the MPS Parameter objects.
    del candidate_optimizer_cpu

    mps_device = torch.device(
        "mps"
    )

    candidate_model = candidate_model.to(
        mps_device
    )

    candidate_model.train()

    candidate_optimizer = (
        preflight
        .build_frozen_adam(
            candidate_model
        )
    )

    require(
        len(
            candidate_optimizer.state
        )
        == 0,
        "Fresh MPS Adam state is not empty.",
    )

    # Keep canonical dense embedding behavior.
    require(
        bool(
            candidate_model.startup_embedding.sparse
        )
        is False,
        "MPS startup embedding unexpectedly sparse.",
    )

    require(
        bool(
            candidate_model.investor_embedding.sparse
        )
        is False,
        "MPS investor embedding unexpectedly sparse.",
    )

    shared_mps = dict(
        shared_cpu
    )

    shared_mps[
        "edge_index"
    ] = (
        shared_cpu[
            "edge_index"
        ]
        .to(
            mps_device
        )
    )

    shared_mps[
        "edge_type"
    ] = (
        shared_cpu[
            "edge_type"
        ]
        .to(
            mps_device
        )
    )

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
        "Candidate pre-transfer state:         BYTE-EXACT"
    )
    print(
        "MPS optimizer state:                  EMPTY / FROZEN ADAM"
    )

    # =========================================================================
    # CPU exact bridge + MPS comparison
    # =========================================================================

    banner(
        "STATEFUL CPU-BRIDGE PROOF + MPS NUMERICAL COMPARISON"
    )

    cpu_proof_rows = []
    mps_batch_rows = []
    policy_audit_rows = []

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

    mps_runtime_error = None

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
        # Exact CPU reference through AST-derived lean executor.
        # ---------------------------------------------------------------------

        set_executor_device(
            "cpu"
        )

        ref_start = time.perf_counter()

        reference_result = lean_execute(
            reference_model,
            reference_optimizer,
            (
                lambda model: (
                    "LEAN_RUNTIME_HASH_SKIPPED"
                )
            ),
            batch,
            shared_cpu,
        )

        ref_seconds = (
            time.perf_counter()
            - ref_start
        )

        reference_logits = (
            reference_capture.cpu_copy()
        )

        reference_gradient_sha = (
            roundtrip
            .gradient_logical_sha256(
                reference_model
            )
        )

        reference_model_sha = (
            reference_hash_fn(
                reference_model
            )
        )

        reference_optimizer_sha = (
            roundtrip
            .optimizer_state_logical_sha256(
                reference_model,
                reference_optimizer,
            )
        )

        reference_exact = bool(
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
            <= 5e-10
            and roundtrip.tensor_sha256(
                reference_logits
            )
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
                "AST-derived lean executor failed "
                f"canonical CPU exactness at batch {batch_index}."
            ),
        )

        cpu_proof_rows.append(
            {
                "batch_index": (
                    batch_index
                ),
                "lean_cpu_seconds": (
                    ref_seconds
                ),
                "loss": (
                    float(
                        reference_result[
                            "loss"
                        ]
                    )
                ),
                "logit_sha256": (
                    roundtrip.tensor_sha256(
                        reference_logits
                    )
                ),
                "gradient_sha256": (
                    reference_gradient_sha
                ),
                "post_model_sha256": (
                    reference_model_sha
                ),
                "optimizer_state_sha256": (
                    reference_optimizer_sha
                ),
                "exact_frozen": (
                    reference_exact
                ),
            }
        )

        print()
        print(
            f"Batch {batch_index} CPU lean bridge:   EXACT"
        )

        # ---------------------------------------------------------------------
        # MPS candidate.
        # ---------------------------------------------------------------------

        set_executor_device(
            "mps"
        )

        try:
            torch.mps.synchronize()

            candidate_start = (
                time.perf_counter()
            )

            candidate_result = lean_execute(
                candidate_model,
                candidate_optimizer,
                (
                    lambda model: (
                        "LEAN_RUNTIME_HASH_SKIPPED"
                    )
                ),
                batch,
                shared_mps,
            )

            torch.mps.synchronize()

            candidate_seconds = (
                time.perf_counter()
                - candidate_start
            )

        except Exception as exc:
            mps_runtime_error = (
                f"{type(exc).__name__}: {exc}"
            )

            print(
                f"Batch {batch_index} MPS ERROR: "
                f"{mps_runtime_error}"
            )

            break

        candidate_logits = (
            candidate_capture.cpu_copy()
        )

        # ---------------------------------------------------------------------
        # Numerical comparison.
        # ---------------------------------------------------------------------

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

        summary = {
            "batch_index": (
                batch_index
            ),
            "cpu_reference_seconds": (
                ref_seconds
            ),
            "mps_candidate_seconds": (
                candidate_seconds
            ),
            "reference_loss": (
                float(
                    reference_result[
                        "loss"
                    ]
                )
            ),
            "candidate_loss": (
                float(
                    candidate_result[
                        "loss"
                    ]
                )
            ),
            "loss_abs_diff": (
                loss_abs_diff
            ),
            "logit_max_abs_diff": (
                logit[
                    "max_abs_diff"
                ]
            ),
            "logit_mean_abs_diff": (
                logit[
                    "mean_abs_diff"
                ]
            ),
            "logit_relative_l2_error": (
                (
                    math.sqrt(
                        logit[
                            "diff_l2_sq"
                        ]
                    )
                    / max(
                        math.sqrt(
                            logit[
                                "reference_l2_sq"
                            ]
                        ),
                        1e-30,
                    )
                )
            ),
            "gradient_max_abs_diff": (
                gradient[
                    "max_abs_diff"
                ]
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
            "gradient_exact_fraction": (
                gradient[
                    "exact_fraction"
                ]
            ),
            "parameter_max_abs_diff": (
                parameter[
                    "max_abs_diff"
                ]
            ),
            "parameter_relative_l2_error": (
                parameter[
                    "relative_l2_error"
                ]
            ),
            "parameter_cosine_similarity": (
                parameter[
                    "cosine_similarity"
                ]
            ),
            "parameter_exact_fraction": (
                parameter[
                    "exact_fraction"
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

        mps_batch_rows.append(
            summary
        )

        batch_policy_rows = policy_rows(
            batch_index,
            summary,
            thresholds,
        )

        policy_audit_rows.extend(
            batch_policy_rows
        )

        batch_policy_pass = bool(
            all(
                row[
                    "result"
                ]
                == "PASS"
                for row in batch_policy_rows
            )
        )

        print(
            f"Batch {batch_index} MPS seconds:      "
            f"{candidate_seconds:.3f}"
        )
        print(
            f"  loss abs diff:                      "
            f"{loss_abs_diff:.12e}"
        )
        print(
            f"  logit max abs diff:                 "
            f"{logit['max_abs_diff']:.12e}"
        )
        print(
            f"  gradient relative L2:               "
            f"{gradient['relative_l2_error']:.12e}"
        )
        print(
            f"  gradient cosine:                    "
            f"{gradient['cosine_similarity']:.12f}"
        )
        print(
            f"  gradient sign agreement:            "
            f"{gradient['sign_agreement']:.12f}"
        )
        print(
            f"  parameter relative L2:              "
            f"{parameter['relative_l2_error']:.12e}"
        )
        print(
            f"  parameter max abs diff:             "
            f"{parameter['max_abs_diff']:.12e}"
        )
        print(
            f"  frozen policy:                      "
            f"{'PASS' if batch_policy_pass else 'FAIL'}"
        )

        reference_capture.clear()
        candidate_capture.clear()

    # =========================================================================
    # Handle unsupported MPS op
    # =========================================================================

    if mps_runtime_error is not None:
        write_unavailable_outputs(
            status="MPS_RUNTIME_UNSUPPORTED",
            reason=mps_runtime_error,
            executor_metadata=(
                executor_metadata
            ),
        )

        banner(
            "PHASE 5.4.8a COMPLETE / "
            "MPS RUNTIME UNSUPPORTED BY CURRENT NUMERICAL PATH"
        )

        return

    # =========================================================================
    # Consolidate decision
    # =========================================================================

    banner(
        "MPS FROZEN-POLICY DECISION + RUNTIME PROJECTION"
    )

    cpu_proof_df = pd.DataFrame(
        cpu_proof_rows
    )

    mps_batch_df = pd.DataFrame(
        mps_batch_rows
    )

    policy_audit_df = pd.DataFrame(
        policy_audit_rows
    )

    mps_policy_pass = bool(
        (
            policy_audit_df[
                "result"
            ]
            == "PASS"
        ).all()
    )

    mps_mean_seconds = float(
        mps_batch_df[
            "mps_candidate_seconds"
        ].mean()
    )

    mps_batch1_seconds = float(
        mps_batch_df.loc[
            mps_batch_df[
                "batch_index"
            ]
            == 1,
            "mps_candidate_seconds",
        ].iloc[
            0
        ]
    )

    speedup_vs_canonical = (
        canonical_lean_seconds
        / mps_mean_seconds
    )

    projected_epoch_seconds = (
        mps_mean_seconds
        * BATCHES_PER_EPOCH
    )

    projected_full_seconds = (
        mps_mean_seconds
        * TOTAL_OPTIMIZER_STEPS
    )

    warm_projected_epoch_seconds = (
        mps_batch1_seconds
        * BATCHES_PER_EPOCH
    )

    warm_projected_full_seconds = (
        mps_batch1_seconds
        * TOTAL_OPTIMIZER_STEPS
    )

    print(
        "MPS numerical equivalence:            "
        + (
            "PASS"
            if mps_policy_pass
            else "FAIL"
        )
    )
    print(
        f"MPS mean seconds / batch:             "
        f"{mps_mean_seconds:.3f}"
    )
    print(
        f"MPS batch-1 warm seconds:             "
        f"{mps_batch1_seconds:.3f}"
    )
    print(
        f"Speedup vs canonical lean CPU:        "
        f"{speedup_vs_canonical:.2f}x"
    )
    print()
    print(
        "Projected one epoch (2-batch mean):   "
        f"{human_duration(projected_epoch_seconds)}"
    )
    print(
        "Projected 20 epochs (2-batch mean):   "
        f"{human_duration(projected_full_seconds)}"
    )
    print(
        "Projected 20 epochs (batch-1 warm):   "
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
                    "mps_two_batch_mean_seconds"
                ),
                "value": (
                    mps_mean_seconds
                ),
                "human": (
                    f"{mps_mean_seconds:.3f} s"
                ),
            },
            {
                "metric": (
                    "mps_batch1_warm_seconds"
                ),
                "value": (
                    mps_batch1_seconds
                ),
                "human": (
                    f"{mps_batch1_seconds:.3f} s"
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
                    "projected_20_epochs_two_batch_mean"
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
                    "projected_20_epochs_batch1_warm"
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
        "FINAL PHASE-5.4.8a INVARIANTS"
    )

    checks = [
        (
            "policy_frozen_before_mps",
            (
                policy[
                    "status"
                ]
                == "FROZEN"
            ),
        ),
        (
            "ast_device_bridge_cpu_batch0_exact",
            bool(
                cpu_proof_df.loc[
                    cpu_proof_df[
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
            "ast_device_bridge_cpu_batch1_exact",
            bool(
                cpu_proof_df.loc[
                    cpu_proof_df[
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
            "mps_two_batches_executed",
            (
                set(
                    mps_batch_df[
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
                    policy_audit_df.groupby(
                        "batch_index"
                    ).size()
                    == 11
                ).all()
            ),
        ),
        (
            "mps_timings_positive_finite",
            bool(
                np.isfinite(
                    mps_batch_df[
                        "mps_candidate_seconds"
                    ].to_numpy(
                        dtype=np.float64
                    )
                ).all()
                and (
                    mps_batch_df[
                        "mps_candidate_seconds"
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
            "At least one Phase-5.4.8a "
            "device-audit invariant failed."
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
        "WRITE PHASE-5.4.8a OUTPUTS"
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTRACT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cpu_proof_df.to_csv(
        CPU_BRIDGE_PROOF_PATH,
        index=False,
    )

    mps_batch_df.to_csv(
        MPS_BATCH_SUMMARY_PATH,
        index=False,
    )

    policy_audit_df.to_csv(
        MPS_POLICY_AUDIT_PATH,
        index=False,
    )

    runtime_df.to_csv(
        MPS_RUNTIME_PATH,
        index=False,
    )

    invariant_df.to_csv(
        FINAL_INVARIANT_PATH,
        index=False,
    )

    contract = {
        "phase": (
            "5.4.8a"
        ),
        "title": (
            "Canonical-Dense Apple MPS Numerical Equivalence + Feasibility Audit"
        ),
        "status": (
            "COMPLETE"
        ),
        "classification": (
            "IMPLEMENTATION_EQUIVALENT_DEVICE_AUDIT"
        ),
        "policy_version": (
            EXPECTED_POLICY_VERSION
        ),
        "policy_status": (
            "FROZEN_BEFORE_MPS"
        ),
        "candidate_runtime": (
            "CANONICAL_DENSE_EMBEDDING_MPS"
        ),
        "lean_executor_metadata": (
            executor_metadata
        ),
        "cpu_bridge_two_step_byte_exact": (
            True
        ),
        "mps_numerically_equivalent": (
            mps_policy_pass
        ),
        "mps_mean_seconds_per_batch": (
            mps_mean_seconds
        ),
        "mps_batch1_warm_seconds": (
            mps_batch1_seconds
        ),
        "speedup_vs_canonical_lean_cpu": (
            speedup_vs_canonical
        ),
        "projected_20_epochs_seconds_two_batch_mean": (
            projected_full_seconds
        ),
        "projected_20_epochs_seconds_batch1_warm": (
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
        "validation_cases_scored": (
            0
        ),
        "test_cases_scored": (
            0
        ),
        "next_phase": (
            (
                "5.4.8b_EXTENDED_MPS_STABILITY_AND_RUNTIME_BENCHMARK"
            )
            if mps_policy_pass
            else (
                "5.4.8b_ALTERNATIVE_ACCELERATED_RUNTIME_AUDIT"
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
            "5.4.8a"
        ),
        "status": (
            (
                "MPS_NUMERICAL_EQUIVALENCE_PASS"
            )
            if mps_policy_pass
            else (
                "MPS_NUMERICAL_EQUIVALENCE_FAIL"
            )
        ),
        "mps_mean_seconds_per_batch": (
            mps_mean_seconds
        ),
        "speedup_vs_canonical_lean_cpu": (
            speedup_vs_canonical
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
        CPU_BRIDGE_PROOF_PATH,
        MPS_BATCH_SUMMARY_PATH,
        MPS_POLICY_AUDIT_PATH,
        MPS_RUNTIME_PATH,
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
    del shared_cpu
    del shared_mps

    torch.mps.empty_cache()
    gc.collect()

    banner(
        "PHASE 5.4.8a FINAL STATUS"
    )

    print(
        "AST-derived CPU bridge:               BYTE-EXACT / TWO BATCHES"
    )
    print(
        "MPS numerical equivalence:            "
        + (
            "PASS"
            if mps_policy_pass
            else "FAIL"
        )
    )
    print(
        f"MPS mean seconds / batch:             "
        f"{mps_mean_seconds:.3f}"
    )
    print(
        f"Speedup vs canonical lean CPU:        "
        f"{speedup_vs_canonical:.2f}x"
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
        "PHASE 5.4.8a COMPLETE / "
        "APPLE MPS DEVICE AUDIT CLOSED"
    )


if __name__ == "__main__":
    main()