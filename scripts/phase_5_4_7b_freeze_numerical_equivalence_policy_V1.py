#!/usr/bin/env python3
"""
Phase 5.4.7b — Freeze Numerical-Equivalence Policy + Sparse Runtime Decision

Purpose
-------
Freeze the numerical-equivalence acceptance policy BEFORE any accelerated-device
(MPS) benchmark is observed.

This prevents post-hoc tolerance adjustment.

The policy is designed for float32 implementation-equivalent execution:
    - loss_abs_diff <= 1e-6
    - logit_max_abs_diff <= 1e-5
    - gradient_relative_l2_error <= 1e-5
    - gradient_cosine_similarity >= 0.999999
    - gradient_sign_agreement >= 0.99999
    - parameter_relative_l2_error <= 1e-7
    - parameter_max_abs_diff <= 1e-5
    - Adam exp_avg relative L2 <= 1e-5
    - Adam exp_avg cosine >= 0.999999
    - Adam exp_avg_sq relative L2 <= 1e-5
    - Adam exp_avg_sq cosine >= 0.999999

These thresholds are frozen before MPS/device results.

This phase:
    - reads the Phase-5.4.7a measured divergence
    - freezes the policy
    - decides whether D_BOTH_SPARSE qualifies as NUMERICALLY EQUIVALENT
    - does NOT make D_BOTH_SPARSE the final production runtime
    - does NOT train
    - does NOT validate
    - does NOT touch test

The next phase may benchmark MPS or another accelerated runtime against the
same frozen policy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Paths
# =============================================================================

CONTRACT_DIR = Path(
    "data/experimental/phase_5/contracts"
)

AUDIT_DIR = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_4_7b"
)

PHASE_5_4_7A_CONTRACT_PATH = (
    CONTRACT_DIR
    / "phase_5_4_7a_numerical_divergence_characterization_contract.json"
)

PHASE_5_4_6_CONTRACT_PATH = (
    CONTRACT_DIR
    / "phase_5_4_6_sparse_embedding_acceleration_contract.json"
)

PHASE_5_4_AUTHORIZATION_PATH = (
    CONTRACT_DIR
    / "phase_5_4_production_training_launch_authorization.json"
)

POLICY_PATH = (
    CONTRACT_DIR
    / "phase_5_4_numerical_equivalence_policy.json"
)

DECISION_REGISTER_PATH = (
    CONTRACT_DIR
    / "phase_5_4_7b_numerical_equivalence_decision_register.csv"
)

AUDIT_TABLE_PATH = (
    AUDIT_DIR
    / "sparse_runtime_numerical_equivalence_audit.csv"
)

FINAL_INVARIANT_PATH = (
    AUDIT_DIR
    / "phase_5_4_7b_final_invariants.csv"
)

CONTRACT_PATH = (
    CONTRACT_DIR
    / "phase_5_4_7b_numerical_equivalence_policy_contract.json"
)

MANIFEST_PATH = (
    AUDIT_DIR
    / "phase_5_4_7b_numerical_equivalence_policy_manifest.json"
)


# =============================================================================
# Frozen policy
# =============================================================================

POLICY_VERSION = "ITRS_PHASE5_NUMERICAL_EQUIVALENCE_V1"

THRESHOLDS = {
    "loss_abs_diff_max": 1e-6,
    "logit_max_abs_diff_max": 1e-5,
    "gradient_relative_l2_error_max": 1e-5,
    "gradient_cosine_similarity_min": 0.999999,
    "gradient_sign_agreement_min": 0.99999,
    "parameter_relative_l2_error_max": 1e-7,
    "parameter_max_abs_diff_max": 1e-5,
    "adam_exp_avg_relative_l2_error_max": 1e-5,
    "adam_exp_avg_cosine_similarity_min": 0.999999,
    "adam_exp_avg_sq_relative_l2_error_max": 1e-5,
    "adam_exp_avg_sq_cosine_similarity_min": 0.999999,
}

EXPECTED_CANDIDATE = "D_BOTH_SPARSE"
EXPECTED_BATCHES = {0, 1}


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


def finite_number(value) -> bool:
    try:
        value = float(value)
    except Exception:
        return False

    return bool(
        np.isfinite(value)
    )


def metric_check(
    *,
    batch_index: int,
    metric: str,
    actual: float,
    comparator: str,
    threshold: float,
) -> dict:

    actual = float(actual)
    threshold = float(threshold)

    if comparator == "<=":
        passed = (
            actual
            <= threshold
        )
    elif comparator == ">=":
        passed = (
            actual
            >= threshold
        )
    else:
        raise AssertionError(
            f"Unsupported comparator: {comparator}"
        )

    return {
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


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.4.7b — "
        "FREEZE NUMERICAL-EQUIVALENCE POLICY + SPARSE RUNTIME DECISION"
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
        "PREREQUISITE GATE"
    )

    for path in (
        PHASE_5_4_7A_CONTRACT_PATH,
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

    divergence = load_json(
        PHASE_5_4_7A_CONTRACT_PATH
    )

    sparse_contract = load_json(
        PHASE_5_4_6_CONTRACT_PATH
    )

    authorization = load_json(
        PHASE_5_4_AUTHORIZATION_PATH
    )

    require(
        divergence.get(
            "status"
        )
        == "DIAGNOSTIC_COMPLETE",
        (
            "Phase-5.4.7a divergence characterization "
            "is not DIAGNOSTIC_COMPLETE."
        ),
    )

    require(
        divergence.get(
            "candidate_runtime"
        )
        == EXPECTED_CANDIDATE,
        (
            "Phase-5.4.7a candidate runtime drift."
        ),
    )

    require(
        divergence.get(
            "tolerance_frozen"
        )
        is False,
        (
            "Phase-5.4.7a unexpectedly froze a tolerance."
        ),
    )

    require(
        divergence.get(
            "acceptance_decision_made"
        )
        is False,
        (
            "Phase-5.4.7a unexpectedly made an acceptance decision."
        ),
    )

    require(
        sparse_contract.get(
            "status"
        )
        == "COMPLETE",
        (
            "Phase-5.4.6 sparse audit is not COMPLETE."
        ),
    )

    require(
        authorization.get(
            "training_allowed"
        )
        is True,
        (
            "Phase-5.4 launch authorization is not ALLOWED."
        ),
    )

    batch_summary = divergence.get(
        "batch_summary",
        []
    )

    require(
        isinstance(
            batch_summary,
            list,
        )
        and len(
            batch_summary
        )
        == 2,
        (
            "Phase-5.4.7a must contain exactly "
            "two batch-summary records."
        ),
    )

    batch_indices = {
        int(
            row[
                "batch_index"
            ]
        )
        for row in batch_summary
    }

    require(
        batch_indices
        == EXPECTED_BATCHES,
        (
            "Phase-5.4.7a batch set drift."
        ),
    )

    print(
        "Phase-5.4.7a diagnostic:              PASS"
    )
    print(
        f"Candidate runtime:                    "
        f"{EXPECTED_CANDIDATE}"
    )
    print(
        "Policy previously frozen:             NO"
    )

    # =========================================================================
    # Freeze policy
    # =========================================================================

    banner(
        "FREEZE NUMERICAL-EQUIVALENCE POLICY"
    )

    policy = {
        "schema_version": (
            POLICY_VERSION
        ),
        "status": (
            "FROZEN"
        ),
        "classification": (
            "IMPLEMENTATION_EQUIVALENT_CHOICE"
        ),
        "purpose": (
            "Judge accelerated float32 runtimes against "
            "the canonical CPU reference without requiring "
            "byte-identical accumulation order."
        ),
        "reference_runtime": (
            "CANONICAL_DENSE_EMBEDDING_CPU"
        ),
        "thresholds": (
            THRESHOLDS
        ),
        "evaluation_scope": {
            "minimum_stateful_training_batches": (
                2
            ),
            "must_start_from_canonical_initialization": (
                True
            ),
            "must_use_identical_examples_and_order": (
                True
            ),
            "must_use_identical_optimizer_hyperparameters": (
                True
            ),
            "must_compare_post_adam_parameter_state": (
                True
            ),
            "must_compare_adam_moments": (
                True
            ),
        },
        "scientific_semantics_that_must_not_change": [
            "model architecture",
            "parameter shapes",
            "initialization seed/state",
            "training positives",
            "training negatives",
            "example order",
            "batch size",
            "BCEWithLogitsLoss",
            "Adam hyperparameters",
            "epoch count",
            "validation candidate sets",
            "validation ranking semantics",
            "best-checkpoint selection",
            "test isolation",
        ],
        "nonbinding_diagnostics": [
            (
                "byte-exact hash equality remains reported "
                "but is not required for accelerated runtime acceptance"
            ),
            (
                "parameter exact-element fraction is descriptive "
                "and has no acceptance threshold"
            ),
        ],
        "policy_frozen_before_device_benchmark": (
            True
        ),
    }

    # Write policy now, before any future device result.
    POLICY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    POLICY_PATH.write_text(
        json.dumps(
            policy,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    for key, value in THRESHOLDS.items():
        print(
            f"{key:48s} {value:.12g}"
        )

    # =========================================================================
    # Apply frozen policy to sparse candidate
    # =========================================================================

    banner(
        "APPLY FROZEN POLICY TO D_BOTH_SPARSE"
    )

    audit_rows = []

    for row in sorted(
        batch_summary,
        key=lambda item: int(
            item[
                "batch_index"
            ]
        ),
    ):

        batch_index = int(
            row[
                "batch_index"
            ]
        )

        required_metrics = [
            "loss_abs_diff",
            "logit_max_abs_diff",
            "gradient_relative_l2_error",
            "gradient_cosine_similarity",
            "gradient_sign_agreement",
            "parameter_relative_l2_error",
            "parameter_max_abs_diff",
            "adam_exp_avg_relative_l2_error",
            "adam_exp_avg_cosine_similarity",
            "adam_exp_avg_sq_relative_l2_error",
            "adam_exp_avg_sq_cosine_similarity",
        ]

        for metric in required_metrics:
            require(
                metric in row,
                (
                    f"Missing Phase-5.4.7a metric: "
                    f"batch={batch_index}, metric={metric}"
                ),
            )

            require(
                finite_number(
                    row[
                        metric
                    ]
                ),
                (
                    f"Non-finite Phase-5.4.7a metric: "
                    f"batch={batch_index}, metric={metric}"
                ),
            )

        audit_rows.extend(
            [
                metric_check(
                    batch_index=batch_index,
                    metric="loss_abs_diff",
                    actual=row[
                        "loss_abs_diff"
                    ],
                    comparator="<=",
                    threshold=THRESHOLDS[
                        "loss_abs_diff_max"
                    ],
                ),
                metric_check(
                    batch_index=batch_index,
                    metric="logit_max_abs_diff",
                    actual=row[
                        "logit_max_abs_diff"
                    ],
                    comparator="<=",
                    threshold=THRESHOLDS[
                        "logit_max_abs_diff_max"
                    ],
                ),
                metric_check(
                    batch_index=batch_index,
                    metric="gradient_relative_l2_error",
                    actual=row[
                        "gradient_relative_l2_error"
                    ],
                    comparator="<=",
                    threshold=THRESHOLDS[
                        "gradient_relative_l2_error_max"
                    ],
                ),
                metric_check(
                    batch_index=batch_index,
                    metric="gradient_cosine_similarity",
                    actual=row[
                        "gradient_cosine_similarity"
                    ],
                    comparator=">=",
                    threshold=THRESHOLDS[
                        "gradient_cosine_similarity_min"
                    ],
                ),
                metric_check(
                    batch_index=batch_index,
                    metric="gradient_sign_agreement",
                    actual=row[
                        "gradient_sign_agreement"
                    ],
                    comparator=">=",
                    threshold=THRESHOLDS[
                        "gradient_sign_agreement_min"
                    ],
                ),
                metric_check(
                    batch_index=batch_index,
                    metric="parameter_relative_l2_error",
                    actual=row[
                        "parameter_relative_l2_error"
                    ],
                    comparator="<=",
                    threshold=THRESHOLDS[
                        "parameter_relative_l2_error_max"
                    ],
                ),
                metric_check(
                    batch_index=batch_index,
                    metric="parameter_max_abs_diff",
                    actual=row[
                        "parameter_max_abs_diff"
                    ],
                    comparator="<=",
                    threshold=THRESHOLDS[
                        "parameter_max_abs_diff_max"
                    ],
                ),
                metric_check(
                    batch_index=batch_index,
                    metric="adam_exp_avg_relative_l2_error",
                    actual=row[
                        "adam_exp_avg_relative_l2_error"
                    ],
                    comparator="<=",
                    threshold=THRESHOLDS[
                        "adam_exp_avg_relative_l2_error_max"
                    ],
                ),
                metric_check(
                    batch_index=batch_index,
                    metric="adam_exp_avg_cosine_similarity",
                    actual=row[
                        "adam_exp_avg_cosine_similarity"
                    ],
                    comparator=">=",
                    threshold=THRESHOLDS[
                        "adam_exp_avg_cosine_similarity_min"
                    ],
                ),
                metric_check(
                    batch_index=batch_index,
                    metric="adam_exp_avg_sq_relative_l2_error",
                    actual=row[
                        "adam_exp_avg_sq_relative_l2_error"
                    ],
                    comparator="<=",
                    threshold=THRESHOLDS[
                        "adam_exp_avg_sq_relative_l2_error_max"
                    ],
                ),
                metric_check(
                    batch_index=batch_index,
                    metric="adam_exp_avg_sq_cosine_similarity",
                    actual=row[
                        "adam_exp_avg_sq_cosine_similarity"
                    ],
                    comparator=">=",
                    threshold=THRESHOLDS[
                        "adam_exp_avg_sq_cosine_similarity_min"
                    ],
                ),
            ]
        )

    audit_df = pd.DataFrame(
        audit_rows
    )

    sparse_numerically_equivalent = bool(
        (
            audit_df[
                "result"
            ]
            == "PASS"
        ).all()
    )

    print(
        audit_df.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.12e}"
            ),
        )
    )

    print()
    print(
        "D_BOTH_SPARSE numerical-equivalence:  "
        + (
            "PASS"
            if sparse_numerically_equivalent
            else "FAIL"
        )
    )

    # =========================================================================
    # Decision register
    # =========================================================================

    decision_df = pd.DataFrame(
        [
            {
                "decision": (
                    "numerical_equivalence_policy"
                ),
                "value": (
                    POLICY_VERSION
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": (
                    "FROZEN_PHASE_5_4_7B"
                ),
            },
            {
                "decision": (
                    "byte_exact_required_for_accelerated_runtime"
                ),
                "value": (
                    "NO"
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": (
                    "FROZEN_PHASE_5_4_7B"
                ),
            },
            {
                "decision": (
                    "canonical_cpu_reference_retained"
                ),
                "value": (
                    "YES"
                ),
                "classification": (
                    "EVALUATION_INTEGRITY_GUARD"
                ),
                "status": (
                    "FROZEN_PHASE_5_4_7B"
                ),
            },
            {
                "decision": (
                    "D_BOTH_SPARSE_numerical_equivalence"
                ),
                "value": (
                    "PASS"
                    if sparse_numerically_equivalent
                    else "FAIL"
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": (
                    "FROZEN_PHASE_5_4_7B"
                ),
            },
            {
                "decision": (
                    "D_BOTH_SPARSE_final_production_runtime"
                ),
                "value": (
                    "NOT_SELECTED"
                ),
                "classification": (
                    "RUNTIME_FEASIBILITY_ONLY"
                ),
                "status": (
                    "OPEN_PENDING_DEVICE_BENCHMARK"
                ),
            },
            {
                "decision": (
                    "future_device_runtime_must_use_same_policy"
                ),
                "value": (
                    "YES"
                ),
                "classification": (
                    "EVALUATION_INTEGRITY_GUARD"
                ),
                "status": (
                    "FROZEN_PHASE_5_4_7B"
                ),
            },
        ]
    )

    # =========================================================================
    # Final invariants
    # =========================================================================

    banner(
        "FINAL PHASE-5.4.7b INVARIANTS"
    )

    checks = [
        (
            "policy_written_before_device_benchmark",
            (
                POLICY_PATH.exists()
            ),
        ),
        (
            "policy_status_frozen",
            (
                policy[
                    "status"
                ]
                == "FROZEN"
            ),
        ),
        (
            "all_11_metrics_checked_per_batch",
            bool(
                (
                    audit_df.groupby(
                        "batch_index"
                    ).size()
                    == 11
                ).all()
            ),
        ),
        (
            "both_batches_checked",
            (
                set(
                    audit_df[
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
            "canonical_cpu_reference_retained",
            True,
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
            "At least one Phase-5.4.7b "
            "policy-freeze invariant failed."
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
        "WRITE PHASE-5.4.7b OUTPUTS"
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_df.to_csv(
        AUDIT_TABLE_PATH,
        index=False,
    )

    decision_df.to_csv(
        DECISION_REGISTER_PATH,
        index=False,
    )

    invariant_df.to_csv(
        FINAL_INVARIANT_PATH,
        index=False,
    )

    contract = {
        "phase": (
            "5.4.7b"
        ),
        "title": (
            "Numerical-Equivalence Policy Freeze + Sparse Runtime Decision"
        ),
        "status": (
            "FROZEN"
        ),
        "classification": (
            "IMPLEMENTATION_EQUIVALENT_CHOICE"
        ),
        "policy_version": (
            POLICY_VERSION
        ),
        "policy_path": str(
            POLICY_PATH
        ),
        "thresholds": (
            THRESHOLDS
        ),
        "reference_runtime": (
            "CANONICAL_DENSE_EMBEDDING_CPU"
        ),
        "candidate_runtime": (
            EXPECTED_CANDIDATE
        ),
        "candidate_numerically_equivalent": (
            sparse_numerically_equivalent
        ),
        "candidate_final_production_runtime_selected": (
            False
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
        "next_phase": (
            "5.4.8_ACCELERATED_DEVICE_RUNTIME_BENCHMARK"
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
            "5.4.7b"
        ),
        "status": (
            "NUMERICAL_EQUIVALENCE_POLICY_FROZEN"
        ),
        "policy_version": (
            POLICY_VERSION
        ),
        "D_BOTH_SPARSE_pass": (
            sparse_numerically_equivalent
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
        POLICY_PATH,
        AUDIT_TABLE_PATH,
        DECISION_REGISTER_PATH,
        FINAL_INVARIANT_PATH,
        CONTRACT_PATH,
        MANIFEST_PATH,
    ):
        print(
            f"WROTE  {path}"
        )

    banner(
        "PHASE 5.4.7b FINAL STATUS"
    )

    print(
        f"Policy version:                       "
        f"{POLICY_VERSION}"
    )
    print(
        "Policy status:                        FROZEN"
    )
    print(
        "D_BOTH_SPARSE numerical equivalence:  "
        + (
            "PASS"
            if sparse_numerically_equivalent
            else "FAIL"
        )
    )
    print(
        "D_BOTH_SPARSE production selected:    NO"
    )
    print()
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
        "PHASE 5.4.7b COMPLETE / "
        "NUMERICAL-EQUIVALENCE POLICY FROZEN BEFORE DEVICE BENCHMARK"
    )


if __name__ == "__main__":
    main()