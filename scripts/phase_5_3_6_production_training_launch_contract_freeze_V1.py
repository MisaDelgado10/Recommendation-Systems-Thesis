#!/usr/bin/env python3
"""
Phase 5.3.6 — Production Training Launch Contract Freeze

Purpose
-------
Freeze the exact operational contract for the real Phase-5.4 20-epoch run.

This phase performs:
    - NO model construction
    - NO optimizer construction
    - NO negative regeneration
    - NO forward/backward
    - NO optimizer.step()
    - NO validation scoring
    - NO test scoring

Why this phase is intentionally small
-------------------------------------
The numerical training path, checkpoint/resume behavior, validation runtime,
best-checkpoint selection, and generalized stream generator have already been
independently proved and frozen.

This phase does NOT duplicate those expensive runtimes. It freezes the exact
production schedule and reporting policy that the Phase-5.4 executable must
obey.

Metric availability
-------------------
During Phase 5.4:
    after EACH completed training epoch:
        - mean training BCE loss for that epoch
        - validation HR@10
        - validation NDCG@10
        - whether the epoch became the new best checkpoint

Therefore the first REAL trained-model evaluation metrics are available after
epoch_index=0 completes (display epoch "1").

Test metrics remain forbidden until:
    - all 20 epochs complete
    - all 20 validation evaluations are committed
    - the best validation checkpoint is selected
    - training_complete == True

Then and only then:
    - load BEST checkpoint
    - score the test split exactly once
    - report final test HR@10 and NDCG@10
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# =============================================================================
# Frozen prerequisite contracts
# =============================================================================

CONTRACT_DIR = Path(
    "data/experimental/phase_5/contracts"
)

AUDIT_DIR = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_6"
)

PREREQUISITE_CONTRACTS = {
    "5.3.2a_training_execution": (
        CONTRACT_DIR
        / "phase_5_3_2a_training_execution_state_contract.json"
    ),
    "5.3.2c_production_controller": (
        CONTRACT_DIR
        / "phase_5_3_2c_production_driver_controller_contract.json"
    ),
    "5.3.3a_ranking": (
        CONTRACT_DIR
        / "phase_5_3_3a_validation_ranking_metric_contract.json"
    ),
    "5.3.3b_validation_preflight": (
        CONTRACT_DIR
        / "phase_5_3_3b_canonical_validation_scoring_preflight_contract.json"
    ),
    "5.3.3c_full_validation": (
        CONTRACT_DIR
        / "phase_5_3_3c_full_validation_runtime_contract.json"
    ),
    "5.3.3d_validation_checkpoint": (
        CONTRACT_DIR
        / "phase_5_3_3d_validation_checkpoint_integration_contract.json"
    ),
    "5.3.4_production_assembly": (
        CONTRACT_DIR
        / "phase_5_3_4_production_trainer_assembly_contract.json"
    ),
    "5.3.5a_representation": (
        CONTRACT_DIR
        / "phase_5_3_5a_epoch0_stream_representation_audit.json"
    ),
    "5.3.5_generalized_stream": (
        CONTRACT_DIR
        / "phase_5_3_5_generalized_training_stream_generator_contract.json"
    ),
}


# =============================================================================
# Frozen production arithmetic
# =============================================================================

NUM_EPOCHS = 20
BATCH_SIZE = 512
POSITIVE_EVENTS_PER_EPOCH = 1_073_249
NEGATIVES_PER_POSITIVE = 4
EXAMPLES_PER_EPOCH = 5_366_245
BATCHES_PER_EPOCH = 10_481
FINAL_BATCH_SIZE = 485
TOTAL_OPTIMIZER_STEPS = 209_620

VALIDATION_CASES = 2_251
TEST_CASES = 20_264
CANDIDATES_PER_EVALUATION_CASE = 100
NEGATIVES_PER_EVALUATION_CASE = 99

EXPECTED_INITIAL_MODEL_SHA256 = (
    "49e822ea7fad35c458f47e134c94c05e"
    "ac099b68c5c468e2c71559c8c88998ab"
)

EXPECTED_POSITIVE_ORDER_SHA256 = (
    "73b074a80675793b811fbdc8a0609883"
    "c857fb2a687a2e01c31865ade5b509d1"
)

EXPECTED_EPOCH0_NEGATIVE_SHA256 = (
    "47015b147b1949562c0f6737a6f3a3f2"
    "d7cabd2d2202e4e57456d884a1e23fe6"
)

EXPECTED_EPOCH0_ORDER_SHA256 = (
    "0156be3ee623ade1ae696557337bfb324"
    "e9011adb7df8be9648ecb0a426c134e"
)

EXPECTED_EPOCH1_NEGATIVE_SHA256 = (
    "f7b415e0f305e049cc94c7e4261b6838"
    "00085f8fa7f7b2df11ff3658dea9d850"
)

EXPECTED_EPOCH1_ORDER_SHA256 = (
    "2da43d28e540ed48cb557ca889190b5d"
    "febc7c1207cbd3882df2fa14ca2a28d8"
)

EXPECTED_EPOCH19_NEGATIVE_SHA256 = (
    "06f9a11d8986ba9b7e0242fc41423478"
    "9ccc7adba91cab6da68ebc21e44682b7"
)

EXPECTED_EPOCH19_ORDER_SHA256 = (
    "c6236cb081ddba7f72eb0d36199500ac"
    "6b7695646e21a824fb84b85fae769329"
)

EXPECTED_EPOCH0_NEGATIVE_DTYPE = "int32"
EXPECTED_EPOCH0_NEGATIVE_DTYPE_STR = "<i4"


# =============================================================================
# Outputs
# =============================================================================

PREREQUISITE_AUDIT_PATH = (
    AUDIT_DIR
    / "production_launch_prerequisite_contract_audit.csv"
)

PRODUCTION_SCHEDULE_PATH = (
    AUDIT_DIR
    / "production_20_epoch_schedule.csv"
)

METRIC_REPORTING_PATH = (
    AUDIT_DIR
    / "production_metric_reporting_contract.csv"
)

TEST_GUARD_PATH = (
    AUDIT_DIR
    / "production_test_access_guard.csv"
)

FINAL_INVARIANT_PATH = (
    AUDIT_DIR
    / "phase_5_3_6_final_invariants.csv"
)

LAUNCH_CONFIG_PATH = (
    CONTRACT_DIR
    / "phase_5_4_production_training_launch_config.json"
)

CONTRACT_PATH = (
    CONTRACT_DIR
    / "phase_5_3_6_production_training_launch_contract.json"
)

DECISION_REGISTER_PATH = (
    CONTRACT_DIR
    / "phase_5_3_6_production_launch_decision_register.csv"
)

MANIFEST_PATH = (
    AUDIT_DIR
    / "phase_5_3_6_production_training_launch_manifest.json"
)


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


def load_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def nested_get(
    payload: dict,
    keys: list[str],
):
    current = payload

    for key in keys:
        require(
            isinstance(
                current,
                dict,
            )
            and key in current,
            (
                "Missing expected contract field: "
                + ".".join(
                    keys
                )
            ),
        )

        current = current[
            key
        ]

    return current


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.3.6 — "
        "PRODUCTION TRAINING LAUNCH CONTRACT FREEZE"
    )

    print(
        "Model instantiated:                   NO"
    )
    print(
        "Optimizer instantiated:               NO"
    )
    print(
        "Training batches executed:            0"
    )
    print(
        "Validation cases scored:              0"
    )
    print(
        "Test cases scored:                    0"
    )

    # =========================================================================
    # Contract gate
    # =========================================================================

    banner(
        "FROZEN PREREQUISITE CONTRACT GATE"
    )

    contracts = {}
    prerequisite_rows = []

    for (
        phase_name,
        path,
    ) in PREREQUISITE_CONTRACTS.items():

        require(
            path.exists(),
            (
                "Missing prerequisite contract: "
                f"{path}"
            ),
        )

        payload = load_json(
            path
        )

        status = str(
            payload.get(
                "status",
                "",
            )
        )

        # 5.3.5a is a diagnostic prerequisite rather than a FROZEN scientific
        # contract; all others must be frozen.
        if phase_name == (
            "5.3.5a_representation"
        ):
            acceptable = (
                status
                == "DIAGNOSIS_COMPLETE"
            )
        else:
            acceptable = (
                status
                == "FROZEN"
            )

        require(
            acceptable,
            (
                f"Prerequisite {phase_name} "
                f"has unacceptable status: {status}"
            ),
        )

        contracts[
            phase_name
        ] = payload

        prerequisite_rows.append(
            {
                "phase": (
                    phase_name
                ),
                "path": str(
                    path
                ),
                "status": (
                    status
                ),
                "gate": (
                    "PASS"
                ),
            }
        )

        print(
            f"{phase_name:40s} "
            f"{status:20s} PASS"
        )

    prerequisite_df = pd.DataFrame(
        prerequisite_rows
    )

    # =========================================================================
    # Stream contract anchors
    # =========================================================================

    banner(
        "GENERALIZED TRAINING STREAM LAUNCH ANCHORS"
    )

    stream_contract = contracts[
        "5.3.5_generalized_stream"
    ]

    require(
        nested_get(
            stream_contract,
            [
                "positive_stream",
                "logical_sha256",
            ],
        )
        == EXPECTED_POSITIVE_ORDER_SHA256,
        (
            "Positive-order fingerprint drift."
        ),
    )

    require(
        nested_get(
            stream_contract,
            [
                "epoch0_exact_regression",
                "negative_sha256",
            ],
        )
        == EXPECTED_EPOCH0_NEGATIVE_SHA256,
        (
            "Epoch-0 negative fingerprint drift."
        ),
    )

    require(
        nested_get(
            stream_contract,
            [
                "epoch0_exact_regression",
                "order_sha256",
            ],
        )
        == EXPECTED_EPOCH0_ORDER_SHA256,
        (
            "Epoch-0 order fingerprint drift."
        ),
    )

    require(
        nested_get(
            stream_contract,
            [
                "epoch0_exact_regression",
                "negative_dtype",
            ],
        )
        == EXPECTED_EPOCH0_NEGATIVE_DTYPE,
        (
            "Epoch-0 negative dtype drift."
        ),
    )

    require(
        nested_get(
            stream_contract,
            [
                "epoch0_exact_regression",
                "negative_dtype_str",
            ],
        )
        == EXPECTED_EPOCH0_NEGATIVE_DTYPE_STR,
        (
            "Epoch-0 negative dtype.str drift."
        ),
    )

    require(
        nested_get(
            stream_contract,
            [
                "epoch1_frozen_fingerprint",
                "negative_sha256",
            ],
        )
        == EXPECTED_EPOCH1_NEGATIVE_SHA256,
        (
            "Epoch-1 negative fingerprint drift."
        ),
    )

    require(
        nested_get(
            stream_contract,
            [
                "epoch1_frozen_fingerprint",
                "order_sha256",
            ],
        )
        == EXPECTED_EPOCH1_ORDER_SHA256,
        (
            "Epoch-1 order fingerprint drift."
        ),
    )

    require(
        nested_get(
            stream_contract,
            [
                "epoch19_frozen_fingerprint",
                "negative_sha256",
            ],
        )
        == EXPECTED_EPOCH19_NEGATIVE_SHA256,
        (
            "Epoch-19 negative fingerprint drift."
        ),
    )

    require(
        nested_get(
            stream_contract,
            [
                "epoch19_frozen_fingerprint",
                "order_sha256",
            ],
        )
        == EXPECTED_EPOCH19_ORDER_SHA256,
        (
            "Epoch-19 order fingerprint drift."
        ),
    )

    print(
        "Positive-order anchor:                PASS"
    )
    print(
        "Epoch-0 negative/order anchors:       PASS"
    )
    print(
        "Epoch-0 int32 representation:         PASS"
    )
    print(
        "Epoch-1 independent anchors:          PASS"
    )
    print(
        "Epoch-19 independent anchors:         PASS"
    )

    # =========================================================================
    # Exact 20-epoch production schedule
    # =========================================================================

    banner(
        "FREEZE EXACT 20-EPOCH PRODUCTION SCHEDULE"
    )

    schedule_rows = []

    for epoch_index in range(
        NUM_EPOCHS
    ):
        first_global_step = (
            epoch_index
            * BATCHES_PER_EPOCH
            + 1
        )

        final_global_step = (
            (
                epoch_index
                + 1
            )
            * BATCHES_PER_EPOCH
        )

        schedule_rows.append(
            {
                "epoch_index": (
                    epoch_index
                ),
                "display_epoch": (
                    epoch_index + 1
                ),
                "positive_events": (
                    POSITIVE_EVENTS_PER_EPOCH
                ),
                "negatives_per_positive": (
                    NEGATIVES_PER_POSITIVE
                ),
                "examples": (
                    EXAMPLES_PER_EPOCH
                ),
                "batches": (
                    BATCHES_PER_EPOCH
                ),
                "batch_size": (
                    BATCH_SIZE
                ),
                "final_batch_size": (
                    FINAL_BATCH_SIZE
                ),
                "first_global_optimizer_step": (
                    first_global_step
                ),
                "final_global_optimizer_step": (
                    final_global_step
                ),
                "validation_after_epoch": (
                    True
                ),
                "checkpoint_latest_after_validation": (
                    True
                ),
                "best_checkpoint_comparison": (
                    True
                ),
                "test_allowed_after_epoch": (
                    epoch_index
                    == (
                        NUM_EPOCHS - 1
                    )
                ),
            }
        )

    schedule_df = pd.DataFrame(
        schedule_rows
    )

    require(
        int(
            schedule_df.iloc[
                -1
            ][
                "final_global_optimizer_step"
            ]
        )
        == TOTAL_OPTIMIZER_STEPS,
        (
            "20-epoch optimizer-step arithmetic drift."
        ),
    )

    require(
        int(
            schedule_df[
                "examples"
            ].sum()
        )
        == (
            NUM_EPOCHS
            * EXAMPLES_PER_EPOCH
        ),
        (
            "20-epoch example arithmetic drift."
        ),
    )

    print(
        schedule_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Metric reporting contract
    # =========================================================================

    banner(
        "TRAINING / VALIDATION METRIC REPORTING CONTRACT"
    )

    metric_df = pd.DataFrame(
        [
            {
                "metric": (
                    "epoch_mean_training_BCE"
                ),
                "available": (
                    "after every completed training epoch"
                ),
                "used_for_model_selection": (
                    False
                ),
                "split": (
                    "training"
                ),
            },
            {
                "metric": (
                    "validation_HR@10"
                ),
                "available": (
                    "after every completed epoch validation"
                ),
                "used_for_model_selection": (
                    True
                ),
                "split": (
                    "validation"
                ),
            },
            {
                "metric": (
                    "validation_NDCG@10"
                ),
                "available": (
                    "after every completed epoch validation"
                ),
                "used_for_model_selection": (
                    True
                ),
                "split": (
                    "validation"
                ),
            },
            {
                "metric": (
                    "is_new_best_checkpoint"
                ),
                "available": (
                    "after every completed epoch validation"
                ),
                "used_for_model_selection": (
                    True
                ),
                "split": (
                    "validation"
                ),
            },
            {
                "metric": (
                    "final_test_HR@10"
                ),
                "available": (
                    "once after all 20 epochs using BEST checkpoint"
                ),
                "used_for_model_selection": (
                    False
                ),
                "split": (
                    "test"
                ),
            },
            {
                "metric": (
                    "final_test_NDCG@10"
                ),
                "available": (
                    "once after all 20 epochs using BEST checkpoint"
                ),
                "used_for_model_selection": (
                    False
                ),
                "split": (
                    "test"
                ),
            },
        ]
    )

    print(
        metric_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Test guard
    # =========================================================================

    banner(
        "FINAL TEST ACCESS GUARD"
    )

    test_guard_df = pd.DataFrame(
        [
            {
                "state": (
                    "before_epoch_1"
                ),
                "training_complete": (
                    False
                ),
                "test_access_allowed": (
                    False
                ),
            },
            {
                "state": (
                    "after_epoch_1_validation"
                ),
                "training_complete": (
                    False
                ),
                "test_access_allowed": (
                    False
                ),
            },
            {
                "state": (
                    "after_epoch_10_validation"
                ),
                "training_complete": (
                    False
                ),
                "test_access_allowed": (
                    False
                ),
            },
            {
                "state": (
                    "after_epoch_20_training_before_validation_commit"
                ),
                "training_complete": (
                    False
                ),
                "test_access_allowed": (
                    False
                ),
            },
            {
                "state": (
                    "after_epoch_20_validation_commit"
                ),
                "training_complete": (
                    True
                ),
                "test_access_allowed": (
                    True
                ),
            },
        ]
    )

    require(
        bool(
            (
                test_guard_df.iloc[
                    :-1
                ][
                    "test_access_allowed"
                ]
                == False
            ).all()
        ),
        (
            "Test guard opens before final validation commit."
        ),
    )

    require(
        bool(
            test_guard_df.iloc[
                -1
            ][
                "test_access_allowed"
            ]
        ),
        (
            "Test guard does not open after final validation commit."
        ),
    )

    print(
        test_guard_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Final invariants
    # =========================================================================

    banner(
        "FINAL PHASE-5.3.6 INVARIANTS"
    )

    checks = [
        (
            "all_prerequisite_contracts_pass",
            bool(
                (
                    prerequisite_df[
                        "gate"
                    ]
                    == "PASS"
                ).all()
            ),
        ),
        (
            "epochs_exactly_20",
            (
                len(
                    schedule_df
                )
                == 20
            ),
        ),
        (
            "batches_per_epoch_10481",
            bool(
                (
                    schedule_df[
                        "batches"
                    ]
                    == 10_481
                ).all()
            ),
        ),
        (
            "examples_per_epoch_5366245",
            bool(
                (
                    schedule_df[
                        "examples"
                    ]
                    == 5_366_245
                ).all()
            ),
        ),
        (
            "total_optimizer_steps_209620",
            (
                int(
                    schedule_df.iloc[
                        -1
                    ][
                        "final_global_optimizer_step"
                    ]
                )
                == 209_620
            ),
        ),
        (
            "validation_every_epoch",
            bool(
                schedule_df[
                    "validation_after_epoch"
                ].all()
            ),
        ),
        (
            "first_real_metrics_after_epoch1",
            True,
        ),
        (
            "best_selection_validation_only",
            True,
        ),
        (
            "test_forbidden_before_final_validation_commit",
            bool(
                (
                    test_guard_df.iloc[
                        :-1
                    ][
                        "test_access_allowed"
                    ]
                    == False
                ).all()
            ),
        ),
        (
            "test_allowed_only_after_training_complete",
            bool(
                test_guard_df.iloc[
                    -1
                ][
                    "training_complete"
                ]
                and test_guard_df.iloc[
                    -1
                ][
                    "test_access_allowed"
                ]
            ),
        ),
        (
            "model_not_instantiated",
            True,
        ),
        (
            "optimizer_not_instantiated",
            True,
        ),
        (
            "training_steps_zero",
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
            "At least one Phase-5.3.6 "
            "launch-contract invariant failed."
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
        "WRITE PHASE-5.3.6 OUTPUTS"
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTRACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prerequisite_df.to_csv(
        PREREQUISITE_AUDIT_PATH,
        index=False,
    )

    schedule_df.to_csv(
        PRODUCTION_SCHEDULE_PATH,
        index=False,
    )

    metric_df.to_csv(
        METRIC_REPORTING_PATH,
        index=False,
    )

    test_guard_df.to_csv(
        TEST_GUARD_PATH,
        index=False,
    )

    invariant_df.to_csv(
        FINAL_INVARIANT_PATH,
        index=False,
    )

    decision_df = pd.DataFrame(
        [
            {
                "decision": (
                    "production_epochs"
                ),
                "value": (
                    "20"
                ),
                "classification": (
                    "INHERITED_FROZEN_PHASE_5_2_2"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_6"
                ),
            },
            {
                "decision": (
                    "metric_reporting_cadence"
                ),
                "value": (
                    "TRAIN_LOSS_AND_VALIDATION_HR10_NDCG10_AFTER_EVERY_EPOCH"
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_6"
                ),
            },
            {
                "decision": (
                    "first_trained_model_metrics"
                ),
                "value": (
                    "AFTER_EPOCH_INDEX_0_VALIDATION_DISPLAY_EPOCH_1"
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_6"
                ),
            },
            {
                "decision": (
                    "final_test_timing"
                ),
                "value": (
                    "ONCE_AFTER_EPOCH19_VALIDATION_COMMIT_USING_BEST_CHECKPOINT"
                ),
                "classification": (
                    "EVALUATION_INTEGRITY_GUARD"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_6"
                ),
            },
            {
                "decision": (
                    "phase_5_3_7_gate"
                ),
                "value": (
                    "FULL_READ_ONLY_INTEGRITY_CLOSURE_REQUIRED_BEFORE_TRAINING"
                ),
                "classification": (
                    "EVALUATION_INTEGRITY_GUARD"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_6"
                ),
            },
        ]
    )

    decision_df.to_csv(
        DECISION_REGISTER_PATH,
        index=False,
    )

    launch_config = {
        "schema_version": (
            "ITRS_PHASE5_PRODUCTION_LAUNCH_V1"
        ),
        "phase": (
            "5.4"
        ),
        "status": (
            "LOCKED_PENDING_PHASE_5_3_7_INTEGRITY_CLOSURE"
        ),
        "reference_device": (
            "CPU"
        ),
        "reference_pytorch": (
            "2.7.0"
        ),
        "initial_model_sha256": (
            EXPECTED_INITIAL_MODEL_SHA256
        ),
        "epochs": (
            NUM_EPOCHS
        ),
        "batch_size": (
            BATCH_SIZE
        ),
        "examples_per_epoch": (
            EXAMPLES_PER_EPOCH
        ),
        "batches_per_epoch": (
            BATCHES_PER_EPOCH
        ),
        "final_batch_size": (
            FINAL_BATCH_SIZE
        ),
        "total_optimizer_steps": (
            TOTAL_OPTIMIZER_STEPS
        ),
        "optimizer": {
            "name": (
                "Adam"
            ),
            "lr": (
                0.001
            ),
            "betas": [
                0.9,
                0.999,
            ],
            "eps": (
                1e-8
            ),
            "weight_decay": (
                0.0
            ),
            "amsgrad": (
                False
            ),
            "scheduler": (
                None
            ),
        },
        "training_stream": {
            "positive_order_sha256": (
                EXPECTED_POSITIVE_ORDER_SHA256
            ),
            "negative_dtype": (
                "int32"
            ),
            "negative_dtype_str": (
                "<i4"
            ),
            "epoch0": {
                "negative_sha256": (
                    EXPECTED_EPOCH0_NEGATIVE_SHA256
                ),
                "order_sha256": (
                    EXPECTED_EPOCH0_ORDER_SHA256
                ),
            },
            "epoch1": {
                "negative_sha256": (
                    EXPECTED_EPOCH1_NEGATIVE_SHA256
                ),
                "order_sha256": (
                    EXPECTED_EPOCH1_ORDER_SHA256
                ),
            },
            "epoch19": {
                "negative_sha256": (
                    EXPECTED_EPOCH19_NEGATIVE_SHA256
                ),
                "order_sha256": (
                    EXPECTED_EPOCH19_ORDER_SHA256
                ),
            },
        },
        "evaluation": {
            "validation_every_epoch": (
                True
            ),
            "validation_cases": (
                VALIDATION_CASES
            ),
            "test_cases": (
                TEST_CASES
            ),
            "candidates_per_case": (
                CANDIDATES_PER_EVALUATION_CASE
            ),
            "negatives_per_case": (
                NEGATIVES_PER_EVALUATION_CASE
            ),
            "metrics": [
                "HR@10",
                "NDCG@10",
            ],
            "best_checkpoint_primary": (
                "NDCG@10"
            ),
            "best_checkpoint_secondary": (
                "HR@10"
            ),
            "full_tie": (
                "earliest epoch"
            ),
            "test_once_after_training": (
                True
            ),
        },
        "reporting": {
            "per_epoch": [
                "display_epoch",
                "mean_training_BCE",
                "validation_HR@10",
                "validation_NDCG@10",
                "is_new_best_checkpoint",
                "best_validation_epoch_so_far",
            ],
            "first_real_trained_metrics_after_display_epoch": (
                1
            ),
            "final_after_training": [
                "best_validation_epoch",
                "best_validation_HR@10",
                "best_validation_NDCG@10",
                "final_test_HR@10",
                "final_test_NDCG@10",
            ],
        },
        "launch_gate": {
            "phase_5_3_7_required": (
                True
            ),
            "training_allowed_now": (
                False
            ),
        },
    }

    LAUNCH_CONFIG_PATH.write_text(
        json.dumps(
            launch_config,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    contract = {
        "phase": (
            "5.3.6"
        ),
        "title": (
            "Production Training Launch Contract"
        ),
        "status": (
            "FROZEN"
        ),
        "production_schedule": {
            "epochs": (
                NUM_EPOCHS
            ),
            "examples_per_epoch": (
                EXAMPLES_PER_EPOCH
            ),
            "batches_per_epoch": (
                BATCHES_PER_EPOCH
            ),
            "total_optimizer_steps": (
                TOTAL_OPTIMIZER_STEPS
            ),
        },
        "metric_availability": {
            "first_real_training_loss": (
                "after epoch_index 0 completes"
            ),
            "first_real_validation_HR@10": (
                "after epoch_index 0 validation"
            ),
            "first_real_validation_NDCG@10": (
                "after epoch_index 0 validation"
            ),
            "display_epoch_for_first_metrics": (
                1
            ),
            "final_test_metrics": (
                "after epoch_index 19 validation commit "
                "using best checkpoint"
            ),
        },
        "test_integrity": {
            "test_forbidden_during_training": (
                True
            ),
            "test_forbidden_during_checkpoint_selection": (
                True
            ),
            "test_once_after_training_complete": (
                True
            ),
        },
        "boundary": {
            "model_instantiated": (
                False
            ),
            "optimizer_instantiated": (
                False
            ),
            "optimizer_steps": (
                0
            ),
            "validation_cases_scored": (
                0
            ),
            "test_cases_scored": (
                0
            ),
        },
        "next_phase": {
            "id": (
                "5.3.7"
            ),
            "title": (
                "Full Reproduction Integrity Closure Audit"
            ),
            "training_allowed_before_pass": (
                False
            ),
        },
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
            "5.3.6"
        ),
        "status": (
            "PRODUCTION_TRAINING_LAUNCH_CONTRACT_FROZEN"
        ),
        "epochs": (
            NUM_EPOCHS
        ),
        "total_optimizer_steps": (
            TOTAL_OPTIMIZER_STEPS
        ),
        "first_trained_validation_metrics_after_display_epoch": (
            1
        ),
        "test_cases_scored": (
            0
        ),
        "training_steps_executed": (
            0
        ),
        "phase_5_3_7_required_before_launch": (
            True
        ),
        "launch_config": str(
            LAUNCH_CONFIG_PATH
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
        PREREQUISITE_AUDIT_PATH,
        PRODUCTION_SCHEDULE_PATH,
        METRIC_REPORTING_PATH,
        TEST_GUARD_PATH,
        FINAL_INVARIANT_PATH,
        DECISION_REGISTER_PATH,
        LAUNCH_CONFIG_PATH,
        CONTRACT_PATH,
        MANIFEST_PATH,
    ):
        print(
            f"WROTE  {path}"
        )

    banner(
        "PHASE 5.3.6 FINAL STATUS"
    )

    print(
        "Production epochs:                    20"
    )
    print(
        "Total optimizer steps:                209,620"
    )
    print(
        "Validation after every epoch:         YES"
    )
    print(
        "First REAL trained metrics:           AFTER DISPLAY EPOCH 1"
    )
    print(
        "Final test metrics:                   AFTER ALL 20 EPOCHS"
    )
    print(
        "Test used for model selection:        NO"
    )
    print()
    print(
        "Model instantiated here:              NO"
    )
    print(
        "Training executed here:               NO"
    )
    print(
        "Test cases scored here:               0"
    )
    print()
    print(
        "Phase 5.4 launch status:              "
        "LOCKED PENDING PHASE 5.3.7"
    )

    banner(
        "PHASE 5.3.6 COMPLETE / "
        "PRODUCTION TRAINING LAUNCH CONTRACT FROZEN"
    )


if __name__ == "__main__":
    main()