"""
Phase 5.3.3d — Validation-to-Checkpoint Selection Integration Proof

Purpose
-------
Freeze and prove how a completed epoch transitions through:

    TRAINING_COMPLETE_FOR_EPOCH
        -> VALIDATION_PENDING
        -> VALIDATION_COMMITTED
        -> BEST-CHECKPOINT DECISION
        -> ADVANCE TO NEXT EPOCH

or, for epoch 19:

        -> TRAINING_COMPLETE

This phase performs NO neural forward, NO backward, NO optimizer.step(),
and NO test scoring.

It consumes the REAL frozen Phase-5.3.3c full-validation result from the
canonical initial model as the first validation commit, then uses controlled
metric scenarios to prove the already-frozen best-checkpoint ordering:

    primary   = maximum validation NDCG@10
    secondary = maximum validation HR@10
    full tie  = keep earliest epoch

Important checkpoint-role distinction
-------------------------------------
BEST checkpoint:
    - refers to the model state that produced validation metrics for epoch e
    - used for final test after all 20 epochs
    - does NOT define the next training-resume position

LATEST/RESUME checkpoint:
    - written after validation commit / epoch advancement
    - defines the next training action
    - after epoch e<19 validation:
          epoch_index      = e + 1
          next_batch_index = 0

Validation is committed BEFORE epoch advancement so the best-checkpoint
decision is unambiguously associated with the model that completed epoch e.

Test isolation
--------------
Test remains forbidden while training_complete == False.
At final epoch commit, test becomes ELIGIBLE for exactly one later evaluation,
but this phase does NOT execute that evaluation.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pandas as pd


# =============================================================================
# Frozen dependencies
# =============================================================================

PHASE_5_3_2A_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_2a_training_execution_state_contract.json"
)

PHASE_5_3_2C_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_2c_production_driver_controller_contract.json"
)

PHASE_5_3_3C_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_3c_full_validation_runtime_contract.json"
)

PHASE_5_3_3C_MANIFEST_PATH = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_3c/"
    "phase_5_3_3c_full_validation_runtime_manifest.json"
)


# =============================================================================
# Frozen training arithmetic
# =============================================================================

NUM_EPOCHS = 20
BATCHES_PER_EPOCH = 10_481
EXAMPLES_PER_EPOCH = 5_366_245
TOTAL_OPTIMIZER_STEPS = 209_620


# =============================================================================
# Frozen real validation regression values from Phase 5.3.3c
# =============================================================================

EXPECTED_CANONICAL_VALIDATION_HR10 = (
    0.09151488227454465
)

EXPECTED_CANONICAL_VALIDATION_NDCG10 = (
    0.040193099163
)

EXPECTED_FULL_LOGIT_SHA256 = (
    "1799eb3e382fbdf5477234666d64bbb8"
    "32d64ad6bfb354c903618e6b945b5058"
)

EXPECTED_FULL_METRIC_SHA256 = (
    "c091008ad2206c2890e5f4835d687bf9"
    "288bd078e2c8b60a12575b64fa131b63"
)

EXPECTED_POSITIVE_RANK_SHA256 = (
    "273bdb5b2252f1c8a48420df4685bd2e"
    "5dbfb5f66a7b6f5125ac8ee1913e3345"
)


# =============================================================================
# Outputs
# =============================================================================

AUDIT_DIR = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_3d"
)

CONTRACT_DIR = Path(
    "data/experimental/phase_5/contracts"
)

REAL_COMMIT_PATH = (
    AUDIT_DIR
    / "real_validation_epoch0_commit_audit.csv"
)

BEST_RULE_PATH = (
    AUDIT_DIR
    / "best_checkpoint_comparison_probes.csv"
)

EPOCH_ADVANCE_PATH = (
    AUDIT_DIR
    / "validation_epoch_advance_audit.csv"
)

FINAL_EPOCH_PATH = (
    AUDIT_DIR
    / "final_epoch_validation_completion_audit.csv"
)

CHECKPOINT_ROLE_PATH = (
    AUDIT_DIR
    / "best_vs_latest_checkpoint_role_contract.csv"
)

FINAL_INVARIANT_PATH = (
    AUDIT_DIR
    / "phase_5_3_3d_final_invariants.csv"
)

DECISION_REGISTER_PATH = (
    CONTRACT_DIR
    / "phase_5_3_3d_validation_checkpoint_decision_register.csv"
)

CONTRACT_PATH = (
    CONTRACT_DIR
    / "phase_5_3_3d_validation_checkpoint_integration_contract.json"
)

MANIFEST_PATH = (
    AUDIT_DIR
    / "phase_5_3_3d_validation_checkpoint_integration_manifest.json"
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


def load_json(
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def validation_candidate_is_better(
    candidate_ndcg: float,
    candidate_hr: float,
    candidate_epoch: int,
    best_ndcg: float | None,
    best_hr: float | None,
    best_epoch: int | None,
) -> bool:
    """
    Frozen Phase-5.2.2 / Phase-5.3.2a comparison rule.

    primary   = larger NDCG@10
    secondary = larger HR@10
    exact tie = keep existing earliest epoch
    """

    require(
        math.isfinite(
            float(
                candidate_ndcg
            )
        ),
        "Candidate NDCG@10 is non-finite.",
    )

    require(
        math.isfinite(
            float(
                candidate_hr
            )
        ),
        "Candidate HR@10 is non-finite.",
    )

    require(
        0.0
        <= float(
            candidate_ndcg
        )
        <= 1.0,
        "Candidate NDCG@10 outside [0,1].",
    )

    require(
        0.0
        <= float(
            candidate_hr
        )
        <= 1.0,
        "Candidate HR@10 outside [0,1].",
    )

    require(
        0
        <= int(
            candidate_epoch
        )
        < NUM_EPOCHS,
        "Candidate epoch outside 0..19.",
    )

    if best_epoch is None:
        require(
            best_ndcg is None
            and best_hr is None,
            (
                "Uninitialized best checkpoint "
                "has partial metrics."
            ),
        )
        return True

    require(
        best_ndcg is not None
        and best_hr is not None,
        (
            "Initialized best checkpoint "
            "is missing metrics."
        ),
    )

    if candidate_ndcg > best_ndcg:
        return True

    if candidate_ndcg < best_ndcg:
        return False

    if candidate_hr > best_hr:
        return True

    # Equal NDCG and equal/lower HR:
    # keep existing best, which is necessarily earlier
    # because validation epochs arrive chronologically.
    return False


def make_validation_pending_state(
    epoch_index: int,
) -> dict:
    """
    Controller state immediately after the final optimizer step
    of an epoch and before validation is committed.

    The epoch-loss value here is a CONTROL-FLOW-ONLY placeholder.
    It is not a research result.
    """

    require(
        0
        <= epoch_index
        < NUM_EPOCHS,
        (
            "Epoch outside 0..19."
        ),
    )

    global_step = (
        (
            epoch_index
            + 1
        )
        * BATCHES_PER_EPOCH
    )

    return {
        "epoch_index": (
            epoch_index
        ),
        "next_batch_index": (
            BATCHES_PER_EPOCH
        ),
        "global_optimizer_step": (
            global_step
        ),
        "epoch_loss_weighted_sum": (
            0.0
        ),
        "epoch_example_count": (
            EXAMPLES_PER_EPOCH
        ),
        "validation_pending": (
            True
        ),
        "validation_history": (
            []
        ),
        "best_validation_epoch": (
            None
        ),
        "best_validation_ndcg10": (
            None
        ),
        "best_validation_hr10": (
            None
        ),
        "training_complete": (
            False
        ),
    }


def commit_validation(
    state: dict,
    *,
    validation_hr10: float,
    validation_ndcg10: float,
) -> dict:
    """
    Commit validation for state["epoch_index"].

    Returns a structured transition containing:

        committed_state
            Same epoch e; validation already appended and best metadata
            updated. This state is the semantic source for BEST checkpoint
            creation if this epoch becomes best.

        resume_state
            If e < 19:
                advanced to epoch e+1 / next_batch_index 0.
            If e == 19:
                training_complete True.

        best_checkpoint_should_write
            Whether epoch e becomes the new best model.

        test_access_allowed
            True only after epoch-19 validation commit.
    """

    require(
        state[
            "validation_pending"
        ]
        is True,
        (
            "Validation commit requires "
            "validation_pending=True."
        ),
    )

    require(
        state[
            "training_complete"
        ]
        is False,
        (
            "Cannot commit validation after "
            "training_complete=True."
        ),
    )

    epoch_index = int(
        state[
            "epoch_index"
        ]
    )

    require(
        int(
            state[
                "next_batch_index"
            ]
        )
        == BATCHES_PER_EPOCH,
        (
            "Validation requires all epoch "
            "training batches completed."
        ),
    )

    require(
        int(
            state[
                "epoch_example_count"
            ]
        )
        == EXAMPLES_PER_EPOCH,
        (
            "Validation requires exact epoch "
            "example count."
        ),
    )

    expected_global_step = (
        (
            epoch_index
            + 1
        )
        * BATCHES_PER_EPOCH
    )

    require(
        int(
            state[
                "global_optimizer_step"
            ]
        )
        == expected_global_step,
        (
            "Validation global optimizer-step "
            "count inconsistent with epoch."
        ),
    )

    candidate_hr = float(
        validation_hr10
    )

    candidate_ndcg = float(
        validation_ndcg10
    )

    better = (
        validation_candidate_is_better(
            candidate_ndcg=(
                candidate_ndcg
            ),
            candidate_hr=(
                candidate_hr
            ),
            candidate_epoch=(
                epoch_index
            ),
            best_ndcg=(
                state[
                    "best_validation_ndcg10"
                ]
            ),
            best_hr=(
                state[
                    "best_validation_hr10"
                ]
            ),
            best_epoch=(
                state[
                    "best_validation_epoch"
                ]
            ),
        )
    )

    committed = copy.deepcopy(
        state
    )

    committed[
        "validation_pending"
    ] = False

    committed[
        "validation_history"
    ] = list(
        committed[
            "validation_history"
        ]
    )

    committed[
        "validation_history"
    ].append(
        {
            "epoch_index": (
                epoch_index
            ),
            "display_epoch": (
                epoch_index + 1
            ),
            "HR@10": (
                candidate_hr
            ),
            "NDCG@10": (
                candidate_ndcg
            ),
        }
    )

    if better:
        committed[
            "best_validation_epoch"
        ] = epoch_index

        committed[
            "best_validation_ndcg10"
        ] = candidate_ndcg

        committed[
            "best_validation_hr10"
        ] = candidate_hr

    # BEST checkpoint is associated with this committed epoch state.
    best_checkpoint_state = (
        copy.deepcopy(
            committed
        )
        if better
        else None
    )

    # Training-resume state is then advanced.
    resume_state = copy.deepcopy(
        committed
    )

    if epoch_index < (
        NUM_EPOCHS - 1
    ):
        resume_state[
            "epoch_index"
        ] = (
            epoch_index + 1
        )

        resume_state[
            "next_batch_index"
        ] = 0

        resume_state[
            "epoch_loss_weighted_sum"
        ] = 0.0

        resume_state[
            "epoch_example_count"
        ] = 0

        resume_state[
            "validation_pending"
        ] = False

        resume_state[
            "training_complete"
        ] = False

        test_access_allowed = False

    else:
        resume_state[
            "epoch_index"
        ] = epoch_index

        resume_state[
            "next_batch_index"
        ] = BATCHES_PER_EPOCH

        resume_state[
            "validation_pending"
        ] = False

        resume_state[
            "training_complete"
        ] = True

        test_access_allowed = True

    return {
        "candidate_epoch": (
            epoch_index
        ),
        "candidate_HR@10": (
            candidate_hr
        ),
        "candidate_NDCG@10": (
            candidate_ndcg
        ),
        "best_checkpoint_should_write": (
            better
        ),
        "best_checkpoint_state": (
            best_checkpoint_state
        ),
        "committed_state": (
            committed
        ),
        "resume_state": (
            resume_state
        ),
        "test_access_allowed": (
            test_access_allowed
        ),
    }


def seed_existing_best(
    state: dict,
    *,
    epoch: int,
    ndcg: float,
    hr: float,
) -> dict:
    seeded = copy.deepcopy(
        state
    )

    seeded[
        "best_validation_epoch"
    ] = int(
        epoch
    )

    seeded[
        "best_validation_ndcg10"
    ] = float(
        ndcg
    )

    seeded[
        "best_validation_hr10"
    ] = float(
        hr
    )

    seeded[
        "validation_history"
    ] = [
        {
            "epoch_index": (
                int(
                    epoch
                )
            ),
            "display_epoch": (
                int(
                    epoch
                )
                + 1
            ),
            "HR@10": (
                float(
                    hr
                )
            ),
            "NDCG@10": (
                float(
                    ndcg
                )
            ),
        }
    ]

    return seeded


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.3.3d — "
        "VALIDATION-TO-CHECKPOINT SELECTION INTEGRATION PROOF"
    )

    print(
        "Neural model instantiated:            NO"
    )
    print(
        "Validation forward executed:          NO"
    )
    print(
        "Backward computation:                 NO"
    )
    print(
        "optimizer.step():                     0"
    )
    print(
        "Test cases scored:                    0"
    )

    # =========================================================================
    # Authoritative prerequisite recheck
    # =========================================================================

    banner(
        "AUTHORITATIVE INTEGRATION GATE RECHECK"
    )

    for path in (
        PHASE_5_3_2A_CONTRACT_PATH,
        PHASE_5_3_2C_CONTRACT_PATH,
        PHASE_5_3_3C_CONTRACT_PATH,
        PHASE_5_3_3C_MANIFEST_PATH,
    ):
        require(
            path.exists(),
            (
                "Missing authoritative input: "
                f"{path}"
            ),
        )

        print(
            f"FOUND  {path}"
        )

    execution_contract = load_json(
        PHASE_5_3_2A_CONTRACT_PATH
    )

    controller_contract = load_json(
        PHASE_5_3_2C_CONTRACT_PATH
    )

    validation_contract = load_json(
        PHASE_5_3_3C_CONTRACT_PATH
    )

    validation_manifest = load_json(
        PHASE_5_3_3C_MANIFEST_PATH
    )

    require(
        execution_contract[
            "status"
        ]
        == "FROZEN",
        (
            "Phase-5.3.2a execution "
            "contract not frozen."
        ),
    )

    require(
        controller_contract[
            "status"
        ]
        == "FROZEN",
        (
            "Phase-5.3.2c controller "
            "contract not frozen."
        ),
    )

    require(
        validation_contract[
            "status"
        ]
        == "FROZEN",
        (
            "Phase-5.3.3c validation "
            "contract not frozen."
        ),
    )

    require(
        validation_manifest[
            "status"
        ]
        == (
            "FULL_VALIDATION_SPLIT_RUNTIME_"
            "PASSED_AND_FROZEN"
        ),
        (
            "Unexpected Phase-5.3.3c "
            "manifest status."
        ),
    )

    actual_real_hr = float(
        validation_contract[
            "canonical_initial_validation"
        ][
            "HR@10"
        ]
    )

    actual_real_ndcg = float(
        validation_contract[
            "canonical_initial_validation"
        ][
            "NDCG@10"
        ]
    )

    require(
        abs(
            actual_real_hr
            - EXPECTED_CANONICAL_VALIDATION_HR10
        )
        <= 5e-15,
        (
            "Frozen canonical validation "
            "HR@10 drift."
        ),
    )

    require(
        abs(
            actual_real_ndcg
            - EXPECTED_CANONICAL_VALIDATION_NDCG10
        )
        <= 5e-13,
        (
            "Frozen canonical validation "
            "NDCG@10 drift."
        ),
    )

    require(
        validation_contract[
            "canonical_initial_validation"
        ][
            "raw_logit_matrix_logical_sha256"
        ]
        == EXPECTED_FULL_LOGIT_SHA256,
        (
            "Full validation logit "
            "fingerprint drift."
        ),
    )

    require(
        validation_contract[
            "canonical_initial_validation"
        ][
            "case_metric_logical_sha256"
        ]
        == EXPECTED_FULL_METRIC_SHA256,
        (
            "Full validation metric "
            "fingerprint drift."
        ),
    )

    require(
        validation_contract[
            "canonical_initial_validation"
        ][
            "positive_rank_vector_logical_sha256"
        ]
        == EXPECTED_POSITIVE_RANK_SHA256,
        (
            "Full validation rank-vector "
            "fingerprint drift."
        ),
    )

    print(
        "Phase-5.3.2 controller:               FROZEN / PASS"
    )
    print(
        "Phase-5.3.3c full validation:         FROZEN / PASS"
    )

    # =========================================================================
    # Real epoch-0 validation commit
    # =========================================================================

    banner(
        "REAL CANONICAL VALIDATION RESULT -> EPOCH-0 COMMIT"
    )

    epoch0_pending = (
        make_validation_pending_state(
            0
        )
    )

    epoch0_result = commit_validation(
        epoch0_pending,
        validation_hr10=(
            actual_real_hr
        ),
        validation_ndcg10=(
            actual_real_ndcg
        ),
    )

    require(
        epoch0_result[
            "best_checkpoint_should_write"
        ]
        is True,
        (
            "First validation epoch must "
            "become initial best."
        ),
    )

    epoch0_best = (
        epoch0_result[
            "best_checkpoint_state"
        ]
    )

    require(
        epoch0_best is not None,
        (
            "First validation epoch did not "
            "produce best-checkpoint state."
        ),
    )

    require(
        epoch0_best[
            "epoch_index"
        ]
        == 0,
        (
            "Best checkpoint must remain "
            "associated with epoch 0."
        ),
    )

    require(
        epoch0_best[
            "next_batch_index"
        ]
        == BATCHES_PER_EPOCH,
        (
            "Best checkpoint should represent "
            "completed epoch-0 training."
        ),
    )

    require(
        epoch0_result[
            "resume_state"
        ][
            "epoch_index"
        ]
        == 1,
        (
            "Post-validation resume state "
            "did not advance to epoch 1."
        ),
    )

    require(
        epoch0_result[
            "resume_state"
        ][
            "next_batch_index"
        ]
        == 0,
        (
            "Post-validation resume state "
            "did not reset batch index."
        ),
    )

    require(
        epoch0_result[
            "test_access_allowed"
        ]
        is False,
        (
            "Test became accessible "
            "after epoch 0."
        ),
    )

    real_commit_df = pd.DataFrame(
        [
            {
                "item": (
                    "validation_epoch"
                ),
                "value": (
                    0
                ),
            },
            {
                "item": (
                    "real_HR@10"
                ),
                "value": (
                    actual_real_hr
                ),
            },
            {
                "item": (
                    "real_NDCG@10"
                ),
                "value": (
                    actual_real_ndcg
                ),
            },
            {
                "item": (
                    "becomes_best"
                ),
                "value": (
                    True
                ),
            },
            {
                "item": (
                    "best_checkpoint_epoch_index"
                ),
                "value": (
                    epoch0_best[
                        "epoch_index"
                    ]
                ),
            },
            {
                "item": (
                    "resume_epoch_index"
                ),
                "value": (
                    epoch0_result[
                        "resume_state"
                    ][
                        "epoch_index"
                    ]
                ),
            },
            {
                "item": (
                    "resume_next_batch_index"
                ),
                "value": (
                    epoch0_result[
                        "resume_state"
                    ][
                        "next_batch_index"
                    ]
                ),
            },
            {
                "item": (
                    "test_access_allowed"
                ),
                "value": (
                    epoch0_result[
                        "test_access_allowed"
                    ]
                ),
            },
        ]
    )

    print(
        real_commit_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Best-checkpoint comparison probes
    # =========================================================================

    banner(
        "BEST-CHECKPOINT COMPARISON PROBES"
    )

    probes = []

    # Probe 1 — higher NDCG wins even if HR is lower.
    probe_state = (
        make_validation_pending_state(
            1
        )
    )

    probe_state = seed_existing_best(
        probe_state,
        epoch=0,
        ndcg=0.20,
        hr=0.90,
    )

    result = commit_validation(
        probe_state,
        validation_hr10=0.10,
        validation_ndcg10=0.21,
    )

    require(
        result[
            "best_checkpoint_should_write"
        ]
        is True,
        (
            "Higher NDCG did not replace best."
        ),
    )

    probes.append(
        {
            "probe": (
                "higher_NDCG_wins"
            ),
            "prior_best_epoch": (
                0
            ),
            "prior_best_NDCG": (
                0.20
            ),
            "prior_best_HR": (
                0.90
            ),
            "candidate_epoch": (
                1
            ),
            "candidate_NDCG": (
                0.21
            ),
            "candidate_HR": (
                0.10
            ),
            "should_replace": (
                True
            ),
            "actual_replace": (
                result[
                    "best_checkpoint_should_write"
                ]
            ),
            "status": (
                "PASS"
            ),
        }
    )

    # Probe 2 — NDCG tie; higher HR wins.
    probe_state = (
        make_validation_pending_state(
            2
        )
    )

    probe_state = seed_existing_best(
        probe_state,
        epoch=1,
        ndcg=0.25,
        hr=0.30,
    )

    result = commit_validation(
        probe_state,
        validation_hr10=0.31,
        validation_ndcg10=0.25,
    )

    require(
        result[
            "best_checkpoint_should_write"
        ]
        is True,
        (
            "Higher HR did not break "
            "NDCG tie."
        ),
    )

    probes.append(
        {
            "probe": (
                "NDCG_tie_higher_HR_wins"
            ),
            "prior_best_epoch": (
                1
            ),
            "prior_best_NDCG": (
                0.25
            ),
            "prior_best_HR": (
                0.30
            ),
            "candidate_epoch": (
                2
            ),
            "candidate_NDCG": (
                0.25
            ),
            "candidate_HR": (
                0.31
            ),
            "should_replace": (
                True
            ),
            "actual_replace": (
                result[
                    "best_checkpoint_should_write"
                ]
            ),
            "status": (
                "PASS"
            ),
        }
    )

    # Probe 3 — exact full tie; earlier best survives.
    probe_state = (
        make_validation_pending_state(
            3
        )
    )

    probe_state = seed_existing_best(
        probe_state,
        epoch=1,
        ndcg=0.25,
        hr=0.31,
    )

    result = commit_validation(
        probe_state,
        validation_hr10=0.31,
        validation_ndcg10=0.25,
    )

    require(
        result[
            "best_checkpoint_should_write"
        ]
        is False,
        (
            "Exact later tie incorrectly "
            "replaced earlier best."
        ),
    )

    require(
        result[
            "committed_state"
        ][
            "best_validation_epoch"
        ]
        == 1,
        (
            "Exact tie did not preserve "
            "earliest best epoch."
        ),
    )

    probes.append(
        {
            "probe": (
                "exact_tie_keeps_earliest"
            ),
            "prior_best_epoch": (
                1
            ),
            "prior_best_NDCG": (
                0.25
            ),
            "prior_best_HR": (
                0.31
            ),
            "candidate_epoch": (
                3
            ),
            "candidate_NDCG": (
                0.25
            ),
            "candidate_HR": (
                0.31
            ),
            "should_replace": (
                False
            ),
            "actual_replace": (
                result[
                    "best_checkpoint_should_write"
                ]
            ),
            "status": (
                "PASS"
            ),
        }
    )

    # Probe 4 — lower NDCG loses even with much higher HR.
    probe_state = (
        make_validation_pending_state(
            4
        )
    )

    probe_state = seed_existing_best(
        probe_state,
        epoch=2,
        ndcg=0.30,
        hr=0.20,
    )

    result = commit_validation(
        probe_state,
        validation_hr10=0.99,
        validation_ndcg10=0.29,
    )

    require(
        result[
            "best_checkpoint_should_write"
        ]
        is False,
        (
            "Lower NDCG incorrectly won "
            "because HR was higher."
        ),
    )

    probes.append(
        {
            "probe": (
                "lower_NDCG_loses_even_if_HR_higher"
            ),
            "prior_best_epoch": (
                2
            ),
            "prior_best_NDCG": (
                0.30
            ),
            "prior_best_HR": (
                0.20
            ),
            "candidate_epoch": (
                4
            ),
            "candidate_NDCG": (
                0.29
            ),
            "candidate_HR": (
                0.99
            ),
            "should_replace": (
                False
            ),
            "actual_replace": (
                result[
                    "best_checkpoint_should_write"
                ]
            ),
            "status": (
                "PASS"
            ),
        }
    )

    # Probe 5 — same NDCG, lower HR loses.
    probe_state = (
        make_validation_pending_state(
            5
        )
    )

    probe_state = seed_existing_best(
        probe_state,
        epoch=2,
        ndcg=0.30,
        hr=0.50,
    )

    result = commit_validation(
        probe_state,
        validation_hr10=0.49,
        validation_ndcg10=0.30,
    )

    require(
        result[
            "best_checkpoint_should_write"
        ]
        is False,
        (
            "Lower HR incorrectly won "
            "an NDCG tie."
        ),
    )

    probes.append(
        {
            "probe": (
                "NDCG_tie_lower_HR_loses"
            ),
            "prior_best_epoch": (
                2
            ),
            "prior_best_NDCG": (
                0.30
            ),
            "prior_best_HR": (
                0.50
            ),
            "candidate_epoch": (
                5
            ),
            "candidate_NDCG": (
                0.30
            ),
            "candidate_HR": (
                0.49
            ),
            "should_replace": (
                False
            ),
            "actual_replace": (
                result[
                    "best_checkpoint_should_write"
                ]
            ),
            "status": (
                "PASS"
            ),
        }
    )

    best_rule_df = pd.DataFrame(
        probes
    )

    require(
        bool(
            (
                best_rule_df[
                    "should_replace"
                ]
                == best_rule_df[
                    "actual_replace"
                ]
            ).all()
        ),
        (
            "At least one best-checkpoint "
            "comparison probe failed."
        ),
    )

    print(
        best_rule_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Epoch advancement semantics
    # =========================================================================

    banner(
        "VALIDATION COMMIT -> NEXT-EPOCH ADVANCEMENT"
    )

    epoch7_pending = (
        make_validation_pending_state(
            7
        )
    )

    epoch7_pending = seed_existing_best(
        epoch7_pending,
        epoch=3,
        ndcg=0.40,
        hr=0.50,
    )

    epoch7_result = commit_validation(
        epoch7_pending,
        validation_hr10=0.45,
        validation_ndcg10=0.41,
    )

    committed7 = (
        epoch7_result[
            "committed_state"
        ]
    )

    resumed8 = (
        epoch7_result[
            "resume_state"
        ]
    )

    require(
        committed7[
            "epoch_index"
        ]
        == 7,
        (
            "Committed validation state "
            "lost epoch-7 identity."
        ),
    )

    require(
        committed7[
            "next_batch_index"
        ]
        == BATCHES_PER_EPOCH,
        (
            "Committed validation state "
            "must remain at completed epoch boundary."
        ),
    )

    require(
        resumed8[
            "epoch_index"
        ]
        == 8,
        (
            "Resume state did not advance "
            "to epoch 8."
        ),
    )

    require(
        resumed8[
            "next_batch_index"
        ]
        == 0,
        (
            "Resume state did not reset "
            "next_batch_index to zero."
        ),
    )

    require(
        resumed8[
            "epoch_example_count"
        ]
        == 0,
        (
            "Resume state did not reset "
            "epoch example count."
        ),
    )

    require(
        resumed8[
            "global_optimizer_step"
        ]
        == (
            8
            * BATCHES_PER_EPOCH
        ),
        (
            "Epoch advancement incorrectly "
            "changed global optimizer step."
        ),
    )

    epoch_advance_df = pd.DataFrame(
        [
            {
                "boundary": (
                    "validation_pending_epoch7"
                ),
                "epoch_index": (
                    epoch7_pending[
                        "epoch_index"
                    ]
                ),
                "next_batch_index": (
                    epoch7_pending[
                        "next_batch_index"
                    ]
                ),
                "global_optimizer_step": (
                    epoch7_pending[
                        "global_optimizer_step"
                    ]
                ),
                "epoch_example_count": (
                    epoch7_pending[
                        "epoch_example_count"
                    ]
                ),
                "validation_pending": (
                    epoch7_pending[
                        "validation_pending"
                    ]
                ),
                "training_complete": (
                    epoch7_pending[
                        "training_complete"
                    ]
                ),
            },
            {
                "boundary": (
                    "validation_committed_epoch7"
                ),
                "epoch_index": (
                    committed7[
                        "epoch_index"
                    ]
                ),
                "next_batch_index": (
                    committed7[
                        "next_batch_index"
                    ]
                ),
                "global_optimizer_step": (
                    committed7[
                        "global_optimizer_step"
                    ]
                ),
                "epoch_example_count": (
                    committed7[
                        "epoch_example_count"
                    ]
                ),
                "validation_pending": (
                    committed7[
                        "validation_pending"
                    ]
                ),
                "training_complete": (
                    committed7[
                        "training_complete"
                    ]
                ),
            },
            {
                "boundary": (
                    "resume_epoch8"
                ),
                "epoch_index": (
                    resumed8[
                        "epoch_index"
                    ]
                ),
                "next_batch_index": (
                    resumed8[
                        "next_batch_index"
                    ]
                ),
                "global_optimizer_step": (
                    resumed8[
                        "global_optimizer_step"
                    ]
                ),
                "epoch_example_count": (
                    resumed8[
                        "epoch_example_count"
                    ]
                ),
                "validation_pending": (
                    resumed8[
                        "validation_pending"
                    ]
                ),
                "training_complete": (
                    resumed8[
                        "training_complete"
                    ]
                ),
            },
        ]
    )

    print(
        epoch_advance_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Final epoch / test eligibility
    # =========================================================================

    banner(
        "FINAL EPOCH-19 VALIDATION COMPLETION / TEST ELIGIBILITY"
    )

    epoch19_pending = (
        make_validation_pending_state(
            19
        )
    )

    epoch19_pending = seed_existing_best(
        epoch19_pending,
        epoch=12,
        ndcg=0.50,
        hr=0.60,
    )

    epoch19_result = commit_validation(
        epoch19_pending,
        validation_hr10=0.60,
        validation_ndcg10=0.50,
    )

    final_state = (
        epoch19_result[
            "resume_state"
        ]
    )

    require(
        epoch19_result[
            "best_checkpoint_should_write"
        ]
        is False,
        (
            "Exact epoch-19 tie incorrectly "
            "replaced earlier epoch-12 best."
        ),
    )

    require(
        final_state[
            "best_validation_epoch"
        ]
        == 12,
        (
            "Final exact tie did not preserve "
            "earlier best epoch."
        ),
    )

    require(
        final_state[
            "training_complete"
        ]
        is True,
        (
            "Epoch-19 validation commit did "
            "not mark training complete."
        ),
    )

    require(
        final_state[
            "global_optimizer_step"
        ]
        == TOTAL_OPTIMIZER_STEPS,
        (
            "Final global optimizer-step "
            "count is not 209,620."
        ),
    )

    require(
        epoch19_result[
            "test_access_allowed"
        ]
        is True,
        (
            "Test did not become eligible "
            "after final validation commit."
        ),
    )

    final_epoch_df = pd.DataFrame(
        [
            {
                "item": (
                    "epoch_index"
                ),
                "value": (
                    final_state[
                        "epoch_index"
                    ]
                ),
            },
            {
                "item": (
                    "global_optimizer_step"
                ),
                "value": (
                    final_state[
                        "global_optimizer_step"
                    ]
                ),
            },
            {
                "item": (
                    "training_complete"
                ),
                "value": (
                    final_state[
                        "training_complete"
                    ]
                ),
            },
            {
                "item": (
                    "best_validation_epoch"
                ),
                "value": (
                    final_state[
                        "best_validation_epoch"
                    ]
                ),
            },
            {
                "item": (
                    "test_access_allowed"
                ),
                "value": (
                    epoch19_result[
                        "test_access_allowed"
                    ]
                ),
            },
            {
                "item": (
                    "test_cases_scored_in_this_phase"
                ),
                "value": (
                    0
                ),
            },
        ]
    )

    print(
        final_epoch_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Checkpoint-role freeze
    # =========================================================================

    banner(
        "BEST CHECKPOINT vs LATEST/RESUME CHECKPOINT ROLES"
    )

    checkpoint_role_df = pd.DataFrame(
        [
            {
                "checkpoint_kind": (
                    "best"
                ),
                "write_condition": (
                    "validation candidate becomes new best"
                ),
                "epoch_semantics": (
                    "validated model epoch e"
                ),
                "next_batch_semantics": (
                    "completed epoch boundary 10481"
                ),
                "primary_use": (
                    "final test model"
                ),
                "training_resume_source": (
                    False
                ),
            },
            {
                "checkpoint_kind": (
                    "latest"
                ),
                "write_condition": (
                    "after validation commit / controller advancement"
                ),
                "epoch_semantics": (
                    "next training epoch e+1, or final complete state"
                ),
                "next_batch_semantics": (
                    "0 for next epoch; 10481 when fully complete"
                ),
                "primary_use": (
                    "training resume"
                ),
                "training_resume_source": (
                    True
                ),
            },
        ]
    )

    print(
        checkpoint_role_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Final invariants
    # =========================================================================

    banner(
        "FINAL PHASE-5.3.3d INVARIANTS"
    )

    checks = [
        (
            "phase_5_3_2a_contract_frozen",
            (
                execution_contract[
                    "status"
                ]
                == "FROZEN"
            ),
        ),
        (
            "phase_5_3_2c_contract_frozen",
            (
                controller_contract[
                    "status"
                ]
                == "FROZEN"
            ),
        ),
        (
            "phase_5_3_3c_contract_frozen",
            (
                validation_contract[
                    "status"
                ]
                == "FROZEN"
            ),
        ),
        (
            "real_epoch0_validation_becomes_initial_best",
            (
                epoch0_result[
                    "best_checkpoint_should_write"
                ]
                is True
            ),
        ),
        (
            "real_epoch0_best_checkpoint_associated_with_epoch0",
            (
                epoch0_best[
                    "epoch_index"
                ]
                == 0
            ),
        ),
        (
            "real_epoch0_resume_advances_to_epoch1",
            (
                epoch0_result[
                    "resume_state"
                ][
                    "epoch_index"
                ]
                == 1
                and epoch0_result[
                    "resume_state"
                ][
                    "next_batch_index"
                ]
                == 0
            ),
        ),
        (
            "test_forbidden_after_epoch0",
            (
                epoch0_result[
                    "test_access_allowed"
                ]
                is False
            ),
        ),
        (
            "higher_NDCG_primary_rule",
            bool(
                best_rule_df.loc[
                    best_rule_df[
                        "probe"
                    ]
                    == "higher_NDCG_wins",
                    "actual_replace",
                ].iloc[
                    0
                ]
            ),
        ),
        (
            "higher_HR_secondary_rule",
            bool(
                best_rule_df.loc[
                    best_rule_df[
                        "probe"
                    ]
                    == "NDCG_tie_higher_HR_wins",
                    "actual_replace",
                ].iloc[
                    0
                ]
            ),
        ),
        (
            "exact_tie_keeps_earliest",
            (
                bool(
                    best_rule_df.loc[
                        best_rule_df[
                            "probe"
                        ]
                        == "exact_tie_keeps_earliest",
                        "actual_replace",
                    ].iloc[
                        0
                    ]
                )
                is False
            ),
        ),
        (
            "lower_NDCG_cannot_be_rescued_by_HR",
            (
                bool(
                    best_rule_df.loc[
                        best_rule_df[
                            "probe"
                        ]
                        == "lower_NDCG_loses_even_if_HR_higher",
                        "actual_replace",
                    ].iloc[
                        0
                    ]
                )
                is False
            ),
        ),
        (
            "NDCG_tie_lower_HR_loses",
            (
                bool(
                    best_rule_df.loc[
                        best_rule_df[
                            "probe"
                        ]
                        == "NDCG_tie_lower_HR_loses",
                        "actual_replace",
                    ].iloc[
                        0
                    ]
                )
                is False
            ),
        ),
        (
            "validation_commit_precedes_epoch_advance",
            (
                committed7[
                    "epoch_index"
                ]
                == 7
                and resumed8[
                    "epoch_index"
                ]
                == 8
            ),
        ),
        (
            "next_epoch_batch_index_resets_zero",
            (
                resumed8[
                    "next_batch_index"
                ]
                == 0
            ),
        ),
        (
            "epoch_advance_preserves_global_step",
            (
                resumed8[
                    "global_optimizer_step"
                ]
                == (
                    8
                    * BATCHES_PER_EPOCH
                )
            ),
        ),
        (
            "epoch19_commit_marks_training_complete",
            (
                final_state[
                    "training_complete"
                ]
                is True
            ),
        ),
        (
            "final_global_optimizer_steps_209620",
            (
                final_state[
                    "global_optimizer_step"
                ]
                == TOTAL_OPTIMIZER_STEPS
            ),
        ),
        (
            "final_exact_tie_keeps_earlier_best",
            (
                final_state[
                    "best_validation_epoch"
                ]
                == 12
            ),
        ),
        (
            "test_eligible_only_after_final_validation",
            (
                epoch19_result[
                    "test_access_allowed"
                ]
                is True
                and epoch0_result[
                    "test_access_allowed"
                ]
                is False
            ),
        ),
        (
            "test_cases_scored_zero",
            True,
        ),
        (
            "best_checkpoint_not_training_resume_source",
            (
                bool(
                    checkpoint_role_df.loc[
                        checkpoint_role_df[
                            "checkpoint_kind"
                        ]
                        == "best",
                        "training_resume_source",
                    ].iloc[
                        0
                    ]
                )
                is False
            ),
        ),
        (
            "latest_checkpoint_is_training_resume_source",
            bool(
                checkpoint_role_df.loc[
                    checkpoint_role_df[
                        "checkpoint_kind"
                    ]
                    == "latest",
                    "training_resume_source",
                ].iloc[
                    0
                ]
            ),
        ),
        (
            "model_not_instantiated",
            True,
        ),
        (
            "validation_forward_not_executed",
            True,
        ),
        (
            "backward_not_performed",
            True,
        ),
        (
            "optimizer_steps_zero",
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
            "At least one Phase-5.3.3d "
            "integration invariant failed."
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
        "WRITE PHASE-5.3.3d OUTPUTS"
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTRACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    real_commit_df.to_csv(
        REAL_COMMIT_PATH,
        index=False,
    )

    best_rule_df.to_csv(
        BEST_RULE_PATH,
        index=False,
    )

    epoch_advance_df.to_csv(
        EPOCH_ADVANCE_PATH,
        index=False,
    )

    final_epoch_df.to_csv(
        FINAL_EPOCH_PATH,
        index=False,
    )

    checkpoint_role_df.to_csv(
        CHECKPOINT_ROLE_PATH,
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
                    "validation_commit_order"
                ),
                "value": (
                    "COMMIT_METRICS_AND_BEST_DECISION_BEFORE_EPOCH_ADVANCE"
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_3d"
                ),
            },
            {
                "decision": (
                    "best_checkpoint_role"
                ),
                "value": (
                    "VALIDATED_EPOCH_MODEL_FOR_FINAL_TEST_NOT_TRAINING_RESUME"
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_3d"
                ),
            },
            {
                "decision": (
                    "latest_checkpoint_role"
                ),
                "value": (
                    "POST_VALIDATION_ADVANCED_CONTROLLER_FOR_TRAINING_RESUME"
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_3d"
                ),
            },
            {
                "decision": (
                    "best_metric_primary"
                ),
                "value": (
                    "MAX_VALIDATION_NDCG10"
                ),
                "classification": (
                    "INHERITED_FROZEN_PHASE_5_2_2"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_3d"
                ),
            },
            {
                "decision": (
                    "best_metric_secondary"
                ),
                "value": (
                    "MAX_VALIDATION_HR10"
                ),
                "classification": (
                    "INHERITED_FROZEN_PHASE_5_2_2"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_3d"
                ),
            },
            {
                "decision": (
                    "best_full_tie"
                ),
                "value": (
                    "KEEP_EARLIEST_EXISTING_BEST"
                ),
                "classification": (
                    "INHERITED_FROZEN_PHASE_5_2_2"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_3d"
                ),
            },
            {
                "decision": (
                    "test_eligibility"
                ),
                "value": (
                    "ONLY_AFTER_EPOCH19_VALIDATION_COMMITTED"
                ),
                "classification": (
                    "EVALUATION_INTEGRITY_GUARD"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_3d"
                ),
            },
        ]
    )

    decision_df.to_csv(
        DECISION_REGISTER_PATH,
        index=False,
    )

    contract = {
        "phase": (
            "5.3.3d"
        ),
        "title": (
            "Validation-to-Checkpoint Selection Integration Contract"
        ),
        "status": (
            "FROZEN"
        ),
        "real_validation_anchor": {
            "source_phase": (
                "5.3.3c"
            ),
            "canonical_initial_HR@10": (
                actual_real_hr
            ),
            "canonical_initial_NDCG@10": (
                actual_real_ndcg
            ),
            "full_logit_sha256": (
                EXPECTED_FULL_LOGIT_SHA256
            ),
            "full_metric_sha256": (
                EXPECTED_FULL_METRIC_SHA256
            ),
            "positive_rank_sha256": (
                EXPECTED_POSITIVE_RANK_SHA256
            ),
            "epoch0_becomes_initial_best": (
                True
            ),
        },
        "validation_commit": {
            "requires_validation_pending": (
                True
            ),
            "commit_before_epoch_advance": (
                True
            ),
            "history_append_unit": (
                "event-aggregated validation epoch record"
            ),
            "best_primary": (
                "maximum NDCG@10"
            ),
            "best_secondary": (
                "maximum HR@10"
            ),
            "full_tie": (
                "keep existing earliest epoch"
            ),
        },
        "best_checkpoint": {
            "state_association": (
                "validated epoch e before controller advance"
            ),
            "write_condition": (
                "candidate epoch becomes new best"
            ),
            "training_resume_source": (
                False
            ),
            "final_test_source": (
                True
            ),
        },
        "latest_checkpoint": {
            "state_association": (
                "post-validation advanced controller state"
            ),
            "after_nonfinal_epoch": {
                "epoch_index": (
                    "e+1"
                ),
                "next_batch_index": (
                    0
                ),
            },
            "training_resume_source": (
                True
            ),
            "final_test_source": (
                False
            ),
        },
        "final_epoch": {
            "epoch_index": (
                19
            ),
            "expected_global_optimizer_steps": (
                TOTAL_OPTIMIZER_STEPS
            ),
            "validation_commit_sets_training_complete": (
                True
            ),
            "test_eligible_after_commit": (
                True
            ),
            "test_scored_in_this_phase": (
                False
            ),
        },
        "boundary": {
            "model_instantiated": (
                False
            ),
            "validation_forward_executed": (
                False
            ),
            "backward_executed": (
                False
            ),
            "optimizer_steps": (
                0
            ),
            "test_cases_scored": (
                0
            ),
        },
        "next_phase": {
            "id": (
                "5.3.4"
            ),
            "title": (
                "Production Trainer Assembly and Bounded End-to-End Dry-Run"
            ),
            "requirement": (
                "Assemble the frozen training stream, exact Adam update, "
                "checkpoint/resume controller, full validation runtime, "
                "validation commit/best-checkpoint logic, and test guard "
                "into one production trainer. Run a bounded integration "
                "dry-run without completing a full epoch, then freeze the "
                "launch-ready 20-epoch driver."
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
            "5.3.3d"
        ),
        "status": (
            "VALIDATION_CHECKPOINT_SELECTION_"
            "INTEGRATION_PROVED_AND_FROZEN"
        ),
        "real_epoch0_HR@10": (
            actual_real_hr
        ),
        "real_epoch0_NDCG@10": (
            actual_real_ndcg
        ),
        "real_epoch0_becomes_initial_best": (
            True
        ),
        "best_rule_probes": (
            len(
                best_rule_df
            )
        ),
        "final_test_eligibility_after_epoch19": (
            True
        ),
        "test_cases_scored": (
            0
        ),
        "optimizer_steps": (
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
        REAL_COMMIT_PATH,
        BEST_RULE_PATH,
        EPOCH_ADVANCE_PATH,
        FINAL_EPOCH_PATH,
        CHECKPOINT_ROLE_PATH,
        FINAL_INVARIANT_PATH,
        DECISION_REGISTER_PATH,
        CONTRACT_PATH,
        MANIFEST_PATH,
    ):
        print(
            f"WROTE  {path}"
        )

    # =========================================================================
    # Final
    # =========================================================================

    banner(
        "PHASE 5.3.3d FINAL STATUS"
    )

    print(
        "Real frozen validation anchor:"
    )
    print(
        f"  epoch 0 HR@10:                      "
        f"{actual_real_hr:.12f}"
    )
    print(
        f"  epoch 0 NDCG@10:                    "
        f"{actual_real_ndcg:.12f}"
    )
    print(
        "  becomes initial best:               YES"
    )
    print()

    print(
        "Best-checkpoint ordering:"
    )
    print(
        "  1. maximum NDCG@10"
    )
    print(
        "  2. maximum HR@10"
    )
    print(
        "  3. exact tie keeps earliest epoch"
    )
    print()

    print(
        "Validation commit:"
    )
    print(
        "  committed before epoch advancement"
    )
    print(
        "  best checkpoint refers to validated epoch model"
    )
    print(
        "  latest checkpoint refers to training-resume state"
    )
    print()

    print(
        "After non-final epoch validation:"
    )
    print(
        "  epoch_index:                        e + 1"
    )
    print(
        "  next_batch_index:                   0"
    )
    print()

    print(
        "After epoch-19 validation:"
    )
    print(
        "  training_complete:                  True"
    )
    print(
        "  global_optimizer_step:              209,620"
    )
    print(
        "  test becomes eligible:              YES"
    )
    print(
        "  test cases scored here:             0"
    )
    print()

    print(
        "Neural model instantiated:            NO"
    )
    print(
        "Validation forward executed:          NO"
    )
    print(
        "Backward computation:                 NO"
    )
    print(
        "optimizer.step():                     0"
    )

    banner(
        "PHASE 5.3.3d COMPLETE / "
        "VALIDATION-CHECKPOINT SELECTION INTEGRATION PROVED AND FROZEN"
    )


if __name__ == "__main__":
    main()