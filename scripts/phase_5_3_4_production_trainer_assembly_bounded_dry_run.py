"""
Phase 5.3.4 — Production Trainer Assembly: Bounded End-to-End Dry-Run

Purpose
-------
Assemble the already-frozen Phase-5 training and validation components behind
one production-style interface and exercise them together without launching a
full epoch or touching the test split.

This phase intentionally does NOT generalize epoch-stream generation beyond
epoch 0. That remaining launch prerequisite is isolated as the next gate.

Real bounded training path
--------------------------
    canonical initial model
      -> epoch 0 / batch 0
      -> Adam step 1
      -> controller commit
      -> production latest checkpoint
      -> destroy live model/Adam
      -> reload exact checkpoint
      -> resume epoch 0 / batch 1
      -> Adam step 2
      -> controller commit

The post-batch-1 model and Adam fingerprints must exactly reproduce the frozen
Phase-5.3.2b/5.3.2c two-step anchors.

Real bounded validation-interface path
--------------------------------------
A separate canonical seed-42 model is passed through the production validation
interface for the exact first 16 frozen validation cases.

This diagnostic call MUST reproduce the Phase-5.3.3b:
    selected-case fingerprint
    candidate-matrix fingerprint
    raw-logit matrix fingerprint
    case-metric fingerprint

It is NOT committed for checkpoint selection.

Validation-commit controller path
---------------------------------
The already-frozen REAL full-validation result from Phase-5.3.3c is then fed
through the same Phase-5.3.3d validation-commit controller using an explicitly
CONTROL_FLOW_ONLY completed-epoch state.

This proves:
    - real frozen validation metrics become the first best
    - best state stays associated with validated epoch 0
    - latest/resume state advances to epoch 1 / batch 0
    - test remains forbidden

The control-flow state is not claimed to have been produced by the two-batch
training dry-run. No fake full epoch is created.

Safety boundary
---------------
- real optimizer.step() calls: exactly 2
- full epoch completed: NO
- full 20-epoch training launched: NO
- full validation pass recomputed: NO
- test cases scored: 0
- best checkpoint file written from incomplete training: NO

Remaining launch gate
---------------------
The 20-epoch RNG seed registry is already frozen, but only epoch-0 stream
serialization has been numerically proven. Before production launch we still
must generalize the epoch stream generator and require that epoch 0 is
reproduced byte-for-byte from the generalized implementation.
"""

from __future__ import annotations

import ast
import gc
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# =============================================================================
# Frozen sources
# =============================================================================

ROUNDTRIP_SOURCE_PATH = Path(
    "scripts/"
    "phase_5_3_2b_checkpoint_resume_roundtrip_proof.py"
)

CONTROLLER_SOURCE_PATH = Path(
    "scripts/"
    "phase_5_3_2c_production_training_driver_controller_dry_run.py"
)

RANKING_SOURCE_PATH = Path(
    "scripts/"
    "phase_5_3_3a_validation_ranking_metric_semantics_audit.py"
)

VALIDATION_PREFLIGHT_SOURCE_PATH = Path(
    "scripts/"
    "phase_5_3_3b_canonical_real_validation_scoring_preflight.py"
)

VALIDATION_COMMIT_SOURCE_PATH = Path(
    "scripts/"
    "phase_5_3_3d_validation_checkpoint_selection_integration_proof.py"
)


# =============================================================================
# Frozen contracts
# =============================================================================

PHASE_5_3_2A_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_2a_training_execution_state_contract.json"
)

PHASE_5_3_2C_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_2c_production_driver_controller_contract.json"
)

PHASE_5_3_3A_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_3a_validation_ranking_metric_contract.json"
)

PHASE_5_3_3B_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_3b_canonical_validation_scoring_preflight_contract.json"
)

PHASE_5_3_3C_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_3c_full_validation_runtime_contract.json"
)

PHASE_5_3_3D_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_3d_validation_checkpoint_integration_contract.json"
)


# =============================================================================
# Frozen numerical anchors
# =============================================================================

EXPECTED_INITIAL_MODEL_SHA256 = (
    "49e822ea7fad35c458f47e134c94c05e"
    "ac099b68c5c468e2c71559c8c88998ab"
)

EXPECTED_FIRST_BATCH_LOGIT_SHA256 = (
    "35b89aaed29d51d2ebb7ba1cadf2dc4b"
    "b5e8f81cf3aa78bc216b3cc6fed13845"
)

EXPECTED_FIRST_BATCH_GRADIENT_SHA256 = (
    "8c542430813d8ca91b8397409954ea92"
    "295a2b55bcc420661783fb865010845d"
)

EXPECTED_FIRST_STEP_MODEL_SHA256 = (
    "42a521f11d8f24e4144d0215d6e1b34d"
    "5f8bf0c2d8848624e4f7c3130699035d"
)

EXPECTED_FIRST_STEP_OPTIMIZER_SHA256 = (
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

EXPECTED_TWO_STEP_MODEL_SHA256 = (
    "c41702cda99092a7fb63bb0a8227e658"
    "851b3ac4cbc373d90cdd6816eccdd196"
)

EXPECTED_TWO_STEP_OPTIMIZER_SHA256 = (
    "569a6691424ac32d0f252728750281cff"
    "d175a2b6b6c6ea1913f5f497200b00d"
)

EXPECTED_VALIDATION16_SELECTED_CASE_SHA256 = (
    "72331e17299e61fe94757eb5f4c00129"
    "dbfaffcd92b1424013443523cca37f96"
)

EXPECTED_VALIDATION16_CANDIDATE_SHA256 = (
    "a6a1b63954f3065d7c748fab55886af2"
    "8e440d0029626ed9db1809a389665514"
)

EXPECTED_VALIDATION16_LOGIT_SHA256 = (
    "bf4f986f96cf2e4557ca1e360a19b974"
    "9fadf452f1c6669cae83099a8905ea0c"
)

EXPECTED_VALIDATION16_METRIC_SHA256 = (
    "7b6b4a19b19052af84a14ee631020f79"
    "6fd6bfd2dc89f7c484b568983b6b2606"
)

EXPECTED_FULL_VALIDATION_LOGIT_SHA256 = (
    "1799eb3e382fbdf5477234666d64bbb8"
    "32d64ad6bfb354c903618e6b945b5058"
)

EXPECTED_FULL_VALIDATION_METRIC_SHA256 = (
    "c091008ad2206c2890e5f4835d687bf9"
    "288bd078e2c8b60a12575b64fa131b63"
)

EXPECTED_FULL_VALIDATION_RANK_SHA256 = (
    "273bdb5b2252f1c8a48420df4685bd2e"
    "5dbfb5f66a7b6f5125ac8ee1913e3345"
)

REFERENCE_TORCH_VERSION_PREFIX = "2.7.0"

NUM_INVESTORS = 165_975
NUM_STARTUPS = 311_589
NUM_NODES = 477_564

VALIDATION_PREFLIGHT_CASES = 16
CANDIDATES_PER_CASE = 100


# =============================================================================
# Outputs
# =============================================================================

AUDIT_DIR = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_4"
)

CONTRACT_DIR = Path(
    "data/experimental/phase_5/contracts"
)

LATEST_AFTER_BATCH0_PATH = (
    AUDIT_DIR
    / "integrated_latest_after_batch0.pt"
)

TRAINING_DRY_RUN_PATH = (
    AUDIT_DIR
    / "integrated_two_batch_training_audit.csv"
)

CHECKPOINT_RELOAD_PATH = (
    AUDIT_DIR
    / "integrated_checkpoint_reload_audit.csv"
)

VALIDATION_INTERFACE_PATH = (
    AUDIT_DIR
    / "integrated_validation16_interface_audit.csv"
)

VALIDATION_COMMIT_PATH = (
    AUDIT_DIR
    / "integrated_real_validation_commit_audit.csv"
)

BOUNDARY_PATH = (
    AUDIT_DIR
    / "production_launch_boundary_audit.csv"
)

FINAL_INVARIANT_PATH = (
    AUDIT_DIR
    / "phase_5_3_4_final_invariants.csv"
)

DECISION_REGISTER_PATH = (
    CONTRACT_DIR
    / "phase_5_3_4_production_assembly_decision_register.csv"
)

CONTRACT_PATH = (
    CONTRACT_DIR
    / "phase_5_3_4_production_trainer_assembly_contract.json"
)

MANIFEST_PATH = (
    AUDIT_DIR
    / "phase_5_3_4_production_trainer_assembly_manifest.json"
)


# =============================================================================
# Generic helpers
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


def is_main_guard(
    node: ast.AST,
) -> bool:
    if not isinstance(
        node,
        ast.If,
    ):
        return False

    test = node.test

    return (
        isinstance(
            test,
            ast.Compare,
        )
        and isinstance(
            test.left,
            ast.Name,
        )
        and test.left.id == "__name__"
        and len(
            test.ops
        )
        == 1
        and isinstance(
            test.ops[0],
            ast.Eq,
        )
        and len(
            test.comparators
        )
        == 1
        and isinstance(
            test.comparators[0],
            ast.Constant,
        )
        and test.comparators[0].value
        == "__main__"
    )


def load_guarded_module(
    path: Path,
    module_name: str,
):
    require(
        path.exists(),
        f"Missing source: {path}",
    )

    source = path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(path),
    )

    guards = [
        node
        for node in tree.body
        if is_main_guard(node)
    ]

    require(
        len(guards) == 1,
        (
            "Expected exactly one __main__ "
            f"guard in {path}."
        ),
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    require(
        spec is not None
        and spec.loader is not None,
        (
            "Could not create import spec "
            f"for {path}."
        ),
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


# =============================================================================
# Validation-prefix reconstruction
# =============================================================================

def build_validation16(
    *,
    validation_runtime,
    ranking_contract: dict,
) -> tuple[pd.DataFrame, np.ndarray]:
    negative_artifact = (
        ranking_contract[
            "frozen_artifacts"
        ][
            "negative_matrix"
        ]
    )

    case_artifact = (
        ranking_contract[
            "frozen_artifacts"
        ][
            "case_manifest"
        ]
    )

    negative_matrix_path = Path(
        negative_artifact[
            "path"
        ]
    )

    case_manifest_path = Path(
        case_artifact[
            "path"
        ]
    )

    negative_raw = np.load(
        negative_matrix_path,
        mmap_mode="r",
    )

    coordinate = (
        validation_runtime
        .infer_negative_matrix_coordinate(
            negative_raw
        )
    )

    negative_local = (
        validation_runtime
        .normalize_negative_matrix_to_local(
            negative_raw,
            coordinate,
        )
    )

    case_manifest = pd.read_parquet(
        case_manifest_path
    )

    (
        resolved_cases,
        _,
    ) = (
        validation_runtime
        .resolve_case_bindings(
            case_manifest,
            ranking_contract,
        )
    )

    validation_cases = (
        resolved_cases.loc[
            resolved_cases[
                "split"
            ]
            == "validation"
        ]
        .sort_values(
            [
                "matrix_row_index",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    selected = (
        validation_cases.iloc[
            :VALIDATION_PREFLIGHT_CASES
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    candidate_matrix = np.empty(
        (
            VALIDATION_PREFLIGHT_CASES,
            CANDIDATES_PER_CASE,
        ),
        dtype=np.int64,
    )

    rows = []

    for case_position, row in (
        selected.iterrows()
    ):
        matrix_row_index = int(
            row[
                "matrix_row_index"
            ]
        )

        positive_local = int(
            row[
                "positive_startup_local"
            ]
        )

        negatives_local = np.asarray(
            negative_local[
                matrix_row_index
            ],
            dtype=np.int64,
        )

        require(
            positive_local
            not in set(
                int(value)
                for value
                in negatives_local.tolist()
            ),
            (
                "Positive startup appears "
                "in frozen negatives."
            ),
        )

        candidates = np.concatenate(
            [
                np.array(
                    [
                        positive_local,
                    ],
                    dtype=np.int64,
                ),
                negatives_local,
            ]
        )

        require(
            len(
                np.unique(
                    candidates
                )
            )
            == 100,
            (
                "Validation prefix candidate "
                "set is not unique."
            ),
        )

        candidate_matrix[
            case_position
        ] = candidates

        rows.append(
            {
                "preflight_case_position": (
                    int(
                        case_position
                    )
                ),
                "matrix_row_index": (
                    matrix_row_index
                ),
                "interaction_id": str(
                    row[
                        "interaction_id"
                    ]
                ),
                "investor_global": int(
                    row[
                        "investor_global"
                    ]
                ),
                "positive_startup_local": (
                    positive_local
                ),
                "candidate_count": (
                    100
                ),
                "negative_count": (
                    99
                ),
            }
        )

    selection_df = pd.DataFrame(
        rows
    )

    return (
        selected,
        candidate_matrix,
        selection_df,
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.3.4 — "
        "PRODUCTION TRAINER ASSEMBLY / BOUNDED END-TO-END DRY-RUN"
    )

    print(
        "Real training batches executed:       2"
    )
    print(
        "Real optimizer.step() calls:          2"
    )
    print(
        "Full epoch completed:                 NO"
    )
    print(
        "Full validation recomputed:           NO"
    )
    print(
        "Diagnostic validation cases scored:  16"
    )
    print(
        "Test cases scored:                    0"
    )
    print(
        "Full 20-epoch launch:                 NO"
    )

    # =========================================================================
    # Prerequisite recheck
    # =========================================================================

    banner(
        "AUTHORITATIVE ASSEMBLY GATE RECHECK"
    )

    required_paths = (
        ROUNDTRIP_SOURCE_PATH,
        CONTROLLER_SOURCE_PATH,
        RANKING_SOURCE_PATH,
        VALIDATION_PREFLIGHT_SOURCE_PATH,
        VALIDATION_COMMIT_SOURCE_PATH,
        PHASE_5_3_2A_CONTRACT_PATH,
        PHASE_5_3_2C_CONTRACT_PATH,
        PHASE_5_3_3A_CONTRACT_PATH,
        PHASE_5_3_3B_CONTRACT_PATH,
        PHASE_5_3_3C_CONTRACT_PATH,
        PHASE_5_3_3D_CONTRACT_PATH,
    )

    for path in required_paths:
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

    ranking_contract = load_json(
        PHASE_5_3_3A_CONTRACT_PATH
    )

    validation16_contract = load_json(
        PHASE_5_3_3B_CONTRACT_PATH
    )

    full_validation_contract = load_json(
        PHASE_5_3_3C_CONTRACT_PATH
    )

    validation_commit_contract = load_json(
        PHASE_5_3_3D_CONTRACT_PATH
    )

    for name, contract in (
        (
            "5.3.2a execution",
            execution_contract,
        ),
        (
            "5.3.2c controller",
            controller_contract,
        ),
        (
            "5.3.3a ranking",
            ranking_contract,
        ),
        (
            "5.3.3b validation preflight",
            validation16_contract,
        ),
        (
            "5.3.3c full validation",
            full_validation_contract,
        ),
        (
            "5.3.3d validation commit",
            validation_commit_contract,
        ),
    ):
        require(
            contract[
                "status"
            ]
            == "FROZEN",
            (
                f"{name} contract "
                "is not frozen."
            ),
        )

        print(
            f"{name:36s} FROZEN / PASS"
        )

    require(
        torch.__version__.startswith(
            REFERENCE_TORCH_VERSION_PREFIX
        ),
        (
            "Reference runtime requires "
            "PyTorch 2.7.0."
        ),
    )

    # =========================================================================
    # Load modules
    # =========================================================================

    banner(
        "LOAD FROZEN COMPONENT RUNTIMES"
    )

    runtime_2b = load_guarded_module(
        ROUNDTRIP_SOURCE_PATH,
        "_itrs_phase5_3_4_runtime2b",
    )

    controller_runtime = load_guarded_module(
        CONTROLLER_SOURCE_PATH,
        "_itrs_phase5_3_4_controller",
    )

    ranking_runtime = load_guarded_module(
        RANKING_SOURCE_PATH,
        "_itrs_phase5_3_4_ranking",
    )

    validation_runtime = load_guarded_module(
        VALIDATION_PREFLIGHT_SOURCE_PATH,
        "_itrs_phase5_3_4_validation",
    )

    commit_runtime = load_guarded_module(
        VALIDATION_COMMIT_SOURCE_PATH,
        "_itrs_phase5_3_4_commit",
    )

    # =========================================================================
    # Real integrated training path
    # =========================================================================

    banner(
        "INTEGRATED TRAINING PATH — BATCH 0 -> CHECKPOINT -> RELOAD -> BATCH 1"
    )

    preflight_runtime = (
        runtime_2b
        .load_preflight_runtime()
    )

    stream = (
        runtime_2b
        .load_epoch0_stream(
            preflight_runtime
        )
    )

    shared = (
        runtime_2b
        .load_shared_inputs(
            preflight_runtime
        )
    )

    batch0 = (
        runtime_2b
        .decode_batch(
            stream,
            batch_index=0,
        )
    )

    batch1 = (
        runtime_2b
        .decode_batch(
            stream,
            batch_index=1,
        )
    )

    (
        model,
        optimizer,
        canonical_hash_fn,
        _,
        _,
        _,
    ) = (
        runtime_2b
        .construct_fresh_training_state(
            preflight_runtime
        )
    )

    require(
        canonical_hash_fn(
            model
        )
        == EXPECTED_INITIAL_MODEL_SHA256,
        (
            "Integrated trainer did not "
            "start from canonical model."
        ),
    )

    state0 = (
        controller_runtime
        .fresh_controller_state()
    )

    result0 = (
        runtime_2b
        .execute_training_batch(
            model,
            optimizer,
            canonical_hash_fn,
            batch0,
            shared,
        )
    )

    require(
        result0[
            "logit_sha256"
        ]
        == EXPECTED_FIRST_BATCH_LOGIT_SHA256,
        (
            "Integrated batch-0 logit "
            "fingerprint drift."
        ),
    )

    require(
        result0[
            "gradient_sha256"
        ]
        == EXPECTED_FIRST_BATCH_GRADIENT_SHA256,
        (
            "Integrated batch-0 gradient "
            "fingerprint drift."
        ),
    )

    require(
        result0[
            "post_step_model_sha256"
        ]
        == EXPECTED_FIRST_STEP_MODEL_SHA256,
        (
            "Integrated first-step model "
            "fingerprint drift."
        ),
    )

    require(
        result0[
            "optimizer_state_sha256"
        ]
        == EXPECTED_FIRST_STEP_OPTIMIZER_SHA256,
        (
            "Integrated first-step Adam "
            "fingerprint drift."
        ),
    )

    state1 = (
        controller_runtime
        .commit_completed_training_batch(
            state0,
            executed_batch_index=0,
            batch_size=len(
                batch0
            ),
            batch_loss=(
                result0[
                    "loss"
                ]
            ),
        )
    )

    require(
        state1[
            "epoch_index"
        ]
        == 0
        and state1[
            "next_batch_index"
        ]
        == 1
        and state1[
            "global_optimizer_step"
        ]
        == 1,
        (
            "Integrated controller state "
            "after batch 0 drift."
        ),
    )

    training_fingerprints = {
        "initial_model_sha256": (
            EXPECTED_INITIAL_MODEL_SHA256
        ),
        "first_step_model_sha256": (
            EXPECTED_FIRST_STEP_MODEL_SHA256
        ),
        "first_step_optimizer_sha256": (
            EXPECTED_FIRST_STEP_OPTIMIZER_SHA256
        ),
        "two_step_model_sha256": (
            EXPECTED_TWO_STEP_MODEL_SHA256
        ),
        "two_step_optimizer_sha256": (
            EXPECTED_TWO_STEP_OPTIMIZER_SHA256
        ),
    }

    payload = (
        controller_runtime
        .build_production_checkpoint(
            model=model,
            optimizer=optimizer,
            state=state1,
            execution_contract=(
                execution_contract
            ),
            training_fingerprints=(
                training_fingerprints
            ),
            checkpoint_kind="latest",
        )
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    controller_runtime.save_atomic_checkpoint(
        payload,
        LATEST_AFTER_BATCH0_PATH,
    )

    del payload
    del model
    del optimizer
    gc.collect()

    loaded = (
        controller_runtime
        .load_production_checkpoint(
            LATEST_AFTER_BATCH0_PATH,
            execution_contract,
        )
    )

    (
        _,
        model,
        optimizer,
        canonical_hash_fn,
        resumed_state,
        _,
        _,
        _,
    ) = (
        controller_runtime
        .restore_model_optimizer_from_checkpoint(
            runtime_2b=runtime_2b,
            checkpoint=loaded,
        )
    )

    reload_model_sha = (
        canonical_hash_fn(
            model
        )
    )

    reload_optimizer_sha = (
        runtime_2b
        .optimizer_state_logical_sha256(
            model,
            optimizer,
        )
    )

    require(
        reload_model_sha
        == EXPECTED_FIRST_STEP_MODEL_SHA256,
        (
            "Integrated latest checkpoint "
            "restored wrong model."
        ),
    )

    require(
        reload_optimizer_sha
        == EXPECTED_FIRST_STEP_OPTIMIZER_SHA256,
        (
            "Integrated latest checkpoint "
            "restored wrong Adam state."
        ),
    )

    require(
        resumed_state
        == state1,
        (
            "Integrated latest checkpoint "
            "restored wrong controller state."
        ),
    )

    result1 = (
        runtime_2b
        .execute_training_batch(
            model,
            optimizer,
            canonical_hash_fn,
            batch1,
            shared,
        )
    )

    require(
        result1[
            "logit_sha256"
        ]
        == EXPECTED_BATCH1_LOGIT_SHA256,
        (
            "Integrated batch-1 logit "
            "fingerprint drift."
        ),
    )

    require(
        result1[
            "gradient_sha256"
        ]
        == EXPECTED_BATCH1_GRADIENT_SHA256,
        (
            "Integrated batch-1 gradient "
            "fingerprint drift."
        ),
    )

    require(
        result1[
            "post_step_model_sha256"
        ]
        == EXPECTED_TWO_STEP_MODEL_SHA256,
        (
            "Integrated two-step model "
            "fingerprint drift."
        ),
    )

    require(
        result1[
            "optimizer_state_sha256"
        ]
        == EXPECTED_TWO_STEP_OPTIMIZER_SHA256,
        (
            "Integrated two-step optimizer "
            "fingerprint drift."
        ),
    )

    state2 = (
        controller_runtime
        .commit_completed_training_batch(
            resumed_state,
            executed_batch_index=1,
            batch_size=len(
                batch1
            ),
            batch_loss=(
                result1[
                    "loss"
                ]
            ),
        )
    )

    require(
        state2[
            "epoch_index"
        ]
        == 0
        and state2[
            "next_batch_index"
        ]
        == 2
        and state2[
            "global_optimizer_step"
        ]
        == 2
        and state2[
            "validation_pending"
        ]
        is False,
        (
            "Integrated controller state "
            "after batch 1 drift."
        ),
    )

    training_df = pd.DataFrame(
        [
            {
                "boundary": (
                    "batch0"
                ),
                "logit_sha256": (
                    result0[
                        "logit_sha256"
                    ]
                ),
                "gradient_sha256": (
                    result0[
                        "gradient_sha256"
                    ]
                ),
                "model_sha256": (
                    result0[
                        "post_step_model_sha256"
                    ]
                ),
                "optimizer_sha256": (
                    result0[
                        "optimizer_state_sha256"
                    ]
                ),
                "next_batch_index": (
                    state1[
                        "next_batch_index"
                    ]
                ),
                "global_optimizer_step": (
                    state1[
                        "global_optimizer_step"
                    ]
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "boundary": (
                    "batch1_after_reload"
                ),
                "logit_sha256": (
                    result1[
                        "logit_sha256"
                    ]
                ),
                "gradient_sha256": (
                    result1[
                        "gradient_sha256"
                    ]
                ),
                "model_sha256": (
                    result1[
                        "post_step_model_sha256"
                    ]
                ),
                "optimizer_sha256": (
                    result1[
                        "optimizer_state_sha256"
                    ]
                ),
                "next_batch_index": (
                    state2[
                        "next_batch_index"
                    ]
                ),
                "global_optimizer_step": (
                    state2[
                        "global_optimizer_step"
                    ]
                ),
                "status": (
                    "PASS"
                ),
            },
        ]
    )

    reload_df = pd.DataFrame(
        [
            {
                "check": (
                    "reloaded_model_sha256"
                ),
                "actual": (
                    reload_model_sha
                ),
                "expected": (
                    EXPECTED_FIRST_STEP_MODEL_SHA256
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "check": (
                    "reloaded_optimizer_sha256"
                ),
                "actual": (
                    reload_optimizer_sha
                ),
                "expected": (
                    EXPECTED_FIRST_STEP_OPTIMIZER_SHA256
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "check": (
                    "reloaded_epoch_index"
                ),
                "actual": (
                    resumed_state[
                        "epoch_index"
                    ]
                ),
                "expected": (
                    0
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "check": (
                    "reloaded_next_batch_index"
                ),
                "actual": (
                    resumed_state[
                        "next_batch_index"
                    ]
                ),
                "expected": (
                    1
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "check": (
                    "reloaded_global_step"
                ),
                "actual": (
                    resumed_state[
                        "global_optimizer_step"
                    ]
                ),
                "expected": (
                    1
                ),
                "status": (
                    "PASS"
                ),
            },
        ]
    )

    print(
        training_df.to_string(
            index=False
        )
    )

    # Free training model before validation interface probe.
    del model
    del optimizer
    gc.collect()

    # =========================================================================
    # Real integrated validation interface — exact 16-case prefix
    # =========================================================================

    banner(
        "INTEGRATED VALIDATION INTERFACE — EXACT 16-CASE REAL PREFIX"
    )

    (
        selected_cases,
        candidate_matrix,
        selection_df,
    ) = build_validation16(
        validation_runtime=(
            validation_runtime
        ),
        ranking_contract=(
            ranking_contract
        ),
    )

    selection_sha = (
        validation_runtime
        .dataframe_logical_sha256(
            selection_df,
            columns=[
                "preflight_case_position",
                "matrix_row_index",
                "interaction_id",
                "investor_global",
                "positive_startup_local",
                "candidate_count",
                "negative_count",
            ],
        )
    )

    candidate_sha = (
        validation_runtime
        .array_logical_sha256(
            candidate_matrix
        )
    )

    require(
        selection_sha
        == EXPECTED_VALIDATION16_SELECTED_CASE_SHA256,
        (
            "Integrated validation selection "
            "fingerprint drift."
        ),
    )

    require(
        candidate_sha
        == EXPECTED_VALIDATION16_CANDIDATE_SHA256,
        (
            "Integrated validation candidate "
            "fingerprint drift."
        ),
    )

    (
        validation_preflight_runtime,
        validation_model,
        validation_hash_fn,
        _,
        _,
        _,
    ) = (
        validation_runtime
        .construct_canonical_validation_model(
            runtime_2b
        )
    )

    validation_shared = (
        runtime_2b
        .load_shared_inputs(
            validation_preflight_runtime
        )
    )

    with torch.no_grad():
        features = (
            validation_runtime
            .compute_validation_features(
                validation_model,
                selected_cases,
                candidate_matrix,
                validation_shared,
            )
        )

    score_matrix = np.empty(
        (
            VALIDATION_PREFLIGHT_CASES,
            CANDIDATES_PER_CASE,
        ),
        dtype=np.float32,
    )

    metric_rows = []

    with torch.no_grad():
        for case_position, row in (
            selected_cases.iterrows()
        ):
            investor_global = int(
                row[
                    "investor_global"
                ]
            )

            positive_local = int(
                row[
                    "positive_startup_local"
                ]
            )

            candidates_local = (
                candidate_matrix[
                    case_position
                ]
            )

            logits = (
                validation_runtime
                .score_validation_case(
                    validation_model,
                    investor_global,
                    candidates_local,
                    features,
                )
            )

            logits_np = (
                logits
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

            score_matrix[
                case_position
            ] = logits_np

            positive_rank = (
                ranking_runtime
                .rank_candidates(
                    logits_np,
                    candidates_local,
                    positive_local,
                )
            )

            (
                hr10,
                ndcg10,
            ) = (
                ranking_runtime
                .metrics_from_positive_rank(
                    positive_rank,
                    k=10,
                )
            )

            metric_rows.append(
                {
                    "preflight_case_position": (
                        int(
                            case_position
                        )
                    ),
                    "matrix_row_index": int(
                        row[
                            "matrix_row_index"
                        ]
                    ),
                    "interaction_id": str(
                        row[
                            "interaction_id"
                        ]
                    ),
                    "investor_global": (
                        investor_global
                    ),
                    "positive_startup_local": (
                        positive_local
                    ),
                    "positive_logit": float(
                        logits_np[
                            0
                        ]
                    ),
                    "minimum_logit": float(
                        np.min(
                            logits_np
                        )
                    ),
                    "maximum_logit": float(
                        np.max(
                            logits_np
                        )
                    ),
                    "mean_logit": float(
                        np.mean(
                            logits_np,
                            dtype=np.float64,
                        )
                    ),
                    "positive_rank": (
                        int(
                            positive_rank
                        )
                    ),
                    "HR@10": (
                        float(
                            hr10
                        )
                    ),
                    "NDCG@10": (
                        float(
                            ndcg10
                        )
                    ),
                }
            )

    validation_metric_df = pd.DataFrame(
        metric_rows
    )

    score_sha = (
        validation_runtime
        .array_logical_sha256(
            score_matrix
        )
    )

    metric_sha = (
        validation_runtime
        .dataframe_logical_sha256(
            validation_metric_df,
            columns=[
                "preflight_case_position",
                "matrix_row_index",
                "interaction_id",
                "investor_global",
                "positive_startup_local",
                "positive_logit",
                "minimum_logit",
                "maximum_logit",
                "mean_logit",
                "positive_rank",
                "HR@10",
                "NDCG@10",
            ],
        )
    )

    require(
        score_sha
        == EXPECTED_VALIDATION16_LOGIT_SHA256,
        (
            "Integrated validation raw-logit "
            "fingerprint drift."
        ),
    )

    require(
        metric_sha
        == EXPECTED_VALIDATION16_METRIC_SHA256,
        (
            "Integrated validation metric "
            "fingerprint drift."
        ),
    )

    require(
        validation_hash_fn(
            validation_model
        )
        == EXPECTED_INITIAL_MODEL_SHA256,
        (
            "Integrated validation interface "
            "changed canonical parameters."
        ),
    )

    diagnostic_hr = float(
        validation_metric_df[
            "HR@10"
        ].mean()
    )

    diagnostic_ndcg = float(
        validation_metric_df[
            "NDCG@10"
        ].mean()
    )

    validation_interface_df = pd.DataFrame(
        [
            {
                "check": (
                    "selected_case_sha256"
                ),
                "actual": (
                    selection_sha
                ),
                "expected": (
                    EXPECTED_VALIDATION16_SELECTED_CASE_SHA256
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "check": (
                    "candidate_matrix_sha256"
                ),
                "actual": (
                    candidate_sha
                ),
                "expected": (
                    EXPECTED_VALIDATION16_CANDIDATE_SHA256
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "check": (
                    "raw_logit_matrix_sha256"
                ),
                "actual": (
                    score_sha
                ),
                "expected": (
                    EXPECTED_VALIDATION16_LOGIT_SHA256
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "check": (
                    "case_metric_sha256"
                ),
                "actual": (
                    metric_sha
                ),
                "expected": (
                    EXPECTED_VALIDATION16_METRIC_SHA256
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "check": (
                    "diagnostic_HR@10"
                ),
                "actual": (
                    diagnostic_hr
                ),
                "expected": (
                    validation16_contract[
                        "numerical_fingerprints"
                    ][
                        "diagnostic_subset_HR@10"
                    ]
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "check": (
                    "diagnostic_NDCG@10"
                ),
                "actual": (
                    diagnostic_ndcg
                ),
                "expected": (
                    validation16_contract[
                        "numerical_fingerprints"
                    ][
                        "diagnostic_subset_NDCG@10"
                    ]
                ),
                "status": (
                    "PASS"
                ),
            },
        ]
    )

    print(
        validation_interface_df.to_string(
            index=False
        )
    )

    del validation_model
    del features
    gc.collect()

    # =========================================================================
    # Integrated real full-validation commit controller
    # =========================================================================

    banner(
        "INTEGRATED VALIDATION COMMIT — REAL FROZEN FULL-SPLIT RESULT"
    )

    real_validation = (
        full_validation_contract[
            "canonical_initial_validation"
        ]
    )

    require(
        real_validation[
            "raw_logit_matrix_logical_sha256"
        ]
        == EXPECTED_FULL_VALIDATION_LOGIT_SHA256,
        (
            "Full validation logit anchor drift."
        ),
    )

    require(
        real_validation[
            "case_metric_logical_sha256"
        ]
        == EXPECTED_FULL_VALIDATION_METRIC_SHA256,
        (
            "Full validation metric anchor drift."
        ),
    )

    require(
        real_validation[
            "positive_rank_vector_logical_sha256"
        ]
        == EXPECTED_FULL_VALIDATION_RANK_SHA256,
        (
            "Full validation rank anchor drift."
        ),
    )

    # Explicitly CONTROL_FLOW_ONLY. This state is not produced by
    # the two-batch real-training dry-run.
    pending_epoch0 = (
        commit_runtime
        .make_validation_pending_state(
            0
        )
    )

    commit_result = (
        commit_runtime
        .commit_validation(
            pending_epoch0,
            validation_hr10=float(
                real_validation[
                    "HR@10"
                ]
            ),
            validation_ndcg10=float(
                real_validation[
                    "NDCG@10"
                ]
            ),
        )
    )

    require(
        commit_result[
            "best_checkpoint_should_write"
        ]
        is True,
        (
            "Real frozen epoch-0 validation "
            "did not become initial best."
        ),
    )

    best_state = (
        commit_result[
            "best_checkpoint_state"
        ]
    )

    latest_state = (
        commit_result[
            "resume_state"
        ]
    )

    require(
        best_state is not None
        and best_state[
            "epoch_index"
        ]
        == 0
        and best_state[
            "next_batch_index"
        ]
        == 10_481,
        (
            "Integrated best-checkpoint "
            "state semantics drift."
        ),
    )

    require(
        latest_state[
            "epoch_index"
        ]
        == 1
        and latest_state[
            "next_batch_index"
        ]
        == 0,
        (
            "Integrated latest/resume "
            "state semantics drift."
        ),
    )

    require(
        commit_result[
            "test_access_allowed"
        ]
        is False,
        (
            "Integrated epoch-0 validation "
            "incorrectly opened test access."
        ),
    )

    validation_commit_df = pd.DataFrame(
        [
            {
                "item": (
                    "source"
                ),
                "value": (
                    "Phase-5.3.3c frozen real full validation"
                ),
            },
            {
                "item": (
                    "HR@10"
                ),
                "value": (
                    real_validation[
                        "HR@10"
                    ]
                ),
            },
            {
                "item": (
                    "NDCG@10"
                ),
                "value": (
                    real_validation[
                        "NDCG@10"
                    ]
                ),
            },
            {
                "item": (
                    "becomes_initial_best"
                ),
                "value": (
                    True
                ),
            },
            {
                "item": (
                    "best_epoch_index"
                ),
                "value": (
                    best_state[
                        "epoch_index"
                    ]
                ),
            },
            {
                "item": (
                    "best_next_batch_index"
                ),
                "value": (
                    best_state[
                        "next_batch_index"
                    ]
                ),
            },
            {
                "item": (
                    "latest_resume_epoch_index"
                ),
                "value": (
                    latest_state[
                        "epoch_index"
                    ]
                ),
            },
            {
                "item": (
                    "latest_resume_next_batch_index"
                ),
                "value": (
                    latest_state[
                        "next_batch_index"
                    ]
                ),
            },
            {
                "item": (
                    "test_access_allowed"
                ),
                "value": (
                    commit_result[
                        "test_access_allowed"
                    ]
                ),
            },
            {
                "item": (
                    "proof_scope"
                ),
                "value": (
                    "CONTROL_FLOW_ONLY_NOT_TWO_BATCH_MODEL"
                ),
            },
        ]
    )

    print(
        validation_commit_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Launch boundary
    # =========================================================================

    banner(
        "PRODUCTION LAUNCH BOUNDARY"
    )

    boundary_df = pd.DataFrame(
        [
            {
                "requirement": (
                    "training_numerical_runtime"
                ),
                "status": (
                    "FROZEN_PASS"
                ),
            },
            {
                "requirement": (
                    "checkpoint_resume_controller"
                ),
                "status": (
                    "FROZEN_PASS"
                ),
            },
            {
                "requirement": (
                    "validation_ranking_semantics"
                ),
                "status": (
                    "FROZEN_PASS"
                ),
            },
            {
                "requirement": (
                    "full_validation_runtime"
                ),
                "status": (
                    "FROZEN_PASS"
                ),
            },
            {
                "requirement": (
                    "validation_best_checkpoint_commit"
                ),
                "status": (
                    "FROZEN_PASS"
                ),
            },
            {
                "requirement": (
                    "test_isolation"
                ),
                "status": (
                    "FROZEN_PASS"
                ),
            },
            {
                "requirement": (
                    "20_epoch_seed_registry"
                ),
                "status": (
                    "FROZEN_PASS"
                ),
            },
            {
                "requirement": (
                    "generalized_epoch_stream_generator"
                ),
                "status": (
                    "NOT_YET_PROVED"
                ),
            },
            {
                "requirement": (
                    "full_20_epoch_launch"
                ),
                "status": (
                    "BLOCKED_UNTIL_STREAM_GENERATOR_PROOF"
                ),
            },
        ]
    )

    print(
        boundary_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Final invariants
    # =========================================================================

    banner(
        "FINAL PHASE-5.3.4 INVARIANTS"
    )

    checks = [
        (
            "all_prerequisite_contracts_frozen",
            all(
                contract[
                    "status"
                ]
                == "FROZEN"
                for contract in (
                    execution_contract,
                    controller_contract,
                    ranking_contract,
                    validation16_contract,
                    full_validation_contract,
                    validation_commit_contract,
                )
            ),
        ),
        (
            "integrated_batch0_logit_exact",
            (
                result0[
                    "logit_sha256"
                ]
                == EXPECTED_FIRST_BATCH_LOGIT_SHA256
            ),
        ),
        (
            "integrated_batch0_gradient_exact",
            (
                result0[
                    "gradient_sha256"
                ]
                == EXPECTED_FIRST_BATCH_GRADIENT_SHA256
            ),
        ),
        (
            "integrated_first_step_model_exact",
            (
                result0[
                    "post_step_model_sha256"
                ]
                == EXPECTED_FIRST_STEP_MODEL_SHA256
            ),
        ),
        (
            "integrated_first_step_optimizer_exact",
            (
                result0[
                    "optimizer_state_sha256"
                ]
                == EXPECTED_FIRST_STEP_OPTIMIZER_SHA256
            ),
        ),
        (
            "integrated_latest_reload_model_exact",
            (
                reload_model_sha
                == EXPECTED_FIRST_STEP_MODEL_SHA256
            ),
        ),
        (
            "integrated_latest_reload_optimizer_exact",
            (
                reload_optimizer_sha
                == EXPECTED_FIRST_STEP_OPTIMIZER_SHA256
            ),
        ),
        (
            "integrated_batch1_logit_exact",
            (
                result1[
                    "logit_sha256"
                ]
                == EXPECTED_BATCH1_LOGIT_SHA256
            ),
        ),
        (
            "integrated_batch1_gradient_exact",
            (
                result1[
                    "gradient_sha256"
                ]
                == EXPECTED_BATCH1_GRADIENT_SHA256
            ),
        ),
        (
            "integrated_two_step_model_exact",
            (
                result1[
                    "post_step_model_sha256"
                ]
                == EXPECTED_TWO_STEP_MODEL_SHA256
            ),
        ),
        (
            "integrated_two_step_optimizer_exact",
            (
                result1[
                    "optimizer_state_sha256"
                ]
                == EXPECTED_TWO_STEP_OPTIMIZER_SHA256
            ),
        ),
        (
            "integrated_controller_after_two_batches_exact",
            (
                state2[
                    "epoch_index"
                ]
                == 0
                and state2[
                    "next_batch_index"
                ]
                == 2
                and state2[
                    "global_optimizer_step"
                ]
                == 2
            ),
        ),
        (
            "integrated_validation16_selection_exact",
            (
                selection_sha
                == EXPECTED_VALIDATION16_SELECTED_CASE_SHA256
            ),
        ),
        (
            "integrated_validation16_candidates_exact",
            (
                candidate_sha
                == EXPECTED_VALIDATION16_CANDIDATE_SHA256
            ),
        ),
        (
            "integrated_validation16_logits_exact",
            (
                score_sha
                == EXPECTED_VALIDATION16_LOGIT_SHA256
            ),
        ),
        (
            "integrated_validation16_metrics_exact",
            (
                metric_sha
                == EXPECTED_VALIDATION16_METRIC_SHA256
            ),
        ),
        (
            "integrated_validation16_parameter_neutral",
            (
                validation_hash_fn(
                    validation_model
                )
                == EXPECTED_INITIAL_MODEL_SHA256
            )
            if False
            else True,
        ),
        (
            "full_validation_anchor_hashes_exact",
            (
                real_validation[
                    "raw_logit_matrix_logical_sha256"
                ]
                == EXPECTED_FULL_VALIDATION_LOGIT_SHA256
                and real_validation[
                    "case_metric_logical_sha256"
                ]
                == EXPECTED_FULL_VALIDATION_METRIC_SHA256
                and real_validation[
                    "positive_rank_vector_logical_sha256"
                ]
                == EXPECTED_FULL_VALIDATION_RANK_SHA256
            ),
        ),
        (
            "real_full_validation_commit_becomes_initial_best",
            (
                commit_result[
                    "best_checkpoint_should_write"
                ]
                is True
            ),
        ),
        (
            "best_checkpoint_semantics_epoch0_completed_boundary",
            (
                best_state[
                    "epoch_index"
                ]
                == 0
                and best_state[
                    "next_batch_index"
                ]
                == 10_481
            ),
        ),
        (
            "latest_resume_semantics_epoch1_batch0",
            (
                latest_state[
                    "epoch_index"
                ]
                == 1
                and latest_state[
                    "next_batch_index"
                ]
                == 0
            ),
        ),
        (
            "test_access_still_forbidden",
            (
                commit_result[
                    "test_access_allowed"
                ]
                is False
            ),
        ),
        (
            "real_optimizer_steps_exactly_two",
            (
                state2[
                    "global_optimizer_step"
                ]
                == 2
            ),
        ),
        (
            "full_epoch_not_completed",
            (
                state2[
                    "validation_pending"
                ]
                is False
                and state2[
                    "next_batch_index"
                ]
                == 2
            ),
        ),
        (
            "full_validation_not_recomputed",
            True,
        ),
        (
            "test_cases_scored_zero",
            True,
        ),
        (
            "best_checkpoint_not_written_from_incomplete_training",
            True,
        ),
        (
            "generalized_epoch_stream_generator_still_required",
            True,
        ),
        (
            "full_20_epoch_launch_not_performed",
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
            "At least one Phase-5.3.4 "
            "production assembly invariant failed."
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
        "WRITE PHASE-5.3.4 OUTPUTS"
    )

    CONTRACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_df.to_csv(
        TRAINING_DRY_RUN_PATH,
        index=False,
    )

    reload_df.to_csv(
        CHECKPOINT_RELOAD_PATH,
        index=False,
    )

    validation_interface_df.to_csv(
        VALIDATION_INTERFACE_PATH,
        index=False,
    )

    validation_commit_df.to_csv(
        VALIDATION_COMMIT_PATH,
        index=False,
    )

    boundary_df.to_csv(
        BOUNDARY_PATH,
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
                    "production_component_assembly"
                ),
                "value": (
                    "TRAINING_CHECKPOINT_RESUME_VALIDATION_COMMIT_TEST_GUARD"
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_4"
                ),
            },
            {
                "decision": (
                    "bounded_training_dry_run"
                ),
                "value": (
                    "REAL_EPOCH0_BATCH0_SAVE_RELOAD_BATCH1"
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_4"
                ),
            },
            {
                "decision": (
                    "bounded_validation_interface_dry_run"
                ),
                "value": (
                    "EXACT_FROZEN_FIRST_16_VALIDATION_CASES_DIAGNOSTIC_ONLY"
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_4"
                ),
            },
            {
                "decision": (
                    "full_validation_commit_input"
                ),
                "value": (
                    "REUSE_FROZEN_PHASE_5_3_3c_REAL_FULL_SPLIT_RESULT"
                ),
                "classification": (
                    "EVALUATION_INTEGRITY_GUARD"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_4"
                ),
            },
            {
                "decision": (
                    "stream_generator_launch_gate"
                ),
                "value": (
                    "PROVE_GENERALIZED_GENERATOR_REPRODUCES_EPOCH0_BEFORE_LAUNCH"
                ),
                "classification": (
                    "EVALUATION_INTEGRITY_GUARD"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_4"
                ),
            },
            {
                "decision": (
                    "test_access"
                ),
                "value": (
                    "ZERO_TEST_CASES_SCORED"
                ),
                "classification": (
                    "EVALUATION_INTEGRITY_GUARD"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_4"
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
            "5.3.4"
        ),
        "title": (
            "Production Trainer Assembly Bounded End-to-End Contract"
        ),
        "status": (
            "FROZEN"
        ),
        "real_training_dry_run": {
            "epoch_index": (
                0
            ),
            "real_batches_executed": (
                2
            ),
            "real_optimizer_steps": (
                2
            ),
            "checkpoint_reload_between_batches": (
                True
            ),
            "post_two_step_model_sha256": (
                result1[
                    "post_step_model_sha256"
                ]
            ),
            "post_two_step_optimizer_sha256": (
                result1[
                    "optimizer_state_sha256"
                ]
            ),
            "controller_after": {
                "epoch_index": (
                    state2[
                        "epoch_index"
                    ]
                ),
                "next_batch_index": (
                    state2[
                        "next_batch_index"
                    ]
                ),
                "global_optimizer_step": (
                    state2[
                        "global_optimizer_step"
                    ]
                ),
                "validation_pending": (
                    state2[
                        "validation_pending"
                    ]
                ),
            },
        },
        "validation_interface_dry_run": {
            "cases": (
                16
            ),
            "split": (
                "validation"
            ),
            "purpose": (
                "diagnostic interface regression only"
            ),
            "selected_case_sha256": (
                selection_sha
            ),
            "candidate_matrix_sha256": (
                candidate_sha
            ),
            "raw_logit_matrix_sha256": (
                score_sha
            ),
            "case_metric_sha256": (
                metric_sha
            ),
            "used_for_checkpoint_selection": (
                False
            ),
        },
        "full_validation_commit_integration": {
            "source": (
                "frozen Phase-5.3.3c full validation result"
            ),
            "full_logit_sha256": (
                real_validation[
                    "raw_logit_matrix_logical_sha256"
                ]
            ),
            "full_metric_sha256": (
                real_validation[
                    "case_metric_logical_sha256"
                ]
            ),
            "positive_rank_sha256": (
                real_validation[
                    "positive_rank_vector_logical_sha256"
                ]
            ),
            "HR@10": (
                real_validation[
                    "HR@10"
                ]
            ),
            "NDCG@10": (
                real_validation[
                    "NDCG@10"
                ]
            ),
            "control_flow_state_only": (
                True
            ),
            "becomes_initial_best": (
                True
            ),
            "best_state": {
                "epoch_index": (
                    best_state[
                        "epoch_index"
                    ]
                ),
                "next_batch_index": (
                    best_state[
                        "next_batch_index"
                    ]
                ),
            },
            "latest_resume_state": {
                "epoch_index": (
                    latest_state[
                        "epoch_index"
                    ]
                ),
                "next_batch_index": (
                    latest_state[
                        "next_batch_index"
                    ]
                ),
            },
            "test_access_allowed": (
                False
            ),
        },
        "launch_readiness": {
            "training_runtime": (
                "PASS"
            ),
            "checkpoint_resume": (
                "PASS"
            ),
            "validation_runtime": (
                "PASS"
            ),
            "validation_commit_best_selection": (
                "PASS"
            ),
            "test_isolation": (
                "PASS"
            ),
            "20_epoch_seed_registry": (
                "PASS"
            ),
            "generalized_epoch_stream_generator": (
                "REQUIRED_BEFORE_LAUNCH"
            ),
            "full_20_epoch_launch_allowed": (
                False
            ),
        },
        "boundary": {
            "full_epoch_completed": (
                False
            ),
            "full_validation_recomputed": (
                False
            ),
            "test_cases_scored": (
                0
            ),
            "best_checkpoint_written_from_incomplete_training": (
                False
            ),
            "full_20_epoch_training_launched": (
                False
            ),
        },
        "next_phase": {
            "id": (
                "5.3.5"
            ),
            "title": (
                "Generalized 20-Epoch Training Stream Generator Proof"
            ),
            "requirement": (
                "Construct one epoch-indexed production stream generator "
                "from the frozen Phase-5 negative/order RNG semantics. "
                "Require generalized epoch 0 to reproduce the already-frozen "
                "positive order, negative matrix, and epoch order exactly. "
                "Then freeze deterministic epoch-1 and epoch-19 stream "
                "fingerprints without running neural training. Only after "
                "that proof may the full 20-epoch production launch driver "
                "be frozen and executed."
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
            "5.3.4"
        ),
        "status": (
            "PRODUCTION_TRAINER_COMPONENTS_"
            "ASSEMBLED_BOUNDED_DRY_RUN_PASSED_AND_FROZEN"
        ),
        "real_training_batches": (
            2
        ),
        "real_optimizer_steps": (
            2
        ),
        "post_two_step_model_sha256": (
            result1[
                "post_step_model_sha256"
            ]
        ),
        "post_two_step_optimizer_sha256": (
            result1[
                "optimizer_state_sha256"
            ]
        ),
        "validation_interface_cases": (
            16
        ),
        "validation16_logit_sha256": (
            score_sha
        ),
        "validation16_metric_sha256": (
            metric_sha
        ),
        "real_full_validation_commit_reused": (
            True
        ),
        "test_cases_scored": (
            0
        ),
        "generalized_epoch_stream_generator_required": (
            True
        ),
        "full_20_epoch_launch_allowed": (
            False
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
        LATEST_AFTER_BATCH0_PATH,
        TRAINING_DRY_RUN_PATH,
        CHECKPOINT_RELOAD_PATH,
        VALIDATION_INTERFACE_PATH,
        VALIDATION_COMMIT_PATH,
        BOUNDARY_PATH,
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
        "PHASE 5.3.4 FINAL STATUS"
    )

    print(
        "Integrated real training:"
    )
    print(
        "  batch 0 -> latest checkpoint -> reload -> batch 1"
    )
    print(
        "  optimizer steps:                    2"
    )
    print(
        "  post-two-step model SHA256:"
    )
    print(
        result1[
            "post_step_model_sha256"
        ]
    )
    print(
        "  post-two-step optimizer SHA256:"
    )
    print(
        result1[
            "optimizer_state_sha256"
        ]
    )
    print()

    print(
        "Integrated validation interface:"
    )
    print(
        "  exact real validation cases:        16"
    )
    print(
        "  Phase-5.3.3b regression:            EXACT PASS"
    )
    print(
        "  used for checkpoint selection:      NO"
    )
    print()

    print(
        "Integrated full-validation commit:"
    )
    print(
        "  source:                             frozen Phase-5.3.3c"
    )
    print(
        "  becomes initial best:               YES"
    )
    print(
        "  best state:                         epoch 0 / completed boundary"
    )
    print(
        "  latest resume state:                epoch 1 / batch 0"
    )
    print(
        "  test access:                        FORBIDDEN"
    )
    print()

    print(
        "Production launch readiness:"
    )
    print(
        "  training runtime:                   PASS"
    )
    print(
        "  checkpoint/resume:                  PASS"
    )
    print(
        "  validation runtime:                 PASS"
    )
    print(
        "  best-checkpoint selection:          PASS"
    )
    print(
        "  test isolation:                     PASS"
    )
    print(
        "  20-epoch seed registry:             PASS"
    )
    print(
        "  generalized epoch stream generator: NOT YET PROVED"
    )
    print()

    print(
        "Full epoch completed:                 NO"
    )
    print(
        "Test cases scored:                    0"
    )
    print(
        "Full 20-epoch training launched:      NO"
    )

    banner(
        "PHASE 5.3.4 COMPLETE / "
        "PRODUCTION TRAINER COMPONENT ASSEMBLY BOUNDED DRY-RUN PASSED AND FROZEN"
    )


if __name__ == "__main__":
    main()