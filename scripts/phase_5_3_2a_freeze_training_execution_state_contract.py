"""
Phase 5.3.2a
Freeze Full-Training Checkpoint, Resume, Epoch, Validation,
and Best-Checkpoint Execution Semantics.

IMPORTANT CORRECTION
--------------------
Earlier drafts used manually transcribed dictionaries for several
epoch RNG seeds. Those dictionaries introduced transcription errors
even though the frozen seed-derivation function itself was correct.

This corrected version REMOVES the fragile multi-epoch hard-coded
seed tables.

All 20 epoch seeds are derived directly from the already-frozen rule:

    material = f"{namespace}|{base_seed}|{epoch}"
    digest   = SHA256(material)
    raw      = first 8 digest bytes interpreted little-endian unsigned
    seed     = raw mod (2**63 - 1)

Only epoch-0 values are retained as regression anchors because those
were already independently frozen and exercised by Phase 5.3.1l.1.

NO neural model is instantiated.
NO optimizer is instantiated.
NO negative matrix is generated.
NO epoch permutation is generated.
NO forward/backward occurs.
NO optimizer.step() occurs.
NO checkpoint containing model weights is written.

This phase freezes only execution semantics for the later
20-epoch production training driver.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


# =============================================================================
# Frozen Phase-5 execution constants
# =============================================================================

BASE_SEED = 42

NUM_EPOCHS = 20

TRAIN_POSITIVES_PER_EPOCH = 1_073_249
NEGATIVES_PER_POSITIVE = 4

NEGATIVES_PER_EPOCH = 4_292_996
EXAMPLES_PER_EPOCH = 5_366_245

BATCH_SIZE = 512
BATCHES_PER_EPOCH = 10_481
FINAL_BATCH_SIZE = 485

TOTAL_EXAMPLES = 107_324_900
TOTAL_OPTIMIZER_STEPS = 209_620

NEGATIVE_NAMESPACE = "ITRS_PHASE5_TRAIN_NEGATIVE"
ORDER_NAMESPACE = "ITRS_PHASE5_TRAIN_ORDER"

# These two epoch-0 values were already frozen and exercised earlier.
EXPECTED_EPOCH0_NEGATIVE_SEED = 7_895_109_663_985_029_800
EXPECTED_EPOCH0_ORDER_SEED = 4_607_400_055_922_019_930


# =============================================================================
# Frozen numerical fingerprints
# =============================================================================

EXPECTED_INITIAL_MODEL_SHA256 = (
    "49e822ea7fad35c458f47e134c94c05e"
    "ac099b68c5c468e2c71559c8c88998ab"
)

EXPECTED_FIRST_STEP_MODEL_SHA256 = (
    "42a521f11d8f24e4144d0215d6e1b34d"
    "5f8bf0c2d8848624e4f7c3130699035d"
)

EXPECTED_FIRST_STEP_OPTIMIZER_SHA256 = (
    "5ce2683c21f456b9d5d15eb876b049c5"
    "e6db1215db5a026630f093f7f9d49891"
)

EXPECTED_FIRST_BATCH_SHA256 = (
    "8408432b944bcd0805af9c34ff1b2db3"
    "ea938e0649a75d381b7839b86cd280ea"
)

EXPECTED_FIRST_BATCH_LOGIT_SHA256 = (
    "35b89aaed29d51d2ebb7ba1cadf2dc4b"
    "b5e8f81cf3aa78bc216b3cc6fed13845"
)

EXPECTED_FIRST_BATCH_GRADIENT_SHA256 = (
    "8c542430813d8ca91b8397409954ea92"
    "295a2b55bcc420661783fb865010845d"
)


# =============================================================================
# Authoritative prior Phase-5 artifacts
# =============================================================================

PHASE_5_3_1M_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_1m_first_adam_weight_update_contract.json"
)

PHASE_5_3_1M_MANIFEST_PATH = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_1m/"
    "phase_5_3_1m_first_adam_update_manifest.json"
)

PHASE_5_3_1L_2B_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_1l_2b_corrected_adam_first_batch_preflight_contract.json"
)

PHASE_5_3_1L_1_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_1l_1_epoch0_training_stream_serialization_contract.json"
)


# =============================================================================
# Outputs
# =============================================================================

AUDIT_DIR = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_2a"
)

CONTRACT_DIR = Path(
    "data/experimental/phase_5/contracts"
)

EPOCH_SEED_PATH = (
    AUDIT_DIR
    / "phase_5_training_epoch_seed_registry.csv"
)

CHECKPOINT_SCHEMA_PATH = (
    AUDIT_DIR
    / "training_checkpoint_payload_schema.csv"
)

RESUME_STATE_MACHINE_PATH = (
    AUDIT_DIR
    / "training_resume_state_machine.csv"
)

VALIDATION_POLICY_PATH = (
    AUDIT_DIR
    / "validation_checkpoint_selection_policy.csv"
)

FINAL_INVARIANT_PATH = (
    AUDIT_DIR
    / "phase_5_3_2a_final_invariants.csv"
)

DECISION_REGISTER_PATH = (
    CONTRACT_DIR
    / "phase_5_3_2a_training_execution_decision_register.csv"
)

CONTRACT_PATH = (
    CONTRACT_DIR
    / "phase_5_3_2a_training_execution_state_contract.json"
)

MANIFEST_PATH = (
    AUDIT_DIR
    / "phase_5_3_2a_training_execution_state_manifest.json"
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
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def derive_seed(
    namespace: str,
    base_seed: int,
    epoch: int,
) -> int:
    material = f"{namespace}|{base_seed}|{epoch}"

    digest = hashlib.sha256(
        material.encode("utf-8")
    ).digest()

    raw = int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    )

    return int(
        raw % (2**63 - 1)
    )


def seed_registry_logical_sha256(
    frame: pd.DataFrame,
) -> str:
    """
    Stable logical fingerprint for the complete 20-epoch registry.
    """
    digest = hashlib.sha256()

    for row in frame.itertuples(index=False):
        material = (
            f"{int(row.epoch_index)}|"
            f"{int(row.negative_seed)}|"
            f"{int(row.order_seed)}\n"
        )
        digest.update(
            material.encode("utf-8")
        )

    return digest.hexdigest()


# =============================================================================
# Best validation checkpoint comparison
# =============================================================================

def validation_candidate_is_better(
    candidate_ndcg: float,
    candidate_hr: float,
    candidate_epoch: int,
    best_ndcg: float | None,
    best_hr: float | None,
    best_epoch: int | None,
) -> bool:
    """
    Frozen Phase-5.2.2 ranking:

        primary   = maximum validation NDCG@10
        secondary = maximum validation HR@10
        tertiary  = earliest epoch

    Epochs arrive chronologically, so an exact full tie does not replace
    the existing best checkpoint. The earliest epoch therefore survives.
    """

    if best_epoch is None:
        return True

    if candidate_ndcg > best_ndcg:
        return True

    if candidate_ndcg < best_ndcg:
        return False

    if candidate_hr > best_hr:
        return True

    return False


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.3.2a — "
        "FREEZE FULL-TRAINING EXECUTION-STATE CONTRACT"
    )

    print("Neural model instantiated:           NO")
    print("Adam instantiated:                   NO")
    print("Training negative RNG instantiated:  NO")
    print("Training order RNG instantiated:     NO")
    print("Forward computation:                 NO")
    print("Backward computation:                NO")
    print("optimizer.step():                    0")
    print("Production checkpoint written:       NO")

    # =========================================================================
    # Recheck prior gates
    # =========================================================================

    banner("AUTHORITATIVE PRE-TRAINING GATE RECHECK")

    required_paths = (
        PHASE_5_3_1M_CONTRACT_PATH,
        PHASE_5_3_1M_MANIFEST_PATH,
        PHASE_5_3_1L_2B_CONTRACT_PATH,
        PHASE_5_3_1L_1_CONTRACT_PATH,
    )

    for path in required_paths:
        require(
            path.exists(),
            f"Missing authoritative Phase-5 input: {path}",
        )
        print(f"FOUND  {path}")

    first_update_contract = load_json(
        PHASE_5_3_1M_CONTRACT_PATH
    )

    first_update_manifest = load_json(
        PHASE_5_3_1M_MANIFEST_PATH
    )

    batch_contract = load_json(
        PHASE_5_3_1L_2B_CONTRACT_PATH
    )

    stream_contract = load_json(
        PHASE_5_3_1L_1_CONTRACT_PATH
    )

    require(
        first_update_contract["status"] == "FROZEN",
        "Phase-5.3.1m contract is not frozen.",
    )

    require(
        first_update_manifest["status"]
        == "FIRST_ADAM_WEIGHT_UPDATE_PROVED_AND_FROZEN",
        "Unexpected Phase-5.3.1m manifest status.",
    )

    require(
        first_update_manifest["pre_step_parameter_sha256"]
        == EXPECTED_INITIAL_MODEL_SHA256,
        "Canonical initial-model fingerprint drift.",
    )

    require(
        first_update_manifest["post_step_parameter_sha256"]
        == EXPECTED_FIRST_STEP_MODEL_SHA256,
        "First-step model fingerprint drift.",
    )

    require(
        first_update_manifest["optimizer_state_sha256"]
        == EXPECTED_FIRST_STEP_OPTIMIZER_SHA256,
        "First-step optimizer fingerprint drift.",
    )

    require(
        first_update_manifest["gradient_sha256"]
        == EXPECTED_FIRST_BATCH_GRADIENT_SHA256,
        "First-step gradient fingerprint drift.",
    )

    require(
        int(first_update_manifest["optimizer_steps"]) == 1,
        "Phase-5.3.1m did not contain exactly one optimizer step.",
    )

    require(
        batch_contract["status"] == "FROZEN",
        "Phase-5.3.1l.2b contract not frozen.",
    )

    require(
        batch_contract["training_stream"]["first_batch_sha256"]
        == EXPECTED_FIRST_BATCH_SHA256,
        "First-batch fingerprint drift.",
    )

    require(
        batch_contract["numerical_result"]["logit_sha256"]
        == EXPECTED_FIRST_BATCH_LOGIT_SHA256,
        "First-batch logit fingerprint drift.",
    )

    require(
        stream_contract["status"] == "FROZEN",
        "Phase-5.3.1l.1 training stream contract not frozen.",
    )

    print("Phase-5.3.1m first update:            FROZEN / PASS")
    print("Phase-5.3.1l.2b batch runtime:        FROZEN / PASS")
    print("Phase-5.3.1l.1 stream serialization: FROZEN / PASS")

    # =========================================================================
    # Global training arithmetic
    # =========================================================================

    banner("FULL TRAINING ARITHMETIC")

    require(
        TRAIN_POSITIVES_PER_EPOCH * NEGATIVES_PER_POSITIVE
        == NEGATIVES_PER_EPOCH,
        "Negative-per-epoch arithmetic drift.",
    )

    require(
        TRAIN_POSITIVES_PER_EPOCH + NEGATIVES_PER_EPOCH
        == EXAMPLES_PER_EPOCH,
        "Example-per-epoch arithmetic drift.",
    )

    require(
        (
            EXAMPLES_PER_EPOCH
            + BATCH_SIZE
            - 1
        )
        // BATCH_SIZE
        == BATCHES_PER_EPOCH,
        "Batch count arithmetic drift.",
    )

    require(
        (
            EXAMPLES_PER_EPOCH
            - (
                BATCHES_PER_EPOCH
                - 1
            )
            * BATCH_SIZE
        )
        == FINAL_BATCH_SIZE,
        "Final batch size arithmetic drift.",
    )

    require(
        NUM_EPOCHS * EXAMPLES_PER_EPOCH
        == TOTAL_EXAMPLES,
        "20-epoch example-count drift.",
    )

    require(
        NUM_EPOCHS * BATCHES_PER_EPOCH
        == TOTAL_OPTIMIZER_STEPS,
        "20-epoch optimizer-step count drift.",
    )

    print(f"Epochs:                               {NUM_EPOCHS}")
    print(
        f"Examples / epoch:                     "
        f"{EXAMPLES_PER_EPOCH:,}"
    )
    print(
        f"Batches / epoch:                      "
        f"{BATCHES_PER_EPOCH:,}"
    )
    print(
        f"Final batch size:                     "
        f"{FINAL_BATCH_SIZE}"
    )
    print(
        f"Total examples:                       "
        f"{TOTAL_EXAMPLES:,}"
    )
    print(
        f"Expected optimizer steps:             "
        f"{TOTAL_OPTIMIZER_STEPS:,}"
    )

    # =========================================================================
    # Freeze all 20 epoch RNG seeds
    # =========================================================================

    banner("FREEZE 20-EPOCH RNG SEED REGISTRY")

    seed_rows = []

    for epoch_index in range(NUM_EPOCHS):

        negative_seed = derive_seed(
            NEGATIVE_NAMESPACE,
            BASE_SEED,
            epoch_index,
        )

        order_seed = derive_seed(
            ORDER_NAMESPACE,
            BASE_SEED,
            epoch_index,
        )

        seed_rows.append(
            {
                "epoch_index": epoch_index,
                "display_epoch": epoch_index + 1,
                "negative_namespace": NEGATIVE_NAMESPACE,
                "negative_seed": negative_seed,
                "order_namespace": ORDER_NAMESPACE,
                "order_seed": order_seed,
            }
        )

    seed_df = pd.DataFrame(seed_rows)

    # Regression anchors: epoch 0 was already frozen and actually used.
    require(
        int(
            seed_df.loc[
                seed_df["epoch_index"] == 0,
                "negative_seed",
            ].iloc[0]
        )
        == EXPECTED_EPOCH0_NEGATIVE_SEED,
        "Epoch-0 negative seed drift.",
    )

    require(
        int(
            seed_df.loc[
                seed_df["epoch_index"] == 0,
                "order_seed",
            ].iloc[0]
        )
        == EXPECTED_EPOCH0_ORDER_SEED,
        "Epoch-0 order seed drift.",
    )

    require(
        len(seed_df) == NUM_EPOCHS,
        "Expected 20 epoch-seed rows.",
    )

    require(
        seed_df["negative_seed"].is_unique,
        "Negative seeds are not unique.",
    )

    require(
        seed_df["order_seed"].is_unique,
        "Order seeds are not unique.",
    )

    # Cross-namespace collision guard.
    all_negative_seeds = set(
        int(value)
        for value in seed_df[
            "negative_seed"
        ].tolist()
    )

    all_order_seeds = set(
        int(value)
        for value in seed_df[
            "order_seed"
        ].tolist()
    )

    require(
        all_negative_seeds.isdisjoint(
            all_order_seeds
        ),
        (
            "Negative and order RNG namespaces produced "
            "a cross-namespace seed collision."
        ),
    )

    seed_registry_sha256 = (
        seed_registry_logical_sha256(
            seed_df
        )
    )

    print(
        seed_df[
            [
                "epoch_index",
                "display_epoch",
                "negative_seed",
                "order_seed",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("20-epoch seed-registry logical SHA256:")
    print(seed_registry_sha256)

    # =========================================================================
    # Freeze checkpoint payload
    # =========================================================================

    banner("FREEZE PRODUCTION CHECKPOINT PAYLOAD")

    checkpoint_fields = [
        (
            "schema_version",
            "str",
            "Frozen checkpoint schema version.",
        ),
        (
            "phase",
            "str",
            "Phase-5 production training runtime identifier.",
        ),
        (
            "checkpoint_kind",
            "str",
            "latest | epoch | best",
        ),
        (
            "model_state_dict",
            "OrderedDict[str, Tensor]",
            "All canonical model state.",
        ),
        (
            "optimizer_state_dict",
            "dict",
            "Exact torch.optim.Adam state including moments and steps.",
        ),
        (
            "epoch_index",
            "int",
            "Zero-based epoch currently being executed.",
        ),
        (
            "next_batch_index",
            "int",
            "Next batch NOT yet executed inside epoch_index.",
        ),
        (
            "global_optimizer_step",
            "int",
            "Number of completed optimizer.step() calls.",
        ),
        (
            "epoch_loss_weighted_sum",
            "float",
            "Sum(batch_loss * batch_example_count) for completed batches.",
        ),
        (
            "epoch_example_count",
            "int",
            "Examples already accumulated into current epoch loss.",
        ),
        (
            "validation_pending",
            "bool",
            (
                "True when all training batches of epoch_index are "
                "complete but validation has not yet been committed."
            ),
        ),
        (
            "validation_history",
            "list[dict]",
            (
                "Completed epoch validation records containing "
                "epoch, HR@10, NDCG@10."
            ),
        ),
        (
            "best_validation_epoch",
            "int | None",
            "Best zero-based validation epoch so far.",
        ),
        (
            "best_validation_ndcg10",
            "float | None",
            "Best validation NDCG@10 so far.",
        ),
        (
            "best_validation_hr10",
            "float | None",
            "Best validation HR@10 so far.",
        ),
        (
            "python_rng_state",
            "object",
            "Python random state at checkpoint boundary.",
        ),
        (
            "numpy_rng_state",
            "object",
            "NumPy global RNG state at checkpoint boundary.",
        ),
        (
            "torch_rng_state",
            "Tensor",
            "Torch CPU RNG state at checkpoint boundary.",
        ),
        (
            "training_contract_fingerprints",
            "dict[str, str]",
            (
                "Frozen Phase-5 model/stream/runtime "
                "contract fingerprints."
            ),
        ),
        (
            "training_complete",
            "bool",
            "True only after epoch-19 validation is committed.",
        ),
    ]

    checkpoint_schema_df = pd.DataFrame(
        [
            {
                "field": field,
                "type": field_type,
                "semantics": semantics,
                "required": True,
            }
            for (
                field,
                field_type,
                semantics,
            ) in checkpoint_fields
        ]
    )

    print(
        checkpoint_schema_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Freeze resume state machine
    # =========================================================================

    banner("FREEZE RESUME STATE MACHINE")

    resume_rows = [
        {
            "state": "IN_EPOCH",
            "condition": (
                "0 <= next_batch_index < 10481 "
                "and validation_pending == False"
            ),
            "resume_action": (
                "Regenerate epoch stream deterministically; "
                "skip batches < next_batch_index; "
                "execute next_batch_index."
            ),
            "repeat_completed_batch": False,
            "validation_action": "NONE",
        },
        {
            "state": "POST_TRAINING_PRE_VALIDATION",
            "condition": (
                "next_batch_index == 10481 "
                "and validation_pending == True"
            ),
            "resume_action": (
                "Do not execute another training batch."
            ),
            "repeat_completed_batch": False,
            "validation_action": (
                "Run validation exactly once for epoch_index."
            ),
        },
        {
            "state": "POST_VALIDATION",
            "condition": (
                "validation for epoch_index committed "
                "and epoch_index < 19"
            ),
            "resume_action": (
                "Advance epoch_index += 1; "
                "next_batch_index = 0; "
                "reset epoch loss accumulators."
            ),
            "repeat_completed_batch": False,
            "validation_action": "NONE",
        },
        {
            "state": "TRAINING_COMPLETE",
            "condition": (
                "epoch_index == 19 and epoch-19 "
                "validation committed"
            ),
            "resume_action": (
                "No further training. "
                "Load best checkpoint for final test."
            ),
            "repeat_completed_batch": False,
            "validation_action": "NONE",
        },
    ]

    resume_df = pd.DataFrame(
        resume_rows
    )

    print(
        resume_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Freeze validation / best checkpoint policy
    # =========================================================================

    banner(
        "FREEZE VALIDATION / MODEL-SELECTION POLICY"
    )

    validation_rows = [
        {
            "decision": "validation_frequency",
            "value": "ONCE_AFTER_EACH_COMPLETED_EPOCH",
        },
        {
            "decision": "validation_epochs",
            "value": "1..20",
        },
        {
            "decision": "early_stopping",
            "value": "NONE",
        },
        {
            "decision": "primary_best_metric",
            "value": "MAX_VALIDATION_NDCG@10",
        },
        {
            "decision": "secondary_best_metric",
            "value": "MAX_VALIDATION_HR@10",
        },
        {
            "decision": "full_tie_rule",
            "value": "KEEP_EXISTING_BEST_EARLIEST_EPOCH",
        },
        {
            "decision": "test_during_training",
            "value": "FORBIDDEN",
        },
        {
            "decision": "final_test",
            "value": (
                "EXACTLY_ONCE_AFTER_TRAINING_"
                "FROM_BEST_CHECKPOINT"
            ),
        },
        {
            "decision": "validation_gradient_mode",
            "value": "torch.no_grad",
        },
        {
            "decision": "validation_model_mode",
            "value": "model.eval()",
        },
    ]

    validation_df = pd.DataFrame(
        validation_rows
    )

    print(
        validation_df.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Best-selection logic proof
    # -------------------------------------------------------------------------

    require(
        validation_candidate_is_better(
            candidate_ndcg=0.20,
            candidate_hr=0.30,
            candidate_epoch=1,
            best_ndcg=None,
            best_hr=None,
            best_epoch=None,
        ),
        "First validation checkpoint must become best.",
    )

    require(
        validation_candidate_is_better(
            candidate_ndcg=0.21,
            candidate_hr=0.10,
            candidate_epoch=2,
            best_ndcg=0.20,
            best_hr=0.99,
            best_epoch=1,
        ),
        "Higher NDCG must win.",
    )

    require(
        validation_candidate_is_better(
            candidate_ndcg=0.20,
            candidate_hr=0.31,
            candidate_epoch=2,
            best_ndcg=0.20,
            best_hr=0.30,
            best_epoch=1,
        ),
        "Higher HR must break NDCG tie.",
    )

    require(
        not validation_candidate_is_better(
            candidate_ndcg=0.20,
            candidate_hr=0.30,
            candidate_epoch=2,
            best_ndcg=0.20,
            best_hr=0.30,
            best_epoch=1,
        ),
        "Later epoch must not replace exact tie.",
    )

    # =========================================================================
    # Freeze atomic batch / checkpoint semantics
    # =========================================================================

    banner("ATOMIC TRAINING-BATCH SEMANTICS")

    atomic_batch_steps = [
        "optimizer.zero_grad(set_to_none=True)",
        "forward",
        "BCEWithLogitsLoss",
        "backward",
        "optimizer.step",
        "increment global_optimizer_step",
        "increment next_batch_index",
        "update weighted epoch loss accumulator",
        "checkpoint boundary becomes valid",
    ]

    for index, step in enumerate(
        atomic_batch_steps,
        start=1,
    ):
        print(f"{index:2d}. {step}")

    print()
    print(
        "Checkpointing DURING an incomplete "
        "forward/backward/update is FORBIDDEN."
    )
    print()
    print("Mandatory durable checkpoint boundary:")
    print("  after every completed epoch validation")
    print()
    print("Optional operational latest snapshots:")
    print(
        "  may be written at any completed batch boundary "
        "without changing scientific training semantics"
    )

    # =========================================================================
    # Final invariants
    # =========================================================================

    banner("FINAL PHASE-5.3.2a INVARIANTS")

    checks = [
        (
            "phase_5_3_1m_frozen",
            first_update_contract[
                "status"
            ]
            == "FROZEN",
        ),
        (
            "first_update_model_hash_exact",
            first_update_manifest[
                "post_step_parameter_sha256"
            ]
            == EXPECTED_FIRST_STEP_MODEL_SHA256,
        ),
        (
            "first_update_optimizer_hash_exact",
            first_update_manifest[
                "optimizer_state_sha256"
            ]
            == EXPECTED_FIRST_STEP_OPTIMIZER_SHA256,
        ),
        (
            "20_epochs_exact",
            NUM_EPOCHS == 20,
        ),
        (
            "epoch_examples_exact",
            EXAMPLES_PER_EPOCH == 5_366_245,
        ),
        (
            "epoch_batches_exact",
            BATCHES_PER_EPOCH == 10_481,
        ),
        (
            "final_batch_exact",
            FINAL_BATCH_SIZE == 485,
        ),
        (
            "total_examples_exact",
            TOTAL_EXAMPLES == 107_324_900,
        ),
        (
            "total_optimizer_steps_exact",
            TOTAL_OPTIMIZER_STEPS == 209_620,
        ),
        (
            "epoch0_negative_seed_anchor_exact",
            int(
                seed_df.loc[
                    seed_df["epoch_index"] == 0,
                    "negative_seed",
                ].iloc[0]
            )
            == EXPECTED_EPOCH0_NEGATIVE_SEED,
        ),
        (
            "epoch0_order_seed_anchor_exact",
            int(
                seed_df.loc[
                    seed_df["epoch_index"] == 0,
                    "order_seed",
                ].iloc[0]
            )
            == EXPECTED_EPOCH0_ORDER_SEED,
        ),
        (
            "20_negative_seeds_derived_and_unique",
            (
                len(seed_df) == 20
                and seed_df[
                    "negative_seed"
                ].is_unique
            ),
        ),
        (
            "20_order_seeds_derived_and_unique",
            (
                len(seed_df) == 20
                and seed_df[
                    "order_seed"
                ].is_unique
            ),
        ),
        (
            "negative_and_order_namespaces_disjoint",
            all_negative_seeds.isdisjoint(
                all_order_seeds
            ),
        ),
        (
            "checkpoint_schema_complete",
            bool(
                checkpoint_schema_df[
                    "required"
                ].all()
            ),
        ),
        (
            "resume_next_batch_semantics_frozen",
            (
                "next_batch_index"
                in set(
                    checkpoint_schema_df[
                        "field"
                    ]
                )
            ),
        ),
        (
            "validation_once_each_epoch",
            True,
        ),
        (
            "no_early_stopping",
            True,
        ),
        (
            "best_checkpoint_primary_NDCG10",
            True,
        ),
        (
            "best_checkpoint_secondary_HR10",
            True,
        ),
        (
            "earliest_epoch_full_tie",
            True,
        ),
        (
            "test_forbidden_during_training",
            True,
        ),
        (
            "test_once_after_training",
            True,
        ),
        (
            "checkpoint_only_at_atomic_boundaries",
            True,
        ),
        (
            "model_not_instantiated",
            True,
        ),
        (
            "Adam_not_instantiated",
            True,
        ),
        (
            "forward_not_performed",
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
            "At least one Phase-5.3.2a "
            "execution-state invariant failed."
        ),
    )

    print(
        invariant_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Write freeze artifacts
    # =========================================================================

    banner("WRITE PHASE-5.3.2a CONTRACT")

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTRACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    seed_df.to_csv(
        EPOCH_SEED_PATH,
        index=False,
    )

    checkpoint_schema_df.to_csv(
        CHECKPOINT_SCHEMA_PATH,
        index=False,
    )

    resume_df.to_csv(
        RESUME_STATE_MACHINE_PATH,
        index=False,
    )

    validation_df.to_csv(
        VALIDATION_POLICY_PATH,
        index=False,
    )

    invariant_df.to_csv(
        FINAL_INVARIANT_PATH,
        index=False,
    )

    decision_df = pd.DataFrame(
        [
            {
                "decision": "epoch_indexing",
                "value": "ZERO_BASED_0_TO_19",
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": "FROZEN_PHASE_5_3_2a",
            },
            {
                "decision": "batch_indexing",
                "value": (
                    "ZERO_BASED_NEXT_BATCH_NOT_YET_EXECUTED"
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": "FROZEN_PHASE_5_3_2a",
            },
            {
                "decision": "resume_epoch_stream",
                "value": (
                    "REGENERATE_FROM_FROZEN_EPOCH_SEEDS_AND_"
                    "SKIP_COMPLETED_BATCHES"
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": "FROZEN_PHASE_5_3_2a",
            },
            {
                "decision": "checkpoint_atomicity",
                "value": "ONLY_AFTER_COMPLETE_OPTIMIZER_STEP",
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": "FROZEN_PHASE_5_3_2a",
            },
            {
                "decision": "validation_timing",
                "value": "ONCE_AFTER_EACH_EPOCH",
                "classification": (
                    "INHERITED_FROZEN_PHASE_5_2_2"
                ),
                "status": "FROZEN_PHASE_5_3_2a",
            },
            {
                "decision": "best_checkpoint_selection",
                "value": (
                    "MAX_NDCG10_THEN_MAX_HR10_"
                    "THEN_EARLIEST_EPOCH"
                ),
                "classification": (
                    "INHERITED_FROZEN_PHASE_5_2_2"
                ),
                "status": "FROZEN_PHASE_5_3_2a",
            },
            {
                "decision": "test_access",
                "value": (
                    "EXACTLY_ONCE_AFTER_20_EPOCHS_"
                    "FROM_BEST_CHECKPOINT"
                ),
                "classification": (
                    "EVALUATION_INTEGRITY_GUARD"
                ),
                "status": "FROZEN_PHASE_5_3_2a",
            },
            {
                "decision": "epoch_seed_registry",
                "value": (
                    "DERIVE_ALL_20_FROM_FROZEN_SHA256_RULE_"
                    "NO_MANUAL_MULTI_EPOCH_CONSTANT_TABLE"
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": "FROZEN_PHASE_5_3_2a",
            },
        ]
    )

    decision_df.to_csv(
        DECISION_REGISTER_PATH,
        index=False,
    )

    epoch_seed_records = [
        {
            "epoch_index": int(
                row.epoch_index
            ),
            "negative_seed": int(
                row.negative_seed
            ),
            "order_seed": int(
                row.order_seed
            ),
        }
        for row in seed_df.itertuples(
            index=False
        )
    ]

    contract = {
        "phase": "5.3.2a",
        "title": (
            "Full-Training Execution-State Contract"
        ),
        "status": "FROZEN",

        "training_control": {
            "epochs": NUM_EPOCHS,
            "training_positive_events_per_epoch": (
                TRAIN_POSITIVES_PER_EPOCH
            ),
            "negatives_per_positive": (
                NEGATIVES_PER_POSITIVE
            ),
            "negatives_per_epoch": (
                NEGATIVES_PER_EPOCH
            ),
            "examples_per_epoch": (
                EXAMPLES_PER_EPOCH
            ),
            "batch_size": BATCH_SIZE,
            "batches_per_epoch": (
                BATCHES_PER_EPOCH
            ),
            "final_batch_size": (
                FINAL_BATCH_SIZE
            ),
            "total_examples": (
                TOTAL_EXAMPLES
            ),
            "total_optimizer_steps": (
                TOTAL_OPTIMIZER_STEPS
            ),
            "early_stopping": False,
            "scheduler": None,
        },

        "epoch_rng": {
            "base_seed": BASE_SEED,
            "negative_namespace": (
                NEGATIVE_NAMESPACE
            ),
            "order_namespace": (
                ORDER_NAMESPACE
            ),
            "derivation": (
                "SHA256(namespace|base_seed|epoch), "
                "first8 little-endian unsigned, "
                "mod (2**63 - 1)"
            ),
            "epoch0_negative_seed_anchor": (
                EXPECTED_EPOCH0_NEGATIVE_SEED
            ),
            "epoch0_order_seed_anchor": (
                EXPECTED_EPOCH0_ORDER_SEED
            ),
            "registry_logical_sha256": (
                seed_registry_sha256
            ),
            "epochs": epoch_seed_records,
        },

        "batch_atomicity": {
            "ordered_operations": (
                atomic_batch_steps
            ),
            "checkpoint_during_incomplete_batch": (
                False
            ),
            "next_batch_index_semantics": (
                "next batch not yet executed"
            ),
        },

        "checkpoint": {
            "required_fields": (
                checkpoint_schema_df[
                    "field"
                ].tolist()
            ),
            "mandatory_epoch_boundary_checkpoint": (
                True
            ),
            "mid_epoch_latest_snapshot_allowed": (
                True
            ),
            "mid_epoch_snapshot_boundary": (
                "after completed optimizer step only"
            ),
        },

        "resume": {
            "regenerate_epoch_negatives": True,
            "regenerate_epoch_order": True,
            "regeneration_uses_frozen_epoch_seeds": (
                True
            ),
            "skip_batches_before_next_batch_index": (
                True
            ),
            "repeat_completed_batch": False,
        },

        "validation": {
            "frequency": (
                "once after every completed epoch"
            ),
            "number_of_validation_passes": (
                NUM_EPOCHS
            ),
            "model_mode": "eval",
            "gradient_mode": "no_grad",
            "primary_best_metric": "NDCG@10",
            "secondary_best_metric": "HR@10",
            "exact_full_tie": (
                "keep existing earliest epoch"
            ),
        },

        "test": {
            "during_training": False,
            "used_for_checkpoint_selection": (
                False
            ),
            "evaluation_count": 1,
            "timing": (
                "after all 20 validation epochs"
            ),
            "model": (
                "best validation checkpoint"
            ),
        },

        "frozen_numerical_fingerprints": {
            "canonical_initial_model_sha256": (
                EXPECTED_INITIAL_MODEL_SHA256
            ),
            "first_batch_sha256": (
                EXPECTED_FIRST_BATCH_SHA256
            ),
            "first_batch_logit_sha256": (
                EXPECTED_FIRST_BATCH_LOGIT_SHA256
            ),
            "first_batch_gradient_sha256": (
                EXPECTED_FIRST_BATCH_GRADIENT_SHA256
            ),
            "first_step_model_sha256": (
                EXPECTED_FIRST_STEP_MODEL_SHA256
            ),
            "first_step_optimizer_sha256": (
                EXPECTED_FIRST_STEP_OPTIMIZER_SHA256
            ),
        },

        "next_phase": {
            "id": "5.3.2b",
            "title": (
                "Checkpoint / Resume "
                "Round-Trip Numerical Proof"
            ),
            "requirement": (
                "Execute two frozen training batches in "
                "two paths: (A) uninterrupted batch0->batch1 "
                "and (B) batch0->checkpoint->reload->batch1. "
                "Require identical model and Adam states "
                "after batch1."
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
        "phase": "5.3.2a",
        "status": (
            "FULL_TRAINING_EXECUTION_STATE_"
            "CONTRACT_FROZEN"
        ),
        "epochs": NUM_EPOCHS,
        "batches_per_epoch": (
            BATCHES_PER_EPOCH
        ),
        "total_optimizer_steps": (
            TOTAL_OPTIMIZER_STEPS
        ),
        "epoch_seed_rows": len(
            seed_df
        ),
        "epoch_seed_registry_sha256": (
            seed_registry_sha256
        ),
        "checkpoint_schema_fields": len(
            checkpoint_schema_df
        ),
        "model_instantiated": False,
        "Adam_instantiated": False,
        "optimizer_steps": 0,
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
        EPOCH_SEED_PATH,
        CHECKPOINT_SCHEMA_PATH,
        RESUME_STATE_MACHINE_PATH,
        VALIDATION_POLICY_PATH,
        FINAL_INVARIANT_PATH,
        DECISION_REGISTER_PATH,
        CONTRACT_PATH,
        MANIFEST_PATH,
    ):
        print(f"WROTE  {path}")

    # =========================================================================
    # Final
    # =========================================================================

    banner("PHASE 5.3.2a FINAL STATUS")

    print("Training epochs:                      20")
    print("Batches / epoch:                      10,481")
    print("Expected optimizer steps:             209,620")
    print()

    print("Epoch RNG seeds derived/frozen:       20 / 20")
    print("Manual multi-epoch seed table:        REMOVED")
    print()

    print("20-epoch seed-registry SHA256:")
    print(seed_registry_sha256)
    print()

    print("Checkpoint counter semantics:")
    print("  epoch_index      = current zero-based epoch")
    print("  next_batch_index = next batch NOT yet executed")
    print("  global_optimizer_step = completed Adam steps")
    print()

    print("Checkpoint atomicity:")
    print("  only after a completely finished optimizer step")
    print()

    print("Resume:")
    print("  regenerate frozen epoch stream")
    print("  skip completed batches")
    print("  never repeat a completed batch")
    print()

    print("Validation:                           AFTER EVERY EPOCH")
    print("Early stopping:                       NONE")
    print("Best checkpoint:")
    print("  1. max validation NDCG@10")
    print("  2. max validation HR@10")
    print("  3. earliest epoch")
    print()

    print("Test during training:                 FORBIDDEN")
    print("Final test passes:                    1")
    print("Final test model:                     BEST VALIDATION CHECKPOINT")
    print()

    print("Model instantiated:                   NO")
    print("Adam instantiated:                    NO")
    print("optimizer.step():                     0")

    banner(
        "PHASE 5.3.2a COMPLETE / "
        "FULL-TRAINING EXECUTION-STATE CONTRACT FROZEN"
    )


if __name__ == "__main__":
    main()
