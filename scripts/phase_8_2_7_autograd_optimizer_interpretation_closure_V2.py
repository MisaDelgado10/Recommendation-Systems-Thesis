#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json


ROOT = Path(
    "data/experimental/phase_8/"
    "phase_8_2_framework_development/"
    "phase_8_2_7_autograd_optimizer_preflight"
)

ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

PHASE_826_CLOSURE = Path(
    "data/experimental/phase_8/"
    "phase_8_2_framework_development/"
    "phase_8_2_6_training_stream_dry_run/"
    "phase_8_2_6_training_stream_integration_closure_manifest_V2.json"
)

PREFLIGHT_CONTRACT = Path(
    "data/experimental/phase_8/"
    "phase_8_2_framework_development/"
    "phase_8_2_0_preflight/"
    "phase_8_2_0_preflight_contract_V2.json"
)

MODEL_MODULE = Path(
    "scripts/phase_8_2_3_candidate_model_modules_V1.py"
)

INITIALIZER = Path(
    "scripts/phase_8_2_4_canonical_initialization_audit_V1.py"
)

FORWARD_AUDIT = Path(
    "scripts/phase_8_2_5_canonical_forward_branch_insertion_audit_V1.py"
)

FIRST_BATCH = Path(
    "data/experimental/phase_8/"
    "phase_8_2_framework_development/"
    "phase_8_2_6_training_stream_dry_run/"
    "DEV_T57/epoch_0/"
    "epoch_0_first_batch_manifest.parquet"
)

FIRST_BATCH_GROUPS = Path(
    "data/experimental/phase_8/"
    "phase_8_2_framework_development/"
    "phase_8_2_6_training_stream_dry_run/"
    "DEV_T57/epoch_0/"
    "epoch_0_first_batch_segment_groups.csv"
)

CONTRACT = (
    ROOT
    / "phase_8_2_7_autograd_optimizer_interpretation_closure_V2.json"
)

HASHES = (
    ROOT
    / "phase_8_2_7_autograd_optimizer_interpretation_closure_sha256_V2.json"
)


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


print("=" * 120)
print("PHASE 8.2.7 — AUTOGRAD / OPTIMIZER INTERPRETATION CLOSURE V2")
print("=" * 120)


# =============================================================================
# 1. PREREQUISITES
# =============================================================================

required_paths = [
    PHASE_826_CLOSURE,
    PREFLIGHT_CONTRACT,
    MODEL_MODULE,
    INITIALIZER,
    FORWARD_AUDIT,
    FIRST_BATCH,
    FIRST_BATCH_GROUPS,
]

for path in required_paths:
    require(
        path.exists(),
        f"Missing prerequisite: {path}",
    )

phase826 = json.loads(
    PHASE_826_CLOSURE.read_text(
        encoding="utf-8"
    )
)

require(
    phase826["status"]
    == "TRAINING_STREAM_INTEGRATION_DRY_RUN_CLOSED",
    "Phase 8.2.6 is not formally closed.",
)

require(
    phase826["training_allowed_now"] is False,
    "Unexpected training authorization after Phase 8.2.6.",
)

preflight = json.loads(
    PREFLIGHT_CONTRACT.read_text(
        encoding="utf-8"
    )
)

require(
    "work_packages" in preflight,
    "V2 preflight contract missing work_packages.",
)

phase827_rows = [
    row
    for row in preflight["work_packages"]
    if row.get("phase") == "8.2.7"
]

require(
    len(phase827_rows) == 1,
    "Could not uniquely identify Phase 8.2.7.",
)

phase827 = phase827_rows[0]

require(
    phase827["order"] == 7,
    "Unexpected Phase 8.2.7 work-package order.",
)

require(
    phase827["performance_allowed"] is False,
    "Phase 8.2.7 unexpectedly permits performance inspection.",
)

require(
    phase827["training_allowed_after"] is False,
    "Phase 8.2.7 unexpectedly authorizes comparative training.",
)

require(
    "Temporary one-batch forward/backward/update"
    in phase827["produces"],
    "Phase 8.2.7 update semantics changed.",
)


# =============================================================================
# 2. FROZEN INTERPRETATION
# =============================================================================

contract = {
    "schema_version":
        "PHASE_8_2_7_AUTOGRAD_OPTIMIZER_INTERPRETATION_CLOSURE_V2",

    "phase":
        "8.2.7",

    "status":
        "FROZEN_BEFORE_MODEL_EXECUTION",

    "scientific_role":
        "TEMPORARY_ONE_BATCH_AUTOGRAD_OPTIMIZER_IMPLEMENTATION_PREFLIGHT",

    "supersession": {
        "V1_attempt":
            "FAILED_BEFORE_ARTIFACT_WRITE",

        "failure_class":
            "READ_ONLY_JSON_SCHEMA_KEY_MISMATCH",

        "incorrect_key":
            "work_package",

        "correct_key":
            "work_packages",

        "model_instantiated":
            False,

        "optimizer_created":
            False,

        "forward_executed":
            False,

        "scientific_state_changed":
            False,
    },

    "authoritative_predecessor":
        str(PHASE_826_CLOSURE),

    "authoritative_work_package": {
        "name":
            phase827["name"],

        "order":
            phase827["order"],

        "produces":
            phase827["produces"],

        "scientific_gate":
            phase827["scientific_gate"],

        "performance_allowed":
            phase827["performance_allowed"],

        "training_allowed_after":
            phase827["training_allowed_after"],
    },

    "scope": {
        "randomness_scope":
            "DEV_T57",

        "origin":
            "T57",

        "epoch_index":
            0,

        "display_epoch":
            1,

        "batch_size":
            512,

        "batch_source":
            "FROZEN_DEV_T57_EPOCH0_FIRST_BATCH",

        "first_batch_manifest":
            str(FIRST_BATCH),

        "first_batch_segment_groups":
            str(FIRST_BATCH_GROUPS),

        "regenerate_negatives":
            False,

        "regenerate_epoch_order":
            False,
    },

    "candidate_families": [
        "B1a",
        "B1b",
        "M1",
        "M2",
    ],

    "family_execution_policy": {
        "same_frozen_batch_for_all_families":
            True,

        "fresh_canonical_model_instance_per_family":
            True,

        "initialization_scope":
            "DEV_T57",

        "share_parameter_state_between_families":
            False,

        "share_optimizer_state_between_families":
            False,
    },

    "forward_policy": {
        "real_frozen_training_examples":
            True,

        "mixed_target_segments":
            True,

        "trend_history_for_target_T_h":
            "T0..T(h-1)",

        "post_h_GRU_padding":
            False,

        "restore_original_batch_positions_before_loss":
            True,

        "cold_start_execution":
            "FIT_TIME",

        "cold_start_features":
            "ZERO_ALLOWED_STARTUP_FEATURE_SLOTS",

        "cold_start_structural_edges":
            "MASK_ALL_FOUNDER_ACQUISITION_EDGES_INCIDENT_TO_COLD_t",

        "evaluation_candidate_matrix_used":
            False,

        "T60_used":
            False,

        "April_used":
            False,
    },

    "loss": {
        "class":
            "torch.nn.BCEWithLogitsLoss",

        "input":
            "RAW_LOGITS",

        "targets":
            "FROZEN_BINARY_TRAINING_LABELS",

        "loss_may_be_recorded":
            True,

        "ranking_metric_may_be_computed":
            False,

        "loss_may_drive_model_selection":
            False,
    },

    "optimizer": {
        "class":
            "torch.optim.Adam",

        "lr":
            0.001,

        "betas":
            [0.9, 0.999],

        "eps":
            1e-8,

        "weight_decay":
            0.0,

        "amsgrad":
            False,

        "foreach":
            False,

        "fused":
            False,

        "maximize":
            False,

        "capturable":
            False,

        "differentiable":
            False,

        "scheduler":
            None,
    },

    "autograd_policy": {
        "backward_calls_per_family":
            1,

        "all_intended_trainable_branches_must_receive_gradient":
            True,

        "all_existing_gradients_must_be_finite":
            True,

        "active_branch_gradients_must_be_nonzero":
            True,

        "inactive_branch_parameters_must_not_exist":
            True,

        "startup_identity_parameter_must_not_exist":
            True,

        "person_identity_parameter_must_not_exist":
            True,
    },

    "optimizer_step_policy": {
        "optimizer_step_calls_per_family":
            1,

        "temporary_update_only":
            True,

        "post_step_parameter_change_required":
            True,

        "Adam_state_expected_after_step":
            True,

        "post_step_scoring_allowed":
            False,

        "post_step_state_reuse_allowed":
            False,

        "post_step_checkpoint_allowed":
            False,

        "updated_state_retained":
            False,

        "model_and_optimizer_discarded_after_family_audit":
            True,
    },

    "state_audit": {
        "record_initial_parameter_hashes":
            True,

        "forward_must_not_change_parameters":
            True,

        "backward_must_not_change_parameters":
            True,

        "one_step_must_change_at_least_one_parameter":
            True,

        "post_step_state_must_differ_from_initial":
            True,

        "trained_checkpoint_retained":
            False,
    },

    "forbidden_actions": [
        "full_epoch_training",
        "development_NDCG",
        "development_HR",
        "epoch_selection",
        "model_selection",
        "architecture_change",
        "feature_change",
        "T60_scoring",
        "April_analysis",
        "checkpoint_persistence",
        "reuse_of_post_step_parameters",
    ],

    "phase5_reference_policy": {
        "reuse": [
            "Adam_hyperparameters",
            "BCEWithLogitsLoss_pattern",
            "mixed_h_execution_pattern",
            "gradient_audit_pattern",
            "pre_step_state_neutrality_pattern",
        ],

        "do_not_reuse": [
            "Phase5_optimizer_step_prohibition",
            "Phase5_model_topology",
            "Phase5_Startup_identity_embedding",
            "Phase5_graph_artifacts",
        ],

        "reason":
            (
                "Phase 8.2.7 explicitly requires a temporary "
                "one-batch update; the historical Phase-5 "
                "preflight stopped before optimizer.step()."
            ),
    },

    "execution_state_at_freeze": {
        "model_instantiated":
            False,

        "optimizer_created":
            False,

        "forward_executed":
            False,

        "BCE_computed":
            False,

        "backward_executed":
            False,

        "optimizer_step_executed":
            False,

        "full_training_epoch_executed":
            False,

        "performance_inspected":
            False,

        "T60_scored":
            False,

        "April_analyzed":
            False,
    },

    "training_allowed_now":
        False,

    "next_step":
        (
            "Execute the frozen DEV_T57 epoch-0 first-batch "
            "forward/BCE/backward/exactly-one-Adam-step audit "
            "for B1a/B1b/M1/M2 and discard every updated state."
        ),
}


CONTRACT.write_text(
    json.dumps(
        contract,
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)


hash_targets = [
    PHASE_826_CLOSURE,
    PREFLIGHT_CONTRACT,
    MODEL_MODULE,
    INITIALIZER,
    FORWARD_AUDIT,
    FIRST_BATCH,
    FIRST_BATCH_GROUPS,
    CONTRACT,
]

HASHES.write_text(
    json.dumps(
        {
            str(path): sha256_file(path)
            for path in hash_targets
        },
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)


print()
print("Phase 8.2.6 closed:             PASS")
print("8.2.7 work-package located:     PASS")
print("Correct schema key:             work_packages")
print("Performance allowed:            NO")
print("Training allowed after 8.2.7:  NO")
print()
print("Scope:                          DEV_T57")
print("Epoch index:                    0")
print("Frozen batch:                   512 examples")
print("Families:                       B1a / B1b / M1 / M2")
print()
print("Forward allowed next:           YES")
print("BCE allowed next:               YES")
print("Backward allowed next:          YES")
print("Adam allowed next:              YES")
print("optimizer.step next:            EXACTLY 1 / family")
print("Post-step checkpoint:           NO")
print("Post-step scoring:              NO")
print("Updated state retention:        NO")
print()
print(f"Contract: {CONTRACT}")
print(f"Hashes:   {HASHES}")
print()
print("=" * 120)
print("STATUS: PHASE_8_2_7_INTERPRETATION_FROZEN")
print("NEXT:   Real frozen-first-batch autograd / Adam one-step preflight")
print("=" * 120)
