from pathlib import Path
import hashlib
import json


# =============================================================================
# PHASE 8.1.12A — OUTCOME-INDEPENDENT COLD-START MASKING AMENDMENT V1
# =============================================================================
#
# Scientific role:
#
#   PRE-TRAINING LEAKAGE / INDUCTIVE-INTEGRITY CORRECTION
#
# PURPOSE
# -------
#
# Correct one clause inherited from Phase 8.1.10A:
#
# OLD:
#   hide only Startup IDs that later appear as positive targets in the
#   frozen PRIMARY cold-start evaluation stratum.
#
# PROBLEM:
#   future evaluation outcome membership determines the fit-time graph mask.
#
# NEW:
#   for each prediction origin O, derive the masked Startup population using
#   ONLY pre-origin information:
#
#       frozen Startup-role node
#       AND origin-eligible under the already frozen existence policy
#       AND zero prior investment interactions strictly before O
#
# This population is independent of:
#
#   - future positive labels;
#   - future negative labels;
#   - evaluation candidate matrices;
#   - eventual evaluation participation.
#
# No feature, model family, neural layer, optimizer, metric, or candidate
# universe is changed.
#
# =============================================================================


BASE = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation"
)

CONSISTENCY_CONTRACT = (
    BASE
    / "phase_8_1_10a_inductive_consistency_closure"
    / "phase_8_1_10a_consistency_amendment_V1.json"
)

BRANCH_CONTRACT = (
    BASE
    / "phase_8_1_11a_model_family_branch_semantics"
    / "phase_8_1_11a_branch_semantics_amendment_V1.json"
)

EXECUTION_CONTRACT = (
    BASE
    / "phase_8_1_12_execution_contract"
    / "phase_8_1_12_execution_contract_V3.json"
)

PREFLIGHT_CONTRACT = Path(
    "data/experimental/phase_8/"
    "phase_8_2_framework_development/"
    "phase_8_2_0_preflight/"
    "phase_8_2_0_preflight_contract_V1.json"
)

FOUNDER_HISTORY_MANIFEST = Path(
    "data/experimental/phase_8/"
    "phase_8_2_framework_development/"
    "phase_8_2_1_feature_materialization/"
    "phase_8_2_1b_manifest_V1.json"
)


OUTPUT_DIR = (
    BASE
    / "phase_8_1_12a_outcome_independent_cold_start_masking"
)

CONTRACT_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_12a_outcome_independent_masking_contract_V1.json"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_12a_manifest_V1.json"
)

HASH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_12a_artifact_sha256_V1.json"
)


def separator(char="=", width=118):
    print(char * width)


def require(condition, message):

    if not bool(condition):
        raise ValueError(message)


def read_json(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def write_json(path, obj):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
            default=str,
        )


def sha256_file(path, chunk_size=1024 * 1024):

    digest = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def main():

    separator()

    print(
        "PHASE 8.1.12A — "
        "OUTCOME-INDEPENDENT COLD-START MASKING "
        "AMENDMENT V1"
    )

    separator()

    print()

    print(
        "Scientific role:             "
        "PRE-TRAINING LEAKAGE / INDUCTIVE-INTEGRITY CORRECTION"
    )

    print("Feature discovery reopened:  NO")
    print("Architecture changed:        NO")
    print("Model family changed:        NO")
    print("Negative protocol changed:   NO")
    print("Model trained:               NO")
    print("T60 scored:                  NO")
    print("April analyzed:              NO")

    # =========================================================================
    # 1. Prerequisite gates
    # =========================================================================

    separator("-")

    print(
        "8.1.12A.1 — PREREQUISITE GATES"
    )

    separator("-")

    required = [
        CONSISTENCY_CONTRACT,
        BRANCH_CONTRACT,
        EXECUTION_CONTRACT,
        PREFLIGHT_CONTRACT,
        FOUNDER_HISTORY_MANIFEST,
    ]

    for path in required:

        require(
            path.exists(),
            f"Missing prerequisite: {path}",
        )

        print(
            f"FOUND {path}"
        )

    consistency = read_json(
        CONSISTENCY_CONTRACT
    )

    branches = read_json(
        BRANCH_CONTRACT
    )

    execution = read_json(
        EXECUTION_CONTRACT
    )

    preflight = read_json(
        PREFLIGHT_CONTRACT
    )

    founder_history = read_json(
        FOUNDER_HISTORY_MANIFEST
    )

    for name, payload in [
        ("8.1.10A", consistency),
        ("8.1.11A", branches),
        ("8.1.12 V3", execution),
        ("8.2.0", preflight),
    ]:

        require(
            payload.get("status")
            ==
            "FROZEN",
            f"{name} is not frozen.",
        )

    require(
        founder_history.get("status")
        ==
        "MATERIALIZED_AND_AUDITED",
        "8.2.1B is not authoritative.",
    )

    require(
        execution.get(
            "training_performed"
        )
        is False,
        "Training already occurred.",
    )

    require(
        execution.get(
            "T60_scored"
        )
        is False,
        "T60 already scored.",
    )

    require(
        execution.get(
            "April_analyzed"
        )
        is False,
        "April already analyzed.",
    )

    # Confirm that the historical clause being amended actually exists
    # somewhere in the frozen 8.1.10A contract.
    old_contract_text = (
        json.dumps(
            consistency,
            sort_keys=True,
        )
        .casefold()
    )

    require(
        (
            "positive target"
            in
            old_contract_text
        )
        or
        (
            "positive_target"
            in
            old_contract_text
        ),
        (
            "Could not locate the historical positive-target "
            "masking semantics in Phase 8.1.10A."
        ),
    )

    print()

    print(
        "Historical outcome-conditioned clause detected: YES"
    )

    print(
        "Training occurred before correction:            NO"
    )

    # =========================================================================
    # 2. Scientific issue
    # =========================================================================

    separator("-")

    print(
        "8.1.12A.2 — SCIENTIFIC ISSUE"
    )

    separator("-")

    issue = {
        "old_mask_definition":
            (
                "Startup IDs appearing as positive targets "
                "in the frozen PRIMARY cold-start stratum."
            ),

        "problem":
            (
                "Future evaluation outcome membership determines "
                "which Startup covariates and graph edges are hidden "
                "during parameter fitting."
            ),

        "classification":
            (
                "OUTCOME_CONDITIONED_FIT_TIME_MASK"
            ),

        "performance_tuning_issue":
            False,

        "feature_discovery_issue":
            False,

        "requires_pretraining_correction":
            True,
    }

    for key, value in issue.items():

        print(
            f"{key:<36} {value}"
        )

    # =========================================================================
    # 3. Corrected cold-start universe
    # =========================================================================

    separator("-")

    print(
        "8.1.12A.3 — OUTCOME-INDEPENDENT COLD-START UNIVERSE"
    )

    separator("-")

    cold_definition = {
        "population":
            "FROZEN_311589_STARTUP_ROLE_NODES",

        "evaluated_separately_per_origin":
            True,

        "required_condition_1":
            (
                "Startup is origin-eligible under the already "
                "frozen Phase-8 existence policy."
            ),

        "required_condition_2":
            (
                "Startup has ZERO canonical investment interactions "
                "strictly before prediction origin O."
            ),

        "T0_counts_as_prior_history":
            True,

        "future_positive_label_used":
            False,

        "future_negative_label_used":
            False,

        "evaluation_case_membership_used":
            False,

        "evaluation_candidate_matrix_used":
            False,

        "eventual_evaluation_participation_used":
            False,

        "startup_UUID_predictive":
            False,

        "startup_UUID_role":
            "bookkeeping/masking only",
    }

    for key, value in cold_definition.items():

        print(
            f"{key:<44} {value}"
        )

    # =========================================================================
    # 4. Fit-time semantics
    # =========================================================================

    separator("-")

    print(
        "8.1.12A.4 — PARAMETER-FIT MASKING"
    )

    separator("-")

    fit_policy = {
        "masked_population":
            "ALL_ORIGIN_ELIGIBLE_INVESTMENT_COLD_STARTUPS",

        "startup_allowed_feature_rows":
            "ZERO_DURING_PARAMETER_FIT",

        "startup_founder_edges":
            "MASK_ALL_INCIDENT_EDGES_DURING_PARAMETER_FIT",

        "startup_acquisition_edges":
            "MASK_ALL_INCIDENT_EDGES_DURING_PARAMETER_FIT",

        "incoming_messages":
            False,

        "outgoing_messages":
            False,

        "node_index_deleted":
            False,

        "node_row_deleted":
            False,

        "person_nodes_deleted_if_orphaned":
            False,

        "parameters_trainable":
            True,

        "future_labels_needed":
            False,
    }

    for key, value in fit_policy.items():

        print(
            f"{key:<44} {value}"
        )

    # =========================================================================
    # 5. Training-negative interaction with cold mask
    # =========================================================================

    separator("-")

    print(
        "8.1.12A.5 — TRAINING-NEGATIVE SEMANTICS"
    )

    separator("-")

    negative_policy = {
        "phase_8_1_12_negative_sampler_changed":
            False,

        "same_training_stream_across_B1a_B1b_M1_M2":
            True,

        "origin_cold_node_can_be_sampled_as_historical_negative":
            True,

        "if_sampled_startup_identity_parameter":
            False,

        "if_sampled_startup_inductive_features":
            "ZERO",

        "if_sampled_startup_structural_representation":
            "ZERO",

        "interpretation":
            (
                "The frozen negative-sampling distribution is preserved. "
                "An origin-cold Startup sampled as a training negative "
                "cannot expose Startup-specific covariates, structural "
                "context, or trainable identity because all are masked."
            ),

        "reason_not_to_change_sampler":
            (
                "Changing negative eligibility now would alter the frozen "
                "training objective and confound attribution relative to "
                "the preregistered execution contract."
            ),
    }

    for key, value in negative_policy.items():

        print(
            f"{key:<52} {value}"
        )

    # =========================================================================
    # 6. Inference semantics
    # =========================================================================

    separator("-")

    print(
        "8.1.12A.6 — INDUCTIVE INFERENCE INSERTION"
    )

    separator("-")

    inference_policy = {
        "population":
            "ALL_ORIGIN_ELIGIBLE_INVESTMENT_COLD_STARTUPS",

        "restore_allowed_feature_rows":
            True,

        "restore_origin_eligible_incoming_edges":
            True,

        "allow_outgoing_messages_from_cold_startups":
            False,

        "cold_startups_are_message_passing_sinks":
            True,

        "model_parameters_trainable":
            False,

        "optimizer_step_allowed":
            False,

        "preprocessing_refit_allowed":
            False,

        "evaluation_matrix_defines_restoration":
            False,

        "positive_target_identity_defines_restoration":
            False,

        "negative_identity_defines_restoration":
            False,
    }

    for key, value in inference_policy.items():

        print(
            f"{key:<48} {value}"
        )

    # =========================================================================
    # 7. Supersession scope
    # =========================================================================

    separator("-")

    print(
        "8.1.12A.7 — SUPERSESSION SCOPE"
    )

    separator("-")

    supersession = {
        "phase_8_1_10a_file_modified":
            False,

        "phase_8_1_11a_file_modified":
            False,

        "phase_8_1_12_v3_file_modified":
            False,

        "superseded_clause":
            (
                "Any fit/inference heldout-set definition based on "
                "future PRIMARY positive-target membership."
            ),

        "replacement_clause":
            (
                "Use all origin-eligible investment-cold Startups "
                "defined solely from pre-origin information."
            ),

        "retained_from_8_1_10a":
            [
                "founder-history pre-focal cutoff",
                "Person ZERO_40 initialization",
                "no Person identity embedding",
                "fit/inference parameter separation",
            ],

        "retained_from_8_1_11a":
            [
                "B1a/B1b no-graph semantics",
                "M1/M2 heterogeneous graph semantics",
                "sink-only cold-start inference messages",
                "280-D pair representation",
            ],

        "retained_from_8_1_12_v3":
            [
                "optimizer",
                "randomness",
                "negative sampling",
                "20-epoch budget",
                "development selection",
                "T60 no-selection",
                "April confirmatory protocol",
            ],
    }

    for key, value in supersession.items():

        print(
            f"{key:<40} {value}"
        )

    # =========================================================================
    # 8. Method-design status
    # =========================================================================

    separator("-")

    print(
        "8.1.12A.8 — METHOD-DESIGN STATUS"
    )

    separator("-")

    status = {
        "feature_discovery_closed":
            True,

        "graph_schema_closed":
            True,

        "architecture_closed":
            True,

        "execution_policy_closed":
            True,

        "method_design_closed":
            True,

        "correction_class":
            "LEAKAGE_INTEGRITY_AMENDMENT",

        "performance_based_change":
            False,

        "new_information_channel_added":
            False,

        "new_model_family_added":
            False,

        "training_performed":
            False,

        "performance_inspected":
            False,

        "T60_scored":
            False,

        "April_analyzed":
            False,
    }

    for key, value in status.items():

        print(
            f"{key:<38} {value}"
        )

    # =========================================================================
    # 9. Persist amendment
    # =========================================================================

    separator("-")

    print(
        "8.1.12A.9 — PERSIST LEAKAGE-INTEGRITY AMENDMENT"
    )

    separator("-")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    contract = {
        "schema_version":
            "PHASE_8_1_12A_OUTCOME_INDEPENDENT_MASKING_V1",

        "phase":
            "8.1.12A",

        "status":
            "FROZEN",

        "scientific_role":
            "PRETRAINING_LEAKAGE_INTEGRITY_CORRECTION",

        "issue":
            issue,

        "authoritative_cold_start_definition":
            cold_definition,

        "parameter_fit_policy":
            fit_policy,

        "training_negative_policy":
            negative_policy,

        "inference_policy":
            inference_policy,

        "supersession":
            supersession,

        "method_design_status":
            status,

        "authoritative_interpretation":
            (
                "For every prediction origin, all origin-eligible "
                "Startups with zero prior investment interactions are "
                "masked during parameter fitting independently of future "
                "evaluation membership. At inference, allowed covariates "
                "and incoming structural context are restored with frozen "
                "parameters; cold Startups remain message-passing sinks."
            ),

        "training_allowed_now":
            False,

        "next_step":
            (
                "Phase 8.2.1C — materialize outcome-independent "
                "origin cold/warm masks and training-only "
                "preprocessing populations before fitting any "
                "feature transformer."
            ),
    }

    write_json(
        CONTRACT_OUTPUT,
        contract,
    )

    manifest = {
        "phase":
            "8.1.12A",

        "status":
            "FROZEN",

        "correction_class":
            "LEAKAGE_INTEGRITY_AMENDMENT",

        "upstream_files_preserved":
            True,

        "feature_discovery_reopened":
            False,

        "architecture_changed":
            False,

        "negative_sampler_changed":
            False,

        "training_performed":
            False,

        "performance_inspected":
            False,

        "T60_scored":
            False,

        "April_analyzed":
            False,

        "contract":
            str(
                CONTRACT_OUTPUT
            ),
    }

    write_json(
        MANIFEST_OUTPUT,
        manifest,
    )

    write_json(
        HASH_OUTPUT,
        {
            str(CONTRACT_OUTPUT):
                sha256_file(
                    CONTRACT_OUTPUT
                ),

            str(MANIFEST_OUTPUT):
                sha256_file(
                    MANIFEST_OUTPUT
                ),
        },
    )

    separator()

    print(
        "PHASE 8.1.12A OUTCOME-INDEPENDENT "
        "COLD-START MASKING FROZEN"
    )

    separator()

    print()

    print(
        "OLD MASK:"
    )

    print(
        "  future PRIMARY positive targets"
    )

    print()

    print(
        "AUTHORITATIVE MASK:"
    )

    print(
        "  ALL origin-eligible Startups"
    )

    print(
        "  with ZERO prior investment interactions"
    )

    print(
        "  defined using PRE-ORIGIN information only"
    )

    print()

    print(
        "EVALUATION POSITIVE IDS USED:      NO"
    )

    print(
        "EVALUATION NEGATIVE IDS USED:      NO"
    )

    print(
        "CANDIDATE MATRIX USED:             NO"
    )

    print(
        "FEATURE / ARCHITECTURE CHANGE:     NO"
    )

    print(
        "NEGATIVE SAMPLER CHANGE:           NO"
    )

    print(
        "METHOD DESIGN REMAINS CLOSED:      YES"
    )

    print(
        "TRAINING PERFORMED:                NO"
    )

    print(
        "T60 SCORED:                        NO"
    )

    print(
        "APRIL ANALYZED:                    NO"
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Phase 8.2.1C — origin cold/warm masks + "
        "training-only preprocessing populations."
    )


if __name__ == "__main__":
    main()
