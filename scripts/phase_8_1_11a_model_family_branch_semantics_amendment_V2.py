from pathlib import Path
from copy import deepcopy
import hashlib
import json


# =============================================================================
# PHASE 8.1.11A — MODEL-FAMILY BRANCH SEMANTICS AMENDMENT V2
# =============================================================================
#
# Scientific role:
#
#   PRE-EXECUTION ARCHITECTURAL SEMANTIC REPAIR
#
# PURPOSE
# -------
#
# Supersede ONLY the outcome-conditioned "held-out primary cold-start Startup"
# wording inherited by Phase 8.1.11A V1.
#
# The model-family architecture itself is NOT changed.
#
# Authoritative population:
#
#   COLD_t =
#       all frozen Startup-role nodes
#       that are origin-eligible at t
#       AND have zero canonical investment interactions strictly before t.
#
# COLD_t is independent of:
#
#   - evaluation positive IDs;
#   - evaluation negative IDs;
#   - candidate matrices;
#   - future evaluation participation.
#
# During parameter fitting:
#
#   - Startup side features for every COLD_t node are zero;
#   - all founder/acquisition structural edges incident to every COLD_t node
#     are masked;
#   - node/index rows remain present.
#
# During inductive inference:
#
#   - model/preprocessing parameters stay frozen;
#   - allowed origin-time Startup features are restored;
#   - origin-eligible incoming structural messages are restored;
#   - COLD_t Startups remain message-passing sinks;
#   - no outgoing message from a COLD_t Startup may alter fitted graph nodes.
#
# =============================================================================


BASE = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation"
)

V1_PATH = (
    BASE
    / "phase_8_1_11a_model_family_branch_semantics"
    / "phase_8_1_11a_branch_semantics_amendment_V1.json"
)

MASKING_CONTRACT = (
    BASE
    / "phase_8_1_12a_outcome_independent_cold_start_masking"
    / "phase_8_1_12a_outcome_independent_masking_contract_V1.json"
)

FOUNDER_HISTORY_CONTRACT = (
    BASE
    / "phase_8_1_12b_origin_referenced_founder_history"
    / "phase_8_1_12b_origin_referenced_founder_history_contract_V1.json"
)

OUTPUT_DIR = (
    BASE
    / "phase_8_1_11a_model_family_branch_semantics"
)

V2_PATH = (
    OUTPUT_DIR
    / "phase_8_1_11a_branch_semantics_amendment_V2.json"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "phase_8_1_11a_branch_semantics_manifest_V2.json"
)

HASH_PATH = (
    OUTPUT_DIR
    / "phase_8_1_11a_branch_semantics_sha256_V2.json"
)


EXPECTED_MODELS = [
    "B0",
    "B1a",
    "B1b",
    "M1",
    "M2",
]


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
        "PHASE 8.1.11A — "
        "MODEL-FAMILY BRANCH SEMANTICS AMENDMENT V2"
    )

    separator()

    print()

    print(
        "Scientific role:             "
        "PRE-EXECUTION ARCHITECTURAL SEMANTIC REPAIR"
    )

    print("Feature discovery reopened:  NO")
    print("Architecture topology changed:NO")
    print("Model family changed:        NO")
    print("Pair dimension changed:      NO")
    print("Training performed:          NO")
    print("Performance inspected:       NO")
    print("T60 scored:                  NO")
    print("April analyzed:              NO")

    # =========================================================================
    # 1. Prerequisite gates
    # =========================================================================

    separator("-")

    print(
        "8.1.11A-V2.1 — PREREQUISITE GATES"
    )

    separator("-")

    required = [
        V1_PATH,
        MASKING_CONTRACT,
        FOUNDER_HISTORY_CONTRACT,
    ]

    for path in required:

        require(
            path.exists(),
            f"Missing prerequisite: {path}",
        )

        print(
            f"FOUND {path}"
        )

    v1 = read_json(
        V1_PATH
    )

    masking = read_json(
        MASKING_CONTRACT
    )

    founder_history = read_json(
        FOUNDER_HISTORY_CONTRACT
    )

    require(
        v1.get("status")
        ==
        "FROZEN",
        "8.1.11A V1 is not frozen.",
    )

    require(
        masking.get("status")
        ==
        "FROZEN",
        "8.1.12A is not frozen.",
    )

    require(
        founder_history.get("status")
        ==
        "FROZEN",
        "8.1.12B is not frozen.",
    )

    require(
        v1.get("pair_dimension")
        ==
        280,
        "V1 pair dimension is not 280.",
    )

    models = [
        row.get("model")
        for row
        in v1.get(
            "model_family_branch_activation",
            []
        )
    ]

    require(
        models
        ==
        EXPECTED_MODELS,
        (
            "Unexpected model-family ordering or membership: "
            f"{models}"
        ),
    )

    old_policy = v1.get(
        "heldout_message_policy"
    )

    require(
        isinstance(
            old_policy,
            dict,
        ),
        "V1 heldout_message_policy is missing.",
    )

    old_policy_text = json.dumps(
        old_policy,
        sort_keys=True,
    )

    require(
        "Held-out primary cold-start"
        in
        old_policy_text,
        (
            "Expected historical held-out primary "
            "cold-start wording was not found."
        ),
    )

    require(
        old_policy.get(
            "evaluation_case_specific_graph"
        )
        is False,
        (
            "V1 unexpectedly permits an "
            "evaluation-case-specific graph."
        ),
    )

    require(
        old_policy.get(
            "negative_sample_specific_graph"
        )
        is False,
        (
            "V1 unexpectedly permits a "
            "negative-specific graph."
        ),
    )

    require(
        old_policy.get(
            "parameters_updated_at_inference"
        )
        is False,
        (
            "V1 unexpectedly permits parameter "
            "updates at inference."
        ),
    )

    print()

    print(
        "Historical outcome-conditioned wording detected: YES"
    )

    print(
        "Architectural topology drift detected:           NO"
    )

    # =========================================================================
    # 2. Freeze authoritative COLD_t semantics
    # =========================================================================

    separator("-")

    print(
        "8.1.11A-V2.2 — AUTHORITATIVE COLD_t POPULATION"
    )

    separator("-")

    cold_population = {
        "population_name":
            "COLD_t",

        "candidate_universe":
            "FROZEN_311589_STARTUP_ROLE_NODES",

        "definition":
            (
                "Startup is origin-eligible at prediction origin t "
                "AND has zero canonical investment interactions "
                "strictly before t."
            ),

        "origin_eligibility_reference":
            (
                "Authoritative Phase 8.2.1C V2 origin-eligibility "
                "materialization."
            ),

        "future_positive_ids_used":
            False,

        "future_negative_ids_used":
            False,

        "evaluation_case_membership_used":
            False,

        "evaluation_candidate_matrix_used":
            False,

        "future_outcome_used":
            False,

        "startup_UUID_predictive":
            False,
    }

    for key, value in cold_population.items():

        print(
            f"{key:<42} {value}"
        )

    # =========================================================================
    # 3. Corrected fitting / inference message policy
    # =========================================================================

    separator("-")

    print(
        "8.1.11A-V2.3 — CORRECTED MESSAGE-PASSING POLICY"
    )

    separator("-")

    cold_policy = {
        "population":
            "ALL_COLD_t_STARTUPS",

        "training_startup_features":
            (
                "ZERO all allowed Startup-side feature values "
                "for every COLD_t node during parameter fitting."
            ),

        "training_structural_edges":
            (
                "Mask all founder/acquisition structural edges "
                "incident to every COLD_t Startup during "
                "parameter fitting."
            ),

        "training_node_rows_retained":
            True,

        "training_node_indices_retained":
            True,

        "evaluation_case_specific_graph":
            False,

        "negative_sample_specific_graph":
            False,

        "future_label_specific_mask":
            False,

        "inference_features":
            (
                "After model/preprocessor fitting is frozen, "
                "restore allowed origin-time Startup features "
                "for every COLD_t node."
            ),

        "inference_incoming_messages":
            True,

        "inference_outgoing_messages":
            False,

        "inference_rule":
            (
                "Every COLD_t Startup is an inductive message-passing "
                "sink: origin-eligible structural messages may flow "
                "INTO the cold Startup, but messages sourced FROM the "
                "cold Startup may not alter Person, Investor, or other "
                "Startup representations fitted without that node's "
                "side information."
            ),

        "parameters_updated_at_inference":
            False,

        "preprocessing_refit_at_inference":
            False,

        "cold_startup_can_modify_fitted_graph_nodes":
            False,

        "reason":
            (
                "Provides an outcome-independent inductive cold-start "
                "simulation while allowing the unseen Startup to "
                "aggregate from the frozen fitted heterogeneous graph."
            ),
    }

    for key, value in cold_policy.items():

        print(
            f"{key:<48} {value}"
        )

    # =========================================================================
    # 4. Preserve branch activation EXACTLY
    # =========================================================================

    separator("-")

    print(
        "8.1.11A-V2.4 — MODEL-FAMILY ACTIVATION PRESERVATION"
    )

    separator("-")

    for row in v1[
        "model_family_branch_activation"
    ]:

        print(
            f"{row['model']:<5} "
            f"graph={str(row['graph_active']):<5} "
            f"trend={row['investor_trend']:<15} "
            f"startup_content={row['startup_content']:<30} "
            f"startup_structured={row['startup_structured']:<35} "
            f"startup_structural={row['startup_structural']}"
        )

    print()

    print(
        "Model-family activation changed: NO"
    )

    print(
        "Comparison semantics changed:   NO"
    )

    print(
        "Pair dimension remains:         280"
    )

    # =========================================================================
    # 5. Construct V2 contract
    # =========================================================================

    separator("-")

    print(
        "8.1.11A-V2.5 — CONSTRUCT SUPERSEDING CONTRACT"
    )

    separator("-")

    v2 = deepcopy(
        v1
    )

    # Remove the semantically stale field entirely.
    removed_policy = v2.pop(
        "heldout_message_policy"
    )

    v2[
        "schema_version"
    ] = (
        "PHASE_8_1_11A_MODEL_FAMILY_BRANCH_SEMANTICS_V2"
    )

    v2[
        "status"
    ] = "FROZEN"

    v2[
        "scientific_role"
    ] = (
        "PRE_EXECUTION_ARCHITECTURAL_CLARIFICATION_CORRECTED"
    )

    v2[
        "cold_start_population"
    ] = cold_population

    v2[
        "cold_start_message_policy"
    ] = cold_policy

    v2[
        "supersession"
    ] = {
        "supersedes":
            str(
                V1_PATH
            ),

        "V1_file_modified":
            False,

        "V1_artifact_preserved":
            True,

        "superseded_scope":
            (
                "Only the heldout_message_policy clauses whose "
                "masked population was defined by future PRIMARY "
                "cold-start evaluation membership."
            ),

        "replacement_scope":
            (
                "All fit/inference masking now applies to the "
                "outcome-independent COLD_t population."
            ),

        "architecture_topology_changed":
            False,

        "branch_activation_changed":
            False,

        "comparison_semantics_changed":
            False,

        "pair_dimension_changed":
            False,

        "feature_discovery_reopened":
            False,

        "performance_based_change":
            False,
    }

    v2[
        "semantic_dependencies"
    ] = {
        "outcome_independent_cold_mask":
            str(
                MASKING_CONTRACT
            ),

        "origin_referenced_founder_history":
            str(
                FOUNDER_HISTORY_CONTRACT
            ),
    }

    v2[
        "method_design_reopened"
    ] = False

    v2[
        "training_performed"
    ] = False

    v2[
        "performance_inspected"
    ] = False

    v2[
        "T60_scored"
    ] = False

    v2[
        "April_analyzed"
    ] = False

    v2[
        "next_step"
    ] = (
        "Phase 8.1.12 V4 — supersede the V3 execution contract "
        "with corrected COLD_t preprocessing semantics, "
        "Phase-8-specific randomness namespaces, explicit "
        "parameter initialization policy and exact two-way "
        "clustered bootstrap specification."
    )

    # =========================================================================
    # 6. Integrity comparisons against V1
    # =========================================================================

    separator("-")

    print(
        "8.1.11A-V2.6 — NON-TARGETED-CONTRACT INTEGRITY"
    )

    separator("-")

    require(
        v2[
            "model_family_branch_activation"
        ]
        ==
        v1[
            "model_family_branch_activation"
        ],
        "Model-family activation changed unexpectedly.",
    )

    require(
        v2[
            "comparison_semantics"
        ]
        ==
        v1[
            "comparison_semantics"
        ],
        "Comparison semantics changed unexpectedly.",
    )

    require(
        v2[
            "pair_dimension"
        ]
        ==
        v1[
            "pair_dimension"
        ]
        ==
        280,
        "Pair dimension changed unexpectedly.",
    )

    require(
        v2[
            "clarification"
        ]
        ==
        v1[
            "clarification"
        ],
        "Branch clarification changed unexpectedly.",
    )

    require(
        "heldout_message_policy"
        not in
        v2,
        "Stale heldout_message_policy survived into V2.",
    )

    serialized_v2 = json.dumps(
        v2,
        sort_keys=True,
    )

    require(
        "Held-out primary cold-start"
        not in
        serialized_v2,
        (
            "Outcome-conditioned held-out wording "
            "survived into V2."
        ),
    )

    require(
        "ALL_COLD_t_STARTUPS"
        in
        serialized_v2,
        "Corrected COLD_t policy missing from V2.",
    )

    print(
        "Branch activation unchanged:       PASS"
    )

    print(
        "Comparison semantics unchanged:    PASS"
    )

    print(
        "Pair dimension unchanged:          PASS"
    )

    print(
        "Old held-out wording removed:      PASS"
    )

    print(
        "Outcome-independent COLD_t added:  PASS"
    )

    # =========================================================================
    # 7. Persist V2 + manifest + hashes
    # =========================================================================

    separator("-")

    print(
        "8.1.11A-V2.7 — PERSIST SUPERSEDING CONTRACT"
    )

    separator("-")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_json(
        V2_PATH,
        v2,
    )

    manifest = {
        "phase":
            "8.1.11A",

        "version":
            "V2",

        "status":
            "FROZEN",

        "scientific_role":
            "ARCHITECTURAL_SEMANTIC_REPAIR",

        "V1_status":
            "FROZEN_SUPERSEDED",

        "V1_file_preserved":
            True,

        "authoritative_contract":
            str(
                V2_PATH
            ),

        "cold_population":
            "ALL_COLD_t_STARTUPS",

        "future_evaluation_membership_used":
            False,

        "architecture_topology_changed":
            False,

        "model_family_changed":
            False,

        "pair_dimension_changed":
            False,

        "feature_discovery_reopened":
            False,

        "performance_based_change":
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

    write_json(
        MANIFEST_PATH,
        manifest,
    )

    write_json(
        HASH_PATH,
        {
            str(V2_PATH):
                sha256_file(
                    V2_PATH
                ),

            str(MANIFEST_PATH):
                sha256_file(
                    MANIFEST_PATH
                ),
        },
    )

    separator()

    print(
        "PHASE 8.1.11A V2 BRANCH SEMANTICS FROZEN"
    )

    separator()

    print()

    print(
        "V1 HELD-OUT POLICY:                SUPERSEDED"
    )

    print(
        "AUTHORITATIVE MASKED POPULATION:  ALL COLD_t STARTUPS"
    )

    print(
        "FUTURE POSITIVE IDS USED:         NO"
    )

    print(
        "EVALUATION CANDIDATE IDS USED:    NO"
    )

    print(
        "COLD FEATURES DURING FIT:         ZERO"
    )

    print(
        "COLD INCIDENT EDGES DURING FIT:   MASKED"
    )

    print(
        "INFERENCE FEATURES RESTORED:      YES"
    )

    print(
        "INFERENCE INCOMING MESSAGES:      YES"
    )

    print(
        "INFERENCE OUTGOING MESSAGES:      NO"
    )

    print(
        "ARCHITECTURE CHANGED:             NO"
    )

    print(
        "BRANCH ACTIVATION CHANGED:        NO"
    )

    print(
        "PAIR DIMENSION:                   280"
    )

    print(
        "TRAINING PERFORMED:               NO"
    )

    print(
        "T60 SCORED:                       NO"
    )

    print(
        "APRIL ANALYZED:                   NO"
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Phase 8.1.12 V4 — corrected final execution contract."
    )


if __name__ == "__main__":
    main()
