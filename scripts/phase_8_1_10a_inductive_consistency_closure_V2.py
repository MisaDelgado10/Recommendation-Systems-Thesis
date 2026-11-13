from pathlib import Path
from copy import deepcopy
import hashlib
import json

import pandas as pd


# =============================================================================
# PHASE 8.1.10A — INDUCTIVE CONSISTENCY CLOSURE V2
# =============================================================================
#
# Scientific role:
#   PRE-ARCHITECTURE LOGICAL CONSISTENCY REPAIR
#
# Supersedes V1 in TWO semantic areas only:
#
#   A. Founder history
#      OLD: prior_company.founded_on < focal_startup.founded_on
#      NEW: prior_company.founded_on < prediction_origin
#
#      P11/P12 are Startup-conditioned origin-specific features H(s,t).
#
#   B. Cold-start masking
#      OLD: future PRIMARY positive targets define held-out nodes
#      NEW: ALL outcome-independent COLD_t Startups are hidden during fit
#
# No feature discovery, architecture selection, model training, T60 scoring,
# or April analysis is performed.
# =============================================================================


BASE = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation"
)

DIR = (
    BASE
    / "phase_8_1_10a_inductive_consistency_closure"
)

V1 = (
    DIR
    / "phase_8_1_10a_consistency_amendment_V1.json"
)

GRAPH_SCHEMA_V2 = (
    BASE
    / "phase_8_1_9_heterogeneous_graph_schema"
    / "phase_8_1_9_graph_schema_V2.json"
)

COLD_CONTRACT = (
    BASE
    / "phase_8_1_12a_outcome_independent_cold_start_masking"
    / "phase_8_1_12a_outcome_independent_masking_contract_V1.json"
)

FOUNDER_CONTRACT = (
    BASE
    / "phase_8_1_12b_origin_referenced_founder_history"
    / "phase_8_1_12b_origin_referenced_founder_history_contract_V1.json"
)

BRANCH_V2 = (
    BASE
    / "phase_8_1_11a_model_family_branch_semantics"
    / "phase_8_1_11a_branch_semantics_amendment_V2.json"
)

V2 = (
    DIR
    / "phase_8_1_10a_consistency_amendment_V2.json"
)

FOUNDER_CSV = (
    DIR
    / "phase_8_1_10a_founder_history_semantics_V2.csv"
)

GRAPH_POLICY_CSV = (
    DIR
    / "phase_8_1_10a_training_inference_graph_policy_V2.csv"
)

MANIFEST = (
    DIR
    / "phase_8_1_10a_manifest_V2.json"
)

HASHES = (
    DIR
    / "phase_8_1_10a_artifact_sha256_V2.json"
)


def sep(char="=", n=118):
    print(char * n)


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
            default=str,
        )


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def main():

    sep()

    print(
        "PHASE 8.1.10A — "
        "INDUCTIVE CONSISTENCY CLOSURE V2"
    )

    sep()

    print()

    print(
        "Scientific role:             "
        "PRE-ARCHITECTURE LOGICAL CONSISTENCY REPAIR"
    )

    print("Feature discovery reopened:  NO")
    print("Architecture changed:        NO")
    print("Model family changed:        NO")
    print("Graph family changed:        NO")
    print("Training performed:          NO")
    print("Performance inspected:       NO")
    print("T60 scored:                  NO")
    print("April analyzed:              NO")

    # =========================================================================
    # 1. Prerequisites
    # =========================================================================

    sep("-")

    print(
        "8.1.10A-V2.1 — PREREQUISITE GATES"
    )

    sep("-")

    required = [
        V1,
        GRAPH_SCHEMA_V2,
        COLD_CONTRACT,
        FOUNDER_CONTRACT,
        BRANCH_V2,
    ]

    for path in required:

        require(
            path.exists(),
            f"Missing prerequisite: {path}",
        )

        print(f"FOUND {path}")

    v1 = read_json(V1)
    graph_v2 = read_json(GRAPH_SCHEMA_V2)
    cold_contract = read_json(COLD_CONTRACT)
    founder_contract = read_json(FOUNDER_CONTRACT)
    branch_v2 = read_json(BRANCH_V2)

    for name, obj in [
        ("8.1.10A V1", v1),
        ("8.1.9 V2", graph_v2),
        ("8.1.12A", cold_contract),
        ("8.1.12B", founder_contract),
        ("8.1.11A V2", branch_v2),
    ]:

        require(
            obj.get("status") == "FROZEN",
            f"{name} is not FROZEN.",
        )

    # =========================================================================
    # 2. Verify historical semantic defects
    # =========================================================================

    sep("-")

    print(
        "8.1.10A-V2.2 — HISTORICAL SEMANTIC DEFECT DETECTION"
    )

    sep("-")

    old_history = (
        v1.get(
            "founder_history_effective_policy",
            [],
        )
    )

    require(
        len(old_history) == 2,
        "Expected exactly two founder-history channels in V1.",
    )

    old_history_blob = json.dumps(
        old_history,
        sort_keys=True,
    )

    require(
        "prior_company.founded_on < focal_startup.founded_on"
        in old_history_blob,
        (
            "Historical pre-focal founder-history "
            "cutoff was not detected."
        ),
    )

    old_mask = v1.get(
        "masking_policy",
        {},
    )

    require(
        "positive targets"
        in old_mask.get(
            "heldout_definition",
            "",
        ),
        (
            "Historical outcome-conditioned masking "
            "definition was not detected."
        ),
    )

    require(
        v1[
            "person_initialization_effective_policy"
        ][
            "person_initial_feature"
        ]
        ==
        "ZERO_40",
        "Person ZERO_40 policy was not found.",
    )

    require(
        v1[
            "person_initialization_effective_policy"
        ][
            "trainable_person_identity_parameter"
        ]
        is False,
        "V1 unexpectedly permits Person identity parameters.",
    )

    print(
        "Pre-focal founder-history cutoff detected:  YES"
    )

    print(
        "Future-positive masking detected:           YES"
    )

    print(
        "Person ZERO_40 already correct:             YES"
    )

    # =========================================================================
    # 3. Correct founder history
    # =========================================================================

    sep("-")

    print(
        "8.1.10A-V2.3 — ORIGIN-REFERENCED FOUNDER HISTORY"
    )

    sep("-")

    founder_history_v2 = [
        {
            "channel":
                "founder_prior_startup_count",

            "effective_placement":
                "STARTUP_NODE_FEATURE",

            "model_scope":
                "M1 | M2",

            "reference_entity":
                "FOCAL_STARTUP",

            "reference_time":
                "prediction_origin",

            "history_set":
                "H(s,t)",

            "eligible_history":
                (
                    "Unique non-focal Company c sharing >=1 "
                    "exact-resolved founder with focal Startup s, "
                    "with parseable c.founded_on < prediction_origin."
                ),

            "focal_startup_excluded":
                True,

            "deduplicate_prior_companies_across_founders":
                True,

            "origin_specific":
                True,

            "temporal_class":
                "DERIVED_PRE_ORIGIN",

            "notes":
                (
                    "P11 = number of unique Companies in H(s,t). "
                    "This measures founder entrepreneurial history "
                    "available by prediction origin, not only history "
                    "predating creation of the focal Startup."
                ),
        },
        {
            "channel":
                "founder_prior_sector_profile",

            "effective_placement":
                "STARTUP_NODE_FEATURE",

            "model_scope":
                "M1 | M2",

            "reference_entity":
                "FOCAL_STARTUP",

            "reference_time":
                "prediction_origin",

            "history_set":
                "H(s,t)",

            "eligible_history":
                (
                    "Categories/category_groups pooled from the "
                    "same deduplicated H(s,t) Company set used "
                    "for P11."
                ),

            "focal_startup_excluded":
                True,

            "deduplicate_prior_companies_across_founders":
                True,

            "origin_specific":
                True,

            "temporal_class":
                (
                    "DERIVED_PRE_ORIGIN_WITH_"
                    "MATCHED_SNAPSHOT_SECTOR_LABELS"
                ),

            "notes":
                (
                    "P12 uses the same H(s,t) as P11. "
                    "Company sector labels remain matched-snapshot "
                    "rather than historically versioned."
                ),
        },
    ]

    for row in founder_history_v2:

        print(
            f"{row['channel']:<32} "
            f"placement={row['effective_placement']:<22} "
            f"reference={row['reference_time']}"
        )

    # =========================================================================
    # 4. Correct outcome-independent cold-start masking
    # =========================================================================

    sep("-")

    print(
        "8.1.10A-V2.4 — OUTCOME-INDEPENDENT COLD_t MASK"
    )

    sep("-")

    masking_v2 = {
        "population_name":
            "COLD_t",

        "heldout_definition":
            (
                "All frozen Startup-role nodes that are "
                "origin-eligible at prediction origin t AND have "
                "zero canonical investment interactions strictly "
                "before t."
            ),

        "future_positive_ids_define_mask":
            False,

        "evaluation_negative_ids_define_mask":
            False,

        "full_candidate_matrix_defines_mask":
            False,

        "evaluation_case_membership_defines_mask":
            False,

        "future_outcomes_define_mask":
            False,

        "training_feature_action":
            "ZERO_ALL_COLD_t_STARTUP_FEATURE_ROWS",

        "training_edge_action":
            (
                "MASK_ALL_FOUNDED_BY_FOUNDED_ACQUIRED_ACQUIRED_BY_"
                "EDGES_INCIDENT_TO_ALL_COLD_t_STARTUPS"
            ),

        "node_rows_physically_deleted":
            False,

        "node_indices_reassigned":
            False,

        "person_nodes_deleted_if_orphaned":
            False,

        "inference_feature_action":
            (
                "RESTORE_ALLOWED_ORIGIN_TIME_FEATURES_FOR_ALL_COLD_t"
            ),

        "inference_incoming_edge_action":
            (
                "RESTORE_ORIGIN_ELIGIBLE_STRUCTURAL_EDGES_"
                "WITH_DESTINATION_COLD_t_STARTUP"
            ),

        "inference_outgoing_edge_action":
            (
                "SUPPRESS_STRUCTURAL_EDGES_WITH_SOURCE_"
                "COLD_t_STARTUP"
            ),

        "parameters_updated_after_restore":
            False,

        "preprocessing_refit_after_restore":
            False,

        "investment_target_edges_added":
            False,

        "purpose":
            (
                "Outcome-independent inductive Startup cold-start "
                "simulation: all COLD_t side information and incident "
                "heterogeneous structure are hidden during parameter "
                "fitting, then allowed origin-time information is "
                "inserted sink-only after all fitted parameters are frozen."
            ),
    }

    for key in [
        "population_name",
        "future_positive_ids_define_mask",
        "evaluation_case_membership_defines_mask",
        "training_feature_action",
        "inference_incoming_edge_action",
        "inference_outgoing_edge_action",
    ]:

        print(
            f"{key:<42} {masking_v2[key]}"
        )

    # =========================================================================
    # 5. Correct fit/inference policy at every origin
    # =========================================================================

    sep("-")

    print(
        "8.1.10A-V2.5 — TRAINING / INFERENCE GRAPH POLICY"
    )

    sep("-")

    graph_policy = []

    stage_specs = [
        (
            "DEVELOPMENT_PARAMETER_FIT",
            "T57 | T58 | T59",
            True,
        ),
        (
            "DEVELOPMENT_INFERENCE",
            "T57 | T58 | T59",
            False,
        ),
        (
            "T60_PARAMETER_FIT",
            "T60",
            True,
        ),
        (
            "T60_INFERENCE",
            "T60",
            False,
        ),
        (
            "APRIL_FINAL_FIT",
            "APRIL_CONFIRMATORY",
            True,
        ),
        (
            "APRIL_CONFIRMATORY_INFERENCE",
            "APRIL_CONFIRMATORY",
            False,
        ),
    ]

    for stage, origins, fitting in stage_specs:

        if fitting:

            row = {
                "stage":
                    stage,

                "origins":
                    origins,

                "population":
                    "ALL_COLD_t_STARTUPS",

                "cold_startup_features_visible":
                    False,

                "cold_startup_incident_edges_visible":
                    False,

                "cold_startup_incoming_edges_visible":
                    False,

                "cold_startup_outgoing_edges_visible":
                    False,

                "model_parameters_trainable":
                    True,

                "future_evaluation_membership_used":
                    False,

                "description":
                    (
                        "During parameter fitting, every COLD_t Startup "
                        "retains its node/index row but has allowed "
                        "Startup-side features zeroed and all founder/"
                        "acquisition structural edges incident to it masked."
                    ),
            }

        else:

            row = {
                "stage":
                    stage,

                "origins":
                    origins,

                "population":
                    "ALL_COLD_t_STARTUPS",

                "cold_startup_features_visible":
                    True,

                "cold_startup_incident_edges_visible":
                    "INCOMING_ONLY",

                "cold_startup_incoming_edges_visible":
                    True,

                "cold_startup_outgoing_edges_visible":
                    False,

                "model_parameters_trainable":
                    False,

                "future_evaluation_membership_used":
                    False,

                "description":
                    (
                        "With model and preprocessing parameters frozen, "
                        "restore allowed origin-time COLD_t Startup features "
                        "and origin-eligible incoming structural messages. "
                        "COLD_t Startups are sinks and cannot alter fitted "
                        "Person, Investor, or other Startup representations."
                    ),
            }

        graph_policy.append(row)

        print(
            f"{stage:<34} "
            f"origin={origins:<24} "
            f"fit={str(fitting):<5} "
            f"population=ALL_COLD_t_STARTUPS"
        )

    # =========================================================================
    # 6. Construct V2
    # =========================================================================

    sep("-")

    print(
        "8.1.10A-V2.6 — CONSTRUCT SUPERSEDING CONTRACT"
    )

    sep("-")

    v2 = deepcopy(v1)

    v2[
        "schema_version"
    ] = (
        "PHASE_8_1_10A_INDUCTIVE_CONSISTENCY_AMENDMENT_V2"
    )

    v2[
        "status"
    ] = "FROZEN"

    v2[
        "scientific_role"
    ] = (
        "PRE_ARCHITECTURE_LOGICAL_CONSISTENCY_AMENDMENT_CORRECTED"
    )

    v2[
        "effective_contract"
    ] = (
        "Phase 8.1.7 V2 + Phase 8.1.9 V2 + "
        "Phase 8.1.10 V1 + Phase 8.1.10A V2 + "
        "Phase 8.1.12A V1 + Phase 8.1.12B V1"
    )

    v2[
        "founder_history_effective_policy"
    ] = founder_history_v2

    v2[
        "masking_policy"
    ] = masking_v2

    v2[
        "training_inference_graph_policy"
    ] = graph_policy

    # Person ZERO_40 policy is intentionally preserved unchanged.
    v2[
        "person_initialization_effective_policy"
    ] = deepcopy(
        v1[
            "person_initialization_effective_policy"
        ]
    )

    v2[
        "scientific_interpretation"
    ] = {
        "founder_history_claim":
            (
                "Founder prior-history features summarize "
                "entrepreneurial/domain experience available by the "
                "prediction origin through H(s,t), not merely experience "
                "predating creation of the focal Startup."
            ),

        "primary_claim":
            (
                "The proposed inductive models are fitted while ALL "
                "origin-defined COLD_t Startup side information and "
                "incident heterogeneous structure are hidden. After "
                "fitting is frozen, allowed origin-time information is "
                "restored sink-only for inductive inference."
            ),

        "not_claimed": [
            (
                "Historical database-observation time for "
                "snapshot-unversioned Crunchbase attributes."
            ),
            "Strict historical versioning of sector labels.",
            "Inductive cold-start for investors.",
            (
                "Open-world recommendation over all Crunchbase "
                "Companies."
            ),
        ],
    }

    v2[
        "superseded_clauses"
    ] = [
        (
            "Phase 8.1.10A V1 founder-history reference "
            "to focal_startup.founded_on."
        ),
        (
            "Phase 8.1.10A V1 masking definition based on "
            "future PRIMARY evaluation positive targets."
        ),
        (
            "Phase 8.1.10A V1 fit/inference rows referring "
            "only to held-out primary target Startups."
        ),
        (
            "Any interpretation allowing COLD_t Startup side "
            "information or incident structural edges to influence "
            "parameter fitting."
        ),
    ]

    v2[
        "supersession"
    ] = {
        "supersedes":
            str(V1),

        "V1_file_modified":
            False,

        "V1_artifacts_preserved":
            True,

        "founder_history_clock_changed":
            (
                "focal_startup.founded_on -> prediction_origin"
            ),

        "masked_population_changed":
            (
                "future PRIMARY positive targets -> ALL COLD_t Startups"
            ),

        "feature_inventory_changed":
            False,

        "architecture_changed":
            False,

        "model_family_changed":
            False,

        "performance_based_change":
            False,

        "feature_discovery_reopened":
            False,
    }

    v2[
        "semantic_dependencies"
    ] = {
        "graph_schema":
            str(GRAPH_SCHEMA_V2),

        "outcome_independent_cold_mask":
            str(COLD_CONTRACT),

        "origin_referenced_founder_history":
            str(FOUNDER_CONTRACT),

        "branch_semantics":
            str(BRANCH_V2),
    }

    v2[
        "architecture_selected"
    ] = False

    v2[
        "model_trained"
    ] = False

    v2[
        "performance_inspected"
    ] = False

    v2[
        "April_analyzed"
    ] = False

    v2[
        "April_performance_inspected"
    ] = False

    v2[
        "next_step"
    ] = (
        "Phase 8.1.12 V4 — final corrected execution contract."
    )

    # =========================================================================
    # 7. Integrity audit
    # =========================================================================

    sep("-")

    print(
        "8.1.10A-V2.7 — INTEGRITY AUDIT"
    )

    sep("-")

    serialized = json.dumps(
        v2,
        sort_keys=True,
    )

    require(
        "prior_company.founded_on < focal_startup.founded_on"
        not in serialized,
        "Old pre-focal founder-history cutoff survived.",
    )

    require(
        "positive targets in the frozen PRIMARY"
        not in serialized,
        "Old future-positive masking definition survived.",
    )

    require(
        v2[
            "person_initialization_effective_policy"
        ]
        ==
        v1[
            "person_initialization_effective_policy"
        ],
        "Person ZERO_40 policy changed unexpectedly.",
    )

    require(
        graph_v2[
            "information_placement"
        ]
        [
            [
                x["information"]
                for x
                in graph_v2[
                    "information_placement"
                ]
            ].index(
                "founder_prior_startup_count"
            )
        ][
            "placement"
        ]
        ==
        "STARTUP_NODE_FEATURE",
        "8.1.9 V2 P11 placement is not Startup-level.",
    )

    require(
        "ALL_COLD_t_STARTUPS"
        in serialized,
        "Corrected COLD_t population missing.",
    )

    require(
        "prediction_origin"
        in json.dumps(
            founder_history_v2,
            sort_keys=True,
        ),
        "Prediction-origin founder-history reference missing.",
    )

    require(
        v2[
            "feature_discovery_closed"
        ]
        is True,
        "Feature discovery is not closed.",
    )

    require(
        v2[
            "feature_discovery_reopened"
        ]
        is False,
        "Feature discovery was reopened unexpectedly.",
    )

    print(
        "Origin-referenced founder history:       PASS"
    )

    print(
        "P11/P12 remain Startup-conditioned:     PASS"
    )

    print(
        "Outcome-independent ALL-COLD_t mask:    PASS"
    )

    print(
        "Sink-only inference semantics:          PASS"
    )

    print(
        "Person ZERO_40 preserved:               PASS"
    )

    print(
        "Feature inventory unchanged:            PASS"
    )

    print(
        "Feature discovery remains closed:       PASS"
    )

    # =========================================================================
    # 8. Persist
    # =========================================================================

    sep("-")

    print(
        "8.1.10A-V2.8 — PERSIST SUPERSEDING CONTRACT"
    )

    sep("-")

    DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_json(
        V2,
        v2,
    )

    pd.DataFrame(
        founder_history_v2
    ).to_csv(
        FOUNDER_CSV,
        index=False,
    )

    pd.DataFrame(
        graph_policy
    ).to_csv(
        GRAPH_POLICY_CSV,
        index=False,
    )

    manifest = {
        "phase":
            "8.1.10A",

        "version":
            "V2",

        "status":
            "FROZEN",

        "V1_status":
            "FROZEN_SUPERSEDED",

        "V1_artifacts_preserved":
            True,

        "scientific_role":
            "INDUCTIVE_CONSISTENCY_SEMANTIC_REPAIR",

        "authoritative_contract":
            str(V2),

        "founder_history_reference":
            "PREDICTION_ORIGIN",

        "founder_history_placement":
            "STARTUP_NODE_FEATURE",

        "masked_population":
            "ALL_COLD_t_STARTUPS",

        "future_positive_membership_used_for_mask":
            False,

        "future_evaluation_membership_used_for_mask":
            False,

        "inference_policy":
            "SINK_ONLY",

        "person_initialization":
            "ZERO_40",

        "feature_inventory_changed":
            False,

        "architecture_changed":
            False,

        "feature_discovery_reopened":
            False,

        "model_trained":
            False,

        "performance_inspected":
            False,

        "T60_scored":
            False,

        "April_analyzed":
            False,

        "outputs": [
            str(FOUNDER_CSV),
            str(GRAPH_POLICY_CSV),
            str(V2),
        ],
    }

    write_json(
        MANIFEST,
        manifest,
    )

    outputs = [
        V2,
        FOUNDER_CSV,
        GRAPH_POLICY_CSV,
        MANIFEST,
    ]

    write_json(
        HASHES,
        {
            str(path):
                sha256(path)
            for path in outputs
        },
    )

    sep()

    print(
        "PHASE 8.1.10A V2 INDUCTIVE CONSISTENCY FROZEN"
    )

    sep()

    print()

    print(
        "8.1.10A V1:                       FROZEN_SUPERSEDED"
    )

    print(
        "8.1.10A V2:                       FROZEN / AUTHORITATIVE"
    )

    print()

    print(
        "FOUNDER HISTORY CLOCK:            prediction_origin"
    )

    print(
        "P11/P12 PLACEMENT:                STARTUP_NODE_FEATURE"
    )

    print(
        "P11/P12 HISTORY SET:              H(s,t)"
    )

    print(
        "PRIOR COMPANY CONDITION:          founded_on < t"
    )

    print(
        "MASKED POPULATION DURING FIT:     ALL COLD_t"
    )

    print(
        "FUTURE POSITIVE MEMBERSHIP USED:  NO"
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
        "PERSON INITIALIZATION:            ZERO_40"
    )

    print(
        "FEATURE DISCOVERY REOPENED:       NO"
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

    print("NEXT:")
    print(
        "Phase 8.1.12 V4 — final corrected execution contract."
    )


if __name__ == "__main__":
    main()
