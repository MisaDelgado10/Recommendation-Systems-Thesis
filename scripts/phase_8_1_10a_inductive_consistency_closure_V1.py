from pathlib import Path
import hashlib
import json

import pandas as pd


# =============================================================================
# PHASE 8.1.10A — INDUCTIVE CONSISTENCY CLOSURE V1
# =============================================================================
#
# Scientific role:
#
#   PRE-ARCHITECTURE LOGICAL-CONSISTENCY AMENDMENT
#
# PURPOSE
# -------
#
# Resolve two issues identified after freezing Phase 8.1.9 and 8.1.10:
#
#   A. Focal-dependent founder history cannot be represented by one
#      origin-wide Person feature vector.
#
#   B. Held-out primary cold-start Startup covariates must not influence
#      parameter fitting through heterogeneous message passing.
#
# NO new information channel is introduced.
# Feature discovery remains CLOSED.
#
# Existing artifacts are preserved and not overwritten.
#
# =============================================================================


BASE = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation"
)

INPUT_CONTRACT = (
    BASE
    / "phase_8_1_7_input_contract_amendment"
    / "phase_8_1_7_contract_amendment_V2.json"
)

GRAPH_SCHEMA = (
    BASE
    / "phase_8_1_9_heterogeneous_graph_schema"
    / "phase_8_1_9_graph_schema_V1.json"
)

ENCODING_CONTRACT = (
    BASE
    / "phase_8_1_10_identity_free_encoding_policy"
    / "phase_8_1_10_identity_free_encoding_contract_V1.json"
)


OUTPUT_DIR = (
    BASE
    / "phase_8_1_10a_inductive_consistency_closure"
)

FOUNDER_HISTORY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_10a_founder_history_semantics_V1.csv"
)

GRAPH_VISIBILITY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_10a_training_inference_graph_policy_V1.csv"
)

AMENDMENT_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_10a_consistency_amendment_V1.json"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_10a_manifest_V1.json"
)

HASH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_10a_artifact_sha256_V1.json"
)


def separator(char="=", width=118):
    print(char * width)


def require(condition, message):

    if not condition:
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
        "PHASE 8.1.10A — "
        "INDUCTIVE CONSISTENCY CLOSURE V1"
    )

    separator()

    print()

    print(
        "Scientific role:             "
        "PRE-ARCHITECTURE LOGICAL-CONSISTENCY AMENDMENT"
    )

    print("New feature introduced:      NO")
    print("Feature discovery reopened:  NO")
    print("Architecture selected:       NO")
    print("Model trained:               NO")
    print("April analyzed:              NO")

    # =========================================================================
    # 1. Prerequisite gates
    # =========================================================================

    separator("-")

    print(
        "8.1.10A.1 — PREREQUISITE GATES"
    )

    separator("-")

    for path in [
        INPUT_CONTRACT,
        GRAPH_SCHEMA,
        ENCODING_CONTRACT,
    ]:

        require(
            path.exists(),
            f"Missing prerequisite artifact: {path}",
        )

        print(
            f"FOUND {path}"
        )

    input_contract = read_json(
        INPUT_CONTRACT
    )

    graph_schema = read_json(
        GRAPH_SCHEMA
    )

    encoding_contract = read_json(
        ENCODING_CONTRACT
    )

    require(
        input_contract.get("status")
        ==
        "FROZEN",
        "Phase 8.1.7 V2 not frozen.",
    )

    require(
        graph_schema.get("status")
        ==
        "FROZEN",
        "Phase 8.1.9 not frozen.",
    )

    require(
        encoding_contract.get("status")
        ==
        "FROZEN",
        "Phase 8.1.10 not frozen.",
    )

    require(
        input_contract.get(
            "feature_discovery_closed"
        )
        is True,
        "Feature discovery unexpectedly open.",
    )

    require(
        encoding_contract.get(
            "model_trained"
        )
        is False,
        "Model unexpectedly trained.",
    )

    require(
        encoding_contract.get(
            "April_analyzed"
        )
        is False,
        "April unexpectedly analyzed.",
    )

    print()

    print(
        "Upstream frozen contracts: PASS"
    )

    # =========================================================================
    # 2. Detect founder-history inconsistency
    # =========================================================================

    separator("-")

    print(
        "8.1.10A.2 — FOUNDER-HISTORY "
        "CONSISTENCY AUDIT"
    )

    separator("-")

    placement_records = {
        row[
            "information"
        ]:
            row
        for row
        in graph_schema[
            "information_placement"
        ]
    }

    for channel in [
        "founder_prior_startup_count",
        "founder_prior_sector_profile",
    ]:

        require(
            channel
            in
            placement_records,
            f"Missing graph placement: {channel}",
        )

        require(
            placement_records[
                channel
            ][
                "placement"
            ]
            ==
            "PERSON_NODE_FEATURE",
            (
                f"Unexpected pre-amendment placement "
                f"for {channel}."
            ),
        )

    old_history_policy = (
        graph_schema[
            "founder_history_policy"
        ]
    )

    require(
        old_history_policy.get(
            "focal_startup_excluded"
        )
        is True,
        (
            "Expected focal-dependent founder-history "
            "definition not found."
        ),
    )

    require(
        encoding_contract[
            "evaluation_independence"
        ][
            "origin_specific_feature_store_built_once"
        ]
        is True,
        (
            "Expected one-origin feature-store policy "
            "not found."
        ),
    )

    print(
        "Detected:"
    )

    print(
        "  founder history placement = Person node"
    )

    print(
        "  history excludes focal startup"
    )

    print(
        "  feature store = once per origin"
    )

    print()

    print(
        "Conclusion:"
    )

    print(
        "  focal-dependent Person state would be "
        "internally inconsistent."
    )

    # =========================================================================
    # 3. Correct founder-history semantics
    # =========================================================================

    separator("-")

    print(
        "8.1.10A.3 — CORRECTED FOUNDER-HISTORY SEMANTICS"
    )

    separator("-")

    founder_rows = [
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
                "focal_startup.founded_on",

            "eligible_history":
                (
                    "unique non-focal companies sharing >=1 "
                    "resolved founder with focal startup AND "
                    "prior_company.founded_on < "
                    "focal_startup.founded_on"
                ),

            "deduplicate_prior_companies_across_founders":
                True,

            "origin_gate":
                "focal_startup.founded_on <= origin",

            "temporal_class":
                "DERIVED_PRE_FOCAL_FOUNDING",

            "notes":
                (
                    "Measures entrepreneurial experience that "
                    "existed before creation of the focal startup."
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
                "focal_startup.founded_on",

            "eligible_history":
                (
                    "categories/category_groups of the same "
                    "deduplicated prior-founder-company set"
                ),

            "deduplicate_prior_companies_across_founders":
                True,

            "origin_gate":
                "focal_startup.founded_on <= origin",

            "temporal_class":
                (
                    "DERIVED_PRE_FOCAL_FOUNDING_WITH_"
                    "MATCHED_SNAPSHOT_SECTOR_LABELS"
                ),

            "notes":
                (
                    "Edge/history timing is pre-focal; "
                    "sector labels remain matched-snapshot."
                ),
        },
    ]

    founder_df = pd.DataFrame(
        founder_rows
    )

    print(
        founder_df[
            [
                "channel",
                "effective_placement",
                "reference_time",
                "temporal_class",
            ]
        ].to_string(
            index=False
        )
    )

    person_initialization_amendment = {
        "person_initial_feature":
            "ZERO_40",

        "trainable_person_identity_parameter":
            False,

        "person_current_snapshot_features":
            False,

        "founder_history_stored_on_person":
            False,

        "person_role":
            (
                "STRUCTURAL_INTERMEDIARY_FOR_FOUNDER_MESSAGE_PASSING"
            ),

        "rationale":
            (
                "A shared zero initialization prevents focal-dependent "
                "history from being stored in one Person vector while "
                "still allowing explicit Person nodes to mediate "
                "Startup↔Founder↔Startup propagation."
            ),
    }

    print()

    print(
        "Person initialization amendment:"
    )

    for key, value in (
        person_initialization_amendment.items()
    ):

        print(
            f"{key:<46} {value}"
        )

    # =========================================================================
    # 4. Strict inductive train/inference graph separation
    # =========================================================================

    separator("-")

    print(
        "8.1.10A.4 — STRICT INDUCTIVE "
        "TRAIN / INFERENCE GRAPH POLICY"
    )

    separator("-")

    graph_visibility_rows = [
        {
            "stage":
                "DEVELOPMENT_PARAMETER_FIT",

            "origins":
                "T57 | T58 | T59",

            "heldout_primary_target_startup_features_visible":
                False,

            "heldout_primary_target_incident_edges_visible":
                False,

            "model_parameters_trainable":
                True,

            "description":
                (
                    "Primary cold-start target Startups for the "
                    "evaluation origin remain index-addressable only "
                    "for bookkeeping; their feature rows are zeroed "
                    "and all incident heterogeneous edges are masked "
                    "during parameter fitting."
                ),
        },

        {
            "stage":
                "DEVELOPMENT_INFERENCE",

            "origins":
                "T57 | T58 | T59",

            "heldout_primary_target_startup_features_visible":
                True,

            "heldout_primary_target_incident_edges_visible":
                True,

            "model_parameters_trainable":
                False,

            "description":
                (
                    "After fitting, restore allowed side information "
                    "and origin-eligible founder/acquisition edges, "
                    "then recompute graph representations with frozen "
                    "parameters."
                ),
        },

        {
            "stage":
                "T60_PARAMETER_FIT",

            "origins":
                "T60",

            "heldout_primary_target_startup_features_visible":
                False,

            "heldout_primary_target_incident_edges_visible":
                False,

            "model_parameters_trainable":
                True,

            "description":
                (
                    "Same strict inductive policy after all architecture "
                    "selection is frozen. T60 is never used for selection."
                ),
        },

        {
            "stage":
                "T60_INFERENCE",

            "origins":
                "T60",

            "heldout_primary_target_startup_features_visible":
                True,

            "heldout_primary_target_incident_edges_visible":
                True,

            "model_parameters_trainable":
                False,

            "description":
                (
                    "Cold-start target nodes are inserted/restored only "
                    "after parameter fitting."
                ),
        },

        {
            "stage":
                "APRIL_FINAL_FIT",

            "origins":
                "APRIL_CONFIRMATORY",

            "heldout_primary_target_startup_features_visible":
                False,

            "heldout_primary_target_incident_edges_visible":
                False,

            "model_parameters_trainable":
                True,

            "description":
                (
                    "April primary cold-start targets remain hidden "
                    "from message passing during final parameter fit. "
                    "No April performance is inspected."
                ),
        },

        {
            "stage":
                "APRIL_CONFIRMATORY_INFERENCE",

            "origins":
                "APRIL_CONFIRMATORY",

            "heldout_primary_target_startup_features_visible":
                True,

            "heldout_primary_target_incident_edges_visible":
                True,

            "model_parameters_trainable":
                False,

            "description":
                (
                    "Restore only allowed side information and "
                    "origin-eligible edges for the one frozen "
                    "confirmatory evaluation."
                ),
        },
    ]

    visibility_df = pd.DataFrame(
        graph_visibility_rows
    )

    print(
        visibility_df[
            [
                "stage",
                "origins",
                "heldout_primary_target_startup_features_visible",
                "heldout_primary_target_incident_edges_visible",
                "model_parameters_trainable",
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # 5. Detailed masking semantics
    # =========================================================================

    separator("-")

    print(
        "8.1.10A.5 — MASKING SEMANTICS"
    )

    separator("-")

    masking_policy = {
        "heldout_definition":
            (
                "Startup IDs appearing as positive targets in the "
                "frozen PRIMARY cold-start stratum for the relevant "
                "evaluation origin."
            ),

        "node_indices_reassigned":
            False,

        "node_rows_physically_deleted":
            False,

        "training_feature_action":
            "ZERO_HELDOUT_PRIMARY_STARTUP_FEATURE_ROWS",

        "training_edge_action":
            (
                "MASK_ALL_FOUNDED_BY_FOUNDED_ACQUIRED_"
                "ACQUIRED_BY_EDGES_INCIDENT_TO_HELDOUT_"
                "PRIMARY_STARTUPS"
            ),

        "person_nodes_deleted_if_orphaned":
            False,

        "person_nodes_if_only_heldout_neighbors":
            (
                "remain zero isolated context nodes during fit"
            ),

        "inference_feature_action":
            "RESTORE_ALLOWED_FEATURE_ROWS",

        "inference_edge_action":
            "RESTORE_ORIGIN_ELIGIBLE_STRUCTURAL_EDGES",

        "parameters_updated_after_restore":
            False,

        "evaluation_negative_ids_define_graph":
            False,

        "full_candidate_matrix_defines_graph":
            False,

        "investment_target_edges_added":
            False,

        "purpose":
            (
                "Prevent held-out primary cold-start covariates from "
                "affecting learned GNN parameters while retaining "
                "genuinely inductive insertion at inference."
            ),
    }

    for key, value in (
        masking_policy.items()
    ):

        print(
            f"{key:<48} {value}"
        )

    # =========================================================================
    # 6. Scientific interpretation freeze
    # =========================================================================

    separator("-")

    print(
        "8.1.10A.6 — SCIENTIFIC INTERPRETATION"
    )

    separator("-")

    interpretation = {
        "primary_claim":
            (
                "The proposed model generates representations for "
                "investment-cold target startups whose allowed side "
                "information and heterogeneous incident edges were "
                "not available to the model during parameter fitting."
            ),

        "founder_history_claim":
            (
                "Founder prior-history features summarize founder "
                "entrepreneurial/domain experience that predates the "
                "creation of the focal startup."
            ),

        "not_claimed":
            [
                (
                    "Historical database-observation time for "
                    "snapshot-unversioned Crunchbase attributes."
                ),
                (
                    "Strict historical versioning of sector labels."
                ),
                (
                    "Inductive cold-start for investors."
                ),
            ],
    }

    print(
        interpretation[
            "primary_claim"
        ]
    )

    print()

    print(
        interpretation[
            "founder_history_claim"
        ]
    )

    # =========================================================================
    # 7. Persist amendment
    # =========================================================================

    separator("-")

    print(
        "8.1.10A.7 — PERSIST CONSISTENCY AMENDMENT"
    )

    separator("-")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    founder_df.to_csv(
        FOUNDER_HISTORY_OUTPUT,
        index=False,
    )

    visibility_df.to_csv(
        GRAPH_VISIBILITY_OUTPUT,
        index=False,
    )

    amendment = {
        "schema_version":
            "PHASE_8_1_10A_INDUCTIVE_CONSISTENCY_AMENDMENT_V1",

        "status":
            "FROZEN",

        "scientific_role":
            "PRE_ARCHITECTURE_LOGICAL_CONSISTENCY_AMENDMENT",

        "effective_contract":
            (
                "Phase 8.1.7 V2 + Phase 8.1.9 V1 + "
                "Phase 8.1.10 V1 + Phase 8.1.10A V1"
            ),

        "new_information_channels_added":
            False,

        "feature_discovery_reopened":
            False,

        "superseded_clauses": [
            (
                "Phase 8.1.9 placement of "
                "founder_prior_startup_count as PERSON_NODE_FEATURE"
            ),
            (
                "Phase 8.1.9 placement of "
                "founder_prior_sector_profile as PERSON_NODE_FEATURE"
            ),
            (
                "Phase 8.1.10 Person initialization source "
                "SHARED_PERSON_HISTORY_ENCODER"
            ),
            (
                "Any interpretation allowing held-out primary "
                "cold-start features/edges to influence parameter fit"
            ),
        ],

        "founder_history_effective_policy":
            founder_df.to_dict(
                orient="records"
            ),

        "person_initialization_effective_policy":
            person_initialization_amendment,

        "training_inference_graph_policy":
            visibility_df.to_dict(
                orient="records"
            ),

        "masking_policy":
            masking_policy,

        "scientific_interpretation":
            interpretation,

        "feature_discovery_closed":
            True,

        "graph_schema_effectively_amended":
            True,

        "architecture_selected":
            False,

        "model_trained":
            False,

        "April_analyzed":
            False,

        "April_performance_inspected":
            False,

        "next_step":
            (
                "Phase 8.1.11 — freeze exact encoder, "
                "R-GCN and scoring architecture under the "
                "corrected inductive contract."
            ),
    }

    write_json(
        AMENDMENT_OUTPUT,
        amendment,
    )

    manifest = {
        "phase":
            "8.1.10A",

        "status":
            "FROZEN",

        "outputs": [
            str(FOUNDER_HISTORY_OUTPUT),
            str(GRAPH_VISIBILITY_OUTPUT),
            str(AMENDMENT_OUTPUT),
        ],

        "new_information_channels_added":
            False,

        "feature_discovery_reopened":
            False,

        "architecture_selected":
            False,

        "model_trained":
            False,

        "April_analyzed":
            False,
    }

    write_json(
        MANIFEST_OUTPUT,
        manifest,
    )

    outputs = [
        FOUNDER_HISTORY_OUTPUT,
        GRAPH_VISIBILITY_OUTPUT,
        AMENDMENT_OUTPUT,
        MANIFEST_OUTPUT,
    ]

    write_json(
        HASH_OUTPUT,
        {
            str(path):
                sha256_file(path)
            for path
            in outputs
        },
    )

    separator()

    print(
        "PHASE 8.1.10A INDUCTIVE "
        "CONSISTENCY CLOSURE FROZEN"
    )

    separator()

    print()

    print(
        "NEW FEATURES ADDED:               NO"
    )

    print(
        "FEATURE DISCOVERY REOPENED:       NO"
    )

    print(
        "FOUNDER HISTORY:"
    )

    print(
        "  Person feature -> Startup feature"
    )

    print(
        "  cutoff = focal startup founding date"
    )

    print()

    print(
        "PERSON INITIAL FEATURES:          ZERO_40"
    )

    print(
        "HELD-OUT COLD STARTUPS DURING FIT:"
    )

    print(
        "  features hidden"
    )

    print(
        "  incident structural edges hidden"
    )

    print()

    print(
        "HELD-OUT COLD STARTUPS AT INFERENCE:"
    )

    print(
        "  allowed features restored"
    )

    print(
        "  eligible structural edges restored"
    )

    print(
        "  model parameters frozen"
    )

    print()

    print(
        "ARCHITECTURE SELECTED:            NO"
    )

    print(
        "MODEL TRAINED:                    NO"
    )

    print(
        "APRIL ANALYZED:                   NO"
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Phase 8.1.11 — exact neural architecture freeze."
    )


if __name__ == "__main__":
    main()