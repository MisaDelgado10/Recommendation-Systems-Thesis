from pathlib import Path
import hashlib
import json

import pandas as pd


# =============================================================================
# PHASE 8.1.9 — HETEROGENEOUS GRAPH SCHEMA FREEZE V1
# =============================================================================
#
# Scientific role:
#
#   PRE-MATERIALIZATION HETEROGENEOUS GRAPH-SCHEMA FREEZE
#
# PURPOSE
# -------
#
# Translate the frozen Phase-8.1.7 V2 information contract into an explicit
# heterogeneous graph representation BEFORE node/edge materialization,
# neural architecture selection, or model training.
#
# This phase freezes:
#
#   - node types;
#   - semantic relation types;
#   - temporal eligibility rules;
#   - information placement;
#   - graph variants / structural ablations;
#   - candidate-universe invariance;
#   - identity restrictions.
#
# This phase DOES NOT:
#
#   - materialize the new graph;
#   - select embedding dimensions;
#   - select a GNN architecture;
#   - initialize trainable node vectors;
#   - train or score any model;
#   - inspect April.
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

INPUT_CHANNELS = (
    BASE
    / "phase_8_1_7_input_contract_amendment"
    / "phase_8_1_7_final_information_channels_V2.csv"
)

MODEL_FAMILY_CONTRACT = (
    BASE
    / "phase_8_1_7_input_contract_amendment"
    / "phase_8_1_7_model_family_contract_V2.csv"
)

PHASE3_GRAPH_CONFIG = Path(
    "configs/phase_3_itrs_graph_schema.json"
)

PHASE3_NODE_INDEX = Path(
    "data/experimental/phase_3/"
    "model_ready/node_index.parquet"
)

PHASE3_EDGE_INDEX = Path(
    "data/experimental/phase_3/"
    "model_ready/edge_index.npy"
)


OUTPUT_DIR = (
    BASE
    / "phase_8_1_9_heterogeneous_graph_schema"
)

NODE_SCHEMA_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_9_node_type_schema_V1.csv"
)

RELATION_SCHEMA_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_9_relation_schema_V1.csv"
)

PLACEMENT_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_9_information_placement_V1.csv"
)

VARIANT_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_9_graph_variants_V1.csv"
)

SCHEMA_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_9_graph_schema_V1.json"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_9_manifest_V1.json"
)

HASH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_9_artifact_sha256_V1.json"
)


EXPECTED_INVESTORS = 165_975
EXPECTED_STARTUPS = 311_589


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
        "PHASE 8.1.9 — "
        "HETEROGENEOUS GRAPH SCHEMA FREEZE V1"
    )

    separator()

    print()

    print(
        "Scientific role:             "
        "PRE-MATERIALIZATION GRAPH-SCHEMA FREEZE"
    )

    print(
        "Feature discovery open:      NO"
    )

    print(
        "Graph materialized:          NO"
    )

    print(
        "Architecture selected:       NO"
    )

    print(
        "Model trained:               NO"
    )

    print(
        "April analyzed:              NO"
    )

    # =========================================================================
    # 1. Prerequisite gates
    # =========================================================================

    separator("-")

    print(
        "8.1.9.1 — PREREQUISITE GATES"
    )

    separator("-")

    required = [
        INPUT_CONTRACT,
        INPUT_CHANNELS,
        MODEL_FAMILY_CONTRACT,
        PHASE3_GRAPH_CONFIG,
        PHASE3_NODE_INDEX,
        PHASE3_EDGE_INDEX,
    ]

    for path in required:

        require(
            path.exists(),
            f"Missing prerequisite artifact: {path}",
        )

        print(
            f"FOUND {path}"
        )

    contract = read_json(
        INPUT_CONTRACT
    )

    require(
        contract.get("status")
        ==
        "FROZEN",
        "Phase 8.1.7 V2 is not FROZEN.",
    )

    require(
        contract.get(
            "feature_discovery_closed"
        )
        is True,
        "Feature discovery is not closed.",
    )

    require(
        contract.get(
            "April_analyzed"
        )
        is False,
        "April unexpectedly analyzed.",
    )

    require(
        contract.get(
            "model_trained"
        )
        is False,
        "Model unexpectedly trained.",
    )

    print()

    print(
        "Phase 8.1.7 V2 gate: PASS"
    )

    # =========================================================================
    # 2. Frozen canonical role-node universe
    # =========================================================================

    separator("-")

    print(
        "8.1.9.2 — CANONICAL ROLE-NODE UNIVERSE"
    )

    separator("-")

    nodes = pd.read_parquet(
        PHASE3_NODE_INDEX,
        columns=[
            "node_index",
            "node_type",
            "raw_entity_id",
        ],
    )

    investor_count = int(
        (
            nodes[
                "node_type"
            ]
            ==
            "investor"
        ).sum()
    )

    startup_count = int(
        (
            nodes[
                "node_type"
            ]
            ==
            "startup"
        ).sum()
    )

    require(
        investor_count
        ==
        EXPECTED_INVESTORS,
        (
            "Investor role-node universe drift: "
            f"{investor_count:,}"
        ),
    )

    require(
        startup_count
        ==
        EXPECTED_STARTUPS,
        (
            "Startup role-node universe drift: "
            f"{startup_count:,}"
        ),
    )

    print(
        f"Investor role nodes: "
        f"{investor_count:,} PASS"
    )

    print(
        f"Startup role nodes:  "
        f"{startup_count:,} PASS"
    )

    print()

    print(
        "Candidate startup universe expanded: NO"
    )

    # =========================================================================
    # 3. Freeze node types
    # =========================================================================

    separator("-")

    print(
        "8.1.9.3 — NODE-TYPE SCHEMA"
    )

    separator("-")

    node_rows = [
        {
            "node_type":
                "investor",

            "namespace":
                "investor::<crunchbase_uuid>",

            "source":
                "frozen Phase-3 investor role universe",

            "candidate_entity":
                False,

            "context_entity":
                True,

            "primary_graph":
                True,

            "identity_policy":
                (
                    "Preserve baseline investor identity/history "
                    "policy; startup-side inductive restriction "
                    "does not require investor cold-start."
                ),

            "temporal_policy":
                "PRESERVE_ITRS_INVESTOR_PROTOCOL",
        },

        {
            "node_type":
                "startup",

            "namespace":
                "startup::<crunchbase_uuid>",

            "source":
                "frozen 311,589 Startup role universe",

            "candidate_entity":
                True,

            "context_entity":
                True,

            "primary_graph":
                True,

            "identity_policy":
                (
                    "NO startup-ID trainable embedding; "
                    "NO per-startup learned document vector."
                ),

            "temporal_policy":
                (
                    "Origin-specific feature construction "
                    "under frozen Phase-8 protocol."
                ),
        },

        {
            "node_type":
                "person",

            "namespace":
                "person::<crunchbase_uuid>",

            "source":
                (
                    "exact-resolved People linked through "
                    "companies.founder_links for frozen "
                    "Startup role entities"
                ),

            "candidate_entity":
                False,

            "context_entity":
                True,

            "primary_graph":
                True,

            "identity_policy":
                (
                    "NO trainable Person-ID embedding; "
                    "Person identity is used only for graph "
                    "connectivity and shared feature transforms."
                ),

            "temporal_policy":
                (
                    "Founder relation semantically gated by "
                    "startup.founded_on <= origin."
                ),
        },
    ]

    node_df = pd.DataFrame(
        node_rows
    )

    print(
        node_df[
            [
                "node_type",
                "candidate_entity",
                "primary_graph",
                "identity_policy",
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # 4. Freeze primary semantic relations
    # =========================================================================

    separator("-")

    print(
        "8.1.9.4 — PRIMARY RELATION SCHEMA"
    )

    separator("-")

    relation_rows = [
        {
            "relation_id":
                "R01",

            "src_type":
                "startup",

            "relation":
                "founded_by",

            "dst_type":
                "person",

            "primary_graph":
                True,

            "inverse_relation":
                "founded",

            "temporal_class":
                "SEMANTIC_EVENT_DATE_PROXY",

            "origin_gate":
                "startup.founded_on <= origin",

            "database_observation_time_known":
                False,

            "target_pair_mask_required":
                False,

            "notes":
                (
                    "Explicit founder relation approved by "
                    "Phase 8.1.7 V2."
                ),
        },

        {
            "relation_id":
                "R02",

            "src_type":
                "person",

            "relation":
                "founded",

            "dst_type":
                "startup",

            "primary_graph":
                True,

            "inverse_relation":
                "founded_by",

            "temporal_class":
                "SEMANTIC_EVENT_DATE_PROXY",

            "origin_gate":
                "startup.founded_on <= origin",

            "database_observation_time_known":
                False,

            "target_pair_mask_required":
                False,

            "notes":
                (
                    "Explicit inverse for message passing."
                ),
        },

        {
            "relation_id":
                "R03",

            "src_type":
                "ROLE_NODE",

            "relation":
                "acquired",

            "dst_type":
                "ROLE_NODE",

            "primary_graph":
                True,

            "inverse_relation":
                "acquired_by",

            "temporal_class":
                "STRICT_TIMESTAMPED_EVENT",

            "origin_gate":
                "acquisition.announced_on <= origin",

            "database_observation_time_known":
                True,

            "target_pair_mask_required":
                True,

            "notes":
                (
                    "Reuse frozen Phase-8 acquisition-safe "
                    "role projection and target-pair exclusion."
                ),
        },

        {
            "relation_id":
                "R04",

            "src_type":
                "ROLE_NODE",

            "relation":
                "acquired_by",

            "dst_type":
                "ROLE_NODE",

            "primary_graph":
                True,

            "inverse_relation":
                "acquired",

            "temporal_class":
                "STRICT_TIMESTAMPED_EVENT",

            "origin_gate":
                "acquisition.announced_on <= origin",

            "database_observation_time_known":
                True,

            "target_pair_mask_required":
                True,

            "notes":
                (
                    "Explicit semantic inverse."
                ),
        },
    ]

    relation_df = pd.DataFrame(
        relation_rows
    )

    print(
        relation_df[
            [
                "relation_id",
                "src_type",
                "relation",
                "dst_type",
                "temporal_class",
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # 5. Freeze information placement
    # =========================================================================

    separator("-")

    print(
        "8.1.9.5 — INFORMATION PLACEMENT"
    )

    separator("-")

    placement_rows = [
        {
            "information":
                "startup_description",

            "placement":
                "STARTUP_NODE_FEATURE",

            "origin_specific":
                False,

            "graph_relation":
                False,

            "notes":
                "Matched-snapshot raw text -> shared encoder.",
        },

        {
            "information":
                "startup_categories",

            "placement":
                "STARTUP_NODE_FEATURE",

            "origin_specific":
                False,

            "graph_relation":
                False,

            "notes":
                "Fine-grained multi-label industry feature.",
        },

        {
            "information":
                "startup_category_groups",

            "placement":
                "STARTUP_NODE_FEATURE",

            "origin_specific":
                False,

            "graph_relation":
                False,

            "notes":
                "Coarse industry hierarchy.",
        },

        {
            "information":
                "startup_country",

            "placement":
                "STARTUP_NODE_FEATURE",

            "origin_specific":
                False,

            "graph_relation":
                False,

            "notes":
                "Matched-snapshot geography caveat.",
        },

        {
            "information":
                "startup_region_location_groups",

            "placement":
                "STARTUP_NODE_FEATURE",

            "origin_specific":
                False,

            "graph_relation":
                False,

            "notes":
                "Hierarchical geography; avoid duplicate city block.",
        },

        {
            "information":
                "startup_company_type",

            "placement":
                "STARTUP_NODE_FEATURE",

            "origin_specific":
                False,

            "graph_relation":
                False,

            "notes":
                "Low-cardinality organizational form.",
        },

        {
            "information":
                "startup_age_at_origin",

            "placement":
                "STARTUP_NODE_FEATURE",

            "origin_specific":
                True,

            "graph_relation":
                False,

            "notes":
                "origin - founded_on.",
        },

        {
            "information":
                "startup_founder_count_resolved",

            "placement":
                "STARTUP_NODE_FEATURE",

            "origin_specific":
                True,

            "graph_relation":
                False,

            "notes":
                (
                    "Derived from eligible resolved founder edges; "
                    "missingness handling frozen later."
                ),
        },

        {
            "information":
                "explicit_startup_founder_person",

            "placement":
                "HETEROGENEOUS_EDGE",

            "origin_specific":
                True,

            "graph_relation":
                True,

            "notes":
                "startup|founded_by|person + inverse.",
        },

        {
            "information":
                "acquisition_relation",

            "placement":
                "HETEROGENEOUS_EDGE",

            "origin_specific":
                True,

            "graph_relation":
                True,

            "notes":
                (
                    "Reuse origin-safe acquired/acquired_by edges."
                ),
        },

        {
            "information":
                "founder_prior_startup_count",

            "placement":
                "PERSON_NODE_FEATURE",

            "origin_specific":
                True,

            "graph_relation":
                False,

            "notes":
                (
                    "Count non-focal companies founded by Person "
                    "with founded_on < origin. "
                    "May use all local Company rows for aggregation "
                    "without adding those companies as graph nodes."
                ),
        },

        {
            "information":
                "founder_prior_sector_profile",

            "placement":
                "PERSON_NODE_FEATURE",

            "origin_specific":
                True,

            "graph_relation":
                False,

            "notes":
                (
                    "Pool categories/category_groups from "
                    "non-focal founder companies with "
                    "founded_on < origin; labels remain "
                    "matched-snapshot semantics."
                ),
        },
    ]

    placement_df = pd.DataFrame(
        placement_rows
    )

    print(
        placement_df[
            [
                "information",
                "placement",
                "origin_specific",
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # 6. Freeze graph-universe boundaries
    # =========================================================================

    separator("-")

    print(
        "8.1.9.6 — GRAPH-UNIVERSE BOUNDARIES"
    )

    separator("-")

    boundaries = {
        "candidate_startup_universe":
            "FROZEN_311589_STARTUP_ROLE_NODES",

        "investor_role_universe":
            "FROZEN_165975_INVESTOR_ROLE_NODES",

        "person_node_universe":
            (
                "EXACT_RESOLVED_FOUNDERS_OF_FROZEN_STARTUP_ROLE_NODES"
            ),

        "external_company_nodes_added":
            False,

        "all_companies_allowed_for_feature_aggregation":
            True,

        "all_companies_aggregation_scope":
            (
                "Founder pre-origin venture/sector feature "
                "construction only."
            ),

        "candidate_universe_expanded":
            False,

        "school_nodes_in_primary_graph":
            False,

        "event_nodes_in_primary_graph":
            False,

        "hub_nodes_in_primary_graph":
            False,

        "contact_nodes_in_primary_graph":
            False,

        "funding_round_nodes_in_primary_graph":
            False,
    }

    for key, value in boundaries.items():

        print(
            f"{key:<48} {value}"
        )

    # =========================================================================
    # 7. Founder-history temporal semantics
    # =========================================================================

    separator("-")

    print(
        "8.1.9.7 — FOUNDER HISTORY SEMANTICS"
    )

    separator("-")

    founder_history_policy = {
        "focal_startup_excluded":
            True,

        "prior_company_condition":
            "prior_company.founded_on < prediction_origin",

        "must_predate_focal_startup_founding":
            False,

        "interpretation":
            (
                "Pre-origin non-focal entrepreneurial history, "
                "not necessarily ventures founded before the focal "
                "startup itself."
            ),

        "investment_history_of_focal_startup_used":
            False,

        "raw_people_num_founded_organizations_used":
            False,

        "external_company_identity_embedding_used":
            False,

        "prior_company_categories_temporal_class":
            "MATCHED_SNAPSHOT_UNVERSIONED",

        "database_founder_observation_time_known":
            False,
    }

    for key, value in (
        founder_history_policy.items()
    ):

        print(
            f"{key:<48} {value}"
        )

    # =========================================================================
    # 8. Structural graph variants
    # =========================================================================

    separator("-")

    print(
        "8.1.9.8 — FROZEN STRUCTURAL VARIANTS"
    )

    separator("-")

    variant_rows = [
        {
            "variant":
                "G_PRIMARY",

            "relations":
                "founded_by | founded | acquired | acquired_by",

            "primary":
                True,

            "purpose":
                (
                    "Primary explicit-founder heterogeneous graph "
                    "for M1/M2."
                ),
        },

        {
            "variant":
                "G_ACQUISITION_ONLY",

            "relations":
                "acquired | acquired_by",

            "primary":
                False,

            "purpose":
                "Founder-structure ablation.",
        },

        {
            "variant":
                "G_FOUNDER_ONLY",

            "relations":
                "founded_by | founded",

            "primary":
                False,

            "purpose":
                "Acquisition-relation ablation.",
        },

        {
            "variant":
                "G_LEGACY_PROJECTED_FOUNDER",

            "relations":
                (
                    "shared_founder | acquired | acquired_by"
                ),

            "primary":
                False,

            "purpose":
                (
                    "Comparator against the existing Phase-8 "
                    "projected shared-founder graph."
                ),
        },

        {
            "variant":
                "G_TYPED_ACQUISITION_DEV_ABLATION",

            "relations":
                (
                    "founded_by | founded | "
                    "typed_acquired | typed_acquired_by"
                ),

            "primary":
                False,

            "purpose":
                (
                    "Development-only acquisition-type ablation; "
                    "not part of the primary graph contract."
                ),
        },
    ]

    variant_df = pd.DataFrame(
        variant_rows
    )

    print(
        variant_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # 9. Explicit anti-duplication / anti-leakage rules
    # =========================================================================

    separator("-")

    print(
        "8.1.9.9 — STRUCTURAL SAFETY RULES"
    )

    separator("-")

    safety_rules = {
        "shared_founder_in_primary_graph":
            False,

        "reason_shared_founder_excluded":
            (
                "Explicit Person paths already encode founder "
                "connectivity; including the projection would "
                "double-count the same founder evidence."
            ),

        "target_pair_direct_edge_masking":
            "REUSE_FROZEN_PHASE8_POLICY",

        "post_origin_acquisition_edges":
            False,

        "post_origin_founder_edges":
            False,

        "startup_investment_edges_in_structural_branch":
            False,

        "focal_startup_prior_investment_features":
            False,

        "startup_id_embedding":
            False,

        "person_id_embedding":
            False,

        "phase4_startup_doc2vec_vector":
            False,

        "current_person_attributes":
            False,

        "school_relation_primary":
            False,
    }

    for key, value in safety_rules.items():

        print(
            f"{key:<48} {value}"
        )

    # =========================================================================
    # 10. Persist graph schema
    # =========================================================================

    separator("-")

    print(
        "8.1.9.10 — PERSIST GRAPH SCHEMA"
    )

    separator("-")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    node_df.to_csv(
        NODE_SCHEMA_OUTPUT,
        index=False,
    )

    relation_df.to_csv(
        RELATION_SCHEMA_OUTPUT,
        index=False,
    )

    placement_df.to_csv(
        PLACEMENT_OUTPUT,
        index=False,
    )

    variant_df.to_csv(
        VARIANT_OUTPUT,
        index=False,
    )

    phase3_config_hash = sha256_file(
        PHASE3_GRAPH_CONFIG
    )

    input_contract_hash = sha256_file(
        INPUT_CONTRACT
    )

    graph_schema = {
        "schema_version":
            "PHASE_8_1_9_HETEROGENEOUS_GRAPH_SCHEMA_V1",

        "status":
            "FROZEN",

        "scientific_role":
            "PRE_MATERIALIZATION_HETEROGENEOUS_GRAPH_SCHEMA",

        "source_input_contract":
            str(INPUT_CONTRACT),

        "source_input_contract_sha256":
            input_contract_hash,

        "baseline_phase3_graph_config":
            str(PHASE3_GRAPH_CONFIG),

        "baseline_phase3_graph_config_sha256":
            phase3_config_hash,

        "node_types":
            node_df.to_dict(
                orient="records"
            ),

        "primary_relations":
            relation_df.to_dict(
                orient="records"
            ),

        "information_placement":
            placement_df.to_dict(
                orient="records"
            ),

        "graph_universe_boundaries":
            boundaries,

        "founder_history_policy":
            founder_history_policy,

        "graph_variants":
            variant_df.to_dict(
                orient="records"
            ),

        "structural_safety_rules":
            safety_rules,

        "candidate_universe_expanded":
            False,

        "feature_discovery_reopened":
            False,

        "graph_materialized":
            False,

        "node_initialization_frozen":
            False,

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
                "Phase 8.1.10 — freeze identity-free startup/Person "
                "node initialization and feature encoding policy."
            ),
    }

    write_json(
        SCHEMA_OUTPUT,
        graph_schema,
    )

    manifest = {
        "phase":
            "8.1.9",

        "status":
            "FROZEN",

        "outputs": [
            str(NODE_SCHEMA_OUTPUT),
            str(RELATION_SCHEMA_OUTPUT),
            str(PLACEMENT_OUTPUT),
            str(VARIANT_OUTPUT),
            str(SCHEMA_OUTPUT),
        ],

        "feature_discovery_reopened":
            False,

        "graph_materialized":
            False,

        "node_initialization_frozen":
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
        NODE_SCHEMA_OUTPUT,
        RELATION_SCHEMA_OUTPUT,
        PLACEMENT_OUTPUT,
        VARIANT_OUTPUT,
        SCHEMA_OUTPUT,
        MANIFEST_OUTPUT,
    ]

    write_json(
        HASH_OUTPUT,
        {
            str(path):
                sha256_file(path)
            for path in outputs
        },
    )

    separator()

    print(
        "PHASE 8.1.9 HETEROGENEOUS GRAPH SCHEMA FROZEN"
    )

    separator()

    print()

    print(
        "NODE TYPES:                      "
        "investor / startup / person"
    )

    print(
        "PRIMARY RELATIONS:               "
        "founded_by / founded / acquired / acquired_by"
    )

    print(
        "SHARED_FOUNDER IN PRIMARY:       NO"
    )

    print(
        "EXTERNAL COMPANY NODES:          NO"
    )

    print(
        "CANDIDATE UNIVERSE EXPANDED:     NO"
    )

    print(
        "FOUNDER HISTORY PLACEMENT:       "
        "PERSON NODE FEATURES"
    )

    print()

    print(
        "FEATURE DISCOVERY REOPENED:      NO"
    )

    print(
        "GRAPH MATERIALIZED:              NO"
    )

    print(
        "NODE INITIALIZATION FROZEN:      NO"
    )

    print(
        "ARCHITECTURE SELECTED:           NO"
    )

    print(
        "MODEL TRAINED:                   NO"
    )

    print(
        "APRIL ANALYZED:                  NO"
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Phase 8.1.10 — identity-free node initialization "
        "and feature-encoding policy."
    )


if __name__ == "__main__":
    main()