from pathlib import Path
import hashlib
import json

import pandas as pd


# =============================================================================
# PHASE 8.1.6G — DATA FOUNDATIONS SYNTHESIS V1
# =============================================================================
#
# Scientific role:
#
#   PRE-ARCHITECTURE DATA-ROLE SYNTHESIS
#
# PURPOSE
# -------
#
# Consolidate the Phase-8.1 data-foundation evidence into an explicit
# research-role ledger BEFORE:
#
#   - input-contract amendment
#   - graph-schema modification
#   - architecture selection
#   - model training
#
# This script DOES NOT select the final model.
# This script DOES NOT approve a new relation.
# April remains untouched.
#
# =============================================================================


BASE = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation"
)

PHASE815 = (
    BASE
    / "phase_8_1_5_relation_semantics"
    / "phase_8_1_5_relation_semantics_summary_V1.json"
)

PHASE816A = (
    BASE
    / "phase_8_1_6_data_foundations"
    / "phase_8_1_6a_entity_foundation_eda"
    / "phase_8_1_6a_eda_summary_V1.json"
)

PHASE816B = (
    BASE
    / "phase_8_1_6_data_foundations"
    / "phase_8_1_6b_startup_attribute_eda"
    / "phase_8_1_6b_eda_summary_V1.json"
)

PHASE816C = (
    BASE
    / "phase_8_1_6_data_foundations"
    / "phase_8_1_6c_founder_people_school_eda"
    / "phase_8_1_6c_eda_summary_V1.json"
)

PHASE816D = (
    BASE
    / "phase_8_1_6_data_foundations"
    / "phase_8_1_6d_explicit_founder_incremental_coverage"
    / "phase_8_1_6d_summary_V2.json"
)

PHASE816E = (
    BASE
    / "phase_8_1_6_data_foundations"
    / "phase_8_1_6e_founder_person_temporal_provenance"
    / "phase_8_1_6e_summary_V2.json"
)

PHASE816F = (
    BASE
    / "phase_8_1_6_data_foundations"
    / "phase_8_1_6f_founder_node_structural_utility"
    / "phase_8_1_6f_summary_V1.json"
)

OUTPUT_DIR = (
    BASE
    / "phase_8_1_6_data_foundations"
    / "phase_8_1_6g_data_foundations_synthesis"
)

ROLE_MATRIX_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6g_research_role_matrix_V1.csv"
)

DECISION_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6g_data_channel_decision_ledger_V1.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6g_summary_V1.json"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6g_manifest_V1.json"
)

HASH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6g_artifact_sha256_V1.json"
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
        "PHASE 8.1.6G — "
        "DATA FOUNDATIONS SYNTHESIS V1"
    )

    separator()

    print()

    print(
        "Scientific role:             "
        "PRE-ARCHITECTURE DATA-ROLE SYNTHESIS"
    )

    print("New relation approved:       NO")
    print("Input contract amended:      NO")
    print("Graph modified:              NO")
    print("Architecture selected:       NO")
    print("Model trained:               NO")
    print("April analyzed:              NO")

    # =========================================================================
    # 1. Evidence gates
    # =========================================================================

    separator("-")

    print(
        "8.1.6G.1 — EVIDENCE GATES"
    )

    separator("-")

    required = [
        PHASE815,
        PHASE816A,
        PHASE816B,
        PHASE816C,
        PHASE816D,
        PHASE816E,
        PHASE816F,
    ]

    for path in required:

        require(
            path.exists(),
            f"Missing evidence artifact: {path}",
        )

        print(
            f"FOUND {path}"
        )

    summaries = {
        "8.1.6B": read_json(PHASE816B),
        "8.1.6C": read_json(PHASE816C),
        "8.1.6D": read_json(PHASE816D),
        "8.1.6E": read_json(PHASE816E),
        "8.1.6F": read_json(PHASE816F),
    }

    for phase, obj in summaries.items():

        require(
            obj.get("status")
            ==
            "COMPLETE",
            f"{phase} is not COMPLETE.",
        )

    require(
        summaries["8.1.6D"].get(
            "April_analyzed"
        )
        is False,
        "April unexpectedly analyzed in 8.1.6D.",
    )

    require(
        summaries["8.1.6E"].get(
            "April_analyzed"
        )
        is False,
        "April unexpectedly analyzed in 8.1.6E.",
    )

    require(
        summaries["8.1.6F"].get(
            "April_analyzed"
        )
        is False,
        "April unexpectedly analyzed in 8.1.6F.",
    )

    print()

    print(
        "All Data Foundations evidence gates: PASS"
    )

    # =========================================================================
    # 2. Key founder evidence
    # =========================================================================

    separator("-")

    print(
        "8.1.6G.2 — KEY FOUNDER EVIDENCE"
    )

    separator("-")

    founder_mean = float(
        summaries[
            "8.1.6F"
        ][
            "mean_founder_coverage_pct"
        ]
    )

    shared_mean = float(
        summaries[
            "8.1.6F"
        ][
            "mean_founder_degree_ge2_pct"
        ]
    )

    leaf_mean = float(
        summaries[
            "8.1.6F"
        ][
            "mean_founder_leaf_only_pct"
        ]
    )

    description_mean = float(
        summaries[
            "8.1.6F"
        ][
            "mean_founder_description_pct"
        ]
    )

    school_mean = float(
        summaries[
            "8.1.6F"
        ][
            "mean_founder_school_field_pct"
        ]
    )

    isolation_recovered = float(
        summaries[
            "8.1.6D"
        ][
            "mean_fraction_current_isolation_recovered_pct"
        ]
    )

    semantic_retention = float(
        summaries[
            "8.1.6E"
        ][
            "mean_semantic_retention_pct"
        ]
    )

    leaf_fraction_given_founder = (
        100.0
        * leaf_mean
        / founder_mean
    )

    print(
        f"Mean primary founder coverage:       "
        f"{founder_mean:.4f}%"
    )

    print(
        f"Founder degree>=2 coverage:          "
        f"{shared_mean:.4f}%"
    )

    print(
        f"Founder leaf-only coverage:          "
        f"{leaf_mean:.4f}%"
    )

    print(
        f"Leaf-only among founder cases:       "
        f"{leaf_fraction_given_founder:.4f}%"
    )

    print(
        f"Current isolation recovered by raw "
        f"Person context: {isolation_recovered:.4f}%"
    )

    print(
        f"Semantic temporal retention:         "
        f"{semantic_retention:.4f}%"
    )

    print(
        f"Founder description coverage:        "
        f"{description_mean:.4f}%"
    )

    print(
        f"Founder school-field coverage:       "
        f"{school_mean:.4f}%"
    )

    # =========================================================================
    # 3. Dataset-level research-role matrix
    # =========================================================================

    separator("-")

    print(
        "8.1.6G.3 — DATASET RESEARCH-ROLE MATRIX"
    )

    separator("-")

    role_rows = [
        {
            "dataset": "companies",
            "primary_research_role":
                "STARTUP_SIDE_INDUCTIVE_INFORMATION",

            "keep_for_research": True,

            "main_evidence":
                (
                    "Description/categories near-universal in "
                    "Startup-role universe; structured geography "
                    "and company semantics also highly covered."
                ),

            "temporal_status":
                "MATCHED_SNAPSHOT_UNVERSIONED",

            "decision":
                "CORE_DATASET",
        },

        {
            "dataset": "investors",
            "primary_research_role":
                "PRESERVE_ITRS_INVESTOR_REPRESENTATION",

            "keep_for_research": True,

            "main_evidence":
                (
                    "Needed for investor identity and trend-aware "
                    "historical preference modeling."
                ),

            "temporal_status":
                "BASELINE_BRANCH_PRESERVED",

            "decision":
                "CORE_DATASET",
        },

        {
            "dataset": "funding",
            "primary_research_role":
                "TEMPORAL_INVESTMENT_INTERACTIONS",

            "keep_for_research": True,

            "main_evidence":
                (
                    "Canonical timestamped investor-startup "
                    "interaction history and evaluation labels."
                ),

            "temporal_status":
                "TIMESTAMPED",

            "decision":
                "CORE_DATASET",
        },

        {
            "dataset": "acquisitions",
            "primary_research_role":
                "NON_TARGET_HETEROGENEOUS_RELATION",

            "keep_for_research": True,

            "main_evidence":
                (
                    "Timestamp-auditable pre-origin acquisition "
                    "relation already used in strict structural track."
                ),

            "temporal_status":
                "TIMESTAMPED_PRE_ORIGIN",

            "decision":
                "CORE_RELATION",
        },

        {
            "dataset": "people",
            "primary_research_role":
                "FOUNDER_IDENTITY_AND_TYPED_CONTEXT",

            "keep_for_research": True,

            "main_evidence":
                (
                    "Founder URL->Person resolution is essentially "
                    "exact; mean primary founder coverage "
                    f"{founder_mean:.2f}%, but "
                    f"{leaf_fraction_given_founder:.2f}% of "
                    "founder-covered cases are leaf-only."
                ),

            "temporal_status":
                (
                    "FOUNDER_EDGE_SEMANTICALLY_ORIGIN_GATABLE; "
                    "PERSON_ATTRIBUTES_SNAPSHOT_UNVERSIONED"
                ),

            "decision":
                "CANDIDATE_NOT_YET_APPROVED",
        },

        {
            "dataset": "schools",
            "primary_research_role":
                "POTENTIAL_SECOND_HOP_FOUNDER_CONTEXT",

            "keep_for_research": True,

            "main_evidence":
                (
                    "Substantial founder-school surface, but "
                    "education dates unavailable and name ambiguity "
                    "exists."
                ),

            "temporal_status":
                "SNAPSHOT_UNVERSIONED_NO_EDUCATION_DATES",

            "decision":
                "SNAPSHOT_SENSITIVITY_ONLY",
        },

        {
            "dataset": "hubs",
            "primary_research_role":
                "NONE_FOR_PRIMARY_MODEL",

            "keep_for_research": False,

            "main_evidence":
                (
                    "Company hub_tags populated in only about "
                    "0.10% of raw Companies."
                ),

            "temporal_status":
                "SNAPSHOT_UNVERSIONED",

            "decision":
                "EXCLUDE_PRIMARY",
        },

        {
            "dataset": "contacts",
            "primary_research_role":
                "NONE_FOR_PRIMARY_MODEL",

            "keep_for_research": False,

            "main_evidence":
                (
                    "organization_uuid relation surface has "
                    "zero populated rows."
                ),

            "temporal_status":
                "NO_USABLE_ORGANIZATION_BRIDGE",

            "decision":
                "EXCLUDE_PRIMARY",
        },

        {
            "dataset": "events",
            "primary_research_role":
                "NONE_FOR_PRIMARY_MODEL",

            "keep_for_research": False,

            "main_evidence":
                (
                    "No explicit participant relation identified "
                    "in available schema."
                ),

            "temporal_status":
                "NO_RESOLVABLE_ENTITY_BRIDGE",

            "decision":
                "EXCLUDE_PRIMARY",
        },
    ]

    role_df = pd.DataFrame(
        role_rows
    )

    print(
        role_df[
            [
                "dataset",
                "primary_research_role",
                "decision",
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # 4. Information-channel decision ledger
    # =========================================================================

    separator("-")

    print(
        "8.1.6G.4 — INFORMATION-CHANNEL "
        "DECISION LEDGER"
    )

    separator("-")

    decision_rows = [
        {
            "channel":
                "startup_description_text",

            "source":
                "companies",

            "research_role":
                "INDUCTIVE_STARTUP_BACKBONE",

            "provenance":
                "MATCHED_SNAPSHOT_UNVERSIONED",

            "current_status":
                "ALLOWED_FROZEN_8_1_2",

            "requires_contract_amendment":
                False,

            "primary_model_allowed_now":
                True,
        },

        {
            "channel":
                "startup_categories",

            "source":
                "companies",

            "research_role":
                "INDUCTIVE_STARTUP_BACKBONE",

            "provenance":
                "MATCHED_SNAPSHOT_UNVERSIONED",

            "current_status":
                "ALLOWED_FROZEN_8_1_2",

            "requires_contract_amendment":
                False,

            "primary_model_allowed_now":
                True,
        },

        {
            "channel":
                "startup_geography",

            "source":
                "companies",

            "research_role":
                "POTENTIAL_STRUCTURED_ATTRIBUTE",

            "provenance":
                "MATCHED_SNAPSHOT_UNVERSIONED",

            "current_status":
                "EDA_SUPPORTED_EXTENSION",

            "requires_contract_amendment":
                True,

            "primary_model_allowed_now":
                False,
        },

        {
            "channel":
                "startup_company_type",

            "source":
                "companies",

            "research_role":
                "POTENTIAL_STRUCTURED_ATTRIBUTE",

            "provenance":
                "MATCHED_SNAPSHOT_UNVERSIONED",

            "current_status":
                "EDA_SUPPORTED_EXTENSION",

            "requires_contract_amendment":
                True,

            "primary_model_allowed_now":
                False,
        },

        {
            "channel":
                "startup_category_groups",

            "source":
                "companies",

            "research_role":
                "POTENTIAL_STRUCTURED_ATTRIBUTE",

            "provenance":
                "MATCHED_SNAPSHOT_UNVERSIONED",

            "current_status":
                "EDA_SUPPORTED_EXTENSION",

            "requires_contract_amendment":
                True,

            "primary_model_allowed_now":
                False,
        },

        {
            "channel":
                "timestamped_acquisition_relation",

            "source":
                "acquisitions",

            "research_role":
                "HETEROGENEOUS_STRUCTURAL_CONTEXT",

            "provenance":
                "TIMESTAMPED_PRE_ORIGIN",

            "current_status":
                "ALLOWED_FROZEN_8_1_2",

            "requires_contract_amendment":
                False,

            "primary_model_allowed_now":
                True,
        },

        {
            "channel":
                "shared_founder_projection",

            "source":
                "companies + people",

            "research_role":
                "HETEROGENEOUS_STRUCTURAL_CONTEXT",

            "provenance":
                "SEMANTIC_FOUNDED_ON_GATE",

            "current_status":
                "EXISTING_PHASE8_TRACK",

            "requires_contract_amendment":
                False,

            "primary_model_allowed_now":
                True,
        },

        {
            "channel":
                "explicit_startup_founder_person",

            "source":
                "companies + people",

            "research_role":
                "CANDIDATE_TYPED_HETEROGENEOUS_CONTEXT",

            "provenance":
                (
                    "SEMANTIC_FOUNDED_ON_GATE_"
                    "DATABASE_OBSERVATION_UNKNOWN"
                ),

            "current_status":
                "CANDIDATE_NOT_APPROVED",

            "requires_contract_amendment":
                True,

            "primary_model_allowed_now":
                False,
        },

        {
            "channel":
                "person_description",

            "source":
                "people",

            "research_role":
                "POTENTIAL_PERSON_NODE_FEATURE",

            "provenance":
                "CURRENT_SNAPSHOT_UNVERSIONED",

            "current_status":
                "SNAPSHOT_ONLY",

            "requires_contract_amendment":
                True,

            "primary_model_allowed_now":
                False,
        },

        {
            "channel":
                "person_geography",

            "source":
                "people",

            "research_role":
                "POTENTIAL_PERSON_NODE_FEATURE",

            "provenance":
                "CURRENT_SNAPSHOT_UNVERSIONED",

            "current_status":
                "SNAPSHOT_ONLY",

            "requires_contract_amendment":
                True,

            "primary_model_allowed_now":
                False,
        },

        {
            "channel":
                "person_current_organizations",

            "source":
                "people",

            "research_role":
                "NONE",

            "provenance":
                "CURRENT_SNAPSHOT",

            "current_status":
                "HIGH_LEAKAGE_RISK_PROHIBIT",

            "requires_contract_amendment":
                False,

            "primary_model_allowed_now":
                False,
        },

        {
            "channel":
                "person_school_relation",

            "source":
                "people + schools",

            "research_role":
                "POTENTIAL_SECOND_HOP_CONTEXT",

            "provenance":
                "SNAPSHOT_UNVERSIONED_NO_EDUCATION_DATES",

            "current_status":
                "SNAPSHOT_SENSITIVITY_ONLY",

            "requires_contract_amendment":
                True,

            "primary_model_allowed_now":
                False,
        },

        {
            "channel":
                "startup_current_funding_state",

            "source":
                "companies + funding",

            "research_role":
                "NONE_FOR_COLD_START_ENCODER",

            "provenance":
                "CURRENT_SNAPSHOT_OUTCOME_DERIVED",

            "current_status":
                "HIGH_LEAKAGE_RISK_PROHIBIT",

            "requires_contract_amendment":
                False,

            "primary_model_allowed_now":
                False,
        },

        {
            "channel":
                "startup_rank_semrush_operating_outcome_fields",

            "source":
                "companies",

            "research_role":
                "NONE",

            "provenance":
                "CURRENT_SNAPSHOT",

            "current_status":
                "HIGH_LEAKAGE_RISK_PROHIBIT",

            "requires_contract_amendment":
                False,

            "primary_model_allowed_now":
                False,
        },
    ]

    decision_df = pd.DataFrame(
        decision_rows
    )

    print(
        decision_df[
            [
                "channel",
                "current_status",
                "primary_model_allowed_now",
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # 5. Persist synthesis
    # =========================================================================

    separator("-")

    print(
        "8.1.6G.5 — PERSIST SYNTHESIS"
    )

    separator("-")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    role_df.to_csv(
        ROLE_MATRIX_OUTPUT,
        index=False,
    )

    decision_df.to_csv(
        DECISION_OUTPUT,
        index=False,
    )

    summary = {
        "schema_version":
            "PHASE_8_1_6G_DATA_FOUNDATIONS_SYNTHESIS_V1",

        "status":
            "COMPLETE",

        "scientific_role":
            "PRE_ARCHITECTURE_DATA_ROLE_SYNTHESIS",

        "core_datasets": [
            "companies",
            "investors",
            "funding",
            "acquisitions",
        ],

        "candidate_extension_datasets": [
            "people",
            "schools",
        ],

        "excluded_primary_datasets": [
            "hubs",
            "contacts",
            "events",
        ],

        "founder_evidence": {
            "mean_primary_founder_coverage_pct":
                founder_mean,

            "mean_primary_founder_degree_ge2_pct":
                shared_mean,

            "mean_primary_founder_leaf_only_pct":
                leaf_mean,

            "leaf_only_fraction_given_founder_pct":
                leaf_fraction_given_founder,

            "mean_isolation_recovered_by_raw_person_pct":
                isolation_recovered,

            "semantic_temporal_retention_pct":
                semantic_retention,
        },

        "key_scientific_conclusion":
            (
                "Explicit founder-Person identity substantially "
                "increases available founder context, but most "
                "founder-covered cold-start cases are leaf-only. "
                "Therefore explicit Person nodes should not be "
                "treated as automatically informative without "
                "careful feature/relation design."
            ),

        "new_relation_approved":
            False,

        "input_contract_amended":
            False,

        "graph_modified":
            False,

        "architecture_selected":
            False,

        "model_trained":
            False,

        "April_analyzed":
            False,

        "next_step":
            (
                "Decide whether a formal Phase-8.1 input-contract "
                "amendment is scientifically justified before "
                "freezing the inductive graph schema."
            ),
    }

    write_json(
        SUMMARY_OUTPUT,
        summary,
    )

    manifest = {
        "phase":
            "8.1.6G",

        "status":
            "COMPLETE",

        "outputs": [
            str(ROLE_MATRIX_OUTPUT),
            str(DECISION_OUTPUT),
            str(SUMMARY_OUTPUT),
        ],

        "prohibited_operations": [
            "new_relation_approval",
            "input_contract_amendment",
            "graph_modification",
            "architecture_selection",
            "training",
            "inference",
            "ranking_scoring",
            "April_analysis",
        ],
    }

    write_json(
        MANIFEST_OUTPUT,
        manifest,
    )

    outputs = [
        ROLE_MATRIX_OUTPUT,
        DECISION_OUTPUT,
        SUMMARY_OUTPUT,
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
        "PHASE 8.1.6G DATA FOUNDATIONS "
        "SYNTHESIS COMPLETE"
    )

    separator()

    print()

    print(
        "ALL 9 DATASET ROLES SYNTHESIZED: YES"
    )

    print(
        "INPUT CONTRACT AMENDED:          NO"
    )

    print(
        "NEW RELATION APPROVED:           NO"
    )

    print(
        "GRAPH MODIFIED:                  NO"
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
        "Next:"
    )

    print(
        "Scientific decision on whether the "
        "Phase-8.1 input contract should be amended "
        "before graph-schema freeze."
    )


if __name__ == "__main__":
    main()