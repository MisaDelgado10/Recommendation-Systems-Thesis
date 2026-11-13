from pathlib import Path
import hashlib
import json

import pandas as pd


# =============================================================================
# PHASE 8.1.7 — FINAL INDUCTIVE INPUT CONTRACT AMENDMENT V2
# =============================================================================
#
# Scientific role:
#
#   FINAL PRE-ARCHITECTURE FEATURE / INFORMATION CONTRACT
#
# PURPOSE
# -------
#
# Supersede, but NEVER overwrite, Phase 8.1.7 V1 after the comprehensive
# Phase 8.1.8 EDA and feature-governance review.
#
# This script freezes WHAT information may be used.
#
# It does NOT yet freeze:
#
#   - exact neural architecture
#   - embedding dimensions
#   - graph message-passing design
#   - normalization strategy
#   - optimizer / training seed
#   - final heterogeneous graph implementation
#
# IMPORTANT
# ---------
#
# No performance-based feature selection has occurred.
# M1/M2 have not been trained.
# April confirmatory data remains untouched.
#
# =============================================================================


BASE = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation"
)

PHASE817_V1 = (
    BASE
    / "phase_8_1_7_input_contract_amendment"
    / "phase_8_1_7_contract_amendment_V1.json"
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

EDA_DIR = Path(
    "analysis/phase_8_1_8_comprehensive_eda"
)

EDA_REPORT = (
    EDA_DIR
    / "README.md"
)

GOVERNANCE_MATRIX = (
    EDA_DIR
    / "feature_governance_matrix.csv"
)

DATASET_SUMMARY = (
    EDA_DIR
    / "dataset_summary.csv"
)


OUTPUT_DIR = (
    BASE
    / "phase_8_1_7_input_contract_amendment"
)

FINAL_CHANNELS_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_7_final_information_channels_V2.csv"
)

MODEL_FAMILY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_7_model_family_contract_V2.csv"
)

PROHIBITIONS_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_7_prohibited_information_V2.csv"
)

AMENDMENT_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_7_contract_amendment_V2.json"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_7_manifest_V2.json"
)

HASH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_7_artifact_sha256_V2.json"
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

            chunk = f.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def main():

    separator()

    print(
        "PHASE 8.1.7 — "
        "FINAL INDUCTIVE INPUT CONTRACT AMENDMENT V2"
    )

    separator()

    print()

    print(
        "Scientific role:             "
        "FINAL PRE-ARCHITECTURE INFORMATION CONTRACT"
    )

    print(
        "V1 overwritten:              NO"
    )

    print(
        "V1 superseded by V2:         YES"
    )

    print(
        "Architecture selected:       NO"
    )

    print(
        "Graph schema frozen:          NO"
    )

    print(
        "Model trained:               NO"
    )

    print(
        "April analyzed:              NO"
    )

    # =========================================================================
    # 1. Evidence gates
    # =========================================================================

    separator("-")

    print(
        "8.1.7.V2.1 — EVIDENCE GATES"
    )

    separator("-")

    required_paths = [
        PHASE817_V1,
        PHASE816D,
        PHASE816E,
        PHASE816F,
        EDA_REPORT,
        GOVERNANCE_MATRIX,
        DATASET_SUMMARY,
    ]

    for path in required_paths:

        require(
            path.exists(),
            f"Missing prerequisite artifact: {path}",
        )

        print(
            f"FOUND {path}"
        )

    v1 = read_json(
        PHASE817_V1
    )

    d_summary = read_json(
        PHASE816D
    )

    e_summary = read_json(
        PHASE816E
    )

    f_summary = read_json(
        PHASE816F
    )

    require(
        v1.get("status")
        ==
        "FROZEN",
        "Phase 8.1.7 V1 is not frozen.",
    )

    require(
        v1.get("April_analyzed")
        is False,
        "Phase 8.1.7 V1 unexpectedly analyzed April.",
    )

    for phase_name, summary in [
        ("8.1.6D", d_summary),
        ("8.1.6E", e_summary),
        ("8.1.6F", f_summary),
    ]:

        require(
            summary.get("status")
            ==
            "COMPLETE",
            f"{phase_name} is not COMPLETE.",
        )

        require(
            summary.get("April_analyzed")
            is False,
            f"{phase_name} unexpectedly analyzed April.",
        )

    governance = pd.read_csv(
        GOVERNANCE_MATRIX
    )

    require(
        len(governance) > 0,
        "Feature governance matrix is empty.",
    )

    required_columns = {
        "channel",
        "source",
        "temporal_class",
        "research_value",
        "recommendation",
        "primary_model",
        "ablation_or_sensitivity",
        "rationale",
    }

    require(
        required_columns.issubset(
            governance.columns
        ),
        (
            "Feature governance matrix schema drift. "
            f"Observed={list(governance.columns)}"
        ),
    )

    print()

    print(
        "V1 preservation gate:        PASS"
    )

    print(
        "Data Foundations gate:       PASS"
    )

    print(
        "Phase 8.1.8 governance gate: PASS"
    )

    # =========================================================================
    # 2. Quantitative founder evidence
    # =========================================================================

    separator("-")

    print(
        "8.1.7.V2.2 — FOUNDER EVIDENCE"
    )

    separator("-")

    founder_coverage = float(
        f_summary[
            "mean_founder_coverage_pct"
        ]
    )

    leaf_founder_coverage = float(
        f_summary[
            "mean_founder_leaf_only_pct"
        ]
    )

    founder_degree_ge2 = float(
        f_summary[
            "mean_founder_degree_ge2_pct"
        ]
    )

    isolation_recovered = float(
        d_summary[
            "mean_fraction_current_isolation_recovered_pct"
        ]
    )

    semantic_retention = float(
        e_summary[
            "mean_semantic_retention_pct"
        ]
    )

    require(
        abs(
            semantic_retention
            -
            100.0
        )
        <
        1e-6,
        "Founder semantic temporal retention drift.",
    )

    leaf_fraction_given_founder = (
        100.0
        *
        leaf_founder_coverage
        /
        founder_coverage
    )

    print(
        f"Mean founder coverage:               "
        f"{founder_coverage:.4f}%"
    )

    print(
        f"Current isolation recovered:         "
        f"{isolation_recovered:.4f}%"
    )

    print(
        f"Semantic retention:                  "
        f"{semantic_retention:.4f}%"
    )

    print(
        f"Founder degree >=2 coverage:         "
        f"{founder_degree_ge2:.4f}%"
    )

    print(
        f"Founder leaf-only coverage:          "
        f"{leaf_founder_coverage:.4f}%"
    )

    print(
        f"Leaf-only among founder cases:       "
        f"{leaf_fraction_given_founder:.4f}%"
    )

    # =========================================================================
    # 3. Governance reconciliation
    # =========================================================================

    separator("-")

    print(
        "8.1.7.V2.3 — GOVERNANCE RECONCILIATION"
    )

    separator("-")

    governance_by_channel = (
        governance
        .set_index(
            "channel"
        )
    )

    required_governance_channels = {
        "startup_description",
        "startup_categories",
        "startup_category_groups",
        "startup_country",
        "startup_region_location_groups",
        "startup_company_type",
        "startup_age_at_origin",
        "startup_founder_count_resolved",
        "explicit_startup_founder_person",
        "founder_prior_startup_count",
        "founder_prior_sector_profile",
        "acquisition_relation",
        "acquisition_type",
        "founder_school_relation",
        "school_stable_context",
        "founder_prior_funding_experience",
        "founder_prior_exit_experience",
        "investor_type",
        "investor_geography",
    }

    missing_governance = (
        required_governance_channels
        -
        set(
            governance_by_channel.index
        )
    )

    require(
        not missing_governance,
        (
            "Governance channels missing: "
            f"{sorted(missing_governance)}"
        ),
    )

    print(
        f"Required governance channels found: "
        f"{len(required_governance_channels):,} PASS"
    )

    # =========================================================================
    # 4. FINAL PRIMARY INFORMATION CONTRACT
    # =========================================================================

    separator("-")

    print(
        "8.1.7.V2.4 — FINAL PRIMARY "
        "INFORMATION CONTRACT"
    )

    separator("-")

    primary_rows = [
        {
            "contract_id":
                "P01",

            "channel":
                "startup_description",

            "source":
                "companies.description / short_description",

            "class":
                "PRIMARY_STARTUP_CONTENT",

            "temporal_policy":
                "MATCHED_SNAPSHOT_UNVERSIONED",

            "allowed":
                True,

            "notes":
                (
                    "Identity-free text representation. "
                    "Historical wording is not claimed."
                ),
        },

        {
            "contract_id":
                "P02",

            "channel":
                "startup_categories",

            "source":
                "companies.categories",

            "class":
                "PRIMARY_STARTUP_CONTENT",

            "temporal_policy":
                "MATCHED_SNAPSHOT_UNVERSIONED",

            "allowed":
                True,

            "notes":
                (
                    "Fine-grained startup industry semantics."
                ),
        },

        {
            "contract_id":
                "P03",

            "channel":
                "startup_category_groups",

            "source":
                "companies.category_groups",

            "class":
                "PRIMARY_STRUCTURED_STARTUP_ATTRIBUTE",

            "temporal_policy":
                "MATCHED_SNAPSHOT_UNVERSIONED",

            "allowed":
                True,

            "notes":
                (
                    "Coarse industry hierarchy. "
                    "Contribution must be ablated because it is "
                    "partly redundant with categories."
                ),
        },

        {
            "contract_id":
                "P04",

            "channel":
                "startup_country",

            "source":
                "companies.country",

            "class":
                "PRIMARY_STRUCTURED_STARTUP_ATTRIBUTE",

            "temporal_policy":
                "SNAPSHOT_SLOW_MOVING",

            "allowed":
                True,

            "notes":
                (
                    "Geographic compatibility signal. "
                    "HQ historical constancy is not claimed."
                ),
        },

        {
            "contract_id":
                "P05",

            "channel":
                "startup_region_location_groups",

            "source":
                "companies.region + companies.location_groups",

            "class":
                "PRIMARY_STRUCTURED_STARTUP_ATTRIBUTE",

            "temporal_policy":
                "SNAPSHOT_SLOW_MOVING",

            "allowed":
                True,

            "notes":
                (
                    "Hierarchical geography. "
                    "Do not simultaneously add every raw location "
                    "field to avoid geographic overweighting."
                ),
        },

        {
            "contract_id":
                "P06",

            "channel":
                "startup_company_type",

            "source":
                "companies.company_type",

            "class":
                "PRIMARY_STRUCTURED_STARTUP_ATTRIBUTE",

            "temporal_policy":
                "INTRINSIC_LOW_CARDINALITY",

            "allowed":
                True,

            "notes":
                (
                    "Organizational form. "
                    "Use unknown token for missing values."
                ),
        },

        {
            "contract_id":
                "P07",

            "channel":
                "startup_age_at_origin",

            "source":
                "prediction_origin - companies.founded_on",

            "class":
                "PRIMARY_DERIVED_TEMPORAL_ATTRIBUTE",

            "temporal_policy":
                "DERIVED_STRICT_TEMPORAL",

            "allowed":
                True,

            "notes":
                (
                    "Use age at origin, not raw founded_on "
                    "calendar value as an unrestricted predictor."
                ),
        },

        {
            "contract_id":
                "P08",

            "channel":
                "startup_founder_count_resolved",

            "source":
                (
                    "companies.founder_links -> people.link "
                    "exact resolution"
                ),

            "class":
                "PRIMARY_DERIVED_RELATIONAL_ATTRIBUTE",

            "temporal_policy":
                (
                    "SEMANTIC_FOUNDED_ON_GATE_"
                    "DATABASE_OBSERVATION_UNKNOWN"
                ),

            "allowed":
                True,

            "notes":
                (
                    "Derive from resolved founder relations; "
                    "do not use raw snapshot num_founders aggregate."
                ),
        },

        {
            "contract_id":
                "P09",

            "channel":
                "explicit_startup_founder_person",

            "source":
                (
                    "companies.founder_links -> people.link"
                ),

            "class":
                "PRIMARY_HETEROGENEOUS_RELATION",

            "temporal_policy":
                (
                    "startup.founded_on <= prediction_origin; "
                    "database observation time unknown"
                ),

            "allowed":
                True,

            "notes":
                (
                    "No trainable Person identity embedding "
                    "may be used as a memorization shortcut."
                ),
        },

        {
            "contract_id":
                "P10",

            "channel":
                "acquisition_relation",

            "source":
                "acquisitions",

            "class":
                "PRIMARY_HETEROGENEOUS_RELATION",

            "temporal_policy":
                "announced_on <= prediction_origin",

            "allowed":
                True,

            "notes":
                (
                    "Strict timestamped non-target relation."
                ),
        },

        {
            "contract_id":
                "P11",

            "channel":
                "founder_prior_startup_count",

            "source":
                (
                    "founder_links + prior company founded_on"
                ),

            "class":
                "PRIMARY_DERIVED_PRE_ORIGIN_FOUNDER_CONTEXT",

            "temporal_policy":
                (
                    "non-focal founded company "
                    "founded_on < prediction_origin"
                ),

            "allowed":
                True,

            "notes":
                (
                    "Use reconstructed prior ventures, "
                    "never people.num_founded_organizations."
                ),
        },

        {
            "contract_id":
                "P12",

            "channel":
                "founder_prior_sector_profile",

            "source":
                (
                    "founder prior companies + "
                    "categories/category_groups"
                ),

            "class":
                "PRIMARY_DERIVED_PRE_ORIGIN_FOUNDER_CONTEXT",

            "temporal_policy":
                (
                    "prior-company edge gated by founded_on; "
                    "sector labels remain matched-snapshot semantics"
                ),

            "allowed":
                True,

            "notes":
                (
                    "Represents founder domain experience "
                    "without using focal-startup investment history."
                ),
        },
    ]

    primary_df = pd.DataFrame(
        primary_rows
    )

    print(
        primary_df[
            [
                "contract_id",
                "channel",
                "class",
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # 5. ABLATION / SENSITIVITY CONTRACT
    # =========================================================================

    separator("-")

    print(
        "8.1.7.V2.5 — ABLATION / "
        "SENSITIVITY CONTRACT"
    )

    separator("-")

    experiment_rows = [
        {
            "channel":
                "startup_category_groups",

            "experimental_role":
                "PRIMARY_COMPONENT_ABLATION",

            "temporal_class":
                "MATCHED_SNAPSHOT_UNVERSIONED",

            "may_enter_primary":
                True,

            "notes":
                "Ablate to test hierarchy beyond categories.",
        },

        {
            "channel":
                "startup_geography",

            "experimental_role":
                "PRIMARY_COMPONENT_ABLATION",

            "temporal_class":
                "SNAPSHOT_SLOW_MOVING",

            "may_enter_primary":
                True,

            "notes":
                "Country + region/location-group block.",
        },

        {
            "channel":
                "startup_company_type",

            "experimental_role":
                "PRIMARY_COMPONENT_ABLATION",

            "temporal_class":
                "INTRINSIC_LOW_CARDINALITY",

            "may_enter_primary":
                True,

            "notes":
                "Quantify organizational-form contribution.",
        },

        {
            "channel":
                "startup_age_at_origin",

            "experimental_role":
                "PRIMARY_COMPONENT_ABLATION",

            "temporal_class":
                "DERIVED_STRICT_TEMPORAL",

            "may_enter_primary":
                True,

            "notes":
                "Quantify lifecycle-maturity contribution.",
        },

        {
            "channel":
                "explicit_founder_person",

            "experimental_role":
                "PRIMARY_COMPONENT_ABLATION",

            "temporal_class":
                "SEMANTIC_FOUNDED_ON_GATE",

            "may_enter_primary":
                True,

            "notes":
                "Remove explicit Person context.",
        },

        {
            "channel":
                "founder_prior_history",

            "experimental_role":
                "PRIMARY_COMPONENT_ABLATION",

            "temporal_class":
                "DERIVED_PRE_ORIGIN",

            "may_enter_primary":
                True,

            "notes":
                (
                    "Remove serial-founder and prior-sector history."
                ),
        },

        {
            "channel":
                "shared_founder_projection",

            "experimental_role":
                "STRUCTURAL_COMPARATOR",

            "temporal_class":
                "SEMANTIC_FOUNDED_ON_GATE",

            "may_enter_primary":
                False,

            "notes":
                (
                    "Compare legacy projected founder structure "
                    "against explicit Person representation."
                ),
        },

        {
            "channel":
                "acquisition_type",

            "experimental_role":
                "RELATIONAL_ABLATION",

            "temporal_class":
                "STRICT_EVENT_ATTRIBUTE",

            "may_enter_primary":
                False,

            "notes":
                (
                    "Typed vs untyped acquisition relation."
                ),
        },

        {
            "channel":
                "acquisition_recency",

            "experimental_role":
                "RELATIONAL_ABLATION",

            "temporal_class":
                "DERIVED_STRICT_TEMPORAL",

            "may_enter_primary":
                False,

            "notes":
                "Optional time-since-acquisition encoding.",
        },

        {
            "channel":
                "founder_prior_funding_experience",

            "experimental_role":
                "SECONDARY_FOUNDER_HISTORY_ABLATION",

            "temporal_class":
                "DERIVED_STRICT_PRE_ORIGIN",

            "may_enter_primary":
                False,

            "notes":
                (
                    "Funding of prior founder companies only; "
                    "never focal-startup prior investments."
                ),
        },

        {
            "channel":
                "founder_prior_exit_experience",

            "experimental_role":
                "SECONDARY_FOUNDER_HISTORY_ABLATION",

            "temporal_class":
                "DERIVED_STRICT_PRE_ORIGIN",

            "may_enter_primary":
                False,

            "notes":
                (
                    "Acquisitions of prior founder companies "
                    "strictly before origin."
                ),
        },

        {
            "channel":
                "founder_school_relation",

            "experimental_role":
                "SNAPSHOT_BIOGRAPHICAL_SENSITIVITY",

            "temporal_class":
                "SNAPSHOT_UNVERSIONED_NO_EDUCATION_DATES",

            "may_enter_primary":
                False,

            "notes":
                (
                    "Report separately; local export cannot prove "
                    "individual attendance timing."
                ),
        },

        {
            "channel":
                "school_stable_context",

            "experimental_role":
                "SNAPSHOT_BIOGRAPHICAL_SENSITIVITY",

            "temporal_class":
                "SNAPSHOT_SLOW_MOVING",

            "may_enter_primary":
                False,

            "notes":
                (
                    "Only if School sensitivity is run; "
                    "restrict to stable institution descriptors."
                ),
        },

        {
            "channel":
                "founder_description_or_current_geography",

            "experimental_role":
                "SNAPSHOT_PERSON_SENSITIVITY",

            "temporal_class":
                "CURRENT_SNAPSHOT_UNVERSIONED",

            "may_enter_primary":
                False,

            "notes":
                (
                    "Never mix into primary result."
                ),
        },

        {
            "channel":
                "investor_type_geography_extension",

            "experimental_role":
                "SECONDARY_INVESTOR_SIDE_EXTENSION",

            "temporal_class":
                "SNAPSHOT_SLOW_MOVING",

            "may_enter_primary":
                False,

            "notes":
                (
                    "Reserve to protect attribution of the "
                    "startup-side thesis contribution."
                ),
        },

        {
            "channel":
                "investor_preorigin_stage_sector_profile",

            "experimental_role":
                "SECONDARY_INVESTOR_SIDE_EXTENSION",

            "temporal_class":
                "DERIVED_STRICT_PRE_ORIGIN",

            "may_enter_primary":
                False,

            "notes":
                (
                    "Potentially valid but changes investor branch."
                ),
        },
    ]

    experiment_df = pd.DataFrame(
        experiment_rows
    )

    print(
        experiment_df[
            [
                "channel",
                "experimental_role",
                "may_enter_primary",
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # 6. EXPLICIT PROHIBITIONS
    # =========================================================================

    separator("-")

    print(
        "8.1.7.V2.6 — EXPLICIT PROHIBITIONS"
    )

    separator("-")

    prohibition_rows = [
        {
            "channel":
                "startup_id_latent_embedding",

            "reason":
                "Violates inductive cold-start representation.",
        },

        {
            "channel":
                "phase4_doc2vec_document_vector",

            "reason":
                (
                    "Per-document learned vector is "
                    "startup-identity transductive."
                ),
        },

        {
            "channel":
                "focal_startup_prior_investment_history",

            "reason":
                (
                    "Primary cohort is investment-cold by definition."
                ),
        },

        {
            "channel":
                "raw_startup_num_founders",

            "reason":
                (
                    "Use relation-derived resolved founder count instead."
                ),
        },

        {
            "channel":
                "current_startup_funding_stage_or_type",

            "reason":
                "Current state can include post-origin rounds.",
        },

        {
            "channel":
                "current_startup_funding_totals",

            "reason":
                "Outcome-derived post-origin leakage risk.",
        },

        {
            "channel":
                "current_startup_num_investors_or_rounds",

            "reason":
                "Direct target-history / future interaction signal.",
        },

        {
            "channel":
                "current_startup_employees",

            "reason":
                "Dynamic growth signal; current snapshot.",
        },

        {
            "channel":
                "current_startup_revenue",

            "reason":
                "Dynamic current outcome/growth signal.",
        },

        {
            "channel":
                "current_operating_ipo_acquisition_status",

            "reason":
                "Direct current/future outcome information.",
        },

        {
            "channel":
                "crunchbase_rank_or_rank_delta",

            "reason":
                "Current popularity signal with future information.",
        },

        {
            "channel":
                "semrush_or_current_web_traction",

            "reason":
                "Current web traction is not historically versioned.",
        },

        {
            "channel":
                "growth_insight_fields",

            "reason":
                "Vendor-derived current growth/outcome-adjacent signal.",
        },

        {
            "channel":
                "person_current_organizations",

            "reason":
                "High post-origin career leakage risk.",
        },

        {
            "channel":
                "person_current_investment_portfolio_exit_counts",

            "reason":
                "Current outcome-derived Person aggregates.",
        },

        {
            "channel":
                "person_primary_job_title",

            "reason":
                "Current title may postdate prediction origin.",
        },

        {
            "channel":
                "person_gender_as_predictor",

            "reason":
                (
                    "Protected characteristic; never ranking input."
                ),
        },

        {
            "channel":
                "school_current_rank_funding_web_employee_metrics",

            "reason":
                (
                    "Current School state irrelevant to historical "
                    "founder education context."
                ),
        },

        {
            "channel":
                "contacts_team_snapshot",

            "reason":
                (
                    "Selected current contact sample with weak "
                    "historical provenance and unusable canonical org ID."
                ),
        },

        {
            "channel":
                "events_context_without_participant_bridge",

            "reason":
                (
                    "No defensible Startup/Person participant relation."
                ),
        },

        {
            "channel":
                "hub_aggregate_context",

            "reason":
                (
                    "Wrong analytical grain and negligible "
                    "company membership surface."
                ),
        },

        {
            "channel":
                "unfrozen_post_origin_structural_edges",

            "reason":
                "Temporal leakage / protocol violation.",
        },
    ]

    prohibition_df = pd.DataFrame(
        prohibition_rows
    )

    for row in prohibition_rows:

        print(
            f"PROHIBITED  {row['channel']}"
        )

    # =========================================================================
    # 7. HISTORICAL RECONSTRUCTION RULE
    # =========================================================================

    separator("-")

    print(
        "8.1.7.V2.7 — HISTORICAL "
        "RECONSTRUCTION PRINCIPLE"
    )

    separator("-")

    reconstruction_rules = {
        "rule":
            (
                "If a current Crunchbase aggregate can be "
                "reconstructed from pre-origin events/relations, "
                "prefer the reconstructed historical value."
            ),

        "examples": {
            "raw_num_founded_organizations":
                "count founder-linked companies founded before origin",

            "raw_num_funding_rounds":
                "count dated Funding events before origin",

            "raw_funding_total":
                "sum dated round amounts before origin",

            "raw_person_exit_count":
                (
                    "count acquisitions of prior founder companies "
                    "announced before origin"
                ),

            "raw_investor_stage":
                (
                    "derive stage distribution from investment events "
                    "announced before origin"
                ),
        },

        "focal_startup_exception":
            (
                "The focal primary startup remains investment-cold; "
                "its pre-origin investment history is prohibited even "
                "when technically reconstructable."
            ),
    }

    print(
        reconstruction_rules[
            "rule"
        ]
    )

    print()

    for raw, derived in (
        reconstruction_rules[
            "examples"
        ].items()
    ):

        print(
            f"{raw:<36} -> {derived}"
        )

    # =========================================================================
    # 8. MODEL FAMILY CONTRACT
    # =========================================================================

    separator("-")

    print(
        "8.1.7.V2.8 — MODEL FAMILY CONTRACT"
    )

    separator("-")

    model_rows = [
        {
            "model":
                "B0",

            "role":
                "FROZEN_REPRODUCED_ITRS",

            "startup_content":
                "BASELINE_AS_REPRODUCED",

            "structured_startup":
                False,

            "explicit_founder":
                False,

            "founder_prior_history":
                False,

            "heterogeneous_context":
                "BASELINE_AS_REPRODUCED",

            "trend":
                True,

            "selection_role":
                "FIXED_BASELINE",
        },

        {
            "model":
                "B1a",

            "role":
                "INDUCTIVE_CONTENT_BASELINE",

            "startup_content":
                "DESCRIPTION_PLUS_CATEGORIES",

            "structured_startup":
                False,

            "explicit_founder":
                False,

            "founder_prior_history":
                False,

            "heterogeneous_context":
                False,

            "trend":
                True,

            "selection_role":
                "PREDECLARED_COMPARATOR",
        },

        {
            "model":
                "B1b",

            "role":
                "INDUCTIVE_STRUCTURED_STARTUP",

            "startup_content":
                "DESCRIPTION_PLUS_CATEGORIES_PLUS_CATEGORY_GROUPS",

            "structured_startup":
                True,

            "explicit_founder":
                False,

            "founder_prior_history":
                False,

            "heterogeneous_context":
                False,

            "trend":
                True,

            "selection_role":
                "PREDECLARED_COMPARATOR",
        },

        {
            "model":
                "M1",

            "role":
                "INDUCTIVE_HETEROGENEOUS_NO_TREND",

            "startup_content":
                "B1B_CONTENT",

            "structured_startup":
                True,

            "explicit_founder":
                True,

            "founder_prior_history":
                True,

            "heterogeneous_context":
                True,

            "trend":
                False,

            "selection_role":
                "PREDECLARED_MAIN_COMPARATOR",
        },

        {
            "model":
                "M2",

            "role":
                "PROPOSED_TREND_AWARE_INDUCTIVE_HETEROGENEOUS",

            "startup_content":
                "B1B_CONTENT",

            "structured_startup":
                True,

            "explicit_founder":
                True,

            "founder_prior_history":
                True,

            "heterogeneous_context":
                True,

            "trend":
                True,

            "selection_role":
                "PROPOSED_MODEL_FAMILY",
        },
    ]

    model_df = pd.DataFrame(
        model_rows
    )

    print(
        model_df[
            [
                "model",
                "role",
                "structured_startup",
                "explicit_founder",
                "founder_prior_history",
                "trend",
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # 9. Freeze contract
    # =========================================================================

    separator("-")

    print(
        "8.1.7.V2.9 — FREEZE FINAL "
        "INFORMATION CONTRACT"
    )

    separator("-")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    primary_df.to_csv(
        FINAL_CHANNELS_OUTPUT,
        index=False,
    )

    model_df.to_csv(
        MODEL_FAMILY_OUTPUT,
        index=False,
    )

    prohibition_df.to_csv(
        PROHIBITIONS_OUTPUT,
        index=False,
    )

    contract = {
        "schema_version":
            "PHASE_8_1_7_FINAL_INDUCTIVE_INPUT_CONTRACT_V2",

        "status":
            "FROZEN",

        "scientific_role":
            "FINAL_PRE_ARCHITECTURE_INFORMATION_CONTRACT",

        "supersedes":
            "PHASE_8_1_7_INDUCTIVE_INPUT_CONTRACT_AMENDMENT_V1",

        "supersedes_v1":
            True,

        "v1_overwritten":
            False,

        "reason_for_v2":
            (
                "Phase 8.1.8 comprehensive EDA and feature-governance "
                "review identified additional semantically defensible "
                "cold-start information channels before model training, "
                "architecture selection, or April inspection."
            ),

        "primary_information_channels":
            primary_df[
                "channel"
            ].tolist(),

        "primary_startup_structured_block": [
            "startup_category_groups",
            "startup_country",
            "startup_region_location_groups",
            "startup_company_type",
            "startup_age_at_origin",
            "startup_founder_count_resolved",
        ],

        "primary_heterogeneous_block": [
            "explicit_startup_founder_person",
            "acquisition_relation",
            "founder_prior_startup_count",
            "founder_prior_sector_profile",
        ],

        "sensitivity_only": [
            "founder_school_relation",
            "school_stable_context",
            "founder_description_or_current_geography",
        ],

        "secondary_only": [
            "founder_prior_funding_experience",
            "founder_prior_exit_experience",
            "investor_type_geography_extension",
            "investor_preorigin_stage_sector_profile",
        ],

        "historical_reconstruction_principle":
            reconstruction_rules,

        "matched_snapshot_caveat":
            (
                "Description, categories, category groups and geography "
                "are not claimed to be historically versioned. "
                "They are matched-snapshot side information."
            ),

        "founder_observation_caveat":
            (
                "startup.founded_on provides a semantic effective-date "
                "proxy for founder relations, not the date Crunchbase "
                "first observed the relation."
            ),

        "feature_discovery_closed":
            True,

        "future_primary_feature_additions_allowed":
            False,

        "exception_policy":
            (
                "After this freeze, a new primary information channel "
                "requires a documented data-integrity/leakage correction "
                "or formal protocol amendment before any affected model "
                "is trained. Performance results cannot justify adding it."
            ),

        "graph_schema_frozen":
            False,

        "node_initialization_frozen":
            False,

        "architecture_selected":
            False,

        "training_seed_frozen":
            False,

        "model_trained":
            False,

        "April_analyzed":
            False,

        "April_performance_inspected":
            False,

        "next_step":
            (
                "Phase 8.1.9: freeze the heterogeneous graph schema "
                "and decide which approved information is represented "
                "as nodes, edges, edge attributes, or pooled features."
            ),
    }

    write_json(
        AMENDMENT_OUTPUT,
        contract,
    )

    manifest = {
        "phase":
            "8.1.7 V2",

        "status":
            "FROZEN",

        "feature_discovery_closed":
            True,

        "supersedes_v1":
            True,

        "v1_overwritten":
            False,

        "outputs": [
            str(FINAL_CHANNELS_OUTPUT),
            str(MODEL_FAMILY_OUTPUT),
            str(PROHIBITIONS_OUTPUT),
            str(AMENDMENT_OUTPUT),
        ],

        "architecture_selected":
            False,

        "graph_schema_frozen":
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
        FINAL_CHANNELS_OUTPUT,
        MODEL_FAMILY_OUTPUT,
        PROHIBITIONS_OUTPUT,
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
        "PHASE 8.1.7 V2 FINAL INPUT CONTRACT FROZEN"
    )

    separator()

    print()

    print(
        "V1 PRESERVED:                    YES"
    )

    print(
        "V1 SUPERSEDED:                   YES"
    )

    print(
        "FEATURE DISCOVERY CLOSED:         YES"
    )

    print(
        "PRIMARY CHANNEL COUNT:            "
        f"{len(primary_df):,}"
    )

    print(
        "CURRENT/OUTCOME FEATURES:         PROHIBITED"
    )

    print(
        "SCHOOL CONTEXT:                   SENSITIVITY ONLY"
    )

    print(
        "RICH INVESTOR EXTENSIONS:         SECONDARY ONLY"
    )

    print()

    print(
        "GRAPH SCHEMA FROZEN:              NO"
    )

    print(
        "NODE INITIALIZATION FROZEN:       NO"
    )

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
        "Phase 8.1.9 — heterogeneous graph-schema freeze."
    )


if __name__ == "__main__":
    main()