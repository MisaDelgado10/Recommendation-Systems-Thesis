from pathlib import Path
import hashlib
import json

import pandas as pd


# =============================================================================
# PHASE 8.1.7 — INDUCTIVE INPUT CONTRACT AMENDMENT V1
# =============================================================================
#
# Scientific role:
#
#   FORMAL PRE-ARCHITECTURE CONTRACT AMENDMENT
#
# PURPOSE
# -------
#
# Extend, but DO NOT overwrite, the frozen Phase-8.1.2 startup input contract
# using evidence generated during Phase-8.1.6 Data Foundations.
#
# APPROVED AMENDMENT
# ------------------
#
# Add one candidate heterogeneous information channel:
#
#     Startup -> Founder Person
#
# Temporal policy:
#
#     startup.founded_on <= prediction origin
#
# Interpretation:
#
#     semantically origin-eligible founder relation
#
# NOT claimed:
#
#     historical Crunchbase database-observation time
#
# IMPORTANT
# ---------
#
# This amendment does NOT:
#
#   - select an architecture
#   - train a model
#   - inspect April
#   - approve Person attributes
#   - approve Person->School
#   - approve additional Company snapshot attributes
#
# =============================================================================


BASE = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation"
)

DATA_FOUNDATIONS_DIR = (
    BASE
    / "phase_8_1_6_data_foundations"
)

SYNTHESIS_SUMMARY = (
    DATA_FOUNDATIONS_DIR
    / "phase_8_1_6g_data_foundations_synthesis"
    / "phase_8_1_6g_summary_V1.json"
)

DECISION_LEDGER = (
    DATA_FOUNDATIONS_DIR
    / "phase_8_1_6g_data_foundations_synthesis"
    / "phase_8_1_6g_data_channel_decision_ledger_V1.csv"
)

PHASE816D_SUMMARY = (
    DATA_FOUNDATIONS_DIR
    / "phase_8_1_6d_explicit_founder_incremental_coverage"
    / "phase_8_1_6d_summary_V2.json"
)

PHASE816E_SUMMARY = (
    DATA_FOUNDATIONS_DIR
    / "phase_8_1_6e_founder_person_temporal_provenance"
    / "phase_8_1_6e_summary_V2.json"
)

PHASE816F_SUMMARY = (
    DATA_FOUNDATIONS_DIR
    / "phase_8_1_6f_founder_node_structural_utility"
    / "phase_8_1_6f_summary_V1.json"
)


OUTPUT_DIR = (
    BASE
    / "phase_8_1_7_input_contract_amendment"
)

AMENDED_CHANNEL_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_7_amended_input_channel_contract_V1.csv"
)

MODEL_FAMILY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_7_model_family_input_contract_V1.csv"
)

AMENDMENT_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_7_contract_amendment_V1.json"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_7_manifest_V1.json"
)

HASH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_7_artifact_sha256_V1.json"
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
        "INDUCTIVE INPUT CONTRACT AMENDMENT V1"
    )

    separator()

    print()

    print(
        "Scientific role:             "
        "FORMAL PRE-ARCHITECTURE CONTRACT AMENDMENT"
    )

    print(
        "Original 8.1.2 overwritten: NO"
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
    # 1. Evidence gates
    # =========================================================================

    separator("-")

    print(
        "8.1.7.1 — EVIDENCE GATES"
    )

    separator("-")

    required = [
        SYNTHESIS_SUMMARY,
        DECISION_LEDGER,
        PHASE816D_SUMMARY,
        PHASE816E_SUMMARY,
        PHASE816F_SUMMARY,
    ]

    for path in required:

        require(
            path.exists(),
            f"Missing evidence artifact: {path}",
        )

        print(
            f"FOUND {path}"
        )

    synthesis = read_json(
        SYNTHESIS_SUMMARY
    )

    d_summary = read_json(
        PHASE816D_SUMMARY
    )

    e_summary = read_json(
        PHASE816E_SUMMARY
    )

    f_summary = read_json(
        PHASE816F_SUMMARY
    )

    require(
        synthesis.get("status")
        ==
        "COMPLETE",
        "Phase 8.1.6G incomplete.",
    )

    require(
        d_summary.get("status")
        ==
        "COMPLETE",
        "Phase 8.1.6D incomplete.",
    )

    require(
        e_summary.get("status")
        ==
        "COMPLETE",
        "Phase 8.1.6E incomplete.",
    )

    require(
        f_summary.get("status")
        ==
        "COMPLETE",
        "Phase 8.1.6F incomplete.",
    )

    for name, obj in [
        ("8.1.6D", d_summary),
        ("8.1.6E", e_summary),
        ("8.1.6F", f_summary),
    ]:

        require(
            obj.get(
                "April_analyzed"
            )
            is False,
            f"{name}: April unexpectedly analyzed.",
        )

    print()

    print(
        "Data Foundations evidence chain: PASS"
    )

    # =========================================================================
    # 2. Quantitative evidence
    # =========================================================================

    separator("-")

    print(
        "8.1.7.2 — AMENDMENT EVIDENCE"
    )

    separator("-")

    founder_coverage = float(
        f_summary[
            "mean_founder_coverage_pct"
        ]
    )

    founder_leaf = float(
        f_summary[
            "mean_founder_leaf_only_pct"
        ]
    )

    founder_shared = float(
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

    raw_person_coverage = float(
        e_summary[
            "mean_raw_founder_person_coverage_pct"
        ]
    )

    require(
        abs(
            founder_coverage
            -
            raw_person_coverage
        )
        <
        1e-6,
        (
            "Founder coverage disagreement across "
            "Data Foundations audits."
        ),
    )

    require(
        abs(
            semantic_retention
            -
            100.0
        )
        <
        1e-6,
        (
            "Expected 100% semantic retention "
            "from completed development audit."
        ),
    )

    leaf_fraction_given_founder = (
        100.0
        * founder_leaf
        /
        founder_coverage
    )

    print(
        f"Mean primary founder coverage:      "
        f"{founder_coverage:.4f}%"
    )

    print(
        f"Current isolation recovered:        "
        f"{isolation_recovered:.4f}%"
    )

    print(
        f"Semantic temporal retention:        "
        f"{semantic_retention:.4f}%"
    )

    print(
        f"Founder degree>=2 coverage:         "
        f"{founder_shared:.4f}%"
    )

    print(
        f"Founder leaf-only coverage:         "
        f"{founder_leaf:.4f}%"
    )

    print(
        f"Leaf-only among founder cases:      "
        f"{leaf_fraction_given_founder:.4f}%"
    )

    # =========================================================================
    # 3. Amended information-channel contract
    # =========================================================================

    separator("-")

    print(
        "8.1.7.3 — AMENDED INFORMATION-CHANNEL CONTRACT"
    )

    separator("-")

    channel_rows = [
        {
            "channel_id":
                "I1",

            "channel":
                "resolved_description_text",

            "source":
                "companies",

            "status":
                "ALLOWED",

            "provenance":
                "MATCHED_SNAPSHOT_UNVERSIONED",

            "role":
                "INDUCTIVE_STARTUP_CONTENT",

            "amendment":
                "UNCHANGED_FROM_8_1_2",
        },

        {
            "channel_id":
                "I2",

            "channel":
                "category_multihot",

            "source":
                "companies",

            "status":
                "ALLOWED",

            "provenance":
                "MATCHED_SNAPSHOT_UNVERSIONED",

            "role":
                "INDUCTIVE_STARTUP_CONTENT",

            "amendment":
                "UNCHANGED_FROM_8_1_2",
        },

        {
            "channel_id":
                "I3",

            "channel":
                "timestamped_acquisition_relations",

            "source":
                "acquisitions",

            "status":
                "ALLOWED",

            "provenance":
                "TIMESTAMPED_PRE_ORIGIN",

            "role":
                "HETEROGENEOUS_RELATIONAL_CONTEXT",

            "amendment":
                "UNCHANGED_FROM_8_1_2",
        },

        {
            "channel_id":
                "I4",

            "channel":
                "shared_founder_projection",

            "source":
                "companies + people",

            "status":
                "ALLOWED_EXISTING_TRACK",

            "provenance":
                (
                    "SEMANTIC_FOUNDED_ON_GATE_"
                    "DATABASE_OBSERVATION_UNKNOWN"
                ),

            "role":
                "LEGACY_PROJECTED_FOUNDER_CONTEXT",

            "amendment":
                "UNCHANGED_EXISTING_TRACK",
        },

        {
            "channel_id":
                "I5",

            "channel":
                "explicit_startup_founder_person",

            "source":
                "companies.founder_links + people.link",

            "status":
                "ALLOWED_BY_AMENDMENT",

            "provenance":
                (
                    "SEMANTIC_FOUNDED_ON_GATE_"
                    "DATABASE_OBSERVATION_UNKNOWN"
                ),

            "role":
                "INDUCTIVE_TYPED_HETEROGENEOUS_CONTEXT",

            "amendment":
                "NEW_8_1_7",
        },
    ]

    channel_df = pd.DataFrame(
        channel_rows
    )

    print(
        channel_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # 4. Explicit prohibitions
    # =========================================================================

    separator("-")

    print(
        "8.1.7.4 — EXPLICITLY PROHIBITED "
        "PRIMARY INPUTS"
    )

    separator("-")

    prohibited = [
        "startup_id_latent_embedding",
        "phase4_doc2vec_document_vector",
        "startup_prior_investment_history",
        "founded_on_as_predictive_feature",
        "post_origin_interaction_information",
        "unfrozen_structural_edges",

        "startup_geography",
        "startup_company_type",
        "startup_category_groups",

        "person_description",
        "person_geography",
        "person_current_organizations",
        "person_school_relation",

        "startup_current_funding_state",
        "startup_operating_status",
        "startup_ipo_status",
        "startup_acquisition_status",

        "startup_rank_fields",
        "startup_semrush_fields",
    ]

    for item in prohibited:

        print(
            f"PROHIBITED  {item}"
        )

    # =========================================================================
    # 5. Founder relation contract
    # =========================================================================

    separator("-")

    print(
        "8.1.7.5 — EXPLICIT FOUNDER-PERSON CONTRACT"
    )

    separator("-")

    founder_contract = {
        "source_field":
            "companies.founder_links",

        "resolution_target":
            "people.link",

        "resolution_method":
            "EXACT_NORMALIZED_CRUNCHBASE_PERSON_URL",

        "edge_semantics":
            "startup_founded_by_person",

        "origin_gate":
            "startup.founded_on <= prediction_origin",

        "founded_on_use":
            "EDGE_ELIGIBILITY_ONLY",

        "database_observation_timestamp_known":
            False,

        "allowed_claim":
            (
                "semantically origin-eligible "
                "founder relation"
            ),

        "prohibited_claim":
            (
                "strict historically observed "
                "Crunchbase founder relation"
            ),

        "person_identity_embedding_allowed":
            False,

        "person_snapshot_attributes_allowed":
            False,

        "person_school_relation_allowed":
            False,

        "target_pair_information_used":
            False,

        "startup_prior_investment_required":
            False,
    }

    for key, value in (
        founder_contract.items()
    ):

        print(
            f"{key:<42} {value}"
        )

    # =========================================================================
    # 6. Model-family input constraints
    # =========================================================================

    separator("-")

    print(
        "8.1.7.6 — MODEL-FAMILY INPUT CONTRACT"
    )

    separator("-")

    model_rows = [
        {
            "model_family":
                "B0",

            "scientific_role":
                "REPRODUCED_ITRS_BASELINE",

            "raw_text_categories":
                False,

            "startup_id_embedding":
                True,

            "phase4_doc2vec":
                True,

            "acquisition_relation":
                "BASELINE_AS_REPRODUCED",

            "shared_founder_projection":
                "BASELINE_AS_REPRODUCED",

            "explicit_founder_person":
                False,

            "trend_aware_investor":
                True,

            "notes":
                "B0 remains unchanged for baseline comparability.",
        },

        {
            "model_family":
                "B1",

            "scientific_role":
                "INDUCTIVE_ATTRIBUTE_ONLY",

            "raw_text_categories":
                True,

            "startup_id_embedding":
                False,

            "phase4_doc2vec":
                False,

            "acquisition_relation":
                False,

            "shared_founder_projection":
                False,

            "explicit_founder_person":
                False,

            "trend_aware_investor":
                True,

            "notes":
                (
                    "Tests whether inductive startup content "
                    "alone improves cold-start ranking."
                ),
        },

        {
            "model_family":
                "M1",

            "scientific_role":
                "INDUCTIVE_HETEROGENEOUS_NO_TREND",

            "raw_text_categories":
                True,

            "startup_id_embedding":
                False,

            "phase4_doc2vec":
                False,

            "acquisition_relation":
                True,

            "shared_founder_projection":
                "ARCHITECTURE_STAGE_DECISION",

            "explicit_founder_person":
                True,

            "trend_aware_investor":
                False,

            "notes":
                (
                    "Founder-Person is permitted by amendment; "
                    "exact handling of legacy projection is frozen later."
                ),
        },

        {
            "model_family":
                "M2",

            "scientific_role":
                "INDUCTIVE_HETEROGENEOUS_PLUS_TREND",

            "raw_text_categories":
                True,

            "startup_id_embedding":
                False,

            "phase4_doc2vec":
                False,

            "acquisition_relation":
                True,

            "shared_founder_projection":
                "ARCHITECTURE_STAGE_DECISION",

            "explicit_founder_person":
                True,

            "trend_aware_investor":
                True,

            "notes":
                (
                    "Primary proposed family: inductive startup "
                    "representation plus preserved ITRS trend signal."
                ),
        },
    ]

    model_df = pd.DataFrame(
        model_rows
    )

    print(
        model_df[
            [
                "model_family",
                "scientific_role",
                "explicit_founder_person",
                "trend_aware_investor",
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # 7. Persist amendment
    # =========================================================================

    separator("-")

    print(
        "8.1.7.7 — PERSIST CONTRACT AMENDMENT"
    )

    separator("-")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    channel_df.to_csv(
        AMENDED_CHANNEL_OUTPUT,
        index=False,
    )

    model_df.to_csv(
        MODEL_FAMILY_OUTPUT,
        index=False,
    )

    amendment = {
        "schema_version":
            "PHASE_8_1_7_INDUCTIVE_INPUT_CONTRACT_AMENDMENT_V1",

        "status":
            "FROZEN",

        "scientific_role":
            "FORMAL_PRE_ARCHITECTURE_INPUT_CONTRACT_AMENDMENT",

        "supersedes_phase_8_1_2":
            False,

        "extends_phase_8_1_2":
            True,

        "approved_new_channel":
            "explicit_startup_founder_person",

        "approved_new_channel_id":
            "I5",

        "founder_contract":
            founder_contract,

        "quantitative_justification": {
            "mean_primary_founder_coverage_pct":
                founder_coverage,

            "mean_current_isolation_recovered_pct":
                isolation_recovered,

            "semantic_temporal_retention_pct":
                semantic_retention,

            "mean_founder_degree_ge2_pct":
                founder_shared,

            "mean_founder_leaf_only_pct":
                founder_leaf,

            "leaf_only_fraction_given_founder_pct":
                leaf_fraction_given_founder,
        },

        "scientific_rationale":
            (
                "The explicit Startup-Founder-Person relation is "
                "identity-resolvable, does not require prior startup "
                "investment history, has substantial primary cold-start "
                "coverage, and can be semantically gated by startup "
                "founding date. It directly operationalizes the proposal's "
                "typed heterogeneous-neighbor component. Its effectiveness "
                "remains an empirical question because most founder-covered "
                "cases are leaf-only."
            ),

        "new_company_snapshot_attributes_approved":
            False,

        "person_snapshot_attributes_approved":
            False,

        "person_school_relation_approved":
            False,

        "architecture_selected":
            False,

        "graph_schema_frozen":
            False,

        "model_trained":
            False,

        "April_analyzed":
            False,

        "next_step":
            (
                "Freeze the Phase-8.1 heterogeneous graph schema "
                "and identity-free node initialization policy."
            ),
    }

    write_json(
        AMENDMENT_OUTPUT,
        amendment,
    )

    manifest = {
        "phase":
            "8.1.7",

        "status":
            "FROZEN",

        "scientific_role":
            "INPUT_CONTRACT_AMENDMENT",

        "outputs": [
            str(AMENDED_CHANNEL_OUTPUT),
            str(MODEL_FAMILY_OUTPUT),
            str(AMENDMENT_OUTPUT),
        ],

        "April_analyzed":
            False,

        "architecture_selected":
            False,

        "graph_schema_frozen":
            False,

        "model_trained":
            False,
    }

    write_json(
        MANIFEST_OUTPUT,
        manifest,
    )

    outputs = [
        AMENDED_CHANNEL_OUTPUT,
        MODEL_FAMILY_OUTPUT,
        AMENDMENT_OUTPUT,
        MANIFEST_OUTPUT,
    ]

    write_json(
        HASH_OUTPUT,
        {
            str(path):
                sha256_file(
                    path
                )
            for path
            in outputs
        },
    )

    separator()

    print(
        "PHASE 8.1.7 INPUT CONTRACT "
        "AMENDMENT FROZEN"
    )

    separator()

    print()

    print(
        "ORIGINAL 8.1.2 OVERWRITTEN:      NO"
    )

    print(
        "NEW ALLOWED CHANNEL:"
    )

    print(
        "  I5 explicit Startup -> Founder Person"
    )

    print()

    print(
        "FOUNDER EDGE TEMPORAL POLICY:"
    )

    print(
        "  startup.founded_on <= prediction origin"
    )

    print()

    print(
        "PERSON SNAPSHOT ATTRIBUTES:       PROHIBITED"
    )

    print(
        "PERSON -> SCHOOL PRIMARY INPUT:   PROHIBITED"
    )

    print(
        "NEW COMPANY SNAPSHOT FEATURES:    PROHIBITED"
    )

    print()

    print(
        "ARCHITECTURE SELECTED:            NO"
    )

    print(
        "GRAPH SCHEMA FROZEN:              NO"
    )

    print(
        "MODEL TRAINED:                    NO"
    )

    print(
        "APRIL ANALYZED:                   NO"
    )

    print()

    print(
        "Next:"
    )

    print(
        "Freeze the identity-free heterogeneous "
        "graph schema and node initialization policy."
    )


if __name__ == "__main__":
    main()