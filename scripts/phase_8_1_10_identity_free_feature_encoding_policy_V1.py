from pathlib import Path
import hashlib
import json

import pandas as pd


# =============================================================================
# PHASE 8.1.10 — IDENTITY-FREE NODE INITIALIZATION &
#                FEATURE-ENCODING POLICY V1
# =============================================================================
#
# Scientific role:
#
#   PRE-ARCHITECTURE REPRESENTATION / LEAKAGE CONTRACT
#
# PURPOSE
# -------
#
# Freeze HOW allowed Phase-8.1 information may be transformed into model
# inputs without introducing entity-identity memorization or evaluation-set
# adaptation.
#
# This phase freezes:
#
#   - Startup / Person identity restrictions;
#   - allowed Investor identity exception;
#   - shared encoder semantics;
#   - preprocessing fit scope;
#   - origin-specific feature semantics;
#   - missing / unknown handling principles;
#   - evaluation-set independence;
#   - common graph-input width requirement.
#
# This phase DOES NOT yet freeze:
#
#   - exact text featurizer family;
#   - exact hidden branch dimensions;
#   - exact MLP layer counts;
#   - exact GNN implementation;
#   - optimizer;
#   - initialization seed;
#   - model training.
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

PHASE4_DESCRIPTION_CONTRACT = Path(
    "data/experimental/phase_4/"
    "description_neural_contract/"
    "description_neural_contract.json"
)

PHASE4_RGCN_CONTRACT = Path(
    "data/experimental/phase_4/"
    "rgcn_neural_contract/"
    "rgcn_neural_contract.json"
)


OUTPUT_DIR = (
    BASE
    / "phase_8_1_10_identity_free_encoding_policy"
)

NODE_INIT_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_10_node_initialization_policy_V1.csv"
)

FEATURE_ENCODING_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_10_feature_encoding_policy_V1.csv"
)

FIT_SCOPE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_10_preprocessing_fit_scope_V1.csv"
)

MISSINGNESS_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_10_missing_unknown_policy_V1.csv"
)

CONTRACT_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_10_identity_free_encoding_contract_V1.json"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_10_manifest_V1.json"
)

HASH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_10_artifact_sha256_V1.json"
)


COMMON_STRUCTURAL_INPUT_DIM = 40


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
        "PHASE 8.1.10 — "
        "IDENTITY-FREE NODE INITIALIZATION & "
        "FEATURE-ENCODING POLICY V1"
    )

    separator()

    print()

    print(
        "Scientific role:             "
        "PRE-ARCHITECTURE REPRESENTATION / LEAKAGE CONTRACT"
    )

    print(
        "Feature discovery reopened:  NO"
    )

    print(
        "Graph schema modified:       NO"
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
    # 1. PREREQUISITE GATES
    # =========================================================================

    separator("-")

    print(
        "8.1.10.1 — PREREQUISITE GATES"
    )

    separator("-")

    required = [
        INPUT_CONTRACT,
        GRAPH_SCHEMA,
        PHASE4_DESCRIPTION_CONTRACT,
        PHASE4_RGCN_CONTRACT,
    ]

    for path in required:

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

    phase4_description = read_json(
        PHASE4_DESCRIPTION_CONTRACT
    )

    phase4_rgcn = read_json(
        PHASE4_RGCN_CONTRACT
    )

    require(
        input_contract.get("status")
        ==
        "FROZEN",
        "Phase 8.1.7 V2 is not frozen.",
    )

    require(
        input_contract.get(
            "feature_discovery_closed"
        )
        is True,
        "Feature discovery is unexpectedly open.",
    )

    require(
        graph_schema.get("status")
        ==
        "FROZEN",
        "Phase 8.1.9 graph schema is not frozen.",
    )

    require(
        graph_schema.get(
            "candidate_universe_expanded"
        )
        is False,
        "Candidate universe unexpectedly expanded.",
    )

    require(
        graph_schema[
            "structural_safety_rules"
        ][
            "startup_id_embedding"
        ]
        is False,
        "Phase 8.1.9 unexpectedly allows Startup ID embeddings.",
    )

    require(
        graph_schema[
            "structural_safety_rules"
        ][
            "person_id_embedding"
        ]
        is False,
        "Phase 8.1.9 unexpectedly allows Person ID embeddings.",
    )

    require(
        phase4_description.get("status")
        ==
        "FROZEN",
        "Phase-4 description contract unavailable.",
    )

    require(
        phase4_rgcn.get("status")
        ==
        "FROZEN",
        "Phase-4 R-GCN contract unavailable.",
    )

    print()

    print(
        "Phase 8.1.7 V2: PASS"
    )

    print(
        "Phase 8.1.9:    PASS"
    )

    print(
        "Phase-4 baseline reference contracts: PASS"
    )

    # =========================================================================
    # 2. NODE INITIALIZATION POLICY
    # =========================================================================

    separator("-")

    print(
        "8.1.10.2 — NODE INITIALIZATION POLICY"
    )

    separator("-")

    node_init_rows = [
        {
            "node_type":
                "investor",

            "model_scope":
                "B1a | B1b | M1 | M2",

            "identity_indexed_parameter_allowed":
                True,

            "initialization_source":
                "BASELINE_TRAINABLE_INVESTOR_LATENT",

            "required_output_dim":
                COMMON_STRUCTURAL_INPUT_DIM,

            "inductive_requirement":
                False,

            "reason":
                (
                    "Primary research target is warm-investor / "
                    "investment-cold startup. Preserve investor-side "
                    "baseline capacity for attribution."
                ),
        },

        {
            "node_type":
                "startup",

            "model_scope":
                "B1a | B1b | M1 | M2",

            "identity_indexed_parameter_allowed":
                False,

            "initialization_source":
                "SHARED_STARTUP_FEATURE_ENCODER",

            "required_output_dim":
                COMMON_STRUCTURAL_INPUT_DIM,

            "inductive_requirement":
                True,

            "reason":
                (
                    "Startup representation must be computable for an "
                    "unseen Startup UUID entirely from allowed side "
                    "information."
                ),
        },

        {
            "node_type":
                "person",

            "model_scope":
                "M1 | M2",

            "identity_indexed_parameter_allowed":
                False,

            "initialization_source":
                "SHARED_PERSON_HISTORY_ENCODER",

            "required_output_dim":
                COMMON_STRUCTURAL_INPUT_DIM,

            "inductive_requirement":
                True,

            "reason":
                (
                    "Person UUID is connectivity only; no founder "
                    "identity memorization is permitted."
                ),
        },

        {
            "node_type":
                "startup",

            "model_scope":
                "B0_ONLY",

            "identity_indexed_parameter_allowed":
                True,

            "initialization_source":
                "FROZEN_REPRODUCED_ITRS_BASELINE",

            "required_output_dim":
                40,

            "inductive_requirement":
                False,

            "reason":
                (
                    "B0 remains unchanged as the reproduced "
                    "transductive comparator."
                ),
        },
    ]

    node_init_df = pd.DataFrame(
        node_init_rows
    )

    print(
        node_init_df[
            [
                "node_type",
                "model_scope",
                "identity_indexed_parameter_allowed",
                "initialization_source",
                "required_output_dim",
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # 3. STARTUP / PERSON FEATURE ENCODING POLICY
    # =========================================================================

    separator("-")

    print(
        "8.1.10.3 — FEATURE ENCODING POLICY"
    )

    separator("-")

    encoding_rows = [
        {
            "channel":
                "startup_description",

            "node_type":
                "startup",

            "encoding_family":
                "SHARED_COMPOSITIONAL_TEXT_ENCODER",

            "identity_free":
                True,

            "fit_on_evaluation_entities":
                False,

            "origin_specific":
                False,

            "policy":
                (
                    "Raw description/short-description text may be "
                    "encoded only through a shared text function. "
                    "Document tags, Startup IDs and learned per-document "
                    "vectors are forbidden. Exact featurizer frozen in "
                    "Phase 8.1.11."
                ),
        },

        {
            "channel":
                "startup_categories",

            "node_type":
                "startup",

            "encoding_family":
                "MULTILABEL_SHARED_ENCODER",

            "identity_free":
                True,

            "fit_on_evaluation_entities":
                False,

            "origin_specific":
                False,

            "policy":
                (
                    "Multi-label category semantics. Parsing may reuse "
                    "the audited Phase-4 parser; data-driven vocabulary "
                    "or projection statistics must not adapt to the "
                    "evaluation cohort."
                ),
        },

        {
            "channel":
                "startup_category_groups",

            "node_type":
                "startup",

            "encoding_family":
                "MULTILABEL_SHARED_ENCODER",

            "identity_free":
                True,

            "fit_on_evaluation_entities":
                False,

            "origin_specific":
                False,

            "policy":
                (
                    "Coarse industry hierarchy encoded independently "
                    "of Startup identity."
                ),
        },

        {
            "channel":
                "startup_country",

            "node_type":
                "startup",

            "encoding_family":
                "CATEGORICAL_SHARED_ENCODER",

            "identity_free":
                True,

            "fit_on_evaluation_entities":
                False,

            "origin_specific":
                False,

            "policy":
                (
                    "Country token with explicit missing/unknown "
                    "handling; no raw Startup-ID interaction."
                ),
        },

        {
            "channel":
                "startup_region_location_groups",

            "node_type":
                "startup",

            "encoding_family":
                "MULTILABEL_GEOGRAPHY_ENCODER",

            "identity_free":
                True,

            "fit_on_evaluation_entities":
                False,

            "origin_specific":
                False,

            "policy":
                (
                    "Region/location-group hierarchy; city excluded "
                    "from the primary block."
                ),
        },

        {
            "channel":
                "startup_company_type",

            "node_type":
                "startup",

            "encoding_family":
                "LOW_CARDINALITY_CATEGORICAL_ENCODER",

            "identity_free":
                True,

            "fit_on_evaluation_entities":
                False,

            "origin_specific":
                False,

            "policy":
                (
                    "Shared categorical encoding with missing token."
                ),
        },

        {
            "channel":
                "startup_age_at_origin",

            "node_type":
                "startup",

            "encoding_family":
                "NUMERIC_ORIGIN_SPECIFIC",

            "identity_free":
                True,

            "fit_on_evaluation_entities":
                False,

            "origin_specific":
                True,

            "policy":
                (
                    "Compute nonnegative age from origin-founded_on. "
                    "Use deterministic log1p age transform; any "
                    "centering/scaling statistics are training-only."
                ),
        },

        {
            "channel":
                "startup_founder_count_resolved",

            "node_type":
                "startup",

            "encoding_family":
                "NUMERIC_ORIGIN_SPECIFIC",

            "identity_free":
                True,

            "fit_on_evaluation_entities":
                False,

            "origin_specific":
                True,

            "policy":
                (
                    "Count only founder edges eligible at origin. "
                    "Apply log1p transform; any normalization "
                    "statistics are training-only."
                ),
        },

        {
            "channel":
                "founder_prior_startup_count",

            "node_type":
                "person",

            "encoding_family":
                "NUMERIC_ORIGIN_SPECIFIC",

            "identity_free":
                True,

            "fit_on_evaluation_entities":
                False,

            "origin_specific":
                True,

            "policy":
                (
                    "Count non-focal companies founded before origin. "
                    "Apply log1p transform. No current raw People "
                    "aggregate is allowed."
                ),
        },

        {
            "channel":
                "founder_prior_sector_profile",

            "node_type":
                "person",

            "encoding_family":
                "POOLED_MULTILABEL_SECTOR_ENCODER",

            "identity_free":
                True,

            "fit_on_evaluation_entities":
                False,

            "origin_specific":
                True,

            "policy":
                (
                    "Aggregate sector labels over eligible non-focal "
                    "pre-origin founder ventures. Normalize pooled "
                    "sector counts independently from venture count "
                    "so prior_startup_count remains a separate signal."
                ),
        },
    ]

    encoding_df = pd.DataFrame(
        encoding_rows
    )

    print(
        encoding_df[
            [
                "channel",
                "node_type",
                "encoding_family",
                "origin_specific",
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # 4. PREPROCESSING FIT-SCOPE POLICY
    # =========================================================================

    separator("-")

    print(
        "8.1.10.4 — PREPROCESSING FIT-SCOPE POLICY"
    )

    separator("-")

    fit_scope_rows = [
        {
            "component":
                "text_vocabulary_or_unsupervised_text_statistics",

            "development_rule":
                (
                    "FIT USING TRAINING-SIDE STARTUP TEXT ONLY "
                    "FOR THE ORIGIN"
                ),

            "final_april_rule":
                (
                    "FIT USING FINAL PRE-APRIL TRAINING DATA ONLY"
                ),

            "evaluation_distribution_access":
                False,

            "negative_sample_access":
                False,
        },

        {
            "component":
                "categorical_vocabulary_if_data_driven",

            "development_rule":
                (
                    "FIT USING TRAINING-SIDE VALUES ONLY; "
                    "UNSEEN VALUES -> UNK"
                ),

            "final_april_rule":
                (
                    "FIT USING FINAL PRE-APRIL TRAINING VALUES ONLY"
                ),

            "evaluation_distribution_access":
                False,

            "negative_sample_access":
                False,
        },

        {
            "component":
                "numeric_scaling_statistics",

            "development_rule":
                (
                    "FIT MEAN/STD OR ROBUST STATISTICS "
                    "ON TRAINING-SIDE ENTITIES ONLY"
                ),

            "final_april_rule":
                (
                    "FIT USING FINAL PRE-APRIL TRAINING ENTITIES ONLY"
                ),

            "evaluation_distribution_access":
                False,

            "negative_sample_access":
                False,
        },

        {
            "component":
                "founder_history_raw_aggregation",

            "development_rule":
                (
                    "COMPUTE FROM ALL ALLOWED SOURCE RECORDS WITH "
                    "EFFECTIVE DATE < ORIGIN; NO LABEL FITTING"
                ),

            "final_april_rule":
                (
                    "COMPUTE USING ONLY EVENTS/RELATIONS "
                    "ELIGIBLE BEFORE FINAL ORIGIN"
                ),

            "evaluation_distribution_access":
                True,

            "negative_sample_access":
                False,
        },

        {
            "component":
                "graph_relation_materialization",

            "development_rule":
                (
                    "BUILD ON THE FULL ORIGIN-ELIGIBLE GRAPH UNIVERSE, "
                    "NOT ON THE 100 CANDIDATES OF AN EVALUATION CASE"
                ),

            "final_april_rule":
                (
                    "BUILD ON THE FULL FROZEN PRE-APRIL "
                    "ELIGIBLE GRAPH UNIVERSE"
                ),

            "evaluation_distribution_access":
                True,

            "negative_sample_access":
                False,
        },
    ]

    fit_scope_df = pd.DataFrame(
        fit_scope_rows
    )

    print(
        fit_scope_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # 5. EVALUATION-INDEPENDENCE CONTRACT
    # =========================================================================

    separator("-")

    print(
        "8.1.10.5 — EVALUATION-INDEPENDENCE CONTRACT"
    )

    separator("-")

    evaluation_policy = {
        "feature_tables_built_per_case":
            False,

        "feature_tables_built_per_negative_sample":
            False,

        "negative_ids_used_to_fit_preprocessing":
            False,

        "positive_ids_used_to_fit_preprocessing":
            False,

        "evaluation_labels_used_in_feature_construction":
            False,

        "candidate_rank_used_in_feature_construction":
            False,

        "full_origin_graph_built_once":
            True,

        "origin_specific_feature_store_built_once":
            True,

        "evaluation_candidates_may_be_encoded":
            True,

        "evaluation_candidate_side_information_allowed":
            True,

        "reason":
            (
                "At inference, an unseen startup may supply its allowed "
                "side information and eligible graph edges, but its "
                "presence in the evaluation sample must not alter learned "
                "preprocessing parameters or other entities' raw features."
            ),
    }

    for key, value in (
        evaluation_policy.items()
    ):

        print(
            f"{key:<52} {value}"
        )

    # =========================================================================
    # 6. MISSING / UNKNOWN POLICY
    # =========================================================================

    separator("-")

    print(
        "8.1.10.6 — MISSING / UNKNOWN POLICY"
    )

    separator("-")

    missing_rows = [
        {
            "channel":
                "startup_description",

            "missing_behavior":
                "ZERO_CONTENT_INPUT",

            "separate_missingness_feature":
                False,

            "notes":
                (
                    "No entity-specific fallback vector."
                ),
        },

        {
            "channel":
                "startup_categories",

            "missing_behavior":
                "ALL_ZERO_MULTILABEL",

            "separate_missingness_feature":
                False,

            "notes":
                "Unseen token maps to UNK only if encoder requires it.",
        },

        {
            "channel":
                "startup_category_groups",

            "missing_behavior":
                "ALL_ZERO_MULTILABEL",

            "separate_missingness_feature":
                False,

            "notes":
                "No completeness score added.",
        },

        {
            "channel":
                "startup_country_region",

            "missing_behavior":
                "EXPLICIT_MISSING_TOKEN",

            "separate_missingness_feature":
                False,

            "notes":
                "Unknown unseen values use distinct UNK token.",
        },

        {
            "channel":
                "startup_company_type",

            "missing_behavior":
                "EXPLICIT_MISSING_TOKEN",

            "separate_missingness_feature":
                False,

            "notes":
                "Unknown unseen values use distinct UNK token.",
        },

        {
            "channel":
                "startup_age_at_origin",

            "missing_behavior":
                "TRAINING_MEDIAN_AFTER_TRANSFORM",

            "separate_missingness_feature":
                False,

            "notes":
                (
                    "No separate founded_on-missing indicator in "
                    "the primary representation."
                ),
        },

        {
            "channel":
                "startup_founder_count_resolved",

            "missing_behavior":
                "ZERO",

            "separate_missingness_feature":
                False,

            "notes":
                (
                    "Zero means no eligible resolved founder relation."
                ),
        },

        {
            "channel":
                "founder_prior_startup_count",

            "missing_behavior":
                "ZERO",

            "separate_missingness_feature":
                False,

            "notes":
                "Zero means no eligible non-focal prior venture.",
        },

        {
            "channel":
                "founder_prior_sector_profile",

            "missing_behavior":
                "ALL_ZERO_MULTILABEL",

            "separate_missingness_feature":
                False,

            "notes":
                (
                    "Expected when founder has no prior eligible venture."
                ),
        },
    ]

    missing_df = pd.DataFrame(
        missing_rows
    )

    print(
        missing_df[
            [
                "channel",
                "missing_behavior",
                "separate_missingness_feature",
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # 7. PARAMETER-SHARING CONTRACT
    # =========================================================================

    separator("-")

    print(
        "8.1.10.7 — PARAMETER-SHARING CONTRACT"
    )

    separator("-")

    parameter_policy = {
        "startup_encoder_shared_across_startups":
            True,

        "startup_encoder_indexed_by_uuid":
            False,

        "person_encoder_shared_across_people":
            True,

        "person_encoder_indexed_by_uuid":
            False,

        "investor_identity_embedding_allowed":
            True,

        "investor_identity_embedding_scope":
            "WARM_INVESTOR_PRIMARY_PROTOCOL_ONLY",

        "startup_document_vector_table_allowed":
            False,

        "startup_latent_table_allowed":
            False,

        "person_latent_table_allowed":
            False,

        "learned_missing_entity_vector_allowed":
            False,

        "common_structural_input_dim":
            COMMON_STRUCTURAL_INPUT_DIM,
    }

    for key, value in (
        parameter_policy.items()
    ):

        print(
            f"{key:<52} {value}"
        )

    # =========================================================================
    # 8. ORIGIN-SPECIFIC FEATURE SEMANTICS
    # =========================================================================

    separator("-")

    print(
        "8.1.10.8 — ORIGIN-SPECIFIC FEATURE SEMANTICS"
    )

    separator("-")

    origin_policy = {
        "T57":
            "construct with T57 origin rules",

        "T58":
            "construct with T58 origin rules",

        "T59":
            "construct with T59 origin rules",

        "T60":
            (
                "construct only after architecture/model selection "
                "is fully frozen; no selection on T60"
            ),

        "APRIL":
            (
                "construct only for final confirmatory execution "
                "after final fit; never use for selection"
            ),

        "startup_age":
            "origin - founded_on",

        "founder_edge":
            "startup.founded_on <= origin",

        "founder_prior_venture":
            (
                "non-focal company founded_on < origin"
            ),

        "acquisition_edge":
            "announced_on <= origin",

        "focal_startup_investment_history":
            "NEVER USED",
    }

    for key, value in (
        origin_policy.items()
    ):

        print(
            f"{key:<30} {value}"
        )

    # =========================================================================
    # 9. ARCHITECTURE ITEMS DELIBERATELY DEFERRED
    # =========================================================================

    separator("-")

    print(
        "8.1.10.9 — DELIBERATELY DEFERRED "
        "TO PHASE 8.1.11"
    )

    separator("-")

    deferred = [
        "exact text featurizer family",
        "text representation dimension",
        "category/category-group projection dimensions",
        "geography projection dimension",
        "structured-feature fusion MLP",
        "Person-history fusion MLP",
        "exact R-GCN relation vocabulary after role projection",
        "R-GCN layer count confirmation/adaptation",
        "R-GCN basis count confirmation/adaptation",
        "dropout / normalization",
        "pair-scoring fusion dimensions",
        "exact parameter initialization",
    ]

    for item in deferred:

        print(
            f"DEFERRED  {item}"
        )

    # =========================================================================
    # 10. PERSIST
    # =========================================================================

    separator("-")

    print(
        "8.1.10.10 — PERSIST REPRESENTATION CONTRACT"
    )

    separator("-")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    node_init_df.to_csv(
        NODE_INIT_OUTPUT,
        index=False,
    )

    encoding_df.to_csv(
        FEATURE_ENCODING_OUTPUT,
        index=False,
    )

    fit_scope_df.to_csv(
        FIT_SCOPE_OUTPUT,
        index=False,
    )

    missing_df.to_csv(
        MISSINGNESS_OUTPUT,
        index=False,
    )

    contract = {
        "schema_version":
            "PHASE_8_1_10_IDENTITY_FREE_ENCODING_POLICY_V1",

        "status":
            "FROZEN",

        "scientific_role":
            "PRE_ARCHITECTURE_REPRESENTATION_LEAKAGE_CONTRACT",

        "source_input_contract":
            str(INPUT_CONTRACT),

        "source_graph_schema":
            str(GRAPH_SCHEMA),

        "common_structural_input_dim":
            COMMON_STRUCTURAL_INPUT_DIM,

        "node_initialization":
            node_init_df.to_dict(
                orient="records"
            ),

        "feature_encoding":
            encoding_df.to_dict(
                orient="records"
            ),

        "preprocessing_fit_scope":
            fit_scope_df.to_dict(
                orient="records"
            ),

        "missing_unknown_policy":
            missing_df.to_dict(
                orient="records"
            ),

        "evaluation_independence":
            evaluation_policy,

        "parameter_sharing":
            parameter_policy,

        "origin_specific_policy":
            origin_policy,

        "deferred_to_architecture_freeze":
            deferred,

        "baseline_exception":
            (
                "B0 remains the unchanged reproduced ITRS baseline "
                "and may retain its original Startup latent table and "
                "Phase-4 Doc2Vec document vectors. These mechanisms "
                "are prohibited in B1a/B1b/M1/M2."
            ),

        "inductive_definition":
            (
                "For B1a/B1b/M1/M2, a Startup or Person representation "
                "must be computable for a previously unseen entity ID "
                "using shared parameters and allowed side information "
                "only. No trainable or stored parameter may be indexed "
                "by Startup or Person identity."
            ),

        "feature_discovery_reopened":
            False,

        "graph_schema_modified":
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
                "Phase 8.1.11 — freeze the exact encoder and "
                "heterogeneous neural architecture."
            ),
    }

    write_json(
        CONTRACT_OUTPUT,
        contract,
    )

    manifest = {
        "phase":
            "8.1.10",

        "status":
            "FROZEN",

        "outputs": [
            str(NODE_INIT_OUTPUT),
            str(FEATURE_ENCODING_OUTPUT),
            str(FIT_SCOPE_OUTPUT),
            str(MISSINGNESS_OUTPUT),
            str(CONTRACT_OUTPUT),
        ],

        "feature_discovery_reopened":
            False,

        "graph_schema_modified":
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
        NODE_INIT_OUTPUT,
        FEATURE_ENCODING_OUTPUT,
        FIT_SCOPE_OUTPUT,
        MISSINGNESS_OUTPUT,
        CONTRACT_OUTPUT,
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
        "PHASE 8.1.10 IDENTITY-FREE "
        "ENCODING POLICY FROZEN"
    )

    separator()

    print()

    print(
        "COMMON STRUCTURAL INPUT DIM:     "
        f"{COMMON_STRUCTURAL_INPUT_DIM}"
    )

    print(
        "STARTUP ID PARAMETERS:           PROHIBITED"
    )

    print(
        "PERSON ID PARAMETERS:            PROHIBITED"
    )

    print(
        "INVESTOR ID PARAMETERS:          "
        "ALLOWED — WARM INVESTOR"
    )

    print(
        "EVAL-FITTED PREPROCESSING:       PROHIBITED"
    )

    print(
        "NEGATIVE-SAMPLE FEATURE FITTING: PROHIBITED"
    )

    print(
        "ORIGIN FEATURE STORE:            "
        "BUILD ONCE PER ORIGIN"
    )

    print()

    print(
        "B0 BASELINE MODIFIED:            NO"
    )

    print(
        "FEATURE DISCOVERY REOPENED:      NO"
    )

    print(
        "GRAPH SCHEMA MODIFIED:           NO"
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
        "Phase 8.1.11 — exact encoder + "
        "heterogeneous neural architecture freeze."
    )


if __name__ == "__main__":
    main()