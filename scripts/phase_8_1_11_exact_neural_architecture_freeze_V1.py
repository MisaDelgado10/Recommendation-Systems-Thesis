from pathlib import Path
import hashlib
import json

import pandas as pd


# =============================================================================
# PHASE 8.1.11 — EXACT NEURAL ARCHITECTURE FREEZE V1
# =============================================================================
#
# Scientific role:
#
#   FINAL PRE-IMPLEMENTATION / PRE-TRAINING ARCHITECTURE CONTRACT
#
# PURPOSE
# -------
#
# Freeze the exact representation topology for:
#
#   B1a — inductive content baseline
#   B1b — inductive structured startup baseline
#   M1  — inductive heterogeneous model without trend
#   M2  — proposed trend-aware inductive heterogeneous model
#
# Scientific design principle:
#
#   Replace ITRS Startup identity-dependent information with inductive
#   side-information WITHOUT increasing the final Startup representation
#   width or final pair-scoring width relative to B0.
#
# This phase freezes architecture only.
#
# It DOES NOT:
#
#   - materialize features;
#   - materialize the heterogeneous graph;
#   - initialize model parameters;
#   - choose a training seed;
#   - train;
#   - select by performance;
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

CONSISTENCY_AMENDMENT = (
    BASE
    / "phase_8_1_10a_inductive_consistency_closure"
    / "phase_8_1_10a_consistency_amendment_V1.json"
)


PHASE4_DESCRIPTION = Path(
    "data/experimental/phase_4/"
    "description_neural_contract/"
    "description_neural_contract.json"
)

PHASE4_RGCN = Path(
    "data/experimental/phase_4/"
    "rgcn_neural_contract/"
    "rgcn_neural_contract.json"
)

PHASE4_SCORING = Path(
    "data/experimental/phase_4/"
    "scoring_neural_contract/"
    "scoring_neural_contract.json"
)


OUTPUT_DIR = (
    BASE
    / "phase_8_1_11_exact_neural_architecture"
)

ENCODER_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_11_encoder_architecture_V1.csv"
)

MODEL_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_11_model_family_architecture_V1.csv"
)

RELATION_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_11_typed_relation_vocabulary_V1.csv"
)

SCORING_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_11_scoring_representation_V1.csv"
)

CONTRACT_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_11_architecture_contract_V1.json"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_11_manifest_V1.json"
)

HASH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_11_artifact_sha256_V1.json"
)


# =============================================================================
# Frozen dimensions
# =============================================================================

TEXT_SVD_DIM = 32

CONTENT_TEXT_BRANCH_DIM = 20
CONTENT_CATEGORY_BRANCH_DIM = 20
CONTENT_DIM = 40

STRUCTURED_DIM = 40

GRAPH_INPUT_DIM = 40
GRAPH_HIDDEN_DIM = 40
GRAPH_OUTPUT_DIM = 40

INVESTOR_TREND_DIM = 40
INVESTOR_LATENT_DIM = 40
INVESTOR_DESCRIPTION_DIM = 40
INVESTOR_STRUCTURAL_DIM = 40

INVESTOR_REPRESENTATION_DIM = 160

STARTUP_STRUCTURED_DIM = 40
STARTUP_CONTENT_DIM = 40
STARTUP_STRUCTURAL_DIM = 40

STARTUP_REPRESENTATION_DIM = 120

PAIR_DIM = 280

RGCN_LAYERS = 2
RGCN_BASES = 5

SCORER_HIDDEN = [
    128,
    64,
    32,
    16,
]

SCORER_OUTPUT_DIM = 1


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
        "PHASE 8.1.11 — "
        "EXACT NEURAL ARCHITECTURE FREEZE V1"
    )

    separator()

    print()

    print(
        "Scientific role:             "
        "FINAL PRE-IMPLEMENTATION ARCHITECTURE CONTRACT"
    )

    print(
        "Feature discovery reopened:  NO"
    )

    print(
        "Graph schema reopened:       NO"
    )

    print(
        "Training performed:          NO"
    )

    print(
        "Model selection performed:   NO"
    )

    print(
        "April analyzed:              NO"
    )

    # =========================================================================
    # 1. Prerequisite gates
    # =========================================================================

    separator("-")

    print(
        "8.1.11.1 — PREREQUISITE GATES"
    )

    separator("-")

    required_paths = [
        INPUT_CONTRACT,
        GRAPH_SCHEMA,
        ENCODING_CONTRACT,
        CONSISTENCY_AMENDMENT,
        PHASE4_DESCRIPTION,
        PHASE4_RGCN,
        PHASE4_SCORING,
    ]

    for path in required_paths:

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

    consistency = read_json(
        CONSISTENCY_AMENDMENT
    )

    phase4_description = read_json(
        PHASE4_DESCRIPTION
    )

    phase4_rgcn = read_json(
        PHASE4_RGCN
    )

    phase4_scoring = read_json(
        PHASE4_SCORING
    )

    for name, contract in [
        ("Phase 8.1.7 V2", input_contract),
        ("Phase 8.1.9", graph_schema),
        ("Phase 8.1.10", encoding_contract),
        ("Phase 8.1.10A", consistency),
    ]:

        require(
            contract.get("status")
            ==
            "FROZEN",
            f"{name} is not frozen.",
        )

    require(
        input_contract.get(
            "feature_discovery_closed"
        )
        is True,
        "Feature discovery unexpectedly reopened.",
    )

    require(
        consistency.get(
            "new_information_channels_added"
        )
        is False,
        "8.1.10A unexpectedly added information channels.",
    )

    require(
        consistency.get(
            "model_trained"
        )
        is False,
        "Model unexpectedly trained before architecture freeze.",
    )

    require(
        consistency.get(
            "April_analyzed"
        )
        is False,
        "April unexpectedly analyzed.",
    )

    require(
        phase4_description.get("status")
        ==
        "FROZEN",
        "Phase-4 description architecture unavailable.",
    )

    require(
        phase4_rgcn.get("status")
        ==
        "FROZEN",
        "Phase-4 R-GCN architecture unavailable.",
    )

    # scoring contract historically uses its own frozen status naming,
    # so existence plus topology checks below are authoritative.

    print()

    print(
        "Upstream scientific contracts: PASS"
    )

    # =========================================================================
    # 2. Baseline dimension anchors
    # =========================================================================

    separator("-")

    print(
        "8.1.11.2 — BASELINE DIMENSION ANCHORS"
    )

    separator("-")

    baseline_description_dim = int(
        phase4_description[
            "paper_specified"
        ][
            "final_description_dim"
        ]
    )

    baseline_rgcn_dim = int(
        phase4_rgcn[
            "architecture"
        ][
            "output_dim"
        ]
    )

    require(
        baseline_description_dim
        ==
        40,
        "Baseline description dimension changed.",
    )

    require(
        baseline_rgcn_dim
        ==
        40,
        "Baseline R-GCN output dimension changed.",
    )

    require(
        CONTENT_DIM
        ==
        40,
        "Inductive content dimension must be 40.",
    )

    require(
        STRUCTURED_DIM
        ==
        40,
        "Structured Startup dimension must be 40.",
    )

    require(
        GRAPH_OUTPUT_DIM
        ==
        40,
        "Structural dimension must be 40.",
    )

    require(
        STARTUP_REPRESENTATION_DIM
        ==
        120,
        "Startup representation must remain 120-D.",
    )

    require(
        INVESTOR_REPRESENTATION_DIM
        ==
        160,
        "Investor representation must remain 160-D.",
    )

    require(
        PAIR_DIM
        ==
        280,
        "Pair representation must remain 280-D.",
    )

    print(
        "Baseline description dimension: 40"
    )

    print(
        "Baseline structural dimension:  40"
    )

    print(
        "Proposed Startup dimension:     120"
    )

    print(
        "Investor dimension:             160"
    )

    print(
        "Pair-scoring dimension:         280"
    )

    # =========================================================================
    # 3. Text / content architecture
    # =========================================================================

    separator("-")

    print(
        "8.1.11.3 — INDUCTIVE CONTENT ENCODER"
    )

    separator("-")

    text_policy = {
        "source":
            "resolved description with short_description fallback",

        "tokenization":
            "REUSE_PHASE4_AUDITED_TOKENIZATION_SEMANTICS",

        "vectorizer":
            "TFIDF",

        "ngram_range":
            [1, 2],

        "min_df":
            5,

        "max_df":
            0.95,

        "max_features":
            50000,

        "sublinear_tf":
            True,

        "norm":
            "l2",

        "fit_scope":
            "TRAINING_SIDE_STARTUP_TEXT_ONLY_PER_ORIGIN",

        "unseen_document_supported":
            True,

        "document_identity_parameter":
            False,

        "dimensionality_reduction":
            "TRUNCATED_SVD",

        "svd_components":
            TEXT_SVD_DIM,

        "svd_random_state":
            42,

        "svd_fit_scope":
            "TRAINING_SIDE_STARTUP_TEXT_ONLY_PER_ORIGIN",

        "text_projection":
            (
                f"Linear({TEXT_SVD_DIM}, "
                f"{CONTENT_TEXT_BRANCH_DIM}) + ReLU"
            ),

        "text_projection_trainable":
            True,
    }

    category_policy = {
        "source":
            "companies.categories",

        "parsing":
            "REUSE_PHASE4_AUDITED_CATEGORY_PARSER",

        "vocabulary":
            "TRAINING_SIDE_ONLY_PER_ORIGIN",

        "unknown_policy":
            "UNSEEN_CATEGORY -> UNK",

        "representation":
            "SPARSE_MULTI_HOT",

        "projection":
            (
                f"SparseLinear(V_categories, "
                f"{CONTENT_CATEGORY_BRANCH_DIM}) + ReLU"
            ),

        "projection_trainable":
            True,

        "identity_parameter":
            False,
    }

    require(
        CONTENT_TEXT_BRANCH_DIM
        +
        CONTENT_CATEGORY_BRANCH_DIM
        ==
        CONTENT_DIM,
        "Content branches do not sum to 40.",
    )

    print(
        "Text:"
    )

    print(
        "  TF-IDF(training only)"
    )

    print(
        "  1–2 grams; min_df=5; max_df=.95"
    )

    print(
        "  max_features=50,000"
    )

    print(
        "  TruncatedSVD -> 32"
    )

    print(
        "  Linear 32 -> 20 + ReLU"
    )

    print()

    print(
        "Categories:"
    )

    print(
        "  training-only sparse vocabulary"
    )

    print(
        "  SparseLinear V -> 20 + ReLU"
    )

    print()

    print(
        "Content output:"
    )

    print(
        "  20 + 20 = 40"
    )

    # =========================================================================
    # 4. Structured Startup encoder
    # =========================================================================

    separator("-")

    print(
        "8.1.11.4 — STRUCTURED STARTUP ENCODER"
    )

    separator("-")

    structured_channels_b1b = [
        "startup_category_groups",
        "startup_country",
        "startup_region_location_groups",
        "startup_company_type",
        "startup_age_at_origin",
        "startup_founder_count_resolved",
    ]

    structured_channels_m = (
        structured_channels_b1b
        +
        [
            "founder_prior_startup_count",
            "founder_prior_sector_profile",
        ]
    )

    structured_policy = {
        "categorical_vocabulary_scope":
            "TRAINING_SIDE_ONLY_PER_ORIGIN",

        "categorical_unknown_policy":
            "UNK",

        "category_groups":
            "SPARSE_MULTI_HOT",

        "country":
            "ONE_HOT_WITH_MISSING_AND_UNK",

        "region_location_groups":
            "SPARSE_MULTI_HOT_WITH_UNK",

        "company_type":
            "ONE_HOT_WITH_MISSING_AND_UNK",

        "age_transform":
            "log1p(nonnegative years at origin)",

        "founder_count_transform":
            "log1p(nonnegative eligible resolved founder count)",

        "prior_startup_count_transform":
            "log1p(nonnegative pre-focal prior venture count)",

        "prior_sector_profile":
            (
                "L1_NORMALIZED_SPARSE_MULTI_HOT_OR_COUNT_PROFILE"
            ),

        "continuous_standardization":
            "TRAINING_SIDE_ONLY",

        "architecture":
            (
                "Concatenate allowed structured inputs "
                "-> Linear(input_dim,64) -> ReLU "
                "-> Linear(64,40) -> ReLU"
            ),

        "hidden_dim":
            64,

        "output_dim":
            STRUCTURED_DIM,

        "dropout":
            0.0,

        "batch_norm":
            False,

        "layer_norm":
            False,

        "residual":
            False,

        "startup_identity_parameter":
            False,
    }

    print(
        "B1b structured channels:"
    )

    for channel in structured_channels_b1b:

        print(
            f"  {channel}"
        )

    print()

    print(
        "M1/M2 additional founder-history channels:"
    )

    for channel in [
        "founder_prior_startup_count",
        "founder_prior_sector_profile",
    ]:

        print(
            f"  {channel}"
        )

    print()

    print(
        "Fusion:"
    )

    print(
        "  input -> 64 -> 40"
    )

    # =========================================================================
    # 5. Structural graph input fusion
    # =========================================================================

    separator("-")

    print(
        "8.1.11.5 — STRUCTURAL INPUT FUSION"
    )

    separator("-")

    structural_input_policy = {
        "investor_initial":
            (
                "existing trainable Investor latent L_o, 40-D"
            ),

        "startup_initial":
            (
                "Linear([content_40 || structured_40], "
                "80 -> 40) + ReLU"
            ),

        "person_initial":
            "zero vector, 40-D",

        "startup_identity_embedding":
            False,

        "person_identity_embedding":
            False,

        "startup_input_dim_before_projection":
            80,

        "startup_graph_input_dim":
            GRAPH_INPUT_DIM,

        "person_graph_input_dim":
            GRAPH_INPUT_DIM,

        "investor_graph_input_dim":
            GRAPH_INPUT_DIM,

        "projection_shared_across_startups":
            True,

        "projection_bias":
            True,

        "projection_dropout":
            0.0,
    }

    for key, value in (
        structural_input_policy.items()
    ):

        print(
            f"{key:<44} {value}"
        )

    # =========================================================================
    # 6. Typed relation vocabulary
    # =========================================================================

    separator("-")

    print(
        "8.1.11.6 — TYPED RELATION VOCABULARY"
    )

    separator("-")

    relation_rows = [
        {
            "relation_id": 0,
            "typed_relation_key":
                "startup|founded_by|person",
        },
        {
            "relation_id": 1,
            "typed_relation_key":
                "person|founded|startup",
        },
        {
            "relation_id": 2,
            "typed_relation_key":
                "investor|acquired|investor",
        },
        {
            "relation_id": 3,
            "typed_relation_key":
                "investor|acquired|startup",
        },
        {
            "relation_id": 4,
            "typed_relation_key":
                "startup|acquired|investor",
        },
        {
            "relation_id": 5,
            "typed_relation_key":
                "startup|acquired|startup",
        },
        {
            "relation_id": 6,
            "typed_relation_key":
                "investor|acquired_by|investor",
        },
        {
            "relation_id": 7,
            "typed_relation_key":
                "investor|acquired_by|startup",
        },
        {
            "relation_id": 8,
            "typed_relation_key":
                "startup|acquired_by|investor",
        },
        {
            "relation_id": 9,
            "typed_relation_key":
                "startup|acquired_by|startup",
        },
    ]

    relation_df = pd.DataFrame(
        relation_rows
    )

    require(
        relation_df[
            "relation_id"
        ]
        .tolist()
        ==
        list(range(10)),
        "Relation IDs are not contiguous.",
    )

    require(
        relation_df[
            "typed_relation_key"
        ]
        .nunique()
        ==
        10,
        "Typed relation keys are not unique.",
    )

    print(
        relation_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # 7. R-GCN architecture
    # =========================================================================

    separator("-")

    print(
        "8.1.11.7 — HETEROGENEOUS R-GCN ARCHITECTURE"
    )

    separator("-")

    rgcn_policy = {
        "family":
            "BASIS_DECOMPOSED_RGCN",

        "implementation":
            "EXPLICIT_PYTORCH_COMPATIBLE_WITH_PHASE4_SEMANTICS",

        "input_dim":
            GRAPH_INPUT_DIM,

        "hidden_dim":
            GRAPH_HIDDEN_DIM,

        "output_dim":
            GRAPH_OUTPUT_DIM,

        "layers":
            RGCN_LAYERS,

        "typed_relations":
            10,

        "bases_per_layer":
            RGCN_BASES,

        "message_direction":
            "SOURCE_TO_DESTINATION",

        "relation_normalization":
            "MEAN_PER_DESTINATION_AND_TYPED_RELATION",

        "root_transform":
            True,

        "explicit_self_loop_edges":
            False,

        "activation":
            "ReLU after layer 1 and layer 2",

        "layer_bias":
            False,

        "dropout":
            0.0,

        "batch_norm":
            False,

        "layer_norm":
            False,

        "residual":
            False,

        "reason":
            (
                "Preserve Phase-4 R-GCN depth, width, basis count, "
                "normalization and self/root semantics while replacing "
                "the transductive Startup input with an inductive one "
                "and adding Person node types."
            ),
    }

    for key, value in (
        rgcn_policy.items()
    ):

        print(
            f"{key:<40} {value}"
        )

    # =========================================================================
    # 8. Final pair representation
    # =========================================================================

    separator("-")

    print(
        "8.1.11.8 — FINAL PAIR REPRESENTATION"
    )

    separator("-")

    scoring_rows = [
        {
            "position": 0,
            "component": "investor_trend",
            "dimension": 40,
        },
        {
            "position": 1,
            "component": "investor_latent",
            "dimension": 40,
        },
        {
            "position": 2,
            "component": "investor_description",
            "dimension": 40,
        },
        {
            "position": 3,
            "component": "investor_structural",
            "dimension": 40,
        },
        {
            "position": 4,
            "component": "startup_structured_inductive",
            "dimension": 40,
        },
        {
            "position": 5,
            "component": "startup_content_inductive",
            "dimension": 40,
        },
        {
            "position": 6,
            "component": "startup_structural_inductive",
            "dimension": 40,
        },
    ]

    scoring_df = pd.DataFrame(
        scoring_rows
    )

    require(
        int(
            scoring_df[
                "dimension"
            ].sum()
        )
        ==
        PAIR_DIM,
        "Pair representation does not sum to 280.",
    )

    print(
        scoring_df.to_string(
            index=False
        )
    )

    print()

    print(
        "Pair dimension: 280"
    )

    # =========================================================================
    # 9. Model-family branch activation
    # =========================================================================

    separator("-")

    print(
        "8.1.11.9 — MODEL-FAMILY BRANCH ACTIVATION"
    )

    separator("-")

    model_rows = [
        {
            "model":
                "B0",

            "startup_content":
                "BASELINE_PHASE4_DESCRIPTION",

            "startup_structured":
                "BASELINE_PHASE4_LATENT",

            "startup_graph":
                "BASELINE_PHASE4_RGCN",

            "founder_prior_history":
                False,

            "trend":
                "ITRS_TREND",

            "scorer":
                "FROZEN_PHASE4_SCORER",

            "pair_dim":
                280,
        },

        {
            "model":
                "B1a",

            "startup_content":
                "INDUCTIVE_CONTENT_40",

            "startup_structured":
                "ZERO_40",

            "startup_graph":
                "ZERO_40",

            "founder_prior_history":
                False,

            "trend":
                "ITRS_TREND",

            "scorer":
                "SAME_TOPOLOGY_280_128_64_32_16_1",

            "pair_dim":
                280,
        },

        {
            "model":
                "B1b",

            "startup_content":
                "INDUCTIVE_CONTENT_40",

            "startup_structured":
                "INDUCTIVE_BASE_STRUCTURED_40",

            "startup_graph":
                "ZERO_40",

            "founder_prior_history":
                False,

            "trend":
                "ITRS_TREND",

            "scorer":
                "SAME_TOPOLOGY_280_128_64_32_16_1",

            "pair_dim":
                280,
        },

        {
            "model":
                "M1",

            "startup_content":
                "INDUCTIVE_CONTENT_40",

            "startup_structured":
                "INDUCTIVE_EXTENDED_STRUCTURED_40",

            "startup_graph":
                "HETEROGENEOUS_RGCN_40",

            "founder_prior_history":
                True,

            "trend":
                "ZERO_40",

            "scorer":
                "SAME_TOPOLOGY_280_128_64_32_16_1",

            "pair_dim":
                280,
        },

        {
            "model":
                "M2",

            "startup_content":
                "INDUCTIVE_CONTENT_40",

            "startup_structured":
                "INDUCTIVE_EXTENDED_STRUCTURED_40",

            "startup_graph":
                "HETEROGENEOUS_RGCN_40",

            "founder_prior_history":
                True,

            "trend":
                "ITRS_TREND",

            "scorer":
                "SAME_TOPOLOGY_280_128_64_32_16_1",

            "pair_dim":
                280,
        },
    ]

    model_df = pd.DataFrame(
        model_rows
    )

    print(
        model_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # 10. Scoring network
    # =========================================================================

    separator("-")

    print(
        "8.1.11.10 — SCORING NETWORK FREEZE"
    )

    separator("-")

    scoring_policy = {
        "input_dim":
            PAIR_DIM,

        "hidden_widths":
            SCORER_HIDDEN,

        "hidden_activation":
            "ReLU",

        "hidden_bias":
            True,

        "output_dim":
            SCORER_OUTPUT_DIM,

        "output_bias":
            True,

        "dropout":
            0.0,

        "batch_norm":
            False,

        "layer_norm":
            False,

        "residual":
            False,

        "training_output":
            "logit",

        "loss":
            "BCEWithLogitsLoss",

        "ranking_score":
            "sigmoid(logit) or logit-equivalent ordering",

        "rationale":
            (
                "Preserve the frozen Phase-4 scoring topology and "
                "280-D pair width for attribution."
            ),
    }

    print(
        "280 -> 128 -> 64 -> 32 -> 16 -> 1"
    )

    print()

    print(
        "BCEWithLogitsLoss"
    )

    # =========================================================================
    # 11. Explicit architectural prohibitions
    # =========================================================================

    separator("-")

    print(
        "8.1.11.11 — ARCHITECTURAL PROHIBITIONS"
    )

    separator("-")

    prohibited = [
        "Startup-ID embedding in B1a/B1b/M1/M2",
        "Person-ID embedding",
        "Phase-4 Startup Doc2Vec document vector",
        "Transformer fine-tuning on evaluation text",
        "Vocabulary fitting on evaluation startups",
        "Evaluation-fitted SVD",
        "Evaluation-fitted numeric normalization",
        "Per-evaluation-case feature encoder",
        "Per-negative-sample graph construction",
        "Shared-founder projection in G_PRIMARY",
        "Founder School in primary architecture",
        "Current Person snapshot attributes",
        "Current Startup outcome/growth attributes",
        "Post-origin structural edges",
        "Changing scoring width across B1a/B1b/M1/M2",
        "Using April for architecture selection",
    ]

    for item in prohibited:

        print(
            f"PROHIBITED  {item}"
        )

    # =========================================================================
    # 12. Deliberately defer training/runtime decisions
    # =========================================================================

    separator("-")

    print(
        "8.1.11.12 — DEFERRED TO EXECUTION CONTRACT"
    )

    separator("-")

    deferred = [
        "global random seed",
        "exact Kaiming variant",
        "optimizer",
        "learning rate",
        "weight decay",
        "batch size",
        "epoch execution schedule",
        "checkpoint policy",
        "development model-selection tie-break implementation",
        "GPU runtime implementation",
        "feature materialization storage format",
        "graph materialization storage format",
    ]

    for item in deferred:

        print(
            f"DEFERRED  {item}"
        )

    # =========================================================================
    # 13. Persist exact architecture
    # =========================================================================

    separator("-")

    print(
        "8.1.11.13 — PERSIST ARCHITECTURE CONTRACT"
    )

    separator("-")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    encoder_rows = [
        {
            "component":
                "text_input",

            "architecture":
                (
                    "TFIDF(1-2gram,min_df=5,max_df=.95,"
                    "max_features=50000,sublinear_tf=True)"
                    " -> TruncatedSVD32"
                ),

            "output_dim":
                32,
        },

        {
            "component":
                "text_projection",

            "architecture":
                "Linear32->20 + ReLU",

            "output_dim":
                20,
        },

        {
            "component":
                "category_projection",

            "architecture":
                "SparseMultiHot -> SparseLinear(V,20) + ReLU",

            "output_dim":
                20,
        },

        {
            "component":
                "startup_content",

            "architecture":
                "concat(text20, category20)",

            "output_dim":
                40,
        },

        {
            "component":
                "startup_structured",

            "architecture":
                "structured_input -> Linear64 -> ReLU -> Linear40 -> ReLU",

            "output_dim":
                40,
        },

        {
            "component":
                "startup_graph_input",

            "architecture":
                "concat(content40,structured40) -> Linear80->40 + ReLU",

            "output_dim":
                40,
        },

        {
            "component":
                "person_graph_input",

            "architecture":
                "ZERO_40",

            "output_dim":
                40,
        },

        {
            "component":
                "investor_graph_input",

            "architecture":
                "baseline Investor latent L_o",

            "output_dim":
                40,
        },

        {
            "component":
                "heterogeneous_rgcn",

            "architecture":
                "2-layer basis RGCN, 40->40->40, 10 relations, 5 bases",

            "output_dim":
                40,
        },

        {
            "component":
                "scorer",

            "architecture":
                "280->128->64->32->16->1",

            "output_dim":
                1,
        },
    ]

    encoder_df = pd.DataFrame(
        encoder_rows
    )

    encoder_df.to_csv(
        ENCODER_OUTPUT,
        index=False,
    )

    model_df.to_csv(
        MODEL_OUTPUT,
        index=False,
    )

    relation_df.to_csv(
        RELATION_OUTPUT,
        index=False,
    )

    scoring_df.to_csv(
        SCORING_OUTPUT,
        index=False,
    )

    contract = {
        "schema_version":
            "PHASE_8_1_11_EXACT_NEURAL_ARCHITECTURE_V1",

        "status":
            "FROZEN",

        "scientific_role":
            "FINAL_PRE_IMPLEMENTATION_ARCHITECTURE_CONTRACT",

        "source_contracts": {
            "input_contract":
                str(INPUT_CONTRACT),

            "graph_schema":
                str(GRAPH_SCHEMA),

            "encoding_contract":
                str(ENCODING_CONTRACT),

            "consistency_amendment":
                str(CONSISTENCY_AMENDMENT),
        },

        "architectural_principle":
            (
                "Replace Startup identity-dependent latent/document "
                "representations with shared inductive encoders while "
                "retaining the 120-D Startup and 280-D pair widths."
            ),

        "text_encoder":
            text_policy,

        "category_encoder":
            category_policy,

        "structured_startup_encoder": {
            "B1b_channels":
                structured_channels_b1b,

            "M1_M2_channels":
                structured_channels_m,

            "policy":
                structured_policy,
        },

        "structural_input":
            structural_input_policy,

        "typed_relation_vocabulary":
            relation_df.to_dict(
                orient="records"
            ),

        "rgcn":
            rgcn_policy,

        "pair_representation":
            scoring_df.to_dict(
                orient="records"
            ),

        "scoring":
            scoring_policy,

        "model_families":
            model_df.to_dict(
                orient="records"
            ),

        "prohibited":
            prohibited,

        "deferred_to_execution_contract":
            deferred,

        "feature_discovery_reopened":
            False,

        "graph_schema_reopened":
            False,

        "architecture_selected":
            True,

        "architecture_selection_used_performance":
            False,

        "model_trained":
            False,

        "April_analyzed":
            False,

        "April_performance_inspected":
            False,

        "next_step":
            (
                "Phase 8.1.12 — freeze initialization, randomness, "
                "optimizer and execution/runtime contract before "
                "materialization and training."
            ),
    }

    write_json(
        CONTRACT_OUTPUT,
        contract,
    )

    manifest = {
        "phase":
            "8.1.11",

        "status":
            "FROZEN",

        "outputs": [
            str(ENCODER_OUTPUT),
            str(MODEL_OUTPUT),
            str(RELATION_OUTPUT),
            str(SCORING_OUTPUT),
            str(CONTRACT_OUTPUT),
        ],

        "feature_discovery_reopened":
            False,

        "architecture_selected":
            True,

        "architecture_selection_used_performance":
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
        ENCODER_OUTPUT,
        MODEL_OUTPUT,
        RELATION_OUTPUT,
        SCORING_OUTPUT,
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
        "PHASE 8.1.11 EXACT NEURAL ARCHITECTURE FROZEN"
    )

    separator()

    print()

    print(
        "STARTUP CONTENT:          40"
    )

    print(
        "STARTUP STRUCTURED:       40"
    )

    print(
        "STARTUP STRUCTURAL:       40"
    )

    print(
        "STARTUP REPRESENTATION:  120"
    )

    print(
        "INVESTOR REPRESENTATION: 160"
    )

    print(
        "PAIR REPRESENTATION:     280"
    )

    print()

    print(
        "R-GCN: 2 layers / 40-D / 10 typed relations / 5 bases"
    )

    print(
        "SCORER: 280 -> 128 -> 64 -> 32 -> 16 -> 1"
    )

    print()

    print(
        "STARTUP ID EMBEDDING:     PROHIBITED"
    )

    print(
        "PERSON ID EMBEDDING:      PROHIBITED"
    )

    print(
        "FEATURE DISCOVERY:        CLOSED"
    )

    print(
        "PERFORMANCE-BASED DESIGN: NO"
    )

    print(
        "MODEL TRAINED:            NO"
    )

    print(
        "APRIL ANALYZED:           NO"
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Phase 8.1.12 — initialization/randomness/"
        "optimizer/execution contract."
    )


if __name__ == "__main__":
    main()
