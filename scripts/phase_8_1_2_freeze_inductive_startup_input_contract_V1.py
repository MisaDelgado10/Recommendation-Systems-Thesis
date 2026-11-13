from pathlib import Path
import hashlib
import json

import pandas as pd


# =============================================================================
# PHASE 8.1.2 — FREEZE INDUCTIVE STARTUP INPUT CONTRACT V1
# =============================================================================
#
# Scientific role:
#
#   PRE-ARCHITECTURE STARTUP INPUT CONTRACT FREEZE
#
# PURPOSE
# -------
#
# Freeze the information channels that MAY and MAY NOT be consumed by
# Phase-8 inductive startup models before architecture design.
#
# CENTRAL PRINCIPLE
# -----------------
#
# The Phase-8 startup representation must be computable without:
#
#   - a startup-ID-indexed learned embedding;
#   - a pre-trained per-document vector tied to the startup identity;
#   - prior startup investment history.
#
# Allowed startup content:
#
#   - frozen resolved Crunchbase description text;
#   - frozen Crunchbase category multi-hot information.
#
# Allowed structure:
#
#   - Phase-8.0 origin-specific acquisition relations;
#   - Phase-8.0 semantically time-eligible founder relation track.
#
# This phase DOES NOT:
#
#   - choose text architecture;
#   - choose graph architecture;
#   - choose hidden dimensions;
#   - train;
#   - infer;
#   - score;
#   - select models;
#   - inspect April performance.
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

PHASE80_CLOSURE = Path(
    "data/experimental/phase_8/"
    "phase_8_0_evaluation_reset/"
    "phase_8_0_13_phase_closure/"
    "phase_8_0_13_closure_manifest_V1.json"
)


PHASE811_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation/"
    "phase_8_1_1_input_availability_provenance"
)

PHASE811_SUMMARY = (
    PHASE811_DIR
    / "phase_8_1_1_audit_summary_V3.json"
)

PHASE811_PROVENANCE = (
    PHASE811_DIR
    / "phase_8_1_1_input_provenance_register_V3.csv"
)

PHASE811_PRIMARY_COVERAGE = (
    PHASE811_DIR
    / "phase_8_1_1_primary_feature_coverage_by_origin_V3.csv"
)

PHASE811_MANIFEST = (
    PHASE811_DIR
    / "phase_8_1_1_analysis_manifest_V3.json"
)


DESCRIPTION_MANIFEST = Path(
    "data/experimental/phase_4/"
    "description_contract/"
    "description_input_manifest.parquet"
)

DESCRIPTION_VOCABULARY = Path(
    "data/experimental/phase_4/"
    "description_contract/"
    "description_label_vocabulary.csv"
)

DESCRIPTION_CONTRACT = Path(
    "data/experimental/phase_4/"
    "description_contract/"
    "description_contract.json"
)

DOC2VEC_CONTRACT = Path(
    "data/experimental/phase_4/"
    "doc2vec_contract/"
    "doc2vec_contract.json"
)

DOC2VEC_STARTUP_VECTORS = Path(
    "data/experimental/phase_4/"
    "doc2vec/"
    "vectors/"
    "startup_doc2vec_vectors.npy"
)


PHASE806_POLICY = Path(
    "data/experimental/phase_8/"
    "phase_8_0_evaluation_reset/"
    "phase_8_0_6_dual_track_graph_policy/"
    "phase_8_0_6_graph_information_policy_V1.json"
)

PHASE806_MASKS = Path(
    "data/experimental/phase_8/"
    "phase_8_0_evaluation_reset/"
    "phase_8_0_6_dual_track_graph_policy/"
    "phase_8_0_6_dual_track_graph_masks_V1.npz"
)


PHASE812_DESIGN = Path(
    "data/experimental/phase_8/"
    "phase_8_0_evaluation_reset/"
    "phase_8_0_12_research_design_contract/"
    "phase_8_0_12_research_design_contract_V1.json"
)

PHASE812_BASELINES = Path(
    "data/experimental/phase_8/"
    "phase_8_0_evaluation_reset/"
    "phase_8_0_12_research_design_contract/"
    "phase_8_0_12_baseline_model_matrix_V1.csv"
)

PHASE812_ABLATIONS = Path(
    "data/experimental/phase_8/"
    "phase_8_0_evaluation_reset/"
    "phase_8_0_12_research_design_contract/"
    "phase_8_0_12_required_ablation_matrix_V1.csv"
)


# =============================================================================
# OUTPUTS
# =============================================================================

OUTPUT_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation/"
    "phase_8_1_2_input_contract"
)

ALLOWED_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_2_allowed_startup_input_register_V1.csv"
)

PROHIBITED_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_2_prohibited_startup_input_register_V1.csv"
)

TRACK_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_2_information_track_register_V1.csv"
)

MODEL_INPUT_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_2_model_family_input_matrix_V1.csv"
)

CONTRACT_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_2_inductive_startup_input_contract_V1.json"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_2_manifest_V1.json"
)

HASH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_2_derived_artifact_sha256_V1.json"
)


# =============================================================================
# EXPECTED AUDITED VALUES
# =============================================================================

EXPECTED_STARTUPS = 311_589
EXPECTED_LABEL_VOCABULARY = 802

EXPECTED_PRIMARY = {
    "T57": 6_850,
    "T58": 6_872,
    "T59": 6_764,
    "APRIL_CONFIRMATORY": 1_690,
}


# =============================================================================
# HELPERS
# =============================================================================

def separator(
    char="=",
    width=118,
):
    print(
        char * width
    )


def require(
    condition,
    message,
):
    if not condition:
        raise ValueError(
            message
        )


def sha256_file(
    path,
    chunk_size=1024 * 1024,
):
    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:

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


def read_json(
    path,
):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(
            f
        )


def write_json(
    path,
    obj,
):
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


# =============================================================================
# MAIN
# =============================================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    separator()

    print(
        "PHASE 8.1.2 — "
        "FREEZE INDUCTIVE STARTUP INPUT CONTRACT V1"
    )

    separator()

    print()

    print(
        "Scientific role:             "
        "PRE-ARCHITECTURE INPUT FREEZE"
    )

    print("Phase-8.0 reopened:          NO")
    print("Architecture selected:       NO")
    print("Model constructed:           NO")
    print("Training performed:          NO")
    print("Inference executed:          NO")
    print("Ranking metrics scored:      NO")
    print("Model selection performed:   NO")
    print("April performance inspected: NO")

    # =========================================================================
    # 1. Phase-8.0 closure gate
    # =========================================================================

    separator("-")

    print(
        "8.1.2.1 — PHASE-8.0 CLOSURE GATE"
    )

    separator("-")

    require(
        PHASE80_CLOSURE.exists(),
        "Missing Phase-8.0 closure manifest.",
    )

    closure = read_json(
        PHASE80_CLOSURE
    )

    require(
        closure.get(
            "phase_8_0_status"
        )
        ==
        "CLOSED",
        "Phase 8.0 is not closed.",
    )

    require(
        closure.get(
            "closure_gate"
        )
        ==
        "PASS",
        "Phase-8.0 closure gate did not pass.",
    )

    require(
        closure.get(
            "April_performance_inspected"
        )
        is False,
        "April performance unexpectedly inspected.",
    )

    print(
        "Phase 8.0: CLOSED PASS"
    )

    print(
        "April: UNTOUCHED PASS"
    )

    # =========================================================================
    # 2. Phase-8.1.1 prerequisite audit
    # =========================================================================

    separator("-")

    print(
        "8.1.2.2 — PHASE-8.1.1 "
        "INPUT-AUDIT GATE"
    )

    separator("-")

    prerequisite_paths = [
        PHASE811_SUMMARY,
        PHASE811_PROVENANCE,
        PHASE811_PRIMARY_COVERAGE,
        PHASE811_MANIFEST,
        DESCRIPTION_MANIFEST,
        DESCRIPTION_VOCABULARY,
        DESCRIPTION_CONTRACT,
        DOC2VEC_CONTRACT,
        DOC2VEC_STARTUP_VECTORS,
        PHASE806_POLICY,
        PHASE806_MASKS,
        PHASE812_DESIGN,
        PHASE812_BASELINES,
        PHASE812_ABLATIONS,
    ]

    source_hashes = {}

    for path in prerequisite_paths:

        require(
            path.exists(),
            f"Missing prerequisite artifact: {path}",
        )

        observed_hash = sha256_file(
            path
        )

        source_hashes[
            str(
                path
            )
        ] = observed_hash

        print(
            f"FOUND  {path}"
        )

        print(
            f"SHA256 {observed_hash}"
        )

    summary811 = read_json(
        PHASE811_SUMMARY
    )

    manifest811 = read_json(
        PHASE811_MANIFEST
    )

    require(
        summary811.get(
            "status"
        )
        ==
        "COMPLETE",
        "Phase-8.1.1 summary not complete.",
    )

    require(
        manifest811.get(
            "status"
        )
        ==
        "COMPLETE",
        "Phase-8.1.1 manifest not complete.",
    )

    require(
        summary811[
            "category_raw_source_field"
        ]
        ==
        "categories",
        (
            "Corrected Crunchbase category "
            "source was not frozen in V3."
        ),
    )

    require(
        summary811[
            "direct_snapshot_update_metadata_detected"
        ]
        is False,
        (
            "Unexpected direct snapshot-update "
            "metadata detected."
        ),
    )

    print()

    print(
        "Phase-8.1.1 V3: COMPLETE PASS"
    )

    # =========================================================================
    # 3. Verify raw resolved startup inputs
    # =========================================================================

    separator("-")

    print(
        "8.1.2.3 — RAW RESOLVED STARTUP "
        "CONTENT INPUTS"
    )

    separator("-")

    description = pd.read_parquet(
        DESCRIPTION_MANIFEST,
        columns=[
            "node_type",
            "raw_entity_id",
            "text",
            "text_source",
            "has_text",
            "has_labels",
            "label_count",
            "labels_json",
        ],
    )

    startup_description = (
        description[
            description[
                "node_type"
            ]
            ==
            "startup"
        ]
        .copy()
    )

    require(
        len(
            startup_description
        )
        ==
        EXPECTED_STARTUPS,
        "Startup description population drift.",
    )

    require(
        not startup_description[
            "raw_entity_id"
        ].astype(str)
        .duplicated()
        .any(),
        "Duplicate startup IDs in description manifest.",
    )

    vocabulary = pd.read_csv(
        DESCRIPTION_VOCABULARY
    )

    require(
        len(
            vocabulary
        )
        ==
        EXPECTED_LABEL_VOCABULARY,
        (
            "Frozen category vocabulary "
            "size drift."
        ),
    )

    print(
        f"Startup resolved documents: "
        f"{len(startup_description):,} PASS"
    )

    print(
        f"Frozen category vocabulary: "
        f"{len(vocabulary):,} PASS"
    )

    # =========================================================================
    # 4. Verify primary-stratum availability
    # =========================================================================

    separator("-")

    print(
        "8.1.2.4 — PRIMARY COLD-START "
        "CONTENT AVAILABILITY"
    )

    separator("-")

    primary_coverage = pd.read_csv(
        PHASE811_PRIMARY_COVERAGE
    )

    for origin, expected_n in (
        EXPECTED_PRIMARY.items()
    ):

        row = primary_coverage.loc[
            primary_coverage[
                "origin"
            ]
            ==
            origin
        ]

        require(
            len(
                row
            )
            ==
            1,
            f"{origin}: missing primary coverage row.",
        )

        row = row.iloc[0]

        require(
            int(
                row[
                    "rows"
                ]
            )
            ==
            expected_n,
            f"{origin}: primary population drift.",
        )

        text_coverage = float(
            row[
                "text_coverage_pct"
            ]
        )

        require(
            text_coverage
            ==
            100.0,
            (
                f"{origin}: expected complete "
                "primary raw-text coverage."
            ),
        )

        print(
            f"{origin}: "
            f"N={expected_n:,} | "
            f"text={text_coverage:.4f}% | "
            f"labels="
            f"{float(row['label_coverage_pct']):.4f}%"
        )

    # =========================================================================
    # 5. Freeze allowed startup information channels
    # =========================================================================

    separator("-")

    print(
        "8.1.2.5 — ALLOWED STARTUP "
        "INFORMATION CHANNELS"
    )

    separator("-")

    allowed = pd.DataFrame(
        [
            {
                "input_id":
                    "I1",

                "input_family":
                    "resolved_description_text",

                "exact_source":
                    (
                        "Phase-4 description_input_manifest."
                        "parquet::text"
                    ),

                "allowed_B1":
                    True,

                "allowed_M1":
                    True,

                "allowed_M2":
                    True,

                "temporal_class":
                    "MATCHED_SNAPSHOT_UNVERSIONED",

                "identity_indexed_parameter":
                    False,

                "primary_cold_start_available":
                    True,

                "contract_role":
                    "BASE_CONTENT_INPUT",

                "caveat":
                    (
                        "Exact historical text value at "
                        "each prediction origin is unknown."
                    ),
            },
            {
                "input_id":
                    "I2",

                "input_family":
                    "category_multihot",

                "exact_source":
                    (
                        "Phase-4 frozen 802-label "
                        "category vocabulary / labels_json"
                    ),

                "allowed_B1":
                    True,

                "allowed_M1":
                    True,

                "allowed_M2":
                    True,

                "temporal_class":
                    "MATCHED_SNAPSHOT_UNVERSIONED",

                "identity_indexed_parameter":
                    False,

                "primary_cold_start_available":
                    True,

                "contract_role":
                    "BASE_CONTENT_INPUT",

                "caveat":
                    (
                        "Missing labels remain an "
                        "all-zero category input; exact "
                        "historical category membership "
                        "is unknown."
                    ),
            },
            {
                "input_id":
                    "I3",

                "input_family":
                    "timestamped_acquisition_relations",

                "exact_source":
                    (
                        "Phase-8.0.6 origin-specific "
                        "strict_temporal graph mask"
                    ),

                "allowed_B1":
                    False,

                "allowed_M1":
                    True,

                "allowed_M2":
                    True,

                "temporal_class":
                    "TIMESTAMPED_PRE_ORIGIN",

                "identity_indexed_parameter":
                    False,

                "primary_cold_start_available":
                    "WHEN_STRUCTURALLY_CONNECTED",

                "contract_role":
                    "GRAPH_RELATION_INPUT",

                "caveat":
                    (
                        "Must use the exact frozen "
                        "origin-specific relation mask."
                    ),
            },
            {
                "input_id":
                    "I4",

                "input_family":
                    "shared_founder_relations",

                "exact_source":
                    (
                        "Phase-8.0.6 founder_proxy "
                        "origin-specific graph mask"
                    ),

                "allowed_B1":
                    False,

                "allowed_M1":
                    True,

                "allowed_M2":
                    True,

                "temporal_class":
                    (
                        "SNAPSHOT_RELATION_WITH_"
                        "ENTITY_EXISTENCE_GATING"
                    ),

                "identity_indexed_parameter":
                    False,

                "primary_cold_start_available":
                    "WHEN_STRUCTURALLY_CONNECTED",

                "contract_role":
                    "GRAPH_RELATION_INPUT",

                "caveat":
                    (
                        "Semantic eligibility is gated "
                        "by endpoint founded_on; database "
                        "observation time is unknown."
                    ),
            },
        ]
    )

    allowed.to_csv(
        ALLOWED_OUTPUT,
        index=False,
    )

    print(
        allowed[
            [
                "input_id",
                "input_family",
                "temporal_class",
                "contract_role",
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # 6. Freeze prohibited startup information channels
    # =========================================================================

    separator("-")

    print(
        "8.1.2.6 — PROHIBITED STARTUP "
        "INFORMATION CHANNELS"
    )

    separator("-")

    prohibited = pd.DataFrame(
        [
            {
                "input_id":
                    "P1",

                "input_family":
                    "startup_id_latent_embedding",

                "prohibited_B1":
                    True,

                "prohibited_M1":
                    True,

                "prohibited_M2":
                    True,

                "reason":
                    (
                        "Learned parameter indexed by "
                        "startup identity; transductive "
                        "and incompatible with the "
                        "primary inductive objective."
                    ),
            },
            {
                "input_id":
                    "P2",

                "input_family":
                    "phase4_doc2vec_document_vector",

                "prohibited_B1":
                    True,

                "prohibited_M1":
                    True,

                "prohibited_M2":
                    True,

                "reason":
                    (
                        "Phase-4 Doc2Vec vectors are "
                        "learned per-document model.dv "
                        "parameters tied to document tags; "
                        "they are not outputs of a reusable "
                        "startup-independent text encoder."
                    ),
            },
            {
                "input_id":
                    "P3",

                "input_family":
                    "startup_prior_investment_history",

                "prohibited_B1":
                    True,

                "prohibited_M1":
                    True,

                "prohibited_M2":
                    True,

                "reason":
                    (
                        "Primary thesis startups are "
                        "investment-cold by definition; "
                        "startup representation must not "
                        "depend on historical startup "
                        "investment events."
                    ),
            },
            {
                "input_id":
                    "P4",

                "input_family":
                    "founded_on_as_predictive_feature",

                "prohibited_B1":
                    True,

                "prohibited_M1":
                    True,

                "prohibited_M2":
                    True,

                "reason":
                    (
                        "founded_on remains frozen as "
                        "entity-existence / relation-"
                        "eligibility evidence. Using age "
                        "predictively would introduce a "
                        "new startup feature channel not "
                        "present in the controlled B0 "
                        "representation comparison."
                    ),
            },
            {
                "input_id":
                    "P5",

                "input_family":
                    "post_origin_interaction_information",

                "prohibited_B1":
                    True,

                "prohibited_M1":
                    True,

                "prohibited_M2":
                    True,

                "reason":
                    "Temporal target leakage.",
            },
            {
                "input_id":
                    "P6",

                "input_family":
                    "unfrozen_structural_edges",

                "prohibited_B1":
                    True,

                "prohibited_M1":
                    True,

                "prohibited_M2":
                    True,

                "reason":
                    (
                        "Graph inputs must come only "
                        "from frozen Phase-8.0.6 masks."
                    ),
            },
        ]
    )

    prohibited.to_csv(
        PROHIBITED_OUTPUT,
        index=False,
    )

    print(
        prohibited[
            [
                "input_id",
                "input_family",
                "reason",
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # 7. Freeze information-track terminology
    # =========================================================================

    separator("-")

    print(
        "8.1.2.7 — INFORMATION-TRACK "
        "TERMINOLOGY"
    )

    separator("-")

    tracks = pd.DataFrame(
        [
            {
                "track_id":
                    "T_CONTENT",

                "track_name":
                    "matched_snapshot_content",

                "content":
                    (
                        "resolved description text + "
                        "frozen category labels"
                    ),

                "historical_observability":
                    "NOT_PROVEN",

                "approved_claim":
                    (
                        "ID-independent matched-snapshot "
                        "startup content"
                    ),

                "prohibited_claim":
                    (
                        "strictly historical startup "
                        "content at every origin"
                    ),
            },
            {
                "track_id":
                    "T_ACQ",

                "track_name":
                    "strict_relation_provenance",

                "content":
                    (
                        "origin-specific timestamped "
                        "acquisition relations"
                    ),

                "historical_observability":
                    "TIMESTAMP_AUDITABLE",

                "approved_claim":
                    (
                        "strictly timestamped "
                        "relation-provenance track"
                    ),

                "prohibited_claim":
                    (
                        "entire model is strictly "
                        "temporal"
                    ),
            },
            {
                "track_id":
                    "T_FOUNDER",

                "track_name":
                    "founder_semantic_eligibility",

                "content":
                    (
                        "shared-founder relations with "
                        "both endpoint organizations "
                        "founded by prediction origin"
                    ),

                "historical_observability":
                    (
                        "SEMANTICALLY_GATED_"
                        "DATABASE_TIME_UNKNOWN"
                    ),

                "approved_claim":
                    (
                        "semantically time-eligible "
                        "founder side information"
                    ),

                "prohibited_claim":
                    (
                        "historically observed founder "
                        "edge in Crunchbase at origin"
                    ),
            },
        ]
    )

    tracks.to_csv(
        TRACK_OUTPUT,
        index=False,
    )

    print(
        tracks.to_string(
            index=False
        )
    )

    # =========================================================================
    # 8. Freeze model-family input matrix
    # =========================================================================

    separator("-")

    print(
        "8.1.2.8 — MODEL-FAMILY "
        "INPUT MATRIX"
    )

    separator("-")

    model_inputs = pd.DataFrame(
        [
            {
                "model_id":
                    "B0",

                "startup_id_embedding":
                    True,

                "phase4_doc2vec_document_vector":
                    True,

                "raw_resolved_text":
                    False,

                "category_multihot":
                    True,

                "graph_relations":
                    "ORIGINAL_PHASE8_COMPATIBLE_ITRS",

                "startup_investment_history":
                    "ORIGINAL_ITRS_SEMANTICS",

                "role":
                    "REPRODUCED_BASELINE_UNCHANGED",
            },
            {
                "model_id":
                    "B1",

                "startup_id_embedding":
                    False,

                "phase4_doc2vec_document_vector":
                    False,

                "raw_resolved_text":
                    True,

                "category_multihot":
                    True,

                "graph_relations":
                    "NONE",

                "startup_investment_history":
                    False,

                "role":
                    "INDUCTIVE_CONTENT_ONLY_BASELINE",
            },
            {
                "model_id":
                    "M1",

                "startup_id_embedding":
                    False,

                "phase4_doc2vec_document_vector":
                    False,

                "raw_resolved_text":
                    True,

                "category_multihot":
                    True,

                "graph_relations":
                    (
                        "PHASE8_FROZEN_RELATION_"
                        "CONFIGURATION"
                    ),

                "startup_investment_history":
                    False,

                "role":
                    (
                        "INDUCTIVE_HETEROGENEOUS_"
                        "STARTUP_MODEL"
                    ),
            },
            {
                "model_id":
                    "M2",

                "startup_id_embedding":
                    False,

                "phase4_doc2vec_document_vector":
                    False,

                "raw_resolved_text":
                    True,

                "category_multihot":
                    True,

                "graph_relations":
                    (
                        "SAME_AS_MATCHED_M1"
                    ),

                "startup_investment_history":
                    False,

                "role":
                    (
                        "M1_PLUS_FROZEN_ITRS_"
                        "TREND_INVESTOR_BRANCH"
                    ),
            },
        ]
    )

    model_inputs.to_csv(
        MODEL_INPUT_OUTPUT,
        index=False,
    )

    print(
        model_inputs.to_string(
            index=False
        )
    )

    # =========================================================================
    # 9. Freeze contract
    # =========================================================================

    separator("-")

    print(
        "8.1.2.9 — FREEZE INDUCTIVE "
        "STARTUP INPUT CONTRACT"
    )

    separator("-")

    contract = {
        "schema_version":
            "PHASE_8_1_2_INDUCTIVE_STARTUP_INPUT_CONTRACT_V1",

        "status":
            "FROZEN",

        "scientific_role":
            "PRE_ARCHITECTURE_INPUT_FREEZE",

        "primary_inductive_requirement":
            (
                "B1/M1/M2 startup representation must "
                "be computable without a parameter or "
                "stored vector indexed by startup identity."
            ),

        "base_startup_content_inputs": {
            "resolved_description_text": {
                "allowed":
                    True,

                "source":
                    str(
                        DESCRIPTION_MANIFEST
                    ),

                "column":
                    "text",

                "temporal_status":
                    "CURRENT_SNAPSHOT_UNVERSIONED",

                "primary_coverage":
                    "100_PERCENT_AT_ALL_FROZEN_ORIGINS",
            },

            "category_labels": {
                "allowed":
                    True,

                "source":
                    str(
                        DESCRIPTION_MANIFEST
                    ),

                "column":
                    "labels_json",

                "vocabulary_size":
                    802,

                "temporal_status":
                    "CURRENT_SNAPSHOT_UNVERSIONED",

                "missing_policy":
                    (
                        "all-zero category input; "
                        "no identity-specific fallback"
                    ),
            },
        },

        "prohibited_startup_identity_inputs": [
            "startup_id_latent_embedding",
            "phase4_doc2vec_document_vector",
        ],

        "phase4_doc2vec_classification": {
            "allowed_for_B0":
                True,

            "allowed_for_B1_M1_M2":
                False,

            "reason":
                (
                    "Document vectors were learned "
                    "per TaggedDocument/node tag and "
                    "retrieved through model.dv; "
                    "therefore they are transductive "
                    "document-specific representations."
                ),
        },

        "startup_investment_history": {
            "allowed_in_B1_M1_M2_startup_encoder":
                False,

            "reason":
                (
                    "Primary thesis population is "
                    "investment-cold."
                ),
        },

        "founded_on": {
            "predictive_feature_allowed":
                False,

            "allowed_uses": [
                "startup_origin_existence_eligibility",
                "founder_relation_semantic_eligibility",
            ],

            "reason":
                (
                    "Maintain controlled information "
                    "comparison rather than add an "
                    "additional startup-age predictor."
                ),
        },

        "graph_relations": {
            "acquisition":
                (
                    "allowed only through frozen "
                    "origin-specific Phase-8.0.6 masks"
                ),

            "shared_founder":
                (
                    "allowed only through frozen "
                    "founder_proxy masks and with "
                    "the frozen provenance caveat"
                ),

            "unfrozen_edges_allowed":
                False,
        },

        "information_claims": {
            "description_categories":
                (
                    "matched-snapshot ID-independent "
                    "content; exact historical values "
                    "are not claimed"
                ),

            "acquisition_relations":
                (
                    "strictly timestamped relation "
                    "provenance"
                ),

            "shared_founder_relations":
                (
                    "semantically time-eligible; "
                    "database observation timestamp "
                    "unknown"
                ),

            "entire_model_strictly_temporal_claim_allowed":
                False,
        },

        "missing_content_fallback": {
            "entity_specific_fallback_allowed":
                False,

            "learned_unknown_startup_id_embedding_allowed":
                False,
        },

        "investor_side_scope": {
            "startup_input_contract_changes_investor_side":
                False,

            "trend_branch":
                (
                    "handled only through frozen "
                    "Phase-8.0 M1/M2 trend ablation"
                ),
        },

        "text_encoder_architecture_frozen":
            False,

        "graph_encoder_architecture_frozen":
            False,

        "hidden_dimensions_frozen":
            False,

        "model_constructed":
            False,

        "training_performed":
            False,

        "ranking_metrics_scored":
            False,

        "April_performance_inspected":
            False,

        "next_step":
            (
                "Phase 8.1.3 — audit and freeze "
                "the inductive startup encoder "
                "architecture candidates."
            ),
    }

    write_json(
        CONTRACT_OUTPUT,
        contract,
    )

    print(
        "Raw resolved text:             ALLOWED"
    )

    print(
        "Category multi-hot:            ALLOWED"
    )

    print(
        "Startup-ID embedding:          PROHIBITED"
    )

    print(
        "Phase-4 Doc2Vec document vec:  PROHIBITED"
    )

    print(
        "Startup investment history:    PROHIBITED"
    )

    print(
        "founded_on predictive use:     PROHIBITED"
    )

    print(
        "Frozen graph relations:        ALLOWED"
    )

    # =========================================================================
    # 10. Manifest / hashes
    # =========================================================================

    separator("-")

    print(
        "8.1.2.10 — MANIFEST & FINGERPRINTS"
    )

    separator("-")

    manifest = {
        "phase":
            "8.1.2",

        "status":
            "FROZEN",

        "authoritative_inputs":
            source_hashes,

        "outputs": [
            str(
                ALLOWED_OUTPUT
            ),
            str(
                PROHIBITED_OUTPUT
            ),
            str(
                TRACK_OUTPUT
            ),
            str(
                MODEL_INPUT_OUTPUT
            ),
            str(
                CONTRACT_OUTPUT
            ),
        ],

        "prohibited_operations": [
            "architecture_selection",
            "model_construction",
            "training",
            "inference",
            "ranking_scoring",
            "model_selection",
            "April_performance_inspection",
        ],
    }

    write_json(
        MANIFEST_OUTPUT,
        manifest,
    )

    output_hashes = {
        str(
            ALLOWED_OUTPUT
        ):
            sha256_file(
                ALLOWED_OUTPUT
            ),

        str(
            PROHIBITED_OUTPUT
        ):
            sha256_file(
                PROHIBITED_OUTPUT
            ),

        str(
            TRACK_OUTPUT
        ):
            sha256_file(
                TRACK_OUTPUT
            ),

        str(
            MODEL_INPUT_OUTPUT
        ):
            sha256_file(
                MODEL_INPUT_OUTPUT
            ),

        str(
            CONTRACT_OUTPUT
        ):
            sha256_file(
                CONTRACT_OUTPUT
            ),

        str(
            MANIFEST_OUTPUT
        ):
            sha256_file(
                MANIFEST_OUTPUT
            ),
    }

    write_json(
        HASH_OUTPUT,
        output_hashes,
    )

    # =========================================================================
    # FINAL
    # =========================================================================

    separator()

    print(
        "PHASE 8.1.2 INDUCTIVE STARTUP "
        "INPUT CONTRACT FREEZE COMPLETE"
    )

    separator()

    print()

    print(
        "INPUT CONTRACT:                   FROZEN"
    )

    print(
        "RAW DESCRIPTION TEXT:             ALLOWED"
    )

    print(
        "CATEGORY MULTI-HOT:               ALLOWED"
    )

    print(
        "STARTUP-ID LEARNED EMBEDDING:     PROHIBITED"
    )

    print(
        "PHASE-4 DOC2VEC DOCUMENT VECTOR:  PROHIBITED"
    )

    print(
        "STARTUP INVESTMENT HISTORY:        PROHIBITED"
    )

    print(
        "FOUNDED_ON AS PREDICTOR:           PROHIBITED"
    )

    print(
        "ORIGIN-SAFE GRAPH RELATIONS:       ALLOWED"
    )

    print()

    print(
        "FULL MODEL STRICTLY TEMPORAL CLAIM:"
        " NOT ALLOWED"
    )

    print()

    print(
        "ARCHITECTURE FROZEN:               NO"
    )

    print(
        "MODEL CONSTRUCTED:                 NO"
    )

    print(
        "TRAINING PERFORMED:                NO"
    )

    print(
        "APRIL PERFORMANCE INSPECTED:       NO"
    )

    print()

    print(
        "Next:"
    )

    print(
        "Phase 8.1.3 — define and audit "
        "a small set of inductive startup "
        "encoder architecture candidates "
        "under this frozen information contract."
    )

    print()

    print(
        f"Outputs written to:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()