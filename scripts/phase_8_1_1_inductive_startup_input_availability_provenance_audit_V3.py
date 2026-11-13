from pathlib import Path
import hashlib
import json

import pandas as pd


# =============================================================================
# PHASE 8.1.1 — INDUCTIVE STARTUP INPUT AVAILABILITY & PROVENANCE AUDIT V3
# =============================================================================
#
# Scientific role:
#
#   PRE-MODEL INPUT AVAILABILITY / TEMPORAL-PROVENANCE AUDIT ONLY
#
# PURPOSE
# -------
#
# Determine which existing startup-side information can support an
# ID-independent inductive representation under the frozen Phase-8 protocol.
#
# This audit examines:
#
#   1. frozen Phase-4 description/text coverage;
#   2. frozen Phase-4 category-label coverage;
#   3. Doc2Vec feature availability;
#   4. coverage in each Phase-8 candidate universe;
#   5. coverage among Phase-8 evaluation targets;
#   6. coverage specifically in the primary cold-start thesis stratum;
#   7. temporal provenance of candidate input families.
#
# IMPORTANT
# ---------
#
# This phase does NOT freeze the final model-input contract.
#
# It does NOT:
#
#   - construct a model;
#   - train;
#   - infer;
#   - score rankings;
#   - select architecture;
#   - change Phase-8.0;
#   - inspect April performance.
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

RAW_DIR = Path(
    "data/raw"
)


PHASE80_CLOSURE = Path(
    "data/experimental/phase_8/"
    "phase_8_0_evaluation_reset/"
    "phase_8_0_13_phase_closure/"
    "phase_8_0_13_closure_manifest_V1.json"
)


DESCRIPTION_MANIFEST_PATH = Path(
    "data/experimental/phase_4/"
    "description_contract/"
    "description_input_manifest.parquet"
)

DOC2VEC_MANIFEST_PATH = Path(
    "data/experimental/phase_4/"
    "doc2vec/"
    "vectors/"
    "doc2vec_vector_manifest.parquet"
)


PHASE809_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_0_evaluation_reset/"
    "phase_8_0_9_origin_eligibility_candidate_universe"
)

CANDIDATE_UNIVERSE_PATH = (
    PHASE809_DIR
    / "phase_8_0_9_origin_candidate_universe_V5.parquet"
)

ELIGIBLE_TARGETS_PATH = (
    PHASE809_DIR
    / "phase_8_0_9_origin_eligible_targets_V5.parquet"
)

STARTUP_EXISTENCE_REGISTRY_PATH = (
    PHASE809_DIR
    / "phase_8_0_9_startup_existence_registry_V5.parquet"
)


PHASE806_POLICY_PATH = Path(
    "data/experimental/phase_8/"
    "phase_8_0_evaluation_reset/"
    "phase_8_0_6_dual_track_graph_policy/"
    "phase_8_0_6_graph_information_policy_V1.json"
)


OUTPUT_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation/"
    "phase_8_1_1_input_availability_provenance"
)

GLOBAL_COVERAGE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_1_global_startup_feature_coverage_V3.csv"
)

CANDIDATE_COVERAGE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_1_candidate_feature_coverage_by_origin_V3.csv"
)

TARGET_COVERAGE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_1_target_feature_coverage_by_origin_V3.csv"
)

PRIMARY_COVERAGE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_1_primary_feature_coverage_by_origin_V3.csv"
)

PROVENANCE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_1_input_provenance_register_V3.csv"
)

RAW_SCHEMA_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_1_company_schema_temporal_metadata_audit_V3.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_1_audit_summary_V3.json"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_1_analysis_manifest_V3.json"
)

HASH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_1_derived_artifact_sha256_V3.json"
)


# =============================================================================
# FROZEN EXPECTATIONS
# =============================================================================

EXPECTED_STARTUPS = 311_589

EXPECTED_STARTUPS_WITH_TEXT = 311_586

EXPECTED_STARTUPS_WITH_LABELS = 303_688

EXPECTED_ORIGINS = [
    "T57",
    "T58",
    "T59",
    "APRIL_CONFIRMATORY",
]

EXPECTED_CANDIDATE_COUNTS = {
    "T57": 308_027,
    "T58": 308_854,
    "T59": 309_611,
    "APRIL_CONFIRMATORY": 311_266,
}

EXPECTED_TARGET_COUNTS = {
    "T57": 21_924,
    "T58": 22_384,
    "T59": 22_650,
    "APRIL_CONFIRMATORY": 6_313,
}

EXPECTED_PRIMARY_COUNTS = {
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


def pct(
    numerator,
    denominator,
):

    if denominator == 0:

        return float("nan")

    return (
        100.0
        * float(
            numerator
        )
        / float(
            denominator
        )
    )


def feature_summary_row(
    label,
    frame,
):

    n = len(
        frame
    )

    text_n = int(
        frame[
            "has_text"
        ].sum()
    )

    labels_n = int(
        frame[
            "has_labels"
        ].sum()
    )

    doc2vec_nonzero_n = int(
        (
            ~frame[
                "doc2vec_zero_vector"
            ]
        ).sum()
    )

    both_n = int(
        (
            frame[
                "has_text"
            ]
            &
            frame[
                "has_labels"
            ]
        ).sum()
    )

    neither_n = int(
        (
            ~frame[
                "has_text"
            ]
            &
            ~frame[
                "has_labels"
            ]
        ).sum()
    )

    return {
        "population":
            label,

        "rows":
            n,

        "with_text":
            text_n,

        "text_coverage_pct":
            pct(
                text_n,
                n,
            ),

        "with_labels":
            labels_n,

        "label_coverage_pct":
            pct(
                labels_n,
                n,
            ),

        "with_text_and_labels":
            both_n,

        "text_and_label_coverage_pct":
            pct(
                both_n,
                n,
            ),

        "doc2vec_nonzero":
            doc2vec_nonzero_n,

        "doc2vec_nonzero_pct":
            pct(
                doc2vec_nonzero_n,
                n,
            ),

        "with_neither_text_nor_labels":
            neither_n,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    separator()

    print(
        "PHASE 8.1.1 — "
        "INDUCTIVE STARTUP INPUT AVAILABILITY "
        "& TEMPORAL-PROVENANCE AUDIT V3"
    )

    separator()

    print()

    print(
        "Scientific role:             "
        "PRE-MODEL INPUT AUDIT ONLY"
    )

    print(
        "Phase-8.0 reopened:          NO"
    )

    print(
        "Model constructed:           NO"
    )

    print(
        "Model trained:               NO"
    )

    print(
        "Checkpoint loaded:           NO"
    )

    print(
        "Inference executed:          NO"
    )

    print(
        "Ranking metrics scored:      NO"
    )

    print(
        "Architecture selected:       NO"
    )

    print(
        "April performance inspected: NO"
    )

    # =========================================================================
    # 1. Phase-8.0 closure gate
    # =========================================================================

    separator("-")

    print(
        "8.1.1.1 — PHASE-8.0 "
        "CLOSURE GATE"
    )

    separator("-")

    require(
        PHASE80_CLOSURE.exists(),
        (
            "Missing Phase-8.0 closure manifest."
        ),
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
        (
            "Phase 8.0 is not formally closed."
        ),
    )

    require(
        closure.get(
            "closure_gate"
        )
        ==
        "PASS",
        (
            "Phase-8.0 closure gate did not pass."
        ),
    )

    require(
        closure.get(
            "April_performance_inspected"
        )
        is False,
        (
            "April performance was unexpectedly "
            "marked as inspected."
        ),
    )

    require(
        closure.get(
            "new_model_training_started"
        )
        is False,
        (
            "Phase-8.0 closure claims new-model "
            "training already started."
        ),
    )

    print(
        "Phase 8.0 status: CLOSED PASS"
    )

    print(
        "April performance: UNTOUCHED PASS"
    )

    # =========================================================================
    # 2. Authoritative input fingerprints
    # =========================================================================

    separator("-")

    print(
        "8.1.1.2 — AUTHORITATIVE "
        "INPUT FINGERPRINTS"
    )

    separator("-")

    required_paths = [
        DESCRIPTION_MANIFEST_PATH,
        DOC2VEC_MANIFEST_PATH,
        CANDIDATE_UNIVERSE_PATH,
        ELIGIBLE_TARGETS_PATH,
        STARTUP_EXISTENCE_REGISTRY_PATH,
        PHASE806_POLICY_PATH,
    ]

    source_hashes = {}

    for path in required_paths:

        require(
            path.exists(),
            f"Missing required input: {path}",
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

    # =========================================================================
    # 3. Frozen Phase-4 feature manifests
    # =========================================================================

    separator("-")

    print(
        "8.1.1.3 — FROZEN PHASE-4 "
        "DESCRIPTION INPUTS"
    )

    separator("-")

    description = pd.read_parquet(
        DESCRIPTION_MANIFEST_PATH,
        columns=[
            "node_type",
            "raw_entity_id",
            "has_text",
            "text_source",
            "has_labels",
            "label_count",
        ],
    )

    doc2vec = pd.read_parquet(
        DOC2VEC_MANIFEST_PATH,
        columns=[
            "node_type",
            "raw_entity_id",
            "doc2vec_zero_vector",
        ],
    )

    description[
        "raw_entity_id"
    ] = (
        description[
            "raw_entity_id"
        ].astype(str)
    )

    doc2vec[
        "raw_entity_id"
    ] = (
        doc2vec[
            "raw_entity_id"
        ].astype(str)
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

    startup_doc2vec = (
        doc2vec[
            doc2vec[
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
        (
            "Startup description-manifest "
            "population drift."
        ),
    )

    require(
        len(
            startup_doc2vec
        )
        ==
        EXPECTED_STARTUPS,
        (
            "Startup Doc2Vec-manifest "
            "population drift."
        ),
    )

    require(
        not startup_description[
            "raw_entity_id"
        ].duplicated().any(),
        (
            "Duplicate Startup IDs in "
            "description manifest."
        ),
    )

    require(
        not startup_doc2vec[
            "raw_entity_id"
        ].duplicated().any(),
        (
            "Duplicate Startup IDs in "
            "Doc2Vec manifest."
        ),
    )

    text_count = int(
        startup_description[
            "has_text"
        ].sum()
    )

    label_count = int(
        startup_description[
            "has_labels"
        ].sum()
    )

    require(
        text_count
        ==
        EXPECTED_STARTUPS_WITH_TEXT,
        (
            "Startup text coverage drift."
        ),
    )

    require(
        label_count
        ==
        EXPECTED_STARTUPS_WITH_LABELS,
        (
            "Startup label coverage drift."
        ),
    )

    print(
        f"Startup role nodes:       "
        f"{len(startup_description):,}"
    )

    print(
        f"Startups with text:       "
        f"{text_count:,} "
        f"({pct(text_count, EXPECTED_STARTUPS):.4f}%)"
    )

    print(
        f"Startups with categories: "
        f"{label_count:,} "
        f"({pct(label_count, EXPECTED_STARTUPS):.4f}%)"
    )

    # =========================================================================
    # 4. Build unified ID-independent feature registry
    # =========================================================================

    separator("-")

    print(
        "8.1.1.4 — STARTUP FEATURE "
        "AVAILABILITY REGISTRY"
    )

    separator("-")

    feature_registry = (
        startup_description[
            [
                "raw_entity_id",
                "has_text",
                "text_source",
                "has_labels",
                "label_count",
            ]
        ]
        .rename(
            columns={
                "raw_entity_id":
                    "startup_id"
            }
        )
        .merge(
            startup_doc2vec[
                [
                    "raw_entity_id",
                    "doc2vec_zero_vector",
                ]
            ].rename(
                columns={
                    "raw_entity_id":
                        "startup_id"
                }
            ),
            on="startup_id",
            how="inner",
            validate="one_to_one",
        )
    )

    require(
        len(
            feature_registry
        )
        ==
        EXPECTED_STARTUPS,
        (
            "Unified feature registry "
            "population drift."
        ),
    )

    global_summary = pd.DataFrame(
        [
            feature_summary_row(
                "GLOBAL_STARTUP_ROLE_UNIVERSE",
                feature_registry,
            )
        ]
    )

    print(
        global_summary.to_string(
            index=False
        )
    )

    # =========================================================================
    # 5. Frozen Phase-8 populations
    # =========================================================================

    separator("-")

    print(
        "8.1.1.5 — PHASE-8 ORIGIN "
        "CANDIDATE FEATURE COVERAGE"
    )

    separator("-")

    candidates = pd.read_parquet(
        CANDIDATE_UNIVERSE_PATH,
        columns=[
            "origin",
            "startup_id",
        ],
    )

    targets = pd.read_parquet(
        ELIGIBLE_TARGETS_PATH,
        columns=[
            "origin",
            "startup_id_str",
            "is_primary_stratum",
        ],
    )

    candidates[
        "startup_id"
    ] = (
        candidates[
            "startup_id"
        ].astype(str)
    )

    targets[
        "startup_id_str"
    ] = (
        targets[
            "startup_id_str"
        ].astype(str)
    )

    candidate_rows = []
    target_rows = []
    primary_rows = []

    feature_lookup = (
        feature_registry
        .set_index(
            "startup_id"
        )
    )

    for origin in (
        EXPECTED_ORIGINS
    ):

        # ---------------------------------------------------------------------
        # Candidate universe
        # ---------------------------------------------------------------------

        origin_candidates = (
            candidates[
                candidates[
                    "origin"
                ]
                ==
                origin
            ]
            .copy()
        )

        require(
            len(
                origin_candidates
            )
            ==
            EXPECTED_CANDIDATE_COUNTS[
                origin
            ],
            (
                f"{origin}: candidate "
                "population drift."
            ),
        )

        candidate_features = (
            origin_candidates
            .join(
                feature_lookup,
                on="startup_id",
                how="left",
            )
        )

        require(
            not candidate_features[
                "has_text"
            ].isna().any(),
            (
                f"{origin}: at least one "
                "candidate startup is absent "
                "from the feature registry."
            ),
        )

        candidate_row = (
            feature_summary_row(
                origin,
                candidate_features,
            )
        )

        candidate_row[
            "origin"
        ] = origin

        candidate_rows.append(
            candidate_row
        )

        # ---------------------------------------------------------------------
        # All eligible targets
        # ---------------------------------------------------------------------

        origin_targets = (
            targets[
                targets[
                    "origin"
                ]
                ==
                origin
            ]
            .copy()
        )

        require(
            len(
                origin_targets
            )
            ==
            EXPECTED_TARGET_COUNTS[
                origin
            ],
            (
                f"{origin}: eligible-target "
                "population drift."
            ),
        )

        target_features = (
            origin_targets
            .join(
                feature_lookup,
                on="startup_id_str",
                how="left",
            )
        )

        require(
            not target_features[
                "has_text"
            ].isna().any(),
            (
                f"{origin}: at least one "
                "target startup is absent "
                "from feature registry."
            ),
        )

        target_row = (
            feature_summary_row(
                origin,
                target_features,
            )
        )

        target_row[
            "origin"
        ] = origin

        target_rows.append(
            target_row
        )

        # ---------------------------------------------------------------------
        # Primary thesis stratum
        # ---------------------------------------------------------------------

        primary = target_features[
            target_features[
                "is_primary_stratum"
            ]
        ].copy()

        require(
            len(
                primary
            )
            ==
            EXPECTED_PRIMARY_COUNTS[
                origin
            ],
            (
                f"{origin}: primary "
                "population drift."
            ),
        )

        primary_row = (
            feature_summary_row(
                origin,
                primary,
            )
        )

        primary_row[
            "origin"
        ] = origin

        primary_rows.append(
            primary_row
        )

        print()

        print(
            f"{origin}"
        )

        print(
            f"  candidates:              "
            f"{len(candidate_features):,}"
        )

        print(
            f"  candidate text coverage: "
            f"{candidate_row['text_coverage_pct']:.4f}%"
        )

        print(
            f"  candidate label coverage:"
            f" {candidate_row['label_coverage_pct']:.4f}%"
        )

        print(
            f"  eligible targets:        "
            f"{len(target_features):,}"
        )

        print(
            f"  primary cold-start:      "
            f"{len(primary):,}"
        )

        print(
            f"  PRIMARY text coverage:   "
            f"{primary_row['text_coverage_pct']:.4f}%"
        )

        print(
            f"  PRIMARY label coverage:  "
            f"{primary_row['label_coverage_pct']:.4f}%"
        )

        print(
            f"  PRIMARY Doc2Vec nonzero: "
            f"{primary_row['doc2vec_nonzero_pct']:.4f}%"
        )

    candidate_coverage = pd.DataFrame(
        candidate_rows
    )

    target_coverage = pd.DataFrame(
        target_rows
    )

    primary_coverage = pd.DataFrame(
        primary_rows
    )

    # =========================================================================
    # 6. Raw company schema / temporal metadata
    # =========================================================================

    separator("-")

    print(
        "8.1.1.6 — RAW COMPANY ATTRIBUTE "
        "TEMPORAL-PROVENANCE AUDIT"
    )

    separator("-")

    company_files = sorted(
        RAW_DIR.glob(
            "companies*.csv"
        )
    )

    require(
        len(
            company_files
        )
        ==
        24,
        (
            "Expected 24 company CSV files."
        ),
    )

    # Exact raw fields used / audited by the Phase-4 description
    # reconstruction. The Crunchbase source field is `categories`,
    # NOT `category_list`.
    required_static_fields = [
        "id",
        "description",
        "short_description",
        "categories",
        "founded_on",
    ]

    temporal_metadata_candidates = [
        "created_at",
        "updated_at",
        "modified_at",
        "last_updated_at",
        "description_updated_at",
        "category_updated_at",
    ]

    schema_rows = []

    union_columns = set()

    for path in (
        company_files
    ):

        header = pd.read_csv(
            path,
            nrows=0,
        )

        columns = set(
            header.columns
        )

        union_columns.update(
            columns
        )

        row = {
            "source_file":
                path.name,

            "column_count":
                len(
                    columns
                ),
        }

        for field in (
            required_static_fields
        ):

            row[
                f"has_{field}"
            ] = (
                field
                in
                columns
            )

        for field in (
            temporal_metadata_candidates
        ):

            row[
                f"has_{field}"
            ] = (
                field
                in
                columns
            )

        schema_rows.append(
            row
        )

    schema_audit = pd.DataFrame(
        schema_rows
    )

    for field in (
        required_static_fields
    ):

        count_present = int(
            schema_audit[
                f"has_{field}"
            ].sum()
        )

        print(
            f"{field:<24} "
            f"{count_present}/24 files"
        )

        require(
            count_present
            ==
            len(
                company_files
            ),
            (
                f"Required raw Company field "
                f"{field!r} is not present in "
                f"all {len(company_files)} files."
            ),
        )

    print()

    print(
        "Exact Phase-4 category source: categories"
    )

    print(
        "Legacy category_list assumption: NOT USED"
    )

    print()

    print(
        "Direct snapshot-update metadata:"
    )

    any_direct_update_metadata = False

    for field in (
        temporal_metadata_candidates
    ):

        count_present = int(
            schema_audit[
                f"has_{field}"
            ].sum()
        )

        if count_present > 0:

            any_direct_update_metadata = True

        print(
            f"{field:<24} "
            f"{count_present}/24 files"
        )

    # =========================================================================
    # 7. Frozen graph-provenance context
    # =========================================================================

    separator("-")

    print(
        "8.1.1.7 — EXISTING GRAPH "
        "PROVENANCE CONTEXT"
    )

    separator("-")

    graph_policy = read_json(
        PHASE806_POLICY_PATH
    )

    require(
        graph_policy[
            "strict_temporal_track"
        ][
            "status"
        ]
        ==
        "FROZEN",
        (
            "Strict graph track no longer frozen."
        ),
    )

    require(
        graph_policy[
            "founder_proxy_track"
        ][
            "status"
        ]
        ==
        "FROZEN",
        (
            "Founder graph track no longer frozen."
        ),
    )

    print(
        "Timestamped acquisition track: FROZEN"
    )

    print(
        "Founder-proxy track:           FROZEN"
    )

    print(
        "Founder DB observation time:   NOT CLAIMED"
    )

    # =========================================================================
    # 8. Construct input-provenance register
    # =========================================================================

    separator("-")

    print(
        "8.1.1.8 — INPUT PROVENANCE REGISTER"
    )

    separator("-")

    provenance = pd.DataFrame(
        [
            {
                "input_family":
                    "startup_id_latent_embedding",

                "source":
                    (
                        "Phase-4 reproduced ITRS "
                        "entity-specific embedding"
                    ),

                "id_independent":
                    False,

                "cold_start_inductive":
                    False,

                "temporal_provenance":
                    (
                        "entity identity / learned "
                        "transductive parameter"
                    ),

                "historical_value_timestamped":
                    False,

                "proposed_phase8_role":
                    "B0_ONLY",

                "scientific_caveat":
                    (
                        "Cannot define a genuinely "
                        "ID-independent inductive startup "
                        "representation."
                    ),
            },
            {
                "input_family":
                    "description_text",

                "source":
                    (
                        "Phase-4 frozen Crunchbase "
                        "description with short-description "
                        "fallback"
                    ),

                "id_independent":
                    True,

                "cold_start_inductive":
                    True,

                "temporal_provenance":
                    "current_snapshot_unversioned",

                "historical_value_timestamped":
                    False,

                "proposed_phase8_role":
                    "CANDIDATE_SHARED_INPUT",

                "scientific_caveat":
                    (
                        "Exact historical description "
                        "at T57/T58/T59 cannot be proven "
                        "from the available snapshot."
                    ),
            },
            {
                "input_family":
                    "category_labels",

                "source":
                    (
                        "Phase-4 frozen Crunchbase "
                        "category_list vocabulary"
                    ),

                "id_independent":
                    True,

                "cold_start_inductive":
                    True,

                "temporal_provenance":
                    "current_snapshot_unversioned",

                "historical_value_timestamped":
                    False,

                "proposed_phase8_role":
                    "CANDIDATE_SHARED_INPUT",

                "scientific_caveat":
                    (
                        "Exact historical category "
                        "membership at each origin "
                        "cannot be reconstructed."
                    ),
            },
            {
                "input_family":
                    "founded_on",

                "source":
                    "Crunchbase Company founded_on",

                "id_independent":
                    True,

                "cold_start_inductive":
                    True,

                "temporal_provenance":
                    "semantic_event_date",

                "historical_value_timestamped":
                    True,

                "proposed_phase8_role":
                    "ELIGIBILITY_GATING_ONLY_PENDING_FREEZE",

                "scientific_caveat":
                    (
                        "Currently frozen for entity "
                        "existence eligibility; adding it "
                        "as a predictive feature would be "
                        "a new information channel and "
                        "must be separately justified."
                    ),
            },
            {
                "input_family":
                    "acquisition_graph",

                "source":
                    "Phase-8.0.6 strict temporal graph",

                "id_independent":
                    True,

                "cold_start_inductive":
                    True,

                "temporal_provenance":
                    "timestamped_pre_origin",

                "historical_value_timestamped":
                    True,

                "proposed_phase8_role":
                    "ALLOWED_RELATION_ABLATION",

                "scientific_caveat":
                    (
                        "Only edges available under each "
                        "origin-specific strict mask "
                        "may be consumed."
                    ),
            },
            {
                "input_family":
                    "shared_founder_graph",

                "source":
                    (
                        "Phase-8.0.6 founder-proxy "
                        "graph track"
                    ),

                "id_independent":
                    True,

                "cold_start_inductive":
                    True,

                "temporal_provenance":
                    (
                        "current_snapshot_relation_with_"
                        "entity_existence_gating"
                    ),

                "historical_value_timestamped":
                    False,

                "proposed_phase8_role":
                    "ALLOWED_RELATION_ABLATION",

                "scientific_caveat":
                    (
                        "Both organizations are required "
                        "to exist by origin, but the date "
                        "Crunchbase recorded the founder "
                        "relationship is unknown."
                    ),
            },
            {
                "input_family":
                    "startup_investment_history",

                "source":
                    "canonical temporal interactions",

                "id_independent":
                    True,

                "cold_start_inductive":
                    False,

                "temporal_provenance":
                    "timestamped",

                "historical_value_timestamped":
                    True,

                "proposed_phase8_role":
                    "NOT_AVAILABLE_IN_PRIMARY_STARTUP_BRANCH",

                "scientific_caveat":
                    (
                        "Primary thesis startups are "
                        "investment-cold by definition."
                    ),
            },
        ]
    )

    print(
        provenance[
            [
                "input_family",
                "id_independent",
                "cold_start_inductive",
                "temporal_provenance",
                "proposed_phase8_role",
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # 9. Persist audit
    # =========================================================================

    separator("-")

    print(
        "8.1.1.9 — PERSIST AUDIT ARTIFACTS"
    )

    separator("-")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    global_summary.to_csv(
        GLOBAL_COVERAGE_OUTPUT,
        index=False,
    )

    candidate_coverage.to_csv(
        CANDIDATE_COVERAGE_OUTPUT,
        index=False,
    )

    target_coverage.to_csv(
        TARGET_COVERAGE_OUTPUT,
        index=False,
    )

    primary_coverage.to_csv(
        PRIMARY_COVERAGE_OUTPUT,
        index=False,
    )

    provenance.to_csv(
        PROVENANCE_OUTPUT,
        index=False,
    )

    schema_audit.to_csv(
        RAW_SCHEMA_OUTPUT,
        index=False,
    )

    summary = {
        "schema_version":
            "PHASE_8_1_1_INPUT_AVAILABILITY_PROVENANCE_V3",

        "status":
            "COMPLETE",

        "scientific_role":
            "PRE_MODEL_INPUT_AUDIT_ONLY",

        "global_startup_nodes":
            EXPECTED_STARTUPS,

        "global_startups_with_text":
            text_count,

        "global_startups_with_labels":
            label_count,

        "company_csv_files":
            len(
                company_files
            ),

        "direct_snapshot_update_metadata_detected":
            any_direct_update_metadata,

        "description_temporal_classification":
            "current_snapshot_unversioned",

        "category_raw_source_field":
            "categories",

        "category_temporal_classification":
            "current_snapshot_unversioned",

        "startup_id_embedding_classification":
            "transductive_not_inductive",

        "founded_on_current_role":
            "eligibility_only_pending_model_input_decision",

        "strict_acquisition_graph":
            "timestamped_origin_specific",

        "founder_graph":
            (
                "snapshot_relation_semantically_gated_"
                "by_entity_existence"
            ),

        "input_contract_frozen":
            False,

        "architecture_frozen":
            False,

        "model_constructed":
            False,

        "training_performed":
            False,

        "ranking_metrics_scored":
            False,

        "April_performance_inspected":
            False,

        "next_decision":
            (
                "Freeze Phase-8.1 inductive startup "
                "input contract before architecture design."
            ),
    }

    write_json(
        SUMMARY_OUTPUT,
        summary,
    )

    manifest = {
        "phase":
            "8.1.1",

        "status":
            "COMPLETE",

        "authoritative_inputs":
            source_hashes,

        "outputs": [
            str(
                GLOBAL_COVERAGE_OUTPUT
            ),
            str(
                CANDIDATE_COVERAGE_OUTPUT
            ),
            str(
                TARGET_COVERAGE_OUTPUT
            ),
            str(
                PRIMARY_COVERAGE_OUTPUT
            ),
            str(
                PROVENANCE_OUTPUT
            ),
            str(
                RAW_SCHEMA_OUTPUT
            ),
            str(
                SUMMARY_OUTPUT
            ),
        ],

        "prohibited_operations": [
            "model_construction",
            "training",
            "inference",
            "ranking_scoring",
            "architecture_selection",
            "April_performance_inspection",
        ],
    }

    write_json(
        MANIFEST_OUTPUT,
        manifest,
    )

    output_hashes = {
        str(
            GLOBAL_COVERAGE_OUTPUT
        ):
            sha256_file(
                GLOBAL_COVERAGE_OUTPUT
            ),

        str(
            CANDIDATE_COVERAGE_OUTPUT
        ):
            sha256_file(
                CANDIDATE_COVERAGE_OUTPUT
            ),

        str(
            TARGET_COVERAGE_OUTPUT
        ):
            sha256_file(
                TARGET_COVERAGE_OUTPUT
            ),

        str(
            PRIMARY_COVERAGE_OUTPUT
        ):
            sha256_file(
                PRIMARY_COVERAGE_OUTPUT
            ),

        str(
            PROVENANCE_OUTPUT
        ):
            sha256_file(
                PROVENANCE_OUTPUT
            ),

        str(
            RAW_SCHEMA_OUTPUT
        ):
            sha256_file(
                RAW_SCHEMA_OUTPUT
            ),

        str(
            SUMMARY_OUTPUT
        ):
            sha256_file(
                SUMMARY_OUTPUT
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
        "PHASE 8.1.1 INPUT AVAILABILITY "
        "& PROVENANCE AUDIT COMPLETE"
    )

    separator()

    print()

    print(
        "Startup-ID embedding:          "
        "TRANSDUCTIVE / NOT INDUCTIVE"
    )

    print(
        "Description inputs:            "
        "ID-INDEPENDENT / SNAPSHOT-UNVERSIONED"
    )

    print(
        "Category inputs:               "
        "ID-INDEPENDENT / SNAPSHOT-UNVERSIONED"
    )

    print(
        "founded_on:                    "
        "SEMANTIC EVENT DATE"
    )

    print(
        "Acquisition graph:             "
        "TIMESTAMPED ORIGIN-SPECIFIC"
    )

    print(
        "Shared-founder graph:          "
        "SEMANTICALLY GATED SNAPSHOT RELATION"
    )

    print()

    print(
        "INPUT CONTRACT FROZEN:         NO"
    )

    print(
        "ARCHITECTURE FROZEN:           NO"
    )

    print(
        "MODEL CONSTRUCTED:             NO"
    )

    print(
        "TRAINING PERFORMED:            NO"
    )

    print(
        "APRIL PERFORMANCE INSPECTED:   NO"
    )

    print()

    print(
        "Next:"
    )

    print(
        "Use these coverage and provenance "
        "results to freeze the Phase-8.1 "
        "inductive startup input contract."
    )

    print()

    print(
        f"Outputs written to:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()