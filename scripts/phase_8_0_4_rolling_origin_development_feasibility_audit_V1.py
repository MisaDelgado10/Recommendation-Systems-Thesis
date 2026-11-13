from pathlib import Path
import hashlib
import json

import pandas as pd


# =============================================================================
# PHASE 8.0.4 — ROLLING-ORIGIN DEVELOPMENT FEASIBILITY AUDIT V1
# =============================================================================
#
# Scientific role:
#
#   OUTCOME-BLIND DEVELOPMENT-WINDOW / TEMPORAL-LEAKAGE AUDIT
#
# This script does NOT:
#
#   - load a model,
#   - load a checkpoint,
#   - run inference,
#   - compute HR@10 / NDCG@10,
#   - generate negatives,
#   - select a model,
#   - modify the Phase-3 graph,
#   - create final development masks,
#   - touch the April-2026 confirmatory outcomes.
#
# PURPOSE
# -------
#
# Determine whether T57, T58, and T59 are suitable rolling-origin
# development folds for Phase 8.
#
# For every fold:
#
#   1. reconstruct the frozen temporal window;
#   2. define history strictly before the fold;
#   3. reconstruct thesis diagnostic strata at that origin;
#   4. measure direct structural target-pair overlap;
#   5. measure timestamped Phase-3 edges that occur after that origin;
#   6. calculate the size of a possible temporally safe graph overlay.
#
# No performance outcomes are inspected.
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

TEMPORAL_PATH = Path(
    "data/experimental/phase_2/"
    "temporal/interactions_itrs_temporal.parquet"
)

PHASE3_EDGES_PATH = Path(
    "data/experimental/phase_3/graph/edges.parquet"
)

OUTPUT_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_0_evaluation_reset/"
    "phase_8_0_4_rolling_origin_development_feasibility"
)

FOLD_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_4_fold_population_summary_V1.csv"
)

GRAPH_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_4_fold_graph_leakage_summary_V1.csv"
)

RELATION_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_4_fold_relation_exclusion_summary_V1.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_4_feasibility_summary_V1.json"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_4_analysis_manifest_V1.json"
)

HASH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_4_derived_artifact_sha256_V1.json"
)


# =============================================================================
# FROZEN DEVELOPMENT-FOLD CANDIDATES
# =============================================================================

FOLDS = [
    {
        "segment_number": 57,
        "segment_label": "T57",
        "history_end": "2025-03-31",
        "fold_start": "2025-04-01",
        "fold_end": "2025-06-30",
    },
    {
        "segment_number": 58,
        "segment_label": "T58",
        "history_end": "2025-06-30",
        "fold_start": "2025-07-01",
        "fold_end": "2025-09-30",
    },
    {
        "segment_number": 59,
        "segment_label": "T59",
        "history_end": "2025-09-30",
        "fold_start": "2025-10-01",
        "fold_end": "2025-12-31",
    },
]


# =============================================================================
# FROZEN SOURCE EXPECTATIONS
# =============================================================================

EXPECTED_PHASE3_EDGES = 158_818

EXPECTED_PHASE3_EDGES_SHA256 = (
    "895871308de416b12d001550838cdb276"
    "3c87cd897774c6a55539a0ebb9a1ee2"
)


# =============================================================================
# UTILITIES
# =============================================================================

def separator(char="=", width=118):
    print(char * width)


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


def ordered_underlying_pair(a, b):

    a = str(a)
    b = str(b)

    if a <= b:
        return (a, b)

    return (b, a)


# =============================================================================
# MAIN
# =============================================================================

def main():

    separator()

    print(
        "PHASE 8.0.4 — "
        "ROLLING-ORIGIN DEVELOPMENT "
        "FEASIBILITY AUDIT V1"
    )

    separator()

    print()
    print(
        "Scientific role:             "
        "OUTCOME-BLIND DEVELOPMENT-DESIGN AUDIT"
    )

    print("Model loaded:                NO")
    print("Checkpoint loaded:           NO")
    print("Inference executed:          NO")
    print("Ranking metrics scored:      NO")
    print("Negatives generated:         NO")
    print("Model selection performed:   NO")
    print("Phase-3 graph modified:      NO")
    print("Development folds frozen:    NO")
    print("April confirmation scored:   NO")

    # =========================================================================
    # 1. Source integrity
    # =========================================================================

    separator("-")
    print(
        "8.0.4.1 — SOURCE EXISTENCE "
        "AND INTEGRITY"
    )
    separator("-")

    for path in [
        TEMPORAL_PATH,
        PHASE3_EDGES_PATH,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Required source missing: {path}"
            )

        print(
            f"FOUND  {path}"
        )

        print(
            f"SHA256 {sha256_file(path)}"
        )

    observed_graph_sha = sha256_file(
        PHASE3_EDGES_PATH
    )

    if (
        observed_graph_sha
        != EXPECTED_PHASE3_EDGES_SHA256
    ):

        raise ValueError(
            "Frozen Phase-3 graph fingerprint drift."
        )

    # =========================================================================
    # 2. Load frozen temporal interaction layer
    # =========================================================================

    separator("-")
    print(
        "8.0.4.2 — LOAD FROZEN "
        "TEMPORAL INTERACTION LAYER"
    )
    separator("-")

    interactions = pd.read_parquet(
        TEMPORAL_PATH,
        columns=[
            "interaction_id",
            "announced_on",
            "investor_id",
            "startup_id",
            "segment_number",
            "segment_label",
        ],
    )

    interactions[
        "announced_on"
    ] = pd.to_datetime(
        interactions[
            "announced_on"
        ],
        errors="raise",
    )

    print(
        f"Temporal rows:     "
        f"{len(interactions):,}"
    )

    print(
        f"Temporal range:    "
        f"{interactions['announced_on'].min().date()} "
        f"-> "
        f"{interactions['announced_on'].max().date()}"
    )

    if interactions[
        "interaction_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate interaction_id values "
            "in temporal layer."
        )

    # =========================================================================
    # 3. Load frozen Phase-3 graph
    # =========================================================================

    separator("-")
    print(
        "8.0.4.3 — LOAD FROZEN "
        "STRUCTURAL GRAPH"
    )
    separator("-")

    edges = pd.read_parquet(
        PHASE3_EDGES_PATH,
        columns=[
            "edge_id",
            "relation",
            "underlying_src_id",
            "underlying_dst_id",
            "temporal_provenance",
            "first_observed_on",
        ],
    )

    if (
        len(edges)
        != EXPECTED_PHASE3_EDGES
    ):

        raise ValueError(
            "Frozen Phase-3 edge count drift."
        )

    if edges[
        "edge_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate structural edge_id."
        )

    edges[
        "underlying_pair"
    ] = [
        ordered_underlying_pair(
            src,
            dst,
        )
        for src, dst
        in zip(
            edges[
                "underlying_src_id"
            ],
            edges[
                "underlying_dst_id"
            ],
        )
    ]

    edges[
        "first_observed_on_parsed"
    ] = pd.to_datetime(
        edges[
            "first_observed_on"
        ],
        errors="coerce",
    )

    print(
        f"Structural edges:  "
        f"{len(edges):,}"
    )

    print(
        "Temporal provenance:"
    )

    print(
        edges[
            "temporal_provenance"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    # =========================================================================
    # 4. Audit candidate development folds
    # =========================================================================

    fold_rows = []
    graph_rows = []
    relation_rows = []

    for fold_spec in FOLDS:

        segment_number = (
            fold_spec[
                "segment_number"
            ]
        )

        label = (
            fold_spec[
                "segment_label"
            ]
        )

        history_end = pd.Timestamp(
            fold_spec[
                "history_end"
            ]
        )

        fold_start = pd.Timestamp(
            fold_spec[
                "fold_start"
            ]
        )

        fold_end = pd.Timestamp(
            fold_spec[
                "fold_end"
            ]
        )

        separator("-")

        print(
            f"8.0.4 — {label} "
            f"DEVELOPMENT-FOLD AUDIT"
        )

        separator("-")

        # ---------------------------------------------------------------------
        # Exact frozen segment reconstruction
        # ---------------------------------------------------------------------

        fold = interactions[
            interactions[
                "segment_number"
            ]
            == segment_number
        ].copy()

        if len(fold) == 0:

            raise ValueError(
                f"{label} contains zero events."
            )

        observed_start = (
            fold[
                "announced_on"
            ].min()
        )

        observed_end = (
            fold[
                "announced_on"
            ].max()
        )

        if (
            fold[
                "announced_on"
            ].min()
            < fold_start
            or
            fold[
                "announced_on"
            ].max()
            > fold_end
        ):

            raise ValueError(
                f"{label} events fall outside "
                "the expected quarter."
            )

        if not (
            fold[
                "segment_label"
            ]
            == label
        ).all():

            raise ValueError(
                f"{label} segment-label mismatch."
            )

        # ---------------------------------------------------------------------
        # Historical information available at prediction origin
        # ---------------------------------------------------------------------

        history = interactions[
            interactions[
                "announced_on"
            ]
            <= history_end
        ].copy()

        historical_investors = set(
            history[
                "investor_id"
            ].astype(str)
        )

        historical_startups = set(
            history[
                "startup_id"
            ].astype(str)
        )

        historical_pairs = set(
            zip(
                history[
                    "investor_id"
                ].astype(str),

                history[
                    "startup_id"
                ].astype(str),
            )
        )

        # ---------------------------------------------------------------------
        # Endpoint-relative thesis strata
        # ---------------------------------------------------------------------

        investor_strings = (
            fold[
                "investor_id"
            ].astype(str)
        )

        startup_strings = (
            fold[
                "startup_id"
            ].astype(str)
        )

        fold[
            "cold_start_investor"
        ] = ~investor_strings.isin(
            historical_investors
        )

        fold[
            "cold_start_startup"
        ] = ~startup_strings.isin(
            historical_startups
        )

        fold[
            "new_to_investor_pair"
        ] = [
            (
                investor_id,
                startup_id,
            )
            not in historical_pairs

            for investor_id, startup_id
            in zip(
                investor_strings,
                startup_strings,
            )
        ]

        def classify(row):

            if not row[
                "new_to_investor_pair"
            ]:
                return "prior_pair"

            if (
                not row[
                    "cold_start_investor"
                ]
                and
                not row[
                    "cold_start_startup"
                ]
            ):
                return (
                    "new_warm_investor_"
                    "warm_startup"
                )

            if (
                not row[
                    "cold_start_investor"
                ]
                and
                row[
                    "cold_start_startup"
                ]
            ):
                return (
                    "new_warm_investor_"
                    "cold_startup"
                )

            if (
                row[
                    "cold_start_investor"
                ]
                and
                not row[
                    "cold_start_startup"
                ]
            ):
                return (
                    "new_cold_investor_"
                    "warm_startup"
                )

            return "new_dual_cold"

        fold[
            "analysis_stratum"
        ] = fold.apply(
            classify,
            axis=1,
        )

        stratum_counts = (
            fold[
                "analysis_stratum"
            ]
            .value_counts()
            .to_dict()
        )

        primary_label = (
            "new_warm_investor_"
            "cold_startup"
        )

        primary_events = int(
            stratum_counts.get(
                primary_label,
                0,
            )
        )

        # ---------------------------------------------------------------------
        # Target-pair structural overlap
        # ---------------------------------------------------------------------

        fold[
            "underlying_pair"
        ] = [
            ordered_underlying_pair(
                investor_id,
                startup_id,
            )
            for investor_id, startup_id
            in zip(
                fold[
                    "investor_id"
                ],
                fold[
                    "startup_id"
                ],
            )
        ]

        target_pairs = set(
            fold[
                "underlying_pair"
            ]
        )

        direct_overlap_mask = (
            edges[
                "underlying_pair"
            ].isin(
                target_pairs
            )
        )

        direct_overlap_edges = edges[
            direct_overlap_mask
        ].copy()

        direct_overlap_pairs = set(
            direct_overlap_edges[
                "underlying_pair"
            ]
        )

        fold[
            "direct_structural_overlap"
        ] = (
            fold[
                "underlying_pair"
            ].isin(
                direct_overlap_pairs
            )
        )

        direct_overlap_events = int(
            fold[
                "direct_structural_overlap"
            ].sum()
        )

        primary_mask = (
            fold[
                "analysis_stratum"
            ]
            == primary_label
        )

        primary_overlap_events = int(
            (
                primary_mask
                &
                fold[
                    "direct_structural_overlap"
                ]
            ).sum()
        )

        # ---------------------------------------------------------------------
        # Timestamped graph leakage relative to fold origin
        # ---------------------------------------------------------------------
        #
        # Any structural edge with a genuine observation date later than
        # the history boundary cannot be used in a historically faithful
        # graph for this fold.
        #
        # Unversioned founder edges are NOT classified as dated future
        # leakage here because no historical observation timestamp exists.
        # They are reported separately as a limitation.
        # ---------------------------------------------------------------------

        future_timestamp_mask = (
            edges[
                "first_observed_on_parsed"
            ].notna()
            &
            (
                edges[
                    "first_observed_on_parsed"
                ]
                > history_end
            )
        )

        future_timestamp_edges = (
            edges[
                future_timestamp_mask
            ]
        )

        unversioned_mask = (
            edges[
                "temporal_provenance"
            ]
            ==
            "current_snapshot_unversioned"
        )

        # ---------------------------------------------------------------------
        # Candidate leakage-safe overlay
        #
        # Union, because an edge may satisfy both exclusion reasons.
        # ---------------------------------------------------------------------

        combined_exclusion_mask = (
            direct_overlap_mask
            |
            future_timestamp_mask
        )

        combined_exclusion_edges = int(
            combined_exclusion_mask.sum()
        )

        safe_core_edges = (
            len(edges)
            -
            combined_exclusion_edges
        )

        # ---------------------------------------------------------------------
        # Population outputs
        # ---------------------------------------------------------------------

        unique_pairs = int(
            fold[
                [
                    "investor_id",
                    "startup_id",
                ]
            ]
            .drop_duplicates()
            .shape[0]
        )

        new_events = int(
            fold[
                "new_to_investor_pair"
            ].sum()
        )

        fold_rows.append(
            {
                "fold":
                    label,

                "history_end":
                    history_end.date(),

                "fold_start":
                    fold_start.date(),

                "fold_end":
                    fold_end.date(),

                "observed_first_event":
                    observed_start.date(),

                "observed_last_event":
                    observed_end.date(),

                "interactions":
                    len(fold),

                "unique_investors":
                    fold[
                        "investor_id"
                    ].nunique(),

                "unique_startups":
                    fold[
                        "startup_id"
                    ].nunique(),

                "unique_pairs":
                    unique_pairs,

                "new_to_investor_events":
                    new_events,

                "prior_pair":
                    int(
                        stratum_counts.get(
                            "prior_pair",
                            0,
                        )
                    ),

                "new_warm_investor_warm_startup":
                    int(
                        stratum_counts.get(
                            "new_warm_investor_"
                            "warm_startup",
                            0,
                        )
                    ),

                "new_warm_investor_cold_startup":
                    primary_events,

                "new_cold_investor_warm_startup":
                    int(
                        stratum_counts.get(
                            "new_cold_investor_"
                            "warm_startup",
                            0,
                        )
                    ),

                "new_dual_cold":
                    int(
                        stratum_counts.get(
                            "new_dual_cold",
                            0,
                        )
                    ),
            }
        )

        graph_rows.append(
            {
                "fold":
                    label,

                "history_end":
                    history_end.date(),

                "target_unique_unordered_pairs":
                    len(
                        target_pairs
                    ),

                "direct_overlap_edge_records":
                    len(
                        direct_overlap_edges
                    ),

                "direct_overlap_unique_pairs":
                    len(
                        direct_overlap_pairs
                    ),

                "fold_events_on_direct_overlap_pairs":
                    direct_overlap_events,

                "primary_events_on_direct_overlap_pairs":
                    primary_overlap_events,

                "future_timestamped_edge_records":
                    int(
                        future_timestamp_mask.sum()
                    ),

                "unversioned_edge_records":
                    int(
                        unversioned_mask.sum()
                    ),

                "combined_exclusion_edge_records":
                    combined_exclusion_edges,

                "candidate_temporally_safe_core_edges":
                    safe_core_edges,

                "direct_overlap_after_candidate_mask":
                    0,
            }
        )

        # ---------------------------------------------------------------------
        # Relation-level exclusion diagnostics
        # ---------------------------------------------------------------------

        for relation in sorted(
            edges[
                "relation"
            ].unique()
        ):

            relation_mask = (
                edges[
                    "relation"
                ]
                == relation
            )

            relation_rows.append(
                {
                    "fold":
                        label,

                    "relation":
                        relation,

                    "original_edges":
                        int(
                            relation_mask.sum()
                        ),

                    "direct_target_pair_exclusions":
                        int(
                            (
                                relation_mask
                                &
                                direct_overlap_mask
                            ).sum()
                        ),

                    "future_timestamp_exclusions":
                        int(
                            (
                                relation_mask
                                &
                                future_timestamp_mask
                            ).sum()
                        ),

                    "combined_unique_exclusions":
                        int(
                            (
                                relation_mask
                                &
                                combined_exclusion_mask
                            ).sum()
                        ),

                    "candidate_safe_edges":
                        int(
                            (
                                relation_mask
                                &
                                ~combined_exclusion_mask
                            ).sum()
                        ),
                }
            )

        # ---------------------------------------------------------------------
        # Terminal summary for this fold
        # ---------------------------------------------------------------------

        print(
            f"Prediction origin:                 "
            f"{history_end.date()}"
        )

        print(
            f"Evaluation period:                 "
            f"{fold_start.date()} "
            f"-> {fold_end.date()}"
        )

        print(
            f"Events:                            "
            f"{len(fold):,}"
        )

        print(
            f"Unique pairs:                      "
            f"{unique_pairs:,}"
        )

        print(
            f"New-to-investor events:            "
            f"{new_events:,}"
        )

        print(
            f"Primary-stratum events:            "
            f"{primary_events:,}"
        )

        print()
        print(
            f"Direct graph-overlap pairs:        "
            f"{len(direct_overlap_pairs):,}"
        )

        print(
            f"Events affected by direct overlap: "
            f"{direct_overlap_events:,}"
        )

        print(
            f"Primary events affected:           "
            f"{primary_overlap_events:,}"
        )

        print()
        print(
            f"Future timestamped graph edges:    "
            f"{int(future_timestamp_mask.sum()):,}"
        )

        print(
            f"Unversioned graph edges:           "
            f"{int(unversioned_mask.sum()):,}"
        )

        print(
            f"Combined exclusions required:      "
            f"{combined_exclusion_edges:,}"
        )

        print(
            f"Candidate safe core edges:         "
            f"{safe_core_edges:,}"
        )

    # =========================================================================
    # 5. Materialize audit tables
    # =========================================================================

    fold_summary = pd.DataFrame(
        fold_rows
    )

    graph_summary = pd.DataFrame(
        graph_rows
    )

    relation_summary = pd.DataFrame(
        relation_rows
    )

    separator("-")
    print(
        "8.0.4.5 — DEVELOPMENT-FOLD "
        "POPULATION SUMMARY"
    )
    separator("-")

    print(
        fold_summary.to_string(
            index=False
        )
    )

    separator("-")
    print(
        "8.0.4.6 — DEVELOPMENT-FOLD "
        "GRAPH SAFETY SUMMARY"
    )
    separator("-")

    print(
        graph_summary.to_string(
            index=False
        )
    )

    separator("-")
    print(
        "8.0.4.7 — RELATION-LEVEL "
        "EXCLUSION SUMMARY"
    )
    separator("-")

    print(
        relation_summary.to_string(
            index=False
        )
    )

    # =========================================================================
    # 6. Persist
    # =========================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_summary.to_csv(
        FOLD_SUMMARY_OUTPUT,
        index=False,
    )

    graph_summary.to_csv(
        GRAPH_SUMMARY_OUTPUT,
        index=False,
    )

    relation_summary.to_csv(
        RELATION_SUMMARY_OUTPUT,
        index=False,
    )

    all_primary_nonzero = bool(
        (
            fold_summary[
                "new_warm_investor_cold_startup"
            ]
            > 0
        ).all()
    )

    all_safe_core_positive = bool(
        (
            graph_summary[
                "candidate_temporally_safe_core_edges"
            ]
            > 0
        ).all()
    )

    summary = {
        "schema_version":
            "PHASE_8_0_4_ROLLING_ORIGIN_FEASIBILITY_V1",

        "scientific_role":
            "OUTCOME_BLIND_DEVELOPMENT_WINDOW_TEMPORAL_LEAKAGE_AUDIT",

        "candidate_folds": [
            "T57",
            "T58",
            "T59",
        ],

        "performance_metrics_inspected":
            False,

        "model_loaded":
            False,

        "checkpoint_loaded":
            False,

        "inference_executed":
            False,

        "negatives_generated":
            False,

        "model_selection_performed":
            False,

        "phase3_graph_modified":
            False,

        "development_folds_frozen":
            False,

        "april_confirmatory_outcomes_touched":
            False,

        "all_folds_have_primary_stratum_events":
            all_primary_nonzero,

        "all_folds_have_positive_safe_graph_size":
            all_safe_core_positive,

        "temporal_graph_policy_tested": (
            "For each candidate development fold, "
            "exclude the union of: "
            "(1) direct structural edges whose "
            "underlying pair matches a fold target pair, "
            "and (2) timestamped structural edges first "
            "observed after that fold's history boundary."
        ),

        "unversioned_founder_policy_status": (
            "NOT YET RESOLVED. "
            "Current-snapshot founder relations cannot "
            "be historically filtered using observation "
            "timestamps and remain a documented "
            "temporal-observability limitation."
        ),

        "decision_status":
            "FEASIBILITY_AUDIT_ONLY_NOT_FROZEN",
    }

    write_json(
        SUMMARY_OUTPUT,
        summary,
    )

    source_hashes = {
        str(TEMPORAL_PATH):
            sha256_file(
                TEMPORAL_PATH
            ),

        str(PHASE3_EDGES_PATH):
            sha256_file(
                PHASE3_EDGES_PATH
            ),
    }

    manifest = {
        "phase":
            "8.0.4",

        "authoritative_inputs":
            source_hashes,

        "candidate_fold_contracts":
            FOLDS,

        "outputs": [
            str(
                FOLD_SUMMARY_OUTPUT
            ),
            str(
                GRAPH_SUMMARY_OUTPUT
            ),
            str(
                RELATION_SUMMARY_OUTPUT
            ),
            str(
                SUMMARY_OUTPUT
            ),
        ],

        "prohibited_operations": [
            "model_loading",
            "checkpoint_loading",
            "inference",
            "ranking_metric_scoring",
            "negative_generation",
            "model_selection",
            "phase3_graph_modification",
            "april_confirmatory_scoring",
        ],
    }

    write_json(
        MANIFEST_OUTPUT,
        manifest,
    )

    derived_hashes = {
        str(FOLD_SUMMARY_OUTPUT):
            sha256_file(
                FOLD_SUMMARY_OUTPUT
            ),

        str(GRAPH_SUMMARY_OUTPUT):
            sha256_file(
                GRAPH_SUMMARY_OUTPUT
            ),

        str(RELATION_SUMMARY_OUTPUT):
            sha256_file(
                RELATION_SUMMARY_OUTPUT
            ),

        str(SUMMARY_OUTPUT):
            sha256_file(
                SUMMARY_OUTPUT
            ),

        str(MANIFEST_OUTPUT):
            sha256_file(
                MANIFEST_OUTPUT
            ),
    }

    write_json(
        HASH_OUTPUT,
        derived_hashes,
    )

    # =========================================================================
    # 7. Final audit summary
    # =========================================================================

    separator()
    print(
        "PHASE 8.0.4 FEASIBILITY AUDIT COMPLETE"
    )
    separator()

    print()

    for row in fold_rows:

        print(
            f"{row['fold']}: "
            f"{row['interactions']:,} events | "
            f"{row['new_warm_investor_cold_startup']:,} "
            f"primary-stratum events"
        )

    print()

    print(
        "All folds contain primary-stratum cases: "
        f"{all_primary_nonzero}"
    )

    print(
        "All folds admit non-empty safe graph:     "
        f"{all_safe_core_positive}"
    )

    print()

    print(
        "DEVELOPMENT FOLDS FROZEN:             NO"
    )

    print(
        "APRIL CONFIRMATORY HOLDOUT SCORED:    NO"
    )

    print()

    print(
        "Next decision:"
    )

    print(
        "Determine whether T57-T59 should be "
        "frozen as the Phase-8 rolling-origin "
        "development/model-selection protocol."
    )

    print()

    print(
        f"Outputs written to:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()