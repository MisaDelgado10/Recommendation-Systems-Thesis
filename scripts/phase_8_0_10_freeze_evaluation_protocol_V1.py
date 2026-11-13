from pathlib import Path
import hashlib
import json

import pandas as pd


# =============================================================================
# PHASE 8.0.10 — FREEZE PHASE-8 EVALUATION PROTOCOL V1
# =============================================================================
#
# Scientific role:
#
#   OUTCOME-BLIND RESEARCH / MODEL-SELECTION / CONFIRMATORY PROTOCOL FREEZE
#
# NO:
#   - model construction
#   - model training
#   - checkpoint loading
#   - inference
#   - ranking scoring
#   - negative generation
#   - model selection
#   - April confirmatory scoring
#
# PURPOSE
# -------
#
# Freeze the complete temporal role of:
#
#   T57  rolling-origin validation fold
#   T58  rolling-origin validation fold
#   T59  rolling-origin validation fold
#   T60  spent diagnostic period; later usable for final fit only
#   April 2026 untouched confirmatory period
#
# Also freeze:
#
#   - training-history semantics;
#   - validation aggregation rule;
#   - primary model-selection endpoint;
#   - tie-break hierarchy;
#   - evaluation-candidate semantics;
#   - final-fit semantics;
#   - April access restrictions.
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

TEMPORAL_PATH = Path(
    "data/experimental/phase_2/"
    "temporal/interactions_itrs_temporal.parquet"
)

PHASE806_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_0_evaluation_reset/"
    "phase_8_0_6_dual_track_graph_policy"
)

PHASE806_SUMMARY = (
    PHASE806_DIR
    / "phase_8_0_6_origin_graph_summary_V1.csv"
)

PHASE806_POLICY = (
    PHASE806_DIR
    / "phase_8_0_6_graph_information_policy_V1.json"
)

PHASE809_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_0_evaluation_reset/"
    "phase_8_0_9_origin_eligibility_candidate_universe"
)

PHASE809_SUMMARY = (
    PHASE809_DIR
    / "phase_8_0_9_origin_contract_summary_V5.csv"
)

PHASE809_CONTRACT = (
    PHASE809_DIR
    / "phase_8_0_9_origin_eligibility_contract_V5.json"
)

PHASE809_CANDIDATES = (
    PHASE809_DIR
    / "phase_8_0_9_origin_candidate_universe_V5.parquet"
)

PHASE809_TARGETS = (
    PHASE809_DIR
    / "phase_8_0_9_origin_eligible_targets_V5.parquet"
)


OUTPUT_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_0_evaluation_reset/"
    "phase_8_0_10_evaluation_protocol"
)

FOLD_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_10_fold_role_register_V1.csv"
)

SELECTION_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_10_model_selection_contract_V1.json"
)

PROTOCOL_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_10_evaluation_protocol_V1.json"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_10_manifest_V1.json"
)

HASH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_10_derived_artifact_sha256_V1.json"
)


# =============================================================================
# FROZEN PHASE-8.0.9 ANCHORS
# =============================================================================

EXPECTED_809 = {
    "T57": {
        "candidate_startups": 308_027,
        "all_target_events": 22_742,
        "origin_eligible_targets": 21_924,
        "eligible_primary_events": 6_850,
    },
    "T58": {
        "candidate_startups": 308_854,
        "all_target_events": 23_337,
        "origin_eligible_targets": 22_384,
        "eligible_primary_events": 6_872,
    },
    "T59": {
        "candidate_startups": 309_611,
        "all_target_events": 23_644,
        "origin_eligible_targets": 22_650,
        "eligible_primary_events": 6_764,
    },
    "APRIL_CONFIRMATORY": {
        "candidate_startups": 311_266,
        "all_target_events": 6_622,
        "origin_eligible_targets": 6_313,
        "eligible_primary_events": 1_690,
    },
}


EXPECTED_GRAPH = {
    "T57": {
        "strict_temporal_edges": 60_144,
        "founder_proxy_edges": 153_034,
    },
    "T58": {
        "strict_temporal_edges": 61_358,
        "founder_proxy_edges": 154_392,
    },
    "T59": {
        "strict_temporal_edges": 62_554,
        "founder_proxy_edges": 155_736,
    },
    "APRIL_CONFIRMATORY": {
        "strict_temporal_edges": 63_964,
        "founder_proxy_edges": 157_444,
    },
}


# =============================================================================
# FOLD CONTRACT
# =============================================================================

FOLDS = [
    {
        "fold": "T57",
        "prediction_origin": "2025-03-31",
        "evaluation_start": "2025-04-01",
        "evaluation_end": "2025-06-30",
        "training_segment_first": 1,
        "training_segment_last": 56,
        "evaluation_segment": 57,
        "role": "rolling_origin_validation",
    },
    {
        "fold": "T58",
        "prediction_origin": "2025-06-30",
        "evaluation_start": "2025-07-01",
        "evaluation_end": "2025-09-30",
        "training_segment_first": 1,
        "training_segment_last": 57,
        "evaluation_segment": 58,
        "role": "rolling_origin_validation",
    },
    {
        "fold": "T59",
        "prediction_origin": "2025-09-30",
        "evaluation_start": "2025-10-01",
        "evaluation_end": "2025-12-31",
        "training_segment_first": 1,
        "training_segment_last": 58,
        "evaluation_segment": 59,
        "role": "rolling_origin_validation",
    },
]


# =============================================================================
# UTILITIES
# =============================================================================

def separator(
    char="=",
    width=118,
):
    print(
        char * width
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
        "PHASE 8.0.10 — "
        "FREEZE PHASE-8 EVALUATION PROTOCOL V1"
    )

    separator()

    print()

    print(
        "Scientific role:             "
        "OUTCOME-BLIND EVALUATION-DESIGN FREEZE"
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
        "Negatives generated:         NO"
    )

    print(
        "Model selection performed:   NO"
    )

    print(
        "April performance inspected: NO"
    )

    # =========================================================================
    # 1. Authoritative input integrity
    # =========================================================================

    separator("-")

    print(
        "8.0.10.1 — AUTHORITATIVE "
        "INPUT INTEGRITY"
    )

    separator("-")

    required_paths = [
        TEMPORAL_PATH,
        PHASE806_SUMMARY,
        PHASE806_POLICY,
        PHASE809_SUMMARY,
        PHASE809_CONTRACT,
        PHASE809_CANDIDATES,
        PHASE809_TARGETS,
    ]

    source_hashes = {}

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

        observed = sha256_file(
            path
        )

        source_hashes[
            str(path)
        ] = observed

        print(
            f"FOUND  {path}"
        )

        print(
            f"SHA256 {observed}"
        )

    # =========================================================================
    # 2. Verify Phase 8.0.9 frozen universe
    # =========================================================================

    separator("-")

    print(
        "8.0.10.2 — PHASE-8.0.9 "
        "UNIVERSE CONTRACT VERIFICATION"
    )

    separator("-")

    summary809 = pd.read_csv(
        PHASE809_SUMMARY
    )

    if set(
        summary809[
            "origin"
        ]
    ) != set(
        EXPECTED_809
    ):

        raise ValueError(
            "Phase-8.0.9 origin set drift."
        )

    for origin, expected in (
        EXPECTED_809.items()
    ):

        row = summary809.loc[
            summary809[
                "origin"
            ]
            ==
            origin
        ]

        if len(row) != 1:

            raise ValueError(
                f"{origin}: expected one "
                "Phase-8.0.9 summary row."
            )

        row = row.iloc[0]

        for field, value in (
            expected.items()
        ):

            observed = int(
                row[
                    field
                ]
            )

            if observed != value:

                raise ValueError(
                    f"{origin}: {field} "
                    f"{observed:,} != "
                    f"{value:,}"
                )

    with open(
        PHASE809_CONTRACT,
        "r",
        encoding="utf-8",
    ) as f:

        contract809 = json.load(
            f
        )

    if (
        contract809.get(
            "status"
        )
        !=
        "FROZEN"
    ):

        raise ValueError(
            "Phase-8.0.9 contract "
            "is not frozen."
        )

    print(
        "Phase-8.0.9 frozen universe: PASS"
    )

    # =========================================================================
    # 3. Verify Phase 8.0.6 graph policy
    # =========================================================================

    separator("-")

    print(
        "8.0.10.3 — PHASE-8.0.6 "
        "GRAPH POLICY VERIFICATION"
    )

    separator("-")

    graph_summary = pd.read_csv(
        PHASE806_SUMMARY
    )

    for origin, expected in (
        EXPECTED_GRAPH.items()
    ):

        row = graph_summary.loc[
            graph_summary[
                "origin"
            ]
            ==
            origin
        ]

        if len(row) != 1:

            raise ValueError(
                f"{origin}: missing graph summary."
            )

        row = row.iloc[0]

        for field, value in (
            expected.items()
        ):

            observed = int(
                row[
                    field
                ]
            )

            if observed != value:

                raise ValueError(
                    f"{origin}: graph field "
                    f"{field} drift."
                )

    with open(
        PHASE806_POLICY,
        "r",
        encoding="utf-8",
    ) as f:

        graph_policy = json.load(
            f
        )

    if (
        graph_policy[
            "strict_temporal_track"
        ][
            "status"
        ]
        !=
        "FROZEN"
    ):

        raise ValueError(
            "Strict graph track not frozen."
        )

    if (
        graph_policy[
            "founder_proxy_track"
        ][
            "status"
        ]
        !=
        "FROZEN"
    ):

        raise ValueError(
            "Founder graph track not frozen."
        )

    print(
        "Phase-8 graph policy: PASS"
    )

    # =========================================================================
    # 4. Reconstruct training-history fold semantics
    # =========================================================================

    separator("-")

    print(
        "8.0.10.4 — ROLLING-ORIGIN "
        "TRAINING HISTORY"
    )

    separator("-")

    temporal = pd.read_parquet(
        TEMPORAL_PATH,
        columns=[
            "interaction_id",
            "segment_number",
            "segment_label",
            "announced_on",
        ],
    )

    temporal[
        "segment_number"
    ] = (
        temporal[
            "segment_number"
        ].astype(int)
    )

    temporal[
        "announced_on"
    ] = pd.to_datetime(
        temporal[
            "announced_on"
        ],
        errors="raise",
    )

    fold_rows = []

    for spec in FOLDS:

        fold = spec[
            "fold"
        ]

        train_mask = (
            temporal[
                "segment_number"
            ].between(
                spec[
                    "training_segment_first"
                ],
                spec[
                    "training_segment_last"
                ],
            )
        )

        eval_mask = (
            temporal[
                "segment_number"
            ]
            ==
            spec[
                "evaluation_segment"
            ]
        )

        training_positive_count = int(
            train_mask.sum()
        )

        raw_fold_events = int(
            eval_mask.sum()
        )

        expected_raw = (
            EXPECTED_809[
                fold
            ][
                "all_target_events"
            ]
        )

        if (
            raw_fold_events
            != expected_raw
        ):

            raise ValueError(
                f"{fold}: evaluation event "
                "count drift."
            )

        fold_rows.append(
            {
                **spec,

                "t0_role":
                    "compressed_history_only",

                "training_positive_events":
                    training_positive_count,

                "raw_evaluation_events":
                    raw_fold_events,

                "eligible_evaluation_events":
                    EXPECTED_809[
                        fold
                    ][
                        "origin_eligible_targets"
                    ],

                "eligible_primary_events":
                    EXPECTED_809[
                        fold
                    ][
                        "eligible_primary_events"
                    ],

                "candidate_startups":
                    EXPECTED_809[
                        fold
                    ][
                        "candidate_startups"
                    ],

                "strict_graph_edges":
                    EXPECTED_GRAPH[
                        fold
                    ][
                        "strict_temporal_edges"
                    ],

                "founder_time_eligible_graph_edges":
                    EXPECTED_GRAPH[
                        fold
                    ][
                        "founder_proxy_edges"
                    ],
            }
        )

        print()

        print(
            f"{fold}"
        )

        print(
            f"  prediction origin:       "
            f"{spec['prediction_origin']}"
        )

        print(
            f"  training positives:      "
            f"{training_positive_count:,}"
        )

        print(
            f"  training segments:       "
            f"T{spec['training_segment_first']}"
            f"-T{spec['training_segment_last']}"
        )

        print(
            f"  validation segment:      "
            f"T{spec['evaluation_segment']}"
        )

        print(
            f"  eligible validation:     "
            f"{EXPECTED_809[fold]['origin_eligible_targets']:,}"
        )

        print(
            f"  eligible primary:        "
            f"{EXPECTED_809[fold]['eligible_primary_events']:,}"
        )

    fold_register = pd.DataFrame(
        fold_rows
    )

    fold_register.to_csv(
        FOLD_OUTPUT,
        index=False,
    )

    # =========================================================================
    # 5. Freeze validation / model-selection rule
    # =========================================================================

    separator("-")

    print(
        "8.0.10.5 — MODEL-SELECTION CONTRACT"
    )

    separator("-")

    selection_contract = {
        "schema_version":
            "PHASE_8_0_10_MODEL_SELECTION_V1",

        "status":
            "FROZEN",

        "development_folds": [
            "T57",
            "T58",
            "T59",
        ],

        "fold_weighting":
            "equal_weight_per_temporal_fold",

        "maximum_training_epochs":
            20,

        "epoch_candidates":
            list(
                range(
                    1,
                    21,
                )
            ),

        "primary_selection_endpoint": {
            "metric":
                "NDCG@10",

            "stratum":
                (
                    "new_to_investor_pair=True; "
                    "warm_investor=True; "
                    "cold_start_startup=True; "
                    "startup_origin_eligible=True"
                ),

            "aggregation":
                (
                    "unweighted arithmetic mean "
                    "across T57, T58, T59"
                ),
        },

        "selection_unit":
            (
                "architecture + hyperparameter "
                "configuration + epoch count"
            ),

        "selection_order": [
            (
                "1_maximize_mean_primary_NDCG_at_10"
            ),
            (
                "2_maximize_mean_primary_HR_at_10"
            ),
            (
                "3_maximize_mean_new_warm_warm_NDCG_at_10"
            ),
            (
                "4_prefer_fewer_trainable_parameters"
            ),
        ],

        "secondary_metrics": [
            (
                "primary-stratum HR@10"
            ),
            (
                "new warm-investor / warm-startup "
                "NDCG@10"
            ),
            (
                "new warm-investor / warm-startup "
                "HR@10"
            ),
            (
                "aggregate eligible NDCG@10"
            ),
            (
                "aggregate eligible HR@10"
            ),
        ],

        "warm_warm_role":
            (
                "reported guardrail; "
                "not a hard non-inferiority "
                "threshold at Phase 8.0"
            ),

        "T60_used_for_model_selection":
            False,

        "April_used_for_model_selection":
            False,

        "selection_uses_only":
            "T57_T58_T59_validation_results",
    }

    write_json(
        SELECTION_OUTPUT,
        selection_contract,
    )

    print(
        "Primary selection metric:"
    )

    print(
        "  mean primary-stratum "
        "NDCG@10 across T57/T58/T59"
    )

    print()

    print(
        "T60 used for selection:    NO"
    )

    print(
        "April used for selection:  NO"
    )

    # =========================================================================
    # 6. Freeze evaluation candidate-generation SEMANTICS
    # =========================================================================

    separator("-")

    print(
        "8.0.10.6 — EVALUATION "
        "CANDIDATE-GENERATION SEMANTICS"
    )

    separator("-")

    candidate_contract = {
        "positive_candidates_per_case":
            1,

        "negative_candidates_per_case":
            99,

        "total_candidates_per_case":
            100,

        "candidate_universe":
            (
                "origin-specific Startup universe "
                "frozen by Phase 8.0.9"
            ),

        "negative_distribution":
            "uniform",

        "replacement":
            False,

        "exclude_from_negative_pool": [
            (
                "focal positive startup"
            ),
            (
                "all startups with a positive "
                "Investor-Startup pair for the "
                "focal investor before the "
                "prediction origin"
            ),
        ],

        "other_same_fold_positive_labels":
            "NOT_USED_FOR_EXCLUSION",

        "future_fold_positive_labels":
            "NOT_USED_FOR_EXCLUSION",

        "candidate_lists":
            (
                "must be generated once and frozen "
                "before model training or model selection"
            ),

        "all_models_share_identical_candidate_lists":
            True,

        "sampling_interpretation":
            (
                "sampled 100-candidate ranking; "
                "not full-catalog top-10"
            ),

        "base_seed":
            42,

        "rng_namespace":
            (
                "ITRS_PHASE8_EVAL_NEGATIVE|"
                "42|<origin>|<interaction_id>"
            ),
    }

    print(
        "Evaluation case = "
        "1 positive + 99 frozen negatives"
    )

    print(
        "Other same-fold positives "
        "used to clean negatives: NO"
    )

    # =========================================================================
    # 7. Final fit and confirmatory lock
    # =========================================================================

    separator("-")

    print(
        "8.0.10.7 — FINAL FIT / "
        "APRIL CONFIRMATORY LOCK"
    )

    separator("-")

    final_fit = {
        "status":
            "FROZEN",

        "allowed_only_after": [
            "architecture frozen",
            "hyperparameters frozen",
            "epoch count frozen",
            "relation configuration frozen",
            "candidate lists frozen",
            "statistical analysis plan frozen",
        ],

        "training_positive_segments": {
            "first":
                1,

            "last":
                60,

            "T0_role":
                "compressed_history_only",
        },

        "uses_T60_as_training_history":
            True,

        "uses_April_as_training_history":
            False,

        "epoch_count":
            (
                "exact epoch count selected "
                "from T57/T58/T59 validation"
            ),

        "April_checkpoint_selection":
            False,

        "April_early_stopping":
            False,
    }

    april_lock = {
        "prediction_origin":
            "2026-03-31",

        "evaluation_start":
            "2026-04-01",

        "evaluation_end":
            "2026-04-30",

        "candidate_startups":
            311_266,

        "eligible_events":
            6_313,

        "eligible_primary_events":
            1_690,

        "performance_status":
            "UNTOUCHED",

        "allowed_primary_confirmatory_models": [
            (
                "B0 reproduced ITRS baseline "
                "re-executed under Phase-8 protocol"
            ),
            (
                "single final Phase-8 selected model"
            ),
        ],

        "model_selection_allowed":
            False,

        "architecture_change_after_scoring":
            False,

        "hyperparameter_change_after_scoring":
            False,

        "checkpoint_change_after_scoring":
            False,

        "negative_resampling_after_scoring":
            False,
    }

    print(
        "Final fit training positives: T1-T60"
    )

    print(
        "April origin:                2026-03-31"
    )

    print(
        "April eligible events:       6,313"
    )

    print(
        "April primary events:        1,690"
    )

    print(
        "April performance:           LOCKED"
    )

    # =========================================================================
    # 8. Complete protocol
    # =========================================================================

    protocol = {
        "schema_version":
            "PHASE_8_0_10_EVALUATION_PROTOCOL_V1",

        "status":
            "FROZEN",

        "scientific_role":
            (
                "outcome_blind_development_selection_"
                "confirmatory_protocol"
            ),

        "temporal_design": {
            "development":
                (
                    "expanding-window rolling-origin "
                    "validation on T57/T58/T59"
                ),

            "T0_role":
                "compressed_history_only",

            "T60_role":
                (
                    "diagnostically spent from Phases 6-7; "
                    "prohibited for Phase-8 model selection; "
                    "allowed as training history only after "
                    "model-selection freeze for April final fit"
                ),

            "confirmatory":
                (
                    "fixed-origin April 2026 "
                    "at prediction origin 2026-03-31"
                ),
        },

        "model_selection":
            selection_contract,

        "evaluation_candidate_generation":
            candidate_contract,

        "graph_information_policy": {
            "strict_track":
                (
                    "timestamped acquisition relations "
                    "available by prediction origin"
                ),

            "founder_track_artifact_name":
                "founder_proxy",

            "founder_track_phase8_interpretation":
                (
                    "shared-founder relation admitted only "
                    "when both endpoint organizations have "
                    "known founded_on <= prediction origin "
                    "and direct target-pair leakage is absent"
                ),

            "founder_effective_time_semantics":
                (
                    "max(src.founded_on, dst.founded_on) "
                    "acts as the earliest semantic time at "
                    "which both organizations exist"
                ),

            "database_observation_time_claimed":
                False,

            "strict_acquisition_only_track_retained":
                True,
        },

        "baseline_comparison": {
            "B0":
                (
                    "reproduced ITRS architecture under "
                    "the Phase-8 protocol"
                ),

            "reuse_Phase6_test_metric_as_Phase8_baseline":
                False,

            "fair_comparison_requires_same": [
                "eligible target cases",
                "candidate universe",
                "negative candidate lists",
                "graph-information policy",
                "ranking metrics",
            ],
        },

        "final_fit":
            final_fit,

        "April_confirmatory_lock":
            april_lock,

        "not_yet_materialized": [
            (
                "evaluation negative candidate lists"
            ),
            (
                "new model architecture"
            ),
            (
                "hyperparameter configurations"
            ),
            (
                "relation ablation matrix"
            ),
            (
                "final statistical analysis plan"
            ),
        ],

        "model_training_performed":
            False,

        "model_selection_performed":
            False,

        "April_performance_inspected":
            False,
    }

    write_json(
        PROTOCOL_OUTPUT,
        protocol,
    )

    # =========================================================================
    # 9. Manifest + hashes
    # =========================================================================

    manifest = {
        "phase":
            "8.0.10",

        "authoritative_inputs":
            source_hashes,

        "outputs": [
            str(
                FOLD_OUTPUT
            ),
            str(
                SELECTION_OUTPUT
            ),
            str(
                PROTOCOL_OUTPUT
            ),
        ],

        "prohibited_operations": [
            "model_training",
            "checkpoint_loading",
            "inference",
            "metric_scoring",
            "negative_generation",
            "model_selection",
            "April_confirmatory_scoring",
        ],
    }

    write_json(
        MANIFEST_OUTPUT,
        manifest,
    )

    derived_hashes = {
        str(FOLD_OUTPUT):
            sha256_file(
                FOLD_OUTPUT
            ),

        str(SELECTION_OUTPUT):
            sha256_file(
                SELECTION_OUTPUT
            ),

        str(PROTOCOL_OUTPUT):
            sha256_file(
                PROTOCOL_OUTPUT
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
    # 10. Final
    # =========================================================================

    separator()

    print(
        "PHASE 8.0.10 EVALUATION "
        "PROTOCOL FREEZE COMPLETE"
    )

    separator()

    print()

    print(
        "ROLLING-ORIGIN VALIDATION:       "
        "FROZEN -> T57 / T58 / T59"
    )

    print(
        "PRIMARY SELECTION ENDPOINT:       "
        "FROZEN -> mean primary NDCG@10"
    )

    print(
        "MAXIMUM TRAINING EPOCHS:          "
        "FROZEN -> 20"
    )

    print(
        "T60 PHASE-8 MODEL SELECTION:      "
        "FORBIDDEN"
    )

    print(
        "T60 FINAL-FIT HISTORY:            "
        "ALLOWED AFTER SELECTION FREEZE"
    )

    print(
        "APRIL CONFIRMATORY ORIGIN:        "
        "FROZEN -> 2026-03-31"
    )

    print(
        "APRIL PRIMARY CASES:              "
        "FROZEN -> 1,690"
    )

    print(
        "APRIL PERFORMANCE:                "
        "UNTOUCHED"
    )

    print(
        "NEGATIVE LISTS GENERATED:         NO"
    )

    print(
        "MODEL TRAINING PERFORMED:         NO"
    )

    print()

    print(
        "Next:"
    )

    print(
        "Materialize and freeze the "
        "T57/T58/T59/April evaluation "
        "candidate lists under this contract."
    )

    print()

    print(
        f"Outputs written to:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()