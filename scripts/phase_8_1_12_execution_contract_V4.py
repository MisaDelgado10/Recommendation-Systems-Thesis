from pathlib import Path
from copy import deepcopy
import hashlib
import json
import math


# =============================================================================
# PHASE 8.1.12 — FINAL EXECUTION CONTRACT V4
# =============================================================================
#
# Scientific role:
#
#   FINAL_METHOD_DESIGN_FREEZE_CORRECTED
#
# V4 supersedes V3 before any Phase-8 candidate model is trained.
#
# Corrections / closures:
#
#   1. Authoritative outcome-independent COLD_t semantics.
#   2. Origin-referenced Startup-conditioned founder history H(s,t).
#   3. Corrected Phase-8 graph schema and branch semantics.
#   4. Exact training-only preprocessing population.
#   5. Phase-8-specific deterministic randomness namespaces.
#   6. Parameter-name / semantic-component initialization policy.
#   7. Exact two-way clustered bootstrap algorithm.
#   8. Transparent prior-performance observation disclosure.
#   9. Correct attribution of the development ablation list.
#
# No training, model scoring, T60 evaluation, April evaluation, feature
# discovery, architecture search, or performance-driven design occurs here.
# =============================================================================


BASE = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation"
)

EXEC_DIR = (
    BASE
    / "phase_8_1_12_execution_contract"
)

V3_PATH = (
    EXEC_DIR
    / "phase_8_1_12_execution_contract_V3.json"
)

GRAPH_SCHEMA_V2 = (
    BASE
    / "phase_8_1_9_heterogeneous_graph_schema"
    / "phase_8_1_9_graph_schema_V2.json"
)

CONSISTENCY_V2 = (
    BASE
    / "phase_8_1_10a_inductive_consistency_closure"
    / "phase_8_1_10a_consistency_amendment_V2.json"
)

ARCHITECTURE_V1 = (
    BASE
    / "phase_8_1_11_exact_neural_architecture"
    / "phase_8_1_11_architecture_contract_V1.json"
)

BRANCH_V2 = (
    BASE
    / "phase_8_1_11a_model_family_branch_semantics"
    / "phase_8_1_11a_branch_semantics_amendment_V2.json"
)

COLD_V1 = (
    BASE
    / "phase_8_1_12a_outcome_independent_cold_start_masking"
    / "phase_8_1_12a_outcome_independent_masking_contract_V1.json"
)

FOUNDER_V1 = (
    BASE
    / "phase_8_1_12b_origin_referenced_founder_history"
    / "phase_8_1_12b_origin_referenced_founder_history_contract_V1.json"
)

PHASE4_INIT = Path(
    "data/experimental/phase_4/"
    "initialization_contract/"
    "phase_4_7_1b_neural_initialization_contract.json"
)

PHASE5_NEGATIVE = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_1_1d_training_negative_sampling_runtime_contract.json"
)

PHASE5_OPTIMIZER = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_2_2_training_control_optimizer_runtime_contract.json"
)

PHASE8_2_1C_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_2_framework_development/"
    "phase_8_2_1_feature_materialization"
)

ORIGIN_MASK_MANIFEST_V2 = (
    PHASE8_2_1C_DIR
    / "phase_8_2_1c_manifest_V2.json"
)

ORIGIN_MASKS_V2 = (
    PHASE8_2_1C_DIR
    / "phase_8_2_1c_origin_startup_masks_V2.parquet"
)

PREPROCESSING_POP_V2 = (
    PHASE8_2_1C_DIR
    / "phase_8_2_1c_preprocessing_fit_population_V2.parquet"
)

ORIGIN_SUMMARY_V2 = (
    PHASE8_2_1C_DIR
    / "phase_8_2_1c_origin_population_summary_V2.csv"
)


V4_PATH = (
    EXEC_DIR
    / "phase_8_1_12_execution_contract_V4.json"
)

V4_MANIFEST = (
    EXEC_DIR
    / "phase_8_1_12_execution_contract_manifest_V4.json"
)

V4_HASHES = (
    EXEC_DIR
    / "phase_8_1_12_execution_contract_sha256_V4.json"
)


BASE_SEED = 42

EXPECTED_MODELS = [
    "B1a",
    "B1b",
    "M1",
    "M2",
]

EXPECTED_DEV_ORIGINS = [
    "T57",
    "T58",
    "T59",
]


def sep(char="=", width=120):
    print(char * width)


def require(condition, message):

    if not bool(condition):
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

        # V4 must be standards-compliant JSON.
        # In particular, no NaN placeholders from V3 survive.
        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )


def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def no_nonfinite(obj):

    if isinstance(obj, float):

        return math.isfinite(obj)

    if isinstance(obj, dict):

        return all(
            no_nonfinite(v)
            for v
            in obj.values()
        )

    if isinstance(obj, list):

        return all(
            no_nonfinite(v)
            for v
            in obj
        )

    return True


def main():

    sep()

    print(
        "PHASE 8.1.12 — "
        "FINAL EXECUTION CONTRACT V4"
    )

    sep()

    print()

    print(
        "Scientific role:             "
        "FINAL_METHOD_DESIGN_FREEZE_CORRECTED"
    )

    print("Feature discovery reopened:  NO")
    print("Architecture search reopened:NO")
    print("Phase-8 model trained:       NO")
    print("Phase-8 performance seen:    NO")
    print("T60 scored:                  NO")
    print("April analyzed:              NO")

    # =========================================================================
    # 1. Prerequisite gates
    # =========================================================================

    sep("-")

    print(
        "8.1.12-V4.1 — PREREQUISITE GATES"
    )

    sep("-")

    required = [
        V3_PATH,
        GRAPH_SCHEMA_V2,
        CONSISTENCY_V2,
        ARCHITECTURE_V1,
        BRANCH_V2,
        COLD_V1,
        FOUNDER_V1,
        PHASE4_INIT,
        PHASE5_NEGATIVE,
        PHASE5_OPTIMIZER,
        ORIGIN_MASK_MANIFEST_V2,
        ORIGIN_MASKS_V2,
        PREPROCESSING_POP_V2,
        ORIGIN_SUMMARY_V2,
    ]

    for path in required:

        require(
            path.exists(),
            f"Missing prerequisite: {path}",
        )

        print(
            f"FOUND {path}"
        )

    v3 = read_json(
        V3_PATH
    )

    graph_v2 = read_json(
        GRAPH_SCHEMA_V2
    )

    consistency_v2 = read_json(
        CONSISTENCY_V2
    )

    architecture_v1 = read_json(
        ARCHITECTURE_V1
    )

    branch_v2 = read_json(
        BRANCH_V2
    )

    cold_v1 = read_json(
        COLD_V1
    )

    founder_v1 = read_json(
        FOUNDER_V1
    )

    require(
        v3.get("schema_version")
        ==
        "PHASE_8_1_12_EXECUTION_CONTRACT_V3",
        "Unexpected V3 execution-contract schema.",
    )

    require(
        v3.get("status")
        ==
        "FROZEN",
        "V3 is not frozen.",
    )

    for name, obj in [
        ("8.1.9 V2", graph_v2),
        ("8.1.10A V2", consistency_v2),
        ("8.1.11 V1", architecture_v1),
        ("8.1.11A V2", branch_v2),
        ("8.1.12A V1", cold_v1),
        ("8.1.12B V1", founder_v1),
    ]:

        require(
            obj.get("status")
            ==
            "FROZEN",
            f"{name} is not FROZEN.",
        )

    print()

    print(
        "Corrected semantic prerequisites: PASS"
    )

    # =========================================================================
    # 2. Detect V3 clauses that require supersession
    # =========================================================================

    sep("-")

    print(
        "8.1.12-V4.2 — V3 SUPERSESSION TARGETS"
    )

    sep("-")

    old_namespaces = {
        row[
            "stream"
        ]:
            row[
                "namespace"
            ]
        for row
        in v3[
            "randomness"
        ]
    }

    require(
        old_namespaces[
            "training_negatives"
        ]
        ==
        "ITRS_PHASE5_TRAIN_NEGATIVE",
        "Expected V3 legacy negative namespace not found.",
    )

    require(
        old_namespaces[
            "training_order"
        ]
        ==
        "ITRS_PHASE5_TRAIN_ORDER",
        "Expected V3 legacy order namespace not found.",
    )

    require(
        v3.get(
            "performance_observed_before_method_freeze"
        )
        is False,
        (
            "Expected historical V3 global performance "
            "disclosure was not found."
        ),
    )

    require(
        (
            "allowed_predeclared_ablations"
            in
            v3[
                "development_ablation_policy"
            ]
        ),
        "Expected V3 ablation attribution key not found.",
    )

    require(
        (
            "phase_8_1_11a_branch_semantics_amendment_V1.json"
            in
            v3[
                "source_contracts"
            ][
                "branch_semantics"
            ]
        ),
        "Expected V3 branch-semantics V1 dependency not found.",
    )

    print(
        "Legacy Phase-5 training RNG namespaces:       DETECTED"
    )

    print(
        "Global no-prior-performance claim:           DETECTED"
    )

    print(
        "Ambiguous predeclared-ablation attribution:  DETECTED"
    )

    print(
        "Superseded 8.1.11A V1 dependency:            DETECTED"
    )

    # =========================================================================
    # 3. Freeze corrected preprocessing execution contract
    # =========================================================================

    sep("-")

    print(
        "8.1.12-V4.3 — PREPROCESSING CONTRACT"
    )

    sep("-")

    preprocessing = {
        "applies_to_models":
            EXPECTED_MODELS,

        "shared_across_models_within_origin":
            True,

        "fit_once_per_origin":
            True,

        "authoritative_fit_population_artifact":
            str(
                PREPROCESSING_POP_V2
            ),

        "authoritative_cold_mask_artifact":
            str(
                ORIGIN_MASKS_V2
            ),

        "fit_population_definition":
            (
                "Unique Startups appearing in supervised positive "
                "training events T1 through the segment immediately "
                "before prediction origin, exactly as materialized "
                "by Phase 8.2.1C V2."
            ),

        "T0_counts_for_cold_warm_history":
            True,

        "T0_used_for_preprocessor_fit":
            False,

        "cold_t_raw_values_may_fit":
            False,

        "evaluation_positive_ids_may_fit":
            False,

        "evaluation_negative_ids_may_fit":
            False,

        "evaluation_candidate_matrix_may_fit":
            False,

        "future_evaluation_membership_may_fit":
            False,

        "tfidf_vocabulary":
            (
                "Fit only on Startup text rows belonging to the "
                "authoritative origin-specific preprocessing-fit "
                "population."
            ),

        "svd_fit":
            (
                "Fit only after TF-IDF is frozen, using the same "
                "origin-specific preprocessing-fit population."
            ),

        "category_vocabulary":
            (
                "Fit only from category values observed in the "
                "origin-specific preprocessing-fit population."
            ),

        "category_group_vocabulary":
            (
                "Fit only from category-group values observed in "
                "the origin-specific preprocessing-fit population."
            ),

        "geography_vocabulary":
            (
                "Fit only from geography values observed in the "
                "origin-specific preprocessing-fit population."
            ),

        "company_type_vocabulary":
            (
                "Fit only from values observed in the "
                "origin-specific preprocessing-fit population."
            ),

        "founder_sector_vocabulary":
            (
                "For M1/M2 P12, fit only from the origin-specific "
                "training-side Startup population using each focal "
                "Startup's H(s,t)."
            ),

        "numeric_scalers":
            (
                "Fit only on numeric values belonging to the "
                "origin-specific preprocessing-fit population."
            ),

        "after_fit_transform_scope":
            (
                "Once all preprocessing parameters for an origin "
                "are frozen, transform all required origin-eligible "
                "Startup rows without refitting."
            ),

        "cold_t_during_parameter_fit":
            (
                "Even if transformed with already-frozen preprocessing, "
                "all COLD_t Startup feature slots remain ZERO during "
                "model parameter fitting."
            ),

        "cold_t_during_inference":
            (
                "After model parameters are frozen, restore the allowed "
                "origin-time transformed features for COLD_t Startups."
            ),

        "preprocessing_refit_at_inference":
            False,

        "B0_policy":
            (
                "B0 preserves its frozen baseline preprocessing/"
                "representation protocol and is not retrofitted with "
                "the inductive Phase-8 preprocessing pipeline."
            ),
    }

    for key in [
        "fit_once_per_origin",
        "cold_t_raw_values_may_fit",
        "evaluation_candidate_matrix_may_fit",
        "preprocessing_refit_at_inference",
    ]:

        print(
            f"{key:<42} {preprocessing[key]}"
        )

    # =========================================================================
    # 4. Freeze exact deterministic seed derivation
    # =========================================================================

    sep("-")

    print(
        "8.1.12-V4.4 — RANDOMNESS NAMESPACES"
    )

    sep("-")

    seed_derivation = {
        "base_seed":
            BASE_SEED,

        "hash":
            "SHA256",

        "encoding":
            "UTF-8",

        "payload_template":
            (
                "{namespace}|base_seed=42|scope={scope_id}|"
                "epoch={epoch_index_or_NA}|component={component_or_NA}|"
                "parameter={parameter_or_NA}|shape={shape_or_NA}|"
                "dtype={dtype_or_NA}|replicate={replicate_or_NA}|"
                "attempt={attempt_or_0}|dimension={dimension_or_NA}"
            ),

        "seed_u64":
            (
                "Interpret the first 8 SHA256 digest bytes as an "
                "unsigned big-endian 64-bit integer."
            ),

        "numpy_PCG64_seed":
            "seed_u64",

        "torch_CPU_Generator_seed":
            "seed_u64",

        "scikit_learn_random_state":
            "seed_u64 modulo 2**32",
    }

    execution_scopes = {
        "DEV_T57":
            "Development fit using T1-T56.",

        "DEV_T58":
            "Development fit using T1-T57.",

        "DEV_T59":
            "Development fit using T1-T58.",

        "T60_FINAL_IMPROVED":
            (
                "B1a/B1b/M1/M2 held-out reporting fit "
                "using T1-T59."
            ),

        "APRIL_FINAL_M2":
            (
                "M2 confirmatory final fit using T1-T60."
            ),
    }

    randomness = {
        "seed_derivation":
            seed_derivation,

        "execution_scopes":
            execution_scopes,

        "streams": {
            "training_negatives": {
                "namespace":
                    "ITRS_PHASE8_TRAIN_NEGATIVE",

                "generator":
                    "numpy.random.PCG64",

                "seed_key":
                    "namespace + scope_id + zero-based epoch_index",

                "same_stream_across_B1a_B1b_M1_M2_within_scope_epoch":
                    True,

                "model_name_enters_seed":
                    False,

                "resample_frequency":
                    "once per epoch",

                "negative_positive_ratio":
                    4,

                "future_labels_used":
                    False,
            },

            "training_order": {
                "namespace":
                    "ITRS_PHASE8_TRAIN_ORDER",

                "generator":
                    "numpy.random.PCG64",

                "seed_key":
                    "namespace + scope_id + zero-based epoch_index",

                "same_stream_across_B1a_B1b_M1_M2_within_scope_epoch":
                    True,

                "model_name_enters_seed":
                    False,

                "shuffle_frequency":
                    "once per epoch",
            },

            "truncated_svd": {
                "namespace":
                    "ITRS_PHASE8_SVD",

                "generator":
                    "scikit-learn random_state",

                "seed_key":
                    "namespace + scope_id",

                "same_preprocessor_across_phase8_candidate_models":
                    True,
            },

            "neural_initialization": {
                "namespace":
                    "ITRS_PHASE8_NEURAL_INIT",

                "generator":
                    "torch CPU Generator",

                "seed_key":
                    (
                        "namespace + scope_id + semantic_component_id + "
                        "parameter_name + shape + dtype"
                    ),

                "parameter_registration_order_affects_seed":
                    False,
            },

            "clustered_bootstrap": {
                "namespace":
                    "ITRS_PHASE8_TWO_WAY_CLUSTERED_BOOTSTRAP",

                "generator":
                    "numpy.random.PCG64",

                "scope_id":
                    "APRIL_PRIMARY_M2_MINUS_B0",

                "replicate_specific_seed":
                    True,
            },
        },
    }

    print(
        "training_negatives  -> ITRS_PHASE8_TRAIN_NEGATIVE"
    )

    print(
        "training_order      -> ITRS_PHASE8_TRAIN_ORDER"
    )

    print(
        "neural_init         -> ITRS_PHASE8_NEURAL_INIT"
    )

    print(
        "SVD                 -> ITRS_PHASE8_SVD"
    )

    print(
        "bootstrap           -> ITRS_PHASE8_TWO_WAY_CLUSTERED_BOOTSTRAP"
    )

    # =========================================================================
    # 5. Freeze exact neural initialization semantics
    # =========================================================================

    sep("-")

    print(
        "8.1.12-V4.5 — NEURAL INITIALIZATION CONTRACT"
    )

    sep("-")

    initialization = {
        "base_seed":
            BASE_SEED,

        "canonical_initialization_device":
            "CPU",

        "startup_identity_parameter":
            False,

        "person_identity_parameter":
            False,

        "matrix_initialization":
            (
                "torch.nn.init.kaiming_normal_"
                "(a=0, mode='fan_in', nonlinearity='relu')"
            ),

        "bias_initialization":
            "zeros",

        "phase8_candidate_model_policy":
            (
                "Each trainable tensor is initialized from a dedicated "
                "torch CPU Generator whose seed is derived from the "
                "frozen semantic parameter key; parameter traversal or "
                "module registration order cannot change initialization."
            ),

        "semantic_parameter_key":
            (
                "scope_id + semantic_component_id + parameter_name + "
                "shape + dtype"
            ),

        "shared_tensor_rule":
            (
                "Two Phase-8 candidate models receive identical initial "
                "values for a tensor if and only if the tensor has the "
                "same semantic_component_id, parameter_name, shape, dtype "
                "and execution scope."
            ),

        "different_architecture_rule":
            (
                "No claim is made that entire model families share one "
                "initialization stream. Model-specific tensors have "
                "distinct semantic component IDs."
            ),

        "implementation_audit_requirement":
            (
                "Before the first optimizer step, emit a complete "
                "trainable-parameter namespace table containing model, "
                "scope, semantic component ID, parameter name, shape, "
                "dtype, derived seed and initialization rule; every "
                "trainable tensor must occur exactly once."
            ),

        "B0_T60_policy":
            (
                "Do not reinitialize or retrain B0 for T60; reuse the "
                "frozen Phase-6 checkpoint/scores."
            ),

        "B0_April_policy":
            (
                "The April B0 reference final fit preserves the frozen "
                "legacy Phase-4 initialization and Phase-5 baseline "
                "runtime protocol rather than adopting Phase-8 "
                "candidate-model initialization."
            ),
    }

    print(
        "Parameter-order-dependent init:       NO"
    )

    print(
        "Shared semantic tensors aligned:      YES"
    )

    print(
        "Different tensors forced identical:   NO"
    )

    print(
        "Startup/Person identity parameters:   NO"
    )

    # =========================================================================
    # 6. Freeze exact April two-way clustered bootstrap
    # =========================================================================

    sep("-")

    print(
        "8.1.12-V4.6 — TWO-WAY CLUSTERED BOOTSTRAP"
    )

    sep("-")

    bootstrap = {
        "name":
            "TWO_WAY_PIGEONHOLE_CLUSTER_BOOTSTRAP",

        "purpose":
            (
                "Uncertainty for the paired M2 minus B0 performance "
                "difference on the frozen April PRIMARY cohort."
            ),

        "population":
            (
                "Frozen April PRIMARY warm-investor / "
                "investment-cold-startup evaluation cases only."
            ),

        "evaluation_cases":
            1690,

        "paired_models": [
            "B0",
            "M2",
        ],

        "replicates":
            10_000,

        "base_seed":
            BASE_SEED,

        "namespace":
            "ITRS_PHASE8_TWO_WAY_CLUSTERED_BOOTSTRAP",

        "cluster_dimensions": [
            "investor_id",
            "startup_id",
        ],

        "sampling_algorithm": [
            (
                "Let I be the unique investor IDs and S the unique "
                "Startup IDs in the frozen April PRIMARY cohort."
            ),
            (
                "For each replicate r, independently sample |I| investor "
                "cluster labels with replacement from I and |S| Startup "
                "cluster labels with replacement from S."
            ),
            (
                "Let a_i be the multiplicity of investor i and b_s the "
                "multiplicity of Startup s in those two independent "
                "cluster samples."
            ),
            (
                "For original evaluation case k=(i,s), assign bootstrap "
                "weight w_k = a_i * b_s."
            ),
            (
                "Compute the replicate paired effect as "
                "sum_k(w_k * d_k) / sum_k(w_k), where d_k is the "
                "per-case M2 minus B0 metric contribution."
            ),
            (
                "If sum_k(w_k)=0, deterministically redraw that replicate "
                "using attempt=1,2,... in the frozen seed payload until "
                "the first positive-denominator draw is obtained."
            ),
        ],

        "investor_and_startup_draws_independent":
            True,

        "point_estimate":
            (
                "Unweighted arithmetic mean across original PRIMARY "
                "evaluation cases of the paired per-case M2 minus B0 "
                "metric contribution."
            ),

        "primary_metric":
            "NDCG@10",

        "secondary_metric":
            "HR@10",

        "same_cluster_draws_used_for_primary_and_secondary_metrics":
            True,

        "confidence_interval":
            {
                "level":
                    0.95,

                "method":
                    "EMPIRICAL_PERCENTILE",

                "lower_percentile":
                    2.5,

                "upper_percentile":
                    97.5,

                "quantile_interpolation":
                    "linear",
            },

        "model_selection_from_bootstrap":
            False,

        "architecture_change_from_bootstrap":
            False,

        "feature_change_from_bootstrap":
            False,
    }

    print(
        "Algorithm:        TWO_WAY_PIGEONHOLE_CLUSTER_BOOTSTRAP"
    )

    print(
        "Replicates:       10,000"
    )

    print(
        "Clusters:         investor_id + startup_id"
    )

    print(
        "CI:               95% empirical percentile"
    )

    print(
        "Estimand:         paired mean M2 - B0"
    )

    # =========================================================================
    # 7. Transparent observation / design provenance
    # =========================================================================

    sep("-")

    print(
        "8.1.12-V4.7 — PRIOR PERFORMANCE OBSERVATION DISCLOSURE"
    )

    sep("-")

    observation_disclosure = {
        "phase6_B0_baseline_performance_observed":
            True,

        "phase7_posthoc_diagnostics_observed":
            True,

        "phase7_diagnostics_informed_problem_focus":
            True,

        "phase8_candidate_models":
            EXPECTED_MODELS,

        "phase8_candidate_model_training_performed_before_V4":
            False,

        "phase8_candidate_model_performance_observed_before_V4":
            False,

        "T60_phase8_candidate_model_performance_observed_before_V4":
            False,

        "April_performance_observed_before_V4":
            False,

        "interpretation":
            (
                "The inductive cold-start research focus was informed by "
                "the frozen B0 reproduction and Phase-7 post-hoc "
                "diagnostics. No B1a/B1b/M1/M2 performance was observed "
                "before this corrected final method-design freeze."
            ),
    }

    for key in [
        "phase6_B0_baseline_performance_observed",
        "phase7_posthoc_diagnostics_observed",
        "phase8_candidate_model_performance_observed_before_V4",
        "April_performance_observed_before_V4",
    ]:

        print(
            f"{key:<58} "
            f"{observation_disclosure[key]}"
        )

    # =========================================================================
    # 8. Correct ablation-list attribution
    # =========================================================================

    sep("-")

    print(
        "8.1.12-V4.8 — DEVELOPMENT ABLATION PROVENANCE"
    )

    sep("-")

    old_ablation_policy = deepcopy(
        v3[
            "development_ablation_policy"
        ]
    )

    frozen_ablations = old_ablation_policy.pop(
        "allowed_predeclared_ablations"
    )

    ablation_policy = old_ablation_policy

    ablation_policy[
        "allowed_frozen_development_ablations"
    ] = frozen_ablations

    ablation_policy[
        "freeze_timing"
    ] = (
        "Frozen no later than Phase 8.1.12 V3 and before any "
        "B1a/B1b/M1/M2 model performance was observed."
    )

    ablation_policy[
        "phase8_0_12_preregistration_attribution"
    ] = (
        "NOT_CLAIMED_BY_THIS_CONTRACT"
    )

    ablation_policy[
        "provenance_note"
    ] = (
        "Do not describe the entire list as preregistered in Phase "
        "8.0.12 unless an individual ablation is independently "
        "documented there. V4 only claims that this list was frozen "
        "before Phase-8 candidate-model performance."
    )

    ablation_policy[
        "phase8_candidate_model_performance_used_to_define_list"
    ] = False

    print(
        "Ablation list changed:                     NO"
    )

    print(
        "Phase-8 performance used to define list:   NO"
    )

    print(
        "Phase-8.0.12 preregistration claimed:      NO"
    )

    # =========================================================================
    # 9. Construct V4 from V3
    # =========================================================================

    sep("-")

    print(
        "8.1.12-V4.9 — CONSTRUCT FINAL CORRECTED CONTRACT"
    )

    sep("-")

    v4 = deepcopy(
        v3
    )

    v4[
        "schema_version"
    ] = (
        "PHASE_8_1_12_EXECUTION_CONTRACT_V4"
    )

    v4[
        "scientific_role"
    ] = (
        "FINAL_METHOD_DESIGN_FREEZE_CORRECTED"
    )

    v4[
        "status"
    ] = "FROZEN"

    # Remove globally misleading V3 disclosure.
    v4.pop(
        "performance_observed_before_method_freeze",
        None,
    )

    v4[
        "prior_performance_observation_disclosure"
    ] = observation_disclosure

    v4[
        "preprocessing"
    ] = preprocessing

    v4[
        "randomness"
    ] = randomness

    v4[
        "initialization"
    ] = initialization

    v4[
        "development_ablation_policy"
    ] = ablation_policy

    # -------------------------------------------------------------------------
    # Exact bootstrap specification.
    # -------------------------------------------------------------------------

    v4[
        "April_confirmatory"
    ][
        "bootstrap"
    ] = bootstrap

    # Remove redundant underspecified V3 bootstrap keys.
    v4[
        "April_confirmatory"
    ].pop(
        "bootstrap_seed",
        None,
    )

    v4[
        "April_confirmatory"
    ].pop(
        "cluster_dimensions",
        None,
    )

    # -------------------------------------------------------------------------
    # Add deterministic execution scopes to development folds.
    # -------------------------------------------------------------------------

    for fold in (
        v4[
            "development_folds"
        ]
    ):

        origin = (
            fold[
                "origin"
            ]
        )

        require(
            origin
            in
            EXPECTED_DEV_ORIGINS,
            (
                "Unexpected development origin: "
                f"{origin}"
            ),
        )

        fold[
            "randomness_scope"
        ] = (
            f"DEV_{origin}"
        )

        fold[
            "preprocessing_source"
        ] = (
            str(
                PREPROCESSING_POP_V2
            )
        )

        fold[
            "cold_mask_source"
        ] = (
            str(
                ORIGIN_MASKS_V2
            )
        )

    # -------------------------------------------------------------------------
    # Clarify T60 / April runtime scopes.
    # -------------------------------------------------------------------------

    for stage in (
        v4[
            "T60_April_execution"
        ]
    ):

        stage_name = (
            stage[
                "stage"
            ]
        )

        if (
            stage_name
            ==
            "T60_IMPROVED_MODELS"
        ):

            stage[
                "randomness_scope"
            ] = (
                "T60_FINAL_IMPROVED"
            )

            stage[
                "preprocessing_policy"
            ] = (
                "PHASE8_ORIGIN_SPECIFIC_TRAINING_ONLY"
            )

            stage[
                "candidate_model_performance_may_change_design"
            ] = False

        elif (
            stage_name
            ==
            "T60_B0"
        ):

            stage[
                "randomness_scope"
            ] = (
                "NONE_REUSE_EXISTING_PHASE6"
            )

            stage[
                "preprocessing_policy"
            ] = (
                "REUSE_FROZEN_PHASE6"
            )

        elif (
            stage_name
            ==
            "APRIL_B0_FINAL_FIT"
        ):

            stage[
                "randomness_scope"
            ] = (
                "LEGACY_PHASE4_PHASE5_BASELINE"
            )

            stage[
                "preprocessing_policy"
            ] = (
                "FROZEN_BASELINE_PROTOCOL"
            )

        elif (
            stage_name
            ==
            "APRIL_M2_FINAL_FIT"
        ):

            stage[
                "randomness_scope"
            ] = (
                "APRIL_FINAL_M2"
            )

            stage[
                "preprocessing_policy"
            ] = (
                "PHASE8_ORIGIN_SPECIFIC_TRAINING_ONLY"
            )

        else:

            raise ValueError(
                f"Unexpected T60/April stage: {stage_name}"
            )

    # -------------------------------------------------------------------------
    # Correct source-contract chain.
    # -------------------------------------------------------------------------

    v4[
        "source_contracts"
    ] = {
        "input_contract":
            (
                "data/experimental/phase_8/"
                "phase_8_1_inductive_startup_representation/"
                "phase_8_1_7_input_contract_amendment/"
                "phase_8_1_7_contract_amendment_V2.json"
            ),

        "graph_schema":
            str(
                GRAPH_SCHEMA_V2
            ),

        "inductive_consistency":
            str(
                CONSISTENCY_V2
            ),

        "architecture":
            str(
                ARCHITECTURE_V1
            ),

        "branch_semantics":
            str(
                BRANCH_V2
            ),

        "outcome_independent_cold_mask":
            str(
                COLD_V1
            ),

        "origin_referenced_founder_history":
            str(
                FOUNDER_V1
            ),

        "origin_population_materialization":
            str(
                ORIGIN_MASK_MANIFEST_V2
            ),

        "legacy_initialization_reference":
            str(
                PHASE4_INIT
            ),

        "legacy_negative_sampling_reference":
            str(
                PHASE5_NEGATIVE
            ),

        "optimizer_reference":
            str(
                PHASE5_OPTIMIZER
            ),
    }

    # -------------------------------------------------------------------------
    # Explicit supersession ledger.
    # -------------------------------------------------------------------------

    v4[
        "supersession"
    ] = {
        "supersedes":
            str(
                V3_PATH
            ),

        "V3_file_modified":
            False,

        "V3_artifact_preserved":
            True,

        "corrected_clauses": [
            (
                "Outcome-independent COLD_t fit/inference "
                "semantics."
            ),
            (
                "Origin-referenced Startup-conditioned "
                "founder history H(s,t)."
            ),
            (
                "Corrected 8.1.9 V2 graph placement and "
                "strict pre-origin relation gates."
            ),
            (
                "Corrected 8.1.11A V2 sink-only branch "
                "semantics."
            ),
            (
                "Training-only preprocessing population."
            ),
            (
                "Phase-8-specific deterministic randomness "
                "namespaces."
            ),
            (
                "Order-independent named-parameter "
                "initialization."
            ),
            (
                "Exact two-way clustered bootstrap "
                "algorithm."
            ),
            (
                "Transparent prior-performance observation "
                "disclosure."
            ),
            (
                "Correct ablation-list provenance."
            ),
        ],

        "feature_inventory_changed":
            False,

        "architecture_topology_changed":
            False,

        "model_family_changed":
            False,

        "primary_endpoint_changed":
            False,

        "candidate_matrix_changed":
            False,

        "development_folds_changed":
            False,

        "maximum_epochs_changed":
            False,

        "optimizer_changed":
            False,

        "performance_based_change":
            False,

        "phase8_candidate_model_performance_seen":
            False,
    }

    # -------------------------------------------------------------------------
    # Method-design gate.
    # -------------------------------------------------------------------------

    v4[
        "method_design_closed"
    ] = True

    v4[
        "implementation_gate"
    ] = {
        "feature_discovery_closed":
            True,

        "graph_schema_closed":
            True,

        "identity_policy_closed":
            True,

        "architecture_closed":
            True,

        "execution_policy_closed":
            True,

        "randomness_policy_closed":
            True,

        "bootstrap_policy_closed":
            True,

        "method_design_closed":
            True,

        "training_allowed_immediately":
            False,

        "reason_training_not_yet_allowed":
            (
                "The corrected V4 execution contract must first be "
                "formally adopted by a corrected Phase-8.2 "
                "implementation/materialization preflight. Learned "
                "preprocessing, graph tensors and model implementation "
                "must then be audited against V4 before the first "
                "optimizer step."
            ),

        "next_phase":
            (
                "Phase 8.2.0 V2 — corrected implementation/"
                "materialization preflight adoption."
            ),

        "next_project_block":
            "FRAMEWORK DEVELOPMENT + INTEGRATION",
    }

    v4[
        "next_step"
    ] = (
        "Phase 8.2.0 V2 — corrected implementation/materialization "
        "preflight adoption against the authoritative repaired contracts."
    )

    v4[
        "model_initialized"
    ] = False

    v4[
        "training_performed"
    ] = False

    v4[
        "T60_scored"
    ] = False

    v4[
        "April_analyzed"
    ] = False

    v4[
        "April_selection_allowed"
    ] = False

    v4[
        "T60_selection_allowed"
    ] = False

    v4[
        "architecture_change_after_dev_allowed"
    ] = False

    v4[
        "feature_change_after_dev_allowed"
    ] = False

    # =========================================================================
    # 10. Integrity audit
    # =========================================================================

    sep("-")

    print(
        "8.1.12-V4.10 — FINAL CONTRACT INTEGRITY AUDIT"
    )

    sep("-")

    serialized = json.dumps(
        v4,
        sort_keys=True,
        allow_nan=False,
    )

    require(
        no_nonfinite(v4),
        "V4 contains non-finite numeric values.",
    )

    for forbidden in [
        '"namespace": "ITRS_PHASE5_TRAIN_NEGATIVE"',
        '"namespace": "ITRS_PHASE5_TRAIN_ORDER"',
        '"namespace": "PHASE4_GLOBAL_NEURAL_SEED"',
        (
            "phase_8_1_11a_branch_semantics_"
            "amendment_V1.json"
        ),
        '"performance_observed_before_method_freeze"',
        '"allowed_predeclared_ablations"',
    ]:

        require(
            forbidden
            not in
            serialized,
            (
                "Superseded V3 clause survived: "
                f"{forbidden}"
            ),
        )

    for required_text in [
        "ITRS_PHASE8_TRAIN_NEGATIVE",
        "ITRS_PHASE8_TRAIN_ORDER",
        "ITRS_PHASE8_NEURAL_INIT",
        "ITRS_PHASE8_SVD",
        "ITRS_PHASE8_TWO_WAY_CLUSTERED_BOOTSTRAP",
        "TWO_WAY_PIGEONHOLE_CLUSTER_BOOTSTRAP",
        "phase_8_1_9_graph_schema_V2.json",
        "phase_8_1_10a_consistency_amendment_V2.json",
        "phase_8_1_11a_branch_semantics_amendment_V2.json",
        "phase_8_2_1c_preprocessing_fit_population_V2.parquet",
        "ALL_COLD_t",
    ]:

        require(
            required_text
            in
            serialized,
            (
                "Required V4 semantic marker missing: "
                f"{required_text}"
            ),
        )

    require(
        v4[
            "prior_performance_observation_disclosure"
        ][
            "phase8_candidate_model_performance_observed_before_V4"
        ]
        is False,
        (
            "Phase-8 candidate-model prior-performance "
            "disclosure is not false."
        ),
    )

    require(
        v4[
            "prior_performance_observation_disclosure"
        ][
            "phase6_B0_baseline_performance_observed"
        ]
        is True,
        (
            "Phase-6 baseline observation is not "
            "transparently disclosed."
        ),
    )

    require(
        v4[
            "April_confirmatory"
        ][
            "bootstrap"
        ][
            "replicates"
        ]
        ==
        10_000,
        "Bootstrap replicate count changed.",
    )

    require(
        v4[
            "April_confirmatory"
        ][
            "bootstrap"
        ][
            "cluster_dimensions"
        ]
        ==
        [
            "investor_id",
            "startup_id",
        ],
        "Bootstrap cluster dimensions changed.",
    )

    require(
        v4[
            "implementation_gate"
        ][
            "training_allowed_immediately"
        ]
        is False,
        "Training was incorrectly authorized.",
    )

    require(
        v4[
            "method_design_closed"
        ]
        is True,
        "Method design is not closed.",
    )

    print(
        "Corrected semantic chain:              PASS"
    )

    print(
        "Training-only preprocessing:           PASS"
    )

    print(
        "Phase-8 RNG namespaces:                PASS"
    )

    print(
        "Order-independent initialization:      PASS"
    )

    print(
        "Exact two-way bootstrap:               PASS"
    )

    print(
        "Prior-performance disclosure:          PASS"
    )

    print(
        "Ablation provenance correction:        PASS"
    )

    print(
        "No NaN JSON placeholders:              PASS"
    )

    print(
        "Training remains blocked:              PASS"
    )

    # =========================================================================
    # 11. Persist V4
    # =========================================================================

    sep("-")

    print(
        "8.1.12-V4.11 — PERSIST FINAL CORRECTED CONTRACT"
    )

    sep("-")

    EXEC_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_json(
        V4_PATH,
        v4,
    )

    manifest = {
        "phase":
            "8.1.12",

        "version":
            "V4",

        "status":
            "FROZEN",

        "scientific_role":
            "FINAL_METHOD_DESIGN_FREEZE_CORRECTED",

        "V3_status":
            "FROZEN_SUPERSEDED",

        "V3_artifact_preserved":
            True,

        "authoritative_execution_contract":
            str(
                V4_PATH
            ),

        "method_design_closed":
            True,

        "feature_discovery_closed":
            True,

        "graph_schema":
            "8.1.9 V2",

        "inductive_consistency":
            "8.1.10A V2",

        "branch_semantics":
            "8.1.11A V2",

        "cold_mask":
            "8.1.12A V1",

        "founder_history":
            "8.1.12B V1",

        "origin_populations":
            "8.2.1C V2",

        "phase8_candidate_model_performance_seen":
            False,

        "training_allowed_immediately":
            False,

        "next_required_gate":
            (
                "Phase 8.2.0 V2 corrected "
                "implementation/materialization preflight"
            ),

        "training_performed":
            False,

        "T60_scored":
            False,

        "April_analyzed":
            False,
    }

    write_json(
        V4_MANIFEST,
        manifest,
    )

    write_json(
        V4_HASHES,
        {
            str(
                V4_PATH
            ):
                sha256_file(
                    V4_PATH
                ),

            str(
                V4_MANIFEST
            ):
                sha256_file(
                    V4_MANIFEST
                ),
        },
    )

    sep()

    print(
        "PHASE 8.1.12 V4 — FINAL METHOD DESIGN FREEZE"
    )

    sep()

    print()

    print(
        "8.1.12 V3:                         FROZEN_SUPERSEDED"
    )

    print(
        "8.1.12 V4:                         FROZEN / AUTHORITATIVE"
    )

    print()

    print(
        "COLD MASK:                         ALL outcome-independent COLD_t"
    )

    print(
        "FOUNDER HISTORY:                   H(s,t), founded_on < origin"
    )

    print(
        "P11/P12 PLACEMENT:                 STARTUP_NODE_FEATURE"
    )

    print(
        "COLD INFERENCE:                    SINK_ONLY"
    )

    print(
        "PREPROCESSING FIT:                 TRAINING-SIDE ONLY"
    )

    print(
        "NEGATIVE RNG:                      ITRS_PHASE8_TRAIN_NEGATIVE"
    )

    print(
        "ORDER RNG:                         ITRS_PHASE8_TRAIN_ORDER"
    )

    print(
        "INITIALIZATION RNG:                ITRS_PHASE8_NEURAL_INIT"
    )

    print(
        "BOOTSTRAP:                         TWO-WAY PIGEONHOLE / 10,000"
    )

    print(
        "PHASE6/7 RESULTS PREVIOUSLY SEEN:  YES / DISCLOSED"
    )

    print(
        "PHASE8 CANDIDATE RESULTS SEEN:      NO"
    )

    print(
        "FEATURE DISCOVERY REOPENED:         NO"
    )

    print(
        "METHOD DESIGN CLOSED:              YES"
    )

    print(
        "TRAINING AUTHORIZED YET:            NO"
    )

    print(
        "T60 SCORED:                        NO"
    )

    print(
        "APRIL ANALYZED:                    NO"
    )

    print()

    print("NEXT:")
    print(
        "Phase 8.2.0 V2 — corrected implementation/materialization "
        "preflight adoption."
    )


if __name__ == "__main__":
    main()
