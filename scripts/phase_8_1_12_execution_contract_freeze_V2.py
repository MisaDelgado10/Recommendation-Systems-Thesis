from pathlib import Path
import hashlib
import json

import pandas as pd


# =============================================================================
# PHASE 8.1.12 — RANDOMNESS / OPTIMIZER / EXECUTION CONTRACT FREEZE V2
# =============================================================================
#
# Scientific role:
#
#   FINAL METHOD-DESIGN FREEZE BEFORE IMPLEMENTATION / MATERIALIZATION
#
# PURPOSE
# -------
#
# Freeze:
#
#   - neural initialization policy;
#   - training RNG policy;
#   - negative-sampling policy;
#   - training-order policy;
#   - optimizer;
#   - epoch budget;
#   - development checkpoint / epoch selection;
#   - T60 no-selection protocol;
#   - April confirmatory execution;
#   - exact boundary between method design and implementation.
#
# This phase performs NO:
#
#   - model initialization;
#   - feature materialization;
#   - graph materialization;
#   - negative generation;
#   - forward pass;
#   - backward pass;
#   - optimizer step;
#   - model selection;
#   - T60 scoring;
#   - April scoring.
#
# =============================================================================


BASE = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation"
)

ARCHITECTURE_CONTRACT = (
    BASE
    / "phase_8_1_11_exact_neural_architecture"
    / "phase_8_1_11_architecture_contract_V2.json"
)

BRANCH_CONTRACT = (
    BASE
    / "phase_8_1_11a_model_family_branch_semantics"
    / "phase_8_1_11a_branch_semantics_amendment_V2.json"
)

PHASE4_INITIALIZATION = Path(
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


OUTPUT_DIR = (
    BASE
    / "phase_8_1_12_execution_contract"
)

DEVELOPMENT_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_12_development_fold_schedule_V2.csv"
)

RANDOMNESS_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_12_randomness_policy_V2.csv"
)

SELECTION_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_12_model_selection_policy_V2.csv"
)

FINAL_EXECUTION_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_12_t60_april_execution_policy_V2.csv"
)

CONTRACT_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_12_execution_contract_V2.json"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_12_manifest_V2.json"
)

HASH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_12_artifact_sha256_V2.json"
)


# =============================================================================
# Frozen inherited training controls
# =============================================================================

BASE_SEED = 42

MAX_EPOCHS = 20
BATCH_SIZE = 512

OPTIMIZER = "Adam"
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0

ADAM_BETAS = [0.9, 0.999]
ADAM_EPS = 1e-8

EARLY_STOPPING = False
LR_SCHEDULER = None

NEGATIVES_PER_POSITIVE = 4

TRAIN_NEGATIVE_NAMESPACE = (
    "ITRS_PHASE5_TRAIN_NEGATIVE"
)

TRAIN_ORDER_NAMESPACE = (
    "ITRS_PHASE5_TRAIN_ORDER"
)

BIT_GENERATOR = "numpy.random.PCG64"

GLOBAL_NEURAL_SEED = 42

KAIMING_DISTRIBUTION = "normal"
KAIMING_MODE = "fan_in"
KAIMING_NONLINEARITY = "relu"
BIAS_INITIALIZATION = "zeros"


# =============================================================================
# Frozen development origins from Phase 8.0 protocol
# =============================================================================

DEVELOPMENT_FOLDS = [
    {
        "origin":
            "T57",

        "training_segments":
            "T1-T56",

        "training_positive_events":
            1_003_526,

        "evaluation_cases":
            21_924,

        "primary_cases":
            6_850,
    },

    {
        "origin":
            "T58",

        "training_segments":
            "T1-T57",

        "training_positive_events":
            1_026_268,

        "evaluation_cases":
            22_384,

        "primary_cases":
            6_872,
    },

    {
        "origin":
            "T59",

        "training_segments":
            "T1-T58",

        "training_positive_events":
            1_049_605,

        "evaluation_cases":
            22_650,

        "primary_cases":
            6_764,
    },
]


IMPROVED_MODEL_FAMILIES = [
    "B1a",
    "B1b",
    "M1",
    "M2",
]


# =============================================================================
# Helpers
# =============================================================================

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


def derive_seed(namespace, epoch_index):

    material = (
        f"{namespace}|"
        f"{BASE_SEED}|"
        f"{epoch_index}"
    ).encode(
        "utf-8"
    )

    digest = hashlib.sha256(
        material
    ).digest()

    return (
        int.from_bytes(
            digest[:8],
            byteorder="little",
            signed=False,
        )
        %
        (
            (2 ** 63)
            - 1
        )
    )


def main():

    separator()

    print(
        "PHASE 8.1.12 — "
        "RANDOMNESS / OPTIMIZER / EXECUTION "
        "CONTRACT FREEZE V2"
    )

    separator()

    print()

    print(
        "Scientific role:             "
        "FINAL METHOD-DESIGN FREEZE"
    )

    print(
        "Architecture modified:       NO"
    )

    print(
        "Feature discovery reopened:  NO"
    )

    print(
        "Model initialized:           NO"
    )

    print(
        "Training performed:          NO"
    )

    print(
        "Model selection performed:   NO"
    )

    print(
        "T60 scored:                  NO"
    )

    print(
        "April analyzed:              NO"
    )

    # =========================================================================
    # 1. Prerequisite gates
    # =========================================================================

    separator("-")

    print(
        "8.1.12.1 — PREREQUISITE GATES"
    )

    separator("-")

    required_paths = [
        ARCHITECTURE_CONTRACT,
        BRANCH_CONTRACT,
        PHASE4_INITIALIZATION,
        PHASE5_NEGATIVE,
        PHASE5_OPTIMIZER,
    ]

    for path in required_paths:

        require(
            path.exists(),
            f"Missing prerequisite artifact: {path}",
        )

        print(
            f"FOUND {path}"
        )

    architecture = read_json(
        ARCHITECTURE_CONTRACT
    )

    branches = read_json(
        BRANCH_CONTRACT
    )

    phase4_init = read_json(
        PHASE4_INITIALIZATION
    )

    phase5_negative = read_json(
        PHASE5_NEGATIVE
    )

    phase5_optimizer = read_json(
        PHASE5_OPTIMIZER
    )

    require(
        architecture.get("status")
        ==
        "FROZEN",
        "Phase 8.1.11 is not frozen.",
    )

    require(
        architecture.get(
            "architecture_selected"
        )
        is True,
        "Architecture is not frozen.",
    )

    require(
        architecture.get(
            "architecture_selection_used_performance"
        )
        is False,
        "Architecture unexpectedly used performance.",
    )

    require(
        branches.get("status")
        ==
        "FROZEN",
        "Phase 8.1.11A is not frozen.",
    )

    require(
        branches.get(
            "model_trained"
        )
        is False,
        "Model unexpectedly trained.",
    )

    require(
        branches.get(
            "April_analyzed"
        )
        is False,
        "April unexpectedly analyzed.",
    )

    print()

    print(
        "Phase 8.1.11 architecture: PASS"
    )

    print(
        "Phase 8.1.11A semantics:    PASS"
    )

    # =========================================================================
    # 2. Recheck inherited initialization contract
    # =========================================================================

    separator("-")

    print(
        "8.1.12.2 — NEURAL INITIALIZATION POLICY"
    )

    separator("-")

    require(
        phase4_init.get("status")
        ==
        "FROZEN",
        "Phase-4 initialization contract not frozen.",
    )

    require(
        int(
            phase4_init[
                "global_neural_seed"
            ][
                "value"
            ]
        )
        ==
        GLOBAL_NEURAL_SEED,
        "Global neural seed drift.",
    )

    require(
        phase4_init[
            "kaiming"
        ][
            "distribution"
        ]
        ==
        KAIMING_DISTRIBUTION,
        "Kaiming distribution drift.",
    )

    require(
        phase4_init[
            "kaiming"
        ][
            "mode"
        ]
        ==
        KAIMING_MODE,
        "Kaiming mode drift.",
    )

    require(
        phase4_init[
            "kaiming"
        ][
            "nonlinearity"
        ]
        ==
        KAIMING_NONLINEARITY,
        "Kaiming nonlinearity drift.",
    )

    require(
        phase4_init[
            "bias"
        ][
            "initialization"
        ]
        ==
        BIAS_INITIALIZATION,
        "Bias initialization drift.",
    )

    initialization_policy = {
        "global_neural_seed":
            GLOBAL_NEURAL_SEED,

        "canonical_initialization_device":
            "CPU",

        "matrix_initialization":
            (
                "torch.nn.init.kaiming_normal_"
                "(a=0, mode=fan_in, nonlinearity=relu)"
            ),

        "bias_initialization":
            "zeros",

        "new_phase8_trainable_matrices":
            (
                "same Kaiming policy as frozen Phase-4"
            ),

        "new_phase8_biases":
            "zeros",

        "startup_or_person_identity_parameters":
            False,

        "implementation_requirement":
            (
                "Before first optimizer step, Phase-8 implementation "
                "must produce an explicit parameter namespace and prove "
                "that every trainable tensor has exactly one frozen "
                "initialization rule."
            ),
    }

    for key, value in (
        initialization_policy.items()
    ):

        print(
            f"{key:<44} {value}"
        )

    # =========================================================================
    # 3. Recheck inherited training-negative semantics
    # =========================================================================

    separator("-")

    print(
        "8.1.12.3 — TRAINING NEGATIVE-SAMPLING POLICY"
    )

    separator("-")

    require(
        phase5_negative.get("status")
        ==
        "FROZEN",
        "Phase-5 negative contract not frozen.",
    )

    neg_choices = (
        phase5_negative[
            "paper_unspecified_reproduction_choices"
        ]
    )

    neg_runtime = (
        phase5_negative[
            "sampling_runtime"
        ]
    )

    require(
        int(
            neg_choices[
                "training_negative_positive_ratio"
            ]
        )
        ==
        NEGATIVES_PER_POSITIVE,
        "Negative ratio drift.",
    )

    require(
        int(
            neg_runtime[
                "sampler_base_seed"
            ]
        )
        ==
        BASE_SEED,
        "Negative sampler base seed drift.",
    )

    require(
        neg_runtime[
            "seed_namespace"
        ]
        ==
        TRAIN_NEGATIVE_NAMESPACE,
        "Negative seed namespace drift.",
    )

    negative_policy = {
        "negative_positive_ratio":
            4,

        "distribution":
            "uniform over historically eligible Startups",

        "replacement":
            (
                "without replacement within each focal positive"
            ),

        "resampling":
            "once per epoch",

        "exclude_prior_positive_pairs":
            True,

        "exclude_same_target_positive_pairs":
            True,

        "exclude_future_positive_pairs":
            False,

        "future_labels_used":
            False,

        "candidate_universe":
            (
                "frozen Startup-role universe subject to "
                "origin-specific existence/eligibility rules"
            ),

        "same_stream_across_models_within_fold":
            True,

        "reason":
            (
                "B1a/B1b/M1/M2 must see identical labeled "
                "training streams for a given fold and epoch."
            ),
    }

    for key, value in (
        negative_policy.items()
    ):

        print(
            f"{key:<44} {value}"
        )

    # =========================================================================
    # 4. Recheck inherited optimizer / training controls
    # =========================================================================

    separator("-")

    print(
        "8.1.12.4 — OPTIMIZER / TRAINING CONTROL"
    )

    separator("-")

    require(
        phase5_optimizer.get("status")
        ==
        "FROZEN",
        "Phase-5 optimizer contract not frozen.",
    )

    paper_specified = (
        phase5_optimizer[
            "paper_specified"
        ]
    )

    choices = (
        phase5_optimizer[
            "paper_unspecified_reproduction_choices"
        ]
    )

    optimizer_runtime = (
        phase5_optimizer[
            "optimizer_runtime"
        ]
    )

    require(
        paper_specified[
            "optimizer"
        ]
        ==
        OPTIMIZER,
        "Optimizer drift.",
    )

    require(
        float(
            paper_specified[
                "learning_rate"
            ]
        )
        ==
        LEARNING_RATE,
        "Learning-rate drift.",
    )

    require(
        int(
            paper_specified[
                "batch_size"
            ]
        )
        ==
        BATCH_SIZE,
        "Batch-size drift.",
    )

    require(
        int(
            choices[
                "fixed_epochs"
            ]
        )
        ==
        MAX_EPOCHS,
        "Epoch-budget drift.",
    )

    require(
        choices[
            "early_stopping"
        ]
        is False,
        "Early-stopping policy drift.",
    )

    require(
        float(
            choices[
                "weight_decay"
            ]
        )
        ==
        WEIGHT_DECAY,
        "Weight-decay drift.",
    )

    require(
        list(
            optimizer_runtime[
                "betas"
            ]
        )
        ==
        ADAM_BETAS,
        "Adam beta drift.",
    )

    require(
        float(
            optimizer_runtime[
                "eps"
            ]
        )
        ==
        ADAM_EPS,
        "Adam epsilon drift.",
    )

    execution_optimizer = {
        "optimizer":
            OPTIMIZER,

        "learning_rate":
            LEARNING_RATE,

        "betas":
            ADAM_BETAS,

        "eps":
            ADAM_EPS,

        "weight_decay":
            WEIGHT_DECAY,

        "scheduler":
            None,

        "batch_size":
            BATCH_SIZE,

        "drop_last":
            False,

        "maximum_epochs":
            MAX_EPOCHS,

        "early_stopping":
            False,

        "loss":
            "BCEWithLogitsLoss",
    }

    for key, value in (
        execution_optimizer.items()
    ):

        print(
            f"{key:<30} {value}"
        )

    # =========================================================================
    # 5. Deterministic RNG streams
    # =========================================================================

    separator("-")

    print(
        "8.1.12.5 — RANDOMNESS CONTRACT"
    )

    separator("-")

    randomness_rows = [
        {
            "stream":
                "neural_initialization",

            "base_seed":
                42,

            "namespace":
                "PHASE4_GLOBAL_NEURAL_SEED",

            "generator":
                "torch CPU Generator",

            "shared_across_models_within_fold":
                True,
        },

        {
            "stream":
                "training_negatives",

            "base_seed":
                42,

            "namespace":
                TRAIN_NEGATIVE_NAMESPACE,

            "generator":
                BIT_GENERATOR,

            "shared_across_models_within_fold":
                True,
        },

        {
            "stream":
                "training_order",

            "base_seed":
                42,

            "namespace":
                TRAIN_ORDER_NAMESPACE,

            "generator":
                BIT_GENERATOR,

            "shared_across_models_within_fold":
                True,
        },

        {
            "stream":
                "truncated_svd",

            "base_seed":
                42,

            "namespace":
                "PHASE8_TFIDF_SVD",

            "generator":
                "scikit-learn random_state",

            "shared_across_models_within_fold":
                True,
        },

        {
            "stream":
                "clustered_bootstrap",

            "base_seed":
                42,

            "namespace":
                "PHASE8_TWO_WAY_CLUSTERED_BOOTSTRAP",

            "generator":
                BIT_GENERATOR,

            "shared_across_model_comparisons":
                True,
        },
    ]

    randomness_df = pd.DataFrame(
        randomness_rows
    )

    # -------------------------------------------------------------------------
    # Engineering fix from V1:
    # V1 attempted to persist the first CSV before OUTPUT_DIR existed.
    # Create the directory before ANY Phase-8.1.12 artifact write.
    # No scientific decision is changed.
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    randomness_df.to_csv(
        RANDOMNESS_OUTPUT,
        index=False,
    )

    print(
        randomness_df.to_string(
            index=False
        )
    )

    print()

    print(
        "Epoch-0 negative seed: "
        f"{derive_seed(TRAIN_NEGATIVE_NAMESPACE, 0)}"
    )

    print(
        "Epoch-0 order seed:    "
        f"{derive_seed(TRAIN_ORDER_NAMESPACE, 0)}"
    )

    # =========================================================================
    # 6. Development-fold schedule
    # =========================================================================

    separator("-")

    print(
        "8.1.12.6 — DEVELOPMENT EXECUTION SCHEDULE"
    )

    separator("-")

    development_rows = []

    for fold in DEVELOPMENT_FOLDS:

        positives = int(
            fold[
                "training_positive_events"
            ]
        )

        negatives = (
            positives
            *
            NEGATIVES_PER_POSITIVE
        )

        examples = (
            positives
            +
            negatives
        )

        batches = (
            examples
            +
            BATCH_SIZE
            -
            1
        ) // BATCH_SIZE

        development_rows.append(
            {
                "origin":
                    fold[
                        "origin"
                    ],

                "training_segments":
                    fold[
                        "training_segments"
                    ],

                "training_positive_events":
                    positives,

                "training_negative_examples_per_epoch":
                    negatives,

                "training_examples_per_epoch":
                    examples,

                "batches_per_epoch":
                    batches,

                "maximum_epochs":
                    MAX_EPOCHS,

                "evaluation_cases":
                    fold[
                        "evaluation_cases"
                    ],

                "primary_cases":
                    fold[
                        "primary_cases"
                    ],

                "candidates_per_case":
                    100,

                "evaluation_negatives_per_case":
                    99,

                "candidate_matrix":
                    "FROZEN_PHASE8_MATRIX",

                "used_for_epoch_selection":
                    True,
            }
        )

    development_df = pd.DataFrame(
        development_rows
    )

    development_df.to_csv(
        DEVELOPMENT_OUTPUT,
        index=False,
    )

    print(
        development_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # 7. Exact development model-selection rule
    # =========================================================================

    separator("-")

    print(
        "8.1.12.7 — DEVELOPMENT MODEL-SELECTION RULE"
    )

    separator("-")

    selection_rows = []

    for model in IMPROVED_MODEL_FAMILIES:

        selection_rows.append(
            {
                "model":
                    model,

                "candidate_epochs":
                    "1..20",

                "folds":
                    "T57 | T58 | T59",

                "selection_population":
                    "PRIMARY_COLD_START_ONLY",

                "primary_metric":
                    "EQUAL_FOLD_MEAN_NDCG@10",

                "secondary_metric":
                    "EQUAL_FOLD_MEAN_HR@10",

                "final_tiebreak":
                    "EARLIEST_EPOCH",

                "fold_weighting":
                    "EQUAL_FOLD_NOT_CASE_WEIGHTED",

                "early_stopping":
                    False,

                "architecture_change_after_metrics":
                    False,

                "feature_change_after_metrics":
                    False,

                "selected_epoch_reused_for_T60":
                    True,

                "selected_epoch_reused_for_April":
                    (
                        model
                        ==
                        "M2"
                    ),
            }
        )

    selection_df = pd.DataFrame(
        selection_rows
    )

    selection_df.to_csv(
        SELECTION_OUTPUT,
        index=False,
    )

    print(
        selection_df.to_string(
            index=False
        )
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "  We select one display epoch per model family."
    )

    print(
        "  We do NOT select a different best epoch independently "
        "for T57, T58 and T59."
    )

    print(
        "  For epoch e, first compute each fold's PRIMARY metric;"
    )

    print(
        "  then average the three fold metrics with equal fold weight."
    )

    # =========================================================================
    # 8. Preprocessing execution boundary
    # =========================================================================

    separator("-")

    print(
        "8.1.12.8 — PREPROCESSING EXECUTION BOUNDARY"
    )

    separator("-")

    preprocessing_policy = {
        "fit_once_per_origin":
            True,

        "tfidf_vocabulary":
            "training-side Startup text only",

        "svd_fit":
            "training-side Startup text only",

        "category_vocabulary":
            "training-side Startup values only",

        "category_group_vocabulary":
            "training-side Startup values only",

        "geography_vocabulary":
            "training-side Startup values only",

        "numeric_scalers":
            "training-side entities only",

        "shared_across_B1a_B1b_M1_M2":
            True,

        "evaluation_entities_may_transform":
            True,

        "evaluation_entities_may_fit":
            False,

        "evaluation_candidate_matrix_may_fit":
            False,

        "negative_sample_identity_may_fit":
            False,
    }

    for key, value in (
        preprocessing_policy.items()
    ):

        print(
            f"{key:<44} {value}"
        )

    # =========================================================================
    # 9. T60 / April execution boundary
    # =========================================================================

    separator("-")

    print(
        "8.1.12.9 — T60 / APRIL EXECUTION POLICY"
    )

    separator("-")

    final_rows = [
        {
            "stage":
                "T60_IMPROVED_MODELS",

            "models":
                "B1a | B1b | M1 | M2",

            "training_segments":
                "T1-T59",

            "training_positive_events":
                1_073_249,

            "epoch_policy":
                "USE_MODEL_SPECIFIC_DEV_SELECTED_EPOCH",

            "evaluation":
                (
                    "FROZEN_PHASE6_T60_FINAL_TEST_ARTIFACT"
                ),

            "evaluation_count":
                1,

            "selection_allowed":
                False,

            "architecture_change_allowed":
                False,

            "purpose":
                "HELDOUT_REPORTING_ONLY",
        },

        {
            "stage":
                "T60_B0",

            "models":
                "B0",

            "training_segments":
                "EXISTING_PHASE6_REPRODUCTION",

            "training_positive_events":
                1_073_249,

            "epoch_policy":
                (
                    "REUSE_FROZEN_PHASE6_SELECTED_CHECKPOINT"
                ),

            "evaluation":
                (
                    "REUSE_EXISTING_PHASE6_FINAL_TEST_SCORES"
                ),

            "evaluation_count":
                0,

            "selection_allowed":
                False,

            "architecture_change_allowed":
                False,

            "purpose":
                "IMMUTABLE_REFERENCE_BASELINE",
        },

        {
            "stage":
                "APRIL_B0_FINAL_FIT",

            "models":
                "B0",

            "training_segments":
                "T1-T60",

            "training_positive_events":
                1_095_764,

            "epoch_policy":
                (
                    "FIXED_19_DISPLAY_EPOCHS_FROM_PRIOR_"
                    "PHASE6_BASELINE_SELECTION"
                ),

            "evaluation":
                "APRIL_CONFIRMATORY",

            "evaluation_count":
                1,

            "selection_allowed":
                False,

            "architecture_change_allowed":
                False,

            "purpose":
                "CONFIRMATORY_REFERENCE",
        },

        {
            "stage":
                "APRIL_M2_FINAL_FIT",

            "models":
                "M2",

            "training_segments":
                "T1-T60",

            "training_positive_events":
                1_095_764,

            "epoch_policy":
                "USE_M2_DEV_SELECTED_EPOCH",

            "evaluation":
                "APRIL_CONFIRMATORY",

            "evaluation_count":
                1,

            "selection_allowed":
                False,

            "architecture_change_allowed":
                False,

            "purpose":
                "PRIMARY_CONFIRMATORY_MODEL",
        },
    ]

    final_df = pd.DataFrame(
        final_rows
    )

    final_df.to_csv(
        FINAL_EXECUTION_OUTPUT,
        index=False,
    )

    print(
        final_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # 10. April confirmatory statistical contract
    # =========================================================================

    separator("-")

    print(
        "8.1.12.10 — APRIL CONFIRMATORY STATISTICAL CONTRACT"
    )

    separator("-")

    april_policy = {
        "primary_models":
            [
                "B0",
                "M2",
            ],

        "evaluation_cases":
            6_313,

        "primary_cold_start_cases":
            1_690,

        "primary_endpoint":
            (
                "paired delta NDCG@10 on April PRIMARY "
                "warm-investor / investment-cold-startup cohort"
            ),

        "secondary_endpoint":
            (
                "paired delta HR@10 on the same PRIMARY cohort"
            ),

        "comparison":
            "M2 - B0",

        "bootstrap":
            (
                "10,000 replicate two-way clustered bootstrap"
            ),

        "cluster_dimensions":
            [
                "investor_id",
                "startup_id",
            ],

        "bootstrap_seed":
            42,

        "candidate_matrix":
            "FROZEN_1_POSITIVE_PLUS_99_NEGATIVES",

        "candidate_regeneration":
            False,

        "model_reselection":
            False,

        "architecture_reselection":
            False,

        "feature_reselection":
            False,

        "April_used_for_selection":
            False,

        "April_primary_execution_count":
            1,
    }

    for key, value in (
        april_policy.items()
    ):

        print(
            f"{key:<34} {value}"
        )

    # =========================================================================
    # 11. Development-only secondary analyses
    # =========================================================================

    separator("-")

    print(
        "8.1.12.11 — DEVELOPMENT-ONLY ABLATION BOUNDARY"
    )

    separator("-")

    ablation_policy = {
        "primary_architecture":
            "M2",

        "ablation_selection_data":
            "T57/T58/T59 ONLY",

        "T60_used_to_select_ablation":
            False,

        "April_used_for_ablation_selection":
            False,

        "April_ablation_scoring":
            False,

        "allowed_predeclared_ablations": [
            "remove geography",
            "remove category_groups",
            "remove company_type",
            "remove startup_age",
            "remove explicit founder relation",
            "remove founder prior history",
            "G_ACQUISITION_ONLY",
            "G_FOUNDER_ONLY",
            "G_LEGACY_PROJECTED_FOUNDER",
            "M2_no_heterogeneous_branch",
            "M1_vs_M2 trend comparison",
        ],

        "rule":
            (
                "Ablation results may explain the mechanism but "
                "cannot alter the frozen M2 architecture before "
                "T60 or April."
            ),
    }

    print(
        "Ablations: T57/T58/T59 only"
    )

    print(
        "T60 ablation selection: NO"
    )

    print(
        "April ablations:         NO"
    )

    # =========================================================================
    # 12. Freeze implementation launch gate
    # =========================================================================

    separator("-")

    print(
        "8.1.12.12 — IMPLEMENTATION LAUNCH GATE"
    )

    separator("-")

    implementation_gate = {
        "method_design_closed":
            True,

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

        "training_allowed_immediately":
            False,

        "reason_training_not_yet_allowed":
            (
                "Feature stores, graph tensors and Phase-8 model "
                "implementation must first be materialized and "
                "audited against the frozen contracts."
            ),

        "next_project_block":
            (
                "FRAMEWORK DEVELOPMENT + INTEGRATION"
            ),

        "next_phase":
            (
                "Phase 8.2.0 — implementation/materialization "
                "preflight and artifact plan"
            ),
    }

    for key, value in (
        implementation_gate.items()
    ):

        print(
            f"{key:<40} {value}"
        )

    # =========================================================================
    # 13. Persist
    # =========================================================================

    separator("-")

    print(
        "8.1.12.13 — PERSIST FINAL EXECUTION CONTRACT"
    )

    separator("-")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    contract = {
        "schema_version":
            "PHASE_8_1_12_EXECUTION_CONTRACT_V2",

        "status":
            "FROZEN",

        "scientific_role":
            "FINAL_METHOD_DESIGN_FREEZE",

        "source_contracts": {
            "architecture":
                str(ARCHITECTURE_CONTRACT),

            "branch_semantics":
                str(BRANCH_CONTRACT),

            "initialization":
                str(PHASE4_INITIALIZATION),

            "negative_sampling":
                str(PHASE5_NEGATIVE),

            "optimizer":
                str(PHASE5_OPTIMIZER),
        },

        "initialization":
            initialization_policy,

        "negative_sampling":
            negative_policy,

        "optimizer_training_control":
            execution_optimizer,

        "randomness":
            randomness_df.to_dict(
                orient="records"
            ),

        "development_folds":
            development_df.to_dict(
                orient="records"
            ),

        "development_model_selection":
            selection_df.to_dict(
                orient="records"
            ),

        "preprocessing":
            preprocessing_policy,

        "T60_April_execution":
            final_df.to_dict(
                orient="records"
            ),

        "April_confirmatory":
            april_policy,

        "development_ablation_policy":
            ablation_policy,

        "implementation_gate":
            implementation_gate,

        "method_design_closed":
            True,

        "performance_observed_before_method_freeze":
            False,

        "architecture_change_after_dev_allowed":
            False,

        "feature_change_after_dev_allowed":
            False,

        "T60_selection_allowed":
            False,

        "April_selection_allowed":
            False,

        "model_initialized":
            False,

        "training_performed":
            False,

        "T60_scored":
            False,

        "April_analyzed":
            False,

        "next_step":
            (
                "Phase 8.2.0 — implementation/materialization "
                "preflight."
            ),
    }

    write_json(
        CONTRACT_OUTPUT,
        contract,
    )

    manifest = {
        "phase":
            "8.1.12",

        "status":
            "FROZEN",

        "method_design_closed":
            True,

        "outputs": [
            str(DEVELOPMENT_OUTPUT),
            str(RANDOMNESS_OUTPUT),
            str(SELECTION_OUTPUT),
            str(FINAL_EXECUTION_OUTPUT),
            str(CONTRACT_OUTPUT),
        ],

        "model_initialized":
            False,

        "training_performed":
            False,

        "T60_scored":
            False,

        "April_analyzed":
            False,

        "next_project_block":
            "FRAMEWORK DEVELOPMENT + INTEGRATION",
    }

    write_json(
        MANIFEST_OUTPUT,
        manifest,
    )

    outputs = [
        DEVELOPMENT_OUTPUT,
        RANDOMNESS_OUTPUT,
        SELECTION_OUTPUT,
        FINAL_EXECUTION_OUTPUT,
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
        "PHASE 8.1.12 EXECUTION CONTRACT FROZEN"
    )

    separator()

    print()

    print(
        "GLOBAL NEURAL SEED:                42"
    )

    print(
        "INITIALIZATION:                    "
        "Kaiming normal / fan_in / ReLU"
    )

    print(
        "OPTIMIZER:                         Adam"
    )

    print(
        "LEARNING RATE:                     0.001"
    )

    print(
        "WEIGHT DECAY:                      0"
    )

    print(
        "BATCH SIZE:                        512"
    )

    print(
        "MAX EPOCHS:                        20"
    )

    print(
        "EARLY STOPPING:                    NO"
    )

    print(
        "TRAIN NEGATIVES:                   4:1"
    )

    print()

    print(
        "DEV SELECTION:"
    )

    print(
        "  T57 / T58 / T59"
    )

    print(
        "  equal-fold mean PRIMARY NDCG@10"
    )

    print(
        "  then mean HR@10"
    )

    print(
        "  then earliest epoch"
    )

    print()

    print(
        "T60 MODEL SELECTION:                FORBIDDEN"
    )

    print(
        "APRIL MODEL SELECTION:              FORBIDDEN"
    )

    print(
        "APRIL PRIMARY:                      M2 vs B0"
    )

    print(
        "APRIL PRIMARY ENDPOINT:             delta NDCG@10"
    )

    print(
        "APRIL BOOTSTRAP:                    "
        "10,000 two-way clustered / seed 42"
    )

    print()

    separator()

    print(
        "INDUCTIVE METHOD STUDY & DESIGN: CLOSED"
    )

    separator()

    print()

    print(
        "TRAINING ALLOWED NOW: NO"
    )

    print(
        "NEXT PROJECT BLOCK:"
    )

    print(
        "FRAMEWORK DEVELOPMENT + INTEGRATION"
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Phase 8.2.0 — implementation/materialization preflight."
    )


if __name__ == "__main__":
    main()
